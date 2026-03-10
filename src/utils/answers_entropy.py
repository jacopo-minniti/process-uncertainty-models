"""
answer_entropy.py

Measure "completion entropy" at the *final-answer level* (not token entropy):
for each (item, sample) in completions.jsonl, parse + normalize the final answer
for every completion, build a categorical distribution over answers, and compute
Shannon entropy + diversity stats.

Now also calculates correctness metrics (accuracy) and entropy conditioned on
correctness (e.g. entropy of incorrect answers) to distinguish difficulty.

Expected input JSONL structure (your completer output):
{
  "original_item": {
      "solution": "...",
      ...
  },
  "samples": [
    {
      "sample_idx": int,
      "cut_cot": str,
      "completions": [str, str, ...]
    },
    ...
  ]
}

Outputs:
- a summary JSON with pooled + mean stats
- optionally a per-sample JSONL with detailed metrics
- optionally a per-item JSONL aggregating samples within each original item
"""

import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rely.utils import load_dataset, extract_final_answer, normalize_answer


# -------------------------
# Helpers
# -------------------------

def shannon_entropy_from_counts(counts: Counter) -> Tuple[float, float]:
    """Return (entropy_nats, entropy_bits) for a categorical distribution."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    ent_nats = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent_nats -= p * math.log(p)
    ent_bits = ent_nats / math.log(2.0) if ent_nats > 0 else 0.0
    return ent_nats, ent_bits


def safe_norm_answer(s: str) -> Optional[str]:
    """Extract + normalize answer. Returns None if unparseable/empty."""
    if not s:
        return None
    ans = extract_final_answer(s)
    if ans is None:
        return None
    ans = str(ans).strip()
    if not ans:
        return None
    try:
        return normalize_answer(ans)
    except Exception:
        # If normalize_answer is strict for some formats, fallback to a light normalization.
        return ans.strip().lower()


def safe_norm_ground_truth(s: str) -> Optional[str]:
    """Normalize the ground truth string directly (no extraction needed usually)."""
    if not s:
        return None
    s = str(s).strip()
    try:
        return normalize_answer(s)
    except Exception:
        return s.strip().lower()


def summarize_array(x: List[float]) -> Dict[str, float]:
    if not x:
        return {}
    arr = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# -------------------------
# Core computation
# -------------------------

def compute_sample_answer_entropy(
    completions: List[str],
    *,
    min_parsed: int,
    include_unparseable: bool,
    solution_str: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Compute answer-level entropy metrics for one sample (one cut_cot with N completions).
    Also computes accuracy and conditional entropy if solution_str is provided.
    Returns None if too few parsed (and not including unparseable as category).
    """
    total = len(completions)
    if total == 0:
        return None

    answers: List[str] = []
    n_unparseable = 0

    for c in completions:
        a = safe_norm_answer(c)
        if a is None:
            n_unparseable += 1
            if include_unparseable:
                answers.append("<UNPARSEABLE>")
        else:
            answers.append(a)

    if not include_unparseable:
        parsed = total - n_unparseable
        if parsed < min_parsed:
            return None
    else:
        parsed = total - n_unparseable
        if parsed < min_parsed:
            return None

    # --- Basic Entropy ---
    counts = Counter(answers)
    entropy_nats, entropy_bits = shannon_entropy_from_counts(counts)

    K = len(counts)
    top1 = max(counts.values()) if counts else 0
    top1_frac = (top1 / sum(counts.values())) if counts else 0.0

    # "Effective number of answers" (perplexity)
    eff_num = float(math.exp(entropy_nats)) if entropy_nats > 0 else 1.0

    # Normalized entropy (0..1)
    norm_ent = float(entropy_nats / math.log(K)) if K > 1 else 0.0

    # --- Correctness & Conditional Entropy ---
    accuracy = None
    ent_incorrect_nats = None
    ent_incorrect_bits = None
    n_incorrect_unique = None

    if solution_str is not None:
        norm_sol = safe_norm_ground_truth(solution_str)
        if norm_sol is not None:
            # Check correctness for every valid answer
            # Note: If unparseable is included, it counts as incorrect (unless sol is <UNPARSEABLE> which is unlikely)
            correct_count = 0
            incorrect_answers = []
            
            for ans in answers:
                if ans == norm_sol:
                    correct_count += 1
                else:
                    incorrect_answers.append(ans)
            
            n_used = len(answers)
            accuracy = float(correct_count / n_used) if n_used > 0 else 0.0
            
            # Entropy of incorrect answers (H(Y | Y != Solution))
            # If accuracy is 1.0, this is undefined (or 0 implicitly), we set to None or 0.
            if incorrect_answers:
                inc_counts = Counter(incorrect_answers)
                ei_nats, ei_bits = shannon_entropy_from_counts(inc_counts)
                ent_incorrect_nats = ei_nats
                ent_incorrect_bits = ei_bits
                n_incorrect_unique = len(inc_counts)
            else:
                # Perfect accuracy implies 0 entropy on incorrect distribution (it doesn't exist)
                ent_incorrect_nats = 0.0
                ent_incorrect_bits = 0.0
                n_incorrect_unique = 0

    return {
        "n_completions_total": total,
        "n_parseable": parsed,
        "n_unparseable": n_unparseable,
        "parse_rate": float(parsed / total) if total > 0 else 0.0,
        "unique_answers": int(K),
        "top1_answer_frac": float(top1_frac),
        "answer_entropy_nats": float(entropy_nats),
        "answer_entropy_bits": float(entropy_bits),
        "answer_entropy_normalized": float(norm_ent),
        "effective_num_answers": float(eff_num),
        # Correctness metrics (will be None if no solution found)
        "accuracy": accuracy,
        "entropy_given_incorrect_nats": ent_incorrect_nats,
        "entropy_given_incorrect_bits": ent_incorrect_bits,
        "n_incorrect_unique": n_incorrect_unique
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", type=str, required=True, help="Path to completions.jsonl")
    ap.add_argument("--output_summary", type=str, required=True, help="Path to write summary JSON")
    ap.add_argument("--output_per_sample", type=str, default=None, help="Optional: write per-sample metrics JSONL")
    ap.add_argument("--output_per_item", type=str, default=None, help="Optional: write per-item aggregated metrics JSONL")
    ap.add_argument("--min_parsed", type=int, default=1, help="Min parseable completions required per sample")
    ap.add_argument(
        "--include_unparseable",
        action="store_true",
        help="If set, treat unparseable as an answer category '<UNPARSEABLE>' (still respects min_parsed by default).",
    )
    args = ap.parse_args()

    data = load_dataset(args.input_path)
    if not data:
        raise SystemExit(f"Could not load any data from {args.input_path}")

    per_sample_rows: List[Dict[str, Any]] = []
    per_item_rows: List[Dict[str, Any]] = []

    # Aggregates across all valid samples
    # We use lists to compute full distribution stats (mean/std/percentiles)
    stats_acc: List[float] = []
    stats_ent_inc_bits: List[float] = []
    
    ent_nats_all: List[float] = []
    ent_bits_all: List[float] = []
    uniq_all: List[float] = []
    top1_all: List[float] = []
    parse_rate_all: List[float] = []
    eff_num_all: List[float] = []
    norm_ent_all: List[float] = []

    # Weighted sums
    w_sum_acc = 0.0
    w_sum_ent_inc_bits = 0.0
    w_sum_inc_weight = 0.0 # Weight for conditional entropy (only where errors exist)

    w_sum_ent_nats = 0.0
    w_sum_ent_bits = 0.0
    w_sum_uniq = 0.0
    w_sum_top1 = 0.0
    w_sum_parse_rate = 0.0
    w_sum_eff = 0.0
    w_sum_norm_ent = 0.0
    w_total = 0.0

    n_items = 0
    n_samples_total = 0
    n_samples_used = 0
    n_completions_total = 0
    n_parseable_total = 0
    n_unparseable_total = 0

    for item_idx, item in enumerate(data):
        n_items += 1
        original = item.get("original_item", {}) or {}
        solution_str = original.get("solution", None) # Retrieve ground truth
        
        samples = item.get("samples", []) or []

        # Per-item temporary lists
        item_ent_nats: List[float] = []
        item_ent_bits: List[float] = []
        item_uniq: List[float] = []
        item_top1: List[float] = []
        item_parse: List[float] = []
        item_eff: List[float] = []
        item_norm_ent: List[float] = []
        item_acc: List[float] = []
        item_ent_inc: List[float] = []

        for s in samples:
            n_samples_total += 1
            completions = s.get("completions", []) or []
            n_completions_total += len(completions)

            metrics = compute_sample_answer_entropy(
                completions,
                min_parsed=args.min_parsed,
                include_unparseable=args.include_unparseable,
                solution_str=solution_str
            )
            if metrics is None:
                continue

            n_samples_used += 1
            n_parseable_total += int(metrics["n_parseable"])
            n_unparseable_total += int(metrics["n_unparseable"])

            row = {
                "item_idx": item_idx,
                "generations_idx": original.get("generations_idx", None),
                "sample_idx": s.get("sample_idx", None),
                "n_completions_total": metrics["n_completions_total"],
                "n_parseable": metrics["n_parseable"],
                "n_unparseable": metrics["n_unparseable"],
                "parse_rate": metrics["parse_rate"],
                "unique_answers": metrics["unique_answers"],
                "top1_answer_frac": metrics["top1_answer_frac"],
                "answer_entropy_nats": metrics["answer_entropy_nats"],
                "answer_entropy_bits": metrics["answer_entropy_bits"],
                "answer_entropy_normalized": metrics["answer_entropy_normalized"],
                "effective_num_answers": metrics["effective_num_answers"],
                # New correctness metrics
                "accuracy": metrics["accuracy"],
                "entropy_given_incorrect_bits": metrics["entropy_given_incorrect_bits"],
                "n_incorrect_unique": metrics["n_incorrect_unique"]
            }
            per_sample_rows.append(row)

            # Unweighted aggregates
            ent_nats_all.append(metrics["answer_entropy_nats"])
            ent_bits_all.append(metrics["answer_entropy_bits"])
            uniq_all.append(metrics["unique_answers"])
            top1_all.append(metrics["top1_answer_frac"])
            parse_rate_all.append(metrics["parse_rate"])
            eff_num_all.append(metrics["effective_num_answers"])
            norm_ent_all.append(metrics["answer_entropy_normalized"])
            
            if metrics["accuracy"] is not None:
                stats_acc.append(metrics["accuracy"])
            if metrics["entropy_given_incorrect_bits"] is not None:
                stats_ent_inc_bits.append(metrics["entropy_given_incorrect_bits"])

            # Weighted aggregates
            w = float(metrics["n_parseable"])
            if w > 0:
                w_sum_ent_nats += w * float(metrics["answer_entropy_nats"])
                w_sum_ent_bits += w * float(metrics["answer_entropy_bits"])
                w_sum_uniq += w * float(metrics["unique_answers"])
                w_sum_top1 += w * float(metrics["top1_answer_frac"])
                w_sum_parse_rate += w * float(metrics["parse_rate"])
                w_sum_eff += w * float(metrics["effective_num_answers"])
                w_sum_norm_ent += w * float(metrics["answer_entropy_normalized"])
                
                if metrics["accuracy"] is not None:
                    w_sum_acc += w * float(metrics["accuracy"])
                
                # For incorrect entropy, we probably want to weight by number of *incorrect* answers
                # or just track it globally. Here we weight by total parseable for general stats, 
                # but technically conditional entropy should be weighted by P(condition).
                # Let's simple-average it in the weighted block if accuracy < 1.0
                if metrics["accuracy"] is not None and metrics["accuracy"] < 1.0:
                    n_inc = w * (1.0 - metrics["accuracy"])
                    if n_inc > 0 and metrics["entropy_given_incorrect_bits"] is not None:
                        w_sum_ent_inc_bits += n_inc * float(metrics["entropy_given_incorrect_bits"])
                        w_sum_inc_weight += n_inc

                w_total += w

            # Per-item buckets
            item_ent_nats.append(metrics["answer_entropy_nats"])
            item_ent_bits.append(metrics["answer_entropy_bits"])
            item_uniq.append(metrics["unique_answers"])
            item_top1.append(metrics["top1_answer_frac"])
            item_parse.append(metrics["parse_rate"])
            item_eff.append(metrics["effective_num_answers"])
            item_norm_ent.append(metrics["answer_entropy_normalized"])
            if metrics["accuracy"] is not None:
                item_acc.append(metrics["accuracy"])
            if metrics["entropy_given_incorrect_bits"] is not None:
                item_ent_inc.append(metrics["entropy_given_incorrect_bits"])

        # Optional per-item aggregation
        if args.output_per_item:
            if item_ent_nats:
                row_item = {
                    "item_idx": item_idx,
                    "generations_idx": original.get("generations_idx", None),
                    "n_samples_used": int(len(item_ent_nats)),
                    "mean_answer_entropy_nats": float(np.mean(item_ent_nats)),
                    "mean_answer_entropy_bits": float(np.mean(item_ent_bits)),
                    "mean_unique_answers": float(np.mean(item_uniq)),
                    "mean_top1_answer_frac": float(np.mean(item_top1)),
                    "mean_parse_rate": float(np.mean(item_parse)),
                    "mean_effective_num_answers": float(np.mean(item_eff)),
                    "mean_answer_entropy_normalized": float(np.mean(item_norm_ent)),
                    "mean_accuracy": float(np.mean(item_acc)) if item_acc else None,
                    "mean_entropy_given_incorrect_bits": float(np.mean(item_ent_inc)) if item_ent_inc else None
                }
                per_item_rows.append(row_item)

    # Build summary
    summary: Dict[str, Any] = {
        "input_path": args.input_path,
        "settings": {
            "min_parsed": args.min_parsed,
            "include_unparseable": bool(args.include_unparseable),
            "entropy_definition": "Shannon entropy over normalized final answers",
        },
        "counts": {
            "n_items": int(n_items),
            "n_samples_total": int(n_samples_total),
            "n_samples_used": int(n_samples_used),
            "n_completions_total": int(n_completions_total),
            "n_parseable_total_used_samples": int(n_parseable_total),
            "n_unparseable_total_used_samples": int(n_unparseable_total),
        },
        "unweighted_per_sample": {
            "answer_entropy_nats": summarize_array(ent_nats_all),
            "answer_entropy_bits": summarize_array(ent_bits_all),
            "unique_answers": summarize_array(uniq_all),
            "top1_answer_frac": summarize_array(top1_all),
            "parse_rate": summarize_array(parse_rate_all),
            "effective_num_answers": summarize_array(eff_num_all),
            "answer_entropy_normalized": summarize_array(norm_ent_all),
            "accuracy": summarize_array(stats_acc),
            "entropy_given_incorrect_bits": summarize_array(stats_ent_inc_bits),
        },
        "weighted_by_n_parseable": (
            {
                "total_weight": float(w_total),
                "answer_entropy_nats_mean": float(w_sum_ent_nats / w_total) if w_total > 0 else None,
                "answer_entropy_bits_mean": float(w_sum_ent_bits / w_total) if w_total > 0 else None,
                "unique_answers_mean": float(w_sum_uniq / w_total) if w_total > 0 else None,
                "top1_answer_frac_mean": float(w_sum_top1 / w_total) if w_total > 0 else None,
                "parse_rate_mean": float(w_sum_parse_rate / w_total) if w_total > 0 else None,
                "effective_num_answers_mean": float(w_sum_eff / w_total) if w_total > 0 else None,
                "answer_entropy_normalized_mean": float(w_sum_norm_ent / w_total) if w_total > 0 else None,
                "accuracy_mean": float(w_sum_acc / w_total) if w_total > 0 else None,
                # Weighted average of incorrect entropy (weighted by number of incorrect samples)
                "entropy_given_incorrect_bits_mean": float(w_sum_ent_inc_bits / w_sum_inc_weight) if w_sum_inc_weight > 0 else None
            }
            if w_total > 0
            else {"total_weight": 0.0}
        ),
    }

    # Write outputs
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary to {args.output_summary}")

    if args.output_per_sample:
        with open(args.output_per_sample, "w", encoding="utf-8") as f:
            for row in per_sample_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-sample metrics to {args.output_per_sample} ({len(per_sample_rows)} rows)")

    if args.output_per_item:
        with open(args.output_per_item, "w", encoding="utf-8") as f:
            for row in per_item_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-item metrics to {args.output_per_item} ({len(per_item_rows)} rows)")

    # Quick console snippet
    if ent_bits_all:
        print("\nQuick view (unweighted per-sample):")
        print(f"  mean accuracy:       {float(np.mean(stats_acc)):.4f}" if stats_acc else "  mean accuracy: N/A")
        print(f"  mean entropy (bits): {float(np.mean(ent_bits_all)):.4f}")
        print(f"  mean unique answers: {float(np.mean(uniq_all)):.4f}")
        print(f"  mean parse rate:     {float(np.mean(parse_rate_all)):.4f}")


if __name__ == "__main__":
    main()