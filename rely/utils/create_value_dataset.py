#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from rely.utils import extract_final_answer, normalize_answer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_norm_answer(text: str) -> str:
    """Extract final answer then normalize. Return '' if extraction fails."""
    if text is None:
        return ""
    text = str(text)

    try:
        ans = extract_final_answer(text)
    except Exception:
        ans = None

    if ans is None:
        return ""

    try:
        return normalize_answer(ans)
    except Exception:
        return str(ans).strip()


def split_steps(attempt: str, step_separator: str) -> List[str]:
    if not attempt:
        return []
    # Strip only whitespace at ends to avoid trailing "" steps
    attempt = attempt.strip()
    steps = attempt.split(step_separator)
    # Drop trailing empties just in case
    while steps and steps[-1].strip() == "":
        steps.pop()
    return steps


def fraction_correct_for_step(samples_for_step: List[Dict[str, Any]], gold_norm: str) -> Optional[float]:
    """
    For a given step, we have multiple sample groups (could happen if you reran / merged).
    Each group provides completions for the same cut_cot.
    We score each completion by running extract_final_answer on (cut_cot + completion),
    falling back to cut_cot alone if completion is empty or extraction fails on the concat.
    """
    if not gold_norm:
        return None

    total = 0
    correct = 0

    for s in samples_for_step:
        cut_cot = s.get("cut_cot", "") or ""
        completions = s.get("completions", []) or []

        for comp in completions:
            comp = "" if comp is None else str(comp)
            full_text = cut_cot + comp

            pred_norm = safe_norm_answer(full_text)

            # If the model didn't produce a final answer in the completion, but cut_cot already contains one,
            # this makes “full CoT” prefixes score correctly.
            if pred_norm == "":
                pred_norm = safe_norm_answer(cut_cot)

            total += 1
            if pred_norm != "" and pred_norm == gold_norm:
                correct += 1

    if total == 0:
        return None
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--step_separator", type=str, default="\n\n")
    ap.add_argument("--attempt_field", type=str, default="attempt")
    ap.add_argument("--solution_field", type=str, default="solution")
    ap.add_argument("--keep_samples", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.completions_jsonl)
    out_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Building value.jsonl"):
        original = row.get("original_item", {}) or {}
        samples = row.get("samples", []) or []

        attempt = original.get(args.attempt_field, "") or ""
        steps = split_steps(attempt, args.step_separator)
        n_steps = len(steps)

        gold_norm = safe_norm_answer(original.get(args.solution_field, "") or "")
        # If solution is already in plain form (e.g. "(15,-29)") extract_final_answer might fail.
        # In that case, normalize the raw solution as a fallback.
        if gold_norm == "":
            try:
                gold_norm = normalize_answer(original.get(args.solution_field, "") or "")
            except Exception:
                gold_norm = str(original.get(args.solution_field, "") or "").strip()

        # Group sample blocks by sample_idx (which in your pipeline is the 0-based step index)
        by_step: List[List[Dict[str, Any]]] = [[] for _ in range(n_steps)]
        for s in samples:
            idx = s.get("sample_idx", None)
            if idx is None:
                continue
            try:
                idx = int(idx)
            except Exception:
                continue
            if 0 <= idx < n_steps:
                by_step[idx].append(s)

        gt_value: List[Optional[float]] = []
        for i in range(n_steps):
            v = fraction_correct_for_step(by_step[i], gold_norm)
            gt_value.append(v)

        out_item = dict(original)
        out_item["steps"] = steps
        out_item["gt_value"] = gt_value

        if args.keep_samples:
            out_item["samples"] = samples

        out_rows.append(out_item)

    write_jsonl(args.output_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} items to {args.output_jsonl}")


if __name__ == "__main__":
    main()
