############################################
# COMPLETIONS --> GOLDEN
############################################

import argparse
import os
from pathlib import Path

from rely.utils import (
    load_dataset,
    save_dataset,
    extract_final_answer,
    normalize_answer,
)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Convert SBS completions into a golden dataset with value and variance labels.")
    p.add_argument(
        "--completions_path",
        type=str,
        required=True,
        help="Path to the SBS completions JSONL file (output of SBS inference).",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the processed dataset. Defaults to the completions directory with a golden_* filename.",
    )
    return p.parse_args()

# ---------------------------------------------------------------------
# Helper: compute variance p(1-p) for a list of completions at one step
# ---------------------------------------------------------------------

def compute_step_variance(completions, correct_answer):
    """
    completions: list[str] – all rollouts from a single step
    correct_answer: normalized gold answer (string)
    Returns: (p, variance, n_correct, n_total)
    """
    n_total = len(completions)
    n_correct = 0

    for c in completions:
        try:
            pred = extract_final_answer(c)
            pred_norm = normalize_answer(pred)
        except Exception as e:
            print(f"[WARN] Failed to extract/normalize from completion, marking as incorrect. Error: {e}")
            pred_norm = None

        if pred_norm == correct_answer:
            n_correct += 1

    if n_total == 0:
        p = 0.0
    else:
        p = n_correct / n_total

    variance = p * (1.0 - p)
    return p, variance, n_correct, n_total


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    completions_path = Path(args.completions_path)

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        base = completions_path.stem
        if base.startswith("completions"):
            base = "golden" + base[len("completions"):]
        else:
            base = f"golden_{base}"
        out_path = completions_path.with_name(base + ".jsonl")

    os.makedirs(out_path.parent, exist_ok=True)

    print(f"[INFO] Loading completions from: {completions_path}")
    comp_ds = load_dataset(completions_path)
    print(f"[INFO] Loaded {len(comp_ds)} completion entries")

    if len(comp_ds) > 0:
        print("[DEBUG] Example completions entry keys:", list(comp_ds[0].keys()))
        if "original_item" in comp_ds[0]:
            print("[DEBUG] original_item keys:", list(comp_ds[0]["original_item"].keys()))
        if "samples" in comp_ds[0]:
            print("[DEBUG] Number of samples (steps) in example 0:", len(comp_ds[0]["samples"]))
            if len(comp_ds[0]["samples"]) > 0:
                print("[DEBUG] Keys in first sample:", list(comp_ds[0]["samples"][0].keys()))

    n_with_solution = 0
    n_missing_solution = 0

    out_examples = []

    for i, ex in enumerate(comp_ds):
        # Extract question and solution from original_item
        if "original_item" not in ex:
            print(f"[WARN] completions[{i}] missing 'original_item'; keys={list(ex.keys())}")
            continue

        original_item = ex["original_item"]

        if "question" not in original_item:
            print(f"[WARN] completions[{i}].original_item missing 'question'; keys={list(original_item.keys())}")
            continue

        if "solution" not in original_item or original_item["solution"] is None:
            n_missing_solution += 1
            if n_missing_solution <= 5:
                print(f"[WARN] completions[{i}].original_item has no 'solution' field or is None.")
                print("       original_item keys:", list(original_item.keys()))
            continue

        question = original_item["question"]
        sol_text = original_item["solution"]

        # Debug for the first few examples
        if i < 3:
            print(f"[DEBUG] Example {i} question:", question[:200], "...")
            print(f"[DEBUG] Example {i} raw solution:", str(sol_text)[:200], "...")

        # Extract and normalize gold answer
        try:
            final_ans = extract_final_answer(sol_text)
            gold_ans = normalize_answer(final_ans)
        except Exception as e:
            print(f"[ERROR] Failed to extract/normalize gold answer for example {i}: {e}")
            continue

        n_with_solution += 1

        if i < 3:
            print(f"[DEBUG] Example {i} extract_final_answer:", final_ans)
            print(f"[DEBUG] Example {i} normalize_answer:", gold_ans)

        samples = ex.get("samples", [])
        if len(samples) == 0:
            print(f"[WARN] completions[{i}] has no 'samples'")
            continue

        step_values = []
        step_variances = []
        step_representative_completions = []

        # Each 'sample' is one step; we use all completions at that step
        for step_idx, step in enumerate(samples):
            completions_step = step.get("completions", [])
            if step_idx < 2 and i < 2:
                print(f"[DEBUG]   Example {i}, step {step_idx}: {len(completions_step)} completions")

            p, var, n_correct, n_total = compute_step_variance(completions_step, gold_ans)
            step_values.append(p)
            step_variances.append(var)

            # Use the first completion as a representative prefix for this step
            if len(completions_step) > 0:
                rep = completions_step[0]
            else:
                rep = ""  # empty placeholder
            step_representative_completions.append(rep)

            if i < 2 and step_idx < 2:
                print(
                    f"[DEBUG]      step {step_idx}: "
                    f"p={p:.3f}, var={var:.3f}, n_correct={n_correct}/{n_total}"
                )

        # Build output record: same style as HF dataset
        out_ex = {
            "prompt": question,
            "completions": step_representative_completions,
            "value": step_values,
            "variance": step_variances,
            "labels": step_values,  # Backward-compatible alias for value
        }
        out_examples.append(out_ex)

    print("[INFO] Finished processing completions.")
    print(f"[INFO] Examples with solution:   {n_with_solution}")
    print(f"[INFO] Examples missing solution: {n_missing_solution}")
    print(f"[INFO] Output examples:           {len(out_examples)}")

    if len(out_examples) > 0:
        print("[DEBUG] First output example:")
        print("  prompt:", out_examples[0]["prompt"][:200], "...")
        print("  #completions:", len(out_examples[0]["completions"]))
        print("  labels (first few):", out_examples[0]["labels"][:5])

    print(f"[INFO] Saving golden dataset to: {out_path}")
    save_dataset(out_examples, out_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
