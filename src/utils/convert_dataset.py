import json
import random
import time
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from rely.utils import load_dataset, save_dataset, extract_final_answer, normalize_answer


# =========================
# Global configuration
# =========================

INPUT_PATH = "data/mmlu/completions_qwen2.5.jsonl"
N = 8
OUTPUT_PREFIX = "qwen2.5_value"

# You can tune this
MIN_PARSABLE_COMPLETIONS = 1
TRAIN_RATIO = 0.85


# =========================
# Core helpers
# =========================

def clean_cot(text: str) -> str:
    """
    Clean CoT text:
    - KEEP <think> and </think>
    - REMOVE ChatML-style control tokens like <|im_start|>, <|im_end|>, etc.
    """
    if not text:
        return ""

    # Remove ChatML-style control tokens
    text = re.sub(r"<\|im_[^|]*\|>", "", text)

    return text


def calculate_value(
    completions: List[str],
    ground_truth: str,
    min_parsed: int = MIN_PARSABLE_COMPLETIONS,
) -> Tuple[Optional[float], int, int, int]:
    """
    Calculate percentage of correct answers across completions, based on parsed answers only.

    Returns:
        value (float or None): percentage of correct answers (0.0 to 1.0). None if we skip this sample (too few parsed answers or no GT).
        parsed_count (int): number of completions with a parseable final answer
        total_completions (int): raw completions seen for this sample
        correct_parsed (int): number of parsed answers matching ground_truth
    """
    total = len(completions)

    norm_gt = normalize_answer(ground_truth)

    answers = []
    for completion in completions:
        extracted = extract_final_answer(completion)
        if extracted is not None:
            norm_ans = normalize_answer(str(extracted))
            if norm_ans:
                answers.append(norm_ans)

    parsed_count = len(answers)
    if not ground_truth or parsed_count < min_parsed:
        # Not enough signal (or no GT) → skip this sample for labels
        return None, parsed_count, total, 0

    correct_parsed = sum(1 for answer in answers if answer == norm_gt)
    p = correct_parsed / parsed_count
    return p, parsed_count, total, correct_parsed


def format_dataset_item(
    item: Dict[str, Any],
    min_parsed: int,
    max_correct_bin: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert a single item from completer format to process-reward format.

    Output:
        {
          "prompt": question,
          "completions": [segment_0, segment_1, ...],
          "labels": [score_0, score_1, ...]
        }

    where "completions" are contiguous segments of the full CoT, and segment
    boundaries are exactly where we have supervised prefixes (after filtering
    by min_parsed).
    """
    original_item = item["original_item"]
    samples = item.get("samples", [])

    question = original_item.get("question", "").strip()
    solution = original_item.get("solution", "")

    # Recover full CoT
    full_cot_raw = original_item.get("attempt", "") or ""
    if not full_cot_raw and samples:
        # Fallback: use longest cut_cot if attempt is missing
        full_cot_raw = max(
            (sample.get("cut_cot", "") or "" for sample in samples),
            key=len,
            default="",
        )
    full_cot = clean_cot(full_cot_raw)

    # Stats to return
    stats: Dict[str, Any] = {
        "samples_total": 0,
        "samples_with_parsed": 0,
        "samples_used": 0,  # parsed_count >= min_parsed
        "total_completions": 0,
        "parsed_answers": 0,
        "correct_answers": 0,
        "parsed_count_hist": Counter(),   # parsed_count -> count
        "count_bins_used": Counter(),     # k_correct (0..max_correct_bin) -> count
        "scores_used": [],                # per-sample score for USED samples
        "steps_used": 0,
    }

    # Map from boundary position in full_cot (end index) -> list of scores
    boundary_scores: Dict[int, List[float]] = defaultdict(list)

    for sample in samples:
        completions_list = sample.get("completions", [])
        if not completions_list:
            continue

        stats["samples_total"] += 1
        stats["total_completions"] += len(completions_list)

        value, parsed_count, total_seen, correct_parsed = calculate_value(
            completions_list, solution, min_parsed=min_parsed
        )

        stats["parsed_count_hist"][parsed_count] += 1
        stats["parsed_answers"] += parsed_count
        stats["correct_answers"] += correct_parsed

        if parsed_count > 0:
            stats["samples_with_parsed"] += 1

        if value is None:
            # Not enough parsed answers (or no GT) → ignore for labels
            continue

        # Sample is used for labels
        stats["samples_used"] += 1
        stats["scores_used"].append(value)
        k_correct = min(correct_parsed, max_correct_bin)
        stats["count_bins_used"][k_correct] += 1

        # Find where this supervised prefix ends in the full CoT
        cut_cot_raw = sample.get("cut_cot", "") or ""
        cut_cot_clean = clean_cot(cut_cot_raw)
        if not cut_cot_clean:
            continue

        # We expect cut_cot_clean to be a substring of full_cot (usually a prefix)
        pos = full_cot.find(cut_cot_clean)
        if pos == -1:
            # If not found, skip this sample (mismatch between attempt and cut_cot)
            continue

        boundary_end = pos + len(cut_cot_clean)
        boundary_scores[boundary_end].append(value)

    # Build contiguous segments of full_cot based on the sorted boundaries
    sorted_bounds = sorted(boundary_scores.keys())
    completions_steps: List[str] = []
    score_values: List[float] = []

    prev_end = 0
    for end in sorted_bounds:
        segment = full_cot[prev_end:end]
        # Strip leading/trailing whitespace so we don't start with many \n\n
        segment = segment.strip()
        if not segment:
            # If this slice is empty after stripping, skip (and keep prev_end)
            continue

        mean_val = float(sum(boundary_scores[end]) / len(boundary_scores[end]))
        completions_steps.append(segment)
        score_values.append(mean_val)
        prev_end = end

    stats["steps_used"] = len(completions_steps)

    formatted_item = {
        "prompt": question,
        "completions": completions_steps,
        "labels": score_values,
    }

    return formatted_item, stats


def convert_dataset(
    input_file: str,
    max_correct_bin: int,
    min_parsed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert the entire dataset from completer format to process-reward format.
    """
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    if not data:
        raise ValueError(f"Could not load data from {input_file}")
    total_items = len(data)
    print(f"Loaded {total_items} items from the dataset")
    print(f"Computing step-level percentage values of completion correctness "
          f"(min_parsed = {min_parsed})")

    aggregate_stats: Dict[str, Any] = {
        "items_total": total_items,
        "items_single_step_removed": 0,
        "samples_total": 0,
        "samples_with_parsed": 0,
        "samples_used": 0,
        "total_completions": 0,
        "parsed_answers": 0,
        "correct_answers": 0,
        "steps_used": 0,
        "count_bins_used": Counter(),
        "parsed_count_hist": Counter(),
        "scores_used": [],
    }

    all_formatted_items: List[Dict[str, Any]] = []

    for item in data:
        formatted_item, stats = format_dataset_item(
            item,
            min_parsed=min_parsed,
            max_correct_bin=max_correct_bin,
        )

        # Drop items with 0 or 1 supervised segment
        if len(formatted_item["completions"]) <= 1:
            aggregate_stats["items_single_step_removed"] += 1
            continue

        all_formatted_items.append(formatted_item)

        # Aggregate stats
        aggregate_stats["samples_total"] += stats["samples_total"]
        aggregate_stats["samples_with_parsed"] += stats["samples_with_parsed"]
        aggregate_stats["samples_used"] += stats["samples_used"]
        aggregate_stats["total_completions"] += stats["total_completions"]
        aggregate_stats["parsed_answers"] += stats["parsed_answers"]
        aggregate_stats["correct_answers"] += stats["correct_answers"]
        aggregate_stats["steps_used"] += stats["steps_used"]

        aggregate_stats["parsed_count_hist"].update(stats["parsed_count_hist"])
        aggregate_stats["count_bins_used"].update(stats["count_bins_used"])
        aggregate_stats["scores_used"].extend(stats["scores_used"])

    print(f"\nItems removed because they had ≤1 supervised segment: "
          f"{aggregate_stats['items_single_step_removed']} "
          f"({aggregate_stats['items_single_step_removed'] / total_items * 100:.2f}%)")

    # ---------- Sample-level statistics ----------
    print("\n--- Sample-level Statistics (after min_parsed filter) ---")
    samples_total = aggregate_stats["samples_total"]
    samples_used = aggregate_stats["samples_used"]
    total_completions = aggregate_stats["total_completions"]
    parsed_answers = aggregate_stats["parsed_answers"]
    correct_answers = aggregate_stats["correct_answers"]
    parsed_hist = aggregate_stats["parsed_count_hist"]
    count_bins_used = aggregate_stats["count_bins_used"]
    scores_used = aggregate_stats["scores_used"]

    kept_items = len(all_formatted_items)
    print(f"Total items kept: {kept_items} (out of {total_items})")
    print(f"Total samples (original, among kept items): {samples_total}")
    print(f"Total completions: {total_completions}")

    print("\n=== Parsed-answer statistics per sample ===")
    zero_parsed = parsed_hist[0]
    one_parsed = parsed_hist[1]
    ge2_parsed = sum(c for k, c in parsed_hist.items() if k >= 2)

    if samples_total > 0:
        print(f"Samples with 0 parsed answers:  {zero_parsed} "
              f"({zero_parsed / samples_total * 100:.2f}%)")
        print(f"Samples with 1 parsed answer:  {one_parsed} "
              f"({one_parsed / samples_total * 100:.2f}%)")
        print(f"Samples with ≥2 parsed answers: {ge2_parsed} "
              f"({ge2_parsed / samples_total * 100:.2f}%)")
        print(f"Average parsed answers per sample: "
              f"{parsed_answers / samples_total:.2f}")
    else:
        print("No samples to report parsed statistics on.")

    if samples_total > 0:
        print(f"\nSamples USED for labels (parsed_count ≥ {min_parsed}): "
              f"{samples_used} "
              f"({samples_used / samples_total * 100:.2f}% of all samples)")
    else:
        print("\nSamples USED for labels: 0 (no samples).")

    # Histogram for USED samples (correct counts on parsed answers)
    print("\n=== Histogram of correct counts (USED samples; parsed-only) ===")
    if samples_used > 0:
        for k in range(max_correct_bin + 1):
            c = count_bins_used[k]
            pct = (c / samples_used * 100) if samples_used else 0.0
            print(f"{k}/{max_correct_bin} correct: {c} samples ({pct:.2f}%)")
    else:
        print("No samples used after min_parsed filtering.")

    # Variance statistics for USED samples
    def summarize_scores(name: str, arr: List[float]) -> None:
        if not arr:
            print(f"{name}: no entries")
            return
        arr_np = np.array(arr)
        print(f"{name}:")
        print(f"  mean:   {arr_np.mean():.4f}")
        print(f"  median: {np.median(arr_np):.4f}")
        print(f"  std:    {arr_np.std():.4f}")
        print(f"  min:    {arr_np.min():.4f}")
        print(f"  max:    {arr_np.max():.4f}")
        frac_nonzero = np.mean(arr_np > 0)
        print(f"  fraction > 0: {frac_nonzero * 100:.2f}%")

    print("\n=== Score statistics per USED sample ===")
    summarize_scores("Parsed-only (parsed_count ≥ min_parsed)", scores_used)

    if parsed_answers > 0:
        print("\n=== Overall correctness (parsed-only view) ===")
        print(f"Parsed completions: correct = {correct_answers} / {parsed_answers} "
              f"({correct_answers / parsed_answers * 100:.2f}%)")
    else:
        print("\nNo parsed answers to compute correctness.")

    # ---------- Step-level score statistics ----------
    steps_used = aggregate_stats["steps_used"]
    print(f"\nTotal supervised steps (after filtering): {steps_used}")

    print(f"\nConverted to {len(all_formatted_items)} formatted items "
          f"with percentage score labels")
    return all_formatted_items, aggregate_stats


# =========================
# Outlier removal & stats
# =========================

def remove_outliers_by_steps(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes outliers from the dataset based on the number of steps per item.
    Uses the 1.5 * IQR rule to identify and remove items with an unusually high number of steps.
    """
    if not data:
        return []

    print("\n--- Removing Outliers based on Step Count ---")

    step_lengths = [len(item['completions']) for item in data]
    if not step_lengths:
        print("No items with completions to process for outlier removal.")
        return data

    q1, q3 = np.percentile(step_lengths, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    print(f"Step count stats for outlier detection:")
    print(f"  - Q1 (25th percentile): {q1:.1f} steps")
    print(f"  - Q3 (75th percentile): {q3:.1f} steps")
    print(f"  - IQR: {iqr:.1f} steps")
    print(f"  - Upper bound for outliers (Q3 + 1.5*IQR): {upper_bound:.2f} steps")

    original_count = len(data)
    filtered_data = [item for item in data if len(item['completions']) <= upper_bound]
    removed_count = original_count - len(filtered_data)

    if removed_count > 0:
        print(f"Removed {removed_count} outliers "
              f"(items with more than {int(upper_bound)} steps).")
    else:
        print("No outliers were detected based on the number of steps.")

    print(f"Dataset size changed from {original_count} to {len(filtered_data)} items.")
    return filtered_data


def print_label_statistics(data: List[Dict[str, Any]], dataset_name: str = "Dataset") -> None:
    """
    Print detailed statistics about the continuous labels (scaled percentage values) in the dataset.
    """
    if not data:
        print(f"{dataset_name} is empty")
        return

    print(f"\n--- {dataset_name} Statistics ---")
    total_items = len(data)
    steps_per_item = [len(item['labels']) for item in data]
    total_steps = sum(steps_per_item)
    print(f"Overall Info:")
    print(f"  - Total items: {total_items}")
    print(f"  - Total supervised steps: {total_steps}")

    steps_np = np.array(steps_per_item)
    print(f"\nSteps Per Item:")
    print(f"  - Mean: {np.mean(steps_np):.2f}")
    print(f"  - Median: {np.median(steps_np):.2f}")
    print(f"  - Std Dev: {np.std(steps_np):.2f}")
    print(f"  - Range: [{np.min(steps_np)}, {np.max(steps_np)}]")

    all_labels = [label for item in data for label in item['labels']]
    if not all_labels:
        print("\nDataset has no labels to analyze.")
        return

    labels_np = np.array(all_labels)
    p25, p75 = np.percentile(labels_np, [25, 75])

    print(f"\nLabel Values (per step, scaled [0,1]):")
    print(f"  - Mean: {np.mean(labels_np):.4f}")
    print(f"  - Median: {np.median(labels_np):.4f}")
    print(f"  - Std Dev: {np.std(labels_np):.4f}")
    print(f"  - Range: [{np.min(labels_np):.4f}, {np.max(labels_np):.4f}]")
    print(f"  - IQR (25th-75th percentile): [{p25:.4f}, {p75:.4f}]")

    avg_labels_per_item = [np.mean(item['labels']) if item['labels'] else 0 for item in data]
    avg_labels_np = np.array(avg_labels_per_item)

    print(f"\nAverage Label Value (per item):")
    print(f"  - Mean: {np.mean(avg_labels_np):.4f}")
    print(f"  - Median: {np.median(avg_labels_np):.4f}")
    print(f"  - Std Dev: {np.std(avg_labels_np):.4f}")
    print(f"  - Range: [{np.min(avg_labels_np):.4f}, {np.max(avg_labels_np):.4f}]")


# =========================
# Train/test split
# =========================

def split_dataset_without_contamination(
    formatted_data: List[Dict[str, Any]],
    train_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split the dataset ensuring no prompt contamination between train and test sets.
    """
    print("\n--- Splitting Dataset (No Contamination) ---")

    prompt_groups = defaultdict(list)
    for item in formatted_data:
        prompt_groups[item['prompt']].append(item)

    print(f"Splitting {len(formatted_data)} items from {len(prompt_groups)} unique prompts.")

    unique_prompts = list(prompt_groups.keys())
    random.seed(42)
    random.shuffle(unique_prompts)

    split_point = int(len(unique_prompts) * train_ratio)
    train_prompts = set(unique_prompts[:split_point])

    new_train, new_test = [], []
    for prompt, items in prompt_groups.items():
        if prompt in train_prompts:
            new_train.extend(items)
        else:
            new_test.extend(items)

    print(f"Split results:")
    print(f"  - Train: {len(new_train)} items ({len(train_prompts)} prompts)")
    print(f"  - Test:  {len(new_test)} items "
          f"({len(prompt_groups) - len(train_prompts)} prompts)")
    print(f"  - Train percentage: {len(new_train) / len(formatted_data) * 100:.1f}%")

    train_prompts_check = set(item['prompt'] for item in new_train)
    test_prompts_check = set(item['prompt'] for item in new_test)
    contamination = train_prompts_check.intersection(test_prompts_check)
    if contamination:
        print(f"WARNING: Found {len(contamination)} overlapping prompts "
              f"between train and test sets!")
    else:
        print("Contamination check passed: No overlapping prompts found.")

    return new_train, new_test


# =========================
# Main pipeline
# =========================

if __name__ == "__main__":
    print("\n--- Step 1: Converting Dataset Format ---")
    formatted_data, aggregate_stats = convert_dataset(
        INPUT_PATH,
        max_correct_bin=N,
        min_parsed=MIN_PARSABLE_COMPLETIONS,
    )
    if not formatted_data:
        raise SystemExit("No items were converted; aborting.")

    # Step 2: Stats on the full converted dataset
    print_label_statistics(formatted_data, "Converted Dataset (before outlier removal)")

    # Step 3: Remove outliers based on number of supervised steps
    print("\n--- Step 3: Removing Outliers ---")
    filtered_data = remove_outliers_by_steps(formatted_data)

    # Step 4: Stats on the filtered dataset
    print_label_statistics(filtered_data, "Filtered Dataset (after outlier removal)")

    time.sleep(1)

    # Step 5: Split into train / test without prompt contamination
    print("\n--- Step 5: Splitting into Train and Test ---")
    train_data, test_data = split_dataset_without_contamination(
        filtered_data,
        train_ratio=TRAIN_RATIO,
    )

    time.sleep(1)

    # Step 6: Save datasets (use OUTPUT_PREFIX directly)
    print("\n--- Step 6: Saving Datasets ---")
    train_path = f"data/mmlu/{OUTPUT_PREFIX}_train.jsonl"
    test_path = f"data/mmlu/{OUTPUT_PREFIX}_test.jsonl"

    save_dataset(train_data, train_path)
    save_dataset(test_data, test_path)

    print(f"Training set saved to: {train_path}")
    print(f"Test set saved to:     {test_path}")

    # Step 7: Show one sample JSON row (for quick sanity check)
    example = train_data[0] if train_data else (filtered_data[0] if filtered_data else None)
    if example is not None:
        print("\n--- Example converted item (truncated) ---")
        print(json.dumps(example, indent=2, ensure_ascii=False)[:4000])
    else:
        print("\nNo example to show (empty dataset).")

    print("\nPipeline complete ✅")
