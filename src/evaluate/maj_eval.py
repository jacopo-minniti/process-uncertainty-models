import json
import os
import glob
from pathlib import Path
from tqdm import tqdm
from rely.utils import extract_final_answer, normalize_answer, load_dataset

# Configuration
RESULTS_DIR = "results/maj/r1/mmlu/n_1"  # Update this if your n=1 run is elsewhere
DATA_PATH = "data/mmlu/inference_test.jsonl"

def main():
    # 1. Load Ground Truths
    print(f"Loading ground truths from {DATA_PATH}...")
    dataset = load_dataset(DATA_PATH)
    # Create a map of index -> solution (ground truth)
    # Assuming the dataset order matches the q_{index} folder numbering
    ground_truths = {i: item["solution"] for i, item in enumerate(dataset)}

    # 2. Find all summary files
    # We look for results/maj/.../q_XXXX/summary.json
    pattern = os.path.join(RESULTS_DIR, "q_*", "summary.json")
    summary_files = glob.glob(pattern)
    
    print(f"Found {len(summary_files)} summary files. Processing...")

    correct_count = 0
    total_processed = 0
    extraction_failures = 0

    # Sort files to process in order (q_0000, q_0001...)
    summary_files.sort()

    for sum_file in tqdm(summary_files):
        try:
            # Parse index from folder name (e.g., .../q_0050/summary.json -> 50)
            folder_name = os.path.basename(os.path.dirname(sum_file))
            idx = int(folder_name.split("_")[1])

            # Load the summary json
            with open(sum_file, 'r') as f:
                data = json.load(f)

            # --- KEY LOGIC REQUESTED ---
            # 1. Get the first solution's path (text)
            # We use [0] because n=1
            if not data["solutions"]:
                # This might happen if the run crashed for this specific sample
                continue

            generated_text_path = data["solutions"][0]["solution_path"]

            # 2. Re-run extract_final_answer explicitly
            extracted = extract_final_answer(generated_text_path)

            # 3. Get Ground Truth
            gt_raw = ground_truths.get(idx)
            if gt_raw is None:
                print(f"Warning: No ground truth found for index {idx}")
                continue

            # 4. Normalize both
            norm_extracted = normalize_answer(extracted) if extracted else "[Not Found]"
            norm_gt = normalize_answer(gt_raw)

            # 5. Compare
            is_correct = (norm_extracted == norm_gt)
            
            if is_correct:
                correct_count += 1
            
            if extracted is None:
                extraction_failures += 1

            total_processed += 1

            # --- DIAGNOSTIC PRINTING ---
            # Print only errors or weird extraction cases to help you debug
            if not is_correct:
                # Optional: Un-comment this to see EVERY mismatch
                # print(f"\n[MISS] Q_{idx:04d}")
                # print(f"  GT Raw: {gt_raw}  -> Norm: {norm_gt}")
                # print(f"  Extracted: {extracted} -> Norm: {norm_extracted}")
                
                # Special check: Did extraction fail completely?
                if extracted is None:
                     print(f"\n[EXTRACTION FAILED] Q_{idx:04d}")
                     print(f"  Model output tail (last 200 chars): ...{generated_text_path[-200:]!r}")

        except Exception as e:
            print(f"Error processing {sum_file}: {e}")

    # Final Stats
    print("-" * 30)
    print(f"Total Processed: {total_processed}")
    print(f"Extraction Failures (None): {extraction_failures}")
    print(f"Correct Matches: {correct_count}")
    if total_processed > 0:
        print(f"Accuracy: {correct_count / total_processed:.2%}")

if __name__ == "__main__":
    main()