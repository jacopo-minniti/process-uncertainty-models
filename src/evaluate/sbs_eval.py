import os
import json
import numpy as np
import argparse

def process_json_file(path):
    """
    Processes a single JSON file by reading pre-calculated metrics from summary.json.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        gt_usable = bool(data.get('gt_usable', False))

        # Majority vote
        majority_vote = data.get('majority_vote')
        maj_applicable = gt_usable and majority_vote not in (None, "", "TIE")
        is_maj_correct = bool(data.get('is_majority_correct', False))

        # Best-of-N
        bon_answer = data.get('best_of_n_answer')
        bon_applicable = gt_usable and bon_answer not in (None, "")
        is_bon_correct = bool(data.get('is_best_of_n_correct', False))

        try:
            total_tokens = int(float(data.get('total_tokens', 0)))
        except (TypeError, ValueError):
            total_tokens = 0

        return is_maj_correct, is_bon_correct, bon_applicable, gt_usable, maj_applicable, total_tokens

    except Exception as e:
        print(f"Warning: Failed to process file {path}: {e}")
        return False, False, False, False, False, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files and compute evaluation statistics.')
    parser.add_argument('input_dir', help='Directory containing JSON files to process')
    parser.add_argument('--skip_invalid', action='store_true', help='Skip invalid samples instead of treating them as incorrect')
    parser.add_argument('--show_invalid', action='store_true', help='Print paths of invalid samples')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Error: The directory does not exist: '{input_dir}'")
        exit(1)

    # Recursively find all .json files
    json_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON files found in '{input_dir}' or its subfolders.")
        exit(1)

    # --- Initialize counters ---
    total_files = 0
    
    # Majority Vote
    maj_correct_count = 0
    
    # Best of N
    bon_correct_count = 0
    
    # Valid/Usable items (GT exists and is usable)
    valid_items_count = 0
    
    # Applicability counts
    bon_valid_items_count = 0
    maj_valid_items_count = 0
    stat_valid_items_count = 0

    token_counts = []
    unusable_gt_count = 0
    invalid_paths = []

    for path in sorted(json_files):
        is_maj, is_bon, bon_app, gt_usable, maj_app, tokens = process_json_file(path)
        
        total_files += 1
        token_counts.append(tokens)
        
        if not gt_usable or (not maj_app and not bon_app):
            unusable_gt_count += 1
            if args.show_invalid:
                invalid_paths.append(path)
            continue

        valid_items_count += 1
        if maj_app:
            maj_valid_items_count += 1
        if bon_app:
            bon_valid_items_count += 1
        if maj_app or bon_app:
            stat_valid_items_count += 1
        
        if is_maj:
            maj_correct_count += 1
        if bon_app and is_bon:
            bon_correct_count += 1

    # Calculate statistics
    
    # Metrics
    # Denominator depends on --skip_invalid flag
    maj_den = maj_valid_items_count if args.skip_invalid else total_files
    bon_den = bon_valid_items_count if args.skip_invalid else total_files

    maj_acc = (maj_correct_count / maj_den) if maj_den > 0 else 0.0
    bon_acc = (bon_correct_count / bon_den) if bon_den > 0 else 0.0
    
    mean_tokens = np.mean(token_counts) if token_counts else 0.0

    # Extract B1 and B3 from directory name
    dir_name = os.path.basename(input_dir.rstrip('/'))
    b1, b3 = 0, 0
    if "max_" in dir_name:
        parts = dir_name.split("max_")[1].split("_")
        if len(parts) >= 2:
            try:
                b1 = int(parts[0])
                b3 = int(parts[1])
            except ValueError:
                pass

    # Only keep requested fields
    valid_items_report = stat_valid_items_count

    result = {
        "B1": b1,
        "B3": b3,
        "tokens_generated": round(mean_tokens),
        "total_items": total_files,
        "valid_items": valid_items_report,
        "bon": round(bon_acc, 4),
        "maj": round(maj_acc, 4)
    }

    if args.show_invalid and invalid_paths:
        print("Invalid samples:")
        for p in invalid_paths:
            print(p)

    print(json.dumps(result))
