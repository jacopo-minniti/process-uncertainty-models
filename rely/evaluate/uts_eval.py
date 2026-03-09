import os
import json
import numpy as np
from rely.utils import normalize_answer
import argparse

def process_json_file(path):
    """
    Processes a single JSON file to check for correctness.
    Compares 'final_answer' with 'ground_truth' or uses the 'accuracy' field.
    
    Returns:
        A tuple containing:
        - is_correct (bool or None): True if correct, False if incorrect, None if should be skipped.
        - total_tokens (int): The total tokens from the file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get('ground_truth')
        final_answer = data.get('final_answer')
        accuracy = data.get('accuracy')
        
        # Get total tokens
        total_tokens = int(data.get('total_tokens', 0))

        # Check correctness
        is_correct = False
        
        # First, try to use the accuracy field if available
        if accuracy is not None:
            if isinstance(accuracy, str):
                is_correct = accuracy.lower() == "correct"
            elif isinstance(accuracy, bool):
                is_correct = accuracy
        # If no accuracy field, compare final_answer with ground_truth
        elif final_answer is not None and ground_truth is not None:
            normalized_answer = normalize_answer(str(final_answer))
            normalized_gt = normalize_answer(str(ground_truth))
            if normalized_answer == normalized_gt and normalized_gt != "":
                is_correct = True
        else:
            # If neither accuracy field nor both answers are available, skip this sample
            return None, total_tokens

        return is_correct, total_tokens

    except Exception as e:
        # For any file reading/parsing error, print warning and skip file
        print(f"Warning: Failed to process file {path}: {e}")
        return None, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files and compute evaluation statistics.')
    parser.add_argument('input_dir', nargs='?', default='results/uts-.35-3', help='Directory containing JSON files to process')
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
    correct_count = 0
    # List to store all token counts for percentile calculation
    token_counts = []

    for path in sorted(json_files):
        is_correct, total_tokens = process_json_file(path)
        
        # Skip entries that should not be considered (no valid solutions)
        if is_correct is None:
            continue
            
        total_files += 1
        
        # Accumulate correctness stats
        if is_correct:
            correct_count += 1
        
        # Store token count for percentile calculation
        token_counts.append(total_tokens)

    # Calculate statistics
    percent_correct = (correct_count / total_files) if total_files > 0 else 0.0
    
    mean_tokens = np.mean(token_counts) if token_counts else 0.0
    percentile_95_tokens = np.percentile(token_counts, 95) if token_counts else 0.0

    # Print the number of samples evaluated
    print(f"Number of samples evaluated: {total_files}")

    # extract threshold and retries from filename
    uncertainty_threshold, retries = input_dir.split(".")[-1][:2], input_dir.split("-")[-1]
    
    # Print results in dictionary format
    result = {
        "Uncertainty Threshold": float(f"0.{uncertainty_threshold}"), 
        "N. Retries": int(retries), 
        "Mean Tokens": round(mean_tokens),
        "Mean Accuracy": round(percent_correct, 4)
    }
    print(json.dumps(result))