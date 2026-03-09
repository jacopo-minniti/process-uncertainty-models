import os
import json
import argparse
from tqdm import tqdm

# Configuration for input and output paths
INPUT_DIR = "results/sbs/qwen2.5/math/uniform-n1-mine/max_2_8"
OUTPUT_FILE = "results/snr/qwen2.5/math/uniform-n1-mine/max_2_8/generations.jsonl"

def process_sbs_to_cots(input_dir, output_file):
    """
    Reads SBS results and converts the best-of-N solution to a JSONL file format.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Recursively find all summary.json files
    summary_files = []
    print(f"Scanning {input_dir} for summary.json files...")
    for root, _, files in os.walk(input_dir):
        if 'summary.json' in files:
            summary_files.append(os.path.join(root, 'summary.json'))
    
    if not summary_files:
        print(f"No summary.json files found in {input_dir}")
        return

    # Sort files to ensure deterministic order (and likely by question index if named q_XXXX)
    summary_files.sort()

    results = []
    print(f"Found {len(summary_files)} summary.json files. Processing...")

    for path in tqdm(summary_files, desc="Processing files"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract basic info
            question = data.get('question')
            solution = data.get('ground_truth')
            solutions = data.get('solutions', [])
            
            if not solutions:
                continue
            
            # Find the best solution (Best-of-N logic)
            # summary.json has 'solutions' which is a list of dicts.
            # We select the one with the maximum value.
            best_sol = max(solutions, key=lambda x: float(x.get('value', float('-inf'))))
            
            # 'solution_path' contains the full text (COT)
            attempt = best_sol.get('solution_path') 
            
            # Try to extract index from directory name (e.g., q_0123)
            parent_dir = os.path.basename(os.path.dirname(path))
            generations_idx = -1
            if parent_dir.startswith("q_"):
                try:
                    generations_idx = int(parent_dir.split("_")[1])
                except ValueError:
                    pass
            
            entry = {
                "generations_idx": generations_idx,
                "question": question,
                "attempt": attempt,
                "solution": solution
            }
            results.append(entry)
            
        except Exception as e:
            print(f"Warning: Failed to process file {path}: {e}")

    # Write to jsonl
    print(f"Writing {len(results)} items to {output_file}")
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SBS results to COTs JSONL format.')
    
    parser.add_argument('--input_dir', type=str, default=None, help=f'Directory containing SBS results (default uses script variable: {INPUT_DIR})')
    parser.add_argument('--output_file', type=str, default=None, help=f'Output JSONL file path (default uses script variable: {OUTPUT_FILE})')
    
    args = parser.parse_args()
    
    # Prioritize arguments, fall back to script variables
    input_dir_to_use = args.input_dir if args.input_dir else INPUT_DIR
    output_file_to_use = args.output_file if args.output_file else OUTPUT_FILE
    
    if input_dir_to_use == "/path/to/sbs/results" and not os.path.exists(input_dir_to_use):
         print(f"Please provide a valid --input_dir or edit the INPUT_DIR variable in the script.")
    else:
        process_sbs_to_cots(input_dir_to_use, output_file_to_use)
