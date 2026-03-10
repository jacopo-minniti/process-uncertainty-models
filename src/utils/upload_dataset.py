#!/usr/bin/env python3
"""
Script to upload v2 dataset files to Hugging Face Hub without overwriting existing data.
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo
import argparse

def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def upload_as_separate_files(repo_id, files_to_upload, token=None):
    """Upload files directly to the repository root."""
    api = HfApi(token=token)
    
    # Try to create the repository if it doesn't exist
    try:
        print(f"Checking if repository {repo_id} exists...")
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"✅ Repository {repo_id} is ready")
    except Exception as e:
        print(f"⚠️  Repository creation info: {e}")
    
    for file_path in files_to_upload:
        if Path(file_path).exists():
            # Extract just the filename for upload (without directory path)
            file_name = Path(file_path).name
            print(f"Uploading {file_path} as {file_name}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"✅ Uploaded {file_name}")
        else:
            print(f"⚠️  File {file_path} not found")

def main():
    parser = argparse.ArgumentParser(description="Upload dataset files to Hugging Face Hub")
    parser.add_argument("repo_id", help="Repository ID (username/dataset-name)")
    parser.add_argument("--files", nargs="+", 
                       default=["data/math_variance_v2_train.jsonl", "data/math_variance_v2_test.jsonl"],
                       help="Files to upload with their paths")
    parser.add_argument("--token", help="Hugging Face token (optional if logged in)")
    
    args = parser.parse_args()
    
    print(f"Repository: {args.repo_id}")
    print(f"Files to upload: {args.files}")
    upload_as_separate_files(args.repo_id, args.files, args.token)

if __name__ == "__main__":
    # run_script rely/utils/upload_dataset.py jacopo-minniti/MMLU-PUM-qwen2.5-1.5B --files data/mmlu/qwen2.5_value_train.jsonl data/mmlu/qwen2.5_value_test.jsonl
    main()
