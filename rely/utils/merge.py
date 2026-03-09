import random
from typing import List, Dict, Any, Union
from pathlib import Path
from .load import load_dataset, save_dataset


def merge(input_files: List[str], output_filename: str = None) -> str:
    """
    Merge a list of input files (either .pt or .jsonl) that contain lists of dictionaries.
    
    Args:
        input_files: List of file paths to merge
        output_filename: Output filename (optional, will auto-generate if not provided)
    
    Returns:
        str: The output filename that was created
    
    Raises:
        ValueError: If files don't contain lists of dictionaries or have incompatible formats
        FileNotFoundError: If any input file doesn't exist
    """
    if not input_files:
        raise ValueError("Input files list cannot be empty")
    
    merged_data = []
    
    for file_path in input_files:
        # Use centralized loading function
        data = load_dataset(file_path)
        merged_data.extend(data)
    
    # Generate output filename if not provided
    if output_filename is None:
        base_name = "merged_data"
        if all(Path(f).suffix.lower() == '.pt' for f in input_files):
            output_filename = f"{base_name}.pt"
        else:
            output_filename = f"{base_name}.jsonl"
    
    # Use centralized saving function
    random.shuffle(merged_data)
    save_dataset(merged_data, output_filename)
    
    print(f"Successfully merged {len(input_files)} files into {output_filename}")
    print(f"Total items: {len(merged_data)}")
    
    return output_filename
