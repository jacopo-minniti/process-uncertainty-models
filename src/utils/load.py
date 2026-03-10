import os
import random
import json
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def load_dataset(file_path: Union[str, Path], *, subset: Optional[str] = None, split: Optional[str] = None, return_hf: bool = False, **hf_kwargs) -> Union[List[Dict[str, Any]], Any]:
    """
    Load a dataset from either .pt, .jsonl file, or Hugging Face hub.
    
    Args:
        file_path: Path to the dataset file or Hugging Face dataset name (str)
        subset: (Optional) Subset name for Hugging Face datasets
        split: (Optional) Split name for Hugging Face datasets (e.g., 'train', 'test')
        **hf_kwargs: Additional keyword arguments for Hugging Face datasets.load_dataset
    
    Returns:
        List[Dict[str, Any]]: The loaded dataset as a list of dictionaries
    
    Raises:
        FileNotFoundError: If the file doesn't exist (for local files)
        ValueError: If the file format is unsupported or data is invalid
    """
    file_path = Path(file_path) if not isinstance(file_path, str) or os.path.exists(file_path) else file_path
    
    # If file_path is a local file
    if isinstance(file_path, Path) and file_path.exists():
        file_ext = file_path.suffix.lower()
        if file_ext == '.pt':
            if torch is None:
                raise ImportError("Loading '.pt' datasets requires PyTorch (`torch`).")
            try:
                data = torch.load(file_path, map_location='cpu', weights_only=False)
            except Exception as e:
                raise ValueError(f"Error loading PyTorch file {file_path}: {e}")
        elif file_ext == '.jsonl':
            try:
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Error loading JSONL file {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Only .pt and .jsonl are supported")
    else:
        # Assume Hugging Face dataset
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError("datasets library is required to load Hugging Face datasets.")
        ds = hf_load_dataset(file_path, name=subset, split=split, **hf_kwargs)
        # Convert to list of dicts
        data = [dict(x) for x in ds]
    # Validate that data is a list of dictionaries
    if not isinstance(data, list):
        raise ValueError(f"File {file_path} does not contain a list. Found: {type(data)}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"File {file_path} contains non-dictionary item at index {i}: {type(item)}")
    return data


def save_dataset(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save a dataset to either .pt or .jsonl file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the dataset
    
    Raises:
        ValueError: If the file format is unsupported
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pt':
        # Save as PyTorch file
        if torch is None:
            raise ImportError("Saving '.pt' datasets requires PyTorch (`torch`).")
        torch.save(data, file_path)
    elif file_ext == '.jsonl':
        # Save as JSONL file
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .pt and .jsonl are supported")


def validate_file_format(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file has a supported format (.pt or .jsonl).
    
    Args:
        file_path: Path to the file to validate
    
    Returns:
        bool: True if format is supported, False otherwise
    """
    file_ext = Path(file_path).suffix.lower()
    return file_ext in ['.pt', '.jsonl']


def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.85) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split a dataset into training and validation sets.
    
    Args:
        data: List of dictionaries representing the dataset
        train_ratio: Proportion of data to use for training (default is 0.85)
    
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Training and validation datasets
    """
    if not (0 < train_ratio < 1):   
        raise ValueError("train_ratio must be between 0 and 1")
    
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
