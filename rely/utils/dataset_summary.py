import os
import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from .load import load_dataset


def show_fields(file_path: str, max_items: int = 10) -> None:
    """
    Print the fields (keys) present in a dataset and their data types.
    
    Args:
        file_path: Path to the dataset file
        max_items: Maximum number of items to check for field analysis
    """
    try:
        data = load_dataset(file_path)
        
        if not data:
            print(f"Dataset {file_path} is empty")
            return
        
        print(f"Dataset: {file_path}")
        print(f"Total items: {len(data)}")
        print(f"Analyzing fields from first {min(max_items, len(data))} items...")
        print("-" * 50)
        
        # Analyze fields from first max_items
        field_info = {}
        items_to_check = min(max_items, len(data))
        
        for i in range(items_to_check):
            item = data[i]
            for key, value in item.items():
                if key not in field_info:
                    field_info[key] = {
                        'types': set(),
                        'sample_values': [],
                        'count': 0
                    }
                
                field_info[key]['types'].add(type(value).__name__)
                field_info[key]['count'] += 1
                
                # Store sample values (up to 3)
                if len(field_info[key]['sample_values']) < 3:
                    # Truncate long values for display
                    if isinstance(value, str) and len(value) > 50:
                        sample_val = value[:50] + "..."
                    elif isinstance(value, (list, tuple)) and len(value) > 5:
                        sample_val = str(value[:5]) + "..."
                    else:
                        sample_val = str(value)
                    field_info[key]['sample_values'].append(sample_val)
        
        # Print field information
        for field_name, info in sorted(field_info.items()):
            types_str = ", ".join(sorted(info['types']))
            presence = f"{info['count']}/{items_to_check} items"
            samples = "; ".join(info['sample_values'])
            
            print(f"Field: {field_name}")
            print(f"  Types: {types_str}")
            print(f"  Presence: {presence}")
            print(f"  Sample values: {samples}")
            print()
            
    except Exception as e:
        print(f"Error analyzing fields: {e}")


def show_first_n(file_path: str, n: int = 5, show_all_fields: bool = False) -> None:
    """
    Print the first n elements of a dataset.
    
    Args:
        file_path: Path to the dataset file
        n: Number of elements to show
        show_all_fields: If True, show all fields. If False, show only first few fields
    """
    try:
        data = load_dataset(file_path)
        
        if not data:
            print(f"Dataset {file_path} is empty")
            return
        
        print(f"Dataset: {file_path}")
        print(f"Total items: {len(data)}")
        print(f"Showing first {min(n, len(data))} items:")
        print("=" * 80)
        
        items_to_show = min(n, len(data))
        
        for i in range(items_to_show):
            item = data[i]
            print(f"\nItem {i + 1}:")
            print("-" * 40)
            
            # Get all fields or just first few
            fields = list(item.keys())
            if not show_all_fields and len(fields) > 5:
                fields = fields[:5]
                print(f"Showing first 5 fields (out of {len(item.keys())} total fields)")
            
            for field_name in fields:
                value = item[field_name]
                
                # Format the value for display
                if isinstance(value, str):
                    if len(value) > 100:
                        display_value = value[:100] + "..."
                    else:
                        display_value = value
                elif isinstance(value, (list, tuple)):
                    if len(value) > 10:
                        display_value = str(value[:10]) + f"... (total: {len(value)} items)"
                    else:
                        display_value = str(value)
                elif isinstance(value, dict):
                    if len(value) > 5:
                        display_value = str(dict(list(value.items())[:5])) + f"... (total: {len(value)} keys)"
                    else:
                        display_value = str(value)
                else:
                    display_value = str(value)
                
                print(f"  {field_name}: {display_value}")
            
            if not show_all_fields and len(item.keys()) > 5:
                remaining = len(item.keys()) - 5
                print(f"  ... and {remaining} more fields")
        
        if len(data) > n:
            print(f"\n... and {len(data) - n} more items")
            
    except Exception as e:
        print(f"Error showing first {n} items: {e}")


def show_summary(file_path: str) -> None:
    """
    Print a summary of the dataset including basic statistics.
    
    Args:
        file_path: Path to the dataset file
    """
    try:
        data = load_dataset(file_path)
        
        if not data:
            print(f"Dataset {file_path} is empty")
            return
        
        print(f"Dataset Summary: {file_path}")
        print("=" * 50)
        print(f"Total items: {len(data)}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Get all unique fields
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        print(f"Total unique fields: {len(all_fields)}")
        print(f"Fields: {', '.join(sorted(all_fields))}")
        
        # Show field presence statistics
        print("\nField presence statistics:")
        field_counts = {}
        for item in data:
            for field in item.keys():
                field_counts[field] = field_counts.get(field, 0) + 1
        
        for field, count in sorted(field_counts.items()):
            percentage = (count / len(data)) * 100
            print(f"  {field}: {count}/{len(data)} items ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error showing summary: {e}")


def main():
    """
    Command-line interface for the show utilities.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset inspection utilities")
    parser.add_argument("file_path", help="Path to the dataset file (.pt or .jsonl)")
    parser.add_argument("--action", choices=["fields", "first", "summary"], 
                       default="fields", help="Action to perform")
    parser.add_argument("--n", type=int, default=5, 
                       help="Number of items to show (for 'first' action)")
    parser.add_argument("--max-items", type=int, default=10,
                       help="Maximum items to analyze for field detection")
    parser.add_argument("--all-fields", action="store_true",
                       help="Show all fields (for 'first' action)")
    
    args = parser.parse_args()
    
    if args.action == "fields":
        show_fields(args.file_path, args.max_items)
    elif args.action == "first":
        show_first_n(args.file_path, args.n, args.all_fields)
    elif args.action == "summary":
        show_summary(args.file_path)


if __name__ == "__main__":
    main()
