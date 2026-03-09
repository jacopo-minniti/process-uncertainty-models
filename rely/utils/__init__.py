"""
Utility modules for the rely package.

This package contains various utility functions and classes for data loading,
text processing, probe management, dataset analysis, and more.
"""

# Import from load.py
from .load import (
    load_dataset,
    save_dataset,
    validate_file_format,
    split_dataset
)

# Import from merge.py
from .merge import merge

# Import from show.py
from .dataset_summary import (
    show_fields,
    show_first_n,
    show_summary
)

# Import from text_utils.py
from .text_utils import (
    get_last_step_pos,
    count_tokens_after_marker,
    format_prompt,
    extract_final_answer,
    normalize_answer,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT,
    MATH_SYSTEM_PROMPT,
    prompt_pattern
)

__all__ = [
    # Load utilities
    "load_dataset",
    "save_dataset", 
    "validate_file_format",
    "split_dataset",
    
    # Merge utilities
    "merge",
    
    # Show utilities
    "show_fields",
    "show_first_n",
    "show_summary",
    
    # Text utilities
    "get_last_step_pos",
    "count_tokens_after_marker",
    "format_prompt",
    "ensure_think_ending",
    "MMLU_SYSTEM_PROMPT",
    "MATH_SYSTEM_PROMPT",
    "extract_final_answer",
    "normalize_answer",
    "prompt_pattern"
]

__version__ = "1.0.0" 