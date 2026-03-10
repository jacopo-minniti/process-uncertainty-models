"""
Score module for calculating various metrics from model outputs.

This module contains functions for calculating entropy scores, semantic isotropy,
and other metrics from model completions and activations.
"""

from .score_metrics import (
    score,
    calculate_semantic_isotropy,
    calculate_entropy_from_completions
)

__all__ = [
    'score',
    'calculate_semantic_isotropy',
    'calculate_entropy_from_completions'
] 