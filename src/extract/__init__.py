"""
Extract module for extracting model activations and creating fork points.

This module contains functions for extracting model activations and creating
fork points based on entropy thresholds.
"""

from .fork import create_forks_from_dataset
# Note: activations.py is meant to be run as a script, so we don't import it here
# from .activations import extract_activations

__all__ = [
    'create_forks_from_dataset',
    # 'extract_activations',
] 