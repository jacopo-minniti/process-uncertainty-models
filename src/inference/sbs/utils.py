# rely/inference/sbs/utils.py

import argparse
import logging
from multiprocessing import Queue
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__) 

@dataclass
class SBSConfig:
    """Configuration for Step-level Beam Search"""
    strategy: str = "uniform"
    step_beam_width: int = 3
    n_total_samples: int = 5
    max_depth: Optional[int] = None
    budget: Optional[int] = None
    temperature: float = 0.6
    max_tokens: int = 256
    remove_duplicate: bool = True
    verbose: bool = True
    evaluate_n_steps: int = 1
    value_method: str = "last_step"
    uncertainty_method: str = "last_step" # For PUM
    uncertainty_temperature: float = 1.0
    # Sampling strategy (PRM-only)
    # Dual-PRM strategy (mean_var)
    dual_prm_weight: float = 0.5
    step_separator_token: str = "<extra_0>"

class SBSNode:
    """Node in the Step-level Beam Search tree"""
    def __init__(self, parent: Optional['SBSNode'] = None, text: str = "", depth: int = 0):
        self.parent = parent
        self.children: List['SBSNode'] = []
        self.text = text
        self.depth = depth
        self.full_text: str = (parent.full_text if parent else "") + text
        self.value: float = -100.0
        self.prm_score: Optional[float] = None
        self.uncertainty: float = 0.5
        self.is_terminal = False
        self.final_answer = ""

    def add_child(self, child_text: str) -> 'SBSNode':
        cleaned = child_text.lstrip('\n')
        child = SBSNode(parent=self, text=cleaned, depth=self.depth + 1)
        self.children.append(child)
        return child