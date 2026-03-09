"""
Inference modules for the rely package.

This package contains inference-related functionality including UATS (Uncertainty-guided Approximate Tree Search)
and Budget Forcing inference strategies.
"""

__all__ = []

try:
    from .majority_voting import (
        SelfConsistencyConfig,
        SelfConsistencyResult,
        SelfConsistencyInference,
        run_self_consistency,
        save_self_consistency_result,
    )

    __all__ += [
        "SelfConsistencyConfig",
        "SelfConsistencyResult",
        "SelfConsistencyInference",
        "run_self_consistency",
        "save_self_consistency_result",
    ]
except ModuleNotFoundError as e:
    # `rely.inference` should remain importable without optional runtime deps (e.g., vLLM).
    if e.name != "vllm":
        raise

__version__ = "1.0.0" 
