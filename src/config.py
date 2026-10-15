"""
Configuration settings for jobsnap-evals evaluation runs and test suites.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Configuration settings for benchmark evaluation and threshold gating."""
    accuracy_threshold: float = Field(
        default=0.85,
        description="Minimum overall accuracy required for regression suite to pass (0.0 to 1.0)",
    )
    company_match_threshold: float = Field(
        default=0.85,
        description="Minimum company match rate required (0.0 to 1.0)",
    )
    status_match_threshold: float = Field(
        default=0.85,
        description="Minimum status match rate required (0.0 to 1.0)",
    )
    golden_dataset_path: Path = Field(
        default=Path("data/golden.jsonl"),
        description="Path to ground truth golden dataset",
    )


def get_eval_config() -> EvalConfig:
    """Load evaluation config with environment variable overrides."""
    env_acc = os.getenv("EVAL_ACCURACY_THRESHOLD")
    threshold = float(env_acc) if env_acc is not None else 0.85
    return EvalConfig(accuracy_threshold=threshold)


# Global config instance
config = get_eval_config()
