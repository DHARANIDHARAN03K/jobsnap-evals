"""
jobsnap-evals: Evaluation framework and benchmark runner.
"""

from .schema import EmailCase, EmailStatus, ModelOutput
from .runner import run_evaluation, TraceRecord
from .checks import (
    DeterministicResult,
    JudgeResult,
    normalize_company,
    check_company_match,
    check_status_match,
    validate_model_output,
    evaluate_deterministic,
    judge_hallucination,
)

__all__ = [
    "EmailCase",
    "EmailStatus",
    "ModelOutput",
    "run_evaluation",
    "TraceRecord",
    "DeterministicResult",
    "JudgeResult",
    "normalize_company",
    "check_company_match",
    "check_status_match",
    "validate_model_output",
    "evaluate_deterministic",
    "judge_hallucination",
]
