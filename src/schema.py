"""
src/schema.py: Pydantic schemas for jobsnap-evals benchmark dataset and model outputs.
Source of truth aligned with JobSnap Lambda Intelligence Engine.
"""

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class EmailStatus(str, Enum):
    """
    Allowed status values (exact casing from JobSnap Lambda).
    """
    APPLIED = "Applied"
    SCREENING = "Screening"
    ASSESSMENT = "Assessment"
    INTERVIEW = "Interview"
    IN_REVIEW = "In Review"
    REJECTED = "Rejected"
    OFFER = "Offer"
    IGNORE = "Ignore"
    UNCERTAIN = "Uncertain"


AllowedStatus = Literal[
    "Applied",
    "Screening",
    "Assessment",
    "Interview",
    "In Review",
    "Rejected",
    "Offer",
    "Ignore",
    "Uncertain",
]


class EmailCase(BaseModel):
    """
    Golden benchmark test case schema representing an email and its evaluation ground-truth.
    """
    id: str = Field(description="Unique identifier for the test case")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body text")
    sender: str = Field(description="Full sender address (e.g., 'Recruiting Team <talent@company.com>')")
    sender_domain: str = Field(description="Extracted domain of the sender (e.g., 'company.com')")
    existing_jobs_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical tracked jobs passed to the prompt to test for context contamination",
    )
    expected_company: str | None = Field(
        default=None,
        description="Ground-truth expected company name",
    )
    expected_status: AllowedStatus = Field(
        description="Ground-truth status (exact case: Applied, Screening, Assessment, Interview, In Review, Rejected, Offer, Ignore, Uncertain)",
    )
    expected_title: str | None = Field(
        default=None,
        description="Optional expected job title/role",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization and edge case tags (e.g., soft_rejection, newsletter, two_companies)",
    )

    class Config:
        use_enum_values = True


class ModelOutput(BaseModel):
    """
    Schema for classification outputs emitted by JobSnap Lambda LLM providers.
    """
    reasoning: str | None = Field(default=None, description="Model explanation for the classification")
    source_of_company: str | None = Field(
        default=None,
        description="Source where company was found: subject, body, sender_domain, context_rows, inferred",
    )
    company_quote: str | None = Field(default=None, description="Exact substring from email")
    is_existing_job: bool | None = Field(default=False)
    matched_job_id: str | None = Field(default=None)
    corrected_company_name: str | None = Field(default=None, description="Normalized company name")
    job_title: str | None = Field(default=None, description="Extracted job role/title")
    status: str = Field(description="One of the 9 allowed statuses")
    interview_date: str | None = Field(default=None)
    deadline: str | None = Field(default=None)
    original_email_date: str | None = Field(default=None)
    body_summary: str | None = Field(default=None)
    ai_model_used: str | None = Field(default=None)
