"""
Tests for EmailCase schema and data/golden.jsonl dataset validation.
Compatible with both pytest and python -m unittest.
"""

from pathlib import Path
import unittest
from pydantic import ValidationError

from src.schema import EmailCase, EmailStatus


class TestSchemaAndGoldenDataset(unittest.TestCase):
    def test_email_case_valid_instantiation(self):
        case = EmailCase(
            id="test_001",
            email_text="Thank you for applying to Acme Corp.",
            expected_is_job_email=True,
            expected_company="Acme Corp",
            expected_status="applied",
            expected_role="Backend Engineer",
            tags=["applied", "test"],
        )
        self.assertEqual(case.id, "test_001")
        self.assertEqual(case.expected_status, "applied")
        self.assertTrue(case.expected_is_job_email)

    def test_email_case_invalid_status_raises(self):
        with self.assertRaises(ValidationError):
            EmailCase(
                id="test_002",
                email_text="Invalid status test",
                expected_is_job_email=True,
                expected_status="pending_review",  # Invalid enum/literal
            )

    def test_golden_dataset_parses_and_satisfies_schema(self):
        golden_path = Path("data/golden.jsonl")
        self.assertTrue(golden_path.exists(), "data/golden.jsonl must exist")

        cases: list[EmailCase] = []
        with open(golden_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                case = EmailCase.model_validate_json(line)
                cases.append(case)

        self.assertEqual(len(cases), 10, f"Expected exactly 10 golden cases, got {len(cases)}")

        # Check for presence of required hard-case tags
        all_tags = {tag for c in cases for tag in c.tags}
        required_tags = [
            "soft_rejection",
            "newsletter",
            "two_companies",
            "forwarded",
            "quoted_printable",
            "context_contamination",
        ]
        for tag in required_tags:
            self.assertIn(tag, all_tags, f"Missing required edge case tag: {tag}")


if __name__ == "__main__":
    unittest.main()
