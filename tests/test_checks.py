"""
Tests for src/checks.py: Deterministic checks against ModelOutput.
"""

import unittest
from src.schema import EmailCase, ModelOutput
from src.checks import (
    normalize_company,
    check_company_match,
    check_status_match,
    validate_model_output,
    evaluate_deterministic,
    _heuristic_judge,
)


class TestChecks(unittest.TestCase):
    def test_normalize_company(self):
        self.assertEqual(normalize_company("Acme Corp."), "acme")
        self.assertEqual(normalize_company("Google, LLC"), "google")
        self.assertEqual(normalize_company("Stripe Technologies"), "stripe")
        self.assertEqual(normalize_company(None), "")

    def test_check_company_match(self):
        matched, _ = check_company_match("Acme Analytics Inc.", "Acme Analytics")
        self.assertTrue(matched)

        matched, _ = check_company_match("Google", "Microsoft")
        self.assertFalse(matched)

        matched, _ = check_company_match(None, None)
        self.assertTrue(matched)

    def test_check_status_match(self):
        matched, _ = check_status_match("Interview", "Interview")
        self.assertTrue(matched)

        matched, _ = check_status_match("Offer", "Rejected")
        self.assertFalse(matched)

    def test_validate_model_output(self):
        valid_dict = {
            "corrected_company_name": "Figma",
            "status": "Interview",
            "source_of_company": "body",
            "job_title": "Designer",
            "reasoning": "Application update",
        }
        ok, out, _ = validate_model_output(valid_dict)
        self.assertTrue(ok)
        self.assertIsInstance(out, ModelOutput)

        invalid_dict = {
            "status": "Interview"
            # Missing corrected_company_name, source_of_company
        }
        ok, out, _ = validate_model_output(invalid_dict)
        self.assertFalse(ok)

    def test_evaluate_deterministic_pass(self):
        case = EmailCase(
            id="c1",
            subject="Interview with Datadog",
            body="We invite you to interview for Backend Engineer at Datadog.",
            sender="talent@datadog.com",
            sender_domain="datadog.com",
            expected_company="Datadog",
            expected_status="Interview",
            expected_title="Backend Engineer",
        )
        model_out = {
            "corrected_company_name": "Datadog Inc.",
            "status": "Interview",
            "source_of_company": "body",
            "job_title": "Backend Engineer",
            "reasoning": "Interview invited",
        }
        res = evaluate_deterministic(case, model_out)
        self.assertTrue(res.all_passed)
        self.assertTrue(res.company_match)
        self.assertTrue(res.status_match)

    def test_heuristic_judge_hallucination_detection(self):
        body = "Hi Chris, we received your application for Backend Engineer at Datadog."

        # Grounded output -> PASS
        good_out = {
            "corrected_company_name": "Datadog",
            "job_title": "Backend Engineer",
            "status": "Applied",
        }
        j_pass = _heuristic_judge(body, good_out)
        self.assertEqual(j_pass.verdict, "PASS")

        # Hallucinated company -> FAIL
        bad_out = {
            "corrected_company_name": "Netflix",
            "job_title": "Backend Engineer",
            "status": "Applied",
        }
        j_fail = _heuristic_judge(body, bad_out)
        self.assertEqual(j_fail.verdict, "FAIL")
        self.assertIn("Netflix", j_fail.reason)


if __name__ == "__main__":
    unittest.main()
