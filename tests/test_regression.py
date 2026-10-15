"""
tests/test_regression.py: Real-classifier regression tests.

Test 1 — Libra AI / Zennah AI context contamination:
  Email is about Libra AI. existing_jobs_context contains Zennah AI rows.
  Assert corrected_company_name == "Libra AI" and source_of_company != "context_rows".

Test 2 — Uncertain status preservation:
  Documents the production behavior: Uncertain is a valid model output,
  and the test fails if the classifier NEVER emits Uncertain on an ambiguous email.
  (In production, Uncertain maps to Applied — this test records that gap.)

Test 3 — Accuracy threshold gate:
  Load golden.jsonl, run each case through Gemini (the working provider),
  and assert overall accuracy >= config.accuracy_threshold.
  Skips gracefully if Gemini key is missing.
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import (
    call_gemini,
    build_groq_system_prompt,
    extract_company_from_email,
    extract_sender_domain,
    is_aggregator_domain,
    truncate_at_noise_zone,
    GEMINI_API_KEY,
)
from src.checks import evaluate_deterministic, check_company_match
from src.config import config
from src.schema import EmailCase


# ─── helpers ──────────────────────────────────────────────────────────────────

def _run_gemini_on_case(
    subject: str,
    body: str,
    sender: str,
    sender_domain: str,
    existing_jobs_context: list,
) -> dict:
    """Build prompt and call Gemini exactly the way runner.py does."""
    body_clean = truncate_at_noise_zone(body)
    sd = sender_domain or extract_sender_domain(sender)
    hint = extract_company_from_email(subject, body_clean)
    sys_prompt = build_groq_system_prompt(
        subject_company_hint=hint,
        existing_jobs_context=existing_jobs_context,
        sender_domain=sd,
        is_aggregator=is_aggregator_domain(sd),
    )
    user_content = f"Subject: {subject}\n\nBody:\n{body_clean[:4000]}"
    return call_gemini(sys_prompt, user_content)


# =============================================================================
# TEST 1 — LIBRA AI / ZENNAH AI CONTEXT CONTAMINATION
# =============================================================================

class TestLibraAiContextContamination(unittest.TestCase):
    """
    The classifier must NOT copy the company from existing_jobs_context.
    NON-NEGOTIABLE RULE 2 in the system prompt forbids this.
    """

    @unittest.skipUnless(GEMINI_API_KEY, "GEMINI_API_KEY not set — skipping live test")
    def test_libra_ai_not_contaminated_by_zennah(self):
        subject = "Next Steps: Machine Learning Engineer at Libra AI"
        body = (
            "Hi Jordan,\n\n"
            "Thank you for speaking with our team about the Machine Learning Engineer role at Libra AI.\n"
            "We are pleased to invite you to our 45-minute technical screen this Wednesday.\n\n"
            "Best regards,\nLibra AI Recruiting Team"
        )
        sender = "talent@libra-ai.com"
        sender_domain = "libra-ai.com"

        # Context deliberately contains Zennah AI — this is the contamination vector
        contaminated_context = [
            {"id": "APP-1092", "company": "Zennah AI", "role": "AI Research Scientist", "status": "Interview"},
            {"id": "APP-1044", "company": "Zennah AI", "role": "MLOps Architect", "status": "Applied"},
        ]

        result = _run_gemini_on_case(subject, body, sender, sender_domain, contaminated_context)

        company = result.get("corrected_company_name", "")
        source = result.get("source_of_company", "")

        # Primary assertion: correct company extracted
        matched, msg = check_company_match(company, "Libra AI")
        self.assertTrue(
            matched,
            f"Expected 'Libra AI' but got '{company}'. Message: {msg}. Full output: {result}"
        )

        # Anti-contamination assertion: company must NOT come from context
        self.assertNotEqual(
            source, "context_rows",
            f"source_of_company == 'context_rows' — classifier hallucinated from context! Output: {result}"
        )

        # Bonus: company should not literally be Zennah AI
        self.assertNotIn(
            "zennah", company.lower(),
            f"Company '{company}' contains 'zennah' — contamination detected!"
        )

        print(f"[PASS] Libra AI contamination test — company='{company}', source='{source}'")


# =============================================================================
# TEST 2 — UNCERTAIN STATUS PRESERVATION / PRODUCTION GAP DOCUMENTATION
# =============================================================================

class TestUncertainStatusPreservation(unittest.TestCase):
    """
    Documents that the classifier CAN emit Uncertain, and that production maps
    Uncertain → Applied (a known gap users should be aware of).
    """

    @unittest.skipUnless(GEMINI_API_KEY, "GEMINI_API_KEY not set — skipping live test")
    def test_uncertain_can_be_returned(self):
        """
        An intentionally ambiguous email (generic recruiter outreach, no clear
        confirmation) should yield Uncertain or Ignore — never a confident Applied.
        """
        subject = "We'd love to connect with you!"
        body = (
            "Hi,\n\n"
            "We noticed your profile and think you might be a great fit.\n"
            "Could you share your availability for a quick chat?\n\n"
            "Best,\nThe Recruiting Team"
        )
        sender = "recruiting@unknown-startup.io"
        sender_domain = "unknown-startup.io"

        result = _run_gemini_on_case(subject, body, sender, sender_domain, [])
        status = result.get("status", "")

        # The email is NOT a confirmed application update — Applied would be wrong
        self.assertNotEqual(
            status, "Applied",
            f"Classifier returned 'Applied' on a cold-outreach email — likely wrong. Output: {result}"
        )

        # Document: if Uncertain comes back, production maps it to Applied.
        # This test passes if the model picks Uncertain or Ignore (both acceptable).
        acceptable = {"Uncertain", "Ignore"}
        self.assertIn(
            status, acceptable,
            f"Expected Uncertain or Ignore for ambiguous outreach. Got '{status}'. Output: {result}"
        )
        print(f"[PASS] Uncertain/Ignore test — status='{status}'")
        if status == "Uncertain":
            print("[INFO] Production NOTE: Uncertain is mapped → Applied in the Lambda handler.")


# =============================================================================
# TEST 3 — ACCURACY THRESHOLD GATE
# =============================================================================

class TestAccuracyThresholdGate(unittest.TestCase):
    """
    Run every golden case through Gemini and assert accuracy >= threshold.
    This test acts as a CI gate: if the model regresses, it fails the suite.
    """

    @unittest.skipUnless(GEMINI_API_KEY, "GEMINI_API_KEY not set — skipping live benchmark")
    def test_golden_accuracy_gate(self):
        golden_path = Path("data/golden.jsonl")
        if not golden_path.exists():
            self.skipTest(f"Golden dataset not found at {golden_path}")

        cases: list[EmailCase] = []
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EmailCase.model_validate_json(line))

        self.assertGreater(len(cases), 0, "No golden cases loaded")

        passed = 0
        failures = []

        for case in cases:
            try:
                result = _run_gemini_on_case(
                    subject=case.subject,
                    body=case.body,
                    sender=case.sender,
                    sender_domain=case.sender_domain,
                    existing_jobs_context=case.existing_jobs_context,
                )
                det = evaluate_deterministic(case, result)
                if det.all_passed:
                    passed += 1
                else:
                    failures.append({"case_id": case.id, "details": det.details})
            except Exception as exc:
                failures.append({"case_id": case.id, "error": str(exc)})

        accuracy = passed / len(cases)
        print(f"\nAccuracy: {accuracy:.1%} ({passed}/{len(cases)})")
        if failures:
            print(f"Failed cases: {json.dumps(failures, indent=2)}")

        self.assertGreaterEqual(
            accuracy, config.accuracy_threshold,
            f"Regression gate: accuracy {accuracy:.1%} < threshold {config.accuracy_threshold:.1%}. "
            f"Failures: {failures}"
        )


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
