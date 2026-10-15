"""
api/index.py: Serverless API endpoint for Vercel deployment.
Provides:
  - GET /api/cases: Returns preset benchmark test cases (including the famous Libra AI vs Zennah AI case)
  - POST /api/classify: Runs real extraction using the Lambda Intelligence Engine with environment keys
"""

import json
from http.server import BaseHTTPRequestHandler
import os
import sys
from pathlib import Path
import time
import urllib.parse

# Add parent directory to path so src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import (
    REAL_PROVIDERS,
    build_groq_system_prompt,
    extract_company_from_email,
    extract_sender_domain,
    is_aggregator_domain,
    truncate_at_noise_zone,
)
from src.checks import check_company_match, check_status_match, judge_hallucination

PRESET_CASES = [
    {
        "id": "case_libra_ai_contamination",
        "title": "Libra AI (Context Contamination Test)",
        "badge": "Bug Reproduction",
        "subject": "Next Steps: Machine Learning Engineer at Libra AI",
        "sender": "talent@libra-ai.com",
        "body": "Hi Jordan,\n\nThank you for speaking with our team about the Machine Learning Engineer role at Libra AI. We are pleased to invite you to our 45-minute technical screen this Wednesday.\n\nBest regards,\nLibra AI Recruiting Team",
        "existing_jobs_context": [
            {"id": "APP-1092", "company": "Zennah AI", "role": "AI Research Scientist", "status": "Interview"},
            {"id": "APP-1044", "company": "Zennah AI", "role": "MLOps Architect", "status": "Applied"}
        ],
        "expected_company": "Libra AI",
        "expected_status": "Interview"
    },
    {
        "id": "case_soft_rejection_noise",
        "title": "Soft Rejection with Top Jobs Noise",
        "badge": "Adversarial Noise",
        "subject": "Your update from Stripe regarding Backend Engineer role",
        "sender": "jobs-noreply@linkedin.com",
        "body": "Thank you for applying to Stripe. While your experience is impressive, we have decided not to move forward with your candidacy at this time.\n\nTop jobs for you:\n- Senior Dev at Datadog\n- Infrastructure Engineer at Figma",
        "existing_jobs_context": [],
        "expected_company": "Stripe",
        "expected_status": "Rejected"
    },
    {
        "id": "case_marketing_course_fake",
        "title": "Marketing Course (Non-job Email)",
        "badge": "False Positive Test",
        "subject": "Complete your application for our AI Bootcamp",
        "sender": "admissions@futuretechacademy.io",
        "body": "Dear candidate,\n\nYou are invited to apply for our 12-week intensive AI engineering program! Click below to enroll and pay your deposit.",
        "existing_jobs_context": [],
        "expected_company": None,
        "expected_status": "Ignore"
    }
]


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.endswith("/api/cases") or path.endswith("/cases"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(PRESET_CASES).encode("utf-8"))
            return

        # Health/status endpoint
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        response = {
            "status": "healthy",
            "available_providers": list(REAL_PROVIDERS.keys()),
            "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
            "has_groq_key": bool(os.environ.get("GROQ_API_KEY")),
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if not (path.endswith("/api/classify") or path.endswith("/classify")):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {str(e)}"})
            return

        subject = req_data.get("subject", "")
        email_body = req_data.get("body", "")
        sender = req_data.get("sender", "")
        provider = req_data.get("provider", "gemini")
        context_rows = req_data.get("existing_jobs_context", [])
        expected_company = req_data.get("expected_company")
        expected_status = req_data.get("expected_status")

        if provider not in REAL_PROVIDERS:
            self._send_json(400, {"error": f"Unsupported provider: {provider}"})
            return

        # Check if the requested provider's key is set; if not, automatically route to Gemini (the active provider)
        key_env_var = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(provider)

        effective_provider = provider
        note = None
        if key_env_var and not os.environ.get(key_env_var):
            effective_provider = "gemini"
            note = f"Key for '{provider}' not configured in Vercel. Automatically routed to Gemini (gemini-3.5-flash)."

        call_fn = REAL_PROVIDERS[effective_provider]

        # Build prompt exactly as production Lambda does
        body_clean = truncate_at_noise_zone(email_body)
        sender_domain = extract_sender_domain(sender)
        hint = extract_company_from_email(subject, body_clean)
        is_agg = is_aggregator_domain(sender_domain)

        sys_prompt = build_groq_system_prompt(
            subject_company_hint=hint,
            existing_jobs_context=context_rows,
            sender_domain=sender_domain,
            is_aggregator=is_agg,
        )
        user_content = f"Subject: {subject}\n\nBody:\n{body_clean[:4000]}"

        start_time = time.perf_counter()
        try:
            result = call_fn(sys_prompt, user_content)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Deterministic checks
            comp_match, comp_msg = check_company_match(
                result.get("corrected_company_name"), expected_company
            ) if expected_company is not None else (True, "No expected company")

            status_match, status_msg = check_status_match(
                result.get("status"), expected_status
            ) if expected_status is not None else (True, "No expected status")

            # Check context contamination
            source_of_company = result.get("source_of_company")
            is_contaminated = (source_of_company == "context_rows")

            # LLM Judge evaluation
            judge = judge_hallucination(subject, email_body, result)

            self._send_json(200, {
                "success": True,
                "provider": effective_provider,
                "requested_provider": provider,
                "note": note,
                "latency_ms": round(latency_ms, 2),
                "model_output": result,
                "evaluation": {
                    "company_match": comp_match,
                    "company_msg": comp_msg,
                    "status_match": status_match,
                    "status_msg": status_msg,
                    "is_contaminated": is_contaminated,
                    "judge_verdict": judge.verdict,
                    "judge_reason": judge.reason,
                }
            })
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._send_json(500, {
                "success": False,
                "provider": provider,
                "latency_ms": round(latency_ms, 2),
                "error": str(e),
            })

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
