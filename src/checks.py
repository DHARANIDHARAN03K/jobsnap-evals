"""
src/checks.py: Deterministic and LLM-as-judge evaluation checks.

Aligned with the real ModelOutput schema (corrected_company_name, status, job_title,
source_of_company) from src/schema.py.

Part A – Deterministic checks:
  1. Output is valid JSON with required fields present.
  2. corrected_company_name matches expected_company (after normalization).
  3. status matches expected_status (exact case, 9-value vocabulary).

Part B – LLM-as-judge:
  "Does the output contain any fact not present in the email?"
  Binary PASS/FAIL + one-line reason.
  Uses Gemini (the only reliably working key) via urllib3 (same stack as classifier).
"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schema import EmailCase, ModelOutput


# =============================================================================
# PART A: DETERMINISTIC CHECKS
# =============================================================================

class DeterministicResult(BaseModel):
    """Outcome of running all Part A checks on one (case, model_output) pair."""
    is_valid_json: bool
    schema_valid: bool
    company_match: bool
    status_match: bool
    all_passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


# ── Normalization ─────────────────────────────────────────────────────────────

_SUFFIX_PATTERN = re.compile(
    r"\b(corporation|corp\.?|incorporated|inc\.?|llc\.?|ltd\.?|"
    r"technologies|tech|systems|co\.?|group|partners|platform|"
    r"analytics|services|labs?|pvt|llp)\b",
    re.IGNORECASE,
)

def normalize_company(name: str | None) -> str:
    """Strip legal suffixes and punctuation for fuzzy comparison."""
    if not name:
        return ""
    text = name.strip()
    text = _SUFFIX_PATTERN.sub("", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def check_company_match(predicted: str | None, expected: str | None) -> tuple[bool, str]:
    """
    Compare predicted corrected_company_name against expected_company.
    Handles None, Unknown, exact normalized match, and substring match.
    """
    # Both absent / None / Unknown → treat as no company expected
    pred_clean = (predicted or "").strip()
    exp_clean = (expected or "").strip()

    if not exp_clean:
        # No ground-truth company: pass trivially
        return True, "No expected company — skipping check"

    if not pred_clean or pred_clean.lower() == "unknown":
        return False, f"Predicted company is '{pred_clean}', expected '{exp_clean}'"

    norm_pred = normalize_company(pred_clean)
    norm_exp = normalize_company(exp_clean)

    if norm_pred == norm_exp:
        return True, f"Exact normalized match: '{norm_pred}'"
    if norm_pred and norm_pred in norm_exp:
        return True, f"Substring match: '{norm_pred}' ⊂ '{norm_exp}'"
    if norm_exp and norm_exp in norm_pred:
        return True, f"Substring match: '{norm_exp}' ⊂ '{norm_pred}'"

    return False, f"Company mismatch: '{pred_clean}' ≠ '{exp_clean}'"


def check_status_match(predicted: str | None, expected: str | None) -> tuple[bool, str]:
    """Exact-case status comparison (e.g. 'Applied' == 'Applied')."""
    pred = (predicted or "").strip()
    exp = (expected or "").strip()
    if pred == exp:
        return True, f"Status match: '{pred}'"
    return False, f"Status mismatch: got '{pred}', expected '{exp}'"


# ── Schema validation ─────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {"corrected_company_name", "status", "source_of_company"}

def validate_model_output(raw_or_parsed: Any) -> tuple[bool, ModelOutput | None, str]:
    """
    Check that the output is valid JSON and contains the required Lambda fields.
    Returns (is_valid, validated_model | None, message).
    """
    parsed_dict: dict[str, Any]

    if isinstance(raw_or_parsed, str):
        try:
            parsed_dict = json.loads(raw_or_parsed)
        except Exception as exc:
            return False, None, f"JSON parse error: {exc}"
    elif isinstance(raw_or_parsed, dict):
        parsed_dict = raw_or_parsed
    else:
        return False, None, f"Unsupported type: {type(raw_or_parsed).__name__}"

    missing = _REQUIRED_FIELDS - set(parsed_dict.keys())
    if missing:
        return False, None, f"Missing required fields: {missing}"

    try:
        model = ModelOutput.model_validate(parsed_dict)
        return True, model, "Schema OK"
    except Exception as exc:
        return False, None, f"Pydantic validation error: {exc}"


# ── Main deterministic evaluator ──────────────────────────────────────────────

def evaluate_deterministic(case: EmailCase, model_output: Any) -> DeterministicResult:
    """Run all Part A checks and return a DeterministicResult."""
    details: dict[str, Any] = {}

    is_json, validated, schema_msg = validate_model_output(model_output)
    details["schema"] = schema_msg

    comp_ok = False
    stat_ok = False

    if validated:
        comp_ok, comp_msg = check_company_match(
            validated.corrected_company_name, case.expected_company
        )
        details["company"] = comp_msg

        stat_ok, stat_msg = check_status_match(
            validated.status, case.expected_status
        )
        details["status"] = stat_msg
    else:
        details["company"] = "Skipped — schema invalid"
        details["status"] = "Skipped — schema invalid"

    return DeterministicResult(
        is_valid_json=is_json,
        schema_valid=(validated is not None),
        company_match=comp_ok,
        status_match=stat_ok,
        all_passed=bool(validated and comp_ok and stat_ok),
        details=details,
    )


# =============================================================================
# PART B: LLM-AS-JUDGE
# =============================================================================

class JudgeResult(BaseModel):
    """Binary verdict from the LLM judge."""
    verdict: Literal["PASS", "FAIL"]
    reason: str


JUDGE_SYSTEM_PROMPT = """You are an objective AI evaluation judge assessing factual faithfulness.

Core question: "Does the extracted output contain any fact NOT present in the source email?"

Rules:
- PASS if all companies, roles, dates, and statuses are directly stated in or
  obviously inferable from the email text.
- FAIL if the output introduces a company name, role title, or status that does not
  appear in the email (hallucination).

Return ONLY valid JSON:
{"verdict": "PASS" | "FAIL", "reason": "<one concise sentence>"}
"""


def judge_hallucination(
    email_subject: str,
    email_body: str,
    output: Any,
) -> JudgeResult:
    """
    Ask Gemini (via urllib3, same stack as classifier) whether the output
    hallucinated any fact not in the email.

    Falls back to a local heuristic if Gemini is unavailable.
    """
    import os
    import urllib3

    # Normalize output
    if isinstance(output, str):
        try:
            output_dict = json.loads(output)
        except Exception:
            output_dict = {"raw": output}
    elif isinstance(output, dict):
        output_dict = output
    else:
        output_dict = {"raw": str(output)}

    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    user_prompt = (
        f"Source Email Subject: {email_subject}\n\n"
        f"Source Email Body:\n---\n{email_body[:3000]}\n---\n\n"
        f"Extracted Output:\n---\n{json.dumps(output_dict, indent=2)}\n---\n\n"
        "Does the output contain any fact not present in the email?"
    )

    if gemini_key:
        try:
            http = urllib3.PoolManager()
            payload = {
                "contents": [{"parts": [{"text": f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.0,
                },
            }
            r = http.request(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={gemini_key}",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=urllib3.Timeout(connect=5, read=20),
            )
            if r.status == 200:
                resp = json.loads(r.data.decode("utf-8"))
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text)
                verdict = "PASS" if str(data.get("verdict", "")).upper() == "PASS" else "FAIL"
                return JudgeResult(verdict=verdict, reason=str(data.get("reason", "LLM judge.")))
            else:
                print(f"[Judge] Gemini returned HTTP {r.status} — using heuristic fallback")
        except Exception as exc:
            print(f"[Judge] Gemini call failed ({exc}) — using heuristic fallback")

    return _heuristic_judge(email_body, output_dict)


def _heuristic_judge(email_body: str, output: dict) -> JudgeResult:
    """Local fallback judge: checks if extracted fields appear in email text."""
    text_lower = email_body.lower()

    company = output.get("corrected_company_name") or ""
    role = output.get("job_title") or ""
    status = output.get("status") or ""

    if company and company.lower() not in ("unknown", ""):
        norm = normalize_company(company)
        tokens = [t for t in norm.split() if len(t) > 2]
        if tokens and not any(t in text_lower for t in tokens):
            return JudgeResult(
                verdict="FAIL",
                reason=f"Company '{company}' not found in email text (possible hallucination).",
            )

    if role:
        role_tokens = [t for t in re.sub(r"[^\w\s]", "", role.lower()).split() if len(t) > 3]
        skip = {"senior", "staff", "lead", "junior"}
        role_tokens = [t for t in role_tokens if t not in skip]
        if role_tokens and not any(t in text_lower for t in role_tokens):
            return JudgeResult(
                verdict="FAIL",
                reason=f"Role '{role}' tokens not found in email text.",
            )

    rejection_words = ["unfortunately", "not moving forward", "will not be proceeding", "not selected"]
    positive_statuses = {"offer", "interview", "assessment", "screening"}
    if status.lower() in positive_statuses and any(w in text_lower for w in rejection_words):
        return JudgeResult(
            verdict="FAIL",
            reason=f"Status '{status}' contradicts rejection language in email.",
        )

    return JudgeResult(
        verdict="PASS",
        reason="All extracted values appear grounded in the email text.",
    )
