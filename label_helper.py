"""
label_helper.py: Tool to ingest raw emails from data/real/, pre-fill labels via LLM,
and export a structured file for human review and correction.
"""

import argparse
import email
from email import policy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import httpx

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.schema import EmailCase, EmailStatus


PROMPT_TEMPLATE = """You are an expert annotator for an evaluation dataset of job application emails.
Analyze the following email and extract the evaluation labels.

Allowed status values:
- "applied": Confirmation that an application was received/submitted.
- "interview": Invitation or scheduling for a phone screen, technical, behavioral, or onsite interview.
- "assessment": Invitation to a coding challenge, take-home test, or automated assessment (e.g., HackerRank, CodeSignal).
- "offer": Formal or informal job offer.
- "rejected": Rejection letter, whether direct or softly worded.
- "none": Not a job application email (e.g., newsletter, marketing, receipt, general update).

Edge case tags to detect and include if applicable:
- "soft_rejection": Polite/indirect rejection (e.g. "keeping resume on file", "impressed by background but moving forward with others").
- "newsletter": Job board digests, tech news, promotional career newsletters.
- "two_companies": Mentions a recruiting agency/headhunter AND the client company hiring.
- "forwarded": Email headers showing it was forwarded or redirected.
- "quoted_printable": Contains MIME quoted-printable artifacts (e.g. =20, =3D).
- "context_contamination": Quoted previous threads, signature references to prior companies, or mixed disparate roles.

Return ONLY a JSON object with this exact structure:
{
  "is_job_email": true or false,
  "company": "Company Name" or null,
  "status": "applied" | "interview" | "assessment" | "offer" | "rejected" | "none",
  "role": "Role Title" or null,
  "tags": ["tag1", "tag2"]
}

Email Content:
---
%s
---
"""


def extract_text_from_file(file_path: Path) -> str:
    """Extract raw text from .txt, .eml, or .json files."""
    suffix = file_path.suffix.lower()

    if suffix == ".eml":
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
            subject = msg.get("Subject", "")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        body += part.get_content()
            else:
                body = msg.get_content()
            return f"Subject: {subject}\n\n{body}".strip()

    elif suffix in (".json", ".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("email_text") or data.get("text") or data.get("body") or json.dumps(data)
            return json.dumps(data)

    else:
        # Default text read (.txt, raw text, etc.)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()


def heuristic_prefill(email_text: str) -> dict[str, Any]:
    """Rule-based heuristic pre-filler used as fallback when no LLM API key is present."""
    text_lower = email_text.lower()
    tags = []
    status = "none"
    is_job = False
    company = None
    role = None

    # Check edge case indicators
    if "=20" in email_text or "=3d" in text_lower or "=3D" in email_text:
        tags.append("quoted_printable")
    if "forwarded message" in text_lower or "subject: fwd:" in text_lower:
        tags.append("forwarded")
    if "newsletter" in text_lower or "unsubscribe" in text_lower or "weekly digest" in text_lower:
        tags.append("newsletter")
    if "on behalf of our client" in text_lower or "recruiting partners" in text_lower:
        tags.append("two_companies")
    if "wrote:" in text_lower and ">" in email_text:
        tags.append("context_contamination")

    # Status heuristics
    if any(k in text_lower for k in ["offer of employment", "pleased to offer", "offer letter"]):
        status = "offer"
        is_job = True
        tags.append("offer")
    elif any(k in text_lower for k in ["interview", "phone screen", "speak with our team", "technical screen"]):
        status = "interview"
        is_job = True
        tags.append("interview")
    elif any(k in text_lower for k in ["assessment", "hackerrank", "codesignal", "coding challenge", "take-home"]):
        status = "assessment"
        is_job = True
        tags.append("assessment")
    elif any(k in text_lower for k in ["unfortunately", "not moving forward", "other candidates", "regret to inform"]):
        status = "rejected"
        is_job = True
        tags.append("rejected")
        if any(k in text_lower for k in ["impressed", "keep your resume on file", "future openings"]):
            tags.append("soft_rejection")
    elif any(k in text_lower for k in ["thank you for applying", "received your application", "application submitted"]):
        status = "applied"
        is_job = True
        tags.append("applied")

    if not is_job and "newsletter" in tags:
        tags.append("non_job_email")

    return {
        "is_job_email": is_job,
        "company": company,
        "status": status,
        "role": role,
        "tags": sorted(list(set(tags))),
    }


def call_llm(email_text: str, provider: str = "auto", model: str | None = None) -> dict[str, Any]:
    """Call LLM API to pre-fill classification labels, falling back to heuristics if keys are absent."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    prompt = PROMPT_TEMPLATE % email_text

    # OpenAI / Compatible API
    if (provider in ("openai", "auto")) and openai_key:
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        target_model = model or "gpt-4o-mini"
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": target_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0,
                    },
                )
                res.raise_for_status()
                data = res.json()["choices"][0]["message"]["content"]
                return json.loads(data)
        except Exception as e:
            print(f"[Warning] OpenAI API call failed ({e}). Falling back to heuristics.")

    # Gemini API
    elif (provider in ("gemini", "auto")) and gemini_key:
        target_model = model or "gemini-1.5-flash"
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0},
                    },
                )
                res.raise_for_status()
                text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception as e:
            print(f"[Warning] Gemini API call failed ({e}). Falling back to heuristics.")

    # Fallback to local heuristic pre-filler
    return heuristic_prefill(email_text)


def interactive_review(review_file: Path, output_file: Path | None = None):
    """Interactive CLI review mode allowing the user to review and correct predictions."""
    if not review_file.exists():
        print(f"Review file {review_file} does not exist.")
        return

    items: list[dict] = []
    with open(review_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        print("No items to review.")
        return

    print(f"\nLoaded {len(items)} items for review. Press Enter to accept existing value or type replacement.\n")
    reviewed_cases: list[EmailCase] = []

    for i, item in enumerate(items, 1):
        print("=" * 60)
        print(f"Item {i}/{len(items)}: ID={item.get('id')}")
        print("--- EMAIL SNIPPET ---")
        text_snippet = item.get("email_text", "")[:300]
        print(text_snippet + ("..." if len(item.get("email_text", "")) > 300 else ""))
        print("---------------------")

        # Review fields
        curr_job = item.get("expected_is_job_email", False)
        inp_job = input(f"is_job_email [{curr_job}] (y/n or Enter): ").strip().lower()
        new_job = curr_job if inp_job == "" else (inp_job in ("y", "true", "1"))

        curr_comp = item.get("expected_company")
        inp_comp = input(f"expected_company [{curr_comp}]: ").strip()
        new_comp = curr_comp if inp_comp == "" else (None if inp_comp.lower() == "none" else inp_comp)

        curr_status = item.get("expected_status", "none")
        inp_status = input(f"expected_status (applied/interview/assessment/offer/rejected/none) [{curr_status}]: ").strip().lower()
        new_status = curr_status if inp_status == "" else inp_status

        curr_role = item.get("expected_role")
        inp_role = input(f"expected_role [{curr_role}]: ").strip()
        new_role = curr_role if inp_role == "" else (None if inp_role.lower() == "none" else inp_role)

        curr_tags = item.get("tags", [])
        inp_tags = input(f"tags (comma-separated) [{', '.join(curr_tags)}]: ").strip()
        new_tags = curr_tags if inp_tags == "" else [t.strip() for t in inp_tags.split(",") if t.strip()]

        validated = EmailCase(
            id=item["id"],
            email_text=item["email_text"],
            expected_is_job_email=new_job,
            expected_company=new_comp,
            expected_status=new_status,
            expected_role=new_role,
            tags=new_tags,
        )
        reviewed_cases.append(validated)

    dest = output_file or review_file
    with open(dest, "w", encoding="utf-8") as f:
        for c in reviewed_cases:
            f.write(c.model_dump_json() + "\n")

    print(f"\n[Success] Saved {len(reviewed_cases)} validated cases to {dest}")


def process_raw_emails(input_dir: Path, output_file: Path, provider: str, model: str | None):
    """Scan raw emails in input_dir, run LLM pre-filling, and export JSONL for user review."""
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_files = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith(".")]

    if not raw_files:
        print(f"No raw email files found in {input_dir}.")
        print("Place raw email files (.txt, .eml, .json) in data/real/ and re-run.")
        return

    print(f"Found {len(raw_files)} raw files in {input_dir}. Starting pre-filling...")
    results: list[dict[str, Any]] = []

    for idx, f in enumerate(raw_files, 1):
        case_id = f"real_{f.stem}"
        email_text = extract_text_from_file(f)
        if not email_text:
            continue

        print(f"[{idx}/{len(raw_files)}] Pre-filling labels for {f.name}...")
        llm_output = call_llm(email_text, provider=provider, model=model)

        # Normalize status to valid literal
        raw_status = str(llm_output.get("status", "none")).lower()
        status_val = raw_status if raw_status in EmailStatus._value2member_map_ else "none"

        case_data = {
            "id": case_id,
            "email_text": email_text,
            "expected_is_job_email": bool(llm_output.get("is_job_email", False)),
            "expected_company": llm_output.get("company"),
            "expected_status": status_val,
            "expected_role": llm_output.get("role"),
            "tags": list(llm_output.get("tags", [])),
            "_review_status": "pending_user_verification",
        }

        # Verify it validates with EmailCase
        try:
            EmailCase(**{k: v for k, v in case_data.items() if k != "_review_status"})
        except Exception as e:
            print(f"[Warning] Validation warning on {case_id}: {e}")

        results.append(case_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[Done] Exported {len(results)} pre-filled cases to {output_file}")
    print(f"You can now review/edit {output_file} directly or run:")
    print(f"  python label_helper.py --review --review-file {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest raw emails from data/real/, pre-fill labels via LLM, and export for review."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/real"),
        help="Directory containing raw emails (.txt, .eml, .json). Default: data/real",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pending_review.jsonl"),
        help="Path to export pre-filled cases for review. Default: data/pending_review.jsonl",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "gemini", "heuristic"],
        help="LLM provider to use for labeling (auto detects env key, or heuristic fallback).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to invoke (e.g. gpt-4o-mini, gemini-1.5-flash).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Launch interactive terminal review mode to verify and correct exported labels.",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        default=Path("data/pending_review.jsonl"),
        help="Target file to review when running with --review.",
    )
    parser.add_argument(
        "--reviewed-output",
        type=Path,
        default=None,
        help="Destination file for reviewed cases (defaults to overwriting the review file).",
    )

    args = parser.parse_args()

    if args.review:
        interactive_review(args.review_file, args.reviewed_output)
    else:
        process_raw_emails(args.input_dir, args.output, args.provider, args.model)


if __name__ == "__main__":
    main()
