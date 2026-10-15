"""
anonymize_and_build_golden.py: Ingest raw emails from data/real/ (git-ignored),
anonymize personal identifiers (people, emails, links, companies), and export
data/golden.jsonl conforming to the new EmailCase schema.
"""

import argparse
import email
from email import policy
import json
from pathlib import Path
import re
import sys
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.schema import EmailCase, AllowedStatus
from src.classifier import (
    extract_sender_domain,
    company_from_sender_domain,
    is_aggregator_domain,
    extract_company_from_email,
    truncate_at_noise_zone,
)


# Anonymization regex patterns
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')

# Known personal names to scrub
SCRUB_NAMES = [
    r'\bDharani\b', r'\bDharani\s+Dharan\b', r'\bJordan\b', r'\bAlex\b',
    r'\bTaylor\b', r'\bCasey\b', r'\bMorgan\b', r'\bSam\b', r'\bJamie\b'
]


def anonymize_text(text: str) -> str:
    """Anonymize emails, links, phone numbers, and candidate names."""
    if not text:
        return ""

    # Replace emails
    cleaned = EMAIL_REGEX.sub("candidate@example.com", text)
    # Replace URLs with clean placeholder links
    cleaned = URL_REGEX.sub("https://careers.example.com/portal/job-ref", cleaned)
    # Replace phone numbers
    cleaned = PHONE_REGEX.sub("+1 (555) 019-2834", cleaned)
    # Replace personal names
    for pat in SCRUB_NAMES:
        cleaned = re.sub(pat, "Alex", cleaned, flags=re.IGNORECASE)

    return cleaned


def extract_raw_email_components(file_path: Path) -> dict[str, str]:
    """Parse raw .eml, .txt, or .json file into sender, subject, and body."""
    suffix = file_path.suffix.lower()

    if suffix == ".eml":
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
            sender = str(msg.get("From", "talent@company.com"))
            subject = str(msg.get("Subject", "Job Application Update"))
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += str(part.get_content())
            else:
                body = str(msg.get_content())
            return {"sender": sender, "subject": subject, "body": body.strip()}

    elif suffix in (".json", ".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "sender": data.get("sender", "recruiting@company.com"),
                "subject": data.get("subject", "Application Update"),
                "body": data.get("body") or data.get("email_text") or "",
            }

    else:
        # Default text file (.txt)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            subject = "Application Update"
            sender = "recruiting@company.com"
            body_lines = []

            for line in lines:
                if line.lower().startswith("subject:"):
                    subject = line.split(":", 1)[1].strip()
                elif line.lower().startswith("from:"):
                    sender = line.split(":", 1)[1].strip()
                else:
                    body_lines.append(line)

            return {
                "sender": sender,
                "subject": subject,
                "body": "".join(body_lines).strip()
            }


def generate_seed_golden_cases() -> list[EmailCase]:
    """
    High-fidelity seed cases conforming strictly to the new schema:
    - 9 allowed statuses: Applied, Screening, Assessment, Interview, In Review, Rejected, Offer, Ignore, Uncertain
    - Hard edge cases: soft_rejection, newsletter, two_companies, forwarded, quoted_printable, context_contamination
    """
    raw_seed = [
        {
            "id": "case_001_soft_rejection",
            "subject": "Update on your Data Platform Engineer application",
            "body": "Hi Alex,\n\nThank you for taking the time to speak with our team about the Data Platform Engineer role at Acme Analytics. While we were genuinely impressed with your technical depth and problem-solving skills, we have decided to move forward with another candidate whose specific background in Kubernetes internals is a closer fit for our immediate needs.\n\nWe would love to stay in touch as our engineering organization scales, and we will keep your profile in mind for upcoming openings.\n\nBest regards,\nSarah Jenkins\nAcme Analytics Talent Team",
            "sender": "Sarah Jenkins <talent@acmeanalytics.com>",
            "sender_domain": "acmeanalytics.com",
            "existing_jobs_context": [
                {"company_name": "Acme Analytics", "job_title": "Data Platform Engineer", "status": "Interview"}
            ],
            "expected_company": "Acme Analytics",
            "expected_status": "Rejected",
            "expected_title": "Data Platform Engineer",
            "tags": ["soft_rejection", "rejected"]
        },
        {
            "id": "case_002_newsletter",
            "subject": "TechCareers Weekly Issue #142: Salaries, System Design & Top Startups",
            "body": "TechCareers Weekly Issue #142\n\nTop Stories This Week:\n- Why Remote Work Salaries Are Stabilizing in 2026\n- 5 Tips to Ace System Design Interviews\n- Featured Sponsor: Try CloudDB Free for 30 Days\n\nClick here to view 25 trending startup engineering openings on our public job board.\n\nUnsubscribe from this newsletter | Update preferences",
            "sender": "TechCareers Newsletter <digest@techcareers-weekly.com>",
            "sender_domain": "techcareers-weekly.com",
            "existing_jobs_context": [],
            "expected_company": "Unknown",
            "expected_status": "Ignore",
            "expected_title": None,
            "tags": ["newsletter", "marketing", "non_job_email"]
        },
        {
            "id": "case_003_two_companies",
            "subject": "Retained Search: Staff Backend Engineer on behalf of NovaScale",
            "body": "Hello Alex,\n\nI am Jordan from Apex Recruiting Partners. We are currently leading an exclusive retained executive search on behalf of our client, NovaScale.\n\nNovaScale is looking for a Staff Backend Engineer to lead their distributed storage architecture. We reviewed your public GitHub and thought your experience with Go and Raft would be an exceptional match for NovaScale's technical leadership.\n\nWould you be open to an introductory 15-minute call this Thursday?\n\nJordan Vance\nSenior Executive Search Consultant, Apex Recruiting Partners",
            "sender": "Jordan Vance <jvance@apexrecruitingpartners.com>",
            "sender_domain": "apexrecruitingpartners.com",
            "existing_jobs_context": [],
            "expected_company": "NovaScale",
            "expected_status": "Applied",
            "expected_title": "Staff Backend Engineer",
            "tags": ["two_companies", "staffing_agency", "client_placement"]
        },
        {
            "id": "case_004_quoted_printable",
            "subject": "Fwd: Interview Invitation: Cloud Infrastructure Architect at HyperCloud",
            "body": "---------- Forwarded message ---------\nFrom: Talent Team <talent@hypercloud-systems.com>\nDate: Mon, Oct 12, 2026 at 9:15 AM\nSubject: Interview Invitation: Cloud Infrastructure Architect at HyperCloud\nTo: candidate@personalmail.com\nContent-Type: text/plain; charset=3D\"UTF-8\"\nContent-Transfer-Encoding: quoted-printable\n\nDear Candidate,=\n\n\nWe are pleased to invite you to a 45-minute technical screen for the =3D\nCloud Infrastructure Architect position at HyperCloud.=20\n\nPlease select a time slot that suits your schedule using our scheduling lin=\nk below:\nhttps://careers.example.com/portal/job-ref\n\nWe look forward to speaking with you!=20\n\nBest regards,=\n\nHyperCloud Recruiting Team",
            "sender": "Talent Team <talent@hypercloud-systems.com>",
            "sender_domain": "hypercloud-systems.com",
            "existing_jobs_context": [
                {"company_name": "HyperCloud", "job_title": "Cloud Infrastructure Architect", "status": "Applied"}
            ],
            "expected_company": "HyperCloud",
            "expected_status": "Interview",
            "expected_title": "Cloud Infrastructure Architect",
            "tags": ["forwarded", "quoted_printable"]
        },
        {
            "id": "case_005_context_contamination",
            "subject": "Next steps: Senior Security Engineer interview panel",
            "body": "Hi Alex,\n\nGreat news! The hiring manager reviewed your take-home project for the Senior Security Engineer role at FinFlow and would like to schedule your onsite interview panel.\n\nOn Fri, Oct 2, 2026 at 3:00 PM, Stripe Recruiting <no-reply@stripe.com> wrote:\n> Thank you for your interest in the Payments Engineer position at Stripe. Unfortunately, we will not be moving forward with your candidacy at this time.\n\nLooking forward to speaking tomorrow,\nDavid Miller\nLead Technical Recruiter, FinFlow\nFormerly VP of Recruiting at LegacyCorp (2018-2022)",
            "sender": "David Miller <dmiller@finflow.io>",
            "sender_domain": "finflow.io",
            "existing_jobs_context": [
                {"company_name": "Stripe", "job_title": "Payments Engineer", "status": "Rejected"},
                {"company_name": "FinFlow", "job_title": "Senior Security Engineer", "status": "Assessment"}
            ],
            "expected_company": "FinFlow",
            "expected_status": "Interview",
            "expected_title": "Senior Security Engineer",
            "tags": ["context_contamination", "mixed_threads"]
        },
        {
            "id": "case_006_application_received",
            "subject": "Thank you for applying to Veloce Robotics",
            "body": "Dear Alex,\n\nThank you for applying for the Embedded Systems Engineer position at Veloce Robotics. We have received your application and resume.\n\nOur recruiting team is currently reviewing submissions. If your qualifications match our current requirements, we will reach out directly regarding next steps.\n\nYou can check the real-time status of your application anytime in our careers portal: https://careers.example.com/portal/job-ref\n\nBest regards,\nVeloce Robotics People Ops",
            "sender": "Veloce Robotics Careers <careers@veloce.io>",
            "sender_domain": "veloce.io",
            "existing_jobs_context": [],
            "expected_company": "Veloce Robotics",
            "expected_status": "Applied",
            "expected_title": "Embedded Systems Engineer",
            "tags": ["applied", "confirmation"]
        },
        {
            "id": "case_007_technical_assessment",
            "subject": "Invitation to Code Assessment - Full Stack Developer",
            "body": "Hi Alex,\n\nAs the next step in your application for Full Stack Developer at Prism Health, you have been invited to complete a 75-minute technical coding assessment on HackerRank.\n\nInstructions:\n- Complete the assessment within 5 calendar days.\n- Ensure a stable internet connection and quiet environment.\n- Test link: https://careers.example.com/portal/job-ref\n\nGood luck,\nEngineering Hiring Team | Prism Health",
            "sender": "Prism Health Hiring <no-reply@prismhealth.org>",
            "sender_domain": "prismhealth.org",
            "existing_jobs_context": [
                {"company_name": "Prism Health", "job_title": "Full Stack Developer", "status": "Applied"}
            ],
            "expected_company": "Prism Health",
            "expected_status": "Assessment",
            "expected_title": "Full Stack Developer",
            "tags": ["assessment", "coding_challenge"]
        },
        {
            "id": "case_008_job_offer",
            "subject": "Offer of Employment: Principal AI Research Scientist at Aether Dynamics",
            "body": "Dear Alex,\n\nOn behalf of Aether Dynamics, I am thrilled to extend a formal offer of employment for the position of Principal AI Research Scientist reporting to our CTO.\n\nPlease find attached your comprehensive offer letter containing details on compensation, equity incentives, health benefits, and proposed start date. Kindly review and sign via DocuSign by Friday at 5:00 PM EST.\n\nWe are immensely excited about the prospect of you joining our research lab!\n\nWarm regards,\nElena Rostova\nChief People Officer, Aether Dynamics",
            "sender": "Elena Rostova <elena@aetherdynamics.ai>",
            "sender_domain": "aetherdynamics.ai",
            "existing_jobs_context": [
                {"company_name": "Aether Dynamics", "job_title": "Principal AI Research Scientist", "status": "Interview"}
            ],
            "expected_company": "Aether Dynamics",
            "expected_status": "Offer",
            "expected_title": "Principal AI Research Scientist",
            "tags": ["offer", "formal"]
        },
        {
            "id": "case_009_screening_stage",
            "subject": "Application Update: Senior DevOps Engineer",
            "body": "Hello Alex,\n\nYour application for the Senior DevOps Engineer position at BluePeak Security has been reviewed by our talent acquisition leads and has officially moved to our Screening stage.\n\nOur recruiter is currently aligning with hiring managers regarding telephone screen schedules and will reach out with scheduling availability soon.\n\nThank you,\nBluePeak Security Talent Acquisition",
            "sender": "BluePeak Recruiting <talent@bluepeak.io>",
            "sender_domain": "bluepeak.io",
            "existing_jobs_context": [
                {"company_name": "BluePeak Security", "job_title": "Senior DevOps Engineer", "status": "Applied"}
            ],
            "expected_company": "BluePeak Security",
            "expected_status": "Screening",
            "expected_title": "Senior DevOps Engineer",
            "tags": ["screening"]
        },
        {
            "id": "case_010_in_review_stage",
            "subject": "Your UX Designer application is now in review",
            "body": "Hi Alex,\n\nThank you for your interest in joining Apple. Your application for the UX Designer position in our Cupertino design studio has successfully moved to the In Review stage.\n\nOur team is actively reviewing your portfolio. We appreciate your patience as we evaluate your work.\n\nBest regards,\nApple Recruiting",
            "sender": "Apple Careers <noreply@apple.com>",
            "sender_domain": "apple.com",
            "existing_jobs_context": [
                {"company_name": "Apple", "job_title": "UX Designer", "status": "Applied"}
            ],
            "expected_company": "Apple",
            "expected_status": "In Review",
            "expected_title": "UX Designer",
            "tags": ["in_review"]
        }
    ]

    return [EmailCase.model_validate(c) for c in raw_seed]


def process_real_directory(input_dir: Path, output_file: Path):
    """Scan data/real/, anonymize contents, and write data/golden.jsonl."""
    raw_files = [f for f in input_dir.iterdir() if f.is_file() and not f.name.startswith(".")]

    cases: list[EmailCase] = []

    if raw_files:
        print(f"Ingesting and anonymizing {len(raw_files)} files from {input_dir}...")
        for idx, f in enumerate(raw_files, 1):
            comp = extract_raw_email_components(f)
            anon_sender = anonymize_text(comp["sender"])
            anon_subject = anonymize_text(comp["subject"])
            anon_body = anonymize_text(comp["body"])
            domain = extract_sender_domain(anon_sender)

            # Extract initial guess for company
            hint_comp = extract_company_from_email(anon_subject, anon_body) or company_from_sender_domain(domain)

            case = EmailCase(
                id=f"real_{idx:03d}_{f.stem}",
                subject=anon_subject,
                body=anon_body,
                sender=anon_sender,
                sender_domain=domain,
                existing_jobs_context=[],
                expected_company=hint_comp if (hint_comp and hint_comp != "NOT_FOUND") else "Unknown",
                expected_status="Applied",
                expected_title=None,
                tags=["real_ingested", "pending_human_review"],
            )
            cases.append(case)
    else:
        print(f"No raw files in {input_dir}. Initializing data/golden.jsonl with 10 seed benchmark cases...")
        cases = generate_seed_golden_cases()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")

    print(f"\n[Success] Exported {len(cases)} cases to {output_file}")
    print("You can now open data/golden.jsonl to review and correct all labels by hand.")


def main():
    parser = argparse.ArgumentParser(description="Anonymize raw emails and build data/golden.jsonl.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--output", type=Path, default=Path("data/golden.jsonl"))
    args = parser.parse_args()

    args.input_dir.mkdir(parents=True, exist_ok=True)
    process_real_directory(args.input_dir, args.output)


if __name__ == "__main__":
    main()
