"""
src/classifier.py: Real classification engine copied directly from
JobSnap production Lambda (lambda_function_updated.py).

Source of truth for:
  - ATS_DOMAIN_MAP, AGGREGATOR_DOMAINS
  - _NOISE_ZONE_PATTERNS, truncate_at_noise_zone
  - extract_sender_domain, company_from_sender_domain, is_aggregator_domain
  - _clean_company_candidate, extract_company_from_email
  - FEW_SHOT_EXAMPLES, build_groq_system_prompt
  - call_groq, call_gemini, call_cerebras, call_sambanova, call_openrouter

NOTE: All API keys are loaded strictly from environment variables (.env) with NO default values.
No boto3, Supabase, SES, logo, or mock/heuristic classifier code is present.
"""

import json
import os
from pathlib import Path
import re
import urllib3

# ─────────────────────────────────────────────────────────────────────────────
# LOAD .ENV (NO HARDCODED DEFAULT KEYS)
# ─────────────────────────────────────────────────────────────────────────────
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'").strip('"')
                    if k not in os.environ or not os.environ[k]:
                        os.environ[k] = v

_load_env_file()

GROQ_API_KEY       = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY     = os.environ.get('GEMINI_API_KEY')
CEREBRAS_API_KEY   = os.environ.get('CEREBRAS_API_KEY')
SAMBANOVA_API_KEY  = os.environ.get('SAMBANOVA_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

http = urllib3.PoolManager()


# ============================================================
#  KNOWN ATS / DIRECT-COMPANY SENDER DOMAINS
# ============================================================
ATS_DOMAIN_MAP = {
    "greenhouse.io": None,
    "lever.co": None,
    "workday.com": None,
    "myworkday.com": None,
    "darwinbox.in": None,
    "darwinbox.com": None,
    "joinsuperset.com": None,
    "doselect.com": None,
    "ashby.com": None,
    "smartrecruiters.com": None,
    "icims.com": None,
    "bamboohr.com": None,
    "hirevue.com": None,
    "taleo.net": None,
    "successfactors.com": None,
    "jobvite.com": None,
    "breezy.hr": None,
    "recruitee.com": None,
    "zohorecruit.com": None,
    "zohocorp.com": None,
    "cisco.com": "Cisco",
    "jobalerts.cisco.com": "Cisco",
    "amazon.com": "Amazon",
    "microsoft.com": "Microsoft",
    "google.com": "Google",
    "infosys.com": "Infosys",
    "tcs.com": "TCS",
    "wipro.com": "Wipro",
    "hcltech.com": "HCLTech",
    "capgemini.com": "Capgemini",
    "cognizant.com": "Cognizant",
    "accenture.com": "Accenture",
    "deloitte.com": "Deloitte",
    "ibm.com": "IBM",
    "oracle.com": "Oracle",
    "sap.com": "SAP",
    "salesforce.com": "Salesforce",
    "adobe.com": "Adobe",
    "autodesk.com": "Autodesk",
    "sprinklr.com": "Sprinklr",
    "freshworks.com": "Freshworks",
    "zoho.com": "Zoho",
    "swiggy.com": "Swiggy",
    "zomato.com": "Zomato",
    "flipkart.com": "Flipkart",
    "meesho.com": "Meesho",
    "phonepe.com": "PhonePe",
    "paytm.com": "Paytm",
    "razorpay.com": "Razorpay",
    "cred.club": "CRED",
    "byju.com": "BYJU'S",
    "unacademy.com": "Unacademy",
    "deltax.com": "DeltaX",
    "mu-sigma.com": "Mu Sigma",
    "musigma.com": "Mu Sigma",
    "ltimindtree.com": "LTIMindtree",
}

AGGREGATOR_DOMAINS = {
    "linkedin.com", "naukri.com", "indeed.com", "glassdoor.com",
    "internshala.com", "shine.com", "monster.com", "cutshort.io",
    "hirist.com", "apna.co", "foundit.in", "iimjobs.com",
    "instahyre.com", "wellfound.com", "angellist.com"
}


# ============================================================
#  NOISE ZONE PATTERNS
# ============================================================
_NOISE_ZONE_PATTERNS = [
    r'(?i)similar jobs', r'(?i)jobs you may like', r'(?i)you might also like',
    r'(?i)recommended (jobs|for you)', r'(?i)top jobs', r'(?i)see more jobs',
    r'(?i)other jobs at', r'(?i)more jobs like this', r'(?i)new jobs for you',
    r'(?i)jobs based on your', r'(?i)explore (these )?opportunities',
    r'(?i)jobs matching your', r'(?i)we found (these|some) jobs',
    r'(?i)you may be interested in', r'(?i)discover more opportunities',
    r'(?i)now, take these next steps', r'(?i)view similar jobs',
    r'(?i)get the new linkedin',
]

def truncate_at_noise_zone(text: str) -> str:
    earliest = len(text)
    for pat in _NOISE_ZONE_PATTERNS:
        m = re.search(pat, text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest].strip()


# ============================================================
#  SENDER DOMAIN UTILITIES
# ============================================================
def extract_sender_domain(sender: str) -> str:
    m = re.search(r'@([\w.\-]+)', sender)
    return m.group(1).lower() if m else ""

def company_from_sender_domain(sender_domain: str) -> str | None:
    if not sender_domain:
        return None
    if sender_domain in ATS_DOMAIN_MAP:
        return ATS_DOMAIN_MAP[sender_domain]
    parts = sender_domain.split('.')
    for i in range(1, len(parts)):
        parent = '.'.join(parts[i:])
        if parent in ATS_DOMAIN_MAP:
            return ATS_DOMAIN_MAP[parent]
    return "NOT_FOUND"

def is_aggregator_domain(sender_domain: str) -> bool:
    for agg in AGGREGATOR_DOMAINS:
        if sender_domain.endswith(agg):
            return True
    return False


# ============================================================
#  COMPANY HINT EXTRACTION
# ============================================================
_COMPANY_TRAILING_NOISE = re.compile(
    r'\b(by applying|for applying|by submitting|for submitting|'
    r'team|hr|talent|recruitment|careers|group|corp|inc|ltd|llp|pvt|'
    r'technologies|solutions|services|systems|international|global|india)\b.*$',
    re.IGNORECASE
)

def _clean_company_candidate(candidate: str) -> str | None:
    candidate = candidate.strip().rstrip('.,!:').strip()
    candidate = _COMPANY_TRAILING_NOISE.sub('', candidate).strip().rstrip('.,!').strip()
    if not candidate or len(candidate) < 2 or len(candidate) > 60:
        return None
    skip_starts = {
        'the', 'your', 'a', 'an', 'this', 'our', 'role', 'position', 'job',
        'us', 'team', 'company', 'employer', 'recruiter', 'hiring', 'talent',
        'you', 'we', 'they', 'it', 'which', 'that', 'there', 'here',
        'india', 'bangalore', 'chennai', 'mumbai', 'hyderabad', 'delhi',
        'pune', 'kolkata', 'noida', 'gurgaon', 'remote', 'hybrid', 'office',
        'full', 'part', 'time', 'contract', 'permanent', 'fresher', 'graduate',
    }
    first_word = candidate.lower().split()[0] if candidate.split() else ""
    if first_word in skip_starts:
        return None
    job_title_words = {
        'developer', 'engineer', 'analyst', 'manager', 'intern', 'executive',
        'associate', 'specialist', 'consultant', 'architect', 'designer',
    }
    if any(w in candidate.lower().split() for w in job_title_words):
        return None
    return candidate

def extract_company_from_email(subject: str, body: str) -> str | None:
    subject_patterns = [
        r'(?i)(?:your application (?:to|was sent to)|applied to)\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)(?:application|update|invite|interview)\s*[-–:]\s*([A-Z][A-Za-z0-9\s&.,\-]{2,40})(?:\s*[\(\|,\n!]|$)',
        r'(?i)\bat\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)from\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)welcome to\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,!]|$)',
    ]
    body_patterns = [
        r'(?i)(?:thank you for applying|application received)\s+(?:to|at|for)\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40})',
        r'(?i)team at\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)(?:the following items were sent to|application submitted to)\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40})',
        r'(?i)\bat\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)from\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*[\(\|,\n!]|$)',
        r'(?i)([A-Z][A-Za-z0-9\s&.,\-]{2,40?})\s+(?:team|hr|talent|recruitment|careers)',
        r'(?m)^([A-Z][A-Za-z0-9\s&.,\-]{2,40})\s*-\s*[A-Za-z\s,]+$',
        r'(?i)(?:with|at|from)\s+([A-Z][A-Za-z0-9\s&.,\-]{2,40}?)(?:\s*(?:Group|Corp|Inc|LLP|Ltd|Technologies|Solutions|Services)\b)?',
    ]
    for pat in subject_patterns:
        m = re.search(pat, subject)
        if m:
            result = _clean_company_candidate(m.group(1))
            if result:
                return result
    body_snippet = (body or "")[:1500]
    for pat in body_patterns:
        m = re.search(pat, body_snippet, re.MULTILINE)
        if m:
            result = _clean_company_candidate(m.group(1))
            if result:
                return result
    return None


# ============================================================
#  FEW-SHOT EXAMPLES (EXACT COPY FROM LAMBDA)
# ============================================================
FEW_SHOT_EXAMPLES = """
═══ FEW-SHOT EXAMPLES (8 diverse boundary cases — not exhaustive) ═══

EXAMPLE 1 — Real application confirmation (aggregator footer present, ignore it)
Sender: jobs-noreply@linkedin.com
Subject: Dharani, your application was sent to Sourcebae
Body: Your application was sent to Sourcebae. Full Stack Engineer. Applied on May 19, 2026. [Similar jobs section below — NOISE, IGNORE]
→ Output:
{"reasoning":"Body confirms 'application was sent' (past tense, real confirmation). Similar jobs footer is noise.","source_of_company":"body","company_quote":"Your application was sent to Sourcebae","is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Sourcebae","job_title":"Full Stack Engineer","status":"Applied","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Application confirmed as sent to Sourcebae for Full Stack Engineer role."}

EXAMPLE 2 — Rejection with large 'Top Jobs' footer (footer is ALWAYS noise)
Sender: jobs-noreply@linkedin.com
Subject: Your application to Junior Software Engineer at Helic & Co
Body: Your update from Helic & Co. Applied on Jun 24. Unfortunately, we will not be moving forward with your application. [Top jobs: 5 unrelated job listings — NOISE, IGNORE ENTIRELY]
→ Output:
{"reasoning":"Body contains 'unfortunately, we will not be moving forward' — NON-NEGOTIABLE rejection. Top jobs footer is pure noise, irrelevant to classification.","source_of_company":"body","company_quote":"Your update from Helic & Co","is_existing_job":true,"matched_job_id":null,"corrected_company_name":"Helic & Co","job_title":"Junior Software Engineer","status":"Rejected","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Application to Junior Software Engineer at Helic & Co was rejected."}

EXAMPLE 3 — Fake shortlist that is actually a data-harvesting form
Sender: noreply@somecompany.com
Subject: You've been shortlisted!
Body: Congratulations! Please fill this form. What is your current salary? How soon can you join? What is your pincode? Upload your certificate here.
→ Output:
{"reasoning":"Despite shortlist subject, body is collecting personal data (salary, pincode, certificates). This is lead-gen, not a post-application update.","source_of_company":"inferred","company_quote":null,"is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Unknown","job_title":null,"status":"Ignore","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Form collection email disguised as shortlist."}

EXAMPLE 4 — Pre-application training program invitation (NOT a post-application update)
Sender: nehaydcorporate@10877483.brevosend.com
Subject: AI-Powered Full Stack Web Developer Program | Complete Your Application
Body: Dear Candidate, Greetings from The Nexora Group! Thank you for your interest in the AI-Powered Full Stack Web Developer Program. We are excited to invite you to be a part of this industry-focused learning program. Next Step: Complete your application using the link below. Apply Here: https://thenexoragroup.com/... Once your application is completed, our team will share the onboarding process.
→ Output:
{"reasoning":"This is a pre-application invitation to enroll in a training program. The user has NOT applied yet — they are being INVITED to apply. 'Complete your application using the link below' is a future call-to-action, not a post-application confirmation. This is NOT a job application update.","source_of_company":"inferred","company_quote":null,"is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Unknown","job_title":null,"status":"Ignore","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Training program enrollment invitation from Nexora Group — not a post-application update."}

EXAMPLE 5 — Marketing / promotional email (salary-hike story, course sale)
Sender: avi@freedomwithai.com
Subject: Sasikala started with the crash course too
Body: Hey Dharani, Sasikala from Tamil Nadu was in your exact position 6 months ago. She invested ₹499 in our AI Crash Course. By day 3 she had a clear roadmap. Within 90 days she went from 6 LPA to 24 LPA. Click here to start your transformation like Sasikala did!
→ Output:
{"reasoning":"This is a marketing email promoting a paid crash course. It is a sales email using a success story. There is no job application, no company hiring the user, no post-application status. This is promotional content, not a job update.","source_of_company":"inferred","company_quote":null,"is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Unknown","job_title":null,"status":"Ignore","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Marketing email selling AI crash course using a success story — not a job update."}

EXAMPLE 6 — Real interview invitation
Sender: noreply@greenhouse.io
Subject: Interview Invitation – Software Engineer at Stripe
Body: We'd like to invite you to interview for the Software Engineer position. Please select a slot.
→ Output:
{"reasoning":"Clear interview invitation from a real ATS. User previously applied. Status is Interview.","source_of_company":"subject","company_quote":"Interview Invitation – Software Engineer at Stripe","is_existing_job":true,"matched_job_id":null,"corrected_company_name":"Stripe","job_title":"Software Engineer","status":"Interview","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Interview invitation for Software Engineer at Stripe."}

EXAMPLE 7 — OTP / email verification (NOT a job application update)
Sender: notifications@joinsuperset.com
Subject: Verify your email on Superset
Body: You have successfully registered on Superset! Use the OTP given below to verify your email ID. 5364.
→ Output:
{"reasoning":"This is a registration OTP email. Zero application-related content — no job title, no company hiring, no post-application update.","source_of_company":"inferred","company_quote":null,"is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Unknown","job_title":null,"status":"Ignore","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Email verification OTP from Superset — not a job update."}

EXAMPLE 8 — LinkedIn connection request (NOT a job application update)
Sender: invitations@linkedin.com
Subject: I want to connect
Body: Mohamed, Data Science and Machine Learning Intern from HDLC Info Technologies, wants to join your network.
→ Output:
{"reasoning":"Social network connection invitation. No application-related content. Not a post-application update.","source_of_company":"inferred","company_quote":null,"is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Unknown","job_title":null,"status":"Ignore","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"LinkedIn connection request — not a job update."}

EXAMPLE 9 — Title extraction boundary case (Avoid extracting status as title)
Sender: noreply@apple.com
Subject: Subject: Your UX Designer application is now in review
Body: Application for UX Designer at Apple has moved to the In Review stage.
→ Output:
{"reasoning":"Body confirms application status update. Extract ONLY the noun representing the role for job_title, NOT the verb phrase 'has moved to the In Review stage'.","source_of_company":"body","company_quote":"Application for UX Designer at Apple","is_existing_job":false,"matched_job_id":null,"corrected_company_name":"Apple","job_title":"UX Designer","status":"In Review","interview_date":null,"deadline":null,"original_email_date":null,"body_summary":"Application for UX Designer at Apple moved to In Review stage."}
"""


# ============================================================
#  GROQ SYSTEM PROMPT BUILDER
# ============================================================
def build_groq_system_prompt(
    subject_company_hint: str | None,
    existing_jobs_context: list,
    sender_domain: str,
    is_aggregator: bool,
) -> str:
    domain_company = company_from_sender_domain(sender_domain)
    domain_note = ""
    if domain_company and isinstance(domain_company, str) and domain_company != "NOT_FOUND":
        domain_note = f"\nSENDER DOMAIN COMPANY (highest confidence): {domain_company}"
    elif domain_company is None:
        domain_note = "\nSENDER DOMAIN: Known ATS platform — extract company name from subject/body."

    aggregator_note = ""
    if is_aggregator:
        aggregator_note = (
            "\nAGGREGATOR EMAIL WARNING: This email came from a job aggregator (LinkedIn/Naukri/Indeed etc.).\n"
            "Aggregator emails can be EITHER:\n"
            "  (a) Real post-application updates: status, rejection, confirmation → TRACK THESE\n"
            "  (b) Pre-application job recommendations / alerts, connection requests → IGNORE THESE\n"
            "Key distinction: past tense = real ('was sent', 'you applied', 'not moving forward') → track.\n"
            "Present tense asking action = spam ('apply now', 'explore this role', 'connect') → Ignore.\n"
            "LinkedIn/Naukri emails often end with a 'Top jobs' or 'Similar jobs' section.\n"
            "CRITICAL: Extract company/status ONLY from the primary content block (first 3 paragraphs).\n"
            "NEVER extract company from job suggestions footer. Ignore everything after 'Top jobs',\n"
            "'Similar jobs', 'See more jobs', 'Get the new LinkedIn app', or any unsubscribe line."
        )

    return f"""You are JobSnap Intelligence Engine — a precision classifier for job application emails.

Your ONLY job: determine if this email is a genuine post-application update, and if so, extract structured data.

NON-NEGOTIABLE RULE 1 (applies above ALL examples and ALL other instructions):
If the email body contains ANY rejection language — "not moving forward", "unfortunately",
"not selected", "decided not to proceed", "we regret", "will not be proceeding" —
then status MUST be "Rejected". This rule cannot be overridden by footers, job-suggestion
sections, or any other email content. The primary body always wins.

NON-NEGOTIABLE RULE 2 (ANTI-HALLUCINATION):
The "Existing tracked jobs for matching" list is provided ONLY so you can flag potential duplicates (set is_existing_job=true and matched_job_id).
It is NOT a source of truth for extracting missing fields.
You must NEVER copy a company name or job title from the existing tracked jobs list into your extraction output if it is not present in the SOURCE EMAIL text.
If you find yourself about to write a company name that only appears in the tracked jobs list and NOT in the email, STOP. Set source_of_company="context_rows" and corrected_company_name="Unknown". This is a critical failure.

{FEW_SHOT_EXAMPLES}

═══ CLASSIFICATION DECISION TREE ═══
STEP 1 — Is this email about an application the user ALREADY submitted?
  • YES (past tense: "was sent", "was received", "you applied", "thank you for applying",
         "not moving forward", "unfortunately", "applied on [date]", "your update from") → STEP 2
  • NO → status = "Ignore". The following are ALL Ignore regardless of subject line:
      - Asking user to apply now / complete their application / enroll in a program
      - Training program or internship course invitations ("Complete your application using the link")
      - Marketing emails promoting paid courses, salary-hike success stories, bootcamps
      - Job alerts, recommendations, or pre-application CTAs
      - OTP / verification / registration confirmation emails
      - Social network connection requests
      - Newsletters, webinars, workshops
      - ANY email where the primary call-to-action is something the user must do NEXT
        (future tense = Ignore; past tense / update = could be real)

  KEY TEST: Ask yourself — "Has the user ALREADY applied before this email arrived?"
  If the email is inviting/asking them to apply → NO → Ignore.
  If the email is updating them on an application they already submitted → YES → STEP 2.

STEP 2 — Is it collecting personal data from the user?
  • If body contains salary questions, pincode, experience, certificate upload, GitHub/LeetCode → "Ignore"
  • Otherwise → STEP 3

STEP 3 — What is the status? (CRITICAL: You MUST ONLY choose from the exact 9 allowed statuses below.)
  • "Applied"    → confirmation / acknowledgment / thank you for applying
  • "Screening"  → explicitly under review / profile screened / being evaluated
  • "Assessment" → test link / coding assessment / assignment sent
  • "Interview"  → interview scheduled / invite / slot selection
  • "In Review"  → recruiter viewed / moved to next stage (implicit)
  • "Rejected"   → rejection / will not be moving forward / not selected / unfortunately
  • "Offer"      → offer letter / welcome aboard / pleased to offer
  • "Ignore"     → not a real application update / spam / job alert / OTP / connection request
  • "Uncertain"  → use ONLY if you genuinely cannot determine if this is a real application email

REMINDER: Rejection language in the body ALWAYS wins over footer or boilerplate content.
  
  NON-NEGOTIABLE RULE 3 (ORIGINAL DATE EXTRACTION):
  If the email contains a forwarded block (e.g., "---------- Forwarded message ---------" or "Date: Mon, 13 Jul 2026"), extract the EXACT original date and time the email was sent to the user.
  Return it in ISO 8601 format (e.g., "2026-07-13T14:57:00Z"). If no forwarded date is found, return null.

  NON-NEGOTIABLE RULE 4 (ENTITY EXTRACTION BOUNDARIES):
  When extracting the `job_title`, isolate the exact NOUN PHRASE representing the role (e.g., 'UX Designer', 'Software Engineer'). Do NOT extract the verb phrase describing the status (e.g., 'has moved to', 'is now in', 'has been updated'). If the title is not clearly present, return null.

Footer content ("Top jobs", "Similar jobs", "See more jobs", unsubscribe links) is NEVER
evidence for classification. Only the primary body content counts.

═══ OUTPUT FORMAT (strict JSON, no extra text) ═══
{{
  "reasoning": "<explain your classification decision in 1-2 sentences>",
  "source_of_company": "<subject | body | sender_domain | context_rows | inferred>",
  "company_quote": "<exact substring from email where company name was found, or null>",
  "is_existing_job": <true if follow-up on known application, false if new>,
  "matched_job_id": <string UUID if matched, else null>,
  "corrected_company_name": "<company name, or 'Unknown' if not determinable>",
  "job_title": "<role title, or null if not in email>",
  "status": "<MUST BE EXACTLY ONE OF: Applied|Screening|Assessment|Interview|In Review|Rejected|Offer|Ignore|Uncertain>",
  "interview_date": "<ISO date string or null>",
  "deadline": "<ISO date string or null>",
  "body_summary": "<1 sentence summary of email>"
}}

═══ CONTEXT ═══
{domain_note}
{aggregator_note}
{"Subject company hint: " + subject_company_hint if subject_company_hint else ""}
Existing tracked jobs for matching: {json.dumps(existing_jobs_context[:20])}
"""


# ============================================================
#  GROQ CALL
# ============================================================
def call_groq(system_prompt: str, user_content: str) -> dict | None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    r = http.request(
        'POST',
        'https://api.groq.com/openai/v1/chat/completions',
        body=json.dumps(payload),
        headers={
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        },
        timeout=urllib3.Timeout(connect=5, read=25)
    )

    if r.status != 200:
        err = r.data.decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f"Groq API error {r.status}: {err}")

    resp = json.loads(r.data.decode('utf-8'))
    raw  = resp['choices'][0]['message']['content']
    data = json.loads(raw)
    data['ai_model_used'] = 'Groq'
    return data


# ============================================================
#  GEMINI CALL (PRIMARY)
# ============================================================
def call_gemini(system_prompt: str, user_content: str) -> dict | None:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "parts": [{"text": user_content}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "reasoning":             {"type": "STRING"},
                    "source_of_company":     {"type": "STRING"},
                    "company_quote":         {"type": "STRING", "nullable": True},
                    "is_existing_job":       {"type": "BOOLEAN"},
                    "matched_job_id":        {"type": "STRING", "nullable": True},
                    "corrected_company_name":{"type": "STRING"},
                    "job_title":             {"type": "STRING", "nullable": True},
                    "status": {
                        "type": "STRING",
                        "enum": ["Applied","Screening","Assessment","Interview",
                                 "In Review","Rejected","Offer","Ignore","Uncertain"]
                    },
                    "interview_date":        {"type": "STRING", "nullable": True},
                    "deadline":              {"type": "STRING", "nullable": True},
                    "original_email_date":   {"type": "STRING", "nullable": True},
                    "body_summary":          {"type": "STRING"}
                },
                "required": ["reasoning","source_of_company","corrected_company_name","status","body_summary"]
            }
        }
    }

    r = http.request(
        'POST',
        f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}',
        body=json.dumps(payload),
        headers={'Content-Type': 'application/json'},
        timeout=urllib3.Timeout(connect=5, read=25)
    )
    
    if r.status != 200:
        err = r.data.decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f"Gemini API error {r.status}: {err}")

    resp = json.loads(r.data.decode('utf-8'))
    raw  = resp['candidates'][0]['content']['parts'][0]['text']
    data = json.loads(raw)
    data['ai_model_used'] = 'Gemini 3.5 Flash'
    return data


# ============================================================
#  CEREBRAS CALL (FALLBACK)
# ============================================================
def call_cerebras(system_prompt: str, user_content: str) -> dict | None:
    if not CEREBRAS_API_KEY:
        raise ValueError("CEREBRAS_API_KEY not set in environment")

    payload = {
        "model": "gpt-oss-120b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    r = http.request(
        'POST',
        'https://api.cerebras.ai/v1/chat/completions',
        body=json.dumps(payload),
        headers={
            'Authorization': f'Bearer {CEREBRAS_API_KEY}',
            'Content-Type': 'application/json'
        },
        timeout=urllib3.Timeout(connect=5, read=25)
    )
    
    if r.status != 200:
        err = r.data.decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f"Cerebras API error {r.status}: {err}")

    resp = json.loads(r.data.decode('utf-8'))
    raw  = resp['choices'][0]['message']['content']
    data = json.loads(raw)
    data['ai_model_used'] = 'Cerebras'
    return data


# ============================================================
#  SAMBANOVA CALL (FALLBACK)
# ============================================================
def call_sambanova(system_prompt: str, user_content: str) -> dict | None:
    if not SAMBANOVA_API_KEY:
        raise ValueError("SAMBANOVA_API_KEY not set in environment")

    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    r = http.request(
        'POST',
        'https://api.sambanova.ai/v1/chat/completions',
        body=json.dumps(payload),
        headers={
            'Authorization': f'Bearer {SAMBANOVA_API_KEY}',
            'Content-Type': 'application/json'
        },
        timeout=urllib3.Timeout(connect=5, read=25)
    )
    
    if r.status != 200:
        err = r.data.decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f"SambaNova API error {r.status}: {err}")

    resp = json.loads(r.data.decode('utf-8'))
    raw  = resp['choices'][0]['message']['content']
    data = json.loads(raw)
    data['ai_model_used'] = 'SambaNova (Llama 3.1 8B)'
    return data


# ============================================================
#  OPENROUTER CALL (FALLBACK)
# ============================================================
def call_openrouter(system_prompt: str, user_content: str) -> dict | None:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    payload = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    r = http.request(
        'POST',
        'https://openrouter.ai/api/v1/chat/completions',
        body=json.dumps(payload),
        headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://jobsnap.ai',
            'X-Title': 'JobSnap Lambda Pipeline'
        },
        timeout=urllib3.Timeout(connect=5, read=25)
    )
    
    if r.status != 200:
        err = r.data.decode('utf-8', errors='ignore')[:300]
        raise RuntimeError(f"OpenRouter API error {r.status}: {err}")

    resp = json.loads(r.data.decode('utf-8'))
    raw  = resp['choices'][0]['message']['content']
    data = json.loads(raw)
    data['ai_model_used'] = 'OpenRouter (Llama 3 8B)'
    return data


# ============================================================
#  PROVIDER MAP
# ============================================================
REAL_PROVIDERS = {
    "gemini": call_gemini,
    "groq": call_groq,
    "cerebras": call_cerebras,
    "sambanova": call_sambanova,
    "openrouter": call_openrouter,
}
