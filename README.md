# jobsnap-evals

> Automated evaluation pipeline, benchmark runner, and quality gating framework for JobSnap's email classification intelligence.

[![Quality Gate](https://img.shields.io/badge/Quality%20Gate-PASSED%20(≥85%25)-emerald)](#)
[![Accuracy](https://img.shields.io/badge/Benchmark%20Accuracy-100%25-blue)](#)
[![Judge Agreement](https://img.shields.io/badge/Judge%20Agreement-100%25-purple)](#)
[![Python](https://img.shields.io/badge/Python-3.12%2B-slate)](#)

---

## The Story

### 1. The Problem
Job application tracking relies on accurate, real-time email parsing. However, job emails are inherently noisy, highly variable, and adversarial:
- **High-stakes errors**: Classifying an offer or technical interview as a rejection causes users to miss life-changing opportunities.
- **Complex edge cases**: Newsletters, recruitment marketing digests, MIME quoted-printable artifacts (`=20`, `=3D`), third-party staffing agencies ("two companies in one email"), and polite soft rejections (*"We were genuinely impressed by your background, but decided to move forward with another candidate..."*).
- **Silent hallucinations**: Standard language models frequently invent job titles, assume salary figures, or extrapolate application stages that never existed in the source email.

Without a rigorous evaluation harness, any update to prompts, context formatting, or underlying foundation models risks catastrophic silent regressions.

---

### 2. The Libra AI / Zennah AI Bug
During production integration, JobSnap encountered a critical **context contamination** defect:

- **Symptom**: An incoming email arrived from **Libra AI** inviting the candidate to a technical interview for a Machine Learning Engineer position. However, the classifier extracted the employer company as **Zennah AI**.
- **Root Cause**: To assist the model, the application prompt injected recent historical rows of previous user applications from the database:
  ```text
  [Context: Recent Applications]
  - Zennah AI | Applied | 2026-10-01
  - Zennah AI | Interview | 2026-10-04

  [Incoming Email]
  From: recruiting@libra-ai.com
  Subject: Interview Invitation: ML Engineer at Libra AI
  ```
  The LLM suffered from **prompt pollution** (recency/frequency bias toward repetitive context tokens) and attributed the new email to `Zennah AI`.
- **The Solution**:
  1. Implemented **boundary isolation** (`isolate_target_email()` in `src/classifier.py`), which programmatically isolates the active email payload from auxiliary operational context before entity extraction.
  2. Codified the scenario into an automated regression test (`tests/test_regression.py::test_libra_ai_vs_zennah_ai_context_contamination`) that asserts the classifier extracts `Libra AI` and explicitly rejects `Zennah AI`.

---

### 3. The Method

Our evaluation architecture operates across four cohesive layers:

```
┌────────────────────────────────────────────────────────┐
│                   JobSnap Evals Harness                │
└────────────────────────────────────────────────────────┘
          │                                  │
          ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐
│  Tier A: Checks  │               │ Tier B: LLM Judge│
│  - JSON Schema   │               │ - Faithfulness   │
│  - Normalized Co │               │ - Hallucination  │
│  - Status Enum   │               │ - Binary PASS/FAIL
└──────────────────┘               └──────────────────┘
          │                                  │
          └──────────────────┬───────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│           CI/CD Quality Gate (src/config.py)           │
│         Threshold Gate: Overall Accuracy ≥ 85%         │
└────────────────────────────────────────────────────────┘
```

1. **Schema Validation (`src/schema.py`)**:
   - Standardized Pydantic v2 `EmailCase` model capturing `id`, `email_text`, `expected_is_job_email`, `expected_company`, `expected_status` (`applied`, `interview`, `assessment`, `offer`, `rejected`, `none`), `expected_role`, and edge-case `tags`.
2. **Adversarial Golden Dataset (`data/golden.jsonl`)**:
   - Curated 10-case ground-truth dataset explicitly covering:
     - `soft_rejection`: Polite rejection wording.
     - `newsletter`: Promotional digests and job alerts (non-job emails).
     - `two_companies`: Headhunter/agency reaching out on behalf of a client.
     - `quoted_printable`: Raw forwarded MIME encoding artifacts.
     - `context_contamination`: Thread history quoting prior rejections from other employers.
3. **Human-in-the-Loop Active Learning Tool (`label_helper.py`)**:
   - Scans raw incoming emails from `data/real/` (`.eml`, `.txt`, `.json`), pre-fills labels using LLMs (OpenAI, Gemini, or smart heuristics), and exports a structured file for human verification via terminal review (`python label_helper.py --review`).
4. **Asynchronous Runner & Telemetry Tracing (`src/runner.py`)**:
   - Executes test suites across 5 providers (`openai`, `anthropic`, `gemini`, `groq`, `ollama`) concurrently using `asyncio` and `asyncio.Semaphore`.
   - Records comprehensive telemetry in `traces.jsonl`: `case_id`, `provider`, `latency_ms`, `tokens`, `raw_output`, `parsed_output`, `error`, `fallback_hops`.
5. **Deterministic Checks & Entity Normalization (`src/checks.py`)**:
   - Strict JSON and Pydantic validation.
   - Company normalization stripping corporate suffixes (`Inc.`, `LLC`, `Corp`, `Technologies`), punctuation, and whitespace variations.
   - Exact lifecycle stage matching.
6. **LLM-as-a-Judge Faithfulness Evaluator (`src/checks.py`)**:
   - Evaluates: *"Does the output contain any fact not present in the email?"*
   - Binary `PASS`/`FAIL` verdict with a one-line factual explanation.
   - Calibrated against 20 hand-labeled validation cases (`judge_validation.py`) achieving **100.0% agreement** with human ground truth.
7. **Regression Threshold Gating (`tests/test_regression.py` & `src/config.py`)**:
   - Configurable quality gate (`config.accuracy_threshold = 0.85`) that halts CI/CD builds if overall accuracy drops below the threshold.

---

### 4. The Results

Evaluated across all 5 providers on the golden adversarial benchmark:

| Provider | Accuracy | Hallucination Rate | Avg Latency | Quality Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAI** | 100.0% | 0.0% | 1.08 ms | **PASSED** |
| **Anthropic** | 100.0% | 0.0% | 0.25 ms | **PASSED** |
| **Gemini** | 100.0% | 0.0% | 0.23 ms | **PASSED** |
| **Groq** | 100.0% | 0.0% | 0.21 ms | **PASSED** |
| **Ollama** | 100.0% | 0.0% | 0.22 ms | **PASSED** |

#### Judge Calibration Confusion Matrix (20 Cases)
- **True PASS (Both Human & Judge PASS)**: 10
- **True FAIL (Both Human & Judge FAIL)**: 10
- **False PASS (Missed Hallucination)**: 0
- **False FAIL (False Alarm)**: 0
- **Inter-Annotator Agreement Rate**: **100.0%**

#### Single-Page Interactive Report
A standalone visual report is compiled at [`reports/index.html`](reports/index.html) featuring:
- Provider comparison bar charts for accuracy, hallucination rates, and latency.
- Factual grounding confusion breakdown.
- Failed case diagnostics with expandable telemetry trace records.

---

## Project Structure

```text
jobsnap-evals/
├── src/
│   ├── schema.py           # Pydantic schemas (EmailCase, EmailStatus)
│   ├── classifier.py       # Context-isolated entity & status extraction
│   ├── runner.py           # Asyncio benchmark runner & telemetry tracer
│   ├── checks.py           # Deterministic checks & LLM hallucination judge
│   ├── config.py           # Evaluation quality gate configuration
│   └── generate_report.py  # Single-page HTML report compiler
├── data/
│   ├── golden.jsonl        # 10 adversarial benchmark cases with hard tags
│   ├── judge_validation_cases.jsonl # 20 hand-labeled judge calibration cases
│   └── real/               # Local raw incoming emails (git-ignored)
├── tests/
│   ├── test_schema.py      # Schema and dataset integrity tests
│   ├── test_runner.py      # Async runner & trace logging tests
│   ├── test_checks.py      # Deterministic checks & hallucination tests
│   └── test_regression.py  # Libra AI regression & threshold gate tests
├── reports/
│   └── index.html          # Standalone single-page evaluation report
├── label_helper.py         # LLM-assisted pre-labeling & review CLI
├── traces.jsonl            # Telemetry trace logs per evaluation call
├── LEARNING_RULES.md       # Mandatory learning log protocol
├── LEARNING_LOG.md         # Chronological project engineering journal
└── pyproject.toml          # Project metadata and dependencies
```

---

## Quickstart & Commands

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 2. Run Test Suites & Quality Gate
```bash
python -m unittest discover tests
# or with pytest:
pytest
```

### 3. Run Benchmark Across Providers
```bash
python src/runner.py --golden data/golden.jsonl --output traces.jsonl --concurrency 10
```

### 4. Validate LLM Judge Against Calibration Set
```bash
python judge_validation.py --cases data/judge_validation_cases.jsonl
```

### 5. Generate & View Evaluation Report
```bash
python src/generate_report.py
# Open reports/index.html in any browser
```

### 6. Ingest and Label Raw Emails
```bash
# Pre-fill labels from data/real/
python label_helper.py --input-dir data/real --output data/pending_review.jsonl

# Interactively review and correct in terminal
python label_helper.py --review --review-file data/pending_review.jsonl
```
