"""
judge_validation.py: Validate the Gemini LLM judge against hand-labeled examples.

How it works:
1. Load traces from traces.jsonl (produced by runner.py).
2. Load human labels from data/judge_labels.jsonl (you must create this).
3. For each labeled trace, ask the Gemini judge: PASS or FAIL?
4. Compare judge verdict to your label and report agreement %.

data/judge_labels.jsonl format (one JSON object per line):
  {
    "case_id": "case_001",
    "provider": "gemini",
    "human_verdict": "PASS"   // or "FAIL"
  }

Run:
  python judge_validation.py --traces traces.jsonl --labels data/judge_labels.jsonl
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.checks import judge_hallucination
from src.schema import EmailCase


def load_golden_map(golden_path: Path) -> dict[str, EmailCase]:
    """Load golden cases into a dict keyed by case_id."""
    cases = {}
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = EmailCase.model_validate_json(line)
                cases[c.id] = c
    return cases


def load_traces_map(traces_path: Path) -> dict[tuple[str, str], dict]:
    """Load traces into a dict keyed by (case_id, provider)."""
    traces = {}
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                traces[(t["case_id"], t["provider"])] = t
    return traces


def load_labels(labels_path: Path) -> list[dict]:
    """Load human-labeled verdicts."""
    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    return labels


def run_validation(
    golden_path: Path = Path("data/golden.jsonl"),
    traces_path: Path = Path("traces.jsonl"),
    labels_path: Path = Path("data/judge_labels.jsonl"),
) -> None:
    # Verify files exist
    for p in (golden_path, traces_path, labels_path):
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    golden = load_golden_map(golden_path)
    traces = load_traces_map(traces_path)
    labels = load_labels(labels_path)

    if not labels:
        print("[ERROR] No labels found. Please populate data/judge_labels.jsonl first.")
        sys.exit(1)

    print(f"Loaded {len(labels)} hand-labeled cases from {labels_path}\n")

    agreement = 0
    disagreement = 0
    skipped = 0
    rows = []

    for label in labels:
        case_id = label.get("case_id", "")
        provider = label.get("provider", "gemini")
        human_verdict = str(label.get("human_verdict", "")).upper()

        if human_verdict not in ("PASS", "FAIL"):
            print(f"  [SKIP] {case_id}/{provider} — invalid human_verdict '{human_verdict}'")
            skipped += 1
            continue

        trace = traces.get((case_id, provider))
        if not trace:
            print(f"  [SKIP] {case_id}/{provider} — no trace found in {traces_path}")
            skipped += 1
            continue

        case = golden.get(case_id)
        if not case:
            print(f"  [SKIP] {case_id} — not in golden.jsonl")
            skipped += 1
            continue

        parsed_output = trace.get("parsed_output") or {}

        judge_result = judge_hallucination(
            email_subject=case.subject,
            email_body=case.body,
            output=parsed_output,
        )
        judge_verdict = judge_result.verdict

        match = (judge_verdict == human_verdict)
        if match:
            agreement += 1
            icon = "✅"
        else:
            disagreement += 1
            icon = "❌"

        rows.append({
            "case_id": case_id,
            "provider": provider,
            "human": human_verdict,
            "judge": judge_verdict,
            "match": match,
            "reason": judge_result.reason,
        })
        print(f"  {icon} [{case_id}/{provider}]  human={human_verdict}  judge={judge_verdict}")
        print(f"       reason: {judge_result.reason}")

    total_evaluated = agreement + disagreement
    if total_evaluated == 0:
        print("\n[ERROR] No valid labels evaluated — check your labels file.")
        return

    pct = agreement / total_evaluated * 100
    print(f"\n{'─'*60}")
    print(f"Judge–Human Agreement: {agreement}/{total_evaluated} = {pct:.1f}%")
    print(f"Skipped: {skipped}")
    print(f"{'─'*60}")

    # Write results to file
    out_path = Path("reports/judge_validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_labeled": len(labels),
            "evaluated": total_evaluated,
            "skipped": skipped,
            "agreement": agreement,
            "disagreement": disagreement,
            "agreement_pct": round(pct, 1),
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {out_path}")


def _create_labels_template(golden_path: Path, traces_path: Path, out_path: Path) -> None:
    """
    Helper: generate a starter labels file from existing traces so you can
    fill in human_verdict = PASS or FAIL without typing case IDs manually.
    Run: python judge_validation.py --create-template
    """
    golden = load_golden_map(golden_path)
    traces = load_traces_map(traces_path)

    rows = []
    for (case_id, provider), trace in traces.items():
        if trace.get("error") or not trace.get("parsed_output"):
            continue  # skip failed traces
        rows.append({
            "case_id": case_id,
            "provider": provider,
            "human_verdict": "PASS",  # <-- you fill this in
            "_note": "Change PASS/FAIL based on whether the output looks faithful",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Template written → {out_path}  ({len(rows)} rows)")
    print("Fill in 'human_verdict' for each row then run judge_validation.py normally.")


def main():
    parser = argparse.ArgumentParser(description="Validate the LLM judge against hand labels.")
    parser.add_argument("--golden", type=Path, default=Path("data/golden.jsonl"))
    parser.add_argument("--traces", type=Path, default=Path("traces.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/judge_labels.jsonl"))
    parser.add_argument(
        "--create-template", action="store_true",
        help="Generate a template labels file from existing traces (then you fill in verdicts)"
    )
    args = parser.parse_args()

    if args.create_template:
        _create_labels_template(args.golden, args.traces, args.labels)
    else:
        run_validation(args.golden, args.traces, args.labels)


if __name__ == "__main__":
    main()
