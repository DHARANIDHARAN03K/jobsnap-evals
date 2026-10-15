"""
src/runner.py: Asynchronous benchmark runner.

Calls each of the 5 real providers (gemini, groq, cerebras, sambanova, openrouter)
directly and independently for every golden case, writing one trace per call to
traces.jsonl.

Design:
- call_* functions from classifier.py are blocking (urllib3), so each is wrapped
  with asyncio.to_thread so the event loop stays alive.
- Per-provider semaphore limits concurrency (default: 3 per provider).
- HTTP 429 is retried once after a 2-second sleep.
- --waterfall mode replays the Lambda fallback chain in order and records
  fallback_hops (stops at first success).
- All failures (auth errors, model-not-found, timeouts) are caught and recorded
  in the trace `error` field; the run never crashes.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema import EmailCase
from src.classifier import (
    REAL_PROVIDERS,
    build_groq_system_prompt,
    extract_company_from_email,
    extract_sender_domain,
    is_aggregator_domain,
    truncate_at_noise_zone,
)

# ── Provider ordering for --waterfall mode (Lambda production order) ──────────
WATERFALL_ORDER = ["gemini", "groq", "cerebras", "sambanova", "openrouter"]

# ── Default providers for normal (parallel) mode ─────────────────────────────
DEFAULT_PROVIDERS = list(REAL_PROVIDERS.keys())


# =============================================================================
# TRACE RECORD
# =============================================================================

class TraceRecord:
    """One execution trace: one provider × one case."""

    __slots__ = (
        "case_id", "provider", "latency_ms", "tokens",
        "raw_output", "parsed_output", "error", "fallback_hops",
    )

    def __init__(
        self,
        case_id: str,
        provider: str,
        latency_ms: float,
        tokens: Any,
        raw_output: str | None,
        parsed_output: dict[str, Any] | None,
        error: str | None,
        fallback_hops: int = 0,
    ):
        self.case_id = case_id
        self.provider = provider
        self.latency_ms = round(latency_ms, 2)
        self.tokens = tokens
        self.raw_output = raw_output
        self.parsed_output = parsed_output
        self.error = error
        self.fallback_hops = fallback_hops

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "error": self.error,
            "fallback_hops": self.fallback_hops,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt_for_case(case: EmailCase) -> tuple[str, str]:
    """Return (system_prompt, user_content) for a golden case."""
    body_clean = truncate_at_noise_zone(case.body)
    subject_company_hint = extract_company_from_email(case.subject, body_clean)
    sender_domain = case.sender_domain or extract_sender_domain(case.sender)
    is_agg = is_aggregator_domain(sender_domain)

    sys_prompt = build_groq_system_prompt(
        subject_company_hint=subject_company_hint,
        existing_jobs_context=case.existing_jobs_context,
        sender_domain=sender_domain,
        is_aggregator=is_agg,
    )
    user_content = f"Subject: {case.subject}\n\nBody:\n{body_clean[:4000]}"
    return sys_prompt, user_content


# =============================================================================
# SINGLE PROVIDER CALL (async wrapper around blocking call_*)
# =============================================================================

async def _call_provider_once(
    provider: str,
    sys_prompt: str,
    user_content: str,
) -> tuple[dict | None, str | None]:
    """
    Call one provider in a thread.  Returns (data, error_str).
    Retries once on HTTP 429.
    """
    call_fn = REAL_PROVIDERS[provider]
    error: str | None = None
    data: dict | None = None

    for attempt in range(2):
        try:
            data = await asyncio.to_thread(call_fn, sys_prompt, user_content)
            error = None
            break
        except RuntimeError as exc:
            msg = str(exc)
            if "429" in msg and attempt == 0:
                await asyncio.sleep(2)
                continue
            error = f"{type(exc).__name__}: {msg}"
            data = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)}"
            data = None
            break

    return data, error


# =============================================================================
# EXECUTE ONE CASE × ONE PROVIDER  (parallel mode)
# =============================================================================

async def execute_case_provider(
    case: EmailCase,
    provider: str,
    semaphore: asyncio.Semaphore,
) -> TraceRecord:
    """Run one case against one provider, guarded by semaphore."""
    async with semaphore:
        sys_prompt, user_content = build_prompt_for_case(case)

        t0 = time.perf_counter()
        data, error = await _call_provider_once(provider, sys_prompt, user_content)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_output = json.dumps(data, ensure_ascii=False) if data else None
        tokens = _extract_tokens(data)

        return TraceRecord(
            case_id=case.id,
            provider=provider,
            latency_ms=latency_ms,
            tokens=tokens,
            raw_output=raw_output,
            parsed_output=data,
            error=error,
            fallback_hops=0,
        )


# =============================================================================
# EXECUTE ONE CASE IN WATERFALL MODE
# =============================================================================

async def execute_case_waterfall(
    case: EmailCase,
    semaphores: dict[str, asyncio.Semaphore],
) -> TraceRecord:
    """
    Try providers in WATERFALL_ORDER, stop at first success.
    Record total fallback_hops and which provider finally succeeded.
    """
    sys_prompt, user_content = build_prompt_for_case(case)
    last_error: str | None = None
    hops = 0

    t0_total = time.perf_counter()

    for provider in WATERFALL_ORDER:
        sem = semaphores.get(provider, asyncio.Semaphore(3))
        async with sem:
            t0 = time.perf_counter()
            data, error = await _call_provider_once(provider, sys_prompt, user_content)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if data is not None:
                total_latency = (time.perf_counter() - t0_total) * 1000.0
                return TraceRecord(
                    case_id=case.id,
                    provider=provider,
                    latency_ms=total_latency,
                    tokens=_extract_tokens(data),
                    raw_output=json.dumps(data, ensure_ascii=False),
                    parsed_output=data,
                    error=None,
                    fallback_hops=hops,
                )
            last_error = error
            hops += 1

    # All providers failed
    total_latency = (time.perf_counter() - t0_total) * 1000.0
    return TraceRecord(
        case_id=case.id,
        provider="all_failed",
        latency_ms=total_latency,
        tokens=None,
        raw_output=None,
        parsed_output=None,
        error=last_error,
        fallback_hops=hops,
    )


# =============================================================================
# TOKEN EXTRACTION HELPER
# =============================================================================

def _extract_tokens(data: dict | None) -> dict | None:
    """Try to pull token usage from the provider response dict."""
    if not data:
        return None
    usage = data.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

async def run_evaluation(
    golden_path: Path = Path("data/golden.jsonl"),
    traces_path: Path = Path("traces.jsonl"),
    providers: list[str] | None = None,
    concurrency_per_provider: int = 3,
    waterfall: bool = False,
) -> list[TraceRecord]:
    """
    Run all golden cases against all providers.
    In waterfall mode, each case goes through providers in Lambda order.
    In parallel mode, all providers run concurrently per case.
    """
    if providers is None:
        providers = DEFAULT_PROVIDERS

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    cases: list[EmailCase] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(EmailCase.model_validate_json(line))

    print(f"Loaded {len(cases)} cases from {golden_path}")

    traces: list[TraceRecord] = []

    if waterfall:
        print(f"Mode: WATERFALL ({' → '.join(WATERFALL_ORDER)})")
        semaphores = {p: asyncio.Semaphore(concurrency_per_provider) for p in WATERFALL_ORDER}
        tasks = [execute_case_waterfall(case, semaphores) for case in cases]
        print(f"Total calls (waterfall, max {len(WATERFALL_ORDER)} per case): {len(cases)}")
        traces = await asyncio.gather(*tasks)
    else:
        print(f"Mode: PARALLEL — providers: {', '.join(providers)}")
        print(f"Total calls: {len(cases)} × {len(providers)} = {len(cases) * len(providers)}")
        semaphores = {p: asyncio.Semaphore(concurrency_per_provider) for p in providers}
        tasks = []
        for case in cases:
            for provider in providers:
                sem = semaphores[provider]
                tasks.append(execute_case_provider(case, provider, sem))
        traces = await asyncio.gather(*tasks)

    # Write traces
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(trace.to_json() + "\n")

    print(f"\n[Done] Wrote {len(traces)} traces → {traces_path}")
    _print_summary(traces)
    return traces


def _print_summary(traces: list[TraceRecord]) -> None:
    """Print per-provider success / error counts to stdout."""
    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0})
    for t in traces:
        if t.error:
            counts[t.provider]["err"] += 1
        else:
            counts[t.provider]["ok"] += 1
    print("\nPer-provider results:")
    for provider, c in sorted(counts.items()):
        total = c["ok"] + c["err"]
        print(f"  {provider:15s}  ok={c['ok']}/{total}  errors={c['err']}/{total}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run jobsnap-evals benchmark across real LLM providers."
    )
    parser.add_argument("--golden", type=Path, default=Path("data/golden.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("traces.jsonl"))
    parser.add_argument(
        "--providers", nargs="+", default=DEFAULT_PROVIDERS,
        help=f"Providers to use (default: {' '.join(DEFAULT_PROVIDERS)})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Semaphore size per provider (default: 3)",
    )
    parser.add_argument(
        "--waterfall", action="store_true",
        help="Replay Lambda waterfall order; stop at first success per case",
    )

    args = parser.parse_args()
    asyncio.run(
        run_evaluation(
            golden_path=args.golden,
            traces_path=args.output,
            providers=args.providers,
            concurrency_per_provider=args.concurrency,
            waterfall=args.waterfall,
        )
    )


if __name__ == "__main__":
    main()
