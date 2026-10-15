"""
Tests for src/runner.py: Asynchronous evaluation execution and trace logging.
"""

import asyncio
import json
from pathlib import Path
import unittest

from src.schema import EmailCase
from src.runner import run_evaluation, TraceRecord, execute_case_with_provider


class TestRunner(unittest.IsolatedAsyncioTestCase):
    async def test_trace_record_structure(self):
        record = TraceRecord(
            case_id="case_001",
            provider="openai",
            latency_ms=123.456,
            tokens={"prompt": 50, "completion": 20, "total": 70},
            raw_output='{"status": "applied"}',
            parsed_output={"status": "applied"},
            error=None,
            fallback_hops=0,
        )
        d = record.to_dict()
        self.assertEqual(d["case_id"], "case_001")
        self.assertEqual(d["provider"], "openai")
        self.assertEqual(d["latency_ms"], 123.46)
        self.assertEqual(d["fallback_hops"], 0)
        self.assertIsNone(d["error"])

    async def test_execute_case_with_provider_error_handling(self):
        def failing_classify(text, provider):
            raise ConnectionError("Simulated network timeout")

        case = EmailCase(
            id="test_err_01",
            email_text="Sample text",
            expected_is_job_email=True,
            expected_status="applied",
        )

        trace = await execute_case_with_provider(
            case=case,
            provider="test_prov",
            classify_fn=failing_classify,
        )

        self.assertEqual(trace.case_id, "test_err_01")
        self.assertEqual(trace.provider, "test_prov")
        self.assertIn("ConnectionError: Simulated network timeout", trace.error)
        self.assertGreater(trace.latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
