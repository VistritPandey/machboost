from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from machboost.team import (
    TeamAccessError,
    TeamAdmissionController,
    TeamPrincipal,
    TeamStore,
    performance_evaluation,
)


class TeamStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = TeamStore(Path(self.temporary.name) / "team.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_key_is_returned_once_and_authenticates_by_hash(self) -> None:
        created = self.store.create_key(
            "VS Code",
            scopes=("inference", "traces:read"),
            allowed_models=("llama3.2:3b",),
            max_concurrent=3,
            requests_per_minute=90,
        )

        principal = self.store.authenticate(created.token)

        self.assertIsNotNone(principal)
        self.assertEqual(principal.name, "VS Code")
        self.assertEqual(principal.max_concurrent, 3)
        self.assertTrue(principal.permits("traces:read"))
        self.assertTrue(principal.permits_model("llama3.2:3b"))
        self.assertFalse(principal.permits_model("qwen3:8b"))
        self.assertNotIn(created.token, repr(self.store.list_keys()))

    def test_revoked_key_cannot_authenticate(self) -> None:
        created = self.store.create_key("Temporary")

        self.assertTrue(self.store.revoke_key(created.principal.id))
        self.assertIsNone(self.store.authenticate(created.token))

    def test_metadata_mode_omits_prompts_and_outputs(self) -> None:
        principal = self.store.create_key("Agent").principal
        started_at = time.time()
        trace_id = self.store.record_trace(
            request_id="request-1",
            principal=principal,
            endpoint="/v1/chat/completions",
            model="test-model",
            status="completed",
            started_at=started_at,
            finished_at=started_at + 2.0,
            input_data={"prompt": "private input"},
            output_text="private output",
        )

        trace = self.store.trace(trace_id)

        self.assertIsNone(trace["input"])
        self.assertIsNone(trace["output"])

    def test_redacted_mode_removes_common_secrets(self) -> None:
        self.store.update_settings(trace_mode="redacted")
        principal = self.store.create_key("Agent").principal
        started_at = time.time()
        trace_id = self.store.record_trace(
            request_id="request-2",
            principal=principal,
            endpoint="/api/chat",
            model="test-model",
            status="completed",
            started_at=started_at,
            finished_at=started_at + 1.0,
            input_data={"token": "abc", "prompt": "api_key=secret-value"},
            output_text="Authorization: Bearer abc.def",
        )

        trace = self.store.trace(trace_id)

        self.assertEqual(trace["input"]["token"], "[REDACTED]")
        self.assertNotIn("secret-value", trace["input"]["prompt"])
        self.assertNotIn("abc.def", trace["output"])

    def test_off_mode_writes_no_trace(self) -> None:
        self.store.update_settings(trace_mode="off")
        principal = self.store.create_key("Agent").principal

        trace_id = self.store.record_trace(
            request_id="request-3",
            principal=principal,
            endpoint="/api/chat",
            model="test-model",
            status="completed",
            started_at=100.0,
            finished_at=101.0,
        )

        self.assertIsNone(trace_id)
        self.assertEqual(self.store.status()["traces"], 0)

    def test_trace_retention_prunes_expired_rows(self) -> None:
        now = [1_000_000.0]
        self.store.close()
        self.store = TeamStore(
            Path(self.temporary.name) / "retention.sqlite3",
            clock=lambda: now[0],
        )
        principal = self.store.create_key("Agent").principal
        self.store.record_trace(
            request_id="old",
            principal=principal,
            endpoint="/api/chat",
            model="test-model",
            status="completed",
            started_at=1_000.0,
            finished_at=1_001.0,
        )

        self.assertEqual(self.store.status()["traces"], 0)

    def test_performance_evaluation_is_persisted(self) -> None:
        traces = [
            {
                "id": "trace-1",
                "status": "completed",
                "duration_seconds": 2.0,
                "time_to_first_token_seconds": 0.2,
                "completion_tokens": 20,
            },
            {
                "id": "trace-2",
                "status": "failed",
                "duration_seconds": 4.0,
                "time_to_first_token_seconds": 0.4,
                "completion_tokens": 0,
            },
        ]
        summary = performance_evaluation(traces)

        evaluation = self.store.create_evaluation(
            name="Weekly agent check",
            trace_ids=("trace-1", "trace-2"),
            evaluator="deterministic",
            summary=summary,
        )

        self.assertEqual(summary["completion_rate"], 0.5)
        self.assertEqual(summary["latency_seconds"]["p50"], 2.0)
        self.assertEqual(self.store.list_evaluations()[0]["id"], evaluation["id"])


class TeamAdmissionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = [10.0]
        self.controller = TeamAdmissionController(clock=lambda: self.now[0])
        self.principal = TeamPrincipal(
            id="key-1",
            name="Developer",
            scopes=("inference",),
            allowed_models=("allowed",),
            max_concurrent=1,
            requests_per_minute=2,
        )

    def test_enforces_scope_and_model_allowlist(self) -> None:
        denied = TeamPrincipal(
            id="key-2",
            name="Read only",
            scopes=("models:read",),
            allowed_models=(),
            max_concurrent=1,
            requests_per_minute=1,
        )

        with self.assertRaises(TeamAccessError) as scope_error:
            with self.controller.slot(denied, "allowed"):
                pass
        with self.assertRaises(TeamAccessError) as model_error:
            with self.controller.slot(self.principal, "denied"):
                pass

        self.assertEqual(scope_error.exception.reason, "scope_denied")
        self.assertEqual(model_error.exception.reason, "model_denied")

    def test_enforces_concurrent_and_per_minute_limits(self) -> None:
        with self.controller.slot(self.principal, "allowed"):
            with self.assertRaises(TeamAccessError) as concurrent:
                with self.controller.slot(self.principal, "allowed"):
                    pass
        with self.controller.slot(self.principal, "allowed"):
            pass
        with self.assertRaises(TeamAccessError) as rate:
            with self.controller.slot(self.principal, "allowed"):
                pass

        self.assertEqual(concurrent.exception.reason, "concurrency_limited")
        self.assertEqual(rate.exception.reason, "rate_limited")


if __name__ == "__main__":
    unittest.main()
