import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from machboost.providers import (
    ProviderBudgetError,
    ProviderError,
    ProviderStore,
    route_with_fallback,
)


class ProviderStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.calls = []

        def transport(config, path, payload, headers):
            self.calls.append((config, path, payload, headers))
            return {
                "id": "chatcmpl_test",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 25},
            }

        self.store = ProviderStore(
            Path(self.temporary.name) / "team.sqlite3",
            transport=transport,
            clock=lambda: 1_725_000_000.0,
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def configure(self, **overrides):
        values = {
            "provider_id": "primary",
            "name": "Primary",
            "base_url": "https://inference.example.com",
            "models": ["qwen-coder", "fallback-model"],
            "api_key": "top-secret",
            "monthly_budget_usd": 10.0,
            "input_cost_per_million": 2.0,
            "output_cost_per_million": 8.0,
        }
        values.update(overrides)
        return self.store.configure(**values)

    def test_configuration_never_persists_or_returns_secret(self):
        config = self.configure()
        raw = (Path(self.temporary.name) / "team.sqlite3").read_bytes()

        self.assertNotIn(b"top-secret", raw)
        listed = self.store.list()[0]
        self.assertTrue(listed["has_secret"])
        self.assertNotIn("api_key", listed)
        self.assertEqual(config.models, ("qwen-coder", "fallback-model"))

    def test_remote_http_is_rejected_but_loopback_is_allowed(self):
        with self.assertRaises(ValueError):
            self.configure(base_url="http://inference.example.com")
        config = self.configure(base_url="http://127.0.0.1:8000")
        self.assertEqual(config.base_url, "http://127.0.0.1:8000")

    def test_chat_forwards_auth_and_records_cost(self):
        self.configure()
        result = self.store.chat(
            {"model": "qwen-coder", "messages": [{"role": "user", "content": "hi"}]}
        )

        self.assertEqual(result.response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(result.prompt_tokens, 100)
        self.assertAlmostEqual(result.cost_usd, 0.0004)
        self.assertEqual(self.calls[0][1], "/chat/completions")
        self.assertEqual(self.calls[0][3]["Authorization"], "Bearer top-secret")
        usage = self.store.usage()["usage"][0]
        self.assertEqual(usage["requests"], 1)
        self.assertEqual(usage["prompt_tokens"], 100)

    def test_environment_secret_is_loaded_without_persistence(self):
        self.configure(api_key=None, api_key_env="MACHBOOST_TEST_PROVIDER_KEY")
        with patch.dict(os.environ, {"MACHBOOST_TEST_PROVIDER_KEY": "from-env"}):
            self.store.chat({"model": "qwen-coder", "messages": []})
        self.assertEqual(self.calls[0][3]["Authorization"], "Bearer from-env")

    def test_unknown_model_and_missing_secret_fail_closed(self):
        self.configure(api_key=None)
        with self.assertRaises(ProviderError) as unknown:
            self.store.chat({"model": "other", "messages": []})
        self.assertEqual(unknown.exception.status, 404)
        with self.assertRaises(ProviderError) as missing:
            self.store.chat({"model": "qwen-coder", "messages": []})
        self.assertEqual(missing.exception.status, 401)

    def test_monthly_budget_blocks_more_requests(self):
        self.configure(monthly_budget_usd=0.0)
        with self.assertRaises(ProviderBudgetError):
            self.store.chat({"model": "qwen-coder", "messages": []})

    def test_delete_removes_metadata_and_secret(self):
        self.configure()
        self.assertTrue(self.store.delete("primary"))
        self.assertFalse(self.store.delete("primary"))
        self.assertEqual(self.store.list(), [])


class RouteFallbackTests(unittest.TestCase):
    def test_modes_choose_expected_primary(self):
        self.assertEqual(
            route_with_fallback("local_only", local=lambda: "local", external=lambda: "remote"),
            ("local", "local"),
        )
        self.assertEqual(
            route_with_fallback(
                "external_first", local=lambda: "local", external=lambda: "remote"
            ),
            ("external", "remote"),
        )

    def test_transient_error_falls_back(self):
        source, value = route_with_fallback(
            "external_first",
            local=lambda: "local",
            external=lambda: (_ for _ in ()).throw(
                ProviderError("rate limited", status=429, transient=True)
            ),
        )
        self.assertEqual((source, value), ("local", "local"))

    def test_auth_and_budget_errors_do_not_fall_back(self):
        called = []
        with self.assertRaises(ProviderError):
            route_with_fallback(
                "external_first",
                local=lambda: called.append(True),
                external=lambda: (_ for _ in ()).throw(
                    ProviderError("unauthorized", status=401, transient=False)
                ),
            )
        self.assertEqual(called, [])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            route_with_fallback("random", local=lambda: None, external=lambda: None)


if __name__ == "__main__":
    unittest.main()
