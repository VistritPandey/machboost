import json
import unittest

from machboost.adapters.ollama import (
    OllamaHTTPAdapter,
    OllamaHTTPError,
    normalize_endpoint,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class RecordingOpener:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.requests = []
        self.timeouts = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        return FakeResponse(self.payload, status=self.status)


class OllamaAdapterTest(unittest.TestCase):
    def test_normalize_endpoint_adds_scheme_and_removes_trailing_slash(self):
        self.assertEqual(normalize_endpoint("127.0.0.1:11434/"), "http://127.0.0.1:11434")
        self.assertEqual(normalize_endpoint("https://example.test/"), "https://example.test")

    def test_generate_posts_non_streaming_payload_and_reports_tps(self):
        opener = RecordingOpener(
            {
                "model": "qwen2.5:3b",
                "response": "ok",
                "done": True,
                "eval_count": 32,
                "eval_duration": 2_000_000_000,
                "prompt_eval_count": 8,
                "prompt_eval_duration": 1_000_000_000,
                "load_duration": 5_000_000,
                "total_duration": 3_000_000_000,
            }
        )
        adapter = OllamaHTTPAdapter(
            "qwen2.5:3b",
            endpoint="127.0.0.1:11434",
            timeout=7,
            default_options={"num_ctx": 2048, "temperature": 0.8},
            opener=opener,
        )

        result = adapter.generate("hello", options={"temperature": 0.0, "num_predict": 16})

        self.assertEqual(result.response, "ok")
        self.assertEqual(result.tokens_per_second, 16.0)
        self.assertEqual(result.prompt_tokens_per_second, 8.0)
        self.assertEqual(result.total_ms, 3000.0)
        self.assertEqual(opener.timeouts, [7])
        req = opener.requests[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(req.get_method(), "POST")
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(payload["prompt"], "hello")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], -1)
        self.assertEqual(payload["options"], {"num_ctx": 2048, "temperature": 0.0, "num_predict": 16})

    def test_benchmark_sets_token_and_context_options(self):
        opener = RecordingOpener({"model": "qwen2.5:3b", "response": "", "done": True})
        adapter = OllamaHTTPAdapter("qwen2.5:3b", endpoint="http://localhost:11434", opener=opener)

        adapter.benchmark("bench me", tokens=12, ctx=1024)

        payload = json.loads(opener.requests[0].data.decode("utf-8"))
        self.assertEqual(payload["options"]["num_predict"], 12)
        self.assertEqual(payload["options"]["num_ctx"], 1024)

    def test_tags_uses_get(self):
        opener = RecordingOpener({"models": [{"name": "qwen2.5:3b"}]})
        adapter = OllamaHTTPAdapter("qwen2.5:3b", endpoint="http://localhost:11434", opener=opener)

        tags = adapter.tags()

        self.assertEqual(tags["models"][0]["name"], "qwen2.5:3b")
        self.assertEqual(opener.requests[0].full_url, "http://localhost:11434/api/tags")
        self.assertEqual(opener.requests[0].get_method(), "GET")

    def test_with_draft_options_sets_ollama_draft_depth(self):
        options = OllamaHTTPAdapter.with_draft_options({"num_predict": 32}, draft_num_predict=8)

        self.assertEqual(options, {"num_predict": 32, "draft_num_predict": 8})

    def test_capabilities_are_wrapper_only(self):
        adapter = OllamaHTTPAdapter("qwen2.5:3b", opener=RecordingOpener({}))
        caps = adapter.capabilities()

        self.assertFalse(caps.native_verification)
        self.assertFalse(caps.token_level_api)
        self.assertEqual(caps.acceleration_mode, "wrapper")
        with self.assertRaises(NotImplementedError):
            adapter.require_native_verifier()

    def test_non_success_status_raises_http_error(self):
        opener = RecordingOpener({"error": "model not found"}, status=404)
        adapter = OllamaHTTPAdapter("missing:model", opener=opener)

        with self.assertRaises(OllamaHTTPError):
            adapter.generate("hello")


if __name__ == "__main__":
    unittest.main()
