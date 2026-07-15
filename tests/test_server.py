from __future__ import annotations

import json
import threading
import unittest
from dataclasses import dataclass
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from machboost.server import MachBoostHTTPServer, RuntimeManager, parse_keep_alive


@dataclass
class FakeStats:
    generated_tokens: int = 2
    baseline_target_calls: int = 2
    target_calls: int = 1
    verify_calls: int = 1
    next_token_calls: int = 0
    candidate_rounds: int = 1
    accepted_draft_tokens: int = 2
    accepted_draft_spans: int = 1
    rejected_candidates: int = 0


class FakeService:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_cache(self) -> None:
        self.reset_calls += 1


class FakeAccelerator:
    def __init__(self) -> None:
        self.service = FakeService()
        self.chat_calls = []
        self.generate_calls = []

    def generate_chat(self, messages, *, max_tokens, context=None, on_text=None):
        self.chat_calls.append((messages, max_tokens, context))
        if on_text is not None:
            on_text("hello ")
            on_text("world")
        return "hello world", FakeStats()

    def generate(self, prompt, *, max_tokens, context=None, on_text=None):
        self.generate_calls.append((prompt, max_tokens, context))
        if on_text is not None:
            on_text("completed")
        return "completed", FakeStats(generated_tokens=1)


class FakeVisionAccelerator:
    def __init__(self) -> None:
        self.chat_calls = []
        self.generate_calls = []
        self.cold_vision_calls = []
        self.cache_hits = 0

    def generate_chat(
        self,
        messages,
        *,
        max_tokens,
        context=None,
        on_text=None,
        use_vision_cache=True,
        temperature=0.0,
        cold_vision_mode="off",
        cold_vision_max_edge=None,
    ):
        self.chat_calls.append((messages, max_tokens, use_vision_cache, temperature))
        self.cold_vision_calls.append((cold_vision_mode, cold_vision_max_edge))
        if use_vision_cache and len(self.chat_calls) > 1:
            self.cache_hits += 1
        if on_text is not None:
            on_text("blue square")
        return "blue square", FakeStats(generated_tokens=2)

    def generate(
        self,
        prompt,
        *,
        max_tokens,
        context=None,
        on_text=None,
        images=None,
        use_vision_cache=True,
        temperature=0.0,
        cold_vision_mode="off",
        cold_vision_max_edge=None,
    ):
        self.generate_calls.append((prompt, tuple(images or ()), use_vision_cache))
        self.cold_vision_calls.append((cold_vision_mode, cold_vision_max_edge))
        if on_text is not None:
            on_text("visual completion")
        return "visual completion", FakeStats(generated_tokens=2)

    def reset_cache(self):
        self.cache_hits = 0

    def cache_info(self):
        return {
            "size": 1 if self.chat_calls or self.generate_calls else 0,
            "max_size": 20,
            "hits": self.cache_hits,
            "misses": 1 if self.chat_calls or self.generate_calls else 0,
            "puts": 1 if self.chat_calls or self.generate_calls else 0,
            "evictions": 0,
        }


class FailingAccelerator(FakeAccelerator):
    def generate(self, prompt, *, max_tokens, context=None, on_text=None):
        raise RuntimeError("intentional streaming failure")


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RuntimeManagerTests(unittest.TestCase):
    def test_parse_keep_alive_supports_ollama_style_durations(self):
        self.assertEqual(parse_keep_alive("90s"), 90.0)
        self.assertEqual(parse_keep_alive("5m"), 300.0)
        self.assertEqual(parse_keep_alive("2h"), 7200.0)
        self.assertEqual(parse_keep_alive("forever"), -1.0)

    def test_model_is_loaded_once_and_reused_until_stopped(self):
        loaded = []

        def loader(config):
            loaded.append(config)
            return FakeAccelerator()

        manager = RuntimeManager(loader=loader)
        messages = [{"role": "user", "content": "hello"}]

        first = manager.chat("mlx-community/example", messages, options={"num_predict": 9})
        second = manager.chat("mlx-community/example", messages, options={"num_predict": 9})

        self.assertEqual(len(loaded), 1)
        self.assertEqual(first.text, "hello world")
        self.assertEqual(second.load_duration_s, 0.0)
        self.assertEqual(manager.ps()[0]["requests"], 2)
        self.assertEqual(manager.ps()[0]["keep_alive_seconds"], -1.0)
        self.assertEqual(manager.stop("mlx-community/example"), 1)
        self.assertEqual(manager.ps(), [])

    def test_finite_keep_alive_evicts_idle_model(self):
        clock = FakeClock()
        manager = RuntimeManager(loader=lambda config: FakeAccelerator(), clock=clock)
        manager.get_or_load("mlx-community/example", keep_alive="5s")

        clock.advance(4.0)
        self.assertEqual(len(manager.ps()), 1)
        clock.advance(2.0)
        self.assertEqual(manager.ps(), [])

    def test_model_configuration_separates_context_indexes(self):
        loaded = []
        manager = RuntimeManager(loader=lambda config: loaded.append(config) or FakeAccelerator())

        manager.get_or_load("mlx-community/example", options={"context_paths": ["docs"]})
        manager.get_or_load("mlx-community/example", options={"context_paths": ["src"]})

        self.assertEqual(len(loaded), 2)
        self.assertEqual({config.context_paths for config in loaded}, {("docs",), ("src",)})

    def test_stopping_alias_unloads_mlx_and_hf_variants(self):
        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        manager.get_or_load("qwen2.5:3b", options={"backend": "mlx"})
        manager.get_or_load("qwen2.5:3b", options={"backend": "hf"})

        self.assertEqual(len(manager.ps()), 2)
        self.assertEqual(manager.stop("qwen2.5:3b"), 2)
        self.assertEqual(manager.ps(), [])


class HTTPServerTests(unittest.TestCase):
    def setUp(self):
        self.loaded = []
        manager = RuntimeManager(loader=lambda config: self._load(config))
        self.server = MachBoostHTTPServer(("127.0.0.1", 0), manager=manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def _load(self, config):
        if config.model.endswith("failing"):
            accelerator = FailingAccelerator()
        else:
            accelerator = FakeVisionAccelerator() if config.backend.endswith("-vlm") else FakeAccelerator()
        self.loaded.append((config, accelerator))
        return accelerator

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urlopen(request, timeout=3.0) as response:
            return response.status, response.headers, response.read().decode("utf-8")

    def test_health_and_ps_endpoints(self):
        status, _, body = self.request("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

        self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        _, _, body = self.request("/api/ps")
        self.assertEqual(json.loads(body)["models"][0]["model"], "mlx-community/example")

    def test_ollama_chat_supports_non_streaming_and_reuses_model(self):
        payload = {
            "model": "mlx-community/example",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "options": {"num_predict": 11},
        }

        _, _, first_body = self.request("/api/chat", payload)
        _, _, second_body = self.request("/api/chat", payload)

        first = json.loads(first_body)
        second = json.loads(second_body)
        self.assertEqual(first["message"]["content"], "hello world")
        self.assertTrue(first["done"])
        self.assertEqual(first["machboost"]["stats"]["accepted_draft_tokens"], 2)
        self.assertEqual(second["load_duration"], 0)
        self.assertEqual(len(self.loaded), 1)
        self.assertEqual(self.loaded[0][1].chat_calls[0][1], 11)

    def test_load_endpoint_preloads_without_generating(self):
        _, _, body = self.request(
            "/api/load",
            {
                "model": "qwen2.5:3b",
                "keep_alive": "1h",
                "options": {"backend": "mlx"},
            },
        )

        response = json.loads(body)
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["instance"]["model"], "mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.assertEqual(response["instance"]["keep_alive_seconds"], 3600.0)
        self.assertEqual(self.loaded[0][1].chat_calls, [])
        self.assertEqual(self.loaded[0][1].generate_calls, [])

    def test_ollama_chat_streams_ndjson_chunks(self):
        _, headers, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        rows = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(headers.get_content_type(), "application/x-ndjson")
        self.assertEqual("".join(row["message"]["content"] for row in rows), "hello world")
        self.assertFalse(rows[0]["done"])
        self.assertTrue(rows[-1]["done"])

    def test_openai_completion_endpoint(self):
        _, _, body = self.request(
            "/v1/completions",
            {"model": "mlx-community/example", "prompt": "def add", "max_tokens": 16},
        )

        response = json.loads(body)
        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "completed")
        self.assertEqual(self.loaded[0][1].generate_calls[0][1], 16)

    def test_stop_endpoint_unloads_resident_model(self):
        self.request(
            "/api/generate",
            {"model": "mlx-community/example", "prompt": "hello", "stream": False},
        )
        _, _, body = self.request("/api/stop", {"model": "mlx-community/example"})

        self.assertEqual(json.loads(body)["unloaded"], 1)
        _, _, ps_body = self.request("/api/ps")
        self.assertEqual(json.loads(ps_body)["models"], [])

    def test_ollama_multimodal_chat_routes_to_vision_backend(self):
        payload = {
            "model": "qwen2.5-vl:3b",
            "messages": [
                {
                    "role": "user",
                    "content": "What shape is shown?",
                    "images": ["iVBORw0KGgo="],
                }
            ],
            "stream": False,
            "options": {"num_predict": 12, "temperature": 0.0},
        }

        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            _, _, first_body = self.request("/api/chat", payload)
            _, _, second_body = self.request("/api/chat", payload)
            _, _, ps_body = self.request("/api/ps")

        self.assertEqual(json.loads(first_body)["message"]["content"], "blue square")
        self.assertTrue(json.loads(second_body)["machboost"]["stats"]["generated_tokens"] > 0)
        config, accelerator = self.loaded[0]
        self.assertEqual(config.backend, "mlx-vlm")
        self.assertEqual(accelerator.chat_calls[0][0][0]["images"], ["iVBORw0KGgo="])
        model = json.loads(ps_body)["models"][0]
        self.assertEqual(model["capabilities"], ["vision", "chat"])
        self.assertEqual(model["vision_cache"]["hits"], 1)

    def test_openai_multimodal_content_parts_are_preserved(self):
        payload = {
            "model": "qwen2.5-vl:3b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            "max_tokens": 8,
        }

        _, _, body = self.request("/v1/chat/completions", payload)

        response = json.loads(body)
        self.assertEqual(response["choices"][0]["message"]["content"], "blue square")
        content = self.loaded[0][1].chat_calls[0][0][0]["content"]
        self.assertEqual(content[1]["type"], "image_url")

    def test_ollama_generate_forwards_images_and_cache_control(self):
        _, _, body = self.request(
            "/api/generate",
            {
                "model": "qwen2.5-vl:3b",
                "prompt": "Describe this.",
                "images": ["image-one"],
                "stream": False,
                "options": {"no_vision_cache": True},
            },
        )

        self.assertEqual(json.loads(body)["response"], "visual completion")
        self.assertEqual(self.loaded[0][1].generate_calls[0], ("Describe this.", ("image-one",), False))

    def test_ollama_generate_forwards_cold_vision_options(self):
        self.request(
            "/api/generate",
            {
                "model": "qwen2.5-vl:3b",
                "prompt": "Read the label.",
                "images": ["image-one"],
                "stream": False,
                "options": {
                    "cold_vision": "adaptive",
                    "vision_max_edge": 512,
                },
            },
        )

        self.assertEqual(self.loaded[0][1].cold_vision_calls[0], ("adaptive", 512))

    def test_text_backend_rejects_image_payload(self):
        with self.assertRaises(HTTPError) as raised:
            self.request(
                "/api/chat",
                {
                    "model": "mlx-community/example",
                    "messages": [{"role": "user", "content": "Look", "images": ["image"]}],
                    "stream": False,
                },
            )

        self.assertEqual(raised.exception.code, 400)
        self.assertIn("vision model", raised.exception.read().decode("utf-8"))

    def test_streaming_failure_is_returned_as_ndjson_error(self):
        _, _, body = self.request(
            "/api/generate",
            {
                "model": "mlx-community/failing",
                "prompt": "fail",
                "stream": True,
            },
        )

        row = json.loads(body)
        self.assertEqual(row["error"], "intentional streaming failure")
        self.assertTrue(row["done"])

    def test_shutdown_endpoint_stops_server_and_releases_models(self):
        self.request(
            "/api/generate",
            {"model": "mlx-community/example", "prompt": "hello", "stream": False},
        )
        _, _, body = self.request("/api/shutdown", {})

        self.assertEqual(json.loads(body)["unloaded"], 1)
        self.thread.join(timeout=2.0)
        self.assertFalse(self.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
