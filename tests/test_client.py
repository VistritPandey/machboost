from __future__ import annotations

import os
import io
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from machboost import __version__
from machboost.client import (
    MachBoostAPIError,
    MachBoostClient,
    api_error,
    default_endpoint,
    ensure_server,
)
from machboost.server import MachBoostHTTPServer, RuntimeManager


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


class FakeAccelerator:
    service = None

    def generate_chat(self, messages, *, max_tokens, context=None, on_text=None):
        if on_text is not None:
            on_text("warm ")
            on_text("chat")
        return "warm chat", FakeStats()

    def generate(self, prompt, *, max_tokens, context=None, on_text=None):
        if on_text is not None:
            on_text("completion")
        return "completion", FakeStats(generated_tokens=1)


class ClientTests(unittest.TestCase):
    def setUp(self):
        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        self.server = MachBoostHTTPServer(("127.0.0.1", 0), manager=manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = MachBoostClient(f"http://{host}:{port}", timeout=3.0)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_default_endpoint_uses_environment(self):
        with patch.dict(os.environ, {"MACHBOOST_HOST": "localhost:9999"}):
            self.assertEqual(default_endpoint(), "http://localhost:9999")

    def test_client_streams_chat_and_lists_warm_model(self):
        rows = list(
            self.client.chat(
                "mlx-community/example",
                [{"role": "user", "content": "hello"}],
                stream=True,
            )
        )

        self.assertEqual("".join(row["message"]["content"] for row in rows), "warm chat")
        self.assertTrue(rows[-1]["done"])
        self.assertEqual(self.client.ps()[0]["requests"], 1)

    def test_client_exposes_catalog_and_runtime_metrics(self):
        catalog = self.client.catalog()
        metrics = self.client.metrics()

        self.assertTrue(any(model["name"] == "llama3.2:3b" for model in catalog))
        self.assertEqual(metrics["schema"], "machboost.metrics.v1")
        self.assertIn("active_count", metrics["operations"])

    def test_client_forwards_request_identifier(self):
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.chat(
                "mlx-community/example",
                [{"role": "user", "content": "hello"}],
                request_id="desktop-message-42",
                stream=False,
            )

        self.assertEqual(post.call_args.args[1]["request_id"], "desktop-message-42")

    def test_client_adds_bearer_token_to_every_request(self):
        client = MachBoostClient(self.client.endpoint, api_token="secret-token")

        self.assertEqual(
            client._headers()["Authorization"],
            "Bearer secret-token",
        )

    def test_cancel_returns_false_for_unknown_request(self):
        self.assertFalse(self.client.cancel("missing-request"))

    def test_client_supports_non_streaming_generation_and_stop(self):
        response = self.client.generate("mlx-community/example", "def add", stream=False)

        self.assertEqual(response["response"], "completion")
        self.assertEqual(self.client.stop("mlx-community/example")["unloaded"], 1)
        self.assertEqual(self.client.ps(), [])

    def test_client_can_preload_model_without_generation(self):
        response = self.client.load(
            "qwen2.5:3b",
            options={"backend": "mlx"},
            keep_alive="2h",
        )

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["instance"]["keep_alive_seconds"], 7200.0)
        self.assertEqual(response["instance"]["requests"], 0)

    def test_http_errors_become_api_errors(self):
        with self.assertRaisesRegex(MachBoostAPIError, "missing required field"):
            self.client.post("/api/chat", {"messages": []})

    def test_health_treats_connection_reset_as_not_running(self):
        with patch("machboost.client.urlopen", side_effect=ConnectionResetError("reset")):
            self.assertFalse(self.client.is_healthy())

    def test_chat_attaches_images_to_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "What is shown?"},
        ]
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            response = self.client.chat(
                "qwen2.5-vl:3b",
                messages,
                images=["image-a", "image-b"],
                stream=False,
            )

        self.assertTrue(response["done"])
        payload = post.call_args.args[1]
        self.assertEqual(payload["messages"][2]["images"], ["image-a", "image-b"])
        self.assertNotIn("images", messages[2])

    def test_generate_forwards_images(self):
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.generate(
                "qwen2.5-vl:3b",
                "Describe this.",
                images="image-a",
                stream=False,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["images"], "image-a")

    def test_client_forwards_affinity_and_request_queue_timeout(self):
        options = {"temperature": 0.0}
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.chat(
                "mlx-community/example",
                [{"role": "user", "content": "hello"}],
                options=options,
                affinity_key="customer-42",
                queue_timeout=1.5,
                stream=False,
            )

        request_options = post.call_args.args[1]["options"]
        self.assertEqual(request_options["affinity_key"], "customer-42")
        self.assertEqual(request_options["queue_timeout"], 1.5)
        self.assertEqual(options, {"temperature": 0.0})

    def test_api_error_preserves_overload_status_and_code(self):
        response = io.BytesIO(b'{"error":"model request queue is full","code":"queue_full"}')
        error = api_error(HTTPError("http://localhost", 503, "busy", {}, response))

        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "queue_full")
        self.assertIn("queue is full", str(error))

    def test_chat_rejects_images_without_messages(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.client.chat("qwen2.5-vl:3b", [], images=["image-a"], stream=False)


class BootstrapTests(unittest.TestCase):
    def test_ensure_server_starts_local_daemon_and_writes_pid(self):
        process = Mock(pid=4321)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("machboost.client.Path.home", return_value=home),
                patch(
                    "machboost.client.MachBoostClient.health",
                    side_effect=[
                        MachBoostAPIError("not running"),
                        {"status": "ok", "version": __version__},
                    ],
                ),
                patch("machboost.client.subprocess.Popen", return_value=process) as popen,
            ):
                client, started = ensure_server(
                    "http://127.0.0.1:11435",
                    timeout=1.0,
                    log_path=home / "server.log",
                )

            self.assertTrue(started)
            self.assertEqual(client.endpoint, "http://127.0.0.1:11435")
            self.assertIn("serve", popen.call_args.args[0])
            self.assertEqual((home / ".cache" / "machboost" / "server.pid").read_text(), "4321\n")

    def test_ensure_server_does_not_auto_start_remote_endpoint(self):
        with patch(
            "machboost.client.MachBoostClient.health",
            side_effect=MachBoostAPIError("not running"),
        ):
            with self.assertRaisesRegex(MachBoostAPIError, "non-local"):
                ensure_server("http://example.com:11435", timeout=0.1)

    def test_ensure_server_reuses_matching_resident_version(self):
        with patch(
            "machboost.client.MachBoostClient.health",
            return_value={"status": "ok", "version": __version__},
        ):
            client, started = ensure_server("http://127.0.0.1:11435")

        self.assertFalse(started)
        self.assertEqual(client.endpoint, "http://127.0.0.1:11435")

    def test_ensure_server_restarts_stale_local_version(self):
        process = Mock(pid=4321)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with (
                patch("machboost.client.Path.home", return_value=home),
                patch(
                    "machboost.client.MachBoostClient.health",
                    side_effect=[
                        {"status": "ok", "version": "0.5.0"},
                        {"status": "ok", "version": __version__},
                    ],
                ),
                patch("machboost.client.MachBoostClient.shutdown") as shutdown,
                patch("machboost.client.MachBoostClient.is_healthy", return_value=False),
                patch("machboost.client.subprocess.Popen", return_value=process),
            ):
                _, started = ensure_server(
                    "http://127.0.0.1:11435",
                    timeout=1.0,
                    log_path=home / "server.log",
                )

        self.assertTrue(started)
        shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
