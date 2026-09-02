from __future__ import annotations

import os
import io
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
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

    def test_client_exposes_workspace_lifecycle_helpers(self):
        with patch.object(
            self.client,
            "post",
            side_effect=[
                {"workspace": {"id": "workspace-1", "name": "repo"}},
                {"workspace": {"id": "workspace-1", "file_count": 12}},
                {"schema": "machboost.workspace-query.v1", "hits": [{"path": "a.py"}]},
                {"removed": True},
            ],
        ) as post:
            registered = self.client.register_workspace(
                Path("/tmp/example-repo"),
                name="Example",
            )
            reindexed = self.client.reindex_workspace("workspace-1")
            query = self.client.query_workspace(
                "workspace-1",
                "where is cancellation",
                top_k=4,
                max_chars=12_000,
            )
            removed = self.client.remove_workspace("workspace-1")

        self.assertEqual(registered["id"], "workspace-1")
        self.assertEqual(reindexed["file_count"], 12)
        self.assertEqual(query["hits"][0]["path"], "a.py")
        self.assertTrue(removed)
        self.assertEqual(post.call_args_list[0].args[0], "/api/workspaces")
        self.assertEqual(
            post.call_args_list[0].args[1]["path"],
            "/tmp/example-repo",
        )
        self.assertEqual(
            post.call_args_list[2].args[1]["max_chars"],
            12_000,
        )

    def test_client_exposes_team_key_and_evaluation_helpers(self):
        with (
            patch.object(
                self.client,
                "post",
                side_effect=[
                    {"token": "mbk_once", "key": {"id": "key-1"}},
                    {"evaluation": {"id": "eval-1", "evaluator": "deterministic"}},
                ],
            ) as post,
            patch.object(
                self.client,
                "get",
                side_effect=[
                    {"traces": [{"id": "trace-1"}]},
                    {"evaluations": [{"id": "eval-old"}]},
                ],
            ) as get,
        ):
            created = self.client.create_team_key(
                "Agent",
                allowed_models=("llama3.2:3b",),
                max_concurrent=3,
            )
            traces = self.client.traces(limit=25)
            evaluations = self.client.evaluations(limit=10)
            evaluation = self.client.evaluate_traces(["trace-1"])

        self.assertEqual(created["token"], "mbk_once")
        self.assertEqual(traces[0]["id"], "trace-1")
        self.assertEqual(evaluations[0]["id"], "eval-old")
        self.assertEqual(evaluation["id"], "eval-1")
        self.assertEqual(post.call_args_list[0].args[1]["max_concurrent"], 3)
        self.assertEqual(get.call_args_list[0].args[0], "/api/traces?limit=25")

    def test_client_exposes_team_desktop_and_model_request_helpers(self):
        client = MachBoostClient(
            self.client.endpoint,
            api_token="mbk_team",
            device_id="device-42",
        )
        with (
            patch.object(
                client,
                "get",
                side_effect=[
                    {"schema": "machboost.team-connect.v1", "models": []},
                    {"clients": [{"device_id": "device-42"}]},
                    {"requests": [{"id": "modelreq_1", "status": "pending"}]},
                ],
            ) as get,
            patch.object(
                client,
                "post",
                side_effect=[
                    {"client": {"device_id": "device-42", "online": True}},
                    {"request": {"id": "modelreq_1", "status": "pending"}},
                    {"request": {"id": "modelreq_1", "status": "downloaded"}},
                ],
            ) as post,
        ):
            connection = client.team_connect()
            clients = client.team_clients(active_within_seconds=90)
            requests = client.team_model_requests(status="pending", limit=20)
            presence = client.report_team_presence(
                "Developer Mac",
                "0.13.0",
                workspace_name="checkout-service",
                workspace_fingerprint="abc123",
            )
            requested = client.request_team_model(
                "mlx-community/Muse-Glimmer-30B-4bit",
                note="Vision work",
            )
            resolved = client.resolve_team_model_request(
                "modelreq_1",
                status="downloaded",
            )

        self.assertEqual(connection["schema"], "machboost.team-connect.v1")
        self.assertEqual(clients[0]["device_id"], "device-42")
        self.assertEqual(requests[0]["status"], "pending")
        self.assertTrue(presence["online"])
        self.assertEqual(requested["status"], "pending")
        self.assertEqual(resolved["status"], "downloaded")
        self.assertIn("active_within_seconds=90.0", get.call_args_list[1].args[0])
        self.assertIn("status=pending", get.call_args_list[2].args[0])
        self.assertEqual(post.call_args_list[0].args[1]["device_id"], "device-42")
        self.assertNotIn("workspace_path", post.call_args_list[0].args[1])
        self.assertEqual(post.call_args_list[1].args[1]["device_id"], "device-42")

    def test_client_adds_device_header_to_inference_requests(self):
        client = MachBoostClient(self.client.endpoint, device_id="device-42")

        self.assertEqual(
            client._headers()["X-MachBoost-Device-ID"],
            "device-42",
        )

    def test_client_forwards_workspace_chat_options(self):
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.chat(
                "qwen2.5:7b",
                [{"role": "user", "content": "Where is cancellation handled?"}],
                workspace_id="workspace-1",
                workspace_query="request cancellation",
                workspace_top_k=6,
                workspace_max_chars=24_000,
                stream=False,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["workspace_id"], "workspace-1")
        self.assertEqual(payload["workspace_query"], "request cancellation")
        self.assertEqual(payload["workspace_top_k"], 6)
        self.assertEqual(payload["workspace_max_chars"], 24_000)

    def test_client_forwards_request_identifier(self):
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.chat(
                "mlx-community/example",
                [{"role": "user", "content": "hello"}],
                request_id="desktop-message-42",
                stream=False,
            )

        self.assertEqual(post.call_args.args[1]["request_id"], "desktop-message-42")

    def test_client_forwards_muse_reasoning_tools_and_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Read current weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        output_schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        }
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.chat(
                "muse-glimmer:30b-mlx",
                [{"role": "user", "content": "Check Chicago."}],
                tools=tools,
                tool_choice="auto",
                format=output_schema,
                think="high",
                request_id="muse-request-1",
                stream=False,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["format"], output_schema)
        self.assertEqual(payload["think"], "high")
        self.assertEqual(payload["request_id"], "muse-request-1")

    def test_generate_forwards_muse_reasoning_and_structured_output(self):
        with patch.object(self.client, "post", return_value={"done": True}) as post:
            self.client.generate(
                "muse-glimmer:30b-mlx",
                "Describe the image.",
                images=["fixture.png"],
                format="json",
                think="medium",
                stream=False,
            )

        payload = post.call_args.args[1]
        self.assertEqual(payload["images"], ["fixture.png"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["think"], "medium")

    def test_client_adds_bearer_token_to_every_request(self):
        client = MachBoostClient(self.client.endpoint, api_token="secret-token")

        self.assertEqual(
            client._headers()["Authorization"],
            "Bearer secret-token",
        )

    def test_client_retries_local_401_with_app_keychain_token(self):
        unauthorized = HTTPError(
            "http://127.0.0.1:11435/api/ps",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"error":"authentication required"}'),
        )
        response = MagicMock()
        response.read.return_value = b'{"models":[]}'
        response.__enter__.return_value = response
        client = MachBoostClient("http://127.0.0.1:11435")

        with (
            patch("machboost.client.machboost_app_api_token", return_value="app-token") as token,
            patch("machboost.client.urlopen", side_effect=[unauthorized, response]) as request,
        ):
            self.assertEqual(client.ps(), [])

        token.assert_called_once_with()
        retried_request = request.call_args_list[1].args[0]
        self.assertEqual(retried_request.get_header("Authorization"), "Bearer app-token")

    def test_stream_retries_local_401_with_app_keychain_token(self):
        unauthorized = HTTPError(
            "http://localhost:11435/api/chat",
            401,
            "unauthorized",
            {},
            io.BytesIO(b'{"error":"authentication required"}'),
        )
        response = MagicMock()
        response.__iter__.return_value = iter([b'{"done":true}\n'])
        response.__enter__.return_value = response
        client = MachBoostClient("http://localhost:11435")

        with (
            patch("machboost.client.machboost_app_api_token", return_value="app-token"),
            patch("machboost.client.urlopen", side_effect=[unauthorized, response]) as request,
        ):
            rows = list(client.stream("/api/chat", {"model": "example"}))

        self.assertEqual(rows, [{"done": True}])
        retried_request = request.call_args_list[1].args[0]
        self.assertEqual(retried_request.get_header("Authorization"), "Bearer app-token")

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

    def test_client_exposes_model_lifecycle_and_embedding_helpers(self):
        with patch.object(
            self.client,
            "post",
            side_effect=[
                {"status": "success", "model": {"name": "coder"}},
                {"status": "success", "model": {"name": "coder-copy"}},
                {"embeddings": [[0.5, 0.25]]},
                {"removed": True},
            ],
        ) as post:
            created = self.client.create_model(
                "coder",
                "qwen2.5-coder:3b",
                system="Use tests.",
                options={"num_ctx": 8192},
            )
            copied = self.client.copy_model("coder", "coder-copy")
            embeddings = self.client.embed("coder", "hello", keep_alive="10m")
            removed = self.client.delete_model("coder-copy")

        self.assertEqual(created["model"]["name"], "coder")
        self.assertEqual(copied["model"]["name"], "coder-copy")
        self.assertEqual(embeddings, [[0.5, 0.25]])
        self.assertTrue(removed)
        self.assertEqual(post.call_args_list[0].args[0], "/api/create")
        self.assertEqual(post.call_args_list[2].args[1]["keep_alive"], "10m")

    def test_client_can_purge_downloaded_model_weights(self):
        with patch.object(
            self.client,
            "post",
            return_value={"removed": True, "bytes_removed": 8_000_000_000},
        ) as post:
            removed = self.client.delete_model("mlx-community/example", purge=True)

        self.assertTrue(removed)
        post.assert_called_once_with(
            "/api/delete",
            {"model": "mlx-community/example", "purge": True},
        )

    def test_client_exposes_memory_and_provider_administration(self):
        with patch.object(
            self.client,
            "get",
            side_effect=[
                {"memories": [{"id": "mem_1"}]},
                {"schema": "machboost.cache-metrics.v1", "totals": {}},
                {"providers": [{"id": "provider_1"}]},
                {"schema": "machboost.provider-usage.v1", "usage": []},
            ],
        ) as get, patch.object(
            self.client,
            "post",
            side_effect=[
                {"memory": {"id": "mem_2"}},
                {"removed": 1},
                {"provider": {"id": "provider_1"}},
                {"provider_id": "provider_1", "has_secret": True},
                {"provider_id": "provider_1", "removed": True},
            ],
        ) as post:
            memories = self.client.memories(workspace_id="repo/a", limit=25)
            metrics = self.client.cache_metrics()
            providers = self.client.providers()
            usage = self.client.provider_usage("provider_1")
            memory = self.client.create_memory(
                "repo/a",
                "Retry policy",
                "Reuse the idempotency key.",
                scope="team",
                validated_by=("ci",),
            )
            removed_memories = self.client.delete_memories(["mem_2"])
            provider = self.client.configure_provider(
                "Fallback",
                "https://api.example.com",
                ("company-coder",),
                api_key_env="COMPANY_API_KEY",
                monthly_budget_usd=25,
                input_cost_per_million=1.5,
            )
            restored = self.client.set_provider_secret("provider_1", "secret")
            removed_provider = self.client.delete_provider("provider_1")

        self.assertEqual(memories[0]["id"], "mem_1")
        self.assertEqual(metrics["schema"], "machboost.cache-metrics.v1")
        self.assertEqual(providers[0]["id"], "provider_1")
        self.assertEqual(usage["usage"], [])
        self.assertEqual(memory["id"], "mem_2")
        self.assertEqual(removed_memories, 1)
        self.assertEqual(provider["id"], "provider_1")
        self.assertTrue(restored)
        self.assertTrue(removed_provider)
        self.assertIn("workspace_id=repo%2Fa", get.call_args_list[0].args[0])
        self.assertEqual(post.call_args_list[2].args[1]["monthly_budget_usd"], 25)
        self.assertNotIn("api_key", post.call_args_list[2].args[1])

    def test_client_forwards_memory_and_provider_route_extensions(self):
        with patch.object(
            self.client,
            "post",
            return_value={"message": {"role": "assistant", "content": "ok"}},
        ) as post:
            self.client.chat(
                "company-coder",
                [{"role": "user", "content": "Review this."}],
                workspace_id="repo",
                machboost={
                    "memory": {"mode": "private", "exact_cache": True},
                    "route": {"mode": "local_first", "provider_id": "provider_1"},
                },
                stream=False,
            )

        extension = post.call_args.args[1]["machboost"]
        self.assertTrue(extension["memory"]["exact_cache"])
        self.assertEqual(extension["route"]["mode"], "local_first")

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
