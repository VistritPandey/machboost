from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from machboost.models import resolve_model
from machboost.scheduler import RequestAdmissionError
from machboost.providers import ProviderStore
from machboost.server import (
    MachBoostHTTPServer,
    ModelConfig,
    OperationRegistry,
    RequestCancelled,
    RuntimeManager,
    load_accelerator,
    model_config,
    parse_keep_alive,
)
from machboost.team import TeamStore
from machboost.workspace import WorkspaceStore


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
    prompt_tokens: int = 12
    prompt_eval_seconds: float = 0.1
    generation_seconds: float = 0.2
    time_to_first_token_seconds: float = 0.12


class FakeService:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.prompt_cache_configs = []

    def reset_cache(self) -> None:
        self.reset_calls += 1

    def encode(self, text):
        return tuple(str(text).split())

    def decode(self, tokens, **_kwargs):
        return " ".join(tokens)

    def embed(self, texts):
        return [[float(len(text.split())), 1.0] for text in texts]

    def configure_native_prompt_cache(
        self, *, enabled, max_size, max_bytes, namespace
    ) -> None:
        self.prompt_cache_configs.append(
            {
                "enabled": enabled,
                "max_size": max_size,
                "max_bytes": max_bytes,
                "namespace": namespace,
            }
        )


class FakeAccelerator:
    def __init__(self) -> None:
        self.service = FakeService()
        self.chat_calls = []
        self.generate_calls = []
        self.chat_options = []
        self.generate_options = []

    def generate_chat(
        self,
        messages,
        *,
        max_tokens,
        context=None,
        on_text=None,
        tools=None,
        enable_thinking=False,
        stop_strings=None,
        generation_options=None,
    ):
        self.chat_calls.append((messages, max_tokens, context))
        self.chat_options.append(
            {
                "enable_thinking": enable_thinking,
                "stop_strings": stop_strings,
                "generation_options": generation_options,
            }
        )
        if on_text is not None:
            on_text("hello ")
            on_text("world")
        return "hello world", FakeStats()

    def generate(
        self,
        prompt,
        *,
        max_tokens,
        context=None,
        on_text=None,
        stop_strings=None,
        generation_options=None,
    ):
        self.generate_calls.append((prompt, max_tokens, context))
        self.generate_options.append(
            {
                "stop_strings": stop_strings,
                "generation_options": generation_options,
            }
        )
        if on_text is not None:
            on_text("completed")
        return "completed", FakeStats(generated_tokens=1)


class ToolCallingAccelerator(FakeAccelerator):
    def generate_chat(
        self,
        messages,
        *,
        max_tokens,
        context=None,
        on_text=None,
        tools=None,
    ):
        self.chat_calls.append((messages, max_tokens, context, tools))
        return (
            '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
            '<tool_call>{"name":"search_repo","arguments":{"query":"cancel"}}</tool_call>',
            FakeStats(generated_tokens=20),
        )


class FakeVisionAccelerator:
    def __init__(self) -> None:
        self.chat_calls = []
        self.generate_calls = []
        self.cold_vision_calls = []
        self.vision_token_calls = []
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
        vision_token_mode="off",
        vision_token_ratio=0.35,
        vision_token_layer=None,
        vision_token_bucket=None,
        vision_calibration=None,
    ):
        self.chat_calls.append((messages, max_tokens, use_vision_cache, temperature))
        self.cold_vision_calls.append((cold_vision_mode, cold_vision_max_edge))
        self.vision_token_calls.append(
            (
                vision_token_mode,
                vision_token_ratio,
                vision_token_layer,
                vision_token_bucket,
                vision_calibration,
            )
        )
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
        vision_token_mode="off",
        vision_token_ratio=0.35,
        vision_token_layer=None,
        vision_token_bucket=None,
        vision_calibration=None,
    ):
        self.generate_calls.append((prompt, tuple(images or ()), use_vision_cache))
        self.cold_vision_calls.append((cold_vision_mode, cold_vision_max_edge))
        self.vision_token_calls.append(
            (
                vision_token_mode,
                vision_token_ratio,
                vision_token_layer,
                vision_token_bucket,
                vision_calibration,
            )
        )
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


class ConcurrencyProbe:
    def __init__(self, target: int = 1) -> None:
        self.target = target
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.target_entered = threading.Event()
        self.active = 0
        self.max_active = 0

    def wait(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.target:
                self.target_entered.set()
        try:
            if not self.release.wait(timeout=3.0):
                raise RuntimeError("test request was not released")
        finally:
            with self.lock:
                self.active -= 1


class BlockingAccelerator(FakeAccelerator):
    def __init__(self, probe: ConcurrencyProbe) -> None:
        super().__init__()
        self.probe = probe

    def generate_chat(self, messages, *, max_tokens, context=None, on_text=None):
        self.chat_calls.append((messages, max_tokens, context))
        self.probe.wait()
        if on_text is not None:
            on_text("hello world")
        return "hello world", FakeStats()


class StreamingAccelerator(FakeAccelerator):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def generate_chat(self, messages, *, max_tokens, context=None, on_text=None):
        self.started.set()
        for _ in range(200):
            if on_text is not None:
                on_text("token ")
            time.sleep(0.005)
        return "token " * 200, FakeStats(generated_tokens=200)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RuntimeManagerTests(unittest.TestCase):
    def test_dflash_model_config_preserves_decoder_selection(self):
        config = model_config(
            "qwen3.5:9b",
            {
                "backend": "dflash",
                "draft_model": "z-lab/custom-draft",
                "draft_quant": "w4:gs64",
                "verify_mode": "adaptive",
            },
        )

        self.assertEqual(config.backend, "dflash")
        self.assertEqual(config.draft_model, "z-lab/custom-draft")
        self.assertEqual(config.draft_quant, "w4:gs64")
        self.assertEqual(config.verify_mode, "adaptive")

    def test_dflash_loader_receives_resident_decoder_options(self):
        sentinel = object()
        config = ModelConfig(
            model="mlx-community/Qwen3.5-9B-MLX-4bit",
            backend="dflash",
            draft_model="z-lab/Qwen3.5-9B-DFlash",
            draft_quant="w4",
            verify_mode="adaptive",
            lazy=True,
        )
        with patch(
            "machboost.adapters.dflash.DFlashAccelerator.from_pretrained",
            return_value=sentinel,
        ) as load:
            result = load_accelerator(config)

        self.assertIs(result, sentinel)
        load.assert_called_once_with(
            config.model,
            draft_model=config.draft_model,
            draft_quant=config.draft_quant,
            verify_mode=config.verify_mode,
            lazy=True,
        )

    def test_generation_throughput_excludes_pull_duration(self):
        now = [0.0]
        registry = OperationRegistry(clock=lambda: now[0])
        generation = registry.begin("chat-1", "chat", "example")
        now[0] = 2.0
        registry.finish(generation, status="completed", generated_tokens=20)
        pull = registry.begin("pull-1", "pull", "example")
        now[0] = 12.0
        registry.finish(pull, status="completed")

        metrics = registry.snapshot()

        self.assertEqual(metrics["generation_tokens_per_second"], 10.0)

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
        self.assertEqual(manager.ps()[0]["keep_alive_seconds"], 300.0)
        metrics = first.ollama_metrics()
        self.assertEqual(metrics["prompt_eval_count"], 12)
        self.assertEqual(metrics["prompt_eval_duration"], 100_000_000)
        self.assertEqual(metrics["eval_duration"], 200_000_000)
        self.assertEqual(
            metrics["machboost"]["time_to_first_token_seconds"],
            0.12,
        )
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

    def test_expiry_waits_for_active_model_lock(self):
        clock = FakeClock()
        manager = RuntimeManager(loader=lambda config: FakeAccelerator(), clock=clock)
        entry, _ = manager.get_or_load("mlx-community/example", keep_alive="1s")
        locked = threading.Event()
        release = threading.Event()

        def hold_model_lock():
            with entry.lock:
                locked.set()
                release.wait(timeout=2.0)

        thread = threading.Thread(target=hold_model_lock)
        thread.start()
        try:
            self.assertTrue(locked.wait(timeout=1.0))
            clock.advance(2.0)
            self.assertEqual(manager.evict_expired(), 0)
        finally:
            release.set()
            thread.join(timeout=2.0)

        self.assertEqual(manager.evict_expired(), 1)

    def test_compile_warmup_runs_once_without_counting_as_request(self):
        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        entry, _ = manager.get_or_load("mlx-community/example")

        _, first_performed = manager.warm(entry)
        second_duration, second_performed = manager.warm(entry)

        self.assertTrue(first_performed)
        self.assertFalse(second_performed)
        self.assertEqual(second_duration, 0.0)
        self.assertEqual(entry.warmups, 1)
        self.assertEqual(entry.requests, 0)
        self.assertEqual(entry.accelerator.chat_calls[0][1], 1)

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

    def test_two_text_replicas_execute_same_model_requests_concurrently(self):
        probe = ConcurrencyProbe(target=2)
        loaded = []
        manager = RuntimeManager(
            loader=lambda config: loaded.append(BlockingAccelerator(probe)) or loaded[-1],
            replicas=2,
        )
        results = []
        errors = []

        def request() -> None:
            try:
                results.append(
                    manager.chat(
                        "mlx-community/example",
                        [{"role": "user", "content": "hello"}],
                    )
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(probe.target_entered.wait(timeout=2.0))
            scheduler = manager.ps()[0]["scheduler"]
            self.assertEqual(scheduler["active_requests"], 2)
            self.assertEqual(scheduler["max_active_requests"], 2)
        finally:
            probe.release.set()
            for thread in threads:
                thread.join(timeout=2.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(probe.max_active, 2)
        self.assertEqual({result.scheduler["replica"] for result in results}, {0, 1})
        self.assertTrue(all(result.scheduler["replicas"] == 2 for result in results))

    def test_full_runtime_queue_rejects_without_waiting(self):
        probe = ConcurrencyProbe()
        manager = RuntimeManager(
            loader=lambda config: BlockingAccelerator(probe),
            max_queue=0,
        )
        first = threading.Thread(
            target=lambda: manager.chat(
                "mlx-community/example",
                [{"role": "user", "content": "first"}],
            )
        )
        first.start()
        try:
            self.assertTrue(probe.target_entered.wait(timeout=1.0))
            with self.assertRaises(RequestAdmissionError) as raised:
                manager.chat(
                    "mlx-community/example",
                    [{"role": "user", "content": "second"}],
                )
            self.assertEqual(raised.exception.reason, "queue_full")
            self.assertEqual(manager.ps()[0]["scheduler"]["rejected_requests"], 1)
        finally:
            probe.release.set()
            first.join(timeout=2.0)

    def test_runtime_queue_timeout_is_visible(self):
        probe = ConcurrencyProbe()
        manager = RuntimeManager(
            loader=lambda config: BlockingAccelerator(probe),
            max_queue=1,
            queue_timeout=0.02,
        )
        first = threading.Thread(
            target=lambda: manager.chat(
                "mlx-community/example",
                [{"role": "user", "content": "first"}],
            )
        )
        first.start()
        try:
            self.assertTrue(probe.target_entered.wait(timeout=1.0))
            with self.assertRaises(RequestAdmissionError) as raised:
                manager.chat(
                    "mlx-community/example",
                    [{"role": "user", "content": "second"}],
                )
            self.assertEqual(raised.exception.reason, "queue_timeout")
            self.assertEqual(manager.ps()[0]["scheduler"]["timed_out_requests"], 1)
        finally:
            probe.release.set()
            first.join(timeout=2.0)

    def test_stop_waits_for_requests_and_releases_every_replica(self):
        probe = ConcurrencyProbe()
        loaded = []
        manager = RuntimeManager(
            loader=lambda config: loaded.append(BlockingAccelerator(probe)) or loaded[-1],
            replicas=2,
        )
        request = threading.Thread(
            target=lambda: manager.chat(
                "mlx-community/example",
                [{"role": "user", "content": "hello"}],
            )
        )
        request.start()
        self.assertTrue(probe.target_entered.wait(timeout=1.0))
        stopped = []
        stop_thread = threading.Thread(
            target=lambda: stopped.append(manager.stop("mlx-community/example"))
        )
        stop_thread.start()
        self.assertTrue(stop_thread.is_alive())

        probe.release.set()
        request.join(timeout=2.0)
        stop_thread.join(timeout=2.0)

        self.assertEqual(stopped, [1])
        self.assertEqual([item.service.reset_calls for item in loaded], [1, 1])

    def test_vlm_uses_one_safe_worker_when_text_server_has_multiple_replicas(self):
        loaded = []
        manager = RuntimeManager(
            loader=lambda config: loaded.append(config) or FakeVisionAccelerator(),
            replicas=2,
        )

        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            entry, _ = manager.get_or_load("qwen2.5-vl:3b")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(entry.config.replicas, 1)
        self.assertEqual(entry.scheduler.snapshot()["replicas"], 1)


class HTTPServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace_store = WorkspaceStore(
            Path(self.temporary.name) / "workspaces"
        )
        self.loaded = []
        manager = RuntimeManager(loader=lambda config: self._load(config))
        self.server = MachBoostHTTPServer(
            ("127.0.0.1", 0),
            manager=manager,
            workspace_store=self.workspace_store,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temporary.cleanup()

    def _load(self, config):
        if config.model.endswith("failing"):
            accelerator = FailingAccelerator()
        elif config.model.endswith("tool-calling"):
            accelerator = ToolCallingAccelerator()
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
        health = json.loads(body)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["serving"]["text_replicas"], 1)

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

    def test_catalog_and_metrics_have_stable_schemas(self):
        _, _, catalog_body = self.request("/api/catalog")
        _, _, metrics_body = self.request("/api/metrics")

        catalog = json.loads(catalog_body)
        metrics = json.loads(metrics_body)
        self.assertEqual(catalog["schema"], "machboost.catalog.v1")
        self.assertTrue(any(item["name"] == "llama3.2:3b" for item in catalog["models"]))
        self.assertEqual(metrics["schema"], "machboost.metrics.v1")
        self.assertIn("peak_resident_memory_bytes", metrics["process"])

    def test_workspace_lifecycle_endpoints(self):
        repository = Path(self.temporary.name) / "repository"
        repository.mkdir()
        source = repository / "payments.py"
        source.write_text(
            "def capture_payment(invoice_id):\n"
            "    return gateway.capture(invoice_id)\n",
            encoding="utf-8",
        )

        status, _, body = self.request(
            "/api/workspaces",
            {"path": str(repository), "name": "Payments"},
        )
        created = json.loads(body)
        workspace_id = created["workspace"]["id"]
        self.assertEqual(status, 201)
        self.assertEqual(created["schema"], "machboost.workspace-index.v1")
        self.assertEqual(created["workspace"]["file_count"], 1)

        _, _, body = self.request("/api/workspaces")
        listed = json.loads(body)
        self.assertEqual(listed["schema"], "machboost.workspaces.v1")
        self.assertEqual(listed["workspaces"][0]["id"], workspace_id)

        _, _, body = self.request(
            "/api/workspaces/query",
            {
                "workspace_id": workspace_id,
                "query": "Where is capture_payment implemented?",
            },
        )
        query = json.loads(body)
        self.assertEqual(query["schema"], "machboost.workspace-query.v1")
        self.assertEqual(query["hits"][0]["path"], "payments.py")

        source.write_text(
            "def settle_invoice(invoice_id):\n"
            "    return gateway.settle(invoice_id)\n",
            encoding="utf-8",
        )
        _, _, body = self.request(
            "/api/workspaces/index",
            {"workspace_id": workspace_id},
        )
        reindexed = json.loads(body)
        self.assertEqual(reindexed["indexed_files"], 1)

        _, _, body = self.request(
            "/api/workspaces/delete",
            {"workspace_id": workspace_id},
        )
        self.assertTrue(json.loads(body)["removed"])
        self.assertTrue(source.exists())

    def test_workspace_chat_injects_retrieved_evidence_and_returns_citations(self):
        repository = Path(self.temporary.name) / "repository"
        repository.mkdir()
        (repository / "auth.py").write_text(
            "def authenticate_user(token):\n"
            "    payload = decode_token(token)\n"
            "    return lookup_user(payload['sub'])\n",
            encoding="utf-8",
        )
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)

        _, _, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "workspace_id": workspace.id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Where is authenticate_user handled?",
                    }
                ],
                "stream": False,
            },
        )

        response = json.loads(body)
        accelerator = self.loaded[0][1]
        messages, _, draft_context = accelerator.chat_calls[0]
        evidence = next(
            message["content"]
            for message in messages
            if message["role"] == "system"
        )
        self.assertIn("auth.py:1-3", evidence)
        self.assertIn("authenticate_user", evidence)
        self.assertTrue(
            any("authenticate_user" in item for item in draft_context)
        )
        workspace_result = response["machboost"]["workspace"]
        self.assertEqual(workspace_result["id"], workspace.id)
        self.assertEqual(workspace_result["retrieved_chunks"], 1)
        self.assertEqual(workspace_result["citations"][0]["path"], "auth.py")
        self.assertEqual(
            response["machboost"]["stats"]["accepted_draft_tokens"], 2
        )
        self.assertEqual(response["machboost"]["backend"], "mlx")
        indexed_workspace = self.workspace_store.get(workspace.id)
        self.assertEqual(
            accelerator.service.prompt_cache_configs[-1]["namespace"],
            f"workspace:{workspace.id}:{indexed_workspace.revision}",
        )

    def test_streaming_workspace_response_retains_runtime_metrics(self):
        repository = Path(self.temporary.name) / "repository"
        repository.mkdir()
        (repository / "stream.py").write_text(
            "def cancel_stream(request_id):\n"
            "    return active_requests.cancel(request_id)\n",
            encoding="utf-8",
        )
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)

        _, _, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "workspace_id": workspace.id,
                "messages": [
                    {"role": "user", "content": "Where is streaming cancelled?"}
                ],
                "stream": True,
            },
        )

        final = json.loads(body.splitlines()[-1])
        self.assertEqual(final["machboost"]["workspace"]["id"], workspace.id)
        self.assertEqual(final["machboost"]["stats"]["generated_tokens"], 2)
        self.assertEqual(final["machboost"]["backend"], "mlx")

    def test_openai_workspace_extension_is_backward_compatible(self):
        repository = Path(self.temporary.name) / "repository"
        repository.mkdir()
        (repository / "jobs.go").write_text(
            "package jobs\nfunc ScheduleCleanup() {}\n",
            encoding="utf-8",
        )
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)

        _, _, body = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/example",
                "messages": [
                    {"role": "user", "content": "Where is cleanup scheduled?"}
                ],
                "machboost": {
                    "workspace_id": workspace.id,
                    "workspace_top_k": 4,
                },
            },
        )

        response = json.loads(body)
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(
            response["machboost"]["workspace"]["citations"][0]["path"],
            "jobs.go",
        )

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

    def test_ollama_num_ctx_truncates_with_loaded_tokenizer(self):
        _, _, body = self.request(
            "/api/generate",
            {
                "model": "mlx-community/example",
                "prompt": "one two three four five",
                "stream": False,
                "options": {"num_ctx": 3, "num_predict": 1},
            },
        )

        response = json.loads(body)
        accelerator = self.loaded[0][1]
        self.assertEqual(accelerator.generate_calls[0][0], "four five")
        self.assertEqual(
            response["machboost"]["stats"]["context_truncated_tokens"], 3
        )

    def test_ollama_context_overflow_can_fail_instead_of_shift(self):
        with self.assertRaises(HTTPError) as raised:
            self.request(
                "/api/generate",
                {
                    "model": "mlx-community/example",
                    "prompt": "one two three four five",
                    "stream": False,
                    "options": {
                        "num_ctx": 3,
                        "num_predict": 1,
                        "truncate": False,
                    },
                },
            )
        self.assertEqual(raised.exception.code, 400)
        self.assertIn("requires 5 tokens", raised.exception.read().decode("utf-8"))

    def test_ollama_system_template_thinking_and_stop_are_forwarded(self):
        self.request(
            "/api/generate",
            {
                "model": "mlx-community/example",
                "prompt": "hello",
                "system": "be concise",
                "template": "SYS={{ .System }} USER={{ .Prompt }}",
                "stream": False,
                "options": {
                    "stop": ["END"],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            },
        )
        self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "solve"}],
                "think": "high",
                "stream": False,
            },
        )

        accelerator = self.loaded[0][1]
        self.assertEqual(
            accelerator.generate_calls[0][0], "SYS=be concise USER=hello"
        )
        self.assertEqual(accelerator.generate_options[0]["stop_strings"], ["END"])
        self.assertEqual(
            accelerator.generate_options[0]["generation_options"],
            {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
        )
        self.assertEqual(accelerator.chat_options[0]["enable_thinking"], "high")

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

    def test_load_endpoint_can_compile_warm_text_model(self):
        _, _, body = self.request(
            "/api/load",
            {
                "model": "qwen2.5:3b",
                "keep_alive": "5m",
                "warmup": True,
                "options": {"backend": "mlx"},
            },
        )

        response = json.loads(body)
        self.assertTrue(response["warmup_performed"])
        self.assertEqual(response["instance"]["warmups"], 1)
        self.assertEqual(self.loaded[0][1].chat_calls[0][1], 1)

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

    def test_client_request_id_is_returned_in_every_stream_event(self):
        _, _, body = self.request(
            "/api/chat",
            {
                "request_id": "desktop-chat-7",
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        rows = [json.loads(line) for line in body.splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(row["request_id"] == "desktop-chat-7" for row in rows))

    def test_pull_stream_reports_progress_with_request_id(self):
        def fake_pull(model, *, revision=None, progress=None, cancel_event=None):
            self.assertEqual(model, "mlx-community/example")
            self.assertEqual(revision, "main")
            progress({"status": "downloading", "file": "weights.safetensors", "completed": 4, "total": 8})
            return {"status": "success", "model": model, "path": "/tmp/example"}

        with patch.object(self.server.manager, "pull", side_effect=fake_pull):
            _, headers, body = self.request(
                "/api/pull",
                {
                    "request_id": "desktop-pull-3",
                    "model": "mlx-community/example",
                    "revision": "main",
                    "stream": True,
                },
            )

        rows = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(headers.get_content_type(), "application/x-ndjson")
        self.assertTrue(all(row["request_id"] == "desktop-pull-3" for row in rows))
        self.assertEqual(rows[0]["completed"], 4)
        self.assertTrue(rows[-1]["done"])
        self.assertEqual(rows[-1]["status"], "success")

    def test_cancel_stops_an_active_pull_stream(self):
        started = threading.Event()
        response_rows = []

        def slow_pull(model, *, revision=None, progress=None, cancel_event=None):
            progress({"status": "resolving", "model": model})
            started.set()
            while not cancel_event.wait(timeout=0.01):
                pass
            raise RequestCancelled("request cancelled")

        def stream_request() -> None:
            payload = json.dumps(
                {
                    "request_id": "cancel-pull",
                    "model": "mlx-community/example",
                    "stream": True,
                }
            ).encode("utf-8")
            request = Request(
                self.base_url + "/api/pull",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=3.0) as response:
                response_rows.extend(
                    json.loads(line) for line in response if line.strip()
                )

        with patch.object(self.server.manager, "pull", side_effect=slow_pull):
            request_thread = threading.Thread(target=stream_request)
            request_thread.start()
            self.assertTrue(started.wait(timeout=1.0))
            _, _, cancel_body = self.request(
                "/api/cancel",
                {"request_id": "cancel-pull"},
            )
            request_thread.join(timeout=2.0)

        self.assertFalse(request_thread.is_alive())
        self.assertTrue(json.loads(cancel_body)["cancelled"])
        self.assertEqual(response_rows[-1]["status"], "cancelled")
        self.assertTrue(response_rows[-1]["done"])

    def test_secured_server_requires_valid_bearer_token(self):
        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        server = MachBoostHTTPServer(
            ("127.0.0.1", 0),
            manager=manager,
            api_token="top-secret",
            require_auth=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            health = Request(f"http://{host}:{port}/healthz")
            with urlopen(health, timeout=2.0) as response:
                self.assertEqual(response.status, 200)

            for protected_path in ("/", "/api/metrics"):
                with self.subTest(path=protected_path):
                    with self.assertRaises(HTTPError) as raised:
                        urlopen(
                            Request(f"http://{host}:{port}{protected_path}"),
                            timeout=2.0,
                        )
                    self.assertEqual(raised.exception.code, 401)

            authorized = Request(
                f"http://{host}:{port}/api/metrics",
                headers={"Authorization": "Bearer top-secret"},
            )
            with urlopen(authorized, timeout=2.0) as response:
                self.assertEqual(json.loads(response.read())["schema"], "machboost.metrics.v1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_cancel_stops_an_active_stream(self):
        accelerator = StreamingAccelerator()
        manager = RuntimeManager(loader=lambda config: accelerator)
        server = MachBoostHTTPServer(("127.0.0.1", 0), manager=manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        endpoint = f"http://{host}:{port}"
        response_rows = []

        def stream_request() -> None:
            payload = json.dumps(
                {
                    "request_id": "cancel-me",
                    "model": "mlx-community/example",
                    "messages": [{"role": "user", "content": "long answer"}],
                    "stream": True,
                }
            ).encode("utf-8")
            request = Request(
                endpoint + "/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=3.0) as response:
                response_rows.extend(
                    json.loads(line) for line in response if line.strip()
                )

        request_thread = threading.Thread(target=stream_request)
        request_thread.start()
        try:
            self.assertTrue(accelerator.started.wait(timeout=1.0))
            cancel_payload = json.dumps({"request_id": "cancel-me"}).encode("utf-8")
            cancel_request = Request(
                endpoint + "/api/cancel",
                data=cancel_payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(cancel_request, timeout=2.0) as response:
                self.assertEqual(response.status, 202)
            request_thread.join(timeout=2.0)
            self.assertFalse(request_thread.is_alive())
            self.assertEqual(response_rows[-1]["done_reason"], "cancelled")
            self.assertEqual(manager.metrics()["operations"]["totals"]["cancelled"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_openai_completion_endpoint(self):
        _, _, body = self.request(
            "/v1/completions",
            {"model": "mlx-community/example", "prompt": "def add", "max_tokens": 16},
        )

        response = json.loads(body)
        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["choices"][0]["text"], "completed")
        self.assertEqual(self.loaded[0][1].generate_calls[0][1], 16)

    def test_openai_chat_returns_parallel_tool_calls(self):
        _, _, body = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/tool-calling",
                "messages": [{"role": "user", "content": "Inspect and search"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_repo",
                            "parameters": {"type": "object"},
                        },
                    },
                ],
                "parallel_tool_calls": True,
            },
        )

        response = json.loads(body)
        choice = response["choices"][0]
        calls = choice["message"]["tool_calls"]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual([call["function"]["name"] for call in calls], ["read_file", "search_repo"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["path"], "a.py")
        self.assertEqual(len(self.loaded[0][1].chat_calls[0][3]), 2)

    def test_ollama_chat_returns_compatible_tool_calls(self):
        _, _, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/tool-calling",
                "messages": [{"role": "user", "content": "Inspect and search"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "stream": False,
            },
        )

        response = json.loads(body)
        calls = response["message"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(calls[0]["function"]["arguments"]["path"], "a.py")

    def test_stop_endpoint_unloads_resident_model(self):
        self.request(
            "/api/generate",
            {"model": "mlx-community/example", "prompt": "hello", "stream": False},
        )
        _, _, body = self.request("/api/stop", {"model": "mlx-community/example"})

        self.assertEqual(json.loads(body)["unloaded"], 1)
        _, _, ps_body = self.request("/api/ps")
        self.assertEqual(json.loads(ps_body)["models"], [])

    def test_ollama_alias_create_copy_show_run_and_delete(self):
        _, _, created_body = self.request(
            "/api/create",
            {
                "model": "company-coder:latest",
                "from": "qwen2.5:3b",
                "system": "Follow company conventions.",
                "parameters": {"num_ctx": 4096, "temperature": 0},
            },
        )
        self.request(
            "/api/copy",
            {"source": "company-coder:latest", "destination": "company-coder:backup"},
        )
        _, _, response_body = self.request(
            "/api/chat",
            {
                "model": "company-coder:latest",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        _, _, show_body = self.request("/api/show", {"model": "company-coder:latest"})
        _, _, tags_body = self.request("/api/tags")
        _, _, deleted_body = self.request(
            "/api/delete", {"model": "company-coder:backup"}
        )

        created = json.loads(created_body)
        response = json.loads(response_body)
        shown = json.loads(show_body)
        names = {item["name"] for item in json.loads(tags_body)["models"]}
        self.assertEqual(created["model"]["source"], "qwen2.5:3b")
        self.assertEqual(response["model"], "company-coder:latest")
        self.assertEqual(
            self.loaded[0][0].model,
            resolve_model("qwen2.5:3b").model,
        )
        injected = self.loaded[0][1].chat_calls[0][0][0]["content"]
        self.assertIn("Follow company conventions", injected)
        self.assertEqual(shown["alias"]["source"], "qwen2.5:3b")
        self.assertIn("company-coder:latest", names)
        self.assertIn("company-coder:backup", names)
        self.assertTrue(json.loads(deleted_body)["removed"])

    def test_alias_load_stop_and_delete_resolve_the_resident_source_model(self):
        self.request(
            "/api/create",
            {"model": "company-coder:latest", "from": "qwen2.5:3b"},
        )

        _, _, loaded_body = self.request(
            "/api/load",
            {
                "model": "company-coder:latest",
                "options": {"num_ctx": 2048},
                "keep_alive": "10m",
            },
        )
        _, _, stopped_body = self.request(
            "/api/stop", {"model": "company-coder:latest"}
        )
        self.request(
            "/api/load", {"model": "company-coder:latest", "keep_alive": "10m"}
        )
        _, _, deleted_body = self.request(
            "/api/delete", {"model": "company-coder:latest"}
        )

        loaded = json.loads(loaded_body)
        self.assertEqual(
            loaded["instance"]["model"],
            resolve_model("qwen2.5:3b").model,
        )
        self.assertEqual(json.loads(stopped_body)["unloaded"], 1)
        self.assertTrue(json.loads(deleted_body)["removed"])
        self.assertEqual(self.server.manager.ps(), [])

    def test_empty_ollama_request_loads_and_keep_alive_zero_unloads(self):
        _, _, load_body = self.request(
            "/api/chat",
            {
                "model": "qwen2.5:3b",
                "messages": [],
                "stream": False,
                "keep_alive": "10m",
            },
        )
        _, _, unload_body = self.request(
            "/api/generate",
            {
                "model": "qwen2.5:3b",
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )

        self.assertEqual(json.loads(load_body)["done_reason"], "load")
        self.assertEqual(json.loads(unload_body)["done_reason"], "unload")
        self.assertEqual(json.loads(unload_body)["unloaded"], 1)

    def test_ollama_and_openai_embedding_routes_share_resident_model(self):
        _, _, ollama_body = self.request(
            "/api/embed",
            {
                "model": "qwen2.5:3b",
                "input": ["one two", "three"],
                "options": {"num_ctx": 16},
                "keep_alive": "10m",
            },
        )
        _, _, legacy_body = self.request(
            "/api/embeddings",
            {"model": "qwen2.5:3b", "prompt": "four five six"},
        )
        _, _, openai_body = self.request(
            "/v1/embeddings",
            {"model": "qwen2.5:3b", "input": "seven eight"},
        )

        ollama = json.loads(ollama_body)
        legacy = json.loads(legacy_body)
        openai = json.loads(openai_body)
        self.assertEqual(ollama["embeddings"], [[2.0, 1.0], [1.0, 1.0]])
        self.assertEqual(legacy["embedding"], [3.0, 1.0])
        self.assertEqual(openai["data"][0]["embedding"], [2.0, 1.0])
        self.assertEqual(len(self.loaded), 1)

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

    def test_ollama_generate_forwards_post_fusion_vision_options(self):
        self.request(
            "/api/generate",
            {
                "model": "qwen3-vl:8b",
                "prompt": "Read the label.",
                "images": ["image-one"],
                "stream": False,
                "options": {
                    "vision_tokens": "adaptive",
                    "vision_token_ratio": 0.35,
                },
            },
        )

        self.assertEqual(
            self.loaded[0][1].vision_token_calls[0],
            ("adaptive", 0.35, None, None, None),
        )

    @patch("machboost.server.load_vision_calibration")
    def test_ollama_generate_forwards_automatic_vision_policy(self, load_calibration):
        calibration = {"schema": "machboost.vision_calibration.v1"}
        load_calibration.return_value = calibration

        self.request(
            "/api/generate",
            {
                "model": "qwen3-vl:8b",
                "prompt": "Read the label.",
                "images": ["image-one"],
                "stream": False,
                "options": {
                    "vision_tokens": "auto",
                    "vision_token_layer": 6,
                    "vision_token_bucket": 32,
                    "vision_calibration": "vision-calibration.json",
                },
            },
        )

        load_calibration.assert_called_once_with("vision-calibration.json")
        self.assertEqual(
            self.loaded[0][1].vision_token_calls[0],
            ("auto", 0.35, 6, 32, calibration),
        )

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

    def test_streaming_request_gets_http_503_when_queue_is_full(self):
        probe = ConcurrencyProbe()
        manager = RuntimeManager(
            loader=lambda config: BlockingAccelerator(probe),
            max_queue=0,
        )
        server = MachBoostHTTPServer(("127.0.0.1", 0), manager=manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        endpoint = f"http://{host}:{port}/api/chat"
        first_done = threading.Event()

        def first_request() -> None:
            payload = json.dumps(
                {
                    "model": "mlx-community/example",
                    "messages": [{"role": "user", "content": "first"}],
                    "stream": False,
                }
            ).encode("utf-8")
            request = Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=3.0) as response:
                    response.read()
            finally:
                first_done.set()

        first = threading.Thread(target=first_request)
        first.start()
        try:
            self.assertTrue(probe.target_entered.wait(timeout=1.0))
            payload = json.dumps(
                {
                    "model": "mlx-community/example",
                    "messages": [{"role": "user", "content": "second"}],
                    "stream": True,
                }
            ).encode("utf-8")
            request = Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=2.0)
            body = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(raised.exception.code, 503)
            self.assertEqual(body["code"], "queue_full")
        finally:
            probe.release.set()
            first.join(timeout=2.0)
            self.assertTrue(first_done.is_set())
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_shutdown_endpoint_stops_server_and_releases_models(self):
        self.request(
            "/api/generate",
            {"model": "mlx-community/example", "prompt": "hello", "stream": False},
        )
        _, _, body = self.request("/api/shutdown", {})

        self.assertEqual(json.loads(body)["unloaded"], 1)
        self.thread.join(timeout=2.0)
        self.assertFalse(self.thread.is_alive())


class TeamGatewayHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.team_store = TeamStore(Path(self.temporary.name) / "team.sqlite3")
        self.workspace_store = WorkspaceStore(Path(self.temporary.name) / "workspaces")
        self.provider_calls = []

        def provider_transport(config, path, payload, headers):
            self.provider_calls.append((config, path, payload, headers))
            return {
                "id": "upstream",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "external answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }

        self.provider_store = ProviderStore(
            Path(self.temporary.name) / "team.sqlite3",
            transport=provider_transport,
        )
        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        self.server = MachBoostHTTPServer(
            ("127.0.0.1", 0),
            manager=manager,
            api_token="admin-secret",
            require_auth=True,
            team_store=self.team_store,
            workspace_store=self.workspace_store,
            provider_store=self.provider_store,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temporary.cleanup()

    def request(self, path, payload=None, *, token="admin-secret", raw=False):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers)
        with urlopen(request, timeout=3.0) as response:
            body = response.read()
            if raw:
                return response.status, response.headers, body
            return response.status, json.loads(body.decode("utf-8"))

    def create_employee_key(self, *, models=("mlx-community/example",)) -> str:
        _, body = self.request(
            "/api/team/keys",
            {
                "name": "Coding agent",
                "scopes": [
                    "inference",
                    "models:read",
                    "traces:read",
                    "evaluations:read",
                    "evaluations:write",
                ],
                "allowed_models": list(models),
                "max_concurrent": 2,
                "requests_per_minute": 30,
            },
        )
        return body["token"]

    def test_employee_key_infers_and_creates_private_trace(self) -> None:
        self.request("/api/team/settings", {"trace_mode": "full"})
        token = self.create_employee_key()

        status, response = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello team"}],
            },
            token=token,
        )
        _, trace_list = self.request("/api/traces", token=token)
        trace_id = trace_list["traces"][0]["id"]
        _, trace_response = self.request(f"/api/traces/{trace_id}", token=token)

        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(trace_response["trace"]["input"][0]["content"], "hello team")
        self.assertEqual(trace_response["trace"]["output"], "hello world")

    def test_employee_cannot_manage_keys_or_use_disallowed_model(self) -> None:
        token = self.create_employee_key()

        with self.assertRaises(HTTPError) as admin_error:
            self.request("/api/team/keys", token=token)
        with self.assertRaises(HTTPError) as model_error:
            self.request(
                "/api/chat",
                {
                    "model": "mlx-community/denied",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
                token=token,
            )

        self.assertEqual(admin_error.exception.code, 403)
        self.assertEqual(model_error.exception.code, 403)

    def test_deterministic_evaluation_summarizes_selected_traces(self) -> None:
        token = self.create_employee_key()
        self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "evaluate this"}],
                "stream": False,
            },
            token=token,
        )
        _, trace_list = self.request("/api/traces", token=token)

        status, body = self.request(
            "/api/evaluations",
            {
                "name": "Agent smoke test",
                "trace_ids": [trace_list["traces"][0]["id"]],
            },
            token=token,
        )

        self.assertEqual(status, 201)
        self.assertEqual(body["evaluation"]["summary"]["completion_rate"], 1.0)
        self.assertEqual(body["evaluation"]["evaluator"], "deterministic")

    def test_integration_catalog_contains_agent_connection_values(self) -> None:
        token = self.create_employee_key()

        _, body = self.request("/api/integrations", token=token)

        self.assertEqual(body["schema"], "machboost.integrations.v1")
        self.assertTrue(body["openai_base_url"].endswith("/v1"))
        self.assertIn("OLLAMA_HOST", body["clients"][1]["environment"])

    def test_workspace_memory_and_exact_cache_skip_second_inference(self) -> None:
        repository = Path(self.temporary.name) / "memory-repo"
        repository.mkdir()
        source = repository / "auth.py"
        source.write_text(
            "def authenticate(token):\n    return token\n", encoding="utf-8"
        )
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)
        payload = {
            "model": "mlx-community/example",
            "messages": [{"role": "user", "content": "Where is authenticate?"}],
            "machboost": {
                "workspace_id": workspace.id,
                "memory": {"mode": "private", "exact_cache": True},
            },
        }

        _, first = self.request("/v1/chat/completions", payload)
        _, second = self.request("/v1/chat/completions", payload)
        _, memories = self.request(f"/api/memory?workspace_id={workspace.id}")
        _, metrics = self.request("/api/cache/metrics")

        entry = next(iter(self.server.manager._models.values()))
        self.assertEqual(len(entry.accelerator.chat_calls), 1)
        self.assertNotIn("cache", first["machboost"])
        self.assertTrue(second["machboost"]["cache"]["hit"])
        self.assertEqual(second["choices"], first["choices"])
        self.assertEqual(len(memories["memories"]), 1)
        self.assertEqual(metrics["totals"]["exact_cache_hits"], 1)
        self.assertEqual(metrics["totals"]["avoided_prompt_tokens"], 12)

        source.write_text(
            "def authenticate(token):\n    return validate(token)\n",
            encoding="utf-8",
        )
        self.request(
            "/api/workspaces/index", {"workspace_id": workspace.id}
        )
        _, after_edit = self.request("/v1/chat/completions", payload)

        self.assertEqual(len(entry.accelerator.chat_calls), 2)
        self.assertNotIn("cache", after_edit["machboost"])

    def test_manual_team_memory_is_retrieved_and_can_be_deleted(self) -> None:
        repository = Path(self.temporary.name) / "shared-repo"
        repository.mkdir()
        (repository / "build.py").write_text("LOCKFILE = 'uv.lock'\n", encoding="utf-8")
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)

        _, created = self.request(
            "/api/memory",
            {
                "workspace_id": workspace.id,
                "scope": "team",
                "kind": "procedure",
                "title": "Repair dependency lock",
                "content": "Run uv lock after changing pyproject.toml.",
                "query_text": "dependency lock build",
                "confidence": 0.9,
                "validated_by": ["ci"],
            },
        )
        _, response = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/example",
                "messages": [
                    {"role": "user", "content": "How do I repair the dependency lock?"}
                ],
                "machboost": {
                    "workspace_id": workspace.id,
                    "memory": {"remember": False},
                },
            },
        )
        memory_id = created["memory"]["id"]
        _, deleted = self.request("/api/memory/delete", {"memory_ids": [memory_id]})

        entry = next(iter(self.server.manager._models.values()))
        messages = entry.accelerator.chat_calls[0][0]
        injected = "\n".join(str(message.get("content") or "") for message in messages)
        self.assertIn("Run uv lock", injected)
        system_messages = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        ]
        self.assertIn("repository evidence", system_messages[0])
        self.assertIn("prior team experience", system_messages[1])
        self.assertEqual(response["machboost"]["memory"]["retrieved"], 1)
        self.assertEqual(deleted["removed"], 1)

    def test_external_provider_configuration_and_routing(self) -> None:
        _, configured = self.request(
            "/api/providers",
            {
                "id": "fallback",
                "name": "Fallback API",
                "base_url": "https://inference.example.com",
                "models": ["mlx-community/example"],
                "api_key": "provider-secret",
                "monthly_budget_usd": 5,
                "input_cost_per_million": 1,
                "output_cost_per_million": 2,
            },
        )
        _, response = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "machboost": {
                    "route": {
                        "mode": "external_only",
                        "provider_id": "fallback",
                    }
                },
            },
        )
        _, providers = self.request("/api/providers")
        _, usage = self.request("/api/providers/usage")

        self.assertTrue(configured["provider"]["has_secret"])
        self.assertNotIn("api_key", configured["provider"])
        self.assertEqual(
            response["choices"][0]["message"]["content"], "external answer"
        )
        self.assertEqual(response["machboost"]["route"]["source"], "external")
        self.assertEqual(
            self.provider_calls[0][3]["Authorization"], "Bearer provider-secret"
        )
        self.assertEqual(providers["providers"][0]["id"], "fallback")
        self.assertEqual(usage["usage"][0]["requests"], 1)

    def test_external_provider_streaming_route_emits_compatible_sse(self) -> None:
        self.request(
            "/api/providers",
            {
                "id": "fallback",
                "name": "Fallback API",
                "base_url": "https://inference.example.com",
                "models": ["mlx-community/example"],
                "api_key": "provider-secret",
            },
        )

        _, headers, body = self.request(
            "/v1/chat/completions",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "machboost": {
                    "route": {
                        "mode": "external_only",
                        "provider_id": "fallback",
                    }
                },
            },
            raw=True,
        )

        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))
        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.decode("utf-8").splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(
            events[0]["choices"][0]["delta"]["content"], "external answer"
        )
        self.assertEqual(events[-1]["machboost"]["route"]["source"], "external")
        self.assertTrue(events[-1]["machboost"]["route"]["buffered_upstream"])
        self.assertEqual(len(self.server.manager._models), 0)

    def test_streaming_local_first_falls_back_before_headers_on_queue_overload(self) -> None:
        self.request(
            "/api/providers",
            {
                "id": "fallback",
                "name": "Fallback API",
                "base_url": "https://inference.example.com",
                "models": ["mlx-community/example"],
                "api_key": "provider-secret",
            },
        )

        with patch.object(
            self.server.manager,
            "chat",
            side_effect=RequestAdmissionError("queue is full", reason="queue_full"),
        ):
            _, headers, body = self.request(
                "/v1/chat/completions",
                {
                    "model": "mlx-community/example",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "machboost": {
                        "route": {
                            "mode": "local_first",
                            "provider_id": "fallback",
                        }
                    },
                },
                raw=True,
            )

        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertIn(b"external answer", body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))
        self.assertEqual(len(self.provider_calls), 1)
        self.assertEqual(len(self.server.manager._models), 0)


if __name__ == "__main__":
    unittest.main()
