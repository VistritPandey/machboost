from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
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
    ToolAwareTextStream,
    configure_native_prompt_cache,
    download_progress_class,
    extract_tool_calls,
    load_accelerator,
    model_config,
    normalize_tools,
    parse_keep_alive,
    request_affinity_key,
    result_content_and_tool_calls,
)
from machboost.team import TeamStore
from machboost.workspace import WorkspaceStore


class ToolCallParsingTests(unittest.TestCase):
    def test_normalize_tools_canonicalizes_semantically_identical_schemas(self):
        ordered = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path"},
                        "limit": {"type": "integer", "description": "Limit"},
                    },
                    "required": ["path"],
                },
            },
        }
        reordered = {
            "function": {
                "parameters": {
                    "required": ["path"],
                    "properties": {
                        "limit": {"description": "Limit", "type": "integer"},
                        "path": {"description": "Path", "type": "string"},
                    },
                    "type": "object",
                },
                "description": "Read a file.",
                "name": "read_file",
            },
            "type": "function",
        }

        first = normalize_tools([ordered])
        second = normalize_tools([reordered])

        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, separators=(",", ":")),
            json.dumps(second, separators=(",", ":")),
        )

    def test_tool_aware_stream_preserves_prose_and_hides_split_protocol(self):
        emitted = []
        stream = ToolAwareTextStream(emitted.append)

        for chunk in (
            "I will inspect it. ",
            "<atem:function_",
            "calls><atem:invoke name=\"read_file\">",
            "<atem:parameter name=\"path\">a.py</atem:parameter>",
            "</atem:invoke></atem:function_calls>",
        ):
            stream.feed(chunk)

        self.assertEqual("".join(emitted), "I will inspect it.")
        self.assertNotIn("atem", "".join(emitted))

    def test_extracts_muse_attribute_call_without_exposing_control_tokens(self):
        content, calls = extract_tool_calls(
            '<|start|>assistant to=list_files<|message|>'
            '<tool_call name="list_files" '
            'arguments={"path":"services/chat_assistant"}></tool_call>'
        )

        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_files")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "services/chat_assistant"},
        )

    def test_extracts_multiple_attribute_calls_and_keeps_visible_answer(self):
        content, calls = extract_tool_calls(
            'Checking both locations. '
            '<tool_call name="read_file" arguments={"path":"a.py"}></tool_call>'
            '<tool_call name="search_code" arguments={"query":"cancel"}></tool_call>'
        )

        self.assertEqual(content, "Checking both locations.")
        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["read_file", "search_code"],
        )

    def test_hides_control_tokens_and_truncated_tool_calls(self):
        content, calls = extract_tool_calls(
            'Visible preface. <|start|><tool_call name="read_file" arguments={'
        )

        self.assertEqual(content, "Visible preface.")
        self.assertEqual(calls, [])

    def test_hides_control_tokens_when_backend_supplies_tool_calls(self):
        content, calls = result_content_and_tool_calls(
            types.SimpleNamespace(
                text="<|start|>",
                tool_calls=(
                    {
                        "id": "call-1",
                        "function": {"name": "list_files", "arguments": {"path": "."}},
                    },
                ),
            )
        )

        self.assertEqual(content, "")
        self.assertEqual(calls[0]["function"]["name"], "list_files")

    def test_extracts_native_muse_atem_calls_and_cleans_recipient_tokens(self):
        content, calls = extract_tool_calls(
            '<|start|>assistant to=list_files<|message|>'
            '<atem:function_calls>\n'
            '<atem:invoke name="list_files">\n'
            '<atem:parameter name="path">services/app</atem:parameter>\n'
            '<atem:parameter name="limit">50</atem:parameter>\n'
            '</atem:invoke>\n'
            '</atem:function_calls><|eot|>'
        )

        self.assertEqual(content, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_files")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"path": "services/app", "limit": 50},
        )

    def test_rejects_invalid_tool_names_and_removes_bare_user_recipient(self):
        content, calls = extract_tool_calls(
            '<tool_call name="&quot;list_files<|message|" arguments={}></tool_call>'
            'to=user'
        )

        self.assertEqual(content, "")
        self.assertEqual(calls, [])


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


class PrefixDroppingAccelerator(FakeAccelerator):
    def generate_chat(self, messages, *, max_tokens, on_text=None, **_kwargs):
        self.chat_calls.append((messages, max_tokens, None))
        if on_text is not None:
            on_text("'re welcome!")
        return "You're welcome!", FakeStats(generated_tokens=4)


class NativePromptCacheConfigurationTests(unittest.TestCase):
    def test_vlm_requests_use_tenant_as_implicit_cache_affinity(self):
        self.assertEqual(
            request_affinity_key({"_tenant_key": "claude-desktop"}),
            "tenant:claude-desktop",
        )

    def test_explicit_and_image_affinity_take_precedence_over_tenant(self):
        options = {
            "_tenant_key": "claude-desktop",
            "affinity_key": "conversation-42",
        }
        self.assertEqual(
            request_affinity_key(options, image_sources=["image.png"]),
            "client:conversation-42",
        )
        self.assertTrue(
            request_affinity_key(
                {"_tenant_key": "claude-desktop"},
                image_sources=["image.png"],
            ).startswith("images:"),
        )

    def test_server_requests_enable_tenant_isolated_prompt_cache(self):
        accelerator = FakeAccelerator()

        configure_native_prompt_cache(
            accelerator,
            {"_tenant_key": "team-member", "prompt_cache_size": 4},
        )

        self.assertEqual(
            accelerator.service.prompt_cache_configs[-1],
            {
                "enabled": True,
                "max_size": 4,
                "max_bytes": 2 * 1024 * 1024 * 1024,
                "namespace": "tenant:team-member",
            },
        )

    def test_request_can_disable_prompt_cache_explicitly(self):
        accelerator = FakeAccelerator()

        configure_native_prompt_cache(
            accelerator,
            {"_tenant_key": "team-member", "prompt_cache": False},
        )

        self.assertFalse(
            accelerator.service.prompt_cache_configs[-1]["enabled"]
        )

    def test_workspace_namespace_takes_precedence_over_tenant(self):
        accelerator = FakeAccelerator()

        configure_native_prompt_cache(
            accelerator,
            {
                "_tenant_key": "team-member",
                "workspace_prefix_cache": True,
                "_prompt_cache_namespace": "workspace:repo:revision",
            },
        )

        self.assertEqual(
            accelerator.service.prompt_cache_configs[-1]["namespace"],
            "workspace:repo:revision",
        )


class TeamResidencyTests(unittest.TestCase):
    def test_team_server_keeps_models_resident_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = MachBoostHTTPServer(
                ("127.0.0.1", 0),
                workspace_store=WorkspaceStore(root / "workspaces"),
                team_store=TeamStore(root / "team.sqlite3"),
            )
            try:
                self.assertEqual(server.manager.default_keep_alive, -1.0)
            finally:
                server.server_close()

    def test_local_server_retains_the_idle_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            server = MachBoostHTTPServer(
                ("127.0.0.1", 0),
                workspace_store=WorkspaceStore(Path(temporary) / "workspaces"),
            )
            try:
                self.assertEqual(server.manager.default_keep_alive, 300.0)
            finally:
                server.server_close()


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


class StreamingToolCallingAccelerator(FakeAccelerator):
    def generate_chat(self, messages, *, max_tokens, on_text=None, **_kwargs):
        self.chat_calls.append((messages, max_tokens))
        chunks = (
            "I will inspect it. ",
            "<tool_",
            'call name="read_file" arguments={"path":"a.py"}></tool_call>',
        )
        if on_text is not None:
            for chunk in chunks:
                on_text(chunk)
        return "".join(chunks), FakeStats(generated_tokens=18)


class FakeVisionAccelerator:
    def __init__(self) -> None:
        self.chat_calls = []
        self.generate_calls = []
        self.cold_vision_calls = []
        self.vision_token_calls = []
        self.cache_keys = []
        self.cache_hits = 0

    def generate_chat(
        self,
        messages,
        *,
        max_tokens,
        context=None,
        on_text=None,
        on_thinking=None,
        use_vision_cache=True,
        temperature=0.0,
        enable_thinking=False,
        cold_vision_mode="off",
        cold_vision_max_edge=None,
        vision_token_mode="off",
        vision_token_ratio=0.35,
        vision_token_layer=None,
        vision_token_bucket=None,
        vision_calibration=None,
        tools=None,
        tool_choice="auto",
        reasoning_strength=None,
        cache_key=None,
    ):
        self.chat_calls.append(
            (
                messages,
                max_tokens,
                use_vision_cache,
                temperature,
                enable_thinking,
                tools,
                tool_choice,
                reasoning_strength,
            )
        )
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
        self.cache_keys.append(cache_key)
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
        on_thinking=None,
        images=None,
        use_vision_cache=True,
        temperature=0.0,
        enable_thinking=False,
        cold_vision_mode="off",
        cold_vision_max_edge=None,
        vision_token_mode="off",
        vision_token_ratio=0.35,
        vision_token_layer=None,
        vision_token_bucket=None,
        vision_calibration=None,
        cache_key=None,
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
        self.cache_keys.append(cache_key)
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


@dataclass
class FakeMuseStats(FakeStats):
    thinking: str = "Checking the evidence."
    tool_calls: tuple[dict, ...] = ()
    done_reason: str = "stop"
    native_speculative_decoding: bool = True


class FakeMuseAccelerator:
    def __init__(self) -> None:
        self.service = self
        self.chat_calls = []
        self.generate_calls = []

    def generate_chat(
        self,
        messages,
        *,
        max_tokens,
        context=None,
        on_text=None,
        on_thinking=None,
        temperature=0.0,
        enable_thinking=False,
        generation_options=None,
        stop_strings=None,
        tools=None,
        format=None,
        reasoning_strength=None,
        cancel_event=None,
        **runtime_options,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "context": context,
                "temperature": temperature,
                "enable_thinking": enable_thinking,
                "generation_options": generation_options,
                "tools": tools,
                "format": format,
                "reasoning_strength": reasoning_strength,
                "runtime_options": runtime_options,
            }
        )
        if on_thinking is not None:
            on_thinking("Checking the evidence.")
        if tools:
            call = {
                "id": "call_muse_1",
                "function": {
                    "index": 0,
                    "name": "lookup_score",
                    "arguments": {"team": "Argentina"},
                },
            }
            return "", FakeMuseStats(generated_tokens=7, tool_calls=(call,))
        if on_text is not None:
            on_text("Muse answer")
        return "Muse answer", FakeMuseStats(generated_tokens=3)

    def generate(self, prompt, **kwargs):
        self.generate_calls.append((prompt, kwargs))
        message = {"role": "user", "content": prompt}
        if kwargs.get("images"):
            message["images"] = kwargs["images"]
        return self.generate_chat(
            [message],
            **{key: value for key, value in kwargs.items() if key != "images"},
        )

    def close(self):
        pass


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
    def test_download_progress_survives_tqdm_initialization_updates(self):
        events = []

        class DisabledProgress:
            def __init__(self, *_args, **kwargs):
                self.disable = bool(kwargs.get("disable"))

            def update(self, _amount=1):
                return None

            def close(self):
                return None

        fake_tqdm = types.ModuleType("tqdm")
        fake_auto = types.ModuleType("tqdm.auto")
        fake_auto.tqdm = DisabledProgress
        fake_tqdm.auto = fake_auto
        with patch.dict(
            sys.modules,
            {"tqdm": fake_tqdm, "tqdm.auto": fake_auto},
        ):
            progress_type = download_progress_class(events.append, None)

        progress = progress_type(total=10, desc="weights.safetensors")
        progress.update(1)
        progress.close()

        self.assertEqual(events[-1]["file"], "weights.safetensors")
        self.assertEqual(events[-1]["completed"], 1)
        self.assertEqual(events[-1]["total"], 10)

    def test_muse_glimmer_uses_ollama_mlx_backend_and_loader(self):
        config = model_config("muse-glimmer:30b-mlx", {})
        sentinel = object()
        with patch(
            "machboost.adapters.ollama_mlx.OllamaMLXAccelerator.from_pretrained",
            return_value=sentinel,
        ) as load:
            result = load_accelerator(config)

        self.assertEqual(config.backend, "ollama-mlx")
        self.assertIs(result, sentinel)
        load.assert_called_once_with(
            "muse-glimmer:30b-mlx",
            context_paths=None,
            max_context_chars=200_000,
            keep_alive="forever",
        )

    def test_muse_glimmer_default_uses_native_mlx_vlm_repository(self):
        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            config = model_config("muse-glimmer:30b", {})

        self.assertEqual(config.backend, "mlx-vlm")
        self.assertEqual(config.model, "mlx-community/Muse-Glimmer-30B-4bit")

    def test_muse_glimmer_runtime_preserves_native_reasoning_and_tools(self):
        manager = RuntimeManager(loader=lambda config: FakeMuseAccelerator())
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_score",
                    "parameters": {"type": "object"},
                },
            }
        ]
        reasoning = []

        result = manager.chat(
            "muse-glimmer:30b-mlx",
            [{"role": "user", "content": "Who won?", "images": ["image"]}],
            options={
                "num_predict": 64,
                "num_ctx": 32_768,
                "_think": True,
                "_reasoning_strength": "high",
                "_tools": tools,
            },
            emit_thinking=reasoning.append,
        )

        self.assertEqual(result.backend, "ollama-mlx")
        self.assertEqual(result.thinking, "Checking the evidence.")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "lookup_score")
        self.assertEqual(reasoning, ["Checking the evidence."])
        entry = manager.ps()[0]
        self.assertIn("reasoning", entry["capabilities"])
        self.assertIn("tools", entry["capabilities"])

    def test_native_muse_glimmer_forwards_reasoning_to_mlx_vlm(self):
        manager = RuntimeManager(loader=lambda config: FakeMuseAccelerator())
        reasoning = []

        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            result = manager.chat(
                "muse-glimmer:30b",
                [{"role": "user", "content": "Inspect this."}],
                options={"_think": "high", "num_predict": 32},
                emit_thinking=reasoning.append,
            )

        self.assertEqual(result.backend, "mlx-vlm")
        self.assertEqual(result.thinking, "Checking the evidence.")
        self.assertEqual(reasoning, ["Checking the evidence."])
        entry = next(iter(manager._models.values()))
        call = entry.accelerator.chat_calls[0]
        self.assertEqual(call["enable_thinking"], "high")

    def test_pull_muse_glimmer_uses_ollama_lifecycle(self):
        events = []

        class PullStatus:
            def __init__(self, status, completed=0, total=0):
                self.status = status
                self.digest = "sha256:model"
                self.completed = completed
                self.total = total

        class Adapter:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def pull(self, *, stream):
                self.assert_stream = stream
                yield PullStatus("pulling model", 5, 10)
                yield PullStatus("success", 10, 10)

        manager = RuntimeManager(loader=lambda config: FakeMuseAccelerator())
        with (
            patch("machboost.adapters.ollama.OllamaHTTPAdapter", Adapter),
            patch("machboost.adapters.ollama_mlx.ensure_ollama_service"),
            patch(
                "machboost.server.ollama_model_manifest",
                return_value=Path("/cache/muse-manifest"),
            ),
        ):
            result = manager.pull("muse-glimmer:30b-mlx", progress=events.append)

        self.assertEqual(result["backend"], "ollama-mlx")
        self.assertEqual(result["resolved_model"], "muse-glimmer:30b-mlx")
        self.assertEqual(events[0]["completed"], 5)
        self.assertEqual(events[-1]["status"], "success")
    def test_dflash_model_config_defaults_to_adaptive_verification(self):
        config = model_config("qwen3.5:9b", {"backend": "dflash"})

        self.assertEqual(config.verify_mode, "adaptive")

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

    def test_dflash_alias_loads_accelerated_backend_without_request_options(self):
        loaded = []
        manager = RuntimeManager(
            loader=lambda config: loaded.append(config) or FakeAccelerator()
        )

        manager.chat(
            "qwen3.5:4b-dflash",
            [{"role": "user", "content": "Explain a mutex."}],
        )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].backend, "dflash")
        self.assertEqual(
            loaded[0].model,
            "mlx-community/Qwen3.5-4B-MLX-bf16",
        )

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

    def test_pull_dflash_alias_downloads_target_and_draft(self):
        calls = []
        events = []

        def download(*, repo_id, revision, tqdm_class):
            calls.append((repo_id, revision, tqdm_class))
            return f"/cache/{repo_id.replace('/', '--')}"

        manager = RuntimeManager(loader=lambda config: FakeAccelerator())
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = download
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            result = manager.pull(
                "qwen3.5:4b-dflash",
                revision="target-revision",
                progress=events.append,
            )

        self.assertEqual(
            [(repo_id, revision) for repo_id, revision, _ in calls],
            [
                ("mlx-community/Qwen3.5-4B-MLX-bf16", "target-revision"),
                ("z-lab/Qwen3.5-4B-DFlash", None),
            ],
        )
        self.assertEqual(result["backend"], "dflash")
        self.assertEqual(len(result["paths"]), 2)
        self.assertEqual(
            [(event["component"], event["repository"]) for event in events],
            [
                ("target", "mlx-community/Qwen3.5-4B-MLX-bf16"),
                ("draft", "z-lab/Qwen3.5-4B-DFlash"),
            ],
        )

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
        elif config.model.endswith("prefix-dropping"):
            accelerator = PrefixDroppingAccelerator()
        elif config.model.endswith("streaming-tool-calling"):
            accelerator = StreamingToolCallingAccelerator()
        elif config.model.endswith("tool-calling"):
            accelerator = ToolCallingAccelerator()
        elif config.backend == "ollama-mlx":
            accelerator = FakeMuseAccelerator()
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

    def test_extensions_crud_and_enabled_skill_injection(self):
        status, _, body = self.request(
            "/api/mcp/servers",
            {
                "name": "Local files",
                "transport": "stdio",
                "command": "example-mcp-server",
                "args": ["--safe"],
            },
        )
        self.assertEqual(status, 201)
        server = json.loads(body)["server"]
        self.assertEqual(server["transport"], "stdio")
        self.assertNotIn("env", server)

        status, _, body = self.request(
            "/api/skills",
            {
                "name": "House style",
                "instructions": "End every answer with TEST-SKILL.",
                "enabled": True,
            },
        )
        self.assertEqual(status, 201)
        skill = json.loads(body)["skill"]

        _, _, body = self.request("/api/extensions")
        extensions = json.loads(body)
        self.assertEqual(extensions["mcp_servers"][0]["id"], server["id"])
        self.assertEqual(extensions["skills"][0]["id"], skill["id"])
        self.assertEqual(len(extensions["gateway_tools"]), 2)

        self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        messages = self.loaded[0][1].chat_calls[0][0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("TEST-SKILL", messages[0]["content"])

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

    def test_open_server_accepts_required_client_placeholder_key(self):
        request = Request(
            self.base_url + "/v1/models",
            headers={"Authorization": "Bearer machboost"},
        )

        with urlopen(request, timeout=3.0) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["object"], "list")

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
        stable_context = next(
            message["content"]
            for message in messages
            if message["role"] == "system"
            and "stable repository map" in message["content"]
        )
        evidence = next(
            message["content"]
            for message in messages
            if message["role"] == "system"
            and "request-specific repository evidence" in message["content"]
        )
        self.assertIn("auth.py: authenticate_user", stable_context)
        self.assertNotIn("return lookup_user", stable_context)
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

    def test_muse_workspace_keeps_evidence_in_messages_without_duplicate_context(self):
        repository = Path(self.temporary.name) / "muse-repository"
        repository.mkdir()
        (repository / "permissions.py").write_text(
            "def can_view_report(user):\n"
            "    return user.has_permission('reports:view')\n",
            encoding="utf-8",
        )
        workspace = self.workspace_store.register(repository)
        self.workspace_store.index(workspace.id)

        _, _, body = self.request(
            "/api/chat",
            {
                "model": "muse-glimmer:30b-mlx",
                "workspace_id": workspace.id,
                "context": ["Caller-provided release policy."],
                "messages": [
                    {
                        "role": "user",
                        "content": "Where is report access checked?",
                    }
                ],
                "stream": False,
            },
        )

        response = json.loads(body)
        accelerator = self.loaded[0][1]
        call = accelerator.chat_calls[0]
        evidence = next(
            message["content"]
            for message in call["messages"]
            if message["role"] == "system"
            and "request-specific repository evidence" in message["content"]
        )
        self.assertIn("permissions.py:1-2", evidence)
        self.assertIn("can_view_report", evidence)
        self.assertEqual(call["context"], ["Caller-provided release policy."])
        self.assertEqual(response["machboost"]["backend"], "ollama-mlx")
        self.assertEqual(
            response["machboost"]["workspace"]["citations"][0]["path"],
            "permissions.py",
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

    def test_plain_chat_reconciles_a_stream_with_a_missing_prefix(self):
        _, _, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/prefix-dropping",
                "messages": [{"role": "user", "content": "thanks"}],
                "stream": True,
            },
        )

        rows = [json.loads(line) for line in body.splitlines()]
        streamed = "".join(row["message"]["content"] for row in rows)
        correction = next(
            row for row in rows if row.get("machboost", {}).get("full_content")
        )

        self.assertEqual(streamed, "'re welcome!")
        self.assertEqual(correction["message"]["content"], "")
        self.assertEqual(correction["machboost"]["full_content"], "You're welcome!")
        self.assertFalse(correction["done"])
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

            with self.assertRaises(HTTPError) as raised:
                urlopen(
                    Request(
                        f"http://{host}:{port}/v1/models",
                        headers={"Authorization": "Bearer machboost"},
                    ),
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

    def test_responses_endpoint_returns_coding_agent_function_calls(self):
        _, _, body = self.request(
            "/v1/responses",
            {
                "model": "mlx-community/tool-calling",
                "instructions": "Work inside the repository.",
                "input": "Inspect and search",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "Read a workspace file",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "search_repo",
                        "parameters": {"type": "object"},
                    },
                ],
                "parallel_tool_calls": True,
            },
        )

        response = json.loads(body)
        calls = [item for item in response["output"] if item["type"] == "function_call"]
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual([call["name"] for call in calls], ["read_file", "search_repo"])
        self.assertEqual(json.loads(calls[0]["arguments"])["path"], "a.py")
        self.assertEqual(self.loaded[0][1].chat_calls[0][0][0]["role"], "system")

    def test_responses_endpoint_streams_native_sse_events(self):
        _, headers, body = self.request(
            "/v1/responses",
            {
                "model": "mlx-community/example",
                "input": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                "request_id": "resp_test_stream",
            },
        )

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        event_types = [event["type"] for event in events]
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertEqual(event_types[0], "response.created")
        self.assertIn("response.output_text.delta", event_types)
        self.assertEqual(event_types[-1], "response.completed")
        self.assertEqual(events[-1]["response"]["id"], "resp_test_stream")

    def test_anthropic_messages_endpoint_maps_tools_and_results(self):
        _, _, body = self.request(
            "/v1/messages",
            {
                "model": "mlx-community/tool-calling",
                "system": "Edit only files in the workspace.",
                "messages": [{"role": "user", "content": "Inspect and search"}],
                "max_tokens": 128,
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "input_schema": {"type": "object"},
                    },
                    {
                        "name": "search_repo",
                        "input_schema": {"type": "object"},
                    },
                ],
            },
        )

        response = json.loads(body)
        calls = [block for block in response["content"] if block["type"] == "tool_use"]
        self.assertEqual(response["type"], "message")
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual([call["name"] for call in calls], ["read_file", "search_repo"])
        self.assertEqual(calls[0]["input"]["path"], "a.py")

    def test_anthropic_messages_endpoint_accepts_prior_thinking_blocks(self):
        status, _, _ = self.request(
            "/v1/messages",
            {
                "model": "mlx-community/example",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "I should inspect the repository.",
                                "signature": "signed-reasoning",
                            },
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "list_files",
                                "input": {"path": "."},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "README.md\nsrc",
                            },
                            {"type": "text", "text": "Summarize the structure."},
                        ],
                    },
                ],
                "max_tokens": 32,
            },
        )

        self.assertEqual(status, 200)
        forwarded = self.loaded[-1][1].chat_calls[0][0]
        self.assertNotIn("I should inspect the repository.", str(forwarded))
        self.assertIn("list_files", str(forwarded))
        self.assertIn("README.md", str(forwarded))

    def test_claude_desktop_discovers_routes_counts_tokens_and_routes_messages(self):
        catalog_patcher = patch(
            "machboost.server.catalog_rows",
            return_value=[
                {
                    "name": "mlx-community/example",
                    "repository": "mlx-community/example",
                    "backend": "mlx",
                    "capabilities": ["chat"],
                    "cached": True,
                }
            ],
        )
        catalog_patcher.start()
        self.addCleanup(catalog_patcher.stop)

        _, _, body = self.request("/v1/models")
        catalog = json.loads(body)
        routes = [item for item in catalog["data"] if item.get("type") == "model"]

        self.assertGreater(len(routes), 0)
        self.assertEqual(catalog["first_id"], routes[0]["id"])
        self.assertEqual(routes[0]["anthropic_family_tier"], "fable")
        self.assertEqual(routes[0]["max_tokens"], 64_000)

        _, _, count_body = self.request(
            "/v1/messages/count_tokens",
            {
                "model": routes[0]["id"],
                "messages": [{"role": "user", "content": "Count this prompt"}],
            },
        )
        self.assertGreater(json.loads(count_body)["input_tokens"], 0)

        _, _, message_body = self.request(
            "/v1/messages",
            {
                "model": routes[0]["id"],
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 16,
            },
        )
        response = json.loads(message_body)
        self.assertEqual(response["model"], routes[0]["display_name"])
        self.assertEqual(self.loaded[-1][0].model, resolve_model(routes[0]["display_name"]).model)

    def test_claude_desktop_deduplicates_aliases_for_the_same_runtime_model(self):
        duplicate_aliases = [
            {
                "name": "example:latest",
                "repository": "mlx-community/Example-4bit",
                "backend": "mlx",
                "capabilities": ["chat"],
                "cached": True,
            },
            {
                "name": "example:4bit",
                "repository": "mlx-community/Example-4bit",
                "backend": "mlx",
                "capabilities": ["chat"],
                "cached": True,
            },
        ]

        with patch("machboost.server.catalog_rows", return_value=duplicate_aliases):
            _, _, body = self.request("/v1/models")

        routes = [
            item
            for item in json.loads(body)["data"]
            if item.get("anthropic_family_tier")
        ]
        self.assertEqual(len(routes), 1)
        self.assertIn(routes[0]["display_name"], {"example:latest", "example:4bit"})

    def test_anthropic_messages_endpoint_streams_text_events(self):
        _, headers, body = self.request(
            "/v1/messages",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 32,
                "stream": True,
                "request_id": "msg_test_stream",
            },
        )

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        event_types = [event["type"] for event in events]
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertEqual(event_types[0], "message_start")
        self.assertIn("content_block_delta", event_types)
        self.assertEqual(event_types[-1], "message_stop")

    def test_anthropic_base64_image_maps_to_native_vision_content(self):
        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            self.request(
                "/v1/messages",
                {
                    "model": "qwen2.5-vl:3b",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aW1hZ2U=",
                                    },
                                },
                                {"type": "text", "text": "What is shown?"},
                            ],
                        }
                    ],
                    "max_tokens": 32,
                },
            )

        content = self.loaded[0][1].chat_calls[0][0][0]["content"]
        image = next(part for part in content if part["type"] == "image_url")
        self.assertTrue(image["image_url"]["url"].startswith("data:image/png;base64,"))

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

    def test_tool_enabled_chat_streams_prose_before_the_tool_event(self):
        _, _, body = self.request(
            "/api/chat",
            {
                "model": "mlx-community/streaming-tool-calling",
                "messages": [{"role": "user", "content": "Inspect it"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "stream": True,
            },
        )

        events = [json.loads(line) for line in body.splitlines()]
        visible = [
            event["message"]["content"]
            for event in events
            if event.get("message", {}).get("content")
        ]
        tool_index = next(
            index
            for index, event in enumerate(events)
            if event.get("message", {}).get("tool_calls")
        )
        visible_index = next(
            index
            for index, event in enumerate(events)
            if event.get("message", {}).get("content")
        )
        self.assertEqual("".join(visible), "I will inspect it.")
        self.assertLess(visible_index, tool_index)
        self.assertNotIn("<tool", body)

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

    def test_native_vlm_receives_tools_through_its_chat_template(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ]
        with patch("machboost.models.native_mlx_vlm_available", return_value=True):
            self.request(
                "/api/chat",
                {
                    "model": "qwen2.5-vl:3b",
                    "messages": [{"role": "user", "content": "Inspect it."}],
                    "tools": tools,
                    "tool_choice": "required",
                    "options": {"affinity_key": "conversation-1"},
                    "stream": False,
                },
            )

        call = self.loaded[0][1].chat_calls[0]
        self.assertEqual(call[0], [{"role": "user", "content": "Inspect it."}])
        self.assertEqual(call[5], tools)
        self.assertEqual(call[6], "required")
        self.assertEqual(self.loaded[0][1].cache_keys[0], "client:conversation-1")

    def test_muse_glimmer_ollama_chat_preserves_reasoning_vision_and_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_score",
                    "description": "Look up a match score",
                    "parameters": {"type": "object"},
                },
            }
        ]
        _, _, body = self.request(
            "/api/chat",
            {
                "model": "muse-glimmer:30b-mlx",
                "messages": [
                    {
                        "role": "user",
                        "content": "Who won?",
                        "images": ["aW1hZ2U="],
                    }
                ],
                "tools": tools,
                "think": "high",
                "stream": False,
                "options": {"num_ctx": 32_768, "num_predict": 64},
            },
        )

        response = json.loads(body)
        message = response["message"]
        self.assertEqual(message["thinking"], "Checking the evidence.")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup_score")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            {"team": "Argentina"},
        )
        config, accelerator = self.loaded[0]
        self.assertEqual(config.backend, "ollama-mlx")
        call = accelerator.chat_calls[0]
        self.assertEqual(call["messages"][0]["images"], ["aW1hZ2U="])
        self.assertEqual(call["enable_thinking"], "high")
        self.assertEqual(call["generation_options"]["num_ctx"], 32_768)

    def test_muse_glimmer_streams_reasoning_separately(self):
        _, _, body = self.request(
            "/api/chat",
            {
                "model": "muse-glimmer:30b-mlx",
                "messages": [{"role": "user", "content": "Explain briefly."}],
                "think": "low",
                "stream": True,
            },
        )

        events = [json.loads(line) for line in body.splitlines()]
        reasoning = [
            event["message"]["thinking"]
            for event in events
            if event.get("message", {}).get("thinking")
        ]
        visible = [
            event["message"]["content"]
            for event in events
            if event.get("message", {}).get("content")
        ]
        self.assertEqual(reasoning, ["Checking the evidence."])
        self.assertEqual(visible, ["Muse answer"])
        self.assertTrue(events[-1]["done"])
        self.assertEqual(
            self.loaded[0][1].chat_calls[0]["reasoning_strength"],
            "low",
        )

    def test_muse_glimmer_openai_chat_maps_reasoning_effort_and_tool_calls(self):
        _, _, body = self.request(
            "/v1/chat/completions",
            {
                "model": "muse-glimmer:30b-mlx",
                "messages": [{"role": "user", "content": "Use the score tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_score",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "reasoning_effort": "xhigh",
            },
        )

        response = json.loads(body)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(
            choice["message"]["reasoning_content"],
            "Checking the evidence.",
        )
        self.assertEqual(
            json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
            {"team": "Argentina"},
        )
        self.assertEqual(
            self.loaded[0][1].chat_calls[0]["reasoning_strength"],
            "xhigh",
        )

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

    def request(
        self,
        path,
        payload=None,
        *,
        token="admin-secret",
        raw=False,
        extra_headers=None,
    ):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {token}"}
        headers.update(extra_headers or {})
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

    def test_desktop_client_enrolls_and_reports_real_inference_requests(self) -> None:
        token = self.create_employee_key()

        _, connect = self.request("/api/team/connect", token=token)
        _, presence = self.request(
            "/api/team/presence",
            {
                "device_id": "desktop-1",
                "device_name": "Developer Mac",
                "app_version": "0.13.0",
                "workspace_name": "checkout-service",
                "workspace_fingerprint": "sha256:private-path-free",
                "model": "mlx-community/example",
            },
            token=token,
        )
        self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            token=token,
            extra_headers={"X-MachBoost-Device-ID": "desktop-1"},
        )
        _, clients = self.request("/api/team/clients")

        self.assertEqual(connect["schema"], "machboost.team-connect.v1")
        self.assertEqual(connect["principal"]["name"], "Coding agent")
        self.assertTrue(all(model["cached"] for model in connect["models"]))
        self.assertEqual(presence["client"]["workspace_name"], "checkout-service")
        self.assertNotIn("workspace_path", presence["client"])
        self.assertEqual(clients["clients"][0]["request_count"], 1)

        with self.assertRaises(HTTPError) as forbidden:
            self.request("/api/team/clients", token=token)
        self.assertEqual(forbidden.exception.code, 403)

    def test_client_can_request_model_but_only_admin_can_resolve_it(self) -> None:
        token = self.create_employee_key()

        status, created = self.request(
            "/api/team/model-requests",
            {
                "model": "mlx-community/Muse-Glimmer-30B-4bit",
                "device_id": "desktop-1",
                "note": "Vision coding work",
            },
            token=token,
        )
        request_id = created["request"]["id"]
        _, requests = self.request("/api/team/model-requests?status=pending")

        self.assertEqual(status, 201)
        self.assertEqual(len(requests["requests"]), 1)
        with self.assertRaises(HTTPError) as forbidden:
            self.request(
                "/api/team/model-requests/resolve",
                {"request_id": request_id, "status": "downloaded"},
                token=token,
            )
        self.assertEqual(forbidden.exception.code, 403)

        _, resolved = self.request(
            "/api/team/model-requests/resolve",
            {"request_id": request_id, "status": "downloaded"},
        )
        self.assertEqual(resolved["request"]["status"], "downloaded")

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
        self.assertTrue(
            any("stable repository map" in message for message in system_messages)
        )
        self.assertTrue(
            any("prior team experience" in message for message in system_messages)
        )
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

    def test_ollama_chat_can_route_directly_to_external_provider(self) -> None:
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

        _, response = self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "machboost": {
                    "route": {
                        "mode": "external_only",
                        "provider_id": "fallback",
                    }
                },
            },
        )

        self.assertEqual(response["message"]["content"], "external answer")
        self.assertTrue(response["done"])
        self.assertEqual(response["machboost"]["backend"], "external")
        self.assertEqual(response["machboost"]["route"]["source"], "external")
        self.assertEqual(response["eval_count"], 2)

    def test_ollama_chat_external_stream_preserves_ndjson_contract(self) -> None:
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
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "request_id": "chat-route-test",
                "machboost": {
                    "route": {
                        "mode": "external_only",
                        "provider_id": "fallback",
                    }
                },
            },
            raw=True,
        )
        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]

        self.assertEqual(headers.get_content_type(), "application/x-ndjson")
        self.assertEqual(events[0]["request_id"], "chat-route-test")
        self.assertEqual(events[0]["message"]["content"], "external answer")
        self.assertFalse(events[0]["done"])
        self.assertTrue(events[-1]["done"])
        self.assertEqual(events[-1]["machboost"]["route"]["provider_id"], "fallback")
        self.assertTrue(events[-1]["machboost"]["route"]["buffered_upstream"])

    def test_external_route_maps_local_model_to_provider_model(self) -> None:
        self.request(
            "/api/providers",
            {
                "id": "fallback",
                "name": "Fallback API",
                "base_url": "https://inference.example.com",
                "models": ["paid-model"],
                "api_key": "provider-secret",
            },
        )

        _, response = self.request(
            "/api/chat",
            {
                "model": "mlx-community/example",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "machboost": {
                    "route": {
                        "mode": "external_only",
                        "provider_id": "fallback",
                        "model": "paid-model",
                    }
                },
            },
        )

        self.assertEqual(response["message"]["content"], "external answer")
        self.assertEqual(self.provider_calls[-1][2]["model"], "paid-model")

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
