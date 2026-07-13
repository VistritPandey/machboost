from __future__ import annotations

import gc
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
from urllib.parse import urlparse

from . import __version__
from .accelerator import Accelerator
from .models import model_targets, resolve_model

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
DEFAULT_KEEP_ALIVE = -1.0


def select_backend(model: str, backend: str = "auto") -> str:
    return resolve_model(model, backend).backend


def parse_keep_alive(value: Any, *, default: float = DEFAULT_KEEP_ALIVE) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "default"}:
        return float(default)
    if text in {"-1", "forever", "infinite", "infinity"}:
        return -1.0
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([smhd]?)", text)
    if match is None:
        raise ValueError(f"invalid keep_alive value: {value!r}")
    amount = float(match.group(1))
    multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[match.group(2)]
    return amount * multiplier


@dataclass(frozen=True)
class ModelConfig:
    model: str
    backend: str
    context_paths: tuple[str, ...] = ()
    max_context_chars: int = 200_000
    ngram: int = 2
    max_draft_tokens: int = 8
    candidate_limit: int = 1
    reentry_probe_tokens: int = 0
    boost_enabled: bool = True
    device: str = "auto"
    local_files_only: bool = False
    cache_enabled: bool = True


@dataclass
class LoadedModel:
    config: ModelConfig
    accelerator: Accelerator
    loaded_at: float
    last_used_at: float
    keep_alive: float
    load_duration_s: float
    requests: int = 0

    def __post_init__(self) -> None:
        self.lock = threading.RLock()

    @property
    def expires_at(self) -> Optional[float]:
        if self.keep_alive < 0:
            return None
        return self.last_used_at + self.keep_alive

    def to_dict(self, now: Optional[float] = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        expires_at = self.expires_at
        return {
            "name": self.config.model,
            "model": self.config.model,
            "backend": self.config.backend,
            "loaded_for_seconds": max(0.0, now - self.loaded_at),
            "idle_seconds": max(0.0, now - self.last_used_at),
            "expires_in_seconds": None if expires_at is None else max(0.0, expires_at - now),
            "keep_alive_seconds": self.keep_alive,
            "load_duration_seconds": self.load_duration_s,
            "requests": self.requests,
            "context_paths": list(self.config.context_paths),
            "boost_enabled": self.config.boost_enabled,
        }


@dataclass(frozen=True)
class GenerationResult:
    model: str
    backend: str
    text: str
    stats: dict[str, Any]
    load_duration_s: float
    total_duration_s: float

    def ollama_metrics(self) -> dict[str, Any]:
        generated = int(self.stats.get("generated_tokens", 0))
        return {
            "done": True,
            "done_reason": "stop",
            "total_duration": int(self.total_duration_s * 1_000_000_000),
            "load_duration": int(self.load_duration_s * 1_000_000_000),
            "eval_count": generated,
            "eval_duration": int(max(0.0, self.total_duration_s - self.load_duration_s) * 1_000_000_000),
            "machboost": {
                "backend": self.backend,
                "stats": self.stats,
            },
        }


class RuntimeManager:
    def __init__(
        self,
        *,
        loader: Optional[Callable[[ModelConfig], Accelerator]] = None,
        clock: Callable[[], float] = time.monotonic,
        default_keep_alive: float = DEFAULT_KEEP_ALIVE,
    ) -> None:
        self.loader = loader or load_accelerator
        self.clock = clock
        self.default_keep_alive = float(default_keep_alive)
        self._models: dict[ModelConfig, LoadedModel] = {}
        self._lock = threading.RLock()

    def get_or_load(
        self,
        model: str,
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = None,
    ) -> tuple[LoadedModel, float]:
        options = dict(options or {})
        config = model_config(model, options)
        ttl = parse_keep_alive(keep_alive, default=self.default_keep_alive)
        self.evict_expired()
        with self._lock:
            entry = self._models.get(config)
            if entry is not None:
                entry.keep_alive = ttl
                entry.last_used_at = self.clock()
                return entry, 0.0

            started = self.clock()
            accelerator = self.loader(config)
            finished = self.clock()
            load_duration = max(0.0, finished - started)
            entry = LoadedModel(
                config=config,
                accelerator=accelerator,
                loaded_at=finished,
                last_used_at=finished,
                keep_alive=ttl,
                load_duration_s=load_duration,
            )
            self._models[config] = entry
            return entry, load_duration

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = None,
        context: Optional[Iterable[str] | str] = None,
        emit: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        options = dict(options or {})
        entry, load_duration = self.get_or_load(model, options=options, keep_alive=keep_alive)
        max_tokens = int(options.get("num_predict", options.get("max_tokens", 128)))
        started = self.clock()
        with entry.lock:
            text, stats = entry.accelerator.generate_chat(
                messages,
                max_tokens=max_tokens,
                context=context,
                on_text=emit,
            )
            entry.requests += 1
            entry.last_used_at = self.clock()
        return GenerationResult(
            model=model,
            backend=entry.config.backend,
            text=text,
            stats=stats_dict(stats),
            load_duration_s=load_duration,
            total_duration_s=max(0.0, self.clock() - started + load_duration),
        )

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = None,
        context: Optional[Iterable[str] | str] = None,
        emit: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        options = dict(options or {})
        entry, load_duration = self.get_or_load(model, options=options, keep_alive=keep_alive)
        max_tokens = int(options.get("num_predict", options.get("max_tokens", 128)))
        started = self.clock()
        with entry.lock:
            text, stats = entry.accelerator.generate(
                prompt,
                max_tokens=max_tokens,
                context=context,
                on_text=emit,
            )
            entry.requests += 1
            entry.last_used_at = self.clock()
        return GenerationResult(
            model=model,
            backend=entry.config.backend,
            text=text,
            stats=stats_dict(stats),
            load_duration_s=load_duration,
            total_duration_s=max(0.0, self.clock() - started + load_duration),
        )

    def pull(self, model: str, *, revision: Optional[str] = None) -> dict[str, Any]:
        path = Path(model).expanduser()
        if path.exists():
            return {"status": "success", "model": model, "path": str(path.resolve())}
        resolution = resolve_model(model)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Model downloads require `huggingface-hub` or a MachBoost MLX/HF extra.") from exc
        downloaded = snapshot_download(repo_id=resolution.model, revision=revision)
        return {
            "status": "success",
            "model": model,
            "resolved_model": resolution.model,
            "backend": resolution.backend,
            "path": str(downloaded),
        }

    def stop(self, model: Optional[str] = None) -> int:
        targets = model_targets(model) if model else None
        with self._lock:
            configs = [
                config for config in self._models if targets is None or config.model in targets
            ]
            entries = [self._models.pop(config) for config in configs]
        for entry in entries:
            release_accelerator(entry.accelerator)
        return len(entries)

    def evict_expired(self) -> int:
        now = self.clock()
        with self._lock:
            configs = [
                config
                for config, entry in self._models.items()
                if entry.expires_at is not None and entry.expires_at <= now
            ]
            entries = [self._models.pop(config) for config in configs]
        for entry in entries:
            release_accelerator(entry.accelerator)
        return len(entries)

    def ps(self) -> list[dict[str, Any]]:
        self.evict_expired()
        now = self.clock()
        with self._lock:
            return [entry.to_dict(now) for entry in self._models.values()]

    def close(self) -> None:
        self.stop()


def model_config(model: str, options: dict[str, Any]) -> ModelConfig:
    resolution = resolve_model(model, str(options.get("backend", "auto")))
    paths = options.get("context_paths") or options.get("context") or ()
    if isinstance(paths, str):
        paths = (paths,)
    return ModelConfig(
        model=resolution.model,
        backend=resolution.backend,
        context_paths=tuple(str(path) for path in paths),
        max_context_chars=int(options.get("max_context_chars", 200_000)),
        ngram=int(options.get("ngram", 2)),
        max_draft_tokens=int(options.get("max_draft_tokens", 8)),
        candidate_limit=max(1, int(options.get("candidate_limit", 1))),
        reentry_probe_tokens=max(0, int(options.get("reentry_probe_tokens", 0))),
        boost_enabled=not bool(options.get("no_boost", False)),
        device=str(options.get("device", "auto")),
        local_files_only=bool(options.get("local_files_only", False)),
        cache_enabled=not bool(options.get("strict", False)),
    )


def load_accelerator(config: ModelConfig) -> Accelerator:
    common = {
        "context_paths": config.context_paths or None,
        "max_context_chars": config.max_context_chars,
        "ngram": config.ngram,
        "max_draft_tokens": config.max_draft_tokens,
        "candidate_limit": config.candidate_limit,
        "reentry_probe_tokens": config.reentry_probe_tokens,
        "boost_enabled": config.boost_enabled,
    }
    if config.backend == "mlx":
        return Accelerator.from_mlx(config.model, cache_enabled=config.cache_enabled, **common)
    if config.backend == "hf":
        return Accelerator.from_huggingface(
            config.model,
            device=config.device,
            local_files_only=config.local_files_only,
            **common,
        )
    raise ValueError(f"unsupported backend: {config.backend}")


def release_accelerator(accelerator: Accelerator) -> None:
    reset_cache = getattr(getattr(accelerator, "service", None), "reset_cache", None)
    if callable(reset_cache):
        reset_cache()
    del accelerator
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (AttributeError, ImportError):
        pass


def stats_dict(stats: Any) -> dict[str, Any]:
    try:
        return asdict(stats)
    except TypeError:
        if isinstance(stats, dict):
            return dict(stats)
        return {
            name: getattr(stats, name)
            for name in dir(stats)
            if not name.startswith("_") and isinstance(getattr(stats, name), (int, float, str, bool, type(None)))
        }


class MachBoostHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, manager: Optional[RuntimeManager] = None) -> None:
        self.manager = manager or RuntimeManager()
        super().__init__(server_address, MachBoostRequestHandler)

    def server_close(self) -> None:
        self.manager.close()
        super().server_close()


class MachBoostRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"MachBoost/{__version__}"

    @property
    def runtime(self) -> RuntimeManager:
        return self.server.manager  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/health", "/healthz"}:
            self.send_json({"status": "ok", "version": __version__})
            return
        if path == "/api/version":
            self.send_json({"version": __version__})
            return
        if path == "/api/ps":
            self.send_json({"models": self.runtime.ps()})
            return
        if path in {"/api/tags", "/v1/models"}:
            models = self.runtime.ps()
            if path == "/v1/models":
                self.send_json(
                    {
                        "object": "list",
                        "data": [
                            {"id": item["model"], "object": "model", "owned_by": "machboost"}
                            for item in models
                        ],
                    }
                )
            else:
                self.send_json({"models": models})
            return
        self.send_error_json(404, f"unknown endpoint: {path}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path == "/api/chat":
                self.handle_ollama_chat(payload)
                return
            if path == "/api/generate":
                self.handle_ollama_generate(payload)
                return
            if path == "/api/pull":
                model = required_string(payload, "model", aliases=("name",))
                result = self.runtime.pull(model, revision=payload.get("revision"))
                self.send_json(result)
                return
            if path == "/api/stop":
                model = payload.get("model") or payload.get("name")
                unloaded = self.runtime.stop(str(model)) if model else self.runtime.stop()
                self.send_json({"status": "success", "unloaded": unloaded})
                return
            if path == "/api/shutdown":
                unloaded = self.runtime.stop()
                self.send_json({"status": "success", "unloaded": unloaded})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if path == "/api/show":
                model = str(payload.get("model") or payload.get("name") or "")
                resolved_model = resolve_model(model).model
                matches = [item for item in self.runtime.ps() if item["model"] == resolved_model]
                self.send_json(
                    {
                        "model": model,
                        "resolved_model": resolved_model,
                        "loaded": bool(matches),
                        "instances": matches,
                    }
                )
                return
            if path == "/v1/chat/completions":
                self.handle_openai_chat(payload)
                return
            if path == "/v1/completions":
                self.handle_openai_completion(payload)
                return
            self.send_error_json(404, f"unknown endpoint: {path}")
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            if not self.wfile.closed:
                self.send_error_json(400, str(exc))
        except Exception as exc:
            if not self.wfile.closed:
                self.send_error_json(500, f"server error: {exc}")

    def handle_ollama_chat(self, payload: dict[str, Any]) -> None:
        model = required_string(payload, "model")
        messages = normalize_messages(payload.get("messages") or ())
        options = dict(payload.get("options") or {})
        context = payload.get("context")
        if not bool(payload.get("stream", True)):
            result = self.runtime.chat(
                model,
                messages,
                options=options,
                keep_alive=payload.get("keep_alive"),
                context=context,
            )
            body = {
                "model": model,
                "created_at": utc_timestamp(),
                "message": {"role": "assistant", "content": result.text},
                **result.ollama_metrics(),
            }
            self.send_json(body)
            return

        self.start_stream("application/x-ndjson")

        def emit(text: str) -> None:
            self.write_json_line(
                {
                    "model": model,
                    "created_at": utc_timestamp(),
                    "message": {"role": "assistant", "content": text},
                    "done": False,
                }
            )

        result = self.runtime.chat(
            model,
            messages,
            options=options,
            keep_alive=payload.get("keep_alive"),
            context=context,
            emit=emit,
        )
        self.write_json_line(
            {
                "model": model,
                "created_at": utc_timestamp(),
                "message": {"role": "assistant", "content": ""},
                **result.ollama_metrics(),
            }
        )

    def handle_ollama_generate(self, payload: dict[str, Any]) -> None:
        model = required_string(payload, "model")
        prompt = str(payload.get("prompt") or "")
        options = dict(payload.get("options") or {})
        context = payload.get("context")
        if not bool(payload.get("stream", True)):
            result = self.runtime.generate(
                model,
                prompt,
                options=options,
                keep_alive=payload.get("keep_alive"),
                context=context,
            )
            self.send_json(
                {
                    "model": model,
                    "created_at": utc_timestamp(),
                    "response": result.text,
                    **result.ollama_metrics(),
                }
            )
            return

        self.start_stream("application/x-ndjson")

        def emit(text: str) -> None:
            self.write_json_line(
                {"model": model, "created_at": utc_timestamp(), "response": text, "done": False}
            )

        result = self.runtime.generate(
            model,
            prompt,
            options=options,
            keep_alive=payload.get("keep_alive"),
            context=context,
            emit=emit,
        )
        self.write_json_line(
            {"model": model, "created_at": utc_timestamp(), "response": "", **result.ollama_metrics()}
        )

    def handle_openai_chat(self, payload: dict[str, Any]) -> None:
        model = required_string(payload, "model")
        messages = normalize_messages(payload.get("messages") or ())
        options = openai_options(payload)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        if not bool(payload.get("stream", False)):
            result = self.runtime.chat(model, messages, options=options, context=payload.get("context"))
            self.send_json(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}
                    ],
                    "usage": usage_from_result(result),
                    "machboost": result.stats,
                }
            )
            return

        self.start_stream("text/event-stream")

        def emit(text: str) -> None:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
            )

        result = self.runtime.chat(model, messages, options=options, context=payload.get("context"), emit=emit)
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "machboost": result.stats,
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def handle_openai_completion(self, payload: dict[str, Any]) -> None:
        model = required_string(payload, "model")
        prompt = str(payload.get("prompt") or "")
        options = openai_options(payload)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        if not bool(payload.get("stream", False)):
            result = self.runtime.generate(model, prompt, options=options, context=payload.get("context"))
            self.send_json(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
                    "usage": usage_from_result(result),
                    "machboost": result.stats,
                }
            )
            return

        self.start_stream("text/event-stream")

        def emit(text: str) -> None:
            self.write_sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": None}],
                }
            )

        result = self.runtime.generate(model, prompt, options=options, context=payload.get("context"), emit=emit)
        self.write_sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "machboost": result.stats,
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, message: str) -> None:
        if self.headers_sent:
            return
        self.send_json({"error": message}, status=status)

    @property
    def headers_sent(self) -> bool:
        return getattr(self, "_headers_buffer", None) == []

    def start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def write_json_line(self, payload: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
        self.wfile.flush()

    def write_sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        return


def required_string(payload: dict[str, Any], key: str, *, aliases: Sequence[str] = ()) -> str:
    raw = payload.get(key)
    if raw is None:
        for alias in aliases:
            raw = payload.get(alias)
            if raw is not None:
                break
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("MachBoost v0.2 text endpoints require string message content")
        normalized.append({"role": role, "content": content})
    return normalized


def openai_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = dict(payload.get("machboost_options") or {})
    if "max_tokens" in payload:
        options["max_tokens"] = payload["max_tokens"]
    return options


def usage_from_result(result: GenerationResult) -> dict[str, int]:
    completion = int(result.stats.get("generated_tokens", 0))
    return {"prompt_tokens": 0, "completion_tokens": completion, "total_tokens": completion}


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    manager: Optional[RuntimeManager] = None,
    ready: Optional[Callable[[MachBoostHTTPServer], None]] = None,
) -> None:
    server = MachBoostHTTPServer((host, int(port)), manager=manager)
    if ready is not None:
        ready(server)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
