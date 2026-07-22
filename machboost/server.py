from __future__ import annotations

import gc
import json
import math
import re
import sys
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
from .scheduler import ReplicaPool, RequestAdmissionError
from .vision_auto import load_vision_calibration

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
DEFAULT_KEEP_ALIVE = 300.0
DEFAULT_REPLICAS = 1
DEFAULT_MAX_QUEUE = 64
DEFAULT_QUEUE_TIMEOUT = 300.0
MAX_REPLICAS = 8


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
    lazy: bool = False
    vision_cache_size: int = 20
    replicas: int = DEFAULT_REPLICAS


@dataclass
class LoadedModel:
    config: ModelConfig
    accelerator: Any
    loaded_at: float
    last_used_at: float
    keep_alive: float
    load_duration_s: float
    requests: int = 0
    warmups: int = 0
    warmup_duration_s: float = 0.0
    replica_accelerators: tuple[Any, ...] = ()
    max_queue: int = DEFAULT_MAX_QUEUE
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT

    def __post_init__(self) -> None:
        self.lock = threading.RLock()
        if not self.replica_accelerators:
            self.replica_accelerators = (self.accelerator,)
        self.scheduler = ReplicaPool(
            self.replica_accelerators,
            max_queue=self.max_queue,
            queue_timeout=self.queue_timeout,
        )

    @property
    def expires_at(self) -> Optional[float]:
        if self.keep_alive < 0:
            return None
        return self.last_used_at + self.keep_alive

    def to_dict(self, now: Optional[float] = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        expires_at = self.expires_at
        result = {
            "name": self.config.model,
            "model": self.config.model,
            "backend": self.config.backend,
            "loaded_for_seconds": max(0.0, now - self.loaded_at),
            "idle_seconds": max(0.0, now - self.last_used_at),
            "expires_in_seconds": None if expires_at is None else max(0.0, expires_at - now),
            "keep_alive_seconds": self.keep_alive,
            "load_duration_seconds": self.load_duration_s,
            "warmup_duration_seconds": self.warmup_duration_s,
            "requests": self.requests,
            "warmups": self.warmups,
            "context_paths": list(self.config.context_paths),
            "boost_enabled": self.config.boost_enabled,
            "scheduler": self.scheduler.snapshot(),
        }
        cache_info = getattr(self.accelerator, "cache_info", None)
        if callable(cache_info):
            result["vision_cache"] = cache_info()
        for worker, accelerator in zip(
            result["scheduler"]["workers"],
            self.replica_accelerators,
        ):
            worker_cache_info = getattr(accelerator, "cache_info", None)
            if callable(worker_cache_info):
                worker["vision_cache"] = worker_cache_info()
        result["capabilities"] = ["vision", "chat"] if self.config.backend.endswith("-vlm") else ["chat", "completion"]
        return result


@dataclass(frozen=True)
class GenerationResult:
    model: str
    backend: str
    text: str
    stats: dict[str, Any]
    load_duration_s: float
    total_duration_s: float
    prompt_eval_count: int = 0
    prompt_eval_duration_s: float = 0.0
    eval_duration_s: float = 0.0
    time_to_first_token_s: Optional[float] = None
    scheduler: Optional[dict[str, Any]] = None

    def ollama_metrics(self) -> dict[str, Any]:
        generated = int(self.stats.get("generated_tokens", 0))
        return {
            "done": True,
            "done_reason": "stop",
            "total_duration": int(self.total_duration_s * 1_000_000_000),
            "load_duration": int(self.load_duration_s * 1_000_000_000),
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": int(
                self.prompt_eval_duration_s * 1_000_000_000
            ),
            "eval_count": generated,
            "eval_duration": int(self.eval_duration_s * 1_000_000_000),
            "machboost": {
                "backend": self.backend,
                "stats": self.stats,
                "time_to_first_token_seconds": self.time_to_first_token_s,
                "scheduler": dict(self.scheduler or {}),
            },
        }


class RuntimeManager:
    def __init__(
        self,
        *,
        loader: Optional[Callable[[ModelConfig], Accelerator]] = None,
        clock: Callable[[], float] = time.monotonic,
        default_keep_alive: float = DEFAULT_KEEP_ALIVE,
        replicas: int = DEFAULT_REPLICAS,
        max_queue: int = DEFAULT_MAX_QUEUE,
        queue_timeout: float = DEFAULT_QUEUE_TIMEOUT,
    ) -> None:
        self.loader = loader or load_accelerator
        self.clock = clock
        self.default_keep_alive = float(default_keep_alive)
        self.replicas = validate_replicas(replicas)
        max_queue = int(max_queue)
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        queue_timeout = float(queue_timeout)
        if math.isnan(queue_timeout):
            raise ValueError("queue_timeout cannot be NaN")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
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
        config = model_config(model, options, replicas=self.replicas)
        ttl = parse_keep_alive(keep_alive, default=self.default_keep_alive)
        self.evict_expired()
        with self._lock:
            entry = self._models.get(config)
            if entry is not None:
                entry.keep_alive = ttl
                entry.last_used_at = self.clock()
                return entry, 0.0

            started = self.clock()
            accelerators = []
            try:
                for _ in range(config.replicas):
                    accelerators.append(self.loader(config))
            except Exception:
                for loaded_accelerator in accelerators:
                    release_accelerator(loaded_accelerator)
                raise
            finished = self.clock()
            load_duration = max(0.0, finished - started)
            entry = LoadedModel(
                config=config,
                accelerator=accelerators[0],
                loaded_at=finished,
                last_used_at=finished,
                keep_alive=ttl,
                load_duration_s=load_duration,
                replica_accelerators=tuple(accelerators),
                max_queue=self.max_queue,
                queue_timeout=self.queue_timeout,
            )
            self._models[config] = entry
            return entry, load_duration

    def warm(self, entry: LoadedModel) -> tuple[float, bool]:
        if entry.config.backend.endswith("-vlm"):
            return 0.0, False
        with entry.lock:
            if entry.warmups > 0:
                entry.last_used_at = self.clock()
                return 0.0, False
            started = self.clock()
            for accelerator in entry.replica_accelerators:
                accelerator.generate_chat(
                    [
                        {"role": "system", "content": "Answer concisely."},
                        {"role": "user", "content": "hi"},
                    ],
                    max_tokens=1,
                )
            finished = self.clock()
            duration = max(0.0, finished - started)
            entry.warmups += 1
            entry.warmup_duration_s = duration
            entry.last_used_at = finished
            return duration, True

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        options: Optional[dict[str, Any]] = None,
        keep_alive: Any = None,
        context: Optional[Iterable[str] | str] = None,
        emit: Optional[Callable[[str], None]] = None,
        on_admitted: Optional[Callable[[], None]] = None,
    ) -> GenerationResult:
        options = dict(options or {})
        entry, load_duration = self.get_or_load(model, options=options, keep_alive=keep_alive)
        max_tokens = int(options.get("num_predict", options.get("max_tokens", 128)))
        started = self.clock()
        first_emit_at: Optional[float] = None

        def timed_emit(text: str) -> None:
            nonlocal first_emit_at
            if text and first_emit_at is None:
                first_emit_at = self.clock()
            if emit is not None:
                emit(text)

        affinity_key = request_affinity_key(
            options,
            image_sources=message_image_sources(messages),
        )
        with entry.scheduler.slot(
            affinity_key=affinity_key,
            timeout=_optional_float(options.get("queue_timeout")),
        ) as lease:
            if on_admitted is not None:
                on_admitted()
            accelerator = lease.resource
            if entry.config.backend.endswith("-vlm"):
                text, stats = accelerator.generate_chat(
                    messages,
                    max_tokens=max_tokens,
                    context=context,
                    on_text=timed_emit if emit is not None else None,
                    use_vision_cache=not bool(options.get("no_vision_cache", False)),
                    temperature=float(options.get("temperature", 0.0)),
                    cold_vision_mode=str(options.get("cold_vision", "off")),
                    cold_vision_max_edge=_optional_int(options.get("vision_max_edge")),
                    vision_token_mode=str(options.get("vision_tokens", "off")),
                    vision_token_ratio=float(options.get("vision_token_ratio", 0.35)),
                    vision_token_layer=_optional_int(options.get("vision_token_layer")),
                    vision_token_bucket=_optional_int(options.get("vision_token_bucket")),
                    vision_calibration=load_vision_calibration(
                        options.get("vision_calibration")
                    ),
                )
            else:
                if messages_have_images(messages):
                    raise ValueError("image inputs require a vision model and VLM backend")
                text, stats = accelerator.generate_chat(
                    messages,
                    max_tokens=max_tokens,
                    context=context,
                    on_text=timed_emit if emit is not None else None,
                )
            with entry.lock:
                entry.requests += 1
                entry.last_used_at = self.clock()
        finished = self.clock()
        stats = stats_dict(stats)
        request_duration = max(0.0, finished - started)
        ttft = stats.get("time_to_first_token_seconds")
        if first_emit_at is not None:
            ttft = max(0.0, first_emit_at - started)
        elif ttft is not None:
            ttft = lease.queue_wait_seconds + float(ttft)
        prompt_eval_duration = float(stats.get("prompt_eval_seconds") or 0.0)
        eval_duration = float(stats.get("generation_seconds") or 0.0)
        if eval_duration <= 0:
            eval_duration = max(
                0.0,
                request_duration - lease.queue_wait_seconds - prompt_eval_duration,
            )
        return GenerationResult(
            model=model,
            backend=entry.config.backend,
            text=text,
            stats=stats,
            load_duration_s=load_duration,
            total_duration_s=request_duration + load_duration,
            prompt_eval_count=int(stats.get("prompt_tokens") or 0),
            prompt_eval_duration_s=prompt_eval_duration,
            eval_duration_s=eval_duration,
            time_to_first_token_s=ttft,
            scheduler=scheduler_result(lease, entry.config.replicas),
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
        images: Optional[Sequence[str]] = None,
        on_admitted: Optional[Callable[[], None]] = None,
    ) -> GenerationResult:
        options = dict(options or {})
        entry, load_duration = self.get_or_load(model, options=options, keep_alive=keep_alive)
        max_tokens = int(options.get("num_predict", options.get("max_tokens", 128)))
        started = self.clock()
        first_emit_at: Optional[float] = None

        def timed_emit(text: str) -> None:
            nonlocal first_emit_at
            if text and first_emit_at is None:
                first_emit_at = self.clock()
            if emit is not None:
                emit(text)

        affinity_key = request_affinity_key(options, image_sources=images)
        with entry.scheduler.slot(
            affinity_key=affinity_key,
            timeout=_optional_float(options.get("queue_timeout")),
        ) as lease:
            if on_admitted is not None:
                on_admitted()
            accelerator = lease.resource
            if entry.config.backend.endswith("-vlm"):
                text, stats = accelerator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    context=context,
                    on_text=timed_emit if emit is not None else None,
                    images=images,
                    use_vision_cache=not bool(options.get("no_vision_cache", False)),
                    temperature=float(options.get("temperature", 0.0)),
                    cold_vision_mode=str(options.get("cold_vision", "off")),
                    cold_vision_max_edge=_optional_int(options.get("vision_max_edge")),
                    vision_token_mode=str(options.get("vision_tokens", "off")),
                    vision_token_ratio=float(options.get("vision_token_ratio", 0.35)),
                    vision_token_layer=_optional_int(options.get("vision_token_layer")),
                    vision_token_bucket=_optional_int(options.get("vision_token_bucket")),
                    vision_calibration=load_vision_calibration(
                        options.get("vision_calibration")
                    ),
                )
            else:
                if images:
                    raise ValueError("image inputs require a vision model and VLM backend")
                text, stats = accelerator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    context=context,
                    on_text=timed_emit if emit is not None else None,
                )
            with entry.lock:
                entry.requests += 1
                entry.last_used_at = self.clock()
        finished = self.clock()
        stats = stats_dict(stats)
        request_duration = max(0.0, finished - started)
        ttft = stats.get("time_to_first_token_seconds")
        if first_emit_at is not None:
            ttft = max(0.0, first_emit_at - started)
        elif ttft is not None:
            ttft = lease.queue_wait_seconds + float(ttft)
        prompt_eval_duration = float(stats.get("prompt_eval_seconds") or 0.0)
        eval_duration = float(stats.get("generation_seconds") or 0.0)
        if eval_duration <= 0:
            eval_duration = max(
                0.0,
                request_duration - lease.queue_wait_seconds - prompt_eval_duration,
            )
        return GenerationResult(
            model=model,
            backend=entry.config.backend,
            text=text,
            stats=stats,
            load_duration_s=load_duration,
            total_duration_s=request_duration + load_duration,
            prompt_eval_count=int(stats.get("prompt_tokens") or 0),
            prompt_eval_duration_s=prompt_eval_duration,
            eval_duration_s=eval_duration,
            time_to_first_token_s=ttft,
            scheduler=scheduler_result(lease, entry.config.replicas),
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
            entry.scheduler.close(wait=True)
            with entry.lock:
                release_entry_accelerators(entry)
        return len(entries)

    def evict_expired(self) -> int:
        now = self.clock()
        entries: list[LoadedModel] = []
        with self._lock:
            for config, entry in list(self._models.items()):
                if entry.expires_at is None or entry.expires_at > now:
                    continue
                if not entry.lock.acquire(blocking=False):
                    continue
                if not entry.scheduler.try_close_idle():
                    entry.lock.release()
                    continue
                entries.append(self._models.pop(config))
        for entry in entries:
            try:
                release_entry_accelerators(entry)
            finally:
                entry.lock.release()
        return len(entries)

    def ps(self) -> list[dict[str, Any]]:
        self.evict_expired()
        now = self.clock()
        with self._lock:
            return [entry.to_dict(now) for entry in self._models.values()]

    def serving_config(self) -> dict[str, Any]:
        return {
            "text_replicas": self.replicas,
            "vision_replicas": 1,
            "max_queue": self.max_queue,
            "queue_timeout_seconds": self.queue_timeout,
        }

    def close(self) -> None:
        self.stop()


def validate_replicas(value: Any) -> int:
    replicas = int(value)
    if replicas < 1 or replicas > MAX_REPLICAS:
        raise ValueError(f"replicas must be between 1 and {MAX_REPLICAS}")
    return replicas


def model_config(
    model: str,
    options: dict[str, Any],
    *,
    replicas: int = DEFAULT_REPLICAS,
) -> ModelConfig:
    resolution = resolve_model(model, str(options.get("backend", "auto")))
    effective_replicas = (
        DEFAULT_REPLICAS
        if resolution.backend.endswith("-vlm")
        else validate_replicas(replicas)
    )
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
        lazy=bool(options.get("lazy", False)),
        vision_cache_size=max(1, int(options.get("vision_cache_size", 20))),
        replicas=effective_replicas,
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
    if config.backend == "mlx-vlm":
        from .adapters.mlx_vlm import MLXVLMAccelerator

        return MLXVLMAccelerator.from_pretrained(
            config.model,
            lazy=config.lazy,
            vision_cache_size=config.vision_cache_size,
        )
    if config.backend == "hf-vlm":
        raise ImportError(
            "The HF-VLM resident adapter is not available yet; use an mlx-community VLM with `--backend mlx-vlm`."
        )
    raise ValueError(f"unsupported backend: {config.backend}")


def release_accelerator(accelerator: Any) -> None:
    close = getattr(accelerator, "close", None)
    if callable(close):
        close()
    else:
        reset_cache = getattr(accelerator, "reset_cache", None)
        if not callable(reset_cache):
            reset_cache = getattr(getattr(accelerator, "service", None), "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
    del accelerator
    gc.collect()
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            mx.clear_cache()
        except (AttributeError, RuntimeError):
            pass


def release_entry_accelerators(entry: LoadedModel) -> None:
    released: set[int] = set()
    for accelerator in entry.replica_accelerators:
        identity = id(accelerator)
        if identity in released:
            continue
        released.add(identity)
        release_accelerator(accelerator)


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
        self._reaper_stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_expired_models,
            name="machboost-model-reaper",
            daemon=True,
        )
        self._reaper.start()

    def _reap_expired_models(self) -> None:
        while not self._reaper_stop.wait(1.0):
            self.manager.evict_expired()

    def server_close(self) -> None:
        self._reaper_stop.set()
        self._reaper.join(timeout=2.0)
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
            self.send_json(
                {
                    "status": "ok",
                    "version": __version__,
                    "serving": self.runtime.serving_config(),
                }
            )
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
            if path == "/api/load":
                model = required_string(payload, "model", aliases=("name",))
                entry, load_duration = self.runtime.get_or_load(
                    model,
                    options=dict(payload.get("options") or {}),
                    keep_alive=payload.get("keep_alive"),
                )
                warmup_duration = 0.0
                warmed = False
                if bool(payload.get("warmup", False)):
                    warmup_duration, warmed = self.runtime.warm(entry)
                self.send_json(
                    {
                        "status": "success",
                        "model": model,
                        "load_duration_seconds": load_duration,
                        "warmup_duration_seconds": warmup_duration,
                        "warmup_performed": warmed,
                        "instance": entry.to_dict(),
                    }
                )
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
        except RequestAdmissionError as exc:
            if not self.wfile.closed:
                self.send_error_json(503, str(exc), code=exc.reason)
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

        stream_started = False

        def on_admitted() -> None:
            nonlocal stream_started
            self.start_stream("application/x-ndjson")
            stream_started = True

        def emit(text: str) -> None:
            self.write_json_line(
                {
                    "model": model,
                    "created_at": utc_timestamp(),
                    "message": {"role": "assistant", "content": text},
                    "done": False,
                }
            )

        try:
            result = self.runtime.chat(
                model,
                messages,
                options=options,
                keep_alive=payload.get("keep_alive"),
                context=context,
                emit=emit,
                on_admitted=on_admitted,
            )
        except RequestAdmissionError:
            raise
        except Exception as exc:
            if not stream_started:
                raise
            self.write_json_line({"error": str(exc), "done": True})
            return
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
                images=normalize_image_list(payload.get("images")),
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

        stream_started = False

        def on_admitted() -> None:
            nonlocal stream_started
            self.start_stream("application/x-ndjson")
            stream_started = True

        def emit(text: str) -> None:
            self.write_json_line(
                {"model": model, "created_at": utc_timestamp(), "response": text, "done": False}
            )

        try:
            result = self.runtime.generate(
                model,
                prompt,
                options=options,
                keep_alive=payload.get("keep_alive"),
                context=context,
                emit=emit,
                images=normalize_image_list(payload.get("images")),
                on_admitted=on_admitted,
            )
        except RequestAdmissionError:
            raise
        except Exception as exc:
            if not stream_started:
                raise
            self.write_json_line({"error": str(exc), "done": True})
            return
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
                    "machboost": openai_machboost_result(result),
                }
            )
            return

        stream_started = False

        def on_admitted() -> None:
            nonlocal stream_started
            self.start_stream("text/event-stream")
            stream_started = True

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

        try:
            result = self.runtime.chat(
                model,
                messages,
                options=options,
                context=payload.get("context"),
                emit=emit,
                on_admitted=on_admitted,
            )
        except RequestAdmissionError:
            raise
        except Exception as exc:
            if not stream_started:
                raise
            self.write_sse({"error": {"message": str(exc), "type": "server_error"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "machboost": openai_machboost_result(result),
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
            result = self.runtime.generate(
                model,
                prompt,
                options=options,
                context=payload.get("context"),
                images=normalize_image_list(payload.get("images")),
            )
            self.send_json(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
                    "usage": usage_from_result(result),
                    "machboost": openai_machboost_result(result),
                }
            )
            return

        stream_started = False

        def on_admitted() -> None:
            nonlocal stream_started
            self.start_stream("text/event-stream")
            stream_started = True

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

        try:
            result = self.runtime.generate(
                model,
                prompt,
                options=options,
                context=payload.get("context"),
                emit=emit,
                images=normalize_image_list(payload.get("images")),
                on_admitted=on_admitted,
            )
        except RequestAdmissionError:
            raise
        except Exception as exc:
            if not stream_started:
                raise
            self.write_sse({"error": {"message": str(exc), "type": "server_error"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.write_sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "machboost": openai_machboost_result(result),
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

    def send_error_json(
        self,
        status: int,
        message: str,
        *,
        code: Optional[str] = None,
    ) -> None:
        if self.headers_sent:
            return
        payload = {"error": message}
        if code is not None:
            payload["code"] = code
        self.send_json(payload, status=status)

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


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError("message content must be text or a multimodal parts list")
        normalized_message: dict[str, Any] = {"role": role, "content": content}
        if "images" in message:
            normalized_message["images"] = normalize_image_list(message.get("images"))
        normalized.append(normalized_message)
    return normalized


def normalize_image_list(images: Any) -> list[str]:
    if images is None:
        return []
    if isinstance(images, (str, bytes)):
        images = (images,)
    if not isinstance(images, (list, tuple)):
        raise ValueError("images must be a string or list")
    return [image.decode("ascii") if isinstance(image, bytes) else str(image) for image in images]


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def request_affinity_key(
    options: dict[str, Any],
    *,
    image_sources: Optional[Iterable[str]] = None,
) -> Optional[str]:
    explicit = options.get("affinity_key")
    if explicit is not None and str(explicit):
        return f"client:{explicit}"
    sources = tuple(str(source) for source in (image_sources or ()) if source)
    if not sources:
        return None
    return "images:" + json.dumps(sources, separators=(",", ":"), ensure_ascii=True)


def message_image_sources(messages: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    sources: list[str] = []
    for message in messages:
        sources.extend(normalize_image_list(message.get("images")))
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type not in {"image_url", "input_image", "image"}:
                continue
            source = part.get("image_url", part.get("image", part.get("url")))
            if isinstance(source, dict):
                source = source.get("url", source)
            sources.append(json.dumps(source, sort_keys=True, default=str))
    return tuple(sources)


def scheduler_result(lease: Any, replicas: int) -> dict[str, Any]:
    return {
        "replica": int(lease.index),
        "replicas": int(replicas),
        "queue_wait_seconds": float(lease.queue_wait_seconds),
    }


def messages_have_images(messages: Sequence[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("images"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict)
            and str(part.get("type") or "") in {"image_url", "input_image", "image"}
            for part in content
        ):
            return True
    return False


def openai_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = dict(payload.get("machboost_options") or {})
    if "max_tokens" in payload:
        options["max_tokens"] = payload["max_tokens"]
    for key in ("affinity_key", "queue_timeout"):
        if key in payload:
            options[key] = payload[key]
    return options


def usage_from_result(result: GenerationResult) -> dict[str, int]:
    prompt = int(result.stats.get("prompt_tokens", 0))
    completion = int(result.stats.get("generated_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def openai_machboost_result(result: GenerationResult) -> dict[str, Any]:
    return {
        **result.stats,
        "backend": result.backend,
        "time_to_first_token_seconds": result.time_to_first_token_s,
        "scheduler": dict(result.scheduler or {}),
    }


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    manager: Optional[RuntimeManager] = None,
    ready: Optional[Callable[[MachBoostHTTPServer], None]] = None,
    replicas: int = DEFAULT_REPLICAS,
    max_queue: int = DEFAULT_MAX_QUEUE,
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT,
) -> None:
    if manager is None:
        manager = RuntimeManager(
            replicas=replicas,
            max_queue=max_queue,
            queue_timeout=queue_timeout,
        )
    server = MachBoostHTTPServer((host, int(port)), manager=manager)
    if ready is not None:
        ready(server)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
