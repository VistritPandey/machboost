from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib import error, request

DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


class OllamaHTTPError(RuntimeError):
    pass


def normalize_ollama_keep_alive(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {
        "forever",
        "infinite",
        "infinity",
    }:
        return -1
    return value


@dataclass(frozen=True)
class OllamaCapabilities:
    backend: str = "ollama-http"
    native_verification: bool = False
    token_level_api: bool = False
    options_api: bool = True
    keep_alive_api: bool = True
    acceleration_mode: str = "wrapper"
    warning: str = (
        "Ollama's public HTTP API does not expose logits, token IDs, KV cache snapshots, "
        "or verifier hooks. MachBoost can benchmark and configure Ollama over HTTP, but "
        "exact draft-token acceleration needs a native runner hook or patched Ollama runner."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OllamaGenerateResult:
    model: str
    response: str
    done: bool
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OllamaGenerateResult":
        return cls(
            model=str(data.get("model", "")),
            response=str(data.get("response", "")),
            done=bool(data.get("done", False)),
            total_duration_ns=int(data.get("total_duration", 0) or 0),
            load_duration_ns=int(data.get("load_duration", 0) or 0),
            prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
            prompt_eval_duration_ns=int(data.get("prompt_eval_duration", 0) or 0),
            eval_count=int(data.get("eval_count", 0) or 0),
            eval_duration_ns=int(data.get("eval_duration", 0) or 0),
        )

    @property
    def tokens_per_second(self) -> float:
        if self.eval_count <= 0 or self.eval_duration_ns <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1_000_000_000)

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_eval_count <= 0 or self.prompt_eval_duration_ns <= 0:
            return 0.0
        return self.prompt_eval_count / (self.prompt_eval_duration_ns / 1_000_000_000)

    @property
    def total_ms(self) -> float:
        return self.total_duration_ns / 1_000_000

    @property
    def load_ms(self) -> float:
        return self.load_duration_ns / 1_000_000

    @property
    def eval_ms(self) -> float:
        return self.eval_duration_ns / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokens_per_second"] = self.tokens_per_second
        data["prompt_tokens_per_second"] = self.prompt_tokens_per_second
        data["total_ms"] = self.total_ms
        data["load_ms"] = self.load_ms
        data["eval_ms"] = self.eval_ms
        return data


@dataclass(frozen=True)
class OllamaChatChunk:
    model: str
    role: str
    content: str
    thinking: str
    tool_calls: tuple[Mapping[str, Any], ...]
    done: bool
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OllamaChatChunk":
        message = data.get("message") or {}
        role = "assistant"
        content = ""
        thinking = ""
        tool_calls: tuple[Mapping[str, Any], ...] = ()
        if isinstance(message, Mapping):
            role = str(message.get("role", "assistant"))
            content = str(message.get("content", ""))
            thinking = str(message.get("thinking", ""))
            raw_tool_calls = message.get("tool_calls") or []
            if isinstance(raw_tool_calls, list):
                tool_calls = tuple(
                    dict(item) for item in raw_tool_calls if isinstance(item, Mapping)
                )
        return cls(
            model=str(data.get("model", "")),
            role=role,
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            done=bool(data.get("done", False)),
            raw=data,
        )


@dataclass(frozen=True)
class OllamaPullStatus:
    status: str
    digest: str = ""
    total: int = 0
    completed: int = 0
    raw: Mapping[str, Any] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OllamaPullStatus":
        return cls(
            status=str(data.get("status", "")),
            digest=str(data.get("digest", "")),
            total=int(data.get("total", 0) or 0),
            completed=int(data.get("completed", 0) or 0),
            raw=data,
        )

    @property
    def progress(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.completed / self.total)


class OllamaHTTPAdapter:
    def __init__(
        self,
        model: str,
        *,
        endpoint: Optional[str] = None,
        timeout: float = 120.0,
        keep_alive: Any = "5m",
        default_options: Optional[Mapping[str, Any]] = None,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.endpoint = normalize_endpoint(endpoint_from_env() if endpoint is None else endpoint)
        self.timeout = float(timeout)
        self.keep_alive = keep_alive
        self.default_options = dict(default_options or {})
        self._opener = opener or request.urlopen

    def capabilities(self) -> OllamaCapabilities:
        return OllamaCapabilities()

    def generate(
        self,
        prompt: str,
        *,
        options: Optional[Mapping[str, Any]] = None,
        keep_alive: Any = None,
        images: Optional[Sequence[str]] = None,
        system: Optional[str] = None,
        format: Any = None,
        think: Any = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> OllamaGenerateResult:
        merged_options = dict(self.default_options)
        merged_options.update(dict(options or {}))
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if keep_alive is None:
            keep_alive = self.keep_alive
        if keep_alive is not None:
            payload["keep_alive"] = normalize_ollama_keep_alive(keep_alive)
        if merged_options:
            payload["options"] = merged_options
        if images:
            payload["images"] = list(images)
        if system is not None:
            payload["system"] = system
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        if logprobs is not None:
            payload["logprobs"] = bool(logprobs)
        if top_logprobs is not None:
            payload["top_logprobs"] = int(top_logprobs)

        data = self._json_request("POST", "/api/generate", payload)
        return OllamaGenerateResult.from_dict(data)

    def benchmark(
        self,
        prompt: str,
        *,
        tokens: int = 64,
        ctx: int = 4096,
        options: Optional[Mapping[str, Any]] = None,
    ) -> OllamaGenerateResult:
        benchmark_options = {"num_predict": int(tokens), "num_ctx": int(ctx)}
        benchmark_options.update(dict(options or {}))
        return self.generate(prompt, options=benchmark_options)

    def tags(self) -> dict[str, Any]:
        return self._json_request("GET", "/api/tags", None)

    def version(self) -> str:
        return str(self._json_request("GET", "/api/version", None).get("version", ""))

    def show(self, model: Optional[str] = None) -> dict[str, Any]:
        return self._json_request("POST", "/api/show", {"model": model or self.model})

    def installed_models(self) -> tuple[str, ...]:
        models = self.tags().get("models", [])
        names: list[str] = []
        if not isinstance(models, list):
            return ()
        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if name:
                names.append(str(name))
        return tuple(names)

    def has_model(self, model: Optional[str] = None) -> bool:
        return model_is_installed(model or self.model, self.installed_models())

    def pull(self, model: Optional[str] = None, *, stream: bool = True) -> Iterable[OllamaPullStatus]:
        payload = {
            "model": model or self.model,
            "stream": bool(stream),
        }
        for item in self._stream_json_request("POST", "/api/pull", payload):
            yield OllamaPullStatus.from_dict(item)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: Optional[Mapping[str, Any]] = None,
        keep_alive: Any = None,
        stream: bool = True,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        format: Any = None,
        think: Any = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> Iterable[OllamaChatChunk]:
        merged_options = dict(self.default_options)
        merged_options.update(dict(options or {}))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": bool(stream),
        }
        if keep_alive is None:
            keep_alive = self.keep_alive
        if keep_alive is not None:
            payload["keep_alive"] = normalize_ollama_keep_alive(keep_alive)
        if merged_options:
            payload["options"] = merged_options
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = think
        if logprobs is not None:
            payload["logprobs"] = bool(logprobs)
        if top_logprobs is not None:
            payload["top_logprobs"] = int(top_logprobs)

        for item in self._stream_json_request("POST", "/api/chat", payload):
            yield OllamaChatChunk.from_dict(item)

    def unload(self) -> None:
        self._json_request(
            "POST",
            "/api/generate",
            {"model": self.model, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def require_native_verifier(self) -> None:
        raise NotImplementedError(self.capabilities().warning)

    @staticmethod
    def with_draft_options(
        options: Optional[Mapping[str, Any]] = None,
        *,
        draft_num_predict: int = 4,
    ) -> dict[str, Any]:
        merged = dict(options or {})
        merged["draft_num_predict"] = int(draft_num_predict)
        return merged

    def _json_request(self, method: str, path: str, payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            self.endpoint + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(req, timeout=self.timeout) as response:
                raw = response.read()
                status = int(getattr(response, "status", getattr(response, "code", 200)))
        except error.HTTPError as exc:
            raw = exc.read()
            message = raw.decode("utf-8", errors="replace").strip()
            raise OllamaHTTPError(f"Ollama returned HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise OllamaHTTPError(f"Could not reach Ollama at {self.endpoint}: {exc.reason}") from exc

        if status < 200 or status >= 300:
            message = raw.decode("utf-8", errors="replace").strip()
            raise OllamaHTTPError(f"Ollama returned HTTP {status}: {message}")
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _stream_json_request(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]],
    ) -> Iterable[dict[str, Any]]:
        body = None
        headers = {"Accept": "application/x-ndjson, application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            self.endpoint + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(req, timeout=self.timeout) as response:
                status = int(getattr(response, "status", getattr(response, "code", 200)))
                if status < 200 or status >= 300:
                    raw = response.read()
                    message = raw.decode("utf-8", errors="replace").strip()
                    raise OllamaHTTPError(f"Ollama returned HTTP {status}: {message}")
                for line in iter_response_lines(response):
                    if not line:
                        continue
                    yield json.loads(line.decode("utf-8"))
        except error.HTTPError as exc:
            raw = exc.read()
            message = raw.decode("utf-8", errors="replace").strip()
            raise OllamaHTTPError(f"Ollama returned HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise OllamaHTTPError(f"Could not reach Ollama at {self.endpoint}: {exc.reason}") from exc


def endpoint_from_env() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_ENDPOINT)


def normalize_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or DEFAULT_OLLAMA_ENDPOINT).strip().rstrip("/")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return "http://" + endpoint


def model_is_installed(model: str, installed_models: Iterable[str]) -> bool:
    requested = normalize_model_name(model)
    for installed in installed_models:
        if normalize_model_name(installed) == requested:
            return True
    return False


def normalize_model_name(model: str) -> str:
    model = str(model).strip()
    if ":" not in model:
        return model + ":latest"
    return model


def iter_response_lines(response) -> Iterable[bytes]:
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            line = readline()
            if not line:
                break
            yield line.strip()
        return
    raw = response.read()
    for line in raw.splitlines():
        yield line.strip()
