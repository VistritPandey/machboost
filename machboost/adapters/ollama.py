from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Callable, Mapping, Optional
from urllib import error, request

DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


class OllamaHTTPError(RuntimeError):
    pass


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


class OllamaHTTPAdapter:
    def __init__(
        self,
        model: str,
        *,
        endpoint: Optional[str] = None,
        timeout: float = 120.0,
        keep_alive: Any = -1,
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
            payload["keep_alive"] = keep_alive
        if merged_options:
            payload["options"] = merged_options

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


def endpoint_from_env() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_ENDPOINT)


def normalize_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or DEFAULT_OLLAMA_ENDPOINT).strip().rstrip("/")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return "http://" + endpoint
