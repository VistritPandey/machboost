from __future__ import annotations

import importlib.util
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelAlias:
    name: str
    mlx: Optional[str]
    hf: Optional[str]
    capability: str = "chat"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelResolution:
    requested: str
    model: str
    backend: str
    alias: Optional[str] = None


MODEL_ALIASES = {
    alias.name: alias
    for alias in (
        ModelAlias(
            "qwen2.5:0.5b",
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
        ModelAlias(
            "qwen2.5:1.5b",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        ModelAlias(
            "qwen2.5:3b",
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
        ModelAlias(
            "qwen2.5:7b",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        ModelAlias(
            "qwen2.5-coder:3b",
            "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "code",
        ),
        ModelAlias(
            "qwen2.5-coder:7b",
            "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "code",
        ),
        ModelAlias(
            "llama3.2:1b",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "meta-llama/Llama-3.2-1B-Instruct",
        ),
        ModelAlias(
            "llama3.2:3b",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
        ModelAlias(
            "qwen3:0.6b",
            "mlx-community/Qwen3-0.6B-4bit",
            "Qwen/Qwen3-0.6B",
        ),
        ModelAlias(
            "qwen3:1.7b",
            "mlx-community/Qwen3-1.7B-4bit",
            "Qwen/Qwen3-1.7B",
        ),
        ModelAlias(
            "qwen3:4b",
            "mlx-community/Qwen3-4B-4bit",
            "Qwen/Qwen3-4B",
        ),
        ModelAlias(
            "qwen3:8b",
            "mlx-community/Qwen3-8B-4bit",
            "Qwen/Qwen3-8B",
        ),
    )
}


def native_mlx_available() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def resolve_model(model: str, backend: str = "auto") -> ModelResolution:
    requested = model.strip()
    if not requested:
        raise ValueError("model name cannot be empty")

    path = Path(requested).expanduser()
    alias = MODEL_ALIASES.get(requested.lower())
    if alias is None:
        selected = select_backend_for_repo(requested, backend)
        return ModelResolution(requested=requested, model=str(path) if path.exists() else requested, backend=selected)

    selected = backend
    if selected == "auto":
        selected = "mlx" if native_mlx_available() and alias.mlx else "hf"
    resolved = alias.mlx if selected == "mlx" else alias.hf
    if not resolved:
        raise ValueError(f"model alias {requested!r} is not available for backend {selected!r}")
    return ModelResolution(requested=requested, model=resolved, backend=selected, alias=alias.name)


def select_backend_for_repo(model: str, backend: str = "auto") -> str:
    if backend != "auto":
        return backend
    normalized = model.lower()
    if normalized.startswith("mlx-community/") or "mlx" in normalized:
        return "mlx"
    return "hf"


def alias_rows() -> list[dict]:
    return [MODEL_ALIASES[name].to_dict() for name in sorted(MODEL_ALIASES)]
