from __future__ import annotations

import importlib
import importlib.util
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


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
        ModelAlias(
            "qwen2-vl:2b",
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "Qwen/Qwen2-VL-2B-Instruct",
            "vision",
        ),
        ModelAlias(
            "qwen2.5-vl:3b",
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "vision",
        ),
        ModelAlias(
            "qwen3-vl:2b",
            "mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "Qwen/Qwen3-VL-2B-Instruct",
            "vision",
        ),
        ModelAlias(
            "qwen3-vl:4b",
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            "Qwen/Qwen3-VL-4B-Instruct",
            "vision",
        ),
        ModelAlias(
            "qwen3-vl:8b",
            "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "Qwen/Qwen3-VL-8B-Instruct",
            "vision",
        ),
        ModelAlias(
            "qwen3.5:0.8b",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "Qwen/Qwen3.5-0.8B",
            "vision",
        ),
        ModelAlias(
            "qwen3.5:4b",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            "Qwen/Qwen3.5-4B",
            "vision",
        ),
        ModelAlias(
            "qwen3.5:9b",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
            "Qwen/Qwen3.5-9B",
            "vision",
        ),
    )
}


MODEL_RESOURCE_HINTS = {
    "qwen2.5:0.5b": (0.5, 4.0),
    "qwen2.5:1.5b": (1.0, 4.0),
    "qwen2.5:3b": (1.9, 8.0),
    "qwen2.5:7b": (4.5, 12.0),
    "qwen2.5-coder:3b": (1.9, 8.0),
    "qwen2.5-coder:7b": (4.5, 12.0),
    "llama3.2:1b": (0.8, 4.0),
    "llama3.2:3b": (2.0, 8.0),
    "qwen3:0.6b": (0.5, 4.0),
    "qwen3:1.7b": (1.2, 6.0),
    "qwen3:4b": (2.5, 8.0),
    "qwen3:8b": (5.0, 12.0),
    "qwen2-vl:2b": (1.8, 8.0),
    "qwen2.5-vl:3b": (2.4, 12.0),
    "qwen3-vl:2b": (2.0, 8.0),
    "qwen3-vl:4b": (3.3, 12.0),
    "qwen3-vl:8b": (5.8, 16.0),
    "qwen3.5:0.8b": (0.8, 6.0),
    "qwen3.5:4b": (3.0, 12.0),
    "qwen3.5:9b": (6.2, 16.0),
}

RECOMMENDED_MODELS = {
    "qwen2.5:0.5b",
    "qwen2.5:3b",
    "qwen2.5-coder:3b",
    "llama3.2:3b",
    "qwen2.5-vl:3b",
    "qwen3-vl:4b",
}


def native_mlx_available() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def native_mlx_vlm_available() -> bool:
    return native_mlx_available() and importlib.util.find_spec("mlx_vlm") is not None


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
    if alias.capability == "vision":
        if selected == "auto":
            selected = "mlx-vlm" if native_mlx_vlm_available() and alias.mlx else "hf-vlm"
        elif selected == "mlx":
            selected = "mlx-vlm"
        elif selected == "hf":
            selected = "hf-vlm"
        if selected not in {"mlx-vlm", "hf-vlm"}:
            raise ValueError(f"vision model alias {requested!r} requires an MLX-VLM or HF-VLM backend")
    else:
        if selected == "auto":
            selected = "mlx" if native_mlx_available() and alias.mlx else "hf"
        if selected not in {"mlx", "hf"}:
            raise ValueError(f"text model alias {requested!r} requires an MLX or HF backend")
    resolved = alias.mlx if selected.startswith("mlx") else alias.hf
    if not resolved:
        raise ValueError(f"model alias {requested!r} is not available for backend {selected!r}")
    return ModelResolution(requested=requested, model=resolved, backend=selected, alias=alias.name)


def select_backend_for_repo(model: str, backend: str = "auto") -> str:
    if backend != "auto":
        return backend
    normalized = model.lower()
    if looks_like_vision_model(normalized):
        return "mlx-vlm" if normalized.startswith("mlx-community/") else "hf-vlm"
    if normalized.startswith("mlx-community/") or "mlx" in normalized:
        return "mlx"
    return "hf"


def looks_like_vision_model(model: str) -> bool:
    normalized = model.lower()
    markers = ("-vl-", "-vl:", "vision", "llava", "pixtral", "florence", "moondream")
    return any(marker in normalized for marker in markers)


def alias_rows() -> list[dict]:
    return [MODEL_ALIASES[name].to_dict() for name in sorted(MODEL_ALIASES)]


def catalog_rows() -> list[dict[str, Any]]:
    rows = []
    for name in sorted(MODEL_ALIASES):
        alias = MODEL_ALIASES[name]
        backend = "mlx-vlm" if alias.capability == "vision" else "mlx"
        cached_path = cached_repo_path(alias.mlx) if alias.mlx else None
        download_size_gb, minimum_memory_gb = MODEL_RESOURCE_HINTS.get(
            name,
            (None, None),
        )
        capabilities = ["chat", "vision"] if alias.capability == "vision" else ["chat", "completion"]
        if alias.capability == "code":
            capabilities.append("code")
        rows.append(
            {
                "name": name,
                "display_name": _display_name(name),
                "repository": alias.mlx,
                "backend": backend,
                "capabilities": capabilities,
                "cached": cached_path is not None,
                "cached_path": str(cached_path) if cached_path is not None else None,
                "recommended": name in RECOMMENDED_MODELS,
                "tested": True,
                "download_size_gb": download_size_gb,
                "minimum_memory_gb": minimum_memory_gb,
                "support": "ready" if backend_available(backend) else "missing_runtime",
            }
        )
    return rows


def preflight_model(
    model: str,
    backend: str = "auto",
    *,
    allow_network: bool = False,
) -> dict[str, Any]:
    resolution = resolve_model(model, backend)
    result: dict[str, Any] = {
        "requested": resolution.requested,
        "model": resolution.model,
        "backend": resolution.backend,
        "alias": resolution.alias,
        "capabilities": (
            ["chat", "vision"]
            if resolution.backend.endswith("-vlm")
            else ["chat", "completion"]
        ),
        "runtime_available": backend_available(resolution.backend),
        "cached": False,
        "cached_path": None,
        "model_type": None,
        "supported": False,
        "reason": None,
    }
    path = Path(resolution.model).expanduser()
    config_path: Optional[Path] = None
    if path.exists():
        result["cached"] = True
        result["cached_path"] = str(path.resolve())
        config_path = path / "config.json" if path.is_dir() else path
    else:
        cached_path = cached_repo_path(resolution.model)
        if cached_path is not None:
            result["cached"] = True
            result["cached_path"] = str(cached_path)
            config_path = cached_path / "config.json"
        elif allow_network:
            try:
                from huggingface_hub import hf_hub_download

                config_path = Path(
                    hf_hub_download(repo_id=resolution.model, filename="config.json")
                )
            except Exception as exc:
                result["reason"] = f"could not read model config: {exc}"
                return result

    if not result["runtime_available"]:
        result["reason"] = f"{resolution.backend} runtime is not installed"
        return result
    if config_path is None or not config_path.is_file():
        result["reason"] = "model config is not cached; run a network preflight before downloading"
        return result

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = str(config.get("model_type") or "").strip().lower()
        if not model_type:
            raise ValueError("config.json does not define model_type")
        result["model_type"] = model_type
        _validate_mlx_architecture(config, resolution.backend)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["reason"] = str(exc)
        return result

    result["supported"] = True
    result["reason"] = "compatible"
    return result


def backend_available(backend: str) -> bool:
    if backend == "mlx":
        return native_mlx_available()
    if backend == "mlx-vlm":
        return native_mlx_vlm_available()
    if backend == "hf":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    return False


def cached_repo_path(model: Optional[str]) -> Optional[Path]:
    if not model:
        return None
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model, "config.json")
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(config, str):
        return None
    return Path(config).resolve().parent


def _validate_mlx_architecture(config: dict[str, Any], backend: str) -> None:
    model_type = str(config["model_type"]).lower()
    if backend == "mlx":
        from mlx_lm.utils import MODEL_REMAPPING

        mapped = MODEL_REMAPPING.get(model_type, model_type)
        module = importlib.import_module(f"mlx_lm.models.{mapped}")
        if not hasattr(module, "Model") or not hasattr(module, "ModelArgs"):
            raise ValueError(f"Model type {mapped} is missing MLX model classes")
        return
    if backend == "mlx-vlm":
        from mlx_vlm.utils import get_model_and_args

        get_model_and_args(config)
        return
    raise ValueError(f"desktop runtime does not support backend {backend!r}")


def _display_name(alias: str) -> str:
    family, _, size = alias.partition(":")
    family = family.replace("qwen", "Qwen").replace("llama", "Llama ")
    return f"{family} {size.upper()}".strip()


def model_targets(model: str) -> set[str]:
    requested = model.strip()
    alias = MODEL_ALIASES.get(requested.lower())
    if alias is None:
        return {str(Path(requested).expanduser()) if Path(requested).expanduser().exists() else requested}
    return {target for target in (alias.mlx, alias.hf) if target}
