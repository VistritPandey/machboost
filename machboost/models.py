from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import platform
import shutil
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


@dataclass(frozen=True)
class DFlashAlias:
    name: str
    target: str
    draft: str
    download_size_gb: float
    minimum_memory_gb: float


@dataclass(frozen=True)
class OllamaMLXAlias:
    name: str
    model: str
    display_name: str
    source_repository: str
    capabilities: tuple[str, ...]
    download_size_gb: float
    minimum_memory_gb: float
    context_length: int


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
        ModelAlias(
            "muse-glimmer:30b",
            "mlx-community/Muse-Glimmer-30B-4bit",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-4bit",
            "mlx-community/Muse-Glimmer-30B-4bit",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-5bit",
            "mlx-community/Muse-Glimmer-30B-5bit",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-6bit",
            "mlx-community/Muse-Glimmer-30B-6bit",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-8bit",
            "mlx-community/Muse-Glimmer-30B-8bit",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-bf16",
            "mlx-community/Muse-Glimmer-30B-bf16",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-mxfp4",
            "mlx-community/Muse-Glimmer-30B-mxfp4",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-mxfp8",
            "mlx-community/Muse-Glimmer-30B-mxfp8",
            "meta-models/Muse-Glimmer-30B",
            "vision",
        ),
        ModelAlias(
            "muse-glimmer:30b-nvfp4",
            "mlx-community/Muse-Glimmer-30B-nvfp4",
            "meta-models/Muse-Glimmer-30B",
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
    "muse-glimmer:30b": (6.0, 32.0),
    "muse-glimmer:30b-4bit": (6.0, 32.0),
    "muse-glimmer:30b-5bit": (7.0, 32.0),
    "muse-glimmer:30b-6bit": (8.0, 36.0),
    "muse-glimmer:30b-8bit": (10.0, 48.0),
    "muse-glimmer:30b-bf16": (30.0, 64.0),
    "muse-glimmer:30b-mxfp4": (7.0, 32.0),
    "muse-glimmer:30b-mxfp8": (10.0, 48.0),
    "muse-glimmer:30b-nvfp4": (9.0, 40.0),
}

RECOMMENDED_MODELS = {
    "qwen2.5:0.5b",
    "qwen2.5:3b",
    "qwen2.5-coder:3b",
    "llama3.2:3b",
    "qwen2.5-vl:3b",
    "qwen3-vl:4b",
    "muse-glimmer:30b",
}

DFLASH_TARGETS = {
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3.5:4b": "mlx-community/Qwen3.5-4B-MLX-bf16",
    "qwen3.5:9b": "mlx-community/Qwen3.5-9B-MLX-bf16",
}

DFLASH_ALIASES = {
    alias.name: alias
    for alias in (
        DFlashAlias(
            "qwen3.5:4b-dflash",
            "mlx-community/Qwen3.5-4B-MLX-bf16",
            "z-lab/Qwen3.5-4B-DFlash",
            10.3,
            16.0,
        ),
        DFlashAlias(
            "qwen3.5:9b-dflash",
            "mlx-community/Qwen3.5-9B-MLX-bf16",
            "z-lab/Qwen3.5-9B-DFlash",
            21.1,
            32.0,
        ),
    )
}

MUSE_GLIMMER = OllamaMLXAlias(
    name="muse-glimmer:30b-mlx",
    model="muse-glimmer:30b-mlx",
    display_name="Muse Glimmer 30B (Ollama MLX)",
    source_repository="meta-models/Muse-Glimmer-30B",
    capabilities=("chat", "completion", "vision", "reasoning", "tools"),
    download_size_gb=21.0,
    minimum_memory_gb=32.0,
    context_length=131_072,
)

OLLAMA_MLX_ALIASES = {
    MUSE_GLIMMER.name: MUSE_GLIMMER,
}

MUSE_GLIMMER_CAPABILITIES = ("chat", "completion", "vision", "reasoning", "tools")
MUSE_GLIMMER_CONTEXT_LENGTH = 131_072


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
    ollama_mlx_alias = OLLAMA_MLX_ALIASES.get(requested.lower())
    if ollama_mlx_alias is not None:
        if backend not in {"auto", "ollama-mlx"}:
            raise ValueError(
                f"model alias {requested!r} requires Ollama's MLX backend, not {backend!r}"
            )
        return ModelResolution(
            requested=requested,
            model=ollama_mlx_alias.model,
            backend="ollama-mlx",
            alias=ollama_mlx_alias.name,
        )
    dflash_alias = DFLASH_ALIASES.get(requested.lower())
    if dflash_alias is not None:
        if backend not in {"auto", "dflash"}:
            raise ValueError(
                f"model alias {requested!r} requires the DFlash backend, not {backend!r}"
            )
        return ModelResolution(
            requested=requested,
            model=dflash_alias.target,
            backend="dflash",
            alias=dflash_alias.name,
        )
    alias = MODEL_ALIASES.get(requested.lower())
    if alias is None:
        selected = select_backend_for_repo(requested, backend)
        return ModelResolution(requested=requested, model=str(path) if path.exists() else requested, backend=selected)

    selected = backend
    if selected == "dflash":
        dflash_target = DFLASH_TARGETS.get(alias.name)
        if dflash_target is None:
            raise ValueError(
                f"model alias {requested!r} is not supported by the DFlash backend; "
                f"supported aliases: {', '.join(sorted(DFLASH_TARGETS))}"
            )
        return ModelResolution(
            requested=requested,
            model=dflash_target,
            backend="dflash",
            alias=alias.name,
        )
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
    markers = (
        "-vl-",
        "-vl:",
        "vision",
        "llava",
        "pixtral",
        "florence",
        "moondream",
        "muse-glimmer",
    )
    return any(marker in normalized for marker in markers)


def alias_rows() -> list[dict]:
    rows = [MODEL_ALIASES[name].to_dict() for name in sorted(MODEL_ALIASES)]
    rows.extend(
        {
            "name": alias.name,
            "mlx": alias.target,
            "hf": None,
            "capability": "chat",
            "backend": "dflash",
            "draft": alias.draft,
        }
        for alias in DFLASH_ALIASES.values()
    )
    rows.extend(
        {
            "name": alias.name,
            "mlx": alias.model,
            "hf": alias.source_repository,
            "capability": "vision",
            "backend": "ollama-mlx",
            "capabilities": list(alias.capabilities),
        }
        for alias in {item.name: item for item in OLLAMA_MLX_ALIASES.values()}.values()
    )
    return sorted(rows, key=lambda row: str(row["name"]))


def catalog_rows(
    *,
    include_cached_repositories: bool = True,
    cache_dirs: Optional[list[Path]] = None,
) -> list[dict[str, Any]]:
    rows = []
    catalog_repositories = {
        alias.mlx for alias in MODEL_ALIASES.values() if alias.mlx is not None
    }
    catalog_repositories.update(
        repository
        for alias in DFLASH_ALIASES.values()
        for repository in (alias.target, alias.draft)
    )
    for name in sorted(MODEL_ALIASES):
        alias = MODEL_ALIASES[name]
        backend = "mlx-vlm" if alias.capability == "vision" else "mlx"
        cached_path = cached_repo_path(alias.mlx) if alias.mlx else None
        download_size_gb, minimum_memory_gb = MODEL_RESOURCE_HINTS.get(
            name,
            (None, None),
        )
        capabilities = (
            list(MUSE_GLIMMER_CAPABILITIES)
            if name.startswith("muse-glimmer:")
            else ["chat", "vision"]
            if alias.capability == "vision"
            else ["chat", "completion"]
        )
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
                "disk_size_gb": _directory_size_gb(cached_path),
                "minimum_memory_gb": minimum_memory_gb,
                "context_length": (
                    MUSE_GLIMMER_CONTEXT_LENGTH
                    if name.startswith("muse-glimmer:")
                    else None
                ),
                "support": "ready" if backend_available(backend) else "missing_runtime",
                "support_reason": None,
            }
        )
    for alias in {item.name: item for item in OLLAMA_MLX_ALIASES.values()}.values():
        manifest_path = ollama_model_manifest(alias.model)
        runtime_available = backend_available("ollama-mlx")
        rows.append(
            {
                "name": alias.name,
                "display_name": alias.display_name,
                "repository": alias.model,
                "source_repository": alias.source_repository,
                "backend": "ollama-mlx",
                "capabilities": list(alias.capabilities),
                "cached": manifest_path is not None,
                "cached_path": str(manifest_path) if manifest_path is not None else None,
                "recommended": True,
                "tested": True,
                "experimental": False,
                "validation_status": "local_smoke_passed",
                "download_size_gb": alias.download_size_gb,
                "disk_size_gb": _ollama_manifest_size_gb(manifest_path),
                "minimum_memory_gb": alias.minimum_memory_gb,
                "context_length": alias.context_length,
                "support": "ready" if runtime_available else "missing_runtime",
                "support_reason": (
                    None
                    if runtime_available
                    else "Muse Glimmer requires current Ollama on Apple Silicon"
                ),
            }
        )
    for name in sorted(DFLASH_ALIASES):
        alias = DFLASH_ALIASES[name]
        validation_status = (
            "passed_bounded_suite"
            if name == "qwen3.5:4b-dflash"
            else "divergence_observed"
        )
        target_path = cached_repo_path(alias.target)
        draft_path = cached_repo_path(alias.draft)
        cached = target_path is not None and draft_path is not None
        rows.append(
            {
                "name": name,
                "display_name": _display_name(name),
                "repository": alias.target,
                "draft_repository": alias.draft,
                "backend": "dflash",
                "capabilities": ["chat", "completion"],
                "cached": cached,
                "cached_path": str(target_path) if cached else None,
                "recommended": name == "qwen3.5:4b-dflash",
                "tested": True,
                "experimental": name != "qwen3.5:4b-dflash",
                "validation_status": validation_status,
                "download_size_gb": alias.download_size_gb,
                "disk_size_gb": sum(
                    _directory_size_gb(path) or 0.0
                    for path in (target_path, draft_path)
                ),
                "minimum_memory_gb": alias.minimum_memory_gb,
                "support": "ready" if backend_available("dflash") else "missing_runtime",
                "support_reason": None,
            }
        )
    if include_cached_repositories:
        rows.extend(
            _cached_repository_rows(
                cache_dirs=cache_dirs,
                excluded_repositories=catalog_repositories,
            )
        )
    return sorted(rows, key=lambda row: str(row["name"]).lower())


def default_hf_cache_dirs() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if value := os.environ.get(variable):
            candidates.append(Path(value).expanduser())
    if value := os.environ.get("HF_HOME"):
        candidates.append(Path(value).expanduser() / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def _cached_repository_rows(
    *,
    cache_dirs: Optional[list[Path]],
    excluded_repositories: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set(excluded_repositories)
    for cache_dir in cache_dirs or default_hf_cache_dirs():
        if not cache_dir.is_dir():
            continue
        for repository_dir in sorted(cache_dir.glob("models--*")):
            repository = _repository_id_from_cache_name(repository_dir.name)
            if not repository or repository in seen:
                continue
            seen.add(repository)
            backend = select_backend_for_repo(repository)
            if not backend.startswith("mlx"):
                continue
            snapshot = _latest_snapshot(repository_dir)
            if snapshot is None:
                continue
            preflight = _preflight_cached_snapshot(repository, backend, snapshot)
            capabilities = (
                ["chat", "vision"]
                if backend.endswith("-vlm")
                else ["chat", "completion"]
            )
            rows.append(
                {
                    "name": repository,
                    "display_name": repository.rsplit("/", 1)[-1],
                    "repository": repository,
                    "backend": backend,
                    "capabilities": capabilities,
                    "cached": True,
                    "cached_path": str(snapshot),
                    "recommended": False,
                    "tested": False,
                    "download_size_gb": None,
                    "disk_size_gb": _directory_size_gb(snapshot),
                    "minimum_memory_gb": None,
                    "support": "ready" if preflight["supported"] else "unsupported",
                    "support_reason": preflight["reason"],
                }
            )
    return rows


def _repository_id_from_cache_name(name: str) -> Optional[str]:
    if not name.startswith("models--"):
        return None
    pieces = name.removeprefix("models--").split("--")
    if len(pieces) < 2 or any(not piece for piece in pieces):
        return None
    return "/".join(pieces)


def _latest_snapshot(repository_dir: Path) -> Optional[Path]:
    snapshots = repository_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [
        path
        for path in snapshots.iterdir()
        if path.is_dir()
        and (path / "config.json").is_file()
        and _snapshot_has_weights(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def _preflight_cached_snapshot(
    repository: str,
    backend: str,
    snapshot: Path,
) -> dict[str, Any]:
    if not backend_available(backend):
        return {
            "supported": False,
            "reason": f"{backend} runtime is not installed",
        }
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        model_type = str(config.get("model_type") or "").strip().lower()
        if not model_type:
            raise ValueError("config.json does not define model_type")
        _validate_cached_mlx_architecture(config, backend)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"supported": False, "reason": str(exc)}
    return {
        "supported": True,
        "reason": f"compatible {backend} architecture ({model_type})",
    }


def _validate_cached_mlx_architecture(config: dict[str, Any], backend: str) -> None:
    package = "mlx_vlm" if backend == "mlx-vlm" else "mlx_lm"
    spec = importlib.util.find_spec(package)
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        raise ValueError(f"{package} runtime is not installed")

    package_root = Path(locations[0])
    model_type = str(config["model_type"]).lower()
    model_type = _model_remapping(package_root).get(model_type, model_type)
    if backend == "mlx-vlm" and config.get("dflash_config") is not None:
        model_type += "_dflash"

    candidates = [package_root / "models" / f"{model_type}.py"]
    candidates.append(package_root / "models" / model_type / "__init__.py")
    if backend == "mlx-vlm":
        candidates.append(
            package_root / "speculative" / "drafters" / f"{model_type}.py"
        )
    if not any(candidate.is_file() for candidate in candidates):
        raise ValueError(f"Model type {model_type} not supported")


def _model_remapping(package_root: Path) -> dict[str, str]:
    utils_path = package_root / "utils.py"
    try:
        tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return {}
        if isinstance(value, dict):
            return {
                str(key).lower(): str(mapped).lower()
                for key, mapped in value.items()
            }
    return {}


def _directory_size_gb(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.is_dir():
        return None
    total = 0
    seen_files: set[tuple[int, int]] = set()
    try:
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            stat = item.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            total += stat.st_size
    except OSError:
        return None
    return total / 1_000_000_000


def _snapshot_has_weights(path: Path) -> bool:
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        try:
            referenced = {
                str(filename)
                for index in indexes
                for filename in json.loads(index.read_text(encoding="utf-8"))
                .get("weight_map", {})
                .values()
            }
        except (OSError, TypeError, json.JSONDecodeError):
            return False
        return bool(referenced) and all((path / filename).is_file() for filename in referenced)
    return any(path.glob("*.safetensors"))


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
        "capabilities": _resolution_capabilities(resolution),
        "runtime_available": backend_available(resolution.backend),
        "cached": False,
        "cached_path": None,
        "model_type": None,
        "supported": False,
        "reason": None,
    }
    if resolution.backend == "ollama-mlx":
        manifest_path = ollama_model_manifest(resolution.model)
        result.update(
            {
                "cached": manifest_path is not None,
                "cached_path": str(manifest_path) if manifest_path is not None else None,
                "model_type": "muse_glimmer",
                "supported": result["runtime_available"],
                "reason": (
                    "compatible with Ollama's Apple Silicon MLX engine"
                    if result["runtime_available"]
                    else "Muse Glimmer requires current Ollama on Apple Silicon"
                ),
            }
        )
        return result
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
        _validate_mlx_architecture(
            config,
            resolution.backend,
            model_ref=resolution.model,
        )
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["reason"] = str(exc)
        return result

    result["supported"] = True
    result["reason"] = "compatible"
    return result


def backend_available(backend: str) -> bool:
    if backend == "ollama-mlx":
        return (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and ollama_executable() is not None
        )
    if backend == "mlx":
        return native_mlx_available()
    if backend == "mlx-vlm":
        return native_mlx_vlm_available()
    if backend == "dflash":
        return native_mlx_available() and importlib.util.find_spec("dflash_mlx") is not None
    if backend == "hf":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    return False


def ollama_executable() -> Optional[str]:
    """Locate Ollama from shells, app bundles, and common macOS installs."""
    candidates = [
        os.environ.get("OLLAMA_BINARY"),
        shutil.which("ollama"),
        "/Applications/Ollama.app/Contents/Resources/ollama",
        str(Path.home() / "Applications/Ollama.app/Contents/Resources/ollama"),
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def ollama_model_manifest(model: str) -> Optional[Path]:
    normalized = model.strip().lower()
    name, separator, tag = normalized.partition(":")
    if not separator:
        tag = "latest"
    if not name or "/" in name or not tag or "/" in tag:
        return None
    root = Path(os.environ.get("OLLAMA_MODELS", "~/.ollama/models")).expanduser()
    path = root / "manifests" / "registry.ollama.ai" / "library" / name / tag
    return path.resolve() if path.is_file() else None


def _ollama_manifest_size_gb(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        layers = manifest.get("layers") or []
        size = sum(int(layer.get("size") or 0) for layer in layers if isinstance(layer, dict))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return size / 1_000_000_000


def _resolution_capabilities(resolution: ModelResolution) -> list[str]:
    alias = OLLAMA_MLX_ALIASES.get((resolution.alias or resolution.requested).lower())
    if alias is not None:
        return list(alias.capabilities)
    if (resolution.alias or resolution.requested).lower().startswith("muse-glimmer:"):
        return list(MUSE_GLIMMER_CAPABILITIES)
    if resolution.backend.endswith("-vlm"):
        return ["chat", "vision"]
    return ["chat", "completion"]


def cached_repo_path(model: Optional[str]) -> Optional[Path]:
    if not model:
        return None
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=model, local_files_only=True)
    except (ImportError, OSError, ValueError):
        return None
    snapshot_path = Path(snapshot)
    if not (snapshot_path / "config.json").is_file() or not _snapshot_has_weights(
        snapshot_path
    ):
        return None
    return snapshot_path.resolve()


def _validate_mlx_architecture(
    config: dict[str, Any],
    backend: str,
    *,
    model_ref: Optional[str] = None,
) -> None:
    model_type = str(config["model_type"]).lower()
    if backend in {"mlx", "dflash"}:
        from mlx_lm.utils import MODEL_REMAPPING

        mapped = MODEL_REMAPPING.get(model_type, model_type)
        module = importlib.import_module(f"mlx_lm.models.{mapped}")
        if not hasattr(module, "Model") or not hasattr(module, "ModelArgs"):
            raise ValueError(f"Model type {mapped} is missing MLX model classes")
        if backend == "dflash":
            from dflash_mlx.runtime.registry import resolve_model_support_spec

            model_name = str(model_ref or config.get("name_or_path") or "")
            if resolve_model_support_spec(model_name) is None:
                raise ValueError(
                    "model is not in the installed DFlash target registry; pass a supported Qwen or Gemma target"
                )
        return
    if backend == "mlx-vlm":
        from mlx_vlm.utils import get_model_and_args

        get_model_and_args(config)
        return
    raise ValueError(f"desktop runtime does not support backend {backend!r}")


def _display_name(alias: str) -> str:
    family, _, size = alias.partition(":")
    if family == "muse-glimmer":
        return f"Muse Glimmer {size.upper()}".strip()
    family = family.replace("qwen", "Qwen").replace("llama", "Llama ")
    return f"{family} {size.upper()}".strip()


def model_targets(model: str) -> set[str]:
    requested = model.strip()
    ollama_mlx_alias = OLLAMA_MLX_ALIASES.get(requested.lower())
    if ollama_mlx_alias is not None:
        return {ollama_mlx_alias.model}
    dflash_alias = DFLASH_ALIASES.get(requested.lower())
    if dflash_alias is not None:
        return {dflash_alias.target}
    alias = MODEL_ALIASES.get(requested.lower())
    if alias is None:
        return {str(Path(requested).expanduser()) if Path(requested).expanduser().exists() else requested}
    return {target for target in (alias.mlx, alias.hf) if target}


def model_repositories(model: str, backend: str = "auto") -> tuple[str, ...]:
    """Return every repository required to load a model selection."""
    requested = model.strip()
    ollama_mlx_alias = OLLAMA_MLX_ALIASES.get(requested.lower())
    if ollama_mlx_alias is not None:
        if backend not in {"auto", "ollama-mlx"}:
            resolve_model(requested, backend)
        return (ollama_mlx_alias.model,)
    dflash_alias = DFLASH_ALIASES.get(requested.lower())
    if dflash_alias is not None:
        if backend not in {"auto", "dflash"}:
            resolve_model(requested, backend)
        return (dflash_alias.target, dflash_alias.draft)
    return (resolve_model(requested, backend).model,)
