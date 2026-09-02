from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

from . import __version__, machboost
from .accelerator import Accelerator, read_context_paths, resolve_context
from .adapters.ollama import OllamaHTTPAdapter, OllamaHTTPError
from .client import (
    MachBoostAPIError,
    MachBoostClient,
    ensure_server,
    machboost_app_api_token,
)
from .claude_desktop import (
    ClaudeDesktopProfileManager,
    save_model_mappings,
)
from .connections import ConnectionStore, normalize_endpoint
from .context_bench import benchmark_context_acceleration, context_fingerprint
from .latency import benchmark_chat_latency
from .models import alias_rows, backend_available, catalog_rows, resolve_model
from .routing import HostTarget, MachBoostHostPool
from .relay import start_claude_gateway_relay, stop_claude_gateway_relay
from .server import (
    DEFAULT_HOST,
    DEFAULT_MAX_QUEUE,
    DEFAULT_PORT,
    DEFAULT_QUEUE_TIMEOUT,
    DEFAULT_REPLICAS,
    MAX_REPLICAS,
    serve as serve_runtime,
)
from .team import TeamStore
from .video import TemporalVideoSampler, VideoSelection
from .vision_auto import VISION_TOKEN_REQUEST_MODES, load_vision_calibration

DEFAULT_CHAT_SYSTEM = "Answer directly and concisely. Do not reveal hidden reasoning."
CHAT_HELP = """Commands:
  /? or /help       show this help
  /status           show model, host, route, and active context
  /stats on|off     toggle per-response performance statistics
  /route            show local/API routing for this session
  /clear            reset chat history
  /image PATH       attach an image
  /video PATH       attach sampled video frames
  /images           list attached images
  /clear-images     remove attached images
  /unload           unload this model and exit
  /bye or /exit     exit and keep the model warm until its idle timeout

Ctrl-C stops the current reply. Ctrl-D unloads the model and exits."""


@dataclass(frozen=True)
class PackageStatus:
    available: bool
    version: Optional[str] = None


@dataclass(frozen=True)
class CachedModel:
    name: str
    backend: str
    source: str
    path: str
    runnable: bool
    reason: str


class ScriptedVerifierService:
    def __init__(self, prompt: str, completion: str) -> None:
        self.prompt_len = len(self.encode(prompt))
        self.completion = tuple(self.encode(completion))
        self.forward_calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def next_token(self, prefix_tokens: Sequence[int]) -> Optional[int]:
        self.forward_calls += 1
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        if offset >= len(self.completion):
            return None
        return self.completion[offset]

    def verify(self, prefix_tokens: Sequence[int], candidate_tokens: Sequence[int]):
        self.forward_calls += 1
        offset = max(0, len(prefix_tokens) - self.prompt_len)
        accepted = 0
        for token in candidate_tokens:
            target_pos = offset + accepted
            if target_pos >= len(self.completion) or token != self.completion[target_pos]:
                break
            accepted += 1
        if accepted == len(candidate_tokens):
            return accepted, None
        residual_pos = offset + accepted
        residual = self.completion[residual_pos] if residual_pos < len(self.completion) else None
        return accepted, residual


def package_status(
    module_name: str,
    version_attr: str = "__version__",
    distribution_name: Optional[str] = None,
) -> PackageStatus:
    if importlib.util.find_spec(module_name) is None:
        return PackageStatus(False)
    try:
        version = metadata.version(distribution_name or module_name.replace("_", "-"))
    except Exception:
        version = None
    return PackageStatus(True, str(version) if version else None)


def doctor_data() -> dict:
    return {
        "schema_version": "machboost.doctor.v1",
        "machboost_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "optional_packages": {
            "torch": asdict(package_status("torch")),
            "transformers": asdict(package_status("transformers")),
            "mlx": asdict(package_status("mlx")),
            "mlx_lm": asdict(package_status("mlx_lm", distribution_name="mlx-lm")),
            "mlx_vlm": asdict(package_status("mlx_vlm", distribution_name="mlx-vlm")),
            "dflash_mlx": asdict(
                package_status("dflash_mlx", distribution_name="dflash-mlx")
            ),
        },
    }


def self_test_data() -> dict:
    prompt = "Complete: "
    completion = "alpha beta gamma"
    service = ScriptedVerifierService(prompt, completion)
    prompt_tokens = service.encode(prompt)
    corpus_tokens = prompt_tokens + service.encode(completion)
    boosted = machboost(service, corpus_tokens=corpus_tokens, ngram=2, max_draft_tokens=8)
    generated, stats = boosted.generate(prompt_tokens, max_tokens=len(completion))
    text = service.decode(generated)
    exact = text == completion
    return {
        "schema_version": "machboost.self_test.v1",
        "ok": exact and stats.accepted_draft_tokens > 0,
        "output_match": exact,
        "output": text,
        "accepted_draft_tokens": stats.accepted_draft_tokens,
        "target_calls": stats.target_calls,
        "baseline_target_calls": stats.baseline_target_calls,
        "estimated_speedup": stats.estimated_speedup,
    }


def model_list_data(
    *,
    backend: str = "all",
    cache_dirs: Optional[Sequence[str]] = None,
    include_unsupported: bool = False,
) -> dict:
    cache_paths = [Path(item).expanduser() for item in cache_dirs] if cache_dirs else default_hf_cache_dirs()
    models = discover_cached_models(cache_paths, backend=backend, include_unsupported=include_unsupported)
    for row in catalog_rows(include_cached_repositories=False):
        if row["backend"] != "ollama-mlx" or not row["cached"]:
            continue
        if backend not in {"all", "ollama-mlx"}:
            continue
        models.append(
            CachedModel(
                name=str(row["name"]),
                backend="ollama-mlx",
                source="ollama_cache",
                path=str(row["cached_path"] or ""),
                runnable=row["support"] == "ready",
                reason=str(row.get("support_reason") or "official Ollama MLX model"),
            )
        )
    models.sort(key=lambda model: (not model.runnable, model.backend, model.name.lower()))
    return {
        "schema_version": "machboost.model_list.v1",
        "machboost_version": __version__,
        "backends": native_backend_status(),
        "cache_dirs": [str(path) for path in cache_paths],
        "models": [asdict(model) for model in models],
        "hidden_unsupported_count": count_hidden_unsupported(cache_paths, backend=backend)
        if not include_unsupported
        else 0,
        "aliases": alias_rows(),
        "examples": [
            {
                "backend": "mlx",
                "model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                "command": "machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx",
            },
            {
                "backend": "hf",
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "command": "machboost run Qwen/Qwen2.5-3B-Instruct --backend hf",
            },
            {
                "backend": "mlx-vlm",
                "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                "command": "machboost run qwen2.5-vl:3b --image ./image.png",
            },
            {
                "backend": "ollama-mlx",
                "model": "muse-glimmer:30b-mlx",
                "command": "machboost run muse-glimmer:30b-mlx --think high --show-stats",
            },
        ],
    }


def default_hf_cache_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return unique_paths(candidates)


def unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def native_backend_status() -> dict:
    torch = package_status("torch")
    transformers = package_status("transformers")
    mlx = package_status("mlx")
    mlx_lm = package_status("mlx_lm", distribution_name="mlx-lm")
    mlx_vlm = package_status("mlx_vlm", distribution_name="mlx-vlm")
    dflash_mlx = package_status("dflash_mlx", distribution_name="dflash-mlx")
    return {
        "hf": {
            "available": torch.available and transformers.available,
            "packages": {
                "torch": asdict(torch),
                "transformers": asdict(transformers),
            },
        },
        "mlx": {
            "available": mlx.available and mlx_lm.available,
            "packages": {
                "mlx": asdict(mlx),
                "mlx_lm": asdict(mlx_lm),
            },
        },
        "mlx-vlm": {
            "available": mlx.available and mlx_vlm.available,
            "packages": {
                "mlx": asdict(mlx),
                "mlx_vlm": asdict(mlx_vlm),
            },
        },
        "dflash": {
            "available": mlx.available and mlx_lm.available and dflash_mlx.available,
            "packages": {
                "mlx": asdict(mlx),
                "mlx_lm": asdict(mlx_lm),
                "dflash_mlx": asdict(dflash_mlx),
            },
        },
        "ollama-mlx": {
            "available": backend_available("ollama-mlx"),
            "packages": {},
        },
    }


def discover_cached_models(
    cache_dirs: Sequence[Path],
    *,
    backend: str = "all",
    include_unsupported: bool = False,
) -> list[CachedModel]:
    models: dict[tuple[str, str], CachedModel] = {}
    for cache_dir in cache_dirs:
        if not cache_dir.exists() or not cache_dir.is_dir():
            continue
        for item in sorted(cache_dir.iterdir()):
            if not item.is_dir() or not item.name.startswith("models--"):
                continue
            model_id = hf_cache_dir_to_model_id(item.name)
            if not model_id:
                continue
            model_backend, runnable, reason = classify_cached_model(model_id, item)
            if backend != "all" and model_backend != backend:
                continue
            if not runnable and not include_unsupported:
                continue
            key = (model_backend, model_id)
            models.setdefault(
                key,
                CachedModel(
                    name=model_id,
                    backend=model_backend,
                    source="huggingface_cache",
                    path=str(item),
                    runnable=runnable,
                    reason=reason,
                ),
            )
    return sorted(models.values(), key=lambda model: (not model.runnable, model.backend, model.name.lower()))


def count_hidden_unsupported(cache_dirs: Sequence[Path], *, backend: str = "all") -> int:
    return len(discover_cached_models(cache_dirs, backend=backend, include_unsupported=True)) - len(
        discover_cached_models(cache_dirs, backend=backend, include_unsupported=False)
    )


def hf_cache_dir_to_model_id(name: str) -> str:
    encoded = name.removeprefix("models--")
    if not encoded or encoded == name:
        return ""
    return encoded.replace("--", "/")


def classify_cached_model(model_id: str, model_dir: Path) -> tuple[str, bool, str]:
    model_backend = select_native_backend(model_id, "auto")
    normalized = model_id.lower()
    if "gguf" in normalized:
        return model_backend, False, "GGUF repo; native llama.cpp/GGUF runner is not supported yet"
    if model_backend in {"mlx", "mlx-vlm"}:
        kind = "MLX vision model cache" if model_backend == "mlx-vlm" else "MLX model cache"
        return model_backend, has_snapshot(model_dir), kind if has_snapshot(model_dir) else "no snapshot"

    if model_backend == "hf-vlm":
        return model_backend, False, "HF vision cache detected; resident HF-VLM adapter is not available yet"

    config = read_cached_config(model_dir)
    if config is None:
        return model_backend, False, "missing config.json"
    architectures = [str(item) for item in config.get("architectures") or ()]
    model_type = str(config.get("model_type") or "")
    if is_hf_causal_lm_config(architectures, model_type):
        return model_backend, True, "Hugging Face causal LM cache"
    return model_backend, False, "not a causal LM supported by AutoModelForCausalLM"


def has_snapshot(model_dir: Path) -> bool:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists() or not snapshots.is_dir():
        return False
    return any(snapshots.iterdir())


def read_cached_config(model_dir: Path) -> Optional[dict]:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists() or not snapshots.is_dir():
        return None
    snapshot_dirs = sorted(
        [item for item in snapshots.iterdir() if item.is_dir()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshot_dirs:
        config_path = snapshot / "config.json"
        if not config_path.exists():
            continue
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def is_hf_causal_lm_config(architectures: Sequence[str], model_type: str) -> bool:
    if any(architecture.endswith("ForCausalLM") or "LMHeadModel" in architecture for architecture in architectures):
        return True
    return model_type in {
        "bloom",
        "falcon",
        "gemma",
        "gemma2",
        "gpt2",
        "gpt_bigcode",
        "gpt_neox",
        "gptj",
        "llama",
        "mistral",
        "mixtral",
        "olmo",
        "phi",
        "phi3",
        "qwen2",
        "qwen3",
        "stablelm",
        "starcoder2",
    }


def print_human_doctor(data: dict) -> None:
    print(f"machboost {data['machboost_version']}")
    print(f"python {data['python']['version']} ({data['python']['executable']})")
    platform_data = data["platform"]
    print(f"platform {platform_data['system']} {platform_data['release']} {platform_data['machine']}")
    print("optional packages:")
    for name, status in data["optional_packages"].items():
        version = f" {status['version']}" if status.get("version") else ""
        state = "found" if status["available"] else "missing"
        print(f"  {name}: {state}{version}")


def print_human_self_test(data: dict) -> None:
    state = "ok" if data["ok"] else "failed"
    print(f"self-test: {state}")
    print(f"output match: {data['output_match']}")
    print(f"accepted draft tokens: {data['accepted_draft_tokens']}")
    print(f"target calls: {data['target_calls']} / baseline {data['baseline_target_calls']}")
    print(f"synthetic target-call reduction: {data['estimated_speedup']:.2f}x")
    print("note: this is an in-memory correctness check, not a wall-clock benchmark")


def print_human_model_list(data: dict) -> None:
    print(f"machboost {data['machboost_version']}")
    print("native backends:")
    for backend, status in data["backends"].items():
        state = "ready" if status["available"] else "missing optional packages"
        print(f"  {backend}: {state}")

    models = data["models"]
    runnable = [model for model in models if model["runnable"]]
    unsupported = [model for model in models if not model["runnable"]]
    print("cached runnable models:")
    if runnable:
        for model in runnable:
            print(f"  {model['backend']:<3} {model['name']}")
            print(f"      run: machboost run {model['name']} --backend {model['backend']}")
    else:
        print("  none found in the inspected Hugging Face caches")

    if unsupported:
        print("unsupported cached repos:")
        for model in unsupported:
            print(f"  {model['backend']:<8} {model['name']} ({model['reason']})")
    elif data.get("hidden_unsupported_count", 0):
        print(f"unsupported cached repos: {data['hidden_unsupported_count']} hidden; use --all to show them")

    print("short aliases:")
    for alias in data.get("aliases", ()):
        target = alias.get("mlx") or alias.get("hf")
        print(f"  {alias['name']:<22} {target}")

    print("remote examples:")
    for example in data["examples"]:
        print(f"  {example['command']}")


def run_connect(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        endpoint = normalize_endpoint(args.endpoint)
        name = args.name or (urlparse(endpoint).hostname or "host").replace(".", "-")
        token = os.environ.get("MACHBOOST_API_TOKEN") or getpass.getpass(
            f"API key for {endpoint}: "
        )
        if not token.strip():
            raise ValueError("an API key is required for a remote MachBoost host")
        client = MachBoostClient(endpoint, api_token=token, timeout=args.timeout)
        health = client.health()
        client.catalog()
        store = ConnectionStore()
        profile = store.save(name, endpoint, api_token=token)
        store.select("auto")
    except (MachBoostAPIError, RuntimeError, ValueError) as exc:
        print(f"machboost connect error: {exc}", file=error_stream)
        return 2
    print(f"connected to {profile.name} ({profile.endpoint})", file=output_stream)
    print(f"server {health.get('version') or health.get('status') or 'ready'}", file=output_stream)
    print("automatic routing enabled across this Mac and saved devices", file=output_stream)
    return 0


def run_connections(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        store = ConnectionStore()
        mode = store.mode()
        active = store.active()
        profiles = store.list()
    except ValueError as exc:
        print(f"machboost connections error: {exc}", file=error_stream)
        return 2
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "machboost.connections.v2",
                    "active": (
                        "auto" if mode == "auto" else active.id if active else "local"
                    ),
                    "connections": [asdict(profile) for profile in profiles],
                    **(
                        {"routing": _connection_route_status(store, args)}
                        if args.probe
                        else {}
                    ),
                },
                indent=2,
            ),
            file=output_stream,
        )
        return 0
    if mode == "auto":
        print("routing: automatic (lowest expected completion time)", file=output_stream)
    print("ACTIVE  NAME                 ENDPOINT", file=output_stream)
    print(
        f"{'*' if mode == 'local' else ' ':<7} "
        f"local                http://{DEFAULT_HOST}:{DEFAULT_PORT}",
        file=output_stream,
    )
    for profile in profiles:
        marker = "*" if mode == "fixed" and active and active.id == profile.id else " "
        print(f"{marker:<7} {profile.name:<20} {profile.endpoint}", file=output_stream)
    if args.probe:
        print_connection_routes(_connection_route_status(store, args), stream=output_stream)
    return 0


def run_use_connection(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        store = ConnectionStore()
        profile = store.select(args.name)
    except (KeyError, ValueError) as exc:
        print(f"machboost use error: connection {exc} was not found", file=error_stream)
        return 2
    if store.mode() == "auto":
        print("using automatic routing across this Mac and saved devices", file=output_stream)
    elif profile is None:
        print("using local MachBoost server", file=output_stream)
    else:
        print(f"using {profile.name} ({profile.endpoint})", file=output_stream)
    return 0


def run_disconnect(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        profile = ConnectionStore().remove(args.name)
    except (KeyError, ValueError) as exc:
        print(f"machboost disconnect error: connection {exc} was not found", file=error_stream)
        return 2
    print(f"forgot {profile.name}; its API key was removed from Keychain", file=output_stream)
    return 0


def run_mcp(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client, _ = ensure_server(args.endpoint, timeout=args.timeout)
        if args.mcp_command == "list":
            rows = list(client.extensions().get("mcp_servers") or ())
            if args.json:
                print(json.dumps(rows, indent=2), file=output_stream)
            elif not rows:
                print("no MCP connectors configured", file=output_stream)
            else:
                print("STATUS  TOOLS  NAME                 CONNECTION", file=output_stream)
                for row in rows:
                    status = row.get("last_status") or ("enabled" if row.get("enabled") else "disabled")
                    connection = row.get("url") or " ".join(
                        [str(row.get("command") or ""), *map(str, row.get("args") or ())]
                    ).strip()
                    print(
                        f"{status:<7} {int(row.get('tool_count') or 0):<6} "
                        f"{str(row.get('name') or ''):<20} {connection}",
                        file=output_stream,
                    )
            return 0
        if args.mcp_command == "add":
            transport = "http" if args.url else "stdio"
            server = client.configure_mcp_server(
                args.name,
                transport=transport,
                url=args.url,
                command=args.mcp_executable,
                args=tuple(args.arg),
                env=_key_value_pairs(args.env),
                headers=_key_value_pairs(args.header),
            )
            print(f"saved MCP connector {server.get('name')} ({server.get('id')})", file=output_stream)
            return 0
        if args.mcp_command == "remove":
            client.delete_mcp_server(args.server_id)
            print(f"removed MCP connector {args.server_id}", file=output_stream)
            return 0
        if args.mcp_command == "test":
            tools = client.test_mcp_server(args.server_id)
            if args.json:
                print(json.dumps(tools, indent=2), file=output_stream)
            else:
                print(f"connected; {len(tools)} tool(s)", file=output_stream)
                for tool in tools:
                    print(f"  {tool.get('name')}: {tool.get('description') or ''}", file=output_stream)
            return 0
        if args.mcp_command == "search":
            tools = client.search_mcp_tools(args.query, limit=args.limit)
            print(json.dumps(tools, indent=2), file=output_stream)
            return 0
        if args.mcp_command == "call":
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("--arguments must be a JSON object")
            result = client.call_mcp_tool(args.server_id, args.tool, arguments)
            if args.json:
                print(json.dumps(result, indent=2), file=output_stream)
            else:
                print(result.get("text") or "tool completed with no text", file=output_stream)
            return 0
    except (json.JSONDecodeError, MachBoostAPIError, RuntimeError, ValueError) as exc:
        print(f"machboost mcp error: {exc}", file=error_stream)
        return 2
    return 0


def run_skill(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client, _ = ensure_server(args.endpoint, timeout=args.timeout)
        if args.skill_command == "list":
            rows = list(client.extensions().get("skills") or ())
            if args.json:
                print(json.dumps(rows, indent=2), file=output_stream)
            elif not rows:
                print("no reusable instructions configured", file=output_stream)
            else:
                for row in rows:
                    state = "on" if row.get("enabled") else "off"
                    print(f"{state:<3} {row.get('id')}  {row.get('name')}", file=output_stream)
            return 0
        if args.skill_command == "add":
            skill = client.configure_skill(args.name, args.instructions)
            print(f"saved instructions {skill.get('name')} ({skill.get('id')})", file=output_stream)
            return 0
        if args.skill_command == "remove":
            client.delete_skill(args.skill_id)
            print(f"removed instructions {args.skill_id}", file=output_stream)
            return 0
    except (MachBoostAPIError, RuntimeError, ValueError) as exc:
        print(f"machboost skill error: {exc}", file=error_stream)
        return 2
    return 0


def _key_value_pairs(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"expected NAME=VALUE, got {value!r}")
        result[key] = item
    return result


def run_launch(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    if args.integration not in {"claude-desktop", "claude-app"}:
        print(
            f"machboost launch error: unsupported integration {args.integration!r}",
            file=error_stream,
        )
        return 2

    manager = ClaudeDesktopProfileManager()
    is_local = True
    try:
        if args.restore:
            status = manager.restore()
            action = "restored"
        else:
            endpoint, api_key, is_local = _claude_desktop_gateway(args)
            if args.model:
                if not is_local:
                    raise ValueError(
                        "--model configures this Mac; shared-host model mappings are managed by the host"
                    )
                save_model_mappings(args.model)

            client = MachBoostClient(
                endpoint,
                api_token=api_key,
                timeout=args.timeout,
            )
            catalog = client.get("/v1/models")
            routes = [
                item
                for item in catalog.get("data", [])
                if item.get("type") == "model"
                and item.get("anthropic_family_tier")
            ]
            if not routes:
                raise RuntimeError(
                    "the selected MachBoost server has no Claude Desktop-compatible models"
                )
            status = manager.configure(endpoint, api_key)
            action = "connected"

        restarted = False
        if not args.no_restart and manager.installed_application() is not None:
            restart = args.yes
            if not args.yes and sys.stdin.isatty():
                answer = input("Restart Claude Desktop now? Any running task will stop. [y/N] ")
                restart = answer.strip().lower() in {"y", "yes"}
            if restart:
                manager.restart_application()
                restarted = True
    except (MachBoostAPIError, OSError, RuntimeError, ValueError) as exc:
        if not is_local:
            stop_claude_gateway_relay()
        print(f"machboost launch error: {exc}", file=error_stream)
        return 2

    if args.json:
        print(json.dumps({**status, "action": action, "restarted": restarted}, indent=2), file=output_stream)
    elif action == "restored":
        print("Claude Desktop restored to its previous inference profile.", file=output_stream)
    else:
        destination = status.get("upstream") or status["endpoint"]
        print(f"Claude Desktop connected to MachBoost at {destination}.", file=output_stream)
        if status.get("relayed"):
            print("A private localhost bridge keeps the shared-host key out of Claude's profile.", file=output_stream)
        print("Models are discovered from the selected MachBoost host.", file=output_stream)
    if manager.installed_application() is not None and not restarted:
        print("Quit and reopen Claude Desktop for the change to take effect.", file=output_stream)
    return 0


def _claude_desktop_gateway(args: argparse.Namespace) -> tuple[str, str, bool]:
    if args.connection and args.endpoint:
        raise ValueError("use either --connection or --endpoint, not both")
    if args.connection:
        store = ConnectionStore()
        profile = store.get(args.connection)
        token = store.token(profile)
        if not token:
            raise ValueError(f"saved connection {profile.name!r} has no API key")
        endpoint, local_token = start_claude_gateway_relay(profile.endpoint, token)
        return endpoint, local_token, False

    endpoint = normalize_endpoint(args.endpoint or f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    host = (urlparse(endpoint).hostname or "").lower()
    is_local = host in {"127.0.0.1", "localhost", "::1"}
    if is_local:
        ensure_server(endpoint, timeout=args.timeout)
    token = (
        str(args.api_key or "").strip()
        or str(os.environ.get("MACHBOOST_API_TOKEN") or "").strip()
        or (_machboost_app_api_token() if is_local else "")
        or ("machboost" if is_local else "")
    )
    if not token:
        raise ValueError("provide --api-key or use a saved --connection for a remote host")
    if not is_local:
        endpoint, token = start_claude_gateway_relay(endpoint, token)
    return endpoint, token, is_local


def _machboost_app_api_token() -> str:
    return machboost_app_api_token()


def select_native_backend(model: str, backend: str) -> str:
    return resolve_model(model, backend).backend


def render_chat_prompt(system: str, turns: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    if system:
        lines.append(f"System: {system.strip()}")
    for turn in turns:
        role = turn["role"].strip().capitalize()
        lines.append(f"{role}: {turn['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_native_accelerator(args: argparse.Namespace, *, stream=None):
    stream = stream or sys.stderr
    resolution = resolve_model(args.model, args.backend)
    backend = resolution.backend
    context_paths = args.context or None
    if resolution.alias:
        print(f"resolved {resolution.alias!r} to {resolution.model!r}", file=stream)
    print(f"loading {resolution.model!r} with native {backend} backend...", file=stream)
    if backend == "ollama-mlx":
        print("Muse Glimmer uses Ollama's native Apple Silicon MLX engine", file=stream)
    else:
        print("if the model is not cached, HF/MLX may download it into its local cache", file=stream)

    common = {
        "context_paths": context_paths,
        "max_context_chars": args.max_context_chars,
        "ngram": args.ngram,
        "max_draft_tokens": args.max_draft_tokens,
        "candidate_limit": args.candidate_limit,
        "reentry_probe_tokens": args.reentry_probe_tokens,
        "boost_enabled": not args.no_boost,
    }
    if backend == "mlx":
        return Accelerator.from_mlx(
            resolution.model,
            lazy=args.lazy,
            cache_enabled=not args.strict,
            **common,
        )
    if backend == "hf":
        return Accelerator.from_huggingface(
            resolution.model,
            device=args.device,
            local_files_only=args.local_files_only,
            torch_dtype=torch_dtype_from_name(args.dtype),
            **common,
        )
    if backend == "mlx-vlm":
        from .adapters.mlx_vlm import MLXVLMAccelerator

        return MLXVLMAccelerator.from_pretrained(
            resolution.model,
            lazy=args.lazy,
            vision_cache_size=args.vision_cache_size,
        )
    if backend == "dflash":
        from .adapters.dflash import DFlashAccelerator

        return DFlashAccelerator.from_pretrained(
            resolution.model,
            draft_model=args.draft_model,
            draft_quant=args.draft_quant,
            verify_mode=args.verify_mode,
            lazy=args.lazy,
        )
    if backend == "ollama-mlx":
        from .adapters.ollama_mlx import OllamaMLXAccelerator

        return OllamaMLXAccelerator.from_pretrained(
            resolution.model,
            context_paths=context_paths,
            max_context_chars=args.max_context_chars,
            keep_alive=args.keep_alive,
            timeout=args.timeout,
        )
    raise ValueError(f"unsupported backend: {backend}")


def run_native_chat(
    args: argparse.Namespace,
    *,
    input_func=input,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        accelerator = load_native_accelerator(args, stream=error_stream)
    except Exception as exc:
        print(f"machboost run error: {exc}", file=error_stream)
        if args.backend == "auto":
            print("try passing an explicit text or VLM backend", file=error_stream)
        return 2

    try:
        active_images, has_video_frames = prepare_visual_inputs(args, stream=error_stream)
    except Exception as exc:
        print(f"video input error: {exc}", file=error_stream)
        return 2
    turns: list[dict[str, object]] = []
    print(f"machboost run: {args.model}", file=output_stream)
    print("Type /bye, /exit, or /quit to leave. Type /clear to reset chat history.", file=output_stream)
    print(
        "Use /image PATH, /video PATH, /images, or /clear-images to manage visual inputs.",
        file=output_stream,
    )

    while True:
        try:
            user_text = input_func(">>> ")
        except EOFError:
            print("", file=output_stream)
            return 0
        except KeyboardInterrupt:
            print("", file=output_stream)
            return 130

        command = user_text.strip()
        if not command:
            continue
        if command in {"/bye", "/exit", "/quit"}:
            return 0
        if command == "/clear":
            turns = []
            print("chat history cleared", file=output_stream)
            continue
        if command.startswith("/image "):
            image = command.partition(" ")[2].strip()
            if image:
                active_images.append(image)
                print(f"image attached: {image}", file=output_stream)
            continue
        if command.startswith("/video "):
            video = command.partition(" ")[2].strip()
            if video:
                try:
                    selection = sample_video(video, args, stream=error_stream)
                except Exception as exc:
                    print(f"video input error: {exc}", file=error_stream)
                    continue
                active_images.extend(selection.images)
                has_video_frames = True
            continue
        if command == "/images":
            print("\n".join(active_images) if active_images else "no images attached", file=output_stream)
            continue
        if command == "/clear-images":
            active_images = []
            has_video_frames = False
            print("images cleared", file=output_stream)
            continue

        content = video_prompt(user_text) if has_video_frames else user_text
        turns.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": args.system or DEFAULT_CHAT_SYSTEM}]
        messages.extend(turns)
        if active_images:
            if not getattr(accelerator, "supports_vision", False):
                turns.pop()
                print("generation error: attached images require a vision model", file=error_stream)
                return 2
            messages[-1]["images"] = list(active_images)
        streamed = False

        def emit(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            print(chunk, end="", flush=True, file=output_stream)

        started = time.perf_counter()
        thinking_started = False

        def emit_thinking(chunk: str) -> None:
            nonlocal thinking_started
            if not chunk:
                return
            if not thinking_started:
                print("thinking> ", end="", flush=True, file=error_stream)
                thinking_started = True
            print(chunk, end="", flush=True, file=error_stream)

        try:
            kwargs = {"max_tokens": args.max_tokens, "on_text": emit}
            if args.think:
                kwargs["enable_thinking"] = args.think
            if args.show_thinking:
                kwargs["on_thinking"] = emit_thinking
            if getattr(accelerator, "supports_vision", False):
                kwargs.update(
                    use_vision_cache=not args.no_vision_cache,
                    temperature=args.temperature,
                    cold_vision_mode=args.cold_vision,
                    cold_vision_max_edge=args.vision_max_edge,
                    vision_token_mode=args.vision_tokens,
                    vision_token_ratio=args.vision_token_ratio,
                    vision_token_layer=args.vision_token_layer,
                    vision_token_bucket=args.vision_token_bucket,
                    vision_calibration=load_vision_calibration(args.vision_calibration),
                )
            response, stats = accelerator.generate_chat(messages, **kwargs)
        except KeyboardInterrupt:
            print("", file=output_stream)
            return 130
        except Exception as exc:
            turns.pop()
            print(f"generation error: {exc}", file=error_stream)
            return 2

        elapsed_s = time.perf_counter() - started
        response = response.strip()
        if thinking_started:
            print("", flush=True, file=error_stream)
        if streamed:
            print("", flush=True, file=output_stream)
        else:
            print(response, flush=True, file=output_stream)
        if args.show_stats:
            tokens_per_second = stats.generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
            stats_backend = getattr(stats, "backend", "")
            if stats_backend == "mlx-vlm":
                print(
                    "stats: "
                    f"elapsed={elapsed_s:.2f}s "
                    f"tokens_per_second={tokens_per_second:.2f} "
                    f"prompt_tps={stats.prompt_tokens_per_second:.2f} "
                    f"vision_cache={'hit' if stats.visual_cache_hit else 'miss' if stats.visual_cache_miss else 'off'}",
                    file=output_stream,
                )
            elif stats_backend == "ollama-mlx":
                ttft = getattr(stats, "time_to_first_token_seconds", None)
                ttft_text = "n/a" if ttft is None else f"{ttft:.3f}s"
                print(
                    "stats: "
                    f"elapsed={elapsed_s:.2f}s "
                    f"ttft={ttft_text} "
                    f"decode_tps={stats.generation_tokens_per_second:.2f} "
                    f"prompt_tps={stats.prompt_tokens_per_second:.2f} "
                    f"tokens={stats.generated_tokens} "
                    f"images={stats.image_count} "
                    f"tool_calls={len(stats.tool_calls)}",
                    file=output_stream,
                )
            else:
                print(
                    "stats: "
                    f"elapsed={elapsed_s:.2f}s "
                    f"tokens_per_second={tokens_per_second:.2f} "
                    f"accepted={stats.accepted_draft_tokens} "
                    f"target_calls={stats.target_calls}/{stats.baseline_target_calls} "
                    f"estimated_speedup={stats.estimated_speedup:.2f}x",
                    file=output_stream,
                )
        turns.append({"role": "assistant", "content": response})


def ollama_options(args: argparse.Namespace) -> dict:
    options = {}
    if args.temperature is not None:
        options["temperature"] = float(args.temperature)
    if args.ctx is not None:
        options["num_ctx"] = int(args.ctx)
    if args.num_predict is not None:
        options["num_predict"] = int(args.num_predict)
    return options


def print_pull_status(status, *, stream=None) -> None:
    stream = stream or sys.stderr
    if status.total > 0:
        percent = int(status.progress * 100)
        print(f"pull: {status.status} {percent}%", file=stream)
        return
    if status.status:
        print(f"pull: {status.status}", file=stream)


def ensure_ollama_model(adapter: OllamaHTTPAdapter, *, no_pull: bool = False, stream=None) -> None:
    stream = stream or sys.stderr
    if adapter.has_model():
        return
    if no_pull:
        raise OllamaHTTPError(
            f"Model {adapter.model!r} is not installed. Run without --no-pull or use `ollama pull {adapter.model}`."
        )
    print(f"model {adapter.model!r} is not installed; pulling with Ollama...", file=stream)
    last_status = None
    for status in adapter.pull():
        last_status = status
        print_pull_status(status, stream=stream)
    if last_status is None:
        print("pull: completed", file=stream)


def run_ollama_chat(
    args: argparse.Namespace,
    *,
    input_func=input,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    adapter = OllamaHTTPAdapter(args.model, endpoint=args.endpoint, timeout=args.timeout)
    options = ollama_options(args)
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    try:
        ensure_ollama_model(adapter, no_pull=args.no_pull, stream=error_stream)
    except OllamaHTTPError as exc:
        print(f"ollama error: {exc}", file=error_stream)
        print("Make sure Ollama is installed and running. Try `ollama serve` in another terminal.", file=error_stream)
        return 2

    print(f"machboost ollama wrapper: {adapter.model}", file=output_stream)
    print("Type /bye, /exit, or /quit to leave. Type /clear to reset chat history.", file=output_stream)

    while True:
        try:
            prompt = input_func(">>> ")
        except EOFError:
            print("", file=output_stream)
            return 0
        except KeyboardInterrupt:
            print("", file=output_stream)
            return 130

        command = prompt.strip()
        if not command:
            continue
        if command in {"/bye", "/exit", "/quit"}:
            return 0
        if command == "/clear":
            messages = [message for message in messages if message.get("role") == "system"]
            print("chat history cleared", file=output_stream)
            continue

        messages.append({"role": "user", "content": prompt})
        chunks: list[str] = []
        try:
            for chunk in adapter.chat(messages, options=options):
                if chunk.content:
                    print(chunk.content, end="", flush=True, file=output_stream)
                    chunks.append(chunk.content)
                if chunk.done:
                    break
        except KeyboardInterrupt:
            print("", file=output_stream)
            return 130
        except OllamaHTTPError as exc:
            messages.pop()
            print(f"\nollama error: {exc}", file=error_stream)
            return 2

        print("", file=output_stream)
        messages.append({"role": "assistant", "content": "".join(chunks)})


def native_server_options(args: argparse.Namespace) -> dict:
    return {
        "backend": args.backend,
        "context_paths": list(args.context or ()),
        "max_context_chars": args.max_context_chars,
        "ngram": args.ngram,
        "max_draft_tokens": args.max_draft_tokens,
        "candidate_limit": args.candidate_limit,
        "reentry_probe_tokens": args.reentry_probe_tokens,
        "no_boost": args.no_boost,
        "strict": args.strict,
        "lazy": args.lazy,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "num_predict": args.max_tokens,
        "temperature": args.temperature,
        "no_vision_cache": args.no_vision_cache,
        "vision_cache_size": args.vision_cache_size,
        "cold_vision": args.cold_vision,
        "vision_max_edge": args.vision_max_edge,
        "vision_tokens": args.vision_tokens,
        "vision_token_ratio": args.vision_token_ratio,
        "vision_token_layer": args.vision_token_layer,
        "vision_token_bucket": args.vision_token_bucket,
        "vision_calibration": args.vision_calibration,
        "draft_model": args.draft_model,
        "draft_quant": args.draft_quant,
        "verify_mode": args.verify_mode,
        "num_ctx": args.ctx,
        "_think": args.think or False,
        "_reasoning_strength": args.think,
    }


def _automatic_host_pool(
    args: argparse.Namespace,
    *,
    error_stream=None,
    autostart: Optional[bool] = None,
) -> Optional[MachBoostHostPool]:
    if args.endpoint is not None or os.environ.get("MACHBOOST_HOST", "").strip():
        return None
    store = ConnectionStore()
    if store.mode() != "auto":
        return None

    error_stream = error_stream or sys.stderr
    targets: list[HostTarget] = []
    local_endpoint = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    should_autostart = (
        not getattr(args, "no_autostart", False)
        if autostart is None
        else autostart
    )
    if should_autostart:
        try:
            _local, started = ensure_server(
                local_endpoint,
                timeout=min(30.0, args.timeout),
            )
            if started:
                print(
                    f"started resident MachBoost server at {local_endpoint}",
                    file=error_stream,
                )
        except MachBoostAPIError as exc:
            print(f"local host unavailable: {exc}", file=error_stream)
    targets.append(
        HostTarget(
            id="local",
            name="This Mac",
            endpoint=local_endpoint,
            api_token=_machboost_app_api_token(),
        )
    )
    targets.extend(
        HostTarget(
            id=profile.id,
            name=profile.name,
            endpoint=profile.endpoint,
            api_token=store.token(profile),
        )
        for profile in store.list()
    )
    return MachBoostHostPool(targets, timeout=args.timeout)


def _connection_route_status(store: ConnectionStore, args: argparse.Namespace) -> dict:
    targets = [
        HostTarget(
            "local",
            "This Mac",
            f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
            _machboost_app_api_token(),
        ),
        *[
            HostTarget(profile.id, profile.name, profile.endpoint, store.token(profile))
            for profile in store.list()
        ],
    ]
    pool = MachBoostHostPool(targets, timeout=args.timeout)
    return pool.route_status(args.model)


def print_connection_routes(status: dict, *, stream=None) -> None:
    stream = stream or sys.stdout
    print("", file=stream)
    print("AUTO ROUTE PROBE", file=stream)
    print(
        f"{'HOST':<20} {'STATE':<8} {'MODEL':<8} {'LOAD':<6} "
        f"{'RTT':>8} {'ACTIVE':>7} {'QUEUE':>6} {'ETA':>8}",
        file=stream,
    )
    selected = status.get("selected")
    for host in status.get("hosts") or ():
        state = "online" if host["online"] else "offline"
        model = "ready" if host["supports_model"] else "missing"
        loaded = "warm" if host["model_loaded"] else "cold"
        rtt = f"{float(host['round_trip_seconds']) * 1000:.0f}ms"
        score = host.get("score")
        eta = f"{float(score):.2f}s" if score is not None else "-"
        marker = "*" if host["id"] == selected else " "
        print(
            f"{marker}{host['name'][:18]:<19} {state:<8} {model:<8} {loaded:<6} "
            f"{rtt:>8} {int(host['active_requests']):>7} {int(host['queued_requests']):>6} {eta:>8}",
            file=stream,
        )


def connect_resident(
    args: argparse.Namespace,
    *,
    error_stream=None,
) -> MachBoostClient | MachBoostHostPool:
    error_stream = error_stream or sys.stderr
    pool = _automatic_host_pool(args, error_stream=error_stream)
    if pool is not None:
        return pool
    if getattr(args, "no_autostart", False):
        client = MachBoostClient(args.endpoint, timeout=args.timeout)
        if not client.is_healthy():
            raise MachBoostAPIError("MachBoost server is not running; start it with `machboost serve`")
        return client
    client, started = ensure_server(args.endpoint, timeout=min(30.0, args.timeout))
    client.timeout = args.timeout
    if started:
        print(f"started resident MachBoost server at {client.endpoint}", file=error_stream)
    return client


def ensure_resident_model(
    client: MachBoostClient,
    args: argparse.Namespace,
    *,
    stream=None,
) -> None:
    stream = stream or sys.stderr
    resolution = resolve_model(args.model, args.backend)
    if resolution.backend != "ollama-mlx":
        return
    response = client.show(
        args.model,
        preflight=True,
        backend=resolution.backend,
    )
    preflight = response.get("preflight") or response
    if not preflight.get("runtime_available", False):
        raise MachBoostAPIError(
            str(preflight.get("reason") or "Muse Glimmer requires current Ollama on Apple Silicon")
        )
    if preflight.get("cached"):
        return
    print(f"model {resolution.model!r} is not installed; pulling 21 GB...", file=stream)
    events = client.pull(args.model, stream=True)
    last_status = None
    for event in events:
        if event.get("error"):
            raise MachBoostAPIError(str(event["error"]))
        status = str(event.get("status") or "")
        if status and status != last_status:
            print(f"pull: {status}", file=stream)
            last_status = status


def prepare_visual_inputs(
    args: argparse.Namespace,
    *,
    stream=None,
) -> tuple[list[str], bool]:
    images = list(args.image or ())
    videos = list(args.video or ())
    for video in videos:
        images.extend(sample_video(video, args, stream=stream).images)
    return images, bool(videos)


def sample_video(video: str, args: argparse.Namespace, *, stream=None) -> VideoSelection:
    stream = stream or sys.stderr
    selection = TemporalVideoSampler().sample(
        video,
        fps=args.video_fps,
        change_threshold=args.video_change_threshold,
        max_frames=args.video_max_frames,
    )
    print(
        "video: "
        f"selected={selection.selected_frames}/{selection.sampled_frames} "
        f"reduction={selection.reduction_rate:.0%} "
        f"elapsed={selection.elapsed_seconds:.2f}s "
        f"extract_cache={'hit' if selection.extraction_cache_hit else 'miss'}",
        file=stream,
    )
    return selection


def video_prompt(prompt: str) -> str:
    return (
        "The attached images are chronological frames selected from a video. "
        "Use their order when answering.\n\n"
        + prompt
    )


def chat_route_options(args: argparse.Namespace) -> Optional[dict[str, dict[str, str]]]:
    mode = str(getattr(args, "route", "local_only"))
    if mode == "local_only":
        return None
    route = {"mode": mode}
    if getattr(args, "provider", None):
        route["provider_id"] = args.provider
    if getattr(args, "provider_model", None):
        route["model"] = args.provider_model
    return {"route": route}


def print_tool_call_summary(tool_calls: Sequence[dict], *, stream=None) -> None:
    stream = stream or sys.stdout
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "tool")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}
        if isinstance(arguments, dict):
            fields = ", ".join(
                f"{key}={str(value)[:80]!r}" for key, value in list(arguments.items())[:4]
            )
        else:
            fields = str(arguments or "")[:160]
        print(f"tool> {name}({fields})", file=stream)


def run_resident_chat(
    args: argparse.Namespace,
    *,
    input_func=input,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    session_started = time.perf_counter()
    try:
        client = connect_resident(args, error_stream=error_stream)
        ensure_resident_model(client, args, stream=error_stream)
    except (MachBoostAPIError, ValueError) as exc:
        print(f"machboost server error: {exc}", file=error_stream)
        return 2

    print(f"loading {args.model}...", file=error_stream, flush=True)
    try:
        preload = client.load(
            args.model,
            options=native_server_options(args),
            keep_alive=args.keep_alive,
            warmup=True,
        )
    except MachBoostAPIError as exc:
        print(f"machboost load error: {exc}", file=error_stream)
        return 2
    preload_wall = time.perf_counter() - session_started
    instance = preload.get("instance") or {}
    backend = str(instance.get("backend") or "unknown")
    resolved_model = str(instance.get("model") or args.model)
    model_load = float(preload.get("load_duration_seconds") or 0.0)
    warmup_duration = float(preload.get("warmup_duration_seconds") or 0.0)

    try:
        active_images, has_video_frames = prepare_visual_inputs(args, stream=error_stream)
    except Exception as exc:
        print(f"video input error: {exc}", file=error_stream)
        return 2
    turns: list[dict[str, str]] = []
    show_stats = bool(args.show_stats)
    state = f"load {model_load:.2f}s" if model_load > 0 else "resident"
    if warmup_duration > 0:
        state += f" | compile {warmup_duration:.2f}s"
    width = max(48, min(88, shutil.get_terminal_size((80, 24)).columns))
    print("-" * width, file=output_stream)
    print(f"MachBoost  {args.model}", file=output_stream)
    print(
        f"{backend} | {state} | wall {preload_wall:.2f}s | {client.endpoint}",
        file=output_stream,
    )
    route_name = str(getattr(args, "route", "local_only"))
    print(f"route {route_name.replace('_', ' ')} | /help for commands", file=output_stream)
    print("-" * width, file=output_stream)
    if resolved_model != args.model and args.show_stats:
        print(f"model: {resolved_model}", file=output_stream)
    print("", file=output_stream)

    while True:
        try:
            user_text = input_func("you> ")
        except EOFError:
            print("", file=output_stream)
            unload_resident_model(client, args.model, stream=output_stream)
            return 0
        except KeyboardInterrupt:
            print("", file=output_stream)
            continue

        command = user_text.strip()
        if not command:
            continue
        if command in {"/bye", "/exit", "/quit"}:
            return 0
        if command in {"/?", "/help"}:
            print(CHAT_HELP, file=output_stream)
            continue
        if command == "/status":
            print(
                f"model={resolved_model} backend={backend} host={client.endpoint} "
                f"route={route_name} turns={len(turns) // 2} images={len(active_images)}",
                file=output_stream,
            )
            continue
        if command == "/route":
            if isinstance(client, MachBoostHostPool):
                print_connection_routes(client.route_status(args.model), stream=output_stream)
            else:
                route = chat_route_options(args)
                print(
                    "route: " + (json.dumps(route["route"], sort_keys=True) if route else "local_only"),
                    file=output_stream,
                )
            continue
        if command.startswith("/stats"):
            setting = command.partition(" ")[2].strip().lower()
            if setting in {"on", "off"}:
                show_stats = setting == "on"
            else:
                show_stats = not show_stats
            print(f"response stats {'on' if show_stats else 'off'}", file=output_stream)
            continue
        if command == "/unload":
            unload_resident_model(client, args.model, stream=output_stream)
            return 0
        if command == "/clear":
            turns = []
            print("chat history cleared", file=output_stream)
            continue
        if command.startswith("/image "):
            image = command.partition(" ")[2].strip()
            if image:
                active_images.append(image)
                print(f"image attached: {image}", file=output_stream)
            continue
        if command.startswith("/video "):
            video = command.partition(" ")[2].strip()
            if video:
                try:
                    selection = sample_video(video, args, stream=error_stream)
                except Exception as exc:
                    print(f"video input error: {exc}", file=error_stream)
                    continue
                active_images.extend(selection.images)
                has_video_frames = True
            continue
        if command == "/images":
            print("\n".join(active_images) if active_images else "no images attached", file=output_stream)
            continue
        if command == "/clear-images":
            active_images = []
            has_video_frames = False
            print("images cleared", file=output_stream)
            continue

        content = video_prompt(user_text) if has_video_frames else user_text
        turns.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": args.system or DEFAULT_CHAT_SYSTEM}, *turns]
        response_parts: list[str] = []
        tool_calls: list[dict] = []
        final_row: dict = {}
        started = time.perf_counter()
        rows = None
        thinking_started = False
        answer_started = False

        def show_thinking(chunk: str) -> None:
            nonlocal thinking_started
            if not args.show_thinking or not chunk:
                return
            if not thinking_started:
                print("thinking> ", end="", flush=True, file=output_stream)
                thinking_started = True
            print(chunk, end="", flush=True, file=output_stream)

        def show_answer(chunk: str) -> None:
            nonlocal answer_started
            if not chunk:
                return
            if not answer_started:
                if thinking_started:
                    print("", flush=True, file=output_stream)
                print("assistant> ", end="", flush=True, file=output_stream)
                answer_started = True
            print(chunk, end="", flush=True, file=output_stream)

        try:
            request_options = {
                "options": native_server_options(args),
                "keep_alive": args.keep_alive,
                "stream": True,
            }
            if active_images:
                request_options["images"] = active_images
            route = chat_route_options(args)
            if route is not None:
                request_options["machboost"] = route
            rows = client.chat(args.model, messages, **request_options)
            for row in rows:
                message = row.get("message") or {}
                chunk = str(message.get("content") or "")
                thinking = str(message.get("thinking") or "")
                show_thinking(thinking)
                if chunk:
                    show_answer(chunk)
                    response_parts.append(chunk)
                if message.get("tool_calls"):
                    tool_calls.extend(message["tool_calls"])
                if row.get("done"):
                    final_row = row
        except KeyboardInterrupt:
            close = getattr(rows, "close", None)
            if callable(close):
                close()
            print("\n[generation stopped]", file=output_stream)
            turns.pop()
            continue
        except MachBoostAPIError as exc:
            turns.pop()
            print(f"\ngeneration error: {exc}", file=error_stream)
            continue

        if thinking_started and not answer_started:
            print("", flush=True, file=output_stream)
        if not answer_started and tool_calls:
            print("assistant> ", file=output_stream)
        elif answer_started:
            print("", flush=True, file=output_stream)
        response = "".join(response_parts).strip()
        if tool_calls:
            print_tool_call_summary(tool_calls, stream=output_stream)
        if show_stats:
            print_resident_stats(final_row, time.perf_counter() - started, stream=output_stream)
        turns.append({"role": "assistant", "content": response})


def unload_resident_model(
    client: MachBoostClient,
    model: str,
    *,
    stream=None,
) -> None:
    stream = stream or sys.stdout
    try:
        result = client.stop(model)
    except MachBoostAPIError as exc:
        print(f"unload warning: {exc}", file=stream)
        return
    print(f"unloaded {int(result.get('unloaded') or 0)} model instance(s)", file=stream)


def run_resident_completion(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        prompt = completion_prompt(args)
        images, has_video_frames = prepare_visual_inputs(args, stream=error_stream)
        if has_video_frames:
            prompt = video_prompt(prompt)
        if args.direct:
            accelerator = load_native_accelerator(args, stream=error_stream)
            thinking_started = False

            def emit_thinking(text: str) -> None:
                nonlocal thinking_started
                if not text:
                    return
                if not thinking_started:
                    print("thinking> ", end="", flush=True, file=error_stream)
                    thinking_started = True
                print(text, end="", flush=True, file=error_stream)

            kwargs = {
                "max_tokens": args.max_tokens,
                "on_text": lambda text: print(text, end="", flush=True, file=output_stream),
            }
            if args.think:
                kwargs["enable_thinking"] = args.think
            if args.show_thinking:
                kwargs["on_thinking"] = emit_thinking
            if images and not getattr(accelerator, "supports_vision", False):
                raise ValueError("attached images require a vision model")
            if getattr(accelerator, "supports_vision", False):
                kwargs.update(
                    images=images or None,
                    use_vision_cache=not args.no_vision_cache,
                    temperature=args.temperature,
                    cold_vision_mode=args.cold_vision,
                    cold_vision_max_edge=args.vision_max_edge,
                    vision_token_mode=args.vision_tokens,
                    vision_token_ratio=args.vision_token_ratio,
                    vision_token_layer=args.vision_token_layer,
                    vision_token_bucket=args.vision_token_bucket,
                    vision_calibration=load_vision_calibration(args.vision_calibration),
                )
            started = time.perf_counter()
            text, stats = accelerator.generate(prompt, **kwargs)
            elapsed_s = time.perf_counter() - started
            if thinking_started:
                print("", file=error_stream)
            print("", file=output_stream)
            if args.show_stats:
                if getattr(stats, "backend", "") == "ollama-mlx":
                    print(
                        "stats: "
                        f"elapsed={elapsed_s:.2f}s "
                        f"decode_tps={stats.generation_tokens_per_second:.2f} "
                        f"prompt_tps={stats.prompt_tokens_per_second:.2f} "
                        f"tokens={stats.generated_tokens} "
                        f"images={stats.image_count} "
                        f"tool_calls={len(stats.tool_calls)}",
                        file=output_stream,
                    )
                else:
                    print(
                        f"stats: generated={stats.generated_tokens} "
                        f"accepted={stats.accepted_draft_tokens}",
                        file=output_stream,
                    )
            return 0

        client = connect_resident(args, error_stream=error_stream)
        ensure_resident_model(client, args, stream=error_stream)
        started = time.perf_counter()
        request_options = {
            "options": native_server_options(args),
            "keep_alive": args.keep_alive,
            "stream": True,
        }
        if images:
            request_options["images"] = images
        rows = client.generate(args.model, prompt, **request_options)
        final_row: dict = {}
        for row in rows:
            chunk = str(row.get("response") or "")
            thinking = str(row.get("thinking") or "")
            if thinking and args.show_thinking:
                print(thinking, end="", flush=True, file=error_stream)
            if chunk:
                print(chunk, end="", flush=True, file=output_stream)
            if row.get("done"):
                final_row = row
        print("", file=output_stream)
        if args.show_stats:
            print_resident_stats(final_row, time.perf_counter() - started, stream=output_stream)
        return 0
    except (MachBoostAPIError, OSError, ValueError) as exc:
        print(f"machboost complete error: {exc}", file=error_stream)
        return 2


def completion_prompt(args: argparse.Namespace) -> str:
    if args.file:
        if args.prompt:
            raise ValueError("pass either a prompt or --file, not both")
        return Path(args.file).expanduser().read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return sys.stdin.read()


def print_resident_stats(row: dict, elapsed_s: float, *, stream=None) -> None:
    stream = stream or sys.stdout
    metrics = row.get("machboost") or {}
    stats = metrics.get("stats") or {}
    generated = int(row.get("eval_count") or stats.get("generated_tokens") or 0)
    total_s = float(row.get("total_duration") or 0) / 1_000_000_000 or elapsed_s
    load_s = float(row.get("load_duration") or 0) / 1_000_000_000
    prompt_count = int(row.get("prompt_eval_count") or stats.get("prompt_tokens") or 0)
    prompt_s = float(row.get("prompt_eval_duration") or 0) / 1_000_000_000
    eval_s = float(row.get("eval_duration") or 0) / 1_000_000_000
    if eval_s <= 0:
        eval_s = max(0.0, elapsed_s - load_s - prompt_s)
    rate = generated / eval_s if eval_s > 0 else 0.0
    prompt_rate = prompt_count / prompt_s if prompt_s > 0 else 0.0
    ttft = metrics.get("time_to_first_token_seconds")
    if ttft is None:
        ttft = stats.get("time_to_first_token_seconds")
    cache_state = ""
    if int(stats.get("image_count") or 0) > 0:
        cache_state = (
            " vision_cache=hit"
            if stats.get("visual_cache_hit")
            else " vision_cache=miss"
            if stats.get("visual_cache_miss")
            else " vision_cache=off"
        )
        cold_vision = stats.get("cold_vision") or {}
        if cold_vision.get("enabled"):
            cache_state += (
                f" cold_vision={cold_vision.get('mode', 'unknown')}"
                f":{int(cold_vision.get('target_max_edge') or 0)}px"
            )
        post_fusion = stats.get("post_fusion_vision") or {}
        if post_fusion.get("enabled"):
            cache_state += (
                f" vision_tokens={post_fusion.get('mode', 'unknown')}"
                f":{float(post_fusion.get('actual_visual_retention_ratio') or 0.0):.0%}"
            )
    print(f"total duration:       {total_s:.3f}s", file=stream)
    print(f"load duration:        {load_s:.3f}s", file=stream)
    if ttft is not None:
        print(f"time to first token:  {float(ttft):.3f}s", file=stream)
    print(f"prompt eval count:    {prompt_count} token(s)", file=stream)
    print(f"prompt eval duration: {prompt_s:.3f}s", file=stream)
    print(f"prompt eval rate:     {prompt_rate:.2f} tokens/s", file=stream)
    print(f"eval count:           {generated} token(s)", file=stream)
    print(f"eval duration:        {eval_s:.3f}s", file=stream)
    print(f"eval rate:            {rate:.2f} tokens/s", file=stream)
    backend = str(metrics.get("backend") or stats.get("backend") or "unknown")
    if backend == "ollama-mlx":
        speculative = "on" if stats.get("native_speculative_decoding", True) else "off"
        print(
            "machboost: "
            f"backend={backend} "
            f"native_speculative={speculative} "
            f"tool_calls={len(stats.get('tool_calls') or ())}"
            f"{cache_state}",
            file=stream,
        )
    else:
        print(
            "machboost: "
            f"backend={backend} "
            f"accepted={int(stats.get('accepted_draft_tokens') or 0)} "
            f"target_calls={int(stats.get('target_calls') or 0)}/"
            f"{int(stats.get('baseline_target_calls') or 0)}"
            f"{cache_state}",
            file=stream,
        )


def run_serve(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    print(f"MachBoost server listening on http://{args.host}:{args.port}", file=output_stream)
    print(
        f"Serving {args.replicas} replica(s) per text model; "
        f"queue={args.max_queue}, timeout={args.queue_timeout:g}s.",
        file=output_stream,
    )
    print(
        "Models unload after their keep-alive or via `machboost stop`/`machboost shutdown`.",
        file=output_stream,
    )
    remote_bind = args.host not in {"127.0.0.1", "localhost", "::1"}
    require_auth = bool(args.require_auth or remote_bind)
    api_token = os.environ.get("MACHBOOST_API_TOKEN")
    if require_auth and not api_token:
        print(
            "machboost serve error: secured or LAN serving requires MACHBOOST_API_TOKEN",
            file=error_stream,
        )
        return 2
    if require_auth:
        print("Bearer authentication is required for API routes.", file=output_stream)
    team_store = None
    if args.team:
        team_path = Path(args.team_db or "~/.machboost/team.sqlite3").expanduser()
        team_store = TeamStore(team_path)
        settings: dict[str, object] = {}
        if args.trace_mode is not None:
            settings["trace_mode"] = args.trace_mode
        if args.trace_retention_days is not None:
            settings["retention_days"] = (
                None if args.trace_retention_days == 0 else args.trace_retention_days
            )
        if args.trace_max_mb is not None:
            settings["max_storage_bytes"] = args.trace_max_mb * 1024 * 1024
        if settings:
            team_store.update_settings(**settings)
        print(
            f"Team gateway enabled; policy and traces: {team_path}",
            file=output_stream,
        )
    try:
        serve_runtime(
            args.host,
            args.port,
            replicas=args.replicas,
            max_queue=args.max_queue,
            queue_timeout=args.queue_timeout,
            api_token=api_token,
            require_auth=require_auth,
            team_store=team_store,
        )
    except KeyboardInterrupt:
        print("\nMachBoost server stopped.", file=error_stream)
    except (OSError, ValueError) as exc:
        if team_store is not None:
            team_store.close()
        print(f"machboost serve error: {exc}", file=error_stream)
        return 2
    return 0


def run_pull(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client = connect_resident(args, error_stream=error_stream)
        print(f"pulling {args.model}...", file=output_stream)
        result = client.pull(args.model, revision=args.revision)
        print(f"success: {result['path']}", file=output_stream)
        return 0
    except MachBoostAPIError as exc:
        print(f"machboost pull error: {exc}", file=error_stream)
        return 2


def run_model_alias_action(
    args: argparse.Namespace,
    *,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client = connect_resident(args, error_stream=error_stream)
        if args.command == "create":
            options = {}
            for value in args.option:
                key, separator, raw = value.partition("=")
                if not separator or not key.strip():
                    raise ValueError("--option must use key=value")
                try:
                    options[key.strip()] = json.loads(raw)
                except json.JSONDecodeError:
                    options[key.strip()] = raw
            result = client.create_model(
                args.model,
                args.source,
                system=args.system,
                template=args.template,
                options=options,
            )
            print(f"created {result['model']['name']} from {result['model']['source']}", file=output_stream)
        elif args.command == "cp":
            result = client.copy_model(args.source, args.destination)
            print(f"copied {args.source} to {result['model']['name']}", file=output_stream)
        else:
            if not client.delete_model(args.model, purge=args.weights):
                kind = "model" if args.weights else "model alias"
                print(f"{kind} not found: {args.model}", file=error_stream)
                return 2
            suffix = " and its downloaded weights" if args.weights else ""
            print(f"removed {args.model}{suffix}", file=output_stream)
        return 0
    except (MachBoostAPIError, ValueError) as exc:
        print(f"machboost {args.command} error: {exc}", file=error_stream)
        return 2


def run_warm(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    started = time.perf_counter()
    try:
        client = connect_resident(args, error_stream=error_stream)
        print(f"loading {args.model}...", file=error_stream, flush=True)
        result = client.load(
            args.model,
            options=native_server_options(args),
            keep_alive=args.keep_alive,
            warmup=True,
        )
        instance = result["instance"]
        wall = time.perf_counter() - started
        load_duration = float(result["load_duration_seconds"])
        warmup_duration = float(result.get("warmup_duration_seconds") or 0.0)
        state = "loaded" if load_duration > 0 else "already resident"
        print(
            f"{state} {instance['model']} on {instance['backend']}; "
            f"model_load={load_duration:.2f}s compile_warmup={warmup_duration:.2f}s "
            f"wall={wall:.2f}s "
            f"keep_alive={args.keep_alive}",
            file=output_stream,
        )
        return 0
    except MachBoostAPIError as exc:
        print(f"machboost warm error: {exc}", file=error_stream)
        return 2


def run_latency_bench(
    args: argparse.Namespace,
    *,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    client = None
    try:
        if args.engine in {"machboost", "both"}:
            client = connect_resident(args, error_stream=error_stream)
        artifact = benchmark_chat_latency(
            args.model,
            prompt=args.prompt,
            system=args.system or DEFAULT_CHAT_SYSTEM,
            runs=args.runs,
            warmups=args.warmups,
            max_tokens=args.max_tokens,
            engine=args.engine,
            backend=args.backend,
            keep_alive=args.keep_alive,
            machboost_client=client,
            ollama_model=args.ollama_model,
            ollama_endpoint=args.ollama_endpoint,
            timeout=args.timeout,
            draft_num_predict=args.draft_num_predict,
        )
    except (MachBoostAPIError, OllamaHTTPError, OSError, ValueError) as exc:
        print(f"machboost bench error: {exc}", file=error_stream)
        return 2
    if args.json:
        print(json.dumps(artifact, indent=2), file=output_stream)
    else:
        print_latency_benchmark(artifact, stream=output_stream)
    return 0


def run_context_bench(
    args: argparse.Namespace,
    *,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        prompt = _context_benchmark_prompt(args)
        contexts = resolve_context(
            args.context_text,
            max_chars=args.max_context_chars,
        )
        remaining = args.max_context_chars - sum(len(text) for text in contexts)
        contexts += read_context_paths(args.context, max_chars=max(0, remaining))
        if not contexts:
            raise ValueError("provide --context PATH or --context-text TEXT")

        resolution = resolve_model(args.model, args.backend)
        if resolution.backend not in {"mlx", "hf"}:
            raise ValueError("bench-context supports text MLX and Hugging Face models")
        print(
            f"loading {resolution.model!r} once for native and MachBoost paths...",
            file=error_stream,
            flush=True,
        )
        load_started = time.perf_counter()
        common = {
            "context": contexts,
            "ngram": args.ngram,
            "max_draft_tokens": args.max_draft_tokens,
            "candidate_limit": args.candidate_limit,
            "reentry_probe_tokens": args.reentry_probe_tokens,
        }
        if resolution.backend == "mlx":
            accelerator = Accelerator.from_mlx(
                resolution.model,
                lazy=args.lazy,
                cache_enabled=not args.strict,
                **common,
            )
        else:
            accelerator = Accelerator.from_huggingface(
                resolution.model,
                device=args.device,
                local_files_only=args.local_files_only,
                torch_dtype=torch_dtype_from_name(args.dtype),
                **common,
            )
        load_seconds = time.perf_counter() - load_started
        artifact = benchmark_context_acceleration(
            accelerator,
            prompt,
            model=resolution.model,
            backend=resolution.backend,
            context_fingerprint=context_fingerprint(contexts),
            context_chars=sum(len(text) for text in contexts),
            runs=args.runs,
            warmups=args.warmups,
            max_tokens=args.max_tokens,
        )
        artifact["model_load_seconds"] = load_seconds
    except Exception as exc:
        print(f"machboost bench-context error: {exc}", file=error_stream)
        return 2

    if args.json:
        print(json.dumps(artifact, indent=2), file=output_stream)
    else:
        print_context_benchmark(artifact, stream=output_stream)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"saved benchmark artifact to {output_path}", file=error_stream)
    return 0 if artifact["summary"]["valid"] else 1


def run_decode_bench(
    args: argparse.Namespace,
    *,
    error_stream=None,
) -> int:
    error_stream = error_stream or sys.stderr
    try:
        from dflash_mlx.benchmark import main as dflash_benchmark_main
        from dflash_mlx.model import DFlashDraftModelArgs

        from .adapters.dflash import _load_runtime_bundle_compat

        resolution = resolve_model(args.model, "dflash")
        validation_valid = True
        benchmark_args = [
            "--model",
            resolution.model,
            "--verify-mode",
            args.verify_mode,
            "--max-tokens",
            str(args.max_tokens),
            "--repeat",
            str(args.runs),
            "--cooldown",
            str(args.cooldown),
        ]
        if args.prompt:
            benchmark_args.extend(["--prompt", args.prompt])
        if args.prompt_file:
            benchmark_args.extend(["--prompt-file", args.prompt_file])
            limit = args.limit or _jsonl_row_count(args.prompt_file)
            benchmark_args.extend(["--limit", str(limit)])
        if args.draft_model:
            benchmark_args.extend(["--draft", args.draft_model])
        if args.draft_quant:
            benchmark_args.extend(["--draft-quant", args.draft_quant])
        if args.no_eos:
            benchmark_args.append("--no-eos")
        if args.output:
            benchmark_args.extend(["--out", args.output])
        try:
            _load_runtime_bundle_compat(
                lambda: dflash_benchmark_main(
                    benchmark_args,
                    prog="machboost bench-decode",
                ),
                DFlashDraftModelArgs,
            )
        except SystemExit as exc:
            return int(exc.code or 0)
        if args.validation_tokens > 0:
            prompts = _decode_validation_prompts(args)
            if prompts:
                validation = validate_decode_outputs(
                    resolution.model,
                    prompts,
                    draft_model=args.draft_model,
                    draft_quant=args.draft_quant,
                    verify_mode=args.verify_mode,
                    max_tokens=args.validation_tokens,
                )
                print(
                    "output validation: "
                    f"{validation['exact_matches']}/{validation['rows']} exact "
                    f"at {args.validation_tokens} token(s)",
                    file=error_stream,
                )
                validation_valid = validation["exact_matches"] == validation["rows"]
                if not validation_valid:
                    print(
                        "output validation failed: accelerated greedy tokens differ from native MLX; "
                        "treat the throughput result as non-equivalent",
                        file=error_stream,
                    )
                if args.output:
                    output_dir = Path(args.output).expanduser()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    validation_path = output_dir / "output_validation.json"
                    validation_path.write_text(
                        json.dumps(validation, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    results_path = output_dir / "results.json"
                    if results_path.is_file():
                        results = json.loads(results_path.read_text(encoding="utf-8"))
                        results["output_validation"] = validation
                        results_path.write_text(
                            json.dumps(results, indent=2) + "\n",
                            encoding="utf-8",
                        )
        return 0 if validation_valid else 1
    except (ImportError, OSError, ValueError) as exc:
        print(f"machboost bench-decode error: {exc}", file=error_stream)
        return 2


def _jsonl_row_count(path: str) -> int:
    count = 0
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"invalid JSONL object in {path} at line {line_number}")
            count += 1
    if count == 0:
        raise ValueError(f"prompt file is empty: {path}")
    return count


def _decode_validation_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt:
        return [{"id": "custom", "prompt": str(args.prompt)}]
    if not args.prompt_file:
        return []
    rows: list[dict[str, str]] = []
    with Path(args.prompt_file).expanduser().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            prompt = str(value.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(
                    f"JSONL prompt is empty in {args.prompt_file} at line {line_number}"
                )
            rows.append(
                {
                    "id": str(value.get("id") or f"row-{line_number}"),
                    "prompt": prompt,
                }
            )
            if args.limit and len(rows) >= args.limit:
                break
    return rows


def validate_decode_outputs(
    model: str,
    prompts: Sequence[dict[str, str]],
    *,
    draft_model: Optional[str],
    draft_quant: Optional[str],
    verify_mode: str,
    max_tokens: int,
) -> dict:
    import mlx.core as mx
    from mlx_lm import generate as native_generate
    from mlx_lm.sample_utils import make_sampler

    from .adapters.dflash import DFlashAccelerator

    accelerator = DFlashAccelerator.from_pretrained(
        model,
        draft_model=draft_model,
        draft_quant=draft_quant,
        verify_mode=verify_mode,
        lazy=True,
    )
    rows = []
    try:
        for item in prompts:
            prompt = accelerator.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            mx.clear_cache()
            native_text = native_generate(
                accelerator.model,
                accelerator.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
            mx.clear_cache()
            accelerated_text, stats = accelerator.generate(
                prompt,
                max_tokens=max_tokens,
            )
            native_tokens = list(accelerator.tokenizer.encode(native_text))
            accelerated_tokens = list(accelerator.tokenizer.encode(accelerated_text))
            common = next(
                (
                    index
                    for index, pair in enumerate(zip(native_tokens, accelerated_tokens))
                    if pair[0] != pair[1]
                ),
                min(len(native_tokens), len(accelerated_tokens)),
            )
            rows.append(
                {
                    "id": item["id"],
                    "exact": native_tokens == accelerated_tokens,
                    "native_tokens": len(native_tokens),
                    "accelerated_tokens": len(accelerated_tokens),
                    "common_prefix_tokens": common,
                    "first_difference": (
                        None
                        if native_tokens == accelerated_tokens
                        else {
                            "index": common,
                            "native_token": native_tokens[common]
                            if common < len(native_tokens)
                            else None,
                            "accelerated_token": accelerated_tokens[common]
                            if common < len(accelerated_tokens)
                            else None,
                        }
                    ),
                    "native_token_hash": _token_hash(native_tokens),
                    "accelerated_token_hash": _token_hash(accelerated_tokens),
                    "acceptance_ratio": stats.acceptance_ratio,
                }
            )
    finally:
        accelerator.close()
    exact_matches = sum(bool(row["exact"]) for row in rows)
    return {
        "schema_version": "machboost.decode-output-validation.v1",
        "model": model,
        "greedy": True,
        "max_tokens": max_tokens,
        "rows": len(rows),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(rows) if rows else None,
        "results": rows,
    }


def _token_hash(tokens: Sequence[int]) -> str:
    payload = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _context_benchmark_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise ValueError("use either --prompt or --prompt-file, not both")
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise ValueError("provide --prompt TEXT or --prompt-file PATH")


def print_context_benchmark(artifact: dict, *, stream=None) -> None:
    stream = stream or sys.stdout
    config = artifact["config"]
    summary = artifact["summary"]
    print(
        f"MachBoost context benchmark: {config['runs']} measured pair(s), "
        f"{config['warmups']} warmup pair(s), max {config['max_tokens']} tokens",
        file=stream,
    )
    print(
        f"same model: {config['model']} ({config['backend']}); "
        f"load={float(artifact.get('model_load_seconds') or 0.0):.3f}s",
        file=stream,
    )
    for row in artifact["rows"]:
        speedup = (
            f"{float(row['speedup']):.3f}x"
            if row["speedup"] is not None
            else "invalid"
        )
        print(
            f"  run {row['run']}: first={row['order'][0]} "
            f"native={row['native']['wall_seconds']:.3f}s "
            f"machboost={row['machboost']['wall_seconds']:.3f}s "
            f"speedup={speedup} exact={'yes' if row['output_match'] else 'NO'} "
            f"accepted={row['accepted_draft_tokens']}",
            file=stream,
        )
    print(
        f"summary: exact={summary['output_match_rate']:.1%} "
        f"engaged={summary['algorithm_engaged_rate']:.1%} "
        f"native={summary['median_native_wall_seconds']:.3f}s "
        f"machboost={summary['median_machboost_wall_seconds']:.3f}s "
        f"accepted={summary['median_accepted_draft_tokens']:.0f} "
        f"target-call-reduction={summary['median_target_call_reduction']:.1%}",
        file=stream,
    )
    if summary["valid"]:
        print(
            f"VALID same-model speedup: {summary['median_speedup']:.3f}x",
            file=stream,
        )
        if summary["algorithm_engaged_rate"] == 0:
            print("note: no draft tokens were accepted; the algorithm did not engage", file=stream)
    else:
        print(
            "INVALID: at least one token sequence differed; no aggregate speedup is claimed",
            file=stream,
        )


def print_latency_benchmark(artifact: dict, *, stream=None) -> None:
    stream = stream or sys.stdout
    config = artifact["config"]
    print(
        f"chat latency: {config['runs']} measured run(s), "
        f"{config['warmups']} warmup(s), max {config['max_tokens']} tokens",
        file=stream,
    )
    for engine, data in artifact["engines"].items():
        summary = data["summary"]
        print(
            f"{engine}: model={data['resolved_model']} backend={data['backend']}",
            file=stream,
        )
        if data.get("load_wall_seconds") is not None:
            print(
                f"  preload wall={float(data['load_wall_seconds']):.3f}s "
                f"model_load={float(data['model_load_seconds']):.3f}s "
                f"compile={float(data.get('compile_warmup_seconds') or 0.0):.3f}s",
                file=stream,
            )
        print(
            f"  median wall={summary['median_wall_seconds']:.3f}s "
            f"ttft={summary['median_client_ttft_seconds']:.3f}s "
            f"decode={summary['median_tokens_per_second']:.2f} tokens/s",
            file=stream,
        )
        for row in data["rows"]:
            print(
                f"  run {row['run']}: wall={row['wall_seconds']:.3f}s "
                f"ttft={float(row['client_ttft_seconds'] or 0.0):.3f}s "
                f"tokens={row['eval_count']} rate={row['tokens_per_second']:.2f}",
                file=stream,
            )
    comparison = artifact.get("comparison")
    if comparison:
        output_equal = comparison.get("median_output_equal")
        output_label = "n/a" if output_equal is None else "yes" if output_equal else "no"
        print(
            "comparison: "
            f"MachBoost wall={comparison['machboost_total_speedup_vs_ollama']:.3f}x "
            f"TTFT={comparison['machboost_ttft_speedup_vs_ollama']:.3f}x "
            f"output_equal={output_label} "
            "relative to Ollama",
            file=stream,
        )
    if config.get("draft_control_method") == "ollama_logprobs_parking":
        print(
            "control: Ollama logprobs park native MLX speculation; logprob materialization overhead is included",
            file=stream,
        )
    print("note: plain chat uses the native backend when no draft context is supplied", file=stream)


def run_ps(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    client = _automatic_host_pool(
        args,
        error_stream=error_stream,
        autostart=False,
    ) or MachBoostClient(args.endpoint, timeout=args.timeout)
    if not client.is_healthy():
        print("MachBoost server is not running.", file=output_stream)
        return 0
    try:
        models = client.ps()
    except MachBoostAPIError as exc:
        print(f"machboost ps error: {exc}", file=error_stream)
        return 2
    if args.json:
        print(json.dumps({"models": models}, indent=2), file=output_stream)
        return 0
    if not models:
        print("No resident models.", file=output_stream)
        return 0
    print(
        f"{'NAME':<38} {'HOST':<14} {'BACKEND':<8} {'REQ':>6} {'REP':>4} "
        f"{'ACTIVE':>6} {'QUEUE':>5} {'IDLE':>9} {'KEEP ALIVE':>12}",
        file=output_stream,
    )
    for model in models:
        keep_alive = (
            "forever"
            if model["keep_alive_seconds"] < 0
            else f"{model['keep_alive_seconds']:.0f}s"
        )
        scheduler = model.get("scheduler") or {}
        print(
            f"{model['model']:<38} {str(model.get('fabric_host_name') or 'local')[:14]:<14} "
            f"{model['backend']:<8} {model['requests']:>6} "
            f"{int(scheduler.get('replicas', 1)):>4} "
            f"{int(scheduler.get('active_requests', 0)):>6} "
            f"{int(scheduler.get('queued_requests', 0)):>5} "
            f"{model['idle_seconds']:>8.1f}s {keep_alive:>12}",
            file=output_stream,
        )
    return 0


def run_server_action(args: argparse.Namespace, action: str, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    client = MachBoostClient(args.endpoint, timeout=args.timeout)
    if not client.is_healthy():
        print("MachBoost server is not running.", file=error_stream)
        return 2
    try:
        if action == "show":
            result = client.show(args.model)
        elif action == "stop":
            result = client.stop(args.model)
        elif action == "shutdown":
            result = client.shutdown()
        else:
            raise ValueError(f"unknown server action: {action}")
    except MachBoostAPIError as exc:
        print(f"machboost {action} error: {exc}", file=error_stream)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2), file=output_stream)
    elif action == "show":
        print(json.dumps(result, indent=2), file=output_stream)
    elif action == "stop":
        print(f"unloaded {result.get('unloaded', 0)} model instance(s)", file=output_stream)
    else:
        print(f"server stopped; unloaded {result.get('unloaded', 0)} model instance(s)", file=output_stream)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MachBoost local inference acceleration utilities.")
    subcommands = parser.add_subparsers(dest="command")

    doctor = subcommands.add_parser("doctor", help="Inspect local optional backend dependencies.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    self_test = subcommands.add_parser("self-test", help="Run an in-memory exactness smoke test.")
    self_test.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    model_list = subcommands.add_parser("list", help="List cached native HF/MLX models.")
    model_list.add_argument(
        "--backend",
        choices=["all", "mlx", "hf", "mlx-vlm", "hf-vlm", "ollama-mlx"],
        default="all",
        help="Filter by backend.",
    )
    model_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    model_list.add_argument("--all", dest="show_all", action="store_true", help="Show unsupported cached repos too.")
    model_list.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        help="Hugging Face hub cache directory to inspect instead of the defaults. Can be repeated.",
    )

    native_run = subcommands.add_parser("run", help="Chat with a native model through the resident MachBoost server.")
    add_native_run_arguments(native_run)
    add_chat_route_arguments(native_run)

    chat = subcommands.add_parser("chat", help="Alias for resident native `machboost run`.")
    add_native_run_arguments(chat)
    add_chat_route_arguments(chat)

    complete = subcommands.add_parser("complete", help="Stream raw text or code completion from a resident model.")
    add_native_run_arguments(complete)
    complete.add_argument("prompt", nargs="?", help="Prompt text. Reads stdin when omitted.")
    complete.add_argument("--file", help="Read the completion prompt from a UTF-8 text file.")

    bench = subcommands.add_parser(
        "bench",
        help="Compare warm serving runtimes; this does not benchmark MachBoost acceleration.",
    )
    bench.add_argument("model", help="MachBoost model alias or repository.")
    bench.add_argument(
        "--engine",
        choices=["machboost", "ollama", "both"],
        default="both",
    )
    bench.add_argument("--ollama-model", help="Ollama model name; defaults to MODEL.")
    bench.add_argument("--ollama-endpoint", help="Ollama server URL.")
    bench.add_argument(
        "--prompt",
        default="Reply to this greeting naturally in one short sentence: hey",
    )
    bench.add_argument("--system", default="")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--max-tokens", type=int, default=32)
    bench.add_argument(
        "--draft-num-predict",
        type=int,
        default=None,
        help=(
            "Ollama MLX DFlash draft block size; 0 uses a logprobs-based "
            "no-speculation diagnostic that includes logprob overhead."
        ),
    )
    bench.add_argument(
        "--backend",
        choices=["auto", "mlx", "hf", "dflash", "ollama-mlx"],
        default="auto",
    )
    bench.add_argument("--keep-alive", default="5m")
    bench.add_argument("--json", action="store_true")
    add_server_connection_arguments(bench, include_autostart=True)

    context_bench = subcommands.add_parser(
        "bench-context",
        help="Benchmark MachBoost context acceleration against same-model native generation.",
    )
    context_bench.add_argument("model", help="Text model alias or repository.")
    context_bench.add_argument("--prompt", help="Raw completion prompt.")
    context_bench.add_argument("--prompt-file", help="Read the raw completion prompt from a UTF-8 file.")
    context_bench.add_argument(
        "--context",
        action="append",
        default=[],
        help="Context file or directory. Can be repeated.",
    )
    context_bench.add_argument(
        "--context-text",
        action="append",
        default=[],
        help="Inline draft context. Can be repeated.",
    )
    context_bench.add_argument("--max-context-chars", type=int, default=200_000)
    context_bench.add_argument("--runs", type=int, default=6, help="Even number of measured pairs.")
    context_bench.add_argument("--warmups", type=int, default=2)
    context_bench.add_argument("--max-tokens", type=int, default=64)
    context_bench.add_argument("--ngram", type=int, default=2)
    context_bench.add_argument("--max-draft-tokens", type=int, default=32)
    context_bench.add_argument("--candidate-limit", type=int, default=1)
    context_bench.add_argument("--reentry-probe-tokens", type=int, default=0)
    context_bench.add_argument("--backend", choices=["auto", "mlx", "hf"], default="auto")
    context_bench.add_argument("--strict", action="store_true", help="MLX: disable prompt-cache reuse for a slow exactness control.")
    context_bench.add_argument("--lazy", action="store_true")
    context_bench.add_argument("--device", default="auto")
    context_bench.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    context_bench.add_argument("--local-files-only", action="store_true")
    context_bench.add_argument("--json", action="store_true")
    context_bench.add_argument("--output", help="Write the JSON artifact to this path.")

    decode_bench = subcommands.add_parser(
        "bench-decode",
        help="Benchmark native MLX against target-verified DFlash decoding.",
    )
    decode_bench.add_argument("model", help="Supported model alias or repository.")
    decode_prompt = decode_bench.add_mutually_exclusive_group()
    decode_prompt.add_argument("--prompt", help="Single unique prompt to benchmark.")
    decode_prompt.add_argument(
        "--prompt-file",
        help="JSONL prompt suite with id, suite, and prompt fields.",
    )
    decode_bench.add_argument("--draft-model", help="DFlash draft repository override.")
    decode_bench.add_argument("--draft-quant", help="Draft quantization such as w4:gs64.")
    decode_bench.add_argument("--max-tokens", type=int, default=512)
    decode_bench.add_argument("--runs", type=int, default=3)
    decode_bench.add_argument(
        "--limit",
        type=int,
        help="Maximum JSONL prompts; defaults to every non-empty row.",
    )
    decode_bench.add_argument("--cooldown", type=int, default=1)
    decode_bench.add_argument(
        "--verify-mode",
        choices=["dflash", "adaptive", "ddtree", "off"],
        default="adaptive",
        help="Target verifier strategy; adaptive avoids costly full blocks when acceptance drops.",
    )
    decode_bench.add_argument(
        "--no-eos",
        action="store_true",
        help="Ignore EOS so every leg measures the requested decode length.",
    )
    decode_bench.add_argument(
        "--validation-tokens",
        type=int,
        default=128,
        help="Compare native and accelerated greedy outputs at this length; zero disables validation.",
    )
    decode_bench.add_argument("--output", help="DFlash benchmark artifact directory.")

    warm = subcommands.add_parser("warm", help="Preload a native model into resident memory.")
    add_native_run_arguments(warm)

    serve = subcommands.add_parser("serve", help="Start the resident MachBoost inference server.")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument(
        "--replicas",
        type=int,
        choices=range(1, MAX_REPLICAS + 1),
        default=DEFAULT_REPLICAS,
        metavar=f"1-{MAX_REPLICAS}",
        help="Independent accelerator instances per text model. Each replica uses additional memory.",
    )
    serve.add_argument(
        "--max-queue",
        type=int,
        default=DEFAULT_MAX_QUEUE,
        help="Maximum waiting requests per loaded model; zero rejects whenever all replicas are busy.",
    )
    serve.add_argument(
        "--queue-timeout",
        type=float,
        default=DEFAULT_QUEUE_TIMEOUT,
        help="Seconds a request may wait for a replica; negative waits indefinitely.",
    )
    serve.add_argument(
        "--require-auth",
        action="store_true",
        help="Require MACHBOOST_API_TOKEN as a bearer token. Enabled automatically for non-loopback hosts.",
    )
    serve.add_argument(
        "--team",
        action="store_true",
        help="Enable scoped employee keys, fair admission, traces, and evaluations.",
    )
    serve.add_argument(
        "--team-db",
        help="Team database path. Defaults to ~/.machboost/team.sqlite3.",
    )
    serve.add_argument(
        "--trace-mode",
        choices=("off", "metadata", "redacted", "full"),
        help="Override the persisted team trace privacy mode.",
    )
    serve.add_argument(
        "--trace-retention-days",
        type=int,
        help="Retain traces for this many days; zero keeps them until the disk cap.",
    )
    serve.add_argument(
        "--trace-max-mb",
        type=int,
        help="Maximum retained trace payload size in MiB.",
    )

    pull = subcommands.add_parser("pull", help="Download a Hugging Face or MLX model into the local cache.")
    pull.add_argument("model")
    pull.add_argument("--revision", default=None)
    add_server_connection_arguments(pull, include_autostart=True)

    create = subcommands.add_parser("create", help="Create or update a local model alias.")
    create.add_argument("model", help="Alias name, for example company-coder:latest.")
    create.add_argument("--from", dest="source", required=True, help="Source model alias or HF/MLX repository.")
    create.add_argument("--system", default="", help="Default system instruction.")
    create.add_argument("--template", default="", help="Optional Ollama-style prompt template.")
    create.add_argument(
        "--option",
        action="append",
        default=[],
        help="Default generation option as key=value; repeat as needed.",
    )
    add_server_connection_arguments(create, include_autostart=True)

    copy_model = subcommands.add_parser("cp", help="Copy a local model alias.")
    copy_model.add_argument("source")
    copy_model.add_argument("destination")
    add_server_connection_arguments(copy_model, include_autostart=True)

    remove_model = subcommands.add_parser(
        "rm",
        help="Remove a local model alias, or add --weights to delete its downloaded cache.",
    )
    remove_model.add_argument("model")
    remove_model.add_argument(
        "--weights",
        action="store_true",
        help="Unload the model and permanently delete its managed cached weights.",
    )
    add_server_connection_arguments(remove_model, include_autostart=True)

    ps = subcommands.add_parser("ps", help="List models currently loaded in MachBoost memory.")
    ps.add_argument("--json", action="store_true")
    add_server_connection_arguments(ps)

    show = subcommands.add_parser("show", help="Show a resident model's runtime state.")
    show.add_argument("model")
    show.add_argument("--json", action="store_true")
    add_server_connection_arguments(show)

    stop = subcommands.add_parser("stop", help="Unload a model from resident memory.")
    stop.add_argument("model", nargs="?", help="Model to unload. Omit to unload every model.")
    add_server_connection_arguments(stop)

    shutdown = subcommands.add_parser("shutdown", help="Stop the resident server and unload every model.")
    add_server_connection_arguments(shutdown)

    ollama = subcommands.add_parser("ollama", help="Ollama-compatible wrapper commands.")
    ollama_subcommands = ollama.add_subparsers(dest="ollama_command")
    ollama_run = ollama_subcommands.add_parser("run", help="Pull if needed, then chat with a model.")
    add_ollama_run_arguments(ollama_run)

    connect = subcommands.add_parser("connect", help="Save and use another MachBoost Mac.")
    connect.add_argument("endpoint", help="Remote MachBoost URL or host:port.")
    connect.add_argument("--name", help="Short name for this connection.")
    connect.add_argument("--timeout", type=float, default=10.0)

    connections = subcommands.add_parser("connections", help="List saved MachBoost devices.")
    connections.add_argument("--json", action="store_true")
    connections.add_argument(
        "--probe",
        action="store_true",
        help="Measure live latency, load, and model readiness.",
    )
    connections.add_argument("--model", help="Model to use when scoring automatic routes.")
    connections.add_argument("--timeout", type=float, default=10.0)

    use_connection = subcommands.add_parser("use", help="Use local or a saved MachBoost device.")
    use_connection.add_argument("name", help="Connection name, local, or auto.")

    disconnect = subcommands.add_parser("disconnect", help="Forget a saved MachBoost device.")
    disconnect.add_argument("name", help="Connection name to forget.")

    launch = subcommands.add_parser("launch", help="Connect MachBoost to another AI application.")
    launch.add_argument("integration", choices=["claude-desktop", "claude-app"])
    launch.add_argument("--endpoint", help="MachBoost server root URL. Defaults to this Mac.")
    launch.add_argument("--connection", help="Saved MachBoost connection name to use.")
    launch.add_argument("--api-key", help="Bearer key for an explicit remote endpoint.")
    launch.add_argument(
        "--model",
        action="append",
        help="Local model to advertise in Claude Desktop; repeat for up to five models.",
    )
    launch.add_argument("--restore", action="store_true", help="Restore Claude Desktop's previous profile.")
    launch.add_argument("--no-restart", action="store_true", help="Do not restart Claude Desktop.")
    launch.add_argument("--yes", action="store_true", help="Restart Claude Desktop without prompting.")
    launch.add_argument("--timeout", type=float, default=30.0)
    launch.add_argument("--json", action="store_true")

    mcp = subcommands.add_parser("mcp", help="Manage Model Context Protocol connectors.")
    mcp_subcommands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_list = mcp_subcommands.add_parser("list", help="List configured MCP connectors.")
    mcp_list.add_argument("--json", action="store_true")
    add_server_connection_arguments(mcp_list)
    mcp_add = mcp_subcommands.add_parser("add", help="Add an MCP connector.")
    mcp_add.add_argument("name")
    mcp_source = mcp_add.add_mutually_exclusive_group(required=True)
    mcp_source.add_argument("--url", help="Streamable HTTP MCP endpoint.")
    mcp_source.add_argument(
        "--command",
        dest="mcp_executable",
        help="Local stdio MCP command.",
    )
    mcp_add.add_argument("--arg", action="append", default=[], help="Command argument; repeat as needed.")
    mcp_add.add_argument("--env", action="append", default=[], help="Environment NAME=VALUE; repeat as needed.")
    mcp_add.add_argument("--header", action="append", default=[], help="HTTP header NAME=VALUE; repeat as needed.")
    add_server_connection_arguments(mcp_add)
    mcp_remove = mcp_subcommands.add_parser("remove", help="Remove an MCP connector.")
    mcp_remove.add_argument("server_id")
    add_server_connection_arguments(mcp_remove)
    mcp_test = mcp_subcommands.add_parser("test", help="Connect and list an MCP server's tools.")
    mcp_test.add_argument("server_id")
    mcp_test.add_argument("--json", action="store_true")
    add_server_connection_arguments(mcp_test)
    mcp_search = mcp_subcommands.add_parser("search", help="Search tools from enabled connectors.")
    mcp_search.add_argument("query")
    mcp_search.add_argument("--limit", type=int, default=8)
    add_server_connection_arguments(mcp_search)
    mcp_call = mcp_subcommands.add_parser("call", help="Call one MCP tool.")
    mcp_call.add_argument("server_id")
    mcp_call.add_argument("tool")
    mcp_call.add_argument("--arguments", default="{}", help="Tool arguments as a JSON object.")
    mcp_call.add_argument("--json", action="store_true")
    add_server_connection_arguments(mcp_call)

    skill = subcommands.add_parser("skill", help="Manage reusable chat instructions.")
    skill_subcommands = skill.add_subparsers(dest="skill_command", required=True)
    skill_list = skill_subcommands.add_parser("list", help="List reusable instructions.")
    skill_list.add_argument("--json", action="store_true")
    add_server_connection_arguments(skill_list)
    skill_add = skill_subcommands.add_parser("add", help="Add reusable instructions.")
    skill_add.add_argument("name")
    skill_add.add_argument("--instructions", required=True)
    add_server_connection_arguments(skill_add)
    skill_remove = skill_subcommands.add_parser("remove", help="Remove reusable instructions.")
    skill_remove.add_argument("skill_id")
    add_server_connection_arguments(skill_remove)

    subcommands.add_parser("version", help="Print the installed MachBoost version.")
    return parser


def add_native_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        help="Hugging Face or MLX model name/path, for example Qwen/Qwen2.5-3B-Instruct.",
    )
    parser.add_argument(
        "--backend",
        choices=[
            "auto",
            "mlx",
            "hf",
            "mlx-vlm",
            "hf-vlm",
            "dflash",
            "ollama-mlx",
        ],
        default="auto",
        help="Model backend. DFlash enables target-verified block-diffusion decoding on supported MLX text models.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Local file or directory to use as MachBoost draft context.",
    )
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--ctx",
        "--num-ctx",
        type=int,
        default=None,
        help="Context-window limit passed to compatible backends.",
    )
    parser.add_argument(
        "--think",
        nargs="?",
        const="medium",
        choices=["low", "medium", "high", "xhigh"],
        help="Enable reasoning, optionally selecting Muse Glimmer reasoning strength.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Print streamed reasoning separately from the visible answer.",
    )
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument(
        "--draft-model",
        help="DFlash draft repository override; supported targets resolve a tested draft automatically.",
    )
    parser.add_argument(
        "--draft-quant",
        help="DFlash draft quantization, for example w4 or w4:gs64.",
    )
    parser.add_argument(
        "--verify-mode",
        choices=["dflash", "adaptive", "ddtree", "off"],
        default="adaptive",
        help="DFlash target verification strategy.",
    )
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument(
        "--reentry-probe-tokens",
        type=int,
        default=0,
        help="MLX experimental: generate this many native seed tokens before context re-entry.",
    )
    parser.add_argument("--system", default="", help="Optional system message.")
    parser.add_argument(
        "--show-stats",
        "--verbose",
        dest="show_stats",
        action="store_true",
        help="Print load, first-token, prompt, decode, and MachBoost timings.",
    )
    parser.add_argument("--no-boost", action="store_true", help="Run the serial baseline path.")
    parser.add_argument("--strict", action="store_true", help="MLX only: disable prompt cache for strict evidence mode.")
    parser.add_argument("--lazy", action="store_true", help="MLX only: use lazy model loading.")
    parser.add_argument("--device", default="auto", help="HF only: auto, cpu, mps, or cuda. Auto prefers MPS on Mac.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto", help="HF only: model dtype. Auto uses float16 on MPS.")
    parser.add_argument("--local-files-only", action="store_true", help="HF only: do not download missing model files.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path, URL, data URL, or base64 payload. Can be repeated.",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video path to sample into chronological change-aware frames. Can be repeated.",
    )
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-change-threshold", type=float, default=0.08)
    parser.add_argument("--video-max-frames", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-vision-cache", action="store_true", help="Re-run the vision encoder for every request.")
    parser.add_argument("--vision-cache-size", type=int, default=20, help="Projected vision feature LRU size.")
    parser.add_argument(
        "--cold-vision",
        choices=["off", "adaptive", "fast", "balanced", "quality"],
        default="off",
        help="Experimental first-view visual budget. Adaptive chooses from image detail and question type.",
    )
    parser.add_argument(
        "--vision-max-edge",
        type=int,
        help="Override the cold-vision maximum image edge in pixels; images are never upscaled.",
    )
    parser.add_argument(
        "--vision-tokens",
        choices=VISION_TOKEN_REQUEST_MODES,
        default="off",
        help="Qwen3-VL post-fusion visual token policy. Auto classifies the prompt and image shape.",
    )
    parser.add_argument(
        "--vision-token-ratio",
        type=float,
        default=0.35,
        help="Requested visual token retention ratio for post-fusion compression.",
    )
    parser.add_argument(
        "--vision-token-layer",
        type=int,
        help="Override the language layer after which visual tokens are compressed.",
    )
    parser.add_argument(
        "--vision-token-bucket",
        type=int,
        help="Round the retained visual-token target to this bucket size; zero disables bucketing.",
    )
    parser.add_argument(
        "--vision-calibration",
        help="Path to a machboost.vision_calibration.v1 JSON policy artifact.",
    )
    parser.add_argument("--direct", action="store_true", help="Load in this process instead of using the resident server.")
    parser.add_argument(
        "--keep-alive",
        default="5m",
        help="Idle resident lifetime, for example 5m, 1h, or forever.",
    )
    add_server_connection_arguments(parser, include_autostart=True)


def add_chat_route_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route",
        choices=["local_only", "local_first", "external_first", "external_only"],
        default="local_only",
        help="Choose local inference, a paid API, or a transient-failure fallback order.",
    )
    parser.add_argument("--provider", help="Configured paid-provider ID.")
    parser.add_argument("--provider-model", help="Paid API model name used instead of the local model ID.")


def add_server_connection_arguments(parser: argparse.ArgumentParser, *, include_autostart: bool = False) -> None:
    parser.add_argument("--endpoint", default=None, help="MachBoost server URL. Defaults to MACHBOOST_HOST or localhost:11435.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Server request timeout in seconds.")
    if include_autostart:
        parser.add_argument("--no-autostart", action="store_true", help="Require an already-running MachBoost server.")


def add_ollama_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Ollama model name, for example qwen2.5:3b or llama3.2.")
    parser.add_argument("--endpoint", default=None, help="Ollama endpoint. Defaults to OLLAMA_HOST or localhost.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds.")
    parser.add_argument("--no-pull", action="store_true", help="Fail if the model is missing instead of pulling it.")
    parser.add_argument("--system", default="", help="Optional system message for the chat.")
    parser.add_argument("--ctx", type=int, default=None, help="Ollama num_ctx option.")
    parser.add_argument("--num-predict", type=int, default=None, help="Ollama num_predict option.")
    parser.add_argument("--temperature", type=float, default=None, help="Ollama temperature option.")


def torch_dtype_from_name(name: str):
    if name == "auto":
        return None
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install Hugging Face support with `pip install machboost[hf]`.") from exc
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        data = doctor_data()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print_human_doctor(data)
        return 0
    if args.command == "self-test":
        data = self_test_data()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print_human_self_test(data)
        return 0 if data["ok"] else 1
    if args.command == "list":
        data = model_list_data(
            backend=args.backend,
            cache_dirs=args.cache_dir or None,
            include_unsupported=args.show_all,
        )
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print_human_model_list(data)
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "connect":
        return run_connect(args)
    if args.command == "connections":
        return run_connections(args)
    if args.command == "use":
        return run_use_connection(args)
    if args.command == "disconnect":
        return run_disconnect(args)
    if args.command == "launch":
        return run_launch(args)
    if args.command == "mcp":
        return run_mcp(args)
    if args.command == "skill":
        return run_skill(args)
    if args.command == "run":
        return run_native_chat(args) if args.direct else run_resident_chat(args)
    if args.command == "chat":
        return run_native_chat(args) if args.direct else run_resident_chat(args)
    if args.command == "complete":
        return run_resident_completion(args)
    if args.command == "bench":
        return run_latency_bench(args)
    if args.command == "bench-context":
        return run_context_bench(args)
    if args.command == "bench-decode":
        return run_decode_bench(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "warm":
        return run_warm(args)
    if args.command == "pull":
        return run_pull(args)
    if args.command in {"create", "cp", "rm"}:
        return run_model_alias_action(args)
    if args.command == "ps":
        return run_ps(args)
    if args.command == "show":
        return run_server_action(args, "show")
    if args.command == "stop":
        return run_server_action(args, "stop")
    if args.command == "shutdown":
        return run_server_action(args, "shutdown")
    if args.command == "ollama" and args.ollama_command == "run":
        return run_ollama_chat(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
