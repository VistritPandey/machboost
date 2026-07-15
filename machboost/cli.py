from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, machboost
from .accelerator import Accelerator
from .adapters.ollama import OllamaHTTPAdapter, OllamaHTTPError
from .client import MachBoostAPIError, MachBoostClient, ensure_server
from .models import alias_rows, resolve_model
from .server import DEFAULT_HOST, DEFAULT_PORT, serve as serve_runtime

DEFAULT_CHAT_SYSTEM = "Answer directly and concisely. Do not reveal hidden reasoning."


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

    turns: list[dict[str, object]] = []
    active_images = list(args.image or ())
    print(f"machboost run: {args.model}", file=output_stream)
    print("Type /bye, /exit, or /quit to leave. Type /clear to reset chat history.", file=output_stream)
    print("Use /image PATH, /images, or /clear-images to manage visual inputs.", file=output_stream)

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
        if command == "/images":
            print("\n".join(active_images) if active_images else "no images attached", file=output_stream)
            continue
        if command == "/clear-images":
            active_images = []
            print("images cleared", file=output_stream)
            continue

        turns.append({"role": "user", "content": user_text})
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
        try:
            kwargs = {"max_tokens": args.max_tokens, "on_text": emit}
            if getattr(accelerator, "supports_vision", False):
                kwargs.update(
                    use_vision_cache=not args.no_vision_cache,
                    temperature=args.temperature,
                    cold_vision_mode=args.cold_vision,
                    cold_vision_max_edge=args.vision_max_edge,
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
        if streamed:
            print("", flush=True, file=output_stream)
        else:
            print(response, flush=True, file=output_stream)
        if args.show_stats:
            tokens_per_second = stats.generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
            if getattr(stats, "backend", "") == "mlx-vlm":
                print(
                    "stats: "
                    f"elapsed={elapsed_s:.2f}s "
                    f"tokens_per_second={tokens_per_second:.2f} "
                    f"prompt_tps={stats.prompt_tokens_per_second:.2f} "
                    f"vision_cache={'hit' if stats.visual_cache_hit else 'miss' if stats.visual_cache_miss else 'off'}",
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
    }


def connect_resident(args: argparse.Namespace, *, error_stream=None) -> MachBoostClient:
    error_stream = error_stream or sys.stderr
    if args.no_autostart:
        client = MachBoostClient(args.endpoint, timeout=args.timeout)
        if not client.is_healthy():
            raise MachBoostAPIError("MachBoost server is not running; start it with `machboost serve`")
        return client
    client, started = ensure_server(args.endpoint, timeout=min(30.0, args.timeout))
    client.timeout = args.timeout
    if started:
        print(f"started resident MachBoost server at {client.endpoint}", file=error_stream)
    return client


def run_resident_chat(
    args: argparse.Namespace,
    *,
    input_func=input,
    output_stream=None,
    error_stream=None,
) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client = connect_resident(args, error_stream=error_stream)
    except MachBoostAPIError as exc:
        print(f"machboost server error: {exc}", file=error_stream)
        return 2

    turns: list[dict[str, str]] = []
    active_images = list(args.image or ())
    print(f"machboost run: {args.model}", file=output_stream)
    print(f"server: {client.endpoint} (model stays warm until stop or shutdown)", file=output_stream)
    print("Type /bye, /exit, or /quit to leave. Type /clear to reset chat history.", file=output_stream)
    print("Use /image PATH, /images, or /clear-images to manage visual inputs.", file=output_stream)

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
        if command == "/images":
            print("\n".join(active_images) if active_images else "no images attached", file=output_stream)
            continue
        if command == "/clear-images":
            active_images = []
            print("images cleared", file=output_stream)
            continue

        turns.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": args.system or DEFAULT_CHAT_SYSTEM}, *turns]
        response_parts: list[str] = []
        final_row: dict = {}
        started = time.perf_counter()
        try:
            request_options = {
                "options": native_server_options(args),
                "keep_alive": args.keep_alive,
                "stream": True,
            }
            if active_images:
                request_options["images"] = active_images
            rows = client.chat(args.model, messages, **request_options)
            for row in rows:
                message = row.get("message") or {}
                chunk = str(message.get("content") or "")
                if chunk:
                    print(chunk, end="", flush=True, file=output_stream)
                    response_parts.append(chunk)
                if row.get("done"):
                    final_row = row
        except KeyboardInterrupt:
            print("", file=output_stream)
            turns.pop()
            return 130
        except MachBoostAPIError as exc:
            turns.pop()
            print(f"\ngeneration error: {exc}", file=error_stream)
            return 2

        print("", flush=True, file=output_stream)
        response = "".join(response_parts).strip()
        if args.show_stats:
            print_resident_stats(final_row, time.perf_counter() - started, stream=output_stream)
        turns.append({"role": "assistant", "content": response})


def run_resident_completion(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        prompt = completion_prompt(args)
        if args.direct:
            accelerator = load_native_accelerator(args, stream=error_stream)
            kwargs = {
                "max_tokens": args.max_tokens,
                "on_text": lambda text: print(text, end="", flush=True, file=output_stream),
            }
            if args.image and not getattr(accelerator, "supports_vision", False):
                raise ValueError("attached images require a vision model")
            if getattr(accelerator, "supports_vision", False):
                kwargs.update(
                    images=args.image or None,
                    use_vision_cache=not args.no_vision_cache,
                    temperature=args.temperature,
                    cold_vision_mode=args.cold_vision,
                    cold_vision_max_edge=args.vision_max_edge,
                )
            text, stats = accelerator.generate(prompt, **kwargs)
            print("", file=output_stream)
            if args.show_stats:
                print(f"stats: generated={stats.generated_tokens} accepted={stats.accepted_draft_tokens}", file=output_stream)
            return 0

        client = connect_resident(args, error_stream=error_stream)
        started = time.perf_counter()
        request_options = {
            "options": native_server_options(args),
            "keep_alive": args.keep_alive,
            "stream": True,
        }
        if args.image:
            request_options["images"] = args.image
        rows = client.generate(args.model, prompt, **request_options)
        final_row: dict = {}
        for row in rows:
            chunk = str(row.get("response") or "")
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
    rate = generated / elapsed_s if elapsed_s > 0 else 0.0
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
    print(
        "stats: "
        f"elapsed={elapsed_s:.2f}s "
        f"tokens_per_second={rate:.2f} "
        f"accepted={int(stats.get('accepted_draft_tokens') or 0)} "
        f"target_calls={int(stats.get('target_calls') or 0)}/{int(stats.get('baseline_target_calls') or 0)} "
        f"backend={metrics.get('backend', 'unknown')}"
        f"{cache_state}",
        file=stream,
    )


def run_serve(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    print(f"MachBoost server listening on http://{args.host}:{args.port}", file=output_stream)
    print("Models remain loaded until `machboost stop`, `machboost shutdown`, or process exit.", file=output_stream)
    try:
        serve_runtime(args.host, args.port)
    except KeyboardInterrupt:
        print("\nMachBoost server stopped.", file=error_stream)
    except OSError as exc:
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


def run_warm(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    try:
        client = connect_resident(args, error_stream=error_stream)
        result = client.load(
            args.model,
            options=native_server_options(args),
            keep_alive=args.keep_alive,
        )
        instance = result["instance"]
        print(
            f"loaded {instance['model']} on {instance['backend']} in "
            f"{result['load_duration_seconds']:.2f}s; keep_alive={args.keep_alive}",
            file=output_stream,
        )
        return 0
    except MachBoostAPIError as exc:
        print(f"machboost warm error: {exc}", file=error_stream)
        return 2


def run_ps(args: argparse.Namespace, *, output_stream=None, error_stream=None) -> int:
    output_stream = output_stream or sys.stdout
    error_stream = error_stream or sys.stderr
    client = MachBoostClient(args.endpoint, timeout=args.timeout)
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
    print(f"{'NAME':<48} {'BACKEND':<8} {'REQUESTS':>8} {'IDLE':>9} {'KEEP ALIVE':>12}", file=output_stream)
    for model in models:
        keep_alive = "forever" if model["keep_alive_seconds"] < 0 else f"{model['keep_alive_seconds']:.0f}s"
        print(
            f"{model['model']:<48} {model['backend']:<8} {model['requests']:>8} "
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
        choices=["all", "mlx", "hf", "mlx-vlm", "hf-vlm"],
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

    chat = subcommands.add_parser("chat", help="Alias for resident native `machboost run`.")
    add_native_run_arguments(chat)

    complete = subcommands.add_parser("complete", help="Stream raw text or code completion from a resident model.")
    add_native_run_arguments(complete)
    complete.add_argument("prompt", nargs="?", help="Prompt text. Reads stdin when omitted.")
    complete.add_argument("--file", help="Read the completion prompt from a UTF-8 text file.")

    warm = subcommands.add_parser("warm", help="Preload a native model into resident memory.")
    add_native_run_arguments(warm)

    serve = subcommands.add_parser("serve", help="Start the resident MachBoost inference server.")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)

    pull = subcommands.add_parser("pull", help="Download a Hugging Face or MLX model into the local cache.")
    pull.add_argument("model")
    pull.add_argument("--revision", default=None)
    add_server_connection_arguments(pull, include_autostart=True)

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

    subcommands.add_parser("version", help="Print the installed MachBoost version.")
    return parser


def add_native_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        help="Hugging Face or MLX model name/path, for example Qwen/Qwen2.5-3B-Instruct.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "hf", "mlx-vlm", "hf-vlm"],
        default="auto",
        help="Model backend. Auto selects text or vision MLX/HF adapters from the model architecture.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Local file or directory to use as MachBoost draft context.",
    )
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument(
        "--reentry-probe-tokens",
        type=int,
        default=0,
        help="MLX experimental: generate this many native seed tokens before context re-entry.",
    )
    parser.add_argument("--system", default="", help="Optional system message.")
    parser.add_argument("--show-stats", action="store_true", help="Print MachBoost draft/verify stats after replies.")
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
    parser.add_argument("--direct", action="store_true", help="Load in this process instead of using the resident server.")
    parser.add_argument("--keep-alive", default="forever", help="Resident lifetime, for example forever, 10m, or 1h.")
    add_server_connection_arguments(parser, include_autostart=True)


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
    if args.command == "run":
        return run_native_chat(args) if args.direct else run_resident_chat(args)
    if args.command == "chat":
        return run_native_chat(args) if args.direct else run_resident_chat(args)
    if args.command == "complete":
        return run_resident_completion(args)
    if args.command == "serve":
        return run_serve(args)
    if args.command == "warm":
        return run_warm(args)
    if args.command == "pull":
        return run_pull(args)
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
