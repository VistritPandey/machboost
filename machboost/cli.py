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
    if model_backend == "mlx":
        return model_backend, has_snapshot(model_dir), "MLX model cache" if has_snapshot(model_dir) else "no snapshot"

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
    print(f"estimated speedup: {data['estimated_speedup']:.2f}x")


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
            print(f"  {model['backend']:<3} {model['name']} ({model['reason']})")
    elif data.get("hidden_unsupported_count", 0):
        print(f"unsupported cached repos: {data['hidden_unsupported_count']} hidden; use --all to show them")

    print("remote examples:")
    for example in data["examples"]:
        print(f"  {example['command']}")


def select_native_backend(model: str, backend: str) -> str:
    if backend != "auto":
        return backend
    normalized = model.lower()
    if normalized.startswith("mlx-community/") or "mlx" in normalized:
        return "mlx"
    return "hf"


def render_chat_prompt(system: str, turns: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    if system:
        lines.append(f"System: {system.strip()}")
    for turn in turns:
        role = turn["role"].strip().capitalize()
        lines.append(f"{role}: {turn['content']}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_native_accelerator(args: argparse.Namespace, *, stream=None) -> Accelerator:
    stream = stream or sys.stderr
    backend = select_native_backend(args.model, args.backend)
    context_paths = args.context or None
    print(f"loading {args.model!r} with native {backend} backend...", file=stream)
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
            args.model,
            lazy=args.lazy,
            cache_enabled=not args.strict,
            **common,
        )
    if backend == "hf":
        return Accelerator.from_huggingface(
            args.model,
            device=args.device,
            local_files_only=args.local_files_only,
            torch_dtype=torch_dtype_from_name(args.dtype),
            **common,
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
            print("try passing --backend mlx or --backend hf explicitly", file=error_stream)
        return 2

    turns: list[dict[str, str]] = []
    print(f"machboost run: {args.model}", file=output_stream)
    print("Type /bye, /exit, or /quit to leave. Type /clear to reset chat history.", file=output_stream)

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

        turns.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": args.system or DEFAULT_CHAT_SYSTEM}]
        messages.extend(turns)
        streamed = False

        def emit(chunk: str) -> None:
            nonlocal streamed
            if not chunk:
                return
            streamed = True
            print(chunk, end="", flush=True, file=output_stream)

        started = time.perf_counter()
        try:
            response, stats = accelerator.generate_chat(messages, max_tokens=args.max_tokens, on_text=emit)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MachBoost local inference acceleration utilities.")
    subcommands = parser.add_subparsers(dest="command")

    doctor = subcommands.add_parser("doctor", help="Inspect local optional backend dependencies.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    self_test = subcommands.add_parser("self-test", help="Run an in-memory exactness smoke test.")
    self_test.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    model_list = subcommands.add_parser("list", help="List cached native HF/MLX models.")
    model_list.add_argument("--backend", choices=["all", "mlx", "hf"], default="all", help="Filter by backend.")
    model_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    model_list.add_argument("--all", dest="show_all", action="store_true", help="Show unsupported cached repos too.")
    model_list.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        help="Hugging Face hub cache directory to inspect instead of the defaults. Can be repeated.",
    )

    native_run = subcommands.add_parser("run", help="Run a native MachBoost HF/MLX model interactively.")
    add_native_run_arguments(native_run)

    chat = subcommands.add_parser("chat", help="Ollama-compatible interactive chat shortcut.")
    add_ollama_run_arguments(chat)

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
        choices=["auto", "mlx", "hf"],
        default="auto",
        help="Model backend. Auto chooses MLX for mlx-community models and HF otherwise.",
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
        return run_native_chat(args)
    if args.command == "chat":
        return run_ollama_chat(args)
    if args.command == "ollama" and args.ollama_command == "run":
        return run_ollama_chat(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
