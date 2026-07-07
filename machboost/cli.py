from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from . import __version__, machboost
from .adapters.ollama import OllamaHTTPAdapter, OllamaHTTPError


@dataclass(frozen=True)
class PackageStatus:
    available: bool
    version: Optional[str] = None


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


def package_status(module_name: str, version_attr: str = "__version__") -> PackageStatus:
    if importlib.util.find_spec(module_name) is None:
        return PackageStatus(False)
    try:
        module = __import__(module_name)
        version = getattr(module, version_attr, None)
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
            "mlx_lm": asdict(package_status("mlx_lm")),
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

    chat = subcommands.add_parser("chat", help="Ollama-compatible interactive chat shortcut.")
    add_ollama_run_arguments(chat)

    ollama = subcommands.add_parser("ollama", help="Ollama-compatible wrapper commands.")
    ollama_subcommands = ollama.add_subparsers(dest="ollama_command")
    ollama_run = ollama_subcommands.add_parser("run", help="Pull if needed, then chat with a model.")
    add_ollama_run_arguments(ollama_run)

    subcommands.add_parser("version", help="Print the installed MachBoost version.")
    return parser


def add_ollama_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Ollama model name, for example qwen2.5:3b or llama3.2.")
    parser.add_argument("--endpoint", default=None, help="Ollama endpoint. Defaults to OLLAMA_HOST or localhost.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds.")
    parser.add_argument("--no-pull", action="store_true", help="Fail if the model is missing instead of pulling it.")
    parser.add_argument("--system", default="", help="Optional system message for the chat.")
    parser.add_argument("--ctx", type=int, default=None, help="Ollama num_ctx option.")
    parser.add_argument("--num-predict", type=int, default=None, help="Ollama num_predict option.")
    parser.add_argument("--temperature", type=float, default=None, help="Ollama temperature option.")


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
    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "chat":
        return run_ollama_chat(args)
    if args.command == "ollama" and args.ollama_command == "run":
        return run_ollama_chat(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
