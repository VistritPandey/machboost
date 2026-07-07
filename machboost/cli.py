from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from . import __version__, machboost


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MachBoost local inference acceleration utilities.")
    subcommands = parser.add_subparsers(dest="command")

    doctor = subcommands.add_parser("doctor", help="Inspect local optional backend dependencies.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    self_test = subcommands.add_parser("self-test", help="Run an in-memory exactness smoke test.")
    self_test.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    subcommands.add_parser("version", help="Print the installed MachBoost version.")
    return parser


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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
