#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machboost.accelerator import render_chat_prompt, service_stop_token_ids
from machboost.adapters import MLXCausalLMService
from machboost.workspace import WorkspaceStore


DEFAULT_QUESTIONS = (
    "Where does the HTTP server route chat requests?",
    "How are resident model replicas scheduled?",
    "Where is request cancellation implemented?",
    "How does the MLX adapter cache prompt state?",
    "Which code indexes repository files and symbols?",
    "How are ignored files excluded from repository search?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native MLX full prefill with MachBoost repository-prefix reuse "
            "using the same model, prompts, and generated tokens."
        )
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=48_000)
    parser.add_argument("--prompt-cache-size", type=int, default=8)
    parser.add_argument(
        "--prompt-cache-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def timed_generate(
    service: MLXCausalLMService,
    prompt_tokens: tuple[int, ...],
    *,
    max_tokens: int,
    stop_tokens: tuple[int, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    tokens = service.generate_tokens(
        prompt_tokens,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
    )
    wall = time.perf_counter() - started
    metrics = service.last_native_metrics
    return {
        "wall_seconds": wall,
        "tokens": list(tokens),
        "token_count": len(tokens),
        "output_sha256": hashlib.sha256(
            ",".join(str(token) for token in tokens).encode("ascii")
        ).hexdigest(),
        "prompt_tokens": int(metrics.get("prompt_tokens") or len(prompt_tokens)),
        "prompt_eval_tokens": int(
            metrics.get("prompt_eval_tokens") or len(prompt_tokens)
        ),
        "cached_prompt_tokens": int(metrics.get("cached_prompt_tokens") or 0),
        "time_to_first_token_seconds": metrics.get("time_to_first_token_seconds"),
        "prompt_eval_seconds": metrics.get("prompt_eval_seconds"),
        "generation_seconds": metrics.get("generation_seconds"),
        "generation_tokens_per_second": metrics.get(
            "generation_tokens_per_second"
        ),
    }


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def command_output(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("machboost", "mlx", "mlx-lm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def system_metadata() -> dict[str, Any]:
    memory_bytes = command_output("sysctl", "-n", "hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": command_output("sysctl", "-n", "machdep.cpu.brand_string")
        or platform.processor(),
        "cpu_count": command_output("sysctl", "-n", "hw.ncpu"),
        "memory_bytes": int(memory_bytes) if memory_bytes else None,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }


def main() -> int:
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    workspace_path = Path(args.workspace).expanduser().resolve()

    print(f"Loading {args.model} once...", flush=True)
    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    if hasattr(model, "eval"):
        model.eval()
    baseline = MLXCausalLMService(
        model,
        tokenizer,
        native_prompt_cache_size=0,
    )
    boosted = MLXCausalLMService(
        model,
        tokenizer,
        native_prompt_cache_size=args.prompt_cache_size,
        native_prompt_cache_bytes=args.prompt_cache_bytes,
    )
    stop_tokens = service_stop_token_ids(boosted)

    with tempfile.TemporaryDirectory(prefix="machboost-workspace-bench-") as temporary:
        store = WorkspaceStore(Path(temporary) / "indexes")
        workspace = store.register(workspace_path)
        report = store.index(workspace.id)
        questions = [
            DEFAULT_QUESTIONS[index % len(DEFAULT_QUESTIONS)]
            for index in range(args.runs)
        ]
        prompts = []
        retrieval_rows = []
        for question in questions:
            retrieval = store.query(
                workspace.id,
                question,
                top_k=args.top_k,
                max_chars=args.max_context_chars,
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Use the repository evidence for repository-specific claims "
                        "and cite path:start-end.\n\n"
                        f"{retrieval.context}"
                    ),
                },
                {"role": "user", "content": question},
            ]
            prompt = render_chat_prompt(boosted, messages)
            prompts.append(boosted.encode(prompt))
            retrieval_rows.append(
                {
                    "question": question,
                    "citations": [
                        f"{hit.path}:{hit.start_line}-{hit.end_line}"
                        for hit in retrieval.hits
                    ],
                }
            )

        print(
            f"Indexed {report.workspace.file_count} files into "
            f"{report.workspace.chunk_count} chunks.",
            flush=True,
        )
        print(
            f"Prompt sizes: {min(map(len, prompts))}-"
            f"{max(map(len, prompts))} tokens.",
            flush=True,
        )

        prime_prompt = prompts[0]
        timed_generate(
            boosted,
            prime_prompt,
            max_tokens=args.max_tokens,
            stop_tokens=stop_tokens,
        )
        rows = []
        for index, prompt_tokens in enumerate(prompts):
            if index % 2 == 0:
                native = timed_generate(
                    baseline,
                    prompt_tokens,
                    max_tokens=args.max_tokens,
                    stop_tokens=stop_tokens,
                )
                cached = timed_generate(
                    boosted,
                    prompt_tokens,
                    max_tokens=args.max_tokens,
                    stop_tokens=stop_tokens,
                )
                order = "native-first"
            else:
                cached = timed_generate(
                    boosted,
                    prompt_tokens,
                    max_tokens=args.max_tokens,
                    stop_tokens=stop_tokens,
                )
                native = timed_generate(
                    baseline,
                    prompt_tokens,
                    max_tokens=args.max_tokens,
                    stop_tokens=stop_tokens,
                )
                order = "machboost-first"
            exact = native["tokens"] == cached["tokens"]
            speedup = (
                native["wall_seconds"] / cached["wall_seconds"]
                if cached["wall_seconds"] > 0
                else 0.0
            )
            row = {
                "run": index + 1,
                "order": order,
                "question": retrieval_rows[index]["question"],
                "repeats_prime": index == 0,
                "citations": retrieval_rows[index]["citations"],
                "exact_tokens": exact,
                "speedup": speedup,
                "native": native,
                "machboost": cached,
            }
            rows.append(row)
            print(
                f"run={index + 1} order={order:15s} "
                f"native={native['wall_seconds']:.3f}s "
                f"machboost={cached['wall_seconds']:.3f}s "
                f"speedup={speedup:.3f}x exact={exact} "
                f"cached={cached['cached_prompt_tokens']}/"
                f"{cached['prompt_tokens']}",
                flush=True,
            )

    exact_rows = [row for row in rows if row["exact_tokens"]]
    valid = len(exact_rows) == len(rows) and any(
        row["machboost"]["cached_prompt_tokens"] > 0 for row in rows
    )
    different_question_rows = [
        row for row in exact_rows if not row["repeats_prime"]
    ]
    summary = {
        "schema": "machboost.workspace-prefix-benchmark.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "system": system_metadata(),
        "packages": package_versions(),
        "settings": {
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "max_context_chars": args.max_context_chars,
            "prompt_cache_size": args.prompt_cache_size,
            "prompt_cache_bytes": args.prompt_cache_bytes,
            "generation": "greedy",
            "prime_question": questions[0],
        },
        "workspace": workspace_path.name,
        "workspace_revision": report.workspace.revision,
        "runs": len(rows),
        "valid": valid,
        "exact_token_rows": len(exact_rows),
        "median_native_wall_seconds": median(
            [row["native"]["wall_seconds"] for row in rows]
        ),
        "median_machboost_wall_seconds": median(
            [row["machboost"]["wall_seconds"] for row in rows]
        ),
        "median_speedup": median([row["speedup"] for row in exact_rows]),
        "different_question_rows": len(different_question_rows),
        "median_different_question_speedup": median(
            [row["speedup"] for row in different_question_rows]
        ),
        "prime_speedup": rows[0]["speedup"] if rows else 0.0,
        "median_cached_prompt_tokens": median(
            [row["machboost"]["cached_prompt_tokens"] for row in rows]
        ),
        "median_prompt_tokens": median(
            [row["machboost"]["prompt_tokens"] for row in rows]
        ),
        "rows": rows,
    }
    print(
        "\n"
        f"VALID={summary['valid']} "
        f"exact={summary['exact_token_rows']}/{summary['runs']} "
        f"median_native={summary['median_native_wall_seconds']:.3f}s "
        f"median_machboost={summary['median_machboost_wall_seconds']:.3f}s "
        f"median_speedup={summary['median_speedup']:.3f}x "
        f"median_cached={summary['median_cached_prompt_tokens']:.0f}/"
        f"{summary['median_prompt_tokens']:.0f} prompt tokens"
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
