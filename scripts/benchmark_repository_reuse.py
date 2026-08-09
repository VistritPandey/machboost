#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA = "machboost.repository-reuse-benchmark.v1"
DEFAULT_SYSTEM = (
    "You are a senior engineer. Answer from repository evidence, cite files, "
    "and do not invent files."
)


class BenchmarkError(RuntimeError):
    pass


def request_json(
    server: str,
    path: str,
    *,
    token: Optional[str],
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(server.rstrip("/") + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return dict(json.load(response))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    except (OSError, URLError, ValueError) as exc:
        raise BenchmarkError(f"{path} failed: {exc}") from exc


def workspace_metadata(
    server: str, workspace_id: str, *, token: Optional[str]
) -> dict[str, Any]:
    payload = request_json(server, "/api/workspaces", token=token)
    for workspace in payload.get("workspaces") or ():
        if workspace.get("id") == workspace_id:
            return {
                key: workspace.get(key)
                for key in (
                    "id",
                    "name",
                    "revision",
                    "file_count",
                    "chunk_count",
                    "total_bytes",
                )
            }
    raise BenchmarkError(f"workspace is not registered: {workspace_id}")


def chat(
    args: argparse.Namespace,
    prompt: str,
    *,
    cache: bool,
    namespace: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ],
        "workspace_id": args.workspace_id,
        "workspace_top_k": args.top_k,
        "workspace_max_chars": args.max_chars,
        "stream": False,
        "keep_alive": args.keep_alive,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
            "workspace_prefix_cache": cache,
            "_prompt_cache_namespace": namespace,
            "prompt_cache_size": args.cache_entries,
            "prompt_cache_bytes": args.cache_bytes,
        },
        "machboost": {"memory": "off"},
    }
    started = time.perf_counter()
    response = request_json(
        args.server,
        "/api/chat",
        token=args.token,
        payload=payload,
        timeout=args.timeout,
    )
    wall_seconds = time.perf_counter() - started
    text = str((response.get("message") or {}).get("content") or "")
    details = response.get("machboost") or {}
    stats = details.get("stats") or {}
    return {
        "wall_seconds": wall_seconds,
        "ttft_seconds": details.get("time_to_first_token_seconds"),
        "prompt_tokens": int(response.get("prompt_eval_count") or 0),
        "prompt_eval_seconds": float(response.get("prompt_eval_duration") or 0)
        / 1_000_000_000.0,
        "completion_tokens": int(response.get("eval_count") or 0),
        "decode_seconds": float(response.get("eval_duration") or 0)
        / 1_000_000_000.0,
        "cached_prompt_tokens": int(stats.get("cached_prompt_tokens") or 0),
        "evaluated_prompt_tokens": int(
            stats.get("prompt_eval_tokens")
            or stats.get("prompt_tokens")
            or response.get("prompt_eval_count")
            or 0
        ),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_chars": len(text),
        "rubric_hits": sum(
            term.casefold() in text.casefold() for term in args.rubric_term
        ),
        "rubric_total": len(args.rubric_term),
        "citations": (
            (details.get("workspace") or {}).get("citations") or []
            if args.include_citations
            else None
        ),
    }


def run_pair(
    args: argparse.Namespace, *, round_number: int, baseline_first: bool
) -> dict[str, Any]:
    namespace = f"repo-bench:{uuid.uuid4().hex}"

    def baseline() -> dict[str, Any]:
        return chat(
            args,
            args.target,
            cache=False,
            namespace=namespace + ":off",
            max_tokens=args.max_tokens,
        )

    def boosted() -> dict[str, Any]:
        chat(
            args,
            args.primer,
            cache=True,
            namespace=namespace,
            max_tokens=args.primer_tokens,
        )
        return chat(
            args,
            args.target,
            cache=True,
            namespace=namespace,
            max_tokens=args.max_tokens,
        )

    if baseline_first:
        baseline_result = baseline()
        boosted_result = boosted()
    else:
        boosted_result = boosted()
        baseline_result = baseline()
    return {
        "round": round_number,
        "first": "baseline" if baseline_first else "machboost",
        "output_equal": (
            baseline_result["output_sha256"] == boosted_result["output_sha256"]
        ),
        "wall_speedup": _ratio(
            baseline_result["wall_seconds"], boosted_result["wall_seconds"]
        ),
        "prefill_speedup": _ratio(
            baseline_result["prompt_eval_seconds"],
            boosted_result["prompt_eval_seconds"],
        ),
        "baseline": baseline_result,
        "machboost": boosted_result,
    }


def summarize(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "measured_rounds": len(rounds),
        "exact_output_pairs": sum(bool(row["output_equal"]) for row in rounds),
        "median_wall_speedup": statistics.median(
            row["wall_speedup"] for row in rounds
        ),
        "median_prefill_speedup": statistics.median(
            row["prefill_speedup"] for row in rounds
        ),
        "median_baseline_wall_seconds": statistics.median(
            row["baseline"]["wall_seconds"] for row in rounds
        ),
        "median_machboost_wall_seconds": statistics.median(
            row["machboost"]["wall_seconds"] for row in rounds
        ),
        "median_baseline_prefill_seconds": statistics.median(
            row["baseline"]["prompt_eval_seconds"] for row in rounds
        ),
        "median_machboost_prefill_seconds": statistics.median(
            row["machboost"]["prompt_eval_seconds"] for row in rounds
        ),
        "median_cached_prompt_tokens": statistics.median(
            row["machboost"]["cached_prompt_tokens"] for row in rounds
        ),
        "median_baseline_rubric_hits": statistics.median(
            row["baseline"]["rubric_hits"] for row in rounds
        ),
        "median_machboost_rubric_hits": statistics.median(
            row["machboost"]["rubric_hits"] for row in rounds
        ),
        "rubric_total": rounds[0]["baseline"]["rubric_total"],
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    workspace = workspace_metadata(
        args.server, args.workspace_id, token=args.token
    )
    rounds = [
        run_pair(args, round_number=index, baseline_first=index % 2 == 1)
        for index in range(1, args.runs + 1)
    ]
    return {
        "schema": SCHEMA,
        "model": args.model,
        "workspace": workspace,
        "summary": summarize(rounds),
        "rounds": rounds,
        "method": {
            "independent_threads": True,
            "greedy_decoding": True,
            "same_resident_model": True,
            "semantic_memory": False,
            "exact_response_cache": False,
            "alternating_order": True,
            "private_outputs_omitted": True,
        },
        "note": (
            "This benchmark measures reuse of a shared repository/system prompt prefix "
            "for a new standalone target request. It does not claim faster decoding."
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Benchmark cross-thread repository prefix reuse."
    )
    result.add_argument("--server", default="http://127.0.0.1:11435")
    result.add_argument("--token", default=os.environ.get("MACHBOOST_API_TOKEN"))
    result.add_argument("--workspace-id", required=True)
    result.add_argument("--model", default="qwen2.5:7b")
    result.add_argument("--system", default=DEFAULT_SYSTEM)
    result.add_argument("--primer", required=True)
    result.add_argument("--target", required=True)
    result.add_argument("--rubric-term", action="append", default=[])
    result.add_argument("--runs", type=int, default=3)
    result.add_argument("--primer-tokens", type=int, default=48)
    result.add_argument("--max-tokens", type=int, default=256)
    result.add_argument("--top-k", type=int, default=10)
    result.add_argument("--max-chars", type=int, default=18_000)
    result.add_argument("--cache-entries", type=int, default=8)
    result.add_argument("--cache-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    result.add_argument("--keep-alive", default="forever")
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--include-citations", action="store_true")
    result.add_argument("--output", type=Path)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.max_tokens < 1 or args.primer_tokens < 1:
        raise SystemExit("token limits must be positive")
    try:
        result = run_benchmark(args)
    except BenchmarkError as exc:
        print(f"repository reuse benchmark failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["summary"]["exact_output_pairs"] == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
