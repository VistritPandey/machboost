#!/usr/bin/env python3
"""Measure concurrent request latency and aggregate throughput on a MachBoost server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from typing import Any, Callable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machboost.client import MachBoostAPIError, MachBoostClient


SCHEMA_VERSION = "machboost.concurrency_benchmark.v1"
DEFAULT_PROMPT = (
    "Request nonce {nonce}. Write a numbered list of eight practical ways to "
    "reduce local inference latency. Use complete sentences."
)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: Sequence[dict[str, Any]], round_seconds: Sequence[float]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("ok")]
    failed = [row for row in rows if not row.get("ok")]
    elapsed = sum(float(value) for value in round_seconds)
    latencies = [float(row["wall_seconds"]) for row in successful]
    queue_waits = [float(row["queue_wait_seconds"]) for row in successful]
    ttfts = [
        float(row["time_to_first_token_seconds"])
        for row in successful
        if row.get("time_to_first_token_seconds") is not None
    ]
    tokens = sum(int(row.get("eval_count") or 0) for row in successful)
    return {
        "requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "overloaded_requests": sum(
            1 for row in failed if row.get("error_code") == "queue_full"
        ),
        "measured_round_seconds": elapsed,
        "requests_per_second": len(successful) / elapsed if elapsed > 0 else 0.0,
        "aggregate_tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
        "generated_tokens": tokens,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": percentile(latencies, 0.95),
        "median_queue_wait_seconds": statistics.median(queue_waits) if queue_waits else None,
        "p95_queue_wait_seconds": percentile(queue_waits, 0.95),
        "median_time_to_first_token_seconds": statistics.median(ttfts) if ttfts else None,
        "p95_time_to_first_token_seconds": percentile(ttfts, 0.95),
        "replicas_used": sorted(
            {int(row["replica"]) for row in successful if row.get("replica") is not None}
        ),
        "unique_output_hashes": len(
            {str(row["output_sha256"]) for row in successful if row.get("output_sha256")}
        ),
    }


def run_request(
    args: argparse.Namespace,
    *,
    round_index: int,
    request_index: int,
    start: threading.Event,
    client_factory: Callable[[], MachBoostClient],
) -> dict[str, Any]:
    nonce = f"r{round_index:03d}-q{request_index:04d}"
    prompt = args.prompt.replace("{nonce}", nonce)
    affinity_key = None
    if args.affinity_prefix:
        affinity_key = f"{args.affinity_prefix}:{request_index % args.clients}"
    start.wait()
    client = client_factory()
    started = time.perf_counter()
    try:
        options = {
            "backend": args.backend,
            "num_predict": args.max_tokens,
            "temperature": args.temperature,
        }
        if args.mode == "chat":
            response = client.chat(
                args.model,
                [{"role": "user", "content": prompt}],
                options=options,
                keep_alive=args.keep_alive,
                affinity_key=affinity_key,
                queue_timeout=args.queue_timeout,
                stream=False,
            )
            output = str((response.get("message") or {}).get("content") or "")
        else:
            response = client.generate(
                args.model,
                prompt,
                options=options,
                keep_alive=args.keep_alive,
                affinity_key=affinity_key,
                queue_timeout=args.queue_timeout,
                stream=False,
            )
            output = str(response.get("response") or "")
        wall = time.perf_counter() - started
        metrics = dict(response.get("machboost") or {})
        scheduler = dict(metrics.get("scheduler") or {})
        return {
            "round": round_index,
            "request": request_index,
            "nonce": nonce,
            "ok": True,
            "wall_seconds": wall,
            "time_to_first_token_seconds": metrics.get(
                "time_to_first_token_seconds"
            ),
            "eval_count": int(response.get("eval_count") or 0),
            "queue_wait_seconds": float(
                scheduler.get("queue_wait_seconds") or 0.0
            ),
            "replica": scheduler.get("replica"),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "output_chars": len(output),
        }
    except MachBoostAPIError as exc:
        return {
            "round": round_index,
            "request": request_index,
            "nonce": nonce,
            "ok": False,
            "wall_seconds": time.perf_counter() - started,
            "error": str(exc),
            "error_status": exc.status,
            "error_code": exc.code,
        }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.endpoint.rstrip("/")
    client_factory = lambda: MachBoostClient(endpoint, timeout=args.timeout)
    preload_started = time.perf_counter()
    preload = client_factory().load(
        args.model,
        options={"backend": args.backend},
        keep_alive=args.keep_alive,
        warmup=True,
    )
    preload_wall = time.perf_counter() - preload_started

    for index in range(args.warmups):
        start = threading.Event()
        start.set()
        run_request(
            args,
            round_index=-1,
            request_index=index,
            start=start,
            client_factory=client_factory,
        )

    rows: list[dict[str, Any]] = []
    round_seconds: list[float] = []
    for round_index in range(args.rounds):
        start = threading.Event()
        with ThreadPoolExecutor(max_workers=args.clients) as executor:
            futures = [
                executor.submit(
                    run_request,
                    args,
                    round_index=round_index,
                    request_index=request_index,
                    start=start,
                    client_factory=client_factory,
                )
                for request_index in range(args.requests)
            ]
            round_started = time.perf_counter()
            start.set()
            round_rows = [future.result() for future in futures]
        round_seconds.append(time.perf_counter() - round_started)
        rows.extend(round_rows)

    models = client_factory().ps()
    resolved = str((preload.get("instance") or {}).get("model") or args.model)
    instance = next(
        (model for model in models if model.get("model") == resolved),
        None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {
            "endpoint": endpoint,
            "model": args.model,
            "resolved_model": resolved,
            "backend": args.backend,
            "mode": args.mode,
            "clients": args.clients,
            "requests_per_round": args.requests,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "queue_timeout_seconds": args.queue_timeout,
            "affinity_prefix": args.affinity_prefix,
        },
        "preload": {
            "wall_seconds": preload_wall,
            "model_load_seconds": float(preload.get("load_duration_seconds") or 0.0),
            "warmup_seconds": float(preload.get("warmup_duration_seconds") or 0.0),
        },
        "round_seconds": round_seconds,
        "summary": summarize(rows, round_seconds),
        "scheduler": (instance or {}).get("scheduler"),
        "rows": rows,
    }


def print_summary(artifact: dict[str, Any]) -> None:
    config = artifact["config"]
    summary = artifact["summary"]
    scheduler = artifact.get("scheduler") or {}
    print(
        f"{config['resolved_model']} | replicas={scheduler.get('replicas', '?')} "
        f"clients={config['clients']} requests={summary['requests']}"
    )
    print(
        f"throughput: {summary['requests_per_second']:.3f} req/s, "
        f"{summary['aggregate_tokens_per_second']:.2f} generated tok/s"
    )
    print(
        f"latency: p50={_seconds(summary['median_latency_seconds'])}, "
        f"p95={_seconds(summary['p95_latency_seconds'])}, "
        f"TTFT p50={_seconds(summary['median_time_to_first_token_seconds'])}"
    )
    print(
        f"queue: p50={_seconds(summary['median_queue_wait_seconds'])}, "
        f"p95={_seconds(summary['p95_queue_wait_seconds'])}, "
        f"failed={summary['failed_requests']}, overloaded={summary['overloaded_requests']}"
    )


def _seconds(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--backend", choices=["auto", "mlx", "hf"], default="auto")
    parser.add_argument("--mode", choices=["chat", "generate"], default="chat")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--queue-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--keep-alive", default="forever")
    parser.add_argument("--affinity-prefix")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("clients", "requests", "rounds"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmups < 0 or args.max_tokens < 1:
        raise SystemExit("--warmups must be nonnegative and --max-tokens must be positive")
    artifact = benchmark(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(artifact, indent=2))
    else:
        print_summary(artifact)
        if args.output is not None:
            print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
