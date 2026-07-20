from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import statistics
import time
from typing import Any, Callable, Mapping, Sequence


CONTEXT_BENCH_SCHEMA = "machboost.context_benchmark.v1"


def benchmark_context_acceleration(
    accelerator,
    prompt: str,
    *,
    model: str,
    backend: str,
    context_fingerprint: str,
    context_chars: int,
    runs: int = 6,
    warmups: int = 2,
    max_tokens: int = 64,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if runs < 2 or runs % 2:
        raise ValueError("runs must be an even number of at least 2")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if not prompt:
        raise ValueError("prompt cannot be empty")

    for round_index in range(warmups):
        _run_pair(
            accelerator,
            prompt,
            max_tokens=max_tokens,
            native_first=round_index % 2 == 0,
            run=0,
            clock=clock,
        )

    rows = []
    for run_index in range(runs):
        rows.append(
            _run_pair(
                accelerator,
                prompt,
                max_tokens=max_tokens,
                native_first=run_index % 2 == 0,
                run=run_index + 1,
                clock=clock,
            )
        )

    exact_rows = [row for row in rows if row["output_match"]]
    all_exact = len(exact_rows) == len(rows)
    engaged_rows = [row for row in rows if row["accepted_draft_tokens"] > 0]
    diagnostic_speedups = [float(row["diagnostic_speedup"]) for row in rows]
    summary = {
        "runs": len(rows),
        "valid": all_exact,
        "output_match_rate": len(exact_rows) / len(rows),
        "algorithm_engaged_rate": len(engaged_rows) / len(rows),
        "median_native_wall_seconds": _median(
            row["native"]["wall_seconds"] for row in rows
        ),
        "median_machboost_wall_seconds": _median(
            row["machboost"]["wall_seconds"] for row in rows
        ),
        "median_speedup": _median(diagnostic_speedups) if all_exact else None,
        "diagnostic_median_speedup": _median(diagnostic_speedups),
        "median_accepted_draft_tokens": _median(
            row["accepted_draft_tokens"] for row in rows
        ),
        "median_target_call_reduction": _median(
            row["target_call_reduction"] for row in rows
        ),
    }
    notes = [
        "Native and MachBoost paths use the same loaded model instance and tokenization.",
        "Measured pairs alternate execution order with equal native-first and MachBoost-first counts.",
        "A speedup is valid only when every accelerated token sequence equals its native pair.",
    ]
    if not all_exact:
        notes.append(
            "At least one output mismatch invalidated the aggregate speedup; diagnostic timing is retained only for debugging."
        )
    if not engaged_rows:
        notes.append(
            "No measured row accepted draft tokens, so this workload did not exercise MachBoost acceleration."
        )
    return {
        "schema_version": CONTEXT_BENCH_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "backend": backend,
            "prompt_sha256": _text_hash(prompt),
            "prompt_chars": len(prompt),
            "context_sha256": context_fingerprint,
            "context_chars": context_chars,
            "runs": runs,
            "warmups": warmups,
            "max_tokens": max_tokens,
            "execution_order": "balanced_alternating_pairs",
        },
        "summary": summary,
        "rows": rows,
        "notes": notes,
    }


def context_fingerprint(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _run_pair(
    accelerator,
    prompt: str,
    *,
    max_tokens: int,
    native_first: bool,
    run: int,
    clock: Callable[[], float],
) -> dict[str, Any]:
    order = ("native", "machboost") if native_first else ("machboost", "native")
    measured: dict[str, dict[str, Any]] = {}
    for mode in order:
        accelerator.boost_enabled = mode == "machboost"
        started = clock()
        result = accelerator.generate_result(prompt, max_tokens=max_tokens)
        elapsed = max(0.0, clock() - started)
        measured[mode] = _measurement(result, elapsed)

    native_tokens = tuple(measured["native"].pop("_tokens"))
    boosted_tokens = tuple(measured["machboost"].pop("_tokens"))
    output_match = native_tokens == boosted_tokens
    mismatch_index = _first_mismatch(native_tokens, boosted_tokens)
    native_calls = int(measured["native"]["target_calls"])
    boosted_calls = int(measured["machboost"]["target_calls"])
    speedup = _ratio(
        measured["native"]["wall_seconds"],
        measured["machboost"]["wall_seconds"],
    )
    return {
        "run": run,
        "order": list(order),
        "output_match": output_match,
        "first_mismatch_token_index": mismatch_index,
        "diagnostic_speedup": speedup,
        "speedup": speedup if output_match else None,
        "accepted_draft_tokens": int(
            measured["machboost"]["accepted_draft_tokens"]
        ),
        "target_call_reduction": (
            1.0 - boosted_calls / native_calls if native_calls > 0 else 0.0
        ),
        "native": measured["native"],
        "machboost": measured["machboost"],
    }


def _measurement(result, elapsed: float) -> dict[str, Any]:
    tokens = tuple(int(token) for token in result.tokens)
    stats = result.stats
    return {
        "wall_seconds": elapsed,
        "generated_tokens": len(tokens),
        "tokens_per_second": _ratio(len(tokens), elapsed),
        "token_sha256": _token_hash(tokens),
        "output_preview": result.text[:160],
        "target_calls": int(stats.target_calls),
        "verify_calls": int(stats.verify_calls),
        "accepted_draft_tokens": int(stats.accepted_draft_tokens),
        "accepted_draft_spans": int(stats.accepted_draft_spans),
        "_tokens": tokens,
    }


def _first_mismatch(left: Sequence[int], right: Sequence[int]):
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _token_hash(tokens: Sequence[int]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _median(values) -> float:
    data = [float(value) for value in values]
    return float(statistics.median(data)) if data else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0
