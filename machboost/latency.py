from __future__ import annotations

from datetime import datetime, timezone
import statistics
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from .adapters.ollama import OllamaHTTPAdapter
from .client import MachBoostClient


LATENCY_SCHEMA = "machboost.chat_latency.v1"


def benchmark_chat_latency(
    model: str,
    *,
    prompt: str,
    system: str,
    runs: int = 3,
    warmups: int = 1,
    max_tokens: int = 32,
    engine: str = "both",
    backend: str = "auto",
    keep_alive: str = "5m",
    machboost_client: Optional[MachBoostClient] = None,
    ollama_adapter: Optional[OllamaHTTPAdapter] = None,
    ollama_model: Optional[str] = None,
    ollama_endpoint: Optional[str] = None,
    timeout: float = 300.0,
    draft_num_predict: Optional[int] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if engine not in {"machboost", "ollama", "both"}:
        raise ValueError(f"unsupported benchmark engine: {engine}")
    if draft_num_predict is not None and draft_num_predict < 0:
        raise ValueError("draft_num_predict cannot be negative")

    nonces = [f"machboost-latency-{index + 1}" for index in range(warmups + runs)]
    artifact: dict[str, Any] = {
        "schema_version": LATENCY_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "ollama_model": ollama_model or model,
            "prompt": prompt,
            "system": system,
            "runs": runs,
            "warmups": warmups,
            "max_tokens": max_tokens,
            "backend": backend,
            "keep_alive": keep_alive,
            "draft_num_predict": draft_num_predict,
            "unique_prompt_nonce": True,
            "execution_order": (
                "alternating_by_round" if engine == "both" else "single_engine"
            ),
            "comparison_kind": (
                "same_engine_gateway_overhead"
                if backend == "ollama-mlx" and engine == "both"
                else "backend_comparison"
            ),
        },
        "engines": {},
        "notes": [
            "Unique system-message nonces prevent exact repeated-prompt cache hits.",
            "Two-engine runs alternate which engine executes first in each round.",
            "Without draft context, MachBoost delegates text generation to the native backend.",
        ],
    }
    if backend == "ollama-mlx":
        artifact["notes"].append(
            "Both paths use the same installed Ollama MLX model; the comparison measures MachBoost gateway overhead."
        )
    else:
        artifact["notes"].append(
            "Ollama and MLX conversions may use different quantization formats and model files."
        )

    if engine == "both":
        if machboost_client is None:
            raise ValueError("machboost_client is required for MachBoost latency runs")
        artifact["engines"].update(
            benchmark_interleaved_chat(
                machboost_client,
                ollama_adapter
                or OllamaHTTPAdapter(
                    ollama_model or model,
                    endpoint=ollama_endpoint,
                    timeout=timeout,
                    keep_alive=keep_alive,
                ),
                model,
                prompt=prompt,
                system=system,
                nonces=nonces,
                warmups=warmups,
                max_tokens=max_tokens,
                backend=backend,
                keep_alive=keep_alive,
                draft_num_predict=draft_num_predict,
                clock=clock,
            )
        )

    if engine == "machboost":
        if machboost_client is None:
            raise ValueError("machboost_client is required for MachBoost latency runs")
        artifact["engines"]["machboost"] = benchmark_machboost_chat(
            machboost_client,
            model,
            prompt=prompt,
            system=system,
            nonces=nonces,
            warmups=warmups,
            max_tokens=max_tokens,
            backend=backend,
            keep_alive=keep_alive,
            draft_num_predict=draft_num_predict,
            clock=clock,
        )

    if engine == "ollama":
        adapter = ollama_adapter or OllamaHTTPAdapter(
            ollama_model or model,
            endpoint=ollama_endpoint,
            timeout=timeout,
            keep_alive=keep_alive,
        )
        artifact["engines"]["ollama"] = benchmark_ollama_chat(
            adapter,
            prompt=prompt,
            system=system,
            nonces=nonces,
            warmups=warmups,
            max_tokens=max_tokens,
            keep_alive=keep_alive,
            draft_num_predict=draft_num_predict,
            clock=clock,
        )

    machboost = artifact["engines"].get("machboost")
    ollama = artifact["engines"].get("ollama")
    if machboost and ollama:
        mach_summary = machboost["summary"]
        ollama_summary = ollama["summary"]
        artifact["comparison"] = {
            "machboost_total_speedup_vs_ollama": safe_ratio(
                ollama_summary["median_wall_seconds"],
                mach_summary["median_wall_seconds"],
            ),
            "machboost_ttft_speedup_vs_ollama": safe_ratio(
                ollama_summary["median_client_ttft_seconds"],
                mach_summary["median_client_ttft_seconds"],
            ),
            "median_output_equal": normalized_outputs(machboost["rows"])
            == normalized_outputs(ollama["rows"]),
            "machboost_gateway_overhead_percent": (
                100.0
                * (
                    safe_ratio(
                        mach_summary["median_wall_seconds"],
                        ollama_summary["median_wall_seconds"],
                    )
                    - 1.0
                )
                if backend == "ollama-mlx"
                else None
            ),
        }
    return artifact


def benchmark_interleaved_chat(
    client: MachBoostClient,
    adapter: OllamaHTTPAdapter,
    model: str,
    *,
    prompt: str,
    system: str,
    nonces: Sequence[str],
    warmups: int,
    max_tokens: int,
    backend: str,
    keep_alive: str,
    draft_num_predict: Optional[int],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, dict[str, Any]]:
    options = generation_options(max_tokens, backend, draft_num_predict)
    machboost = preload_machboost(
        client,
        model,
        options=options,
        keep_alive=keep_alive,
        clock=clock,
    )
    machboost_rows = []
    ollama_rows = []
    for index, nonce in enumerate(nonces):
        messages = benchmark_messages(system, prompt, nonce)
        run = index - warmups + 1
        engines = ("machboost", "ollama") if index % 2 == 0 else ("ollama", "machboost")
        for current in engines:
            if current == "machboost":
                row = measure_machboost_chat(
                    client,
                    model,
                    messages,
                    run=run,
                    options=options,
                    keep_alive=keep_alive,
                    clock=clock,
                )
                if index >= warmups:
                    machboost_rows.append(row)
            else:
                row = measure_ollama_chat(
                    adapter,
                    messages,
                    run=run,
                    max_tokens=max_tokens,
                    keep_alive=keep_alive,
                    draft_num_predict=draft_num_predict,
                    clock=clock,
                )
                if index >= warmups:
                    ollama_rows.append(row)

    machboost["rows"] = machboost_rows
    machboost["summary"] = summarize_latency(machboost_rows)
    return {
        "machboost": machboost,
        "ollama": {
            "requested_model": adapter.model,
            "resolved_model": adapter.model,
            "backend": "ollama",
            "model_load_seconds": median(row["load_seconds"] for row in ollama_rows),
            "load_wall_seconds": None,
            "rows": ollama_rows,
            "summary": summarize_latency(ollama_rows),
        },
    }


def benchmark_machboost_chat(
    client: MachBoostClient,
    model: str,
    *,
    prompt: str,
    system: str,
    nonces: Sequence[str],
    warmups: int,
    max_tokens: int,
    backend: str,
    keep_alive: str,
    draft_num_predict: Optional[int],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    options = generation_options(max_tokens, backend, draft_num_predict)
    result = preload_machboost(
        client,
        model,
        options=options,
        keep_alive=keep_alive,
        clock=clock,
    )
    rows = []
    for index, nonce in enumerate(nonces):
        if index >= warmups:
            rows.append(
                measure_machboost_chat(
                    client,
                    model,
                    benchmark_messages(system, prompt, nonce),
                    run=index - warmups + 1,
                    options=options,
                    keep_alive=keep_alive,
                    clock=clock,
                )
            )
        else:
            measure_machboost_chat(
                client,
                model,
                benchmark_messages(system, prompt, nonce),
                run=0,
                options=options,
                keep_alive=keep_alive,
                clock=clock,
            )
    result["rows"] = rows
    result["summary"] = summarize_latency(rows)
    return result


def benchmark_ollama_chat(
    adapter: OllamaHTTPAdapter,
    *,
    prompt: str,
    system: str,
    nonces: Sequence[str],
    warmups: int,
    max_tokens: int,
    keep_alive: str,
    draft_num_predict: Optional[int],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    rows = []
    for index, nonce in enumerate(nonces):
        if index >= warmups:
            rows.append(
                measure_ollama_chat(
                    adapter,
                    benchmark_messages(system, prompt, nonce),
                    run=index - warmups + 1,
                    max_tokens=max_tokens,
                    keep_alive=keep_alive,
                    draft_num_predict=draft_num_predict,
                    clock=clock,
                )
            )
        else:
            measure_ollama_chat(
                adapter,
                benchmark_messages(system, prompt, nonce),
                run=0,
                max_tokens=max_tokens,
                keep_alive=keep_alive,
                draft_num_predict=draft_num_predict,
                clock=clock,
            )
    return {
        "requested_model": adapter.model,
        "resolved_model": adapter.model,
        "backend": "ollama",
        "model_load_seconds": median(row["load_seconds"] for row in rows),
        "load_wall_seconds": None,
        "rows": rows,
        "summary": summarize_latency(rows),
    }


def generation_options(
    max_tokens: int,
    backend: str,
    draft_num_predict: Optional[int] = None,
) -> dict[str, Any]:
    options = {
        "backend": backend,
        "num_predict": max_tokens,
        "temperature": 0.0,
        "_think": False,
    }
    if draft_num_predict is not None:
        options["draft_num_predict"] = int(draft_num_predict)
    return options


def preload_machboost(
    client: MachBoostClient,
    model: str,
    *,
    options: Mapping[str, Any],
    keep_alive: str,
    clock: Callable[[], float],
) -> dict[str, Any]:
    load_started = clock()
    load = client.load(
        model,
        options=dict(options),
        keep_alive=keep_alive,
        warmup=True,
    )
    load_wall = max(0.0, clock() - load_started)
    instance = load.get("instance") or {}
    return {
        "requested_model": model,
        "resolved_model": instance.get("model", model),
        "backend": instance.get("backend", options.get("backend", "auto")),
        "model_load_seconds": float(load.get("load_duration_seconds") or 0.0),
        "compile_warmup_seconds": float(load.get("warmup_duration_seconds") or 0.0),
        "load_wall_seconds": load_wall,
    }


def measure_machboost_chat(
    client: MachBoostClient,
    model: str,
    messages: Sequence[Mapping[str, str]],
    *,
    run: int,
    options: Mapping[str, Any],
    keep_alive: str,
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    first_text_at = None
    output = []
    final: Mapping[str, Any] = {}
    for item in client.chat(
        model,
        messages,
        options=dict(options),
        keep_alive=keep_alive,
        stream=True,
    ):
        content = str((item.get("message") or {}).get("content") or "")
        if content:
            if first_text_at is None:
                first_text_at = clock()
            output.append(content)
        if item.get("done"):
            final = item
    finished = clock()
    return latency_row(
        "machboost",
        run,
        final,
        output="".join(output),
        wall_seconds=max(0.0, finished - started),
        client_ttft_seconds=None if first_text_at is None else first_text_at - started,
    )


def measure_ollama_chat(
    adapter: OllamaHTTPAdapter,
    messages: Sequence[Mapping[str, str]],
    *,
    run: int,
    max_tokens: int,
    keep_alive: str,
    draft_num_predict: Optional[int],
    clock: Callable[[], float],
) -> dict[str, Any]:
    started = clock()
    first_text_at = None
    output = []
    final: Mapping[str, Any] = {}
    options: dict[str, Any] = {"num_predict": max_tokens, "temperature": 0.0}
    if draft_num_predict is not None:
        options["draft_num_predict"] = int(draft_num_predict)
    for chunk in adapter.chat(
        messages,
        options=options,
        keep_alive=keep_alive,
        stream=True,
        think=False,
    ):
        if chunk.content:
            if first_text_at is None:
                first_text_at = clock()
            output.append(chunk.content)
        if chunk.done:
            final = chunk.raw
    finished = clock()
    return latency_row(
        "ollama",
        run,
        final,
        output="".join(output),
        wall_seconds=max(0.0, finished - started),
        client_ttft_seconds=None if first_text_at is None else first_text_at - started,
    )


def benchmark_messages(system: str, prompt: str, nonce: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"{system}\nBenchmark nonce: {nonce}."},
        {"role": "user", "content": prompt},
    ]


def latency_row(
    engine: str,
    run: int,
    final: Mapping[str, Any],
    *,
    output: str,
    wall_seconds: float,
    client_ttft_seconds: Optional[float],
) -> dict[str, Any]:
    machboost = final.get("machboost") or {}
    eval_count = int(final.get("eval_count") or 0)
    eval_seconds = float(final.get("eval_duration") or 0) / 1_000_000_000
    prompt_count = int(final.get("prompt_eval_count") or 0)
    prompt_seconds = float(final.get("prompt_eval_duration") or 0) / 1_000_000_000
    return {
        "engine": engine,
        "run": run,
        "wall_seconds": wall_seconds,
        "client_ttft_seconds": client_ttft_seconds,
        "backend_ttft_seconds": machboost.get("time_to_first_token_seconds"),
        "total_seconds": float(final.get("total_duration") or 0) / 1_000_000_000,
        "load_seconds": float(final.get("load_duration") or 0) / 1_000_000_000,
        "prompt_eval_count": prompt_count,
        "prompt_eval_seconds": prompt_seconds,
        "prompt_tokens_per_second": safe_ratio(prompt_count, prompt_seconds),
        "eval_count": eval_count,
        "eval_seconds": eval_seconds,
        "tokens_per_second": safe_ratio(eval_count, eval_seconds),
        "output": output.strip(),
    }


def summarize_latency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ttfts = [
        float(row["client_ttft_seconds"])
        for row in rows
        if row.get("client_ttft_seconds") is not None
    ]
    return {
        "runs": len(rows),
        "median_wall_seconds": median(float(row["wall_seconds"]) for row in rows),
        "median_client_ttft_seconds": median(ttfts),
        "median_tokens_per_second": median(
            float(row["tokens_per_second"]) for row in rows
        ),
        "median_prompt_tokens_per_second": median(
            float(row["prompt_tokens_per_second"]) for row in rows
        ),
        "median_eval_count": median(float(row["eval_count"]) for row in rows),
    }


def normalized_outputs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [" ".join(str(row.get("output") or "").lower().split()) for row in rows]


def median(values) -> float:
    data = list(values)
    return float(statistics.median(data)) if data else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0
