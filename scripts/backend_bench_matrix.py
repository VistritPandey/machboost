#!/usr/bin/env python3
"""Benchmark HF, MLX, and Ollama paths with fresh prompts.

The goal is not to hide backend differences. Hugging Face has a cache-aware
verifier prototype, MLX uses the package cache-aware verifier adapter, and
Ollama HTTP is measured as a wrapper because its public API does not expose
verifier hooks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from machboost import Accelerator
from machboost.adapters import MLXCausalLMService, OllamaHTTPAdapter

SCHEMA_VERSION = "machboost.backend_bench_matrix.v2"


@dataclass(frozen=True)
class Fixture:
    name: str
    workflow: str
    expectation: str
    prompt: str
    context: str
    nonce: str


@dataclass(frozen=True)
class BenchRow:
    backend: str
    model: str
    fixture: str
    workflow: str
    expectation: str
    nonce: str
    mode: str
    output_match: bool
    baseline_ms: float
    boosted_ms: float
    baseline_tokens_per_second: float
    boosted_tokens_per_second: float
    speedup: float
    baseline_forwards: int
    boosted_forwards: int
    accepted_draft_tokens: int
    generated_tokens: int
    baseline_output_preview: str
    boosted_output_preview: str
    note: str


def make_nonce(seed: str, repeat: int, fixture: str) -> str:
    digest = hashlib.sha256(f"{seed}:{repeat}:{fixture}:{time.time_ns()}".encode("utf-8")).hexdigest()
    return f"mb-{digest[:10]}"


def build_fixture(name: str, nonce: str) -> Fixture:
    if name == "real_readme_api":
        context = read_repo_snippet("README.md", "Benchmark and calibrate before turning the layer on", max_chars=900)
        prefix = "Benchmark and calibrate before turning the layer on"
        prompt = (
            "Continue this README excerpt exactly from the repository source. Do not add commentary.\n\n"
            "<readme>\n"
            f"{context}"
            "\n</readme>\n\n"
            f"Continuation:\n{prefix}"
        )
        return Fixture(
            name=name,
            workflow="real_readme_continuation",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "real_core_code":
        context = read_repo_snippet("machboost/core.py", "class BoostedService:", max_chars=1200)
        prefix = "class BoostedService:"
        prompt = (
            "Continue this Python source excerpt exactly from the repository file.\n\n"
            "<source>\n"
            f"{context}"
            "\n</source>\n\n"
            f"{prefix}"
        )
        return Fixture(
            name=name,
            workflow="real_code_continuation",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "real_paper_method":
        context = read_repo_snippet("paper/machboost.tex", "\\section{Method}", max_chars=1200)
        prefix = "\\section{Method}"
        prompt = (
            "Continue this LaTeX paper excerpt exactly from the source file.\n\n"
            "<paper>\n"
            f"{context}"
            "\n</paper>\n\n"
            f"{prefix}"
        )
        return Fixture(
            name=name,
            workflow="real_paper_continuation",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "policy":
        context = (
            f"Benchmark memo {nonce}\n"
            "Policy: MachBoost benchmark records must include backend, model, fixture, nonce, "
            "exact output match, baseline tokens per second, boosted tokens per second, and accepted draft tokens.\n"
            "Reason: the record must prove speed without changing model output.\n"
        )
        prompt = (
            "Continue the memo exactly from the document. Do not add commentary.\n\n"
            "<document>\n"
            f"{context}"
            "</document>\n\n"
            "Continuation:\nPolicy: MachBoost benchmark records must include"
        )
        return Fixture(
            name=name,
            workflow="policy_quote",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "json":
        context = (
            "{\n"
            f'  "nonce": "{nonce}",\n'
            '  "backend": "machboost",\n'
            '  "accuracy_rule": "boosted output must match baseline output exactly",\n'
            '  "metrics": ["tokens_per_second", "wall_clock_ms", "accepted_draft_tokens"]\n'
            "}\n"
        )
        prompt = (
            "Continue this JSON exactly from the source document.\n\n"
            "<source>\n"
            f"{context}"
            "</source>\n\n"
            "{\n"
            f'  "nonce": "{nonce}",\n'
            '  "backend":'
        )
        return Fixture(
            name=name,
            workflow="structured_config",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "rag":
        context = (
            f"Retrieval note {nonce}\n"
            "Answer sentence: The acceleration layer verifies local-context draft tokens against the target model before accepting them.\n"
            "Support sentence: Rejected draft tokens fall back to the target model path.\n"
        )
        prompt = (
            "Answer by copying the answer sentence exactly from the retrieved note.\n\n"
            "<retrieved>\n"
            f"{context}"
            "</retrieved>\n\n"
            "Question: How does the acceleration layer stay exact?\n"
            "Answer:"
        )
        return Fixture(
            name=name,
            workflow="rag_answer",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "code":
        context = (
            f"# nonce: {nonce}\n"
            "def format_backend_row(row):\n"
            "    return f\"{row['backend']} | {row['model']} | {row['speedup']:.2f}x | {row['output_match']}\"\n\n"
            "def format_summary(rows):\n"
            "    return \"\\n\".join(format_backend_row(row) for row in rows)\n"
        )
        prompt = (
            "Continue this Python code exactly from the source file.\n\n"
            "<source>\n"
            f"{context}"
            "</source>\n\n"
            f"# nonce: {nonce}\n"
            "def format_backend_row(row):\n"
            "    return"
        )
        return Fixture(
            name=name,
            workflow="code_completion",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "repo_quote":
        context = (
            f"Command note {nonce}\n"
            "Run the Python benchmark suite with:\n"
            "python3 scripts/backend_bench_matrix.py --backends mlx --fixtures policy,json,rag,code --repeat 5\n"
        )
        prompt = (
            "Copy the benchmark command exactly from the command note.\n\n"
            "<note>\n"
            f"{context}"
            "</note>\n\n"
            "Benchmark command:"
        )
        return Fixture(
            name=name,
            workflow="repo_quote",
            expectation="positive",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "creative_open":
        context = (
            f"Cache note {nonce}: benchmark rows, fixture metadata, and JSON output paths are stored locally."
        )
        prompt = (
            f"Write two fresh sentences about why local developer tools can feel empowering. Include nonce {nonce}."
        )
        return Fixture(
            name=name,
            workflow="creative_generation",
            expectation="negative",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    if name == "short_answer":
        context = f"{nonce} yes no maybe benchmark profile context"
        prompt = f"Answer with exactly one word, yes or no. Nonce: {nonce}."
        return Fixture(
            name=name,
            workflow="short_answer",
            expectation="negative",
            prompt=prompt,
            context=context,
            nonce=nonce,
        )
    raise ValueError(f"unknown fixture: {name}")


def read_repo_snippet(path: str, marker: str, *, max_chars: int) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"marker not found in {path}: {marker}")
    return text[start : start + max_chars].strip()


def preview(text: str, chars: int = 180) -> str:
    return " ".join(text.split())[:chars]


def tokens_per_second(tokens: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return tokens / elapsed


def run_hf(args: argparse.Namespace, fixtures: list[Fixture]) -> list[BenchRow]:
    import hf_corpus_speculate as hfs

    runtime_args = argparse.Namespace(
        model=args.hf_model,
        device=args.device,
        local_files_only=True,
    )
    tokenizer, model, device = hfs.load_runtime(runtime_args)
    rows: list[BenchRow] = []

    with tempfile.TemporaryDirectory(prefix="machboost-backend-hf-") as tmp:
        tmp_path = Path(tmp)
        for fixture in fixtures:
            prompt_path = tmp_path / f"{fixture.name}-{fixture.nonce}-prompt.txt"
            context_path = tmp_path / f"{fixture.name}-{fixture.nonce}-context.txt"
            prompt_path.write_text(fixture.prompt, encoding="utf-8")
            context_path.write_text(fixture.context, encoding="utf-8")

            bench_args = argparse.Namespace(
                model=args.hf_model,
                prompt=str(prompt_path),
                context=str(context_path),
                max_new_tokens=args.max_new_tokens,
                ngram=args.ngram,
                max_draft_tokens=args.max_draft_tokens,
                draft_policy="fixed",
                initial_draft_tokens=2,
                min_draft_tokens=1,
                draft_step=2,
                candidate_limit=args.candidate_limit,
                warmup_tokens=0,
                auto_draft=False,
                draft_sweep=str(args.max_draft_tokens),
                source_mode=args.source_mode,
                verify_mode=args.verify_mode,
                anchor_tokens=args.anchor_tokens,
                min_verify_margin=0.0,
                max_context_chars=100_000,
                device=args.device,
                local_files_only=True,
                json=True,
            )
            prompt_ids, source_ids, context_token_count = hfs.prepare_inputs(bench_args, tokenizer)
            result = hfs.compare_with_runtime(
                bench_args,
                tokenizer,
                model,
                device,
                prompt_ids,
                source_ids,
                context_token_count,
            )
            baseline = result.baseline
            boosted = result.speculative
            rows.append(
                BenchRow(
                    backend="huggingface",
                    model=args.hf_model,
                    fixture=fixture.name,
                    workflow=fixture.workflow,
                    expectation=fixture.expectation,
                    nonce=fixture.nonce,
                    mode="native_verifier_kv_cache",
                    output_match=result.output_match,
                    baseline_ms=float(baseline.elapsed_ms),
                    boosted_ms=float(boosted.elapsed_ms),
                    baseline_tokens_per_second=baseline.tokens_per_second,
                    boosted_tokens_per_second=boosted.tokens_per_second,
                    speedup=result.wall_clock_speedup,
                    baseline_forwards=baseline.model_forwards,
                    boosted_forwards=boosted.model_forwards,
                    accepted_draft_tokens=boosted.accepted_draft_tokens,
                    generated_tokens=boosted.generated_tokens,
                    baseline_output_preview=preview(baseline.output),
                    boosted_output_preview=preview(boosted.output),
                    note="HF path uses the cache-aware prototype verifier.",
                )
            )
    return rows


def native_mlx_generate(service: MLXCausalLMService, prompt: str, max_tokens: int):
    service.reset_cache()
    service.forward_calls = 0
    started = time.perf_counter()
    prompt_tokens = service.encode(prompt)
    generated = service.generate_tokens(prompt_tokens, max_tokens=max_tokens)
    elapsed = time.perf_counter() - started
    return tuple(generated), elapsed, service.forward_calls


def run_mlx(args: argparse.Namespace, fixtures: list[Fixture]) -> list[BenchRow]:
    service = MLXCausalLMService.from_pretrained(
        args.mlx_model,
        lazy=args.mlx_lazy,
        cache_enabled=not args.mlx_disable_cache,
    )
    rows: list[BenchRow] = []
    for index, fixture in enumerate(fixtures):
        accelerator = Accelerator(
            service,
            context_texts=[source_text(fixture, args.source_mode)],
            ngram=args.ngram,
            max_draft_tokens=args.max_draft_tokens,
            candidate_limit=args.candidate_limit,
        )

        def run_baseline():
            return native_mlx_generate(service, fixture.prompt, args.max_new_tokens)

        def run_boosted():
            service.reset_cache()
            service.forward_calls = 0
            started = time.perf_counter()
            result = accelerator.generate_result(fixture.prompt, max_tokens=args.max_new_tokens)
            return result, time.perf_counter() - started, service.forward_calls

        if index % 2 == 0:
            baseline_tokens, baseline_elapsed, baseline_forwards = run_baseline()
            boosted_result, boosted_elapsed, boosted_forwards = run_boosted()
        else:
            boosted_result, boosted_elapsed, boosted_forwards = run_boosted()
            baseline_tokens, baseline_elapsed, baseline_forwards = run_baseline()

        baseline_output = service.decode(baseline_tokens)
        boosted_tokens = boosted_result.tokens
        boosted_output = boosted_result.text
        stats = boosted_result.stats
        service.reset_cache()

        rows.append(
            BenchRow(
                backend="mlx",
                model=args.mlx_model,
                fixture=fixture.name,
                workflow=fixture.workflow,
                expectation=fixture.expectation,
                nonce=fixture.nonce,
                mode="native_fallback" if stats.accepted_draft_tokens == 0 else "adaptive_context_verifier",
                output_match=baseline_tokens == boosted_tokens,
                baseline_ms=baseline_elapsed * 1000,
                boosted_ms=boosted_elapsed * 1000,
                baseline_tokens_per_second=tokens_per_second(len(baseline_tokens), baseline_elapsed),
                boosted_tokens_per_second=tokens_per_second(len(boosted_tokens), boosted_elapsed),
                speedup=(baseline_elapsed / boosted_elapsed) if boosted_elapsed > 0 else 0.0,
                baseline_forwards=baseline_forwards,
                boosted_forwards=boosted_forwards,
                accepted_draft_tokens=stats.accepted_draft_tokens,
                generated_tokens=len(boosted_tokens),
                baseline_output_preview=preview(baseline_output),
                boosted_output_preview=preview(boosted_output),
                note=(
                    "Paired wall time compares native mlx-lm generation with the adaptive package path; "
                    "fixture order alternates baseline-first and boosted-first."
                ),
            )
        )
    return rows


def ollama_direct_generate(endpoint: str, model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": options,
    }
    req = request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_tps(data: dict[str, Any]) -> float:
    eval_count = int(data.get("eval_count", 0) or 0)
    eval_duration = int(data.get("eval_duration", 0) or 0)
    if eval_count <= 0 or eval_duration <= 0:
        return 0.0
    return eval_count / (eval_duration / 1_000_000_000)


def ollama_ms(data: dict[str, Any]) -> float:
    return int(data.get("total_duration", 0) or 0) / 1_000_000


def run_ollama(args: argparse.Namespace, fixtures: list[Fixture]) -> list[BenchRow]:
    adapter = OllamaHTTPAdapter(args.ollama_model, endpoint=args.ollama_endpoint, keep_alive=0, timeout=300)
    rows: list[BenchRow] = []
    for fixture in fixtures:
        options = {
            "num_predict": args.max_new_tokens,
            "num_ctx": args.ollama_ctx,
            "temperature": 0,
            "seed": int(hashlib.sha256(fixture.nonce.encode("utf-8")).hexdigest()[:8], 16),
        }
        baseline_started = time.perf_counter()
        baseline = ollama_direct_generate(adapter.endpoint, args.ollama_model, fixture.prompt, options)
        baseline_elapsed = time.perf_counter() - baseline_started

        boosted_started = time.perf_counter()
        boosted = adapter.benchmark(fixture.prompt, tokens=args.max_new_tokens, ctx=args.ollama_ctx, options=options)
        boosted_elapsed = time.perf_counter() - boosted_started

        baseline_output = str(baseline.get("response", ""))
        boosted_output = boosted.response
        baseline_tps = ollama_tps(baseline)
        boosted_tps = boosted.tokens_per_second
        rows.append(
            BenchRow(
                backend="ollama",
                model=args.ollama_model,
                fixture=fixture.name,
                workflow=fixture.workflow,
                expectation=fixture.expectation,
                nonce=fixture.nonce,
                mode="http_wrapper_no_native_verifier",
                output_match=baseline_output == boosted_output,
                baseline_ms=ollama_ms(baseline) or baseline_elapsed * 1000,
                boosted_ms=boosted.total_ms or boosted_elapsed * 1000,
                baseline_tokens_per_second=baseline_tps,
                boosted_tokens_per_second=boosted_tps,
                speedup=(boosted_tps / baseline_tps) if baseline_tps > 0 else 0.0,
                baseline_forwards=int(baseline.get("eval_count", 0) or 0),
                boosted_forwards=boosted.eval_count,
                accepted_draft_tokens=0,
                generated_tokens=boosted.eval_count,
                baseline_output_preview=preview(baseline_output),
                boosted_output_preview=preview(boosted_output),
                note=(
                    "Ollama HTTP exposes wrapper metrics only; speedup is eval-token/sec ratio, "
                    "not total wall time, to avoid model-load/cache effects."
                ),
            )
        )
    return rows


def source_text(fixture: Fixture, mode: str) -> str:
    if mode == "context":
        return fixture.context
    if mode == "prompt":
        return fixture.prompt
    return fixture.prompt + "\n" + fixture.context


def selected_backends(value: str) -> set[str]:
    values = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not values or "all" in values:
        return {"hf", "mlx", "ollama"}
    return values


def summarize(rows: Iterable[BenchRow]) -> list[dict[str, Any]]:
    by_backend: dict[str, list[BenchRow]] = {}
    for row in rows:
        by_backend.setdefault(row.backend, []).append(row)
    summaries = []
    for backend, backend_rows in sorted(by_backend.items()):
        count = len(backend_rows)
        summaries.append(
            {
                "backend": backend,
                "rows": count,
                "output_match_rate": sum(1 for row in backend_rows if row.output_match) / count if count else 0,
                "median_speedup": median([row.speedup for row in backend_rows]),
                "p90_speedup": percentile([row.speedup for row in backend_rows], 90),
                "mean_speedup": mean([row.speedup for row in backend_rows]),
                "median_baseline_tokens_per_second": median(
                    [row.baseline_tokens_per_second for row in backend_rows]
                ),
                "median_boosted_tokens_per_second": median([row.boosted_tokens_per_second for row in backend_rows]),
                "p90_boosted_tokens_per_second": percentile(
                    [row.boosted_tokens_per_second for row in backend_rows], 90
                ),
                "median_accepted_draft_tokens": median([row.accepted_draft_tokens for row in backend_rows]),
                "median_forward_reduction_percent": median(
                    [
                        ((row.baseline_forwards - row.boosted_forwards) / row.baseline_forwards) * 100
                        for row in backend_rows
                        if row.baseline_forwards > 0
                    ]
                ),
            }
        )
    return summaries


def summarize_by_fixture(rows: Iterable[BenchRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[BenchRow]] = {}
    for row in rows:
        grouped.setdefault((row.backend, row.fixture), []).append(row)

    summaries = []
    for (backend, fixture), fixture_rows in sorted(grouped.items()):
        count = len(fixture_rows)
        first = fixture_rows[0]
        summaries.append(
            {
                "backend": backend,
                "fixture": fixture,
                "workflow": first.workflow,
                "expectation": first.expectation,
                "rows": count,
                "output_match_rate": sum(1 for row in fixture_rows if row.output_match) / count if count else 0,
                "median_speedup": median([row.speedup for row in fixture_rows]),
                "p90_speedup": percentile([row.speedup for row in fixture_rows], 90),
                "median_baseline_tokens_per_second": median(
                    [row.baseline_tokens_per_second for row in fixture_rows]
                ),
                "median_boosted_tokens_per_second": median([row.boosted_tokens_per_second for row in fixture_rows]),
                "median_accepted_draft_tokens": median([row.accepted_draft_tokens for row in fixture_rows]),
                "median_forward_reduction_percent": median(
                    [
                        ((row.baseline_forwards - row.boosted_forwards) / row.baseline_forwards) * 100
                        for row in fixture_rows
                        if row.baseline_forwards > 0
                    ]
                ),
            }
        )
    return summaries


def median(values: list[float | int]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def percentile(values: list[float | int], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MachBoost across HF, MLX, and Ollama backends.")
    parser.add_argument("--backends", default="all", help="Comma-separated: hf,mlx,ollama,all")
    parser.add_argument("--fixtures", default="policy,json,rag,code,repo_quote,creative_open,short_answer")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", default="machboost")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=1)
    parser.add_argument("--source-mode", choices=["prompt-context", "context", "prompt"], default="context")
    parser.add_argument("--verify-mode", choices=["block", "hybrid", "sequential"], default="hybrid")
    parser.add_argument("--anchor-tokens", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mlx-model", default="mlx-community/Qwen3.5-0.8B-MLX-4bit")
    parser.add_argument("--mlx-lazy", action="store_true")
    parser.add_argument("--mlx-disable-cache", action="store_true")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-endpoint", default=None)
    parser.add_argument("--ollama-ctx", type=int, default=4096)
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_names = [item.strip() for item in args.fixtures.split(",") if item.strip()]
    fixtures = [
        build_fixture(name, make_nonce(args.seed, repeat, name))
        for repeat in range(1, args.repeat + 1)
        for name in fixture_names
    ]
    rows: list[BenchRow] = []
    backends = selected_backends(args.backends)
    if "hf" in backends:
        rows.extend(run_hf(args, fixtures))
    if "mlx" in backends:
        rows.extend(run_mlx(args, fixtures))
    if "ollama" in backends:
        rows.extend(run_ollama(args, fixtures))
    data_rows = [asdict(row) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "fixtures": [asdict(fixture) for fixture in fixtures],
        "settings": {
            "backends": sorted(backends),
            "repeat": args.repeat,
            "max_new_tokens": args.max_new_tokens,
            "ngram": args.ngram,
            "max_draft_tokens": args.max_draft_tokens,
            "candidate_limit": args.candidate_limit,
            "source_mode": args.source_mode,
            "verify_mode": args.verify_mode,
            "mlx_disable_cache": args.mlx_disable_cache,
        },
        "summaries": summarize(rows),
        "fixture_summaries": summarize_by_fixture(rows),
        "rows": data_rows,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = {
            "schema_version": SCHEMA_VERSION + ".self_test",
            "ok": build_fixture("policy", "mb-test").nonce == "mb-test"
            and build_fixture("creative_open", "mb-test").expectation == "negative"
            and selected_backends("all") == {"hf", "mlx", "ollama"},
        }
    else:
        result = run(args)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not args.self_test or result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
