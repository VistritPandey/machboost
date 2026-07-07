from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping, Optional, Tuple

from .core import (
    DEFAULT_MAX_DRAFT_TOKENS,
    DEFAULT_MAX_SUFFIX_TOKENS,
    DEFAULT_NGRAM,
    RunStats,
    Token,
    TokenSeq,
    machboost,
)


@dataclass(frozen=True)
class GatePolicy:
    min_speedup: float = 1.05
    min_acceptance_rate: float = 0.10
    require_exact_match: bool = True


@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str


@dataclass(frozen=True)
class GenerationMeasurement:
    tokens: Tuple[Token, ...]
    text: str
    elapsed_ms: float
    tokens_per_second: float
    target_calls: int
    forward_calls: int
    stats: Optional[RunStats] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "text": self.text,
            "elapsed_ms": self.elapsed_ms,
            "tokens_per_second": self.tokens_per_second,
            "target_calls": self.target_calls,
            "forward_calls": self.forward_calls,
            "stats": _stats_to_dict(self.stats),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    output_match: bool
    speedup: float
    acceptance_rate: float
    forward_reduction_rate: float
    baseline: GenerationMeasurement
    boosted: GenerationMeasurement
    decision: GateDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_match": self.output_match,
            "speedup": self.speedup,
            "acceptance_rate": self.acceptance_rate,
            "forward_reduction_rate": self.forward_reduction_rate,
            "baseline": self.baseline.to_dict(),
            "boosted": self.boosted.to_dict(),
            "decision": {
                "enabled": self.decision.enabled,
                "reason": self.decision.reason,
            },
        }


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    context: str = ""
    max_tokens: int = 128


def benchmark(
    service,
    prompt: str,
    *,
    context: Optional[str] = None,
    context_tokens: Optional[Iterable[Token]] = None,
    max_tokens: int = 128,
    ngram: int = DEFAULT_NGRAM,
    max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
    max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
    candidate_limit: int = 1,
    gate_policy: Optional[GatePolicy] = None,
) -> BenchmarkResult:
    prompt_tokens = service.encode(prompt)
    if context_tokens is None:
        context_tokens = service.encode(context or "")
    corpus_tokens = tuple(prompt_tokens) + tuple(context_tokens)
    policy = gate_policy or GatePolicy()

    baseline = measure_baseline(service, prompt_tokens, max_tokens=max_tokens)
    boosted = measure_boosted(
        service,
        prompt_tokens,
        corpus_tokens=corpus_tokens,
        max_tokens=max_tokens,
        ngram=ngram,
        max_suffix_tokens=max_suffix_tokens,
        max_draft_tokens=max_draft_tokens,
        candidate_limit=candidate_limit,
    )

    output_match = baseline.tokens == boosted.tokens
    speedup = _ratio(baseline.elapsed_ms, boosted.elapsed_ms)
    acceptance_rate = _acceptance_rate(boosted.stats)
    forward_reduction_rate = _forward_reduction_rate(baseline.target_calls, boosted.target_calls)
    decision = decide_gate(
        output_match=output_match,
        speedup=speedup,
        acceptance_rate=acceptance_rate,
        generated_tokens=len(boosted.tokens),
        policy=policy,
    )
    return BenchmarkResult(
        output_match=output_match,
        speedup=speedup,
        acceptance_rate=acceptance_rate,
        forward_reduction_rate=forward_reduction_rate,
        baseline=baseline,
        boosted=boosted,
        decision=decision,
    )


def benchmark_cases(
    service,
    cases: Iterable[BenchmarkCase],
    *,
    gate_policy: Optional[GatePolicy] = None,
    ngram: int = DEFAULT_NGRAM,
    max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
    max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
    candidate_limit: int = 1,
) -> Tuple[BenchmarkResult, ...]:
    return tuple(
        benchmark(
            service,
            case.prompt,
            context=case.context,
            max_tokens=case.max_tokens,
            gate_policy=gate_policy,
            ngram=ngram,
            max_suffix_tokens=max_suffix_tokens,
            max_draft_tokens=max_draft_tokens,
            candidate_limit=candidate_limit,
        )
        for case in cases
    )


def measure_baseline(
    service,
    prompt_tokens: TokenSeq,
    *,
    max_tokens: int,
    stop_tokens: Optional[Iterable[Token]] = None,
) -> GenerationMeasurement:
    _reset_service(service)
    stop_set = {int(token) for token in stop_tokens or ()}
    generated: list[Token] = []
    started = time.perf_counter()
    while len(generated) < max_tokens:
        token = service.next_token(tuple(prompt_tokens) + tuple(generated))
        if token is None:
            break
        if stop_set and int(token) in stop_set:
            break
        generated.append(int(token))
    elapsed_ms = (time.perf_counter() - started) * 1000
    tokens = tuple(generated)
    forward_calls = _forward_calls(service, fallback=len(tokens))
    stats = RunStats(
        generated_tokens=len(tokens),
        baseline_target_calls=len(tokens),
        target_calls=forward_calls,
        verify_calls=0,
        next_token_calls=forward_calls,
        candidate_rounds=0,
        accepted_draft_tokens=0,
        accepted_draft_spans=0,
        rejected_candidates=0,
    )
    return GenerationMeasurement(
        tokens=tokens,
        text=_decode(service, tokens),
        elapsed_ms=elapsed_ms,
        tokens_per_second=_tokens_per_second(len(tokens), elapsed_ms),
        target_calls=forward_calls,
        forward_calls=forward_calls,
        stats=stats,
    )


def measure_boosted(
    service,
    prompt_tokens: TokenSeq,
    *,
    corpus_tokens: Iterable[Token],
    max_tokens: int,
    ngram: int = DEFAULT_NGRAM,
    max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
    max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
    candidate_limit: int = 1,
) -> GenerationMeasurement:
    _reset_service(service)
    boosted = machboost(
        service,
        corpus_tokens=corpus_tokens,
        ngram=ngram,
        max_suffix_tokens=max_suffix_tokens,
        max_draft_tokens=max_draft_tokens,
        candidate_limit=candidate_limit,
    )
    started = time.perf_counter()
    tokens, stats = boosted.generate(prompt_tokens, max_tokens=max_tokens)
    elapsed_ms = (time.perf_counter() - started) * 1000
    forward_calls = _forward_calls(service, fallback=stats.target_calls)
    return GenerationMeasurement(
        tokens=tokens,
        text=_decode(service, tokens),
        elapsed_ms=elapsed_ms,
        tokens_per_second=_tokens_per_second(len(tokens), elapsed_ms),
        target_calls=stats.target_calls,
        forward_calls=forward_calls,
        stats=stats,
    )


def decide_gate(
    *,
    output_match: bool,
    speedup: float,
    acceptance_rate: float,
    generated_tokens: int,
    policy: Optional[GatePolicy] = None,
) -> GateDecision:
    policy = policy or GatePolicy()
    if generated_tokens <= 0:
        return GateDecision(False, "no tokens generated")
    if policy.require_exact_match and not output_match:
        return GateDecision(False, "boosted output did not match baseline")
    if speedup < policy.min_speedup:
        return GateDecision(False, "measured speedup is below policy threshold")
    if acceptance_rate < policy.min_acceptance_rate:
        return GateDecision(False, "draft acceptance rate is below policy threshold")
    return GateDecision(True, "boost meets exactness and speed policy")


def summarize_results(results: Iterable[BenchmarkResult]) -> dict[str, Any]:
    rows = tuple(results)
    if not rows:
        return {
            "rows": 0,
            "output_match_rate": 0.0,
            "median_speedup": 0.0,
            "median_acceptance_rate": 0.0,
            "enabled_rate": 0.0,
        }
    return {
        "rows": len(rows),
        "output_match_rate": sum(1 for row in rows if row.output_match) / len(rows),
        "median_speedup": _median(row.speedup for row in rows),
        "median_acceptance_rate": _median(row.acceptance_rate for row in rows),
        "enabled_rate": sum(1 for row in rows if row.decision.enabled) / len(rows),
    }


def _reset_service(service) -> None:
    reset_cache = getattr(service, "reset_cache", None)
    if callable(reset_cache):
        reset_cache()
    if hasattr(service, "forward_calls"):
        service.forward_calls = 0


def _forward_calls(service, *, fallback: int) -> int:
    return int(getattr(service, "forward_calls", fallback))


def _decode(service, tokens: Iterable[Token]) -> str:
    decode = getattr(service, "decode", None)
    if callable(decode):
        return str(decode(tuple(tokens)))
    return repr(tuple(tokens))


def _tokens_per_second(tokens: int, elapsed_ms: float) -> float:
    if tokens <= 0 or elapsed_ms <= 0:
        return 0.0
    return tokens / (elapsed_ms / 1000)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _acceptance_rate(stats: Optional[RunStats]) -> float:
    if stats is None or stats.generated_tokens <= 0:
        return 0.0
    return stats.accepted_draft_tokens / stats.generated_tokens


def _forward_reduction_rate(baseline_calls: int, boosted_calls: int) -> float:
    if baseline_calls <= 0:
        return 0.0
    return (baseline_calls - boosted_calls) / baseline_calls


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _stats_to_dict(stats: Optional[RunStats]) -> Optional[Mapping[str, Any]]:
    if stats is None:
        return None
    return {
        "generated_tokens": stats.generated_tokens,
        "baseline_target_calls": stats.baseline_target_calls,
        "target_calls": stats.target_calls,
        "verify_calls": stats.verify_calls,
        "next_token_calls": stats.next_token_calls,
        "candidate_rounds": stats.candidate_rounds,
        "accepted_draft_tokens": stats.accepted_draft_tokens,
        "accepted_draft_spans": stats.accepted_draft_spans,
        "rejected_candidates": stats.rejected_candidates,
        "estimated_speedup": stats.estimated_speedup,
    }
