from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

from .bench import BenchmarkResult, GatePolicy, benchmark as benchmark_service, measure_baseline, summarize_results
from .core import (
    DEFAULT_MAX_DRAFT_TOKENS,
    DEFAULT_MAX_SUFFIX_TOKENS,
    DEFAULT_NGRAM,
    RunStats,
    Token,
    machboost,
)

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {".git", ".cache", "build", "dist", "node_modules", "target", "vendor"}


@dataclass(frozen=True)
class AcceleratorResult:
    text: str
    tokens: Tuple[Token, ...]
    stats: RunStats


@dataclass(frozen=True)
class CalibrationResult:
    enabled: bool
    summary: dict[str, Any]
    results: Tuple[BenchmarkResult, ...]


class Accelerator:
    def __init__(
        self,
        service,
        *,
        context_texts: Optional[Iterable[str]] = None,
        ngram: int = DEFAULT_NGRAM,
        max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
        max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
        boost_enabled: bool = True,
    ) -> None:
        self.service = service
        self.context_texts = tuple(context_texts or ())
        self.ngram = ngram
        self.max_suffix_tokens = max_suffix_tokens
        self.max_draft_tokens = max_draft_tokens
        self.boost_enabled = bool(boost_enabled)
        self.context_tokens = self._encode_many(self.context_texts)

    @classmethod
    def from_mlx(
        cls,
        model: str,
        *,
        context: Optional[Union[Iterable[str], str]] = None,
        context_paths: Optional[Union[Iterable[str], str]] = None,
        max_context_chars: int = 200_000,
        ngram: int = DEFAULT_NGRAM,
        max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
        max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
        tokenizer_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        adapter_path: Optional[str] = None,
        lazy: bool = False,
        revision: Optional[str] = None,
        min_verify_margin: float = 0.0,
        boost_enabled: bool = True,
    ) -> "Accelerator":
        from machboost.adapters import MLXCausalLMService

        service = MLXCausalLMService.from_pretrained(
            model,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            revision=revision,
            min_verify_margin=min_verify_margin,
        )
        context_texts = resolve_context(context, max_chars=max_context_chars)
        context_texts += read_context_paths(context_paths, max_chars=max_context_chars - sum(map(len, context_texts)))
        return cls(
            service,
            context_texts=context_texts,
            ngram=ngram,
            max_suffix_tokens=max_suffix_tokens,
            max_draft_tokens=max_draft_tokens,
            boost_enabled=boost_enabled,
        )

    @classmethod
    def from_huggingface(
        cls,
        model: str,
        *,
        context: Optional[Union[Iterable[str], str]] = None,
        context_paths: Optional[Union[Iterable[str], str]] = None,
        max_context_chars: int = 200_000,
        ngram: int = DEFAULT_NGRAM,
        max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
        max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
        device: Optional[str] = None,
        local_files_only: bool = False,
        torch_dtype=None,
        model_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        min_verify_margin: float = 0.0,
        boost_enabled: bool = True,
    ) -> "Accelerator":
        from machboost.adapters import HuggingFaceCausalLMService

        service = HuggingFaceCausalLMService.from_pretrained(
            model,
            device=device,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            min_verify_margin=min_verify_margin,
        )
        context_texts = resolve_context(context, max_chars=max_context_chars)
        context_texts += read_context_paths(context_paths, max_chars=max_context_chars - sum(map(len, context_texts)))
        return cls(
            service,
            context_texts=context_texts,
            ngram=ngram,
            max_suffix_tokens=max_suffix_tokens,
            max_draft_tokens=max_draft_tokens,
            boost_enabled=boost_enabled,
        )

    def generate(self, prompt: str, *, max_tokens: int = 128, context: Optional[Union[Iterable[str], str]] = None):
        result = self.generate_result(prompt, max_tokens=max_tokens, context=context)
        return result.text, result.stats

    def generate_result(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        context: Optional[Union[Iterable[str], str]] = None,
    ) -> AcceleratorResult:
        prompt_tokens = self.service.encode(prompt)
        if not self.boost_enabled:
            measurement = measure_baseline(self.service, prompt_tokens, max_tokens=max_tokens)
            return AcceleratorResult(text=measurement.text, tokens=measurement.tokens, stats=measurement.stats)

        run_context_tokens = self.context_tokens + self._encode_many(resolve_context(context))
        corpus_tokens = tuple(prompt_tokens) + run_context_tokens
        reset_cache = getattr(self.service, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()

        boosted = machboost(
            self.service,
            corpus_tokens=corpus_tokens,
            ngram=self.ngram,
            max_suffix_tokens=self.max_suffix_tokens,
            max_draft_tokens=self.max_draft_tokens,
        )
        tokens, stats = boosted.generate(prompt_tokens, max_tokens=max_tokens)
        return AcceleratorResult(text=self.service.decode(tokens), tokens=tokens, stats=stats)

    def benchmark(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        context: Optional[Union[Iterable[str], str]] = None,
        gate_policy: Optional[GatePolicy] = None,
    ) -> BenchmarkResult:
        run_context_tokens = self.context_tokens + self._encode_many(resolve_context(context))
        return benchmark_service(
            self.service,
            prompt,
            context_tokens=run_context_tokens,
            max_tokens=max_tokens,
            ngram=self.ngram,
            max_suffix_tokens=self.max_suffix_tokens,
            max_draft_tokens=self.max_draft_tokens,
            gate_policy=gate_policy,
        )

    def calibrate(
        self,
        prompts: Union[Iterable[str], str],
        *,
        max_tokens: int = 24,
        context: Optional[Union[Iterable[str], str]] = None,
        gate_policy: Optional[GatePolicy] = None,
    ) -> CalibrationResult:
        policy = gate_policy or GatePolicy()
        results = tuple(
            self.benchmark(prompt, max_tokens=max_tokens, context=context, gate_policy=policy)
            for prompt in _items(prompts)
        )
        summary = summarize_results(results)
        enabled = (
            summary["rows"] > 0
            and summary["output_match_rate"] == 1.0
            and summary["median_speedup"] >= policy.min_speedup
            and summary["median_acceptance_rate"] >= policy.min_acceptance_rate
        )
        self.boost_enabled = enabled
        return CalibrationResult(enabled=enabled, summary=summary, results=results)

    def _encode_many(self, texts: Iterable[str]) -> Tuple[Token, ...]:
        tokens = []
        for text in texts:
            tokens.extend(self.service.encode(text))
        return tuple(tokens)


def resolve_context(context: Optional[Union[Iterable[str], str]], *, max_chars: int = 200_000) -> list[str]:
    texts: list[str] = []
    remaining = max(0, max_chars)
    for item in _items(context):
        if remaining <= 0:
            break
        path = Path(item).expanduser()
        chunks = read_context_paths([str(path)], max_chars=remaining) if path.exists() else [item]
        for chunk in chunks:
            if remaining <= 0:
                break
            excerpt = chunk[:remaining]
            texts.append(excerpt)
            remaining -= len(excerpt)
    return texts


def read_context_paths(paths: Optional[Union[Iterable[str], str]], *, max_chars: int = 200_000) -> list[str]:
    texts: list[str] = []
    remaining = max(0, max_chars)
    for item in _items(paths):
        if remaining <= 0:
            break
        path = Path(item).expanduser()
        for file_path in _iter_text_files(path):
            if remaining <= 0:
                break
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if not text:
                continue
            excerpt = text[:remaining]
            texts.append(f"\n\n# file: {file_path}\n{excerpt}")
            remaining -= len(excerpt)
    return texts


def _items(value: Optional[Union[Iterable[str], str]]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _iter_text_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in TEXT_SUFFIXES or path.suffix == "":
            yield path
        return
    if not path.is_dir():
        return
    for item in sorted(path.rglob("*")):
        if any(part in SKIP_DIRS for part in item.parts):
            continue
        if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES:
            yield item
