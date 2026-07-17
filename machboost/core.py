from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

Token = int
TokenSeq = Sequence[Token]

DEFAULT_NGRAM = 4
DEFAULT_MAX_SUFFIX_TOKENS = 32
DEFAULT_MAX_DRAFT_TOKENS = 32


@dataclass(frozen=True)
class Candidate:
    tokens: Tuple[Token, ...]
    matched_suffix_tokens: int
    source_match_start: int
    source_start: int
    score: float


@dataclass(frozen=True)
class RunStats:
    generated_tokens: int
    baseline_target_calls: int
    target_calls: int
    verify_calls: int
    next_token_calls: int
    candidate_rounds: int
    accepted_draft_tokens: int
    accepted_draft_spans: int
    rejected_candidates: int
    prompt_tokens: int = 0
    prompt_eval_seconds: float = 0.0
    generation_seconds: float = 0.0
    time_to_first_token_seconds: Optional[float] = None

    @property
    def estimated_speedup(self) -> float:
        if self.target_calls <= 0:
            return 1.0
        return self.baseline_target_calls / self.target_calls


@runtime_checkable
class StepService(Protocol):
    def next_token(self, prefix_tokens: TokenSeq) -> Optional[Token]:
        ...


VerifyResult = Union[int, Tuple[int, Optional[Token]]]


@runtime_checkable
class VerifierService(StepService, Protocol):
    def verify(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> VerifyResult:
        ...


class CorpusDrafter:
    def __init__(
        self,
        corpus_tokens: Iterable[Token],
        *,
        ngram: int = DEFAULT_NGRAM,
        max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
        max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
    ) -> None:
        self.ngram = ngram if ngram > 0 else DEFAULT_NGRAM
        self.max_suffix_tokens = max_suffix_tokens if max_suffix_tokens > 0 else DEFAULT_MAX_SUFFIX_TOKENS
        self.max_draft_tokens = max_draft_tokens if max_draft_tokens > 0 else DEFAULT_MAX_DRAFT_TOKENS
        self.corpus = tuple(int(token) for token in corpus_tokens)
        self.index = self._build_index(self.corpus, self.ngram)
        self.history: Tuple[Token, ...] = ()

    def reset(self, prompt_tokens: Iterable[Token]) -> None:
        self.history = tuple(int(token) for token in prompt_tokens)

    def observe(self, committed_tokens: Iterable[Token]) -> None:
        self.history = self.history + tuple(int(token) for token in committed_tokens)

    def propose(self, current_tokens: Iterable[Token] = (), *, max_tokens: Optional[int] = None) -> Tuple[Token, ...]:
        best = self.best(current_tokens, max_tokens=max_tokens)
        if best is None:
            return ()
        return best.tokens

    def best(self, current_tokens: Iterable[Token] = (), *, max_tokens: Optional[int] = None) -> Optional[Candidate]:
        candidates = self.candidates(current_tokens, max_tokens=max_tokens, limit=1)
        if not candidates:
            return None
        return candidates[0]

    def candidates(
        self,
        current_tokens: Iterable[Token] = (),
        *,
        max_tokens: Optional[int] = None,
        limit: int = 4,
    ) -> Tuple[Candidate, ...]:
        if limit <= 0 or not self.corpus or not self.index:
            return ()

        max_draft = self.max_draft_tokens
        if max_tokens is not None:
            max_draft = min(max_draft, max(0, max_tokens))
        if max_draft <= 0:
            return ()

        prefix = self.history + tuple(int(token) for token in current_tokens)
        if len(prefix) < self.ngram:
            return ()

        by_tokens: dict[Tuple[Token, ...], Candidate] = {}
        longest = min(self.max_suffix_tokens, len(prefix))
        for suffix_len in range(longest, self.ngram - 1, -1):
            suffix = prefix[-suffix_len:]
            needle = suffix[-self.ngram :]
            for ngram_pos in self.index.get(needle, ()):
                match_start = ngram_pos - (suffix_len - self.ngram)
                if match_start < 0 or match_start + suffix_len >= len(self.corpus):
                    continue
                if self.corpus[match_start : match_start + suffix_len] != suffix:
                    continue

                source_start = match_start + suffix_len
                source_end = min(source_start + max_draft, len(self.corpus))
                if source_start >= source_end:
                    continue

                tokens = self.corpus[source_start:source_end]
                candidate = Candidate(
                    tokens=tokens,
                    matched_suffix_tokens=suffix_len,
                    source_match_start=match_start,
                    source_start=source_start,
                    score=float(suffix_len) + len(tokens) / 100,
                )
                existing = by_tokens.get(tokens)
                if existing is None or _better(candidate, existing):
                    by_tokens[tokens] = candidate

            if len(by_tokens) >= limit:
                break

        ranked = sorted(by_tokens.values(), key=_sort_key)
        return tuple(ranked[:limit])

    @staticmethod
    def _build_index(tokens: Tuple[Token, ...], ngram: int) -> dict[Tuple[Token, ...], list[int]]:
        index: dict[Tuple[Token, ...], list[int]] = {}
        if ngram <= 0:
            return index
        for pos in range(0, max(0, len(tokens) - ngram + 1)):
            key = tokens[pos : pos + ngram]
            index.setdefault(key, []).append(pos)
        return index


class BoostedService:
    def __init__(
        self,
        service: StepService,
        drafter: CorpusDrafter,
        *,
        candidate_limit: int = 1,
        reentry_probe_tokens: int = 0,
    ) -> None:
        self.service = service
        self.drafter = drafter
        self.candidate_limit = max(1, int(candidate_limit))
        self.reentry_probe_tokens = max(0, int(reentry_probe_tokens))

    def generate(
        self,
        prompt_tokens: Iterable[Token],
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]] = None,
        on_tokens: Optional[Callable[[Tuple[Token, ...]], None]] = None,
    ) -> Tuple[Tuple[Token, ...], RunStats]:
        prompt = tuple(int(token) for token in prompt_tokens)
        stop_set = {int(token) for token in stop_tokens or ()}
        generated: list[Token] = []
        self.drafter.reset(prompt)

        verify_calls = 0
        next_token_calls = 0
        candidate_rounds = 0
        accepted_draft_tokens = 0
        accepted_draft_spans = 0
        rejected_candidates = 0
        stopped = False
        probe_remaining = min(self.reentry_probe_tokens, max_tokens) if self.drafter.corpus else 0

        while len(generated) < max_tokens and not stopped:
            remaining = max_tokens - len(generated)
            prefix = prompt + tuple(generated)
            candidates = self.drafter.candidates(max_tokens=remaining, limit=self.candidate_limit)

            if candidates:
                fallback_residual: Optional[Token] = None
                committed = False
                for candidate in candidates:
                    candidate_rounds += 1
                    accepted, residual, bonus = self._verify(prefix, candidate.tokens)
                    verify_calls += 1 if hasattr(self.service, "verify") else 0
                    if not hasattr(self.service, "verify"):
                        next_token_calls += min(
                            len(candidate.tokens),
                            accepted + (1 if residual is not None else 0),
                        )

                    if accepted > 0:
                        accepted_tokens = candidate.tokens[:accepted]
                        visible_tokens, stopped = _without_stop_tokens(accepted_tokens, stop_set)
                        committed_tokens = list(visible_tokens)
                        continuation = residual if residual is not None else bonus
                        if not stopped and continuation is not None and len(generated) + len(committed_tokens) < max_tokens:
                            if _is_stop_token(continuation, stop_set):
                                stopped = True
                            else:
                                committed_tokens.append(int(continuation))
                        if visible_tokens:
                            accepted_draft_tokens += len(visible_tokens)
                            accepted_draft_spans += 1
                        if committed_tokens:
                            committed_chunk = tuple(committed_tokens)
                            generated.extend(committed_chunk)
                            self.drafter.observe(committed_chunk)
                            if on_tokens is not None:
                                on_tokens(committed_chunk)
                        committed = True
                        break

                    rejected_candidates += 1
                    if fallback_residual is None and residual is not None:
                        fallback_residual = residual

                if committed:
                    probe_remaining = min(self.reentry_probe_tokens, max_tokens - len(generated))
                    continue

                if fallback_residual is not None:
                    if _is_stop_token(fallback_residual, stop_set):
                        break
                    generated.append(fallback_residual)
                    self.drafter.observe((fallback_residual,))
                    if on_tokens is not None:
                        on_tokens((fallback_residual,))
                    probe_remaining = min(self.reentry_probe_tokens, max_tokens - len(generated))
                    continue

            if not candidates and probe_remaining > 0:
                token = self.service.next_token(prefix)
                next_token_calls += 1
                probe_remaining -= 1
                if token is None or _is_stop_token(token, stop_set):
                    break
                generated.append(int(token))
                self.drafter.observe((int(token),))
                if on_tokens is not None:
                    on_tokens((int(token),))
                continue

            continue_tokens = getattr(self.service, "continue_tokens", None)
            if not candidates and callable(continue_tokens):
                tail = continue_tokens(
                    prefix,
                    max_tokens=remaining,
                    stop_tokens=stop_set,
                    on_tokens=on_tokens,
                )
                if tail is not None:
                    tail = tuple(int(token) for token in tail)
                    generated.extend(tail)
                    self.drafter.observe(tail)
                    next_token_calls += len(tail)
                    break

            token = self.service.next_token(prefix)
            next_token_calls += 1
            if token is None:
                break
            if _is_stop_token(token, stop_set):
                break
            generated.append(int(token))
            self.drafter.observe((int(token),))
            if on_tokens is not None:
                on_tokens((int(token),))

        target_calls = verify_calls + next_token_calls
        stats = RunStats(
            generated_tokens=len(generated),
            baseline_target_calls=len(generated),
            target_calls=target_calls,
            verify_calls=verify_calls,
            next_token_calls=next_token_calls,
            candidate_rounds=candidate_rounds,
            accepted_draft_tokens=accepted_draft_tokens,
            accepted_draft_spans=accepted_draft_spans,
            rejected_candidates=rejected_candidates,
        )
        return tuple(generated), stats

    def _verify(
        self,
        prefix: Tuple[Token, ...],
        candidate: Tuple[Token, ...],
    ) -> Tuple[int, Optional[Token], Optional[Token]]:
        verification = getattr(self.service, "verification", None)
        if callable(verification):
            detail = verification(prefix, candidate)
            if hasattr(detail, "accepted") and hasattr(detail, "residual_token"):
                accepted = max(0, min(int(detail.accepted), len(candidate)))
                return accepted, detail.residual_token, getattr(detail, "bonus_token", None)

        verify = getattr(self.service, "verify", None)
        if callable(verify):
            result = verify(prefix, candidate)
            if isinstance(result, tuple):
                accepted, residual = result
                return max(0, min(int(accepted), len(candidate))), residual, None
            return max(0, min(int(result), len(candidate))), None, None

        accepted = 0
        for token in candidate:
            predicted = self.service.next_token(prefix + candidate[:accepted])
            if predicted is None:
                return accepted, None, None
            predicted = int(predicted)
            if predicted != token:
                return accepted, predicted, None
            accepted += 1
        return accepted, None, None


class MachBoost:
    def __init__(
        self,
        *,
        corpus_tokens: Optional[Iterable[Token]] = None,
        context_tokens: Optional[Iterable[Token]] = None,
        ngram: int = DEFAULT_NGRAM,
        max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
        max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
        candidate_limit: int = 1,
        reentry_probe_tokens: int = 0,
    ) -> None:
        corpus = corpus_tokens if corpus_tokens is not None else context_tokens
        if corpus is None:
            corpus = ()
        self.drafter = CorpusDrafter(
            corpus,
            ngram=ngram,
            max_suffix_tokens=max_suffix_tokens,
            max_draft_tokens=max_draft_tokens,
        )
        self.candidate_limit = max(1, int(candidate_limit))
        self.reentry_probe_tokens = max(0, int(reentry_probe_tokens))

    def wrap(self, service: StepService) -> BoostedService:
        return BoostedService(
            service,
            self.drafter,
            candidate_limit=self.candidate_limit,
            reentry_probe_tokens=self.reentry_probe_tokens,
        )

    def generate(
        self,
        service: StepService,
        prompt_tokens: Iterable[Token],
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]] = None,
        on_tokens: Optional[Callable[[Tuple[Token, ...]], None]] = None,
    ) -> Tuple[Tuple[Token, ...], RunStats]:
        return self.wrap(service).generate(
            prompt_tokens,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            on_tokens=on_tokens,
        )


def machboost(
    service: Optional[StepService] = None,
    *,
    corpus_tokens: Optional[Iterable[Token]] = None,
    context_tokens: Optional[Iterable[Token]] = None,
    ngram: int = DEFAULT_NGRAM,
    max_suffix_tokens: int = DEFAULT_MAX_SUFFIX_TOKENS,
    max_draft_tokens: int = DEFAULT_MAX_DRAFT_TOKENS,
    candidate_limit: int = 1,
    reentry_probe_tokens: int = 0,
) -> Union[MachBoost, BoostedService]:
    boost = MachBoost(
        corpus_tokens=corpus_tokens,
        context_tokens=context_tokens,
        ngram=ngram,
        max_suffix_tokens=max_suffix_tokens,
        max_draft_tokens=max_draft_tokens,
        candidate_limit=candidate_limit,
        reentry_probe_tokens=reentry_probe_tokens,
    )
    if service is None:
        return boost
    return boost.wrap(service)


def _is_stop_token(token: Token, stop_tokens: set[Token]) -> bool:
    return bool(stop_tokens) and int(token) in stop_tokens


def _without_stop_tokens(tokens: Iterable[Token], stop_tokens: set[Token]) -> Tuple[Tuple[Token, ...], bool]:
    visible: list[Token] = []
    for token in tokens:
        token = int(token)
        if _is_stop_token(token, stop_tokens):
            return tuple(visible), True
        visible.append(token)
    return tuple(visible), False


def _better(left: Candidate, right: Candidate) -> bool:
    return _sort_key(left) < _sort_key(right)


def _sort_key(candidate: Candidate) -> Tuple[float, int, int, int]:
    return (
        -candidate.score,
        -candidate.matched_suffix_tokens,
        -len(candidate.tokens),
        candidate.source_start,
    )
