from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

from machboost.core import Token, TokenSeq


@dataclass(frozen=True)
class Verification:
    accepted: int
    residual_token: Optional[Token]


class MLXCausalLMService:
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        mx_module=None,
        min_verify_margin: float = 0.0,
        cache_enabled: bool = True,
        cache_factory: Optional[Callable[[object], object]] = None,
        cache_trimmer: Optional[Callable[[object, int], object]] = None,
        cache_can_trim: Optional[Callable[[object], bool]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx_module
        self.min_verify_margin = float(min_verify_margin)
        self.cache_enabled = cache_enabled
        self.cache_factory = cache_factory
        self.cache_trimmer = cache_trimmer
        self.cache_can_trim = cache_can_trim
        self.forward_calls = 0
        self._cache = None
        self._cache_prefix: Tuple[Token, ...] = ()
        self._cache_logits = None
        self._cache_supported: Optional[bool] = None
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
        adapter_path: Optional[str] = None,
        lazy: bool = False,
        revision: Optional[str] = None,
        min_verify_margin: float = 0.0,
        cache_enabled: bool = True,
    ) -> "MLXCausalLMService":
        try:
            from mlx_lm.utils import load
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc

        model, tokenizer = load(
            model_name_or_path,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            adapter_path=adapter_path,
            lazy=lazy,
            revision=revision,
        )
        return cls(model, tokenizer, min_verify_margin=min_verify_margin, cache_enabled=cache_enabled)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Tuple[Token, ...]:
        if self.tokenizer is None:
            raise ValueError("encode requires a tokenizer")
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            tokens = self.tokenizer.encode(text)
        return tuple(int(token) for token in tokens)

    def decode(self, tokens: Iterable[Token], *, skip_special_tokens: bool = True) -> str:
        if self.tokenizer is None:
            raise ValueError("decode requires a tokenizer")
        try:
            return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)
        except TypeError:
            return self.tokenizer.decode(list(tokens))

    def next_token(self, prefix_tokens: TokenSeq) -> Optional[Token]:
        if len(prefix_tokens) == 0:
            return None
        row = self._cached_next_logits(prefix_tokens)
        if row is not None:
            return self._argmax(row)
        logits = self._logits(prefix_tokens)
        return self._argmax(self._last_row(logits))

    def generate_tokens(
        self,
        prompt_tokens: TokenSeq,
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]] = None,
        on_tokens=None,
    ) -> Tuple[Token, ...]:
        if len(prompt_tokens) == 0 or max_tokens <= 0:
            return ()
        self.reset_cache()
        stop_set = {int(token) for token in stop_tokens or ()}
        generated: list[Token] = []
        for _ in range(max_tokens):
            token = self.next_token(tuple(prompt_tokens) + tuple(generated))
            if token is None:
                break
            token = int(token)
            if token in stop_set:
                break
            generated.append(token)
            if on_tokens is not None:
                on_tokens((token,))
        return tuple(generated)

    def verify(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Tuple[int, Optional[Token]]:
        result = self.verification(prefix_tokens, candidate_tokens)
        return result.accepted, result.residual_token

    def verification(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Verification:
        if len(prefix_tokens) == 0 or len(candidate_tokens) == 0:
            return Verification(0, None)

        prefix = tuple(int(token) for token in prefix_tokens)
        candidate = tuple(int(token) for token in candidate_tokens)
        cached_logits = self._cached_next_logits(prefix)
        if cached_logits is not None:
            return self._verification_cached(prefix, candidate, cached_logits)

        return self._verification_stateless(prefix, candidate)

    def reset_cache(self) -> None:
        self._cache = None
        self._cache_prefix = ()
        self._cache_logits = None

    def _verification_stateless(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Verification:
        sequence = tuple(int(token) for token in prefix_tokens) + tuple(int(token) for token in candidate_tokens)
        logits = self._logits(sequence)
        start = len(prefix_tokens) - 1
        accepted = 0
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens):
            row = self._row(logits, start + offset)
            predicted = self._argmax(row)
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        return Verification(accepted, residual)

    def _verification_cached(
        self,
        prefix_tokens: Tuple[Token, ...],
        candidate_tokens: Tuple[Token, ...],
        current_logits,
    ) -> Verification:
        predicted = self._argmax(current_logits)
        first_candidate = int(candidate_tokens[0])
        if predicted != first_candidate or not self._passes_margin(current_logits, first_candidate):
            return Verification(0, predicted)

        trial_cache = self._clone_cache(self._cache)
        if trial_cache is None:
            return self._verification_cached_mutating(prefix_tokens, candidate_tokens, current_logits)

        logits = self._logits(candidate_tokens, cache=trial_cache)
        accepted = 1
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens[1:], start=1):
            row = self._row(logits, offset - 1)
            predicted = self._argmax(row)
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        committed = candidate_tokens[:accepted]
        rejected = len(candidate_tokens) - accepted
        if rejected == 0 or self._trim_cache(trial_cache, rejected):
            self._cache = trial_cache
            self._cache_prefix = prefix_tokens + committed
            self._cache_logits = self._row(logits, accepted - 1)
        else:
            commit_logits = self._logits(committed, cache=self._cache)
            self._cache_prefix = prefix_tokens + committed
            self._cache_logits = self._row(commit_logits, accepted - 1)

        return Verification(accepted, residual)

    def _verification_cached_mutating(
        self,
        prefix_tokens: Tuple[Token, ...],
        candidate_tokens: Tuple[Token, ...],
        current_logits,
    ) -> Verification:
        predicted = self._argmax(current_logits)
        first_candidate = int(candidate_tokens[0])
        if predicted != first_candidate or not self._passes_margin(current_logits, first_candidate):
            return Verification(0, predicted)

        trial_cache = self._cache
        logits = self._logits(candidate_tokens, cache=trial_cache)
        accepted = 1
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens[1:], start=1):
            row = self._row(logits, offset - 1)
            predicted = self._argmax(row)
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        committed = candidate_tokens[:accepted]
        rejected = len(candidate_tokens) - accepted
        if rejected > 0 and not self._trim_cache(trial_cache, rejected):
            self.reset_cache()
            self._cached_next_logits(prefix_tokens + committed)
        else:
            self._cache = trial_cache
            self._cache_prefix = prefix_tokens + committed
            self._cache_logits = self._row(logits, accepted - 1)

        return Verification(accepted, residual)

    def _clone_cache(self, cache):
        if cache is None:
            return None
        cloned = []
        for item in cache:
            clone = self._clone_cache_item(item)
            if clone is None:
                return None
            cloned.append(clone)
        return cloned

    def _clone_cache_item(self, item):
        from_state = getattr(item.__class__, "from_state", None)
        if not callable(from_state) or not hasattr(item, "state"):
            return None
        state = item.state
        if isinstance(state, list):
            state = list(state)
        meta_state = getattr(item, "meta_state", None)
        try:
            return item.__class__.from_state(state, meta_state)
        except TypeError:
            return None

    def _logits(self, tokens: Sequence[Token], *, cache=None):
        mx = self._mx()
        input_ids = self._array([[int(token) for token in tokens]], mx)
        if cache is None:
            logits = self.model(input_ids)
        else:
            logits = self.model(input_ids, cache=cache)
        self.forward_calls += 1
        if hasattr(mx, "eval"):
            self._eval(mx, logits, cache)
        return logits

    def _eval(self, mx, logits, cache) -> None:
        if cache is None:
            mx.eval(logits)
            return
        try:
            mx.eval(logits, [item.state for item in cache])
        except Exception:
            mx.eval(logits)

    def _cached_next_logits(self, prefix_tokens: TokenSeq):
        prefix = tuple(int(token) for token in prefix_tokens)
        if len(prefix) == 0 or not self._can_use_cache():
            return None
        if self._cache is not None:
            if prefix == self._cache_prefix:
                return self._cache_logits
            if len(prefix) > len(self._cache_prefix) and prefix[: len(self._cache_prefix)] == self._cache_prefix:
                delta = prefix[len(self._cache_prefix) :]
                logits = self._logits(delta, cache=self._cache)
                self._cache_prefix = prefix
                self._cache_logits = self._last_row(logits)
                return self._cache_logits

        self.reset_cache()
        try:
            self._cache = self._new_cache()
            logits = self._logits(prefix, cache=self._cache)
        except (AttributeError, ImportError, TypeError):
            self._cache_supported = False
            self.reset_cache()
            return None
        self._cache_prefix = prefix
        self._cache_logits = self._last_row(logits)
        return self._cache_logits

    def _can_use_cache(self) -> bool:
        if not self.cache_enabled:
            return False
        if self._cache_supported is not None:
            return self._cache_supported
        self._cache_supported = bool(
            self.cache_factory is not None
            or hasattr(self.model, "make_cache")
            or hasattr(self.model, "layers")
        )
        return self._cache_supported

    def _new_cache(self):
        if self.cache_factory is not None:
            return self.cache_factory(self.model)
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc
        return make_prompt_cache(self.model)

    def _trim_cache(self, cache, num_tokens: int) -> bool:
        if num_tokens <= 0:
            return True
        if self.cache_can_trim is not None and not self.cache_can_trim(cache):
            return False
        if self.cache_trimmer is not None:
            self.cache_trimmer(cache, num_tokens)
            return True
        try:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        except ImportError:
            return False
        if not can_trim_prompt_cache(cache):
            return False
        trim_prompt_cache(cache, num_tokens)
        return True

    def _mx(self):
        if self.mx is not None:
            return self.mx
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc
        self.mx = mx
        return mx

    def _array(self, values, mx):
        dtype = getattr(mx, "int32", None)
        if dtype is None:
            return mx.array(values)
        return mx.array(values, dtype=dtype)

    def _row(self, logits, pos: int):
        try:
            return logits[0, pos]
        except (TypeError, IndexError):
            return logits[0][pos]

    def _last_row(self, logits):
        return self._row(logits, -1)

    def _argmax(self, row) -> Token:
        mx = self._mx()
        if hasattr(mx, "argmax"):
            value = mx.argmax(row)
            if hasattr(value, "item"):
                return int(value.item())
            return int(value)
        return max(range(len(row)), key=lambda i: row[i])

    def _passes_margin(self, row, token: Token) -> bool:
        if self.min_verify_margin <= 0:
            return True
        predicted = self._argmax(row)
        if predicted != int(token):
            return False
        top = sorted((float(v) for v in row), reverse=True)[:2]
        if len(top) < 2:
            return True
        return top[0] - top[1] >= self.min_verify_margin
