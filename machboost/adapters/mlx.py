from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from machboost.core import Token, TokenSeq


def _optional_float(value):
    return None if value is None else float(value)


@dataclass(frozen=True)
class Verification:
    accepted: int
    residual_token: Optional[Token]
    bonus_token: Optional[Token] = None


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
        native_prompt_cache_size: int = 0,
        native_prompt_cache_bytes: int = 2 * 1024 * 1024 * 1024,
        native_prompt_cache_namespace: str = "default",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx_module
        self.min_verify_margin = float(min_verify_margin)
        self.cache_enabled = cache_enabled
        self.cache_factory = cache_factory
        self.cache_trimmer = cache_trimmer
        self.cache_can_trim = cache_can_trim
        self.native_prompt_cache_size = max(0, int(native_prompt_cache_size))
        self.native_prompt_cache_bytes = max(0, int(native_prompt_cache_bytes))
        self.native_prompt_cache_namespace = str(native_prompt_cache_namespace or "default")
        self.forward_calls = 0
        self._cache = None
        self._cache_prefix: Tuple[Token, ...] = ()
        self._cache_logits = None
        self._cache_supported: Optional[bool] = None
        self._native_prompt_cache = None
        self._native_prompt_cache_model_key = (
            type(model).__module__,
            type(model).__qualname__,
            id(model),
        )
        self._native_prompt_cache_supported: Optional[bool] = None
        self._last_native_metrics: dict[str, float | int | None] = {}
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
        on_text=None,
        generation_options: Optional[dict[str, Any]] = None,
    ) -> Tuple[Token, ...]:
        if len(prompt_tokens) == 0 or max_tokens <= 0:
            return ()
        if self.tokenizer is not None and self.cache_factory is None:
            return self._generate_tokens_native(
                prompt_tokens,
                max_tokens=max_tokens,
                stop_tokens=stop_tokens,
                on_tokens=on_tokens,
                on_text=on_text,
                generation_options=generation_options,
            )

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

    def continue_tokens(
        self,
        prefix_tokens: TokenSeq,
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]] = None,
        on_tokens=None,
    ) -> Optional[Tuple[Token, ...]]:
        prefix = tuple(int(token) for token in prefix_tokens)
        if max_tokens <= 0:
            return ()
        if (
            self._cache is None
            or len(prefix) != len(self._cache_prefix) + 1
            or prefix[:-1] != self._cache_prefix
        ):
            return None

        try:
            from mlx_lm.generate import generate_step
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc

        mx = self._mx()
        stop_set = {int(token) for token in stop_tokens or ()}
        prompt = self._array([prefix[-1]], mx)
        generated: list[Token] = []
        try:
            for token, _ in generate_step(
                prompt,
                self.model,
                max_tokens=max_tokens,
                prompt_cache=self._cache,
            ):
                token = int(token)
                if token in stop_set:
                    break
                generated.append(token)
                self.forward_calls += 1
                if on_tokens is not None:
                    on_tokens((token,))
        finally:
            self.reset_cache()
        return tuple(generated)

    def _generate_tokens_native(
        self,
        prompt_tokens: TokenSeq,
        *,
        max_tokens: int,
        stop_tokens: Optional[Iterable[Token]],
        on_tokens,
        on_text,
        generation_options: Optional[dict[str, Any]],
    ) -> Tuple[Token, ...]:
        try:
            from mlx_lm import stream_generate
        except ImportError as exc:
            raise ImportError("Install MLX support with `pip install machboost[mlx]`.") from exc

        self.reset_cache()
        self._last_native_metrics = {}
        stop_set = {int(token) for token in stop_tokens or ()}
        generated: list[Token] = []
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        last_response = None
        full_prompt = [int(token) for token in prompt_tokens]
        prompt = full_prompt
        prompt_cache = None
        prompt_cache_store = self._native_prompt_cache_store()
        cached_prompt_tokens = 0
        if prompt_cache_store is not None:
            prompt_cache, prompt = prompt_cache_store.fetch_nearest_cache(
                self._native_prompt_cache_key,
                full_prompt,
            )
            cached_prompt_tokens = len(full_prompt) - len(prompt)
            if prompt_cache is None:
                try:
                    from mlx_lm.models.cache import make_prompt_cache

                    prompt_cache = make_prompt_cache(self.model)
                except (ImportError, RuntimeError, TypeError):
                    prompt_cache_store = None
                    prompt_cache = None
                    prompt = full_prompt
                    cached_prompt_tokens = 0
        cache_key = list(full_prompt)
        stream_kwargs = {"max_tokens": max_tokens}
        generation_options = dict(generation_options or {})
        if generation_options:
            try:
                from mlx_lm.sample_utils import make_logits_processors, make_sampler
            except ImportError as exc:
                raise ImportError(
                    "Installed mlx-lm does not expose sampling utilities; upgrade mlx-lm."
                ) from exc
            stream_kwargs["sampler"] = make_sampler(
                temp=float(generation_options.get("temperature", 0.0)),
                top_p=float(generation_options.get("top_p", 0.0)),
                min_p=float(generation_options.get("min_p", 0.0)),
                top_k=int(generation_options.get("top_k", 0)),
            )
            processors = make_logits_processors(
                repetition_penalty=_optional_float(
                    generation_options.get("repeat_penalty")
                ),
                repetition_context_size=int(
                    generation_options.get("repeat_last_n", 64)
                ),
                presence_penalty=_optional_float(
                    generation_options.get("presence_penalty")
                ),
                presence_context_size=int(
                    generation_options.get("repeat_last_n", 64)
                ),
                frequency_penalty=_optional_float(
                    generation_options.get("frequency_penalty")
                ),
                frequency_context_size=int(
                    generation_options.get("repeat_last_n", 64)
                ),
            )
            if processors:
                stream_kwargs["logits_processors"] = processors
            if generation_options.get("seed") is not None:
                try:
                    import mlx.core as mx

                    mx.random.seed(int(generation_options["seed"]))
                except (ImportError, AttributeError, TypeError, ValueError):
                    pass
        if prompt_cache_store is not None and prompt_cache is not None:
            stream_kwargs["prompt_cache"] = prompt_cache
        try:
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                **stream_kwargs,
            ):
                last_response = response
                token = int(response.token)
                cache_key.append(token)
                if token not in stop_set and first_token_at is None:
                    first_token_at = time.perf_counter()
                text = str(getattr(response, "text", "") or "")
                if on_text is not None and text:
                    on_text(text)
                if token in stop_set:
                    break
                generated.append(token)
                self.forward_calls += 1
                if on_tokens is not None and on_text is None:
                    on_tokens((token,))
        finally:
            if prompt_cache_store is not None and prompt_cache is not None:
                prompt_cache_store.insert_cache(
                    self._native_prompt_cache_key,
                    cache_key,
                    prompt_cache,
                )
            elapsed = max(0.0, time.perf_counter() - started)
            backend_prompt_count = int(
                getattr(last_response, "prompt_tokens", 0) or len(prompt)
            )
            prompt_count = (
                len(full_prompt)
                if prompt_cache_store is not None
                else backend_prompt_count
            )
            evaluated_prompt_count = (
                len(prompt)
                if prompt_cache_store is not None
                else backend_prompt_count
            )
            generation_count = int(
                getattr(last_response, "generation_tokens", 0) or len(generated)
            )
            prompt_tps = float(getattr(last_response, "prompt_tps", 0.0) or 0.0)
            generation_tps = float(
                getattr(last_response, "generation_tps", 0.0) or 0.0
            )
            ttft = None if first_token_at is None else first_token_at - started
            prompt_seconds = (
                evaluated_prompt_count / prompt_tps
                if prompt_tps > 0
                else (ttft or 0.0)
            )
            generation_seconds = (
                generation_count / generation_tps
                if generation_tps > 0
                else max(0.0, elapsed - (ttft or 0.0))
            )
            self._last_native_metrics = {
                "prompt_tokens": prompt_count,
                "prompt_eval_tokens": evaluated_prompt_count,
                "cached_prompt_tokens": cached_prompt_tokens,
                "prompt_eval_seconds": prompt_seconds,
                "generation_seconds": generation_seconds,
                "time_to_first_token_seconds": ttft,
                "prompt_tokens_per_second": prompt_tps,
                "generation_tokens_per_second": generation_tps,
                "prompt_cache_namespace": self.native_prompt_cache_namespace,
            }
        return tuple(generated)

    def clear_prompt_cache(self) -> None:
        self._native_prompt_cache = None

    def configure_native_prompt_cache(
        self,
        *,
        enabled: bool,
        max_size: int = 8,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        namespace: str = "default",
    ) -> None:
        next_size = max(0, int(max_size)) if enabled else 0
        next_bytes = max(0, int(max_bytes)) if enabled else 0
        if (
            next_size != self.native_prompt_cache_size
            or next_bytes != self.native_prompt_cache_bytes
        ):
            self.clear_prompt_cache()
        self.native_prompt_cache_size = next_size
        self.native_prompt_cache_bytes = next_bytes
        self.native_prompt_cache_namespace = str(namespace or "default")

    @property
    def _native_prompt_cache_key(self):
        return (*self._native_prompt_cache_model_key, self.native_prompt_cache_namespace)

    def _native_prompt_cache_store(self):
        if self.native_prompt_cache_size <= 0 or self.native_prompt_cache_bytes <= 0:
            return None
        if self._native_prompt_cache_supported is False:
            return None
        if self._native_prompt_cache is not None:
            return self._native_prompt_cache
        try:
            from mlx_lm.models.cache import LRUPromptCache

            self._native_prompt_cache = LRUPromptCache(
                max_size=self.native_prompt_cache_size,
                max_bytes=self.native_prompt_cache_bytes,
            )
        except (ImportError, RuntimeError, TypeError):
            self._native_prompt_cache_supported = False
            return None
        self._native_prompt_cache_supported = True
        return self._native_prompt_cache

    @property
    def last_native_metrics(self) -> dict[str, float | int | None]:
        return dict(self._last_native_metrics)

    @property
    def supports_native_text_streaming(self) -> bool:
        return self.tokenizer is not None and self.cache_factory is None

    def verify(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Tuple[int, Optional[Token]]:
        result = self.verification(prefix_tokens, candidate_tokens)
        return result.accepted, result.residual_token

    def verification(self, prefix_tokens: TokenSeq, candidate_tokens: TokenSeq) -> Verification:
        if len(prefix_tokens) == 0 or len(candidate_tokens) == 0:
            return Verification(0, None)

        prefix = tuple(int(token) for token in prefix_tokens)
        candidate = tuple(int(token) for token in candidate_tokens)
        if (
            self._cache is not None
            and len(prefix) == len(self._cache_prefix) + 1
            and prefix[:-1] == self._cache_prefix
        ):
            return self._verification_cached_extension(prefix, candidate)

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

        bonus = self._argmax(self._row(logits, start + accepted)) if accepted == len(candidate_tokens) else None
        return Verification(accepted, residual, bonus)

    def _verification_cached_extension(
        self,
        prefix_tokens: Tuple[Token, ...],
        candidate_tokens: Tuple[Token, ...],
    ) -> Verification:
        trial_cache = self._cache
        input_tokens = prefix_tokens[-1:] + candidate_tokens
        logits = self._logits(input_tokens, cache=trial_cache)
        predictions = self._argmax_rows(logits, len(input_tokens))
        accepted = 0
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens):
            predicted = predictions[offset]
            row = self._row(logits, offset)
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        committed = candidate_tokens[:accepted]
        rejected = len(candidate_tokens) - accepted
        bonus = predictions[-1] if rejected == 0 else None
        if rejected > 0 and not self._trim_cache(trial_cache, rejected):
            self.reset_cache()
            self._cached_next_logits(prefix_tokens + committed)
        else:
            self._cache = trial_cache
            self._cache_prefix = prefix_tokens + committed
            self._cache_logits = self._row(logits, accepted)

        return Verification(accepted, residual, bonus)

    def _verification_cached(
        self,
        prefix_tokens: Tuple[Token, ...],
        candidate_tokens: Tuple[Token, ...],
        current_logits,
    ) -> Verification:
        return self._verification_cached_mutating(prefix_tokens, candidate_tokens, current_logits)

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
        predictions = self._argmax_rows(logits, len(candidate_tokens))
        accepted = 1
        residual: Optional[Token] = None

        for offset, candidate in enumerate(candidate_tokens[1:], start=1):
            row = self._row(logits, offset - 1)
            predicted = predictions[offset - 1]
            if predicted != int(candidate) or not self._passes_margin(row, int(candidate)):
                residual = predicted
                break
            accepted += 1

        committed = candidate_tokens[:accepted]
        rejected = len(candidate_tokens) - accepted
        bonus = predictions[-1] if rejected == 0 else None
        if rejected > 0 and not self._trim_cache(trial_cache, rejected):
            self.reset_cache()
            self._cached_next_logits(prefix_tokens + committed)
        else:
            self._cache = trial_cache
            self._cache_prefix = prefix_tokens + committed
            self._cache_logits = self._row(logits, accepted - 1)

        return Verification(accepted, residual, bonus)

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
            if len(prefix) > 1:
                self._logits(prefix[:-1], cache=self._cache)
                logits = self._logits(prefix[-1:], cache=self._cache)
            else:
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

    def _argmax_rows(self, logits, count: int) -> list[Token]:
        mx = self._mx()
        if count > 0 and hasattr(mx, "argmax"):
            try:
                values = mx.argmax(logits[0, :count], axis=-1)
                if hasattr(mx, "eval"):
                    mx.eval(values)
                return [int(value) for value in values.tolist()]
            except (AttributeError, IndexError, TypeError):
                pass
        return [self._argmax(self._row(logits, pos)) for pos in range(count)]

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
