from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Optional, Sequence


@dataclass(frozen=True)
class DFlashRunStats:
    generated_tokens: int
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    peak_memory_gb: float
    total_duration_seconds: float
    time_to_first_token_seconds: Optional[float]
    prompt_eval_seconds: float
    generation_seconds: float
    accepted_draft_tokens: int
    acceptance_ratio: float
    target_calls: int
    baseline_target_calls: int
    tokens_per_cycle: float
    backend: str = "dflash"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DFlashAccelerator:
    """Resident, target-verified block-diffusion decoder for MLX text models."""

    supports_vision = False
    _generation_lock = threading.RLock()
    _load_lock = threading.RLock()

    def __init__(
        self,
        bundle: Any,
        runtime_context: Any,
        *,
        stream_generate_fn: Callable[..., Iterable[Any]],
        stop_token_ids_fn: Callable[[Any], list[int]],
        token_event_type: type,
        summary_event_type: type,
    ) -> None:
        self.bundle = bundle
        self.runtime_context = runtime_context
        self.model = bundle.target_model
        self.tokenizer = bundle.tokenizer
        self.model_name = str(bundle.resolved_model_ref)
        self.draft_model_name = str(bundle.resolved_draft_ref)
        self._stream_generate = stream_generate_fn
        self._get_stop_token_ids = stop_token_ids_fn
        self._token_event_type = token_event_type
        self._summary_event_type = summary_event_type
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        draft_model: Optional[str] = None,
        draft_quant: Optional[str] = None,
        verify_mode: str = "dflash",
        lazy: bool = True,
    ) -> "DFlashAccelerator":
        try:
            import mlx.core as mx
            from dflash_mlx.engine.events import SummaryEvent, TokenEvent
            from dflash_mlx.model import DFlashDraftModelArgs
            from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
            from dflash_mlx.runtime.bundle import load_runtime_bundle
            from dflash_mlx.runtime.context import build_offline_runtime_context
        except ImportError as exc:
            raise ImportError(
                "DFlash decoding requires `pip install machboost[dflash]`."
            ) from exc

        runtime_context = build_offline_runtime_context(
            verify_mode=verify_mode,
            copyspec_mode="off",
        )
        with cls._load_lock:
            bundle = _load_runtime_bundle_compat(
                load_runtime_bundle,
                DFlashDraftModelArgs,
                model_ref=model_name,
                draft_ref=draft_model,
                draft_quant=draft_quant,
                verify_config=runtime_context.verify,
                lazy=lazy,
            )
        mx.eval(bundle.target_model.parameters(), bundle.draft_model.parameters())
        return cls(
            bundle,
            runtime_context,
            stream_generate_fn=stream_dflash_generate,
            stop_token_ids_fn=get_stop_token_ids,
            token_event_type=TokenEvent,
            summary_event_type=SummaryEvent,
        )

    def generate_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        generation_options: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> tuple[str, DFlashRunStats]:
        if context:
            raise ValueError(
                "DFlash repository context is not wired yet; include retrieved context in the messages."
            )
        options = dict(generation_options or {})
        effective_temperature = float(options.get("temperature", temperature))
        self._require_greedy(effective_temperature)
        prompt = self._chat_prompt(messages, enable_thinking=enable_thinking)
        return self._generate_prompt(prompt, max_tokens=max_tokens, on_text=on_text)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        generation_options: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> tuple[str, DFlashRunStats]:
        if context:
            raise ValueError(
                "DFlash repository context is not wired yet; include retrieved context in the prompt."
            )
        options = dict(generation_options or {})
        self._require_greedy(float(options.get("temperature", 0.0)))
        return self._generate_prompt(prompt, max_tokens=max_tokens, on_text=on_text)

    def _chat_prompt(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        enable_thinking: bool,
    ) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return str(self.tokenizer.apply_chat_template(list(messages), **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking")
            return str(self.tokenizer.apply_chat_template(list(messages), **kwargs))

    def _generate_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
    ) -> tuple[str, DFlashRunStats]:
        if self._closed:
            raise RuntimeError("DFlash accelerator is closed")
        if max_tokens < 1:
            return "", self._empty_stats()

        with self._generation_lock:
            started = time.perf_counter()
            first_token_at: Optional[float] = None
            summary = None
            detokenizer = self.tokenizer.detokenizer
            detokenizer.reset()
            stream = self._stream_generate(
                target_model=self.bundle.target_model,
                target_ops=self.bundle.target_ops,
                tokenizer=self.tokenizer,
                draft_model=self.bundle.draft_model,
                draft_backend=self.bundle.draft_backend,
                prompt=prompt,
                max_new_tokens=max_tokens,
                use_chat_template=False,
                stop_token_ids=self._get_stop_token_ids(self.tokenizer),
                runtime_context=self.runtime_context,
            )
            try:
                for event in stream:
                    if isinstance(event, self._token_event_type):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        detokenizer.add_token(int(event.token_id))
                        chunk = str(detokenizer.last_segment or "")
                        if chunk and on_text is not None:
                            on_text(chunk)
                    elif isinstance(event, self._summary_event_type):
                        summary = event
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

            detokenizer.finalize()
            tail = str(detokenizer.last_segment or "")
            if tail and on_text is not None:
                on_text(tail)
            finished = time.perf_counter()
            if summary is None:
                raise RuntimeError("DFlash generation ended without a summary event")
            text = str(detokenizer.text)
            return text, self._stats(
                summary,
                elapsed=finished - started,
                ttft=None if first_token_at is None else first_token_at - started,
            )

    def _stats(
        self,
        summary: Any,
        *,
        elapsed: float,
        ttft: Optional[float],
    ) -> DFlashRunStats:
        phase_us = dict(getattr(summary, "phase_timings_us", {}) or {})
        prefill_seconds = max(0.0, float(phase_us.get("prefill", 0.0)) / 1_000_000.0)
        measured_seconds = max(0.0, float(getattr(summary, "elapsed_us", 0.0)) / 1_000_000.0)
        generation_seconds = max(0.0, measured_seconds - prefill_seconds)
        generated = int(getattr(summary, "generation_tokens", 0) or 0)
        prompt_tokens = int(getattr(summary, "prompt_token_count", 0) or 0)
        cycles = int(getattr(summary, "cycles_completed", 0) or 0)
        return DFlashRunStats(
            generated_tokens=generated,
            prompt_tokens=prompt_tokens,
            prompt_tokens_per_second=(
                prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
            ),
            generation_tokens_per_second=(
                generated / generation_seconds if generation_seconds > 0 else 0.0
            ),
            peak_memory_gb=float(getattr(summary, "peak_memory_gb", 0.0) or 0.0),
            total_duration_seconds=elapsed,
            time_to_first_token_seconds=ttft,
            prompt_eval_seconds=prefill_seconds,
            generation_seconds=generation_seconds,
            accepted_draft_tokens=int(getattr(summary, "accepted_from_draft", 0) or 0),
            acceptance_ratio=float(getattr(summary, "acceptance_ratio", 0.0) or 0.0),
            target_calls=cycles,
            baseline_target_calls=generated,
            tokens_per_cycle=float(getattr(summary, "tokens_per_cycle", 0.0) or 0.0),
        )

    def _empty_stats(self) -> DFlashRunStats:
        return DFlashRunStats(
            generated_tokens=0,
            prompt_tokens=0,
            prompt_tokens_per_second=0.0,
            generation_tokens_per_second=0.0,
            peak_memory_gb=0.0,
            total_duration_seconds=0.0,
            time_to_first_token_seconds=None,
            prompt_eval_seconds=0.0,
            generation_seconds=0.0,
            accepted_draft_tokens=0,
            acceptance_ratio=0.0,
            target_calls=0,
            baseline_target_calls=0,
            tokens_per_cycle=0.0,
        )

    @staticmethod
    def _require_greedy(temperature: float) -> None:
        if temperature != 0.0:
            raise ValueError(
                "DFlash currently supports greedy decoding only; set temperature to 0."
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.bundle = None
        self.model = None
        self.tokenizer = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, RuntimeError):
            pass


def _load_runtime_bundle_compat(
    load_runtime_bundle: Callable[..., Any],
    draft_args_type: type,
    **kwargs: Any,
) -> Any:
    """Bridge current HF DFlash configs until the published runtime catches up."""
    descriptor = draft_args_type.__dict__["from_dict"]
    original = draft_args_type.from_dict

    def from_dict(cls, params):
        data = dict(params)
        dflash_config = dict(data.get("dflash_config") or {})
        rope_parameters = dict(data.get("rope_parameters") or {})
        if "block_size" not in data and "block_size" in dflash_config:
            data["block_size"] = dflash_config["block_size"]
        if "rope_theta" not in data and "rope_theta" in rope_parameters:
            data["rope_theta"] = rope_parameters["rope_theta"]
        return original(data)

    draft_args_type.from_dict = classmethod(from_dict)
    try:
        return load_runtime_bundle(**kwargs)
    finally:
        draft_args_type.from_dict = descriptor
