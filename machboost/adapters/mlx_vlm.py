from __future__ import annotations

import importlib
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..vision import (
    ContentAddressedVisionCache,
    VisualAssetStore,
    normalize_multimodal_messages,
)
from ..vision_auto import choose_vision_token_policy
from ..vision_policy import choose_cold_vision
from ..vision_tokens import configure_post_fusion_vision


@dataclass(frozen=True)
class VisionRunStats:
    generated_tokens: int
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    peak_memory_gb: float
    total_duration_seconds: float
    time_to_first_token_seconds: Optional[float]
    visual_cache_enabled: bool
    visual_cache_hit: bool
    visual_cache_miss: bool
    visual_cache_entries: int
    visual_cache_hits_total: int
    visual_cache_misses_total: int
    prompt_cache_enabled: bool
    prompt_cache_prefix_tokens: int
    image_count: int
    cold_vision: dict[str, Any]
    post_fusion_vision: dict[str, Any]
    mean_token_logprob: Optional[float]
    minimum_token_logprob: Optional[float]
    backend: str = "mlx-vlm"
    accepted_draft_tokens: int = 0
    target_calls: int = 0
    baseline_target_calls: int = 0
    thinking: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThinkingDelta:
    reasoning: str = ""
    content: str = ""


class ThinkingStreamSplitter:
    """Separate model reasoning markers without leaking partial tags."""

    DEFAULT_MARKERS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
    )
    CONTENT_MARKERS = (
        "<|start|>assistant to=user<|message|>",
        "<|start|>assistant<|message|>",
        "<|START_TEXT|>",
        "<|END_TEXT|>",
        "<|eot|>",
    )

    def __init__(
        self,
        *,
        starts_in_thinking: bool = False,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
    ) -> None:
        markers: list[tuple[str, str]] = []
        if start_marker and end_marker:
            if start_marker.startswith("to="):
                markers.append((f"<|start|>assistant {start_marker}", end_marker))
            markers.append((start_marker, end_marker))
        markers.extend(pair for pair in self.DEFAULT_MARKERS if pair not in markers)
        self.markers = tuple(markers)
        self.open_markers = tuple(pair[0] for pair in self.markers)
        self.close_markers = tuple(pair[1] for pair in self.markers)
        self.in_thinking = starts_in_thinking
        self.thinking_done = False
        self.buffer = ""

    def feed(self, text: str, *, final: bool = False) -> ThinkingDelta:
        self.buffer += text
        reasoning: list[str] = []
        content: list[str] = []
        while self.buffer:
            if self.in_thinking:
                index, marker = self._find_first(self.buffer, self.close_markers)
                if index < 0:
                    value, self.buffer = self._split_partial(
                        self.buffer, self.close_markers, final=final
                    )
                    if value:
                        reasoning.append(self._strip_open_marker(value))
                    break
                value = self._strip_open_marker(self.buffer[:index])
                if value:
                    reasoning.append(value)
                self.buffer = self.buffer[index + len(marker) :].lstrip("\n")
                self.in_thinking = False
                self.thinking_done = True
                continue

            if self.thinking_done:
                value, self.buffer = self._split_partial(
                    self.buffer, self.CONTENT_MARKERS, final=final
                )
                if value:
                    content.append(self._clean_content(value))
                break

            index, marker = self._find_first(self.buffer, self.open_markers)
            if index < 0:
                value, self.buffer = self._split_partial(
                    self.buffer,
                    self.open_markers + self.CONTENT_MARKERS,
                    final=final,
                )
                if value:
                    content.append(self._clean_content(value))
                break
            if index:
                content.append(self._clean_content(self.buffer[:index]))
            self.buffer = self.buffer[index + len(marker) :].lstrip("\n")
            self.in_thinking = True

        return ThinkingDelta("".join(reasoning), "".join(content))

    @staticmethod
    def _find_first(text: str, markers: Sequence[str]) -> tuple[int, str]:
        matches = ((text.find(marker), marker) for marker in markers)
        return min(
            ((index, marker) for index, marker in matches if index >= 0),
            default=(-1, ""),
            key=lambda match: match[0],
        )

    @staticmethod
    def _split_partial(
        text: str, markers: Sequence[str], *, final: bool
    ) -> tuple[str, str]:
        if final or not markers:
            return text, ""
        hold = 0
        for marker in markers:
            for length in range(1, min(len(text), len(marker) - 1) + 1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def _strip_open_marker(self, text: str) -> str:
        for marker in self.open_markers:
            text = text.replace(marker, "")
        return text

    def _clean_content(self, text: str) -> str:
        for marker in self.CONTENT_MARKERS:
            text = text.replace(marker, "")
        return text


class InitialReasoningEchoFilter:
    """Remove prompt echoes from the beginning of a reasoning stream."""

    _PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
    _FRAGMENT_BREAK = re.compile(r"(?:\n+|(?<=[.!?])\s+)")

    def __init__(self, candidates: Sequence[str]) -> None:
        normalized = tuple(
            value
            for value in (_collapsed_whitespace(value) for value in candidates)
            if value
        )
        fragments = [
            fragment
            for value in candidates
            for fragment in (
                _collapsed_whitespace(part)
                for part in self._FRAGMENT_BREAK.split(value)
            )
            if len(fragment) >= 8
        ]
        self.candidates = tuple(
            sorted(dict.fromkeys((*normalized, *fragments)), key=len, reverse=True)
        )
        self.buffer = ""
        self.decided = not self.candidates

    def feed(self, text: str, *, final: bool = False) -> str:
        if self.decided:
            return text
        self.buffer += text
        match = self._PARAGRAPH_BREAK.search(self.buffer)
        if match is not None:
            paragraph = self.buffer[: match.start()]
            remainder = self.buffer[match.end() :]
            self.buffer = ""
            cleaned, removed = self._strip_leading_echo(paragraph)
            if removed and not cleaned:
                trailing = remainder.lstrip("\n")
                if trailing:
                    return self.feed(trailing, final=final)
                if final:
                    self.decided = True
                return ""
            self.decided = True
            if removed:
                return cleaned + self.buffered_separator(match) + remainder
            return paragraph + self.buffered_separator(match) + remainder

        current = _collapsed_whitespace(self.buffer)
        can_still_match = any(candidate.startswith(current) for candidate in self.candidates)
        already_continued = any(
            current.startswith(candidate) and current != candidate
            for candidate in self.candidates
        )
        if current and (not can_still_match or already_continued):
            cleaned, removed = self._strip_leading_echo(self.buffer)
            if removed:
                cleaned_current = _collapsed_whitespace(cleaned)
                if not cleaned_current or any(
                    candidate.startswith(cleaned_current)
                    for candidate in self.candidates
                ):
                    self.buffer = cleaned
                    return ""
                self.buffer = ""
                self.decided = True
                return cleaned
            return self._release()
        if final:
            cleaned, removed = self._strip_leading_echo(self.buffer)
            if removed:
                self.buffer = ""
                self.decided = True
                return cleaned
            return self._release()
        return ""

    def _strip_leading_echo(self, value: str) -> tuple[str, bool]:
        remainder = value
        removed = False
        while remainder:
            matched = False
            for candidate in self.candidates:
                words = candidate.split()
                pattern = r"^\s*" + r"\s+".join(re.escape(word) for word in words)
                match = re.match(pattern + r"(?=\s|$)", remainder)
                if match is None:
                    continue
                remainder = remainder[match.end() :].lstrip()
                removed = True
                matched = True
                break
            if not matched:
                break
        return remainder, removed

    def _release(self) -> str:
        value = self.buffer
        self.buffer = ""
        self.decided = True
        return value

    @staticmethod
    def buffered_separator(match: re.Match[str]) -> str:
        return match.group(0)

class MLXVLMAccelerator:
    """Resident MLX-VLM runner with content-addressed projected-feature reuse."""

    supports_vision = True
    _generation_lock = threading.RLock()

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        model_name: str,
        vision_cache_size: int = 20,
        asset_cache_dir: Optional[Path] = None,
        stream_generate_fn: Optional[Callable[..., Iterable[Any]]] = None,
        apply_chat_template_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        self._configure(
            model_name=model_name,
            vision_cache_size=vision_cache_size,
            asset_cache_dir=asset_cache_dir,
            stream_generate_fn=stream_generate_fn,
            apply_chat_template_fn=apply_chat_template_fn,
        )
        self.model = model
        self.processor = processor

    def _configure(
        self,
        *,
        model_name: str,
        vision_cache_size: int,
        asset_cache_dir: Optional[Path],
        stream_generate_fn: Optional[Callable[..., Iterable[Any]]],
        apply_chat_template_fn: Optional[Callable[..., str]],
    ) -> None:
        self.model_name = model_name
        self.revision: Optional[str] = None
        self.vision_cache = ContentAddressedVisionCache(max_size=vision_cache_size)
        self._prompt_caches: OrderedDict[str, Any] = OrderedDict()
        self._apc_manager: Any = None
        self.assets = VisualAssetStore(asset_cache_dir)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="machboost-vlm")
        self._closed = False
        if stream_generate_fn is None or apply_chat_template_fn is None:
            try:
                from mlx_vlm import stream_generate
                from mlx_vlm.prompt_utils import apply_chat_template
            except ImportError as exc:
                raise ImportError(
                    "MLX vision support requires `pip install machboost[vision]`."
                ) from exc
            stream_generate_fn = stream_generate_fn or stream_generate
            apply_chat_template_fn = apply_chat_template_fn or apply_chat_template
        self._stream_generate = stream_generate_fn
        self._apply_chat_template = apply_chat_template_fn

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        lazy: bool = False,
        revision: Optional[str] = None,
        vision_cache_size: int = 20,
        asset_cache_dir: Optional[Path] = None,
        **load_kwargs: Any,
    ) -> "MLXVLMAccelerator":
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise ImportError(
                "MLX vision support requires `pip install machboost[vision]`."
            ) from exc
        instance = cls.__new__(cls)
        instance._configure(
            model_name=model_name,
            vision_cache_size=vision_cache_size,
            asset_cache_dir=asset_cache_dir,
            stream_generate_fn=None,
            apply_chat_template_fn=None,
        )
        instance.revision = revision
        try:
            model, processor = instance._executor.submit(
                load,
                model_name,
                lazy=lazy,
                revision=revision,
                **load_kwargs,
            ).result()
        except Exception:
            instance._closed = True
            instance._executor.shutdown(wait=True, cancel_futures=True)
            raise
        instance.model = model
        instance.processor = processor
        return instance

    def generate_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        use_vision_cache: bool = True,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        cold_vision_mode: str = "off",
        cold_vision_max_edge: Optional[int] = None,
        vision_token_mode: str = "off",
        vision_token_ratio: float = 0.35,
        vision_token_layer: Optional[int] = None,
        vision_token_bucket: Optional[int] = None,
        vision_calibration: Optional[dict[str, Any]] = None,
        tools: Optional[Sequence[dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        reasoning_strength: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> tuple[str, VisionRunStats]:
        normalized, image_sources = normalize_multimodal_messages(messages)
        images = self.assets.materialize_all(image_sources)
        prompt = self._format_chat_prompt(
            normalized,
            image_count=len(images),
            enable_thinking=enable_thinking,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_strength=reasoning_strength,
        )
        return self._generate(
            prompt,
            images=images,
            max_tokens=max_tokens,
            on_text=on_text,
            on_thinking=on_thinking,
            enable_thinking=enable_thinking,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
            cold_vision_mode=cold_vision_mode,
            cold_vision_max_edge=cold_vision_max_edge,
            vision_token_mode=vision_token_mode,
            vision_token_ratio=vision_token_ratio,
            vision_token_layer=vision_token_layer,
            vision_token_bucket=vision_token_bucket,
            vision_calibration=vision_calibration,
            policy_prompt=_latest_user_text(normalized),
            reasoning_echoes=_user_texts(normalized),
            cache_key=cache_key,
            thinking_budget=(
                _thinking_budget(reasoning_strength) if enable_thinking else None
            ),
        )

    def _format_chat_prompt(
        self,
        messages: Sequence[dict[str, str]],
        *,
        image_count: int,
        enable_thinking: bool = False,
        tools: Optional[Sequence[dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        reasoning_strength: Optional[str] = None,
    ) -> str:
        model_type = config_value(self.model.config, "model_type", "")
        module_name = str(getattr(self._apply_chat_template, "__module__", ""))
        qwen_types = {
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        }
        template_options: dict[str, Any] = {
            "enable_thinking": enable_thinking,
        }
        if tools:
            template_options["tools"] = list(tools)
            template_options["tool_choice"] = tool_choice
        if reasoning_strength:
            template_options["reasoning_strength"] = reasoning_strength

        if image_count < 1 or model_type not in qwen_types or not module_name.startswith("mlx_vlm"):
            return self._apply_chat_template(
                self.processor,
                self.model.config,
                messages,
                num_images=image_count,
                **template_options,
            )

        prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")
        image_owner = next(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") not in {"system", "assistant", "tool"}
            ),
            len(messages) - 1,
        )
        formatted = []
        for index, message in enumerate(messages):
            role = str(message.get("role") or "user")
            formatted.append(
                prompt_utils.get_message_json(
                    model_type,
                    str(message.get("content") or ""),
                    role,
                    skip_image_token=index != image_owner,
                    num_images=image_count,
                )
            )
        return prompt_utils.get_chat_template(
            self.processor,
            formatted,
            add_generation_prompt=True,
            **template_options,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        images: Optional[Sequence[str]] = None,
        use_vision_cache: bool = True,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        cold_vision_mode: str = "off",
        cold_vision_max_edge: Optional[int] = None,
        vision_token_mode: str = "off",
        vision_token_ratio: float = 0.35,
        vision_token_layer: Optional[int] = None,
        vision_token_bucket: Optional[int] = None,
        vision_calibration: Optional[dict[str, Any]] = None,
        cache_key: Optional[str] = None,
    ) -> tuple[str, VisionRunStats]:
        materialized = self.assets.materialize_all(images or ())
        templated = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=len(materialized),
            enable_thinking=enable_thinking,
        )
        return self._generate(
            templated,
            images=materialized,
            max_tokens=max_tokens,
            on_text=on_text,
            on_thinking=on_thinking,
            enable_thinking=enable_thinking,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
            cold_vision_mode=cold_vision_mode,
            cold_vision_max_edge=cold_vision_max_edge,
            vision_token_mode=vision_token_mode,
            vision_token_ratio=vision_token_ratio,
            vision_token_layer=vision_token_layer,
            vision_token_bucket=vision_token_bucket,
            vision_calibration=vision_calibration,
            policy_prompt=prompt,
            reasoning_echoes=(prompt,),
            cache_key=cache_key,
            thinking_budget=None,
        )

    def _generate(
        self,
        prompt: str,
        *,
        images: Sequence[str],
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
        on_thinking: Optional[Callable[[str], None]],
        enable_thinking: bool,
        use_vision_cache: bool,
        temperature: float,
        cold_vision_mode: str,
        cold_vision_max_edge: Optional[int],
        vision_token_mode: str,
        vision_token_ratio: float,
        vision_token_layer: Optional[int],
        vision_token_bucket: Optional[int],
        vision_calibration: Optional[dict[str, Any]],
        policy_prompt: str,
        reasoning_echoes: Sequence[str],
        cache_key: Optional[str],
        thinking_budget: Optional[int],
    ) -> tuple[str, VisionRunStats]:
        if self._closed:
            raise RuntimeError("MLX vision accelerator is closed")
        future = self._executor.submit(
            self._generate_on_worker,
            prompt,
            images=images,
            max_tokens=max_tokens,
            on_text=on_text,
            on_thinking=on_thinking,
            enable_thinking=enable_thinking,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
            cold_vision_mode=cold_vision_mode,
            cold_vision_max_edge=cold_vision_max_edge,
            vision_token_mode=vision_token_mode,
            vision_token_ratio=vision_token_ratio,
            vision_token_layer=vision_token_layer,
            vision_token_bucket=vision_token_bucket,
            vision_calibration=vision_calibration,
            policy_prompt=policy_prompt,
            reasoning_echoes=reasoning_echoes,
            cache_key=cache_key,
            thinking_budget=thinking_budget,
        )
        return future.result()

    def _generate_on_worker(
        self,
        prompt: str,
        *,
        images: Sequence[str],
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
        on_thinking: Optional[Callable[[str], None]],
        enable_thinking: bool,
        use_vision_cache: bool,
        temperature: float,
        cold_vision_mode: str,
        cold_vision_max_edge: Optional[int],
        vision_token_mode: str,
        vision_token_ratio: float,
        vision_token_layer: Optional[int],
        vision_token_bucket: Optional[int],
        vision_calibration: Optional[dict[str, Any]],
        policy_prompt: str,
        reasoning_echoes: Sequence[str],
        cache_key: Optional[str],
        thinking_budget: Optional[int],
    ) -> tuple[str, VisionRunStats]:
        with self._generation_lock:
            self._bind_thread_local_stream()
            cold_vision = choose_cold_vision(
                policy_prompt,
                images,
                mode=cold_vision_mode,
                max_edge=cold_vision_max_edge,
            )
            vision_token_decision = choose_vision_token_policy(
                policy_prompt,
                images,
                mode=vision_token_mode,
                retain_ratio=vision_token_ratio,
                prune_after_layer=vision_token_layer,
                token_bucket=vision_token_bucket,
                calibration=vision_calibration,
            )
            post_fusion = configure_post_fusion_vision(
                self.model,
                mode=vision_token_decision.mode,
                retain_ratio=vision_token_decision.retain_ratio,
                prune_after_layer=vision_token_decision.prune_after_layer,
                token_bucket=vision_token_decision.token_bucket,
                policy=vision_token_decision.to_dict(),
            )
            post_fusion_enabled = (
                post_fusion is not None and post_fusion.mode != "off"
            )
            if post_fusion_enabled and cold_vision.mode != "off":
                raise ValueError(
                    "post-fusion visual tokens cannot be combined with cold vision resizing"
                )
            effective_vision_cache = use_vision_cache and not post_fusion_enabled
            cache_before = self.vision_cache.info()
            started = time.perf_counter()
            first_text_at: Optional[float] = None
            parts: list[str] = []
            thinking_parts: list[str] = []
            token_logprobs: list[float] = []
            generated_token_ids: list[int] = []
            observed_generation_steps: set[int] = set()
            last: Any = None
            stream_image: Optional[list[str]] = list(images) or None
            stream_options: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if thinking_budget is not None:
                stream_options["thinking_budget"] = thinking_budget
                if config_value(self.model.config, "model_type", "") == "muse_glimmer":
                    stream_options["thinking_start_token"] = "to=self"
                    stream_options["thinking_end_token"] = "<|eom|>"
            prompt_cache_enabled = False
            prompt_cache_prefix_tokens = 0
            module_name = str(getattr(self._stream_generate, "__module__", ""))
            if not images and cache_key and module_name.startswith("mlx_vlm"):
                stream_options["prompt_cache_state"] = self._prompt_cache_for_key(
                    f"affinity:{cache_key}"
                )
                prompt_cache_enabled = True
            apc_manager = None
            apc_matched_before = 0
            prepared = (
                self._prepare_cached_vision(
                    prompt,
                    images,
                    resize_shape=cold_vision.resize_shape,
                )
                if effective_vision_cache
                else None
            )
            if prepared is not None:
                stream_image, prepared_options, prompt_cache_prefix_tokens = prepared
                stream_options.update(prepared_options)
                prompt_cache_enabled = True
                apc_manager = prepared_options.get("apc_manager")
            elif effective_vision_cache and cold_vision.resize_shape is None:
                stream_options["vision_cache"] = self.vision_cache
            if (
                apc_manager is None
                and module_name.startswith("mlx_vlm")
            ):
                try:
                    apc_manager = self._get_apc_manager()
                except (AttributeError, ImportError):
                    apc_manager = None
                if apc_manager is not None:
                    stream_options["apc_manager"] = apc_manager
                    prompt_cache_enabled = True
            if apc_manager is not None:
                if cache_key:
                    stream_options["apc_tenant"] = cache_key
                apc_matched_before = int(
                    apc_manager.stats_snapshot().get("matched_tokens", 0)
                )
            if prepared is None and cold_vision.resize_shape is not None:
                stream_options["resize_shape"] = cold_vision.resize_shape
            rows = self._stream_generate(
                self.model,
                self.processor,
                prompt,
                image=stream_image,
                **stream_options,
            )
            thinking_start = config_value(
                self.model.config, "thinking_start_token", None
            )
            thinking_end = config_value(
                self.model.config, "thinking_end_token", None
            )
            splitter = ThinkingStreamSplitter(
                starts_in_thinking=bool(enable_thinking)
                and thinking_start in {None, "<think>", "<|START_THINKING|>"},
                start_marker=thinking_start,
                end_marker=thinking_end,
            )
            echo_filter = InitialReasoningEchoFilter(reasoning_echoes)
            has_visible_content = False

            def emit_split(delta: ThinkingDelta, *, final: bool = False) -> None:
                nonlocal has_visible_content
                reasoning = echo_filter.feed(delta.reasoning, final=final)
                if reasoning:
                    thinking_parts.append(reasoning)
                    if on_thinking is not None:
                        on_thinking(reasoning)
                content = delta.content
                if content and not has_visible_content:
                    content = content.lstrip()
                    has_visible_content = bool(content)
                if content:
                    parts.append(content)
                    if on_text is not None:
                        on_text(content)

            for row in rows:
                last = row
                generation_step = int(getattr(row, "generation_tokens", 0) or 0)
                token = getattr(row, "token", None)
                logprobs = getattr(row, "logprobs", None)
                if (
                    generation_step > 0
                    and generation_step not in observed_generation_steps
                    and token is not None
                ):
                    observed_generation_steps.add(generation_step)
                    token_id = _token_int(token)
                    if token_id is not None:
                        generated_token_ids.append(token_id)
                        if logprobs is not None:
                            try:
                                token_logprobs.append(
                                    float(logprobs[token_id].item())
                                )
                            except (AttributeError, IndexError, TypeError, ValueError):
                                pass
                text = str(getattr(row, "text", "") or "")
                if not text:
                    continue
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                emit_split(splitter.feed(text))
            emit_split(splitter.feed("", final=True), final=True)
            rebuilt = self._decode_complete_generation(
                generated_token_ids,
                enable_thinking=enable_thinking,
                thinking_start=thinking_start,
                thinking_end=thinking_end,
                reasoning_echoes=reasoning_echoes,
            )
            if rebuilt is not None:
                rebuilt_content, rebuilt_thinking = rebuilt
                streamed_content = "".join(parts)
                streamed_thinking = "".join(thinking_parts)
                if rebuilt_content != streamed_content:
                    if rebuilt_content.startswith(streamed_content):
                        suffix = rebuilt_content[len(streamed_content) :]
                        if suffix and on_text is not None:
                            on_text(suffix)
                    parts = [rebuilt_content] if rebuilt_content else []
                if rebuilt_thinking != streamed_thinking:
                    if rebuilt_thinking.startswith(streamed_thinking):
                        suffix = rebuilt_thinking[len(streamed_thinking) :]
                        if suffix and on_thinking is not None:
                            on_thinking(suffix)
                    thinking_parts = [rebuilt_thinking] if rebuilt_thinking else []
            prompt_cache_prefix_tokens = max(
                prompt_cache_prefix_tokens,
                int(getattr(last, "cached_tokens", 0) or 0),
            )
            if apc_manager is not None:
                apc_matched_after = int(
                    apc_manager.stats_snapshot().get("matched_tokens", 0)
                )
                prompt_cache_prefix_tokens = max(
                    prompt_cache_prefix_tokens,
                    apc_matched_after - apc_matched_before,
                )
            finished = time.perf_counter()
        cache_after = self.vision_cache.info()
        stats = VisionRunStats(
            generated_tokens=int(getattr(last, "generation_tokens", 0) or 0),
            prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            prompt_tokens_per_second=float(getattr(last, "prompt_tps", 0.0) or 0.0),
            generation_tokens_per_second=float(getattr(last, "generation_tps", 0.0) or 0.0),
            peak_memory_gb=float(getattr(last, "peak_memory", 0.0) or 0.0),
            total_duration_seconds=finished - started,
            time_to_first_token_seconds=None if first_text_at is None else first_text_at - started,
            visual_cache_enabled=bool(effective_vision_cache),
            visual_cache_hit=cache_after.hits > cache_before.hits,
            visual_cache_miss=cache_after.misses > cache_before.misses,
            visual_cache_entries=cache_after.size,
            visual_cache_hits_total=cache_after.hits,
            visual_cache_misses_total=cache_after.misses,
            prompt_cache_enabled=prompt_cache_enabled,
            prompt_cache_prefix_tokens=prompt_cache_prefix_tokens,
            image_count=len(images),
            cold_vision=cold_vision.to_dict(),
            post_fusion_vision=(
                {
                    "mode": vision_token_decision.mode,
                    "enabled": False,
                    "requested_retention_ratio": vision_token_decision.retain_ratio,
                    "prune_after_layer": vision_token_decision.prune_after_layer,
                    "applied_after_layer": 0,
                    "token_bucket": vision_token_decision.token_bucket,
                    "target_visual_tokens": 0,
                    "original_sequence_tokens": 0,
                    "retained_sequence_tokens": 0,
                    "original_visual_tokens": 0,
                    "retained_visual_tokens": 0,
                    "actual_visual_retention_ratio": None,
                    "policy": vision_token_decision.to_dict(),
                }
                if post_fusion is None
                else post_fusion.info()
            ),
            mean_token_logprob=(
                None if not token_logprobs else sum(token_logprobs) / len(token_logprobs)
            ),
            minimum_token_logprob=(None if not token_logprobs else min(token_logprobs)),
            thinking="".join(thinking_parts).strip(),
        )
        return "".join(parts), stats

    def _decode_complete_generation(
        self,
        token_ids: Sequence[int],
        *,
        enable_thinking: bool,
        thinking_start: Optional[str],
        thinking_end: Optional[str],
        reasoning_echoes: Sequence[str],
    ) -> Optional[tuple[str, str]]:
        if not token_ids:
            return None
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return None
        try:
            raw = decode(list(token_ids), skip_special_tokens=False)
        except TypeError:
            raw = decode(list(token_ids))
        except Exception:
            return None
        if not raw:
            return None
        splitter = ThinkingStreamSplitter(
            starts_in_thinking=bool(enable_thinking)
            and thinking_start in {None, "<think>", "<|START_THINKING|>"},
            start_marker=thinking_start,
            end_marker=thinking_end,
        )
        delta = splitter.feed(str(raw), final=True)
        echo_filter = InitialReasoningEchoFilter(reasoning_echoes)
        reasoning = echo_filter.feed(delta.reasoning, final=True)
        return delta.content.lstrip(), reasoning

    def _bind_thread_local_stream(self) -> None:
        module_name = str(getattr(self._stream_generate, "__module__", ""))
        if not module_name.startswith("mlx_vlm"):
            return
        import mlx.core as mx

        generation = importlib.import_module("mlx_vlm.generate")
        generation.generation_stream = mx.new_thread_local_stream(mx.default_device())

    def _prepare_cached_vision(
        self,
        prompt: str,
        images: Sequence[str],
        *,
        resize_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[tuple[None, dict[str, Any], int]]:
        module_name = str(getattr(self._stream_generate, "__module__", ""))
        if not images or not module_name.startswith("mlx_vlm"):
            return None
        try:
            import mlx.core as mx
            from mlx_vlm.utils import prepare_inputs
        except ImportError:
            return None

        model_type = config_value(self.model.config, "model_type", "")
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if model_type in {"gemma3", "gemma3n", "gemma4"}
            else True
        )
        inputs = prepare_inputs(
            self.processor,
            images=list(images),
            prompts=prompt,
            image_token_index=config_value(self.model.config, "image_token_index", None),
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return None

        options = {
            "input_ids": inputs.get("input_ids"),
            "pixel_values": pixel_values,
            "mask": inputs.get("attention_mask"),
        }
        options.update(
            {
                key: value
                for key, value in inputs.items()
                if key not in {"input_ids", "pixel_values", "attention_mask"}
            }
        )
        cache_source = _resolution_scoped_images(images, resize_shape)
        prompt_cache_state = self._prompt_cache_for(cache_source)
        input_ids = inputs.get("input_ids")
        prefix_tokens = 0
        if input_ids is not None:
            token_ids = input_ids.flatten().tolist()
            prefix_tokens = prompt_cache_state.find_prefix_length(token_ids)
            if prefix_tokens >= len(token_ids):
                prefix_tokens = 0
        if prefix_tokens and model_type in {
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        }:
            # MLX-VLM trims input_ids after priming Qwen mRoPE state but leaves
            # this full-length mask untouched, which breaks accumulated chat turns.
            options["mask"] = None
        if model_type in {"qwen3_5", "qwen3_5_moe"}:
            # Qwen3.5 interleaves ordinary KV layers with recurrent ArraysCache
            # layers. A token-only trim cannot roll the recurrent state backward,
            # so use MLX-VLM's whole-prefix snapshot path for exact state restore.
            prompt_cache_state.cache = None
            options["apc_manager"] = self._get_apc_manager()
        if model_type != "qwen3_vl":
            features = self.vision_cache.get(cache_source)
            if features is None:
                features = self._encode_vision_features(pixel_values, inputs, model_type)
                if features is None:
                    return None
                mx.eval(features)
                self.vision_cache.put(cache_source, features)
            options["cached_image_features"] = features
        options["prompt_cache_state"] = prompt_cache_state
        return None, options, prefix_tokens

    def _get_apc_manager(self) -> Any:
        if self._apc_manager is None:
            apc = importlib.import_module("mlx_vlm.apc")
            try:
                num_blocks = max(
                    64,
                    int(os.environ.get("MACHBOOST_MLX_APC_BLOCKS", "2048")),
                )
            except ValueError:
                num_blocks = 2048
            manager_options: dict[str, Any] = {
                "num_blocks": num_blocks,
                "block_size": 16,
            }
            disk_enabled = os.environ.get(
                "MACHBOOST_MLX_APC_DISK", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            disk_store = getattr(apc, "DiskBlockStore", None)
            if disk_enabled and callable(disk_store):
                configured_root = os.environ.get("MACHBOOST_MLX_APC_DISK_PATH")
                if configured_root:
                    disk_root = Path(configured_root).expanduser()
                elif sys.platform == "darwin":
                    disk_root = Path.home() / "Library" / "Caches" / "MachBoost" / "apc"
                else:
                    disk_root = Path(
                        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
                    ).expanduser() / "machboost" / "apc"
                try:
                    max_gb = max(
                        0.25,
                        float(os.environ.get("MACHBOOST_MLX_APC_DISK_GB", "8")),
                    )
                except ValueError:
                    max_gb = 8.0
                try:
                    manager_options["disk"] = disk_store(
                        disk_root,
                        namespace=self._apc_disk_namespace(),
                        num_workers=1,
                        max_bytes=int(max_gb * (1 << 30)),
                    )
                except Exception:
                    # Disk persistence is an acceleration tier, never a reason
                    # to make local inference unavailable.
                    pass
            self._apc_manager = apc.APCManager(**manager_options)
        return self._apc_manager

    def _apc_disk_namespace(self) -> str:
        try:
            hub = importlib.import_module("huggingface_hub")
            cached_config = hub.try_to_load_from_cache(
                self.model_name,
                "config.json",
                revision=self.revision or "main",
            )
            config_path = Path(cached_config)
            snapshot = config_path.parent
            if snapshot.parent.name == "snapshots" and snapshot.name:
                return f"{snapshot.name[:16]}-{self.model_name}"
        except (AttributeError, ImportError, TypeError, ValueError):
            pass
        return self.model_name

    def _prompt_cache_for(self, images: Sequence[str]) -> Any:
        key = f"vision:{self.vision_cache.key_for(list(images))}"
        return self._prompt_cache_for_key(key)

    def _prompt_cache_for_key(self, key: str) -> Any:
        state = self._prompt_caches.get(key)
        if state is not None:
            self._prompt_caches.move_to_end(key)
            return state
        generation = importlib.import_module("mlx_vlm.generate")
        state = generation.PromptCacheState()
        if len(self._prompt_caches) >= self.vision_cache.max_size:
            self._prompt_caches.popitem(last=False)
        self._prompt_caches[key] = state
        return state

    def _encode_vision_features(
        self,
        pixel_values: Any,
        inputs: dict[str, Any],
        model_type: str,
    ) -> Any:
        encode_image = getattr(self.model, "encode_image", None)
        if callable(encode_image):
            return encode_image(pixel_values)
        if model_type not in {
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "qwen3_5_moe",
        }:
            return None
        vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is None:
            return None
        grid = inputs.get("image_grid_thw")
        if grid is None:
            grid = inputs.get("video_grid_thw")
        patch_embed = getattr(vision_tower, "patch_embed", None)
        projection = getattr(patch_embed, "proj", None)
        weight = getattr(projection, "weight", None)
        if weight is not None:
            pixel_values = pixel_values.astype(weight.dtype)
        features = vision_tower(pixel_values, grid, output_hidden_states=False)
        if model_type in {"qwen3_5", "qwen3_5_moe"} and isinstance(features, tuple):
            return features[0]
        return features

    def reset_cache(self) -> None:
        if self._closed:
            return
        self._executor.submit(self._clear_caches).result()

    def _clear_caches(self) -> None:
        self.vision_cache.clear()
        self._prompt_caches.clear()
        if self._apc_manager is not None:
            self._apc_manager.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.reset_cache()
        manager = self._apc_manager
        self._apc_manager = None
        close_manager = getattr(manager, "close", None)
        if callable(close_manager):
            self._executor.submit(close_manager).result()
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def cache_info(self) -> dict[str, int]:
        return self.vision_cache.info().to_dict()


def config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _token_int(token: Any) -> Optional[int]:
    try:
        value = token.item() if hasattr(token, "item") else token
        return int(value)
    except (TypeError, ValueError):
        return None


def _thinking_budget(reasoning_strength: Optional[str]) -> Optional[int]:
    return {
        "low": 64,
        "medium": 256,
        "high": 768,
    }.get(str(reasoning_strength or "").lower())


def _latest_user_text(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return str(messages[-1].get("content") or "") if messages else ""


def _user_texts(messages: Sequence[dict[str, str]]) -> tuple[str, ...]:
    return tuple(
        str(message.get("content") or "")
        for message in reversed(messages)
        if message.get("role") == "user" and message.get("content")
    )


def _collapsed_whitespace(value: str) -> str:
    return " ".join(value.split())


def _resolution_scoped_images(
    images: Sequence[str],
    resize_shape: Optional[tuple[int, int]],
) -> list[str]:
    scoped = list(images)
    if resize_shape is not None:
        scoped.append(f"machboost:cold-vision:{resize_shape[0]}x{resize_shape[1]}")
    return scoped
