from __future__ import annotations

import importlib
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..vision import (
    ContentAddressedVisionCache,
    VisualAssetStore,
    normalize_multimodal_messages,
)


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
    image_count: int
    backend: str = "mlx-vlm"
    accepted_draft_tokens: int = 0
    target_calls: int = 0
    baseline_target_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.vision_cache = ContentAddressedVisionCache(max_size=vision_cache_size)
        self.assets = VisualAssetStore(asset_cache_dir)
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
        model, processor = load(model_name, lazy=lazy, revision=revision, **load_kwargs)
        return cls(
            model,
            processor,
            model_name=model_name,
            vision_cache_size=vision_cache_size,
            asset_cache_dir=asset_cache_dir,
        )

    def generate_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        use_vision_cache: bool = True,
        temperature: float = 0.0,
    ) -> tuple[str, VisionRunStats]:
        normalized, image_sources = normalize_multimodal_messages(messages)
        images = self.assets.materialize_all(image_sources)
        prompt = self._apply_chat_template(
            self.processor,
            self.model.config,
            normalized,
            num_images=len(images),
        )
        return self._generate(
            prompt,
            images=images,
            max_tokens=max_tokens,
            on_text=on_text,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        context: Optional[Iterable[str] | str] = None,
        on_text: Optional[Callable[[str], None]] = None,
        images: Optional[Sequence[str]] = None,
        use_vision_cache: bool = True,
        temperature: float = 0.0,
    ) -> tuple[str, VisionRunStats]:
        materialized = self.assets.materialize_all(images or ())
        templated = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=len(materialized),
        )
        return self._generate(
            templated,
            images=materialized,
            max_tokens=max_tokens,
            on_text=on_text,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
        )

    def _generate(
        self,
        prompt: str,
        *,
        images: Sequence[str],
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
        use_vision_cache: bool,
        temperature: float,
    ) -> tuple[str, VisionRunStats]:
        with self._generation_lock:
            self._bind_thread_local_stream()
            cache_before = self.vision_cache.info()
            started = time.perf_counter()
            first_text_at: Optional[float] = None
            parts: list[str] = []
            last: Any = None
            rows = self._stream_generate(
                self.model,
                self.processor,
                prompt,
                image=list(images) or None,
                max_tokens=max_tokens,
                temperature=temperature,
                vision_cache=self.vision_cache if use_vision_cache else None,
            )
            for row in rows:
                last = row
                text = str(getattr(row, "text", "") or "")
                if not text:
                    continue
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                parts.append(text)
                if on_text is not None:
                    on_text(text)
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
            visual_cache_enabled=bool(use_vision_cache),
            visual_cache_hit=cache_after.hits > cache_before.hits,
            visual_cache_miss=cache_after.misses > cache_before.misses,
            visual_cache_entries=cache_after.size,
            visual_cache_hits_total=cache_after.hits,
            visual_cache_misses_total=cache_after.misses,
            image_count=len(images),
        )
        return "".join(parts), stats

    def _bind_thread_local_stream(self) -> None:
        module_name = str(getattr(self._stream_generate, "__module__", ""))
        if not module_name.startswith("mlx_vlm"):
            return
        import mlx.core as mx

        generation = importlib.import_module("mlx_vlm.generate")
        generation.generation_stream = mx.new_thread_local_stream(mx.default_device())

    def reset_cache(self) -> None:
        self.vision_cache.clear()

    def cache_info(self) -> dict[str, int]:
        return self.vision_cache.info().to_dict()
