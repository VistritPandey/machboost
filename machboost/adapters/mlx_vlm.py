from __future__ import annotations

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
        self.vision_cache = ContentAddressedVisionCache(max_size=vision_cache_size)
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
        if self._closed:
            raise RuntimeError("MLX vision accelerator is closed")
        future = self._executor.submit(
            self._generate_on_worker,
            prompt,
            images=images,
            max_tokens=max_tokens,
            on_text=on_text,
            use_vision_cache=use_vision_cache,
            temperature=temperature,
        )
        return future.result()

    def _generate_on_worker(
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
            stream_image: Optional[list[str]] = list(images) or None
            stream_options: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            prepared = self._prepare_cached_vision(prompt, images) if use_vision_cache else None
            if prepared is not None:
                stream_image, prepared_options = prepared
                stream_options.update(prepared_options)
            elif use_vision_cache:
                stream_options["vision_cache"] = self.vision_cache
            rows = self._stream_generate(
                self.model,
                self.processor,
                prompt,
                image=stream_image,
                **stream_options,
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

    def _prepare_cached_vision(
        self,
        prompt: str,
        images: Sequence[str],
    ) -> Optional[tuple[None, dict[str, Any]]]:
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
            add_special_tokens=add_special_tokens,
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return None

        features = self.vision_cache.get(list(images))
        if features is None:
            features = self._encode_vision_features(pixel_values, inputs, model_type)
            if features is None:
                return None
            mx.eval(features)
            self.vision_cache.put(list(images), features)

        options = {
            "input_ids": inputs.get("input_ids"),
            "pixel_values": pixel_values,
            "mask": inputs.get("attention_mask"),
            "cached_image_features": features,
        }
        options.update(
            {
                key: value
                for key, value in inputs.items()
                if key not in {"input_ids", "pixel_values", "attention_mask"}
            }
        )
        return None, options

    def _encode_vision_features(
        self,
        pixel_values: Any,
        inputs: dict[str, Any],
        model_type: str,
    ) -> Any:
        encode_image = getattr(self.model, "encode_image", None)
        if callable(encode_image):
            return encode_image(pixel_values)
        if model_type not in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
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
        return vision_tower(pixel_values, grid, output_hidden_states=False)

    def reset_cache(self) -> None:
        if self._closed:
            return
        self._executor.submit(self.vision_cache.clear).result()

    def close(self) -> None:
        if self._closed:
            return
        self.reset_cache()
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def cache_info(self) -> dict[str, int]:
        return self.vision_cache.info().to_dict()


def config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)
