from __future__ import annotations

import math
import random
from typing import Any, Optional


POST_FUSION_MODES = ("off", "merge", "adaptive", "random")


class PostFusionVisionModel:
    """Qwen3-VL language wrapper that compresses visual states after deep-stack fusion."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.mode = "off"
        self.retain_ratio = 0.35
        self.prune_after_layer = 3
        self.token_bucket = 0
        self.policy: dict[str, Any] = {}
        self.reset_stats()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def configure(
        self,
        *,
        mode: str,
        retain_ratio: float,
        prune_after_layer: int = 3,
        token_bucket: int = 0,
        policy: Optional[dict[str, Any]] = None,
    ) -> None:
        normalized = str(mode or "off").strip().lower()
        if normalized not in POST_FUSION_MODES:
            raise ValueError(
                "post-fusion vision mode must be one of: "
                + ", ".join(POST_FUSION_MODES)
            )
        ratio = float(retain_ratio)
        if not 0.1 <= ratio <= 1.0:
            raise ValueError("post-fusion visual token ratio must be between 0.1 and 1.0")
        if int(prune_after_layer) < 1:
            raise ValueError("post-fusion prune layer must be at least 1")
        if int(token_bucket) < 0:
            raise ValueError("post-fusion visual token bucket must be zero or greater")
        self.mode = normalized
        self.retain_ratio = ratio
        self.prune_after_layer = int(prune_after_layer)
        self.token_bucket = int(token_bucket)
        self.policy = dict(policy or {})
        self.reset_stats()

    def reset_stats(self) -> None:
        self.original_sequence_tokens = 0
        self.retained_sequence_tokens = 0
        self.original_visual_tokens = 0
        self.retained_visual_tokens = 0
        self.target_visual_tokens = 0
        self.applied_after_layer = 0

    def info(self) -> dict[str, Any]:
        original = self.original_visual_tokens
        retained = self.retained_visual_tokens
        return {
            "mode": self.mode,
            "enabled": self.mode != "off" and original > 0,
            "requested_retention_ratio": self.retain_ratio,
            "prune_after_layer": self.prune_after_layer,
            "applied_after_layer": self.applied_after_layer,
            "token_bucket": self.token_bucket,
            "target_visual_tokens": self.target_visual_tokens,
            "original_sequence_tokens": self.original_sequence_tokens,
            "retained_sequence_tokens": self.retained_sequence_tokens,
            "original_visual_tokens": original,
            "retained_visual_tokens": retained,
            "actual_visual_retention_ratio": (
                None if original == 0 else retained / original
            ),
            "policy": dict(self.policy),
        }

    def __call__(
        self,
        inputs: Any,
        inputs_embeds: Optional[Any] = None,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        visual_pos_masks: Optional[Any] = None,
        deepstack_visual_embeds: Optional[Any] = None,
    ) -> Any:
        if (
            self.mode == "off"
            or visual_pos_masks is None
            or inputs.shape[0] != 1
            or inputs.shape[1] <= 1
        ):
            return self.base(
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )

        import mlx.core as mx
        from mlx_vlm.models.base import create_attention_mask

        hidden = (
            self.base.embed_tokens(inputs)
            if inputs_embeds is None
            else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.base.layers)
        if mask is None:
            mask = create_attention_mask(
                hidden,
                cache[0] if cache and cache[0] is not None else cache,
            )

        deepstack_layers = (
            0 if deepstack_visual_embeds is None else len(deepstack_visual_embeds)
        )
        compress_after = min(
            max(self.prune_after_layer, deepstack_layers),
            max(1, len(self.base.layers) - 1),
        )
        for layer_index, (layer, layer_cache) in enumerate(
            zip(self.base.layers, cache)
        ):
            hidden = layer(hidden, mask, layer_cache, position_ids)
            if deepstack_visual_embeds is not None and layer_index < deepstack_layers:
                hidden = self.base._deepstack_process(
                    hidden,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_index],
                )
            if layer_index + 1 != compress_after:
                continue

            original_length = hidden.shape[1]
            hidden, keep, visual_before, visual_after = self._compress(
                hidden,
                visual_pos_masks[0],
                position_ids,
            )
            if position_ids is not None:
                position_ids = mx.take(position_ids, keep, axis=-1)
            if (
                mask is not None
                and hasattr(mask, "shape")
                and mask.shape[-1] == original_length
            ):
                mask = mx.take(mask, keep, axis=-1)
                if mask.ndim >= 2 and mask.shape[-2] == original_length:
                    mask = mx.take(mask, keep, axis=-2)
            self.original_sequence_tokens = int(original_length)
            self.retained_sequence_tokens = int(hidden.shape[1])
            self.original_visual_tokens = visual_before
            self.retained_visual_tokens = visual_after
            self.target_visual_tokens = bucket_token_target(
                visual_before,
                self.retain_ratio,
                self.token_bucket,
            )
            self.applied_after_layer = layer_index + 1

        return self.base.norm(hidden)

    def _compress(
        self,
        hidden: Any,
        visual_mask: Any,
        position_ids: Optional[Any],
    ) -> tuple[Any, Any, int, int]:
        import mlx.core as mx
        import numpy as np

        visual_flags = np.asarray(visual_mask.tolist(), dtype=bool)
        visual_indices = np.flatnonzero(visual_flags)
        text_indices = np.flatnonzero(~visual_flags)
        if len(visual_indices) < 4 or position_ids is None:
            keep = mx.arange(hidden.shape[1], dtype=mx.uint32)
            return hidden, keep, len(visual_indices), len(visual_indices)

        trailing_text = text_indices[text_indices > visual_indices[-1]][-48:]
        query_indices = trailing_text if len(trailing_text) else text_indices[-48:]
        query = mx.mean(
            mx.take(
                hidden[0],
                mx.array(query_indices, dtype=mx.uint32),
                axis=0,
            ),
            axis=0,
        )
        query = query / mx.maximum(mx.linalg.norm(query), 1e-6)

        visual_index_array = mx.array(visual_indices, dtype=mx.uint32)
        visual_hidden = mx.take(hidden[0], visual_index_array, axis=0)
        normalized_visual = visual_hidden / mx.maximum(
            mx.linalg.norm(visual_hidden, axis=-1, keepdims=True),
            1e-6,
        )
        relevance = normalized_visual @ query

        temporal = np.asarray(
            mx.take(position_ids[0, 0], visual_index_array, axis=0).tolist(),
            dtype=int,
        )
        heights = np.asarray(
            mx.take(position_ids[1, 0], visual_index_array, axis=0).tolist(),
            dtype=int,
        )
        widths = np.asarray(
            mx.take(position_ids[2, 0], visual_index_array, axis=0).tolist(),
            dtype=int,
        )
        spatial_factor = max(1, round(math.sqrt(1.0 / self.retain_ratio)))
        segments = _contiguous_segments(visual_indices)
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        for position, (segment, time_index, height, width) in enumerate(
            zip(segments, temporal, heights, widths)
        ):
            key = (
                int(segment),
                int(time_index),
                int(height) // spatial_factor,
                int(width) // spatial_factor,
            )
            groups.setdefault(key, []).append(position)

        group_values = list(groups.values())
        preserve_groups = self._detail_groups(
            visual_hidden,
            group_values,
            visual_count=len(visual_indices),
        )
        representatives = [
            int(visual_indices[positions[0]]) for positions in group_values
        ]
        retained_visual: list[int] = []
        for group_index, (positions, representative) in enumerate(
            zip(group_values, representatives)
        ):
            if group_index in preserve_groups:
                retained_visual.extend(
                    int(visual_indices[position]) for position in positions
                )
            else:
                retained_visual.append(representative)

        keep_values = sorted([*text_indices.tolist(), *retained_visual])
        output_positions = {
            source: index for index, source in enumerate(keep_values)
        }
        merged_rows = []
        merged_destinations = []
        for group_index, (positions, representative) in enumerate(
            zip(group_values, representatives)
        ):
            if group_index in preserve_groups:
                continue
            group_positions = mx.array(positions, dtype=mx.uint32)
            group_hidden = mx.take(visual_hidden, group_positions, axis=0)
            group_scores = mx.take(relevance, group_positions, axis=0)
            weights = mx.softmax(group_scores * 4.0)
            merged_rows.append(
                mx.sum(group_hidden * weights[:, None], axis=0)
            )
            merged_destinations.append(output_positions[representative])

        keep = mx.array(keep_values, dtype=mx.uint32)
        compressed = mx.take(hidden, keep, axis=1)
        if merged_rows:
            compressed[
                0,
                mx.array(merged_destinations, dtype=mx.uint32),
            ] = mx.stack(merged_rows, axis=0)
        return compressed, keep, len(visual_indices), len(retained_visual)

    def _detail_groups(
        self,
        visual_hidden: Any,
        groups: list[list[int]],
        *,
        visual_count: int,
    ) -> set[int]:
        if self.mode not in {"adaptive", "random"}:
            return set()

        import mlx.core as mx

        target = bucket_token_target(
            visual_count,
            self.retain_ratio,
            self.token_bucket,
        )
        base_retained = len(groups)
        if target <= base_retained:
            return set()
        if self.mode == "random":
            ranked = list(range(len(groups)))
            random.Random(visual_count * 1009 + target).shuffle(ranked)
        else:
            max_group = max(len(group) for group in groups)
            padded = [
                group + [group[0]] * (max_group - len(group))
                for group in groups
            ]
            grouped = mx.take(
                visual_hidden,
                mx.array(padded, dtype=mx.uint32),
                axis=0,
            )
            normalized = grouped / mx.maximum(
                mx.linalg.norm(grouped, axis=-1, keepdims=True),
                1e-6,
            )
            detail = 1.0 - mx.linalg.norm(mx.mean(normalized, axis=1), axis=-1)
            ranked = mx.argsort(detail).tolist()[::-1]
        selected: set[int] = set()
        retained = base_retained
        for group_index in ranked:
            gain = len(groups[group_index]) - 1
            if gain <= 0:
                continue
            selected.add(int(group_index))
            retained += gain
            if retained >= target:
                break
        return selected


def configure_post_fusion_vision(
    model: Any,
    *,
    mode: str = "off",
    retain_ratio: float = 0.35,
    prune_after_layer: int = 3,
    token_bucket: int = 0,
    policy: Optional[dict[str, Any]] = None,
) -> Optional[PostFusionVisionModel]:
    normalized = str(mode or "off").strip().lower()
    if normalized not in POST_FUSION_MODES:
        raise ValueError(
            "post-fusion vision mode must be one of: "
            + ", ".join(POST_FUSION_MODES)
        )
    language_model = getattr(model, "language_model", None)
    inner = getattr(language_model, "model", None)
    wrapper = inner if isinstance(inner, PostFusionVisionModel) else None
    if normalized == "off" and wrapper is None:
        return None
    if wrapper is None:
        model_type = _config_value(getattr(model, "config", None), "model_type")
        if model_type != "qwen3_vl":
            raise ValueError(
                "post-fusion visual token compression currently supports Qwen3-VL only"
            )
        required = ("embed_tokens", "layers", "norm", "_deepstack_process")
        missing = [name for name in required if not hasattr(inner, name)]
        if missing:
            raise RuntimeError(
                "installed mlx-vlm Qwen3-VL internals are incompatible: missing "
                + ", ".join(missing)
            )
        wrapper = PostFusionVisionModel(inner)
        language_model.model = wrapper
    wrapper.configure(
        mode=normalized,
        retain_ratio=retain_ratio,
        prune_after_layer=prune_after_layer,
        token_bucket=token_bucket,
        policy=policy,
    )
    return wrapper


def bucket_token_target(visual_count: int, retain_ratio: float, bucket: int = 0) -> int:
    count = max(0, int(visual_count))
    if count == 0:
        return 0
    target = max(1, math.ceil(count * float(retain_ratio)))
    width = int(bucket)
    if width > 1:
        target = math.ceil(target / width) * width
    return min(count, target)


def _contiguous_segments(indices: Any) -> list[int]:
    segments = []
    segment = 0
    previous = None
    for raw in indices:
        index = int(raw)
        if previous is not None and index != previous + 1:
            segment += 1
        segments.append(segment)
        previous = index
    return segments


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)
