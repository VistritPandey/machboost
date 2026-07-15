from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


_STRUCTURED_DETAIL_TERMS = (
    "bar graph",
    "chart",
    "difference",
    "graph",
    "highest",
    "how many",
    "lowest",
    "percent",
    "percentage",
    "sum of",
    "table",
    "value",
)

_TEXT_DETAIL_TERMS = (
    "aged",
    "brand",
    "date",
    "jersey",
    "label",
    "license plate",
    "name",
    "number",
    "photographer",
    "read",
    "small text",
    "spell",
    "text",
    "time",
    "word",
)

_MODE_EDGES = {
    "fast": 336,
    "balanced": 512,
    "quality": 672,
}


@dataclass(frozen=True)
class ImageDetail:
    width: int
    height: int
    entropy: float
    edge_density: float

    @property
    def max_edge(self) -> int:
        return max(self.width, self.height)


@dataclass(frozen=True)
class ColdVisionDecision:
    mode: str
    enabled: bool
    target_max_edge: Optional[int]
    resize_shape: Optional[tuple[int, int]]
    source_max_edge: int
    image_entropy: float
    image_edge_density: float
    question_class: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.resize_shape is not None:
            value["resize_shape"] = list(self.resize_shape)
        return value


def choose_cold_vision(
    prompt: str,
    images: Sequence[str],
    *,
    mode: str = "off",
    max_edge: Optional[int] = None,
) -> ColdVisionDecision:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode not in {"off", "adaptive", *_MODE_EDGES}:
        raise ValueError(
            "cold vision mode must be one of: off, adaptive, fast, balanced, quality"
        )
    if max_edge is not None and int(max_edge) < 56:
        raise ValueError("cold vision max edge must be at least 56 pixels")
    if not images or normalized_mode == "off":
        return _disabled_decision(normalized_mode, "cold first-view acceleration is disabled")

    details = tuple(_inspect_image(path) for path in images)
    source_max_edge = max(detail.max_edge for detail in details)
    entropy = max(detail.entropy for detail in details)
    edge_density = max(detail.edge_density for detail in details)
    question_class = _question_class(prompt)

    if max_edge is not None:
        target = int(max_edge)
        reason = "explicit first-view visual edge budget"
    elif normalized_mode in _MODE_EDGES:
        target = _MODE_EDGES[normalized_mode]
        reason = f"{normalized_mode} first-view visual edge budget"
    else:
        target, reason = _adaptive_target(
            question_class=question_class,
            entropy=entropy,
            edge_density=edge_density,
        )

    if source_max_edge <= target:
        return ColdVisionDecision(
            mode=normalized_mode,
            enabled=False,
            target_max_edge=target,
            resize_shape=None,
            source_max_edge=source_max_edge,
            image_entropy=entropy,
            image_edge_density=edge_density,
            question_class=question_class,
            reason="source image is already within the selected visual budget",
        )

    return ColdVisionDecision(
        mode=normalized_mode,
        enabled=True,
        target_max_edge=target,
        resize_shape=(target, target),
        source_max_edge=source_max_edge,
        image_entropy=entropy,
        image_edge_density=edge_density,
        question_class=question_class,
        reason=reason,
    )


def _disabled_decision(mode: str, reason: str) -> ColdVisionDecision:
    return ColdVisionDecision(
        mode=mode,
        enabled=False,
        target_max_edge=None,
        resize_shape=None,
        source_max_edge=0,
        image_entropy=0.0,
        image_edge_density=0.0,
        question_class="none",
        reason=reason,
    )


def _question_class(prompt: str) -> str:
    normalized = " ".join(str(prompt).lower().split())
    if any(term in normalized for term in _STRUCTURED_DETAIL_TERMS):
        return "structured-detail"
    if any(term in normalized for term in _TEXT_DETAIL_TERMS):
        return "text-detail"
    return "general"


def _adaptive_target(
    *,
    question_class: str,
    entropy: float,
    edge_density: float,
) -> tuple[int, str]:
    simple_layout = entropy < 0.32 and edge_density < 0.14
    if simple_layout and question_class != "structured-detail":
        return 336, "simple low-entropy layout permits a compact visual pass"
    if question_class == "structured-detail":
        return 672, "structured visual reasoning retains a larger detail budget"
    if question_class == "text-detail":
        return 512, "text-oriented question uses a medium visual detail budget"
    if entropy >= 0.75:
        return 512, "high-entropy image uses a medium visual detail budget"
    return 448, "general first-view request uses a balanced visual detail budget"


def _inspect_image(path: str) -> ImageDetail:
    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise ImportError("Cold vision policy requires Pillow; install machboost[vision].") from exc

    image_path = Path(path).expanduser()
    with Image.open(image_path) as image:
        width, height = image.size
        grayscale = image.convert("L")
        grayscale.thumbnail((256, 256))
        entropy = float(grayscale.entropy()) / 8.0
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        if edges.width > 2 and edges.height > 2:
            edges = edges.crop((1, 1, edges.width - 1, edges.height - 1))
        histogram = edges.histogram()
        samples = max(1, sum(histogram))
        edge_density = sum(histogram[32:]) / samples

    return ImageDetail(
        width=int(width),
        height=int(height),
        entropy=entropy,
        edge_density=edge_density,
    )
