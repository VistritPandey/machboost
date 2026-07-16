from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


VISION_TOKEN_REQUEST_MODES = ("off", "auto", "merge", "adaptive", "random")
CALIBRATION_SCHEMA = "machboost.vision_calibration.v1"

_CHART_TERMS = (
    "axis",
    "bar chart",
    "chart",
    "graph",
    "legend",
    "percentage",
    "plot",
    "trend",
)
_DOCUMENT_TERMS = (
    "amount",
    "date",
    "document",
    "invoice",
    "label",
    "name",
    "number",
    "read",
    "receipt",
    "sign",
    "spell",
    "table",
    "text",
    "time",
    "total",
    "word",
)
_SPATIAL_TERMS = (
    "above",
    "below",
    "center",
    "closest",
    "farthest",
    "left",
    "next to",
    "position",
    "relative",
    "right",
    "where",
)


@dataclass(frozen=True)
class VisionImageSignals:
    count: int
    max_edge: int
    entropy: float
    edge_density: float


@dataclass(frozen=True)
class VisionTokenDecision:
    requested_mode: str
    mode: str
    enabled: bool
    workload: str
    retain_ratio: float
    prune_after_layer: int
    token_bucket: int
    source: str
    reason: str
    image_signals: VisionImageSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "mode": self.mode,
            "enabled": self.enabled,
            "workload": self.workload,
            "retain_ratio": self.retain_ratio,
            "prune_after_layer": self.prune_after_layer,
            "token_bucket": self.token_bucket,
            "source": self.source,
            "reason": self.reason,
            "image_signals": {
                "count": self.image_signals.count,
                "max_edge": self.image_signals.max_edge,
                "entropy": self.image_signals.entropy,
                "edge_density": self.image_signals.edge_density,
            },
        }


def choose_vision_token_policy(
    prompt: str,
    images: Sequence[str],
    *,
    mode: str = "off",
    retain_ratio: float = 0.35,
    prune_after_layer: Optional[int] = None,
    token_bucket: Optional[int] = None,
    calibration: Optional[Mapping[str, Any]] = None,
    image_signals: Optional[VisionImageSignals] = None,
) -> VisionTokenDecision:
    requested = str(mode or "off").strip().lower()
    if requested not in VISION_TOKEN_REQUEST_MODES:
        raise ValueError(
            "vision token mode must be one of: " + ", ".join(VISION_TOKEN_REQUEST_MODES)
        )
    ratio = float(retain_ratio)
    if not 0.1 <= ratio <= 1.0:
        raise ValueError("visual token retention ratio must be between 0.1 and 1.0")
    if prune_after_layer is not None and int(prune_after_layer) < 1:
        raise ValueError("post-fusion prune layer must be at least 1")
    if token_bucket is not None and int(token_bucket) < 0:
        raise ValueError("visual token bucket must be zero or greater")

    signals = image_signals or inspect_vision_images(images)
    workload = classify_vision_workload(prompt, image_count=signals.count)
    if requested == "off" or not images:
        return VisionTokenDecision(
            requested_mode=requested,
            mode="off",
            enabled=False,
            workload=workload,
            retain_ratio=ratio,
            prune_after_layer=int(prune_after_layer or 3),
            token_bucket=int(token_bucket or 0),
            source="request",
            reason="post-fusion visual token compression is disabled",
            image_signals=signals,
        )

    if requested != "auto":
        return VisionTokenDecision(
            requested_mode=requested,
            mode=requested,
            enabled=True,
            workload=workload,
            retain_ratio=ratio,
            prune_after_layer=int(prune_after_layer or 3),
            token_bucket=int(token_bucket or 0),
            source="request",
            reason="explicit post-fusion visual token policy",
            image_signals=signals,
        )

    selected = _builtin_profile(workload, signals)
    source = "builtin"
    calibrated = _calibrated_profile(calibration, workload)
    if calibrated is not None:
        selected.update(calibrated)
        source = "calibration"
    if prune_after_layer is not None:
        selected["prune_after_layer"] = int(prune_after_layer)
        source += "+request"
    if token_bucket is not None:
        selected["token_bucket"] = int(token_bucket)
        source += "+request"

    return VisionTokenDecision(
        requested_mode="auto",
        mode=str(selected.get("mode", "adaptive")),
        enabled=bool(selected.get("enabled", True)),
        workload=workload,
        retain_ratio=float(selected["retain_ratio"]),
        prune_after_layer=int(selected["prune_after_layer"]),
        token_bucket=int(selected.get("token_bucket", 32)),
        source=source,
        reason=str(selected.get("reason", f"automatic {workload} visual token policy")),
        image_signals=signals,
    )


def classify_vision_workload(prompt: str, *, image_count: int = 1) -> str:
    if image_count > 1:
        return "multi-image"
    normalized = " ".join(str(prompt).lower().split())
    if _contains_any(normalized, _CHART_TERMS):
        return "chart"
    if _contains_any(normalized, _DOCUMENT_TERMS):
        return "document-text"
    if _contains_any(normalized, _SPATIAL_TERMS):
        return "spatial"
    return "general"


def inspect_vision_images(images: Sequence[str]) -> VisionImageSignals:
    if not images:
        return VisionImageSignals(count=0, max_edge=0, entropy=0.0, edge_density=0.0)
    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise ImportError("Automatic vision token policy requires Pillow.") from exc

    max_edge = 0
    entropy = 0.0
    edge_density = 0.0
    for source in images:
        path = Path(source).expanduser()
        if not path.is_file():
            continue
        with Image.open(path) as image:
            max_edge = max(max_edge, *image.size)
            grayscale = image.convert("L")
            grayscale.thumbnail((256, 256))
            entropy = max(entropy, float(grayscale.entropy()) / 8.0)
            edges = grayscale.filter(ImageFilter.FIND_EDGES)
            if edges.width > 2 and edges.height > 2:
                edges = edges.crop((1, 1, edges.width - 1, edges.height - 1))
            histogram = edges.histogram()
            samples = max(1, sum(histogram))
            edge_density = max(edge_density, sum(histogram[32:]) / samples)
    return VisionImageSignals(
        count=len(images),
        max_edge=max_edge,
        entropy=entropy,
        edge_density=edge_density,
    )


def load_vision_calibration(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(f"unsupported vision calibration schema in {source}")
    workloads = payload.get("workloads")
    if not isinstance(workloads, dict):
        raise ValueError("vision calibration must contain a workloads object")
    return payload


def _builtin_profile(workload: str, signals: VisionImageSignals) -> dict[str, Any]:
    profiles = {
        "general": (0.35, 3),
        "spatial": (0.45, 6),
        "document-text": (0.50, 6),
        "chart": (0.55, 6),
        "multi-image": (0.55, 6),
    }
    ratio, layer = profiles[workload]
    if signals.entropy >= 0.78 or signals.edge_density >= 0.30:
        ratio = min(0.65, ratio + 0.05)
        layer = max(layer, 6)
    return {
        "mode": "adaptive",
        "enabled": True,
        "retain_ratio": ratio,
        "prune_after_layer": layer,
        "token_bucket": 32,
        "reason": f"automatic {workload} policy selected from prompt and image detail",
    }


def _calibrated_profile(
    calibration: Optional[Mapping[str, Any]],
    workload: str,
) -> dict[str, Any] | None:
    if calibration is None:
        return None
    workloads = calibration.get("workloads")
    if not isinstance(workloads, Mapping):
        return None
    selected = workloads.get(workload, workloads.get("default"))
    if not isinstance(selected, Mapping):
        return None
    return dict(selected)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        if re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None:
            return True
    return False
