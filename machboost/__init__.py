__version__ = "0.4.0"

from .accelerator import Accelerator, AcceleratorResult, CalibrationResult
from .bench import (
    BenchmarkCase,
    BenchmarkResult,
    GateDecision,
    GatePolicy,
    benchmark,
    benchmark_cases,
    summarize_results,
)
from .core import (
    BoostedService,
    Candidate,
    CorpusDrafter,
    MachBoost,
    RunStats,
    machboost,
)
from .client import MachBoostAPIError, MachBoostClient, ensure_server
from .adapters.mlx_vlm import MLXVLMAccelerator, VisionRunStats
from .models import ModelAlias, ModelResolution, resolve_model
from .server import RuntimeManager
from .video import TemporalVideoSampler, VideoFrame, VideoSelection
from .vision import ContentAddressedVisionCache, VisualAssetStore, VisionCacheInfo

__all__ = [
    "Accelerator",
    "AcceleratorResult",
    "BenchmarkCase",
    "BenchmarkResult",
    "BoostedService",
    "CalibrationResult",
    "Candidate",
    "ContentAddressedVisionCache",
    "CorpusDrafter",
    "GateDecision",
    "GatePolicy",
    "MachBoost",
    "MachBoostAPIError",
    "MachBoostClient",
    "MLXVLMAccelerator",
    "ModelAlias",
    "ModelResolution",
    "RunStats",
    "RuntimeManager",
    "TemporalVideoSampler",
    "VideoFrame",
    "VideoSelection",
    "VisualAssetStore",
    "VisionCacheInfo",
    "VisionRunStats",
    "benchmark",
    "benchmark_cases",
    "ensure_server",
    "machboost",
    "resolve_model",
    "summarize_results",
    "__version__",
]
