__version__ = "0.13.3"

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
from .context_bench import (
    CONTEXT_BENCH_SCHEMA,
    benchmark_context_acceleration,
    context_fingerprint,
)
from .latency import LATENCY_SCHEMA, benchmark_chat_latency
from .adapters.dflash import DFlashAccelerator, DFlashRunStats
from .adapters.mlx_vlm import MLXVLMAccelerator, VisionRunStats
from .adapters.ollama_mlx import (
    OllamaMLXAccelerator,
    OllamaMLXCancelled,
    OllamaMLXRunStats,
)
from .models import ModelAlias, ModelResolution, resolve_model
from .server import RuntimeManager
from .video import TemporalVideoSampler, VideoFrame, VideoSelection
from .vision import ContentAddressedVisionCache, VisualAssetStore, VisionCacheInfo
from .workspace import (
    IndexReport,
    SearchHit,
    Workspace,
    WorkspaceError,
    WorkspaceQuery,
    WorkspaceStore,
)

__all__ = [
    "Accelerator",
    "AcceleratorResult",
    "BenchmarkCase",
    "BenchmarkResult",
    "BoostedService",
    "CalibrationResult",
    "Candidate",
    "ContentAddressedVisionCache",
    "CONTEXT_BENCH_SCHEMA",
    "CorpusDrafter",
    "DFlashAccelerator",
    "DFlashRunStats",
    "GateDecision",
    "GatePolicy",
    "LATENCY_SCHEMA",
    "MachBoost",
    "MachBoostAPIError",
    "MachBoostClient",
    "MLXVLMAccelerator",
    "ModelAlias",
    "ModelResolution",
    "OllamaMLXAccelerator",
    "OllamaMLXCancelled",
    "OllamaMLXRunStats",
    "RunStats",
    "SearchHit",
    "RuntimeManager",
    "TemporalVideoSampler",
    "VideoFrame",
    "VideoSelection",
    "VisualAssetStore",
    "VisionCacheInfo",
    "VisionRunStats",
    "Workspace",
    "WorkspaceError",
    "WorkspaceQuery",
    "WorkspaceStore",
    "IndexReport",
    "benchmark",
    "benchmark_cases",
    "benchmark_chat_latency",
    "benchmark_context_acceleration",
    "context_fingerprint",
    "ensure_server",
    "machboost",
    "resolve_model",
    "summarize_results",
    "__version__",
]
