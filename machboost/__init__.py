__version__ = "0.2.0"

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
from .models import ModelAlias, ModelResolution, resolve_model
from .server import RuntimeManager

__all__ = [
    "Accelerator",
    "AcceleratorResult",
    "BenchmarkCase",
    "BenchmarkResult",
    "BoostedService",
    "CalibrationResult",
    "Candidate",
    "CorpusDrafter",
    "GateDecision",
    "GatePolicy",
    "MachBoost",
    "MachBoostAPIError",
    "MachBoostClient",
    "ModelAlias",
    "ModelResolution",
    "RunStats",
    "RuntimeManager",
    "benchmark",
    "benchmark_cases",
    "ensure_server",
    "machboost",
    "resolve_model",
    "summarize_results",
    "__version__",
]
