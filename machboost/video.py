from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class VideoFrame:
    path: str
    sample_index: int
    timestamp_seconds: float
    change_score: float


@dataclass(frozen=True)
class VideoSelection:
    video: str
    frames: tuple[VideoFrame, ...]
    sampled_frames: int
    selected_frames: int
    reduction_rate: float
    sample_fps: float
    change_threshold: float
    max_frames: int
    extraction_cache_hit: bool
    elapsed_seconds: float

    @property
    def images(self) -> tuple[str, ...]:
        return tuple(frame.path for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "frames": [asdict(frame) for frame in self.frames],
        }


class TemporalVideoSampler:
    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.cache_dir = (
            cache_dir or Path.home() / ".cache" / "machboost" / "video" / "frames"
        ).expanduser()
        self.ffmpeg = ffmpeg

    def sample(
        self,
        video: str | Path,
        *,
        fps: float = 1.0,
        change_threshold: float = 0.08,
        max_frames: int = 12,
    ) -> VideoSelection:
        started = time.perf_counter()
        source = Path(video).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"video does not exist: {source}")
        if fps <= 0:
            raise ValueError("video sample FPS must be greater than zero")
        if not 0.0 <= change_threshold <= 1.0:
            raise ValueError("video change threshold must be between 0 and 1")
        if max_frames < 1:
            raise ValueError("video max frames must be at least 1")
        if shutil.which(self.ffmpeg) is None:
            raise RuntimeError(
                "FFmpeg is required for video inputs; install it or pass image frames instead"
            )

        extraction_dir = self.cache_dir / _video_cache_key(source, fps)
        frame_paths, cache_hit = self._extract_frames(source, extraction_dir, fps)
        if not frame_paths:
            raise RuntimeError(f"FFmpeg extracted no frames from {source}")
        changes = frame_change_scores(frame_paths)
        selected_indices = select_temporal_frames(
            changes,
            threshold=change_threshold,
            max_frames=max_frames,
        )
        frames = tuple(
            VideoFrame(
                path=str(frame_paths[index]),
                sample_index=index,
                timestamp_seconds=index / fps,
                change_score=changes[index],
            )
            for index in selected_indices
        )
        return VideoSelection(
            video=str(source),
            frames=frames,
            sampled_frames=len(frame_paths),
            selected_frames=len(frames),
            reduction_rate=1.0 - len(frames) / len(frame_paths),
            sample_fps=float(fps),
            change_threshold=float(change_threshold),
            max_frames=int(max_frames),
            extraction_cache_hit=cache_hit,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _extract_frames(
        self,
        source: Path,
        extraction_dir: Path,
        fps: float,
    ) -> tuple[list[Path], bool]:
        manifest = extraction_dir / "manifest.json"
        existing = sorted(extraction_dir.glob("frame-*.jpg"))
        if manifest.is_file() and existing:
            return existing, True

        extraction_dir.mkdir(parents=True, exist_ok=True)
        for stale in extraction_dir.glob("frame-*.jpg"):
            stale.unlink()
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps={fps:g}",
            "-q:v",
            "2",
            str(extraction_dir / "frame-%06d.jpg"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip() or f"FFmpeg exited with {result.returncode}"
            raise RuntimeError(f"could not sample video frames: {message}")
        frames = sorted(extraction_dir.glob("frame-*.jpg"))
        manifest.write_text(
            json.dumps(
                {
                    "video": str(source),
                    "fps": fps,
                    "frame_count": len(frames),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return frames, False


def frame_change_scores(frame_paths: Sequence[Path]) -> tuple[float, ...]:
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError as exc:
        raise ImportError("Video frame selection requires Pillow.") from exc

    scores = []
    previous = None
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            current = image.convert("L").resize((64, 64))
            if previous is None:
                scores.append(1.0)
            else:
                difference = ImageChops.difference(previous, current)
                scores.append(float(ImageStat.Stat(difference).mean[0]) / 255.0)
            previous = current.copy()
    return tuple(scores)


def select_temporal_frames(
    change_scores: Sequence[float],
    *,
    threshold: float,
    max_frames: int,
) -> tuple[int, ...]:
    if not change_scores:
        return ()
    if max_frames < 1:
        raise ValueError("video max frames must be at least 1")
    last = len(change_scores) - 1
    selected = {0, last}
    selected.update(
        index
        for index, score in enumerate(change_scores[1:], start=1)
        if float(score) >= threshold
    )
    if len(selected) > max_frames:
        boundaries = {0} if max_frames == 1 else {0, last}
        budget = max_frames - len(boundaries)
        ranked = sorted(
            (index for index in selected if index not in boundaries),
            key=lambda index: (-float(change_scores[index]), index),
        )
        selected = boundaries | set(ranked[: max(0, budget)])
    return tuple(sorted(selected))


def _video_cache_key(path: Path, fps: float) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(f"fps={fps:g}".encode("ascii"))
    return digest.hexdigest()
