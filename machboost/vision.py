from __future__ import annotations

import base64
import binascii
import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import unquote_to_bytes, urlparse


DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024
DATA_URL_PATTERN = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


@dataclass(frozen=True)
class VisionCacheInfo:
    size: int
    max_size: int
    hits: int
    misses: int
    puts: int
    evictions: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ContentAddressedVisionCache:
    """Thread-safe LRU for projected vision features keyed by source content."""

    def __init__(self, max_size: int = 20) -> None:
        if max_size < 1:
            raise ValueError("vision cache size must be at least 1")
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._file_digests: dict[tuple[str, int, int], str] = {}
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def get(self, image_source: Any) -> Any:
        key = self.key_for(image_source)
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def put(self, image_source: Any, features: Any) -> None:
        key = self.key_for(image_source)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = features
            self._puts += 1

    def contains(self, image_source: Any) -> bool:
        key = self.key_for(image_source)
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._file_digests.clear()

    def info(self) -> VisionCacheInfo:
        with self._lock:
            return VisionCacheInfo(
                size=len(self._cache),
                max_size=self.max_size,
                hits=self._hits,
                misses=self._misses,
                puts=self._puts,
                evictions=self._evictions,
            )

    def key_for(self, image_source: Any) -> str:
        if isinstance(image_source, (list, tuple)):
            digest = hashlib.sha256()
            for item in image_source:
                encoded = self.key_for(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            return f"list:sha256:{digest.hexdigest()}"
        if isinstance(image_source, bytes):
            return digest_key(image_source)
        if isinstance(image_source, Path):
            return self._file_key(image_source)
        if isinstance(image_source, str):
            data_url = decode_data_url(image_source)
            if data_url is not None:
                return digest_key(data_url[1])
            path = Path(image_source).expanduser()
            if path.is_file():
                return self._file_key(path)
            return f"ref:{image_source}"
        if hasattr(image_source, "tobytes"):
            digest = hashlib.sha256()
            digest.update(str(getattr(image_source, "mode", "")).encode("utf-8"))
            digest.update(repr(getattr(image_source, "size", None)).encode("utf-8"))
            digest.update(image_source.tobytes())
            return f"pil:sha256:{digest.hexdigest()}"
        return f"object:{id(image_source)}"

    def _file_key(self, path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        cache_key = (str(resolved), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            existing = self._file_digests.get(cache_key)
        if existing is not None:
            return existing
        digest = digest_key(resolved.read_bytes())
        with self._lock:
            self._file_digests = {
                key: value for key, value in self._file_digests.items() if key[0] != str(resolved)
            }
            self._file_digests[cache_key] = digest
        return digest

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, image_source: Any) -> bool:
        return self.contains(image_source)


class VisualAssetStore:
    """Materialize base64 image inputs to stable content-addressed local paths."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        self.root = (root or Path.home() / ".cache" / "machboost" / "vision" / "assets").expanduser()
        self.max_image_bytes = int(max_image_bytes)

    def materialize(self, source: str | bytes | Path) -> str:
        if isinstance(source, Path):
            return self._local_path(source)
        if isinstance(source, bytes):
            return self._write_bytes(source, media_type=None)
        if not isinstance(source, str):
            raise TypeError(f"unsupported image source type: {type(source).__name__}")

        data_url = decode_data_url(source)
        if data_url is not None:
            media_type, data = data_url
            return self._write_bytes(data, media_type=media_type)

        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            return source
        path = Path(source).expanduser()
        if path.is_file():
            return self._local_path(path)

        try:
            data = base64.b64decode(source, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image source is not a file, URL, data URL, or valid base64: {source[:80]!r}") from exc
        return self._write_bytes(data, media_type=None)

    def materialize_all(self, sources: Iterable[str | bytes | Path]) -> list[str]:
        return [self.materialize(source) for source in sources]

    def _local_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.stat().st_size > self.max_image_bytes:
            raise ValueError(f"image exceeds {self.max_image_bytes} byte limit: {resolved}")
        return str(resolved)

    def _write_bytes(self, data: bytes, *, media_type: Optional[str]) -> str:
        if not data:
            raise ValueError("image payload is empty")
        if len(data) > self.max_image_bytes:
            raise ValueError(f"image exceeds {self.max_image_bytes} byte limit")
        suffix = image_suffix(data, media_type)
        digest = hashlib.sha256(data).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{digest}{suffix}"
        if not target.exists():
            temporary = target.with_suffix(target.suffix + f".{threading.get_ident()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        return str(target)


def normalize_multimodal_messages(
    messages: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    normalized: list[dict[str, str]] = []
    images: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("multimodal message parts must be objects")
                part_type = str(part.get("type") or "text")
                if part_type in {"text", "input_text"}:
                    text_parts.append(str(part.get("text") or ""))
                elif part_type in {"image_url", "input_image", "image"}:
                    image = part.get("image_url", part.get("image"))
                    if isinstance(image, dict):
                        image = image.get("url")
                    if not image:
                        raise ValueError("image message part is missing its URL or payload")
                    images.append(str(image))
                else:
                    raise ValueError(f"unsupported multimodal message part: {part_type}")
        else:
            raise ValueError("message content must be text or a multimodal parts list")

        raw_images = message.get("images") or ()
        if isinstance(raw_images, (str, bytes)):
            raw_images = (raw_images,)
        images.extend(str(image) for image in raw_images)
        normalized.append({"role": role, "content": "\n".join(part for part in text_parts if part)})
    return normalized, images


def decode_data_url(value: str) -> Optional[tuple[str, bytes]]:
    match = DATA_URL_PATTERN.match(value)
    if match is None:
        return None
    media_type = match.group(1) or "application/octet-stream"
    payload = match.group(3)
    try:
        data = base64.b64decode(payload, validate=True) if match.group(2) else unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid image data URL") from exc
    return media_type, data


def digest_key(data: bytes) -> str:
    return f"bytes:sha256:{hashlib.sha256(data).hexdigest()}"


def image_suffix(data: bytes, media_type: Optional[str]) -> str:
    normalized = (media_type or "").lower()
    known_types = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }
    if normalized in known_types:
        return known_types[normalized]
    signatures = (
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"GIF87a", ".gif"),
        (b"GIF89a", ".gif"),
        (b"BM", ".bmp"),
    )
    for signature, suffix in signatures:
        if data.startswith(signature):
            return suffix
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError(f"unsupported image media type: {media_type or 'unknown'}")
