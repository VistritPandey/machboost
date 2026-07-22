#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$ROOT/apps/macos"
MANIFEST="$APP_ROOT/Resources/RuntimeManifest.json"
LOCKFILE="$APP_ROOT/runtime-requirements.lock"
OUTPUT="${MACHBOOST_RUNTIME_OUTPUT:-$APP_ROOT/Resources/runtime}"
CACHE="${MACHBOOST_BUILD_CACHE:-$HOME/Library/Caches/MachBoost/build}"

read_manifest() {
  python3 - "$MANIFEST" "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

PYTHON_RELEASE="$(read_manifest python.release)"
PYTHON_VERSION="$(read_manifest python.version)"
PYTHON_ASSET="$(read_manifest python.asset)"
PYTHON_SHA256="$(read_manifest python.sha256)"
MACHBOOST_VERSION="$(read_manifest packages.machboost)"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${PYTHON_ASSET}"
ARCHIVE="$CACHE/$PYTHON_ASSET"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MachBoost desktop runtime builds require Apple Silicon macOS." >&2
  exit 2
fi

mkdir -p "$CACHE"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading pinned CPython runtime..."
  curl --fail --location --retry 3 --output "$ARCHIVE.partial" "$PYTHON_URL"
  mv "$ARCHIVE.partial" "$ARCHIVE"
fi

ACTUAL_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$PYTHON_SHA256" ]]; then
  echo "CPython archive checksum mismatch." >&2
  echo "expected: $PYTHON_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 3
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/machboost-runtime.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
tar -xzf "$ARCHIVE" -C "$WORK"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"
mv "$WORK/python" "$OUTPUT/python"
PYTHON="$OUTPUT/python/bin/python3"
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --only-binary=:all: \
  --platform macosx_14_0_arm64 \
  --python-version 3.13 \
  --implementation cp \
  --abi cp313 \
  --target "$SITE_PACKAGES" \
  --require-hashes \
  --requirement "$LOCKFILE"
"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-deps \
  --target "$SITE_PACKAGES" \
  "$ROOT"

find "$OUTPUT" -type d -name __pycache__ -prune -exec rm -rf {} +
"$PYTHON" - "$PYTHON_VERSION" "$MACHBOOST_VERSION" <<'PY'
import json
import platform
import sys

import machboost
import mlx
import mlx_lm
import mlx_vlm

expected_python, expected_machboost = sys.argv[1:]
assert platform.machine() == "arm64"
assert platform.python_version() == expected_python
assert machboost.__version__ == expected_machboost
print(json.dumps({
    "machboost": machboost.__version__,
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "mlx": getattr(mlx, "__version__", "installed"),
    "mlx_lm": getattr(mlx_lm, "__version__", "installed"),
    "mlx_vlm": getattr(mlx_vlm, "__version__", "installed"),
}, sort_keys=True))
PY

echo "Embedded runtime ready at $OUTPUT"
