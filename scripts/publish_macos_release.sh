#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"
NOTES_FILE="${2:-}"
DIST="$ROOT/dist/macos"
TAG="v${VERSION}"
DMG="$DIST/MachBoost-${VERSION}-arm64.dmg"
CHECKSUM="$DMG.sha256"
APPCAST="$DIST/appcast.xml"

if [[ -z "$VERSION" || -z "$NOTES_FILE" ]]; then
  echo "usage: $0 VERSION RELEASE_NOTES.md" >&2
  exit 2
fi
if [[ ! -f "$NOTES_FILE" ]]; then
  echo "release notes not found: $NOTES_FILE" >&2
  exit 3
fi
for artifact in "$DMG" "$CHECKSUM" "$APPCAST"; do
  if [[ ! -f "$artifact" ]]; then
    echo "release artifact not found: $artifact" >&2
    echo "run ./scripts/release_macos.sh $VERSION first" >&2
    exit 4
  fi
done

gh auth status >/dev/null
git rev-parse --verify "${TAG}^{commit}" >/dev/null
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "GitHub release already exists: $TAG" >&2
  exit 5
fi

gh release create "$TAG" \
  "$DMG" \
  "$CHECKSUM" \
  "$APPCAST" \
  --verify-tag \
  --title "MachBoost ${VERSION} for macOS" \
  --notes-file "$NOTES_FILE"

echo "Published $TAG with the notarized DMG, checksum, and signed appcast."
