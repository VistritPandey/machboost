#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$ROOT/apps/macos"
VERSION="${1:-}"
MODE="${2:-release}"
BUILD_NUMBER="${BUILD_NUMBER:-1}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
DIST="$ROOT/dist/macos"
ARCHIVE="$DIST/MachBoost.xcarchive"
DERIVED_DATA="$DIST/DerivedData"
DMG="$DIST/MachBoost-${VERSION}-arm64.dmg"
LOCAL_BUILD=false

if [[ -z "$VERSION" ]]; then
  echo "usage: $0 VERSION [--local]" >&2
  exit 2
fi
case "$MODE" in
  release)
    : "${MACHBOOST_DEVELOPMENT_TEAM:?Set MACHBOOST_DEVELOPMENT_TEAM to the Apple team ID}"
    : "${MACHBOOST_DEVELOPER_ID:?Set MACHBOOST_DEVELOPER_ID to the Developer ID Application identity}"
    : "${MACHBOOST_NOTARY_PROFILE:?Set MACHBOOST_NOTARY_PROFILE to a notarytool Keychain profile}"
    : "${SPARKLE_PUBLIC_ED_KEY:?Set SPARKLE_PUBLIC_ED_KEY to the Sparkle public key}"
    : "${SPARKLE_PRIVATE_KEY:?Set SPARKLE_PRIVATE_KEY to the Sparkle private key file}"
    ;;
  --local)
    LOCAL_BUILD=true
    SPARKLE_PUBLIC_ED_KEY="${SPARKLE_PUBLIC_ED_KEY:-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=}"
    ;;
  *)
    echo "usage: $0 VERSION [--local]" >&2
    exit 2
    ;;
esac

export DEVELOPER_DIR
"$ROOT/scripts/build_macos_runtime.sh"
command -v xcodegen >/dev/null || {
  echo "Install XcodeGen 2.46.0 before releasing." >&2
  exit 3
}

rm -rf "$DIST"
mkdir -p "$DIST"
(cd "$APP_ROOT" && xcodegen generate)

if $LOCAL_BUILD; then
  SIGNING_ARGUMENTS=(
    CODE_SIGN_STYLE=Manual
    DEVELOPMENT_TEAM=
    CODE_SIGN_IDENTITY=-
  )
else
  SIGNING_ARGUMENTS=(
    DEVELOPMENT_TEAM="$MACHBOOST_DEVELOPMENT_TEAM"
    CODE_SIGN_IDENTITY="$MACHBOOST_DEVELOPER_ID"
  )
fi

xcodebuild \
  -project "$APP_ROOT/MachBoost.xcodeproj" \
  -scheme MachBoost \
  -configuration Release \
  -destination "generic/platform=macOS" \
  -archivePath "$ARCHIVE" \
  -derivedDataPath "$DERIVED_DATA" \
  "${SIGNING_ARGUMENTS[@]}" \
  ARCHS=arm64 \
  MARKETING_VERSION="$VERSION" \
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  SPARKLE_PUBLIC_ED_KEY="$SPARKLE_PUBLIC_ED_KEY" \
  archive

APP="$ARCHIVE/Products/Applications/MachBoost.app"
SOURCE_RUNTIME="$APP_ROOT/Resources/runtime"
BUNDLED_RUNTIME="$APP/Contents/Resources/runtime"
if [[ ! -x "$SOURCE_RUNTIME/python/bin/python3" ]]; then
  echo "Embedded runtime was not built at $SOURCE_RUNTIME." >&2
  exit 4
fi
rm -rf "$BUNDLED_RUNTIME"
ditto "$SOURCE_RUNTIME" "$BUNDLED_RUNTIME"
if [[ ! -x "$BUNDLED_RUNTIME/python/bin/python3" ]]; then
  echo "Embedded runtime was not copied into the archived app." >&2
  exit 4
fi

SIGN_IDENTITY="${MACHBOOST_DEVELOPER_ID:--}"
SIGN_ARGUMENTS=(--force --timestamp --options runtime --sign "$SIGN_IDENTITY")
if $LOCAL_BUILD; then
  SIGN_ARGUMENTS=(--force --sign -)
fi
while IFS= read -r -d '' binary; do
  description="$(file -b "$binary")"
  if [[ "$description" == *Mach-O* ]]; then
    codesign "${SIGN_ARGUMENTS[@]}" "$binary"
  fi
done < <(find "$BUNDLED_RUNTIME" -type f -print0)
codesign "${SIGN_ARGUMENTS[@]}" --deep \
  --entitlements "$APP_ROOT/MachBoost/MachBoost.entitlements" \
  "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/machboost-dmg.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
ditto "$APP" "$STAGING/MachBoost.app"
ln -s /Applications "$STAGING/Applications"
hdiutil create \
  -volname MachBoost \
  -srcfolder "$STAGING" \
  -format UDZO \
  -ov \
  "$DMG"

hdiutil verify "$DMG"
shasum -a 256 "$DMG" > "$DMG.sha256"
if $LOCAL_BUILD; then
  echo "Local DMG ready:"
  echo "  $DMG"
  echo "  $DMG.sha256"
  exit 0
fi

NOTARY_ARGUMENTS=(--keychain-profile "$MACHBOOST_NOTARY_PROFILE")
if [[ -n "${MACHBOOST_RELEASE_KEYCHAIN:-}" ]]; then
  NOTARY_ARGUMENTS+=(--keychain "$MACHBOOST_RELEASE_KEYCHAIN")
fi
xcrun notarytool submit "$DMG" "${NOTARY_ARGUMENTS[@]}" --wait
xcrun stapler staple "$DMG"
spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"

GENERATE_APPCAST="$(find "$DERIVED_DATA/SourcePackages/artifacts" -type f -name generate_appcast -perm -111 | head -1)"
if [[ -z "$GENERATE_APPCAST" ]]; then
  echo "Sparkle generate_appcast tool was not found in DerivedData." >&2
  exit 4
fi
"$GENERATE_APPCAST" \
  --ed-key-file "$SPARKLE_PRIVATE_KEY" \
  --download-url-prefix "https://github.com/VistritPandey/machboost/releases/download/v${VERSION}/" \
  "$DIST"

echo "Release artifacts:"
echo "  $DMG"
echo "  $DMG.sha256"
echo "  $DIST/appcast.xml"
