#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$ROOT/apps/macos"
VERSION="${1:-}"
BUILD_NUMBER="${BUILD_NUMBER:-1}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
DIST="$ROOT/dist/macos"
ARCHIVE="$DIST/MachBoost.xcarchive"
DERIVED_DATA="$DIST/DerivedData"
DMG="$DIST/MachBoost-${VERSION}-arm64.dmg"

if [[ -z "$VERSION" ]]; then
  echo "usage: $0 VERSION" >&2
  exit 2
fi
: "${MACHBOOST_DEVELOPMENT_TEAM:?Set MACHBOOST_DEVELOPMENT_TEAM to the Apple team ID}"
: "${MACHBOOST_DEVELOPER_ID:?Set MACHBOOST_DEVELOPER_ID to the Developer ID Application identity}"
: "${MACHBOOST_NOTARY_PROFILE:?Set MACHBOOST_NOTARY_PROFILE to a notarytool Keychain profile}"
: "${SPARKLE_PUBLIC_ED_KEY:?Set SPARKLE_PUBLIC_ED_KEY to the Sparkle public key}"
: "${SPARKLE_PRIVATE_KEY:?Set SPARKLE_PRIVATE_KEY to the Sparkle private key file}"

export DEVELOPER_DIR
"$ROOT/scripts/build_macos_runtime.sh"
command -v xcodegen >/dev/null || {
  echo "Install XcodeGen 2.46.0 before releasing." >&2
  exit 3
}

rm -rf "$DIST"
mkdir -p "$DIST"
(cd "$APP_ROOT" && xcodegen generate)

xcodebuild \
  -project "$APP_ROOT/MachBoost.xcodeproj" \
  -scheme MachBoost \
  -configuration Release \
  -destination "generic/platform=macOS,arch=arm64" \
  -archivePath "$ARCHIVE" \
  -derivedDataPath "$DERIVED_DATA" \
  DEVELOPMENT_TEAM="$MACHBOOST_DEVELOPMENT_TEAM" \
  CODE_SIGN_IDENTITY="$MACHBOOST_DEVELOPER_ID" \
  MARKETING_VERSION="$VERSION" \
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  SPARKLE_PUBLIC_ED_KEY="$SPARKLE_PUBLIC_ED_KEY" \
  archive

APP="$ARCHIVE/Products/Applications/MachBoost.app"
while IFS= read -r binary; do
  codesign --force --timestamp --options runtime --sign "$MACHBOOST_DEVELOPER_ID" "$binary"
done < <(
  find "$APP/Contents/Resources/runtime" -type f -perm -111 -print0 \
    | xargs -0 file \
    | awk -F: '/Mach-O/ {print $1}' \
    | awk '{ print length, $0 }' \
    | sort -rn \
    | cut -d' ' -f2-
)
codesign --force --deep --timestamp --options runtime \
  --entitlements "$APP_ROOT/MachBoost/MachBoost.entitlements" \
  --sign "$MACHBOOST_DEVELOPER_ID" "$APP"
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

xcrun notarytool submit "$DMG" \
  --keychain-profile "$MACHBOOST_NOTARY_PROFILE" \
  --wait
xcrun stapler staple "$DMG"
spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
shasum -a 256 "$DMG" > "$DMG.sha256"

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
