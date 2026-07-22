# MachBoost for macOS

The native MachBoost app is a local chat client and controller for the same
daemon used by the Python package and CLI. It targets Apple Silicon and macOS
14 or newer. The release bundle includes a pinned CPython, MLX, `mlx-lm`,
`mlx-vlm`, and MachBoost runtime, so end users do not need Python, Homebrew, or
a separate MachBoost installation.

## Product Scope

- SwiftUI chat with streaming Markdown, code copy, stop, regenerate, edit and
  resend, persistent attachments, and generation statistics
- local SwiftData conversation history with search, rename, Markdown export,
  and delete
- tested-model catalog, explicit downloads, architecture preflight, load state,
  memory guidance, cancellation, and unload controls
- one resident daemon for several models, bounded queues, optional replicas,
  OpenAI-compatible and Ollama-compatible routes, and live metrics
- menu-bar lifecycle, optional launch at login, and Sparkle 2 updates
- localhost serving by default; authenticated LAN serving is opt-in

Chat history and imported attachment copies stay under MachBoost Application
Support. Model weights stay in the normal Hugging Face cache. The app has no
account, cloud sync, telemetry upload, or automatic model-weight download.

## Local Development

Install Xcode 16 or newer and XcodeGen 2.46.0, then accept the Xcode license:

```sh
sudo xcodebuild -license accept
brew install xcodegen
cd apps/macos
xcodegen generate
open MachBoost.xcodeproj
```

A Debug build can start the Python package from the repository, which keeps UI
iteration quick. Install the package into the Python selected by the app, or
set `MACHBOOST_SOURCE_ROOT` when launching tests:

```sh
python3 -m pip install -e .
xcodebuild test \
  -project apps/macos/MachBoost.xcodeproj \
  -scheme MachBoost \
  -destination 'platform=macOS,arch=arm64'
```

The checked-in Xcode project is generated from `project.yml`. Update the YAML
first and regenerate the project whenever target or package settings change.

## Embedded Runtime

The runtime manifest pins the CPython artifact and checksum. Python wheels are
hash-locked and constrained to CPython 3.13, arm64, and the macOS 14 platform:

```sh
./scripts/build_macos_runtime.sh
```

The resulting `apps/macos/Resources/runtime` directory is intentionally ignored
by Git. The build verifies the architecture and imports MachBoost, MLX,
`mlx-lm`, and `mlx-vlm` before succeeding.

## Serving And Security

The app normally binds the daemon to `127.0.0.1`. LAN mode binds to `0.0.0.0`,
generates a bearer token locally, and stores it in Keychain. Every LAN endpoint
except `/`, `/health`, and `/healthz` requires
`Authorization: Bearer <token>`. The token is passed to the daemon through its
process environment and is not placed in arguments or logs.

LAN traffic is authenticated but not encrypted. Use only a trusted private
network, or terminate TLS in an authenticated reverse proxy. Never expose the
plain HTTP listener directly to the public internet.

The app uses these stable discovery and control routes in addition to existing
MachBoost APIs:

- `GET /api/catalog`
- `GET /api/metrics`
- `POST /api/cancel`
- streaming `POST /api/pull`

Chat, generation, and pull requests may provide a client-generated
`request_id`; streaming events echo the same identifier.

## Release

Unsigned local builds do not need Apple credentials. Public DMG releases need
Vistrit Pandey's Developer ID, Apple team and notarization credentials, and
Sparkle EdDSA keys:

```sh
export MACHBOOST_DEVELOPMENT_TEAM=...
export MACHBOOST_DEVELOPER_ID='Developer ID Application: ...'
export MACHBOOST_NOTARY_PROFILE=...
export SPARKLE_PUBLIC_ED_KEY=...
export SPARKLE_PRIVATE_KEY=/secure/path/to/sparkle-private-key
./scripts/release_macos.sh 0.1.0
```

The release script builds the embedded runtime, archives the arm64 app, signs
nested Mach-O files and the app, creates and notarizes a DMG, staples the
ticket, runs Gatekeeper verification, writes a SHA-256 checksum, and produces a
signed Sparkle appcast. Publish the DMG, checksum, appcast, and release notes in
the matching GitHub release.

Before release, test a small text model, a 3B text model, and a supported vision
model on Apple Silicon. Also verify a clean-machine install, concurrent OpenAI
and Ollama requests, cancellation, overload behavior, token rotation, graceful
quit, and an upgrade from the previous signed build.
