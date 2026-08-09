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
- advanced MLX and MLX-VLM repository entry, with compatibility validation
  before download and automatic discovery after a compatible model is cached
- repository workspaces with local indexing, per-conversation selection,
  query-specific code retrieval, file and line citations, and manual reindexing
- one resident daemon for several models, bounded queues, optional replicas,
  OpenAI-compatible and Ollama-compatible routes, and live queue, latency,
  throughput, and memory metrics
- Team and Logs & evals views for scoped employee keys, model allowlists,
  per-key concurrency/rate limits, trace retention, and local evaluations
- Memory & fallback view for private/shared workspace memory, exact-reuse
  savings, external OpenAI-compatible providers, monthly budgets, and Keychain
  secrets
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
The macOS UI-test runner has hardened runtime disabled so Xcode can inject its
ad-hoc XCTest bundle on unsigned local and CI builds. The application target and
release archives retain hardened runtime; this test-only setting is required to
avoid a mismatched-Team-ID library-validation failure before XCTest starts.

The desktop code is split into explicit targets:

- `MachBoost` owns SwiftUI presentation, app state, daemon lifecycle, Keychain,
  menu-bar behavior, and update controls.
- `MachBoostDaemonClient` owns HTTP/NDJSON transport and stable API schemas.
- `MachBoostPersistence` owns SwiftData chat models, attachment imports, and
  conversation export.
- `MachBoostTests` and `MachBoostUITests` cover module contracts and user flows.

The Swift package manifest mirrors these boundaries for non-Xcode builds.

## Repository Workspaces

Choose **Repository > Open Repository...** in the chat toolbar to index a local
codebase. The selected workspace is stored with that conversation and remains
active for follow-up questions until **No Repository** is selected. The same
menu can refresh the index after code changes or remove MachBoost's local index.
Removing a workspace never deletes or modifies the source repository.

MachBoost does not place every file into the model context. The bundled daemon:

1. follows Git ignore rules and excludes symlinks, likely credentials,
   binaries, and oversized files;
2. stores bounded code chunks and extracted symbols in a local SQLite FTS5
   index;
3. sends a stable repository map plus focused line windows relevant to the
   current question, with dynamic evidence capped independently; and
4. returns file and line citations with the response.

On compatible MLX text models, workspace requests also reuse the longest exact
prompt prefix held by the resident model. This can reduce prefill for later
questions even when each question and retrieved suffix is different. It does
not accelerate the first workspace request, unrelated short prompts, or
output-token decoding. Plain chat does not opt into this cache.

Workspace metadata and indexes stay in MachBoost Application Support. Source
files remain in their original location and no repository content is uploaded.

## Team Memory And Provider Fallback

Open **Server → Memory & fallback** to inspect visible memory entries, exact
reuse counters, avoided-token accounting, and configured external providers.
Workspace chats write private bounded summaries by default. Shared entries are
administrator-controlled, and repository revisions plus dependency digests
invalidate stale records before retrieval. Exact-response reuse remains opt-in
and is restricted to deterministic non-streaming requests without tools or
images.

External providers must expose an OpenAI-compatible HTTPS endpoint. Provider
metadata, budgets, and usage counters live in the local team database. API keys
are stored in macOS Keychain and restored to daemon process memory through a
secret-only API after launch; restoring a key does not rewrite model lists,
pricing, timeouts, or budget configuration.

The app does not silently redirect ordinary chat to a paid provider. Routing is
selected per API request with `machboost.route`; the default is `local_only`.
The provider UI is an administrator configuration surface, while route choice
and workload policy remain in the calling application.

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
except `/health` and `/healthz` requires
`Authorization: Bearer <token>`. The token is passed to the daemon through its
process environment and is not placed in arguments or logs.

`127.0.0.1` is reachable only from the host Mac. To serve another computer,
open **Server → Developer**, choose **Enable authenticated LAN access**, and
copy the LAN endpoint and token shown by the app. MachBoost discovers the Mac's
active IPv4 address and produces remote-client settings such as:

```sh
export OPENAI_BASE_URL="http://192.168.1.50:11435/v1"
export OPENAI_API_KEY="YOUR_MACHBOOST_KEY"
export OLLAMA_HOST="http://192.168.1.50:11435"
```

The Models view and **Server → Overview/Developer** can explicitly load a
downloaded model, choose its keep-alive window, and run a compile warm-up before
the first client request.

LAN traffic is authenticated but not encrypted. Use only a trusted private
network, or terminate TLS in an authenticated reverse proxy. Never expose the
plain HTTP listener directly to the public internet.

The bundled daemon always enables Team Mode. Employee tokens are shown once
and stored only as hashes. Request traces default to metadata-only with bounded
retention; prompt and response content is saved only after the operator selects
redacted or full mode. The database stays in MachBoost Application Support.

The app uses these stable discovery and control routes in addition to existing
MachBoost APIs:

- `GET /api/catalog`
- `GET /api/metrics`
- `GET /api/workspaces`
- `GET /api/team/status`
- `GET /api/team/keys`, `POST /api/team/keys`
- `GET /api/memory`, `POST /api/memory/delete`
- `GET /api/cache/metrics`
- `GET`, `POST /api/providers`
- `POST /api/providers/secret`, `POST /api/providers/delete`
- `POST /api/team/keys/revoke`
- `POST /api/team/settings`
- `GET /api/traces`
- `GET /api/evaluations`, `POST /api/evaluations`
- `GET /api/integrations`
- `POST /api/workspaces`
- `POST /api/workspaces/index`
- `POST /api/workspaces/query`
- `POST /api/workspaces/delete`
- `POST /api/cancel`
- streaming `POST /api/pull`

Chat, generation, and pull requests may provide a client-generated
`request_id`; streaming events echo the same identifier.

## Release

Build an ad-hoc signed DMG for local packaging and runtime tests without Apple
credentials:

```sh
./scripts/release_macos.sh 0.9.0-local --local
open dist/macos/MachBoost-0.9.0-local-arm64.dmg
```

Local mode builds the locked runtime, archives the arm64 app, embeds and signs
every native runtime binary, verifies the app and disk image, and writes a
SHA-256 checksum. It deliberately skips notarization, stapling, and Sparkle
appcast generation. These builds expose **View Latest Release** instead of
starting Sparkle. Gatekeeper does not recognize them as notarized software, so
community users must approve the app manually after each downloaded update.

Public DMG releases need Vistrit Pandey's Developer ID, Apple team and
notarization credentials, and Sparkle EdDSA keys:

```sh
export MACHBOOST_DEVELOPMENT_TEAM=...
export MACHBOOST_DEVELOPER_ID='Developer ID Application: ...'
export MACHBOOST_NOTARY_PROFILE=...
export SPARKLE_PUBLIC_ED_KEY=...
export SPARKLE_PRIVATE_KEY=/secure/path/to/sparkle-private-key
./scripts/release_macos.sh 0.1.0
./scripts/publish_macos_release.sh 0.1.0 ./release-notes/0.1.0.md
```

The release script builds the embedded runtime, archives the arm64 app, signs
nested Mach-O files and the app, creates and notarizes a DMG, staples the
ticket, runs Gatekeeper verification, writes a SHA-256 checksum, and produces a
signed Sparkle appcast. The publisher requires an existing `v0.1.0` tag, an
authenticated GitHub CLI session, and an explicit release-notes file. It refuses
to overwrite an existing release and uploads the DMG, checksum, and appcast to
the matching GitHub release.

The `macOS Release` GitHub Actions workflow performs the same process for an
existing `v*` tag. Configure these repository secrets before pushing a release
tag or starting the workflow manually:

- `APPLE_DEVELOPMENT_TEAM`
- `APPLE_NOTARY_ISSUER_ID`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_KEY_P8_BASE64`
- `MACOS_CERTIFICATE_P12_BASE64`
- `MACOS_CERTIFICATE_PASSWORD`
- `MACOS_CI_KEYCHAIN_PASSWORD`
- `MACOS_DEVELOPER_ID`
- `SPARKLE_PRIVATE_KEY_BASE64`
- `SPARKLE_PUBLIC_ED_KEY`

For automatic signed releases on `v*` tag pushes, also set the repository
variable `MACHBOOST_SIGNED_RELEASES` to `true`. Without that variable, tag
pushes skip the credential-gated signing job so unsigned community releases can
be published separately. Manual workflow dispatch remains available regardless
of the variable.

The workflow imports signing material into a temporary Keychain, builds and
notarizes the DMG, validates the bundled runtime without invoking host Python,
generates release notes, publishes the GitHub release, and then removes the
temporary Keychain. A green ordinary CI run does not imply that signing,
notarization, or clean-machine hardware validation has run; those checks occur
in the credential-gated release workflow and the pre-release hardware pass.

Before release, test a small text model, a 3B text model, and a supported vision
model on Apple Silicon. Also verify a clean-machine install, concurrent OpenAI
and Ollama requests, cancellation, overload behavior, token rotation, graceful
quit, and an upgrade from the previous signed build.
