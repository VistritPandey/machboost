# Team Gateway

MachBoost Team Gateway turns one Apple Silicon Mac into a private inference
endpoint for a small team. It keeps supported MLX text and vision models
resident, accepts concurrent OpenAI Chat/Responses, Anthropic Messages, and
Ollama-compatible requests, and adds
employee keys, limits, fair admission, revision-aware memory, budgeted provider
fallback, local traces, and evaluations.

It is not a public edge gateway. MachBoost serves plain HTTP, does not terminate
TLS, and does not provide an internet-facing identity provider. Bind it only to
a trusted private network or place an authenticated TLS proxy in front of it.

## Start A Team Node

```sh
export MACHBOOST_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

machboost serve \
  --team \
  --host 0.0.0.0 \
  --port 11435 \
  --replicas 2 \
  --max-queue 64 \
  --queue-timeout 120
```

The environment token is the administrator credential. It is never placed in
process arguments or response logs. Team metadata defaults to
`~/.machboost/team.sqlite3`, uses SQLite WAL mode, and never contains plaintext
employee tokens.

The native macOS app starts its bundled daemon in Team Mode automatically. It
stores the database in `~/Library/Application Support/MachBoost/team.sqlite3`
and exposes Team, Memory & fallback, and Logs & evals controls under Server.

## Issue Employee Keys

Use the app, or create a key with the Python client:

```python
import os

from machboost import MachBoostClient

admin = MachBoostClient(
    "http://127.0.0.1:11435",
    api_token=os.environ["MACHBOOST_API_TOKEN"],
)
created = admin.create_team_key(
    "Alice - coding agent",
    scopes=(
        "inference",
        "models:read",
        "workspaces:read",
        "traces:read",
        "evaluations:read",
        "evaluations:write",
    ),
    allowed_models=("qwen2.5-coder:7b",),
    max_concurrent=2,
    requests_per_minute=60,
)
print(created["token"])
```

The token is returned once. The database stores only its SHA-256 digest. Revoke
the key from the app or with `revoke_team_key(key_id)`.

Available scopes are:

| Scope | Access |
|---|---|
| `inference` | Chat and generation on allowed models |
| `models:read` | Catalog, resident models, and integration settings |
| `models:write` | Pull, preload, and unload models |
| `workspaces:read` | Query registered repository workspaces |
| `workspaces:write` | Register, index, and remove workspaces |
| `traces:read` | Read traces owned by the employee key |
| `evaluations:read` | Read evaluation summaries |
| `evaluations:write` | Evaluate selected traces |
| `team:admin` or `*` | Manage keys and team policy |

## Connect Coding Agents

For OpenAI-compatible clients such as Cline or Kilo Code, choose their custom
OpenAI-compatible provider and set:

```sh
export OPENAI_BASE_URL="http://TEAM-MAC:11435/v1"
export OPENAI_API_KEY="mbk_employee_key"
```

For Ollama-compatible clients:

```sh
export OLLAMA_HOST="http://TEAM-MAC:11435"
export OLLAMA_API_KEY="mbk_employee_key"
```

Codex-style clients can use MachBoost as a Responses provider:

```toml
model = "muse-glimmer:30b"
model_provider = "machboost"

[model_providers.machboost]
name = "MachBoost"
base_url = "http://TEAM-MAC:11435/v1"
env_key = "MACHBOOST_API_KEY"
wire_api = "responses"
```

Claude Code can use the Anthropic Messages route:

```sh
export ANTHROPIC_BASE_URL="http://TEAM-MAC:11435"
export ANTHROPIC_AUTH_TOKEN="mbk_employee_key"
export ANTHROPIC_MODEL="muse-glimmer:30b"
claude
```

Claude Desktop has a separate native third-party inference gateway. It is not
an MCP server. The MachBoost app exposes it under **Apps → Claude Desktop**, or
the CLI can configure it directly:

```sh
# Use models resident on this Mac.
machboost launch claude-desktop

# Use models resident on a saved team host.
machboost launch claude-desktop --connection studio

# Restore the Claude provider that was active before MachBoost.
machboost launch claude-desktop --restore
```

Claude Desktop currently requests a fixed set of Claude model route names.
MachBoost advertises those routes and maps them to compatible, accessible
MachBoost models on the selected host. The app shows that mapping before it
changes Claude's configuration. It writes a reversible provider profile under
Claude's Application Support directory, keeps the gateway key out of command
arguments and logs, and restarts Claude after confirmation. Claude only accepts
plain HTTP gateways on loopback. When a saved LAN host uses HTTP, MachBoost
therefore starts a private authenticated loopback bridge, forwards requests to
the selected host with its saved key, and removes the bridge on restore. The
Claude profile never receives the remote host credential.

`GET /api/integrations` returns the same connection values for the active host.
`POST /v1/responses`, `POST /v1/messages`, and `POST /v1/chat/completions`
accept function tools and preserve streaming tool-call events. `POST /api/chat`
accepts Ollama tool schemas. The gateway returns requested tool calls but does
not execute arbitrary tools; execution remains inside the employee's coding
agent and its permission system.

### Connect MachBoost Desktop Apps

An employee opens **Connections** in the native app and chooses **Connect** next
to an automatically discovered device. The app asks for the employee key and
stores it in that Mac's Keychain. If Bonjour discovery is unavailable,
**Connect by address** accepts the LAN endpoint and key. The host enrollment response contains only compatible models that are
already cached on the host and allowed for that key. The employee can submit a
model request, but only the host administrator can approve the download from
**Server → Team**.

The app sends a random device identifier and periodic presence record. Presence
contains the device name, app version, selected model, optional workspace name,
and workspace revision fingerprint. It does not contain the workspace path,
file content, prompts, or credentials. The host records actual inference calls
tagged with that device identifier separately from heartbeat traffic.

Desktop coding mode is client-executed. Bounded file listing, reading, and
literal search run on the employee Mac. Exact replacement and file creation
pause for approval and also run on the employee Mac. Paths are canonicalized
under the selected repository; traversal, `.git`, symlink escapes, binary files,
large files, and ambiguous replacements are rejected. Only tool results enter
the chat request sent to the host. The host therefore does not need a mount of
every employee repository.

This is a hub-and-spoke topology, not peer-to-peer synchronization. The host
owns inference admission, resident models, traces, and optional reusable state;
each employee owns local source access and write approval.

### Serve Muse Glimmer 30B MLX

Muse Glimmer can be exposed through the same endpoint when the host has Apple
Silicon and at least 32 GB unified memory:

```sh
machboost pull muse-glimmer:30b
machboost warm muse-glimmer:30b --keep-alive -1
```

The default alias resolves to `mlx-community/Muse-Glimmer-30B-4bit` and runs
through MachBoost's native MLX-VLM backend. Image content parts, reasoning, and
function-tool schemas are preserved across all four protocol surfaces. The
model stays resident until its keep-alive expires or an administrator unloads
it. Ollama is not required for this path.

The explicit legacy alias `muse-glimmer:30b-mlx` still bridges the older Ollama
artifact for compatibility and historical benchmarking.

## Fairness And Concurrency

Each employee key has independent concurrent-request and requests-per-minute
limits. The model queue rotates between tenant keys, so one busy employee cannot
occupy every queued position. Within admission, `affinity_key` still chooses a
preferred replica when it is available, preserving useful session and workspace
cache locality.

Replicas are independent model instances. They improve isolation and can reduce
queue latency, but consume additional unified memory and do not promise a linear
GPU-throughput increase. MLX-VLM remains one replica per model because its
mutable visual state is not replica-safe. MachBoost does not implement
continuous batching. Simultaneous clients queue around each native replica.

## What Repository Sharing Actually Reuses

Registering a repository and retrieving relevant files is not, by itself, a
MachBoost-specific advantage. Coding agents already inspect repositories. The
server-side workspace path is available only when the host itself can read and
index that repository. Its advantage is that independent threads using the same
workspace content revision can reuse the resident MLX state for the exact
system and repository-map prefix. Query-specific evidence and the new question
are still evaluated normally. Private memories and exact responses remain in
their own access namespaces.

In an August 15 same-model audit, ten repository questions used the same loaded
Qwen2.5 3B MLX weights, tokenizer, prompts, and greedy settings. Median wall
time fell from 3.501 to 1.186 seconds (2.894x), with exact token equality in all
ten pairs. The median request reused 8,409.5 of 10,265 prompt tokens. Decode
speed itself did not improve.

A separate 10-client hosted diagnostic improved throughput from 0.267 to 1.351
requests/s and reduced median time to first token from 19.821 to 3.342 seconds.
Only 16 of 20 cached outputs were byte-identical to the uncached control, so
that 5.06x throughput observation is not an exactness or quality claim and is
not the default public performance contract.

This path works across chat threads and employee keys because the cache
namespace follows the workspace content revision. The revision combines Git
HEAD, when available, with the indexed path/digest manifest, so uncommitted and
untracked eligible source changes also invalidate reuse boundaries. It does not
cross workspace revisions, model instances, incompatible cache architectures,
or daemon restarts.

Run the same privacy-preserving benchmark against a registered workspace:

```sh
python3 scripts/benchmark_repository_reuse.py \
  --workspace-id WORKSPACE_ID \
  --model qwen2.5:7b \
  --primer "Explain the existing subsystem and cite its implementation." \
  --target "Design an adjacent change and cite the files to edit." \
  --runs 3
```

The script omits source paths, prompts, citations, and model text from its JSON
unless the caller explicitly requests citations.

## Team Memory And Exact Reuse

Team memory is available only on workspace-backed requests. It is deliberately
not one global chat transcript. Every entry is partitioned by workspace, scope,
principal where private, repository revision, and dependency digests. This
prevents one employee's unfinished work or one repository's facts from leaking
into another request.

Two scopes are supported:

| Scope | Write policy | Read policy |
|---|---|---|
| `private` | Any authenticated employee; the default | Only the same employee key and administrators |
| `team` | Administrators only | Authenticated workspace users |

Automatic memory records a bounded, redacted summary of a completed exchange,
not the full conversation. Retrieval uses local FTS ranking and a character
budget. Entries with a mismatched repository revision or changed dependency
digest are rejected as stale. Administrators can pin or publish reviewed facts,
fixes, procedures, decisions, and summaries through `POST /api/memory`.

```json
{
  "model": "qwen2.5-coder:7b",
  "messages": [{"role": "user", "content": "How should checkout retries work?"}],
  "stream": false,
  "machboost": {
    "workspace_id": "WORKSPACE_ID",
    "memory": {
      "mode": "private",
      "search": true,
      "remember": true,
      "max_chars": 12000,
      "exact_cache": true
    }
  }
}
```

Exact-response reuse is off by default. It applies only to deterministic,
non-streaming requests with temperature zero and no tools or images. Cache keys
include the model, request, workspace, scope, principal, and content revision. A hit
returns the previously recorded response without model execution and increments
avoided-token and avoided-cost counters. Sampling, visual inputs, tool calls,
streaming, repository changes, or a different employee namespace bypass it.

Inspect the ledger and savings locally:

```sh
curl -H "Authorization: Bearer $MACHBOOST_API_TOKEN" \
  'http://127.0.0.1:11435/api/memory?workspace_id=WORKSPACE_ID'
curl -H "Authorization: Bearer $MACHBOOST_API_TOKEN" \
  http://127.0.0.1:11435/api/cache/metrics
python3 scripts/benchmark_team_memory.py
```

The benchmark's prompt/completion savings are deterministic fixture accounting,
not a model-throughput measurement. It also verifies private isolation,
workspace isolation, shared retrieval, and revision invalidation.

The model-backed private-repository probe provides a separate memory result.
One prior exchange was retrieved in all three independent-thread rounds and a
narrow required-concept rubric increased from 4/8 to 5/8. The additional memory
raised the prompt from 4,243 to 4,657 tokens and median wall time from 5.341 to
5.581 seconds. Memory is therefore presented as optional historical context,
not as automatic token savings or faster inference.

## External Provider Fallback

Administrators can register OpenAI-compatible HTTPS providers for overflow,
outage, or model fallback. Provider metadata and monthly usage are stored in
SQLite. Plaintext API keys are not: the CLI daemon reads them from an environment
variable or process memory, while the native app stores them in Keychain and
restores them through the secret-only endpoint after daemon restart.

Requests choose a policy under `machboost.route`:

```json
{
  "model": "company-coder",
  "messages": [{"role": "user", "content": "Review this patch."}],
  "machboost": {
    "route": {"mode": "local_first", "provider_id": "provider_123"}
  }
}
```

| Mode | Behavior |
|---|---|
| `local_only` | Never calls a provider; default |
| `local_first` | Uses external inference only after a transient local failure before a response starts |
| `external_first` | Tries the selected provider, then local only after a transient provider failure |
| `external_only` | Requires the selected provider and never falls back locally |

Fallback is intentionally narrow. Queue overload, timeout, connection failure,
and selected 5xx/429 responses are transient. Authentication, request
validation, unsupported models, and exhausted monthly budgets fail closed.
Provider usage records request count, input/output tokens, configured unit
prices, and estimated USD cost. The check prevents new calls once recorded spend
reaches the monthly cap; it is not a prepaid billing guarantee for concurrent
in-flight requests.

External responses honor the OpenAI streaming contract. The current provider
transport requests a buffered upstream response and emits it as a valid SSE
chunk with `machboost.route.buffered_upstream=true`, rather than claiming native
upstream token streaming.

## Private Traces

Trace storage has four modes:

| Mode | Stored data |
|---|---|
| `off` | Nothing |
| `metadata` | Identity, endpoint, model, status, timing, and token counts |
| `redacted` | Metadata plus prompt/response content with common credentials removed |
| `full` | Metadata and complete prompt/response content |

The default is metadata-only, seven days, and 256 MiB of retained payload data.
Retention can be set to forever while keeping the disk cap. Redaction is a
best-effort safeguard, not a data-loss-prevention system; use metadata or off for
sensitive workloads.

Employee keys can read only their own traces. Administrators can inspect and
delete all traces. Trace writes occur after generation and are skipped entirely
in off mode; they are not part of token streaming.

## Evaluations

The deterministic evaluator summarizes completion rate, p50/p95 latency, time
to first token, and observed generation throughput for selected traces. An
optional resident model can score retained request/response pairs from 0 to 1
for relevance and correctness. Local judging requires redacted or full trace
content and remains model-based evaluation, not ground truth.

```python
traces = admin.traces(limit=25)
report = admin.evaluate_traces(
    [trace["id"] for trace in traces],
    name="Weekly coding-agent check",
    model="qwen2.5:7b",
)
print(report["summary"])
```

## API Surface

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/team/status` | Counts and trace policy |
| `GET` | `/api/team/connect` | Enroll a desktop client and return its permitted cached models |
| `POST` | `/api/team/presence` | Refresh privacy-bounded device presence |
| `GET` | `/api/team/clients` | List enrolled devices; administrator only |
| `GET`, `POST` | `/api/team/model-requests` | List requests as administrator or request a model as an employee |
| `POST` | `/api/team/model-requests/resolve` | Resolve a pending model request; administrator only |
| `GET`, `POST` | `/api/team/keys` | List or create employee keys |
| `POST` | `/api/team/keys/revoke` | Revoke a key |
| `POST` | `/api/team/settings` | Update trace policy |
| `GET` | `/api/traces` | List traces |
| `GET` | `/api/traces/{id}` | Read retained content |
| `POST` | `/api/traces/delete` | Delete selected or all traces |
| `GET`, `POST` | `/api/evaluations` | List or run evaluations |
| `GET` | `/api/integrations` | Client connection settings |
| `GET`, `POST` | `/api/memory` | List visible entries or publish an entry |
| `POST` | `/api/memory/delete` | Delete authorized memory entries |
| `GET` | `/api/cache/metrics` | Reuse, avoided-token, and namespace counters |
| `GET`, `POST` | `/api/providers` | List or configure external providers |
| `POST` | `/api/providers/secret` | Restore one process-only provider key |
| `POST` | `/api/providers/delete` | Remove provider metadata and in-memory key |
| `GET` | `/api/providers/usage` | Monthly provider token and cost accounting |

Existing `/api/chat`, `/api/generate`, `/v1/chat/completions`, model lifecycle,
workspace, metrics, streaming, and cancellation routes remain available.
