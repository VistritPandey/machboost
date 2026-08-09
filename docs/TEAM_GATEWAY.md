# Team Gateway

MachBoost Team Gateway turns one Apple Silicon Mac into a private inference
endpoint for a small team. It keeps supported MLX text and vision models
resident, accepts concurrent OpenAI- and Ollama-compatible requests, and adds
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

`GET /api/integrations` returns the same connection values for the active host.
`POST /v1/chat/completions` accepts OpenAI function tools, `tool_choice`, and
`parallel_tool_calls`. `POST /api/chat` accepts Ollama tool schemas. MachBoost
returns requested tool calls but never executes them; execution remains inside
the employee's coding agent and its permission system.

## Fairness And Concurrency

Each employee key has independent concurrent-request and requests-per-minute
limits. The model queue rotates between tenant keys, so one busy employee cannot
occupy every queued position. Within admission, `affinity_key` still chooses a
preferred replica when it is available, preserving useful session and workspace
cache locality.

Replicas are independent model instances. They improve isolation and can reduce
queue latency, but consume additional unified memory and do not promise a linear
GPU-throughput increase. MLX-VLM remains one replica per model because its
mutable visual state is not replica-safe. MachBoost v0.9.0 does not implement
continuous batching.

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
include the model, request, workspace, scope, principal, and revision. A hit
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
