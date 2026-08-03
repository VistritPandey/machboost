# Team Gateway

MachBoost Team Gateway turns one Apple Silicon Mac into a private inference
endpoint for a small team. It keeps supported MLX text and vision models
resident, accepts concurrent OpenAI- and Ollama-compatible requests, and adds
employee keys, limits, fair admission, local traces, and evaluations.

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
and exposes Team and Logs & evals controls under Server.

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
mutable visual state is not replica-safe. MachBoost v0.8.0 does not implement
continuous batching.

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

Existing `/api/chat`, `/api/generate`, `/v1/chat/completions`, model lifecycle,
workspace, metrics, streaming, and cancellation routes remain available.
