# machboost

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MachBoost is an alpha-stage, local-first inference server, team gateway, native macOS app, and Python package for MLX, MLX-VLM, and Hugging Face models. It offers an Ollama-like model workflow, keeps models resident between requests, and streams text and visual chat. Team Mode adds scoped employee keys, fair admission, revision-aware private/shared memory, exact-request reuse, local traces, evaluations, and budgeted external-provider fallback. Optional acceleration paths target fresh text decoding on selected Qwen models, reusable local text, repeated image inputs, and selected Qwen3-VL visual-prefill workloads. Muse Glimmer runs through its native Hugging Face MLX-VLM conversions; Ollama is not required for that path.

The paths have different contracts. Plain chat delegates generation to the selected backend and mainly provides residency and API compatibility. The optional DFlash backend proposes blocks for fresh prompts and emits only tokens approved by the target model. Context drafting instead proposes tokens from caller-supplied text; its cache-enabled MLX path remains experimental because a recent Llama 3.2 audit found one token-sequence mismatch in 21 pairs. Repeated-image acceleration reuses process-local visual work for unchanged image bytes. First-view Qwen3-VL compression is explicitly approximate and can change answers.

MachBoost does not upload telemetry, mutate global runtime settings, or change model weights. It does not claim universal speedups, file-identical equivalence across model conversions, or quality preservation for approximate visual compression.

### Performance Contract

MachBoost is not a universal `2x-8x` switch. A speedup measured on a context-backed completion or repeated image must not be applied to unrelated prompts, new images, different models, or different machines.

Context drafting helps only when the model's next tokens are recoverable from caller-supplied local text and the target model accepts those draft tokens. Repository workspaces use a separate mechanism: a stable file/symbol map and query-specific code retrieval stay within a bounded prompt, while MLX can reuse the exact stable prefix on later workspace requests. The question itself can be new, but the first request still pays normal indexing and prefill costs. Without a supported verified-decoding backend, a novel message outside a workspace falls back to native generation, where expected algorithmic speedup is about `1.0x` and the server layer can add latency.

Native MLX-VLM also keeps a model-revision-scoped prefix cache on local disk, bounded to 8 GiB by default. A previously seen system, tool, or repository prefix can therefore survive a daemon or app restart; generated answers are not cached. In a controlled Muse Glimmer 30B 4-bit restart test, a 3,448-token coding prompt measured `30.04s` TTFT cold and `1.79s` after restoring 3,432 prefix tokens from disk. Including a fresh model load in both processes, wall time fell from `34.66s` to `6.03s`. This is reuse of an identical stable prefix, not a speedup for never-seen prompt tokens. Set `MACHBOOST_MLX_APC_DISK=0` to disable persistence, or tune its location and bound with `MACHBOOST_MLX_APC_DISK_PATH` and `MACHBOOST_MLX_APC_DISK_GB`.

DFlash is the first MachBoost path aimed directly at unique output decoding. It is explicit opt-in, greedy-only, text-only, and limited to published model/draft pairs. On this Apple M5 Pro, the same Qwen3.5 4B BF16 target reached a `1.65x` median decode-throughput speedup across three fresh 512-token prompt families; per-prompt medians ranged from `1.31x` for code to `2.43x` for reasoning, and all three 128-token validation prefixes matched native greedy output. The same-weight 9B BF16 row reached `1.61x`, with one prompt at `2.42x`, but only 2/3 validation prefixes matched. Both used the shippable `dflash-mlx==0.1.8` wheel. A practical Qwen3.5 9B 4-bit control reached `1.32x` with adaptive verification and also diverged; fixed 16-token verification regressed to `0.84x` overall. These are decode results, not universal end-to-end, quality-equivalence, or short-answer claims.

Muse Glimmer uses native `mlx-vlm` 0.6.13 or newer. `muse-glimmer:30b` resolves to `mlx-community/Muse-Glimmer-30B-4bit`, and explicit 4-bit, 5-bit, 6-bit, 8-bit, BF16, MXFP4, MXFP8, and NVFP4 aliases are available. The compatibility name `muse-glimmer:30b-mlx` still reaches the older Ollama MLX artifact, but only when explicitly requested. The archived Ollama DFlash diagnostic is retained as historical evidence and is not a claim about the native MLX-VLM path.

| Likely fit | Why it can help |
|---|---|
| RAG and internal knowledge assistants | Answers often quote or closely follow retrieved source passages. |
| Repository-aware code completion | Generated code can continue patterns already present in the repository. |
| Policy, checklist, and runbook assistants | Responses frequently reproduce stable approved wording. |
| Config, JSON, and template generation | Outputs often contain predictable local structures and repeated fields. |
| Fresh reasoning on a supported DFlash target | Parallel draft blocks can reduce expensive target-model decode passes when acceptance is high. |
| Repeated questions over the same image | Visual encoding and matching prompt-prefix work may be reusable. |

| Usually not a fit | Expected behavior |
|---|---|
| A first workspace question or a unique question without a supported decode pair | Normal prefill; no reusable prefix exists yet. A topically unrelated question in the same workspace can still reuse the stable repository map. |
| Context-only drafting for brainstorming or creative writing | Little continuation is recoverable from supplied text, so this path is usually near native speed. DFlash has a separate workload-dependent contract. |
| Unsupported or low-acceptance DFlash workloads | Native generation can be faster; use the adaptive verifier and a paired workload benchmark. |
| A changed or first-seen image | Repeated-image cache does not apply. |
| An external backend without verifier hooks | Wrapper and measurement only; no native MachBoost token verification. |

Treat every workload as uncalibrated until it passes a same-model paired benchmark. `machboost bench-context` alternates execution order, checks generated token IDs, and withholds the aggregate speedup if any output differs. See [examples/python](examples/python/) for RAG, internal knowledge, code-continuation, and workload-evaluation examples.

### Current Status

| Path | Current evidence | Product status |
|---|---|---|
| Plain resident text chat | Native MLX decode through a local server; no drafting without context | usable, with measurable server/streaming overhead versus direct `mlx-lm` |
| Muse Glimmer 30B MLX-VLM | native Hugging Face MLX conversions with reasoning, vision, and tool-call transport; 131,072-token advertised context | usable with `mlx-vlm>=0.6.13`; the default 4-bit alias recommends at least 32 GB unified memory |
| Fresh-prompt DFlash decode | Qwen3.5 4B BF16 median was 1.65x with 3/3 validation prefixes exact; Qwen3.5 9B BF16 reached 1.61x with 2/3 exact | 4B alias is opt-in; 9B remains experimental; greedy-only and workload-dependent |
| Concurrent text API serving | bounded tenant-fair admission, explicit overload responses, per-key limits, and isolated model replicas | usable; replicas consume additional memory and do not guarantee higher GPU throughput |
| Team gateway | hashed scoped keys, model allowlists, enrolled desktop clients, model requests, fair queueing, configurable local traces, and evaluations | usable on a trusted private network; MachBoost does not terminate TLS |
| Team memory | private or administrator-published shared entries, workspace/revision/dependency isolation, bounded retrieval, and opt-in deterministic exact-response reuse | useful for recurring team work; not a decode-throughput speedup and not enabled as a universal response cache |
| External fallback | OpenAI-compatible providers with process-only secrets, monthly budgets, cost accounting, and transient-failure routing | usable for resilience; remote providers require HTTPS and streaming may be buffered when the upstream response is buffered |
| Repository workspace prefix reuse | latest 8,843-file Qwen2.5 3B audit reached 2.894x median with 10/10 exact token pairs; earlier 3B and 7B audits reached 3.021x and 3.282x with 6/6 exact pairs | promising for later questions over a stable indexed repo; not a first-request, arbitrary-model, or decode-throughput claim |
| Context-backed MLX text | latest broad Llama 3.2 3B suite was 1.008x aggregate with 20/21 exact pairs; favorable controlled continuations can be materially faster | experimental; never generalize a fixture result beyond its workload |
| Repeated unchanged image | 5.14x-17.44x model-level paired medians on one synthetic image and short extraction prompts | promising for repeated-image prefill; not a first-view or decode result |
| New-image Qwen3-VL compression | 1.70x median on ten TextVQA rows, with 70% normalized output equality and equal 8/10 aggregate task scores | approximate, opt-in, and not quality-equivalence evidence |

## Install

From a local checkout or downloaded source archive:

```sh
cd machboost
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional backends as needed:

```sh
pip install -e ".[mlx]"
pip install -e ".[hf]"
pip install -e ".[vision]"
pip install -e ".[dflash]"
pip install -e ".[video]"
pip install -e ".[all]"
```

Install the current CLI directly from its GitHub release tag:

```sh
python3 -m pip install "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git@v0.16.13"
python3 -m pip install "machboost[vision] @ git+https://github.com/VistritPandey/machboost.git@v0.16.13"
python3 -m pip install "machboost[dflash] @ git+https://github.com/VistritPandey/machboost.git@v0.16.13"
```

Update an existing install:

```sh
python3 -m pip uninstall -y machboost
python3 -m pip install "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git@v0.16.13"
machboost version
```

The explicit uninstall removes stale editable installs that otherwise continue
loading code from an older checkout. MachBoost is not currently distributed on
PyPI; the native app and tagged GitHub source are the supported release paths.

Check the install:

```sh
machboost doctor
machboost self-test
machboost list
python -m machboost self-test --json
```

### Native macOS App

The repository also contains a SwiftUI app for Apple Silicon Macs running
macOS 14 or newer. It provides streaming chat, local conversation history,
repository workspaces, text/code/folder/image attachments, model downloads,
resident-model controls, automatic long-chat summarization near the selected
context limit, server metrics, employee-key management, trace/evaluation views,
a direct device connection screen, Bonjour host discovery, load-aware request
routing, a Claude Desktop integration screen for local or shared models,
repo-scoped coding tools, connected-device and model-request
views, coding permission modes, a Git working-tree review panel,
a developer API view, and a menu-bar controller. Chats and
imported attachments remain local, model downloads always require confirmation,
and closing the window leaves the selected models available until they expire,
are unloaded, or MachBoost is quit.

Chat generation controls show an estimated context-token count and fractional
percentage, plus the last request's input-token count when reported by the
backend. The estimate includes retained chat text, enabled instructions, tool
schemas, and text attachment sizes; it is not an exact tokenizer measurement
or an image-token count. Automatic summarization checks the configured threshold
before the next request and after a reply. **Summarize Now** also works on short
chats. A successful summary replaces older turns in future requests without
deleting them from history; click the summary indicator to inspect it. Interrupted,
empty, or token-limited summaries leave the previous context unchanged and report
an error. Summaries are model-generated and may omit details.

The Models view and chat model picker search the bundled catalog and local cache
first, then query the public Hugging Face catalog for MLX repositories as the
user types. Live results are marked unverified: MachBoost validates the model
architecture and required files before offering a download, and never downloads
weights without confirmation.

For every downloaded repository, MachBoost reads the local architecture,
context limit, tokenizer response schema, and chat template. That metadata
drives text, vision, reasoning, and tool capability labels for arbitrary local
models, including compatible Qwen 3.x and DeepSeek variants; support does not
depend on a model having a built-in MachBoost alias.

Gemma 4 conversions that include Google's native `<|tool_call>` chat protocol
are also available in Dev mode and through the OpenAI/Ollama-compatible tool
APIs. MachBoost parses the model's structured calls and returns tool results to
the same resident model for the final answer.

Downloaded models can be loaded and compile-warmed explicitly from **Models**
or **Server → Developer**. The Server view also makes the network boundary
visible: `127.0.0.1` is local to the host Mac, while authenticated LAN mode
shows the active LAN IPv4 address and copyable OpenAI Responses, Anthropic
Messages, Chat Completions, and Ollama client settings for other computers.
Active downloads remain visible at the top of **Models** with aggregate bytes,
completed files, transfer rate, ETA, and the currently active Hugging Face/Xet
shards. A downloaded model can be unloaded and permanently removed from its
managed cache with the row's trash button.

Release builds bundle pinned arm64 CPython 3.13, MLX, `mlx-lm`, `mlx-vlm`, and
MachBoost dependencies. They do not depend on Homebrew, system Python, or an
existing package install. The source is under [apps/macos](apps/macos/); see the
[native app guide](apps/macos/README.md) for local builds, runtime verification,
signing, notarization, DMG creation, and update delivery. Version 0.15.0 and
later community builds embed a Sparkle EdDSA public key and consume a signed
appcast, so **Settings → Updates → Check Now** downloads, verifies, installs,
and relaunches in the app. Apple notarization remains separate from Sparkle
update authentication.

The bundled runtime remains self-contained for MLX and MLX-VLM models.
`muse-glimmer:30b` downloads the native 4-bit Hugging Face conversion after
confirmation. Partially downloaded repositories are not shown as runnable: the
catalog verifies that every indexed weight shard exists before marking a model
cached.

The app's **Extensions** view manages MCP connectors and reusable instructions.
Connectors may use a local stdio command or a remote Streamable HTTP endpoint.
MachBoost keeps credentials local and redacts their values from its API. Chats
see only two stable gateway tools for searching and calling connected tools, so
adding connectors does not inject every external schema into every prompt.
External tool calls require one-time approval in chat. Enabled reusable
instructions follow app requests to local or shared inference hosts. Connected
tools are enabled per chat with the **Tools** control, so ordinary chat keeps
its smaller, cache-friendly prompt.

An unsigned Apple Silicon community preview is available from
[GitHub Releases](../../releases/latest).
Drag the app from the DMG into Applications and attempt to open it once. Because
the community build is not Apple-notarized, open **System Settings → Privacy &
Security** and choose **Open Anyway**. This creates a local exception for future
launches. Verify the accompanying SHA-256 checksum before overriding Gatekeeper.

A public Developer ID-signed and notarized DMG is not claimed until those
credential-gated release steps have completed with the project owner's Apple
credentials.

## Quick Start

Start a native local chat from the command line. Short aliases select MLX on a compatible Apple Silicon installation and Hugging Face elsewhere:

```sh
machboost list
machboost pull qwen2.5:3b
machboost run qwen2.5:3b
```

`machboost run` starts a local server automatically when needed, loads the model,
and schedules its one-token MLX warmup in the background. The prompt appears after
weights load; if a message arrives before a first-ever kernel compile finishes, that
message waits behind the warmup and its TTFT includes the remaining compile time.
Large models can take tens of seconds on their first compile for a given model shape,
while later launches can reuse MLX's system kernel cache. Use `--warmup sync` to wait
before showing the prompt or `--warmup off` to compile on the first generation.
Models stay resident for five idle minutes by default; a background reaper releases
expired models even when no later command is issued. Preload explicitly with:

```sh
machboost warm qwen2.5:3b
machboost ps
```

Interactive terminals use a compact green chat layout with separate user,
answer, reasoning, tool, and performance rows. Redirected output stays plain
for scripts and logs. Set `NO_COLOR=1` to keep the layout without ANSI colors.
Reasoning effort is opt-in with `--think low|medium|high|xhigh` or `/think LEVEL`.
When a model inherently emits a reasoning channel, MachBoost detects and renders
that channel separately even without the flag. Some reasoning models can spend
the entire output budget before reaching an answer; increase `--max-tokens` or
use `/think off` where the model supports disabling reasoning.

Run a workspace-bounded coding session from the terminal:

```sh
cd /path/to/repository
machboost code muse-glimmer:30b \
  --workspace . \
  --permission-mode manual \
  --think low \
  --show-stats
```

The coding loop can list and read files, search with ripgrep, make exact block
replacements, create or delete files, run shell commands, and show the Git diff.
`manual` asks before edits and commands, `accept-edits` approves file edits but
asks before commands, `plan` blocks mutations, and `bypass` allows workspace
actions without prompts. Paths cannot escape the selected workspace and `.git`
metadata cannot be edited. Use `/mode`, `/diff`, `/workspace`, and `/tools` during
the session. This is an early local coding harness, not a claim of feature parity
with mature hosted coding agents.

Connect a local MCP server or save reusable instructions from the CLI:

```sh
machboost mcp add filesystem \
  --command npx \
  --arg -y \
  --arg @modelcontextprotocol/server-filesystem \
  --arg "$HOME/Projects"
machboost mcp list
machboost mcp test SERVER_ID

machboost skill add concise \
  --instructions "Answer directly and include exact file paths when relevant."
machboost skill list
```

MachBoost bundles the MCP client runtime. A configured connector runs only when
a user or approved model tool call invokes it; merely adding one does not send
chat content to that server.

Inside chat, `Ctrl-C` stops only the current reply and `Ctrl-D` unloads the current model and exits. `/bye` exits while preserving the five-minute idle window. Use `--keep-alive forever` only when indefinite residency is intentional.

### Muse Glimmer 30B MLX

Muse Glimmer is a roughly 30B multimodal agent model with a 131,072-token
context window, controllable reasoning, vision, and function tools. The default
MachBoost alias uses the native 4-bit MLX-VLM conversion and recommends at least
32 GB unified memory. Higher-bit variants require more memory.

```sh
python3 -m pip install "machboost[vision] @ git+https://github.com/VistritPandey/machboost.git@v0.16.13"
machboost pull muse-glimmer:30b
machboost run muse-glimmer:30b --think high --show-thinking --show-stats
machboost run muse-glimmer:30b --image ./screenshot.png --think medium
machboost run mlx-community/Muse-Glimmer-30B-4bit --backend mlx
```

`--backend mlx` selects the MLX runtime family. Vision repositories such as
Muse Glimmer are routed to MLX-VLM automatically.

The model stays behind the same OpenAI- and Ollama-compatible MachBoost
endpoint. Connected agents may provide tool schemas and execute returned calls;
ordinary API routes transport tool requests but do not grant file or shell access.
The explicit `machboost code` command is the exception: it executes only its own
workspace-bounded tools under the selected permission mode.
Run the complete local reasoning/tool/vision example with:

```sh
python3 examples/python/muse_glimmer_agent.py --image ./screenshot.png
```

The older Ollama MLX bridge remains available under its explicit compatibility
name for reproducing the archived DFlash diagnostic:

```sh
machboost bench muse-glimmer:30b-mlx \
  --engine both --backend ollama-mlx \
  --ollama-model muse-glimmer:30b-mlx \
  --runs 5 --warmups 2 --max-tokens 256 \
  --draft-num-predict 15
```

Use `--draft-num-predict 0` only as a diagnostic. Ollama's current public MLX
API has no direct DFlash-off switch, so MachBoost requests token logprobs to park
speculation; the result includes logprob materialization overhead. Meta's
published M4/M5 figures used ExecuTorch and are not MachBoost measurements. See
the [local evidence artifact](results/muse_glimmer_30b_mlx_20260811.json) for
the commands, hardware, feature smokes, concurrency behavior, and limitations.

Compare warm MachBoost and Ollama chat latency with unique prompts:

```sh
machboost bench llama3.2:3b --ollama-model llama3.2:3b --runs 3 --warmups 1
```

The benchmark reports client-observed time to first text, wall time, backend prompt timing, and decode tokens per second. Two-engine runs alternate which runtime executes first in each round, and every round uses a fresh prompt nonce. Cross-runtime output equality is recorded but is not an accuracy test: MLX and Ollama can use different templates, conversions, and quantization formats. This command measures serving/runtime suitability, not MachBoost's context-drafting algorithm.

Benchmark the actual MachBoost context algorithm against optimized native generation from the same loaded model:

```sh
machboost bench-context qwen2.5:3b \
  --prompt-file ./completion-prompt.txt \
  --context ./src \
  --runs 6 \
  --warmups 2 \
  --max-tokens 64
```

`bench-context` alternates native-first and MachBoost-first pairs, requires an even measured-run count, and compares generated token IDs. Its aggregate speedup is invalidated if any pair differs. The report also shows accepted draft tokens and logical target-call reduction. A valid result with zero accepted drafts means the tested context did not engage the algorithm.

Run target-verified decoding for a fresh prompt on a supported pair:

```sh
machboost pull qwen3.5:4b-dflash
machboost run qwen3.5:4b-dflash --show-stats
machboost bench-decode qwen3.5:4b \
  --prompt-file benchmarks/unique_decode_prompts.jsonl \
  --runs 3 --max-tokens 512 --no-eos
```

The `-dflash` aliases work in the ordinary OpenAI/Ollama `model` field and
download both their target and draft repositories. They intentionally select BF16 Qwen3.5 targets so the paired
benchmark compares the same target weights without a quantization confound.
They therefore download and use substantially more memory than the normal
4-bit aliases, plus a separate draft model. `bench-decode` measures native and
DFlash throughput from the same target and runs every JSONL row by default. It
also validates native and accelerated greedy token sequences for 128 tokens and
exits nonzero on any mismatch; set `--validation-tokens 0` only for an intentional
non-equivalent throughput diagnostic.
Remove `--no-eos` for realistic complete-answer latency. See the
[unique-request acceleration contract](docs/unique-request-acceleration.md)
and [Python example](examples/python/dflash_unique_prompt.py).
The normalized [unique-decode evidence matrix](results/unique_decode_qwen35_20260810.json)
contains the 4B and 9B BF16 rows, strict output-gate metadata, memory measurements,
and the quantized control.

Use full repository IDs when a model has no short alias. If the model is not cached, the selected backend may download it through its normal Hugging Face or MLX loader. Use `--local-files-only` with Hugging Face to require an existing cache.

Stream a raw completion for an editor or code tool:

```sh
machboost complete qwen2.5-coder:3b "def fibonacci(n):" --max-tokens 128
machboost complete qwen2.5:3b --file ./prompt.txt --context ./docs --show-stats
```

The resident server exposes Ollama-compatible routes, OpenAI Chat Completions
and Responses, and Anthropic Messages on `http://127.0.0.1:11435`:

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

### Claude Desktop

Claude Desktop can use MachBoost as its native third-party inference gateway.
This is not an MCP connection: Claude keeps its chat, Cowork, coding, tools,
permissions, and workspace UI, while MachBoost supplies the model inference.

Connect Claude Desktop to the MachBoost server on this Mac:

```sh
machboost launch claude-desktop
```

Connect it to a previously saved shared host instead:

```sh
machboost connect http://TEAM-MAC:11435 --name studio
machboost launch claude-desktop --connection studio
```

To advertise a specific local subset, repeat `--model` up to five times:

```sh
machboost launch claude-desktop \
  --model muse-glimmer:30b \
  --model qwen2.5-coder:7b
```

The macOS app exposes the same flow under **Apps → Claude Desktop**, with a
picker for **This Mac** or any saved MachBoost host. Claude discovers the
selected host's available models through `/v1/models`; requests arrive through
`/v1/messages` and `/v1/messages/count_tokens`. MachBoost uses Claude-compatible
route IDs for discovery and rewrites each request to the displayed MLX/HF model
before inference. Claude requires HTTPS for non-loopback gateways, so MachBoost
automatically gives Claude an authenticated `127.0.0.1` bridge when the selected
team host uses private-network HTTP. The bridge forwards streams to the selected
host, keeps the host key out of Claude's profile, and stops when the integration
is restored. Restore the profile at any time with:

```sh
machboost launch claude-desktop --restore
```

The Claude gateway keeps title-generation helpers off the model queue, trims
repeated client harness text, compacts tool contracts, and preserves the reusable
prompt prefix across tool rounds. On the development Apple Silicon machine, one
captured Claude Code request with resident Muse Glimmer 30B measured `9.42s` to
the first model token for a fresh session and `1.35s` after its coding prefix was
warm, down from `235.6s` before these gateway fixes. Because the model reasoned
before answering, visible text arrived at `12.01s` and `3.95s` respectively.
Those numbers describe that captured workload and machine; they are not a claim
that every new prompt or model receives the same speedup.

MachBoost preserves the previously active Claude inference profile during this
round trip, including an existing Ollama gateway configuration.

Each native-app conversation can pin **Automatic**, **This Mac**, or a named
team device. Automatic routing considers model availability, residency, network
latency, active work, queue depth, and temporary host failures; it only falls
back before visible output begins. The response footer records the device that
actually served the request. In Developer mode, MachBoost primes the stable
system/tool prefix before the first message and reports load, queue, prefill,
and cached-prefix time separately. This can substantially reduce TTFT for a
reused coding prefix, but it does not make a novel prompt or output decoding
free. The Anthropic gateway keeps core coding tools and query-relevant MCP tools
within a bounded schema budget instead of prefilling every connected tool. Set
`MACHBOOST_ANTHROPIC_TOOL_LIMIT` only when a deployment needs a different cap.

Coding agents can use their native tool protocol. MachBoost returns function
calls while the client keeps responsibility for file access, shell commands,
edits, and permission prompts:

```sh
# OpenAI Responses
curl http://127.0.0.1:11435/v1/responses -d '{
  "model":"qwen2.5-coder:7b",
  "input":"Inspect the repository and propose the next tool call."
}'

# Anthropic Messages / Claude Code gateway
export ANTHROPIC_BASE_URL="http://127.0.0.1:11435"
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_MODEL="qwen2.5-coder:7b"
```

### Repository Workspaces

A workspace indexes a repository outside the model context window. Git
repositories use `git ls-files --cached --others --exclude-standard`, so normal
ignore rules apply. MachBoost also rejects symlinks, likely credential files,
binaries, and files larger than the configured limit. SQLite FTS stores bounded
line chunks and extracted symbols locally. The model receives a deterministic
repository map plus only the best query-specific chunks, never every file.

Register and index a repository:

```sh
curl http://127.0.0.1:11435/api/workspaces \
  -H "Content-Type: application/json" \
  -d '{"path":"/absolute/path/to/repository","name":"My service"}'
```

Use the returned workspace ID with either the Ollama-compatible or
OpenAI-compatible chat route:

```sh
curl http://127.0.0.1:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5:7b",
    "workspace_id":"0123456789abcdef",
    "messages":[{"role":"user","content":"Where is request cancellation handled?"}],
    "stream":false
  }'
```

The response includes retrieved `path`, `start_line`, and `end_line` citations
under `machboost.workspace`. `GET /api/workspaces`,
`POST /api/workspaces/index`, `POST /api/workspaces/query`, and
`POST /api/workspaces/delete` provide lifecycle and direct-search operations.
OpenAI-compatible requests can place `workspace_id`, `workspace_top_k`, and
`workspace_max_chars` in a top-level `machboost` object.

The Python client exposes the same workflow:

```python
from machboost import MachBoostClient

client = MachBoostClient()
workspace = client.register_workspace("/absolute/path/to/repository")
response = client.chat(
    "qwen2.5:7b",
    [{"role": "user", "content": "Where is cancellation handled?"}],
    workspace_id=workspace["id"],
    stream=False,
)
print(response["message"]["content"])
print(response["machboost"]["workspace"]["citations"])
```

Workspace requests opt into a bounded MLX prompt-prefix cache and use workspace
affinity. The cache is shared by workspace content revision, not by chat thread
or employee, while private memory and exact responses retain their separate
access namespaces. Plain MLX chat keeps native cache behavior.

An August 9 production-shaped audit indexed a private 8,754-file monorepo and
used Qwen2.5 7B for standalone 192- and 256-token answers. Three adjacent coding
pairs reused a median 3,269 of 4,243 prompt tokens, reduced prefill from `2.306s`
to `0.605s` (`3.835x`), and reduced total wall time from `7.034s` to `5.377s`
(`1.314x`). A separate control primed a frontend question and then asked about
an unrelated scheduler subsystem. It reduced total time from `5.995s` to
`4.217s` (`1.426x`). All six new-question pairs produced byte-identical greedy
outputs; decode time was unchanged. This demonstrates centralized prefill reuse
beyond what merely attaching a repository to each coding-agent chat provides.

The same audit measured deterministic identical-request replay separately:
`2.474s` generated versus `0.091s` cached (`28.557x`) across three exact hits,
avoiding 8,985 prompt and 384 completion tokens. That number applies only to
identical deterministic requests. Semantic memory recovered one omitted
implementation concept in a small rubric, but added 414 prompt tokens and
`0.239s` median latency; it is a context-quality feature, not a speed claim.

The earlier short-answer prefix experiment remains useful as a prefill-heavy
upper bound. On one Apple M5 Pro run over a 183-file snapshot, six
alternating-order Qwen2.5 3B pairs reduced
median wall time from `3.144s` to `1.024s` (`3.021x`), and Qwen2.5 7B reduced it
from `6.587s` to `1.998s` (`3.282x`). All 12 pairs matched generated token IDs.
Five rows per model used different questions and retrieved evidence; the
remaining row repeated the priming question. Across only those five different
questions, the median paired speedup was `2.971x` on 3B and `3.232x` on 7B;
the repeated-prime row reached `13.055x` and `16.637x`, respectively. These are
short, warm,
prefill-heavy requests with 10.2K-10.6K-token prompts, not claims about first
requests, decode rate, every repository, or every architecture. Qwen3.5 9B
could not safely trim its hybrid recurrent cache and produced no valid speedup,
so that path remains native.

Reproduce the same-model comparison:

```sh
python3 scripts/benchmark_repository_reuse.py \
  --workspace-id WORKSPACE_ID \
  --model qwen2.5:7b \
  --primer "Explain the existing subsystem and cite its implementation." \
  --target "Design an adjacent change and cite the files to edit." \
  --runs 3

python3 scripts/benchmark_workspace_prefix.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --workspace . \
  --runs 6 \
  --max-tokens 16 \
  --max-context-chars 32000
```

See the [3B artifact](results/workspace_prefix_qwen25_3b_20260729.json) and
[7B artifact](results/workspace_prefix_qwen25_7b_20260729.json), plus the
[sanitized private-repository audit](results/team_repository_reuse_qwen25_7b_20260809.json).

### Concurrent API Serving

Run MachBoost as a long-lived inference endpoint for multiple application clients:

```sh
export MACHBOOST_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
machboost serve \
  --host 0.0.0.0 \
  --port 11435 \
  --replicas 2 \
  --max-queue 64 \
  --queue-timeout 120
```

Binding to a non-loopback address automatically requires the token. Send it on
every API request except the minimal health routes:

```sh
curl -H "Authorization: Bearer $MACHBOOST_API_TOKEN" \
  http://127.0.0.1:11435/api/metrics
```

Each text-model replica owns an independent accelerator and mutable generation cache. Requests are admitted through a bounded queue that rotates between authenticated tenant keys, then routed to an available replica. This prevents one employee's queued burst from blocking every other employee while preserving FIFO behavior for ordinary local clients. When every replica is busy and the queue is full, MachBoost returns HTTP `503` with `{"code":"queue_full"}` before starting a streaming response. Per-request queue wait and replica selection are returned under `machboost.scheduler`; aggregate active, queued, rejected, timed-out, per-tenant, and per-worker counts are available from `/api/ps` and `machboost ps --json`.

Use `affinity_key` for best-effort routing of a user, document, or session to the same available replica:

```python
response = client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Summarize the current incident."}],
    affinity_key="incident-4821",
    queue_timeout=2.0,
    stream=False,
)
```

Replicas are an opt-in concurrency and isolation control, not a promised throughput multiplier. They load another model instance and therefore use additional unified memory. On one local Qwen2.5 3B MLX audit with four clients, 24 requests, greedy decoding, and a 64-token limit, two replicas improved median time to first token from `2.089s` to `1.234s` and median total latency from `2.681s` to `2.435s`; aggregate decode throughput changed only from `94.15` to `96.12 tok/s`, while p95 total latency regressed from `2.791s` to `3.132s`. All `24/24` same-request output hashes matched. See [the concurrency artifact](results/concurrency_qwen25_3b_mlx_20260721.json) and reproduce it with:

```sh
python3 scripts/benchmark_concurrency.py qwen2.5:3b \
  --endpoint http://127.0.0.1:11435 \
  --backend mlx \
  --mode generate \
  --clients 4 \
  --requests 8 \
  --rounds 3 \
  --max-tokens 64
```

MLX-VLM currently remains limited to one replica because its visual and prompt-state caches are mutable. Even when the server is started with a higher text replica count, a visual model receives one worker and concurrent visual requests queue safely on it. MachBoost does not yet implement continuous batching, which is the more promising route to higher aggregate GPU utilization with one weight copy.

Loopback serving remains backward compatible and does not require a token. LAN
serving requires bearer authentication but does not provide TLS; use a trusted
private network or an authenticated TLS reverse proxy, and never expose the
plain HTTP listener directly to the public internet. The native app generates
its LAN token locally, stores it in Keychain, and passes it to the daemon without
placing it in process arguments or logs.

To connect from another machine, enable authenticated LAN access under
**Server → Developer** and use the displayed address instead of loopback:

```sh
export OPENAI_BASE_URL="http://192.168.1.50:11435/v1"
export OPENAI_API_KEY="YOUR_MACHBOOST_KEY"
export OLLAMA_HOST="http://192.168.1.50:11435"
export ANTHROPIC_BASE_URL="http://192.168.1.50:11435"
export ANTHROPIC_AUTH_TOKEN="YOUR_MACHBOOST_KEY"
```

The MachBoost CLI can save the same host without persistent shell variables.
The token prompt writes to macOS Keychain; the profile file contains only the
name and endpoint:

```sh
machboost connect 192.168.1.50:11435 --name studio
machboost connections --probe --model qwen2.5:7b
machboost use auto
machboost run qwen2.5:7b
machboost ps
machboost use studio  # pin commands to one host when needed
machboost use local
machboost disconnect studio
```

Saving a host enables automatic routing by default. In `auto` mode, the CLI
probes this machine and every saved host concurrently, checks whether the
requested model is cached and resident, and estimates completion time from
round-trip latency, replicas, active requests, queued requests, and requests
already reserved by this client. A transient failure is retried on the next
ranked host only when no output has been emitted; a response is never replayed
mid-stream. `machboost connections --probe --model MODEL` prints the live
ranking, and `/route` shows it inside interactive chat. The connection profile
format is portable; non-macOS clients can provide a saved host key through
`MACHBOOST_API_TOKEN_<CONNECTION_NAME>`.

The address above is illustrative; the app displays the current host Mac's
reachable LAN address. The client and server must be able to reach each other
on the selected network and port.

Discovery and control clients can use `GET /api/catalog`, `GET /api/metrics`,
`GET /api/workspaces`, and `POST /api/cancel`. Chat, generation, and pull requests accept an optional
`request_id`, which is echoed in streaming events. `/api/pull` supports NDJSON
progress and cancellation while retaining its non-streaming response contract.

### Team Gateway

One Apple Silicon Mac can serve several employees through their existing coding
agents without sharing the administrator credential:

```sh
export MACHBOOST_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
machboost serve \
  --team \
  --host 0.0.0.0 \
  --replicas 2 \
  --max-queue 64
```

Create employee keys from the native app's **Server → Team** view or the Python
client. Each key can limit scopes, allowed models, concurrent requests, and
requests per minute. Plaintext employee tokens are returned only once; the local
SQLite database stores their hashes.

```python
from machboost import MachBoostClient

admin = MachBoostClient(api_token="ADMIN_TOKEN")
created = admin.create_team_key(
    "Alice - coding agent",
    allowed_models=("qwen2.5-coder:7b",),
    max_concurrent=2,
    requests_per_minute=60,
)
print(created["token"])
```

Connect OpenAI-compatible tools with:

```sh
export OPENAI_BASE_URL="http://TEAM-MAC:11435/v1"
export OPENAI_API_KEY="mbk_employee_key"
```

MachBoost desktop clients connect from **Connections**. The inference-pool view
shows this Mac and saved remote hosts separately; this Mac is never offered as a
remote connection. Nearby hosts appear through Bonjour, while **Connect by
address** remains available when discovery is blocked. The app stores keys in
Keychain and shows each device's loaded models, active requests, and queue
depth. A request is routed only to an online host where the
selected model is ready. A resident copy is preferred until measured queue
pressure or immediately reserved in-flight requests make an idle compatible host
the better choice. The employee Mac can also join the pool when it has the model.

Routing is client-side and occurs before generation begins. MachBoost does not
migrate a partially streamed response, combine GPU memory across Macs, or make a
model available on a host where it is not downloaded. Generic API clients still
use the single endpoint they target. An employee can request another repository
or alias; the host must approve and download it from **Server → Team**. The host
view reports device presence, selected model, workspace name/revision
fingerprint, and actual inference request counts. It does not receive the
client's repository path.

In desktop coding mode, repository list/read/search operations execute on the
employee Mac. **Manual** asks before every write, **Auto** approves bounded
exact replacements but asks before broader changes, **Accept edits** approves
file writes, **Plan** exposes read-only tools, and **Bypass permissions**
approves all available repository tools. Every mode retains the selected-folder
boundary. Only bounded tool results are sent to the selected inference host.
This avoids granting the host filesystem access and keeps the same coding UI
available to employees who cannot run the model locally.
The chat preserves the streamed order of reasoning, visible prose, and one or
more tool-call rounds. Tool activity is shown in collapsible human-readable
rows instead of raw model protocol. Approved edits include a bounded patch preview plus Open File
and Reveal in Finder actions. A trailing `branch -> working tree` panel shows
the final repository-wide Git diff, while per-message patches retain the change
history. Model protocol tokens are not shown as messages. Reasoning is disabled
by default where the model supports that choice. Muse Glimmer always reasons, so
MachBoost uses its documented `low` setting as the fast default and exposes
`low`, `medium`, `high`, and `xhigh` in Generation controls.
Throughput shown in chat uses total model tokens divided by backend decode time
across the complete assistant turn, including hidden reasoning, tool protocol,
and follow-up rounds. It is not a visible-word rate.

Codex-style clients can select MachBoost as a Responses provider:

```toml
model = "qwen2.5-coder:7b"
model_provider = "machboost"

[model_providers.machboost]
name = "MachBoost"
base_url = "http://TEAM-MAC:11435/v1"
env_key = "MACHBOOST_API_KEY"
wire_api = "responses"
```

Claude Code can use the Anthropic Messages gateway:

```sh
export ANTHROPIC_BASE_URL="http://TEAM-MAC:11435"
export ANTHROPIC_AUTH_TOKEN="mbk_employee_key"
export ANTHROPIC_MODEL="qwen2.5-coder:7b"
claude
```

Responses, Anthropic Messages, OpenAI chat, and Ollama chat routes accept
function tool definitions and return multiple requested tool calls. MachBoost
does not execute arbitrary tools at the gateway. Generic callers retain their
normal permission boundary; the native app implements only its documented,
repo-scoped coding tools and requires approval before writes.

Repository-aware requests can also opt into the team memory ledger. Private
entries remain visible only to their employee key. Shared entries must be
published by an administrator, and retrieval rejects entries whose repository
revision or file dependency digests are stale. Deterministic exact-response
reuse is separately opt-in with `machboost.memory.exact_cache`; sampled, tool,
image, and streaming requests are never served from that cache.

```json
{
  "model": "qwen2.5-coder:7b",
  "messages": [{"role": "user", "content": "How do we retry checkout timeouts?"}],
  "stream": false,
  "machboost": {
    "workspace_id": "WORKSPACE_ID",
    "memory": {"mode": "private", "remember": true, "exact_cache": true}
  }
}
```

Administrators can configure an OpenAI-compatible provider as `local_first`,
`external_first`, or `external_only`. Local fallback occurs only for transient
failures such as queue overload or timeout; authentication, validation, and
budget errors fail closed. API keys live in process memory, an environment
variable, or the macOS Keychain, never in the team database.

The native chat route menu and CLI can map a local model to a different paid
provider model. The response records whether local or external inference was
used, provider latency, and configured cost. External responses are currently
buffered before MachBoost emits compatible stream events, so this fallback is a
resilience/capacity control rather than a local-inference speedup claim.

```sh
machboost run qwen2.5:7b \
  --route local_first \
  --provider production \
  --provider-model paid-model
```

```json
{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "Summarize the incident."}],
  "machboost": {
    "route": {
      "mode": "local_first",
      "provider_id": "production",
      "model": "paid-model"
    }
  }
}
```

Run the deterministic five-developer isolation/reuse benchmark with:

```sh
python3 scripts/benchmark_team_memory.py
```

The committed test checks private and workspace isolation, revision invalidation,
shared retrieval, and token/cost accounting. Its fixture records five exact
hits avoiding 12,000 prompt tokens and 600 completion tokens. Those values are
synthetic accounting inputs; the benchmark explicitly does not claim faster
model decoding.

The private-repository audit adds a model-backed memory probe. Shared memory
retrieved the prior exchange in `3/3` independent-thread runs and increased a
narrow required-concept rubric from `4/8` to `5/8`, while median wall time rose
from `5.341s` to `5.581s`. This is evidence that memory can recover relevant
prior work, not evidence that it saves tokens or improves general answer quality.

Trace storage is opt-in by content level: `off`, `metadata`, `redacted`, or
`full`. Metadata-only is the default, with seven-day retention and a 256 MiB
payload cap. Selected traces can receive deterministic latency/throughput
evaluations or an optional score from a resident local judge model. Redaction is
best effort, not a DLP system.

See the [Team Gateway guide](docs/TEAM_GATEWAY.md) and the
[administration example](examples/python/team_gateway_admin.py) for scopes,
API routes, private-network deployment, retention, and client setup. Team Mode
does not change the acceleration contract: unique first prompts still run at
native model speed unless they qualify for a documented reuse path.

### Visual Chat

Install the vision extra and run a supported MLX-VLM model with a local image:

```sh
pip install -e ".[vision]"
machboost pull qwen2.5-vl:3b
machboost run qwen2.5-vl:3b --image ./invoice.png --show-stats
machboost run qwen3-vl:4b --image ./invoice.png --show-stats
machboost run qwen3.5:4b --image ./invoice.png --show-stats
```

The interactive session keeps the model and attached image warm. Use `/image PATH` to attach another image, `/video PATH` to attach selected video frames, `/images` to inspect attachments, and `/clear-images` to remove them. The same path is available to Python applications:

```python
from machboost import MachBoostClient, ensure_server

client, _ = ensure_server()
response = client.chat(
    "qwen2.5-vl:3b",
    [{"role": "user", "content": "Return only the invoice total."}],
    images=["./invoice.png"],
    options={"temperature": 0.0, "num_predict": 32},
    stream=False,
)
print(response["message"]["content"])
```

Image reuse is content-addressed: changing the file bytes creates a new cache identity. Set `--no-vision-cache` or `options={"no_vision_cache": True}` for an uncached control. This optimization targets repeated questions over the same image; the first image request still performs normal vision encoding and prefill. Cache capabilities are model-specific, and MachBoost disables projected-feature reuse when a model requires additional visual tensors that cannot be cached safely.

### Experimental First-View Acceleration

Qwen3-VL can opt into prompt- and image-aware post-fusion visual-token compression for a new image and question:

```sh
machboost run qwen3-vl:8b \
  --image ./document.png \
  --vision-tokens auto \
  --show-stats
```

The image still passes through the full-resolution vision encoder. Qwen3-VL processes every visual token through its required deep-stack injections and the selected number of early language layers. MachBoost then groups the visual states spatially, preserves internally diverse groups, merges the rest with query-weighted pooling, and sends the shorter sequence through the remaining layers. The request bypasses visual and prompt caches, so the reported gain is independent of prior images or prompts.

`auto` classifies the request as general, document/text, chart, spatial, or multi-image. It selects a retention ratio, layer boundary, and token bucket from built-in experimental profiles. Manual experiments can use `merge`, `adaptive`, or the deterministic `random` control with `--vision-token-ratio`, `--vision-token-layer`, and `--vision-token-bucket`. Do not deploy a profile without calibrating it on representative data:

```sh
python3 scripts/benchmark_vision_tokens.py \
  --model qwen3-vl:8b \
  --datasets chartqa,docvqa,mmmu,textvqa \
  --samples-per-dataset 20 \
  --output results/local/vision-token-ablation.json

python3 scripts/calibrate_vision_tokens.py \
  results/local/vision-token-ablation.json \
  --output vision-calibration.json

machboost run qwen3-vl:8b \
  --image ./document.png \
  --vision-tokens auto \
  --vision-calibration ./vision-calibration.json
```

The calibrator excludes random pruning from deployment and requires minimum sample count, a speedup confidence bound, task-accuracy retention, and normalized output agreement. If no candidate passes, that workload resolves to `off`.

This path is approximate and disabled by default. It currently supports batch-one Qwen3-VL requests only, cannot be combined with `--cold-vision`, and can change wording or answers. On a 10-image TextVQA pilot with Qwen3-VL 8B, 35% visual retention produced a 1.67x aggregate wall-time speedup and 1.70x median paired speedup. Baseline and compressed paths each matched the dataset answer on 8 of 10 questions; normalized outputs were equal on 7 of 10. A 30% follow-up retained the same task score but was slower, so 35% remains the measured profile rather than assuming that more pruning is always better.

### Video Inputs

Video input requires FFmpeg plus Pillow:

```sh
brew install ffmpeg
pip install -e ".[vision,video]"

machboost run qwen3-vl:8b \
  --video ./clip.mp4 \
  --video-fps 2 \
  --video-change-threshold 0.08 \
  --video-max-frames 12 \
  --vision-tokens auto
```

MachBoost samples frames into a content-keyed local cache, computes RGB frame-to-frame change on compact thumbnails, keeps scene changes plus the first and final frame, and caps the result by change strength. The selected images are passed to the VLM in chronological order. This is a generic frame-selection adapter, not a native video encoder, and aggressive selection can miss motion or short events.

The sampler is also public Python API:

```python
from machboost import TemporalVideoSampler

selection = TemporalVideoSampler().sample(
    "./clip.mp4",
    fps=2.0,
    change_threshold=0.08,
    max_frames=12,
)
print(selection.to_dict())
```

On the repository's three-scene integration fixture, the selector keeps four chronological frames from a 12-frame uniform budget, including both color transitions. This verifies frame reduction and transition coverage only; it is not a VLM speed or quality benchmark.

Use the high-level `Accelerator` when you want MachBoost to load a model and build the draft corpus from strings, files, or directories. Calibrate before enabling the boosted path for a workflow:

```python
from machboost import Accelerator, GatePolicy

boost = Accelerator.from_mlx(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    context_paths=["./docs", "./src"],
    ngram=2,
    max_draft_tokens=8,
    candidate_limit=1,
)

calibration = boost.calibrate(
    [
        "Copy the exact release checklist item from the local docs:",
        "Emit the deployment JSON field from the local config:",
    ],
    max_tokens=32,
    gate_policy=GatePolicy(min_speedup=1.05, min_acceptance_rate=0.10),
)

print(calibration.summary)

if calibration.enabled:
    text, stats = boost.generate(
        "Copy the exact release checklist item from the local docs:",
        max_tokens=64,
    )
    print(text)
    print(stats.estimated_speedup)
```

For chat models, use the chat-aware API. It applies the tokenizer's chat template and stops on special end-of-turn tokens:

```python
from machboost import Accelerator

boost = Accelerator.from_huggingface(
    "Qwen/Qwen2.5-3B-Instruct",
    context_paths=["./docs", "./src"],
    local_files_only=True,
)

text, stats = boost.generate_chat(
    [{"role": "user", "content": "Summarize the local release checklist in one sentence."}],
    max_tokens=64,
)

print(text)
```

For custom runtimes, wrap a verifier-capable service:

```python
from machboost import machboost

boosted = machboost(
    some_service,
    corpus_tokens=local_context_tokens,
    ngram=4,
    max_draft_tokens=8,
)

tokens, stats = boosted.generate(prompt_tokens, max_tokens=128)
```

Real speedups require the wrapped service to expose a verifier hook such as:

```python
verify(prefix_tokens, candidate_tokens) -> accepted_count
```

A black-box service with only `next_token(prefix_tokens)` remains exact, but it cannot skip target-model work.

Applications can control a resident server directly:

```python
from machboost import MachBoostClient

client = MachBoostClient()
client.load("qwen2.5:3b", keep_alive="5m", warmup=True)

for chunk in client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Summarize the deployment policy."}],
    options={"context_paths": ["./docs"], "num_predict": 128},
    stream=True,
):
    print(chunk.get("message", {}).get("content", ""), end="", flush=True)
```

## When It Helps

MachBoost is designed for workloads where the model is likely to continue with text already present nearby:

- repo or source-code continuation
- config and JSON generation
- policy or documentation copying
- RAG answers that quote retrieved context
- repeated logs, templates, checklists, and structured artifacts
- repeated extraction, QA, or agent turns over the same image
- short Qwen3-VL first-view requests where visual prefill dominates and approximate token merging is acceptable
- videos with long static spans where selected chronological frames retain the task evidence

It is usually neutral or slower for open-ended creative writing, one-word answers, and prompts where the next tokens are not recoverable from local context. Eligibility is model- and prompt-dependent; benchmark and calibration APIs let applications keep the boosted path off when it does not pass latency and output checks.

## Command Line

The Python package installs an Ollama-style resident model workflow:

```sh
machboost list
machboost list --json
machboost pull qwen2.5:3b
machboost create company-coder:latest --from qwen2.5-coder:7b --option num_ctx=8192
machboost cp company-coder:latest company-coder:staging
machboost warm qwen2.5:3b
machboost run qwen2.5:3b
machboost run qwen2.5-vl:3b --image ./image.png
machboost run qwen3-vl:8b --video ./clip.mp4 --vision-tokens auto
machboost chat qwen2.5:3b
machboost connect 192.168.1.50:11435 --name studio
machboost connections --probe --model qwen2.5:3b
machboost use auto
machboost complete qwen2.5-coder:3b "def parse_config(text):"
machboost ps
machboost show qwen2.5:3b
machboost stop qwen2.5:3b
machboost rm company-coder:staging
machboost rm mlx-community/unused-model --weights
machboost shutdown
```

`machboost list` shows cached Hugging Face and MLX models, backend readiness, and available short aliases. `machboost run MODEL` uses the selected local, fixed, or automatic host mode, loads the model before accepting input, builds a draft corpus from any `--context` files or directories, and opens a streaming interactive chat. New shared connections enable the automatic pool; use `machboost use studio` to pin one host or `machboost use local` to opt out. Use `/?` for commands, `/status` for the active host and route, `/route` for a fresh latency/load ranking, `/stats on|off` for response metrics, `/clear` to reset history, `/bye` to exit while keeping the idle window, `Ctrl-C` to stop a reply, and `Ctrl-D` or `/unload` to unload and exit. Plain `machboost rm` removes a local alias only; add `--weights` to unload the selected model and permanently delete its managed cache.

Run the server in the foreground when integrating it with another application or process manager:

```sh
machboost serve --host 127.0.0.1 --port 11435 --replicas 1 --max-queue 64
```

Increase `--replicas` only after measuring representative concurrent traffic and checking unified-memory use. `machboost ps` shows active and queued work; `machboost ps --json` includes the full scheduler counters.

By default, models remain warm for five idle minutes. The lifetime can be selected per load or run:

```sh
machboost warm qwen2.5:3b --keep-alive 1h
machboost run qwen2.5:3b --keep-alive 10m
machboost run qwen2.5:3b --keep-alive forever
```

Plain open-ended chat without local context delegates to the backend's native greedy generator and should report `estimated_speedup=1.00x`. It still traverses the resident HTTP/streaming layer, so direct in-process `mlx-lm` can have lower first-token latency. Algorithmic text speedups require useful `--context` that predicts upcoming tokens.

### Ollama Compatibility

MachBoost is a replacement-compatible server for common local and agent flows,
not a binary clone of Ollama. Existing clients can point `OLLAMA_HOST` at the
MachBoost endpoint and use chat, generation, model discovery, lifecycle,
embeddings, images, tools, streaming, cancellation, and keep-alive behavior.

| Surface | MachBoost support |
|---|---|
| `/api/chat`, `/api/generate` | streaming/non-streaming, system/templates, thinking toggle, tools, images on compatible VLMs |
| `/api/tags`, `/api/ps`, `/api/show` | catalog, aliases, loaded instances, preflight |
| `/api/pull`, `/api/create`, `/api/copy`, `/api/delete` | HF/MLX downloads with aggregate shard progress, persistent local aliases, and opt-in managed weight deletion with `purge: true` |
| `/api/embed`, `/api/embeddings` | mean-pooled normalized embeddings from the resident model input layer |
| Ollama options | `num_ctx`, `num_predict`, `num_keep`, truncation policy, seed, temperature, top-k/top-p/min-p, repetition/presence/frequency penalties, stop strings, format/schema |
| Not equivalent | GGUF Modelfile builds, Ollama registry push, Ollama's internal model format, and every undocumented client assumption |

`/api/push` returns `501` rather than pretending to publish weights. Structured
output is validated after generation. Context limits are enforced with the
loaded tokenizer, preserving system content and the latest user turn while
dropping older turns when truncation is enabled.

Useful native options:

```sh
machboost list --backend mlx
machboost list --all
machboost run qwen2.5:3b --verbose
machboost run qwen2.5:3b --direct
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --device mps --max-tokens 128
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --dtype float16 --show-stats
machboost run Qwen/Qwen2.5-3B-Instruct --backend hf --local-files-only
machboost run mlx-community/Qwen3.5-0.8B-MLX-4bit --backend mlx --strict
machboost run mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx --context ./docs --ngram 1 --reentry-probe-tokens 1
machboost run qwen3-vl:8b --image ./document.png --vision-tokens adaptive --vision-token-ratio 0.35 --show-stats
machboost run qwen3-vl:8b --image ./chart.png --vision-tokens auto --vision-calibration ./vision-calibration.json
machboost run qwen3-vl:8b --video ./clip.mp4 --video-fps 2 --video-max-frames 12
machboost bench qwen2.5:3b --engine machboost --runs 5 --json
machboost bench-context qwen2.5:3b --prompt-file ./prompt.txt --context ./src --runs 6
```

`--reentry-probe-tokens` is experimental and disabled by default. `--direct` restores the one-process behavior for debugging. On Apple Silicon, a short alias prefers the MLX 4-bit model; explicit Hugging Face models default to `--device auto --dtype auto`, which selects MPS with float16 when available.

The package also includes install checks:

```sh
machboost doctor --json
machboost self-test --json
machboost version
```

An explicit Ollama wrapper remains available for compatibility with an existing Ollama installation:

```sh
machboost ollama run qwen2.5:3b
```

If the model is missing, MachBoost asks the local Ollama server to pull it first, then opens an interactive chat. Inside the chat, use `/bye`, `/exit`, `/quit`, EOF, or Ctrl-C to leave, and `/clear` to reset chat history.

Useful options:

```sh
machboost ollama run qwen2.5:3b --ctx 4096 --temperature 0
machboost ollama run llama3.2 --system "Answer concisely." --no-pull
```

This wrapper uses Ollama's public HTTP API and is not native MachBoost verifier
acceleration. `machboost run` and `machboost chat` normally use the
MachBoost-owned MLX/HF runtime. `muse-glimmer:30b` is native MLX-VLM;
`muse-glimmer:30b-mlx` is retained as an explicit backward-compatible Ollama
bridge for users who already installed that artifact.

The repository also includes the original Go CLI for diagnostics, command wrapping, and local benchmark experiments:

```sh
go run ./cmd/machboost doctor
go run ./cmd/machboost doctor --json
go run ./cmd/machboost run --profile sustained --workload generic -- echo ok
go run ./cmd/machboost bench command -- sleep 1
go run ./cmd/machboost overlap --prompt prompt.txt --output output.txt --context .
```

The Go CLI is useful for local systems experiments. The Python package is the product-facing inference layer.

## Backends

| Backend | Status | Notes |
|---|---|---|
| MLX / `mlx-lm` | native adapter | Primary Apple Silicon path. Cache-enabled drafting is experimental; strict mode disables prompt cache for slower exactness controls. |
| MLX-VLM | native visual adapter | Provides architecture-aware repeated-image reuse, opt-in approximate post-fusion compression, and chronological video-frame inputs for supported VLMs. |
| Hugging Face Transformers | native adapter | Useful for research and broad model coverage. |
| MachBoost resident server | native control plane | Keeps MLX/HF models warm and exposes OpenAI Chat/Responses, Anthropic Messages, and Ollama-compatible streaming APIs. |
| Custom Python service | native if verifier exists | Implement `next_token`, `verify`, `encode`, and `decode` as needed. |
| Ollama HTTP | compatibility wrapper | General models remain wrapper-only. `muse-glimmer:30b-mlx` preserves the older resident bridge; the default `muse-glimmer:30b` alias is native MLX-VLM. |

## Evidence

Public benchmark artifacts live in [results](results/), with methods and limitations in [results/README.md](results/README.md). Representative results are:

| Artifact | Model | Path | Pairs | Output/task result | Median paired speedup |
|---|---|---:|---:|---:|---|
| `mlx_native_default_qwen25_3b_20260713.json` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | default code continuation | 5 | 100% | 1.96x |
| `mlx_native_reentry_qwen25_3b_20260713.json` | same | experimental RAG re-entry | 5 | 100% | 1.58x |
| `context_bench_llama32_3b_20260720.json` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | same-model controlled code boundary | 6 | 100% exact output | 1.412x |
| `llama32_3b_mlx_context_benchmark_20260716.json` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | seven-fixture context suite | 21 | 95.24% exact output | 1.008x |
| `native_workspace_team_qwen25_3b_20260815.json` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | ten later questions over one stable repository | 10 | 10/10 exact token pairs | 2.894x |
| `team_repository_reuse_qwen25_7b_20260809.json` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | new cross-thread private-repo questions | 6 | 6/6 byte-identical output | 1.31x related; 1.43x unrelated |
| `muse_glimmer_30b_mlx_20260811.json` | legacy `muse-glimmer:30b-mlx` Ollama bridge | native DFlash vs logprobs no-spec diagnostic | 10 requests per mode | 2/10 same-prompt pairs byte-identical; no equivalence claim | 1.25x MachBoost decode; 1.30x direct Ollama decode |
| `vision_cache_qwen25_3b_20260714.json` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | repeated questions over one image | 12 | 100% | 18.33x |
| `vision_cache_qwen3vl_2b_20260714.json` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | repeated questions over one image | 12 | 100% | 11.41x |
| `vision_cache_qwen3vl_4b_20260714.json` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | same | 12 | 100% | 12.73x |
| `vision_cache_qwen3vl_8b_20260714.json` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | same | 12 | 100% | 16.69x |
| `vision_cache_qwen35_08b_20260714.json` | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | same | 12 | 75% | 5.14x |
| `vision_cache_qwen35_4b_20260714.json` | `mlx-community/Qwen3.5-4B-MLX-4bit` | same | 12 | 100% | 14.29x |
| `vision_cache_qwen35_9b_20260714.json` | `mlx-community/Qwen3.5-9B-MLX-4bit` | same | 12 | 100% | 17.44x |
| `cold_vision_qwen3vl_8b_postfusion_20260715.json` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | unique-image TextVQA, 35% visual retention | 10 | 70% normalized output equality; 80%/80% task match | 1.70x |

The July 20 `bench-context` artifact is the cleanest packaged algorithm check: one loaded Llama 3.2 model, balanced execution order, six exact pairs, a 1.412x median, 32 accepted tokens, and 50% logical target-call reduction. It repeats one controlled code boundary and does not replace the broader generalization check. In that seven-fixture Llama audit, code and policy subsets reached 1.33x and 1.23x medians with 3/3 exact outputs; the JSON subset reached 1.35x but only 2/3 exact outputs. The suite-wide median was 1.008x because four fixture families accepted no useful drafts. A cache-disabled control restored 9/9 equality on code, JSON, and policy, but its 0.207x median made it roughly 4.8x slower than native generation. These results are why cache-enabled MLX text drafting remains experimental.

Resident serving is measured separately. In `chat_latency_llama32_3b_20260717.json`, seven warm, alternating-order requests reached 0.679s median wall time and 144.00 decode tok/s through MachBoost versus 0.803s and 96.65 tok/s through Ollama. MachBoost was 1.18x faster end to end and 1.49x faster in reported decode throughput, while Ollama delivered first text sooner (0.198s versus 0.247s). The runtimes use different 4-bit model files and prompt tokenizations, and MachBoost accepted no context drafts, so this is a backend/serving comparison rather than an algorithmic or quality result.

The Qwen2.5 code path accepted a median 51 of 64 tokens and reduced logical target forwards by 76.6%. One-token re-entry broadened coverage to copied RAG answers, accepting a median 30 tokens. Those medians remain below 2x and workload-specific. Older `strict` and 9B artifacts compared against synchronous or cache-disabled baselines and remain diagnostics only; they do not establish an improvement over optimized `mlx-lm` or Ollama.

The visual artifact compares 12 uncached requests with 12 accelerated requests on the same resident Qwen2.5-VL 3B model. Median wall time fell from 2.818 seconds uncached to 0.152 seconds on the accelerated path; median paired speedup was 18.33x and median TTFT speedup was 19.45x. All paired outputs and expected fixture answers matched. Eleven of 12 accelerated rows reused a 1,018-token visual prefix. The remaining row deliberately repeated the priming prompt, so it only reused projected image features and reached 1.33x. These are warm repeated-image results on one machine and model, not a claim about first-view latency, decode throughput, changed images, or arbitrary visual workloads.

The cross-model artifact `vision_cache_qwen_matrix_20260714.json` applies the same image, prompts, resident-process policy, generation settings, and alternating pair order to six Qwen models. Across 72 pairs, both modes answer every fixture correctly. The median of the six model-level paired medians is 13.51x, ranging from 5.14x to 17.44x. Literal output equality is 95.83%: the only drift is three Qwen3.5 0.8B rows that differ by a semicolon inside a JSON fence while returning the same expected answer. The median reusable-prefix pair is 12.96x. Qwen3-VL's three genuine no-prefix cache controls have a 0.99x median, confirming that cache reuse alone does not improve first-view work. Every Qwen3.5 row uses a guarded whole-state checkpoint, including the repeated priming prompt. Qwen3.6 is excluded because its official releases are 28B and 36B total parameters. Variant names are not always total multimodal size: Qwen3-VL-8B is listed as 9B total and Qwen3.5-9B as 10B total.

The first-view artifact `cold_vision_qwen3vl_8b_postfusion_20260715.json` instead uses 10 unique public TextVQA images, disables both visual and prompt caches, alternates pair order, and excludes a held-out warm-up. Adaptive post-fusion compression retains a median 35.12% of visual states after layer 3. Median wall time falls from 4.078 to 2.368 seconds; aggregate speedup is 1.67x and median paired speedup is 1.70x. Both paths match an accepted dataset answer on 80% of rows, but normalized output equality is 70% and literal equality is 50%. Equal aggregate task score in this small pilot is not quality-equivalence evidence or a universal first-view result.

The research paper source and PDF are included in [paper](paper/). Keeping `paper/` and `results/` in the public repository is intentional: they make the claims auditable. They are not imported by the package at runtime.

## Reproduce Benchmarks

Run the paired native-MLX suite:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures code,rag,creative_open \
  --repeat 5 \
  --max-new-tokens 64 \
  --max-draft-tokens 32 \
  --ngram 3 \
  --source-mode context \
  --mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --output results/local/mlx_native_default.json
```

Test the opt-in one-token re-entry profile:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures code,rag,creative_open \
  --repeat 5 \
  --max-new-tokens 64 \
  --ngram 1 \
  --max-draft-tokens 32 \
  --reentry-probe-tokens 1 \
  --source-mode context \
  --mlx-model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --output results/local/mlx_native_reentry.json
```

The harness includes prompt processing in both paths, alternates baseline-first and boosted-first ordering, uses fresh nonces, and records environment provenance. Rerun the broader Llama context audit and the serving comparison with:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --mlx-model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --fixtures policy,json,rag,code,repo_quote,creative_open,short_answer \
  --repeat 3 \
  --max-new-tokens 64 \
  --source-mode context \
  --output results/local/llama32-context.json

machboost bench llama3.2:3b \
  --engine both \
  --ollama-model llama3.2:3b \
  --runs 7 \
  --warmups 2 \
  --max-tokens 64 \
  --json
```

The first command tests MachBoost's context path against native `mlx-lm`. The second compares warm serving runtimes and does not exercise context drafting.

For historical comparison with Hugging Face prompt lookup:

```sh
python3 scripts/hf_prompt_lookup_compare.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --fixtures real_readme_api,real_core_code,policy,json,rag,code \
  --max-new-tokens 32 \
  --prompt-lookup-sweep 4,8,16 \
  --machboost-source-modes prompt,context,prompt-context \
  --output results/local/hf_prompt_lookup_compare.json
```

Use `results/local/` for new local runs; it is ignored by git.

Run the repeated-image VLM benchmark:

```sh
python3 -m scripts.benchmark_vision_cache \
  --model qwen2.5-vl:3b \
  --repeats 3 \
  --max-tokens 16 \
  --output results/local/vision_cache_qwen25_3b.json
```

Run additional models with aliases such as `qwen3-vl:2b`, `qwen3-vl:4b`, `qwen3-vl:8b`, `qwen3.5:0.8b`, `qwen3.5:4b`, and `qwen3.5:9b`. Consolidate compatible artifacts with:

```sh
python3 scripts/summarize_vision_matrix.py results/local/vision_cache_*.json \
  --output results/local/vision_cache_matrix.json
```

Run the unique-image Qwen3-VL post-fusion benchmark:

```sh
python3 scripts/benchmark_cold_vision.py \
  --model qwen3-vl:8b \
  --datasets textvqa \
  --samples-per-dataset 10 \
  --max-tokens 16 \
  --cold-mode off \
  --vision-tokens adaptive \
  --vision-token-ratio 0.35 \
  --output results/local/cold_vision_qwen3vl_8b_postfusion.json
```

Run the shared-baseline policy ablation and derive an offline calibration artifact:

```sh
python3 scripts/benchmark_vision_tokens.py \
  --model qwen3-vl:8b \
  --datasets chartqa,docvqa,mmmu,textvqa \
  --samples-per-dataset 20 \
  --output results/local/vision-token-ablation.json

python3 scripts/calibrate_vision_tokens.py \
  results/local/vision-token-ablation.json \
  --min-pairs 10 \
  --output results/local/vision-calibration.json
```

The ablation runner rotates method order, reuses one native baseline per image, reports paired bootstrap confidence intervals, prints request progress, and writes incremental checkpoints. A native backend timeout or Metal failure invalidates the affected dataset run; it must not be reported as an acceleration result.

Compare uniform and temporal-change video frames:

```sh
python3 scripts/benchmark_video_frames.py ./clip.mp4 \
  --model qwen3-vl:8b \
  --question "What changes in the clip?" \
  --answer "the light changes from blue to red" \
  --repeats 5 \
  --output results/local/video-frame-benchmark.json
```

## Examples

Runnable examples live in [examples/python](examples/python/). Start with the workload evaluator before enabling acceleration in an application:

```sh
python3 examples/python/benchmark_context_workload.py \
  --context ./docs \
  --prompt "Continue the exact deployment checklist from the retrieved documentation:"
python3 examples/python/rag_knowledge_bot.py \
  --docs ./docs \
  "What does the deployment policy require?"
python3 examples/python/repository_completion.py \
  --repo . \
  --file ./machboost/context_bench.py
python3 examples/python/verifier_service_demo.py
python3 examples/python/black_box_service_demo.py
python3 examples/python/accelerator_calibration_demo.py
python3 examples/python/hf_adapter_demo.py
python3 examples/python/mlx_adapter_demo.py
python3 examples/python/vision_client_demo.py --image ./image.png
python3 examples/python/video_sampler_demo.py ./clip.mp4
python3 examples/python/ollama_adapter_demo.py
```

The context examples use MLX by default and accept `--backend hf` for Hugging Face. They print accepted draft tokens so a native fallback is visible. The workload evaluator compares both paths on one loaded model and refuses to report a valid aggregate speedup when outputs differ. The HF, MLX, and vision examples require the matching optional dependencies and locally available models.

## Development

Run tests:

```sh
python3 -m unittest discover -s tests
go test ./...
cd apps/macos
xcodegen generate
xcodebuild test \
  -project MachBoost.xcodeproj \
  -scheme MachBoost \
  -destination 'platform=macOS,arch=arm64'
```

CI uses Go 1.24 on current macOS runners. The older Go 1.17 toolchain can build the module, but its test binaries may be rejected by macOS 26 `dyld` because they lack the required Mach-O `LC_UUID` command.

Check packaging:

```sh
python3 -m pip install --dry-run .
python3 -m pip wheel . -w /tmp/machboost-wheel --no-deps
```

Render the paper:

```sh
tectonic paper/machboost.tex --outdir paper
```

## Safety And Scope

MachBoost v1 does not:

- change global shell config
- mutate `launchctl`
- change Ollama service state
- change Docker Desktop settings
- change system power settings
- upload telemetry
- modify model weights

Text drafting requires verifier access to the target model. Repeated-image reuse and approximate visual compression follow separate, explicitly documented contracts.

## License

MIT. See [LICENSE](LICENSE).
