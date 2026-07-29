# machboost

[![CI](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml/badge.svg)](https://github.com/VistritPandey/machboost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MachBoost is an alpha-stage, local-first inference server and Python package for MLX, MLX-VLM, and Hugging Face models. It offers an Ollama-like model workflow, keeps models resident between requests, and streams text and visual chat. Optional acceleration paths target reusable local text, repeated image inputs, and selected Qwen3-VL visual-prefill workloads.

The paths have different contracts. Plain chat delegates generation to the selected backend and mainly provides residency and API compatibility. Text drafting proposes tokens from caller-supplied context and verifies them with the target model, but the cache-enabled MLX path remains experimental because a recent Llama 3.2 audit found one token-sequence mismatch in 21 pairs. Repeated-image acceleration reuses process-local visual work for unchanged image bytes. First-view Qwen3-VL compression is explicitly approximate and can change answers.

MachBoost does not upload telemetry, mutate global runtime settings, or change model weights. It does not claim universal speedups, file-identical equivalence across model conversions, or quality preservation for approximate visual compression.

### Performance Contract

MachBoost is not a universal `2x-8x` switch. A speedup measured on a context-backed completion or repeated image must not be applied to unrelated prompts, new images, different models, or different machines.

Text drafting helps only when the model's next tokens are recoverable from caller-supplied local context and the target model accepts those draft tokens. Repository workspaces use a separate mechanism: a stable file/symbol map and query-specific code retrieval stay within a bounded prompt, while MLX can reuse the exact stable prefix on later workspace requests. The question itself can be new, but the first request still pays normal indexing and prefill costs. A novel message outside a workspace normally falls back to native generation, where expected algorithmic speedup is about `1.0x` and the server layer can add latency.

| Likely fit | Why it can help |
|---|---|
| RAG and internal knowledge assistants | Answers often quote or closely follow retrieved source passages. |
| Repository-aware code completion | Generated code can continue patterns already present in the repository. |
| Policy, checklist, and runbook assistants | Responses frequently reproduce stable approved wording. |
| Config, JSON, and template generation | Outputs often contain predictable local structures and repeated fields. |
| Repeated questions over the same image | Visual encoding and matching prompt-prefix work may be reusable. |

| Usually not a fit | Expected behavior |
|---|---|
| A first workspace question or unrelated unique question | Normal prefill; no reusable prefix exists yet. |
| Brainstorming, creative writing, or novel reasoning | Little recoverable continuation, so usually near native speed. |
| A changed or first-seen image | Repeated-image cache does not apply. |
| An external backend without verifier hooks | Wrapper and measurement only; no native MachBoost token verification. |

Treat every workload as uncalibrated until it passes a same-model paired benchmark. `machboost bench-context` alternates execution order, checks generated token IDs, and withholds the aggregate speedup if any output differs. See [examples/python](examples/python/) for RAG, internal knowledge, code-continuation, and workload-evaluation examples.

### Current Status

| Path | Current evidence | Product status |
|---|---|---|
| Plain resident text chat | Native MLX decode through a local server; no drafting without context | usable, with measurable server/streaming overhead versus direct `mlx-lm` |
| Concurrent text API serving | bounded FIFO admission, explicit overload responses, and isolated model replicas | usable; replicas consume additional memory and do not guarantee higher GPU throughput |
| Repository workspace prefix reuse | same-snapshot Qwen2.5 3B and 7B audits reached 2.659x and 2.867x medians with 6/6 exact token pairs each | promising for later questions over a stable indexed repo; not a first-request, arbitrary-model, or decode-throughput claim |
| Context-backed MLX text | latest broad Llama 3.2 3B suite was 1.008x aggregate with 20/21 exact pairs; favorable controlled continuations can be materially faster | experimental; never generalize a fixture result beyond its workload |
| Repeated unchanged image | 5.14x-17.44x model-level paired medians on one synthetic image and short extraction prompts | promising for repeated-image prefill; not a first-view or decode result |
| New-image Qwen3-VL compression | 1.70x median on ten TextVQA rows, with 70% normalized output equality and equal 8/10 aggregate task scores | approximate, opt-in, and not quality-equivalence evidence |

## Install

From a local checkout:

```sh
git clone https://github.com/VistritPandey/machboost.git
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
pip install -e ".[video]"
pip install -e ".[all]"
```

Install directly from GitHub:

```sh
pip install "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git"
pip install "machboost[vision] @ git+https://github.com/VistritPandey/machboost.git"
```

Update an existing install:

```sh
pip install --upgrade "machboost[mlx] @ git+https://github.com/VistritPandey/machboost.git"
```

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
repository workspaces, text/code/folder/image attachments, model downloads, resident-model controls,
server metrics, a developer API view, and a menu-bar controller. Chats and
imported attachments remain local, model downloads always require confirmation,
and closing the window leaves the selected models available until they expire,
are unloaded, or MachBoost is quit.

Release builds bundle pinned arm64 CPython 3.13, MLX, `mlx-lm`, `mlx-vlm`, and
MachBoost dependencies. They do not depend on Homebrew, system Python, or an
existing package install. The source is under [apps/macos](apps/macos/); see the
[native app guide](apps/macos/README.md) for local builds, runtime verification,
signing, notarization, DMG creation, and Sparkle updates. A public signed DMG is
not claimed until those release steps have completed with the project owner's
Apple credentials.

## Quick Start

Start a native local chat from the command line. Short aliases select MLX on a compatible Apple Silicon installation and Hugging Face elsewhere:

```sh
machboost list
machboost pull qwen2.5:3b
machboost run qwen2.5:3b
```

`machboost run` starts a local server automatically when needed, loads the model, and performs a one-token compile warmup before showing the chat prompt. The header separates model load, compile warmup, and total wall time, so startup work is not hidden behind the first message. Models stay resident for five idle minutes by default; a background reaper releases expired models even when no later command is issued. Preload explicitly with:

```sh
machboost warm qwen2.5:3b
machboost ps
```

Inside chat, `Ctrl-C` stops only the current reply and `Ctrl-D` unloads the current model and exits. `/bye` exits while preserving the five-minute idle window. Use `--keep-alive forever` only when indefinite residency is intentional.

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

Use full repository IDs when a model has no short alias. If the model is not cached, the selected backend may download it through its normal Hugging Face or MLX loader. Use `--local-files-only` with Hugging Face to require an existing cache.

Stream a raw completion for an editor or code tool:

```sh
machboost complete qwen2.5-coder:3b "def fibonacci(n):" --max-tokens 128
machboost complete qwen2.5:3b --file ./prompt.txt --context ./docs --show-stats
```

The resident server also exposes Ollama-compatible and OpenAI-compatible HTTP endpoints on `http://127.0.0.1:11435`:

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
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

Workspace requests opt into a bounded MLX prompt-prefix cache and use workspace
affinity. Plain MLX chat keeps native cache behavior. On one Apple Silicon run
over the same 181-file snapshot, six alternating-order Qwen2.5 3B pairs reduced
median wall time from `2.258s` to `0.853s` (`2.659x`), and Qwen2.5 7B reduced it
from `4.785s` to `1.660s` (`2.867x`). All 12 pairs matched generated token IDs.
Five rows per model used different questions and retrieved evidence; the
remaining row repeated the priming question. These are short, warm,
prefill-heavy requests with 7.5K-9.0K-token prompts, not claims about first
requests, decode rate, every repository, or every architecture. Qwen3.5 9B
could not safely trim its hybrid recurrent cache and produced no valid speedup,
so that path remains native.

Reproduce the same-model comparison:

```sh
python3 scripts/benchmark_workspace_prefix.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --workspace . \
  --runs 6 \
  --max-tokens 16 \
  --max-context-chars 32000
```

See the [3B artifact](results/workspace_prefix_qwen25_3b_20260729.json) and
[7B artifact](results/workspace_prefix_qwen25_7b_20260729.json).

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

Each text-model replica owns an independent accelerator and mutable generation cache. Requests are admitted through a bounded FIFO queue, then routed to an available replica. When every replica is busy and the queue is full, MachBoost returns HTTP `503` with `{"code":"queue_full"}` before starting a streaming response. Per-request queue wait and replica selection are returned under `machboost.scheduler`; aggregate active, queued, rejected, timed-out, and per-worker counts are available from `/api/ps` and `machboost ps --json`.

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

Discovery and control clients can use `GET /api/catalog`, `GET /api/metrics`,
`GET /api/workspaces`, and `POST /api/cancel`. Chat, generation, and pull requests accept an optional
`request_id`, which is echoed in streaming events. `/api/pull` supports NDJSON
progress and cancellation while retaining its non-streaming response contract.

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
machboost warm qwen2.5:3b
machboost run qwen2.5:3b
machboost run qwen2.5-vl:3b --image ./image.png
machboost run qwen3-vl:8b --video ./clip.mp4 --vision-tokens auto
machboost chat qwen2.5:3b
machboost complete qwen2.5-coder:3b "def parse_config(text):"
machboost ps
machboost show qwen2.5:3b
machboost stop qwen2.5:3b
machboost shutdown
```

`machboost list` shows cached Hugging Face and MLX models, backend readiness, and available short aliases. `machboost run MODEL` connects to the resident server, loads the model before accepting input, builds a draft corpus from any `--context` files or directories, and opens a streaming interactive chat. Use `/?` for commands, `/clear` to reset history, `/bye` to exit while keeping the idle window, `Ctrl-C` to stop a reply, and `Ctrl-D` or `/unload` to unload and exit.

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

This wrapper uses Ollama's public HTTP API and is not native MachBoost verifier acceleration. `machboost run` and `machboost chat` use the MachBoost resident runtime.

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
| MachBoost resident server | native control plane | Keeps MLX/HF models warm and exposes Ollama/OpenAI-compatible streaming APIs. |
| Custom Python service | native if verifier exists | Implement `next_token`, `verify`, `encode`, and `decode` as needed. |
| Ollama HTTP | wrapper only | Useful for benchmarking/capability detection; public HTTP does not expose logits/token IDs/KV hooks needed for exact acceleration. |

## Evidence

Public benchmark artifacts live in [results](results/), with methods and limitations in [results/README.md](results/README.md). Representative results are:

| Artifact | Model | Path | Pairs | Output/task result | Median paired speedup |
|---|---|---:|---:|---:|---|
| `mlx_native_default_qwen25_3b_20260713.json` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | default code continuation | 5 | 100% | 1.96x |
| `mlx_native_reentry_qwen25_3b_20260713.json` | same | experimental RAG re-entry | 5 | 100% | 1.58x |
| `context_bench_llama32_3b_20260720.json` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | same-model controlled code boundary | 6 | 100% exact output | 1.412x |
| `llama32_3b_mlx_context_benchmark_20260716.json` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | seven-fixture context suite | 21 | 95.24% exact output | 1.008x |
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
