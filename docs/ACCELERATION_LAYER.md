# MachBoost Acceleration Layer

MachBoost is a backend-aware local inference package with four distinct optimization contracts. Text generation can draft candidate tokens from nearby context and ask the target runtime to verify them before committing them. Repository workspaces can retrieve bounded code context while reusing an unchanged prompt prefix. Repeated-image question answering can reuse process-local work derived from unchanged image bytes. Experimental first-view Qwen3-VL compression is approximate and is evaluated separately.

The central question is:

> Can local context safely draft tokens that the target model would have generated anyway?

If yes, MachBoost can reduce target-model work. If no, the package falls back to normal generation and records why.

For visual workloads, the corresponding question is whether the exact image bytes and visual prompt prefix have already been processed by the same resident model instance. Version 0.3 implements this path for MLX-VLM.

## Non-Goals

- No hidden quality tradeoff: exactness and task-score results are reported separately, and approximate paths are opt-in.
- No quantization requirement.
- No global system mutation.
- No hosted service dependency.
- No claim that black-box inference can be accelerated without runtime support.

## Runtime Classes

### Native Acceleration

This is where algorithmic text speedups are possible. The runtime must expose enough internals to:

- tokenize and detokenize text
- compute next-token logits or greedy decisions
- verify a candidate token span against the target model
- advance or rebuild KV/cache state after accepting a verified prefix

Current package adapters:

- Hugging Face Transformers
- MLX / `mlx-lm`
- MLX-VLM
- custom Python services that implement `next_token` and `verify`

Future native targets:

- llama.cpp / llama-server
- an Ollama runner patch or fork

### Resident Runtime

MachBoost 0.2 added a long-running control plane around the native adapters, and later releases extended it to visual models and bounded concurrent serving. It:

- loads each model once and retains it in unified memory
- compiles the text generation path with a one-token warmup before interactive use
- streams generated text without re-decoding the entire prefix per token
- applies finite or indefinite model keep-alive policies with background expiry
- serves independent models concurrently and can load multiple isolated replicas of a text model
- applies bounded FIFO admission with queue timeouts and HTTP `503` overload responses
- reports active, queued, rejected, timed-out, and per-replica request metrics
- exposes both Ollama-compatible and OpenAI-compatible HTTP endpoints
- supports explicit preload, inspection, stop, and shutdown operations

Resident serving removes repeated model-loading costs and makes MachBoost usable by editors, chat clients, scripts, and internal assistants. It is an operational latency improvement; it is separate from speculative token acceleration and should be measured separately.

Text replicas own separate accelerator and mutable cache instances. This prevents one request's KV or verifier state from contaminating another request, at the cost of another model allocation in unified memory. An optional affinity key prefers the same available replica for related requests but falls back to another idle replica instead of introducing head-of-line blocking. Model stop and expiry close admission first, drain active work, and only then release accelerator resources.

MLX-VLM remains at one replica. Its projected-feature and prompt-state caches are mutable, and its generation path has a process-wide safety lock. Concurrent visual requests therefore queue on one worker. MachBoost does not claim that replica concurrency is continuous batching: independent MLX replicas can contend for the same GPU, and increased replica count may improve queueing or first-token latency without materially improving aggregate decode throughput.

### Repository Workspace Reuse

A repository is not copied wholesale into every model request. MachBoost keeps a
local SQLite FTS5 index of bounded, line-addressable code chunks and a
deterministic repository map containing paths and extracted symbols. Git
repositories are enumerated with `git ls-files --cached --others
--exclude-standard`; symlinks, likely credentials, binaries, and oversized
files are excluded.

Each workspace request has two context regions:

1. A stable repository map that remains byte-identical while the indexed
   revision is unchanged.
2. Focused query-specific line windows, capped independently at 8,000
   characters and carrying file and line citations, that follow the map.

The resident MLX adapter keeps a bounded LRU of native prompt states. It finds
the longest exact token prefix for the current request, restores that state,
and evaluates only the unmatched suffix. New questions can therefore benefit
when they share the repository map and part of the request wrapper; they do not
need to repeat the previous question or retrieved chunks. The generated token
loop is unchanged.

Prefix reuse is opt-in for workspace requests and disabled for ordinary chat.
An index revision change produces a different repository map and naturally
reduces or invalidates the reusable prefix. Models whose cache representation
cannot be safely trimmed remain on native prefill. The current Qwen3.5 hybrid
cache is in that category.

This mechanism lowers prefill and time to first token for long, repeated
repository requests. It does not help the first request, a short unrelated
prompt, or generation dominated by a long answer. Index files remain on the
machine and repository contents are not uploaded.

### Repeated-Image Acceleration

The MLX-VLM adapter selects from two bounded, per-model cache layers according to model capability:

1. A content-addressed LRU stores projected vision features. This skips the vision tower when the same image bytes are submitted again.
2. An image-scoped prompt cache stores language-model state associated with the visual token span. A later question over the same image can skip the matching visual-token prefill and process only the changed text suffix.

Qwen2.5-VL can use both layers. Qwen3-VL's vision tower also returns deep-stack visual tensors, so MachBoost disables projected-feature reuse for that family and uses only complete prompt-state reuse. Qwen3.5 can safely reuse its projected tensor, but its language model mixes ordinary attention KV layers with recurrent linear-attention state. MachBoost therefore restores a whole-prefix checkpoint for Qwen3.5 instead of trimming only K/V. A KV-only Qwen3.5 smoke run was rejected after it changed answers.

Local file identities are derived from image content, with file metadata used only to avoid unnecessary rehashing. Data URLs and in-memory images are also hashed by content. A changed image therefore receives a different feature entry and prompt state. Cache entries remain local to one resident model process and are discarded on model unload, explicit cache reset, or server shutdown.

This path does not improve first-view latency. It benefits repeated extraction, QA, and agent turns over unchanged visual inputs. It also does not increase decode tokens per second after prefill; its primary effect is lower time to first token.

### First-View Visual Compression

Version 0.5 adds a Qwen3-VL-specific post-fusion wrapper. It leaves the vision encoder and required deep-stack injections intact, then shortens the visual hidden-state sequence before the remaining language layers. Four request modes are available:

- `merge`: one query-weighted representative per spatial group
- `adaptive`: merge spatial groups while preserving groups with high internal feature diversity
- `random`: deterministic token-count control for ablation only
- `auto`: classify the prompt and image signals, then select mode, retention ratio, layer, and token bucket

The automatic classifier distinguishes general, document/text, chart, spatial, and multi-image requests. A calibration artifact can replace built-in profiles per workload. The offline selector requires a minimum paired sample count, a lower confidence bound on speedup, bounded task-score loss, and normalized output agreement. Random controls are never deployable. A workload with no passing candidate resolves to native inference.

This path is approximate. It can change outputs and remains disabled unless requested. It cannot be combined with first-view image resizing, and the current layer hook is specific to the MLX-VLM Qwen3-VL implementation.

### Temporal Video Frames

The video adapter uses FFmpeg to sample frames, computes RGB frame-to-frame change on 64 by 64 thumbnails, and keeps the first frame, scene changes, and final frame under a fixed frame budget. When too many changes pass the threshold, the largest changes win while chronological order is preserved. Extracted frames are cached by video metadata and sample rate.

Selected frames enter the existing multi-image VLM path. The adapter is therefore model-agnostic at the file boundary, but it is not equivalent to a native video encoder. It can remove redundant static frames before vision encoding; it can also miss short motion events. Uniform-frame and temporal-frame benchmark paths are kept separate so frame-count reduction is not presented as a model-quality result.

### Wrapper / Policy Mode

Black-box local servers can still be wrapped for diagnostics, benchmarking, and option management. They cannot receive exact MachBoost speedups unless they expose a verifier API.

Examples:

- Ollama HTTP
- OpenAI-compatible local servers
- existing long-running model daemons

For these, MachBoost can still provide:

- workload classification
- context-overlap analysis
- benchmark comparison
- capability reports
- native-adapter recommendations

## Architecture

```text
Prompt + local context
        |
        v
Context Router ---> Policy Gate ------ no ----> serial generation
        |              |
        |             yes
        v              v
Candidate Drafter -> Runtime Verifier -> accepted prefix -> output
        |              |
        v              v
Results Recorder <----+
```

The visual path is independent of the text drafter:

```text
Image bytes + question
        |
        v
Content key ---> capability gate ---> projected-feature LRU (when safe)
        |                              |
        |                              +---> vision tower on miss
        v
Image-scoped prompt state ---> KV prefix or whole-state checkpoint
        |
        v
Native MLX-VLM decoder ---> streamed output + cache metrics
```

Repository workspaces use a separate retrieval and prefill path:

```text
Repository files ---> local FTS index ---> stable repository map
                              |                     |
New question ----------------+--> retrieved chunks |
                                                    v
                                      complete grounded prompt
                                                    |
                              longest exact MLX prefix cache hit
                                                    |
                                                    v
                                    unmatched prefill + native decode
```

First-view and video requests add two optional transformations before native decoding:

```text
Video file ---> FFmpeg samples ---> RGB temporal selector ---> chronological images
                                                               |
New image(s) + question ---> policy classifier ----------------+
                |                                              |
                +--> native visual sequence                    |
                         |                                     |
                         +--> post-fusion merge after layer N --+
                                                               |
                                                               v
                                                    remaining LM layers
```

## Core Interfaces

### Candidate Drafter

The current drafter indexes token n-grams from a caller-provided corpus. At generation time it matches suffixes of the current prefix and proposes the tokens that followed the best matching span.

Current behavior:

- n-gram local-context lookup
- longest suffix preference
- configurable maximum draft length
- optional multiple candidate attempts

Likely improvements:

- suffix arrays or suffix automata for faster lookup
- source locality scoring
- prompt-visible source priority
- trie/tree candidate packing
- retrieval-score weighting

### Runtime Verifier

The verifier checks candidate tokens against the target model.

Current behavior:

- greedy exact-match verification
- accepted-prefix commits
- residual-token fallback on mismatch where supported
- MLX strict/stateless mode for clean evidence runs

Important limitations:

- Sampling-compatible corpus verification is not claimed. Current text evidence uses greedy decoding.
- Verification of a proposed block does not by itself prove that every later token will remain identical when batched and serial cache trajectories use different floating-point reduction paths.
- A July 2026 Llama 3.2 3B audit matched 20 of 21 cache-enabled pairs. Cache-disabled strict mode matched 9 of 9 tested pairs but was about 4.8x slower than native generation. Cache-enabled MLX drafting therefore remains experimental and should be enabled only after model/workload calibration.

### Policy Gate

The policy gate decides whether speculation should run.

Inputs:

- benchmark speedup
- exact-match status
- acceptance rate
- accepted draft span length
- target-call or forward-call reduction
- visual-token retention, layer boundary, and task class
- paired task-score delta and output agreement
- lower confidence bound on paired wall-time speedup

Outputs:

- `enabled`: speculation is likely useful
- `disabled`: serial generation is safer or faster

The policy gate is product-critical because local-context drafting is not universal. It should accelerate grounded workflows and stay out of the way for open-ended prompts.

### Results Recorder

Evidence and benchmark rows should be machine-readable:

- model and backend
- fixture or workload type
- baseline and boosted tokens/sec
- exact-match rate
- accepted draft tokens
- target-model forward reduction
- policy decision
- warnings and mode flags

This keeps future dashboards, CI budgets, and technical reports possible without adding SaaS to v1.

## Public Interfaces

### Python API

```python
from machboost import Accelerator

accel = Accelerator.from_mlx(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    context_paths=["README.md", "docs/"],
)

text, stats = accel.generate(prompt, max_tokens=128)
```

### Repository Workspace API

Register and index a local repository:

```sh
curl http://127.0.0.1:11435/api/workspaces \
  -H "Content-Type: application/json" \
  -d '{"path":"/absolute/path/to/repository"}'
```

Use the returned identifier in an Ollama-compatible request:

```sh
curl http://127.0.0.1:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen2.5:7b",
    "workspace_id":"0123456789abcdef",
    "messages":[{"role":"user","content":"Where is cancellation handled?"}],
    "stream":false
  }'
```

The final response includes retrieved file and line citations under
`machboost.workspace`. Lifecycle and direct-search routes are:

- `GET /api/workspaces`
- `POST /api/workspaces`
- `POST /api/workspaces/index`
- `POST /api/workspaces/query`
- `POST /api/workspaces/delete`

OpenAI-compatible requests place the same fields in a top-level `machboost`
object. `workspace_top_k` and `workspace_max_chars` bound retrieved evidence.

### Resident Client

```python
from machboost import MachBoostClient

client = MachBoostClient()
client.load("qwen2.5:3b", keep_alive="5m", warmup=True)

for event in client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Explain the retry logic."}],
):
    print((event.get("message") or {}).get("content", ""), end="", flush=True)
```

For visual input, attach image paths to the request and use a VLM alias:

```python
client.load("qwen2.5-vl:3b", keep_alive="5m")

response = client.chat(
    "qwen2.5-vl:3b",
    [{"role": "user", "content": "Return only the invoice total."}],
    images=["./invoice.png"],
    options={"temperature": 0.0, "num_predict": 32},
    stream=False,
)
```

### Custom Service

```python
from machboost import machboost

boosted = machboost(
    service,
    corpus_tokens=local_context_tokens,
    ngram=4,
    max_draft_tokens=8,
)

tokens, stats = boosted.generate(prompt_tokens, max_tokens=128)
```

### CLI

The Python package exposes a resident native model runner:

```sh
machboost list
machboost pull qwen2.5:3b
machboost warm qwen2.5:3b --keep-alive 5m
machboost run qwen2.5:3b --context ./docs --verbose
machboost run qwen2.5-vl:3b --image ./image.png --show-stats
machboost complete qwen2.5-coder:3b --file ./prompt.txt
machboost bench qwen2.5:3b --ollama-model qwen2.5:3b --runs 3
machboost bench-context qwen2.5:3b --prompt-file ./prompt.txt --context ./src --runs 6
machboost ps
machboost stop qwen2.5:3b
```

`machboost list` reports cached Hugging Face and MLX models plus portable short aliases. `machboost run MODEL` auto-starts the local server, resolves the best available backend, loads and compile-warms text models, builds a draft corpus from local context files or directories, and then opens an interactive streaming chat. The header reports weight load, compile warmup, and total wall time separately. The default idle lifetime is five minutes, enforced by a background reaper. `Ctrl-C` cancels the current reply, `/bye` exits while retaining the idle window, and `Ctrl-D` or `/unload` unloads the model immediately. Use `--direct` for the earlier one-process behavior.

`machboost bench` measures client-observed time to first text, wall time, prompt evaluation, and decode throughput. It uses a unique prompt nonce per request and alternates which runtime executes first in each two-engine round. It can compare the native MachBoost runtime with Ollama. That comparison measures serving/runtime suitability, not context drafting, and is not file-identical because MLX and Ollama may use different templates, conversions, and quantization formats.

`machboost bench-context` measures the MachBoost algorithm itself. It loads one model instance, alternates optimized native generation with context-backed verification, and compares the resulting token IDs. Measured run counts must be even so execution order is balanced. Any output mismatch invalidates the aggregate speedup instead of reporting a fast but behaviorally different result.

The package also exposes lightweight install checks:

```sh
machboost doctor
machboost self-test
machboost version
```

The native server listens on `http://127.0.0.1:11435` and implements Ollama-compatible chat, generate, model-listing, and lifecycle endpoints plus OpenAI-compatible chat and text completion endpoints. A concurrent text configuration can be started with:

```sh
machboost serve --replicas 2 --max-queue 64 --queue-timeout 120
python3 scripts/benchmark_concurrency.py qwen2.5:3b --clients 4 --requests 8 --rounds 3
```

The server returns queue and replica metadata with each completed request and exposes aggregate scheduler state through `/api/ps`. A full queue produces HTTP `503` before NDJSON or SSE headers are emitted. The server has no authentication or TLS and must not be exposed directly to an untrusted network.

The external Ollama wrapper remains available separately:

```sh
machboost ollama run qwen2.5:3b
```

That explicit wrapper can pull and chat with models owned by an Ollama daemon, but it is not the native verifier-accelerated path. `machboost chat` is an alias for native `machboost run`.

The Go CLI remains available from source for local systems experiments:

```sh
go run ./cmd/machboost doctor
go run ./cmd/machboost bench command -- sleep 1
```

## Adapter Capability Matrix

| Backend | Resident serving | Native speculation | Status |
|---|---:|---:|---|
| Hugging Face | yes | yes | native adapter, streaming server, and research benchmarks exist |
| MLX / `mlx-lm` | yes | workspace prefix reuse is exact on tested Qwen2.5 models; token drafting remains experimental | native adapter, bounded native prompt cache, fast text streaming, cache-enabled drafting, and slower strict controls exist |
| MLX-VLM | yes | repeated-image reuse and approximate first-view compression | image/video-frame chat, streaming, policy calibration, and paired benchmarks exist |
| Custom Python service | caller-owned | yes, if verifier exists | supported through `machboost(...)` |
| External Ollama HTTP | already resident | no | compatibility wrapper and benchmarks only |
| llama.cpp | planned | possible | needs verifier/KV hooks or patch |

Protocol compatibility does not imply full feature parity. MachBoost does not provide Ollama model creation, copy, deletion, embeddings, tool calling, or thinking-field semantics. Image input is supported for Ollama-style chat/generate requests and OpenAI-style content parts when the selected backend is MLX-VLM. Video is accepted by the MachBoost CLI and expanded into image frames before the request.

## Text And Serving Evidence

The repository-prefix artifacts
`results/workspace_prefix_qwen25_3b_20260729.json` and
`results/workspace_prefix_qwen25_7b_20260729.json` compare native full prefill
with bounded prompt-prefix reuse on one M5 Pro with 48 GB unified memory. Both
paths use the same loaded model, tokenizer, complete prompts, and greedy token
loop. All 12 generated token sequences match exactly.

| Model | All-row median | Different-question median | Median native | Median MachBoost |
|---|---:|---:|---:|---:|
| Qwen2.5 3B | 3.021x | 2.971x | 3.144s | 1.024s |
| Qwen2.5 7B | 3.282x | 3.232x | 6.587s | 1.998s |

The first of six rows repeats the priming question and reaches 13.055x on 3B
and 16.637x on 7B. The other five questions are distinct and range from
2.873x to 3.078x on 3B and 3.090x to 3.365x on 7B. The median complete prompt
contains 10,405 tokens and the accelerated path reuses 7,901. These measurements
show reusable repository prefill, not universal first-request or decode
acceleration. A Qwen3.5 9B probe could not safely trim the hybrid cache and
produced no valid gain.

The latest cache-enabled text audit is `results/llama32_3b_mlx_context_benchmark_20260716.json`: 21 Llama 3.2 3B pairs across seven fixture families produced a 1.008x aggregate median and 95.24% exact output equality. Code and policy were exact in all three repeats and reached 1.33x and 1.23x medians. One of three JSON rows diverged. RAG, repo quote, creative, and short-answer fixtures accepted no useful drafts and stayed near native speed.

The cache-disabled control `results/llama32_3b_mlx_context_strict_benchmark_20260716.json` matched all nine tested code, JSON, and policy pairs, but its 0.207x median was about 4.8x slower than native generation. It is an exactness diagnostic, not a performance mode.

Serving is independent of the drafting algorithm. In `results/chat_latency_llama32_3b_20260717.json`, seven warm alternating-order requests reached 0.679 seconds median wall time and 144.00 decode tok/s through MachBoost, versus 0.803 seconds and 96.65 tok/s through Ollama. Ollama reached first text sooner: 0.198 seconds versus 0.247 seconds. The model family and 4-bit class match, but the model files, templates, and token counts do not; no algorithmic or quality claim follows from the ratio.

## Visual Evidence

The artifact `results/vision_cache_qwen25_3b_20260714.json` contains 12 uncached and 12 accelerated requests to one resident `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` instance on an Apple M1 Max. Four deterministic questions were repeated three times over a generated 1024 by 768 image. Request order alternated within each pair.

- Uncached median wall time: 2.818 seconds.
- Accelerated median wall time: 0.152 seconds.
- Median paired wall-time speedup: 18.33x.
- Median time-to-first-token speedup: 19.45x.
- Paired output equality: 100%.
- Expected-answer accuracy in both modes: 100%.
- Projected-feature cache hit rate: 100% after the unrecorded prime.
- Partial visual-prefix hit rate: 91.7%, with a median 1,018 matching tokens.

The one accelerated row without a partial prefix hit reused projected image features only and reached 1.33x. The other 11 rows reused both cache levels and ranged from 13.32x to 21.36x. The run excludes model load from request latency and does not establish performance on changed images, first-view requests, longer answers, other VLM architectures, or video.

The follow-up matrix `results/vision_cache_qwen_matrix_20260714.json` applies the same protocol to six Qwen models and 72 request pairs:

| Model | Official total | Median paired speedup | TTFT speedup | Exact output | Expected answer |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL 2B | 2B | 11.41x | 12.23x | 100% | 100% |
| Qwen3-VL 4B | 4B | 12.73x | 13.32x | 100% | 100% |
| Qwen3-VL 8B | 9B | 16.69x | 17.30x | 100% | 100% |
| Qwen3.5 0.8B | 0.9B | 5.14x | 5.48x | 75% | 100% |
| Qwen3.5 4B | 5B | 14.29x | 16.43x | 100% | 100% |
| Qwen3.5 9B | 10B | 17.44x | 18.82x | 100% | 100% |

The median of the six model medians is 13.51x. The Qwen3-VL no-prefix controls have a 0.99x median, while reusable-prefix pairs drive the reported gains. Qwen3.5 0.8B's three literal mismatches differ only by punctuation around the correct `BLUE SQUARE` answer. Qwen3.6 is excluded because the official releases are 28B and 36B total parameters.

## Milestones

### P0: Evidence Runner

Status: done.

- Hugging Face verifier prototype.
- MLX package adapter.
- Repeatable benchmark suites.
- Direct Hugging Face prompt-lookup comparison.
- Strict MLX evidence artifacts.
- Exact-match JSON reports.

### P1: Package Layer

Status: resident lifecycle done for 0.2; bounded replica serving added later.

- Public Python API.
- Optional backend extras.
- Install doctor and self-test CLI.
- Examples and package docs.
- Calibration and gate policy APIs.

### P2: Resident Serving

Status: done for 0.2.

- Long-running Hugging Face/MLX model ownership.
- Streaming CLI chat and raw completion.
- Model alias resolution and backend selection.
- Pull, preload, inspect, unload, and shutdown lifecycle.
- Ollama-compatible and OpenAI-compatible local APIs.
- Isolated text replicas, bounded admission, overload responses, and scheduler metrics.
- End-to-end resident latency evidence.

### P2.5: Repository Workspaces

Status: local index, native app selection, API retrieval, and MLX prefix reuse
implemented.

- Git-aware local file discovery and incremental SQLite FTS5 indexing.
- Stable repository maps plus bounded query-specific code chunks.
- File and line citations in Ollama-compatible and OpenAI-compatible responses.
- Native macOS repository picker with persisted conversation association.
- Bounded, opt-in MLX prompt-prefix reuse for workspace requests.
- Exact same-model 3B and 7B benchmark artifacts.

### P3: Runtime Expansion

Next targets:

- repeated larger-model evaluations
- cache-enabled MLX exactness and cache-trajectory work
- llama.cpp verifier hook investigation
- Ollama MLX runner patch or fork

### P4: Multimodal Runtime

Status: image path, first-view policy, and temporal frame adapter done for 0.5.

- image request schemas for the CLI, Python client, and local HTTP APIs
- multimodal model alias and capability discovery
- content-addressed projected-feature reuse
- image-scoped visual-prefix KV reuse
- paired time-to-first-token and end-to-end Qwen2.5-VL benchmark
- exact output and fixture-answer comparisons under greedy decoding
- Qwen3-VL post-fusion merge, adaptive, random-control, and automatic policies
- shared-baseline public-dataset ablations with paired bootstrap intervals
- offline workload calibration with quality and latency gates
- FFmpeg video sampling with RGB temporal-change selection
- uniform-versus-temporal video benchmark harness

Remaining work:

- complete multi-dataset first-view evidence after the observed native MLX high-resolution stability failure is resolved
- parallel VLM cache isolation and load testing; current visual serving queues on one worker
- video task-level quality evidence on real benchmarks
- query-aware temporal token selection and temporal feature reuse
- non-Qwen post-fusion adapters

### P5: Throughput Scheduling

Next target:

- integrate continuous batching for compatible MLX text models so concurrent requests share one weight copy and batched decode steps
- compare replica scheduling, continuous batching, and the backend's native server under identical prompts and token limits
- preserve context-verifier semantics for eligible requests while routing plain generation through the batch engine
- add cancellation, per-tenant limits, and request deadlines before recommending internet-facing multi-tenant deployment

## Product Principle

MachBoost should be predictable:

- An exact path is enabled only after model- and workload-specific output checks pass.
- If an approximate visual path is requested, it reports the applied policy and measured quality boundary.
- If it cannot help, it falls back to the backend's native generator and reports that decision; resident-server overhead may still remain.
- It always leaves an audit trail explaining what happened.
