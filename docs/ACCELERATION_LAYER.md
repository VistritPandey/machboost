# MachBoost Acceleration Layer

MachBoost is a backend-aware local inference acceleration layer. Text generation can draft candidate tokens from nearby context and ask the target runtime to verify them before committing them. Visual question answering can reuse deterministic work derived from an unchanged image.

The central question is:

> Can local context safely draft tokens that the target model would have generated anyway?

If yes, MachBoost can reduce target-model work. If no, the package falls back to normal generation and records why.

For visual workloads, the corresponding question is whether the exact image bytes and visual prompt prefix have already been processed by the same resident model instance. Version 0.3 implements this path for MLX-VLM.

## Non-Goals

- No quality tradeoff.
- No quantization requirement.
- No global system mutation.
- No hosted service dependency.
- No claim that black-box inference can be accelerated without runtime support.

## Runtime Classes

### Native Acceleration

This is where real speedups happen. The runtime must expose enough internals to:

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

MachBoost 0.2 added a long-running control plane around the native adapters. Version 0.3 extends it to visual models. It:

- loads each model once and retains it in unified memory
- streams generated text without re-decoding the entire prefix per token
- applies finite or indefinite model keep-alive policies
- serializes generation per model while serving independent models concurrently
- exposes both Ollama-compatible and OpenAI-compatible HTTP endpoints
- supports explicit preload, inspection, stop, and shutdown operations

Resident serving removes repeated model-loading costs and makes MachBoost usable by editors, chat clients, scripts, and internal assistants. It is an operational latency improvement; it is separate from speculative token acceleration and should be measured separately.

### Repeated-Image Acceleration

The MLX-VLM adapter maintains two bounded, per-model caches:

1. A content-addressed LRU stores projected vision features. This skips the vision tower when the same image bytes are submitted again.
2. An image-scoped prompt state stores the language-model KV prefix associated with the visual token span. A later question over the same image can skip the matching visual-token prefill and process only the changed text suffix.

Local file identities are derived from image content, with file metadata used only to avoid unnecessary rehashing. Data URLs and in-memory images are also hashed by content. A changed image therefore receives a different feature entry and prompt state. Cache entries remain local to one resident model process and are discarded on model unload, explicit cache reset, or server shutdown.

This path does not improve first-view latency. It benefits repeated extraction, QA, and agent turns over unchanged visual inputs. It also does not increase decode tokens per second after prefill; its primary effect is lower time to first token.

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
Content key ---> projected-feature LRU ---> vision tower on miss
        |
        v
Image-scoped KV state ---> visual-prefix prefill on miss
        |
        v
Native MLX-VLM decoder ---> streamed output + cache metrics
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

Important limitation: sampling-compatible verification is not claimed in v1. Current public evidence is for greedy decoding and exact token equality.

### Policy Gate

The policy gate decides whether speculation should run.

Inputs:

- benchmark speedup
- exact-match status
- acceptance rate
- accepted draft span length
- target-call or forward-call reduction

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

### Resident Client

```python
from machboost import MachBoostClient

client = MachBoostClient()
client.load("qwen2.5:3b", keep_alive="forever")

for event in client.chat(
    "qwen2.5:3b",
    [{"role": "user", "content": "Explain the retry logic."}],
):
    print((event.get("message") or {}).get("content", ""), end="", flush=True)
```

For visual input, attach image paths to the request and use a VLM alias:

```python
client.load("qwen2.5-vl:3b", keep_alive="forever")

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
machboost warm qwen2.5:3b --keep-alive forever
machboost run qwen2.5:3b --context ./docs --show-stats
machboost run qwen2.5-vl:3b --image ./image.png --show-stats
machboost complete qwen2.5-coder:3b --file ./prompt.txt
machboost ps
machboost stop qwen2.5:3b
```

`machboost list` reports cached Hugging Face and MLX models plus portable short aliases. `machboost run MODEL` auto-starts the local server, resolves the best available backend, loads the model when needed, builds a draft corpus from local context files or directories, and opens an interactive streaming chat. Leaving chat does not unload the model. Use `--direct` for the earlier one-process behavior.

The package also exposes lightweight install checks:

```sh
machboost doctor
machboost self-test
machboost version
```

The native server listens on `http://127.0.0.1:11435` and implements Ollama-compatible chat, generate, model-listing, and lifecycle endpoints plus OpenAI-compatible chat and text completion endpoints. The external Ollama wrapper remains available separately:

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
| Hugging Face | yes | yes | native adapter, streaming server, and benchmarks exist |
| MLX / `mlx-lm` | yes | yes | native adapter, fast text streaming, and strict evidence mode exist |
| MLX-VLM | yes | repeated-image feature and prefix reuse | image chat, streaming, and paired benchmark exist |
| Custom Python service | caller-owned | yes, if verifier exists | supported through `machboost(...)` |
| External Ollama HTTP | already resident | no | compatibility wrapper and benchmarks only |
| llama.cpp | planned | possible | needs verifier/KV hooks or patch |

Protocol compatibility does not imply full feature parity. Version 0.3 does not yet provide Ollama model creation/copy/deletion or embeddings. Image input is supported for Ollama-style chat/generate requests and OpenAI-style content parts when the selected backend is MLX-VLM.

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

Status: done for 0.2.

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
- End-to-end resident latency evidence.

### P3: Runtime Expansion

Next targets:

- repeated larger-model evaluations
- cache-enabled MLX exactness work
- llama.cpp verifier hook investigation
- Ollama MLX runner patch or fork

### P4: Multimodal Runtime

Status: initial image path done for 0.3.

- image request schemas for the CLI, Python client, and local HTTP APIs
- multimodal model alias and capability discovery
- content-addressed projected-feature reuse
- image-scoped visual-prefix KV reuse
- paired time-to-first-token and end-to-end Qwen2.5-VL benchmark
- exact output and fixture-answer comparisons under greedy decoding

Remaining work:

- broader VLM architecture and model-size coverage
- concurrent-session cache isolation and load testing
- configurable image resizing and preprocessing policies
- video input, frame sampling, and temporal feature reuse
- real image datasets and task-level quality metrics

## Product Principle

MachBoost should feel boring:

- If it can help, it speeds up exact output.
- If it cannot help, it stays out of the way.
- It always leaves an audit trail explaining what happened.
