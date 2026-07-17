# Ollama Compatibility and Adapter Notes

## Summary

MachBoost 0.5.1 has its own resident Hugging Face/MLX runtime and exposes an Ollama-compatible HTTP surface. That compatibility lets existing clients use a MachBoost-owned decode path; it does not make an external Ollama process faster.

An installed Ollama runtime could become a native MachBoost target, but not through Ollama's public HTTP API alone. The required integration point is inside the model runner, where tokenization, logits, sampling state, KV cache state, accepted-prefix commits, and rollback are available.

The product-facing commands are native:

```sh
machboost pull qwen2.5:3b
machboost warm qwen2.5:3b
machboost run qwen2.5:3b
machboost complete qwen2.5-coder:3b "def fibonacci(n):"
```

They auto-start the MachBoost server on `127.0.0.1:11435`, resolve short aliases to MLX on Apple Silicon when possible, and keep loaded models resident until their keep-alive expires or the user stops them.

The default idle keep-alive is five minutes. `machboost run` preloads before showing its prompt, `Ctrl-C` cancels the active response, and `Ctrl-D` unloads the current model. Latency can be measured against Ollama with:

```sh
machboost bench llama3.2:3b --ollama-model llama3.2:3b --runs 3 --warmups 1
```

This reports client time to first text and backend throughput with unique request nonces and alternating execution order. It is not a file-identical comparison when the MLX and Ollama templates, conversions, or quantizations differ, and it does not exercise MachBoost context drafting.

The July 17 Llama 3.2 3B artifact records 0.679s median wall time and 144.00 decode tok/s for resident MachBoost/MLX, versus 0.803s and 96.65 tok/s for Ollama. Ollama reached first text sooner, at 0.198s versus 0.247s. Treat those numbers as one-machine runtime selection evidence, not proof that the MachBoost algorithm accelerated Ollama or that the two model files have equal quality.

Ollama's current `main` branch and the local source archive contain several relevant hooks:

- GGUF models are routed to the `llama-server` subprocess.
- MLX-format models are routed to Ollama's Go/MLX runner.
- `draft_num_predict` already exists as a load-time runner option.
- llama-server launch arguments already support MTP/draft-model speculative decoding.
- The MLX runner already has a `drafter` abstraction, accepted-prefix validation, sampler commits, cache snapshots, rollback, and adaptive draft-depth control.

This makes a native runner patch technically plausible: a MachBoost candidate source could reuse existing validation and rollback machinery. The current package does not include that patch, so the external Ollama integration remains a compatibility wrapper without MachBoost acceleration.

## Source Map

Paths below are from the Ollama repository root and were rechecked against official `main` on July 17, 2026. Ollama internals are not a stable public API; verify the linked source again before building a patch.

| Area | File | Relevant behavior |
|---|---|---|
| Scheduler routing | `server/sched.go` | Chooses GGUF `llama-server` vs MLX runner. Non-MLX models load with `llm.NewLlamaServer`; MLX completion models load with `mlxrunner.NewClient`. |
| Common runner interface | `llm/server.go` | Defines `LlamaServer`; documents that all GGML models are served via the upstream llama-server subprocess. |
| API option | `api/types.go` | Defines `draft_num_predict` on runner options. Default is currently 4. |
| Option defaulting | `server/routes.go` | Disables default draft depth unless the model has a draft path or `draft_num_predict` is explicitly set. |
| llama-server launch | `llm/llama_server.go` | Builds llama-server launch args. Adds MTP/draft flags with `--spec-type draft-mtp`, `--spec-draft-n-max`, `--spec-draft-backend-sampling`, and optional `--spec-draft-model`. |
| Draft detection | `llm/llama_server.go` | Auto-enables MTP when the GGUF has embedded MTP draft metadata. |
| MLX load path | `x/mlxrunner/runner.go` | Loads target model and optional draft model, then creates the persistent speculation subsystem. |
| MLX decode loop | `x/mlxrunner/pipeline.go` | Selects speculative vs plain decoder and owns streaming, stopping, and token accounting. |
| MLX drafter contract | [`x/mlxrunner/speculate.go`](https://github.com/ollama/ollama/blob/main/x/mlxrunner/speculate.go) | Defines the current `drafter` lifecycle (`propose`, `committed`, `settle`, and `close`). |
| MLX verifier | `x/mlxrunner/speculate.go` | Fuses current token plus candidates, validates candidate tokens against target distributions, commits the accepted prefix, and rolls back rejected cache writes. |
| MLX MTP drafter | `x/mlxrunner/mtp.go` | Implements the current draft source using MTP/draft heads. |
| MLX depth policy | `x/mlxrunner/speculate_depth.go` | Learns validation cost and acceptance rates, then chooses draft depth adaptively. |
| MLX stats | `x/mlxrunner/speculate_stats.go` | Logs drafted, accepted, acceptance rate, max draft depth, and expected throughput diagnostics. |

## What This Means

### MachBoost Ollama-Compatible Server

MachBoost owns the model and decode loop in this mode. Supported endpoints include:

- `/api/version`, `/api/tags`, `/api/ps`, and `/api/show`
- `/api/pull`, `/api/load`, `/api/stop`, and `/api/shutdown`
- streaming and non-streaming `/api/chat` and `/api/generate`

The same process also exposes OpenAI-compatible `/v1/models`, `/v1/chat/completions`, and `/v1/completions` endpoints. This is protocol compatibility, not complete Ollama feature parity. MachBoost 0.5.1 supports image content on chat/generate requests when the selected backend is MLX-VLM, but it does not implement Ollama model creation, copy, deletion, embeddings, tool calling, or thinking-field semantics.

### External Ollama HTTP Wrapper

Useful for:

- Benchmarking.
- Diagnostics.
- Sending ordinary generation options supported by the wrapper/API.
- Choosing prompts and chat history.
- Pulling missing models through `/api/pull`.
- Interactive chat through streaming `/api/chat`.

Not enough for:

- True corpus-verified speculative decoding.
- KV-cache rollback.
- Exact output preservation under candidate acceptance.

The public API does not expose the target logits and cache controls needed to accept or reject drafted tokens safely.

MachBoost exposes this wrapper through:

```sh
machboost ollama run qwen2.5:3b
```

The flow mirrors the common `ollama run MODEL` experience against an external Ollama daemon: check installed models with `/api/tags`, pull a missing model with `/api/pull`, then stream chat responses with `/api/chat`. It intentionally remains wrapper mode. `machboost run` and `machboost chat` use the native resident MachBoost runtime instead.

### GGUF / llama-server Track

Best short-term path for Ollama-supported speculation:

1. Detect whether a model has embedded MTP or a separate draft model.
2. Set or recommend `draft_num_predict`.
3. Benchmark with and without draft support.
4. Record llama-server timings and Ollama-level token throughput.

Custom MachBoost corpus drafting for GGUF needs a llama.cpp/llama-server patch or upstream extension, because the validation loop lives inside the subprocess rather than the Ollama Go HTTP layer.

### MLX Runner Track

Proposed native MachBoost path:

1. Add a MachBoost drafter that implements Ollama's existing `drafter` interface.
2. Feed it local-context candidate tokens instead of MTP head predictions.
3. Reuse Ollama's existing target verifier, sampler commits, cache rollback, and adaptive depth controller.
4. Add an opt-in request option or environment flag for development.
5. Emit structured stats for benchmark comparison.

This is the most direct integration route identified so far. It still requires a maintained Ollama fork or accepted upstream extension, plus output and throughput tests against Ollama's own plain decoder.

## Proposed Adapter Shape

```text
MachBoost context index
        |
        v
CandidateSource
        |
        v
Ollama MLX drafter implementation
        |
        v
Existing Ollama speculative verifier
        |
        +--> accepted prefix -> stream
        +--> rejection -> rollback -> target token
```

Suggested Go interfaces:

```go
type CandidateSource interface {
    Reset(prompt []int32)
    Propose(current []int32, maxTokens int) []int32
    Observe(committed []int32)
}

type CandidatePolicy interface {
    Enable(stats CandidateStats) bool
    DraftLimit(stats CandidateStats) int
}
```

The same conceptual interface can back the Hugging Face prototype, MLX, and future llama.cpp patches.

## Development Plan

1. Build a small standalone Go package for corpus candidate lookup.
2. Add tests that reproduce the current Python/HF fixture behavior at the token-id level.
3. Patch Ollama MLX runner with a second drafter implementation behind an explicit opt-in flag.
4. Compare three modes on the same prompts:
   - Plain decode.
   - Ollama built-in MTP/draft mode when available.
   - MachBoost corpus drafter.
5. Record exact output equality, decode tokens/sec, accepted draft tokens, rejection rate, and fallback frequency.
6. Only after MLX proof, design the llama.cpp/llama-server patch for GGUF.

## Product Guidance

For public users, the supported path is the standalone resident MachBoost server. It gives clients familiar streaming APIs without mutating an installed Ollama app. A future native Ollama integration should ship one of:

- A documented fork for research builds.
- A patch file users apply to a known Ollama commit.
- A separate runner binary once Ollama runner selection can support it cleanly.
- A compatibility sidecar for wrappers and benchmarking, with native acceleration only when the backend supports verifier hooks.

This keeps the project honest: fast where it owns or patches the decode path, useful but not magical where it only wraps a black-box server.

## Runner Patch Spike

The first pass should be carried as a documented patch or a proper fork rather than mutating an installed Ollama app in place.

The MLX runner has the right validation machinery already:

- `x/mlxrunner/speculate.go` defines the `drafter` interface and owns speculative validation.
- `x/mlxrunner/speculate.go` `accept` fuses the current token plus drafted tokens into one target forward.
- `scheduleSpeculation` / `commitSpeculation` snapshot and roll back target KV cache writes.
- `x/mlxrunner/pipeline.go` opens speculation before prefill, so a drafter can observe prompt chunks through `committed`.

The first MachBoost runner patch should be narrower than full sampling support:

1. Enable only for greedy-compatible requests first: `temperature == 0`, no logprobs, no top-logprobs.
2. Build the corpus source from prompt-visible tokens already in `request.Tokens`. This proves RAG/code/doc continuation use cases without adding a public API field yet.
3. Let `newSpeculation` return a speculation subsystem even when the model has no MTP draft head, as long as an explicit development flag enables the corpus drafter.
4. In `bind`, when `draft == nil`, set `targets = caches` and `draftKV = nil`.
5. In `open`, choose `newCorpusDrafter(request.Tokens)` instead of `newMTPDrafter` when the flag is enabled.
6. Add `draftCandidates` mode for deterministic token candidates with no draft distribution.
7. In `accept`, when candidates have no draft distribution, sample/argmax the target distribution rows directly and accept the longest prefix whose target token equals the corpus candidate token. The next token is the first mismatch token or the bonus row.

Why greedy first: Ollama's current MTP path uses rejection-sampling acceptance with `p/q`, where `q` is the draft model probability. A local corpus drafter has token guesses but no calibrated draft probability distribution. Reusing `p/q` with fake probabilities would not preserve the target sampling distribution. Deterministic greedy verification is the honest first patch; sampling-compatible acceptance needs either a calibrated draft distribution or a separate proof.

Minimal new runner pieces:

```go
type corpusDrafter struct {
    source *candidate.CorpusSource
    history []int32
}

func (d *corpusDrafter) propose(current *mlx.Array, maxTokens int) *draftCandidates {
    currentID := int32(current.Int())
    tokens := d.source.Propose([]int32{currentID}, maxTokens)
    if len(tokens) == 0 {
        return nil
    }
    return &draftCandidates{
        tokens: mlx.FromValues(tokens, 1, len(tokens)),
        dist: nil, // deterministic corpus candidate, greedy accept path only
    }
}
```

The existing Go `internal/candidate.CorpusSource` in this repo already has the token-level lookup behavior needed for that drafter. The Ollama fork can either copy that small package into `x/mlxrunner` for the spike or vendor a shared package later.
