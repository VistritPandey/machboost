# Ollama Adapter Notes

## Summary

Ollama can be a real native target for MachBoost, but not through the public HTTP generate API alone. The useful integration point is inside the model runner, where tokenization, logits, sampling state, KV cache state, accepted-prefix commits, and rollback are available.

The local Ollama checkout already contains several relevant hooks:

- GGUF models are routed to the `llama-server` subprocess.
- MLX-format models are routed to Ollama's Go/MLX runner.
- `draft_num_predict` already exists as a load-time runner option.
- llama-server launch arguments already support MTP/draft-model speculative decoding.
- The MLX runner already has a `drafter` abstraction, accepted-prefix validation, sampler commits, cache snapshots, rollback, and adaptive draft-depth control.

This means the Ollama path should be an adapter layer over existing runner hooks, plus a new MachBoost candidate source. It should not be a black-box HTTP wrapper promising speedups it cannot verify.

## Source Map

Paths below are from the Ollama repository root.

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
| MLX drafter contract | `x/mlxrunner/speculate.go` | Defines `drafter` with `propose`, `committed`, `finish`, and `flush`. |
| MLX verifier | `x/mlxrunner/speculate.go` | Fuses current token plus candidates, validates candidate tokens against target distributions, commits the accepted prefix, and rolls back rejected cache writes. |
| MLX MTP drafter | `x/mlxrunner/mtp.go` | Implements the current draft source using MTP/draft heads. |
| MLX depth policy | `x/mlxrunner/speculate_depth.go` | Learns validation cost and acceptance rates, then chooses draft depth adaptively. |
| MLX stats | `x/mlxrunner/speculate_stats.go` | Logs drafted, accepted, acceptance rate, max draft depth, and expected throughput diagnostics. |

## What This Means

### External Ollama HTTP Wrapper

Useful for:

- Benchmarking.
- Diagnostics.
- Setting options like `draft_num_predict`.
- Choosing prompts and context.
- Pulling missing models through `/api/pull`.
- Interactive chat through streaming `/api/chat`.
- Recording acceptance-like proxy metrics when available.

Not enough for:

- True corpus-verified speculative decoding.
- KV-cache rollback.
- Exact output preservation under candidate acceptance.

The public API does not expose the target logits and cache controls needed to accept or reject drafted tokens safely.

MachBoost exposes this wrapper through:

```sh
machboost ollama run qwen2.5:3b
machboost chat qwen2.5:3b
```

The flow mirrors the common `ollama run MODEL` experience: check installed models with `/api/tags`, pull a missing model with `/api/pull`, then stream chat responses with `/api/chat`. It intentionally remains wrapper mode.

### GGUF / llama-server Track

Best short-term path:

1. Detect whether a model has embedded MTP or a separate draft model.
2. Set or recommend `draft_num_predict`.
3. Benchmark with and without draft support.
4. Record llama-server timings and Ollama-level token throughput.

Custom MachBoost corpus drafting for GGUF needs a llama.cpp/llama-server patch or upstream extension, because the validation loop lives inside the subprocess rather than the Ollama Go HTTP layer.

### MLX Runner Track

Best native path:

1. Add a MachBoost drafter that implements Ollama's existing `drafter` interface.
2. Feed it local-context candidate tokens instead of MTP head predictions.
3. Reuse Ollama's existing target verifier, sampler commits, cache rollback, and adaptive depth controller.
4. Add an opt-in request option or environment flag for development.
5. Emit structured stats for benchmark comparison.

This is the fastest path to proving the generic MachBoost algorithm inside Ollama without re-solving cache safety.

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

For public users, do not mutate an installed Ollama app in place. Ship one of:

- A documented fork for research builds.
- A patch file users apply to a known Ollama commit.
- A separate runner binary once Ollama runner selection can support it cleanly.
- A MachBoost sidecar for wrappers and benchmarking, with native acceleration only when the backend supports verifier hooks.

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
