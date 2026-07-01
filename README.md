# machboost

`machboost` is a Mac-first CLI for running heavy local workloads under safe performance profiles. It is generic by design: LLM inference is the first smart preset, but the same runner works for builds, renders, exports, data jobs, and arbitrary commands.

It does not claim to allocate fake hardware percentages like `gpu: 90%`. Instead it applies real local controls: keep-awake wrapping, per-process environment hints, diagnostics, benchmarking, and reusable YAML profiles.

## Commands

```sh
machboost doctor
machboost doctor --json
machboost run --profile sustained --workload generic -- echo ok
machboost bench command -- sleep 1
machboost bench compare --profile sustained --workload build --repeat 3 -- make test
machboost bench ollama --model qwen3:8b --tokens 32
machboost overlap --prompt prompt.txt --output output.txt --context .
machboost draft --prefix prefix.txt --context .
machboost simulate-draft --prompt prompt.txt --output output.txt --context .
python3 scripts/hf_corpus_speculate.py --prompt prompt.txt --context . --model local-or-hf-model
python3 scripts/hf_corpus_speculate.py --prompt prompt.txt --context . --model local-or-hf-model --auto-draft --verify-mode hybrid --anchor-tokens 1
python3 scripts/hf_bench_suite.py --model local-or-hf-model --repeat 5 --local-files-only
python3 scripts/hf_bench_suite.py --runner in-process --fixtures use_cases,negative_controls --repeat 3 --local-files-only --output results.json
machboost profile init
```

## Profiles

- `sustained`: keeps the Mac awake and uses full-thread hints for supported workloads.
- `balanced`: conservative defaults for normal local work.
- `quiet`: avoids keep-awake behavior and uses reduced thread hints.

## Workloads

- `generic`: process wrapper only.
- `llm`: Ollama/llama.cpp-oriented hints.
- `build`: common build parallelism environment hints.
- `render`: common numerical/render thread environment hints.

`machboost` v1 is local-only. It does not change global shell config, `launchctl`, Ollama service state, Docker Desktop settings, system power settings, or upload telemetry.

## With vs without benchmark

Use `bench compare` to run the same command once without `machboost` and once with a selected profile, repeated as many times as you choose:

```sh
machboost bench compare --profile sustained --workload generic --repeat 3 -- ./your-heavy-job
```

For existing services like an already-running Ollama daemon, profile env vars will not affect that service unless `machboost` launches it. Use `bench compare` for commands that run inside the measured process, and use `bench ollama` to measure current Ollama API performance.

## Research tools

`machboost overlap` measures how much of a generated output can be recovered from the prompt and optional local context. High overlap means a future corpus-lookup speculative decoder may be able to reduce serial token generation steps.

`machboost draft` proposes candidate continuations from local context by matching the current prefix against repo or document text. This is the local corpus drafter that can later feed a decoder-level verifier.

`machboost simulate-draft` estimates how many serial decode steps a local corpus drafter could save on a known prompt/output transcript. The simulation is idealized; real acceleration requires runtime verification integration.

`scripts/hf_corpus_speculate.py` is an experimental Hugging Face verifier loop. It compares KV-cache baseline greedy generation against local-corpus speculative generation for causal language models where MachBoost can inspect logits directly.

`scripts/hf_bench_suite.py` runs repeatable benchmark fixtures and reports median total/decode tokens per second, exact-match rate, accepted draft tokens, forward reduction, and selected draft length.
Fixture aliases include `default`, `use_cases`, `negative_controls`, and `all`.

Useful verifier options:

- `--auto-draft --draft-sweep 2,4,6,8,10`: benchmark multiple draft lengths in one model load.
- `--source-mode prompt-context|context|prompt`: choose where local draft candidates come from.
- `--verify-mode block|hybrid|sequential`: trade off speed and strictness. `hybrid --anchor-tokens 1` verifies a short prefix step-by-step, then block-verifies the rest.
- `--draft-policy fixed|adaptive`: use fixed draft lengths or shrink/grow draft length during generation.
- `--min-verify-margin 1.0`: reject low-confidence draft tokens when testing safer block verification.

## Acceleration layer

See `docs/ACCELERATION_LAYER.md` for the adapter-layer plan: runtime capabilities, policy gate, sidecar shape, and backend roadmap.
