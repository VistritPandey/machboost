# MachBoost Evidence Runs

This directory stores public benchmark artifacts for MachBoost text and visual acceleration paths.

## Cross-Model Qwen Vision Matrix, July 14-15 2026

Aggregate artifact: `vision_cache_qwen_matrix_20260714.json`

Raw artifacts: `vision_cache_qwen3vl_{2b,4b,8b}_20260714.json` and `vision_cache_qwen35_{08b,4b,9b}_20260714.json`

All six runs use the same Apple M1 Max, generated 1024 by 768 image, four extraction prompts, three repeats, 16-token limit, greedy decoding, and one resident MLX-VLM instance at a time. Each pair compares visual caching disabled with MachBoost enabled. Pair order alternates by repeat; model download and load are excluded from request timings.

| Variant | Official total | Cache path | Baseline median | Accelerated median | Paired median | TTFT ratio | Exact output | Task accuracy |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL 2B | 2B | visual prompt state | 1.524s | 0.132s | 11.41x | 12.23x | 100% | 100% |
| Qwen3-VL 4B | 4B | visual prompt state | 2.743s | 0.214s | 12.73x | 13.32x | 100% | 100% |
| Qwen3-VL 8B | 9B | visual prompt state | 5.152s | 0.307s | 16.69x | 17.30x | 100% | 100% |
| Qwen3.5 0.8B | 0.9B | projected features + hybrid checkpoint | 0.656s | 0.125s | 5.14x | 5.48x | 75% | 100% |
| Qwen3.5 4B | 5B | projected features + hybrid checkpoint | 3.199s | 0.203s | 14.29x | 16.43x | 100% | 100% |
| Qwen3.5 9B | 10B | projected features + hybrid checkpoint | 5.572s | 0.311s | 17.44x | 18.82x | 100% | 100% |

Across 72 pairs, the median of the six model-level paired medians is 13.51x. Both modes answer every fixture correctly. Literal equality is 95.83%; the three mismatches are all Qwen3.5 0.8B returning the same `BLUE SQUARE` answer with or without a semicolon inside a JSON fence.

Qwen3-VL does not use projected-feature caching in these runs. Its vision tower returns deep-stack tensors that the current cached-feature interface cannot preserve, so MachBoost keeps the safe full prompt-state path. The three Qwen3-VL rows without a reusable prefix have a 0.99x median. The other 33 Qwen3-VL rows and all 36 Qwen3.5 rows reuse a 776-token visual prefix. Qwen3.5 uses a whole-state checkpoint because its language model interleaves ordinary KV layers with recurrent linear-attention state; trimming only K/V changed answers in a rejected smoke run.

Qwen3.6 is not measured under this matrix's small-model constraint. The official collection currently contains 27B (28B total) and 35B-A3B (36B total, 3B active) variants. Quantized file size and active MoE parameters do not make either model a sub-10B total-parameter model.

Recreate the aggregate after running the six per-model benchmarks:

```sh
python3 scripts/summarize_vision_matrix.py results/local/vision_cache_*.json \
  --output results/local/vision_cache_matrix.json
```

## Repeated-Image VLM Evidence, July 14 2026

Artifact: `vision_cache_qwen25_3b_20260714.json`

Model: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`

Hardware: Apple M1 Max, 32 GB unified memory

Workload: four deterministic extraction questions over one generated 1024 by 768 image, repeated three times. Each pair compares an uncached request with a request using content-addressed projected-image features and image-scoped prompt state. Pair order alternates. Generation is greedy, both modes use the same resident model instance, and model load is excluded from request latency.

| Metric | Uncached | Accelerated | Ratio |
|---|---:|---:|---:|
| Median wall time | 2.818s | 0.152s | 18.58x ratio of medians |
| Median paired wall time | n/a | n/a | 18.33x median pair ratio |
| Median time to first text | 2.807s | 0.144s | 19.45x |
| Median effective prompt throughput | 379.43 tok/s | 12,928.44 tok/s | 34.07x |

All 12 accelerated outputs exactly match their paired uncached outputs. Both modes answer all fixture questions correctly. The projected-feature cache hits in all 12 recorded accelerated rows. The partial visual-prefix cache hits in 11 rows, reusing a median 1,018 prefix tokens. Those 11 rows range from 13.32x to 21.36x paired wall-time speedup.

The first accelerated pair repeats the exact cache-priming prompt. MLX-VLM does not trim a complete prompt match through its partial-prefix path, so that row reuses projected image features only and reaches 1.33x. It is retained in the aggregate rather than discarded.

Reproduce the run with:

```sh
python3 -m scripts.benchmark_vision_cache \
  --model qwen2.5-vl:3b \
  --repeats 3 \
  --max-tokens 16 \
  --output results/local/vision_cache_qwen25_3b.json
```

This experiment measures warm repeated-image question answering on one machine, model, image, and short-answer workload. It does not measure first-view acceleration, changed images, long-form decode speed, other VLM architectures, video, or concurrent clients. The prompt-throughput value is effective throughput: MLX-VLM reports the full logical prompt length while the accelerated request computes only the unmatched suffix.

## Current Native-Baseline Evidence, July 13 2026

Artifacts:

- `mlx_native_default_qwen25_3b_20260713.json`
- `mlx_native_reentry_qwen25_3b_20260713.json`

The current harness compares MachBoost against `mlx-lm` native streaming generation, includes prompt processing in both wall-clock measurements, alternates baseline-first and boosted-first order, records fresh nonces, and preserves every raw row. Each artifact also records package versions, hardware, memory, timestamps, and thermal status.

Model: `mlx-community/Qwen2.5-3B-Instruct-4bit`

Hardware: Apple M1 Max, 32 GB unified memory

Generation: greedy, 64 requested tokens, five fresh-nonce repeats per fixture

### Default profile

Settings: 3-gram context lookup, 32-token drafts, no native-token re-entry.

| Fixture | Selected Path | Exact Match | Median Speedup | Baseline tok/s | MachBoost tok/s |
|---|---|---:|---:|---:|---:|
| `code` | adaptive context verifier | 100% | 1.96x | 87.46 | 167.28 |
| `rag` | native fallback | 100% | 1.04x | 89.98 | 91.61 |
| `creative_open` | native fallback | 100% | 1.00x | 99.95 | 98.27 |

The code fixture accepted a median 51 draft tokens and reduced logical target forwards by 76.6%. Its paired speedups were 2.44x, 1.15x, 2.08x, 1.96x, and 1.91x, showing substantial short-run variance.

### Experimental one-token re-entry

Settings: 1-gram context lookup, 32-token drafts, one native seed token before re-entry.

| Fixture | Selected Path | Exact Match | Median Speedup | Baseline tok/s | MachBoost tok/s |
|---|---|---:|---:|---:|---:|
| `code` | adaptive context verifier | 100% | 1.62x | 72.54 | 121.38 |
| `rag` | adaptive context verifier | 100% | 1.58x | 88.92 | 140.09 |
| `creative_open` | native fallback | 100% | 1.08x | 94.25 | 101.57 |

Re-entry broadens useful coverage: RAG accepts a median 30 draft tokens after one native seed and reduces logical target forwards by 43.8%. A longer 3-token re-entry probe produced one mismatch in a separate 15-row exploratory run, so re-entry is opt-in in 0.1.4.

The current repeated default result is close to, but below, 2x. This is not a universal acceleration result. The implementation fuses verifier continuation with the next draft block, rewinds the MLX KV cache in place, matches native MLX prompt prefill, and resumes native asynchronous decoding when context candidates end.

## Resident Server Evidence, July 13 2026

Artifact: `resident_qwen25_3b_20260713.json`

MachBoost 0.2.0 adds a resident HTTP server that keeps native MLX/Hugging Face models loaded between CLI, Ollama-compatible, and OpenAI-compatible requests. The model was preloaded with `machboost warm qwen2.5:3b --keep-alive forever` and resolved to `mlx-community/Qwen2.5-3B-Instruct-4bit`.

| Workload | Rows | Median TTFT | Median Total | Median End-to-End tok/s |
|---|---:|---:|---:|---:|
| forced 64-token completion | 5 | not measured | 0.657s | 97.47 |
| short streaming chat | 5 | 0.298s | 0.358s | not comparable across variable output lengths |

The first completion after loading took 0.973 seconds because the Metal execution path still required initialization; the next four took 0.650-0.663 seconds. Every row used native fallback with zero accepted draft tokens, so this experiment measures warm serving and corrected streaming detokenization rather than context-backed speculation.

The historical Ollama same-family probe recorded a 0.883-second median total duration for the forced 64-token shape, versus 0.657 seconds for the resident MachBoost run, a 1.35x wall-time ratio. The repositories use different 4-bit formats and the runs were not interleaved, so this is a runtime-suitability observation rather than exact-weight or paper-grade superiority evidence.

## Runtime Suitability Probe, July 13 2026

Artifact: `runtime_probe_qwen25_3b_20260713.json`

A separate forced 64-token decode probe compared Qwen2.5 3B runtimes on the same M1 Max. Native `mlx-lm` reached a median 127.41 tok/s, while Ollama reached 94.68 tok/s, a 1.35x MLX advantage. `vllm-mlx` 0.4.0 reached 15.80 tok/s for five serial prompts and 93.19 tok/s aggregate for five concurrent prompts in its built-in benchmark. The latter improves concurrent completion time but does not improve single-request decode latency.

This is not an exact-weight comparison: MLX and Ollama use different 4-bit formats. The artifact records raw rates, versions, commands' workload shape, and additional caveats. It supports selecting native MLX for Apple Silicon single-stream chat; it does not support a universal 2x claim.

## Legacy Diagnostic Artifacts

Artifacts below this point predate the native MLX baseline and the adaptive fallback path. In particular, the July 6 "strict" MLX runs compared against a synchronous/stateless path near 10 tok/s and must not be used to claim a 3x to 8x improvement over optimized `mlx-lm`, Ollama, or another production runtime. They remain tracked for implementation history and regression analysis.

## Qwen2.5-3B Use Cases, Repeat 3

Command:

```sh
python3 scripts/hf_bench_suite.py \
  --runner in-process \
  --repeat 3 \
  --fixtures use_cases,negative_controls \
  --draft-policy fixed \
  --local-files-only \
  --output results/qwen25_3b_use_cases_r3.json
```

Model: `Qwen/Qwen2.5-3B-Instruct`

Runner: in-process Hugging Face/MPS, one model load reused across fixtures.

Generation: greedy, 32 new tokens, exact-match checked against baseline output.

Artifacts:

- `qwen25_3b_use_cases_r3.json`: full machine-readable results.
- `qwen25_3b_use_cases_r3.txt`: terminal summary table.

| Fixture | Workflow | Expected | Exact Match | Decode Speedup | Forwards |
|---|---|---:|---:|---:|---:|
| `prompt_visible_readme` | docs continuation | positive | 100% | 1.78x | 32 -> 9 |
| `prompt_visible_code` | code completion | positive | 100% | 1.44x | 32 -> 18 |
| `rag_answer` | RAG answer | positive | 100% | 2.32x | 32 -> 9 |
| `log_template` | log generation | positive | 100% | 1.55x | 32 -> 15 |
| `json_config` | structured config | positive | 100% | 2.25x | 32 -> 9 |
| `test_boilerplate` | test generation | positive | 100% | 1.75x | 32 -> 15 |
| `repo_chat_quote` | repo chat | positive | 100% | 1.09x | 32 -> 27 |
| `controlled_context` | controlled | positive | 100% | 1.97x | 32 -> 13 |
| `hidden_readme` | hidden context | negative | 100% | 1.01x | 32 -> 32 |
| `creative_open` | creative generation | negative | 100% | 1.00x | 32 -> 32 |
| `short_answer` | short answer | negative | 100% | 1.04x | 32 -> 32 |

Takeaways:

- Strongest repeatable wins are RAG, structured config, controlled/reference-heavy continuation, docs, logs, and raw test/code continuation.
- Negative controls stayed neutral, which supports the policy goal: enable speculation only for context-grounded workflows.
- Repo-chat quote is weak in this fixture because the model does not reliably copy the intended command.

## Hugging Face Prompt Lookup Comparison, Qwen2.5-3B

Command:

```sh
python3 scripts/hf_prompt_lookup_compare.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --fixtures real_readme_api,real_core_code,policy,json,rag,code \
  --repeat 1 \
  --max-new-tokens 32 \
  --prompt-lookup-sweep 4,8,16 \
  --machboost-source-modes prompt,context,prompt-context \
  --output results/hf_prompt_lookup_compare_qwen25_3b.json
```

Model: `Qwen/Qwen2.5-3B-Instruct`

Runner: in-process Hugging Face/MPS, one model load reused across fixtures.

Generation: greedy, 32 requested new tokens. Output match compares token IDs over the requested 32-token budget. The artifact also records `raw_generated_tokens`; Hugging Face prompt lookup emitted extra tail tokens in some rows after the matching 32-token prefix.

Artifact:

- `hf_prompt_lookup_compare_qwen25_3b.json`: full machine-readable comparison against Hugging Face `prompt_lookup_num_tokens`.

Overall:

| Method | Rows | Exact Match | Median Speedup | P90 Speedup | Median tok/s | Median Forwards | Forward Reduction | Accepted Draft Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hf_serial_generate` | 6 | 100% | 1.00x | 1.00x | 21.74 | 32.0 | 0.0% | 0.0 |
| `hf_prompt_lookup_4` | 6 | 100% | 1.83x | 1.96x | 38.68 | 8.5 | 73.4% | 0.0 |
| `hf_prompt_lookup_8` | 6 | 100% | 1.86x | 2.42x | 41.13 | 5.5 | 82.8% | 0.0 |
| `hf_prompt_lookup_16` | 6 | 100% | 2.11x | 3.61x | 46.59 | 3.0 | 90.6% | 0.0 |
| `machboost_prompt` | 6 | 100% | 2.14x | 2.41x | 45.77 | 10.5 | 67.2% | 30.5 |
| `machboost_context` | 6 | 100% | 2.47x | 2.70x | 51.06 | 10.5 | 67.2% | 30.5 |
| `machboost_prompt-context` | 6 | 100% | 2.26x | 2.44x | 47.63 | 11.0 | 65.6% | 30.0 |

Selected per-fixture comparison:

| Fixture | HF Prompt Lookup 16 | MachBoost Context | Note |
|---|---:|---:|---|
| `real_readme_api` | 2.03x | 2.64x | local README continuation favors context corpus |
| `real_core_code` | 2.12x | 2.19x | roughly tied |
| `policy` | 2.10x | 2.43x | context corpus wins |
| `json` | 3.91x | 2.50x | HF prompt lookup wins |
| `rag` | 1.95x | 1.68x | HF prompt lookup wins |
| `code` | 3.32x | 2.77x | HF prompt lookup wins |

Takeaways:

- Hugging Face prompt lookup is a strong baseline, not a straw man. On this 3B run, `prompt_lookup_num_tokens=16` reaches a 2.11x median speedup with exact 32-token prefix agreement.
- MachBoost context mode is still competitive and wins the median in this fixture mix at 2.47x, especially on local README and policy continuation.
- The defensible MachBoost product difference is not "n-gram lookup exists." It is local-corpus source control, adapter packaging, calibration/gating, and machine-readable evidence around when to enable the layer.
- The comparison should be repeated before making paper-grade claims; this artifact is a first direct baseline check with one repeat.

## MLX Strict Evidence V2, 64 Tokens, Repeat 3

Command:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --fixtures real_readme_api,real_core_code,real_paper_method,policy,json,rag,code,repo_quote,creative_open,short_answer \
  --repeat 3 \
  --max-new-tokens 64 \
  --max-draft-tokens 8 \
  --ngram 2 \
  --candidate-limit 1 \
  --source-mode context \
  --mlx-disable-cache \
  --mlx-model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --output results/mlx_evidence_v2_strict_c1_20260706.json
```

Model: `mlx-community/Qwen3.5-0.8B-MLX-4bit`

Runner: MLX package adapter with prompt cache disabled. This strict mode compares boosted generation against the same stateless greedy baseline, avoiding cache-trajectory drift seen in longer cache-enabled diagnostic runs.

Generation: greedy, 64 new tokens, exact token-match checked against baseline output.

Artifact:

- `mlx_evidence_v2_strict_c1_20260706.json`: full machine-readable results.

Overall:

| Rows | Exact Match | Median Speedup | Mean Speedup | Baseline tok/s | Boosted tok/s | Median Accepted Draft Tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 100% | 3.03x | 3.55x | 10.53 | 26.17 | 49.0 |

Per-fixture medians:

| Fixture | Workflow | Expected | Exact Match | Speedup | Accepted Draft Tokens |
|---|---|---:|---:|---:|---:|
| `real_readme_api` | real README continuation | positive | 100% | 5.68x | 64 |
| `real_core_code` | real code continuation | positive | 100% | 5.93x | 64 |
| `real_paper_method` | real paper continuation | positive | 100% | 8.54x | 64 |
| `json` | structured config | positive | 100% | 3.79x | 57 |
| `code` | code completion | positive | 100% | 3.72x | 55 |
| `policy` | policy quote | positive | 100% | 2.31x | 41 |
| `rag` | RAG answer | positive | 100% | 1.91x | 27 |
| `repo_quote` | repo quote | positive | 100% | 1.78x | 24 |
| `creative_open` | creative generation | negative | 100% | 1.09x | 0 |
| `short_answer` | short answer | negative | 100% | 1.01x | 0 |

Takeaways:

- Real local artifacts are the strongest v2 use case: README, core code, and paper-source continuation all accept the full 64-token draft window and exceed 5.6x speedup in strict mode.
- Strict stateless MLX mode is not the fastest raw runtime path, but it provides clean exactness evidence for longer generations.
- Cache-enabled diagnostic runs were faster in raw tok/s but showed boundary mismatches on longer synthetic quote fixtures, so cache-mode evidence should remain diagnostic until cache trajectory handling is improved further.

## MLX Strict Evidence V2, Three Independent Runs

Additional strict runs:

- `mlx_evidence_v2_strict_run2_20260706.json`
- `mlx_evidence_v2_strict_run3_20260706.json`
- `mlx_evidence_v2_strict_aggregate_20260706.json`

The aggregate combines the original strict v2 run with two fresh runs using different seeds.

| Runs | Rows | Exact Match | Median Speedup | Mean Speedup | P10 Speedup | P90 Speedup | Baseline tok/s | Boosted tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 90 | 100% | 3.00x | 3.56x | 1.05x | 6.93x | 10.61 | 25.66 |

Per-run medians:

| Run | Exact Match | Median Speedup | Mean Speedup | Baseline tok/s | Boosted tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 3.03x | 3.55x | 10.53 | 26.17 |
| 2 | 100% | 3.15x | 3.43x | 11.03 | 25.57 |
| 3 | 100% | 3.00x | 3.70x | 10.15 | 25.53 |

Aggregate per-fixture medians:

| Fixture | Exact Match | Median Speedup | P10-P90 Speedup | Accepted Draft Tokens |
|---|---:|---:|---:|---:|
| `real_paper_method` | 100% | 8.54x | 6.50x-9.57x | 64 |
| `real_core_code` | 100% | 6.61x | 5.22x-6.99x | 64 |
| `real_readme_api` | 100% | 5.65x | 4.23x-5.97x | 64 |
| `json` | 100% | 3.89x | 3.44x-4.11x | 57 |
| `code` | 100% | 3.71x | 3.57x-3.82x | 55 |
| `policy` | 100% | 2.48x | 2.29x-2.96x | 41 |
| `rag` | 100% | 1.66x | 1.55x-2.49x | 27 |
| `repo_quote` | 100% | 1.60x | 1.31x-1.86x | 24 |
| `creative_open` | 100% | 1.07x | 1.04x-1.19x | 0 |
| `short_answer` | 100% | 1.01x | 0.98x-1.03x | 0 |

Takeaways:

- The strict v2 median speedup is stable across independent runs: 3.03x, 3.15x, and 3.00x.
- Exactness held across all 90 rows.
- Real artifact continuations remain the strongest evidence: the README, core-code, and paper-source fixtures accepted the full 64-token draft budget in aggregate.
- Negative controls stayed close to neutral and accepted zero draft tokens, which supports the benchmark gate design.

## MLX Qwen3.5-9B Strict Smoke

Command:

```sh
python3 scripts/backend_bench_matrix.py \
  --backends mlx \
  --mlx-model mlx-community/Qwen3.5-9B-MLX-4bit \
  --source-mode context \
  --mlx-disable-cache \
  --fixtures policy,json \
  --repeat 1 \
  --max-new-tokens 16 \
  --output results/mlx_qwen35_9b_strict_smoke.json
```

Model: `mlx-community/Qwen3.5-9B-MLX-4bit`

Runner: MLX package adapter with prompt cache disabled, matching the strict evidence mode.

Generation: greedy, 16 new tokens, exact token-match checked against baseline output.

Artifact:

- `mlx_qwen35_9b_strict_smoke.json`: two-row larger-model smoke artifact.

| Rows | Exact Match | Median Speedup | Baseline tok/s | Boosted tok/s | Accepted Draft Tokens | Forward Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100% | 7.50x | 1.77 | 13.26 | 16.0 | 87.5% |

Per-fixture:

| Fixture | Exact Match | Speedup | Baseline tok/s | Boosted tok/s | Accepted Draft Tokens |
|---|---:|---:|---:|---:|---:|
| `policy` | 100% | 6.99x | 1.86 | 13.00 | 16 |
| `json` | 100% | 8.01x | 1.69 | 13.51 | 16 |

Takeaway:

- This is only a smoke test, but it supports the expected scaling behavior: when accepted draft spans are long, larger/slower target models benefit more because each avoided serial target step is more expensive.
