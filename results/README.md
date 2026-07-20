# MachBoost Evidence Runs

This directory stores public benchmark artifacts for MachBoost text and visual acceleration paths.

## Evidence Contracts

The artifacts measure different mechanisms and should not be combined into one headline speedup:

- **Serving/runtime:** warm model residency, client latency, and backend throughput. These rows do not exercise context drafting unless context is explicitly supplied.
- **Context-backed text:** paired native and drafted generation using one model conversion. Exact output equality is required for an exact-path claim.
- **Repeated-image reuse:** later questions over identical image bytes after an explicit prime. These are not first-view results.
- **First-view visual compression:** new images with caches disabled. This path is approximate, so task score and output equality are reported separately.
- **Video frame selection:** preprocessing/frame-count evidence only until a completed VLM task benchmark is committed.

Results below are single-machine experiments unless stated otherwise. Ratios of medians and medians of paired ratios are different statistics and are labeled separately.

## Same-Model Context Benchmark, July 20 2026

Artifact: `context_bench_llama32_3b_20260720.json`

Model: `mlx-community/Llama-3.2-3B-Instruct-4bit`

The `machboost bench-context` harness loads one model instance and compares optimized native `mlx-lm` generation with MachBoost context verification. Six measured 64-token pairs alternate execution order after two warm-up pairs. Every native and accelerated token sequence matches exactly.

| Metric | Native | MachBoost |
|---|---:|---:|
| Median wall time | 0.640s | 0.451s |
| Median paired speedup | - | 1.412x |
| Median accepted draft tokens | 0 | 32 |
| Median logical target-call reduction | - | 50.0% |

This is a controlled code-continuation boundary repeated six times, not a six-prompt generalization suite. It proves that the packaged benchmark can isolate and measure the MachBoost algorithm without substituting Ollama or another model conversion. The broader seven-fixture audit below remains the better coverage result.

## Llama 3.2 Text Generalization Audit, July 16 2026

Artifacts:

- `llama32_3b_mlx_context_benchmark_20260716.json`
- `llama32_3b_mlx_context_strict_benchmark_20260716.json`

Model: `mlx-community/Llama-3.2-3B-Instruct-4bit`

Hardware: Apple M1 Max, 32 GB unified memory

Generation: greedy, 64 requested tokens, three repeats per fixture, alternating pair order, native `mlx-lm` baseline

The cache-enabled suite covers seven fixture families and 21 pairs:

| Fixture | Pairs | Exact output | Median speedup | Median accepted drafts |
|---|---:|---:|---:|---:|
| `code` | 3 | 100% | 1.325x | 47 |
| `policy` | 3 | 100% | 1.234x | 34 |
| `json` | 3 | 66.7% | 1.352x | 34 |
| `rag` | 3 | 100% | 0.998x | 0 |
| `repo_quote` | 3 | 100% | 0.980x | 0 |
| `creative_open` | 3 | 100% | 1.004x | 0 |
| `short_answer` | 3 | 100% | 1.007x | 0 |
| **Overall** | **21** | **95.24%** | **1.008x** | **0** |

One JSON pair produced a different 64-token sequence after an identical stored preview. The artifact does not retain enough tail text to characterize the exact differing token, so no narrower explanation is claimed. The mismatch is consistent with batched verification and serial native generation leaving MLX cache state on different numerical trajectories, but the artifact does not establish a root cause.

The cache-disabled strict control reruns code, policy, and JSON for nine pairs. It reaches 100% output equality but only a 0.207x median speedup, or roughly 4.8x slower than native generation. It is useful for diagnosing cache behavior, not for serving. Together, these artifacts make cache-enabled MLX text drafting experimental pending a cache-trajectory fix and broader exactness suite.

## Warm Llama 3.2 Serving Comparison, July 17 2026

Artifact: `chat_latency_llama32_3b_20260717.json`

The harness sends two warmups and seven measured requests, adds a unique system-message nonce to each round, and alternates which runtime executes first. Both use the Llama 3.2 3B family and 4-bit files, but MachBoost resolves to an MLX conversion while Ollama uses its own model artifact and prompt rendering.

| Runtime | Median wall | Median client TTFT | Median decode rate | Median output tokens |
|---|---:|---:|---:|---:|
| MachBoost resident MLX | 0.679s | 0.247s | 144.00 tok/s | 60 |
| Ollama 0.31.2 | 0.803s | 0.198s | 96.65 tok/s | 59 |

MachBoost is 1.18x faster by median wall time and 1.49x faster by reported decode throughput; Ollama reaches first text 1.25x sooner. MachBoost uses native fallback with no draft context in this benchmark. Cross-runtime outputs differ, as expected for non-identical files and prompt tokenization. This is a serving/backend comparison, not an algorithmic speedup or model-quality result.

## First-View Evaluation Status, July 16 2026

Version 0.5 adds a shared-baseline post-fusion ablation runner, paired bootstrap confidence intervals, deterministic random controls, workload-aware automatic policies, offline calibration gates, and a uniform-versus-temporal video harness. These tools do not replace the committed evidence below until complete artifacts are reviewed and added to this directory.

A planned ChartQA, DocVQA, MMMU, and TextVQA matrix did not complete on the evaluation M1 Max. A fresh native Qwen3-VL 8B request for the held-out 2257 by 1764 DocVQA page reached the end of a 3,945-token prefill but did not return generation metrics before the explicit 120-second timeout. This occurred with post-fusion compression disabled. After terminating that native request, later model initialization attempts failed in Metal with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. Those sessions are invalid performance runs and are not committed as evidence.

The runner now prints every dataset/sample/method boundary, writes an atomic checkpoint after each completed sample, and marks caught failures with a type and message. Future dataset runs should be isolated so one native backend failure does not erase valid measurements from other datasets.

The temporal sampler was integration-tested on a generated nine-second, three-scene color video. At two sampled frames per second and a 12-frame uniform budget, RGB change detection retained four chronological frames: the beginning, both color transitions, and the end. This is a 66.7% frame-count reduction test, not a VLM latency or quality result. No video speedup is claimed here.

## Unique-Image Post-Fusion Pilot, July 15 2026

Artifact: `cold_vision_qwen3vl_8b_postfusion_20260715.json`

Model: `mlx-community/Qwen3-VL-8B-Instruct-4bit`

Hardware: Apple M1 Max, 32 GB unified memory

Workload: ten unique TextVQA images with one short question each. Every pair compares the native full-token path with adaptive post-fusion visual-token compression at a requested 35% retention ratio. Both visual and prompt caches are disabled, pair order alternates, and one held-out unique image warms each mode before measurement. Model load is excluded.

| Metric | Native | Post-fusion | Ratio |
|---|---:|---:|---:|
| Median wall time | 4.078s | 2.368s | 1.72x ratio of medians |
| Aggregate wall time | 43.771s | 26.170s | 1.67x |
| Median paired wall time | n/a | n/a | 1.70x |
| Median time to first text | 4.029s | 2.351s | 1.71x |
| Accepted-answer match | 80% | 80% | unchanged in this sample |
| Normalized output equality | n/a | 70% paired | approximate |

The path keeps the original image and vision encoder unchanged. Qwen3-VL processes the complete visual sequence through its first three language layers and all required deep-stack visual injections. MachBoost then spatially groups the visual hidden states, preserves the most internally diverse groups, merges the remainder with query-weighted pooling, and runs the remaining 33 language layers on the shortened sequence. The measured median retained 35.12% of visual states, and all ten accelerated rows report zero visual-cache hits and zero reused prompt-prefix tokens.

This is not exact decoding: literal output equality is 50%, normalized equality is 70%, and equal 80% task scores on ten rows do not establish quality parity. It is also not a 2x result under the committed harness. A separate same-session 30% probe preserved the 8/10 task score but reached only 1.50x aggregate speedup and a 2.523-second median, slower than the 35% run. Metal compilation and tensor-shape effects make latency non-monotonic in the retained-token count.

Reproduce the committed run with:

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

## Qwen2.5 Native-Baseline Evidence, July 13 2026

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

Re-entry broadens useful coverage: RAG accepts a median 30 draft tokens after one native seed and reduces logical target forwards by 43.8%. A longer 3-token re-entry probe produced one mismatch in a separate 15-row exploratory run, so re-entry remains opt-in in 0.5.1.

This favorable repeated result is close to, but below, 2x. It is not a universal acceleration result and should be read alongside the newer Llama 3.2 audit above. The implementation fuses verifier continuation with the next draft block, rewinds the MLX KV cache in place, matches native MLX prompt prefill, and resumes native asynchronous decoding when context candidates end.

## Resident Server Evidence, July 13 2026

Artifact: `resident_qwen25_3b_20260713.json`

MachBoost 0.2.0 adds a resident HTTP server that keeps native MLX/Hugging Face models loaded between CLI, Ollama-compatible, and OpenAI-compatible requests. The model was preloaded with `machboost warm qwen2.5:3b --keep-alive forever` and resolved to `mlx-community/Qwen2.5-3B-Instruct-4bit`.

| Workload | Rows | Median TTFT | Median Total | Median End-to-End tok/s |
|---|---:|---:|---:|---:|
| forced 64-token completion | 5 | not measured | 0.657s | 97.47 |
| short streaming chat | 5 | 0.298s | 0.358s | not comparable across variable output lengths |

The first completion after loading took 0.973 seconds because the Metal execution path still required initialization; the next four took 0.650-0.663 seconds. Every row used native fallback with zero accepted draft tokens, so this experiment measures warm serving and corrected streaming detokenization rather than context-backed speculation.

The historical Ollama same-family probe recorded a 0.883-second median total duration for the forced 64-token shape, versus 0.657 seconds for the resident MachBoost run, a 1.35x wall-time ratio. The repositories use different 4-bit formats and the runs were not interleaved, so the newer July 17 artifact above supersedes this probe for runtime comparison.

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
