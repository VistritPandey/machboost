# Python Examples

Install the package before running examples:

```sh
pip install -e .
```

## Start With Your Workload

MachBoost text acceleration is context-dependent. It can be useful when the expected answer or continuation overlaps retrieved documents, repository code, policies, templates, logs, or other local text. A unique user question can still qualify when its answer is grounded in that material. A genuinely novel answer usually does not qualify and should fall back to native generation.

Run the same-model evaluator before enabling the accelerated path:

```sh
pip install -e ".[mlx]"
python3 examples/python/benchmark_context_workload.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --context ./docs \
  --prompt "Continue the exact deployment checklist from the retrieved documentation:" \
  --runs 6 \
  --warmups 2
```

Add several representative `--prompt` or `--prompt-file` arguments. The script loads one model, alternates native-first and MachBoost-first execution, and compares token IDs. It reports a valid aggregate speedup only if every pair is exact. An engagement rate of zero means the context did not help; it is not evidence of acceleration.

Test a unique-message control with the same context:

```sh
python3 examples/python/benchmark_context_workload.py \
  --context ./docs \
  --prompt "Invent a completely new bedtime story about a lighthouse."
```

This control will normally remain near native performance because the novel continuation is not recoverable from the documentation.

## RAG And Internal Knowledge

The knowledge-bot example performs a small keyword retrieval step, includes the selected passages in the model prompt, and also exposes those passages to MachBoost's verified drafter:

```sh
python3 examples/python/rag_knowledge_bot.py \
  --docs ./docs \
  --show-context \
  "What does the release policy require before deployment?"
```

This shape fits internal policy assistants, support knowledge bases, runbook helpers, and extractive RAG. It is most eligible when the answer follows or quotes retrieved wording. The script prints accepted draft tokens and explicitly reports native fallback. Its lightweight retriever is educational, not a replacement for a production search or vector database.

## Repository Completion

Use source files other than the file being edited as draft context:

```sh
python3 examples/python/repository_completion.py \
  --repo . \
  --file ./machboost/context_bench.py \
  --max-tokens 64
```

The target file is excluded from the context corpus to avoid reading text after the cursor. This is useful for repositories with repeated APIs, schemas, tests, and implementation patterns. A one-off algorithm with no nearby analogue may accept no drafts and use native generation.

None of these examples establishes a universal `2x-8x` improvement. Results apply only to the measured model, context, prompts, settings, machine, and backend version.

Dependency-free demos:

```sh
python3 examples/python/verifier_service_demo.py
python3 examples/python/black_box_service_demo.py
python3 examples/python/accelerator_calibration_demo.py
```

Backend demos:

```sh
pip install -e ".[mlx]"
python3 examples/python/resident_client_demo.py --model qwen2.5:3b
python3 examples/python/resident_client_demo.py \
  --model qwen2.5-coder:3b \
  --prompt "def fibonacci(n):" \
  --max-tokens 128
```

The resident client demo starts the local MachBoost server when needed, loads and compile-warms the selected text model, streams the response, and leaves the model in memory for the five-minute default idle window. Pass one or more `--context PATH` arguments to enable local-context drafting. Use `machboost ps`, `machboost stop MODEL`, and `machboost shutdown` to manage the runtime.

Team gateway administration:

```sh
export MACHBOOST_API_TOKEN="your-admin-token"
machboost serve --team --host 0.0.0.0
python3 examples/python/team_gateway_admin.py
```

The example creates one scoped employee key, prints its one-time token, lists
recent traces, and runs a deterministic performance evaluation when traces are
available. Run it on a private network; MachBoost does not terminate TLS. See
[the team gateway guide](../../docs/TEAM_GATEWAY.md) for key scopes, retention,
coding-agent configuration, and local-model judging.

Team memory and optional provider fallback:

```sh
export MACHBOOST_API_TOKEN="your-admin-token"
python3 examples/python/team_memory_fallback.py /absolute/path/to/repository \
  --model qwen2.5-coder:7b
```

To add a budgeted external provider, also set `EXTERNAL_BASE_URL` and
`EXTERNAL_API_KEY`. The provider must accept the same public model alias passed
with `--model`; MachBoost does not silently rewrite model names. The example
publishes one reviewed team procedure, sends a workspace request with private
memory and deterministic exact reuse, and prints local cache metrics. It never
prints the provider key. Exact-reuse counters represent avoided model work only
for eligible repeated requests; they are not a decode-throughput speedup.

Warm chat latency comparison:

```sh
python3 examples/python/chat_latency_benchmark.py llama3.2:3b \
  --ollama-model llama3.2:3b \
  --runs 3
```

The benchmark records client time to first text, wall time, backend prompt evaluation, and decode throughput. Each request receives a unique nonce, and two-engine runs alternate which runtime executes first. Ollama and MLX may use different templates, converted files, token counts, and quantization formats; cross-runtime output equality is recorded for visibility but is not an accuracy comparison.

Repeated-image visual chat:

```sh
pip install -e ".[vision]"
python3 examples/python/vision_client_demo.py --image ./invoice.png
python3 examples/python/vision_client_demo.py \
  --image ./dashboard.png \
  "Return only the current status." \
  "Return only the displayed total."
```

The visual client sends separate deterministic questions over one image and prints the resident backend's feature-cache hit, matching visual-prefix token count, and request latency. The second and later questions are eligible for repeated-image reuse; actual hits remain model- and prompt-dependent.

Temporal video frame selection:

```sh
brew install ffmpeg
pip install -e ".[video]"
python3 examples/python/video_sampler_demo.py ./clip.mp4 --fps 2 --max-frames 12
```

The video sampler compares a uniform frame budget with RGB change-aware selection and prints the selected chronological frame paths, timestamps, change scores, cache state, and reduction rate. It does not load a model. Use `machboost run qwen3-vl:8b --video ./clip.mp4` to pass selected frames to a resident VLM.

```sh
pip install -e ".[hf]"
python3 examples/python/hf_adapter_demo.py
python3 examples/python/hf_adapter_demo.py --model Qwen/Qwen2.5-3B-Instruct --local-files-only
```

```sh
pip install -e ".[mlx]"
python3 examples/python/mlx_adapter_demo.py
python3 examples/python/mlx_adapter_demo.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit
```

```sh
python3 examples/python/ollama_adapter_demo.py
python3 examples/python/ollama_adapter_demo.py --run --model qwen2.5:3b
```

The Ollama HTTP demo connects to an external Ollama process and is only a wrapper/capability demo. It does not claim native MachBoost acceleration because Ollama's public HTTP API does not expose the verifier hooks needed for exact draft-token acceptance. The resident client demo uses the MachBoost-owned native runtime instead.
