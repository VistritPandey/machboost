# Python Examples

Install the package before running examples:

```sh
pip install -e .
```

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

The resident client demo starts the local MachBoost server when needed, preloads the selected model, streams the response, and leaves the model in memory for the next request. Pass one or more `--context PATH` arguments to enable local-context drafting. Use `machboost ps`, `machboost stop MODEL`, and `machboost shutdown` to manage the runtime.

Repeated-image visual chat:

```sh
pip install -e ".[vision]"
python3 examples/python/vision_client_demo.py --image ./invoice.png
python3 examples/python/vision_client_demo.py \
  --image ./dashboard.png \
  "Return only the current status." \
  "Return only the displayed total."
```

The visual client sends separate deterministic questions over one image and prints the resident backend's feature-cache hit, matching visual-prefix token count, and request latency. The second and later questions are the intended accelerated path.

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
