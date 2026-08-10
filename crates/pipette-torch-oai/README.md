# pipette-torch-oai

Library backend for the unified `pipette` client: Docker-hosted or uv-managed
OpenAI-compatible inference servers (vLLM, SGLang) that serve
PyTorch/HuggingFace models.

Operator docs live outside this crate:

- [Usage guide](../../docs/pipette-cli/usage.md): init → register → install → run → sync
- [torch-oai runtime](../../docs/pipette-cli/torch-oai.md): engine details, flags, plan integration

## Quickstart

```bash
pipette init
pipette auth register \
    --organization YourOrg --contact-email you@example.com \
    --client-details "ci-box"
pipette runtimes pull \
    --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
pipette benchmarks init-local
export PIPETTE_HF_TOKEN=hf_…   # only for gated/private models
pipette benchmarks run \
    --benchmark local/end_to_end_latency_smoke \
    --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \
    --model 'torch://repo=LiquidAI/LFM2-350M'
```

Runtime URI schemes:

| Scheme | Install |
|--------|---------|
| `docker-vllm://` / `docker-sglang://` | `runtimes pull` (or auto on first `benchmarks run`) |
| `uv-vllm://` / `uv-sglang://` | auto-install on first `benchmarks run`; list versions with `pipette runtimes catalog uv_vllm` / `uv_sglang` |

## Flags that matter

- **`--model`**: `torch://repo=org/name` URI, or a JSON `Model` object (including a local directory source).
- **`--runtime-flags`**: JSON array of plan-style `RuntimeFlags` cells (typed knobs + `raw` passthrough), not a shell-split string. Example:

  ```bash
  --runtime-flags '[{"runtime_type":"docker_vllm","model_type":"torch","benchmark_type":"end_to_end_latency","dtype":"bfloat16","gpus":"all","raw":["--max-model-len","4096"]}]'
  ```

- **`PIPETTE_HF_TOKEN`**: gated download token (not `--hf-token` / `--hf-home`).

There is no separate `server start` / `chat` / `memprobe` surface: `benchmarks run` owns the full container/venv lifecycle.

## Benchmarks

Supported: `end_to_end_latency`, `max_memory_usage`, `eval`.

Linux host recommended. Docker Engine for docker schemes; `uv` for uv schemes; `nvidia-container-toolkit` for NVIDIA GPUs under Docker.
