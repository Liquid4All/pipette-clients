# torch-oai runtime (vLLM / SGLang)

Backend library: `pipette-torch-oai`, driven by the unified `pipette` CLI.

Runs OpenAI-compatible servers as either:

| Scheme | Artifact |
|--------|----------|
| `docker-vllm://` / `docker-sglang://` | Docker image in the daemon + workspace record |
| `uv-vllm://` / `uv-sglang://` | uv-managed Python venv under `.pipette/runtimes/` |

Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

**Host:** Linux. GPU depends on flavor (`nvidia_gpu`, `amd_gpu`, `cpu`). NVIDIA
Docker needs `nvidia-container-toolkit`. Tested primarily with
`vllm/vllm-openai`.

There is **no** standalone `server start` / `stop` / `chat` / `memprobe`
surface. `benchmarks run` owns the full container or uv-server lifecycle.

## Runtime install

```bash
# Docker — pulls the image into the daemon (optional warm-up). benchmarks run
# will docker pull + record the runtime itself if the image is missing.
pipette runtimes pull \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
pipette runtimes pull \
  --runtime 'docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=nvidia_gpu'

# uv — shared store ensure (pull and benchmarks run); list bundled versions
pipette runtimes catalog uv_vllm
pipette runtimes catalog uv_sglang
pipette runtimes pull --runtime 'uv-vllm://server=0.21.0&build=cu121&python=3.12'
```

`runtimes pull` for docker and UV records a list-visible store entry (and for
docker ensures the daemon has the image). `benchmarks run` uses the same
`RuntimeArtifactStore::ensure` path. A prior pull means run does not re-pull /
reinstall.

### Engine defaults (Docker)

| Scheme | Container port | Ready path | Model CLI flag |
|--------|----------------|------------|----------------|
| `docker-vllm://` | 8000 | `/v1/models` | `--model …` |
| `docker-sglang://` | 30000 | `/v1/models` | `--model-path …` |

Flavor selects GPU access wiring (`nvidia_gpu` → `--gpus`, `amd_gpu` → device
nodes, `cpu` → no GPU flags).

`runtimes remove --runtime '<uri>'` drops the **list-visible** workspace record
only; remove the image with `docker rmi` if needed. It does not tear down a
record that only `benchmarks run` wrote on the torch ensure path.

## Model

```bash
--model 'torch://repo=LiquidAI/LFM2-350M'
# or JSON Model with a local directory source (bare paths are not accepted)
```

Flow:

1. Resolve into the shared `models/` store (HF snapshot or local dir).
2. For Docker: bind-mount the resolved directory at the fixed container path
   `/models/model` and point the server model flag there. The container does not
   call HF Hub.
3. For uv: pass the host path into the server process.

Gated download: `PIPETTE_HF_TOKEN` (not `--hf-token` / `--hf-home`). Prefer
resolving the model **before** a multi-GB image pull: `benchmarks run` does
that order deliberately.

## Benchmark execution

### Docker lifecycle

1. `docker run -d --rm` with model path + `RuntimeFlags`  
2. Poll ready path until 200 or timeout  
3. Drive the benchmark over HTTP  
4. `docker stop` (Drop guard + SIGINT/SIGTERM handler)  
5. Next run reaps orphans labeled `pipette-torch-oai.workspace=<workspace-root>`

### Launcher defaults (Docker)

| Setting | Default | Notes |
|------|---------|--------|
| host bind | `127.0.0.1` + ephemeral port | measurement only |
| `gpus` | `all` | `RuntimeFlags` field, not a CLI flag |
| `shm_size` | `16g` | ditto |
| `ipc` | `host` | ditto |
| `envs` | — | `K` inherit or `K=V` literal on the cell |
| `max_model_len` | derived from the benchmark | vLLM only; sglang leaves `--context-length` to the operator |
| `prefix_caching` | `false` | fixed by every benchmark; a cell asking for `true` is refused |

### `--runtime-flags`

One JSON **object** of settings for this cell: typed fields plus `raw`
passthrough for the **server** argv, while launcher fields configure
`docker run`. The `(benchmark, runtime, model)` axes a plan entry carries are
**not** written here: the CLI derives them from `--benchmark`, `--runtime` and
`--model`, and rejects an object that restates them.

```bash
export PIPETTE_HF_TOKEN=hf_…   # if gated
pipette benchmarks run \
  --benchmark local/end_to_end_latency_smoke \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \
  --model 'torch://repo=Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --runtime-flags '{"dtype":"bfloat16","gpus":"all","max_model_len":4096}'
```

`max_model_len` and the prefix-cache flags are typed, so `raw` rejects them:
set the typed field instead.

Env forwards, on the CLI and in a plan:

```bash
# CLI: settings only
--runtime-flags '{"envs":["NCCL_DEBUG","VLLM_USE_V1=1"]}'
```

```toml
# plan: the same settings, with the axes that select the cell
runtime_flags = [
  { benchmark_type = "eval", runtime_type = "docker_vllm", model_type = "torch",
    envs = ["NCCL_DEBUG", "VLLM_USE_V1=1"] },
]
```

The submitted result's `runtime_flags` is the cell as it ran (the plan's entry
with every value the client derived folded in) as canonical JSON, the same
shape a plan authors and the same treatment the model/runtime descriptors get:

```json
{
  "runtime_type": "docker_vllm",
  "model_type": "torch",
  "benchmark_type": "eval",
  "tensor_parallel_size": 2,
  "max_model_len": 8448,
  "prefix_caching": false,
  "gpus": "all",
  "shm_size": "16g",
  "ipc": "host",
  "envs": ["NCCL_DEBUG"]
}
```

`gpus` is absent for the flavors that address GPUs by device mount, matching the
container that ran. Env forwards carry names only (a value may be a token, and
the record is uploaded), so put secrets in the bare-`K` inherit form, never in
`raw`, which is submitted verbatim. Flags the benchmark fixes for every run
aren't in the record (the type carries no reserved flag).

### Supported benchmark types

| Type | Status |
|------|--------|
| `end_to_end_latency` | supported (`/v1/completions`, exact-token prefill via `/tokenize`) |
| `max_memory_usage` | **docker only** (cgroup v2 `memory.peak` + `nvidia-smi` GPU peak when available); a `uv-*` runtime errors with `max_memory_usage benchmark is not yet supported on uv runtimes` |
| `eval` | supported (`/v1/chat/completions`, doomloop, checkpoint resume) |
| `prefill_throughput` / `decode_throughput` / `vl_throughput` | not implemented (clear error) |

Eval resume: [eval-checkpoint.md](eval-checkpoint.md).

## Quickstart

```bash
pipette init
pipette auth register \
  --organization YourOrg --contact-email you@example.com \
  --client-details "edge-ci-linux1"
pipette benchmarks init-local

pipette runtimes pull \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
# optional: pipette models pull --model 'torch://repo=LiquidAI/LFM2-350M'

pipette benchmarks run \
  --benchmark local/end_to_end_latency_smoke \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \
  --model 'torch://repo=LiquidAI/LFM2-350M'
```

## Plan integration

```toml
plan_id = "vllm-smoke"
benchmarks = ["eval_smoke"]

[[transports]]
client_id   = "edge-ci-linux1"
type        = "ssh"
host        = "edge-ci-linux1"
user        = "yuri"
binary_path = "/home/yuri/bin/pipette"
work_dir    = "/home/yuri/edge-evals"
shell       = "posix"

[[variants]]
clients = ["edge-ci-linux1"]
models = [
  { type = "torch", source = "huggingface", org = "Qwen", repo_name = "Qwen2.5-0.5B-Instruct" },
]
runtimes = [
  { type = "docker_vllm", image_name = "vllm/vllm-openai", image_tag = "v0.20.2", flavor = "nvidia_gpu" },
]
runtime_flags = [
  { benchmark_type = "eval", runtime_type = "docker_vllm", model_type = "torch", max_model_len = 4096 },
]
```

`auth_token = "hf_…"` on a model is forwarded to the target as
`PIPETTE_HF_TOKEN`. Full plan docs: [plan-runner.md](../pipette-plan/plan-runner.md).

## Result identity (sketch)

Wire payload carries model/runtime reporting fields such as image repo as
`runtime_name` and tag as `runtime_version`. `model_quant` is derived from the
checkpoint `config.json` when present (`dtype=…`, `quant=…`).

## Caveats

- **`max_memory_usage` needs Linux cgroup v2** inside the container path.
- **Broken NVIDIA host stack** → GPU channel unavailable; host memory may still report.
- **Docker Desktop (macOS/Windows)** is untested for measurement paths; treat as
  Linux-host-only for real runs.
- To inspect a live engine mid-debug, use `docker exec` / `curl` against the
  daemon. The CLI will not leave a long-lived server up after the run.
