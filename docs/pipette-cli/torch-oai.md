# torch-oai runtime (vLLM / SGLang)

Backend library: `pipette-torch-oai`, driven by the unified `pipette` CLI.

Runs OpenAI-compatible servers as either:

| Scheme | Artifact |
|--------|----------|
| `docker-vllm://` / `docker-sglang://` | Docker image in the daemon + workspace record |
| `uv-vllm://` / `uv-sglang://` | uv-managed Python venv under `.pipette/runtimes/` |

Shared operator flow and quickstart: [usage.md](usage.md). URI and per-cell
flag notation: [models-and-runtimes.md](models-and-runtimes.md).

**Host:** Linux, the only host it has run on. The CLI does not refuse other
hosts. GPU depends on flavor (`nvidia_gpu`, `amd_gpu`, `cpu`). NVIDIA
Docker needs `nvidia-container-toolkit`. Tested primarily with
`vllm/vllm-openai`. The `docker` and `uv` a run spawns are located when the
command starts; see
[usage.md](usage.md#where-the-external-programs-come-from) to point pipette at a copy
that is not on `PATH`.

`benchmarks run` starts and stops the container or uv server for each
cell.

## Runtime install

```bash
# Docker: pulls the image into the daemon (optional warm-up). benchmarks run
# pulls and records the runtime itself if the image is missing.
pipette runtimes pull \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
pipette runtimes pull \
  --runtime 'docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=nvidia_gpu'

# uv: list the bundled versions, then install one
pipette runtimes catalog uv_vllm
pipette runtimes catalog uv_sglang
pipette runtimes pull --runtime 'uv-vllm://server=0.32.0&build=cu129&python=3.12'
```

An explicit `runtimes pull` is a warm-up only; see
[usage.md §3](usage.md#3-install-a-runtime).

### Engine defaults (Docker)

| Scheme | Container port | Ready path | Container command |
|--------|----------------|------------|-------------------|
| `docker-vllm://` | 8000 | `/v1/models` | the image's `vllm serve`, then `--model …` |
| `docker-sglang://` | 30000 | `/v1/models` | `python3 -m sglang.launch_server --host 0.0.0.0 --model-path …` |

`lmsysorg/sglang` images enter at NVIDIA's wrapper, which runs its arguments
as a command (CUDA tags), or at a shell (ROCm, CPU). SGLang binds 127.0.0.1
unless told otherwise. So the launch names the server command and binds every
interface inside the container. The host still publishes the port on its
loopback only.

Flavor selects GPU access wiring (`nvidia_gpu` → `--gpus`, `amd_gpu` → device
nodes, `cpu` → no GPU flags).

`runtimes remove --runtime '<uri>'` drops the workspace record and the
runtime's compile cache only; remove the image with `docker rmi` if needed.

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

Gated models: see [usage.md](usage.md#gated-models).

## Benchmark execution

### Docker lifecycle

1. `docker pull` if the daemon lacks the image, so a pull is not timed as
   startup and does not count against `attempt_timeout_seconds`
   ([usage.md](usage.md#5c-the-attempt-limit))  
2. `docker run --rm` with model path + `RuntimeFlags`, attached, so the
   container's output is kept; on a Linux host every cell but the eval family
   (`eval`, `vlm_eval`, `asr_eval`) starts its memory probe here, so the model
   load counts  
3. Poll the ready path until it answers 2xx, the container exits, or 1800 s pass  
4. Drive the benchmark over HTTP  
5. `docker stop -t 30`; a ^C sends `docker kill` instead  
6. Next run reaps orphans labeled `pipette.workspace=<workspace-root>` whose
   `pipette.owner` run has exited or is this one, so a run in parallel keeps
   its container

A uv server is started, waited on and stopped the same way, leading a session
of its own: SIGTERM to its process group, then a kill of its session if it is
still up 30 s later. A session a killed run left behind is killed before the
next start, even once its leader is gone (see
[a run that is killed](usage.md#8-planner-worker-pull-model)). The session
kill also reaches what the server put in a group of its own, such as the build
steps `ninja` runs for a JIT compile. Either way the server's stdout and stderr
land on the result, a failed cell quotes their last lines, and `-v` logs each
line as it arrives.

### Launcher defaults (Docker)

| Setting | Default | Notes |
|------|---------|--------|
| host bind | `127.0.0.1` + a free port in 20000..=32767 | the server port is published on loopback only, for the client's own requests |
| `gpus` | `all` | `RuntimeFlags` field, not a CLI flag; to pin *this client* to one card set `CUDA_VISIBLE_DEVICES` in its `--config` entry instead ([usage.md §8a](usage.md#8a-client-configuration)), which stays out of the cell |
| `shm_size` | `16g` | ditto |
| `ipc` | `host` | ditto |
| `envs` | none | `K` inherit or `K=V` literal on the cell |
| `max_model_len` | derived from the benchmark (see below) | vLLM only; SGLang leaves `--context-length` to the operator |
| `prefix_caching` | `false` | fixed by every benchmark; a cell asking for `true` is refused |

The default `max_model_len` depends on the benchmark:

- Timing and max-memory cells: the prompt plus the generation.
- `eval`: `8192 + parameter_max_tokens`. An eval cell that pins
  `max_model_len` at or under `parameter_max_tokens` sends no `max_tokens`;
  see [Context bound](../methodology/evals.md#context-bound).
- `vlm_eval` and `asr_eval`: unset, so the model's own context applies. A
  cell that pins `max_model_len` at or under its output budget is refused.

### `--runtime-flags`

One JSON object of settings for this cell: typed fields plus `raw`
passthrough for the **server** argv, while the launcher fields above configure
`docker run`. The notation, and why the axes are not written here, is in
[models-and-runtimes.md](models-and-runtimes.md#--runtime-flags).

```bash
pipette benchmarks run \
  --benchmark local/end_to_end_latency_smoke \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \
  --model 'torch://repo=Qwen/Qwen2.5-7B-Instruct-AWQ' \
  --runtime-flags '{"dtype":"bfloat16","envs":["NCCL_DEBUG","VLLM_USE_V1=1"],"max_model_len":4096}'
```

`max_model_len` and the prefix-cache flags are typed, so `raw` rejects them:
set the typed field instead.

The submitted result's `runtime_flags` is the cell as it ran: the plan's entry
with every value the client derived folded in (`max_model_len` on vLLM,
`prefix_caching`, `gpus`, `shm_size` and `ipc` on Docker,
`speculative_enabled` on `vl_spec_decode`). `gpus` is absent for the flavors
that address GPUs by device mount, matching the container that ran. Env
forwards are recorded by name only, `GGML_*` aside (a value may be a token, and the record is
uploaded), so put secrets in the bare-`K` inherit form, never in `raw`, which
is submitted verbatim.

### Supported benchmark types

| Type | Status |
|------|--------|
| `prefill_throughput` | supported (streamed `/v1/completions`, `max_tokens = 1`; the sample is the client-clock time to the first token), batched. See [prefill-throughput.md](../methodology/prefill-throughput.md) |
| `decode_throughput` | supported (streamed `/v1/completions`; the sample is first-token-to-last-token wall time scaled to the full decode count), batched. See [decode-throughput.md](../methodology/decode-throughput.md) |
| `end_to_end_latency` | supported (streamed `/v1/completions` timed to the finish chunk, exact-token prefill via `/tokenize`), batched. See [end-to-end-latency.md](../methodology/end-to-end-latency.md) |
| `max_memory_usage` | **Linux host only**, docker or uv: the server's cgroup memory on the host, plus the GPU figure (NVML on NVIDIA, used VRAM on AMD). Method, refusals and caveats: [peak-memory-linux.md](../methodology/peak-memory-linux.md#torch-oai-docker-and-uv) |
| `eval` | supported (`/v1/chat/completions`, checkpoint resume). A vLLM row records the reasoning a reasoning parser streams apart, in front of the answer; see [torch-oai](../methodology/evals.md#torch-oai). A vLLM cell may set `repetition_detection` to send vLLM's repetition stop; see [Repetition stop](../methodology/evals.md#repetition-stop-repetition_detection) |
| `vl_spec_decode` | **sglang only** (`/v1/chat/completions` streamed, SGLang's `sglext.spec_tokens_details` for the draft counters); vLLM errors because its OpenAI endpoint reports no per-request draft counts. See [vl-spec-decode.md](../methodology/vl-spec-decode.md) |
| `vlm_eval` | **uv vLLM only** (the `uv_vllm` × `torch` cell, a repo's own weights; `/v1/chat/completions` streamed, checkpoint resume, every sample answered or the cell refused). See [vLLM](../methodology/vlm-eval.md#vllm) |
| `asr_eval` | **uv vLLM only** (the `uv_vllm` × `torch` cell, a repo's own weights; `/v1/chat/completions` streamed, Voxtral's transcription clips on `/v1/audio/transcriptions`, the corpus read in chunks, checkpoint resume, every clip answered or the cell refused). See [vLLM](../methodology/asr-eval.md#vllm) |
| `vl_throughput` / `vl_max_memory` / `asr_throughput` / `asr_max_memory` | not implemented (clear error) |

Eval resume: [eval-checkpoint.md](eval-checkpoint.md).

On docker on a Linux host, every cell but the eval family (`eval`,
`vlm_eval`, `asr_eval`) also records the container's memory as an
observation; a uv timing cell records none. See
[peak-memory-usage.md](../methodology/peak-memory-usage.md#observations-vs-metrics)
and [peak-memory-linux.md](../methodology/peak-memory-linux.md#torch-oai-docker-and-uv).

### Batched timing cells

The three timing cells run `parameter_batch_size` requests in flight at once;
the shared rules are in
[Batched](../methodology/end-to-end-latency.md#batched). The server is sized
for the batch at launch, one flag per engine, derived from the benchmark
rather than authored on the cell:

| Engine | Slot cap | Reserved in `raw` |
|--------|----------|-------------------|
| vLLM | `--max-num-seqs <N>` | yes, on the three timing cells |
| SGLang | `--max-running-requests <N>` | yes, on the three timing cells |

The eval cells leave them to the operator: eval's `parallel` knob is a
client-side fan-out that reaches no argv. At `N = 1` neither flag is emitted,
so an unbatched run records the same argv as one that never set the field.
The per-request HTTP timeout default scales with `N`; a pinned
`http_timeout_seconds` passes through unscaled.

### `vl_spec_decode` on SGLang

The corpus benchmark on a `torch` model (speculation off) or a
`torch_speculative` pair (the target plus its DSpark drafter). With speculation
on the server is launched with `--speculative-algorithm DSPARK`, the drafter
directory (mounted at `/models/draft` under docker) and
`--speculative-dspark-block-size` from the benchmark's `parameter_draft_length`;
those flags are reserved in `raw`. Everything else about the launch is the
cell's own `raw`; the reference configuration is
`--mem-fraction-static 0.60 --speculative-draft-attention-backend flashinfer`.

`parameter_batch_size` reaches no argv: SGLang batches continuously and sizes
its own slots, so the traversal keeps that many samples in flight. The row's
counters are the same sums; the run's stdout adds the traversal wall clock and
the aggregate tok/s over it, which is what a batch buys and the per-turn sums
cannot express.

The DSpark path needs an SGLang build that carries it (`sglang-kernel>=0.4.7`);
a stock `uv-sglang://` catalog row does not. Pin the build as a
`pip_requirements_text` body, the way
[the example plan](../../examples/plans/sglang-linux-vl-spec-decode.toml) does.

```bash
pipette benchmarks run \
  --benchmark vl_spec_decode_mmspec_test_gqa_k8 \
  --runtime '<uv_sglang JSON with the DSpark build>' \
  --model 'torch-speculative://repo=LiquidAI/LFM2.5-VL-3B&draft_repo=LiquidAI/LFM2.5-VL-3B-DSpark' \
  --runtime-flags '{"raw":["--mem-fraction-static","0.60","--speculative-draft-attention-backend","flashinfer"]}'
```

## Plans and results

Plans: [plan-runner.md](../pipette-ctl/plan-runner.md); the plan vocabulary is
pipette-mgmt
[`crates/pipette-plan-types`](https://github.com/Liquid4All/pipette-mgmt/tree/main/crates/pipette-plan-types).
What a result carries: pipette-mgmt
[`docs/benchmarks.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md).

## Caveats

- **`max_memory_usage` needs pipette on the Linux host itself** (Linux kernel
  5.18 or later, cgroup v2); Docker Desktop, cgroup v1 and a host's root cgroup for uv are
  refused. The full list is in
  [peak-memory-linux.md](../methodology/peak-memory-linux.md#host-counter).
- To inspect a live engine mid-debug, use `docker exec` / `curl` against the
  daemon. The CLI will not leave a long-lived server up after the run.
