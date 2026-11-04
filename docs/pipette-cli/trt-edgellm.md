# TensorRT Edge-LLM runtime

Backend library: `pipette-trt-edgellm`, driven by the unified `pipette` CLI.

Runs vision-language cells on an NVIDIA Jetson through a TensorRT Edge-LLM
server archive.
The server reads the same GGUF vision pair llama.cpp does, builds TensorRT
engines from it on the device, and speaks llama-server's HTTP dialect, so the
cells measure it with llama.cpp's request and timing definitions. Shared
operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

**Host:** a Jetson running the JetPack release that the archive was built for.

- **Flavor.** A flavor names the SoC (system on a chip) and the CUDA major
  version that the plugin is built for. `jetson-orin-cuda-13` is a Jetson Orin
  on CUDA 13, as JetPack 7.2 ships it. On an AGX Orin that is L4T (Linux for
  Tegra) R39, CUDA 13.2, TensorRT 10.16 and GPU architecture sm_87.
- **TensorRT.** The engines also depend on the host's TensorRT version, and
  the server keys its cache by it.
- **Host software.** The archive needs only what JetPack installs: TensorRT
  with its `python3-libnvinfer` bindings, CUDA, and Ubuntu's
  `/usr/bin/python3.12`.
- **Checks.** On any OS other than Linux, pipette refuses the cell before
  anything installs. The preflight also refuses a host whose
  `/proc/device-tree/compatible` does not name the flavor's SoC:
  - `jetson-orin-cuda-13` needs `nvidia,tegra234`, so pipette refuses a Thor
    (`nvidia,tegra264`).
  - A custom flavor needs any `nvidia,tegra` SoC.
- **Install.** A named flavor installs only on Linux arm64. A custom flavor
  installs on any host and leaves the check to the server.

## Runtime

Upstream publishes no server archive, so there is no catalog: a runtime names
an archive by URL, or a vault upload.

```bash
pipette runtimes pull --runtime 'trt-edgellm://url=https://example.com/trt-edgellm-server-0.11.0-jetson-orin-cuda-13-4e3f9d23f052.tar.gz&flavor=jetson-orin-cuda-13'
```

An archive name such as the one above carries the tag, the flavor and a
content hash, so the URL pins the build. A vault archive (`pipette-ctl` upload type
`trt_edgellm`) travels as JSON:

```json
{"type": "trt_edgellm", "source": "vault_remote_archive",
 "url": "/vault/resources/<id>/content", "principal_id": "org_...",
 "custom_label": "TensorRT Edge-LLM 0.11.0", "flavor": "jetson-orin-cuda-13"}
```

The install unpacks the archive under the store and checks its top-level
`trt-edgellm-server/` for `bin/trt-edgellm-server` and
`lib/libNvInfer_edgellm_plugin.so`
([layout](../../crates/pipette-artifacts-catalog/src/runtime/trt_edgellm.rs)).
The worker advertises `runtime:trt_edgellm` and
`runtime:trt_edgellm:<archive url or vault resource id>`.

## Model

A `gguf_vision` pair, the only model type the runtime takes: the language
model GGUF and its `mmproj`. For a vision-language-action (VLA) model, the
projector carries the action head and the state projection (`robo.*`
tensors). The server converts the pair to its
builder's checkpoint once per pair. It converts BF16, F16 and F32 GGUFs only
and refuses a quantized one by tensor and type: TensorRT's INT8 is per channel,
so a Q8_0 GGUF would be quantized twice. Run the BF16 GGUF with
`precision = "w8a16"` for the Q8_0 counterpart. A row's weights are its
`runtime_flags.precision`, not the model's GGUF type. The converter knows the
LFM3-VL layout; another architecture fails the build with a message naming it.

## Cells and flags

Only `vl_throughput` and `vl_max_memory` run; every other benchmark type is
refused by name before anything starts. Both cells take these runtime flags:

| Flag | Meaning |
|------|---------|
| `precision` | The precision of the language model's engine. See the list below the table. |
| `envs` | Env forwards for the build and the server. |
| `raw` | The server's own flags. See the list below the table. |

`precision` takes one of these values:

- `fp16`: the default. The row records `fp16` when the cell leaves it unset.
- `bf16`: BF16 matrix multiplications (GEMMs), norms and residual stream.
  Attention, convolution and the key-value (KV) cache stay FP16, because
  sm_87 has only FP16 attention kernels.
- `w8a16`: INT8 weights per output channel with FP16 activations. The build
  quantizes them from the BF16 weights. The scales are per channel, not
  Q8_0's block scales.

The vision encoder always builds at FP16, and a VLA head always runs in FP32.

`raw` holds the server's own flags:

- The client appends them to the argv of the measured server, never to the
  build's argv.
- Examples: the robot state and embodiment of a VLA run (`--robo-state
  0.1,...`, `--robo-embodiment 0`), or `--ctx-size N` to refuse longer
  requests.
- Plan parsing refuses the flags that the harness sets: `--model`, `-m`,
  `--mmproj`, `--host`, `--port`, `--cache-dir` and `--build-only`. It also
  refuses `--precision`, which is the same setting as `precision`.

The benchmark flags are llama.cpp's: `http_timeout_seconds` and `readiness` on
`vl_throughput`, `http_timeout_seconds` and `bench_timeout_seconds` on
`vl_max_memory`.

## Engine build

Each cell first runs

```text
trt-edgellm-server --model M --mmproj J --precision P --cache-dir C --build-only
```

with `M` the language model GGUF, `J` its `mmproj`, `P` the precision, and `C`
the runtime's compile cache (`.pipette/cache/<runtime key>/`, outside
the storage quota, reclaimed by `pipette runtimes remove`). The build has a
deadline of its own, one hour, and runs before the server starts, so it sits
outside every measured window and every memory sampler. The server keys what it
builds by both GGUFs' sha256, a hash of its plugin, runtime, builder and
converter, the TensorRT, CUDA runtime and driver versions, the L4T release, the
SoC, the precision and the engine profile:

```text
C/checkpoints/<id>/   the converted GGUF pair
C/engines/<key>/      llm/ and vla/ for one precision
C/engines/<vkey>/     the FP16 vision engine every precision shares
```

A cold build of LFM3-VL-2B on an AGX Orin takes about 5 minutes for the first
precision (conversion, the vision, language and VLA engines) and 2 to 3 for
each other one. On a warm cache the step hashes the two GGUFs and returns, a
few seconds. The measured server never builds: with no engines for its key it
exits before binding its port, and the cell fails quoting the build command.

## Measurement

The cells are llama.cpp's
([vl_throughput](../methodology/vl-throughput.md#runtime-details),
[vl_max_memory](../methodology/vl-max-memory.md#platform-status)) through the
shared client in
[`pipette_completions::llama_server`](../../crates/pipette-completions/src/llama_server.rs):

- `prompt_ms` and `predicted_ms` are the server's clock, `predicted_ms` scaled
  from its `n - 1` decode steps to `n`; `total_time_ms` is this client's.
  `prompt_ms` includes the on-GPU image resize and normalization, which
  llama.cpp does on the CPU before its timer.
- `image_tokens` is the server's `timings.image_n`. Pipette does not run
  llama.cpp's image-token calibration helper on this runtime.
- `client_extras.trt_edgellm` is the engine stack, the server's `/props`
  `trt_edgellm` object as it reports it: `engine_key` (its cache key),
  `edgellm_commit` and `dirty` (the Edge-LLM tree the archive was built from),
  `tensorrt`, `cuda_runtime`, `cuda_driver`, `l4t` and `soc`, each null when
  the server cannot tell. A server whose `/props` has none is an archive too
  old for pipette, and the cell fails.
- Memory is the swap-aware `/proc/<pid>/status` peak of the server process.
  The archive's launcher `exec`s the server, so the pid is the server's. On a
  Jetson the GPU watch reads only that, because the integrated GPU's CUDA
  buffers are in the process's resident set (RSS); `max_gpu_bytes` is null.
  The engine profile (8192 tokens) fixes the KV cache and activation memory,
  so the peak does not shrink with a short prompt the way llama.cpp's does.

## Measured

LFM3-VL-2B with its VLA head, as a BF16 GGUF pair with an F16 projector, on
the Orin host (AGX Orin 64 GB, JetPack 7.2, MAXN with `jetson_clocks`), local
cells, a TensorRT Edge-LLM 0.11.0 server archive. Manipulation ran in VLA mode (`--robo-state`, `--robo-embodiment 0`),
orchestration on the plain language engine.

| Cell | Precision | Result |
|------|-----------|--------|
| manipulation `vl_throughput`, 2x256x256, 238 tokens (128 image) | fp16 | `prompt_ms` 49.68 (sd 0.03) |
| manipulation `vl_throughput` | w8a16 | `prompt_ms` 93.58 (sd 0.02) |
| orchestration `vl_throughput`, 3x512x512, 4182 tokens (768 image), 47 decoded | fp16 | `prompt_ms` 588.4, 44.7 tok/s decode |
| manipulation `vl_max_memory` | fp16 | 6.19 GB |
| manipulation `vl_max_memory` | w8a16 | 4.73 GB |
| orchestration `vl_max_memory` | fp16 | 6.01 GB |

The cold build took 299 s for fp16 (conversion, vision, language and VLA
engines) and 158 s for w8a16; each later cell's build step took 3.9 s.

## Code References

- [`pipette-trt-edgellm`](../../crates/pipette-trt-edgellm/src/lib.rs): the
  build step in
  [`engines`](../../crates/pipette-trt-edgellm/src/engines.rs), the command
  lines in [`server`](../../crates/pipette-trt-edgellm/src/server.rs), the
  cells in
  [`execute`](../../crates/pipette-trt-edgellm/src/execute/mod.rs).
- The install:
  [`pipette-artifacts` `trt_edgellm`](../../crates/pipette-artifacts/src/runtime/trt_edgellm.rs).
- The preflight:
  [`pipette-cli` `preflight`](../../crates/pipette-cli/src/preflight.rs).
