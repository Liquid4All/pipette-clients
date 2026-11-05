# ONNX GenAI runtime

The unified `pipette` CLI drives the `pipette-onnx-genai` backend library. It runs ONNX
Runtime GenAI through a uv-managed venv. Shared operator flow: [usage.md](usage.md).
Coordinates: [models-and-runtimes.md](models-and-runtimes.md). Timing and validation
details: [ONNX GenAI measurement](../onnx-genai-measurement.md).

**Host:** The CPU provider runs on macOS, Linux, or Windows with a compatible
wheel, including macOS arm64. The CUDA provider runs on Linux x86_64 and Windows
amd64. `onnxruntime-genai-cuda` publishes no macOS wheel and no Linux arm64
wheel. A Windows arm64 install fails to resolve, because `onnxruntime-gpu`
publishes no `win_arm64` wheel. The CUDA rows' driver floor and the libraries
the venv carries are in
[The CUDA library set](../onnx-genai-measurement.md#the-cuda-library-set). Put
[`uv`](https://docs.astral.sh/uv/) on `PATH`. See [usage.md](usage.md#where-the-external-programs-come-from) to point pipette at a copy that is not on `PATH`.

## Runtime

```bash
pipette runtimes catalog uv_onnx_genai
pipette runtimes pull --runtime 'uv-onnx-genai://version=0.17.1&ep=cpu'
```

Specify both `version` and `ep`, where `ep` is the execution provider. The
pair must name a bundled catalog row. `pipette runtimes catalog uv_onnx_genai`
lists the rows: the examples on this page use `0.17.1`, and the example plan
below uses `0.18.0`. The catalog pins the whole requirements set, which
includes the underlying ONNX Runtime engine. A requirements set outside the
catalog needs a structural JSON runtime.

## Model

A GenAI export contains these files:

- `genai_config.json`
- The decoder graph that the config names
- External weights, if the graph references them
- A tokenizer, such as `tokenizer.json` or `tokenizer.model`

A plain ONNX export is insufficient.

```bash
--model 'onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32'
```

The Phi-3 weights are about 2.7 GB. Use `prefix` to select the provider variant.

## The provider is per install

Each execution provider has its own wheel and venv: CPU uses `onnxruntime-genai`, CUDA
uses `onnxruntime-genai-cuda`. The `ep` on the runtime selects the provider for every
cell that uses that installation. OpenVINO instead picks a device per cell over one
wheel.

Windows ML is refused everywhere, since GenAI's `append_provider` takes no `winml` name
([Model validation](../onnx-genai-measurement.md#model-validation)).

Verification covers the CPU provider on macOS arm64 and Linux x86_64, and the CUDA
provider on a Linux x86_64 NVIDIA host. Windows stays unverified.

The management server's planner gives a CUDA cell `accel:gpu`
([The CUDA capability rule](../onnx-genai-measurement.md#the-cuda-capability-rule)).

## Hardware coverage

This backend covers Intel and AMD CPUs through the CPU provider. It covers NVIDIA
GPUs through the CUDA provider. It covers no Intel GPU and no AMD GPU. PyPI carries
no `onnxruntime-genai-openvino` wheel, and the upstream `build.py` declares no
`--use_openvino` flag, so an OpenVINO-provider leg needs a source build. Use the
[OpenVINO backend](openvino.md) for Intel CPU, GPU and NPU hardware.

## Runtime flags

There are no ONNX GenAI runtime flags. The shared flag validation rejects
`--runtime-flags '{"device":"cpu"}'`. Select `ep=cpu` in the runtime coordinate. An
empty `{}` means no flags.

## Supported benchmarks

| Type | Status |
| --- | --- |
| `prefill_throughput`, `decode_throughput`, `end_to_end_latency`, `max_memory_usage` | supported |
| `eval` | not implemented; no eval checkpoints |
| `vl_throughput`, `vl_max_memory`, `vlm_eval`, `vl_spec_decode` | not implemented |
| `asr_throughput`, `asr_max_memory`, `asr_eval` | not implemented |

A batched entry is refused before the model loads, since the server generates
one sequence at a time ([Batched](../methodology/end-to-end-latency.md#batched)).
`decode_throughput` needs `parameter_decode_tokens` of at least 2
([decode-throughput.md](../methodology/decode-throughput.md#benchmark-shape-and-reported-fields)).
Host and GPU memory recording, and which providers are refused for it, are in
[Host memory](../onnx-genai-measurement.md#host-memory).

## Weight loading and validation

Each cell's server loads the model before it listens, so no request time
includes the load. The memory cell also runs a warm-up server first. There is no compiled blob cache.

The harness checks the config, graph, builder-named external data and provider
before the server loads the model
([Model validation](../onnx-genai-measurement.md#model-validation)). A missing
config error suggests that the model may be a plain ONNX export, to rebuild with
`python -m onnxruntime_genai.models.builder`.

## Setup sketch

```bash
pipette init
pipette benchmarks init-local
pipette runtimes pull --runtime 'uv-onnx-genai://version=0.17.1&ep=cpu'
pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32' \
  --runtime 'uv-onnx-genai://version=0.17.1&ep=cpu'
```

A CUDA host pulls the CUDA row and names the CUDA model variant:

```bash
pipette runtimes pull --runtime 'uv-onnx-genai://version=0.17.1&ep=cuda'
pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cuda/cuda-int4-rtn-block-32' \
  --runtime 'uv-onnx-genai://version=0.17.1&ep=cuda'
```

The CPU export and the CUDA export are separate directories of one repository.
A CPU directory against the CUDA runtime fails the provider check before the
model loads.

The four-cell example is
[`examples/plans/onnx-genai-cpu.toml`](../../examples/plans/onnx-genai-cpu.toml).
