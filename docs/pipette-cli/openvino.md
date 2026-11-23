# OpenVINO runtime

Backend library: `pipette-openvino`, driven by the unified `pipette` CLI.

Runs benchmarks on an Intel CPU, GPU or NPU (neural processing unit) through
`openvino-genai` in a uv-managed venv. Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

Deeper background lives in [openvino-ir.md](../openvino-ir.md) (the layout
of the IR, or intermediate representation; precision identification; NPU
constraints) and
[openvino-measurement.md](../openvino-measurement.md) (timing, compile, cache).

**Host:** Linux or Windows. On any other OS, macOS included, pipette refuses
the cell before anything installs. Requires [`uv`](https://docs.astral.sh/uv/)
on `PATH`. See [usage.md](usage.md#where-the-external-programs-come-from) to point pipette at a copy that is not on `PATH`.

## Runtime

One venv serves every device, so the install carries no device and no flavor:

```bash
pipette runtimes catalog uv_openvino
pipette runtimes pull --runtime 'uv-openvino://version=2026.4.1'
```

`version` is the only key, and it must name a row of the bundled catalog
([bundled-catalog/openvino](../../crates/pipette-artifacts-catalog/bundled-catalog/openvino/README.md)).
An off-catalog pin needs a JSON `--runtime` carrying a full
`pip_requirements_text` source.

## Model

An OpenVINO IR bundle: `openvino_model.xml`/`.bin` plus the tokenizer and
detokenizer IR pairs.

```bash
--model 'openvino://repo=OpenVINO/LFM2.5-350M-int8-ov'
```

A VL cell takes a VL IR under the same model type: the language model, text
embeddings and vision embeddings IR pairs (`openvino_language_model`,
`openvino_text_embeddings_model`, `openvino_vision_embeddings_model`), the
tokenizer and detokenizer IRs, `config.json`, and the image processor's
settings in `processor_config.json` or `preprocessor_config.json`, as
optimum-intel exports them. A text cell refuses a VL IR, and a VL cell a text
IR, before the server starts.

LFM3-VL needs openvino-genai 2026.5, which no catalog row ships yet (the
newest is 2026.4.1). Pin the nightly, tested on 2026.5.1.0.dev20261116, as a
JSON `--runtime` with a `pip_requirements_text` body and OpenVINO's nightly
index in `install_flags`
([when you must use JSON](models-and-runtimes.md#when-you-must-use-json)), as
[`examples/plans/intel-openvino-vl.toml`](../../examples/plans/intel-openvino-vl.toml)
does.

Weight precision is never inferred from the directory; it is authored into the
repo name or the `prefix`
([openvino-ir.md](../openvino-ir.md#identifying-weight-precision)).

## The device is per cell, not per install

`uv-openvino://` takes no `device` key. The device is a runtime flag on the
cell instead (`--runtime-flags '{"device":"cpu"}'`, as in the
[setup sketch](#setup-sketch)), so one installed venv covers cpu, gpu and npu.

Omitting it is an error, raised at execute time after the venv build and the
model download, since OpenVINO would otherwise silently pick CPU:

```text
this OpenVINO cell names no `device`; set one on the cell's runtime_flags ...
```

## Runtime flags

The flag fields and their values are defined in pipette-mgmt
[`crates/pipette-plan-types/src/cell/runtime_flags.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/runtime_flags.rs)
and
[`crates/pipette-plan-types/src/runtime/openvino.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/runtime/openvino.rs).
What this client does with them (`crates/pipette-openvino/src/flags.rs`):

- `max_prompt_len`, `min_response_len` and `generate_hint` reach the pipeline
  only on `npu`. On `cpu`, `gpu` and a custom device string, the client drops
  them from the pipeline and from the recorded flags. The record then names
  only the settings that the pipeline got.
- On `npu`:
  - If the cell's decode length is above GenAI's 128-token default, the client
    raises `min_response_len` to the decode length.
  - The client refuses a prompt over the 1024-token `max_prompt_len` bound
    with an error that explains why. It does not truncate the prompt. A VL
    prompt counts its image tokens and the chat template's: on LFM3-VL it is
    10 + text + N x (image tokens + 2), so five 512x512 images make 1300
    tokens. The client counts it for LFM3-VL only; on another model GenAI
    refuses an over-long prompt itself, after the compile.
  - `generate_hint` defaults to `BEST_PERF`.

Set them yourself only to override that. See
[openvino-ir.md](../openvino-ir.md#npu-constraints) for the constraints.

## Supported benchmarks

| Type | Status |
|------|--------|
| `prefill_throughput`, `decode_throughput`, `end_to_end_latency`, `max_memory_usage` | supported |
| `vl_throughput`, `vl_max_memory` | supported, through `VLMPipeline` ([vl-throughput.md](../methodology/vl-throughput.md#runtime-details)) |
| `eval` | refused: `eval benchmarks are not yet supported for OpenVINO` |
| `vlm_eval`, `vl_spec_decode` | refused: `vlm_eval and vl_spec_decode are not yet supported for OpenVINO` |
| `asr_throughput`, `asr_max_memory`, `asr_eval` | refused: `ASR benchmarks are not yet supported for OpenVINO` (`asr_eval is not yet supported for OpenVINO`) |

A batched entry (`parameter_batch_size` above 1) is refused before the model
loads, since the server generates one sequence at a time; the rule is in
[end-to-end-latency.md](../methodology/end-to-end-latency.md#batched).
Batched timing cells run on vLLM or SGLang
([torch-oai.md](torch-oai.md#batched-timing-cells)), and on `ovms` for the
same IR on CPU and GPU ([ovms.md](../ovms.md#batched-entries)).

## Compile cache

Compiled blobs and GPU kernels land in the workspace's
`cache/<runtime-key>/`, outside the artifact stores. `storage status` does not
count them, and `storage gc` does not remove them. `runtimes remove` reclaims
them. For a 350M model, a blob is about 94 to 96 MB on the NPU and 358 to
363 MB on the CPU or GPU for its int8 IR. Why
the cache exists: [openvino-measurement.md](../openvino-measurement.md).

## Setup sketch

```bash
pipette init
pipette benchmarks init-local
pipette runtimes pull --runtime 'uv-openvino://version=2026.4.1'

pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'openvino://repo=OpenVINO/LFM2.5-350M-int8-ov' \
  --runtime 'uv-openvino://version=2026.4.1' \
  --runtime-flags '{"device":"cpu"}'
```

A worked multi-device configuration, sweeping the same model across cpu, gpu and
npu, is [`examples/plans/intel-openvino.toml`](../../examples/plans/intel-openvino.toml).
