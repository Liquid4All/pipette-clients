# OpenVINO runtime

Backend library: `pipette-openvino`, driven by the unified `pipette` CLI.

Runs Intel CPU / GPU / NPU benchmarks through `openvino-genai` in a uv-managed
venv. Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

Deeper background lives in [openvino-ir.md](../openvino-ir.md) (IR layout,
precision identification, NPU constraints) and
[openvino-measurement.md](../openvino-measurement.md) (timing, compile, cache).

**Host:** Linux or Windows x86_64. Requires [`uv`](https://docs.astral.sh/uv/)
on `PATH`.

## Runtime

One venv serves every device, so the install carries no device and no flavor:

```bash
pipette runtimes catalog uv_openvino
#  REF                            | FLAVOR
# --------------------------------+------------
#  uv-openvino://version=2026.2.1 | any device

pipette runtimes pull --runtime 'uv-openvino://version=2026.2.1'
```

`version` is the only key, and it must name a bundled-catalog row. An
off-catalog pin needs a JSON `--runtime` carrying a full
`pip_requirements_text` source.

## Model

An OpenVINO IR bundle: `openvino_model.xml`/`.bin` plus the tokenizer and
detokenizer IR pairs.

```bash
--model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw'
```

Weight precision is **never inferred** from the directory: `openvino_config.json`
does not record it, and an fp16 export omits the file entirely. Precision has to
be authored into the repo name or the `prefix`, which is why a multi-variant repo
is addressed as `prefix=int4-sym-cw`.

## The device is per cell, not per install

This is the one thing that trips people up. `uv-openvino://` takes no `device`
key:

```text
unknown key `device` for scheme `uv-openvino`
```

The device is a runtime flag on the run instead, so a single installed venv
covers cpu, gpu and npu:

```bash
pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw' \
  --runtime 'uv-openvino://version=2026.2.1' \
  --runtime-flags '{"device":"cpu"}'
```

Omitting it is an error, and one raised at execute time, after the venv build
and the model download:

```text
this OpenVINO cell names no `device`; set one on the cell's runtime_flags
```

## Runtime flags

`device`, `max_prompt_len`, `min_response_len`, `generate_hint`. There is no
`raw` and no `envs` on OpenVINO cells.

| Setting | Values | Notes |
|---------|--------|-------|
| `device` | `cpu`, `gpu`, `npu`, or a verbatim custom string | required |
| `max_prompt_len` | tokens | NPU static-shape prompt bound; GenAI defaults to 1024 |
| `min_response_len` | tokens | NPU output reservation; GenAI's 128 default truncates a 256-token cell |
| `generate_hint` | `best-perf`, `fast-compile` | rendered as `BEST_PERF` / `FAST_COMPILE` |

On the NPU the client derives both rather than leaving you to guess:
`min_response_len` is raised to the cell's decode length whenever that exceeds
GenAI's 128-token default, and a prompt over the 1024-token bound is rejected
with an explanatory error rather than truncated. Set them yourself only to
override that. See [openvino-ir.md](../openvino-ir.md) for the constraints.

## Supported benchmarks

| Type | Status |
|------|--------|
| `prefill_throughput`, `decode_throughput`, `end_to_end_latency`, `max_memory_usage` | supported |
| `eval` | not implemented: `eval benchmarks are not yet supported for OpenVINO` |
| `vl_throughput` | not implemented |

Because eval never runs here, OpenVINO has no eval checkpoints and no doom-loop
settings.

## Compile cache

Compiled blobs land in `.pipette/cache/<runtime-key>/`, outside the artifact
stores so a storage sweep cannot delete what a run depends on. They are
therefore not counted by `storage status` and not swept by `storage gc`;
`runtimes remove` is what reclaims them. Sizes are substantial, roughly 94 MB
per blob for a 350M model. Details:
[openvino-measurement.md](../openvino-measurement.md).

## Setup sketch

```bash
pipette init
pipette benchmarks init-local
pipette runtimes pull --runtime 'uv-openvino://version=2026.2.1'

pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw' \
  --runtime 'uv-openvino://version=2026.2.1' \
  --runtime-flags '{"device":"cpu"}'
```

A worked multi-device configuration, sweeping the same model across cpu, gpu and
npu, is [`examples/plans/intel-openvino.toml`](../../examples/plans/intel-openvino.toml).
