# OpenVINO IR

This page describes what an OpenVINO IR (intermediate representation) model is
on disk, how to identify its weight precision, and the device constraints that
the `uv_openvino` and `ovms` runtimes work within. How cells are measured
against those constraints (warm-up, compile time, what the readiness gate can
see) is in [openvino-measurement.md](openvino-measurement.md).

The asymmetric spellings were read off the published `OpenVINO/` exports
(`LFM2-24B-A2B-int4-ov`, `LFM2.5-350M-int8-ov`); the symmetric ones off our own.

Everything below marked *measured* was run on one Windows box: Intel Core Ultra 7
258V (Lunar Lake), Arc 140V iGPU (sharing the CPU's on-package memory), Intel
AI Boost NPU (driver `32.0.100.4778`), **31.6 GiB RAM**, Windows 11 build
26200, `openvino-genai` 2026.2.1. The 12 GB `LFM2-24B-A2B` IR therefore fits,
though with the iGPU drawing on the same pool it is tight.

The pipette timings on this page (the NPU speed table) came from a per-rep
driver: each rep was its own driver process that timed itself. Do not compare
them with server-path rows
([openvino-measurement.md](openvino-measurement.md)).

## The artifact

A **directory**, not a weights file; closest to `Model::Mlx`, unlike the
single-file gguf models:

```
lfm2.5-350m-int4-sym-cw/
  openvino_model.xml / .bin          # graph + weights
  openvino_tokenizer.xml / .bin      # tokenizer, compiled to IR
  openvino_detokenizer.xml / .bin    # detokenizer, compiled to IR
  config.json  generation_config.json
  tokenizer.json  tokenizer_config.json  chat_template.jinja
```

The tokenizer pair is **required**. `openvino_genai.LLMPipeline` does not use
the `tokenizers` library at runtime. It runs the tokenizer as a compiled
OpenVINO model. `optimum-cli export openvino` emits those files only when
`openvino-tokenizers` is installed in the export environment, version-matched to
`openvino`. OpenVINO loads a directory without them, but cannot generate from
it. pipette refuses such a directory before the server starts.

## Identifying weight precision

Precision is part of the authored coordinate (the repo name, or a `prefix`
subdirectory when one repo carries several variants) the way a gguf quant is
part of its filename. The client does not infer it from the directory.

Do not read `openvino_config.json` for it. That file records `optimum_version`
and `transformers_version`; its `quantization_config` is empty of the weight
format in every quantized export, and the file is absent entirely from fp16
exports.

The IR itself is authoritative. Read it by hand, because the directory name
can disagree with it. See "Which precisions the NPU actually
runs" below for what each one does on hardware:

| Signal in `openvino_model.xml` | Means |
|---|---|
| `element_type="i4"` | int4 **symmetric** |
| `element_type="u4"` | int4 asymmetric |
| `element_type="i8"` | int8 symmetric |
| `element_type="u8"` with no narrower weight type | int8 **asymmetric** |
| `element_type="f16"` alone | fp16 |
| int4 constant rank 2, e.g. `(4608, 1024)` | channel-wise (`--group-size -1`, the `cw` suffix) |
| int4 constant rank 3, e.g. `(4608, 8, 128)` | grouped; trailing dim **is** the group size (`gq128` is a group size of 128) |

### Why signedness identifies the mode

Signedness identifies the mode because of how NNCF (Neural Network Compression
Framework, the OpenVINO weight compression tool) defines the modes. NNCF's
[weight-compression documentation](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
defines `INT4_SYM` as *"quantizing weights to a **signed** 4-bit integer
symmetrically **without zero point**"* and `INT4_ASYM` as the same *"but weights
are quantized … asymmetrically with a typical non-fixed **zero point**"*; the
symmetric quantizer *"restricts the zero-point to 0"*. A signed grid with no
zero-point tensor is `i4`/`i8`; an unsigned grid plus a zero-point tensor per
group is `u4`/`u8`.

The tensor counts corroborate it. A matched pair; same base model
(`LiquidAI/LFM2.5-350M`), same bit width, differing only in mode, each labeled
independently of its IR (ours by the `--sym` export flag, theirs by the
published model card's `mode: INT8_ASYM`):

| export | `i8` | `u8` |
|---|---|---|
| ours, `--weight-format int8 --sym` | 103 | 0 |
| `OpenVINO/LFM2.5-350M-int8-ov` | 0 | 206 |

206 = 103 weight tensors + 103 zero-points, exactly as the spec predicts, and
the symmetric export carries none: matching *"restricts the zero-point to 0"*.

Two caveats. `u8` also appears as zero-points **inside** a 4-bit graph (22 in
our `int4-sym-cw`, 62 in the published asymmetric 24B), which is why it
identifies int8-asym only when no narrower weight type is present: reading
widest-first calls every int4 export int8. And a card may state a `ratio`
below 1.0 (the 24B is `ratio: 0.8`) meaning a *mixed* int4/int8 graph carrying
`u4`, `u8` and `f16` together; the narrowest-type rule still yields the right
answer, and there it is the rule that decides.

Asymmetric does not mean NPU-unsafe: see "Which precisions the NPU actually
runs" below.

## Device support

*Measured.* `ov.Core().available_devices` reports `['CPU', 'GPU', 'NPU']`, and
`LFM2.5-350M` int4-sym-cw runs on all three; greedy, 16-token prompt, 100 new
tokens:

TTFT is the time to first token, and TPOT the time per output token.

| Device | Compile | TTFT | TPOT | Throughput |
|---|---|---|---|---|
| NPU | 17.5s | 96.6ms | 6.22ms | 160.7 tok/s |
| GPU (Arc 140V) | 6.0s | 43.0ms | 8.56ms | 116.9 tok/s |
| CPU | 0.8s | 33.0ms | 6.70ms | 149.4 tok/s |

Indicative only. The prompt is far too short to say anything about prefill, and
see the NPU stability note below before treating any NPU figure as a baseline.

Per OpenVINO's support matrix, only the **dense** LFM2 models have an NPU
path; the mixture-of-experts (MoE) models `LFM2-8B-A1B` and `LFM2-24B-A2B` are
CPU/GPU only. One MoE model, `LFM2.5-8B-A1B`, is measured on the NPU below
([Mixture-of-experts on the NPU](#which-precisions-the-npu-actually-runs)).

## NPU constraints

**Static shapes.** The NPU pipeline is static-shape. Default `MAX_PROMPT_LEN`
is 1024, and exceeding it is a clean refusal rather than a truncation:

```
RuntimeError: Stateful LLM pipeline on NPU may only process prompts or hold
chat history up to 1024 tokens. 1509 is passed.
```

**Raising it costs compile time**, superlinearly (*measured*):

| `MAX_PROMPT_LEN` | Compile |
|---|---|
| 1024 (default) | 17.8s |
| 2048 | 26.9s |
| 4096 | 62.8s |

Every prompt of 1024 tokens or fewer runs on the default bound. pipette refuses
an NPU cell whose prompt is longer than the bound, unless the cell sets
`max_prompt_len`. A larger `max_prompt_len` compiles a larger static shape, at
the cost above. A prefill ladder that reaches 8k tokens is therefore not
practical on the NPU. This limit belongs to plan authoring: a backend cannot
work around it.

**One pipeline per process.** Compiling three pipelines with different
`MAX_PROMPT_LEN` in a single process took the device down:

```
ZE_RESULT_ERROR_DEVICE_LOST — device hung, reset, was removed, or driver update occurred
```

It recovered without intervention, but throughput came back degraded and
climbed over subsequent runs: 91.5 → 121.0 → 133.7 tok/s against a 160.7
baseline. NPU cells need a fresh process each, repeated reps, and a settle
period before the numbers mean anything. Separate processes are safe: the
harness compiles one pipeline in each server process
([openvino-measurement.md](openvino-measurement.md#what-each-cell-samples)).

## Runtime environment

Inference needs `openvino-genai` only: no `transformers`, `optimum-intel`, or
`nncf`, which are export-side. Match the runtime `openvino` version to whatever
produced the IR. The bundled
[runtime catalog](../crates/pipette-artifacts-catalog/bundled-catalog/openvino/catalog.toml)
pins those sets.

## How the backend answers these

- **One compile per process**, outside every timed window:
  [openvino-measurement.md](openvino-measurement.md#what-each-cell-samples).
- **Raw prompt tokens, not a chat-templated prompt**:
  [openvino-measurement.md](openvino-measurement.md#no-chat-template).
- **NPU static-shape settings are derived from the cell**, with authored
  overrides on top:
  [pipette-cli/openvino.md](pipette-cli/openvino.md#runtime-flags).
- **No precision or architecture is rejected at preflight**: see
  [Why there are no preflight guards](#why-there-are-no-preflight-guards).
- **Per-rep driver rows for `LFM2.5-350M`** on CPU and NPU:
  [openvino-measurement.md](openvino-measurement.md#per-rep-driver-rows).

## Which precisions the NPU actually runs

Measured on the same Core Ultra 7 258V across the full LFM2.5-350M
precision set; same architecture and size throughout, greedy decoding, CPU as
the reference. Correctness first:

| precision | NPU correctness |
|---|---|
| `fp16` | matches CPU |
| `int4-sym-cw` | matches CPU |
| `int4-sym-gq128` | matches CPU |
| `int8-asym` | coherent, tracks CPU |
| `int4-asym-gq128` | **fails to compile**: `vpux-compiler ... Found 168 duplicated names` |
| `int8-sym` | **compiles, runs, returns incoherent output** |

`int8-sym` on NPU answered `相追` / `相` / `相` to three prompts whose CPU
answers were correct, reproducibly across separate runs. An
`LFM2-1.2B/int4-sym-cw` control produced byte-identical output on both devices,
so the harness was sound.

Then speed, 512 prefill / 100 decode, best of three:

| precision | NPU prefill | NPU decode | decode tok/s |
|---|---|---|---|
| `int4-sym-cw` | 88.7 ms | 1133 ms | 94.8 |
| `int8-sym` | 89.3 ms | 1262 ms | 84.4 |
| `int8-asym` | 176.1 ms | 9749 ms | **10.3** |

**Asymmetric int8 works on the NPU but is 8× slower than its peers**,
probably because no optimized kernel exists for that layout.

`int8-sym`'s timing sits in band with `int4-sym` rather than looking
anomalously fast, so its numbers are not obviously the product of skipped work,
but its output is wrong, so the row must not be read as a usable
configuration.

### Why there are no preflight guards

A support-matrix guard that refused asymmetric weights on the NPU got two of
its five verdicts wrong: asymmetric int8 runs correctly, asymmetric int4
refuses on its own, and the precision that silently misbehaves is a
*symmetric* one. So the client refuses no precision or architecture:

- A failure that throws needs no preflight: `int4-asym` already reports a
  clear compiler error, and a guard only swaps one clear message for another.
- Refusing a cell suppresses findings. The 10.3 tok/s row above is exactly what
  the campaign exists to measure.
- Detecting the one silent case cheaply does not work. Generated-token count
  fails (the broken pair runs to full length while a healthy one stops at 4);
  reply length works only under one prompt formatting. The one reliable signal
  is running the same prompt on CPU and comparing, which costs a second
  compile. Hardcoding the answer instead is a fact about one model on one
  chassis.

Which precision/device pairs are worth measuring belongs to the plan, authored
per campaign.

**Scope.** One model, one chassis. LFM2.5-350M is the only export carrying both
an int8-sym and an int8-asym variant, so the int8 rows could not be replicated
at another size.

**Not a truncation bug.** With `ignore_eos` at its default the incoherent NPU
runs also stopped after one or two tokens, which looked like it would skew a
throughput measurement. It does not: the client sends `ignore_eos: true` on
every timing request, and under that setting every variant returns the full
requested count.

**Mixture-of-experts on the NPU fails loudly.**
`LFM2.5-8B-A1B/int8-asym` (`model_type: lfm2_moe`, 32 experts) refuses at
compile on the NPU:

```
[vpux-compiler] InitialLowPrecisionTransformationsPipelineRewriterExecutor
                Pass failed : Got illegal group-wise pattern!
RuntimeError: ... src\plugins\intel_npu\src\plugin ...
```

and generates correctly on CPU, so a preflight would add nothing but an earlier
error message.

`int8-asym` is the right probe here because it is *known good* on the NPU for
the dense LFM2.5-350M, which leaves the MoE architecture as the variable.
Every MoE export we have is asymmetric, so this is the only clean isolation
available. The compiler blames a group-wise weight pattern rather than the
expert routing, so the precise mechanism is not established; what is
established is that the pairing fails, and fails with an error.

## Open questions

- At `MAX_PROMPT_LEN` 2048 and 4096 generation returned a single token where
  1024 returned the requested 8. Could be EOS on the synthetic prompt used, or a
  real cliff at larger static shapes. **Unresolved**: settle it before
  measuring decode on NPU.
- These figures are Lunar Lake. Intel's published LFM2 perf sheets are Panther
  Lake, a different NPU generation, so the two are not comparable row-for-row.
