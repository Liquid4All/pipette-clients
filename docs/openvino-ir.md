# OpenVINO IR

What an OpenVINO IR model is on disk, how its weight precision is identified,
and the device constraints the `uv_openvino` runtime has to live with. Both the
`Model::Openvino` coordinate and the runtime exist and execute benchmarks; the
constraints below are what shaped that backend's design.

How cells are measured against those constraints (warm-up, compile time,
what the readiness gate can see) is
[openvino-measurement.md](openvino-measurement.md).

The asymmetric spellings were read off the published `OpenVINO/` exports
(`LFM2-24B-A2B-int4-ov`, `LFM2.5-350M-int8-ov`); the symmetric ones off our own.

Everything below marked *measured* was run on `devcloud`: Intel Core Ultra 7
258V (Lunar Lake), Arc 140V iGPU (sharing the CPU's on-package memory), Intel
AI Boost NPU (driver `32.0.100.4778`), **31.6 GiB RAM**, Windows 11 build
26200, `openvino-genai` 2026.2.1. The 12 GB `LFM2-24B-A2B` IR therefore fits,
though with the iGPU drawing on the same pool it is tight.

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
`openvino`. A directory without them loads but cannot generate.

## Identifying weight precision

Precision is part of the authored coordinate (the repo name, or a `prefix`
subdirectory when one repo carries several variants) the way a gguf quant is
part of its filename. It is **not** inferred from the directory.

Do not read `openvino_config.json` for it. That file records `optimum_version`
and `transformers_version`; its `quantization_config` is empty of the weight
format in every quantized export, and the file is absent entirely from fp16
exports.

The IR itself is authoritative, and is worth reading because the directory name
can disagree with it. Each cell logs the classification it found. See "Which
precisions the NPU actually runs" below for what each one does on hardware:

| Signal in `openvino_model.xml` | Means |
|---|---|
| `element_type="i4"` | int4 **symmetric** |
| `element_type="u4"` | int4 asymmetric: CPU/GPU only |
| `element_type="i8"` | int8 symmetric |
| `element_type="u8"` with no narrower weight type | int8 **asymmetric**: CPU/GPU only |
| `element_type="f16"` alone | fp16 |
| int4 constant rank 2, e.g. `(4608, 1024)` | channel-wise (`--group-size -1`) |
| int4 constant rank 3, e.g. `(4608, 8, 128)` | grouped; trailing dim **is** the group size |

### Why signedness identifies the mode

This is not pattern-matching on observed exports. It follows from what the
compression modes are. NNCF's
[weight-compression documentation](https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md)
defines `INT4_SYM` as *"quantizing weights to a **signed** 4-bit integer
symmetrically **without zero point**"* and `INT4_ASYM` as the same *"but weights
are quantized … asymmetrically with a typical non-fixed **zero point**"*; the
symmetric quantizer *"restricts the zero-point to 0"*. A signed grid with no
zero-point tensor is `i4`/`i8`; an unsigned grid plus a zero-point tensor per
group is `u4`/`u8`.

The tensor counts corroborate it. A matched pair; same base model
(`LiquidAI/LFM2.5-350M`), same bit width, differing only in mode, each labelled
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
identifies int8-asym only when no narrower weight type is present. And a card
may state a `ratio` below 1.0 (the 24B is `ratio: 0.8`) meaning a *mixed*
int4/int8 graph carrying `u4`, `u8` and `f16` together; the narrowest-type rule
still yields the right NPU answer, but it is doing real work there.

**Settled since:** an asymmetric export does *not* degrade on the NPU. That
premise, inherited from the export notes rather than measured, was wrong in
both directions. See "Which precisions the NPU actually runs" below. The
classification rules above still hold; only the conclusion drawn from them
was mistaken.

## Device support

*Measured.* `ov.Core().available_devices` reports `['CPU', 'GPU', 'NPU']`, and
`LFM2.5-350M` int4-sym-cw runs on all three; greedy, 16-token prompt, 100 new
tokens:

| Device | Compile | TTFT | TPOT | Throughput |
|---|---|---|---|---|
| NPU | 17.5s | 96.6ms | 6.22ms | 160.7 tok/s |
| GPU (Arc 140V) | 6.0s | 43.0ms | 8.56ms | 116.9 tok/s |
| CPU | 0.8s | 33.0ms | 6.70ms | 149.4 tok/s |

Indicative only. The prompt is far too short to say anything about prefill, and
see the NPU stability note below before treating any NPU figure as a baseline.

Per OpenVINO's support matrix (**not** measured here): only the **dense** LFM2
models have an NPU path; the MoE models `LFM2-8B-A1B` and `LFM2-24B-A2B` are
CPU/GPU only.

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

Each prefill size therefore needs its own compiled pipeline. A prefill ladder
reaching 8k is impractical on NPU, which is a plan-authoring constraint rather
than something a backend can work around.

**One pipeline per process.** Compiling three pipelines with different
`MAX_PROMPT_LEN` in a single process took the device down:

```
ZE_RESULT_ERROR_DEVICE_LOST — device hung, reset, was removed, or driver update occurred
```

It recovered without intervention, but throughput came back degraded and
climbed over subsequent runs: 91.5 → 121.0 → 133.7 tok/s against a 160.7
baseline. NPU cells need a fresh process each, repeated reps, and a settle
period before the numbers mean anything.

## Runtime environment

Inference needs `openvino-genai` only: no `transformers`, `optimum-intel`, or
`nncf`, which are export-side. Keeping the runtime env minimal avoids the
pinned-export-stack version risk entirely. Match the runtime `openvino` version
to whatever produced the IR.

```
uv venv --python 3.11 <venv>
uv pip install --python <venv> openvino-genai==2026.2.1
```

## How the backend answers these

- **One driver process per rep.** `pipette-openvino` runs a one-shot Python
  driver rather than a server, so no process ever compiles twice. Confirmed on
  NPU: five sequential compiles, no device loss, throughput flat at
  153.8-154.0 tok/s where in-process recompilation had degraded it to 91.5.
- **Metrics come from the driver's counters, not the process wall clock**,
  which contains the compile; ~1s on CPU against ~18s on NPU, both comparable
  to the workload itself.
- **Raw seed tokens, not a chat-templated prompt.** Handing `generate` a string
  makes OpenVINO apply the model's chat template (9 tokens for LFM2); prefill
  and decode are raw-continuation measurements in every pipette backend, so the
  driver passes `TokenizedInputs`. Measured 512/100 decode on CPU: 782.78 ms
  raw against 794.78 ms templated.
- **`MIN_RESPONSE_LEN` is raised automatically** when a cell generates more than
  GenAI's 128 default, which `end_to_end_latency_512_256` does. `MAX_PROMPT_LEN`
  is deliberately left at 1024 and a cell over that bound is rejected rather
  than the bound raised.
- **Nothing is rejected at preflight.** Two rules used to live here: no
  asymmetric weights on NPU, and no mixture-of-experts on NPU. Both were
  support-matrix beliefs compiled into the harness; the first was measured and
  found wrong, the second was never measured at all. A runtime that cannot run
  a pairing says so itself.

Still missing: authorable `RuntimeFlags` variants, so a plan cannot yet override
`MAX_PROMPT_LEN` / `MIN_RESPONSE_LEN` / `GENERATE_HINT`. The values above are
derived from the cell rather than authored.

## Measured through pipette

`LFM2.5-350M` int4-sym-cw, 5 reps per cell, `decode_throughput_512_100` and
`end_to_end_latency_512_256`:

| Cell | CPU | NPU |
|---|---|---|
| `prefill_throughput_512` | — | 173.12 +/- 14.41 ms |
| `decode_throughput_512_100` | 782.78 +/- 3.74 ms | 660.37 +/- 24.61 ms |
| `end_to_end_latency_512_256` | — | 1841.44 +/- 50.37 ms |
| `max_memory_usage_512` | — | 923,475,968 B |

NPU wins decode (154 vs 127 tok/s) and pays ~17s of compile per rep for it.

## Which precisions the NPU actually runs

Measured on the devcloud Core Ultra 7 258V across the full LFM2.5-350M
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

Then speed, 512 prefill / 100 decode, best of three, mirroring the driver:

| precision | NPU prefill | NPU decode | decode tok/s |
|---|---|---|---|
| `int4-sym-cw` | 88.7 ms | 1133 ms | 94.8 |
| `int8-sym` | 89.3 ms | 1262 ms | 84.4 |
| `int8-asym` | 176.1 ms | 9749 ms | **10.3** |

**Asymmetric int8 works on the NPU but is 8× slower than its peers**:
evidently no optimised kernel for that layout. That is a result worth having,
and an earlier preflight guard was suppressing it by refusing the cell
outright.

`int8-sym`'s timing sits in band with `int4-sym` rather than looking
anomalously fast, so its numbers are not obviously the product of skipped work,
but its output is wrong, so the row must not be read as a usable
configuration.

### Why there are no preflight guards

There was one. It refused asymmetric weights on the NPU on the theory that they
"load and generate badly rather than failing". Measured, two of its five
verdicts were wrong: asymmetric int8 runs correctly, asymmetric int4 refuses on
its own, and the precision that silently misbehaves is a *symmetric* one.

It is gone rather than corrected, because:

- A failure that throws needs no preflight: `int4-asym` already reports a
  clear compiler error, and a guard only swaps one clear message for another.
- Refusing a cell suppresses findings. The 10.3 tok/s row above is exactly what
  the campaign exists to measure, and the guard hid it.
- Detecting the one silent case cheaply does not work. Generated-token count
  fails (the broken pair runs to full length while a healthy one stops at 4);
  reply length works only under one prompt formatting. The one reliable signal
  is running the same prompt on CPU and comparing, which costs a second
  compile. Hardcoding the answer instead is a fact about one model on one
  chassis, which is how the original guard went wrong.

Which precision/device pairs are worth measuring belongs to the plan, authored
per campaign. The IR precision is still classified and logged per cell, so a
result records what actually ran rather than what a directory name claimed.

**Scope.** One model, one chassis. LFM2.5-350M is the only export carrying both
an int8-sym and an int8-asym variant, so the int8 rows could not be replicated
at another size.

**Not a truncation bug.** With `ignore_eos` at its default the incoherent NPU
runs also stopped after one or two tokens, which looked like it would skew a
throughput measurement. It would not: the driver sets `ignore_eos = True` in
both prefill and decode, and under that setting every variant returns the full
requested count. Timings already collected are unaffected.

**The mixture-of-experts rule is gone too, and measured.**
`LFM2.5-8B-A1B/int8-asym` (`model_type: lfm2_moe`, 32 experts) refuses at
compile on the NPU:

```
[vpux-compiler] InitialLowPrecisionTransformationsPipelineRewriterExecutor
                Pass failed : Got illegal group-wise pattern!
RuntimeError: ... src\plugins\intel_npu\src\plugin ...
```

and generates correctly on CPU. That is a loud failure, so the preflight was
adding nothing but an earlier error message.

`int8-asym` is the right probe here because it is *known good* on the NPU for
the dense LFM2.5-350M, which leaves the MoE architecture as the variable.
every MoE export we have is asymmetric, so this is the only clean isolation
available. Note the compiler blames a group-wise weight pattern rather than the
expert routing, so the precise mechanism is not established; what is
established is that the pairing fails, and fails audibly.

## Open questions

- At `MAX_PROMPT_LEN` 2048 and 4096 generation returned a single token where
  1024 returned the requested 8. Could be EOS on the synthetic prompt used, or a
  real cliff at larger static shapes. **Unresolved**: settle it before
  measuring decode on NPU.
- These figures are Lunar Lake. Intel's published LFM2 perf sheets are Panther
  Lake, a different NPU generation, so the two are not comparable row-for-row.
- Order matters when reading these. A 4-bit graph carries `u8` zero-points
  beside its weights (22 in our `int4-sym-cw`, 62 in the published asymmetric
  24B), so `u8` identifies asymmetric int8 only when no narrower weight type is
  present. Reading widest-first calls every int4 export int8.
