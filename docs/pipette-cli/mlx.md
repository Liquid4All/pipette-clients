# MLX runtime

Backend library: `pipette-mlx`, driven by the unified `pipette` CLI.

Runs **Apple Silicon** benchmarks via a uv-managed `mlx-lm` (and, for the vision-language (VL)
cells, `mlx-vlm`) venv and a short-lived local HTTP server. Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

## Differences from llama.cpp

The table that compares MLX with llama.cpp and Core AI is in
[coreai.md](coreai.md#differences-from-llamacpp--mlx).

## Runtime

MLX installs through the same shared runtime store as llama.cpp / UV:

```bash
pipette runtimes catalog mlx_macos_pipette
pipette runtimes pull --runtime 'mlx-macos-pipette://version=0.32.0&flavor=macos-arm64'
```

`flavor` defaults to `macos-arm64`. How a row's `version` key is chosen, and
why a shipped row is never edited in place, is in the catalog's
[README](../../crates/pipette-artifacts-catalog/bundled-catalog/mlx/README.md).

```bash
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'mlx://repo=mlx-community/LFM2-350M-4bit' \
  --runtime 'mlx-macos-pipette://version=0.32.0&flavor=macos-arm64'
```

Requires `uv` on `PATH`. See [usage.md](usage.md#where-the-external-programs-come-from) to point pipette at a copy that is not on `PATH`. The relocatable venv lands under
`.pipette/runtimes/<key>/blobs/venv/`. If the runtime is not in the store,
the first `benchmarks run` installs it. A requirement set outside the catalog
has no URI form: pass it as a JSON `--runtime` with a full
`pip_requirements_text` source.

## Model

```bash
--model 'mlx://repo=mlx-community/LFM2-350M-4bit'
# optional: &prefix=… &rev=…
pipette models pull --model 'mlx://repo=…'   # optional pre-fetch
```

`pipette` materializes the Hugging Face (HF) snapshot into the shared
`models/` store and starts the server with that **local directory**. It does
not read the `~/.cache/huggingface` cache. Gated repos need `PIPETTE_HF_TOKEN`; see
[usage.md](usage.md#gated-models).

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`, `eval`, `vl_throughput`, `vl_max_memory`, `vl_spec_decode`.

Not supported: `vlm_eval`, `asr_eval`, `asr_throughput`, `asr_max_memory`.

Each run starts an upstream server on the materialized model directory, drives
the task over its OpenAI-compatible API, then stops the server. Text cells use
`mlx_lm.server` (prompt cache disabled); the VL cells use `mlx_vlm.server`
(prefix cache disabled with `APC_ENABLED=0`), the runtime's own
OpenAI-compatible server, which reports the speculation counters
`vl_spec_decode` reads under `llama-server`'s field names.
Both servers are unmodified upstream code. Exact-length text prompts and the suppression of the end-of-sequence (EOS)
token use a small one-shot tokenizer probe in the same venv.

The three timing cells run one request at a time here: this client drives
`mlx_lm.server` with one `/v1/completions` per repetition and launches it with
no slot count, so an entry whose `parameter_batch_size` is above 1 is refused
before the server comes up. See
[../methodology/end-to-end-latency.md](../methodology/end-to-end-latency.md#batched).

`vl_throughput` runs a VL model (`mlx`) on a runtime row
that ships `mlx-vlm`. The rows that ship it are `0.7.4`, `0.7.11` and
`0.7.1+lfm2-vl-speculative`; `pipette runtimes catalog mlx_macos_pipette`
lists them. Each repetition sends one streamed chat request with a gray image.
This client's clock times it, because the prefill timer of `mlx_vlm.server`
leaves the image encode out on some model families:

- `prompt_ms`: from the send to the first token.
- `predicted_ms`: from the first token to the last token, scaled to every
  decode step.
- `total_time_ms`: from the send to the chunk that carries `finish_reason`.

`--http-timeout-seconds` bounds the wait for the first token and the wait
between tokens (default 3600 s). It does not bound the whole request;
`--attempt-timeout-seconds` bounds each repetition, 300 s by default
([usage.md](usage.md#5c-the-attempt-limit)). The `--readiness-*` flags apply
as on the text timing cells. See
[../methodology/vl-throughput.md](../methodology/vl-throughput.md#runtime-details).

```bash
pipette benchmarks run --benchmark local/vl_throughput_smoke \
  --model 'mlx://repo=mlx-community/LFM2-VL-450M-8bit' \
  --runtime 'mlx-macos-pipette://version=0.7.4'
```

`vl_max_memory` runs on the same runtime row: one chat request with the cell's
gray images and text and a one-token reply, reporting the server's peak
`phys_footprint` as `max_host_bytes` (`max_gpu_bytes` is null: unified memory).
Like `max_memory_usage` it takes `attempt_timeout_seconds` and no other
benchmark flag, so `--http-timeout-seconds` is refused. Both memory cells first
send their request to a warm-up server of their own, its reading dropped, so
the measured server finds its Metal kernels compiled. Each server is one
attempt from its launch (300 s by default), and its request waits up to
3600 s. See
[../methodology/vl-max-memory.md](../methodology/vl-max-memory.md#mlx-macos)
and [../methodology/peak-memory-macos.md](../methodology/peak-memory-macos.md#mlx).

Every cell but `eval` also records the server's peak `phys_footprint` as an
observation; see
[../methodology/peak-memory-usage.md](../methodology/peak-memory-usage.md#observations-vs-metrics).

```bash
pipette benchmarks run --benchmark local/vl_max_memory_smoke \
  --model 'mlx://repo=mlx-community/LFM2-VL-450M-8bit' \
  --runtime 'mlx-macos-pipette://version=0.7.4'
```

`vl_spec_decode` is the speculative-decoding corpus benchmark: one model, one
server, one traversal, one row per run. DSpark is the drafter family it
measures (for example `LiquidAI/LFM2.5-VL-3B-DSpark`). Run it once with a regular model (`mlx`) and once
with a spec-decoding pair (`mlx_speculative`, a VL target and a drafter as two
separate HF repos) against the same benchmark id. The speedup
between the two rows is computed outside the server, from the stored rows. The runtime flag `speculative_enabled`
decides the condition. Unset, it follows the model: off launches the target
alone, on launches it with `--draft-model` and `--draft-block-size`. It is the
only MLX runtime flag, and only on this benchmark. The benchmark's
`parameter_batch_size` (default 1) is how many samples the traversal keeps in
flight; above 1 the server is launched with `--max-num-seqs` at that count, so
its continuous batch is sized for the benchmark's concurrency. Both runs need a
runtime row that ships `mlx-vlm` (`0.7.4`, `0.7.11` or
`0.7.1+lfm2-vl-speculative`), because `mlx-lm` cannot load a VL bundle. See
[../methodology/vl-spec-decode.md](../methodology/vl-spec-decode.md) and
`examples/plans/mlx-macos-vl-spec-decode.toml`.

```bash
RUNTIME='mlx-macos-pipette://version=0.7.4'

# The regular model: speculation off.
pipette benchmarks run --benchmark vl_spec_decode_mmspec_test_gqa_k8 \
  --model 'mlx://repo=LiquidAI/LFM2.5-VL-3B-MLX-bf16' --runtime "$RUNTIME" \
  --runtime-flags '{"speculative_enabled":false}' \
  --http-timeout-seconds 600 --sync

# The spec-decoding pair: speculation on, same benchmark id and corpus.
pipette benchmarks run --benchmark vl_spec_decode_mmspec_test_gqa_k8 \
  --model 'mlx-speculative://repo=LiquidAI/LFM2.5-VL-3B-MLX-bf16&draft_repo=LiquidAI/LFM2.5-VL-3B-DSpark' \
  --runtime "$RUNTIME" --runtime-flags '{"speculative_enabled":true}' \
  --http-timeout-seconds 600 --sync
```

The MLX path runs upstream `mlx-vlm`'s `dflash` loop over the separate drafter,
with none of the DSpark fork's switches, so its accept length (≈3.6 at `k = 8`)
compares with other MLX rows and not with the llama.cpp release figure.

The corpus comes from the management server, so there is no local form of this
benchmark. The run refuses to start on battery or while charging, since the two
arms must run under the same power state. The first import of `mlx_vlm`
in a venv takes about two minutes, because Python compiles the bytecode of a
large package tree. This time is part of the server start, before the warm-up
of the arm, so the cell does not measure it, though a cell with an attempt
limit counts it against the first attempt
([usage.md](usage.md#5c-the-attempt-limit)).

`--runtime-flags` is **refused** for MLX, not ignored, on every benchmark but
`vl_spec_decode`: the `mlx_macos_pipette` runtime has no other flag cell, so
any non-empty object fails before the runtime is even fetched (an empty `{}`
means "no flags" and is accepted). On `vl_spec_decode` it takes
`speculative_enabled` and nothing else.

```text
no runtime flags defined for <benchmark> × MlxMacosPipette × Mlx
```

`--model-enable-thinking` / `--model-flags` do apply on eval. There
`--http-timeout-seconds` bounds the gap between server-sent events (SSE), not
the whole request, and never goes below 1800 s: a slow first token alone can outlast a
short pin.

## Eval resume

Eval checkpoints each completed sample, and a sample that fails is recorded
as failed rather than ending the run. See
[eval-checkpoint.md](eval-checkpoint.md).
