# Decode Throughput Methodology

## Measured Quantity

Decode throughput is the speed of autoregressive token generation after a
prompt context already exists. The `decode_throughput` benchmark times the
decode phase for a fixed number of generated tokens at a fixed prompt depth.

The submitted result is an elapsed time, not a token rate:

- `decode_time_ms`: mean time to generate the requested decode tokens.
- `decode_time_ms_stddev`: sample standard deviation across the five measured
  repetitions.

Lower is better. Pipette stores the time because the benchmark shape already
carries the decode-token count, and elapsed time keeps the payload aligned with
the latency benchmarks. The management service derives the token rate; the
formula is in
[pipette-mgmt docs/benchmarks.md §2.2](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#22-decode-throughput).

Runtime installation, model download, model load, server startup, warm-up,
readiness waits, prompt-context setup and result upload are outside the decode
timing. The scope is model decode work, not TTFT, end-to-end request latency or
streaming API latency.

## Measurement Challenges and Controls

The controls every timing cell shares (a fresh runtime state per repetition,
five repetitions, the readiness gate before each) are written once in
[prefill-throughput.md](prefill-throughput.md#measurement-challenges-and-controls).
The readiness criteria per platform are in
[end-to-end-latency.md](end-to-end-latency.md#system-readiness-control). Decode
adds three controls of its own.

Decode cost depends on both the generated-token count and the existing context
length: 100 tokens after an 8-token prompt is different work from 100 tokens
after a 512-token prompt. Prompt depth and decode count are both part of the
comparison key, and each runtime establishes the prompt context outside the
timed window, then times only the generated tokens.

Models can stop early on EOS or another end-of-generation token, which would
shorten the workload and inflate the apparent rate. Each path therefore forces
the count (a benchmark-native count, an ignore-EOS request, or a fixed-step
loop) and fails when the runtime reports a different decode-token count.

On a server the first token is sampled off the prefill pass, so the server
paths time the interval after the first token and scale it to the full count
(see [Benchmark Shape](#benchmark-shape-and-reported-fields)).

## Measurement Protocol

1. Resolve `parameter_prefill_tokens` and `parameter_decode_tokens`.
2. Prepare the runtime and model.
3. Run the backend's warm-up outside the submitted measurement.
4. For each measured repetition, wait on the readiness gate, then establish the
   requested prompt context outside the timed window.
5. Time generation of the requested decode-token count.
6. Validate the decode token count when the runtime reports it.
7. Report the mean and sample standard deviation of the five decode times.

The timed window starts immediately before the backend's decode operation and
ends when the requested decode completes. llama.cpp uses `llama-bench` depth to
set up the context before its timed block. MLX, torch-oai, OpenVINO and ONNX
GenAI stream a `<D>`-token generation from the server's `/v1/completions` (OVMS
from its `/v3/completions`) and report the after-first-token wall time scaled
to `<D>`, as Apple Foundation does with its response stream. The in-process
iOS llama.cpp, iOS MLX and Android engines time the decode call after an
untimed prefill. iOS ONNX GenAI reports its mean per-token interval after the
first token, times `<D>`. Core AI reports the sidecar's first-token-to-last-token
span without the scaling to `<D>` (see [Core AI](#core-ai)).

## Benchmark Shape and Reported Fields

- `parameter_prefill_tokens`: prompt depth before decode timing starts.
- `parameter_decode_tokens`: number of generated tokens to time.
- `parameter_batch_size`: requests kept in flight at once. Optional, default
  `1`; see [Batched](#batched).

The standard ladder uses IDs such as `decode_throughput_512_100`, 100 generated
tokens after a 512-token prompt; a batched entry carries the count as a
`_bs<n>` suffix, `decode_throughput_512_100_bs4`. The local smoke benchmark,
`decode_throughput_smoke`, uses 8 prompt tokens and 8 generated tokens. The
server catalog also has a depth-zero entry, `decode_throughput_0_32`, which
generates from an empty cache. Only `llama-bench` and the in-process engines
can run it: the server paths need `parameter_prefill_tokens >= 1`, because a
request carries at least one prompt token.

The server paths also require `parameter_decode_tokens >= 2`. There the first
token is sampled off the prefill pass, so `<D>` generated tokens hold `<D> - 1`
decode steps and a one-token cell holds none: its only reading is TTFT, which
is the prefill benchmark under another name. Those paths time the `<D> - 1`
steps after the first token and scale to `<D>`, the same quantity
`llama-bench` and the iOS llama.cpp engine time directly: `<D>` single-token
steps after a depth pass that samples nothing.

## Batched

At `parameter_batch_size` `N` above `1` the same request is measured under
load, each request on its own slot of the one server. The repetition's value is
the per-request mean over the `N` readings, not the batch's wall clock, so
`decode_time_ms` stays one stream's decode time. On llama.cpp,
`llama-batched-bench` generates on the `N` sequences in lockstep, so the one
time it reports is each stream's (see [llama.cpp](#llamacpp)).

Server sizing, gating, warm-up, validation and which engines refuse a batched
entry are shared by the three timing cells:
[end-to-end-latency.md](end-to-end-latency.md#batched).

## Token Control

- llama.cpp: `llama-bench --n-prompt 0 --n-gen <D> --n-depth <P>`, and for a
  batch of `N`, `llama-batched-bench -npp <P> -ntg <D> -npl <N>,<N>`. Both
  generate exactly the requested tokens regardless of EOS.
- MLX: a prompt string the server tokenizes to exactly `<P>` tokens, EOS ids
  suppressed via `logit_bias`.
- torch-oai, OVMS, OpenVINO and ONNX GenAI: the same kind of prompt string,
  with `ignore_eos: true`.
- iOS llama.cpp: the shared seed passage (`PromptSeed.promptSeedText`, the
  bundled copy of `crates/pipette-ops/src/prompt_seed.txt`) tiled to exactly
  `<P>` tokens, then `<D>` tokens fed through the decoder with no sampler, as
  `llama-bench`'s `test_gen` does. The context window is
  `parameter_prefill_tokens + parameter_decode_tokens`, from
  `LlamaRuntimeFlags.ctxSize(for:)`.
- iOS MLX: `<P>` synthetic token ids prefilled, then a fixed loop of `<D>`
  greedy steps.
- iOS ONNX GenAI: minimum and maximum total length both `<P> + <D>`.
- Android app: the seed passage tiled to `<P>` tokens, then `<D>` greedy tokens
  with end-of-generation ignored.
- Core AI: `<P>` deterministic pseudo-random token ids, with no EOS id among
  them, then a greedy generation capped at `<D>` tokens. The sidecar reports
  the count that it generated, and a count other than `<D>` fails the run.

A run is invalid if `llama-bench` prints anything but one row or its row has
the wrong `n_gen` or `n_depth`, if `llama-batched-bench` prints anything but
its two rows or its reading's `pl × pp` and `pl × tg` are not `N × P` and
`N × D`, if a server reports the wrong counts in the stream's trailing `usage`
block or a `finish_reason` other than `"length"`, if an in-process engine
decodes a different count, or if the measured time is missing, non-finite or
non-positive.

## Runtime Details

| Path | Workload | Timing source | Notes |
| --- | --- | --- | --- |
| llama.cpp CLI | `llama-bench --output json --model <path> --n-prompt 0 --n-gen <D> --n-depth <P> --repetitions 1` per measured repetition, after one untimed run of the same command | row `avg_ns`, converted to milliseconds | The depth run establishes context before the timed block. The untimed run keeps a shape's first kernel builds out of rep 1. |
| llama.cpp CLI, batch of `N` | `llama-batched-bench --output-format jsonl --model <path> --ctx-size <N × (P + D)> -npp <P> -ntg <D> -npl <N>,<N>` per measured repetition | second row's `t_tg`, converted to milliseconds | The first row warms the process at the cell's shape; the prompts are prefilled untimed. |
| MLX CLI | streamed `/v1/completions`, `<P>`-token prompt, `max_tokens = <D>` | first-token-to-last-token wall time, scaled to `<D>` | Warm-up at the cell's shape; server prompt cache disabled. |
| torch-oai (vLLM / SGLang, docker or uv) | streamed `/v1/completions`, `<P>`-token prompt, `max_tokens = <D>`, `ignore_eos` | first-token-to-last-token wall time, scaled to `<D>` | Warm-up at the cell's shape; prefix caching disabled at launch. |
| OVMS | streamed `/v3/completions`, `<P>`-token prompt, `<D>` tokens, `ignore_eos` | first chunk to finish chunk, scaled to `<D>` | See [ovms.md](../ovms.md#readings). |
| OpenVINO | in-house server, streamed `/v1/completions`, `<P>`-token prompt, `max_tokens = <D>`, `ignore_eos` | first-token-to-last-token wall time, scaled to `<D>` | Pipeline compiled before the server listens; warm-up at the cell's shape; SDPA backend off NPU. |
| ONNX GenAI | in-house server, streamed `/v1/completions`, `<P>`-token prompt, `max_tokens = <D>`, `ignore_eos` | first-token-to-last-token wall time, scaled to `<D>` | Model loaded before the server listens; warm-up at the cell's shape. |
| iOS llama.cpp | in-process prefill, then `<D>` sampler-free decode steps | Swift-side elapsed time around the decode only | Context and sampler reset per repetition; one untimed warm-up. |
| iOS MLX | in-process prefill into a fresh KV cache, then `<D>` greedy steps | Swift-side elapsed time around the decode only | One untimed warm-up. |
| iOS ONNX GenAI | in-process generation of `<P> + <D>` total tokens | mean per-token interval after the first token, times `<D>` | One untimed warm-up. |
| Apple Foundation (iOS app) | streamed greedy `LanguageModelSession` response to a `<P>`-token prompt ending in a counting seed, `maximumResponseTokens = <D>` | first-snapshot-to-last-snapshot wall time, scaled to `<D>` | One-token warm-up, then a fresh session per repetition; needs `<D> >= 2`, as the server paths do. |
| Android app | in-process llama.cpp prefill, then `<D>` greedy tokens ignoring end-of-generation | elapsed time around the decode only | Context reset and re-prefilled untimed per repetition; one untimed warm-up; refuses a batched entry. |
| Core AI (macOS CLI) | Swift sidecar, `POST /decode_throughput` with `<P>` token ids and `<D>` tokens | sidecar's first-token-to-last-token span, **not** scaled to `<D>` | One untimed warm-up; KV reset untimed per repetition; needs `<D> >= 2`; does not refuse a batched entry. See [Core AI](#core-ai). |

### OpenVINO and ONNX GenAI

Both runtimes are measured through pipette's in-house OpenAI-compatible
server, one per cell, with the same request, reading and checks as
[torch-oai](#torch-oai). The server streams one chunk per generated token, text
or not, flushed as the token exists, so the first and last stamps fall on those
tokens. OpenVINO generates through a token-level streamer and, off NPU, on
GenAI's SDPA backend, because detokenizing each token on the default
paged-attention backend stalls generation. ONNX GenAI forces the count with
equal minimum and maximum total lengths, both `<P> + <D>`. The measurements
behind these choices are in
[OpenVINO measurement](../openvino-measurement.md) and
[ONNX GenAI measurement](../onnx-genai-measurement.md).

### llama.cpp

`pipette-llamacpp` runs one `llama-bench` process per measured sample with
`--repetitions 1`. Before the five it runs the same command once more,
ungated; its row is checked and dropped. The run must print exactly one JSON
row (`n_prompt == 0`), with `n_gen == parameter_decode_tokens` and
`n_depth == parameter_prefill_tokens`. `n_depth` runs a prompt pass before the
timed block (none at depth zero), the timed block generates `n_gen` tokens, and
the row's `avg_ns` is converted to milliseconds.

The extra run is the cell's warm-up. `llama-bench` warms each process itself
(`--no-warmup` is reserved), but for a decode that warm-up is one generated
token on an empty cache, so a kernel that only the cell's depth selects is
built inside the timed generation. For example, on Metal the attention kernel is
tuned per KV length from 1,024 cells up. A shape's first run on a machine
builds it from scratch and later runs take it from the system's shader cache,
so without the extra run, rep 1 alone pays the cold build. Measured with
`llama-bench` b11228 on an M3 Ultra (macOS 27.0.1), depth 4,096, 128 tokens:
six one-repetition runs in a row of a shape that build had not run there
before, reduced without the extra run (runs 1 to 5) and with it (runs 2 to 6):

| Model, KV cache | Run 1 minus the median of the rest | Runs 1 to 5 | Runs 2 to 6 |
| --- | --- | --- | --- |
| gemma-4-E4B Q4_0, `q8_0` | +356 ms | 1,638.8 ± 158.8 ms | 1,566.7 ± 3.4 ms |
| gemma-3-1b Q4_K_M, `q4_0` | +307 ms | 818.9 ± 137.7 ms | 759.4 ± 5.3 ms |
| LFM2-350M Q4_K_M, `q8_0` | +24 ms | 308.2 ± 20.0 ms | 305.0 ± 16.1 ms |

With the default `f16` cache the first build cost 35 to 58 ms (Qwen3.5-0.8B,
gemma-4-E4B). Run again, the same shapes showed no first-run cost, and neither
did a CPU-only build, which compiles nothing: there the extra run only costs
time.

The cached build still happens inside each sample. Taking it out would need a
second decode in the same process, which reads slower for a reason of its own:
with `--repetitions 2` on the same machine, LFM2-350M's second sample ran about
30% slower than its first.

At `parameter_batch_size` `N` above 1 the runner uses `llama-batched-bench`,
still one process per measured sample:

```
llama-batched-bench --output-format jsonl --model <path> <knobs> \
  --ctx-size <N × (P + D)> -npp <P> -ntg <D> -npl <N>,<N>
```

As for [prefill](prefill-throughput.md#llamacpp), the batch is listed twice:
the first row warms the process at the cell's shape and the second is the
reading. The tool prefills the `N` prompts, untimed for this cell, then runs `D`
decode steps, each taking one token from every sequence; its `t_tg` is
converted to milliseconds. A batch needs a prompt: at
`parameter_prefill_tokens = 0` the cell is refused by name before the tool
runs, since the totals carry a positive prompt count and the tool's rate over
an empty prompt can print `nan`, which is not JSON.

The reserved `llama-bench` and `llama-batched-bench` flags and the default
`--load-mode none` are listed once in
[prefill-throughput.md](prefill-throughput.md#llamacpp).

### MLX

`pipette-mlx` starts `mlx_lm.server` and warms it up as for
[prefill](prefill-throughput.md#mlx), with one warm-up request at the cell's own
shape. Each measured repetition streams:

```json
{ "model": "default_model", "prompt": "<P-token text>", "max_tokens": <D>,
  "temperature": 0.0, "logit_bias": {"<eos>": -100.0},
  "stream": true, "stream_options": {"include_usage": true} }
```

The prompt is built from the benchmark prompt seed and sized with the venv's
tokenizer so the server counts exactly `<P>` tokens. The sample is the
streamed wall time from the first token to the last, scaled from that
`<D> - 1` interval to `<D>`; the trailing `usage` chunk, `[DONE]` and the body
close are outside it. `mlx_lm.server` sends a chunk only for a step with text,
so a first token whose text its detokenizer held back (an incomplete UTF-8
sequence, a lone space) would start the interval a token late and read the
rate high. On the prompt seed it does not happen: probed on eight fleet models
at 16 to 8192 prompt tokens, the first token always carried text.

### torch-oai

`pipette-torch-oai` launches the vLLM or SGLang server with prefix caching off,
as for [prefill](prefill-throughput.md#torch-oai), then sends one warm-up
request at the cell's own shape. Each measured repetition streams:

```json
{ "model": "<model>", "prompt": "<P-token text>", "max_tokens": <D>,
  "temperature": 0.0, "ignore_eos": true,
  "stream": true, "stream_options": {"include_usage": true} }
```

The prompt is sized with the server's own `/tokenize`. Neither engine reports a
per-request generation time on the OpenAI surface, so the sample is the
client-measured wall time from the first streamed token to the last, scaled
from `<D> - 1` to `<D>`. Both servers send a chunk per decode step, text or
not, so the interval starts at the first token even when its text is held
back.

vLLM's `--max-model-len`, when the cell does not pin it, is `<P> + <D>`.
SGLang's context length is left to the operator (`--context-length` in `raw`).

### iOS

The iOS llama.cpp engine loads the GGUF model with a context of `<P> + <D>`,
resets the context and sampler for each repetition, prefills the seed prompt,
then times `<D>` sampler-free decode steps, the same work as `llama-bench`'s
timed block. The iOS app waits on its own device-side readiness gate before
each repetition rather than the CLI's `pipette_readiness` gate (see
[end-to-end-latency.md](end-to-end-latency.md#ios)), so compare iOS results
only against other iOS results.

### Core AI

The Core AI runner (`pipette-coreai`, macOS 27 or later) drives a Swift sidecar
over local HTTP. The sidecar loads the model before it prints its ready marker,
so load and specialization are outside every window
([Core AI specialization cache](coreai-specialization.md)). The runner gates
readiness once before it starts the sidecar, then sends one untimed warm-up
request at the cell's shape and checks its count. Each measured repetition
then runs in this order:

1. The readiness gate.
2. An untimed `POST /prepare`, which resets the KV cache and waits 50 ms.
3. `POST /decode_throughput` with `prompt_tokens = <P>` and
   `decode_tokens = <D>`.

The sidecar builds the prompt from `<P>` deterministic pseudo-random token ids
and generates greedily up to `<D>` tokens. It stamps the first streamed token
and the end of the stream on its own clock. It reports
`generation_tps = (<D> - 1) / span`. The runner converts the rate back to the
span: `decode_time_ms = (<D> - 1) / generation_tps × 1000`.

The Core AI value is therefore the `<D> - 1` steps after the first token,
**not** scaled to `<D>`. Every other server path scales the same interval to
`<D>`. A Core AI row reads lower than a scaled row of the same work by the
factor `(<D> - 1) / <D>`, for example 1% lower at `<D> = 100`. Compare Core AI
decode rows only with other Core AI rows.

Core AI needs `<D> >= 2`. A generation that stops early fails the count check.
The runner does not read `parameter_batch_size`, so a batched entry runs one
request per repetition under the batched ID.

## Result Validity and Comparability

The shared validity and comparability rules are in
[prefill-throughput.md](prefill-throughput.md#result-validity-and-comparability).
For decode, also compare only results with the same prompt depth and
decode-token count, and treat flags that affect the KV cache or sampling as
part of the benchmark condition.

Some stored rows come from client versions that timed decode another way.
They do not compare with rows that the client times on its stream clock:

- OpenVINO and ONNX GenAI rows whose `client_version` is dated before
  2026-09-25. A per-process driver timed them.
- MLX rows dated before 2026-09-19. They read the in-process `generation_tps`
  sidecar.
- MLX rows dated 2026-09-19, and some MLX rows dated 2026-09-20. Their interval
  ran to `[DONE]`, so they carry that tail and read slightly high.

The `client_version` of a row tells the methods apart.

## Code References

The benchmark definition and result fields are in
[`pipette_plan_types::benchmark`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs);
the standard ladder is
[`pipette_cli::benchmarks`](../../crates/pipette-cli/src/benchmarks).

The reps, warm-up, checks and reduction are the typed entry
[`pipette_ops::measurement::decode`](../../crates/pipette-ops/src/measurement/decode.rs),
which every CLI runner calls from its crate's `execute/decode_throughput.rs`.
The Core AI runner
[`pipette-coreai execute::decode_throughput`](../../crates/pipette-coreai/src/execute/decode_throughput.rs)
calls `pipette_ops::measurement::run` instead. The llama.cpp runner is
[`pipette-llamacpp execute::decode_throughput`](../../crates/pipette-llamacpp/src/execute/decode_throughput.rs),
through [`bench`](../../crates/pipette-llamacpp/src/bench.rs) or, for a batch,
[`batched_bench`](../../crates/pipette-llamacpp/src/batched_bench.rs). The MLX
request is in
[`pipette-mlx completions`](../../crates/pipette-mlx/src/completions.rs) and the
stream is read by
[`pipette-completions`](../../crates/pipette-completions/src/lib.rs).

The iOS llama.cpp implementation is
[`LlamaExecute.decodeThroughput`](../../ios/Pipette/Pipette/Runtimes/Llama/Execute/LlamaDecodeThroughput.swift),
through the entry's Swift port, `BenchmarkMeasurement.decode`. The Android app
implementation is `runDecodeThroughputImpl` in
[`LlamaBenchmark.kt`](../../android/Pipette/app/src/kotlinEngine/java/ai/liquid/pipette/llama/LlamaBenchmark.kt)
(`kotlinEngine` flavor) and `run_decode_throughput` in
[`android/native/benchmarks.rs`](../../android/native/benchmarks.rs)
(`rustEngine` flavor).
