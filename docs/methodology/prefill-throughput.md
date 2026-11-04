# Prefill Throughput Methodology

## Measured Quantity

Prefill throughput is the speed of processing prompt tokens into the model
state used for generation: the prompt phase that builds the KV cache before
autoregressive decode begins. In pipette it is the `prefill_throughput`
benchmark family.

The submitted result is an elapsed time, not a rate: `prefill_time_ms` (mean
over the measured repetitions) and `prefill_time_ms_stddev` (sample standard
deviation). The benchmark shape already carries the token count, and an
elapsed time keeps the payload aligned with the latency benchmarks. The
management service derives `prefill_throughput` (tokens/s) and `ttft` from
these fields; the formulas are in
[pipette-mgmt docs/benchmarks.md §2.1](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#21-prefill-throughput).

Runtime installation, model download, model load, server startup, warmup,
readiness waits, and result sync are outside the timing. What the window holds
depends on how the backend is read:

- The llama.cpp CLI reads `llama-bench`'s own timing row: the prefill alone,
  with no tokenization or HTTP.
- The in-process app engines (iOS, Android) time the prefill call on token ids
  they built before the window; iOS ONNX GenAI instead reads a one-token
  generation's time to first token, so its sampling is inside.
- Server paths (MLX, torch-oai, OVMS, OpenVINO, ONNX GenAI) read the client's
  time to the first streamed token of a one-token generation. That reading
  includes the HTTP exchange, the server's tokenization of the text prompt and
  the first token's sampling.
- Core AI reads its sidecar's own time from the start of a one-token generation
  over token ids to the first token. That window holds the first token's
  sampling, but no HTTP and no tokenization (see [Core AI](#core-ai)).

So a llama.cpp CLI row, a server row and a Core AI row of the same shape do
not measure the same window, even though all three are filed as
`prefill_time_ms`.

## Measurement Challenges and Controls

Prefill cost scales with prompt-token count, batch sizing, attention
implementation, KV-cache format, quantization, and accelerator placement. A
result is comparable only for the same benchmark shape, model, runtime, and
relevant runtime flags.

Text prompts are not required for this benchmark. Runtimes can use token IDs or
a benchmark-native token-count control when that path most directly represents
the requested work.

Warmup and thermal state can dominate prompt-processing timings, especially on
small devices. The runners separate warmup from measured samples and run the
platform readiness gate before each measured repetition, so a single long
prefill cannot leave residual heat that silently changes the next sample.

Cached state can also distort prompt-processing timings. A second call into a
runtime that retained KV state from the previous call would measure a different
workload from a cold prefill. Each measured repetition therefore starts from a
fresh runtime state:

- the llama.cpp CLI runs one `llama-bench` process per sample
  (`llama-batched-bench` for a batch);
- the iOS and Android llama.cpp engines reset the context and sampler before
  every iteration, and iOS MLX builds a new cache per iteration;
- the MLX server runs with `--prompt-cache-size 0`;
- the torch-oai servers launch with prefix caching off
  (`--no-enable-prefix-caching` on vLLM, `--disable-radix-cache` on SGLang),
  and a cell asking for it on is refused;
- OpenVINO pins GenAI's SDPA backend off NPU, which keeps no state across
  requests ([OpenVINO measurement](../openvino-measurement.md));
- ONNX GenAI builds a fresh generator, and so a fresh KV cache, per request.

Individual samples can still be affected by scheduler noise, runtime caches,
or transient background load. The benchmark measures five repetitions and
reports both mean and sample standard deviation.

## Measurement Protocol

1. Resolve `parameter_prefill_tokens`.
2. Prepare the runtime and model.
3. Warm up at the cell's own shape, outside the submitted measurement: one
   untimed attempt before the reps, or on the llama.cpp CLI, inside each
   sample's own process.
4. Run five measured prefill repetitions, each behind the readiness gate.
5. Validate the prefill token count when the runtime reports it.
6. Report the mean and sample standard deviation of the five measured times.

Token construction, runtime startup, model load, warmup, readiness checks,
result serialization, and sync are outside the timed window.

## Benchmark Shape and Reported Fields

- `parameter_prefill_tokens`: number of prompt tokens to prefill.
- `parameter_batch_size`: requests kept in flight at once. Optional, default
  `1`; see [Batched](#batched).

The standard ladder uses IDs such as `prefill_throughput_512`, meaning a
512-token prefill; a batched entry carries the count as a `_bs<n>` suffix,
`prefill_throughput_512_bs4`. The local smoke benchmark
(`prefill_throughput_smoke`) uses 8 prompt tokens.

Lower `prefill_time_ms` is better.

## Batched

Above `parameter_batch_size = 1` the same request is measured under load and
the value is the per-request mean, not the batch's wall clock; the rules, and
which engines refuse a batched entry, are in
[end-to-end-latency.md](end-to-end-latency.md#batched). llama.cpp has no
per-request reading to average; see [llama.cpp](#llamacpp).

## Token Control

The prefill-token count defines the work:

- llama.cpp uses `llama-bench --n-prompt <P> --n-gen 0`, and for a batch of
  `N`, `llama-batched-bench -npp <P> -ntg 1 -npl <N>,<N>`.
- MLX builds a prompt string the server tokenizes to exactly `<P>` tokens
  (counted with the venv's tokenizer) and confirms the count against the
  stream's trailing `usage` block, requiring `finish_reason = "length"`.
- torch-oai builds the same prompt string, counted with the server's own
  `/tokenize`, and confirms the count against the trailing `usage` block.
- OpenVINO and ONNX GenAI do the same against pipette's in-house server, whose
  `/tokenize` applies the encoding its `/v1/completions` feeds the model.
- The iOS llama.cpp engine tiles `PromptSeed.promptSeedText` (the bundled copy
  of `crates/pipette-ops/src/prompt_seed.txt`), tokenizes it and truncates to
  exactly `parameter_prefill_tokens` ids (`LlamaExecute.seedTokens`). iOS ONNX
  GenAI sizes a seed prompt with `PromptSeed.buildPromptText`. iOS MLX feeds
  synthetic in-vocab ids (`PromptSeed.syntheticTokenIds`).

The context window for iOS llama.cpp is `parameter_prefill_tokens`, derived by
`LlamaRuntimeFlags.ctxSize(for:)`, so the measured prefill is never truncated by
a context sized for something else.

A run is invalid if `llama-bench` prints anything but one row or its row has the
wrong `n_prompt`, if `llama-batched-bench` prints anything but its two rows or
its reading's `pl × pp` is not `N × P`, if a server reports the wrong
`prompt_tokens` or a `completion_tokens` other than 1 in `usage` or a
`finish_reason` other than `"length"`, or if the measured time is missing,
non-finite, or non-positive.

## System Readiness Control

The CLI runners wait on `pipette_readiness` before each measured repetition;
unsupported platforms use its no-op fallback. The iOS and Android apps run
their own device-side gate before each repetition. The per-platform criteria
are in
[end-to-end-latency.md](end-to-end-latency.md#system-readiness-control).

Readiness checks are outside the timed window. If the host or device does not
reach readiness before the platform deadline, the run fails rather than
submitting a timing from an unstable condition.

## Runtime Details

| Path | Workload | Timing source | Notes |
| --- | --- | --- | --- |
| llama.cpp CLI | `llama-bench --output json --model <path> --n-prompt <P> --n-gen 0 --repetitions 1` per measured repetition | row `avg_ns`, converted to milliseconds | Five separate one-repetition invocations so readiness can run between samples. |
| llama.cpp CLI, batch of `N` | `llama-batched-bench --output-format jsonl --model <path> --ctx-size <N × (P + 1)> -npp <P> -ntg 1 -npl <N>,<N>` per measured repetition | second row's `t_pp`, converted to milliseconds | The first row warms the process at the cell's shape. |
| MLX CLI | streamed `/v1/completions` with a `<P>`-token prompt, `max_tokens = 1` | client time to first token | Server prompt cache disabled. |
| torch-oai (vLLM / SGLang, docker or uv) | streamed `/v1/completions` with a `<P>`-token prompt, `max_tokens = 1`, `ignore_eos` | client time to first token | Prefix caching disabled at launch. |
| OVMS | streamed one-token request | client time from send to first chunk | See [ovms.md](../ovms.md#readings). |
| OpenVINO | in-house server, streamed `/v1/completions` with a `<P>`-token prompt, `max_tokens = 1`, `ignore_eos` | client time to first token | Pipeline compiled before the server listens; SDPA backend off NPU. |
| ONNX GenAI | in-house server, same request | client time to first token | Model loaded before the server listens. |
| iOS llama.cpp | `inf.prefill(tokens)` after context and sampler reset | `BenchmarkMeasurement.timed` around the call | One warmup plus five measured repetitions in the app process. |
| iOS MLX | `MlxRuntime.prefill` of synthetic ids into a new cache | `BenchmarkMeasurement.timed` around the call | Model loaded fresh for the cell. |
| iOS ONNX GenAI | one-token generation over the seed prompt ids | the generation's `ttftMs` | Includes the first token's sampling. |
| Apple Foundation | not run | none | The streaming API exposes only TTFT, so AFM refuses the cell (`AFMRuntime.swift`). |
| Android app | llama.cpp prefill of seed prompt ids after context and sampler reset (Rust and Kotlin engines alike) | elapsed time around the call | One warmup plus five measured repetitions. |
| Core AI (macOS CLI) | Swift sidecar, `POST /prefill_throughput`: a one-token generation over `<P>` token ids | `<P> / prompt_tps`, from the sidecar's clock | One untimed warm-up; KV reset untimed per repetition. See [Core AI](#core-ai). |

### OpenVINO and ONNX GenAI

Both runtimes are measured the way MLX and torch-oai are: one in-house
OpenAI-compatible server per cell (`pipette_openvino_server.py`,
`pipette_onnx_genai_server.py`), a streamed `/v1/completions` per repetition,
and the time to the first streamed token on the client's clock. The server
loads the model before it listens, so the load (and OpenVINO's pipeline
compile) is startup time, outside every request. It streams one chunk per token
as soon as the token exists.

Both servers speak HTTP/1.1 and keep a connection open between requests, so the
repetitions reuse the client's connection. A new connection inside each reading
costs about 18 ms on Windows: timing an OpenVINO prefill of 512 tokens on an
integrated GPU on both sides, the client read a median of 18.5 ms more than the
server's own time on HTTP/1.0 and 0.9 ms more on HTTP/1.1, over 60 repetitions
each. On Linux and macOS a new connection is 0.1 to 0.25 ms and readings stayed
within 1%.

Backend choice, token handling and the measured cache effects are in
[OpenVINO measurement](../openvino-measurement.md) and
[ONNX GenAI measurement](../onnx-genai-measurement.md).

### llama.cpp

`pipette-llamacpp` runs one `llama-bench` process per measured sample and
passes `--repetitions 1`. Each process first warms itself with one untimed pass
over the cell's own prompt (`--no-warmup` is reserved), which is not part of
the row's `avg_ns`. The run must print exactly one JSON row, the one test its
arguments ask for (`n_gen == 0`), with `n_prompt == parameter_prefill_tokens`.
The row's `avg_ns` is converted to milliseconds for that sample.

At `parameter_batch_size` `N` above 1 the runner uses `llama-batched-bench`
instead, still one process per measured sample:

```
llama-batched-bench --output-format jsonl --model <path> <knobs> \
  --ctx-size <N × (P + 1)> -npp <P> -ntg 1 -npl <N>,<N>
```

The tool prints a row per `-npl` value, and the batch is listed twice. The
first row warms the process at the cell's own shape, which the tool's own
16-token warm-up is not; the second is the reading. Its `t_pp` is the seconds
from submitting the `N` prompts to the last of them processed, converted to
milliseconds. One token is generated after the prompts because with none the
tool's generation rate can divide zero by zero and print `nan`, which is not
JSON; the prefill time stops before it. `--ctx-size` holds the batch exactly,
since the tool skips a row whose context does not fit without saying so. The
run must print exactly those two rows, and the typed entry checks
`pl × pp == N × parameter_prefill_tokens`. The tool parses llama.cpp's common
arguments, so the typed knobs render as they do for `llama-server`.

The batch has one time, not a per-request mean. When the `N × P` tokens fit one
logical batch (`--batch-size`, 2048 by default) every prompt finishes with the
batch, so that time is each prompt's. Above it the tool processes the batch in
chunks, prompts in the first chunk finish earlier, and the value is the slowest
prompt's rather than the mean.

The plan refuses user overrides for the workload-shaping flags:
`--output`, `-o`, `--model`, `-m`, `--n-prompt`, `-p`, `--n-gen`, `-n`,
`--n-depth`, `-d`, `--repetitions`, `-r` and `--no-warmup`, and for
`llama-batched-bench` `--output-format`, `--ctx-size`, `-c`, `-npp`, `-ntg`,
`-npl`, `-pps` (one prompt shared across the batch) and `-tgs` (the sequences
generated one after another). The list is `LlamacppCliStockTools` in
[pipette-mgmt crates/pipette-plan-types/src/cell/reserved_flags.rs](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/reserved_flags.rs).

Both tools get the weights pinned in anonymous RAM (`--load-mode none`) unless
the cell sets `load_mode` or the deprecated `mmap`. User runtime flags that are
not reserved remain part of the runtime condition and must be kept identical
across compared runs.

### MLX

`pipette-mlx` starts a local `mlx_lm.server` (prompt cache disabled), then
sends one warmup request to `/v1/completions` at the benchmark's own
`prompt_tokens`, so kernel selection and the lazy model load are paid before
the first measured rep. Each measured repetition waits for readiness, then
streams a one-token generation:

```json
{ "model": "default_model", "prompt": "<P-token text>", "max_tokens": 1,
  "temperature": 0.0, "logit_bias": {"<eos>": -100.0},
  "stream": true, "stream_options": {"include_usage": true} }
```

The prompt text is built from the benchmark prompt seed and sized with the
venv's tokenizer so the server counts exactly `<P>` tokens; EOS ids are
suppressed via `logit_bias` for symmetry with the other benchmarks (with
`max_tokens = 1` the call returns after one generated token regardless of EOS).
The recorded sample is the client time to the first streamed token. The
trailing `usage` block must confirm `prompt_tokens = <P>` and
`completion_tokens = 1`, and the terminal chunk must report
`finish_reason = "length"`.

The MLX runtime rows differ in one per-request cost. A row key names the
version of the package that the row moved, so the key is not always the
mlx-lm version (see the
[mlx catalog README](../../crates/pipette-artifacts-catalog/bundled-catalog/mlx/README.md)).

- Rows `0.31.3`, `0.32.2` and `0.7.1+lfm2-vl-speculative` pin mlx-lm 0.31.3.
  That server builds its streaming detokenizer for every request. For a
  byte-level BPE vocabulary, this adds tens of milliseconds to the time to
  first token, and so to prefill and end-to-end cells: 43 ms for the 128K
  vocabulary of Llama 3.2 1B on an M3 Ultra. LFM2 and LFM2.5 use the naive
  detokenizer and pay nothing.
- Rows `0.32.0` and `0.7.4` (mlx-lm 0.32.0) and rows `0.32.4` and `0.7.11`
  (mlx-lm 0.32.2) build the detokenizer once per load
  (ml-explore/mlx-lm#1879).

### torch-oai

`pipette-torch-oai` launches the vLLM or SGLang server (Docker or uv) with
prefix caching off, waits for its readiness URL, then sends one warmup request
to `/v1/completions` at the benchmark's own `prompt_tokens`, so kernel
selection and any lazy compilation are paid before the first measured rep.
Each measured repetition waits for readiness, then streams a one-token
generation:

```json
{ "model": "<model>", "prompt": "<P-token text>", "max_tokens": 1,
  "temperature": 0.0, "ignore_eos": true,
  "stream": true, "stream_options": {"include_usage": true} }
```

The prompt text is built from the benchmark prompt seed and sized with the
server's own `/tokenize` (`add_special_tokens: true`, the same accounting
`/v1/completions` applies to a string prompt) so the server counts exactly
`<P>` tokens. Neither engine reports a per-request prefill time on the OpenAI
surface, so the recorded sample is the client time to the first streamed token.
Both servers send a chunk per decode step, so every chunk with a choice is a
token: one sent empty while the detokenizer holds back an incomplete UTF-8
sequence still stamps its token. The `usage` and `finish_reason` checks are the
MLX path's.

vLLM's `--max-model-len`, when the cell does not pin it, is derived as
`<P> + 1` so the prompt is never truncated by a context sized for something
else. SGLang's context length is left to the operator (`--context-length` in
`raw`).

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
3. `POST /prefill_throughput` with `prompt_tokens = <P>`.

The sidecar builds the prompt from `<P>` deterministic pseudo-random token ids,
with no EOS id among them, and generates one token. On its own clock it times
the span from the start of the generation to the first token, and it reports
`prompt_tps = <P> / span`. The runner converts the rate back:
`prefill_time_ms = <P> / prompt_tps × 1000`. The sidecar reports the prompt
count, and a count other than `<P>` fails the run. The runner does not read
`parameter_batch_size`, so a batched entry runs one request per repetition
under the batched ID.

## Result Validity and Comparability

A run is invalid if the selected runtime row is missing or duplicated, a
reported token count differs from the benchmark shape, a measured time is
non-finite or non-positive, the backend fails, a request times out, or the
readiness gate fails.

Some stored rows come from client versions that timed prefill another way:
OpenVINO and ONNX GenAI rows whose `client_version` is dated before
2026-09-25, which a per-process driver timed over seed token ids, and MLX rows
dated before 2026-09-19, read from the in-process `prompt_tps` sidecar. These
rows carry no HTTP or server-side tokenization cost, so they do not compare
with rows that the client times on its stream clock.

To compare `prefill_throughput` results:

- Use the same benchmark ID, model, runtime, quantization, and relevant runtime
  flags.
- Compare submitted `prefill_time_ms` values directly, or convert all compared
  values to tokens per second with the same formula.
- Treat runtime flags that affect batching, KV cache, mmap behavior, thread
  count, accelerator placement, or memory placement as part of the benchmark
  condition.
- Use the mean and sample standard deviation together; a large standard
  deviation means the run is noisy even when the mean looks plausible.
- Keep llama.cpp CLI rows apart from server rows (see
  [Measured Quantity](#measured-quantity)), and keep CLI and app results apart
  unless the runtime boundary and device readiness conditions are intentionally
  being compared.

The result artifacts preserve enough context to audit the run later:

- Benchmark ID and resolved benchmark parameters.
- Model name, quantization, runtime name, and runtime version.
- Effective runtime flags when the runtime exposes configurable flags.
- Command preview or runtime request context when recorded.
- Captured benchmark stdout or stderr when the runner records it.

## Code References

The benchmark definition and result fields are
[`pipette_plan_types::benchmark`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs);
the CLI's built-in ladder is
[`pipette_cli::benchmarks`](../../crates/pipette-cli/src/benchmarks).

The reps, the warm-up, the checks and the reduction are the typed entry
[`pipette_ops::measurement::prefill`](../../crates/pipette-ops/src/measurement/prefill.rs).
Each CLI engine measures through it from its crate's
`execute/prefill_throughput.rs`: the llama.cpp runner
[`pipette-llamacpp`](../../crates/pipette-llamacpp/src/execute/prefill_throughput.rs)
runs [`bench`](../../crates/pipette-llamacpp/src/bench.rs) or, for a batch,
[`batched_bench`](../../crates/pipette-llamacpp/src/batched_bench.rs); the MLX
runner
[`pipette-mlx`](../../crates/pipette-mlx/src/execute/prefill_throughput.rs)
sends its request through
[`completions`](../../crates/pipette-mlx/src/completions.rs) and reads the
stream with
[`pipette-completions`](../../crates/pipette-completions/src/lib.rs); torch-oai,
OVMS, OpenVINO and ONNX GenAI follow the same shape. The Core AI runner,
[`pipette-coreai`](../../crates/pipette-coreai/src/execute/prefill_throughput.rs),
calls `pipette_ops::measurement::run` instead.

The iOS engines are native Swift under
[`ios/Pipette/Pipette/Runtimes/`](../../ios/Pipette/Pipette/Runtimes/)
(`Llama/Execute/LlamaPrefillThroughput.swift` and its MLX and ONNX GenAI
counterparts). The Android app's are
[`android/native/benchmarks.rs`](../../android/native/benchmarks.rs) and the
Kotlin engine's `LlamaBenchmark.kt`.
