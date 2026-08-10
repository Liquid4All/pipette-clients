# Decode Throughput Methodology

## Measured Quantity

Decode throughput is the speed of autoregressive token generation after a
prompt context already exists. The benchmark measures the decode phase for a
fixed number of generated tokens while conditioning on a fixed prompt depth.

In pipette, this metric is reported by the `decode_throughput` benchmark
family.

The submitted result is an elapsed time, not a token-rate field:

- `decode_time_ms`: mean time to generate the requested decode tokens.
- `decode_time_ms_stddev`: sample standard deviation across measured
  repetitions.

A token-per-second rate can be derived as
`parameter_decode_tokens * 1000 / decode_time_ms`. Pipette stores the time
field because the benchmark shape already carries the decode-token count, and
elapsed time keeps the submitted payload aligned with the latency benchmarks.

Runtime installation, model download, model load, server startup, benchmark
warmup, platform readiness waits, prompt-context setup, and result
sync/upload are outside the decode timing.

The scope is model decode work for a fixed number of generated tokens at a
fixed prompt depth. It is not TTFT, end-to-end request latency, or streaming
API latency.

## Measurement Challenges and Controls

Decode cost depends on both generated-token count and existing context length.
A model generating 100 tokens after an 8-token prompt is doing different work
from the same model generating 100 tokens after a 512-token prompt. The
benchmark therefore treats both prompt depth and decode count as the comparison
key.

The main methodological requirement is to keep prefill outside the timed
window. Each runtime must establish the requested prompt context first, then
time only the generated-token loop.

Models can stop early when they emit EOS or another end-of-generation token.
Early stop would shorten the measured decode workload and inflate the apparent
throughput. Implementations therefore use benchmark-native token counts or an
ignore-EOS path and fail when the runtime reports a different decode-token
count.

Warmup and thermal state can dominate repeated decode timings. The CLI runners
separate warmup from measured samples and run the platform readiness gate
before each measured repetition. Individual samples can still be affected by
scheduler noise, runtime caches, or transient background load, so the benchmark
reports both mean and sample standard deviation across five repetitions.

Cached state can also distort decode timings. A measured decode that reused the
previous repetition's KV cache would be doing different work from a fresh
decode at the requested depth. Each measured repetition therefore starts from a
fresh runtime state: the llama.cpp CLI path runs one `llama-bench` process per
sample (the per-process `n_depth` pass rebuilds the prompt context before the
timed block), the iOS app resets the llama context and sampler before every
iteration, and the MLX sidecar calls `mlx_lm.stream_generate` without a
`prompt_cache` argument so no KV state carries across requests within the
long-lived sidecar process.

## Measurement Protocol

The benchmark has a shared measurement contract:

1. Resolve `parameter_prefill_tokens` and `parameter_decode_tokens`.
2. Prepare the runtime and model.
3. Run the backend's warmup path outside the submitted measurement.
4. For each measured repetition, establish the requested prompt context outside
   the timed window.
5. Time generation of the requested decode-token count.
6. Before each measured CLI repetition, run the platform readiness check.
7. Validate the decode token count when the runtime reports it.
8. Report the mean and sample standard deviation of the five measured decode
   times.

The timed window starts immediately before the backend's decode operation and
ends immediately after the requested decode operation completes. Runtime
startup, model load, warmup, readiness checks, prompt-context setup, result
serialization, and sync are outside that window.

The exact implementation of the timed window is runtime-specific. llama.cpp
uses `llama-bench` depth to set up the prompt context before the timed block.
MLX reports generation tokens per second from the Python sidecar and the Rust
runner converts that rate back to milliseconds. The iOS app measures elapsed
time around the native-Swift llama decode call after prefill.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use two token-count fields:

- `parameter_prefill_tokens`: prompt depth used before decode timing starts.
- `parameter_decode_tokens`: number of generated tokens to time.

The standard ladder uses IDs such as `decode_throughput_512_100`, meaning
100 generated tokens after a 512-token prompt context. The local smoke
benchmark uses 8 prompt tokens and 8 generated tokens. The benchmark requires
`parameter_prefill_tokens >= 1` because decode is defined as autoregressive
generation conditioned on an existing prompt context; a depth of zero is the
TTFT shape and is not in scope here.

The submitted result reports:

- `decode_time_ms`: mean decode time across measured repetitions.
- `decode_time_ms_stddev`: sample standard deviation across measured
  repetitions.

Lower `decode_time_ms` is better. If a report presents this benchmark as
tokens per second, higher is better after applying the derived-rate conversion.

The token-rate metric is derived by the management service:

```text
decode_throughput = parameter_decode_tokens / decode_time_ms * 1000
value_stddev(decode_throughput) =
    decode_throughput * decode_time_ms_stddev / decode_time_ms
```

For measured times `t_1..t_5`, `decode_time_ms` is `sum(t_i) / 5`, and
`decode_time_ms_stddev` is `sqrt(sum((t_i - mean)^2) / 4)`.

## Token Control

The prompt-depth and decode-token counts define the work. The current
implementations use the most direct token-control path available for each
runtime:

- llama.cpp uses `llama-bench --n-prompt 0 --n-gen <D> --n-depth <P>`.
- MLX builds a deterministic token sequence of length `<P>` and calls the
  sidecar generation path for `<D>` tokens.
- The iOS app tokenizes an app-local seed prompt, prefills that context, then
  runs a native-Swift ignore-EOG decode for `<D>` tokens.

The invariant for the CLI runners is the requested decode-token count at the
requested prompt depth. A run is invalid if the selected `llama-bench` row has
the wrong `n_prompt`, `n_gen`, or `n_depth` shape, if the MLX sidecar returns
the wrong `decode_tokens`, or if the reported throughput value is missing,
non-finite, or non-positive.

`llama-bench` itself generates exactly `n_gen` tokens regardless of EOS, so
the CLI path does not need an explicit ignore-EOS flag. The MLX sidecar
suppresses EOS before calling `mlx_lm.stream_generate`, and the iOS app uses
its native-Swift `Inference.decodeIgnoringEoG` primitive. Each path keeps the
measured decode length equal to the requested count.

The iOS app builds the prompt context differently from the CLI runners. It
tiles an app-local seed corpus (`PromptSeed.corpus`) and takes the first
`parameter_prefill_tokens` tokens. The required context window (
`parameter_prefill_tokens + parameter_decode_tokens`) is computed from the
benchmark shape and validated up front by `check_ctx_size` before the model
loads, so a context too small for the benchmark fails with a readable error
instead of running at a shorter depth. iOS results should still be compared only
against other iOS results, because the prompt is a repeated filler string rather
than the shared seed sequence and the app does not use the shared readiness
gate.

## System Readiness Control

The CLI runners use the same platform readiness infrastructure as
`end_to_end_latency`. Android, Linux, macOS, and Windows wait for host or
device readiness before measured repetitions; unsupported platforms use the
no-op fallback from `pipette_readiness`. See the platform deadline,
thermal, and CPU criteria table in
[`end-to-end-latency.md`](end-to-end-latency.md#system-readiness-control).

Readiness checks are outside the timed window. Their purpose is to make each
measured repetition start from a comparable thermal and CPU-load state. If the
host or device fails to reach readiness before the platform deadline, the run
fails rather than submitting a decode timing from an unstable condition.

The iOS app path currently runs its own in-process measurements and does not
use the shared `pipette_readiness` gate.

## Runtime Details

The current client implementations use these paths:

| Path | Workload | Timing source | Notes |
| --- | --- | --- | --- |
| llama.cpp CLI | `llama-bench --output json --model <path> --n-prompt 0 --n-gen <D> --n-depth <P> -r 1` per measured repetition | selected row `avg_ns`, converted to milliseconds | The depth run establishes context before the timed generation block. |
| MLX CLI | Python sidecar `/decode_throughput` with `prompt_tokens = <P>` and `decode_tokens = <D>` | sidecar `generation_tps`, converted to milliseconds | Warmup runs the benchmark's own shape. |
| iOS app | native-Swift llama prefill followed by ignore-EOG decode | Swift-side elapsed time around decode only | One warmup plus five measured repetitions inside the app process. |

### llama.cpp

`pipette-llamacpp` runs one `llama-bench` process per measured sample and
passes `-r 1`. `llama-bench` warmup remains enabled by default unless the user
explicitly changes it through runtime flags; warmup is not part of the selected
row's `avg_ns`.

The runner selects exactly one JSON row matching:

- `n_prompt == 0`.
- `n_gen == parameter_decode_tokens`.
- `n_depth == parameter_prefill_tokens`.

In the vendored `llama-bench` implementation, `n_depth` runs a prompt pass
before the timed block. The timed block then runs generation for `n_gen`
tokens. The row's `avg_ns` is converted to milliseconds for that sample, and
five samples are aggregated by the Rust harness.

The runner controls these workload-shaping flags and rejects user overrides for
them: `--output`, `-o`, `--model`, `-m`, `--n-prompt`, `-p`, `--n-gen`, `-n`,
`--n-depth`, `-d`, `--repetitions`, and `-r`.

The runner adds `--mmap 0` by default unless the user supplies an mmap flag.
User runtime flags that are not reserved remain part of the runtime condition
and must be kept identical across compared runs.

### MLX

`pipette-mlx` starts a local Python HTTP sidecar, then sends one warmup request
to `/decode_throughput` at the benchmark's own `prompt_tokens` and
`decode_tokens`, so kernel selection is paid before the first measured rep.

For each measured repetition, the Rust runner waits for platform readiness and
then sends:

```json
{
  "prompt_tokens": "<P>",
  "decode_tokens": "<D>"
}
```

The sidecar repeats the benchmark prompt-seed token IDs until it has `<P>`
tokens, suppresses EOS, calls `mlx_lm.stream_generate` with
`max_tokens = <D>`, and returns:

- `generation_tps`: MLX generation throughput for that call. This is
  `mlx_lm`'s decode-phase rate (generated tokens divided by generation time);
  it excludes the prompt-processing phase, which `mlx_lm` reports separately as
  `prompt_tps`. Reading `generation_tps` is therefore what isolates decode from
  prefill on the MLX path, the same separation the llama.cpp path gets from the
  `--n-prompt 0 --n-depth <P>` row.
- `decode_tokens`: the decode-token count used by the sidecar.

The Rust runner validates `decode_tokens`, checks that `generation_tps` is
finite and positive, converts `generation_tps` to milliseconds, and aggregates
five converted samples.

### iOS

The iOS app path loads the GGUF model, resets the llama context and sampler for
each repetition, prefills the prompt context, and then times the native-Swift
`Inference.decodeIgnoringEoG` decode primitive. The ignore-EOG decode path is used so the
timed section covers the requested token count instead of stopping early on a
repetitive dummy prompt. The app runs one warmup iteration and five measured
iterations, then reports the same result fields as the CLI runners.

This path is useful for app-integrated measurements, but it should be compared
in its own runtime context because it does not use the shared CLI readiness
gate.

## Unsupported Runtime Paths

`pipette-torch-oai` currently rejects `decode_throughput` benchmarks. The
OpenAI-compatible Docker and uv runners support `end_to_end_latency`,
`max_memory_usage`, and `eval` today, but not prefill/decode throughput.

## Result Validity and Comparability

A run is invalid if the selected runtime row is missing or duplicated, a
reported token count differs from the benchmark shape, a throughput value is
non-finite or non-positive, the backend fails, the timing deadline elapses, or
the readiness probe fails.

To compare `decode_throughput` results:

- Use the same benchmark ID, model, runtime, quantization, and relevant runtime
  flags.
- Compare only results with the same prompt depth and decode-token count.
- Compare submitted `decode_time_ms` values directly, or convert all compared
  values to tokens per second with the same formula.
- Treat runtime flags that affect KV cache, batch sizing, thread count,
  accelerator placement, sampling behavior, or memory placement as part of the
  benchmark condition.
- Use the mean and sample standard deviation together; a large standard
  deviation means the run is noisy even when the mean looks plausible.
- Keep CLI and iOS app results separate unless the runtime boundary and device
  readiness conditions are intentionally being compared.

The result artifacts preserve enough context to audit the run later:

- Benchmark ID and resolved benchmark parameters.
- Model name, quantization, runtime name, and runtime version.
- Effective runtime flags when the runtime exposes configurable flags.
- Command preview or runtime request context when recorded.
- Captured benchmark stdout or stderr when the runner records it.

## Code References

The benchmark definition and result fields are implemented in
[`pipette_plan_types::benchmark`](../../crates/pipette-plan-types/src/benchmark/mod.rs) and
[`pipette_cli::benchmarks`](../../crates/pipette-cli/src/benchmarks).

The llama.cpp runner is implemented in
[`pipette-llamacpp decode_throughput`](../../crates/pipette-llamacpp/src/execute/decode_throughput.rs)
and the shared `llama-bench` repetition helper is
[`bench`](../../crates/pipette-llamacpp/src/bench.rs).

The MLX runner and sidecar endpoint are implemented in
[`pipette-mlx decode_throughput`](../../crates/pipette-mlx/src/execute/decode_throughput.rs)
and
[`pipette_mlx_server.py`](../../crates/pipette-mlx/src/python/pipette_mlx_server.py).

The iOS app implementation is
[`LlamaBenchmark`](../../ios/Pipette/Pipette/Runtimes/Llama/LlamaBenchmark.swift)
(native Swift).
