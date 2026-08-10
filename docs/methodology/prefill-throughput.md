# Prefill Throughput Methodology

## Measured Quantity

Prefill throughput is the speed of processing prompt tokens into the model
state used for generation. In transformer runtimes, this is the prompt phase
that builds the KV cache before autoregressive decode begins.

In pipette, this metric is reported by the `prefill_throughput` benchmark
family.

The submitted result is an elapsed time, not a token-rate field:

- `prefill_time_ms`: mean time to prefill the requested prompt tokens.
- `prefill_time_ms_stddev`: sample standard deviation across measured
  repetitions.

A token-per-second rate can be derived as
`parameter_prefill_tokens * 1000 / prefill_time_ms`. Pipette stores the time
field because the benchmark shape already carries the token count, and elapsed
time keeps the submitted payload aligned with the latency benchmarks.

Runtime installation, model download, model load, server startup, benchmark
warmup, platform readiness waits, and result sync/upload are outside the
prefill timing.

The scope is model prefill work for a fixed number of prompt tokens. This is
not end-to-end prompt latency: tokenizer latency, HTTP request overhead, and
decode are outside the measured quantity on every backend. The timed window is
always the prefill operation alone (see the protocol below); what varies by
backend is only how that window is sourced: a native benchmark timing row, a
runtime counter, or a direct in-process measurement.

## Measurement Challenges and Controls

Prefill cost scales with prompt-token count, batch sizing, attention
implementation, KV-cache format, quantization, and accelerator placement. A
result is comparable only for the same benchmark shape, model, runtime, and
relevant runtime flags.

Text prompts are not required for this benchmark. Unlike end-to-end latency,
prefill throughput is intended to isolate model prompt-processing work rather
than caller-visible request latency. Runtimes can therefore use token IDs or a
benchmark-native token-count control when that path most directly represents
the requested work.

Warmup and thermal state can dominate prompt-processing timings, especially on
small devices. The CLI runners separate warmup from measured samples and run
the platform readiness gate before each measured repetition. This keeps a
single long prefill from leaving residual heat that silently changes the next
sample.

Cached state can also distort prompt-processing timings. A second call into a
runtime that retained KV state from the previous call would measure a different
workload from a cold prefill. Each measured repetition therefore starts from a
fresh runtime state: the llama.cpp CLI path runs one `llama-bench` process per
sample, the iOS app resets the llama context and sampler before every iteration,
and the MLX sidecar calls `mlx_lm.stream_generate` without a `prompt_cache`
argument so no KV state carries across requests within the long-lived sidecar
process.

Individual samples can still be affected by scheduler noise, runtime caches,
or transient background load. The benchmark measures five repetitions and
reports both mean and sample standard deviation.

## Measurement Protocol

The benchmark has a shared measurement contract:

1. Resolve `parameter_prefill_tokens`.
2. Prepare the runtime and model.
3. Run the backend's warmup path outside the submitted measurement.
4. Run five measured prefill repetitions.
5. Before each measured CLI repetition, run the platform readiness check.
6. Validate the prefill token count when the runtime reports it.
7. Report the mean and sample standard deviation of the five measured prefill
   times.

The timed window starts immediately before the backend's prefill operation and
ends immediately after that prefill operation completes. Token construction,
runtime startup, model load, warmup, readiness checks, result serialization,
and sync are outside that window.

The exact implementation of the timed window is runtime-specific. llama.cpp
uses `llama-bench`'s own timing row. MLX reports prompt tokens per second from
the Python sidecar and the Rust runner converts that rate back to milliseconds.
The iOS app measures elapsed time around the direct llama prefill call.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use one token-count field:

- `parameter_prefill_tokens`: number of prompt tokens to prefill.

The standard ladder uses IDs such as `prefill_throughput_512`, meaning a
512-token prefill. The local smoke benchmark uses 8 prompt tokens.

The submitted result reports:

- `prefill_time_ms`: mean prefill time across measured repetitions.
- `prefill_time_ms_stddev`: sample standard deviation across measured
  repetitions.

Lower `prefill_time_ms` is better. If a report presents this benchmark as
tokens per second, higher is better after applying the derived-rate conversion.

The token-rate metric is derived by the management service:

```text
prefill_throughput = parameter_prefill_tokens / prefill_time_ms * 1000
value_stddev(prefill_throughput) =
    prefill_throughput * prefill_time_ms_stddev / prefill_time_ms
```

Prefill throughput is reported as a tokens-per-second rate. Time-to-first-token
(`ttft`) is a separate, caller-visible metric derived from the end-to-end
latency benchmark (which sends real text and includes tokenization); it is not
this benchmark's isolated `prefill_time_ms`. The two are intentionally kept
apart: `prefill_time_ms` isolates model prompt-processing work, while `ttft`
captures the full caller-visible delay; tokenization, request overhead, and
scheduling on top of prefill. Do not read `prefill_time_ms` as a wall-clock TTFT
or compare it against one; a real TTFT is always the larger of the two.

For measured times `t_1..t_5`, `prefill_time_ms` is `sum(t_i) / 5`, and
`prefill_time_ms_stddev` is `sqrt(sum((t_i - mean)^2) / 4)`.

## Token Control

The prefill-token count defines the work. The current implementations use the
most direct token-control path available for each runtime:

- llama.cpp uses `llama-bench --n-prompt <P> --n-gen 0`.
- MLX builds a deterministic token sequence of length `<P>` in the Python
  sidecar.
- The iOS app tokenizes an app-local dummy prompt and preloads the resulting
  token slice.

The invariant for the CLI runners is the requested prefill-token count. A run
is invalid if the selected `llama-bench` row has the wrong `n_prompt` or `n_gen`
shape, if the MLX sidecar returns the wrong `prompt_tokens`, or if the reported
throughput value is missing, non-finite, or non-positive.

The iOS app builds the prompt differently from the CLI runners. It tokenizes
`"hello "` repeated text and takes the first `parameter_prefill_tokens` tokens.
The required context window (`parameter_prefill_tokens`) is computed from the
benchmark shape and validated up front by `check_ctx_size` before the model
loads, so a context too small for the benchmark fails with a readable error
instead of running a shorter prefill. iOS results should still be compared only
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
fails rather than submitting a prefill timing from an unstable condition.

The iOS app path currently runs its own in-process measurements and does not
use the shared `pipette_readiness` gate.

## Runtime Details

The current client implementations use these paths:

| Path | Workload | Timing source | Notes |
| --- | --- | --- | --- |
| llama.cpp CLI | `llama-bench --output json --model <path> --n-prompt <P> --n-gen 0 -r 1` per measured repetition | selected row `avg_ns`, converted to milliseconds | Rust runs five separate one-repetition invocations so readiness can run between samples. |
| MLX CLI | Python sidecar `/prefill_throughput` with `prompt_tokens = <P>` | sidecar `prompt_tps`, converted to milliseconds | Warmup runs the benchmark's own shape. |
| iOS app | direct llama prefill after context reset | Rust-side elapsed time around `llama::prefill` | One warmup plus five measured repetitions inside the app process. |

### llama.cpp

`pipette-llamacpp` runs one `llama-bench` process per measured sample and
passes `-r 1`. `llama-bench` warmup remains enabled by default unless the user
explicitly changes it through runtime flags; warmup is not part of the selected
row's `avg_ns`.

The runner selects exactly one JSON row matching:

- `n_prompt == parameter_prefill_tokens`.
- `n_gen == 0`.

The row's `avg_ns` is converted to milliseconds for that sample. Five samples
are aggregated by the Rust harness.

The runner controls these workload-shaping flags and rejects user overrides for
them: `--output`, `-o`, `--model`, `-m`, `--n-prompt`, `-p`, `--n-gen`, `-n`,
`--repetitions`, and `-r`.

The runner adds `--mmap 0` by default unless the user supplies an mmap flag.
User runtime flags that are not reserved remain part of the runtime condition
and must be kept identical across compared runs.

### MLX

`pipette-mlx` starts a local Python HTTP sidecar, then sends one warmup request
to `/prefill_throughput` at the benchmark's own `prompt_tokens`, so kernel
selection is paid before the first measured rep.

For each measured repetition, the Rust runner waits for platform readiness and
then sends:

```json
{ "prompt_tokens": "<P>" }
```

The sidecar repeats the benchmark prompt-seed token IDs until it has `<P>`
tokens, suppresses EOS, and calls `mlx_lm.stream_generate` with
`max_tokens = 1`. EOS suppression is applied for symmetry with the other
benchmarks; with `max_tokens = 1` the call returns after one generated token
regardless of EOS, so suppression has no effect on the measured prefill. The
sidecar then returns:

- `prompt_tps`: MLX prompt throughput for that call.
- `prompt_tokens`: the prompt-token count used by the sidecar.

The Rust runner validates `prompt_tokens`, checks that `prompt_tps` is finite
and positive, converts `prompt_tps` to milliseconds, and aggregates five
converted samples.

### iOS

The iOS app path loads the GGUF model, resets the llama context and sampler for
each repetition, and times the direct `llama::prefill` call. It runs one warmup
iteration and five measured iterations, then reports the same result fields as
the CLI runners.

This path is useful for app-integrated measurements, but it should be compared
in its own runtime context because it does not use the shared CLI readiness
gate.

## Unsupported Runtime Paths

`pipette-torch-oai` currently rejects `prefill_throughput` benchmarks. The
OpenAI-compatible Docker and uv runners support `end_to_end_latency`,
`max_memory_usage`, and `eval` today, but not prefill/decode throughput.

## Result Validity and Comparability

A run is invalid if the selected runtime row is missing or duplicated, a
reported token count differs from the benchmark shape, a throughput value is
non-finite or non-positive, the backend fails, the timing deadline elapses, or
the readiness probe fails.

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
[`pipette-llamacpp prefill_throughput`](../../crates/pipette-llamacpp/src/execute/prefill_throughput.rs)
and the shared `llama-bench` repetition helper is
[`bench`](../../crates/pipette-llamacpp/src/bench.rs).

The MLX runner and sidecar endpoint are implemented in
[`pipette-mlx prefill_throughput`](../../crates/pipette-mlx/src/execute/prefill_throughput.rs)
and
[`pipette_mlx_server.py`](../../crates/pipette-mlx/src/python/pipette_mlx_server.py).

The iOS app implementation is native Swift under
[`ios/Pipette/Pipette/Runtimes/`](../../ios/Pipette/Pipette/Runtimes/).
