# Measuring OpenVINO cells

What the OpenVINO backend measures, and the decisions behind it that were
settled by running the device rather than reasoning about it. Export, precision
and NPU support live in [openvino-ir.md](openvino-ir.md); this page is about the
harness.

Everything marked *measured* here was run on `devcloud`: Intel Core Ultra 7
258V (Lunar Lake), Intel AI Boost NPU, Windows 11, `openvino-genai` 2026.2.1,
against `LFM2.5-350M/int4-sym-cw` unless stated. Reps are separate processes, as
a real run does them.

## What each cell samples

| cell | sample | first token? | compile? |
|---|---|---|---|
| `prefill_throughput` | `ttft_ms` (one generated token, so TTFT *is* prefill) | — | no |
| `decode_throughput` | `tpot_ms × decode_tokens` | **no** | no |
| `end_to_end_latency` | `ttft_ms + tpot_ms × (n − 1)` | **yes** | no |
| `max_memory_usage` | peak RSS of the driver process | — | **yes** |

The timing cells all sample OpenVINO's own `perf_metrics`, measured inside
`generate()`, and no cell reads the harness's per-rep `Rep.elapsed`, which
would have carried process spawn, python import and the compile. So the compile
is outside every measured *time*.

`max_memory_usage` is the exception, and not by oversight: it samples the
driver process's peak RSS, and the compile allocates in that same process.

`decode_throughput` excluding the first token is deliberate and matches the
other backends: llama.cpp runs `llama-bench --n-prompt 0`, MLX reads `mlx_lm`'s
`generation_tps`. A change that moves only the first token will show in
`end_to_end_latency` and not in `decode_throughput`, which is exactly what the
warm-up does.

## The warm-up

Every timing cell runs one untimed pass before its measured reps, at the cell's
own prefill and decode counts.

The NPU was exempt from this until 2026-08. The exemption read:

> Warm up on CPU and GPU, but not on NPU: there the compile dominates and an
> extra pass only adds device pressure of the kind that took the device down
> during bring-up.

The incident it refers to is real: [openvino-ir.md](openvino-ir.md) records
`ZE_RESULT_ERROR_DEVICE_LOST` from **compiling three pipelines in one process**.
But a warm-up is a second *generate* on an already-compiled pipeline; it
compiles nothing, and one-process-per-rep bounds compiles at one regardless. The
same paragraph asks for "a settle period before the numbers mean anything",
which is what a warm-up is.

*Measured* through the harness itself; `decode_throughput_512_100`, 350M on
NPU, 5 reps each, one binary with the exemption and one without:

| | `decode_time_ms` | stddev | CV |
|---|---|---|---|
| no warm-up | 647.9 | 36.0 | **5.6%** |
| warm-up | 623.4 | 15.2 | **2.4%** |

The warm-up more than halves the deviation (stddev −58%) and moves the mean
−3.8%. The un-warmed number was inflated, not merely noisy. Per-rep the
un-warmed run wanders (166.4, 155.3, 154.2, 142.4, 155.3 tok/s) where the warmed
one holds (163.1, 156.8, 163.7, 163.0, 155.8).

### Measure this through `measurement::run`, not a lookalike

Two standalone reproductions of the above were wrong, in ways worth recording
because both look reasonable from outside:

- **Timing `generate()` with wall clock.** That includes prefill, while
  `decode_throughput` records `tpot × decode_tokens` with the first token
  excluded. It reported the warm-up as worth +18.5% with CV collapsing 8.6% →
  0.3%; most of that was time-to-first-token moving inside a window the cell
  does not report.
- **Constructing `LLMPipeline` with no properties.** The harness derives
  `GENERATE_HINT=BEST_PERF` for NPU, so a bare construction compiles a
  *different graph*: 12.3s compile and ~89 tok/s against the harness's 17.2s and
  ~155 tok/s. Every conclusion drawn from it described a pipeline that never
  runs, including "the NPU has no decode variance to fix". It has 5.6%.

A script that skips the property derivation is not measuring this backend.

## Compile time

The compile is `LLMPipeline(...)` construction. It is timed separately as
`compile_s`, sits outside every measured window, and dominates the cell:

| model | device | compile | measured decode |
|---|---|---|---|
| 350M int4-sym-cw | NPU | 12.4s | 1.1s |
| 1.2B int4-sym-cw | NPU | 32.5s | 1.7s |

A warm-up cannot help this. It runs after the compile, and every rep is a fresh
process, so nothing carries.

`compile_s` currently reaches only a `log::info!` line. `BenchmarkResultData`
has no field for it, so ~95% of an NPU cell's wall clock is unrecorded.

### `CACHE_DIR`

*Measured.* Setting the `CACHE_DIR` pipeline property makes OpenVINO persist the
compiled blob and load it on the next construction:

| | compile |
|---|---|
| cold (writes the blob) | 12.4s |
| warm (loads it) | **0.58s** |

The cache keys on the pipeline properties, not just the model; verified by
compiling two static shapes into one directory:

```
MAX_PROMPT_LEN=1024   12.87s   cold
MAX_PROMPT_LEN=1024    0.84s   hit
MAX_PROMPT_LEN=2048   21.41s   cold, not a false hit
MAX_PROMPT_LEN=2048    0.87s   hit
MAX_PROMPT_LEN=1024    0.79s   still hit, both blobs coexist
```

So two cells differing only in `MIN_RESPONSE_LEN` cannot silently share a graph.
(The 1024 → 2048 compile cost also confirms the superlinear growth
[openvino-ir.md](openvino-ir.md) records.)

The cache lives under `.pipette/cache/<runtime-key>/`, outside the artifact
stores so a storage sweep cannot delete what a run depends on, and shared across
cells and runs. `CACHE_DIR` is added to the pipeline properties by `Cell::properties`, not
by `flags.rs`. It is this host's scratch path, not something the cell declared,
and it must not reach the recorded flags.

**Every measured rep must load from it, not just reps 2–5.** A cache alone would
leave rep 1 compiling (17.2s under `BEST_PERF`) while the rest load (~0.6s),
which is the same "one rep met the device differently" problem the warm-up
result above is about. So `Cell::bind` calls `precompile` (one throwaway compile in its own process
(`Mode::Compile`)) before any measured rep. Distinct from `warmup`, which is an
untimed *generate*; the two are easy to confuse and do different jobs.

*Measured*, same NPU decode cell, cache wiped then repeated:

```
cold   cache warm: compile 18.1s   reps: 0.6 0.6 0.6 0.6 0.6
warm   cache warm: compile  0.6s   reps: 0.6 0.6 0.6 0.6 0.6
```

Compile per cell drops from 5×17.2s ≈ 86s to ~21s cold and ~4s warm. The blob is
94 MB for a 350M IR, so the cache is worth sweeping like any other derived
bytes.

The cache sits at `.pipette/cache/<runtime-key>/`, under the same key as
`runtimes/<key>/`: flat, with no engine segment, since the key already names
the runtime type and a runtime has exactly one engine. The key digests the
requirements body, so an `openvino-genai` bump lands elsewhere and a stale blob
is unreachable rather than merely version-checked, and `runtimes remove`
reclaims it by looking in one place.

The cache buys wall clock; its correctness value is that every rep now starts
from the same place.

#### It moves `max_memory_usage`, and only that cell

The three timing cells sample `perf_metrics` from inside `generate()`, so a
shorter compile cannot reach them. `max_memory_usage` samples peak RSS of the
process the compile runs in, so it can and does.

*Measured*, same NPU cell, `max_memory_usage_512`:

| | compile in the measured process | `max_host_bytes` |
|---|---|---|
| no cache | 17.7s | 927,793,152 |
| cache | 0.6s (blob load) | 817,033,216 |

−110 MB, −11.9%: compiling costs transient memory that loading a blob does not.

The cached figure is the better one. It is what running the model costs, where
the uncached figure conflated inference with a one-off compilation. But it is a
**discontinuity**: NPU `max_memory_usage` results from before and after the
cache are not comparable, and the gap is wide enough to read as a regression in
a trend. Anything comparing across that boundary has to say which side it is on.

### The n=5 stddev is itself unstable

Two runs of the *same* configuration reported `598.1 ± 1.8` and `615.3 ± 22.8`;
a 12x spread in the stddev estimate. So a single 5-rep run cannot settle a
question about variance, and every comparison here should be read as one sample.
The warm-up result above survives that caveat only because the per-rep values
back it: the un-warmed run wanders 166→142 tok/s where the warmed one holds
inside 156-164. Anything smaller than that deserves repeated runs before it is
believed.

## What the readiness gate cannot see

`pipette-readiness`'s Windows gate samples CPU load, GPU compute and die
temperature. None of them see NPU work (`gpu=compute:0%` throughout an NPU cell
), and `windows.rs` says so directly: *"Nothing catches an OpenVINO NPU graph
compile, ~90% of an NPU cell's wall clock."*

Worse, the gate runs *before* the driver process, and the compile happens
*inside* it. On a 1.2B NPU cell the gate certifies a cool idle device, then the
process spends 32.5s compiling before the measured generate. The verdict is
stale by the largest thermal event in the cell.

Gating the NPU properly would mean settling *after* the compile, inside the
driver. Unmeasured; the `\Energy Meter(*)\Power` counter that does see NPU draw
(6.6W against a 3.3W floor) is verified on one laptop and reads as idle when
absent.

## Open

- `compile_s` has nowhere to be recorded.
- `pipette storage` does not see `.pipette/cache`. `runtimes remove` reclaims a
  runtime's blobs, but nothing evicts them under quota pressure. They should be
  a storage kind that is always the first candidate, being pure derived bytes.
- The NPU has no readiness signal, and the gate's verdict predates the compile.
- Whether the warm-up's 2.4% residual CV can be brought down further, and
  whether it holds on larger models: the 1.2B showed 7.7% before the warm-up
  landed and has not been re-measured since.
