# End-to-End Latency Methodology

## Measured Quantity

End-to-end latency is the time a caller waits from submitting a text prompt to
receiving the complete generated response. The benchmark measures that interval
around one local text-completion request. Because the prompt is sent as text,
the timed path includes runtime tokenization, prefill, decode, and local
request/response overhead.

In pipette, this metric is reported by the `end_to_end_latency` benchmark
family.

Runtime installation, model download, model load, server startup, server
readiness polling, warmup, readiness waits before measured requests, and result
sync/upload are setup or orchestration costs. They remain outside the
request-latency metric.

Each benchmark shape is defined by a target prompt length and a target
completion length, both measured in tokens. A result is comparable only when
the runtime reports those exact token counts.

The scope is local full-request latency for text-only completion. The per-phase
rates are measured by the sibling [prefill-throughput](prefill-throughput.md)
and [decode-throughput](decode-throughput.md) benchmarks; the derived metrics
the management service computes from each (time-to-first-token among them) are
defined in pipette-mgmt
[`docs/benchmarks.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#23-end-to-end-latency). The
[vision-language benchmark](vl-throughput.md#end-to-end-latency) sends a
`total_time_ms` of the same kind, around a request that also uploads and
preprocesses an image and that `llama-server` answers in one reply rather than
a stream.

## Controls at a Glance

End-to-end latency is sensitive to many factors unrelated to model quality.
Each known source of measurement bias is controlled as summarized below; the
rest of this document is the detailed implementation of these controls.

- **Tokenizer differences across model families**: comparisons use exact
  prompt-token and completion-token counts under each runtime's own tokenizer,
  not prompt text or byte length. A token-count mismatch on warmup or any
  measured request invalidates the run.
- **Early stop on EOS**: end-of-generation tokens are suppressed so every
  measured request generates exactly the requested number of tokens. Fixing the
  decode length holds every compared run to the same amount of work. Core AI
  does not suppress them; an early stop there fails the count check.
- **Prefix and KV reuse across repeated requests**: server-side prefix reuse
  is disabled (llama.cpp `cache_prompt: false`, MLX with no prompt cache,
  vLLM/SGLang with prefix caching off), so each measured request pays the full
  prefill cost and the engines stay comparable when a prompt repeats.
- **Thermal state and throttling**: each measured request is gated by a
  readiness probe that combines a temperature signal with a CPU busy signal.
  Phones run on mains power and under active cooling
  ([Device conditions](device-conditions.md#active-cooling)), so phone results
  compare only with runs under the same setup.
- **Cold start and warmup**: model load, server startup, readiness polling,
  and one warmup request happen before timing begins and are excluded from the
  reported latency.
- **Scheduler noise and per-request variance**: each cell runs five measured
  requests and reports the mean alongside the sample standard deviation, so an
  unstable run is visible in the result rather than hidden in the headline
  number.

## Measurement Challenges and Controls

Tokenizers differ across model families, runtime implementations, and
special-token policies. The same prompt text may be 512 tokens for one runtime
and a different length for another. The benchmark therefore uses exact
prompt-token count and exact completion-token count as the comparison key. Each
run builds the prompt with the tokenizer path that matches the inference
request.

Sending token IDs directly would bypass server-side tokenization and measure a
narrower prefill-plus-decode path. The benchmark sends prompt text so tokenizer
cost remains inside the timed request.

> **Implementation status.** Every runner tokenizes inside the timed window:
> the server-backed CLI runners, the in-process iOS llama.cpp, MLX and AFM
> engines, and both Android app engines (`kotlinEngine`, the default, in
> `LlamaBenchmark.kt`, and `rustEngine` in `android/native/benchmarks.rs`).
> There are two exceptions:
>
> - iOS ONNX GenAI tokenizes the prompt once before the repetitions and starts
>   its clock at `appendTokens`. Its e2e leaves out tokenizing and building the
>   generator.
> - Core AI sends token ids, not text, so no tokenizer runs in its window (see
>   [Core AI](#core-ai)).

Models can emit EOS or another stop token before the requested decode length.
A shorter completion invalidates the requested benchmark shape. The benchmark
forces or validates generation to the requested completion-token count and
fails the run if the runtime reports a different count.

Forcing generation past a natural stop is deliberate. The metric is latency at
a fixed `(prompt_tokens, completion_tokens)` shape, so every compared run must
do the same amount of decode work; letting one model stop at 40 tokens and
another at 200 would compare different workloads, not different speeds. The
cost is that suppression can exercise decode steps a well-aligned model would
not normally take, so end-to-end latency measures per-token serving speed at a
fixed shape, not how concisely a model chooses to answer.

Thermal state and background load can dominate latency, especially on phones,
laptops, and small desktops. A device that is still cooling down from a
previous run may start fast and then throttle during the measured request
series. The benchmark gates every measured request on platform readiness
outside the timed window ([System Readiness Control](#system-readiness-control))
and fails the cell rather than record latency under unstable conditions.

Individual requests can still be affected by scheduler noise, transient runtime
state, or background work. The primary defense against per-request variance is
the readiness gate rather than the repetition count: because each measured
request first waits for the thermal and CPU-load criteria, every sample starts
from a comparable controlled state. The reported value is the plain mean of the
five; no trimming, outlier rejection, or median. Five is a deliberate
trade-off: on Android phones and CPU-only laptops each gated repetition is
slow, since the readiness wait and the decode work both run on the constrained
device. Five is enough that a single throttled request surfaces as an inflated
standard deviation instead of passing unnoticed. The standard deviation is
therefore part of the result. A high value means that the mean is not a stable
estimate: read the run again, or run it again.

## Measurement Protocol

Every runner first starts or attaches to its serving endpoint (a local
`llama-server` or `mlx_lm.server`, a Docker or uv engine, an in-house server,
the Core AI sidecar, or an in-process engine) and waits for it to answer. The MLX, OVMS, OpenVINO,
ONNX GenAI and Core AI runners also gate readiness once before they launch their
server, and the torch-oai runner gates once before its first request; the
llama.cpp runner has no gate before the warmup. The shared measurement phase
is then:

1. Build a text prompt that tokenizes to exactly the requested prompt-token
   count under the runtime tokenizer used for inference.
2. Run one warmup request at the full benchmark shape (the same prompt and
   decode count as the measured requests) and validate its prompt-token and
   completion-token counts. Warming at the real shape is what makes the first
   measured repetition comparable to the other four: runtimes select and compile
   kernels per tensor shape, so a lighter warmup leaves that cost to be paid
   inside repetition one. The warmup is excluded from the reported statistics
   and has no readiness check of its own; the per-repetition check runs before
   each measured repetition.
3. For each of five measured repetitions, run the platform readiness check.
4. Submit one measured request with the same prompt and decode count.
5. Validate prompt-token and completion-token counts for each measured
   request.
6. Report the mean and sample standard deviation of the five measured request
   latencies.

Every server engine streams the measured request and times it the same way.
Core AI is the exception: it times one unstreamed request (see
[Core AI](#core-ai)).
The timed request window starts once the request's body is encoded,
immediately before it is sent, and ends when the chunk that says why the
prompt stopped arrives: llama.cpp's final chunk (`stop: true`), or the chunk
carrying `finish_reason` on the other engines' completions endpoint. That chunk
is stamped as it arrives, before it is parsed; the usage trailer and `[DONE]`
after it are outside the window.

Stored rows of every engine but llama.cpp whose `client_version` is dated
before 2026-10-02 are timed unstreamed, to the fully parsed reply. At 256
generated tokens, the unstreamed rows differ from streamed ones as follows:

- vLLM and MLX: by under 0.3%.
- OVMS on a CPU: 1 to 2% longer.
- pipette's OpenVINO server on a CPU: by about 1% in either direction.

The unstreamed OpenVINO and ONNX GenAI rows also include a new connection per
request, because those servers answered HTTP/1.0 in those client versions
([prefill throughput](prefill-throughput.md#openvino-and-onnx-genai) has what
that costs on Windows).

The client keeps its connection between requests wherever the server allows
it, so a new connection falls inside the window only where the server closes
the connection after each reply. A new loopback connection adds about 0.1 to
0.25 ms to a request on Linux and macOS, but 13 to 18 ms on Windows.
`llama-server` closes every streamed reply's connection, and `mlx_lm.server`
answers HTTP/1.0, which closes every connection. A pooled client can also open
a fresh connection now and then, while its previous one is still on its way
back to the pool.

Readiness waits, warmup, server startup, model load, prompt construction,
token-count validation after the response, and result sync are outside that
window. The readiness check runs before every measured repetition, including
the first one after warmup, so any heat generated by setup or warmup must clear
before a measured latency is recorded.

That wait has a cost on a host whose GPU driver evicts idle allocations. On the
GMKtec EVO-X2 (Radeon 8060S, Windows, AMD driver 32.0.31041.1004), eviction
starts about 0.73 s after the last response. The readiness wait before each
measured repetition is longer than that: about 1.5 s on an idle box, and 32 to
34 s under the temperature-flatness gate that the investigation used. The next
measured request therefore pays the restore inside its window: about 450 ms of
extra prefill at 512 prompt tokens. A comparison that adds standalone prefill
and decode times to predict E2E must account for this:

- Run the readiness wait, then a full-shape warmup, immediately before each
  timed repetition.
- Report the first request after idle separately, against a matching
  cold-prefill estimate.
Those boxes run [`tools/gpu-keepalive`](../../tools/gpu-keepalive/README.md),
which removes the penalty; the measurements are in
[GMKtec GPU idle residency](../fleet-perf-troubleshooting/gmktec-evo-x2-gpu-idle.md).

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use these fields:

- `parameter_prefill_tokens`: exact number of prompt tokens.
- `parameter_decode_tokens`: exact number of generated tokens.
- `parameter_batch_size`: requests kept in flight at once. Optional, default
  `1`; see [Batched](#batched).

The standard ladder uses IDs such as `end_to_end_latency_512_256`, meaning
512 prompt tokens and 256 generated tokens; a batched entry carries the count
as a `_bs<n>` suffix, `end_to_end_latency_512_256_bs4`. The local smoke
benchmark uses 8 prompt tokens and 8 generated tokens.

The submitted result reports:

- `total_time_ms`: mean latency across the measured requests.
- `total_time_ms_stddev`: sample standard deviation across the measured
  requests.

Milliseconds are the unit because this benchmark measures the wall-clock wait
for one complete request, the latency a caller observes at the completion API
boundary. The prompt and completion token counts define the amount of model
work, so end-to-end latency can stay in request-time units while the throughput
benchmarks cover token-rate units.

For measured latencies `t_1..t_5`, `total_time_ms` is
`sum(t_i) / 5`, and `total_time_ms_stddev` is
`sqrt(sum((t_i - mean)^2) / 4)`. The submission schema and the derived metric
are in pipette-mgmt
[`docs/benchmarks.md` §2.3](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#23-end-to-end-latency).

## Batched

`parameter_batch_size` is the requests kept in flight at once, each on its own
slot of the one server. `1` is the reference procedure and what an entry that
omits the field means. Above `1` the same request is measured under load: the
benchmark id carries the count as `_bs<n>`, so two rows at different
concurrency cannot collide under one id.

**Sizing the server.** A batch that the server admits one at a time is not a
batch, so the slot cap is derived from the parameter rather than left to the
scheduler: vLLM `--max-num-seqs N`, SGLang `--max-running-requests N`,
llama-server `--parallel N`, OVMS `--pipeline_type LM_CB --max_num_seqs N`. On
vLLM and SGLang the flag is reserved in the cell's `raw` on the three timing
cells, for the same reason the token counts are: a cell restating the
concurrency could put the server at one count while the id promises another. On
llama-server `--parallel` stays out of the `raw` denylist so stored payloads
that carry it keep deserializing; the argv builder passes a raw `--parallel`
through with a warning at `N = 1` and refuses the typed-plus-raw pair above it.
`--max-model-len` (llama-server: `--ctx-size` per slot) still bounds one
sequence's prompt plus generation, which the batch does not change;
llama-server divides its `--ctx-size` across slots, so the harness multiplies
the per-cell default by `N` and each slot gets what a serial run gave the only
request. At `N = 1` the batch emits no slot flag; a cell that sets `no_cache`
renders `--parallel 1` itself ([no cache](../pipette-cli/llamacpp.md#execution-notes)).

**Measuring it.** Readiness is gated once per batch, never inside one: with
every slot busy a gate taken mid-batch would stall the slots it is supposed to
describe and measure the stall it caused. The warm-up puts a prompt on every
slot rather than one in total, so a batched server pays its per-slot
first-shape cost before repetition one rather than inside it. Each measured
repetition then submits `N` identical prompts together and times each **from
the submission to its own end**. llama.cpp sends them as one streamed
`/completion` request whose `prompt` holds `N` copies and reads each prompt's
end off its own final chunk; the torch-oai and OVMS servers take `N` separate
streamed requests, each timed from its own submission to its own finish
chunk.

**What the number means.** The repetition's value is the **per-prompt mean**
over those `N` readings, not the batch's wall clock: the wall clock grows with
the batch and describes the host, while `total_time_ms` stays what one caller
waited, measured while the host served `N` of them at once. The standard
deviation is across the five repetitions. The host's aggregate rate follows
downstream from the per-prompt figure and `parameter_batch_size`; it is not
counted here. Every prompt of a batch is validated against its own counts, so
one slot that came back the wrong shape fails the cell rather than being
averaged in.

**llama.cpp's prefill and decode.** `llama-bench` runs one sequence at a
time, so the llama.cpp CLI runs a batched `prefill_throughput` or
`decode_throughput` on `llama-batched-bench` instead: one process per
repetition that puts all `N` sequences through the model together and reports
one time for the batch. There is no server, so there are no slots to size and
no requests to time apart; the batch's time is the repetition's value. See
[prefill-throughput.md](prefill-throughput.md#llamacpp) and
[decode-throughput.md](decode-throughput.md#llamacpp).

**Engines that refuse it.** An engine that runs one sequence at a time refuses
a batched entry by name, before anything loads: `mlx_lm.server` as this client
drives it, the OpenVINO and ONNX GenAI in-house servers, and the in-process iOS
and Android engines (one KV cache, one conversation). Running such an entry
serially would file the reference measurement under an id that promises the
measurement under load, which is the one outcome this parameter exists to
prevent. Core AI also runs one sequence at a time, but it does not read
`parameter_batch_size`: a batched entry runs one request per repetition under
the batched ID (see [Core AI](#core-ai)).

## Exact-Token Prompt Synthesis

The benchmark needs a prompt that satisfies two constraints at the same time:
it must be ordinary text, so tokenizer work stays inside the measured request,
and it must have the requested token length under the runtime tokenizer. Fixed
prompt text produces different token counts across models because tokenizers
and special-token policies differ.

The implementation starts from a shared natural-language seed passage compiled
into each benchmark client binary from a published seed file
([`prompt_seed.txt`](../../crates/pipette-ops/src/prompt_seed.txt)), so the
exact prompts are reproducible from the repository. A prompt builder calls the
runtime tokenizer repeatedly until it finds text that tokenizes to exactly the
requested prompt-token count. Failure to find such a prompt invalidates the run
before any latency is recorded.

The builder is deterministic for a given tokenizer:

1. Tokenize the full seed text to estimate characters per token.
2. Repeat the seed text until the candidate is longer than the target.
3. Use bisection over character boundaries to find a prefix near the target.
4. Scan a small prefix window for an exact match.
5. If the prefix is still short, append simple suffixes such as spaces,
   digits, comma, or period until the target token count is reached.
6. Fail if no exact prompt is found within the tokenize-call budget.

The tokenizer callback must match the inference request's special-token policy.
The final prompt text may differ across runtime or model families; the
invariant is the prompt-token count reported by the runtime used for inference.

The benchmark uses raw completion-style prompts without chat templates or chat
messages.

## System Readiness Control

This section is the one statement of the readiness gate; the other timing
benchmarks gate the same way and link here.

The readiness probe runs on the host or device that can throttle the runtime.
For Docker, the readiness probe runs on the physical Linux host. Readiness
checks are outside the timed window.

Power delivery and cooling are fixed for a run rather than controlled by the
probe: phones run on mains power under active cooling, the MacBook Neo with an
added passive pad inside the case and an active pad under it
([Cooling on the MacBook Neo](device-conditions.md#cooling-on-the-macbook-neo)),
other machines on stock cooling and their normal supply. Both change the absolute numbers, so results
compare only across runs under the same setup; see
[Device conditions](device-conditions.md#active-cooling). The power state each
result ran under is recorded with it
([Recorded run environment](device-conditions.md#recorded-run-environment)).

Phones need stricter thermal handling than larger machines. Phones have limited
thermal mass, so one benchmark cell can leave enough residual heat to affect the
next cell. A phone can also start a run below the throttle point and cross it
during the five-request series, which shows up as high standard deviation or
drift across repetitions.

| Platform | Representative hardware | Default deadline | Thermal criterion | CPU criterion |
| --- | --- | ---: | --- | --- |
| Android (CLI) | Galaxy S25 Ultra (Snapdragon 8 Elite), Galaxy S26 Ultra | 5 min | `dumpsys thermalservice` reports thermal status `0` (`NONE`), the hottest real CPU-cluster thermal zone is `< 34C`, and an opt-in live HAL skin sensor is below its configured bound | Busy share from a 1-second `/proc/stat` delta is `< 0.30`. The IRQ-triggered check detects no held touch. |
| Android (app) | as above | 5 min | `PowerManager.getThermalHeadroom` is `<= 0.85`; where headroom reads `NaN`, thermal status is `THERMAL_STATUS_NONE` | none |
| iOS (app) | iPhone 17 Pro | 5 min | `ProcessInfo.thermalState` is `.nominal`, and the SoC die is `< 36C` (private-API build) or a calibrated IMU estimate is `< 38C` (distributable build); with neither, the state alone | Busy cores from a 1-second tick delta are `< 1.0` |
| Linux | x86 / ARM benchmark host | 5 min | Hottest readable `/sys/class/thermal/thermal_zone*/temp` is `< 70C`; a host that exposes no zone at all gates on load alone | First `/proc/loadavg` field divided by available CPU count is `< 0.30`; for a process confined to some of the host's CPUs, their busy share from a 1-second `/proc/stat` delta instead |
| Raspberry Pi 5 | Raspberry Pi 5 (BCM2712) | 5 min | No active-now bit (under-voltage, frequency capped, throttled, soft limit) in `vcgencmd get_throttled`, and the hottest zone is `< 80C` (the soft limit; hard is 85C) | As on Linux |
| Jetson Orin | NVIDIA Jetson AGX Orin (Tegra234) | 5 min | Hottest zone `< 70C` (the hot-surface-alert trip; throttle 99C, critical 104.5C); no `cpufreq-cpu*`, `devfreq-*gpu*` or `*-throttle-alert` cooling device engaged; no new `soctherm_oc` over-current event | As on Linux, plus GPU load `< 5%` where the kernel exposes it |
| macOS | MacBook Neo (A18 Pro); MacBook Pro (M5 Max) | 7 min | `OSThermalPressureLevel` is `0` (`nominal`), read from `com.apple.system.thermalpressurelevel`, and the die is `< 50C` where its sensors read | Busy cores from a 1-second Mach `host_statistics(HOST_CPU_LOAD_INFO)` tick delta are `< 1.0` |
| Windows | GMKtec EVO-X2 (Ryzen AI MAX+ 395); Core Ultra 7 258V; Zenbook S 16 (Ryzen AI 9 HX 370) | 5 min | No sensor within `15C` of the platform's ACPI `CriticalTripPoint`, or at `95C` or above where the box reports none; no active throttle-reason flag; summed GPU-compute utilization `< 5%`; each sensor the host's `[readiness.windows] ceilings` name at or below its ceiling. Each applies only where the box exposes it | Busy share of all logical CPUs from a 1-second kernel32 `GetSystemTimes` delta is `<= 40%` (the only always-required signal) |
| Other | | 0 sec | No probe | No probe |

The Android (CLI) row is the `pipette` CLI's gate running on the phone. The
Android app cannot read sysfs, procfs or `dumpsys` from its sandbox, so it
gates on the app-SDK `PowerManager` instead
([`Readiness.kt`](../../android/Pipette/app/src/main/java/ai/liquid/pipette/Readiness.kt)).
Its `0.85` headroom ceiling sits above the ~0.70 that a fan-cooled Pixel 10
Pro Fold settles at under continuous inference without throttling, so the gate
does not stall a cooled phone at its working equilibrium. The CLI row's `34C`
and busy-share criteria apply only where the `pipette` CLI itself runs on the
phone (started over `adb`).
The iOS app runs its own Swift peer of the macOS gate
([`ios/Pipette/Pipette/Readiness/`](../../ios/Pipette/Pipette/Readiness/Ios.swift));
see [iOS](#ios).

The Android CLI, Linux, Pi 5, Orin and macOS loops sleep 3 seconds between failed
readings, then retry until the criteria pass or the platform deadline is
reached; one log line per check. The interval is granularity, not criteria: it
sets only how soon a host that has already cooled is noticed to have done so.
The iOS app polls at the same 3 s and the Android app at 2 s. Where the CPU
sample already spends ~1 s inside that interval (~2 s if the tick counters did
not advance and it resampled), the loop tops the gap up rather than sleeping a
further 3 s on top. Windows runs its own loop every 5 seconds, after a probe
that itself takes about 1.5 s (the 1 s CPU sample, then PowerShell's
start-up); an idle box releases on the first
poll, so the 5 s only paces a box that is still busy or over a ceiling.

Probe I/O, parse, or unavailable-data errors fail immediately, with these
exceptions: on the generic Linux gate a host that exposes no `thermal_zone*` at
all gates on load alone (the Pi 5 and Orin gates fail there), and on every Linux
gate a process that cannot read whether it is confined, or the busy share of its
CPUs, reads the load average instead (see the Linux notes below);
on Windows a failed probe is retried, and three in a row end the wait. Threshold
comparisons are strict: a value equal to the threshold fails readiness, except
where the table says `<=` or "at or below" (the Windows CPU share and
ceilings, the Android app's headroom).

A host may replace a gate's temperature or load number with one measured for
its device type, in the same comparison; the keys and their rules are in
[Tuning the readiness gate](../pipette-cli/usage.md#tuning-the-readiness-gate).
A cell's `benchmark_flags.readiness` (`max_wait_secs`, `skip_thermal`, and Android
`skin_below_c`) or the matching CLI flags can change the deadline, set the skin
threshold, or waive the thermal criterion; the
load criterion still applies, and a run with the thermal criterion waived is not
comparable with a gated one ([The readiness gate](../pipette-cli/usage.md#5b-the-readiness-gate)).
The result records what the gate saw before each rep and the effective Android
skin threshold, where set, rather than the rest of the host configuration
([Per-rep readings](device-conditions.md#per-rep-readings)).

This polling is the only retry. The gate runs before each of the five measured
repetitions, and if the device never reaches the band before the deadline, that
repetition's gate returns a timeout that **aborts the whole cell**. The
remaining repetitions are not attempted, no partial or throttled samples are
recorded, and the run is stored as a benchmark error rather than a measurement.
There is no outer retry and no "N of 5 failed" count, because the cell stops at
the first readiness failure; a failed cell is simply absent from the published
results and visible as an error in the run log.

This trade-off has a bias. Heavier models heat the device more, so they are
likelier to hit the deadline, and the gate can drop more of their *unfavorable*
(throttled) samples. Two things keep that bias from selecting favorable
results. The readiness band is identical for
every model (nothing model-specific tightens or loosens it), and failures are
recorded as errors rather than discarded silently, so the readiness-failure rate
is observable per cell instead of hidden. A model that only ever produces fast
numbers by failing the gate often shows that pattern in its error count, which
should be read alongside its latencies.

These thresholds are readiness bands, not throttle limits. The desktop and
server ceiling of 70C sits above a typical 30-50C idle baseline while staying
well below the points where hardware actually throttles (around 95-100C on x86
and ~110C on ARM SoCs), so it admits a cooled-down box without waiting for
ambient. The 0.30 CPU criterion requires the machine to be under roughly 30%
busy, which clears an idle floor that is normally well under 10% while still
catching another benchmark that is already running. The Android thermal band is
deliberately tighter and is sourced separately below.

The `Other` row applies no thermal or load gate, so latencies from a no-probe
platform are not controlled for device state. Treat them as advisory: they
should not be published or compared alongside controlled-platform results
without that caveat.

### macOS

The gate requires three things: the thermal-pressure level `nominal`, the die
`< 50C` where its sensors read, and fewer than one busy core.

The thermal-pressure enum is not a temperature signal. On a MacBook Neo it is a
fixed hold-off of about 318 s from the moment the CPU goes quiet, whatever the
load or cooling; the measurements are in
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md). The 7-minute
deadline is sized so that hold-off fits inside it; a deadline below it would
time out and fail the cell after any repetition that warmed the machine. The
gate keeps the enum because it is conservative, not because it is a good
signal.

The die criterion was measured on the M5 Max, where the enum engages only after
minutes of continuous load and a batch heats underneath it; see
[MacBook Pro (M5 Max) thermal behavior](macbook-m5-thermal-behavior.md) for the
measurements behind `50C`. Die temperature is read through the same private
IOHID sensors iOS uses and recorded with every result
(`device_apple_soc_temp_c_before` / `_after`). An unreadable die passes the
gate's own `50C`; a die bound the host set itself holds the gate instead.

The 1-minute load average stays inflated by background work on a developer
Mac, so the CPU probe samples an instantaneous busy ratio from Mach tick
counters 1 second apart, over the user, system, idle and nice counters:

```text
busy = (total_delta - idle_delta) / total_delta
```

If those counters do not advance across the window (the one degradation ever
measured, and only at windows shorter than the 1 s used here) the probe
resamples once and then reports the cell unready rather than guessing a ratio.
The budget is in busy cores rather than a share because a share does not
survive core count: on an 18-core M5 Max an all-core 0.30 lets a process pinning
a whole core through.

### Windows

Windows has no relative temperature test. Measured on 2026-09-30, waiting for
every temperature counter to go flat moved no GMKtec EVO-X2 cell by more than
0.2% and took 3.4 times as long, and it missed the one effect that does move
numbers: the Zenbook S 16's firmware power cut, which follows a slow ACPI zone
(`\_TZ.THZ0`) that a flatness test reads as settled while the cut is still on.
The next prefill ran 25% slow.

So temperature holds the gate only where a host's configuration says so:
`[readiness.windows] ceilings` names sensors to wait on until they read at or
below a measured ceiling (`"acpi:THZ0" = 43` on the Zenbook S 16, where the
next prefill then ran at full speed). No single ceiling would fit the fleet: the
GMKtec rests at 33-36C and saturates at 98C, the 258V rests at 42-46C and
saturates at 55C. Counters read, and printed on every poll, are
`\EsifDeviceInformation(*)\Temperature` (Intel Dynamic Tuning, already Celsius),
every `\Thermal Zone Information(*)\Temperature` zone by name (Kelvin,
converted with `- 273`), and the GPU's own temperature from
`D3DKMTQueryAdapterInfo`. Which one is live is a property of the box: ESIF is
absent on the AMD boxes, and on the 258V the ACPI zone reads a constant 301 K,
so every one is read and reported and none outranks another.

One absolute check holds regardless: a reading within `15C` of the platform's
own ACPI `CriticalTripPoint` (measured `110C` on the GMKtec, `105C` on the
258V), read from `MSAcpi_ThermalZoneTemperature`. That trip is
hardware-declared and means the same thing on every chassis.

CPU busy comes from kernel32 `GetSystemTimes`, sampled 1 second apart before
the PowerShell call that reads the other signals, so the interpreter's start-up
is not in the window. Kernel time already includes idle time, so the same delta
form applies with `total = kernel + user`:

```text
busy = (total_delta - idle_delta) / total_delta
```

Busy at or under `40%` (or the host's `max_cpu_busy_pct`) is the only
always-required signal; everything else is evaluated only when the box exposes
it. Two other CPU counters were rejected. `Win32_Processor.LoadPercentage` read
`1-19%` on the 258V under a 16-job burn against `1-6%` idle, so its load and
idle bands overlap. PDH's `\Processor Information(_Total)\% Processor Time`
refused 10-20% of idle samples with `PDH_CALC_NEGATIVE_DENOMINATOR` and read up
to `47%` on an idle GMKtec. GPU-offloaded inference barely moves CPU load, so
the GPU-compute signal carries the real workload check: idle is `0%`, active
Vulkan inference sits at `71-105%`, and the gate requires summed utilization
`< 5%`. A non-zero processor throttle-reason flag also holds readiness, though
that flag has never been observed to fire on either box, including while the
GMKtec die sat pinned at its own limit. A box with no temperature counter at
all gates on load alone. See
[`pipette_readiness::windows`](../../crates/pipette-readiness/src/windows.rs).

### Android

The CLI gate uses both the OS thermal-status enum and raw CPU-cluster
temperature. The OS enum is a power-management signal and stays nominal even
while Snapdragon 8 Elite CPU dies reach 75-80C. The raw temperature gate reads
`/sys/class/thermal/thermal_zone*/{type,temp}`, keeps real CPU-family zones
whose `type` matches `cpu-<digit>...` or `cpullc-<digit>...`, and requires the
hottest readable zone to be below 34C. It runs before every measured request,
so if the device reheats during a cell, the next measured request waits for the
device to return to the band or the cell fails at the deadline.

This sensor choice is deliberate. The Android text-generation path measured
here runs llama.cpp on CPU cores, so the relevant thermal budget is the CPU
clusters' ability to hold frequency during prefill and decode. Android exposes
many thermal zones: CPU core or cluster zones, CPU low-level cache zones,
GPU/NPU/modem zones, battery, skin, camera, and board sensors. Battery and skin
temperatures are useful device-level safety signals, but they respond more
slowly than the CPU die. GPU, NPU, modem, and camera sensors describe other
blocks. CPU-family zones are closest to the cores doing the measured work and
react quickly enough to catch residual heat from the previous cell.

The implementation therefore keeps CPU-family zone names such as `cpu-0-0-0`
and `cpullc-1-1`, then gates on the hottest readable value. It deliberately
excludes pseudo-zones such as `cpu-hw-trip-*`, which report fixed hardware trip
points around 105C instead of live die temperature and would otherwise keep a
cool phone waiting forever. The hottest cluster is used because one overheated
cluster can still cause scheduler or frequency changes that affect request
latency.

The 34C threshold is a readiness band, not a safety limit. It sets how much
thermal budget the next repetition gets before the 75-80C performance-throttle
band. On S25-class phones a 50C band left 20 to 40 s repetitions at ±30 to 35%
standard deviation, and 35C brought most of them to ±1 to 8%. A lower band
holds the gate at the idle temperature: a tethered, charging phone sustains
about 32C, and a 32C gate stalled three phones there for minutes. The strict `< 34C` releases at 33C,
about 1C above that floor. Cooling from a typical post-rep peak of ~75C to 34C
takes about 30 to 40 s. Very long repetitions can still overrun the available
budget; high standard deviation on those shapes remains an invalid or follow-up
signal rather than evidence that the cooldown is unnecessary.

The CLI gate also holds a rep while the touchscreen has a held touch. A
screen cooler can cause a "phantom touch" that sends a continuous stream of
input events, and the phone processes them during the measurement. The touch
stays until an operator removes the cause. During the 1-second CPU sample, the
gate counts the interrupts of the touch controller in `/proc/interrupts`
(`synaptics_tcm` on the Pixel 11 Pro, `synaptics_ts` on the Galaxy S26 Ultra).
At 20 or more per second, it reads `Last Cooked Touch: pointerCount` for each
touchscreen in `dumpsys input`. It also reads the pointer count if the summed
IRQ count decreases, which can happen when a CPU goes offline, or if a known
controller's count is malformed. A pointer count
above zero is a held touch. Both phones read 0 interrupts per second at idle.
This check does not detect a held touch
below 20 interrupts per second or on a controller with another IRQ name. The
contact rate has not been measured on these phones. The host selects if the
gate waits for the touch to clear, fails the cell, or disables touch detection
with `[readiness.android] held_touch = "off"`. Disabling it allows a rep to
start with a held touch; thermal and CPU criteria still apply
([The readiness gate](../pipette-cli/usage.md#5b-the-readiness-gate)).

CPU busy is calculated from the aggregate `cpu` line in `/proc/stat`:

```text
busy = (total_delta - idle_delta) / total_delta
idle = idle + iowait
```

The probe samples a short instantaneous busy ratio rather than a load average
because the 1-minute load average on a phone stays inflated by background work
(the resting normalized load sat around 0.35-0.45 from One UI's background-AI
stack and `adb` traffic), so a lagging average stays above the threshold even
when the device is otherwise idle. An instantaneous `%busy` reads near the true
resting floor and only crosses the threshold when something is actively using
cores.

### Linux

Linux keeps the normalized 1-minute `/proc/loadavg` for a process that may run
on every online CPU, because a dedicated benchmark host has a near-zero idle
baseline, so the background inflation that disqualified the average on phones
does not occur. The read is also free of a sampling window, and the 1-minute
smoothing ignores sub-second housekeeping spikes. The cost is that the average
lags (after a previous cell it decays over about a minute), but that lag is
conservative: it makes the gate over-wait for residual load to clear rather
than release into it. This choice is therefore sound for dedicated Linux hosts;
it would not be appropriate on a shared or background-busy machine, where an
instantaneous probe should be used instead.

The gate uses one for the shared case it can detect: a process confined to some
of a host's CPUs, as a SLURM job's cpuset, `taskset` or `docker --cpuset-cpus`
confines it. That is read from `Cpus_allowed_list` in `/proc/self/status` naming
some of the CPUs in `/sys/devices/system/cpu/online` and not all of them. For
such a process the load average is the wrong quantity, not a lagging one: it
counts runnable tasks on the whole host and is divided by only the CPUs the
process may use, so other tenants' work on other CPUs reads as load. Measured on
2026-10-11 from inside a 2-CPU job on a 192-CPU SLURM worker that other jobs
shared, `/proc/loadavg` read 9.87 while the job's own two CPUs were 0% busy
over a second and the whole host 4.4%. Over a 16-CPU job that is 62%, which held
every wait there to its deadline. A confined process therefore reads the busy
share of its own CPUs, from their per-CPU lines in `/proc/stat` sampled 1 second
apart and summed:

```text
busy = 1 - idle_delta / total_delta
idle = idle + iowait
total = user + nice + system + idle + iowait + irq + softirq + steal
```

The same `0.30`, or the host's `cpu_load_below_pct`, holds either reading, and
the poll's line says which it was: `cpu=busy:N%` for the busy share,
`cpu=load:N%` for the load average. If the counters do not advance across the
window the probe resamples once. A process that cannot read its confinement or
those counters reads the load average and logs that once per process: at info
level under a sandbox whose `/proc/self/status` has no `Cpus_allowed_list`
(gVisor), where nothing says the process is confined, and as a warning
otherwise. The allowed CPUs have to be among the online ones to be sampled. A
`/proc` that numbers a container's CPUs from 0 while the mask keeps the host's
ids (lxcfs) is not supported, so a mask that holds a CPU outside the online
list without holding the whole list counts as a confinement that cannot be
read. A cgroup CPU quota (`cpu.max`) on a process that may still run on every
CPU is not confinement here. The Raspberry Pi 5 and Jetson Orin gates read CPU
load the same way.

Linux separates a missing thermal subsystem from a broken one. A rented VM
commonly exposes `/sys/class/thermal` with cooling devices and no `thermal_zone*`
under it; the rented H100 hosts that we measured are that shape, with 26
cooling devices and zero zones. Such a host has no temperature to compare against the `70C` ceiling, so
enforcing the ceiling there does not measure a cool host: it fails every cell
before the first sample. The gate therefore treats an absent zone set as a
criterion that does not apply, records the poll as
`thermal=hottest:unreadable[NO SENSORS]`, and holds the load criterion alone,
which is how Windows reads a box with no temperature counter. A host whose
table sets `hottest_below_c` fails the wait there instead. Zones that exist and
do not read are the other case and still fail: there the host's temperature is
unknown rather than nonexistent. An ungated run is not comparable with a
thermally gated one, and what records which it was is the run's own thermal
telemetry, empty of zones for such a host.

On a Raspberry Pi 5 the firmware's `vcgencmd get_throttled` signal covers what a
temperature threshold cannot, including under-voltage, which is how an
undersized power supply shows up. Only the active-now bits count (mask `0xF`:
under-voltage, ARM frequency capped, currently throttled, soft temperature
limit): the since-boot bits are sticky until a reboot and would hold the gate forever once
the Pi had ever throttled. If `vcgencmd` is missing or `/dev/vcio` is
unreadable, the gate degrades to the temperature bound alone. That bound is the
`80C` soft limit: the Pi 5 throttles from 80C up to its 85C hard limit, and the
5-series has no `temp_soft_limit` setting to move it. See
[`pi5.rs`](../../crates/pipette-readiness/src/linux/pi5.rs).

A Jetson Orin throttles at its 99C passive trip, so the `70C` bound is its
hot-surface-alert trip, which is bound to nothing that cuts a clock. The three
`cv*` thermal zones fail their read with `EAGAIN` and are skipped. Inference
runs on the GPU, where CPU load barely moves, so the gate also reads the GPU's
busy share (sysfs `load`, permille); a kernel that does not expose it leaves GPU
load ungated, with one warning. Any CPU-frequency, GPU-devfreq or
throttle-alert cooling device with a non-zero `cur_state` holds the gate, as
does a `soctherm_oc` over-current counter (on a channel with throttling enabled)
that grew since the previous poll: the counters are cumulative since boot, so
only a change is an event. The previous poll may be the last one of the wait
before, so an event during a rep holds the next wait. Neither is waived by
skipping the thermal criterion. See
[`orin.rs`](../../crates/pipette-readiness/src/linux/orin.rs).

## Runtime Details

The general method is implemented across these runtime paths:

| Path | Serving process | Readiness probe |
| --- | --- | --- |
| macOS llama.cpp | local `llama-server` | macOS host |
| macOS MLX | local `mlx_lm.server` | macOS host |
| Windows llama.cpp | local `llama-server` | Windows host |
| Linux llama.cpp | local `llama-server` | Linux host |
| Android llama.cpp (CLI) | `llama-server` launched by the `pipette` CLI on the device, driven over `adb` | Android device (CLI gate) |
| Android app | in-process llama.cpp (`kotlinEngine` or `rustEngine`) | Android app gate |
| Linux torch-oai Docker | OpenAI-compatible server in a Docker container | Linux host running the container |
| Linux torch-oai uv | OpenAI-compatible server launched by `uv` | Linux host running the process |
| OVMS | local OpenVINO Model Server | host that runs the server |
| iOS llama.cpp | in-process `llama.cpp` (no server) | iOS app gate (see [iOS](#ios)) |
| OpenVINO | local in-house server (`pipette_openvino_server.py`) | host that runs the server |
| ONNX GenAI | local in-house server (`pipette_onnx_genai_server.py`) | host that runs the server |
| macOS Core AI | local Swift sidecar over Apple's Core AI framework (see [Core AI](#core-ai)) | macOS host |

### OpenVINO and ONNX GenAI

Both runtimes run pipette's in-house OpenAI-compatible server, one per cell,
and are measured like torch-oai: the prompt is sized through the server's
`/tokenize`, and each repetition is one streamed `/v1/completions` with
`temperature = 0` and `ignore_eos`, timed from the send to the finish chunk.
Both `usage` counts must match the cell and `finish_reason` must be `"length"`.
The server loads (and OpenVINO compiles) the model before it listens, so that
stays outside every request. Neither server applies a chat template; both keep
the special tokens their tokenizer adds (a BOS where it prepends one), as MLX
does, and `/tokenize` encodes the same way, so the sized prompt is the prompt
the completion counts. Warm-up and the server path are in
[OpenVINO measurement](../openvino-measurement.md) and
[ONNX GenAI measurement](../onnx-genai-measurement.md).

### llama.cpp

`pipette-llamacpp` starts `llama-server` and waits for `/health`. Prompt
construction uses `/tokenize`; each repetition is one streamed `/completion`
request with:

- `prompt`: the exact-token text prompt, `parameter_batch_size` times.
- `stream: true`.
- `temperature: 0.0`.
- `n_predict`: `parameter_decode_tokens`.
- `cache_prompt: false`.
- `ignore_eos: true`.

The `/tokenize` request body is:

```json
{ "content": "<prompt text>", "add_special": true }
```

`add_special: true` keeps the pre-flight token count exactly in sync with the
count `/completion` reports back: it matches the inference path on tokenizers
that auto-prepend a BOS (such as Gemma and LFM2) and is a no-op on models that
do not (such as Granite and Qwen2). A mismatch here would fail the run on the
token-count check.

Each prompt is timed from the request's send to its own final chunk, the one
with `stop: true`, matched by its `index`. From b9425 llama-server opens every
prompt after the first with a `data: null` event, which the client skips. The
final chunk must report:

- `timings.prompt_n == parameter_prefill_tokens`.
- `timings.predicted_n == parameter_decode_tokens`.
- `stopped_limit == true`, or `stop_type == "limit"`.

`llama-server` samples the last token without decoding it, so a request of N
tokens runs N - 1 decode steps after the prefill.

Each request asks for its own connection (`Connection: close`), so every
repetition's time includes a new loopback connection, 13 to 18 ms of it on
Windows: `llama-server` closes a streamed reply's connection although its
headers keep it alive, and reusing one fails the next repetition.

For EOS handling, the runner sets `ignore_eos: true`, under which
`llama-server` biases every end-of-generation token to never be sampled, so
models that would normally stop early still generate the requested decode
count. Builds b7508 to b8717 ignore the flag (fixed upstream in PR #21203); on
those, a model that stops early fails the `stop_type == "limit"` check rather
than reporting a short run.

The server command carries `--no-warmup`, a `--ctx-size` derived for prefill
plus decode unless the cell sets `ctx_size`, and `--load-mode none` unless the
cell sets `load_mode` or `mmap`. A cell that sets either keeps it in its
effective runtime flags, which must match across compared runs. The load modes
are in [llamacpp.md](../pipette-cli/llamacpp.md#execution-notes), the reserved
server flags in [Per-cell flags](../pipette-cli/models-and-runtimes.md#per-cell-flags).

**Like-for-like with `llama-bench`.** A plan that wants its rows comparable
with `llama-bench`, which the prefill and decode benchmarks run, sets
`no_cache = true` (what it renders:
[llamacpp.md](../pipette-cli/llamacpp.md#execution-notes)). Every request
already sends `cache_prompt: false`, so here it changes the context checkpoints
and, at `N = 1`, the slot count.

The checkpoints are what move the reading. On a hybrid or sliding-window model
llama-server saves the state it cannot roll back while it prefills, decoding
the prompt's last tokens in batches of their own to do so, and reads a saved
state back only to reuse a cached prompt. `llama-bench` saves none. On
llama.cpp b10216 (Metal) at a 1024-token prompt and 64 decode tokens, turning
them off lowers `total_time_ms` by 19.1 ms (2.7%) on gemma-4-E2B, 10.2 ms
(2.0%) on Qwen3.5-2B and 3.5 ms (2.0%) on LFM2.5-VL-450M, the last at the edge
of run-to-run noise; the dense InternVL3.5-2B does not change. At `N = 4` it
lowers the per-prompt `total_time_ms` by 17.6 ms (1.2%) on Qwen3.5-2B and
10.9 ms (0.6%) on gemma-4-E2B. Each percentage is of the reading with them on.
The recorded memory drops by the checkpoints that the server does not hold:
38 MiB (two of 19.3 MiB) on Qwen3.5-2B, and at `N = 4`, 196 MiB on Qwen3.5-2B
and 46 MiB on gemma-4-E2B.

One slot instead of llama-server's default four lowers the recorded memory by
61 to 63 MiB on Qwen3.5-2B and moves the timings by at most 3 ms, within
run-to-run noise, on all four models. Some stored `no_cache` rows ran four
slots ([no cache](../pipette-cli/llamacpp.md#execution-notes)).

### MLX

`pipette-mlx` starts a local `mlx_lm.server` with `--prompt-cache-size 0`.
Prompt construction counts candidates with the venv's tokenizer helper;
measured requests stream `/v1/completions` with:

- `prompt`: the exact-token text prompt.
- `max_tokens`: `parameter_decode_tokens`.
- `temperature: 0.0`.
- `logit_bias`: every tokenizer EOS id at `-100.0`, so the server emits the
  full count instead of stopping early.

The stream must report `finish_reason = "length"`, and its usage trailer:

- `prompt_tokens == parameter_prefill_tokens`.
- `completion_tokens == parameter_decode_tokens`.

### torch-oai

`pipette-torch-oai` runs an OpenAI-compatible engine such as vLLM or SGLang.
Prompt construction uses the engine's `/tokenize`; measured requests stream
`/v1/completions` with:

- `prompt`: the exact-token text prompt.
- `max_tokens`: `parameter_decode_tokens`.
- `temperature: 0.0`.
- `ignore_eos: true`.

The `/tokenize` request body includes:

```json
{
  "model": "<model name>",
  "prompt": "<candidate prompt text>",
  "add_special_tokens": true
}
```

The stream must end with a `usage` trailer (`stream_options.include_usage`)
reporting `prompt_tokens` and `completion_tokens` equal to the benchmark shape,
and `finish_reason = "length"`, the same stopped-at-limit check the llama.cpp
path makes on `stop_type`. If the trailer is missing, the run fails.

`pipette-torch-oai` disables server-side prefix reuse at launch for **every**
benchmark type (not just latency), because they all measure cold prefill plus
decode, so a reused cache would silently turn any of them into a warm-prefix
measurement. vLLM launches add `--no-enable-prefix-caching`; SGLang launches add
`--disable-radix-cache`. A cell that sets `prefix_caching = true` is refused.

### Core AI

The Core AI runner (`pipette-coreai`, macOS 27 or later) drives a Swift sidecar
over local HTTP. The sidecar loads the model before it prints its ready marker,
so load and specialization are outside every window
([Core AI specialization cache](coreai-specialization.md)). The runner gates
readiness once before it starts the sidecar, then sends one untimed warm-up
request at the cell's shape and checks its counts. Each measured repetition
then runs in this order:

1. The readiness gate.
2. An untimed `POST /prepare`, which resets the KV cache and waits 50 ms.
3. `POST /end_to_end_latency` with `prompt_tokens` and `decode_tokens`, timed
   on the client's clock from the send to the arrival of the whole JSON reply.
   The reply is not streamed.

The Core AI path differs from the server paths in two ways that affect
comparability:

- The prompt is `parameter_prefill_tokens` deterministic pseudo-random token
  ids, with no EOS id among them, not text. No tokenizer runs in the window.
- The sidecar generates greedily up to `parameter_decode_tokens` tokens and
  does not mask end-of-generation tokens. A generation that stops early fails
  the count check.

The reply must report `prompt_tokens` and `completion_tokens` equal to the
cell. The runner does not read `parameter_batch_size`, so a batched entry runs
one request per repetition under the batched ID.

### iOS

The iOS app runs `llama.cpp` in process rather than launching a server, in
native Swift over the llama.cpp C API, and its latency path differs from the CLI
and server runtimes in ways that affect comparability:

- The prompt is built from the same seed passage as the CLI runners, by this
  client's own synthesizer: `PromptSeed.buildPromptText` tiles
  `PromptSeed.promptSeedText` (the bundled copy of
  `crates/pipette-ops/src/prompt_seed.txt`) and tokenizes to exactly
  `parameter_prefill_tokens` tokens. The context window
  (`parameter_prefill_tokens + parameter_decode_tokens` for latency) is derived
  from the benchmark shape by `LlamaRuntimeFlags.ctxSize(for:)`, and the cell
  loads the model with it, so the measured prompt is always the full requested
  prefill length rather than a slice of a context sized for something else. A
  cell that names its own `ctx_size` in `runtime_flags` gets that one instead,
  which is the one way the two can disagree.
- Each measured repetition waits on the app's own readiness gate (the iOS row
  of the table above) rather than the Rust `pipette_readiness` crate, so heat
  from the previous repetition has to clear before the next one.
- Before each repetition (warmup included) the app resets the llama context and
  sampler. The timed window then tokenizes the prompt, prefills it, and
  generates exactly `parameter_decode_tokens` tokens greedily with every
  end-of-generation token masked, as `ignore_eos` asks `llama-server` to. It
  stops where `llama-server` stops a request: after sampling the last token,
  without decoding it, so a repetition of N tokens runs N - 1 decode steps
  after the prefill, as the CLI's request does. It runs one warmup repetition
  and five measured repetitions and reports the same `total_time_ms` mean and
  sample standard deviation as the other runtimes.

iOS ONNX GenAI also stops after the last sample. iOS MLX samples one token from
the prefill and then runs N decode steps, so on one device an iOS MLX row reads
about one decode step higher than an iOS llama.cpp row of the same shape.

Stored iOS llama.cpp `end_to_end_latency` rows whose `client_version` is dated
before 2026-10-17 also decode the last token. Those rows read one decode step
higher: the deepest step, at a depth of the prompt plus
`parameter_decode_tokens` - 1. The
`decode_time_ms / parameter_decode_tokens` of a same-model `decode_throughput`
cell, or a little more, gives the size of that step: well under 1% of
`total_time_ms` for the 256-token cells.

**Thermal detection on iOS.** iOS exposes no public SoC temperature API, and the
public `ProcessInfo.thermalState` enum can stay `.nominal` while the SoC has
already down-clocked. Published iOS results therefore come from the
[private-API build](../pipette-ios/private-thermal-release-build.md) of this app (`PIPETTE_PRIVATE_THERMAL`,
`PIPETTE_PRIVATE_MEMORY`), which reads the exact SoC die temperature. App Review
rejects such a build, so it runs only side-loaded on registered development
devices; anyone can make the same build from this repository, and the
distributable build falls back to a calibrated IMU-drift estimate. The full
explanation is in
[iOS results and the private-API build](README.md#ios-results-and-the-private-api-build).

Because iOS gates with its own client and runs an in-process engine, iOS latency
results should be compared only against other iOS results under the same
cooling conditions, not against the CLI or server paths.

## Result Validity and Comparability

A run is invalid if warmup or any measured request reports the wrong
prompt-token count, the wrong completion-token count, missing usage or timing
data, unexpected early stop, readiness timeout, or readiness-probe failure.

Every request's token counts are checked against the cell, and a mismatch
fails the run, so a recorded result used exactly the counts its benchmark
names. The run log records each repetition's sample:

```text
end-to-end reading cell="end_to_end_latency_512_256 rep 1/5" prompt=1 prompts=1 total_ms=123.456 prefill_tokens=512 decode_tokens=256
```

To compare `end_to_end_latency` results:

- Use the same benchmark ID, model, runtime, quantization, and relevant runtime
  flags.
- Keep the prompt as text so tokenizer cost remains included.
- Compare by exact prompt-token and completion-token counts; prompt bytes may
  differ across tokenizers. This equalizes the compute *shape* (same token
  counts), not the *content* delivered: at equal token counts a model with a
  denser tokenizer carries more actual text per token, so an equal-token
  comparison slightly favors the model with the more efficient tokenizer. The
  effect is inherent to token-based comparison; where it matters, read latency
  alongside each model's chars-per-token rather than in isolation.
- Treat any prompt or completion token mismatch as invalid data.
- Use the mean and sample standard deviation together; a large standard
  deviation means the run is noisy even when the mean looks plausible.
- Compare only rows from the same client measurement path: a change in how a
  client times or drives a request moves its rows, and `client_version` tells
  the two sides apart.

The result artifacts preserve enough context to audit the run later:

- Benchmark ID and resolved benchmark parameters.
- Model name, quantization, runtime name, and runtime version.
- Effective runtime flags when the runtime exposes configurable flags.
- Command preview for the runtime request or server command.
- Per-repetition output lines with elapsed time and token counts.
- Captured benchmark stdout or stderr when the runner records it for the
  benchmark.

## Code References

The shared Rust implementation for repetition count, gating, per-repetition
bracketing, timing, mean, and sample standard deviation is
[`pipette_ops::measurement`](../../crates/pipette-ops/src/measurement/mod.rs).

The per-repetition thermal readings it brackets each repetition with are paired
by
[`pipette_ops::thermal_series`](../../crates/pipette-ops/src/thermal_series.rs).

The shared exact-token prompt builder is
[`pipette_ops::prompt_seed`](../../crates/pipette-ops/src/prompt_seed.rs).

The platform readiness probes are implemented in
[`pipette_readiness`](../../crates/pipette-readiness/src/lib.rs).

The Core AI runner is
[`pipette-coreai execute::end_to_end_latency`](../../crates/pipette-coreai/src/execute/end_to_end_latency.rs),
and its sidecar is
[`pipette-coreai-sidecar`](../../crates/pipette-artifacts/swift/Sources/pipette-coreai-sidecar/main.swift).
