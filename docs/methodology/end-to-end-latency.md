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

The scope is local full-request latency for text-only completion. The same
measured requests also yield time-to-first-token (TTFT) (the caller-visible
delay before the first generated token), which the management service surfaces
as a derived metric on these end-to-end latency rows; it is a by-product of this
benchmark, not a separately gated measurement. The per-phase rates are measured
by the sibling [prefill-throughput](prefill-throughput.md) and
[decode-throughput](decode-throughput.md) benchmarks.

## Controls at a Glance

End-to-end latency is sensitive to many factors unrelated to model quality.
Each known source of measurement bias is controlled as summarized below; the
rest of this document is the detailed implementation of these controls.

- **Tokenizer differences across model families**: comparisons use exact
  prompt-token and completion-token counts under each runtime's own tokenizer,
  not prompt text or byte length. A token-count mismatch on warmup or any
  measured request invalidates the run.
- **Early stop on EOS**: end-of-generation tokens are suppressed so every
  measured request decodes exactly the requested number of tokens. Fixing the
  decode length holds every compared run to the same amount of work.
- **Prefix and KV reuse across repeated requests**: server-side prefix reuse
  is disabled (llama.cpp `cache_prompt: false`, MLX with no prompt cache,
  vLLM/SGLang with prefix caching off), so each measured request pays the full
  prefill cost and the engines stay comparable when a prompt repeats.
- **Thermal state and throttling**: each measured request is gated by a
  readiness probe that combines a temperature signal with a CPU busy signal.
  Phones are benchmarked on mains power and with aluminum heat sinks so a
  multi-run session stays repeatable instead of drifting slower as the device
  heat-soaks; the cooling reduces run-to-run variance, it does not advantage any
  one model. On iOS the gate runs on an internal thermal-aware build that reads
  the real SoC die temperature, because the public thermal state is too coarse
  to catch throttling. Because the power and cooling setup changes the absolute
  numbers, phone results are only comparable to other runs collected under the
  same setup.
- **Cold start and warmup**: model load, server startup, readiness polling,
  and one warmup request happen before timing begins and are excluded from the
  reported latency.
- **Scheduler noise and per-request variance**: each cell runs five measured
  requests and reports the mean alongside the sample standard deviation, so an
  unstable run is visible in the result rather than hidden in the headline
  number.

## Measurement Challenges and Controls

Input shape, runtime state, and device state all affect end-to-end latency. The
benchmark controls those sources of bias directly in the measurement method.

Tokenizers differ across model families, runtime implementations, and
special-token policies. The same prompt text may be 512 tokens for one runtime
and a different length for another. The benchmark therefore uses exact
prompt-token count and exact completion-token count as the comparison key. Each
run builds the prompt with the tokenizer path that matches the inference
request.

Sending token IDs directly would bypass server-side tokenization and measure a
narrower prefill-plus-decode path. The benchmark sends prompt text so tokenizer
cost remains inside the timed request.

> **Implementation status.** The server-backed runners (llama.cpp / MLX CLIs)
> and the iOS in-process engines tokenize inside the timed window as described.
> The in-process **Android** kernel (`native/benchmarks.rs`) still times
> prefill + decode only: its e2e excludes tokenization until PIP-256 lands, so
> Android e2e is not yet directly comparable to the CLI / iOS figures.

Serving lifecycle can also change what is being measured. Some runners own a
local server process, while others attach to a Docker or uv process. Prompt
synthesis can require the live serving endpoint because tokenizer accounting
has to match the inference endpoint. The benchmark allows runtime-specific
startup order while keeping one metric boundary: endpoint readiness, prompt
synthesis, warmup, and platform readiness happen outside measured requests.

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

Setup and warmup can distort request latency. The benchmark performs runtime
setup, model load, server startup, readiness polling, and one warmup request
before measurement, then excludes that time from the reported latency
statistics.

Thermal state and background load can dominate latency, especially on phones,
laptops, and small desktops. A device that is still cooling down from a
previous run may start fast and then throttle during the measured request
series. External cooling and the device's power source also change the
experimental condition: cooling increases the thermal budget available to the
model, and running on a charger sized above the chip's sustained draw keeps the
device from being power- or charge-throttled mid-series. The benchmark treats
thermal state, CPU load, power delivery, and cooling setup as part of the
measurement environment. It uses platform readiness checks outside the timed
window and records the power and cooling setup when it differs from stock. If
the host or device fails to reach the readiness criteria, the run fails before
recording latency under unstable conditions.

Individual requests can still be affected by scheduler noise, transient runtime
state, or background work. The benchmark measures five requests and reports both
the mean and sample standard deviation. The primary defense against per-request
variance is the readiness gate rather than the repetition count: because each
measured request first waits for the thermal and CPU-load criteria, every sample
starts from a comparable controlled state. The reported value is the plain mean
of the five; no trimming, outlier rejection, or median; the standard deviation
is what exposes a sample that the mean alone would hide. Five is then a
deliberate trade-off; on Android phones and CPU-only laptops each gated
repetition is slow, since the readiness wait and the decode work both run on the
constrained device, so repetitions are expensive in wall-clock time. Five is
enough that a single throttled request surfaces as an inflated standard
deviation instead of passing unnoticed. The standard deviation is therefore part
of the result, not a footnote: a high value means the mean is not a stable
estimate and the run should be re-read or rerun.

## Measurement Protocol

The benchmark has a shared measurement contract. Before prompt synthesis and
measurement, every runner resolves the requested token counts, makes a serving
endpoint available, and passes platform readiness. Runtime startup order can
vary because some runners own the serving process and some attach to a process
started by Docker or uv. The current llama.cpp and MLX runners gate host or
device readiness before launching their local process. The torch-oai runner
launches the Docker or uv server first, waits for its readiness URL, then gates
host readiness inside the end-to-end latency runner.

After the setup conditions pass, the shared measurement phase is:

1. Build a text prompt that tokenizes to exactly the requested prompt-token
   count under the runtime tokenizer used for inference.
2. Run one warmup request at the full benchmark shape (the same prompt and
   decode count as the measured requests) and validate its prompt-token and
   completion-token counts. Warming at the real shape is what makes the first
   measured repetition comparable to the other four: runtimes select and compile
   kernels per tensor shape, so a lighter warmup leaves that cost to be paid
   inside repetition one. It costs one extra full-size request per cell. The
   warmup is excluded from the reported statistics, and the readiness check runs
   before the measured repetitions, not before the warmup.
3. For each of five measured repetitions, run the platform readiness check.
4. Submit one measured request with the same prompt and decode count.
5. Validate prompt-token and completion-token counts for each measured
   request.
6. Report the mean and sample standard deviation of the five measured request
   latencies.

The timed request window starts immediately before each measured inference HTTP
request and ends when that request returns successfully. Readiness waits,
warmup, server startup, model load, prompt construction, token-count validation
after the response, and result sync are outside that window. The readiness
check runs before every measured repetition, including the first one after
warmup, so any heat generated by setup or warmup must clear before a measured
latency is recorded. Warmup token counts are validated, but only measured
repetitions produce per-repetition output and submitted statistics.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use two token-count fields:

- `parameter_prefill_tokens`: exact number of prompt tokens.
- `parameter_decode_tokens`: exact number of generated tokens.

The standard ladder uses IDs such as `end_to_end_latency_512_256`, meaning
512 prompt tokens and 256 generated tokens. The local smoke benchmark uses
8 prompt tokens and 8 generated tokens.

The submitted result reports:

- `total_time_ms`: mean latency across the measured requests.
- `total_time_ms_stddev`: sample standard deviation across the measured
  requests.

Milliseconds are the unit because this benchmark measures the wall-clock wait
for one complete request. That is the latency a caller observes at the
completion API boundary. The prompt and completion token counts define the
amount of model work, so end-to-end latency can stay in request-time units
while the throughput benchmarks cover token-rate units.

The primary metric is the mean across a fixed five measured repetitions. A
single request can be affected by scheduler noise or transient runtime state;
the mean gives the central request latency for the fixed benchmark shape. The
standard deviation uses the same millisecond unit and reports how stable those
five measurements were. The warmup request is a separate setup request and is
excluded from the five measured repetitions.

For measured latencies `t_1..t_5`, `total_time_ms` is
`sum(t_i) / 5`, and `total_time_ms_stddev` is
`sqrt(sum((t_i - mean)^2) / 4)`.

The management service carries `total_time_ms_stddev` through as the derived
end-to-end latency metric's `value_stddev`.

## Exact-Token Prompt Synthesis

The benchmark needs a prompt that satisfies two constraints at the same time:
it must be ordinary text, so tokenizer work stays inside the measured request,
and it must have the requested token length under the runtime tokenizer. Fixed
prompt text produces different token counts across models because tokenizers
and special-token policies differ.

The prompt-synthesis step solves this before measurement. It asks the runtime
tokenizer how many tokens each candidate prompt would produce, then searches
for text with exactly the requested prompt-token count. Failure to find such a
prompt invalidates the run before any latency is recorded.

The implementation starts from a shared natural-language seed passage compiled
into each benchmark client binary from a published seed file
([`prompt_seed.txt`](../../crates/pipette-ops/src/prompt_seed.txt)), so the
exact prompts are reproducible from the repository. A prompt builder calls the
runtime tokenizer repeatedly until it finds text that tokenizes to exactly the
requested prompt-token count.

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

The readiness probe runs on the host or device that can throttle the runtime.
For Docker, the readiness probe runs on the physical Linux host. Readiness
checks are outside the timed window.

Phone measurements are run on mains power with aluminum thermal sinks attached
to the device; laptops, desktops, and servers use stock cooling and their normal
power supply. These externally managed conditions (power delivery and active
cooling) are fixed for the duration of a run and recorded with it; they are
part of the test condition, not something the readiness probe controls.

Phone measurements need stricter thermal handling than larger machines. Phones
have limited thermal mass, so one benchmark cell can leave enough residual heat
to affect the next cell. A phone can also start a run below the throttle point
and cross it during the five-request series, which shows up as high standard
deviation or drift across repetitions. Cooling history therefore matters as
much as instantaneous model configuration.

Our Android runs showed that the OS thermal-status enum alone is a limited
readiness signal for this benchmark. It can remain nominal while
CPU-cluster die temperatures are already high enough to reduce the thermal
budget for the measured requests. The Android readiness check therefore gates
on both `dumpsys thermalservice` and raw CPU-cluster thermal zones. It runs the
readiness check before every measured request. If the device reheats during a
cell, the next measured request waits for the device to return to the readiness
band or the cell fails at the readiness deadline.

macOS turned out to have the same problem from the other direction, and worse.
Measuring die temperature against the thermal-pressure enum on a MacBook Neo
(A18 Pro) showed the enum is not a temperature signal at all but a **fixed
hold-off anchored to the moment the CPU goes quiet**: after a 10-second load it
cleared 317 s later, and after a 123-second load (twelve times the work) it
cleared 318 s later. Repeating the same schedule with and without an external
cooler cleared at the *identical sample* both times, with the die at 34.84C in
one arm and 38.52C in the other. It also engaged on a die change of 0.10C, a
single sensor quantization step. Because the old macOS deadline was 5 minutes
and the hold-off is ~318 s, the gate did not merely over-wait on such a host: it
hit the deadline and **failed the cell** after every measured repetition that
warmed the machine.

The macOS gate still waits on that enum, and the 7-minute deadline is sized so
the ~318 s hold-off fits inside it. That is the defect the measurements
unambiguously fixed: the previous 5-minute deadline sat *below* the hold-off, so
the gate did not merely over-wait. It timed out and failed the cell after any
repetition that warmed the machine.

Replacing the criterion with a die-temperature threshold was considered and
rejected on measurement. Die temperature is read on macOS through the same
private IOHID sensors iOS uses and recorded with every result
(`device_apple_soc_temp_c_before` / `_after`, the same columns iOS reports), but
it does not gate anything, for three reasons:

- **Idle die noise is large and host-specific.** σ ≈ 0.4C with 3.8C
  peak-to-peak on the Neo, against 0.26–0.41C on a stock-cooled MacBook Pro
  (M4 Max). The reading is a max over a per-host sensor count (7 vs 20), so the
  two aren't directly comparable, and the noise is autocorrelated, so averaging
  recovers less than it appears to. Any constant below the wander measures
  noise, which rules out the absolute bands Android (`< 34C`) and iOS
  (`< 36C`) can use on narrow hardware with a known idle floor.
- **Starting temperature does not move the results.** Across the ~3C spread a
  batch actually produces, benchmark numbers did not change measurably, and a
  14-cell soak held 0.27 % between-cell variation. There is no threshold worth
  enforcing because the quantity a threshold would control has been shown not
  to matter at the magnitudes available.
- **Per-repetition cooling does not survive plan scale.** 1000 benchmarks × 5
  repetitions at ~3.5 s is ~5 hours of measurement; cooling 147 s before each
  would add over 200 hours: to control a variable measured not to affect the
  result.

So the temperature column is kept as an audit trail rather than a gate: it
records the condition each repetition ran under, so the assumption above stays
checkable at no cost. The enum gate remains because it is conservative and its
deadline now accommodates the hold-off: not because it is a good signal. The
full characterization, cooled and uncooled, is in
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md).

The aluminum sinks make the phone setup more repeatable by increasing the heat
dissipation available during a run, which also pulls the device back down to the
readiness band faster between cells. Without them, thermal throttling partway
through the series makes phone runs hard to repeat and the readiness wait can
stall for a long time before a device cools back into the band. The sinks raise
the available thermal budget for every model equally, so within a session the
numbers stay stable instead of drifting slower as the device heat-soaks. They
do change the absolute latencies, though: cooled and uncooled runs are not
interchangeable. The sinks therefore define the phone test condition: phone
results should be compared only against other runs with the same cooling setup.

Phones are also benchmarked on mains power rather than battery, and this is the
second externally managed condition. Each device class (iPhone, Galaxy S26
Ultra, and so on) is paired with a charger rated comfortably above the SoC's
sustained peak draw, so inference is never power- or charge-limited and a
depleting battery never confounds timing across the five-request series. Like
the cooling setup, the power source is held constant for a run and changes the
absolute numbers, so phone results are comparable only across runs collected
under the same power and cooling setup. The power state each result actually ran
under is recorded with it: for every benchmark kind, not just latency (see
[Device conditions](device-conditions.md)), so this condition can be verified
per run rather than assumed.

| Platform | Default deadline | Thermal criterion | CPU criterion |
| --- | ---: | --- | --- |
| Android | 10 min | `dumpsys thermalservice` reports thermal status `0` (`NONE` / nominal), and hottest real CPU-cluster thermal zone is `< 34C` | Instantaneous busy ratio from a 1-second `/proc/stat` delta is `< 0.30` |
| Linux | 5 min | Hottest readable `/sys/class/thermal/thermal_zone*/temp` is `< 70C` | First `/proc/loadavg` field divided by available CPU count is `< 0.30` |
| macOS | 7 min | `OSThermalPressureLevel` is `0` (`nominal`), read from `com.apple.system.thermalpressurelevel` | Instantaneous busy ratio from a 1-second Mach `host_statistics(HOST_CPU_LOAD_INFO)` tick delta is `< 0.30` |
| Windows | 5 min | Every exposed temperature counter must go **flat** (spread `<= 3C` across 3 polls) with none within `15C` of the platform's own `CriticalTripPoint`; plus summed GPU-compute utilization `< 5%` and no active throttle-reason flags. See the Windows note below | `\Processor Information(_Total)\% Processor Time` is `< 40` (the only always-required signal) |
| Other | 0 sec | No probe | No probe |

Android, Linux, and macOS readiness loops sleep 3 seconds between
failed readings, then retry until the criteria pass or the platform deadline is
reached; one log line per check. The interval is granularity, not criteria: it
sets only how soon a host that has already cooled is noticed to have done so, and
the iOS client polls at the same 3 s. Windows polls every 5 seconds instead,
because its thermal criterion is a three-sample window rather than a single
comparison, so there the spacing is half the criterion. It also gives Windows a
floor of ~10 s even on an already-cold box. On macOS the CPU sample already
spends ~1 s
inside that interval (~2 s if the tick counters did not advance and it resampled),
so the loop tops the gap up rather than sleeping a further 3 s on top. Probe I/O,
parse, or unavailable-data errors fail
immediately. Threshold comparisons for the thermal and CPU gates are strict: a
value equal to the threshold fails readiness. The one exception is the Windows
flatness span, which is inclusive: a spread of exactly `3C` counts as settled.

This polling is the only retry. The gate runs before each of the five measured
repetitions, and if the device never reaches the band before the deadline, that
repetition's gate returns a timeout that **aborts the whole cell**. The
remaining repetitions are not attempted, no partial or throttled samples are
recorded, and the run is stored as a benchmark error rather than a measurement.
There is no outer retry and no "N of 5 failed" count, because the cell stops at
the first readiness failure; a failed cell is simply absent from the published
results and visible as an error in the run log.

This is a deliberate trade-off with a survivorship caveat worth stating: because
heavier models heat the device more, they are likelier to hit the deadline, so
the gate can preferentially drop *unfavorable* (throttled) samples. Two things
keep that from becoming cherry-picking. The readiness band is identical for
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

Windows gates on more than the two table columns, and its thermal criterion is a
decay test rather than a ceiling. A fixed ceiling is not portable between the two
boxes in the fleet: the gmktec EVO-X2 rests at `33-36C` and saturates at `98C`,
while the Core Ultra 7 258V rests at `42-46C` and saturates at `55C`. One rests
`9C` hotter than the other and uses a fifth of the range, so any single threshold
is either unreachable on one box or a no-op on the other. The `70C` ceiling this
replaced was both.

What is portable is the shape of the curve: both boxes shed junction heat fast
and then crawl (the gmktec `98 -> 44C` in 30 s, then ~600 s for the last `11C`).
So readiness requires every temperature counter the box exposes to go flat:
a spread of at most `3C` across three consecutive 5 s polls. Replayed against the
captured curves that releases 30 s after load ends on the gmktec and 10 s on the
258V. Counters read are `\EsifDeviceInformation(*)\Temperature` (Intel Dynamic
Tuning, already Celsius) and `\Thermal Zone Information(*)\Temperature` (hottest
zone, Kelvin, converted with `− 273`); neither outranks the other, and a counter
stuck at a constant is trivially flat and so contributes nothing.

Flatness means "at steady state", not "cool", so two absolute checks sit above
it. A reading within `15C` of the platform's own ACPI `CriticalTripPoint`
(measured `110C` on the gmktec, `105C` on the 258V) holds readiness. A reading at
or above `60C` only warns and is recorded: blocking a box that has already
reached steady state is futile, and failing the cell loses the measurement
instead of flagging it.

`\Processor Information(_Total)\% Processor Time < 40` is the only
always-required signal; everything else is evaluated only when the box exposes
it. (`Win32_Processor.LoadPercentage` was rejected: it read `1-19%` on the 258V
under a 16-job burn against `1-6%` idle, so its load and idle bands overlap.)
GPU-offloaded inference barely moves CPU load, so the GPU-compute signal carries
the real workload check: idle is `0%`, active Vulkan inference sits at `71-105%`,
and the gate requires summed utilization `< 5%`. A non-zero processor
throttle-reason flag also holds readiness, though that flag has never been
observed to fire on either box; including while the gmktec die sat pinned at its
own limit. A box with no temperature counter at all gates on load alone, paying
only the three-poll window. See
[`pipette_readiness::windows`](../../crates/pipette-readiness/src/windows.rs).

Android uses both the OS thermal-status enum and raw CPU-cluster temperature.
The OS enum is a power-management signal and can remain nominal even when CPU
dies are hot. The raw temperature gate reads
`/sys/class/thermal/thermal_zone*/{type,temp}`, keeps real CPU-family zones
whose `type` matches `cpu-<digit>...` or `cpullc-<digit>...`, and requires the
hottest readable zone to be below 34C.

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

The 34C threshold is a readiness band, not a safety limit. Field runs on
S25-class Snapdragon 8 Elite phones showed that a 70C gate, and then an
interim 50C gate, could still release long prefill/decode repetitions with too
little headroom before the 75-80C performance-throttling band. Starting below
34C gives moderate-duration repetitions roughly 40C of thermal budget while
still releasing reliably above the observed 27-29C tethered idle floor. Very
long repetitions can still overrun the available budget; high standard
deviation on those shapes remains an invalid or follow-up signal rather than
evidence that the cooldown is unnecessary.

Android CPU busy is calculated from the aggregate `cpu` line in `/proc/stat`:

```text
busy = (total_delta - idle_delta) / total_delta
idle = idle + iowait
```

macOS CPU busy comes from Mach `host_statistics(HOST_CPU_LOAD_INFO)` tick
counters sampled 1 second apart; the same delta form as Android, over the user,
system, idle, and nice counters:

```text
busy = (total_delta - idle_delta) / total_delta
```

If those counters do not advance across the window (the one degradation ever
measured, and only at windows shorter than the 1 s used here) the probe
resamples once and then reports the cell unready rather than guessing a ratio.

Windows gathers all fields in one PowerShell call. The `\Thermal Zone
Information(*)\Temperature` counter is reported in whole Kelvin (as a double);
the loop takes the maximum across zones and subtracts `273` to get Celsius.

The Android and macOS CPU probes sample a short instantaneous busy ratio rather
than a load average. On a phone or a developer Mac the 1-minute load average
stays inflated by background work (Android's resting normalized load sat around
0.35–0.45 from the OS background-AI stack and `adb` traffic), so a lagging
average parks above the gate even when the device is otherwise idle. An
instantaneous `%busy` reads near the true resting floor and only crosses the
threshold when something is actively using cores.

Linux keeps the normalized 1-minute `/proc/loadavg` because a dedicated
benchmark host has a near-zero idle baseline, so the background inflation that
disqualified the average on phones does not occur. The read is also free of a
sampling window, and the 1-minute smoothing ignores sub-second housekeeping
spikes. The cost is that the average lags (after a previous cell it decays over
about a minute), but that lag is conservative: it makes the gate over-wait for
residual load to clear rather than release into it. This choice is therefore
sound for dedicated Linux hosts; it would not be appropriate on a shared or
background-busy machine, where an instantaneous probe should be used instead.
Windows uses the aggregate processor load reported by WMI.

## Runtime Details

The general method is implemented across these runtime paths:

| Path | Serving process | Readiness probe |
| --- | --- | --- |
| macOS llama.cpp | local `llama-server` | macOS host |
| macOS MLX | local Python MLX sidecar | macOS host |
| Windows llama.cpp | local `llama-server` | Windows host |
| Linux llama.cpp | local `llama-server` | Linux host |
| Android llama.cpp | `llama-server` on the Android device via `adb` | Android device |
| Linux torch-oai Docker | OpenAI-compatible server in a Docker container | Linux host running the container |
| Linux torch-oai uv | OpenAI-compatible server launched by `uv` | Linux host running the process |
| iOS llama.cpp | in-process `llama.cpp` (no server) | device-side thermal cooldown before each rep (see [iOS](#ios)) |

### llama.cpp

`pipette-llamacpp` starts `llama-server` and waits for `/health`. Prompt
construction uses `/tokenize`; measured requests use `/completion` with:

- `prompt`: the exact-token text prompt.
- `temperature: 0.0`.
- `n_predict`: `parameter_decode_tokens`.
- `cache_prompt: false`.
- `ignore_eos: true`.
- `logit_bias` entries that suppress discovered EOG token IDs.

The `/tokenize` request body is:

```json
{ "content": "<prompt text>", "add_special": true }
```

`add_special: true` keeps the pre-flight token count exactly in sync with the
count `/completion` reports back: it matches the inference path on tokenizers
that auto-prepend a BOS (such as Gemma and LFM2) and is a no-op on models that
do not (such as Granite and Qwen2). A mismatch here would fail the run on the
token-count check.

The `/completion` response must report:

- `timings.prompt_n == parameter_prefill_tokens`.
- `timings.predicted_n == parameter_decode_tokens`.
- `stopped_limit == true`, or `stop == true` with `stop_type == "limit"`.

For EOS handling, the runner parses `llama-server` stderr for `EOG token`
lines. It sends those token IDs in `logit_bias` with value `false`, while also
setting `ignore_eos: true`, so models that would normally stop early still
generate the requested decode count.

The llama-server runner also adds benchmark defaults to the server command:
`--no-warmup`, a derived `--ctx-size` large enough for prefill plus decode
unless the user supplies one, and `--no-mmap` unless the user supplies an mmap
flag. `--no-warmup` is controlled by the benchmark and is rejected from a cell's
runtime flags. Context size and memory-mapping stay operator-overridable via
the cell's typed `ctx_size` / `mmap` runtime flags (rendered to `--ctx-size` /
`--mmap` / `--no-mmap` on the server command); when set they become part of the
effective runtime flags and must be kept identical across compared runs.

These server-wiring flags are also controlled by the benchmark and are rejected
from a cell's runtime flags: `--model`, `-m`, `--mmproj`, `--host`, `--port`,
and `--no-warmup`.

### MLX

`pipette-mlx` starts a local Python HTTP sidecar. Prompt construction uses
`/tokenize`; measured requests call `/end_to_end_latency` with:

- `prompt`: the exact-token text prompt.
- `decode_tokens`: `parameter_decode_tokens`.

The sidecar `/tokenize` request body is:

```json
{ "prompt": "<candidate prompt text>" }
```

The sidecar tokenizes with `add_special_tokens=True`. The response from
`/end_to_end_latency` must report:

- `prompt_tokens == parameter_prefill_tokens`.
- `completion_tokens == parameter_decode_tokens`.
- finite positive latency.

The Python sidecar uses greedy `mlx_lm.stream_generate` and suppresses EOS. The
sidecar also returns its own timing fields, but the submitted benchmark result
uses the Rust-side elapsed time around the outer HTTP request. This keeps MLX
aligned with the llama.cpp and torch-oai timing path.

### torch-oai

`pipette-torch-oai` runs an OpenAI-compatible engine such as vLLM or SGLang.
Prompt construction uses the engine's `/tokenize`; measured requests use
`/v1/completions` with:

- `prompt`: the exact-token text prompt.
- `max_tokens`: `parameter_decode_tokens`.
- `temperature: 0.0`.
- `ignore_eos: true`.

`pipette-torch-oai` disables server-side prefix reuse at launch for **every**
benchmark type (not just latency), because they all measure cold prefill plus
decode, so a reused cache would silently turn any of them into a warm-prefix
measurement. vLLM launches add `--no-enable-prefix-caching`; SGLang launches add
`--disable-radix-cache`. Passing vLLM's positive `--enable-prefix-caching` flag
is rejected because it would turn the benchmark into a warm-prefix measurement
while llama.cpp is explicitly run with `cache_prompt: false`.

The `/tokenize` request body includes:

```json
{
  "model": "<model name>",
  "prompt": "<candidate prompt text>",
  "add_special_tokens": true
}
```

The `/v1/completions` response must include a `usage` block, and the runner
validates:

- `usage.prompt_tokens == parameter_prefill_tokens`.
- `usage.completion_tokens == parameter_decode_tokens`.

If the usage block is missing, the run fails.

### iOS

`pipette-ios` runs `llama.cpp` in process rather than launching a server, and
its latency path differs from the CLI and server runtimes in ways that affect
comparability:

- The prompt is not built by the shared exact-token synthesizer. The app
  repeats the text `"hello "`, tokenizes it, and takes the first
  `parameter_prefill_tokens` tokens. Before the benchmark runs, `check_ctx_size`
  checks the required context window
  (`parameter_prefill_tokens + parameter_decode_tokens` for latency) against the
  context the model was loaded with: on the fresh-load path that size is the one
  the app is about to allocate; on the reuse path (a benchmark run against a
  model the job already loaded) it is the caller-supplied size the model already
  holds. Either way a context too small for the benchmark fails with a readable
  error rather than running. The measured prompt is therefore always the full
  requested prefill length; the `min(parameter_prefill_tokens, context_size)`
  slice in the runner is a floor that the up-front check keeps from being
  exercised.
- The measured loop does not call the shared readiness probe per request. The
  CLI and server paths wait for the thermal/CPU criteria before every measured
  request; the iOS app instead runs a device-side thermal cooldown via a
  per-repetition `readiness` callback, so heat from the previous repetition has
  to clear before the next one. It is a cooldown gate, not "no gate." The
  temperature signal that gate uses depends on the build (see below).
- Before each repetition (warmup included) the app resets the llama context and
  sampler, prefills the prompt, and times a greedy ignore-end-of-generation
  decode of exactly `parameter_decode_tokens` tokens. It runs one warmup
  repetition and five measured repetitions and reports the same `total_time_ms`
  mean and sample standard deviation as the other runtimes.

**Thermal detection on iOS.** iOS exposes no public SoC temperature API, and the
public `ProcessInfo.thermalState` enum is too coarse to gate this benchmark: it
can stay `.nominal` while the SoC has already down-clocked, so a cooldown that
trusts it alone would release reps with too little thermal headroom. A
shipping/Release build is limited to that coarse signal (optionally refined by a
per-device, calibrated IMU-drift temperature estimate, accurate to roughly
±1.5C while the device is stationary). Published iOS numbers are therefore
collected with an internal, thermal-aware build (enabled by the
`PIPETTE_PRIVATE_THERMAL` flag) that reads the actual SoC die temperature (the
`PMU tdie*` sensors) through private IOKit thermal services. With that build the
cooldown requires both `thermalState == .nominal` and a die temperature below
36C (about 1C above the ~35C tethered idle floor) before each repetition,
giving the iOS path a real temperature gate comparable in intent to the raw
CPU-cluster gate used on Android. The private read is compiled out of public
builds, so it adds no private symbols to shipping binaries; it exists only to
make the benchmark's cooldown trustworthy.

Because the prompt is constructed differently (a repeated filler string rather
than the seed synthesizer) and the per-repetition gate is a device-side thermal
cooldown rather than the shared thermal/CPU probe, iOS latency results should be
compared only against other iOS results under the same cooling conditions, not
against the gated CLI or server paths.

## Result Validity and Comparability

A run is invalid if warmup or any measured request reports the wrong
prompt-token count, the wrong completion-token count, missing usage or timing
data, unexpected early stop, readiness timeout, or readiness-probe failure.

Per-repetition output lines follow this shape:

```text
rep 1/5: 123.456 ms (prompt_tokens=512, completion_tokens=256)
```

Those lines are the audit trail for exact token counts used in the run.

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
- Inspect the run output when auditing a run. The per-repetition lines should
  all show the requested prompt and completion token counts.

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
[`pipette_ops::measurement`](../../crates/pipette-ops/src/measurement.rs).

The per-repetition thermal readings it brackets each repetition with are paired
by
[`pipette_ops::thermal_series`](../../crates/pipette-ops/src/thermal_series.rs).

The shared exact-token prompt builder is
[`pipette_ops::prompt_seed`](../../crates/pipette-ops/src/prompt_seed.rs).

The platform readiness probes are implemented in
[`pipette_readiness`](../../crates/pipette-readiness/src/lib.rs).
