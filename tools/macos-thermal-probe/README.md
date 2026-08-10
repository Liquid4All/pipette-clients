# macos-thermal-probe

Characterizes how a Mac sheds heat, and how badly the macOS thermal-pressure
enum reflects that.

It started as a narrow question about the readiness gate in
[`crates/pipette-readiness/src/macos.rs`](../../crates/pipette-readiness/src/macos.rs)
(which proceeds only when
`notify_get_state(kOSThermalNotificationPressureLevelName)` reads 0
(`nominal`)), and how long that gate keeps waiting after the SoC has already
cooled. It samples that level, `ProcessInfo.thermalState`, and IOHID die
temperature side by side across an idle baseline → all-core load → cooldown,
and reports the interval between "die back at baseline" and "enum cleared".

It has since turned out to be more useful than that, because measuring the
question properly required measuring the **idle noise floor** of the die
sensors first, and that number (σ ≈ 1 °C on a cooled Neo, wandering nearly
4 °C peak-to-peak with no load) invalidates most intuitive temperature
thresholds, including several of this tool's own earlier ones. If you are
writing anything in this repo that compares a die temperature against a
constant, read
"[Idle die noise is enormous](#idle-die-noise-is-enormous-and-host-specific)"
and "[Seven calibration mistakes](#seven-calibration-mistakes-all-instructive)"
first.

Two distinct uses now:

- **Gate evidence**: is the enum worth waiting on, and would die temperature
  do better?
- **Batch reproducibility**: how long between repeated benchmark runs to get a
  consistent thermal starting point. See
  "[Using this to make benchmark batches reproducible](#using-this-to-make-benchmark-batches-reproducible)".

Both questions are now answered for the MacBook Neo. See
**[Conclusions](#conclusions)**, immediately below.

## Conclusions

Measured end to end on a MacBook Neo (A18 Pro) against a real benchmark
workload: a ~2 s I/O-bound load, a 3.5 s warm-up, and five 3.5 s benchmarks per
variant, with a ~6 s break between sets.

> ### The test unit is thermally modified; twice
>
> The Neo ships with no internal fan. The unit measured here has **two**
> additions, and every number below was taken on the modified machine:
>
> 1. **A passive thermal pad bonded inside the case**, coupling the CPU and GPU
>    to the chassis. Permanent, present in every run, and the larger of the two
>    interventions. It gives the SoC a heat sink it does not ship with.
> 2. **An external active cooling pad** under the case. This is the only
>    variable; "cooled"/"uncooled" below mean with and without *this* one.
>
> **So there is no stock-Neo data here, and none can be produced from this
> unit.** A stock machine would be expected to run hotter; plausibly into a
> throttling regime neither arm reaches. See
> [If a second MacBook Neo arrives](#if-a-second-macbook-neo-arrives).
>
> On the external pad specifically, paired on/off runs have been done
> ([Cooler on versus off](#cooler-on-versus-off)) and **the conclusions
> survive**: it lowers equilibrium temperature by ~3.4 °C and changes nothing
> else; not the noise floor, not the cooling rate, not peak temperature
> (1.7 °C), not the enum by even one sample. Rep-to-rep scatter is identical.
>
> Is the external pad worth using? Not for speed; nothing in this workflow
> waits on cooling. It buys **margin**: 0.82 °C batch spread instead of 3.00 °C,
> headroom on the very assumption the "don't gate" conclusion rests on, plus a
> much faster route to a trustworthy idle baseline.
>
> The internal pad is also the likeliest explanation for the **multi-timescale
> cooling** and the **long baseline settling times** measured throughout:
> coupling the die to the chassis puts a large slow thermal mass in series with
> it. That is inference from the shape of the data, not something measurable
> without an unmodified machine.

### Thermal behavior of the Neo

| | cooled | uncooled |
|---|---|---|
| idle die noise, de-trended | 0.43 °C | 0.40 °C. **the cooler does not change it** |
| sensor quantization | 0.10 °C | 0.10 °C |
| heating from a 3.5 s benchmark | +10 to +12.5 °C (+6.9 °C in the first 2 s) | same |
| peak, 120 s all-core load | 58.2 °C | 60.0 °C |
| idle equilibrium | 35.0 °C | **38.4 °C** |
| rate of heat shedding | — | **the same** (curves overlay vs own equilibrium) |
| time to reach the *cooled* baseline temp | 126 s | never. It settles above it |
| ceiling under continuous benchmarking | ~49 °C peak / ~60 °C worst case | ~51 °C peak |
| thermal pressure, continuous benchmarking | **pinned at `moderate`** | **pinned at `moderate`**: no escalation either way |
| duty-cycle start spread, 15 reps | 0.82 °C | **3.00 °C** |
| rep-to-rep scatter (start / peak) | 0.10 / 0.46 °C | 0.11 / 0.51 °C |

**Heat arrives in seconds and leaves over minutes.** That asymmetry is the
single most important property here, and it is why return-to-baseline gating is
so expensive on short workloads.

### The answer: don't gate this workload at all

**The ~3 °C thermal spread across a batch does not measurably affect the
benchmark numbers.** That was measured against the real workload, not the
synthetic burn loop, and it is the finding that settles everything else.

So:

- **Do not gate on `pressurelevel`.** It trips 2.9 s into the *first* 3.5 s
  benchmark and never clears: 99% of an 8.5-minute batch reads non-nominal,
  because every rep re-arms it well inside its ~318 s hold-off. A "wait for
  nominal" gate therefore never passes; it waits out `DEFAULT_MAX_WAIT` (420 s)
  and times out. Six gates per variant is **~42 minutes of pure timeout**, and
  if a timeout fails the cell, it fails cells.
- **Don't replace it with a die-temperature gate either.** The variation such a
  gate would control is real but demonstrably inconsequential here, and the
  machine self-limits at ~60 °C, so there is no runaway to protect against.
- **Record die temperature alongside results instead of gating on it.** That
  keeps the assumption under continuous check at zero cost, and turns a
  correctness question into a monitoring one.

The correct gate for this workload on this host is **no gate**.

### What that conclusion does *not* license

- **It is cooling-configuration-specific, and this is the big one.** The 60 °C
  ceiling and the never-past-`moderate` behavior were measured with an external
  cooler attached. Remove it and the machine may well reach a throttling regime,
  at which point the temperature spread across a batch would exceed the ~3 °C
  shown to be harmless, and the "don't gate" conclusion could invert outright.
- **It is workload-specific.** A longer or heavier job could reach the
  throttling regime this one never approaches. The 60 °C ceiling is where *this*
  duty cycle settles under *this* cooling, not a property of the machine alone.
- **It is host-specific.** The 14-core actively cooled Mac never leaves
  `nominal` at all, so its thermal criterion is inert for the opposite reason.
  Two hosts, two ways of the same gate being useless, and neither generalizes.
- **`moderate` is not `nominal`.** The Neo sits permanently in a raised pressure
  state during benchmarking. Nothing here says that state is harmless: only
  that within it, a 3 °C swing does not move these numbers.

## Build / run

macOS only. No dependencies beyond the SDK.

Two modes.

**Single-shot**; one load → cooldown, for gate evidence and the settling curve:

```bash
clang -O0 -framework Foundation -framework IOKit -o thermal-probe thermal-probe.m
./thermal-probe                    # 300s baseline, 300s load, 60s steady window, 2s samples
./thermal-probe 300 600 60 2       # longer load if the host resists heating
./thermal-probe 300 300 60 2 4     # pin to 4 burn threads
./thermal-probe 300 0 60 2 1       # no load: characterize this host's idle noise
```

Args:
`[baseline_secs] [load_secs] [steady_secs] [interval_secs] [threads] [nominal_cap_secs]
[max_cool_secs]`.

**Cycle**; repeated load/rest, for choosing the gap between batch runs:

```bash
./thermal-probe --cycle 5 --load 60 --rest 45
./thermal-probe --cycle 8 --load 30 --rest 90 --threads 4 --baseline 120
```

Flags: `--cycle N --load L --rest R [--interval I] [--threads T] [--baseline B]`.
See "[Using this to make benchmark batches reproducible](#using-this-to-make-benchmark-batches-reproducible)".

**All durations are fractional seconds**, in both modes: `--load 3.5` and an
interval of `0.1` are the point, not an afterthought. Real benchmark units are
seconds long, and the tool was originally scaled for minutes; see
"[Short workloads](#short-workloads-need-a-different-scale)".

**`baseline_secs` should be at least 3× `steady_secs`**. The band is calibrated
from windows slid over the baseline, so a short baseline cannot produce one. The
default is 300 s for that reason.

To avoid paying that every run, set `THERMAL_PROBE_CALIB` to a file path: a
baseline long enough to calibrate writes its noise figures there, and later runs
with a short baseline reuse them. Only the *noise* is cached. The baseline mean
is always re-measured, because ambient moves and `recovered` is judged against
it. A cached calibration is rejected (with the reason printed) if it was taken
for a different window or sample interval, since both statistics depend on
window length.

> **The cache validates the window, not the machine.** It has no idea whether
> the fans changed, a cooling pad was attached, or the laptop moved off a desk.
> Use a **separate `THERMAL_PROBE_CALIB` path per physical configuration** (
> `calib-cooled.txt` and `calib-uncooled.txt`), or it will silently apply one
> setup's noise floor to another and every derived threshold will be wrong in a
> way nothing reports.

`-O0` is deliberate. The burn loop must not be optimized away.

**The run blocks until both enums read nominal before it starts measuring**,
up to `nominal_cap_secs` (default 900; 0 disables). This matters: a run that
begins at `fair` is measuring the tail of some earlier thermal event, not the
load applied here, and on a fanless host that tail is minutes long. If the cap
expires the verdict is flagged `[CONFOUNDED]` rather than silently reported.
Budget for it: back-to-back runs on a fanless machine will spend several
minutes here.

It saturates every core, so don't run it alongside a benchmark you care about.

### The cooldown is adaptive

The cooldown doesn't run for a fixed span. It ends when **both**:

- the die has **recovered**: a smoothed mean (over a fixed duration, not a
  fixed sample count) back within the derived threshold of the pre-load
  baseline, and
- `pressurelevel` has returned to nominal.

`recovered` gates the exit rather than `steady` because an honest steadiness
test needs a window too long to be the fastest signal; see
"[What a die-temperature gate can actually buy](#what-a-die-temperature-gate-can-actually-buy)".
`steady` is still measured and reported alongside, for comparison.

**`steady` requires both a range and a trend test.** A partly-filled window
never counts, so there is always a full `steady_secs` of history behind the
call, and the window's least-squares slope must sit inside a separately
calibrated slope band. A range test alone cannot tell "flat" from "descending".
If the die leaves the band the settle clock restarts and the log says so, so a
sequence of "die steady" lines never reads as though it held when it didn't. The
summary distinguishes *held from Ns*, *reached but did NOT hold*, and *never
settled*.

### The steady band is measured, not hardcoded

The band is calibrated by sliding the **same window** over the idle baseline and
taking the 90th percentile of the ranges it sees there:

```
band = max(p90 of idle window ranges, 0.20 °C)
```

That is the identical statistic the steadiness test computes, measured on data
known to be idle, so it needs no distributional assumption, no conversion
factor, and no ceiling. The die has settled once it is no jumpier than it was
before the load.

This requires `baseline_secs` to be comfortably longer than `steady_secs` (a
baseline of 3× the window gives plenty of overlapping windows). If the baseline
can't hold at least 10 full windows, the tool falls back to `4 × sigma` and says
so in brackets. The floor of 0.20 °C is hardware: the sensors quantize at
~0.10 °C, so a tighter band asks for resolution they don't have.

The summary reports the whole calculation:

```
die temp noise (baseline)    sigma=0.54C spread=2.68C trend=-0.93 C/min over 150 idle samples
noise autocorrelation        lag-1 +0.68 at 2s spacing -> correlation time ~5.0s
                             8 samples over 2s are worth 1.0 independent ones
idle window stats            range med=1.39C p90=1.93C | |slope| med=0.30 p90=0.71 C/min (120 windows)
steady band (derived)        1.93C = p90 of idle window range
slope band (derived)         0.71 C/min = p90 of idle window |slope|
                             tested over a 31-sample window (60s)
recovered threshold          within 1.00C of baseline on a 8-sample mean
```

Every threshold in that block is measured from this host's own idle behavior;
none is a constant. The autocorrelation line is there because it decides whether
sampling faster would buy anything. See the fifth calibration mistake.

A band that is wide relative to the load-induced rise is caught **after** the
run, as `[WEAK]`, where the rise is actually known: not clamped in advance.
See "Seven calibration mistakes" below for why that distinction cost a run.

Waiting on the enum as well as the die is the point: it's the gap between the
two that this tool exists to measure. Exiting the moment the die settled would
end a Neo run around five minutes before the enum cleared, and every verdict
would degrade to "pressurelevel never cleared".

`max_cool_secs` (default 1800; the cap is measured from load stop) bounds the
whole thing. Hitting it is reported as `[PARTIAL]` naming which condition was
still outstanding: never silently folded into the numbers. If die sensors are
unavailable, steadiness is unknowable and the enum alone ends the cooldown.

A longer `baseline_secs` is close to free and strictly improves the
calibration. It adds windows to take the percentile over.

## Interpreting the output

| Column | What it is |
|---|---|
| `ProcessInfo` | `ProcessInfo.thermalState`, 0–3 (nominal/fair/serious/critical). What the gate used to read; still what telemetry records. |
| `pressurelevel` | `notify_get_state` on the linked `kOSThermalNotificationPressureLevelName`. **What the gate reads now.** 0–4 on macOS (nominal/moderate/heavy/trapping/sleeping). |
| `tdie_max` | Hottest `PMU tdie*` IOHID sensor, °C. The ground truth the two enums are being judged against. |
| `pressure` | The undocumented, shorter `com.apple.system.thermalpressure`. |
| `bogus` | A nonsense notify name. Must stay 0; if it doesn't, ignore the whole run. |

Recovery times are reported both absolutely and **relative to load stop**.
the latter is what matters, since that's when a real readiness gate starts
counting against its deadline.

The `=== verdict ===` block states the conclusion outright:

- **`[CRITICAL]`**: the enum needed longer than `DEFAULT_MAX_WAIT` (420 s) to
  clear. The gate doesn't merely wait on such a host; it times out and
  **fails the cell**.
- **`[STRONG]`**: the die recovered N s before the enum cleared. That N is
  the per-cooldown waste, and the case for gating on `PMU tdie`. If the die
  was already at baseline when the enum cleared, it adds that the enum is on
  a **timer, not a temperature threshold**; cooling harder won't help.
- **`[NOTE]` over-wait within one interval**: the enum and the die agree;
  leave the gate alone.
- **`[NOTE]` enum cleared *before* the die recovered**: the enum is the
  *looser* gate, so switching to die temp would make cooldowns **longer**.
  The outcome that argues against the change.
- **`[CONFOUNDED]`**: the run never reached nominal before measuring; every
  recovery number is an upper bound.
- **`[PARTIAL]`**: the cooldown ended on `max_cool_secs` rather than on its
  exit condition, or the die never came back within the recovered threshold of
  baseline. Says which, and raising `max_cool_secs` is the fix.
- **`[WEAK]`**: the steady band is more than half the load-induced rise, so
  "steady" barely resolves anything. Either the load was too small or the host's
  idle noise is too large for a die-trend gate to mean much on it.
- **`[FAIL]`**: `ProcessInfo` moved but `pressurelevel` stayed 0. The notify
  read isn't tracking and `macos.rs` should revert to `ProcessInfo`.
- **`[INCONCLUSIVE]`**: neither enum left nominal. It reports whether die
  temp rose anyway, which separates "the load didn't land" from "the load
  landed but the enums never tripped".

## Findings so far

**MacBook Neo (A18 Pro), external active cooler attached.** `pressurelevel` reaches `moderate` under
load, so the notify signal is live and not a constant 0: worth proving,
since `notify(3)` accepts *any* name and reports 0 for it (hence the `bogus`
control). It moves in **lock-step with `ProcessInfo` in both directions**, so
the switch to notify bought no latency, only the removal of a `swift -e`
compiler spawn from the poll loop. The undocumented `pressure` name stayed 0
throughout and is not a usable substitute.

### The enum is a fixed hold-off timer: confirmed across a 12× load range

Two runs, deliberately very different loads:

| | 10 s load (confounded start) | 123 s load, clean start |
|---|---|---|
| `pressurelevel` cleared, after load stop | 317 s | **318 s** |
| die peak | — | 59.76 °C (from 36.63 baseline) |
| enum tripped, after load *start* | — | 4 s |

**A 12× difference in heat input moved the clear time by one second.** That is
about as clean a demonstration of a fixed hold-off timer as this hardware will
give. It also engages in 4 s and releases in 318 s: strongly asymmetric, which
is what a release timer looks like and not what a temperature threshold looks
like. The earlier single-point observation (38.20 °C reading `moderate` at
335 s and `nominal` at 337 s, having held that temperature for ~5 minutes) is
the same conclusion seen from the other side.

### It trips on one quantization step, and one short benchmark arms it

Measured on the Neo during a cycle run: `pressurelevel` went from `nominal` to
`moderate` when the die moved **43.72 → 43.83 °C; a single 0.11 °C
quantization step.** No temperature threshold crosses on one LSB. Combined with
the release behavior, the enum is an *integrator over accumulated exposure*,
not a thermometer, in both directions.

It tripped ~4 s after load start; the same latency seen in the 123 s run, at a
completely different load duration. So:

> **A single 3.5 s benchmark is enough to arm the enum, which then holds for
> ~318 s.**

For a harness that gates on the enum, that is the whole ballgame. A variant with
six gated units trips on the first one, and every gate after it waits the full
hold-off or times out against the 420 s `DEFAULT_MAX_WAIT`. The gate is not
pacing the benchmark; it is a five-minute stall triggered by the first
three-and-a-half seconds of work.

One caveat on the trip latency: the run that produced it had a **timing bug**
(see below) that stretched a requested 3.5 s load to ~4.1 s. Since the trip
consistently lands ~4 s after load start, a genuinely 3.5 s load may sit just
under it. That is worth re-measuring now the timing is exact, but note the
margin is a few hundred milliseconds, which is not a margin to build a harness
on.

**318 s exceeded the old 300 s `DEFAULT_MAX_WAIT`**, so the gate used to time
out and fail the cell here, which is what `pipette-readiness/src/lib.rs`
already warns about when it names the Neo as needing
`PIPETTE_READINESS_MAX_WAIT_SECS`. The raise to 420 s is what makes this host
pass, and the 318 s measurement is the evidence for that number.

The die, meanwhile, was back within 1 °C of its settled value at 147 s. So the
gate waits roughly twice as long as the hardware needs, but see
"[What a die-temperature gate can actually buy](#what-a-die-temperature-gate-can-actually-buy)"
before concluding that a die-trend gate recovers all of it.

**14-core actively-cooled Mac.** Neither enum ever leaves `nominal`, even at
full load: die temp rose 49 °C → 60 °C over 40 s and both enums stayed 0. On
this class of host the thermal criterion is effectively inert and `top`'s CPU
check does all the gating. Don't generalize gate changes from these machines.

Levels above `nominal` remain unexercised by the Rust test suite, so any run
reaching `moderate`+ is also the first real confirmation that
`format_pressure_word`'s mapping matches hardware. The numbering is
macOS-specific: iOS uses 0/10/20/30/40/50 for the same enum.

## Idle die noise is enormous, and host-specific

From a 300 s no-load run on the Neo (150 samples, `./thermal-probe 300 0 60 2 1`):

| | Neo + cooling pad (7 sensors) | Mac (internal fans, 20 sensors) |
|---|---|---|
| idle sigma | **0.80 °C** | 0.26–0.41 °C |
| idle peak-to-peak | **3.79 °C** (35.92–39.71) | ~1.2–2.0 °C |
| median 60 s window range | **2.71 °C** | 1.39 °C |
| quantization step | 0.10 °C | 0.10 °C |

**The Neo's die wanders nearly 4 °C with no load on it.** Any threshold smaller
than that (on a single sample) is measuring noise. This is the finding that
invalidates hardcoded temperature constants anywhere in the repo.

Three properties matter for anyone writing a die-temperature gate:

1. **The noise is stationary, not drift** *when the machine really is settled.*
   De-trending a clean 300 s baseline moved sigma from 0.801 → 0.799 °C, and the
   two halves differ by 0.155 °C. It is not residual cooling, so waiting longer
   does not reduce it. But see the drift caveat below: not every baseline is
   clean, and the tool cannot currently tell on its own.
2. **The noise is often autocorrelated**: lag-1 has ranged +0.07 to +0.80
   across Neo baselines. When it is high, any statistic assuming independent
   samples (the Gaussian range factor, `σ/√N`) is wrong. Since it is *sometimes*
   low, the correction has to be measured per run rather than assumed either
   way. Note that drift inflates lag-1 as well as sigma, so a high value on a
   `[DRIFT]`-flagged baseline may be the drift talking.
3. **`tdie_max` is a max over sensors**, which is upward-biased and noisier than
   any single sensor, and the sensor count differs per host (7 vs 20). That is
   the right thing to measure (it's what a gate would read), but it means noise
   is not comparable across machines.

### How reproducible is the calibration? Less than first claimed.

Four 300 s Neo baselines, all nominally the same machine:

| baseline | σ (raw) | σ (de-trended) | lag-1 | drift / 300 s |
|---|---|---|---|---|
| run A | 0.80 °C | 0.799 °C | +0.68 | +0.20 °C |
| run 3 | 1.03 °C | 0.925 °C | +0.80 | −1.53 °C |
| idle, cooler ON | 0.43 °C | 0.426 °C | **+0.07** | −0.25 °C |
| idle, cooler OFF | 0.64 °C | **0.404 °C** | +0.63 | +1.74 °C |

An earlier revision of this file, working from the first two rows only,
concluded that "idle noise is a host property, not a session property" and that
the 300 s characterization was therefore worth caching. **With four samples that
claim does not hold.** De-trended sigma still ranges 0.40–0.93 °C and lag-1
+0.07 to +0.80 on one machine.

Drift explains some of it (the two worst-behaved baselines are the two that
`[DRIFT]` flagged), but not all: run A and the cooler-ON idle run are both clean
by that test and still differ 2× in sigma and 10× in autocorrelation.

Practical consequence: **the calibration cache is a convenience, not a
substitute for measuring.** Use it to avoid re-paying 300 s within a session or
a stable setup; re-measure when anything about the machine's situation changes,
and treat a cached band as suspect if the run behaves unexpectedly. Two rows
were not enough to generalize from, and it is worth noticing that the
generalization went in the direction that saved time.

### Drift: `nominal` does not mean settled

Run 3's baseline was **not** clean: it fell 1.53 °C over its 300 s, and
de-trending dropped sigma from 1.026 to 0.925. The cause is visible in its first
line of output; `nominal after 0s`. The enum gate passed instantly while the
die was still descending from earlier activity.

**So `waitForNominal` does not guarantee a settled die.** Given the enum is a
release timer, it can read nominal long before, or long after, the hardware is
actually at rest. On the Neo the two are barely related.

The consequences are mild but real: drift inflates sigma, which inflates the
band by ~10%, and it biases the baseline mean. Run 3's die settled at 37.45 °C,
about 0.5 °C *above* the 36.63 °C mean of its own drifting baseline. Any
"recovered vs baseline" test inherits that bias.

### Seven calibration mistakes, all instructive

Recorded because each looked completely reasonable and each was wrong, in the
same way: **a threshold chosen by judgement rather than measured against the
noise it has to beat.**

**A fixed multiple of sigma doesn't survive.** Converting sigma to an expected
window range needs a factor that depends on window length *and* on the
autocorrelation. Measured on the Neo, p90/sigma was 3.51 at a 16-sample window
but 4.73 at 31: no single multiple serves both. Sliding the real window over
the baseline sidesteps the whole problem.

**An a-priori ceiling is worse than no ceiling.** A 2.00 °C clamp, added to
catch "contaminated" baselines and calibrated on the quiet Mac, cut the Neo's
correct 3.21 °C band by 38%. That put it *below* the 2.71 °C median idle window
range, so `steady` could never fire on a perfectly idle machine (the run burned
to its cap), and the summary labelled the good data "contaminated". A band is
only too wide relative to the thermal swing it must resolve, which isn't known
until the load has run. Hence `[WEAK]` in the verdict instead of a clamp.

**A range test cannot see drift.** Run 3's `steady` fired 97 s after load stop, with a 60 s window range of
3.57 °C, inside the 4.22 °C band. The die at that moment was **still falling at
−3.21 °C/min**. Max − min answers "how far did it move", never "was it moving",
so a slow monotonic descent is indistinguishable from stationary noise. Exactly
the same confound that inflated run 3's baseline sigma, now inside the criterion
itself.

The obvious repair (also add a slope limit) walks straight back into mistake
two if the limit is chosen rather than measured. Idle |slope| over a 60 s window
has a **median of 1.27 °C/min** on this host, so a sensible-sounding
"1.0 °C/min" threshold sits *below* the idle median and would rarely fire on a
genuinely idle machine.

The general lesson for the rest of the repo: **calibrate against the noise the
target host actually has, and prefer a post-hoc sanity check over an a-priori
clamp.** A clamp tuned on one machine silently destroys the measurement on
another and reports the destruction as a property of the data.

**And a fourth, caught while writing the batch mode.** Cycle convergence was
first written as "consecutive rep-to-rep deltas all inside the noise floor",
which is the *same blind spot mirrored*. Five reps creeping +0.9 °C each would
pass every consecutive check against a 1.0 °C floor while drifting 3.6 °C in
total. It now tests the **spread of the remaining reps**, which cannot be fooled
that way. Verified against three cases: the observed accumulating batch (not
converged), the observed stable batch (converged throughout), and a synthetic
slow creep, which the delta test passed from rep 1 and the spread test correctly
holds back to rep 4.

**And a fifth, exposed by adding sub-second sampling.** The `recovered`
threshold was derived as `2σ/√N` over an N-sample mean, which assumes the
samples are *independent*. They are emphatically not: lag-1 autocorrelation is
+0.68 at 2 s spacing and **+0.92 at 0.25 s**. For an AR(1) process the variance
of a mean inflates by `(1+r)/(1-r)`, so the honest count is `N(1-r)/(1+r)`:

```
8 samples over 2s are worth 1.0 independent ones
```

Sampling faster buys **time resolution and no precision at all**. The old
formula divided by √8 and claimed a 0.84 °C threshold where the data supports
2.39 °C: roughly 3× overconfident. Smoothing is now specified as a *duration*
rather than a sample count, since noise decorrelates on a timescale (~3–5 s
here) and expressing it in samples silently changed the statistics whenever the
interval changed.

**A sixth; in the code written because of the fourth.** Cycle convergence, having
been fixed from consecutive-deltas to spread, was still a *spread* test. It
produced a confident false positive on real Neo data: reps 2–5 spanning 0.90 °C,
"inside the 1.79 °C noise floor", while start temperature climbed
**+0.310 ± 0.046 °C/rep (t = 6.7)** and peak **+0.554 ± 0.028 (t = 20.1)**. Four
points of a sustained climb span very little; fifteen span a lot.

Two separate errors were tangled there:

1. **Spread cannot see a trend**: the same blind spot as the range test, at a
   longer timescale, in code written to fix that very thing.
2. **The noise floor was the wrong yardstick.** It is 2σ of *single idle
   samples* (0.895 °C), compared against *smoothed means taken at the same phase
   of a repeating cycle*, whose real scatter was **0.10 °C: 18× tighter**. Idle
   sigma is dominated by slow free-running wander that simply isn't present
   between phase-locked measurements. Importing a number measured under one set
   of conditions into a question posed under another.

Convergence now fits a slope to each candidate tail and asks whether it differs
from zero using **that tail's own residuals**; self-calibrating, no imported
constant. It also tests the **peak**, not just the start, because the benchmark
runs *through* the excursion. That mattered immediately: on the same data the
full-batch tail gives start t = 2.9, which passes a 3.0 threshold, and only the
peak (t = 7.8) catches it.

**A seventh, and the sharpest.** The trend test that replaced the spread test
asked whether the slope was *statistically significant*. On a real 15-rep Neo
batch it declared `[CONVERGED] from rep 8`. It was wrong. The peak slope across
successive candidate tails was:

```
tail from rep  6      7      8      9     10     11
slope       +0.193 +0.216 +0.203 +0.193 +0.195 +0.074   C/rep   <- barely moves
t            4.19   3.91   2.87   2.04   1.45   0.42            <- collapses
```

The slope estimate is essentially constant; only `t` falls, purely from lost
degrees of freedom as the tail shortens. **A search that walks forward shrinking
the window declares convergence exactly when it loses the power to detect the
trend, not when the trend stops.** Any sustained climb passes eventually.

*Non-significance is not evidence of flatness; it is usually absence of data.*
The fix inverts the question; bound the slope from above at 95% and require the
worst-case drift it permits to be small next to the scatter already present.
This is equivalence testing rather than significance testing, and it has the
right incentive: **more reps narrow the interval and make convergence easier to
demonstrate**, where the old rule made it easier with fewer. Replayed against
that batch, every tail correctly reads "still drifting", and the bound now
*grows* as the tail shortens instead of shrinking.

Seven times now, the same shape: *a statistic that answers a slightly different
question than the one being asked.* Range answers "how far did it move", not
"was it moving". Consecutive deltas answer "was each step small", not "did it go
anywhere". `√N` answers "how much noise averages out", but only for independent
samples. Worth being suspicious of any threshold whose statistic wasn't chosen
by asking what exactly it would fail to notice.

### Consequence for `recovered`

The old "within 3.0 °C of baseline" test compared a **single sample** against a
threshold *below* the Neo's 3.79 °C idle peak-to-peak, so one noise dip could
read "recovered" while the die was genuinely hot. It now uses a 5-sample
smoothed mean with a threshold derived from measured noise
(`max(1.0 °C, 2σ/√5)`), reported in the summary.

Smoothing is deliberately short: a full `steady_secs` window would lag a fast
descent by half its length. **The older 24 s recovery figure was measured with
the single-sample 3.0 °C test and is not comparable to current output**. The
equivalent clean measurement is run 3's 147 s, from a much larger load.

### What a die-temperature gate can actually buy

Adding a slope test fixes correctness but exposes a harder limit: **resolving
"slope is zero" against this noise needs a long window, and a long window cannot
report sooner than its own length.**

Idle p90 |slope| on the Neo, by window:

| window | idle p90 \|slope\| | `steady` fires, after load stop | trend at that moment |
|---|---|---|---|
| 60 s, range only | — | 97 s | −3.21 °C/min (false positive) |
| 60 s, + slope | 2.38–2.80 °C/min | 103 s | −2.75 °C/min |
| 120 s | 0.98–1.55 °C/min | — | — |
| **180 s, + slope** | **0.15–0.49 °C/min** | **224 s** | −0.45 °C/min |

Slope resolution improves by an order of magnitude between 60 s and 180 s, but
the 180 s window imposes a 180 s floor on detection. Against an enum that clears
at 318 s, a trend-based gate therefore recovers only **~94 s**: not the 221 s
the false positive advertised.

By contrast `recovered` (a short smoothed mean compared against a *known
absolute reference*) fired at **147 s**. Detecting "close to a known value" is
a far easier estimation problem than detecting "the slope is zero", and it needs
no long window.

**If you must gate on die temperature, prefer an absolute threshold against a
cached per-host idle baseline over a steadiness or trend test**, and the catch
is that an absolute test needs a trustworthy baseline, which run 3 showed is not
automatic, hence caching a clean one per host.

But for the workload that motivated all of this, the answer turned out to be
**don't gate at all**. See [Conclusions](#conclusions). The header comment in
`macos.rs` claiming the die-trend path "settles in well under a minute" is not
supported by any of these runs; that figure came from the same class of
noise-blind measurement as the 24 s below.

## Using this to make benchmark batches reproducible

A different goal from the gate: for N identical benchmark runs you want each run
to *start in the same thermal state*, which is **not** the same as starting
cold. Equilibrium is expensive; consistency is cheap.

Settling curve from run 3 (123 s all-core load, 59.76 °C peak, settling to
37.45 °C):

| tolerance vs settled | first reaches | **and stays within** |
|---|---|---|
| 4.0 °C | 45 s | 45 s |
| 3.0 °C | 57 s | **57 s** |
| 2.0 °C | 73 s | 203 s |
| 1.5 °C | 83 s | 207 s |
| 1.0 °C | 105 s | 318 s |
| 0.5 °C | 121 s | **never** |

The gap between those columns is noise, not heat: the die *reaches* 1 °C at
105 s but keeps wandering back out until 318 s.

1. **~60 s buys 3 °C, reliably.** Tightening to 1 °C costs a further ~260 s per
   run (over 20 minutes across five runs) to gain 2 °C.
2. **±1 °C is a hard floor.** 0.5 °C never holds, because idle sigma is 1.03 °C.
   No amount of waiting controls start temperature more tightly than the sensor
   wanders on its own.
3. **Find out whether it matters before paying for it.** Run the batch
   back-to-back with no cooldown, record die temp at each run's start, and
   correlate against the result. No correlation across the resulting ~15 °C
   spread means stop gating entirely. A correlation gives a °C-per-unit
   sensitivity, which is the only non-arbitrary way to pick the tolerance, and
   if it implies tighter than ±1 °C, thermal gating cannot deliver it and the
   answer is more repetitions or randomized ordering instead.

### Short workloads need a different scale

Real benchmark units are **seconds**, not minutes, and heat arrives far faster
than it leaves. From the load phase of the 123 s Neo run:

| after load start | die rise |
|---|---|
| 2 s | **+6.9 °C** |
| 4 s | +9.6 °C |
| 6 s | +13.7 °C |
| 8 s | +14.3 °C |
| 123 s | +23.1 °C |

**A 3.5 s benchmark heats the die about 10 °C**: ten times the noise floor. So
gating a short workload is not chasing noise; the thermal difference between an
ungated first and fifth run is large and real.

But cooling is not the mirror image. Fitting the decay gives τ ≈ 15 s for the
bulk of the excursion, 44 s for the next stretch, and 55 s+ approaching
baseline; **at least two thermal masses**. Heat arrives in seconds and the last
degree leaves over minutes. That asymmetry is what makes return-to-baseline
gating so expensive on short workloads: a loop of six 3.5 s units behind six
`recovered` gates spends ~15 minutes waiting for ~23 s of work, a 38:1 ratio,
and nearly all of it in the slow tail where waiting buys least.

The conclusion is the same as the batch one, only sharper: **target a repeatable
warm limit cycle, not a return to baseline.**

Two practical settings follow:

- **Sample at 0.1 s** for second-scale work: 35 samples per 3.5 s unit. The
  sampler costs **~0.8 ms of CPU per sample** (measured, flat across intervals;
  it is the sensor read, not the loop), so 0.1 s is 0.8% of one core and cannot
  meaningfully heat what it measures. 0.02 s is 4% and is the floor the tool
  enforces.
- **Do not oversample hoping for precision.** See the fifth calibration mistake:
  at 0.25 s spacing, eight samples are worth one.

### Cycle mode

Cooling to idle between every run is the expensive way to buy a consistent
start. Letting the duty cycle reach its own **limit cycle** is usually cheaper:
it costs only the reps spent getting there, after which every remaining run
starts from the same warm temperature.

`--cycle N --load L --rest R` runs that duty cycle and records the die
temperature at the instant each rep begins:

```
=== cycle results ===
  rep   start_die   delta     peak
  1        50.66C   --         51.02C
  2        50.38C      -0.28   50.92C
  3        50.30C      -0.09   50.70C
  4        49.93C      -0.36   50.17C

=== cycle verdict ===
  start-temp spread across all 4 reps: 0.73C
  [CONVERGED] the whole batch spans 0.73C, inside the 1.03C noise floor.
              Every rep is thermally comparable; nothing to discard, and
              no warm-up needed. This rest period is already long enough.
```

Convergence is judged by **equivalence, not significance**, on both start and
peak: bound each tail's slope from above at 95% and require the worst-case drift
it permits across the tail to be within `2×` the rep-to-rep scatter already
there. "No detectable trend" is not good enough. See the seventh calibration
mistake for the false positive that forced this. The verdict is one of:

- **`[CONVERGED]` whole batch**: the rest is already long enough; use it as is.
- **`[CONVERGED]` from rep K**: discard the first K−1, or pre-warm with that
  many throwaway runs. The reported rep-to-rep scatter is the comparability you
  actually get, and it is a floor you cannot beat by waiting longer.
- **`[NOT CONVERGED]`**: the duty cycle keeps accumulating heat. Reports both
  the observed drift and the 95% worst case, and warns that a shorter tail may
  *look* flat purely for lack of data.

Peak is tested as well as start because the benchmark runs *through* the
excursion. On real Neo data the full-batch start trend scored t = 2.9 (under a
3.0 threshold), and only the peak (t = 7.8) caught it.

At a realistic short-workload duty cycle (`--cycle 5 --load 3.5 --rest 10` on
the actively-cooled Mac) it does **not** converge:

```
rep   start_die   delta
  1      50.94C   --
  2      50.93C      -0.01
  3      51.38C      +0.45
  4      52.22C      +0.84
  5      53.34C      +1.12     spread 2.40C over a 1.08C floor
```

Two things to notice. A 10 s gap is not enough even on a machine with a fan, so
expect worse on a fanless host. And the deltas are *accelerating*. This is
diverging, not settling. It is also a live example of why convergence tests the
spread: the first two deltas are −0.01 and +0.45, so a delta-based test would
have declared convergence at rep 1.

This measures a synthetic burn loop, not your benchmark. If your workload's duty
cycle differs much from `--load`/`--rest`, prefer measuring the real thing,
which is what the correlate-first experiment above does anyway.

## Cooler on versus off

Three paired runs on the Neo, identical schedules, with and without the external
cooling pad.

| | cooler ON | cooler OFF |
|---|---|---|
| **idle σ (raw)** | 0.43 °C | 0.64 °C |
| **idle σ (de-trended)** | **0.426 °C** | **0.404 °C** |
| idle baseline drift over 300 s | −0.25 °C | **+1.74 °C** (`[DRIFT]` fired) |
| peak, 120 s all-core load | 58.24 °C | 59.97 °C |
| settled / equilibrium temp | **35.01 °C** | **38.42 °C** |
| enum trips | **304 s** | **304 s** |
| enum clears | **738 s (318 s after load stop)** | **738 s (318 s after load stop)** |
| die temp when enum cleared | 34.84 °C | 38.52 °C |
| max pressure level reached | `moderate` | `moderate` |
| duty-cycle start drift | +0.041 °C/rep (~0.6 °C over 15) | +0.159 °C/rep (~2.2 °C over 15) |
| duty-cycle start spread, 15 reps | 0.82 °C | 3.00 °C |

### The enum is a pure timer: settled

Trip at **304 s** and clear at **738 s** in *both* arms. Not approximately: the
same sample. And the die temperature at the moment it cleared was **34.84 °C
cooled versus 38.52 °C uncooled**: two materially different thermal states,
one identical instant of release.

Combined with the earlier result that a 12× change in load duration moved the
clear time by one second, this is as conclusive as this hardware will allow.
`pressurelevel` is an integrator with a fixed hold-off. It is not a thermometer
in either direction, and no amount of cooling (or heating) moves it.

### Which gate is stricter *inverts* with cooling

This is the finding with teeth:

- **Cooled**: die recovered 126 s after load stop, enum cleared at 318 s →
  `[STRONG] the gate over-waits 192 s past a recovered SoC`.
- **Uncooled**: `[NOTE] pressurelevel cleared 1106 s BEFORE the die recovered`.
  The enum is now the *looser* gate.

A fixed timer is too slow when cooling is good and too fast when cooling is bad.
There is no cooling configuration for which it is correct, which is a stronger
argument against enum gating than "it waits too long" ever was.

### What the cooler actually changes

Less than expected, and not what was expected:

- **Not the noise floor.** De-trended idle σ is 0.426 vs 0.404 °C: identical.
  The apparent 0.43 → 0.64 difference was *entirely* the uncooled baseline still
  warming. Sensor noise is a property of the sensor.
- **Not peak temperature**, much: 58.24 vs 59.97 °C under a 120 s all-core load,
  1.7 °C apart.
- **Not the cooling dynamics.** Measured as excess over each arm's own settled
  point, the curves nearly overlay (t+20 s: +6.4 vs +5.4 °C; t+120 s: +2.1 vs
  +1.4 °C).
- **The equilibrium**, by 3.4 °C (35.01 vs 38.42 °C). That is the whole effect.

**It does not cool faster. It settles lower.** Worth stating plainly, because
"recovery took 10× longer uncooled" is easy to misread as slower heat shedding.
Measured as excess over each arm's *own* settled point the curves nearly
overlay; the uncooled machine sheds a given excess just as quickly. `recovered`
took 10× longer only because it is measured against a fixed pre-load baseline
that, uncooled, sat *below* the machine's own equilibrium, so it was waiting
for a temperature that arm never reaches.

Time to shed a given excess: the same. Time to reach a given *absolute*
temperature: much longer, or never. Which of those matters depends entirely on
whether anything is actually waiting on it.

### The uncooled "1424 s recovery" is an artifact, not a measurement

The uncooled run reports `recovered` 1424 s after load stop. That number is not
real. Its pre-load baseline drifted **+2.82 °C during the 300 s baseline**
(`[DRIFT]` fired at 3.1× sigma) because the machine was still warming from cold,
so the 35.94 °C "baseline" was recorded *below the machine's own idle
equilibrium of 38.42 °C*. `recovered` was therefore chasing a temperature the
uncooled machine never returns to, and only fired at all on a noise dip at the
very end of the run.

Two lessons. **An uncooled Neo needs far longer than 300 s of idle before its
baseline means anything**: budget several times that. And this is exactly the
failure the `[DRIFT]` warning exists to catch; it caught it, and the number
would have been believed without it.

## If a second MacBook Neo arrives

Everything here is one physical unit, and that unit is **thermally modified**:
a passive pad bonded inside the case coupling the SoC to the chassis, plus an
external active pad under it. The internal pad cannot be removed between runs,
so **no measurement of a stock Neo exists and none can be produced from this
unit.**

A second machine is the only chance to fix that, and most of the value is in the
first hour of owning it.

### Measure it stock, before touching anything

> **This is the irreversible step.** Once the internal pad is bonded in, the
> stock configuration is gone permanently. Whatever else gets skipped, run the
> three commands below on the machine as it arrives.

```bash
export THERMAL_PROBE_CALIB=calib-unit2-stock.txt
./thermal-probe 300 0 60 2 1                                      # idle noise
./thermal-probe 300 120 60 2 <cores> 900 3600                     # load + cooldown
./thermal-probe --cycle 15 --load 3.5 --rest 30 --interval 0.1 --baseline 300
```

Then, if the unit is going to be modified anyway, repeat after **each**
modification: internal pad, then external pad. Three configurations measured
separately give the contribution of each intervention, where unit 1 can only
ever show the last one. Use a **separate `THERMAL_PROBE_CALIB` file per
configuration**; the cache validates the sampling window but has no idea the
hardware changed.

### What is likely to be boring, and what is not

Worth setting expectations, because they differ sharply by finding.

**Expect no news from the enum.** The 318 s hold-off, the ~3 s trip latency, the
engagement on a single quantization step, and the lock-step with
`ProcessInfo.thermalState` are OS behavior, and they survived a 12× load range
and a large change in cooling on unit 1. A second unit reproducing them adds
little. *A second unit **not** reproducing them would be a significant finding*.
it would mean the hold-off is not a fixed OS constant but varies by unit or by
OS build, which changes how much any of this generalizes. Record the macOS
version alongside the result either way; unit 1's data does not have it.

**Expect genuine news from the noise floor.** This is the number every derived
threshold rests on, and it is the least stable thing measured: de-trended σ
ranged 0.40–0.93 °C *on one machine* across sessions, with lag-1 autocorrelation
from +0.07 to +0.80, and drift explains only part of that. What is unknown is
whether that scatter is within-unit session variance or whether units differ
systematically. If a second unit sits reliably outside unit 1's range, per-unit
calibration stops being a convenience and becomes mandatory.

**Expect the real news from stock.** Unit 1 never approached throttling and
never escalated past `moderate`, but it had a heat sink it does not ship with.
A stock Neo is the first chance to see whether the machine as sold reaches a
regime where temperature actually moves benchmark numbers. Two outcomes would
matter a great deal:

- **Pressure escalates above `moderate`.** That would be the first observation
  of `heavy` or beyond on any hardware here, finally exercising four of
  `format_pressure_word`'s five cases, which remain unverified against silicon.
- **Results start tracking temperature.** The "don't gate" conclusion was
  established on a modified machine across a ~3 °C spread. A stock unit with a
  wider spread, or one that throttles, could invalidate it for stock fleet
  devices while leaving it true for this rig.

### Comparison targets from unit 1

Values to diff against, all cooled-with-external-pad unless noted:

| Quantity | Unit 1 |
| --- | --- |
| `PMU tdie` sensor count | 7 |
| Sensor quantization | 0.10 °C |
| Idle σ, de-trended, settled baseline | 0.40–0.43 °C |
| Idle σ across sessions (same unit) | 0.40–0.93 °C |
| Idle lag-1 autocorrelation | +0.07 to +0.80 |
| Idle peak-to-peak, 300 s | 3.79 °C |
| Idle equilibrium: external pad on / off | 35.0 / 38.4 °C |
| Peak, 120 s all-core load: on / off | 58.2 / 60.0 °C |
| Enum trip, after load start | ~2.9–4 s |
| Enum clear, after load stop | 317–318 s |
| Max pressure level ever observed | `moderate` |
| Duty-cycle start spread, 15 reps: on / off | 0.82 / 3.00 °C |
| Rep-to-rep scatter, start / peak | 0.10 / 0.46 °C |

### Before generating a fresh dataset

The [replay and self-test work](#known-gaps) is still unbuilt, and a second unit
is the moment it pays for itself. Every statistical bug in this tool's history
was found by running it and squinting at the output (seven of them), and a new
machine means a new dataset with no prior expectations to check the numbers
against. Being able to replay unit 1's recorded traces and confirm the tool
still reproduces its known-good answers is worth more when there is fresh data
that nobody can sanity-check by eye.

Keep unit 1's raw logs. They are the regression corpus.

## Private API note

Die temperature uses the same private IOHID path as the iOS client's
`pipette_soc_temp`
([`ios/Pipette/Pipette/Native/PipetteThermal.m`](../../ios/Pipette/Pipette/Native/PipetteThermal.m)):
symbols via `dlsym`, `PrimaryUsagePage 0xff00` / `PrimaryUsage 0x0005`
matching, `PMU tdie*` services, max reading. Those sensor names are identical
on Apple Silicon Macs, and no root is needed. On iOS this path is gated
behind `PIPETTE_PRIVATE_THERMAL` because App Review rejects it; that
constraint doesn't apply to a locally built CLI, but it's why this stays a
diagnostic and nothing links it into a shipped binary.

## Status

Diagnostic; not wired into CI, not built by the workspace. **Keep it**, and no
longer as a one-off: the question that prompted it is answered, but the thing it
turned out to be good at is characterizing a machine's thermal behavior, and
that keeps being needed.

Reach for it when:

- a new host class joins the fleet and its noise floor is unknown (the Neo and
  the 14-core Mac differ by 3× in idle sigma and behave oppositely at the enum);
- something in the repo wants to compare a die temperature against a number;
- a benchmark's numbers look unstable and you need to rule thermals in or out;
- a workload changes shape enough that the "don't gate" conclusion above should
  be re-derived rather than inherited.

The pressure-level mapping above `nominal` is still only confirmed at `moderate`
on real hardware. `heavy`, `trapping`, and `sleeping` remain unexercised, so
`format_pressure_word` is verified for exactly one of its five cases.

### Known gaps

- **The Neo numbers above predate the slope test and `recovered` exit.** Run 3's
  97 s `steady` is the false positive that motivated the fix; the corrected
  figure (224 s at a 180 s window) is inferred from replaying its trace, not
  from a fresh run. Re-running the Neo would confirm it directly.
- **`steady_secs` still defaults to 60 s**, which the same analysis shows is too
  short to resolve trend on a noisy host. It is left as the default because a
  180 s window imposes a 180 s reporting floor; pass it explicitly when the
  steadiness figure matters.
- **Cycle mode drives a synthetic burn loop**, not a real benchmark, so its
  limit cycle only transfers insofar as the duty cycles match.
- **The duty-cycle logs used the pre-equivalence binary.** Both cooler-on and
  cooler-off `--cycle` runs printed `[CONVERGED]` from the old significance
  test; re-derived with the corrected rule, **neither converges** (cooled drift
  bound 0.81 °C vs 0.27 °C scatter; uncooled 3.02 vs 0.88). The cooled arm is
  close (+0.041 °C/rep, ~0.6 °C over 15 reps), but neither is provably flat.
  Worth re-running both with the current binary.
- **Still no observation above `moderate`.** Even uncooled at 60 °C peak, the
  enum never escalates, so four of `format_pressure_word`'s five cases remain
  unverified against hardware.
- **No replay or self-test yet**, and this is the significant one. Every bug in
  the list above (seven statistical mistakes, a hung loop from a truncated
  fractional interval, an 18% timing drift) was found by *running* the tool,
  never by reading it. A `--replay` mode over the recorded traces would turn
  each past run into a regression test with known-good answers, and a synthetic
  self-test would cover the statistics directly. Both are still unbuilt.
- **The conclusions rest on one host and one workload.** Everything in
  [Conclusions](#conclusions) was measured on a single Neo running one benchmark
  shape, and that unit is thermally modified in two ways. **No stock-Neo data
  exists**, and the internal pad is permanent, so none can be produced from this
  machine. If a second unit ever arrives, see
  [If a second MacBook Neo arrives](#if-a-second-macbook-neo-arrives). The
  stock measurement has to happen before it is modified or the chance is gone.

Closed since the last revision:

- ~~Does a 3.5 s load trip the enum?~~ Yes: 2.9 s in, and it never clears
  during a batch.
- ~~Does die temperature affect the benchmark result?~~ No, not across the ~3 °C
  a batch produces. Measured against the real workload.
