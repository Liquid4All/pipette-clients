# macos-thermal-probe

Characterizes how a Mac sheds heat and how the macOS thermal-pressure enum
relates to it. It samples `notify_get_state(kOSThermalNotificationPressureLevelName)`
(the level the macOS readiness gate reads), `ProcessInfo.thermalState`, and IOHID
die temperature side by side, and derives every threshold it reports from the
host's own idle noise rather than from constants.

What it found on the MacBook Neo, and why the benchmark setup looks the way it
does, is in
[MacBook Neo thermal behavior](../../docs/methodology/macbook-neo-thermal-behavior.md).
The gate's criteria are in
[System Readiness Control](../../docs/methodology/end-to-end-latency.md#system-readiness-control).

It answers two questions:

- **Gate evidence**: how long does the enum keep reading non-nominal after the
  die has cooled, and would die temperature do better?
- **Batch spacing**: how long between repeated runs to get a consistent thermal
  starting point?

## Build / run

macOS only. No dependencies beyond the SDK.

```bash
clang -O0 -framework Foundation -framework IOKit -o thermal-probe thermal-probe.m
```

`-O0` is deliberate: the burn loop must not be optimized away.

It saturates every core. Do not run it at the same time as a benchmark.

**Single-shot**: idle baseline, one all-core load, then cooldown. For gate
evidence and the settling curve.

```bash
./thermal-probe                    # 300s baseline, 300s load, 60s steady window, 2s samples
./thermal-probe 300 600 60 2       # longer load if the host resists heating
./thermal-probe 300 300 60 2 4     # pin to 4 burn threads
./thermal-probe 300 0 60 2 1       # no load: characterize this host's idle noise
```

Positional args, with defaults:
`[baseline_secs=300] [load_secs=300] [steady_secs=60] [interval_secs=2]
[threads=all cores] [nominal_cap_secs=900] [max_cool_secs=1800]`.

**Cycle**: repeated load/rest, for choosing the gap between batch runs.

```bash
./thermal-probe --cycle 15 --load 3.5 --rest 30 --interval 0.1 --baseline 300
```

Flags, with defaults: `--cycle 5 --load 60 --rest 45 [--interval 2]
[--threads <all cores>] [--baseline 60]`. Convergence needs at least 5 reps.

All durations are fractional seconds in both modes. The interval is floored at
0.02 s. For second-scale work sample at 0.1 s: the sampler costs about 0.8 ms of
CPU per sample (the sensor read, flat across intervals), so 0.1 s is 0.8% of one
core. Sampling faster gives more time resolution, not more precision: the noise is
autocorrelated, so closely spaced samples are worth far fewer independent ones.

### Before measuring

The single-shot run blocks until both enums read nominal, up to
`nominal_cap_secs` (0 disables). A run that starts above nominal is measuring
the tail of an earlier thermal event, so if the cap expires the verdict is
flagged `[CONFOUNDED]`. On a fanless host back-to-back runs spend several
minutes here.

Nominal does not mean settled: the enum is a release timer, so it can read
nominal while the die is still moving. The baseline is checked for drift and
`[DRIFT]` is printed when the total change implied by its linear fit exceeds
its own sigma. Every figure derived from a drifting baseline is biased; re-run
from idle. On an uncooled Neo, 300 s of idle is not enough.

### Calibration cache

`baseline_secs` should be at least 3x `steady_secs`, because the steady band is
calibrated from windows slid over the baseline. To avoid paying that every run,
set `THERMAL_PROBE_CALIB` to a file path: a baseline long enough to calibrate
writes its noise figures there, and later runs with a short baseline reuse them.
Only the noise is cached; the baseline mean is always re-measured. A cache taken
for a different window or interval is rejected, with the reason printed.

The cache validates the window, not the machine. Use a separate file per
physical configuration (for example `calib-cooled.txt` and
`calib-uncooled.txt`), or one setup's noise floor is silently applied to
another. Idle noise also varies between sessions on one machine, so treat the
cache as a convenience within a stable setup and re-measure when anything
changes.

## How it decides

Every threshold is derived from the idle baseline and printed in the summary,
so it can be audited. Example at the default 2 s interval:

```
die temp noise (baseline)    sigma=0.54C spread=2.68C trend=-0.93 C/min over 150 idle samples
noise autocorrelation        lag-1 +0.68 at 2s spacing -> correlation time ~5.2s
                             1 samples over 2s are worth 1.0 independent ones
idle window stats            range med=1.39C p90=1.93C | |slope| med=0.30 p90=0.71 C/min (120 windows)
steady band (derived)        1.93C = p90 of idle window range
slope band (derived)         0.71 C/min = p90 of idle window |slope|
                             tested over a 31-sample window (60s)
recovered threshold          within 1.08C of baseline on a 1-sample mean
```

- **`recovered`**: a 2 s smoothed mean back within
  `max(1.0 °C, 2σ/√N_eff)` of the pre-load baseline, where
  `N_eff = N(1-r)/(1+r)` corrects for lag-1 autocorrelation `r`. Smoothing is a
  duration, not a sample count, because the noise decorrelates on a timescale.
- **`steady`**: a full `steady_secs` window whose range is inside the steady
  band and whose least-squares slope is inside the slope band. The steady band
  is the p90 of the ranges of the same window slid over the idle baseline,
  floored at 0.20 °C (the sensors quantize at about 0.10 °C), with a `4σ`
  fallback when the baseline holds fewer than 10 windows. The slope band is the
  p90 of idle window |slope|. A range test alone cannot tell flat from
  descending, hence both. The default 60 s window is too short to resolve
  trend on a noisy host (on the Neo, idle p90 |slope| was 2.4 to 2.8 °C/min at
  60 s and 0.15 to 0.49 at 180 s); pass `steady_secs=180` when the `steady`
  figure matters.
- **Cooldown exit**: `recovered` and `pressurelevel` nominal, or
  `max_cool_secs` from load stop. `steady` is reported but does not gate the
  exit: resolving a zero slope against this noise needs a long window, and a
  long window cannot report sooner than its own length. If die sensors are
  unavailable, the enum alone ends the cooldown.
- **Settling curve**: for tolerances of 4.0 down to 0.5 °C, when the smoothed die
  first came within, and then stayed within, that distance of where it settled
  (the mean of the last 90 s of cooldown, not the pre-load baseline). That is
  the batch-spacing question; `recovered` is the return-to-idle one.
  Tolerances under about 2σ cannot be held at all.
- **Cycle convergence**: equivalence, not significance. For each tail of reps,
  bound the slope of start and peak temperature from above at 95% and require
  the worst-case drift across the tail to be within 2x the rep-to-rep scatter
  already in it. "No detectable trend" is not enough, because a shrinking tail
  loses the power to detect a trend before the trend stops. Peak is tested as
  well as start because the benchmark runs through the excursion.

## Interpreting the output

| Column | What it is |
|---|---|
| `ProcessInfo` | `ProcessInfo.thermalState`, 0 to 3 (nominal/fair/serious/critical). |
| `pressurelevel` | `notify_get_state` on the linked `kOSThermalNotificationPressureLevelName`, 0 to 4 on macOS (nominal/moderate/heavy/trapping/sleeping). What the macOS readiness gate reads. |
| `tdie_max` | Hottest `PMU tdie*` IOHID sensor, °C. |
| `pressure` | The undocumented `com.apple.system.thermalpressure`. |
| `bogus` | A nonsense notify name. `notify(3)` accepts any name and reports 0, so this must stay 0; if it doesn't, ignore the run. |

Recovery times are reported absolutely and relative to load stop, since load
stop is when a readiness gate starts counting against its deadline.

Single-shot `=== verdict ===` labels:

- **`[CRITICAL]`**: the enum took longer than the macOS gate's 420 s
  `DEFAULT_MAX_WAIT` to clear, or never cleared. The gate would time out and
  fail the cell.
- **`[STRONG]`**: the die recovered N s before the enum cleared; N is the
  per-cooldown over-wait. A following `[NOTE]` that the die was already at
  baseline when the enum cleared means the enum is on a timer, so cooling
  harder will not clear it sooner.
- **`[NOTE]` enum cleared before the die recovered**: the enum is the looser
  criterion here.
- **`[NOTE]` agree within one sample interval**: enum and die agree.
- **`[CONFOUNDED]`**: the run never reached nominal before measuring; every
  recovery number is an upper bound.
- **`[PARTIAL]`**: the cooldown hit `max_cool_secs`, or the die never came back
  within the recovered threshold. Raise `max_cool_secs`.
- **`[WEAK]`**: the steady band is more than half the load-induced rise, so
  `steady` resolves little on this run.
- **`[FAIL]`**: `ProcessInfo` moved but `pressurelevel` stayed 0, so the notify
  read is not tracking.
- **`[INCONCLUSIVE]`**: neither enum left nominal. It reports whether the die
  rose anyway, separating "the load didn't land" from "the enums never tripped".
- **`[BAD]`**: the bogus control went non-zero.
- **`[SKIP]`**: die temperature unavailable.
- **`[OK]`**: the notify signal is live, and how its clear time compares with
  `ProcessInfo` and the deadline.

Cycle `=== cycle verdict ===` labels:

- **`[CONVERGED]` from rep K**: discard the first K-1 reps or pre-warm with that
  many throwaway runs (rep 1 means the rest is already long enough). The
  reported rep-to-rep scatter is the comparability you get.
- **`[NOT CONVERGED]`**: the duty cycle keeps accumulating heat; lengthen the
  rest. Reports observed drift and the 95% worst case.
- **`[SKIP]`**: fewer than 5 reps.

Cycle mode drives a synthetic burn loop, so its limit cycle transfers to a real
benchmark only insofar as the duty cycles match.

## Pitfalls

- A threshold chosen by judgment rather than measured against the host's noise
  is the recurring mistake: fixed sigma multiples, a-priori band ceilings, and
  range or spread tests that cannot see a slow trend all gave false results here.
- Idle noise differs between hosts and between sessions; a calibration from one
  does not carry to the other.
- There is no replay or self-test mode. Every statistical bug so far was found
  by running the tool and reading its output, so keep raw logs of past runs to
  check a change to the statistics against. The two 15-rep `--cycle` logs of the
  MacBook Neo study (cooler on and off) came from a build without the
  equivalence test and print `[CONVERGED]`. The equivalence test finds that
  neither converges.

## Private API note

Die temperature uses the same private IOHID path as the iOS client's
`pipette_soc_temp`
([`ios/Pipette/Pipette/Native/PipetteThermal.m`](../../ios/Pipette/Pipette/Native/PipetteThermal.m)):
symbols via `dlsym`, `PrimaryUsagePage 0xff00` / `PrimaryUsage 0x0005`
matching, `PMU tdie*` services, max reading. No root is needed. On iOS this path
is gated behind `PIPETTE_PRIVATE_THERMAL` because App Review rejects it. macOS
has no such gate: the shipped CLI reads the same sensors through pipette-device's
die-temperature module, which the macOS readiness gate uses.

## Status

Diagnostic: the workspace does not build it, and CI does not run it. Use it
when:

- a host class joins the fleet and its noise floor is unknown;
- code needs to compare a die temperature against a number;
- benchmark numbers look unstable and thermal effects are a possible cause;
- a workload changes shape enough that the Neo conclusions need to be measured
  again for it.

If an unmodified Neo arrives, measure it before fitting any thermal pad: the
measured unit's internal pad is permanent, so stock data can only come from a
new unit. Run the three commands in
[Reproducing these measurements](../../docs/methodology/macbook-neo-thermal-behavior.md#reproducing-these-measurements)
once per configuration, and record the macOS version (the first unit's data
lacks it, so a different enum hold-off could not be told apart from an OS
change).
