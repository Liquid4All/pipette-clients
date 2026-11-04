# MacBook Pro (M5 Max) Thermal Behavior

The readiness gate holds each rep (one timed run of a cell) until the machine
is in a known state, so that two runs are comparable. On the M5 Max, the macOS
thermal-pressure level alone does not give that state. A benchmark batch heated
the die from 35 °C to 72 °C while the level read `nominal`. Cells measured near
the top of that climb came out up to 11 % slower than the hardware is.

This article records what we measured on that machine, and why the macOS gate
reads the die temperature as well as the level. It is the M5 Max counterpart to
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md), and the detailed
backing for the die criterion in
[End-to-end latency → macOS](end-to-end-latency.md#macos).

## Key findings

**The machine loses about 13 % throughput under sustained load, and does not get
it back while the load continues.** The loss comes between minutes 2 and 4 and
stays flat after that. It is a second, lower steady state, not a transient that
averaging removes.

**Recovery is fast and consistent: about 65 s, at about 49 °C, on all three
hosts.**

**The thermal-pressure level cannot give a comparable start state.** It engages
only after 92–118 s of *continuous* load. The gaps between cells (model binding,
dispatch, result recording) restart that count, while the heat continues to
build.

**Die temperature is hysteretic, but the gate reads only the cooling branch.**
One reading can mean full speed (while the die warms) or throttled (while it
cools), so temperature is not a general throttle indicator. The gate samples
only between reps on an idle machine, and the load criterion holds a machine
that another process loads. The gate therefore reads only the cooling branch,
where the relation is monotonic.

**The `die < 50 °C` criterion removes a systematic bias, not only noise.** With
the level alone, the heavy cells were measured hot and reported slower than the
hardware is.

## Measurement rig

- **Hosts:** three MacBook Pro, each with an 18-core M5 Max and 48 GB, stock
  cooling, on mains power.
- **Runtime:** llama.cpp on Metal (`-ngl 99`, flash attention on).
- **Models:** LFM2.5-2.6B, Qwen3.5-4B and Qwen3.5-9B, all Q4_K_M.
- **Cells:** the tables use short names. "e2e 4096" is end-to-end latency with a
  4,096-token prompt, "prefill 8192" is prefill throughput at 8,192 tokens, and
  "4B" and "9B" are the Qwen3.5 models.
- **Signals:** the die temperature is the maximum over the host's `PMU tdie*`
  sensors (42 on these machines). The thermal-pressure level is
  `OSThermalPressureLevel`: 0 `nominal`, 1 `moderate`, 2 `heavy`.

## The failure with the level alone

These are the per-rep die temperatures from four consecutive cells of one plan,
on one host. Each cell measures five reps. The gate, with the level as its only
thermal criterion, ran before each rep and passed each time.

| cell             | die °C at each of 5 reps | rep spread |
|------------------|--------------------------|-----------:|
| e2e 4096, 4B     | 35, 37, 38, 40, 42       |      0.07% |
| prefill 8192, 4B | 45, 48, 51, 55, 58       |      3.3%  |
| e2e 4096, 9B     | 62, 63, 64, 66, 68       |      2.1%  |
| prefill 8192, 9B | 69, 70, 71, 72, **45**   |      6.8%  |

The table shows two defects.

**The batch heats from cell to cell.** Each cell starts hotter than the previous
cell finished, because no criterion of the gate holds it. From the third cell
on, every rep starts at 62–68 °C, where throughput is measurably lower.

**The last cell mixes two states.** Four reps ran at 69–72 °C and the fifth at
45 °C, because the level engaged during the cell and forced a wait. The reported
mean of that cell averages four throttled reps with one cold rep. It measures
neither state, and its 6.8 % spread comes from that step, not from noise.

The rep spread follows the temperature span of each cell. The only cell that
stayed cold is the only tight one.

## Why the level behaves this way

Measured during a 600 s continuous load from a cooled start:

| t, load starts at 0 | prefill vs cold | event                        |
|--------------------:|----------------:|------------------------------|
|             0–20 s  |          +0.02% | full speed                   |
|            40–60 s  |          −7.62% | transient dip, level nominal |
|           80–100 s  |          −1.13% | recovers; level → `moderate` |
|          100–120 s  |          −1.53% | level → `heavy` (118 s)      |
|          120–140 s  |          −6.22% | sustained decline begins     |
|          220–240 s  |         −14.74% |                              |

On this host, the level engages at the right time: at 92–118 s, just before the
sustained decline at about 120 s. The problem is that it needs about two minutes
of continuous load to engage at all. A benchmark plan does not supply that, even
while it heats the machine.

The transient dip at 40–60 s has no known cause. It recovers to −1.1 % before
the sustained decline starts. If it is real and not an artifact of one run, no
available signal detects it.

## Recovery, and why 50 °C

The machine was driven into the throttled state. Then a ~1 s canary cell
(prefill at 2,048 tokens, 3 reps) probed the recovery. The cold reference was
6979 ± 3 tokens per second (t/s). The level column uses the codes in
[Measurement rig](#measurement-rig).

| t after load stops | canary t/s | vs cold | die °C | level |
|-------------------:|-----------:|--------:|-------:|------:|
|                0 s |       5954 |  −14.7% |   68.8 |     2 |
|               17 s |       6665 |   −4.5% |   63.1 |     2 |
|               33 s |       6919 |   −0.9% |   56.8 |     2 |
|               65 s |       6978 |  −0.02% |   48.5 |     1 |
|               96 s |       6984 |  +0.07% |   43.8 |     0 |
|              613 s |       6981 |    0.0% |   34.9 |     0 |
|      three hosts   |            |         | 49.4 / 48.5 / 49.1 at recovery |

A `die < 50 °C` rule admits only probes within 0.02 % of the cold reference. The
threshold is about 15 °C above the idle baseline of these hosts. That is far
outside the ~4 °C peak-to-peak idle sensor noise, which rules out tighter
thresholds. The rule has the same form as the other platforms' criteria
(Android `< 34 °C`, iOS `< 36 °C`).

At the recovery point, the level still read `moderate` on all three hosts, and
it cleared at about 96 s. In a separate 600 s soak, it cleared 52 s after the
load stopped, which is *before* the 65 s recovery measured here. The level can
therefore clear before or after full speed returns, so it is not a conservative
signal.

## The die criterion and its effect

The macOS gate requires the level `nominal` and `die < 50 °C`. Both must pass,
so the gate is never less strict than the level alone.

The same twelve cells (four cells on each of the three hosts) ran once with the
level alone and once with both criteria:

| cell             | level only, mean (spread) | level + die, mean (spread) |
|------------------|--------------------------:|---------------------------:|
| e2e 4096, 4B     |          3151 ms (0.09%)  |           3151 ms (0.10%)  |
| prefill 8192, 4B |          2383 ms (3.25%)  |           2313 ms (0.06%)  |
| e2e 4096, 9B     |          5012 ms (2.12%)  |           4877 ms (0.09%)  |
| prefill 8192, 9B |          4016 ms (7.17%)  |           3628 ms (0.03%)  |

Within-cell spread fell from **3.16 % to 0.07 %**, 45 times tighter. On the
heaviest cell, the spread across the three hosts fell from 2.7 % to 0.06 %: the
three machines land within 2 ms of each other.

The means moved too, and that is the more important result. With the level
alone, the heavy cells were reported **up to 11 % slower than the hardware is**.
That error was a systematic bias from measuring at 69–72 °C, not noise, and more
reps would not remove it.

`e2e 4096, 4B` is the control. It did not go above 42 °C with the level alone,
so the die criterion had nothing to correct, and its result did not change. This
shows that the difference comes from temperature, and not from the gate itself.

The per-rep temperatures with both criteria, for the same cells as the failure
table, were `50, 49, 50, 50, 50`, against `69, 70, 71, 72, 45` with the level
alone. The recorded value is rounded to whole °C, so each `50` was below 50 °C.
Neither defect remains. The gate releases each rep just below the threshold, so
each wait is short. The long ~65 s recovery is the cost of reaching 72 °C, and
the criterion prevents that.

## Cost

The same set of cells took **2.2 times as long**: 238 s with the level alone and
522 s with both criteria. Each host had 37 gate waits, no timeouts and no failed
cells. These cells are the heaviest work in the matrix (8192-token prefill on a
9B model), and the control cell did not wait at all. This is therefore a worst
case, not a typical one.

## Open questions

**The 50 °C threshold is calibrated for the M5 Max and does not transfer.**
`die_temp_max_c` is a max over a per-host sensor count (42 on these machines,
against 7 and 20 on the machines in
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md)), so the same
number means different things on different hardware. A Mac of another type
needs its own threshold, measured on that type. Until it has one, a die that
reads is gated on a number that does not fit it, and a die that does not read
passes. To set it, see
[Tuning the readiness gate](../pipette-cli/usage.md#tuning-the-readiness-gate).

**The mechanism is not established.** Classic thermal throttling is ruled out:
the die *falls* from 73.9 °C to 65.9 °C while throughput continues to drop, so
the chip does not respond to heat. A sustained power or energy budget fits the
observations, but no experiment separated it from heat soak of the chassis. The
decisive test is forced cooling during recovery:

- If performance returns in step with a faster temperature drop, temperature is
  causal.
- If recovery still takes about 65 s, the die reading only tracks elapsed idle
  time, and it would decouple from performance on an actively cooled machine.

**The 40–60 s transient dip comes from one run**, and its cause is not known.

**All runs used llama.cpp on Metal** with the models in
[Measurement rig](#measurement-rig). A CPU-bound or memory-bound workload may
throttle differently.
