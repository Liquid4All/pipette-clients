# MacBook Pro (M5 Max) Thermal Behavior

The readiness gate exists so that two runs are comparable: it holds a
measurement until the machine is in a known state. On the M5 Max it was not
doing that. The operating system's thermal-pressure enum let a benchmark batch
heat from 35 °C to 72 °C without ever leaving `nominal`, and cells measured near
the top of that climb came out up to 11% slower than the hardware actually is.

This article records what we measured on that machine and why the macOS gate now
reads die temperature as well as the enum. It is the M5 Max counterpart to
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md), and the detailed
backing for the macOS row in
[Device conditions](device-conditions.md#per-device-we-measured).

Three hosts were measured (`boston-mbp-m5-1/2/3`), all 18-core M5 Max, 48 GB,
stock cooling, on mains power.

## Key findings

**The machine loses ~13% throughput under sustained load, and keeps it.** The
loss lands between minutes 2 and 4 and is flat after: a second, lower steady
state rather than a transient to average out.

**Recovery is fast and consistent: ~65 s, at ~49 °C, on all three hosts.**

**The pressure enum cannot deliver a comparable starting state.** It engages
only after ~118 s of *continuous* load, and the gaps between cells (model
binding, dispatch, result recording) keep resetting that clock while the heat
keeps accumulating.

**Die temperature is hysteretic, but the gate never sees the ambiguous branch.**
The same reading means "full speed" warming and "throttled" cooling, so
temperature is useless as a general throttle indicator. The gate samples only
between repetitions on an idle machine, and the load criterion rejects a machine
something else is loading, so it reads only the cooling branch, where the
relation is monotonic.

**Adding a `die < 50 °C` criterion removed a systematic bias, not just noise.**
The heavy cells had been reported slower than the hardware is, because they were
measured hot.

## The failure, as recorded

Per-repetition die temperatures from four consecutive cells of one plan slice,
on one host. Each cell measures five repetitions; the gate ran before every one
of them and passed every time.

| cell             | die °C at each of 5 reps | rep spread |
|------------------|--------------------------|-----------:|
| e2e 4096, 4B     | 35, 37, 38, 40, 42       |      0.07% |
| prefill 8192, 4B | 45, 48, 51, 55, 58       |      3.3%  |
| e2e 4096, 9B     | 62, 63, 64, 66, 68       |      2.1%  |
| prefill 8192, 9B | 69, 70, 71, 72, **45**   |      6.8%  |

Two distinct defects are visible.

The batch **ratchets**: each cell starts hotter than the last finished, because
nothing in the gate objects. By the third cell every repetition begins at
62–68 °C, where throughput is measurably down.

The last row **straddles**: four repetitions ran at 69–72 °C and the fifth at
45 °C, after the enum finally tripped mid-cell and forced a wait. That cell's
reported mean averages four throttled repetitions with one cold one. It is not a
measurement of either state, and its 6.8% spread is that discontinuity rather
than noise.

Rep spread tracks the temperature span of the cell throughout. The only cell that
stayed cold is the only one that is tight.

## Why the enum behaves this way

Measured against a 600 s continuous load from a cooled start:

| t, load starts at 0 | prefill vs cold | event                       |
|--------------------:|----------------:|-----------------------------|
|             0–20 s  |          +0.02% | full speed                  |
|            40–60 s  |          −7.62% | transient dip, enum nominal |
|           80–100 s  |          −1.13% | recovers; enum → `moderate` |
|          100–120 s  |          −1.53% | enum → `heavy` (118 s)      |
|          120–140 s  |          −6.22% | sustained decline begins    |
|          220–240 s  |         −14.74% |                             |

On this host the enum's *timing* is good. It engages at 92–118 s, just before
the sustained decline at ~120 s. The problem is not that it fires late. It is
that it needs two minutes of uninterrupted load to fire at all, and a benchmark
plan does not supply that even while it heats the machine.

The transient dip at 40–60 s is unexplained. It recovers to −1.1% before the
permanent decline starts. If it is real rather than an artifact of one run, it is
a window no available signal covers.

## Recovery, and why 50 °C

Driving the machine into the throttled state and probing recovery with a ~1 s
canary (prefill 2048, `r=3`) against a cold reference of 6979 ± 3 t/s:

| t after load stops | canary t/s | vs cold | die °C | enum |
|-------------------:|-----------:|--------:|-------:|-----:|
|                0 s |       5954 |  −14.7% |   68.8 |    2 |
|               17 s |       6665 |   −4.5% |   63.1 |    2 |
|               33 s |       6919 |   −0.9% |   56.8 |    2 |
|               65 s |       6978 |  −0.02% |   48.5 |    1 |
|               96 s |       6984 |  +0.07% |   43.8 |    0 |
|              613 s |       6981 |    0.0% |   34.9 |    0 |
|      three hosts   |            |         | 49.4 / 48.5 / 49.1 at recovery |

A `die < 50 °C` rule would have admitted only probes within 0.02% of the cold
reference. It sits ~15 °C above these hosts' idle baseline (far outside the
~4 °C peak-to-peak idle sensor noise that rules out tighter thresholds), and
follows the same shape as the other platforms' criteria (Android `< 34 °C`, iOS
`< 36 °C`).

Note the enum still read `moderate` at the recovery point on all three hosts,
clearing around 96 s. It is close, but it brackets rather than clears the
recovery point: in a separate 600 s soak it cleared 52 s after load stopped,
*earlier* than the 65 s recovery measured here. It is not reliably conservative.

## What changed, and what it bought

The gate now requires `die < 50 °C` in addition to `nominal`. Both must pass, so
it is strictly stricter than before.

Re-running the identical twelve cells across the three hosts:

| cell             | before, mean (spread) | after, mean (spread) |
|------------------|----------------------:|---------------------:|
| e2e 4096, 4B     |      3151 ms (0.09%)  |     3151 ms (0.10%)  |
| prefill 8192, 4B |      2383 ms (3.25%)  |     2313 ms (0.06%)  |
| e2e 4096, 9B     |      5012 ms (2.12%)  |     4877 ms (0.09%)  |
| prefill 8192, 9B |      4016 ms (7.17%)  |     3628 ms (0.03%)  |

Within-cell spread fell **3.16% → 0.07%**, 45× tighter. Cross-host agreement on
the heaviest cell went from a 2.7% spread to 0.06%: the three machines now land
within 2 ms of each other.

The means moved too, and that is the more important result: the heavy cells were
reported **up to 11% slower than the hardware is**. That was systematic bias from
measuring at 69–72 °C, not noise, and averaging more repetitions would never have
removed it.

`e2e 4096, 4B` is the control. It never exceeded 42 °C even under the old gate,
so it had nothing to fix, and it is unchanged. That is what makes the difference
attributable to temperature rather than to the change itself.

Per-repetition temperatures after the change, same cells as the failure table:
`50, 49, 50, 50, 50` where they had been `69, 70, 71, 72, 45`. The ratchet and
the straddle are both gone. The machine pins itself just under the threshold,
which is why cooling stays cheap. The expensive ~65 s recovery is the cost of
reaching 72 °C in the first place, and the criterion prevents that.

## Cost

The same slice took **2.2× longer**: 238 s → 522 s, with 37 gate waits per host,
no timeouts and no failed cells. That slice is deliberately the heaviest work in
the matrix (8192-token prefill on a 9B model); the control cell waits not at all,
so this is a worst case rather than a typical one.

## Limits of these measurements

**The 50 °C threshold is calibrated for the M5 Max and does not transfer.**
`die_temp_max_c` is a max over a per-host sensor count (42 on these machines
against 7 and 20 on the machines in
[MacBook Neo thermal behavior](macbook-neo-thermal-behavior.md)), so the same
number means different things on different hardware. A portable version needs a
per-host idle baseline plus a delta. Until that exists, an unreadable sensor
waives the criterion rather than failing the cell, which keeps uncalibrated hosts
on the previous behavior; but a Mac with *readable* sensors and a different
baseline would be gated on a number that does not fit it.

**The mechanism is not established.** Classic thermal throttling is ruled out;
the die *falls* from 73.9 °C to 65.9 °C while throughput keeps dropping, so heat
is not what the chip is responding to. A sustained power/energy budget fits the
observations, but no experiment has distinguished it from chassis heat-soak
equilibrium. The decisive test is forced cooling during recovery: if performance
returns in step with an accelerated temperature drop, temperature is causal; if
it still takes ~65 s, the die reading is a correlate of elapsed idle time and
would decouple on an actively cooled machine.

**The 40–60 s transient dip is one run**, and unexplained.

**Everything here is llama.cpp on Metal** (`-ngl 99`, flash attention on) with
LFM2.5-2.6B and Qwen3.5-4B/9B at Q4_K_M. A CPU-bound or memory-bound workload may
throttle differently.
