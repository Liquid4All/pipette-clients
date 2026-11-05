# iPhone 18 Pro Thermal Behavior

On the private-API build
([Private thermal release build](../pipette-ios/private-thermal-release-build.md)),
the iOS readiness gate holds a rep (one timed run of a cell) until the OS
thermal state is `nominal` and the SoC die is below `maxDieC` (36 °C in
`ios/Pipette/Pipette/Readiness/Ios.swift`). The comment on that constant gives an
idle die temperature of about 35 °C, so 36 °C is about 1 °C above it. The
criteria for every platform are in
[System Readiness Control](end-to-end-latency.md#system-readiness-control).
This article records what we measured on the cooled iPhone 18 Pro handsets in
our fleet, and why the same default applies to them.

For the same study on Android, see
[Pixel 11 Pro thermal behavior](pixel-11-pro-thermal-behavior.md).

## Key findings

**The 36 °C threshold is correct for this phone, and the gate has almost no
effect on rep times.** With the gate turned off, reps started 15–20 °C hotter.
They ran 0.05 % and 0.11 % slower on the two phones under test. Neither
difference is statistically significant.

**The gate cannot stick on this phone.** The cooled handsets rest at a die
temperature of about 15–25 °C, which is 10–20 °C below the idle temperature in
the comment on `maxDieC`. After a heavy rep, the die falls below 36 °C within
about 6 s.

**Rep times on this phone are very consistent.** The coefficient of variation
(CV) of a cell is 0.02–0.37 % with a 4B model. With a 350M model, it is
0.12–0.64 % for end-to-end latency and 0.35–3.6 % for prefill.

## Measurement rig

The fleet's current cooling is one Black Shark Live Streaming Cooler on the
back of each phone ([Active cooling](device-conditions.md#active-cooling)).
This section lists the setups that this study measured.

- **Handsets:** five iPhone 18 Pro (iPhone19,2), phone 1 to phone 5, paired to
  a Mac host. Jobs run through
  `pipette-ios-proxy`.
- **Cooling:** each handset has one active cooler.
- **Mode:** Single App Mode, which turns off the touch screen and the side
  buttons ([Accidental input](device-conditions.md#accidental-input)). Water on
  the glass therefore cannot deliver a touch, as it does on the
  [Pixel 11 Pro](pixel-11-pro-thermal-behavior.md#phantom-touches).
- **Room:** an office at about 23 °C (74 °F).
- **App:** the private-API build of 2026-10-18, which reads the
  private SoC die sensor (`private_thermal = true`).
- **Runtime:** llama.cpp b11349 (`llamacpp_ios_pipette`, `ios-arm64`), 2 threads,
  99 GPU layers, `swa_full = false`. These are the fleet's flags for this phone.

All values are from 2026-10-16 to 2026-10-19.

## The gate's temperature signal on this phone

Each rep logs one readiness line for each poll, for example:

```text
readiness: ios thermal=state:nominal die:33C(ok) cpu=busy:0.0cores(ok) → proceeding
readiness: ios thermal=state:nominal[SKIPPED] die:52C(hot) cpu=busy:0.1cores(ok) → proceeding
```

The gate compares the unrounded die value with a strict `<`, but the line shows
the value rounded to a whole degree. So `die:36C(ok)` is a value just below
36 °C, and `die:36C(hot)` is a value just above it.

`[SKIPPED]` marks a cell that runs with `skip_thermal = true`. The die reading
is still logged, so the start temperature of every rep is known in both cases.
The proxy logs these lines.

The 36 °C threshold is a compile-time constant, and no host table overrides it.
A plan can set only `readiness = { skip_thermal, max_wait_secs }` for an iOS
cell. To test another threshold, build and sign a custom app.

## Production reps

From onboarding on 2026-10-16 to 2026-10-19, the five phones ran 250 reps of
LFM2.5-350M Q4_K_M:

| Benchmark | Cells | Mean | CV per cell |
|---|---|---|---|
| `end_to_end_latency_256_256` | 25 | 780–787 ms | 0.12–0.64 % |
| `prefill_throughput_256` | 25 | 35.6–36.5 ms | 0.35–3.6 % |

The die temperature at rep start was 15–36 °C (median 21.5–23.5 °C per phone).
The OS thermal state was `nominal` for every rep. The gate held only 3 of the
250 reps, each for one poll, and no gate reached its deadline. Within a phone,
the rep time does not depend on the start temperature: −0.006 % per °C for
end-to-end latency (r = −0.03) and +0.027 % per °C for prefill (r = +0.09).

## Gate test with a heavy model

These cells are light, so a second test used a heavier model: Qwen3.5-4B
Q4_K_M (`unsloth/Qwen3.5-4B-GGUF`, revision `e87f1764`), the same model as the
Pixel 11 Pro heavy check. The cell was `end_to_end_latency_256_256`, and a rep
took about 8.5 s on the GPU. The plan compares two arms on phone 2 and
phone 4:

- **Gate:** `readiness = { skip_thermal = false }`.
- **No gate:** `readiness = { skip_thermal = true }`. The CPU-load criterion
  still applies.

Each arm ran twice on each phone (10 reps per arm per phone).

| Phone | Arm | Die at rep start | Mean | CV |
|---|---|---|---|---|
| phone 2 | gate | 28–36 °C | 8481.8 ms | 0.071 % |
| phone 2 | no gate | 46–55 °C | 8485.8 ms | 0.049 % |
| phone 4 | gate | 32–36 °C | 8479.7 ms | 0.263 % |
| phone 4 | no gate | 41–55 °C | 8489.1 ms | 0.142 % |

Without the gate, the reps were 4.0 ms (+0.047 %, Welch t = 1.71) slower on
phone 2 and 9.4 ms (+0.111 %, Welch t = 1.17) slower on phone 4. Across the 20
reps of each phone, the rep time changes by +0.003 % to +0.005 % per °C of start
temperature. At that rate, a die that is 20 °C hotter costs less than 0.1 %.

Part of the difference on phone 4 is an order effect. Its first cell (no gate,
directly after the model download) had a mean of 8499.3 ms. Its second no-gate
cell had a mean of 8478.8 ms, which is the same as its gate cells.

Without the gate, the die rises by about 3 °C per rep, from about 41–46 °C to
55 °C over 5 reps. With the gate, the die reads about 49 °C directly after a
rep. The gate waits a median of 6 s (at most 9 s, over 2–4 polls), and the next
rep starts at 28–36 °C. The gate adds about 5.5 s to each rep: the time from one
result to the next is 15 s with the gate and 9.5 s without it.

## Plans for this test

One plan merges identical cells into one job, also with
`merge_duplicates = false`, so a cell listed twice runs once. A repeat comes
from a second filing with `merge_duplicates = false`, or from a campaign
re-run. The merge rules are in [Job generation](../pipette-ctl/job-generation.md) and in
pipette-mgmt
[plan-ingestion.md §7](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md#7-the-job-body-handoff).

## Open questions

- **Other thresholds.** Only 36 °C and no threshold were tested. A lower
  threshold needs a custom build.
- **Other phones.** Only phone 2 and phone 4 ran the heavy model. The production
  data covers all five.
- **Longer reps.** The heavy rep took 8.5 s. A cell that holds the GPU for
  minutes was not tested.
- **Colder rooms.** All runs were in an office at about 23 °C. A colder room
  lowers the resting temperature and gives the 36 °C threshold more margin.
