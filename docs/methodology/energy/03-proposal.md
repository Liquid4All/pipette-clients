# Energy benchmarks: proposal

What to measure, on which devices, with what hardware, and what it costs in
money, device time and engineering. Built on the survey in
[01-landscape.md](01-landscape.md) and the probe evidence in
[02-device-capabilities.md](02-device-capabilities.md), which also define the
abbreviations used here.

**Status: proposal, first written 2026-09-19, with probe findings up to
2026-10-06. No part of the energy benchmark is implemented.** Every statement
below describes the proposed design, except where it says "today" or "already".
Those statements describe the current clients, and most name the code.

The proposal relies on powerd, the device lab's power service. powerd owns the
off-device instruments (smart plugs, inline USB-C meters and the Anker charging
stations), and §4 lists what this proposal needs from it.

## 1. Shape of the benchmark

**Energy would be its own benchmark type, and its parameters would be the
timing benchmark's.** Today `prefill_throughput_N` and `decode_throughput_N_M`
fix the prompt and output lengths. They run an untimed warm-up and five
measured repetitions behind the readiness gate. That is the steady-state,
fixed-workload structure every method in the survey requires. An energy
benchmark (`energy_prefill_N`, `energy_decode_N_M`) would take those parameters
unchanged, with the same tokens and the same depth. Its workload is then the one
the timing row measured, and the two rows compare.

What differs is the protocol:

- The window is reached by repetition count.
- The llama-bench path runs a process pair.
- The repetitions inside a window are not individually gated.
- The gate also checks idle power, charge level and controller state.

That protocol yields a timing figure of its own. It is not the same measurement
as the gated fresh-process timing row, so the two cannot share a benchmark id
without the server losing the distinction. An energy row would carry both its
energy and its timing; a timing row carries timing only.

**Three energy benchmarks per cell, not a twin for every timing id:** the
twins of the largest prefill, the largest decode and the largest end-to-end
run that the fleet's plan runs on that device class. For example, on a plan
whose largest timing ids are `prefill_throughput_2048`,
`decode_throughput_512_100` and `end_to_end_latency_512_256`, the three would
be `energy_prefill_2048`, `energy_decode_512_100` and `energy_e2e_512_256`. The
client's own local ladder runs from 100 to 8192 tokens
(`crates/pipette-cli/src/benchmarks/standard.rs`), so the largest ids depend on
the plan.

The first two would produce the coefficients a reader can apply generally: J
per prompt token and J per generated token, each measured directly with no
subtraction. The third would produce the headline J per request at a realistic
shape, and it covers the depth trajectory a real request walks. It also checks
the coefficients: the server would predict its energy from the two
coefficients, the idle power and the timing row's duration, and record the
residual. If the residual lies inside
the measured precision band the coefficients apply generally for that cell;
if not, the row says so. One blended figure would not do: on the M3 Ultra a
prompt token costs about 17 mJ and a generated token about 320 mJ
([02](02-device-capabilities.md) §7), so an end-to-end J/token is right only
at the shape it was measured at, and the 512:256 blend applied to an 8192:256
request is off by more than 4×.

Batched variants would be separate cells with their own coefficients, not a
correction. Choosing the largest shapes lowers the repetition count the window needs
(item 2) on slow devices; on fast devices the window mechanism still does the
work, and more so as hardware gets faster.

How the reading would bracket the timed block depends on the execution path:

- **In-process runtimes and the server paths** would bracket the timed block
  directly. The server paths are llama.cpp's end-to-end cells and the
  runtimes that drive a server, MLX, OpenVINO and ONNX GenAI among them.
  - The Android app runs the model in the client's own process. It would read
    its fuel gauge at the instants the timer starts and stops.
  - The iOS app also runs in-process. Its power source publishes every 20 s,
    and the iOS proxy on the Mac host would read it (§4, §6). On an iPhone the
    client would record the window's timestamps, and the window would be a
    whole number of 20 s publishes.
  - The servers hold the model resident and stream tokens. For the decode
    shape the client would read at the first and last token of each request.
    For the end-to-end shape, where prefill is part of the work, it would read
    from the request's start to its last token.

  Model load and prompt setup would stay outside the bracket on these paths, as
  they are outside the timing today.
- **The one-process-per-repetition path** is llama.cpp's prefill and decode
  cells. Today they run one `llama-bench` process per repetition
  (`--repetitions 1`, `crates/pipette-llamacpp/src/bench.rs`). Each process
  loads the model and sets up the prompt context before its timed block, so a
  bracket around the whole process over-reads a short decode by that setup.
  Inside one `llama-bench` process run with more repetitions, the model loads
  once, the depth context is computed on the first repetition and restored
  from a saved state on the later ones, and memory is cleared before each
  timed block. An energy repetition on this path would therefore be a **pair
  of processes** at `N` and `2N` repetitions (`-r N` / `-r 2N` for
  `llama-bench`), with the energy of one repetition taken as the difference
  divided by N. The load, the prompt setup and the process overhead are
  identical in both and cancel. The scaling sweeps in
  [02-device-capabilities.md](02-device-capabilities.md) §7 used exactly this
  pair and are the evidence that it isolates the timed work. Timing for the
  row would come from the per-repetition samples of the same `2N` process, so
  energy and timing come from one run. The readiness gate would run once per
  process, not before each inner repetition, which a window of seconds requires
  anyway (item 2 below). The thermal readings before and after each process
  would be kept.

Three additions on top of that, all proposed:

1. **An idle baseline** that the device owns, not the run. The worker would
   keep one stored baseline reading per power condition (charger connected or
   disconnected): 60 s of idle in the benchmark state, with the run's own
   sampler running, recorded as mean, spread and the time it was taken. The
   baseline feeds only the idle-subtracted figure; the total energy row does
   not depend on it. Its unit is the device because nothing larger exists on
   the client side: a run is one cell, jobs from different plans interleave
   in the claim loop, and the worker never sees a plan begin or end. Within a
   cell the same model and runtime are in the same state for prefill and
   decode, so a baseline per run measures the same quantity twice. The
   reading would be kept fresh three ways:
   - The worker takes it whenever it waits for a job longer than the baseline
     window, which is free device time in exactly the benchmark state.
   - It expires by age. Past the limit the worker takes a scheduled baseline
     before the next energy job. The limit is a tuning knob on the order of
     hours; the gate check below is what makes the exact value forgiving.
   - The readiness gate reads a few seconds of idle power before every
     repetition and compares it with the stored baseline's band. A reading
     outside the band means the baseline is stale or the device is not idle.
     The worker then re-takes the baseline or fails readiness, as the gate
     fails readiness for temperature today.

   Every row would copy the baseline's mean, spread and age, so the server-side
   subtraction is unchanged and a reader can see how fresh it was. The
   baseline is not keyed by runtime or model: the llama.cpp CLI path holds
   nothing resident between repetitions, and resident weights on the server
   paths sit in DRAM whose refresh cost does not depend on contents. Phase 0
   checks that claim once, with and without a resident model, and adds the
   runtime to the key only if the difference is outside the baseline's own
   spread. The stored reading must survive a worker restart, because the
   farm's scheduler restarts workers on its own schedule.
2. **A minimum measured window per repetition, reached by repetition count.**
   Today a decode or prefill rep on the Mac Studio lasts a fraction of a
   second. The polled sources update at about 1 Hz (`ryzenadj`, `PSTR`, the
   Pixel 4 XL's fuel gauge; [02](02-device-capabilities.md) Summary), and NVML's
   duty-cycle problem argues the same way. The iPhone's source is cumulative
   and publishes every 20 s; the iOS bullet above covers it.

   An energy window would last at least 10 s where the source is polled. The
   window would be `N` back-to-back repetitions of the benchmark's own shape,
   with `N` derived per device from the timing row already in hand
   (`N = ceil(10 s / rep time)`, at least 1). On the CLI path `N` is the smaller
   count of the pair above. On the other paths it is the number of requests or
   timed blocks bracketed as one window. A cumulative counter (IOReport, NVML)
   does not need the window, so a counter-read source may take `N = 1` and
   report per repetition.

   The token count and depth stay the benchmark's, so the energy row keeps the
   workload its timing row measured. An energy id with a larger token count than
   its timing counterpart does not work, for three reasons:
   - A fixed count cannot give a fixed time across a fleet whose decode rates
     span more than 10×.
   - A 1000-token decode on any Mac still falls short of the window.
   - A longer generation moves the key-value (KV) cache depth that the
     fixed-depth rule (§3) exists to hold.

   The window would be sized per scope from the slowest instrument in the row.
   A smart plug updates its cumulative counter about once a second, so a `wall`
   window is about 60 s for a few percent on the edges, which is also MLPerf's
   minimum. The on-device counters still report per repetition inside it.
3. **A sustained run**, separate and optional: 10–15 minutes of continuous
   decode on one flagship model per device, sampling throughput and
   temperature every 10 s. Produces the sustained/peak ratio and time-to-throttle
   that the phone studies report. Costly in device time, so it is a
   per-device-class campaign, not part of every plan.
4. **A power condition on every row: charger connected or disconnected.** A
   battery device runs a different power budget in each state, and the budget
   sets the DVFS point, so the two states are two machines and there is no
   single "power consumption" of the device. Both conditions would be characterized
   for every device with a battery, and the condition is part of the
   comparison key, next to scope. The default condition is **connected**,
   because every timing row in the fleet runs on external power (the
   device-conditions article requires it) and energy comes from the same
   repetition as the timing. It is also the only condition the Mac Studios,
   GMKtec boxes and Pi 5s have, so it is the only one where classes compare. The
   **disconnected** condition is the device as a user carries it. Its rows are
   `battery` scope, because with no external power the battery is the only
   boundary, and they give "requests per full charge". It is a per-class
   campaign like the sustained run, because it drains, recharges and cycles
   the pack, and the device is unavailable to every benchmark until the charge
   gate (§3) sees it settled again.
   Laptops and phones are one class here. A battery device on external power
   draws from the connector and from the pack at once, in a split that the
   charge controller chooses, on the MacBook Pro and the S26 alike
   ([02](02-device-capabilities.md) §1, §3, §7). The SMC reports the sum on a
   Mac (`PSTR` is `PDTR` plus `PPBR`). A phone has no chip that does the sum, so
   the inline meter would read the connector term and the fuel gauge the
   battery term.

   The clients already record a finer state once per run
   (`device_power_state`: `charging`, `plugged_in_not_charging` or
   `not_charging`). It is an observation, not a key: on the S26 it changes
   several times inside one repetition, so a key on it would put identical runs
   in different buckets. The net energy into the battery over the window,
   recorded per row, carries the same information in a usable form.

## 2. Metrics and result fields

Per run, the client would emit raw quantities in the proposed fields below. The
derived rates and the stored columns would be the management service's, as
tokens/s is today
([pipette-mgmt docs/benchmarks.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md)).
The clients already send per-rep device readings in
`device_hardware_readings` and `device_os_throttling_readings`
([Per-rep readings](../device-conditions.md#per-rep-readings)), but no reader
records power in them today. The proposed fields are:

| Field | Unit | Notes |
|---|---|---|
| `energy_scope` | enum | `battery`, `system`, `wall`, `vendor_internal`. Mandatory. Rows with different scopes are never compared |
| `energy_source` | enum | `ioreport`, `smc_pstr`, `ryzenadj`, `pi_pmic`, `battery_manager`, `ios_power_telemetry`, `usb_meter`, `nvml`, `rapl`, `plug` |
| `energy_j` / `energy_j_stddev` | J | mean over reps of energy in the measured window |
| `energy_window_s` | s | mean measured window |
| `energy_idle_w` / `energy_idle_w_stddev` | W | the device's stored idle baseline, mean and spread over the baseline window |
| `energy_idle_age_s` | s | age of the stored baseline when the run started |
| `energy_gate_idle_w` | W | mean of the readiness gate's short idle readings across the run's repetitions |
| `energy_peak_w` | W | max sample in the window, where the source is polled |
| `energy_soc_pct_before` / `_after` | % | phones and laptops; state of charge |
| `energy_battery_temp_c` | °C | battery temperature at the start of the window, where the platform exposes it (Android does) |
| `energy_charger_connected` | bool | the power condition. Part of the comparison key: rows with different values are never compared. Must be false for battery-scope rows |
| `energy_battery_net_j` | J | signed net energy into the battery over the window, from the fuel gauge or `PPBR`; positive is charging. Records what the charge controller did during the run |
| `energy_adapter_w` | W | the adapter's negotiated power on connected rows, from `PDTR`, the PD contract the Pi 5's firmware exposes, or the iPhone's `AdapterDetails`; the meter's voltage where the device has none; the ceiling the row ran under |
| `energy_power_session` | string | the powerd session that bracketed the window, on rows whose scope came from powerd (§4); the key for an audit against powerd's stored samples |
| `energy_estimated_s` | s | the powerd report's `estimated_s`. Zero on every published row; a non-zero value fails the run |
| `energy_power_mode` | string | Low Power / High Power / Android performance mode / governor if readable (the S26 Ultra's `scaling_governor` reads from the app) |

Derived by the management service: J per prompt token, J per generated token, tokens/J,
average W, dynamic energy (`energy_j − energy_idle_w × energy_window_s`, with
`energy_idle_w_stddev` propagated onto it), J per request at a declared shape.
For phones also mWh per 1k tokens and "requests per full charge" from the
recorded nominal capacity. Per cell, the linear model: with `a` the dynamic
J per prompt token from the prefill benchmark, `b` the dynamic J per generated
token from the decode benchmark, and `t` the duration from the timing row, a
request of `P` prompt and `D` generated tokens costs `a·P + b·D +
energy_idle_w·t`. The end-to-end benchmark's measured energy against that
prediction is the cell's **residual**, stored with the coefficients, and the
coefficients are published as generally applicable only while the residual
sits inside the cell's precision band.

Where two sources exist on one device (Mac: IOReport SoC *and* `PSTR` system;
GMKtec: `ryzenadj` socket *and* plug wall) the client would submit **two rows
with different `energy_scope`**, not one blended number.

`battery`, `system` and `wall` are physical boundaries: at the cell terminals,
DC into the device after the PSU or adapter plus the battery's contribution,
and AC at the outlet. A scope names where the instrument sits and nothing
else. Two instruments at the same boundary measure the same quantity.

Which boundary a device is *compared at* is a separate rule, about attribution
rather than physics: **a device is measured at the boundary of what its
manufacturer ships.** A desktop or a laptop ships with its own PSU or adapter,
so the conversion loss is the device's whether the conversion sits inside the
case or in a brick, and the attributable boundary is `wall`. A phone ships
without a charger, so the charger's loss belongs to whoever sold the charger,
and the attributable boundary is the phone's own connector, which the inline
meter reads on Android and which an iPhone reports itself as `SystemPowerIn`,
and which is submitted as `system`. A Raspberry Pi 5 ships as a bare board
and runs from whatever USB-C supply its owner adds, so it falls under the
same rule as a phone: its attributable boundary is its USB-C connector, read
by the inline meter and submitted as `system`, and the rail sum its PMIC
reports is `vendor_internal` ([02](02-device-capabilities.md) §4).
Cross-class comparison happens between rows at each class's attributable
boundary. A reader will notice that
a PSU loss is counted on one side and not the other; the answer is that the
rule attributes what the maker shipped, and the doc says so rather than
claiming the two boundaries are the same. The supply the phone negotiated is
recorded beside its row, because the phone's own charge IC efficiency depends
on the voltage the charger offered; the adapter model is recorded on laptop
rows for the same reason, and the fleet's MacBooks run on Apple's own 140 W
unit. A Pi 5 row records the PD contract the firmware negotiated, which the
device tree exposes without root, because a supply that offers less than
5 A at 5 V makes the firmware cap USB current and flag under-voltage.

Two consequences. Desktops with their own PSU compare only at `wall`: the
Mac Studio's `PSTR` is a post-PSU DC reading with no counterpart on the
GMKtec, so on a desktop
`system` is a cross-check rather than a comparison figure. And on a battery
device the attributable row and the complete figure differ by the battery
term. A laptop's `wall` row and a phone's connector row are each published
with the net battery energy beside them, so the part of the work the pack
supplied is visible. The complete same-instant figure is `PSTR` on a Mac, and
on a phone the meter's energy less the net energy into the battery; both raw
figures are submitted so the subtraction stays auditable. The only method
that attributes a battery device fully under this rule is drain and recharge,
metering the recharge after a disconnected run; that is a campaign-level
measurement like the sustained run, and is noted here as the consistent
extension, not proposed.

`vendor_internal` is not a boundary. It is whatever a silicon vendor's own
counter happens to include, and that differs by vendor and by model. Apple's
IOReport rollup includes DRAM, because unified memory is on-package;
`ryzenadj`'s socket figure excludes the GMKtec's 128 GB, because it is not.
Decode is memory-bound, so the difference is first-order rather than a rounding
error. RAPL is a third answer again, covering the package but not DRAM on client
parts. The value is therefore useful and real, but portable only within one
device model: **`vendor_internal` rows compare only when `energy_source` and the
device model also match.** What the counter covers on that model belongs in the
device's notes.

The phone subtraction removes the energy that reaches the cell. It does not remove
the loss in the charge path between the connector and the cell, which is
proportional to the charging current. That loss is inside the device, so it
belongs in the `system` figure in the same way a Mac's PSU floor and board
losses belong in `PSTR`: the device pays it to do the work under that
condition. What makes it a problem is repeatability, not attribution. The
charging current is set by where the level sat against the charge cap when
the run began, not by the inference, so two identical runs report different
system energy when the controller charges during one and holds during the
other. The readiness gate therefore requires the controller to be idle at the
start of every repetition (§3), and `energy_battery_net_j` records what it did
afterwards, so a reader can see how much of a row was charging.

## 3. Fairness

The rules, and the reason each is here:

| Rule | Why |
|---|---|
| Same model, quantization, runtime version and token counts across a comparison, recorded per row | already the timing rule; energy inherits it |
| Scope is part of the key. A `vendor_internal` number and a `wall` number for the same device are two results, and two `vendor_internal` numbers from different device models are not one result either | Google's 2.4× accelerator-vs-full gap ([01](01-landscape.md)); the M3 Ultra's SoC counters miss most of its idle draw ([02](02-device-capabilities.md) §1); Apple's counter includes DRAM and `ryzenadj`'s excludes 128 GB of it ([02](02-device-capabilities.md) §7) |
| Report total *and* idle-subtracted energy | dynamic energy flatters high-idle boxes (Mac Studio, GMKtec); total penalizes them for their PSU. Readers need both |
| Report J/token *and* J/request at a declared input:output shape | AI Energy Score v2: per-token efficiency hides 300–800× output-length differences |
| Prefill and decode reported separately | compute-bound vs memory-bound; a blended figure depends on prompt length |
| A coefficient applied beyond its own benchmark carries four things: whether it is dynamic or total, the depth and count it was measured at, its batch size, and the cell's end-to-end residual | AI Energy Score v2's reasoning result; decode slows as the KV cache grows ([02](02-device-capabilities.md) §7); idle adds ~8 % to a decode token on the Mac Studio and ~5 % on a phone; a batched token costs a different amount; the residual is what says the linear model held for that cell |
| Cumulative counters preferred; polled sources sampled at 1–2 Hz over windows ≥10 s; the window sized per scope from the slowest instrument in the row (`wall` ≈ 60 s); first run of a session discarded | NVML's duty cycle ([01](01-landscape.md)); a polled source holds a handful of samples in a sub-second run, polling `ryzenadj` fast raises its own idle, and the cold first run of a session pays the model load ([02](02-device-capabilities.md) §7); a plug's ~1 s cadence is a 10 % edge on a 10 s window |
| Decode energy quoted at the benchmark's fixed depth and token count, never per-token extrapolated | decode slows as the KV cache grows ([02](02-device-capabilities.md) §7) |
| A scaling sweep (E_rep vs n, two rep counts) is run once per device model before its rows are published | it is what told a per-token law apart from a coincidence ([02](02-device-capabilities.md) §7) |
| Idle baseline owned by the device, one per power condition, in the benchmark state (screen state, radios) with the run's own sampler running; mean, spread and age recorded on every row; validated by the readiness gate's short idle reading before each repetition and re-taken when that reading leaves the band or the age limit passes | every method subtracts an idle figure, and the state is what the client can reproduce. Per run it would double the energy run's device time and measure the same quantity twice within a cell; per device with a gate check catches the drift that per-run guarded against. The GMKtec's idle reading scatters widely with the poller visible in it ([02](02-device-capabilities.md) §6), which moves dynamic energy by ~45 J on a ~250 J rep, so the spread travels with the mean |
| Power condition is part of the key, and both conditions are characterized for every battery device. A connected row and a disconnected row for the same device are two results. Connected is the default and the only condition where classes compare; disconnected rows are `battery` scope | a battery device runs a different power budget in each state, and the budget sets the DVFS point; every timing row runs connected, and energy comes from the same rep; mini-PCs, Mac Studios and Pi 5s exist only connected; with no external power the pack is the only boundary |
| Readiness gate before every rep, thermal readings kept | already enforced; energy under throttling is a different quantity |
| Charge level in the general readiness gate, on every device with a battery and for every benchmark: the start condition is the level within a band just under the device's charge cap *and* the controller idle (next row). Below the band the gate waits, under a charge-specific deadline longer than the thermal one, and fails the run rather than producing a clamped number. The worker also checks charge before claiming a job. The floor comes from the M5 Max sweep at full charge and near 25 % (Phase 0); the cap is a managed condition the rig sets and the row records. Battery temperature is among the gate's thermal readings where the platform exposes it | state of charge is a second throttle beside thermal: as the buffer runs down, macOS clamps SoC power to what the adapter delivers, and the OS power-saving modes engage near 20 %. Connected runs drain too (the MBP on AC, the S26 at its cap), so the check is general and a disconnected run needs no special recovery step: power returns, the next gate sees the level below the band and waits. "Exactly the cap" is the wrong target because charge tapers near the top and the S26 flips between 79 and 80 %. A recharge is tens of minutes on a phone and longer on a laptop, so the thermal deadline would fail every run after a disconnected one, and a worker that claims while recharging holds a lease it cannot start. Fast charging warms the pack, and the pack's deliverable power depends on its temperature |
| Charge controller idle at the start of every repetition on connected battery devices: net battery current near zero for a few seconds, enforced by the readiness gate. Per-rep net battery energy recorded on the row | on the S26 the controller holds at 80 % and resumes charging at 79 %, so a floor at the cap invites it to flip inside the rep. The gate controls the start state and the recorded net energy shows what followed, the same division thermal uses. The deadline is longer than the thermal one: a laptop recovers a percent in a minute or two |
| Phones connected: the connector term from the inline meter on Android, and from the phone's own power controller on iOS with the meter as the like-for-like check; airplane mode with Wi‑Fi on, screen off, cooler and charge cap recorded. Phones disconnected: the charger port switched off from the Linux host through powerd's output control (§4), the disconnected campaign only | the fuel gauge reads net current, so a battery row alone needs the charger gone (MELTing Point's VBUS cut); the S26 sweep read one workload very differently depending on the charge controller ([02](02-device-capabilities.md) §7). The iPhone's own power controller reports the connector and battery terms and their sum, and held its cadence through a benchmark ([02](02-device-capabilities.md) §5, §6) |
| Laptops: `wall` is the attributable row, published with the net battery energy beside it; `PSTR` is the complete same-instant figure; `PDTR` is checked against the inline meter on the adapter cable and `PPBR` against the smart battery's own amperage and voltage in the IORegistry | the MBP draws from its battery under load on its adapter ([02](02-device-capabilities.md) §1), so a `wall` row under load misses that much and the row has to show it. `PSTR` is `PDTR` plus `PPBR`; the meter reads the first term at the same boundary, and the battery gauge is an instrument independent of the SMC at the same terminals as the second. Whether `PSTR` counts energy going into the battery when it charges is unknown; the Phase 0 meter settles it, and if it does, the controller-idle gate applies to the laptops too |
| Pi 5: the metered connector is the attributable row; the PMIC rail sum is the `vendor_internal` reading below it; the negotiated PD contract and the firmware's throttle flags are recorded on the row, and a window during which a throttle flag sets fails the run | the board ships without a supply, so the connector is its boundary (§2); the rail sum omits the regulators, the fan and any HAT, so it is a floor and not the figure; the firmware's flags cover under-voltage and the frequency cap as well as temperature, and a capped run is a different machine |
| Power mode / governor recorded | up to 40 % latency spread at equal energy in the DVFS study |
| Off-device instruments are required where the attributable boundary needs one. Before the window the gate asks powerd for the client's current power: every bound meter must be connected with a fresh sample, and the session start must succeed, which powerd refuses while a meter is disconnected. A row whose off-device scope is missing, or whose report carries estimated time, fails the run | without the attributable boundary the row's numbers cannot be trusted, so a partial row is not a result; the check runs before a multi-minute window, not after it |
| Every source records a measured **precision** (the repeat spread of identical runs) and a stated **accuracy**. Accuracy is a bracket where two boundaries enclose the source (Mac Studio `PSTR`: SoC counters below, plug above), a like-for-like check where the boundary is reachable (laptop `PDTR` and `PPBR`, previous rows), and the instrument's specification class for the reference instruments (`wall` plugs, the inline meter, the phone fuel gauge). Nothing transfers between device models. No source is described as calibrated against a plug | a plug reads AC and `PSTR` and `ryzenadj` read DC, so the plug can bound them and cannot calibrate them. `vendor_internal` rows compare only within one device model and source, where precision is the whole requirement and absolute accuracy is irrelevant. GreenBench's implausible package figure is what skipping the check looks like; the M5's window-cadence spread ([02](02-device-capabilities.md) §1) is what a band borrowed from the M3 Ultra would have hidden |
| Repetitions ≥5 with mean ± SD; energy and timing from the same rep. On the llama.cpp CLI path a rep is the `-r N` / `-r 2N` process pair and the timing comes from the `2N` process | MLPerf and ML.ENERGY; already the harness default. The pair is the only bracket the parent of a stock `llama-bench` can take that excludes the load and the depth pass |
| Inside an energy window the back-to-back repetitions are not individually gated; the readiness gate runs once per window (per process on the CLI path) and the thermal readings before and after are kept | a window of seconds is the point; MLPerf runs minutes ungated. The gate still decides when the window may start, and the readings show what happened inside it |
| Cross-class comparisons only between rows at each class's attributable boundary (§2): `wall` for desktops and laptops, the metered connector (`system`) for phones and Pi 5s, in the connected condition. `battery`, `vendor_internal`, desktop `system` and disconnected rows are never compared across classes | the boundary of what the maker ships is the attribution rule; the Mac Studio's post-PSU `PSTR` has no counterpart on the GMKtec; mini-PCs have no battery and no disconnected condition; a vendor counter is not a boundary |

A modified device (coolers, capped charge) would be recorded as such, as
[Device conditions](../device-conditions.md) requires for timing rows. Energy
numbers from a clamped, cooled S26 are not what a user's S26 draws in a pocket,
and the row must say so.

## 4. Hardware and cost

| Item | Qty | Unit | Purpose |
|---|---|---|---|
| Metering smart plug (Shelly Plug US Gen4, the model powerd drives; one is on the Linux host today) | 6 | ~$25 | Mac Studio ×4 wall scope; GMKtec ×4 wall scope (share across boxes by moving plugs, or buy 8); one reserved for calibration |
| USB-C inline meter with a PC API (ChargerLAB KM003C; one is on the Linux host today, on Pi 9) | 4 | ~$100 | the attributable boundary of the Android phones and of the Pi 5s; the like-for-like check of `PDTR` on the laptops and Neo, inline on the adapter cable, and of `SystemPowerIn` on the iPhones; reports the negotiated supply; reads zero to confirm a power cut |
| Monsoon HV or Otii with battery bypass | 0 | ~$1k+ | not proposed: destructive to a flagship handset, and the fleet phones are production units |

The plug count covers the desktops only. The laptops' attributable boundary is
also `wall` (§2), and the table does not yet count plugs for the three MacBook
Pros and the Neo.

powerd's design switches the charger ports from the Linux host, so the
disconnected campaign buys nothing. The Pi 5s get no plug: their boundary is the
connector, and a board whose rails draw about 2 W idle and 8 W loaded
([02](02-device-capabilities.md) §6) sits at the floor of what a plug resolves.

Total: roughly **$550**, all commodity, of which one plug and one meter are
on the Linux host already. The Yokogawa/PTDaemon class of analyzer that MLPerf
requires costs ~$3k per channel and is not needed to publish a stated-method
result. The plug's absolute accuracy is unverified (powerd's
findings record how it is read and what remains to check). Until a reference
check, the `wall` rows would carry the plug's specification class and the
measured precision, as §3 requires.

**A meter for the connected condition, powerd's switch for the disconnected
one.** The handsets share Anker Prime 250W stations (A2345) at two to four
phones per station. A smart plug on the wall side would therefore cut a group
rather than one phone, and it would take the group's other handsets off charge
during a run.

The connected condition needs no switching. The meter would read the connector
term for one handset while the client reads the battery term, and the two close
the energy balance, with the controller held idle at the start of each
repetition (§2, §3). Every handset stays at its 80 % cap, so a reading cycles no
batteries. The disconnected condition needs the charger gone, and powerd would
switch the handset's port from the Linux host. One switch on the station
controls both USB-A ports, so a handset in the disconnected campaign would sit
on a USB-C port.

**powerd.** The off-device instruments (the KM003C, the Shelly plug and the
Anker chargers) are owned by `powerd`, the device lab's per-site power service.
Its docs own how it reads, stores and serves each meter. This proposal relies
on the following parts of powerd's design. powerd's own plan, not this
proposal, decides when each part exists:

- **Bindings.** A powerd client is a device, not a pipette workspace. This
  proposal needs one powerd client per farm device, with the device's pipette
  `client_id` recorded on it.
- **Sessions.** A client would start a session before its window and stop it
  after, both timed on powerd's clock, so the client needs no synchronized
  clock. The stop returns the report. The report's `point` maps onto this
  proposal's scopes (`ac-input` is `wall`, `dc-input` is `system`), and its
  `estimated_s` would be copied to the row. One complete row would leave the
  device through the existing submit path, carrying the session id and
  `estimated_s`, so the management service change stays schema-only. A
  non-zero `estimated_s` would fail the run, under the rule that a
  partial row is not a result. The 60 s `wall` window of §1 rests on powerd's
  measurement of the plug at 60 s windows.
- **The PD contract does not come from powerd**, because the meter stream
  carries voltage and current and not the negotiation. `energy_adapter_w` would
  come from the device side where it has one, and from the meter's voltage
  otherwise.
- **Output control for the power cut.** In powerd's design, powerd switches an
  Anker port or a Shelly plug off and on, keeps integrating across the switch,
  and refuses a switch that also controls another client's meter. The
  disconnected campaign also needs the off state to be a **lease** (PIP-698):
  - The client renews the lease each repetition.
  - powerd restores the output by itself when the lease lapses, so a run that
    dies mid-window leaves a handset draining for at most one lease interval.
  - Open leases survive a powerd restart.

  powerd's output request has no lease today. What stays outside powerd is what
  needs the battery level, which powerd does not read. The readiness gate would
  check the charge floor before the client asks for the cut, and the farm side
  would own the recharge wait and the run-and-recharge alternation of the
  campaign.

The iPhones' power controller is not a powerd instrument. powerd runs on its
own host, and the phones' native tunnel rides the pairing that the Mac host
holds. The host that drives the phones would therefore read them: the iOS
proxy on the Mac would read the `AppleSmartBattery` entry through the
diagnostics relay at the window's ends, with Apple's own pairing and no root.
Today only the probe script
[`tools/energy-probes/ios_power_sampler.sh`](../../../tools/energy-probes/ios_power_sampler.sh)
reads the entry; `pipette-ios-proxy` does not. The entry publishes every 20 s,
so the window would be aligned to the publishes (§1).

The meter count follows how many handsets a campaign measures at once rather
than the fleet size, because §5 already schedules energy runs with those device
types out of the farm's normal scheduling. Three cover a handset campaign and leave one for
calibration. The fourth is for the Pi 5s, which share the same meter model
and whose campaigns run beside the handset ones. Each meter takes its own
supply from a separate host connector, so the Linux host needs one USB port and one
cable run per meter and the meter adds no load to the measured line. Still to
confirm: that the meter passes the Pi 5's 5 V 5 A contract through, since the
official supply declares 5 A in its 5 V PDO, which is outside the standard PD
table, and the Pi falls back to 3 A without it.

If powerd's Anker adapter passes its two experiments (links to ten chargers
at once, and the port readings against a KM003C on the same cable; see
powerd's findings), every handset on a station is instrumented for no further
hardware, and the KM003C count above becomes the like-for-like check rather
than the row's source.

Software cost: zero licenses. Every source is free:

- a public OS API or firmware tool (`BatteryManager`, `vcgencmd`)
- a private, undocumented macOS interface (IOReport, the SMC keys)
- a tool already installed on the box (`ryzenadj`)
- an open-source tool (`pymobiledevice3`, run with `uvx`)

## 5. Device time

Per energy benchmark, three per cell, on top of today's run:

| Component | Time | Applies to |
|---|---|---|
| idle baseline | 60 s, paid only when the stored baseline is missing or past its age limit; otherwise taken while the worker waits for a job | per device and power condition, not per run |
| idle check in the readiness gate | a few seconds per repetition | every energy run |
| five reps × ≥10 s window | ~1–2 min on the in-process and server paths; ~3 min on the llama.cpp CLI path, where each rep is a 10 s and a 20 s process plus two model loads | every energy run |
| five reps × ~60 s window | ~5–6 min | rows whose attributable boundary is a plug (`wall` on desktops and laptops) |
| cool-down via readiness gate | unchanged | |
| charge recovery via readiness gate | zero when the level is in the band and the controller idle; tens of minutes on a phone and longer on a laptop when it is not, during which the device takes no benchmark of any kind | battery devices, every benchmark |
| disconnected campaign: cut, drain, recharge, settle | 5–15 min drain per benchmark, plus the recharge and the wait for the controller to settle; the campaign alternates a run and a recharge because the gate cannot recover charge with the connector off | battery devices, and only for `battery`-scope rows |
| sustained run | 10–15 min plus cool-down | one model per device class, campaigns only |

A Mac or GMKtec energy benchmark therefore costs about **2 to 3 minutes** of
device time for its on-device scopes and about **6 minutes** where the `wall`
row is required, against seconds for the timing benchmark, plus one minute per
device whenever the baseline has to be re-taken on a busy queue. A full cell,
three energy benchmarks with `wall` rows, is therefore **15 to 20 minutes** of
desktop time, which is why the family is three ids and not a twin per timing
id. A metered
phone benchmark costs the on-device figure, because nothing is cut and nothing
recharges. A Pi 5 benchmark costs the same, and its repetitions are nearly windows
already ([02](02-device-capabilities.md) §7). A 2048-token prefill takes about
33 s on the Pi, so `N` is 1 and the window is the timing repetition itself. A
100-token decode takes about 8 s, so `N` is 2. A disconnected run costs
**~20 minutes** or more including the recharge and the settle, and takes the
device out of every queue for that time. On the laptops a plan of connected
energy runs drains the battery as it goes, so the charge gate's waits have to
be budgeted or the later runs fail on the deadline by design. Energy
benchmarks should be scheduled as a plan with the relevant device types
excluded from the farm's normal scheduling, not mixed into the normal queue.

## 6. Engineering time

| Phase | Work | Estimate |
|---|---|---|
| **0. Characterize** | plugs and meters in the rack; powerd's proof of concept, which logs the KM003C and the plug on the Linux host today; the spare plug in series with a working one to confirm the plugs agree; one MLX decode loop on a Mac Studio to bracket `PSTR` between IOReport and the plug; same for a GMKtec, plug above `ryzenadj`; one metered handset run to close the energy balance against `BatteryManager`; one metered Pi 5 sweep to bracket the PMIC rail sum under the connector reading and to confirm the 5 A contract survives the meter; on an MBP a meter inline on the adapter cable to check `PDTR` under load and to settle whether `PSTR` counts battery charging, and a probe of the `AppleSmartBattery` amperage and voltage keys for cadence, to check `PPBR`; the scaling sweep of [02-device-capabilities.md](02-device-capabilities.md) §7 on an M5 Max at full charge and again near 25 %, which sets the charge-gate floor; record precision and the accuracy bracket per device model; on one Mac server path, an idle baseline with and without a resident model, to settle whether the runtime joins the baseline key | 5–6 days once hardware arrives |
| **powerd (the lab's power service)** | planned in powerd's own project. This proposal needs the bindings, the sessions and the output control before the first `wall` and connector rows, and the output lease (PIP-698, §4) before the disconnected campaign | in powerd's plan; on the critical path for every `wall` and connector row |
| **1. macOS, Windows and Pi 5 in the client** | the `energy_*` benchmark type in [pipette-mgmt `crates/pipette-plan-types`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs), three ids per device class mirrored from the largest prefill, decode and end-to-end timing ids; `EnergySampler` trait in `pipette-device` with IOReport, SMC `PSTR`, `ryzenadj` and `pi_pmic` implementations, the last polling `vcgencmd pmic_read_adc` at a few Hz and averaging, with the PD contract from the device tree and the throttle flags on the row; called from the per-rep observer beside `detect_thermal`; the stored idle baseline in the client store, keyed by power condition, taken during worker idle or on expiry; the short idle reading in `pipette-readiness` and its comparison with the stored band; the charge band, the controller-idle check and the pre-claim charge check in the same gate, which also cover the timing rows; the powerd `power` check in the gate and the session start before the window; the `-r N` / `-r 2N` process pair in `pipette-llamacpp`'s bench path, with `N` derived from the timing row and both processes' `samples_ns` parsed; the window bracket at the first and last streamed token on the server paths; the session stop after the window and the fatal path when it fails or reports estimated time; new result fields (or raw samples in the existing `device_hardware_readings` field, which no reader fills with power today). In pipette-mgmt: the stored columns, the derived rates, the linear model with its residual, and site display | 2 weeks, one engineer, plus the management service change |
| **2. Android** | `BatteryManager` sampler in both engine flavors: beside the `ThermalSampler` callback that `android/native/benchmarks.rs` uses in the `rustEngine` flavor, and in the Kotlin port of the shipping `kotlinEngine` flavor (`LlamaBenchmark.kt`); the powerd session for the connector term; the charge band and the controller-idle check in the Android gate; net battery energy per rep on the row; the S26 charge-cap and cooler recorded as conditions. The disconnected campaign waits for powerd's output lease | 1–2 weeks, client side |
| **3. iOS** | no on-device power code. The connector term, the battery term and their sum come from the phone's own power controller. The iOS proxy on the Mac host would read them over the native tunnel ([02](02-device-capabilities.md) §5); the cumulative accumulators publish every 20 s and held that cadence under a five-repetition benchmark (§6 there). The Swift client would record the window timestamps, align the window to the 20 s publishes and carry the result fields. The proxy would read the accumulators at the window's ends and fill the fields in before the row is submitted. The inline meter would be the like-for-like check of `SystemPowerIn` once it arrives, and it would confirm the units of the accumulators | the proxy-side read and the Swift fields, about 1 week |
| **Sustained campaign** | one flagship model per device class, results as a methodology article like the M5 thermal one | 1 day of farm time per class |

Client parity is the standing risk. A change to the shared Rust crates reaches
only the `rustEngine` Android flavor. The shipping `kotlinEngine` flavor and the
iOS app each need their own port, and every Swift type is hand-mirrored
([Client parity](../../client-parity.md)). Phase 1 should land the result fields
in the Swift and Kotlin mirrors as optional-and-absent even before Phases 2 and
3, so the rows from every client are well-formed. The iOS power source itself
carries no such risk, because the host would read it and the app never touches
it.

## 7. Recommendation

Start with Phase 0, powerd's first steps and Phase 1 together. They cover 12
of the 46 client devices in the 2026-09-19 summary of
[02](02-device-capabilities.md) (8 Macs, 4 GMKtec). They need no change to how
the farm powers anything, and they use sources that the probes proved. They
would produce the first cross-runtime (llama.cpp vs MLX vs OpenVINO/ONNX on the
same silicon) energy numbers, which is where the survey suggests the rankings
invert. powerd is on the critical path from the start, because a desktop's
attributable row is the plug's.

Android and the Pi 5s follow once the meters arrive, because the connector is
the attributable boundary for both. iOS needs no meter for its first rows: the
phone reports its own connector and battery terms, and the host can read them
(the probe script does). The iOS rows therefore wait only on the proxy-side
read, and the iPhones would become the best-instrumented phones in the fleet.

Decisions needed:

1. **Scopes to publish:** every scope a device has, labeled, with cross-class
   comparison at each class's attributable boundary (recommended), or the
   attributable boundary alone. `vendor_internal` carries the best-behaved
   source in the fleet and the only cross-runtime comparison worth
   publishing, but it never travels between device models.
2. **Hardware:** approve ~$550 of plugs and inline meters.
3. **Phones:** metered connected runs at the 80 % cap as the default
   (recommended), which cycle no batteries and add no device time, with the
   controller held idle at the start of each repetition; the disconnected
   campaign, switched per port by powerd, for the
   `battery`-scope rows later.
4. **Benchmark ids:** an `energy_*` benchmark type whose parameters mirror the
   timing ids exactly, three per cell (largest prefill, largest decode,
   largest end-to-end), with the window reached by repetition count
   (recommended), versus `energy_*` ids with a larger token count, or a twin
   for every timing id. The larger counts fail the window rule on every Mac
   and move the KV depth (§1 item 2); a twin per id costs 6 minutes of desktop
   time each and adds nothing the three coefficients and the residual do not
   already give.
5. **Ungated inner repetitions:** accept that the repetitions inside an energy
   window run without the readiness gate between them (recommended, with the
   thermal readings before and after kept), or gate each one and give up the
   window on the polled sources.
