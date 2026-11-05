# Energy benchmarks: what each farm device can measure

What power and energy signals each probed device class in the device lab
exposes. The probes ran on one device of each class, from 2026-09-19 to
2026-10-06; each section gives its own date. The probe sources are in
[`tools/energy-probes/`](../../../tools/energy-probes/). Numbers below are the
raw probe output. They show that a source works and how it behaves; they are not
benchmark results.

No client reads power or energy from any of these sources. The clients record:

- the thermal state before and after every repetition (`RunThermal`)
- the battery level, the power state and the power-save mode once per run
- on the Raspberry Pi 5, the readiness gate's `vcgencmd get_throttled` flags
  before each repetition (§4)
- on the Android CLI, the readiness gate's optional skin-temperature reading

The farm reads the GMKtec power limits. No client records watts or joules.

Terms used below:

- SoC: system on chip.
- SMC: the Apple System Management Controller. SMU: the AMD System Management
  Unit.
- PMIC: power management integrated circuit.
- ODPM: Android's On-Device Power Monitor, the per-rail energy counters behind
  `IPowerStats`.
- NVML: the NVIDIA Management Library. RAPL: Intel's Running Average Power
  Limit counters.
- PD: USB Power Delivery. A PDO (power data object) is one voltage and current
  pair that a PD supply offers.

## Summary

| Class (farm hosts, 2026-09-19) | Best software source | Scope of that source | Rate / form | Root | Whole-system source | Verdict |
|---|---|---|---|---|---|---|
| Mac Studio M3 Ultra ×4 | IOReport `Energy Model` counters | SoC: CPU, GPU, ANE, DRAM, fabric, per die | cumulative mJ/nJ counters, any window | none | SMC `PSTR` (W, ~1 Hz) reads total system power without a meter | **ready** |
| MacBook Pro M5 Max ×3 | same | same | same; M5 batch cadence caveat | none | `PSTR` plus `PPBR` (battery) and `PDTR` (adapter) | **ready**, laptops draw from battery under load |
| MacBook Neo (A18 Pro) | same | same | same | none | `PSTR` | **ready** |
| GMKtec EVO-X2 ×4 (Windows) | `ryzenadj --info` `PPT VALUE FAST/SLOW`, `STAPM VALUE` | socket (SMU-averaged) | polled W, not a counter; poll at 1–2 Hz (§7) | ring-0 driver, already installed | none; needs a metering plug | **usable**, plug for system scope |
| Galaxy S26 Ultra ×10, S25 Ultra | `BatteryManager` `CURRENT_NOW` + `ACTION_BATTERY_CHANGED` voltage | battery terminals (net of charger) | sub-second updates, signed µA | none | battery is the system | **usable with the charger cut or metered** |
| Nothing 4a Pro | same | same | ~0.4 s updates | none | same | same |
| Pixel 4 XL | battery current only, no ODPM (pre-Tensor) | battery | 1 Hz class | none | same | low value |
| iPhone 17 Pro ×10, 16 Pro Max | `AppleSmartBattery` `PowerTelemetryData`, read from the host over the CoreDevice tunnel | connector power in, battery power, and their sum, from the phone's own power controller | 1 Hz integration into cumulative mW·s accumulators, published every 20 s | none (Apple's own pairing, through `pymobiledevice3`'s native tunnel) | the phone reports the connector term itself | **ready**, read from the host; see §5 |
| Raspberry Pi 5 ×10 | `vcgencmd pmic_read_adc`: current and voltage per PMIC rail | board rails: SoC core, DRAM, I/O and the WiFi radio; not the 5 V input, the regulators, the fan or a HAT | instantaneous snapshot per call, ~25 ms, no counter | none | none on the board; the USB-C input is the attributable boundary, read by the inline meter | **usable**, meter for the connector; see §4 |
| lambda-h100 ×2 | NVML total-energy counter | GPU board | cumulative mJ | none | RAPL + plug | out of scope |

## 1. macOS Apple Silicon

Probed on Mac Studio 1 (M3 Ultra), MacBook Pro 1 (M5 Max) and the MacBook Neo
(A18 Pro).

### IOReport `Energy Model` (no root)

`ioreport_energy.c` subscribes to the `Energy Model` channel group and prints
the delta over N ms. The counters are cumulative energy, so any start/stop
window works and nothing has to be polled during a repetition. This is what
`macmon`, `asitop` and the Zeus Apple-Silicon RFC use.

| Device | State | `CPU Energy` | `GPU Energy` | DRAM | Notes |
|---|---|---|---|---|---|
| M3 Ultra | idle | 0.03 W (die 0 + die 1) | not reported (0) | 0.07 W ×2 | channels are per die: `DIE_0_CPU Energy`, `DIE_1_CPU Energy`, `DRAM0_0`, `DRAM0_1`, `GPU0_0`, `GPU CS0_0` |
| M3 Ultra | 8× `yes` | 2.4 W + 6.5 W | — | 2.3 W ×2 | |
| M3 Ultra | MLX 4096² matmul loop | 0.16 W | **171.0 W** (`GPU0_0` 158.2 + `GPU CS0_0` 12.8) | | |
| M5 Max | idle | 0.05 W | — | 0.04 W | single-die names: `CPU Energy`, `GPU Energy`, `DRAM`, `AMCC`, `FAB` |
| M5 Max | 8× `yes` | 10.2 W | 0.004 W | 0.33 W | |
| M5 Max | MLX matmul | 0.6 W | **73.6 W** (5 s window), 56.9 W (3 s window) | | the 25 % spread between two windows is the M5 "energy batch cadence" issue (all-smi #410); use windows ≥ 5 s |
| Neo | idle | 0.005 W | — | 0.001 W | |
| Neo | MLX matmul | 0.1 W | **8.1 W** | | |

`sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power` also works
non-interactively on the lab Macs, under an admin account with passwordless
`sudo` (M5 idle: CPU 73 mW, GPU 0, ANE 0). It
reports the same SoC quantities as polled averages and needs root, so IOReport
is the better client-side source; `powermetrics` is the cross-check.

### SMC `PSTR`: total system power without a meter (no root)

`smc_power.c` opens the `AppleSMC` user client and reads float keys. `PSTR`
(system total), `PDTR` (DC-in from the adapter) and `PPBR` (battery) answer on
every Mac probed.

| Device | State | `PSTR` | `PDTR` | `PPBR` | SoC counters at the same time |
|---|---|---|---|---|---|
| M3 Ultra | idle | 8.6–11.3 W | = PSTR | n/a (desktop) | ~0.5 W total |
| M3 Ultra | MLX matmul | **248.8–249.7 W** | = PSTR | n/a | GPU 171.9 W |
| M5 Max | idle | 3.4–4.8 W | 3.2–4.8 W | 0.56 W | |
| M5 Max | MLX matmul | **201.7–208.0 W** | 119–126 W | **42–81 W** | GPU 56.9 W |
| Neo | idle | 0.5–1.3 W | 0.5–1.1 W | 0.03 W | |
| Neo | MLX matmul | **14.3–14.9 W** | 14.3–14.9 W | 0.03–0.05 W | GPU 8.1 W |

Three findings that shape the method:

1. **SoC counters miss most of idle and a third of load.** M3 Ultra idle is
   ~10 W at `PSTR` against ~0.5 W in the counters; under GPU load the GPU
   counter is 69 % of `PSTR`. An idle-subtracted SoC J/token would hide 10 W of
   always-on draw. Both scopes have to be reported.
2. **The MacBook Pro draws 40–80 W from its battery under GPU load while on
   AC.** `PDTR` caps near 125 W and `PPBR` supplies the rest. A wall meter on a
   laptop would under-read the load by that amount, and long runs drain the
   battery even on mains. `PSTR` is the right laptop-system figure, and battery
   state of charge must be recorded before and after.
3. **`PSTR` is a polled reading, not a counter,** so it needs ≥1 Hz sampling
   through the window. It is undocumented; a one-off comparison against a
   metering plug per model gives its error band.

`PHPC` also answers on the Mac Studio and MBP (37 W and 105–136 W under load)
but its meaning is unverified; it is not used.

### Rejected: `AppleSmartBattery` `PowerTelemetryData` (ioreg, no root)

Exposes `AccumulatedSystemEnergyConsumed`, `AccumulatedSystemPowerIn`,
`SystemPowerIn` and their sample counts. Across every 15–20 s probe window the
accumulators and counts did not advance at all, and `SystemPowerIn` on the M3
Ultra read 9.4 W while `PSTR` read 249 W. The update cadence is minutes and the
desktop value is meaningless. Not usable per repetition on a Mac. The same
block is live on the iPhone, where it publishes every 20 s and tracks load;
see §5.

## 2. Windows on AMD APU (GMKtec EVO-X2)

Probed on one GMKtec EVO-X2 box. The farm already runs `ryzenadj --info` before
every worker start to read back the power limits, so the ring-0 driver it needs
(`inpoutx64`) is installed. Windows' own `Energy Meter` RAPL counters read
0.00 W on these boxes.

`ryzenadj_load.ps1` samples `--info` idle, during a 12 s 16-thread CPU load,
and after:

| Sample | `STAPM VALUE` | `PPT VALUE FAST` | `PPT VALUE SLOW` | `THM VALUE CORE` |
|---|---|---|---|---|
| idle | 6.7 W | 26.3 W | 9.0 W | 29.0 °C |
| load + 5 s | 14.2 W | 137.5 W | 98.9 W | 37.5 °C |
| load + 9 s | 19.1 W | 118.0 W | **120.0 W** (= slow limit) | 38.8 °C |
| after + 3 s | 21.5 W | 14.9 W | 60.6 W | 34.3 °C |

- These are the SMU's own socket-power averages, each over a different time
  constant: `FAST` reacts within a second, `SLOW` over several seconds, `STAPM`
  over tens of seconds. None is a cumulative counter, so energy has to be
  integrated by polling; `FAST` at ~1 Hz is the right one for windows of
  10 s or more.
- Scope is the socket (CPU + iGPU + memory controller). The 128 GB of DRAM,
  SSD, fans and PSU losses are outside it, and there is no system-level source
  on the box. A metering smart plug is the only way to get system scope here.
- Each `ryzenadj --info` invocation takes under a second. The `pipette` worker
  process was idle (no inference child) during the probe.
- Limits reset on reboot, and
  [`gpu-keepalive`](../../../tools/gpu-keepalive/README.md) holds the GPU warm.
  Check and record both before an energy run.

## 3. Android

Probed on a Galaxy S26 Ultra (One UI 8.5, Android 16) and the Nothing 4a Pro
(Nothing OS 4), through the Linux host's adb server, as the shell user.

`android_probe.sh` results, both devices unless noted:

| Source | Result |
|---|---|
| `/sys/class/power_supply/*/current_now` | **permission denied** (S26 denies the directory listing; Nothing lists `battery`, `usb`, `wireless` under `qcom,pmic_glink` but denies reads) |
| `/sys/class/powercap` | permission denied (S26); absent (Nothing) |
| `android.hardware.power.stats.IPowerStats` (ODPM rails) | service absent (S26); `FAILED_TRANSACTION` (Nothing) |
| `dumpsys powerstats` | no channels, no energy consumers (Nothing); empty (S26) |
| `dumpsys battery` | works: `voltage` (mV), `temperature`, `Charge counter` (µAh), `AC powered`, and on Samsung a `current now` line plus a broadcast log with `current_avg` |

`android_bp.sh` calls `IBatteryPropertiesRegistrar.getProperty`, the binder
behind the public `BatteryManager.getIntProperty` API, at 5 Hz:

| Property | S26 Ultra | Nothing 4a Pro |
|---|---|---|
| `CHARGE_COUNTER` (µAh) | 3,947,090 (matches `dumpsys`) | 5,730,000 (matches) |
| `CURRENT_NOW` (µA, signed) | 2,343 → 0 → 10,156 → **−10,156** → 5,468 over 2 s | 1,019,007 → 1,068,446 → 1,008,020 (≈1.0–1.07 A charging at level 100) |
| `CURRENT_AVERAGE` (µA) | 781 | 866,959 |
| `ENERGY_COUNTER` | unsupported (error status) | unsupported |

So on both phones the public API delivers signed battery current at better
than 1 Hz. Voltage comes from `dumpsys battery` in the shell; an app cannot
call `dumpsys` and reads it from the `ACTION_BATTERY_CHANGED` sticky broadcast
(`EXTRA_VOLTAGE`, mV, no permission), which carries the same `voltage` and
`temperature` the broadcast log shows. Power at the battery terminals is
`CURRENT_NOW × voltage`. Positive is into the battery.

The client reads these from inside its own app, not from the shell, so the
S26 Ultra (SM-S948U1, Android 16 `user` build BP4A.251205.006, SELinux
enforcing) was rechecked from an app's sandbox by running `android_bp.sh`
under `run-as` on a debuggable build of the app. The answers match the
shell's: `CHARGE_COUNTER`, `CURRENT_NOW` and `CURRENT_AVERAGE` read,
`ENERGY_COUNTER` is unsupported, and `/sys/class/power_supply` and
`/sys/class/powercap` are denied. `dumpsys battery` is refused there ("Can't
find service: battery"). The CPU governor
(`/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`) reads `walt` from
both.

The constraint is the charger. The S26 sits at its 80 % protect cap on AC, so
the gauge sees a few mA of trickle; the Nothing is being charged at 1 A. **The
fuel gauge measures net current, and while a charger is attached that is not the
load.** The S26's own broadcast log shows what a benchmark run looks like:
at 02:16 on 09-17 `current_avg` went to **−687 mA** at 4.17 V (≈2.9 W drawn
from the battery) while temperature rose from 20.5 °C to 27.8 °C, with `ac:true`
throughout. The charge controller sourced part of the system load from the pack
while the handset sat at its 80 % protect cap. The phone reports no figure for
the charger's contribution, so this number is the controller's split, not the
device's consumption.

Consequences:

- A battery-side reading alone is valid only with the charger out of the
  circuit. `dumpsys battery unplug` fakes the framework state and leaves the
  hardware controller running, so only a physical cut counts. adb runs over
  Wi‑Fi, so cutting USB power costs nothing in connectivity. A cut run needs a
  starting state of charge and drains the battery; the 80 % cap and the clip-on
  coolers are conditions to record.
- The alternative is to meter the charger side. The phone cannot see the
  charger's contribution, but an inline USB-C meter between the charger port and
  the handset reads it directly. With both halves measured, system energy is the
  meter's energy less the net energy into the battery, and the controller's
  split cancels.
- The rack decides which route is cheap. The handsets share Anker Prime 250W
  stations, four USB-C and two USB-A ports each. A cooler pad uses a port of its
  own, so a station serves fewer phones than it has ports: in October 2026, each
  Galaxy S26 Ultra used three ports, one for the phone and one for each pad. A
  smart plug on the wall side therefore cuts a group, not one phone, and takes
  the group's other handsets off charge during a run. The stations switch ports
  individually over Bluetooth.
- ODPM-style per-rail energy is unavailable on every phone in the farm. The
  Pixel 4 XL predates it and nothing else exposes it to the shell or to an app.

## 4. Linux (the Linux host, Pi 5, private-network boxes)

### x86 (the Linux host, private-network boxes)

Not probed: none of them runs a measured client. For the record, RAPL through
`/sys/class/powercap/intel-rapl:0/energy_uj` (AMD Zen included since 5.11) is
a cumulative package counter; it is root-only on recent kernels and covers the
package, not DRAM on client parts.

### Raspberry Pi 5 (on the device, no root)

Probed on Pi 1 on 2026-09-30: Pi 5 Model B rev 1.1, 16 GB, Debian 13
on kernel 6.18, firmware of 2026-09-25, an active cooler and an NVMe HAT, on
WiFi with the Ethernet port down, governor `performance` at 2.4 GHz. Ten Pi 5s
run the benchmark worker.

**`vcgencmd pmic_read_adc`.** The DA9091 PMIC exposes twelve current and
fourteen voltage channels, one pair per regulator rail, through the firmware
mailbox. The rails are `VDD_CORE` (the Cortex-A76 cluster), `0V8_SW`,
`1V1_SYS`, `DDR_VDD2` and `DDR_VDDQ`, `1V8_SYS`, `3V3_SYS`, `3V7_WL_SW` (the
WiFi radio), `HDMI` at 5 V and the analog rails. The 5 V input has a voltage
channel (`EXT5V_V`) and no current channel; `BATT_V` is the RTC cell. The sum
of V×I over the rails is a board-rail figure: it covers the SoC, DRAM, I/O and
the radio, and omits the buck-regulator losses, the fan, the USB ports and
anything on a HAT, including this unit's NVMe. It is the Pi's
`vendor_internal` reading, the floor under the connector figure, in the
position IOReport holds on a Mac.

Each call is a snapshot: about 25 ms through the mailbox, no accumulator, no
averaging and no timestamp of its own. Sampled at 4 Hz during a live worker
benchmark the rail sum held 7.4–8.8 W with single rows at 3.7–6.5 W while the
core rail read in the same loop did not move, so a single row is noise and a
window is the mean of its rows. `vcgencmd power_monitor` exists in this
firmware as a rail-trace hook but rejects its own documented argument form,
so it was not pursued.

**The supply contract.** `/proc/device-tree/chosen/power` holds the
negotiated current limit and the supply's PDO list, readable without root.
This unit negotiated 5 V at 5 A from the official 27 W supply, whose PDOs are
5 V/5 A, 9 V/3 A, 12 V/2.25 A and 15 V/1.8 A. The 5 A entry is outside the
standard PD table; with a supply that offers 3 A the firmware caps USB
current at 600 mA and flags under-voltage under load. The contract belongs on
every row, because the board ships without a supply.

**`vcgencmd get_throttled`.** A bitfield: under-voltage (bit 0), ARM frequency
capped (1), currently throttled (2), soft temperature limit (3), each with a
sticky "has occurred" bit at 16–19. It read `0x0` throughout every probe here.
The Pi's readiness gate waits before each throughput-class repetition until
the four active-now bits (mask `0xF`) are clear and ignores the sticky ones
([`pi5.rs`](../../../crates/pipette-readiness/src/linux/pi5.rs)). Nothing reads
it inside a window, so a throttle event during a repetition goes unrecorded.

**Nothing else.** No `powercap` (RAPL is x86), no `power_supply` class, no
perf energy events. `hwmon` has the CPU thermal zone, the NVMe temperatures,
the fan tachometer and PWM, an under-voltage alarm (`rpi_volt`) and the RP1
I/O chip's general-purpose ADC, which is not a power channel.

The sampler and the sweep are `tools/energy-probes/pi_power_sampler.sh` and
`pi_sweep.sh`; the sweep's result is in §7.

## 5. iOS (readable from the host over the network tunnel)

Probed on 2026-09-19 and again on 2026-09-23 from Mac Studio 2, which hosted the
fleet's eleven iPhones at that time.

The phones are **not on USB**: `devicectl` reports `transportType:
localNetwork` for every one, and `system_profiler SPUSBHostDataType` lists no
iPhone. They charge from their own chargers and talk to the host over the
CoreDevice network tunnel. Every Apple-tool route over that tunnel fails:

- Xcode 26 `xctrace` with the Power Profiler template cannot record over it
  (`--all-processes` rejected, `--attach` cannot find the process, `--launch`
  hangs), and Power Profiler reports "impact" scores rather than joules anyway.
- `log collect --device-name` fails with "Device not configured": it needs USB.
- `devicectl` has no power or battery subcommand, and its file service cannot
  list the app container over the tunnel.
- The lockdown pairing records in `/var/db/lockdown` are TCC-protected even
  for root, and the tunnel's RSD discovery port refuses a direct connection.

**What works: the diagnostics relay, through Apple's own pairing.** The host's
`remoted` advertises `com.apple.mobile.diagnostics_relay.shim.remote` for every
phone. `pymobiledevice3` (11.17, run with `uvx`, nothing installed) has a
native-tunnel mode on macOS that piggybacks the pairing `remotepairingd`
already holds and publishes an RSD address, with no root:

```sh
uvx pymobiledevice3 remote start-tunnel --udid <udid> --script-mode   # prints HOST PORT
uvx pymobiledevice3 diagnostics ioregistry --ioclass AppleSmartBattery --rsd HOST PORT
```

One read takes about 0.6 s end to end. The `AppleSmartBattery` entry carries
the fuel gauge and, unlike on a Mac, a live `PowerTelemetryData` block:

| Group | Fields | Phone 2 at idle |
|---|---|---|
| gauge | `InstantAmperage` (mA, signed), `Voltage` (mV), `Temperature` (centi-°C), `CurrentCapacity`, `AppleRawCurrentCapacity` (mAh), `IsCharging`, `ExternalConnected`, `FullyCharged`, `CycleCount`, `UpdateTime` | 3 mA, 4391 mV, 19.2 °C, 100 %, holding |
| `PowerTelemetryData` | `SystemPowerIn`, `SystemVoltageIn`, `SystemCurrentIn`, `SystemLoad`, `BatteryPower`, `AdapterEfficiencyLoss`, `WallEnergyEstimate`; cumulative `Accumulated*` for each with sample counts | 292 mW in, 292 mW load, 0 battery |
| `AdapterDetails` | negotiated volts and amps, watts, charger type, the charger's offered PD menu | 5 V, 3 A, 15 W, "pd charger" |
| `ChargerData` | charge current and voltage limits, not-charging reason | 1000 mA, 4441 mV |

`SystemPowerIn` is power at the connector: `SystemVoltageIn × SystemCurrentIn`
reproduces it. `BatteryPower` is the battery term, negative when the pack
supplies the system. `SystemLoad` is `SystemPowerIn − BatteryPower` exactly in
every sample taken. That is the laptop's `PSTR = PDTR + PPBR` identity, computed
by the phone's power controller.

**Cadence.** The entry publishes every 20 s, all fields together. Between two
publishes the accumulator count advances by 20, so the controller integrates at
1 Hz. The accumulator delta divided by the count matches the published
instantaneous reading (468 vs 476 mW, 485 vs 480 mW), so
`AccumulatedSystemPowerIn` is a cumulative energy counter in mW·s, exact over
any window aligned to the 20 s ticks. `AccumulatedBatteryDischarge` is the same
for the battery term. Units are inferred from that consistency, not documented.
`AdapterEfficiencyLoss` and `WallEnergyEstimate` are Apple's estimate of the
brick's loss, not a measurement, and their units were not established.

**Fleet state.** The two phones read sit at 100 %, `FullyCharged`, controller
idle at 2–3 mA, negotiated at 5 V / 3 A. There is no 80 % cap on the iPhones.
Idle system power is 0.3–0.5 W.

The in-app IOKit read of the same entry from the
[private-API build](../../pipette-ios/private-thermal-release-build.md) is not needed
for a first iOS row: the host reads it. The load test in §6 shows the counter
holds under inference.

## 6. Validation: do the sources track load?

A single idle/load pair shows a source moves; it does not show it moves with
the load or returns. Each usable source was given a stepped ramp (idle →
increasing thread count → idle) with a compute-bound loop per thread.

**M3 Ultra, IOReport vs SMC `PSTR`** (`mac_ramp.sh`; SoC "all" sums the CPU,
GPU, DRAM, AMCC, DCS, ANE and FAB rollups):

| threads | SoC CPU | SoC all | `PSTR` system | system − SoC |
|---|---|---|---|---|
| 0 | 0.04 W | 0.71 W | 8.5 W | 7.8 W |
| 1 | 6.1 | 7.3 | 17.1 | 9.8 |
| 2 | 11.9 | 13.0 | 22.4 | 9.4 |
| 4 | 18.7 | 19.8 | 28.2 | 8.4 |
| 8 | 39.7 | 40.7 | 52.1 | 11.4 |
| 16 | 75.2 | 76.3 | 89.3 | 13.0 |
| 28 | 112.6 | 113.7 | 127.6 | 13.9 |
| 0 | 0.04 | 0.72 | 8.5 | 7.8 |

Monotonic, ~4–6 W per thread, back to the exact baseline. The two sources are
independent (IOReport energy counters vs the SMC's system-input reading) and
their difference is a stable 8 W floor plus a slowly growing conversion loss,
which is what a board and PSU look like. Over the 28-thread step `PSTR` rose
119 W and the SoC counters 113 W: the counters see 95 % of the added draw.
`powermetrics` agreed with the IOReport CPU figure within 3 % on every step
(8.79 vs 9.00, 10.54 vs 10.86, 13.45 vs 13.82 W).

Two agreeing sensors do not validate the workload: BSD `seq 1 0` still prints
`1 0`, so a naive zero-thread step runs two loops and both SoC sources read 9 W
"idle"; and `yes > /dev/null` is syscall-bound, so its energy does not scale
with threads while `PSTR` does.

**GMKtec EVO-X2, `ryzenadj` `PPT VALUE FAST`** (three samples 2 s apart per
step, 16-core Strix Halo):

| threads | samples (W) |
|---|---|
| 0 | 12.3, 24.2, 13.6 |
| 2 | 42.7, 48.7, 43.3 |
| 4 | 63.1, 64.1, 63.8 |
| 8 | 92.5, 92.7, 94.4 |
| 16 | 118.0, 119.9, 117.1 |
| 32 | 120.5, 117.9, 122.6 |
| 0 | 26.8, 13.8, 23.3 |

Monotonic, saturating at the 120 W slow limit once every physical core is
busy, back to idle. Under load the three samples agree within 2 W; **idle
scatters 12–27 W**, because `FAST` is a short-window average and the
PowerShell job runner and `ryzenadj` itself are visible in it. Take the idle
baseline from `PPT VALUE SLOW` or from many `FAST` samples, never from one.

**Galaxy S25 Ultra, `CURRENT_NOW` with the charger attached** (4 busy loops
for 16 s, 79 % at the protect cap):

| phase | `CURRENT_NOW` samples |
|---|---|
| idle | +2.3, −2.3, −10.2 mA |
| load | 0, −15.6, +5.5, −10.2, 0, 0 mA |
| after | +2.3, −4.7, +10.2 mA |

**The load is invisible.** A few watts of CPU work does not move the gauge
out of its ±15 mA noise because the charger supplies it. Only a load beyond
what the charger delivers reaches the battery (the S26's −687 mA event). So
on a plugged-in phone the direction of consumption cannot be seen at all,
which is the concrete reason the charger has to be cut for battery-scope
rows, not a precaution.

**Galaxy S26 Ultra, real inference on the charger** (S26 Ultra phone 4,
benchmark worker stopped; LFM2.5-1.2B Q4_0 on llama.cpp b10516; 2 Hz battery
sampler; 79 % at the protect cap, `AC powered: true`). A single run of the
client's own `decode_throughput_512_100` reads a plausible 0.15 J per generated
token, but **one run is one state of the charge controller**: in the §7 sweep
the same workload at other lengths read 0, 37, 77, 106 and 139 mJ/token, and
prefill came out negative, because the Samsung charge-cap
controller switches between feeding the SoC from the charger and from the
battery on its own hysteresis, and resumes charging mid-run when the level dips
to 79 %. The fuel gauge only ever sees the net. Nothing on the phone reports the
charger's share, so **a battery-side reading with a charger attached is not a
measurement on this fleet**, however plausible any single run looks. The
charger has to be physically cut, or the charger side metered, before an
Android energy row means anything.

**iPhone 17 Pro, real inference, `PowerTelemetryData` from the host**
(iPhone 17 Pro phone 2, benchmark worker stopped during the run;
`headlessrun benchmarks run benchmark=end_to_end_latency_8192_256` with
granite-4.2-3b Q4_0 on llama.cpp b10216, `sync` off; `ios_power_sampler.sh`
polling the registry every 1.5 s through a native tunnel, 75 s idle before and
after). The run completed normally: 31.74 s ± 0.06 s per request over five
repetitions. The counter kept its cadence throughout, 20 samples per 20 s
publish with no gap, and per 20 s block:

| Phase | Connector in | Battery | System load | Gauge |
|---|---|---|---|---|
| idle | 0.39–0.44 W | 0 | = in | +3 mA |
| a repetition | 10.6–12.4 W | −0.2 to −2.1 W | 11.0–14.5 W | −30 to −515 mA at 4.35–4.39 V |
| readiness wait between reps | 0.34–0.48 W | 0 | = in | +2–3 mA |

- The adapter stayed at 5 V / 3 A for the whole run. Connector input peaked at
  12.4 W and the battery supplied the rest, up to 2.1 W: the laptop pattern at
  phone scale. `BatteryPower` agreed with `InstantAmperage × Voltage` within
  5–10 % in every heavy block, and `SystemLoad = SystemPowerIn − BatteryPower`
  held exactly.
- Over the 380 s load window the connector accumulator read 2235 J (2078 J
  above the 0.41 W idle baseline) and the battery discharge accumulator 140 J,
  so about 2220 J of dynamic system energy for the model load, the warm-up and
  five repetitions, roughly 430–460 J per 8192-in / 256-out request at
  13–14.5 W.
- The readiness gate's die-temperature reading hit 73–74.5 °C after every
  repetition and the phone spent 24–36 s cooling to the 36 °C threshold before
  the next one, so the window's mean power is not a duty-cycle figure. Battery
  temperature rose from 19.3 to 29.2 °C across the run.
- The battery accumulator's unit is inferred as mW·s from its agreement with the
  instantaneous values. No token sweep was run on the iPhone (§8).

**Raspberry Pi 5, PMIC rail sum** (`pi_sweep.sh`: 60 s idle before the
sweep, 21 minutes of runs, 30 s idle after; the sampler's four mailbox reads
a second are inside every figure):

| phase | rails | `VDD_CORE` | DDR | WiFi rail | `EXT5V_V` |
|---|---|---|---|---|---|
| idle, 60 s | 2.19 W (sd 0.20) | 0.94 W | 0.02 W | 0.36 W | 5.157 V |
| rows above 5 W across the sweep | 7.56 W | 5.90 W | 0.31 W | 0.37 W | ≥5.116 V |
| idle, 30 s after | 2.27 W | | | | |

Back to within 0.08 W of the baseline after 21 minutes of load, with the
input holding above 5.1 V and `get_throttled` at `0x0` throughout. The radio
draws the same 0.36 W idle and loaded, so it belongs in the baseline; the
unit is on WiFi with its Ethernet port down. Decode puts 0.3 W on the DRAM
rails against 0.02 W idle, which is the memory-bound phase showing in the
one place the Pi separates it out.

**Trust summary.** Mac SoC counters and `PSTR`: trustworthy, cross-validated
against each other and `powermetrics`. The absolute `PSTR` figure can be
bracketed, SoC counters below and a wall plug above, but not calibrated: the
plug reads AC and `PSTR` reads DC, and the PSU's loss sits between them.
GMKtec `ryzenadj`: trustworthy under load, noisy at idle, no independent
cross-check on the box until a plug bounds it from above. Android
battery current: the API is reliable, the number is not while a charger is
attached; the S26 sweep (§7) read the same work at anywhere between 0 and
139 mJ/token depending on what the charge controller was doing; metering the
charger side recovers it, untested here. iOS: the phone's own power controller
reports the connector term, the battery term and their sum at 1 Hz, published
every 20 s, and held that cadence through a five-repetition benchmark; the two
terms agree with the fuel gauge; units inferred, not documented, and no sweep
yet. Pi 5 rail sum: an extensive quantity with R² of at least 0.998 on both phases
and a clean return to baseline, but single reads dip on the core channel and
the mean of a loaded window sits 9 % under its median, and no second source
exists on the board until the meter is on the connector.

## 7. Scaling law per device

A source that moves with load and returns to idle can still be reporting the
wrong quantity. The check that separates a measurement from a coincidence is
that energy behaves like a physical extensive quantity: linear in the work,
with a fixed overhead that separates cleanly, and repeatable. Each device ran
the same engine the client uses (`llama-bench` from the pipette runtime store,
b10516, LFM2.5-1.2B Q4_0; on the Pi, Llama-3.2-1B-Instruct Q4_K_M, the model
in its own store) over token sweeps, each length twice with `-r 2` and
`-r 6`; the difference is exactly four measured repetitions, so
E_rep = (E₆ − E₂)/4 cancels model load and warm-up. Decode is at depth 512.
The sweep scripts and the fit are in `tools/energy-probes/`.

**Mac Studio M3 Ultra** (IOReport SoC counters bracketing the process; `PSTR`
polled at 4 Hz):

| decode n | rep time | E_rep (SoC) | mJ/token | W | | prefill n | rep time | E_rep (SoC) | mJ/token | W |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 121 ms | 15.7 J | 314 | 131 | | 256 | 30 ms | 4.5 J | 17.6 | 152 |
| 100 | 239 ms | 31.6 J | 316 | 132 | | 512 | 52 ms | 8.4 J | 16.5 | 162 |
| 200 | 476 ms | 63.7 J | 318 | 134 | | 1024 | 105 ms | 16.4 J | 16.0 | 157 |
| 400 | 938 ms | 126.9 J | 317 | 135 | | 2048 | 211 ms | 34.0 J | 16.6 | 161 |
| 800 | 1908 ms | 257.7 J | 322 | 135 | | 4096 | 435 ms | 71.2 J | 17.4 | 164 |

- Linear fit R² = 0.999 on prefill; decode is flat to ±1.3 % over 16× in
  length. A repeat of n=200 agreed to 0.1 % (63.7 vs 63.7 J).
- The first run of the session paid a cold model load (9.3 s, 188 J) and
  would have put −21 J on the n=50 point had it not been re-run warm. The
  harness must discard the first run of a session, not only the warm-up rep.
- `PSTR` needs runs of seconds: at 4 Hz a 300 ms run holds 2 samples. On the
  long runs (decode 800 × 14 = 27 s; prefill 4096 × 14 = 6.7 s) the system
  reading is **1.27× and 1.28× the SoC counter** (412 vs 324 mJ/token decode,
  22.3 vs 17.4 prefill), the same ratio on both phases. The ratio holds at
  this load only. The ramp in §6 shows the relation between the two sources is
  a floor of about 8 W plus a slope, so the ratio grows as power falls and is
  20× at idle. Carry the offset and the slope from the ramp if a conversion is
  ever needed.

**GMKtec EVO-X2** (`ryzenadj --info` `PPT VALUE FAST` integrated over each
run; benchmark worker stopped):

| decode n | rep time | E_rep | mJ/token | W | | prefill n | rep time | E_rep | mJ/token | W |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 192 ms | 19.4 J | 389 | 101 | | 256 | 32 ms | 4.0 J | 15.5 | 124 |
| 100 | 384 ms | 39.0 J | 390 | 101 | | 512 | 65 ms | 8.6 J | 16.7 | 132 |
| 200 | 768 ms | 74.9 J | 374 | 97 | | 1024 | 130 ms | 17.6 J | 17.2 | 135 |
| 400 | 1541 ms | 148.4 J | 371 | 96 | | 2048 | 264 ms | 37.3 J | 18.2 | 141 |
| 800 | 3090 ms | 307.3 J | 384 | 100 | | 4096 | 550 ms | 70.9 J | 17.3 | 129 |

- Decode flat to ±2.5 %, prefill to ±8 %. The 44 s long run (800 × 14) read
  407 mJ/token, 5 % above the short runs, at 105 W: the box runs warmer over a
  long window, which is real and is what the sustained test is for.
- `ryzenadj --info` answers at ~27 Hz in a tight loop, far faster than the
  1 Hz polling that §2 recommends, **but polling that hard is itself a load**: idle read
  27.8 W with the sampler running against 9–13 W without. The client should
  poll at 1–2 Hz and record the sampler's own cost in the idle baseline.
- No independent system-scope source exists on the box; the socket figure
  stands alone until a plug is fitted.

**Raspberry Pi 5** (PMIC rail sum sampled at 4 Hz and averaged over each
run; 4 threads, flash attention on, the worker's own `llama-bench`; benchmark
worker stopped):

| decode n | rep time | E_rep | mJ/token | W | | prefill n | rep time | E_rep | mJ/token | W |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 4.01 s | 22.5 J | 449 | 5.6 | | 256 | 3.47 s | 27.6 J | 108 | 8.0 |
| 100 | 7.94 s | 47.1 J | 471 | 5.9 | | 512 | 7.12 s | 58.0 J | 113 | 8.1 |
| 200 | 16.6 s | 96.7 J | 483 | 5.8 | | 1024 | 14.9 s | 120.9 J | 118 | 8.1 |
| 400 | 33.5 s | 201.4 J | 504 | 6.0 | | 2048 | 32.8 s | 267.2 J | 131 | 8.1 |

- Linear fit R² = 0.9998 on decode (512 mJ/token slope) and 0.998 on prefill
  (134 mJ/token). Power is flat within each phase, so the per-token rise
  (12 % over 8× on decode, 21 % on prefill) is time and not power: decode
  slows from 80 to 84 ms/token as the KV cache grows, and prefill from 13.5 to
  16.0 ms/token as attention grows with the prompt. The S26 showed the same
  thing: J per token is quoted at the benchmark's fixed depth and count.
- The fixed overhead per run is 55–70 J on decode and equals one repetition
  on prefill (36, 63, 132, 274 J against 28, 58, 121, 267 J), which is
  `llama-bench`'s warm-up repetition; the pair differences it out.
- The reads are noisy in one direction. About a tenth of the rows sit 1.5 W
  or more below their neighbors, in single-row events, and 96 % of the
  deficit is on the `VDD_CORE` channel. Over a loaded window the mean is
  7.3 W and the median 8.0 W. Whether the dips are real (a brief core idle
  between tokens) or an ADC artifact is what the meter on the connector
  settles; the table uses the mean.
- A Pi decode token costs 1.6× the rail energy of an M3 Ultra SoC token
  (504 against 314–322 mJ) at a twentieth of the power, because it takes
  thirty times longer. Different models and different `vendor_internal`
  scopes, so a shape rather than a comparison, but it is the per-watt point
  of the landscape's survey entry ([01](01-landscape.md) §2) in one row:
  watts favor the Pi and joules do not.

**Galaxy S26 Ultra** (battery current × voltage integrated; charger attached
at the 80 % cap; benchmark worker stopped):

| decode n | rep time | E_rep | mJ/token | | prefill n | rep time | E_rep | mJ/token |
|---|---|---|---|---|---|---|---|---|
| 50 | 677 ms | 7.0 J | 139 | | 256 | 473 ms | −3.2 J | −13 |
| 100 | 1325 ms | 10.6 J | 106 | | 512 | 1310 ms | −5.5 J | −11 |
| 200 | 2761 ms | 7.4 J | 37 | | 1024 | 3640 ms | −18.0 J | −18 |
| 400 | 7421 ms | −0.2 J | 0 | | 2048 | 10.6 s | 7.0 J | 3 |
| 800 | 14.0 s | 61.5 J | 77 | | 4096 | 32.4 s | 6.7 J | 2 |

No law at all. The raw series explains it: in the 84 s decode-800 run the
battery supplied 2.2 A, then ~0.7 A for 40 s, then 1.4 A, for one constant
workload; in decode-400 the battery read ~0 mA after the first 3 s while the
charger carried the load; in prefill-1024 the level dipped to 79 %, charging
resumed at +1.9 A during the run, and the integral went negative. The charge
controller's share is invisible to the phone. The rep times also show a real
effect the other devices hide: decode slowed from 75 tok/s at n=100 to 57 at
n=800 as the KV cache grew, so "J per generated token" on a CPU phone depends
on the generation length and must be quoted at the benchmark's fixed depth
and count, never extrapolated.

**What the four sweeps establish.** On the Mac, the GMKtec and the Pi the
sources report an extensive quantity:

- The energy is linear in the work: R² of at least 0.998 where fitted, and the
  GMKtec's decode energy per token is flat to ±2.5 %.
- Per-token energy is constant on the two accelerators, and it tracks the
  repetition time on the CPU-bound Pi.
- The fixed overheads difference out, and repeats agree at the percent level.
- On the Mac a second, independent reading holds a constant ratio.

Those numbers can be published with their scope. They cannot be published against each other: the
Mac's 314–322 mJ/token and the GMKtec's 371–390 read as directly comparable
and are not, because Apple's counter includes DRAM while `ryzenadj`'s excludes
the box's 128 GB, and decode is memory-bound. Only the system and wall figures
compare across the two. The Pi's rail sum carries a 9 % question between its
mean and its median that the meter on its connector answers. On the S26 the
source is fine and the measurement is not, until the charger is out of the
circuit or the charger side is metered. For the harness: discard the first
run of a session, keep every measured window above a few seconds for polled
sources, poll gently, and define decode energy at a fixed depth and length.

## 8. What the probes did not cover

- No client code reads a power source.
- No metering plug was available, so `PSTR` and `ryzenadj` have no wall
  bracket yet. A plug
  bounds a DC-side source from above; it does not calibrate it, because the
  PSU's loss sits between the two. On a laptop the plug also misses the
  battery's contribution under load: it reads the adapter's ~125 W while the
  machine draws 200 W. The like-for-like checks on a laptop are the inline
  USB-C meter against `PDTR` at the adapter output, and the smart battery's
  own gauge against `PPBR`; neither was available or probed here.
- No sweep was run on the MacBook Pro. §7 covers the M3 Ultra, the GMKtec, the
  Pi 5 and the S26; the M5 Max has only the idle and matmul pairs in §1. Two things turn
  on that. The 201–208 W in §1 comes from an MLX matmul loop, and on the M3
  Ultra real llama.cpp decode drew 131–135 W against 171 W for the same loop,
  so the laptop's battery draw under a benchmark is unknown. State of charge is
  also a second throttle beside thermal: as the buffer runs down under sustained
  load, macOS clamps SoC power to what the adapter delivers, silently, so a run
  at 95 % and a run at 30 % measure different machines. Run the §7 sweep on an
  M5 Max at full charge and again near 25 %, compare tok/s and mJ/token, and
  watch `PPBR`.
- No inline USB-C meter was available, so the metered route in §3 is reasoned
  from the battery-side data rather than demonstrated.
- No token sweep was run on the iPhone; §6 has one end-to-end shape. The 20 s
  publish cadence means the §7 pair on an iPhone needs windows that are whole
  multiples of 20 s, or a repetition count that makes the edge blocks small.
  The inline USB-C meter, once present, is the like-for-like check of
  `SystemPowerIn` at the connector.
- No inline meter was on Pi 1, so its rail sum has no connector
  bracket yet, and whether the meter passes the 5 V 5 A contract through is
  untested. A sweep on a Pi with an inline meter gives the bracket.
- Mac Studio 1 was used for the M3 Ultra; `sudo -n
  powermetrics` was not exercised there.
