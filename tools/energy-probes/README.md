# Energy probes

Probes used to establish which power sources each farm device class exposes, and
whether what they report scales with the work done. The results and their
interpretation are in
[`docs/methodology/energy/02-device-capabilities.md`](../../docs/methodology/energy/02-device-capabilities.md);
this directory is how to reproduce them. Nothing here ships in a client.

Sweep output and the compiled probes are gitignored: rerun a sweep to regenerate
its numbers rather than reading a stale copy.

## Reading a power source

| File | Platform | What it reads | Privilege |
|---|---|---|---|
| `ioreport_energy.c` | macOS | IOReport `Energy Model` counters (CPU, GPU, ANE, DRAM …) as a delta over N ms | none |
| `smc_power.c` | macOS | SMC float keys: `PSTR` (system total power), `PPBR` (battery), `PDTR` (DC-in) | none |
| `ryzenadj_load.ps1` | Windows, AMD APU | `ryzenadj --info` socket power (`PPT VALUE FAST/SLOW`, `STAPM VALUE`) idle vs a 12 s CPU load | ring-0 driver already on the GMKtec boxes |
| `android_probe.sh` | Android (adb shell) | survey: `dumpsys battery`, `/sys/class/power_supply`, powercap, ODPM `IPowerStats`, `PowerStatsService` | none |
| `android_bp.sh` | Android (adb shell) | `IBatteryPropertiesRegistrar.getProperty` for CHARGE_COUNTER, CURRENT_NOW, CURRENT_AVERAGE, ENERGY_COUNTER, then CURRENT_NOW ten times at 5 Hz | none |
| `pi_power_sampler.sh` | Raspberry Pi 5, on the device | `vcgencmd pmic_read_adc`: current and voltage per PMIC rail, summed to a board-rail figure, with the firmware's throttle flags, one CSV row per ~250 ms | none |
| `ios_power_sampler.sh`, `ios_power_parse.py` | iOS, run on the host Mac | the `AppleSmartBattery` entry over the CoreDevice network tunnel: `PowerTelemetryData` (connector power in, battery power, system load, cumulative accumulators, published every 20 s), the fuel gauge, and the adapter negotiation, one CSV row per poll | none (Apple's own pairing, through `pymobiledevice3`'s native tunnel via `uvx`) |

## Checking that a source scales

`mac_ramp.sh` steps a compute load from 0 to 28 threads and back. This is the
quick check: a usable source rises monotonically and returns to its baseline.

The sweeps are the full check: each runs `llama-bench` over a range of token
counts at `-r 2` and `-r 6` and differences the pair, as described in
[§7 of the capabilities doc](../../docs/methodology/energy/02-device-capabilities.md#7-scaling-law-per-device).

| File | Platform |
|---|---|
| `ewrap.c` | macOS: runs a command, reports IOReport SoC energy bracketing it and `PSTR` polled at 4 Hz |
| `mac_sweep.sh` | macOS sweep, using `ewrap` |
| `gmktec_sweep.ps1` | Windows sweep, sampling `ryzenadj` alongside |
| `android_sweep.sh`, `android_sampler.sh` | Android sweep plus its 2 Hz battery sampler |
| `pi_sweep.sh` | Raspberry Pi 5 sweep, with `pi_power_sampler.sh` running throughout and 60 s of idle before the first run |
| `fit.py` | per-repetition energy and the linear fit, over `mac_sweep.sh` or `pi_sweep.sh` output |

## Running

The macOS ramp and sweep run from `/tmp/ep`. Copy this directory there first.
`mac_sweep.sh` sets `LB` (the `llama-bench` path) and `M` (the model path) at
its top; edit them to point at your `llama-bench` and model.

```sh
# macOS (on the host, from /tmp/ep)
clang -O1 -o e ioreport_energy.c -framework CoreFoundation -lIOReport && ./e 5000
clang -O1 -o smc smc_power.c -framework IOKit && ./smc 3 PSTR PPBR PDTR
bash mac_ramp.sh                  # needs e and smc, built above
bash mac_sweep.sh > sweep.txt && python3 fit.py sweep.txt

# Android (from the adb host, the Linux host)
adb -s <serial> shell sh -s < android_probe.sh
adb -s <serial> shell sh -s < android_bp.sh
# the same reads from an app's sandbox (a debuggable build of the app)
adb -s <serial> shell run-as <package> sh -s < android_bp.sh
# the sweep: push the sampler under the name the sweep starts, then run the sweep
# (it expects the pipette store under /data/local/tmp/pipette)
adb -s <serial> push android_sampler.sh /data/local/tmp/energy_sampler.sh
adb -s <serial> shell sh -s < android_sweep.sh > sweep.txt

# Raspberry Pi 5 (on the device; llama-bench and the model from the worker's runtime store,
# override with LB=, M= and CSV=)
bash pi_sweep.sh > sweep.txt && python3 fit.py sweep.txt

# iOS (on the Mac that hosts the phones; udid from `uvx pymobiledevice3 remote browse`)
./ios_power_sampler.sh <udid> samples.csv &    # stop with: touch samples.csv.stop

# Windows (from a bash host over ssh): encode the script as base64 UTF-16LE, then run it
ssh <box> powershell -NoProfile -EncodedCommand \
  "$(iconv -f UTF-8 -t UTF-16LE ryzenadj_load.ps1 | base64 | tr -d '\n')"
# gmktec_sweep.ps1 runs the same way; it reads ryzenadj, llama-bench and the
# model from the paths at its top
```

Before you put load on a device, stop the service that runs the pipette worker
for it. Start it again afterwards.
`ryzenadj_load.ps1` and `gmktec_sweep.ps1` also check that the worker has no
inference child before they start, so a stray run cannot corrupt a live
benchmark. `android_sweep.sh` sets `LD_LIBRARY_PATH` to the runtime blob
directory itself.
