# Private-API builds (`PIPETTE_PRIVATE_THERMAL`, `PIPETTE_PRIVATE_MEMORY`)

What the flags are and the one rule around them. For building/deploying the app in
general, see [build.md](build.md).

> [!CAUTION]
> **Never submit a `PIPETTE_PRIVATE_THERMAL` or `PIPETTE_PRIVATE_MEMORY` build to the
> App Store or TestFlight.** They read the SoC die temperature (`soc_temp`), the CPU
> and GPU clock-state residency (`libIOReport`) and the kernel's footprint peak
> (`proc_reset_footprint_interval`, `proc_pid_rusage`), and flush libmalloc's reclaim
> ring (`mach_vm_reclaim_*`), through private APIs, which App Review rejects. These
> builds are for on-device benchmarking only: side-loaded onto registered
> development devices. The distributable build has both flags off.

Published iOS results come from a build of this app with both flags on. The flags read
the exact SoC die temperature and the kernel's footprint peak, which the public iOS API
does not expose. Anyone can make the same build from this repository, as below. Why
this matters for reproducing results is in [methodology](../methodology/README.md).

## What they do

`PIPETTE_PRIVATE_THERMAL` enables the exact SoC die-temp read used to (a) gate
benchmark readiness (cool the device between measured reps so throttling doesn't skew
numbers), and (b) calibrate the public IMU thermometer (`headlessrun … metrics=calibrate`).
It also logs each measured rep's CPU-cluster and GPU clock-state residency (`libIOReport`),
because `ProcessInfo.thermalState` stays `.nominal` while the GPU down-clocks, so only the
log ties a slow rep to a down-clock. Nothing of it reaches the result.
With it off, the readiness gate falls back to `ProcessInfo.thermalState` + the
calibrated IMU estimate, and nothing touches the private API.

`PIPETTE_PRIVATE_MEMORY` makes `max_memory_usage` include the kernel's exact footprint
peak for each measured bracket: `proc_reset_footprint_interval` as the bracket opens,
`ri_interval_max_phys_footprint` from `proc_pid_rusage` as it closes. The reported
figure is the larger of that and the 20 ms `phys_footprint` sample, which can miss a
short spike ([memory carry-over](memory-carry-over.md) has the measured miss). If the
reset or the read fails, the sample alone is reported. With the flag off, the sample is
the figure. `AppLog.memory` logs both (`sampled=… kernel=…`).

It also flushes libmalloc's deferred-reclaim ring before each bracket
(`mach_vm_reclaim_get_rings_for_task`, `mach_vm_reclaim_ring_capacity`,
`mach_vm_reclaim_ring_flush`). iOS 27 parks freed memory there, still counted in
`phys_footprint`, so without the flush a cell inherits what earlier cells in the
process freed. See
[cross-cell isolation](../methodology/peak-memory-ios.md#cross-cell-isolation-settle-to-floor)
and [memory carry-over](memory-carry-over.md). A failed flush is logged and the
measurement goes ahead.

With the flag off the memory benchmark still runs, but a cell that follows other cells
in the same app can read high; how far each build's figure can be trusted is in
[private-API and public builds](../methodology/peak-memory-ios.md#private-api-and-public-builds).

## How a build gets them

**Off by default, opt in explicitly.** The project defines nothing, so a plain
build (including the Xcode Cloud archive that ships to TestFlight) never compiles
them in. Pass `--internal` to `ios/build.sh` (setting `PIPETTE_PRIVATE_THERMAL=1` in
the environment does the same). For that build only, the script adds
`PIPETTE_PRIVATE_THERMAL` to `GCC_PREPROCESSOR_DEFINITIONS` and both conditions to
`SWIFT_ACTIVE_COMPILATION_CONDITIONS`. One switch turns on both, so a private-API
build stays a single decision. The CI `ios-archive` job sets both flags directly on `xcodebuild`
(see [build.md § Installing a released archive](build.md#installing-a-released-archive)). The opt-in is configuration-independent: it applies to whichever
`-configuration` you build (Debug or Release), so a distributable archive is
private-API-free unless a human deliberately asks for the flag.

`headlessrun version` reports each: `private_thermal=on|off private_memory=on|off`.
Either one makes the version carry `-internal`. The runtime descriptors say the same:
`private_thermal: true` on a build with both, which is what one switch produces.
`private_memory` defaults to `private_thermal`, so it is written only by a build with one
and not the other (`private_memory: false` beside `private_thermal: true`, or
`private_memory: true` alone). A cell that asks for a capability is refused by a build
without it.

## Building it

Pass `--internal` to opt in; otherwise build and install per
[build.md](build.md#running-on-a-physical-device-from-the-cli). The build defaults to
Release, which is what benchmark numbers need:

```bash
./ios/build.sh device --internal -derivedDataPath /tmp/pipette-dd
xcrun devicectl device install app --device <udid> \
  /tmp/pipette-dd/Build/Products/Release-iphoneos/Pipette.app
```

For a Debug build, pass `-configuration Debug` (the product path becomes
`Debug-iphoneos`). Uninstall first for a clean slate (clears downloaded models and all app data):
`xcrun devicectl device uninstall app --device <udid> ai.liquid.liquid-pipette`.

## Distributable builds

The flags are absent unless you opt in, so a normal archive (Xcode Cloud, or
`xcodebuild archive` without the flags) contains no private API. Do not pass
`--internal` for a TestFlight or App Store build.

The private APIs are resolved with `dlsym`, so there is no undefined symbol to look
for; the names appear as strings in `__TEXT` instead. Check the archive:

```bash
strings /tmp/Pipette.xcarchive/Products/Applications/Pipette.app/Pipette \
  | grep -i -e IOHIDEventSystemClient -e proc_reset_footprint_interval -e proc_pid_rusage \
      -e mach_vm_reclaim -e 'IOReport[a-z]'
```

Expect no output. `IOReport[a-z]` matches the private `IOReport...` function names but not the
`IOReport unreadable` log line, which every build carries.
