# Private thermal builds (`PIPETTE_PRIVATE_THERMAL`)

What the flag is and the one rule around it. For building/deploying the app in
general, see [build.md](build.md).

> [!CAUTION]
> **Never submit a `PIPETTE_PRIVATE_THERMAL` build to the App Store or TestFlight.**
> It reads the SoC die temperature through a private API (`soc_temp`), which App
> Review rejects. These builds are internal, for on-device benchmarking only:
> side-loaded onto registered development devices. The distributable build has the
> flag off.

## What it does

The flag enables the exact SoC die-temp read used to (a) gate benchmark readiness (
cool the device between measured reps so throttling doesn't skew numbers), and
(b) calibrate the public IMU thermometer (`headlessrun … metrics=calibrate`). With
it off, the readiness gate falls back to `ProcessInfo.thermalState` + the calibrated
IMU estimate, and nothing touches the private API.

## How a build gets it

**Off by default, opt in explicitly.** The project defines nothing, so a plain
build (including the Xcode Cloud archive that ships to TestFlight) never compiles
it in. Set `PIPETTE_PRIVATE_THERMAL=1` in the environment when you run
`ios/build.sh` and it adds the define to `GCC_PREPROCESSOR_DEFINITIONS` for that
build only. The opt-in is configuration-independent: it applies to whichever
`-configuration` you build (Debug or Release), so a distributable archive is
private-API-free unless a human deliberately asks for the flag.

## Building it

Prefix the build with `PIPETTE_PRIVATE_THERMAL=1` to opt in; otherwise build and
install per [build.md](build.md#running-on-a-physical-device-from-the-cli). The
build defaults to Release, which is what benchmark numbers need:

```bash
PIPETTE_PRIVATE_THERMAL=1 ./ios/build.sh device -derivedDataPath /tmp/pipette-dd
xcrun devicectl device install app --device <udid> \
  /tmp/pipette-dd/Build/Products/Release-iphoneos/Pipette.app
```

For a Debug build, pass `-configuration Debug` (the product path becomes
`Debug-iphoneos`). Uninstall first for a clean slate (clears downloaded models and all app data):
`xcrun devicectl device uninstall app --device <udid> ai.liquid.liquid-pipette`.

## Distributable builds

Nothing to strip: the flag is absent unless you opt in, so a normal archive (Xcode
Cloud, or `xcodebuild archive` without the env var) is already App-Review safe;
just don't set `PIPETTE_PRIVATE_THERMAL=1` when producing a TestFlight / App Store
build.
