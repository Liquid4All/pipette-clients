# Building the Pipette iOS App

How to compile the vendored llama.cpp into the app, and build/run it. For the
architecture behind these artifacts (the native Swift engines, vendored
llama.cpp, core concepts), see [architecture.md](architecture.md).

> The iOS app is **pure Swift**. There is no Rust in the iOS build. llama.cpp is
> compiled directly into the app from the shared `vendor/llama.cpp` submodule
> (the same one Android builds) by `ios/build-llama.sh`.

## How to build

There are exactly two supported ways to build Pipette, and you only need one:

- **`ios/build.sh`**: the command-line entry point. Best for CI, scripted
  loops, and simulator work. One command picks the build destination, sets the
  right flags, checks your toolchain, and builds the app.
- **The Xcode GUI**: open `ios/Pipette/Pipette.xcodeproj`, pick a destination,
  and Run (⌘R). Best for interactive development, and **required at least once
  to deploy to a physical device** (see below).

Both produce the same app: pick whichever fits your workflow. Jump to
[Building from the command line](#building-from-the-command-line) or
[Building from the Xcode GUI](#building-from-the-xcode-gui).

### Deploying to a device needs the GUI once

**Deploying to a physical device requires the Xcode GUI at least once** to set
up code signing: select a development team and register the device. After that
the command line can build and install for the device, but it cannot bootstrap
signing on its own. The simulator path is fully command-line-only.

### What each path needs from you

A handful of one-time setup items differ between the two paths. (The llama.cpp
static library, vendored sources, and the `llama` module are *not* in this
list: they're produced automatically by either build; see
[Build pipeline](#build-pipeline).)

| Setup item | Command line | Xcode GUI |
| --- | --- | --- |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` | Xcode → Settings → Components → install |
| **LicenseList plugin trust** | `-skipPackagePluginValidation` (baked into `ios/build.sh`) | click **Trust & Enable** once when prompted |
| **Device code signing** | *uses* signing once configured: cannot bootstrap it | **required**: one-time team + device setup under Signing & Capabilities |
| **Which slice builds** | `sim` / `device` argument | the scheme's run destination |

## Prerequisites

### Install these yourself (both paths)

The build can't bootstrap these; install them before your first build:

- **Xcode** with command-line tools (`xcrun`, `xcodebuild`, `simctl`, `clang`,
  `libtool`). `build-llama.sh` compiles llama.cpp with these.
- **CMake** (`brew install cmake`) and **Ninja** (`brew install ninja`).
  `build-llama.sh` drives llama.cpp's own CMake build; CMake owns the source
  list, the compile flags, and the embedded Metal library.
- **The Metal toolchain.** Xcode 16+ unbundles the Metal compiler into a
  separate, on-demand component. Without it the `mlx-swift` dependency fails
  with a wall of `CompileMetalFile` errors (the real message is `cannot execute
  tool 'metal' due to missing Metal Toolchain`).
  - CLI: `xcodebuild -downloadComponent MetalToolchain`
  - GUI: **Xcode → Settings → Components**, install "Metal Toolchain"
  - Verify either way: `xcodebuild -showComponent MetalToolchain` → `Status: installed`

  > CI runners ship the Metal toolchain preinstalled, so this only bites local
  > machines on a fresh Xcode.
### Handled automatically (both paths)

The build initializes these on every run (GUI or command line), so you normally
don't touch them. Listed here for when you build the llama.cpp library by hand
or need to debug a failure:

- **Vendored llama.cpp.** The shared `vendor/llama.cpp` submodule (also built by
  Android). `build-llama.sh` initializes it if missing, then builds from a copy
  so the submodule itself is never modified.

### Optional

- **[`sccache`](https://github.com/mozilla/sccache)** (`cargo install sccache`
  or `brew install sccache`) caches the ~200 llama.cpp/ggml C/C++ compiles
  across clean rebuilds, worktree switches, and device↔simulator flips.
  `build-llama.sh` wires it as CMake's compiler launcher when present; set
  `PIPETTE_NO_SCCACHE=1` to opt out.

## Build pipeline

You never invoke the llama.cpp build yourself. It runs automatically inside
every app build. A helper script, `ios/build-llama.sh`, is registered as the
**"Build llama.cpp"** run-script phase in the Xcode project, so it runs as a
dependency of the app build whether you start that build from `ios/build.sh` or
the GUI:

```
ios/build.sh  /  Xcode GUI
  ├─ "Build llama.cpp" phase ──→ ios/build-llama.sh  (compiles libllama.a + `llama` module)
  └─ compiles Swift, links, produces Pipette.app
```

The order matters (the `llama` module and `LlamaCppBuildInfo.swift` must exist
before the Swift that imports/uses them compiles), and the phase guarantees it.
What that phase does:

1. Syncs a build **copy** of the `vendor/llama.cpp` submodule (so the submodule
   stays pristine), and applies our carried patch series (`ios/patches/NNN-*.patch`,
   in numeric order (currently just `001-ggml-metal-oom-nullcheck.patch`)) to the
   **copy**. Fails loudly if a patch no longer applies (see pitfalls).
2. Configures + builds the copy through **llama.cpp's own CMake**, with inline
   `-D` flags (Metal + embedded shader library on device, CPU on the simulator;
   Accelerate): exactly like upstream `build-xcframework.sh`. The Android
   repo-root CMake (`CMakeLists.txt` / `CMakePresets.json`) is **not** involved:
   iOS configures the vendored tree directly. CMake owns the source list, the
   per-target compile flags, and the `GGML_METAL_EMBED_LIBRARY` metallib embed:
   nothing is hand-transcribed.
3. Combines the per-target static libs (`libllama.a` + `libggml*.a`) into one
   `libllama.a` with `xcrun libtool -static`, then writes
   `Generated/Llama/{libllama.a, module.modulemap, *.h, LlamaCppBuildInfo.swift}`.
4. Copies the shared benchmark prompt corpus
   (`crates/pipette-ops/src/prompt_seed.txt`) into
   `Generated/PromptSeed/prompt_seed.txt`, bundled as an app resource so the
   native llama benchmarks tokenize the same text as the Rust CLIs / Android.
5. The Xcode project links `-lllama` and Swift does `import llama`.

It also stamps the resolved llama.cpp commit into `LlamaCppBuildInfo.swift`
(surfaced to Swift as `llamaCppCommit()` → `runtime_version`) and honors the
`sccache` opt-in. For *how* the native Swift engines fit together, see
[architecture.md](architecture.md).

## Building from the command line

`ios/build.sh` builds the whole app in one command. It selects the destination,
passes the required `xcodebuild` flags, pre-flights the Metal toolchain, and
triggers the llama.cpp build automatically.

```bash
./ios/build.sh           # build for the simulator (default)
./ios/build.sh sim       # simulator
./ios/build.sh device    # physical device (needs a signing identity)
./ios/build.sh --help
```

Extra arguments after the mode are forwarded to `xcodebuild`, e.g.
`./ios/build.sh sim -derivedDataPath /tmp/pipette-dd`.

### Lower level: invoking xcodebuild directly

If you need flags `build.sh` doesn't expose, call `xcodebuild` yourself. This
is essentially what the wrapper runs:

```bash
xcodebuild build \
  -project ios/Pipette/Pipette.xcodeproj \
  -scheme Pipette \
  -destination 'generic/platform=iOS Simulator' \
  -skipPackagePluginValidation \
  ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO
```

`-skipPackagePluginValidation` lets the `LicenseList` build-tool plugin
(`PrepareLicenseList`) run without interactive trust approval. A headless
`xcodebuild` has no way to answer the "Trust & Enable" prompt Xcode's GUI shows,
so the build otherwise fails at `Validate plug-in "PrepareLicenseList"`.

### Building the llama.cpp library on its own

You only need `build-llama.sh` directly to warm caches or populate the `llama`
module *without* a full app build; e.g. so your editor/LSP resolves `import
llama`, or to pre-compile under `sccache` before a clean Xcode build:

```bash
./ios/build-llama.sh sim      # simulator slice (CPU, Metal off)
./ios/build-llama.sh device   # device slice (arm64, Metal on)
./ios/build-llama.sh          # auto-detect from Xcode's PLATFORM_NAME, else device
./ios/build-llama.sh --help
```

When the build phase runs it with no argument, the mode is auto-detected from
Xcode's `PLATFORM_NAME` (simulator → `sim`, device → `device`), so exactly the
slice being built is compiled. Set `PIPETTE_SKIP_BUILD_LLAMA=1` on the
`xcodebuild` step to skip a redundant rebuild when `Generated/Llama/` is already
populated (e.g. a CI pre-build step).

### The Settings Debugging card

Settings ends with a **Debugging** card: client id, collector approval `Status`,
Clerk user/session ids, the data-root path, storage counts. It is compiled in only
when `DEBUG` or `PIPETTE_DEBUG_UI` is an active Swift condition, so an App Store
archive does not contain it at all. This mirrors Android, where the same block sits
behind `if (state.isDebug)` in `SettingsScreen.kt`.

| Build | Card |
|---|---|
| Local Debug (`-configuration Debug`) | shown (`DEBUG`) |
| Local Release | hidden; pass `--debug-ui` to get it |
| Xcode Cloud **Internal Testing** | shown, via `PIPETTE_DEBUG_UI=1` |
| Xcode Cloud **App Store** | hidden, and not in the binary |

Locally:

```bash
./ios/build.sh device --debug-ui
```

On Xcode Cloud, set **`PIPETTE_DEBUG_UI=1`** as an environment variable on the
Internal Testing workflow (workflow → Environment → Environment Variables) and leave
it unset on the App Store workflow. Nothing else needs changing: no extra build
configuration, and no edit to the workflow's Archive action.

> [!NOTE]
> Why a variable alone isn't enough. `xcodebuild` does **not** promote shell
> environment variables into build settings, so `PIPETTE_DEBUG_UI=1` cannot become a
> `#if` by itself, and because Xcode Cloud owns the `xcodebuild` invocation, we
> can't pass `SWIFT_ACTIVE_COMPILATION_CONDITIONS=…` on its command line the way
> `ios/build.sh` does. `ci_scripts/ci_pre_xcodebuild.sh` bridges the gap: it reads the
> variable and writes `Config/Distribution.generated.xcconfig`, which
> `Release.xcconfig` pulls in with `#include?`. The optional include is what lets the
> file be absent for every other build without failing. The generated file is
> gitignored: never commit or hand-edit it.
>
> Do not reuse `BuildFlavor.isInternal` for this. It means `PIPETTE_PRIVATE_THERMAL`,
> which [must never reach TestFlight](private-thermal-release-build.md), so it is
> `false` in exactly the Internal Testing builds that should show the card.

## Building from the Xcode GUI

1. Open `ios/Pipette/Pipette.xcodeproj`.
2. One-time: if you haven't installed the Metal toolchain, do so via
   **Xcode → Settings → Components** (see [Prerequisites](#prerequisites)).
3. Select the **Pipette** scheme and a simulator (or your connected device) as
   the run destination.
4. Build/Run (⌘R). The first build prompts to trust the `LicenseList` package
   plugin: click **Trust & Enable**. The approval persists, so this is a
   one-time step.

The "Build llama.cpp" phase runs `build-llama.sh` as part of the build, so the
static library and `llama` module are produced automatically. There is no
separate step.

## Clerk configuration

The app gates its UI behind a [Clerk](https://clerk.com) sign-in (see
[architecture.md](architecture.md#authentication-clerk) for the role it plays),
so these two settings must be present or the app shows a configuration-error
screen at launch instead of running. They are compiled into `Info.plist` and the
associated-domains entitlement (`webcredentials:$(CLERK_FRONTEND_API_DOMAIN)`).
Both iOS build configurations use the live Clerk app from
`ios/Pipette/Config/Shared.xcconfig`:

```xcconfig
CLERK_PUBLISHABLE_KEY = pk_live_Y2xlcmsubGlxdWlkLmFpJA
CLERK_FRONTEND_API_DOMAIN = clerk.liquid.ai
```

`CLERK_PUBLISHABLE_KEY` is a *publishable* (client-side) key, not a secret:
it's designed to ship inside the app binary, which is why it's committed here
rather than injected from a secret store.

For a TestFlight or App Store iPhone archive:

```bash
xcodebuild archive \
  -project ios/Pipette/Pipette.xcodeproj \
  -scheme Pipette \
  -destination 'generic/platform=iOS' \
  -archivePath /tmp/Pipette.xcarchive \
  -skipPackagePluginValidation \
  -allowProvisioningUpdates
```

## Running in the simulator from the CLI

The bundle identifier is `ai.liquid.liquid-pipette`. The full loop (boot a sim,
build, install, launch, observe) without ever opening Xcode:

```bash
# 1. Pick + boot a simulator. `booted` in later commands targets whichever
#    sim is currently running, so you only need the UDID here.
xcrun simctl list devices available | grep iPhone
xcrun simctl boot "iPhone 17 Pro"
open -a Simulator                     # show the sim window

# 2. Build to a known DerivedData path so the .app is easy to find.
./ios/build.sh sim -derivedDataPath /tmp/pipette-dd

# 3. Install + launch.
xcrun simctl install booted /tmp/pipette-dd/Build/Products/Release-iphonesimulator/Pipette.app
xcrun simctl launch booted ai.liquid.liquid-pipette    # prints the launched pid

# 4. Observe.
xcrun simctl io booted screenshot /tmp/pipette.png
xcrun simctl spawn booted log stream --level=debug --style=compact \
  --predicate 'processImagePath CONTAINS "Pipette"'

# 5. Reset between runs.
xcrun simctl terminate booted ai.liquid.liquid-pipette
xcrun simctl uninstall booted ai.liquid.liquid-pipette   # also clears app data
```

`build.sh sim` builds an arm64-only, unsigned `.app` (it passes
`CODE_SIGNING_ALLOWED=NO`, safe for the simulator only) that `simctl install`
accepts. The simulator slice builds llama.cpp **CPU-only** (no Metal backend in
the simulator), so it's for functional/UI work: not benchmark numbers. Device
builds still need a real signing identity, which is why `build.sh device` omits
the no-signing flag.

## Running on a physical device from the CLI

`build.sh device` produces a *signed* `.app` but does not deploy it: a `build`
action stops at DerivedData. Use `xcrun devicectl` (Xcode 15+) to install and
launch it on a connected device.

> **The Xcode GUI is required for device signing. There is no CLI workaround.**
> A device build only succeeds once a development team is selected for the target
> and the target device is registered in a provisioning profile. That setup is
> interactive: it happens when you open the project in Xcode, pick a team under
> **Signing & Capabilities**, and trust the connected device. The CLI
> (`build.sh`, `xcodebuild -allowProvisioningUpdates`) can *use* signing once
> it's configured, but it cannot bootstrap it. The simulator path has no such
> requirement.

```bash
# 1. Find the device UDID (the column labelled "Identifier").
xcrun devicectl list devices

# 2. Build to a known DerivedData path so the .app is easy to find.
./ios/build.sh device -derivedDataPath /tmp/pipette-dd

# 3. Install + launch on the device. The path segment matches the build
#    configuration — Release by default, Debug-iphoneos if you pass
#    `-configuration Debug` above.
xcrun devicectl device install app --device <udid> \
  /tmp/pipette-dd/Build/Products/Release-iphoneos/Pipette.app
xcrun devicectl device process launch --device <udid> ai.liquid.liquid-pipette

# 4. Uninstall (clears app data too).
xcrun devicectl device uninstall app --device <udid> ai.liquid.liquid-pipette
```

For a TestFlight / App Store build instead, archive and export an `.ipa`. See
[Clerk configuration](#clerk-configuration) above for the `xcodebuild archive`
invocation.

## Common pitfalls

- **`No such module 'llama'`**: the "Build llama.cpp" phase hasn't populated
  `Generated/Llama/` (or its module map / search paths are missing). Run
  `./ios/build-llama.sh sim` (or `device`) once, or do a full clean build.
- **x86_64 linker errors**: `libllama.a` is built arm64-only. Use
  `ONLY_ACTIVE_ARCH=YES` or target a specific arm64 simulator device to avoid
  x86_64 link failures.
- **`Generated/Llama/` directory**: auto-generated; do not edit manually. It is
  overwritten by `build-llama.sh`.
- **Patch series**: `build-llama.sh` applies `ios/patches/NNN-*.patch` (in numeric
  order) to the build copy; currently `001-ggml-metal-oom-nullcheck.patch`, a
  null-check after `ggml_metal_buffer_init`. If a llama.cpp bump moves that code,
  the apply fails loudly rather than silently shipping the unsafe path: refresh
  the patch for the new commit (ideally upstream the fix).

## Bumping llama.cpp

1. Update the `vendor/llama.cpp` submodule to the new commit (shared with Android).
2. No source list to maintain: llama.cpp's CMake owns the source list and
   compile flags, so a bump tracks upstream automatically. If upstream changes a
   `GGML_*` / `LLAMA_*` option name, update the inline `-D` flags in
   `ios/build-llama.sh`.
3. Re-verify the `ios/patches/NNN-*.patch` series still applies (see pitfalls
   above).
4. Build on device and re-run the benchmark parity check before relying on
   numbers.

## Installing a released archive

Every release carries `pipette-ios-<version>-internal.xcarchive.zip`: a Release build for
arm64 devices, built by the `ios-archive` CI job. That job runs on every CI run, not
only releases, so any commit (including one under review in a PR) has a downloadable
build attached to its run; the version carries the short sha, so you can tell which. It unpacks to a
`pipette-ios-<version>-internal/` folder holding the `.xcarchive` and an
`INTERNAL-BUILD.md` restating the two caveats below.

Release on both halves: `build-llama.sh` pins `-DCMAKE_BUILD_TYPE=Release` for the
static lib and the job passes `-configuration Release` for the Swift side, so the
numbers it produces are not an unoptimized build's.

**It is an internal, private-thermal build (`-internal`).** The archive is built *with*
`PIPETTE_PRIVATE_THERMAL`, so the readiness gate waits on a real SoC die temperature
and `device_apple_soc_temp_c_*` is populated, which is what makes its numbers
comparable to the rest of the fleet. The binary states this about itself: the version
carries `-internal`, so `client_version` on every submitted row records how the run was
gated. Confirm what a device is running with `headlessrun version`.

> [!CAUTION]
> Never submit this archive to TestFlight or the App Store: App Review rejects the
> private API it reads. Side-load onto registered development devices only; see
> [private-thermal-release-build.md](private-thermal-release-build.md).

**It is unsigned, deliberately.** Signing in CI would mean keeping a certificate and
a provisioning profile in repository secrets, and it would freeze the set of devices
that can run the build: a profile lists device UDIDs at *build* time, so a phone added
to the fleet afterwards could never install that asset, and the published file would
carry every registered device's identifier. An unsigned archive has neither problem.
you sign it with your own identity at install time, so it stays installable on devices
that did not exist when it was built, and it never expires as an artifact.

Two things about this app make signing more than a single `codesign` call, and both
will fail the install if skipped:

- It embeds **nested code** (`Frameworks/Sentry.framework`), which has to be signed
  before the bundle that contains it.
- An unsigned archive has **no `embedded.mobileprovision`**. `codesign` does not add
  one, and iOS will not install a development-signed app without it.

```bash
# 1. Unpack (yields pipette-ios-<version>-internal/ with the archive and the note).
unzip -q pipette-ios-<version>-internal.xcarchive.zip
cd pipette-ios-<version>-internal
APP="Pipette.xcarchive/Products/Applications/Pipette.app"

# 2. Pick a profile that covers the target device and this bundle id. Xcode leaves the
#    ones it has fetched here; `-allowProvisioningUpdates` on any device build creates one.
#    `-a` matters: a .mobileprovision is a CMS blob, and grep skips it as binary without it.
PROF=$(grep -a -l ai.liquid.liquid-pipette \
  ~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/*.mobileprovision | head -1)

# 3. Take the entitlements from the *profile*, not from the repo's .entitlements file —
#    they must match what the profile grants or the install is rejected.
security cms -D -i "$PROF" > /tmp/prof.plist
/usr/libexec/PlistBuddy -x -c 'Print :Entitlements' /tmp/prof.plist > /tmp/ent.plist

# 4. Embed the profile, then sign nested code first and the app last.
cp "$PROF" "$APP/embedded.mobileprovision"
codesign --force --sign "Apple Development: You (TEAMID)" "$APP/Frameworks/Sentry.framework"
codesign --force --sign "Apple Development: You (TEAMID)" \
  --entitlements /tmp/ent.plist --generate-entitlement-der "$APP"
codesign --verify --deep --strict "$APP"     # silence means valid

# 5. Install.
xcrun devicectl list devices                 # find the identifier
xcrun devicectl device install app --device <device-id> "$APP"
```

If you have the team account signed into Xcode, the supported route is easier and
does all of the above for you: open the `.xcarchive` (Window → Organizer → Archives)
and run **Distribute App → Development**, or from the CLI:

```bash
xcodebuild -exportArchive -archivePath Pipette.xcarchive \
  -exportPath out -exportOptionsPlist export.plist -allowProvisioningUpdates
```

with `method: development` and your `teamID` in `export.plist`. It needs an Xcode
*account* for the team, not just a certificate in the keychain: without one it fails
with `No Account for Team`, which is when the manual steps above are the way through.

Sanity checks on an unpacked archive:

```bash
lipo -archs Pipette.xcarchive/Products/Applications/Pipette.app/Pipette   # arm64
vtool -show-build-version .../Pipette | grep platform                    # platform IOS
```

`platform IOS` (not `IOSSIMULATOR`) is what distinguishes an installable build
from a simulator one.
