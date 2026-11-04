# Building the Pipette iOS App

How to compile the vendored llama.cpp into the app, and build/run it. For the
architecture behind these artifacts (the native Swift engines, vendored
llama.cpp, core concepts), see [architecture.md](architecture.md).

> The iOS app is **pure Swift**. There is no Rust in the iOS build. llama.cpp is
> built into a framework the app embeds, from the shared `vendor/llama.cpp`
> submodule (the same one Android builds), by `ios/build-llama.sh`.

## How to build

There are two supported ways to build Pipette. Use either one:

- **`ios/build.sh`**: the command-line entry point, for CI, scripted
  loops, and simulator work. One command picks the build destination, sets the
  right flags, checks your toolchain, and builds the app.
- **The Xcode GUI**: open `ios/Pipette/Pipette.xcodeproj`, pick a destination,
  and Run (⌘R), for interactive development. It is **required at least once
  to deploy to a physical device** (see below).

Both produce the same app. See
[Building from the command line](#building-from-the-command-line) or
[Building from the Xcode GUI](#building-from-the-xcode-gui).

### Deploying to a device needs the GUI once

**Deploying to a physical device requires the Xcode GUI at least once** to set
up code signing: select a development team and register the device. After that
the command line can build and install for the device, but it cannot bootstrap
signing on its own. The simulator path is fully command-line-only.

### What each path needs from you

A handful of one-time setup items differ between the two paths. (The llama.cpp
framework, vendored sources, and the `llama` module are *not* in this
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
  `build-onnx-genai.sh` drives both ONNX trees' CMake builds with the Xcode
  generator, which their iOS toolchains require.
- **uv** (`brew install uv`). ONNX Runtime's build scripts are Python;
  `build-onnx-genai.sh` keeps their environment inside the build tree rather than
  in yours, and uses `uv` to create it.
- **The Metal toolchain.** Xcode 16+ unbundles the Metal compiler into a
  separate, on-demand component. Without it the `mlx-swift` dependency fails
  with a wall of `CompileMetalFile` errors (the real message is `cannot execute
  tool 'metal' due to missing Metal Toolchain`).
  - CLI: `xcodebuild -downloadComponent MetalToolchain`
  - GUI: **Xcode → Settings → Components**, install "Metal Toolchain"
  - Verify either way: `xcodebuild -showComponent MetalToolchain` → `Status: installed`

  > CI runners ship the Metal toolchain preinstalled, so this affects only local
  > machines with a fresh Xcode.

### Handled automatically (both paths)

The build initializes these on every run (GUI or command line), so you normally
don't touch them. Listed here for when you build llama.cpp by hand
or need to debug a failure:

- **Vendored llama.cpp.** The shared `vendor/llama.cpp` submodule (also built by
  Android). `build-llama.sh` initializes it if missing, then builds from a copy
  so the submodule itself is never modified.
- **Vendored ONNX Runtime and ONNX Runtime GenAI.** `vendor/onnxruntime` and
  `vendor/onnxruntime-genai`. `build-onnx-genai.sh` initializes both recursively,
  because onnxruntime has nested submodules of its own, then builds genai from a
  patched copy. The engine tree is large; a first checkout pulls roughly a
  gigabyte.

The full submodule list and how to check them out are in
[CONTRIBUTING.md](../../CONTRIBUTING.md#getting-started).

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
  Pipette target
  ├─ "Build llama.cpp" phase ──────────→ ios/build-llama.sh       (llamacpp.framework + `llama` module)
  ├─ "Build ONNX Runtime GenAI" phase ─→ ios/build-onnx-genai.sh  (libonnxruntime{,-genai}.a + headers)
  ├─ "Generate MLX build info" phase ──→ ios/gen-mlx-build-info.sh
  ├─ compiles Swift, links, copies resources
  ├─ "Embed llama.cpp" phase ──────────→ ios/embed-llama.sh       (Frameworks/<pin>.framework)
  ├─ "Upload Sentry dSYMs" phase ──────→ sentry-cli, Release only, skipped without it or a token
  └─ "Stamp git commit" phase ─────────→ ios/stamp-git-commit.sh  (built Info.plist)
  PipetteTests target (test builds only: xcodebuild test or build-for-testing, or Cmd-U)
  ├─ "Check contract variants" phase ──→ ios/gen-contract-variants.py
  └─ "Copy the ONNX GenAI test fixture" → ios/copy-onnx-test-fixture.sh
```

The order matters (the `llama` module must exist before the Swift that imports
it compiles), and the phase guarantees it. What the "Build llama.cpp" phase does:

1. Syncs a build **copy** of the `vendor/llama.cpp` submodule (so the submodule
   stays pristine), and applies our carried patch series (`ios/patches/NNN-*.patch`,
   in numeric order) to the **copy**. The build fails if a patch does not apply
   (see pitfalls).
2. Configures + builds the copy through **llama.cpp's own CMake**, with inline
   `-D` flags (Metal + embedded shader library on device, CPU on the simulator;
   Accelerate): exactly like upstream `build-xcframework.sh`. The Android
   repo-root CMake (`CMakeLists.txt` / `CMakePresets.json`) is **not** involved:
   iOS configures the vendored tree directly. CMake owns the source list, the
   per-target compile flags, and the `GGML_METAL_EMBED_LIBRARY` metallib embed:
   nothing is hand-transcribed.
3. Links every static library that CMake produced, with the `ios/native`
   veneers, into one dynamic library. The library exports one symbol,
   `pipette_llama_table` (`ios/native/pipette_llama.h`). The script stages the
   library and its dSYM as
   `ios/.build-llama/frameworks/<iphoneos|iphonesimulator>/llamacpp.framework`.
   Its `Info.plist` names the repository, the commit, and the tag when the pin
   has one (see [architecture.md](architecture.md#2-how-the-ios-app-uses-llamacpp)
   for how the app reads them back). Then writes `Generated/Llama/{module.modulemap, *.h}`.
4. Copies the shared benchmark prompt corpus
   (`crates/pipette-ops/src/prompt_seed.txt`) into
   `Generated/PromptSeed/prompt_seed.txt`, bundled as an app resource so the
   native llama benchmarks tokenize the same text as the Rust CLIs / Android.

The Swift code then does `import llama`. The app links no llama.cpp:
`Native/PipetteLlama.c` defines each C call Swift makes as a forwarder to the
framework the process loaded ([llama.cpp variants](#llamacpp-variants)).

The phase also honors the `sccache` opt-in. For *how* the native Swift engines fit together, see
[architecture.md](architecture.md).

The "Embed llama.cpp" phase (`ios/embed-llama.sh`) copies each pin that
`PIPETTE_LLAMA_PINS` names from that staging directory into the app's
`Frameworks/`. The setting is a space-separated list, and `llamacpp` is the
default. In a `dwarf-with-dsym` build, the phase also copies the dSYMs. If the
build signs, the phase signs each framework. The build fails if a named pin was
not staged. It also fails if the pin was built against another revision of
`ios/native/pipette_llama.h`; rebuild that pin with `ios/build-llama.sh`.

## Building from the command line

`ios/build.sh` builds the whole app in one command. It selects the destination,
passes the required `xcodebuild` flags, pre-flights the Metal toolchain, and
triggers the llama.cpp build automatically.

The mode is `sim` (the default) or `device` (needs a signing identity);
`./ios/build.sh --help` lists the modes and its own flags. Extra arguments after
the mode are forwarded to `xcodebuild`, e.g.
`./ios/build.sh sim -derivedDataPath /tmp/pipette-dd`.

### Lower level: invoking xcodebuild directly

If you need flags `build.sh` doesn't expose, call `xcodebuild` yourself. This
is essentially what the wrapper runs:

```bash
xcodebuild build \
  -project ios/Pipette/Pipette.xcodeproj \
  -scheme Pipette \
  -destination 'generic/platform=iOS Simulator' \
  -configuration Release \
  -skipPackagePluginValidation -skipMacroValidation \
  ARCHS=arm64 ONLY_ACTIVE_ARCH=YES CODE_SIGNING_ALLOWED=NO
```

`-skipPackagePluginValidation` lets the `LicenseList` build-tool plugin
(`PrepareLicenseList`) run without interactive trust approval. A headless
`xcodebuild` has no way to answer the "Trust & Enable" prompt Xcode's GUI shows,
so the build otherwise fails at `Validate plug-in "PrepareLicenseList"`.
`-skipMacroValidation` does the same for the swift-syntax macros.

### Building the ONNX Runtime GenAI libraries on their own

Same shape as the llama.cpp script below (`sim`, `device`, or no argument to
auto-detect; see `./ios/build-onnx-genai.sh --help`), and the same reasons to
run it by hand.

Both slices write the same filenames into `Generated/OnnxGenai/`, so the script
stamps which one is published and rebuilds only when the slice or a submodule pin
changes. Switching between a device and a simulator build therefore costs a full
rebuild once and nothing thereafter. `PIPETTE_SKIP_BUILD_ONNX_GENAI=1` skips the
phase outright, as its llama.cpp counterpart does.

A first build compiles both source trees and takes several minutes. What it does,
and the three non-default flags it needs, is
[onnx-genai-runtime.md](onnx-genai-runtime.md#build-pipeline).

### Building a llama.cpp framework on its own

You only need `build-llama.sh` directly to build a [variant](#llamacpp-variants),
to warm caches, or to populate the `llama` module *without* a full app build; e.g.
so your editor/LSP resolves `import llama`, or to pre-compile under `sccache`
before a clean Xcode build. Pass `sim` (CPU, Metal off) or `device` (Metal on);
`./ios/build-llama.sh --help` has the rest.

When the build phase runs it with no argument, the mode is auto-detected from
Xcode's `PLATFORM_NAME` (simulator → `sim`, device → `device`), so exactly the
slice being built is compiled. Set `PIPETTE_SKIP_BUILD_LLAMA=1` on the
`xcodebuild` step to skip a redundant rebuild when the frameworks are already
staged and `Generated/Llama/` populated (e.g. a CI pre-build step).

### llama.cpp variants

Some models need a llama.cpp other than the `vendor/llama.cpp` pin. Each
`vendor/llamacpp-<name>` submodule pins one more for iOS only, built into a
framework of its own, `llamacpp-<name>.framework`, that the
[private-API build](private-thermal-release-build.md) carries beside main's
`llamacpp.framework`. There are two:

| Variant | Repository | Pin | For |
|---|---|---|---|
| `prismml` | `github.com/PrismML-Eng/llama.cpp` | tag `prism-b10777-e3de715` | PrismML's 1-bit and ternary formats |
| `lfm3` | `github.com/ggml-org/llama.cpp` | commit `418f905a4efe910d2a75636bbe65d48eca38790a`, the head of its `master` branch, which has no tag (after `b11986`) | the LFM3 architecture, which upstream added after the main pin (#31252, #31257) |

A process that runs cells loads one pin: the one that a headless run's
`runtime=` names, or main if the run names none
([architecture.md](architecture.md#2-how-the-ios-app-uses-llamacpp)). The UI and
the planner worker run main.

Only the private-API build carries variants. GitHub's `ios-archive` job builds the
framework for every pin and archives one `-internal` app that carries all of
them, for the GitHub release. Xcode
Cloud (TestFlight and the App Store) builds only `vendor/llama.cpp`, so its app
carries only main. Its `ci_pre_xcodebuild.sh` fails a workflow that sets
`PIPETTE_LLAMA_VARIANT` or `PIPETTE_LLAMA_PINS`.

- `<name>` is `[a-z0-9][a-z0-9.-]*`, and the pin is `llamacpp-<name>`. The app's
  `client_version` names no pin: a result names its llama.cpp in its runtime
  descriptor.
- The submodule's URL in `.gitmodules` may name a fork; the pin's runtime
  descriptor then names that repository.
- `vendor/llamacpp-<name>.tag` records its tag as `<tag> <full-sha>`, the same form
  as `vendor/llama.cpp.tag`, in any tag scheme. The first field is what the
  descriptor carries as `repository_version`. A pin with no tag records its
  commit in both fields, as `lfm3` does, so the descriptor names the full commit.
  An untagged commit stays fetchable only while a branch contains it, so a
  force-push or a deletion upstream can fail the `ios-archive` checkout.
- The patch series is `ios/patches/llamacpp-<name>/` when that directory exists,
  else the main `ios/patches/` series. Both variants carry their own series.
  `prismml` sits on an upstream base older than the main pin's and patches it
  differently. `lfm3` sits on a base later than the main pin's. Upstream
  `ggml-org/llama.cpp#31321` changed the no-mmap weight load there, so its `003`
  is rebased onto it. Its `002` is the main one with its hunk offsets moved.
- Swift compiles once, against main's headers. It calls the loaded pin through
  the table that the pin exports, and each pin fills its table from its own
  headers (`ios/native/pipette_llama_table.cpp`). If the prototypes,
  passed-through records or constants of a pin differ from main's, that file
  fails to compile. The patch series of the pin can adapt the fork.
- Two veneers build against more than one llama.cpp API:
  - `ios/native/pipette_vlspec.cpp` builds against either batch API.
    `build-llama.sh` checks the tree for upstream's `common_batch`; a tree
    without it (before `b11236`) takes the veneer's `llama_batch` path.
  - `ios/native/pipette_chat.cpp` needs no such check. It uses only `common/chat.h`
    declarations that every vendored variant shares. The one exception is the
    rendered prompt: a `std::string` before upstream `ggml-org/llama.cpp#30389`
    and a `common_chat_prompt` after it. Its `prompt_text` reads either form.

  A variant that changes another of these declarations fails that file's
  compile in its build step.
- When a variant does not build, or its pin cannot be fetched, `ios-archive`
  fails. With it fail `build-complete` and the whole release (the CLI, the
  Android apps, the proxy and the iOS app), until the pin is bumped or removed.

To add one:

1. Add the submodule (shallow, at the commit) and its `.tag` file.
2. Build its framework, then an app that carries it (`sim` for the simulator):
   `ios/build.sh` passes the `PIPETTE_LLAMA_PINS` setting through to `xcodebuild`.

   ```bash
   PIPETTE_LLAMA_VARIANT=<name> ./ios/build-llama.sh device
   ./ios/build.sh device 'PIPETTE_LLAMA_PINS=llamacpp llamacpp-<name>'
   ```

3. In the `ios-archive` job in `.github/workflows/build.yml`:
   - Add `vendor/llamacpp-<name>` to the checkout step's `submodule update`.
   - Add a `Build llama.cpp (<name>)` step that sets `PIPETTE_LLAMA_VARIANT: <name>`.
   - Add `llamacpp-<name>` to the archive step's `PIPETTE_LLAMA_PINS`.

`ci/checks/llamacpp-variants.py` checks that each variant in `.gitmodules`
agrees with:

- its `.tag` file, which must name the commit that the submodule is pinned at
- the checkout step's paths
- the build steps
- `PIPETTE_LLAMA_PINS`

The check fails until they agree.

To remove one, as when its pin can no longer be fetched:

1. In the `ios-archive` job in `.github/workflows/build.yml`:
   - Delete its path from the checkout step.
   - Delete its build step.
   - Delete its `PIPETTE_LLAMA_PINS` entry.
2. Run `git rm vendor/llamacpp-<name>`, which drops its `.gitmodules` entry but
   not a comment above it, and delete `vendor/llamacpp-<name>.tag`,
   `ios/patches/llamacpp-<name>/` if it has one, and its row in the table above.

`ci/checks/llamacpp-variants.py` fails while the submodules and `ios-archive`
disagree, so both steps go in one commit.

On the farm, pipette-ios-proxy picks a build whose manifest carries the job's
runtime. It passes that runtime to the app, and the app loads the pin that the
runtime names. A plan reaches a variant when it names the llama.cpp of that
variant in the runtime ([DESIGN.md section 9](../../tools/pipette-ios-proxy/DESIGN.md#9-version-mode-the-build-a-job-pins)).
`headlessrun runtimes list` loads every pin and prints a row for each, so the
proxy's check after an install proves each pin loads on that phone.

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
> Do not reuse `BuildFlavor.isInternal` for this. It means a private-API build
> (`PIPETTE_PRIVATE_THERMAL` or `PIPETTE_PRIVATE_MEMORY`), which
> [must never reach TestFlight](private-thermal-release-build.md), so it is
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

The "Build llama.cpp" phase runs `build-llama.sh` as part of the build, so main's
framework and the `llama` module are produced automatically. There is no
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

## Archiving for TestFlight and the App Store

The scheme archives the Release configuration:

```bash
xcodebuild archive \
  -project ios/Pipette/Pipette.xcodeproj \
  -scheme Pipette \
  -destination 'generic/platform=iOS' \
  -archivePath /tmp/Pipette.xcarchive \
  -skipPackagePluginValidation -skipMacroValidation \
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

This needs the one-time signing setup in the Xcode GUI first; see
[Deploying to a device needs the GUI once](#deploying-to-a-device-needs-the-gui-once).

```bash
# 1. Find the device UDID (the column labeled "Identifier").
xcrun devicectl list devices

# 2. Build to a known DerivedData path so the .app is easy to find.
./ios/build.sh device -derivedDataPath /tmp/pipette-dd

# 3. Install + launch on the device. The path segment matches the build
#    configuration: Release by default, Debug-iphoneos if you pass
#    `-configuration Debug` above.
xcrun devicectl device install app --device <udid> \
  /tmp/pipette-dd/Build/Products/Release-iphoneos/Pipette.app
xcrun devicectl device process launch --device <udid> ai.liquid.liquid-pipette

# 4. Uninstall (clears app data too).
xcrun devicectl device uninstall app --device <udid> ai.liquid.liquid-pipette
```

For a TestFlight or App Store build, archive and export an `.ipa` instead. See
[Archiving for TestFlight and the App Store](#archiving-for-testflight-and-the-app-store)
for the `xcodebuild archive` command.

## Common pitfalls

- **`No such module 'llama'`**: the "Build llama.cpp" phase hasn't populated
  `Generated/Llama/` (or its module map / search paths are missing). Run
  `./ios/build-llama.sh sim` (or `device`) once, or do a full clean build.
- **x86_64 linker errors**: the vendored engines are built arm64-only. Use
  `ONLY_ACTIVE_ARCH=YES` or target a specific arm64 simulator device to avoid
  x86_64 link failures.
- **`Generated/Llama/` directory**: auto-generated; do not edit manually. It is
  overwritten by `build-llama.sh`.
- **Patch series**: `build-llama.sh` applies `ios/patches/NNN-*.patch` (in numeric
  order, with no fuzz) to the build copy; what each patch fixes is in
  [memory-carry-over.md](memory-carry-over.md). If a llama.cpp bump moves the
  patched code, the build fails at the apply step. Refresh the patch for the new
  commit, or retire it if the fix is upstream (send the fix upstream where you can).

## Bumping ONNX Runtime GenAI

The two submodules are versioned together: genai names the ONNX Runtime version it
builds against, so move both or neither.
[onnx-genai-runtime.md](onnx-genai-runtime.md#bumping-the-pins) has the steps and
the one behavioral thing to re-check: whether `eval` has become supportable.

## Bumping llama.cpp

1. Update the `vendor/llama.cpp` submodule to the new commit (shared with Android)
   and record it in `vendor/llama.cpp.tag` as `<tag> <full-sha>`. The submodule is
   a shallow clone with no tags, so that file is where the iOS stamp and the
   Android build read the release name from; a stale one reports the build untagged.
2. No source list to maintain: llama.cpp's CMake owns the source list and
   compile flags, so a bump tracks upstream automatically. If upstream changes a
   `GGML_*` / `LLAMA_*` option name, update the inline `-D` flags in
   `ios/build-llama.sh`.
3. Re-verify the `ios/patches/NNN-*.patch` series still applies (see pitfalls
   above).
4. A new tag changes what the app advertises and reports. A cell that names the
   old `repository_version` then fares differently by path:
   - the app's own worker (`headlessrun worker`) is no longer offered a plan that
     requires `runtime:llama_cpp:<old tag>`; a cell it does claim that names the old
     version runs, logs the mismatch and records the build that ran;
   - `pipette-ios-proxy` runs it only on a build that still carries it
     ([DESIGN.md section 9](../../tools/pipette-ios-proxy/DESIGN.md#9-version-mode-the-build-a-job-pins));
   - a headless run given another build's `runtime=` (a local plan) is refused.
5. Build on device and re-run the benchmark parity check before relying on
   numbers.

## Installing a released archive

Every release carries `pipette-ios-<version>-internal.xcarchive.zip`: a Release build for
arm64 devices, built by the `ios-archive` CI job. It is the
[private-API build](private-thermal-release-build.md). The job runs on every push to
`main` or a `release/` branch, and on a pull request that changes the iOS app, so
each such commit has a downloadable build attached to its run. The version carries
the short sha, so you can tell which commit built it. The zip unpacks to a
`pipette-ios-<version>-internal/` folder that holds the `.xcarchive` and an
`INTERNAL-BUILD.md` that restates the two caveats below. The run and the release
also carry `pipette-ios-<version>-internal.json` beside the zip: the manifest of
the runtimes the build carries, one `llamacpp_ios_pipette` runtime per
[llama.cpp pin](#llamacpp-variants) among them. The farm's phones get these builds without the
steps below; see [On the farm](#on-the-farm-pipette-ios-proxy).

Release on both halves: `build-llama.sh` pins `-DCMAKE_BUILD_TYPE=Release` for the
llama.cpp frameworks and the job passes `-configuration Release` for the Swift side, so the
numbers it produces are not an unoptimized build's.

**It is the private-API build (`-internal`).** The archive is built with
`PIPETTE_PRIVATE_THERMAL` and `PIPETTE_PRIVATE_MEMORY`, which is what makes its
numbers comparable to the rest of the fleet; what each one changes is in
[private-thermal-release-build.md](private-thermal-release-build.md). The version
carries `-internal`, so `client_version` on every submitted row records how the
run was gated. Confirm what a device is running with `headlessrun version`.

> [!CAUTION]
> Never submit this archive to TestFlight or the App Store: App Review rejects the
> private APIs it reads. Side-load onto registered development devices only; see
> [private-thermal-release-build.md](private-thermal-release-build.md).

**It is not signed for any device, deliberately.** Signing in CI has two problems:

- CI would have to keep a certificate and a provisioning profile in repository secrets.
- A profile lists device UDIDs at *build* time. A phone added to the fleet later could
  never install that asset, and the published file would carry the identifier of every
  registered device.

You sign an archive that is signed for no device with your own identity at install
time. It stays installable on devices added after the build, and it never expires as
an artifact.

The app does carry an **ad-hoc signature** with its own entitlements
(`ios/adhoc-sign-archive.sh`, run by the same job). It needs no certificate and names no
device, and it is what lets a re-sign that keeps the binary's entitlements, as
`xcodebuild -exportArchive` does, keep `increased-memory-limit`. Without it the exported
app gets iOS's default per-app memory ceiling. The script also adds
`increased-debugging-memory-limit` to these private-API builds: on a 12 GB iPhone it raises the
per-app limit from 6144 to 6656 MiB, which is what lets LFM2.5-8B-A1B Q5_K_M run at 4096
tokens. `Pipette.entitlements` leaves it out because App Store builds may not carry it.
Check a signed bundle with `codesign -d --entitlements - --xml Pipette.app`.

Two things about this app make signing more than a single `codesign` call, and both
will fail the install if skipped:

- It embeds **nested code** (`Frameworks/Sentry.framework` and a framework per
  llama.cpp pin), which has to be signed before the bundle that contains it.
- The archive has **no `embedded.mobileprovision`**. `codesign` does not add
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

# 3. Take the entitlements from the *profile*, not from the repo's .entitlements file:
#    they must match what the profile grants or the install is rejected.
security cms -D -i "$PROF" > /tmp/prof.plist
/usr/libexec/PlistBuddy -x -c 'Print :Entitlements' /tmp/prof.plist > /tmp/ent.plist

# 4. Embed the profile, then sign nested code first and the app last.
cp "$PROF" "$APP/embedded.mobileprovision"
for fw in "$APP"/Frameworks/*.framework; do
  codesign --force --sign "Apple Development: You (TEAMID)" "$fw"
done
codesign --force --sign "Apple Development: You (TEAMID)" \
  --entitlements /tmp/ent.plist --generate-entitlement-der "$APP"
codesign --verify --deep --strict "$APP"     # no output means valid

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

### On the farm: `pipette-ios-proxy`

The farm's iPhones are driven by
[`pipette-ios-proxy`](../../tools/pipette-ios-proxy/README.md) on the Mac they are
tethered to. In version mode it installs the release build each job runs on, so
nobody signs or installs by hand.

A release publishes the archive, its manifest, and a `.sha256` beside each as
GitHub release assets ([Releasing](../../CONTRIBUTING.md#releasing)). The proxy
reads them from a releases bucket, `s3://<bucket>/<tag>/`, that the device
lab's deployment fills from the release. `ios/release-manifest.py` writes
the manifest from what the archive compiled, so the proxy can match a job's
runtime to a build before it downloads anything. A plan pins a build with
`client_versions`
([Pinning an iOS app build](../pipette-ctl/job-generation.md#pinning-an-ios-app-build)).

How the proxy advertises builds, picks one for a job, and signs and installs it
is in its `DESIGN.md` sections
[8](../../tools/pipette-ios-proxy/DESIGN.md#8-app-builds-from-the-releases-bucket)
and [9](../../tools/pipette-ios-proxy/DESIGN.md#9-version-mode-the-build-a-job-pins);
its [README](../../tools/pipette-ios-proxy/README.md#installing-the-version-a-job-pins)
covers the config, the keys and what the signing Mac needs.
