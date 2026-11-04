# Building the Pipette Android App

How to build the native library `libpipette_android.so`, and how to build, test,
and run the app. For the architecture behind these artifacts (the engine
flavors, the process boundary, storage, the job runner), see
[architecture.md](architecture.md).

The app targets `arm64-v8a` only.

The app has two engine flavors (see
[architecture.md](architecture.md#engine-flavors)). `kotlinEngine` is the
default, and every command on this page builds it unless it names `rustEngine`.
With flavors, each Gradle task name includes the flavor:
`assembleKotlinEngineDebug`, `installKotlinEngineDebug`,
`testKotlinEngineDebugUnitTest`. A name without the flavor, such as
`installDebug`, is ambiguous and fails.

## Quickstart

Get a debug build running on a connected arm64 device or emulator. This is the
fast path; each step links to its full treatment below.

1. **Install the toolchain.** A fresh checkout needs a JDK, the Android SDK and
   NDK, `cmake`, `ninja`, and Rust (rustup). `android/build.sh` checks for each
   tool. It stops at the first missing tool and prints the fix for that tool
   only. Install everything in [Prerequisites](#prerequisites) first, so one
   run finds everything.

   The wrapper finds a JDK and the Android SDK in the standard install
   locations and exports `JAVA_HOME` and `ANDROID_HOME`. Set them yourself only
   if your installs are in other locations; see
   [Android SDK + components](#android-sdk--components).

2. **Build and install** onto the device/emulator:

   ```bash
   ./android/build.sh installKotlinEngineDebug
   ```

   The first run is slow: it builds the vendored llama.cpp/ggml native libraries.

3. **Launch it.** Tap the **Pipette** icon, or start the activity explicitly. A
   debug build installs under the package `ai.liquid.pipette.debug`, but the
   activity class keeps the namespace `ai.liquid.pipette`. The component is thus
   `…debug/ai.liquid.pipette.ComposeMainActivity` (see
   [Inspecting devices](#inspecting-devices)):

   ```bash
   "$ANDROID_HOME/platform-tools/adb" shell am start -n ai.liquid.pipette.debug/ai.liquid.pipette.ComposeMainActivity
   ```

4. **Get past the sign-in screen.** The app is gated behind a Clerk sign-in, but
   a debug build needs **no Clerk setup** to get in:

   - **No Clerk key configured (the default):** the gate is open, and the app
     opens directly to the UI with no sign-in screen.
   - **A Clerk key is configured:** tap **"Skip sign-in (debug only)"** on the
     sign-in screen, or toggle **Settings → Account → "Bypass auth gate (debug
     only)"**. The bypass is debug-only and persists across restarts.
   - **To actually sign in** against a dev Clerk instance, set
     `clerk.publishableKey.debug` (see [Clerk sign-in](#clerk-sign-in)) and use a
     `+clerk_test` email with the fixed code `424242`.

5. **Confirm the engine is packaged.** If the native library is missing from
   the APK, the Jobs tab shows *"Native benchmark engine missing"* (see
   [Inspecting devices](#inspecting-devices)). If it shows no such message, add
   models on the Models tab and plan a job on the Jobs tab.

A release build with a Clerk key has no bypass. See [Clerk sign-in](#clerk-sign-in).

### Running on an emulator

No physical device? Create an emulator. Because the app is **`arm64-v8a` only**,
the AVD must use an **`arm64-v8a` system image**. An x86_64 image won't run the
native library. Arm images run natively on Apple Silicon Macs; on x86_64 hosts
they're software-emulated and far too slow to be useful here.

```bash
SDKM="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"
AVDM="$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager"

# the emulator runtime + an arm64 image matching the project's android-36 target
"$SDKM" "emulator" "system-images;android-36;google_apis;arm64-v8a"
"$AVDM" create avd -n pipette -k "system-images;android-36;google_apis;arm64-v8a"

"$ANDROID_HOME/emulator/emulator" -avd pipette &   # boot it (leave running)
"$ANDROID_HOME/platform-tools/adb" devices          # confirm it shows as emulator-5554
```

Once it's booted, `./android/build.sh installKotlinEngineDebug` installs onto it like any
other device. An emulator is fine for exercising the UI and the job/download
flow, but **its benchmark numbers are meaningless**. It runs on host CPU and
doesn't throttle like real silicon, so use a physical device for any measurement
you care about.

## How to build

```bash
./android/build.sh          # assemble the debug APK (:app:assembleKotlinEngineDebug)
./android/build.sh test     # compile + unit tests + assemble
./android/build.sh --help
```

`android/build.sh` is the recommended entry point. Before it starts Gradle, it:

- resolves and exports `JAVA_HOME` and `ANDROID_HOME`
- initializes the vendored submodules that the Android build uses
- checks that the NDK, CMake, Ninja, `cargo`, and `rustup` are present, and
  prints the exact fix for a missing one

`test` runs `:app:compileKotlinEngineDebugKotlin`,
`:app:testKotlinEngineDebugUnitTest`, and `:app:assembleKotlinEngineDebug`. It installs nothing, runs
no package managers, and does not write configuration files (notably no
`local.properties`). Extra arguments are forwarded to `gradlew`.

You still have to *install* the tools it checks for; it only finds them and tells
you what's absent. See [Prerequisites](#prerequisites) for the install steps, or
[Manual Gradle build](#manual-gradle-build) if you'd rather drive Gradle
directly.

## Prerequisites

`android/build.sh` checks for everything in this section and points you at
whatever is missing, but it can't install these for you, so set them up first.

Two parts of the build use different tools. Gradle and AGP (the Android Gradle
Plugin) compile the Kotlin app. A shell script does a CMake build of the native
library and the vendored llama.cpp: `build-native-android.sh` for
`kotlinEngine` (C++ only), and `build-rust-android.sh` for `rustEngine` (C++ and
Rust). The Gradle wrapper (`./gradlew`) downloads Gradle itself, and
`build-rust-android.sh` adds the `aarch64-linux-android` Rust target. You
provide everything else below.

### A JDK to run Gradle

Install any **JDK 17–26** and make sure it's discoverable via `JAVA_HOME` or
`PATH` (e.g. `brew install openjdk`). That's all you need: Gradle is a JVM app
and can't bootstrap its own runtime, but it manages every other Java version
itself.

(Why no specific version: Gradle 9.4.1 itself runs on JDK 17–26; it then
auto-provisions a Java 17 toolchain to *compile* the app (via the foojay
resolver), and a Java 21 JVM to run its build *daemon* (pinned by
`gradle/gradle-daemon-jvm.properties`). The JDK you install only has to launch
Gradle.)

> **macOS note (manual `./gradlew` only):** a Homebrew JDK isn't always
> discoverable until you symlink it where macOS looks (Homebrew prints the exact
> `ln -s …` command at install time); without it, `./gradlew` falls through to
> Apple's `/usr/bin/java` stub and fails with *"Unable to locate a Java
> Runtime."* The `android/build.sh` wrapper sidesteps this. It probes the
> Homebrew location directly, so the symlink isn't needed when you build through
> it.

### Android SDK + components

Install the Android SDK (via Android Studio (which bundles a JDK too), or the
`android-commandlinetools` Homebrew cask), and point Gradle at it with **either**
an `ANDROID_HOME` env var **or** a `sdk.dir` line in
`android/Pipette/local.properties`:

```bash
echo "sdk.dir=/path/to/android-sdk" > android/Pipette/local.properties
```

A bare command-line-tools install ships only `cmdline-tools/`; install the rest
with `sdkmanager` (accept licenses first):

```bash
export ANDROID_HOME=/path/to/android-sdk
SDKM="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"
yes | "$SDKM" --licenses
"$SDKM" "platform-tools" "platforms;android-36.1" "build-tools;36.0.0" "ndk;<version>"
```

- `platforms;android-36.1` matches the project's `compileSdk` (36, minor
  level 1). `targetSdk` is 36 and `minSdk` is 31.
- The **NDK** is required for the native build. Both native build scripts find
  it from `ANDROID_NDK_HOME` / `ANDROID_NDK`, else the newest under `$SDK/ndk`,
  so any recent version works (`"$SDKM" --list | grep ndk` shows the options).
- AGP can auto-download the platform and build-tools once licenses are accepted,
  but **not** the NDK: install that one explicitly.

### CMake and Ninja (native build)

Both native build scripts drive the vendored llama.cpp/ggml build through CMake
and check for **`cmake` and `ninja` on `PATH`**. These are the *system* tools,
**not** the SDK's `cmake;` package (which the script does not use):

```bash
brew install cmake ninja
```

### Rust

**Rust via [rustup](https://rustup.rs)** on `PATH`. Only the `rustEngine` build
compiles Rust, but `android/build.sh` requires `cargo` and `rustup` for every
build. `build-rust-android.sh` adds the `aarch64-linux-android` target, or you
can add it first:

```bash
rustup target add aarch64-linux-android
```

### Vendored submodules

The native library compiles three git submodules under `vendor/`: `llama.cpp`,
`kleidiai`, and `sentry-native`. `android/build.sh` initializes them. To do it
yourself:

```bash
git submodule update --init --recursive -- vendor/llama.cpp vendor/kleidiai vendor/sentry-native
```

The other submodules under `vendor/` are for other clients.

## Clerk sign-in

The app can sit behind a Clerk sign-in gate, but **no key is baked into the
repo**. Sign-in is enabled by the presence of a publishable key and disabled by
its absence, in every build type, so a fresh checkout builds and runs with the
gate open and needs no Clerk instance of its own.

The key is public by design (the same value ships inside an APK anyone can
unzip). Keeping it out of the tree is not about secrecy: it stops a fork from
silently authenticating against Liquid's instance.

It is injected into `BuildConfig.CLERK_PUBLISHABLE_KEY` by
`app/build.gradle.kts`, sourced from `local.properties` first, then the env var,
then the fallback:

| build type | `local.properties` key | env var | fallback |
|---|---|---|---|
| debug | `clerk.publishableKey.debug` | `CLERK_PUBLISHABLE_KEY_DEBUG` | the configured release key, else blank |
| release | `clerk.publishableKey.release` | `CLERK_PUBLISHABLE_KEY` | blank |

Debug falls back to the release key so a single `clerk.publishableKey.release`
opts both variants into the same instance.

What the gate does at runtime depends only on whether a key is set:

- **No key (the default checkout, debug or release):** the SDK is never
  initialized and the gate falls straight to `Ready`: the app opens to the UI
  with no sign-in screen.
- **Key configured, debug:** the sign-in screen appears, but you can skip it with
  the **"Skip sign-in (debug only)"** button on that screen, or the
  **Settings → Account → "Bypass auth gate (debug only)"** toggle. The bypass is
  debug-only and persists across restarts.
- **Key configured, release:** the gate demands a real sign-in and there is no
  bypass.

Every build logs which of these it is (`Clerk: sign-in ENABLED …` / `Clerk:
sign-in DISABLED …`), which is also how a CI run shows whether the
`CLERK_PUBLISHABLE_KEY` secret reached the build.

`verifyReleaseClerkKey` runs before every release `assemble` and `bundle` task
(for example `assembleKotlinEngineRelease`). It does not require a key, because a
build without one is supported. It fails a key that is set but malformed, and it
warns when a release has sign-in disabled. Liquid's own distribution workflow
also fails when the secret is missing, so only a fork can ship with auth off by
accident.

Three places check the key. None of them can call the others, so each has its
own copy of the rule:

| Checked by | Where | Rule | On a failing key |
|---|---|---|---|
| `isCompleteKey` | `ClerkConfiguration.kt` (runtime) | non-blank, no unexpanded `$(…)` or `${…}` placeholder | treats the build as having no auth |
| `verifyReleaseClerkKey` | `app/build.gradle.kts` (build time) | blank is allowed (with a warning); otherwise starts with `pk_` and passes `isCompleteKey` | fails every `assemble*Release` / `bundle*Release` task |
| "Verify the Clerk key is present" | `android-release-distribution.yml` (CI) | non-empty and starts with `pk_` | fails the distribution run |

The runtime check does not test for `pk_`: a non-blank key without a placeholder
goes to `Clerk.initialize`. The placeholder check matters because a prefix check
alone accepts `pk_$(CLERK_PUBLISHABLE_KEY)`. The runtime treats that value as no
key, so a release would ship with the gate open. If you change the rule, change
it in all three places.

### Forking

Nothing to do: clone and build. To put your own Clerk instance behind the gate,
set `clerk.publishableKey.release` locally, or add a `CLERK_PUBLISHABLE_KEY`
secret to your fork for CI builds.

To **actually sign in** against a development Clerk instance, set a dev key:

```properties
# android/Pipette/local.properties
clerk.publishableKey.debug=pk_test_…
```

A dev instance accepts Clerk's test identities: an email containing `+clerk_test`
(e.g. `you+clerk_test@example.com`) verifies with the fixed code `424242`, so you
never touch production while developing.

Those test identities are a **development-instance** feature. The production
instance rejects them (it blocks email subaddresses outright), so against
production the gate's second route is the one to use: enter the address, tap
**"Sign in with a password"**, and enter the account's password. That is also the
credential to put in Play Console → App content → App access, since a reviewer
cannot receive an emailed code. It signs in only; it never creates an account.

### Clerk verbose logging

One further property turns on the SDK's verbose logging, which is **off by
default**:

| BuildConfig field | env var | `local.properties` key | default |
|---|---|---|---|
| `CLERK_DEBUG_LOGGING` | `CLERK_DEBUG_LOGGING` | `clerk.debugLogging` | `false`, and always `false` for release |

Turning it on logs every Frontend-API operation and error code, which helps when
an auth failure is unclear. It also logs every request and response **body**,
which puts credentials in logcat in the clear: the email-code OTP (one-time
password), the session JWT (JSON Web Token), and the password from the password
step. So switch it on
while you are debugging sign-in, and off again afterwards:

```properties
# android/Pipette/local.properties
clerk.debugLogging=true
```

## Manual Gradle build

`android/build.sh` (above) wraps this. Use it unless you specifically want to
drive Gradle yourself. With the SDK location configured and a JDK on `PATH`,
build from the app module:

```bash
cd android/Pipette
./gradlew :app:compileKotlinEngineDebugKotlin :app:testKotlinEngineDebugUnitTest :app:assembleKotlinEngineDebug
```

For the `rustEngine` flavor, use the same task names with `RustEngine` in place
of `KotlinEngine`. `:app:assembleDebug` builds both flavors.

The app build runs the native build automatically. Each flavor has its own
Gradle task:

| Flavor | Gradle task | Script | Output directory |
|---|---|---|---|
| `kotlinEngine` | `buildNativeAndroidArm64` | `build-native-android.sh` | `app/build/generated/kotlinJniLibs` |
| `rustEngine` | `buildRustAndroidArm64` | `build-rust-android.sh` | `app/build/generated/rustJniLibs` |

Each script CMake-builds the vendored llama.cpp/ggml for
`aarch64-linux-android`, builds `libpipette_android.so` (from `jni.cpp` and the
shim for `kotlinEngine`, from the `pipette-android` crate and the shim for
`rustEngine`), and stages the `.so` files, with `libc++_shared.so`, into the
output directory.

To check the Rust crate without Gradle (it covers `rustEngine` only; on the host
the crate compiles to stubs):

```bash
cargo check -p pipette-android
```

### Native build options

Both native build scripts honor these environment overrides:

- `ANDROID_NDK_HOME` / `ANDROID_NDK`: use a specific NDK instead of the newest
  under `$SDK/ndk`.
- `ANDROID_API_LEVEL`: override the native min API level (default `31`).
- `PIPETTE_ENABLE_KLEIDIAI=1` opts into the KleidiAI CPU kernels. Off by
  default, and measured *slower* than off on every SoC tested so far. See
  [KleidiAI](#kleidiai-opt-in) below before enabling it.

## Native build internals

[Manual Gradle build](#manual-gradle-build) covers how the native build tasks
connect to Gradle. This section covers what the native build produces: several
native libraries. The build script CMake-builds llama.cpp/ggml as shared
libraries, with one CPU-backend variant per ARM feature level. It links
`libpipette_android.so` against them and stages every `.so` into Gradle's
generated JNI libs directory. `useLegacyPackaging = true`
keeps them extracted to disk on install (the variant loader scans the lib
directory).

### Per-CPU-variant backend dispatch

The Android build builds one CPU backend per ARM feature level: llama.cpp/ggml
is CMake-built with
`GGML_CPU_ALL_VARIANTS=ON` + `GGML_BACKEND_DL=ON` + `BUILD_SHARED_LIBS=ON`, which
produces a separate runtime-loadable CPU backend per ARM feature level. The
shipped set is an explicit allowlist (`SHIP_VARIANTS` in
`cmake/stage_android_jnilibs.cmake`):

| variant `.so` | feature level |
|---|---|
| `libggml-cpu-android_armv8.0_1.so` | `armv8-a` (baseline) |
| `libggml-cpu-android_armv8.2_1.so` | `armv8.2-a+dotprod` |
| `libggml-cpu-android_armv8.2_2.so` | `armv8.2-a+dotprod+fp16` |
| `libggml-cpu-android_armv8.6_1.so` | `armv8.6-a+dotprod+fp16+i8mm` |
| `libggml-cpu-android_armv9.0_1.so` | `armv9-a` (SVE2) |
| `libggml-cpu-android_armv9.2_1.so` | `armv9.2-a` (SVE + SME) |
| `libggml-cpu-android_armv9.2_2.so` | `armv9.2-a` (SVE + SVE2 + SME) |

SVE and SVE2 are the Arm Scalable Vector Extensions; SME is the Scalable Matrix
Extension.

The base engine (`libllama.so` / `libmtmd.so` / `libggml.so` / `libggml-base.so`)
ships alongside them. At startup `android/native/native_loader.cpp`
(called from the shim's `ensure_backend_init`, before `llama_backend_init`) scans
the app's native-lib dir, calls each variant's `ggml_backend_score()` (returns 0
when the CPU lacks the variant's required features), and registers the
highest-scoring supported one. A single APK thus uses the best variant on each
device (for example `i8mm` where present). It never stops with SIGILL (an
illegal-instruction signal) on a CPU below a fixed feature floor, which a single
`-march=...+i8mm` build would do on pre-armv8.6 silicon. The threadpool API (perf-core affinity) lives in the
variant `.so`, so the shim reaches it through the loader's
`pipette_ggml_threadpool_*` wrappers. All `.so`s link `-Wl,-z,max-page-size=16384`
for Android 15+ 16 KB pages.

**The armv9 variants ship, and Pipette reports what they do, including wrong
output.** On an armv9 device the loader may select an SVE-based backend, but the SVE configuration
llama.cpp assumes isn't guaranteed across mobile armv9 SoCs: SVE2 is mandatory in
the ARMv9 spec, yet vendors diverge; Qualcomm omits it entirely, and vector
length / feature support varies elsewhere. When the selected backend doesn't match
the silicon, the symptom depends on the workload: some hit an outright crash (the
`svcntb() == QK8_0` abort,
[ggml-org/llama.cpp#8109](https://github.com/ggml-org/llama.cpp/issues/8109)),
while the **LFM2-Audio (`lfm2a`) audio encoder instead returns incorrect,
off-topic transcripts** (TTS and vision are unaffected). The wrong-output case occurs on the armv9 Tensor test
devices (Tensor G5 / Pixel 10); its exact root cause is not known.

This is a backend-selection and configuration issue in upstream llama.cpp, and it
is not specific to the model. Pipette ships and measures the selected backend on
purpose: it is a benchmark *harness*, and its job is to report how a device's
stock inference behaves. Treat affected
armv9 devices' audio-encoder accuracy numbers with this caveat until the upstream
issue is fixed.

Staging is **fail-closed** for unknown variants:
`cmake/stage_android_jnilibs.cmake` copies only the variants in the explicit
`SHIP_VARIANTS` allowlist. The build *fails* if ggml builds a variant that is not
in the allowlist, or if it stages no variant at all.

### KleidiAI (opt-in)

Arm's KleidiAI GEMM microkernels are **off by default** and enabled at build time
with `PIPETTE_ENABLE_KLEIDIAI=1` (e.g. `PIPETTE_ENABLE_KLEIDIAI=1 ./android/build.sh`),
which adds `-DGGML_CPU_KLEIDIAI=ON` to the ggml CMake build.

**Leave it off.** It has been measured slower than the stock ggml kernels on every
SoC tested. The numbers below come from an in-app test harness on SM8750,
running a 2.6B-parameter model as a **Q4_0** GGUF, with greedy output
token-identical between arms:

| arm (SM8750, in-app, 12 turns/arm) | prefill tok/s | decode tok/s |
|---|---|---|
| KleidiAI **on** | 154.4 ± 14.5 | 27.82 ± 1.21 |
| KleidiAI **off** | **193.6 ± 23.5** | **32.18 ± 0.75** |

`llama-bench` agrees, with the arms interleaved in one session and the order
rotated (n=4). Its two workloads are the same two columns as the table above:
`pp` (prompt processing) is prefill, `tg` (text generation) is decode. Comparing
like with like on CPU-variant dispatch: on SM8750 with a fixed-ISA build,
28.59 ± 0.41 tg / 170.10 pp with KleidiAI on against **36.81 ± 0.67 tg /
217.22 pp** off; on a Tensor G5 with variant dispatch on, 13.22 ± 0.71 tg against
**14.53 ± 0.83 tg** off.

The flag stays because the result can differ per SoC and upstream recommends the
kernels. Re-measure before enabling it anywhere, and interleave the
arms: these differences are smaller than the drift across a warming device.

**It also costs memory.** KleidiAI registers as a ggml *extra buffer type*, so its
tensors cannot take llama.cpp's zero-copy mmap path (`is_default_buft` is false in
`llama-model.cpp`). They are allocated and repacked instead, which leaves the file
mapping resident but never read: a 1.5 GB model measured `CPU_Mapped 1503.20 MiB`
plus `CPU_KLEIDIAI 1306.14 MiB`, and turning mmap off recovered 1.27 GB of PSS
(proportional set size, -41%). By default the shim loads with
`LLAMA_LOAD_MODE_NONE` (no mmap), so a cell pays this cost only if it sets the
`load_mode` runtime flag to `mmap`. On SME hardware there is a second cost: KleidiAI
packs each eligible tensor twice, once for the SME kernel and once for a non-SME
fallback slot.

Coverage is narrow. It only engages `Q4_0`, `Q8_0`, and `F32` weights, never
k-quants like Q4_K_M, so most of the catalog is unaffected either way. KleidiAI is
also the only place ggml has SME kernels, so a build with it off gets
no SME acceleration. This does not
matter on the test hardware: the Tensor G5 reports
no `sme` in `/proc/cpuinfo` and has no `smidr_el1` sysfs node, so its two SME CPU
variants score 0 and are never selected.

## Inspecting devices

```bash
"$ANDROID_HOME/platform-tools/adb" start-server
"$ANDROID_HOME/platform-tools/adb" devices
```

(or just `adb …` if `$ANDROID_HOME/platform-tools` is on your `PATH`)

The release applicationId is `ai.liquid.pipette`, but the **debug build installs
as `ai.liquid.pipette.debug`** (`applicationIdSuffix = ".debug"`, so a dev build
can be installed next to a release build). The activity *class* keeps the
namespace `ai.liquid.pipette` in both builds. The launch component for a debug
build is thus `ai.liquid.pipette.debug/ai.liquid.pipette.ComposeMainActivity`,
as in the Quickstart's launch step. The short form `pkg/.ComposeMainActivity`
resolves the class against the suffixed id and fails with *"Activity class …
does not exist."* If you are not sure, `resolve-activity` (below) prints the
exact component.

To confirm what's actually installed and what its launchable component is:

```bash
# which variant(s) are installed
"$ANDROID_HOME/platform-tools/adb" shell pm list packages | grep pipette
# the exact launchable component for a package
"$ANDROID_HOME/platform-tools/adb" shell cmd package resolve-activity --brief ai.liquid.pipette.debug | tail -1
```

If `libpipette_android.so` is not packaged in the APK, the Jobs tab shows
*"Native benchmark engine missing"*, and cells fail with an explicit error. The
check is a class-loader path lookup, so it confirms that the library is packaged,
not that an engine is loaded. Benchmarks load the library in the `:benchmark`
process on the first benchmark call.

Read crash logs and the app's files (debug build; `run-as` needs the installed
package id, so use the `.debug` package):

```bash
"$ANDROID_HOME/platform-tools/adb" logcat -b crash -d
"$ANDROID_HOME/platform-tools/adb" shell run-as ai.liquid.pipette.debug find files -maxdepth 5 -type f | sort
```
