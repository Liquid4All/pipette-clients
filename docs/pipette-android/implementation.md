# Pipette Android Implementation Ledger

This is the detailed reference for the Android app under `android/Pipette/`: which
files own each piece, the full per-subsystem feature list, the platform choices in
practice, the test suite, and the known gaps. For the higher-level design and
rationale, see [architecture.md](architecture.md); for building and running, see
[build.md](build.md).

## Current Status

The Android app is a native Kotlin client with working local state, model
management, benchmark planning, native benchmark execution, result persistence,
CSV export, and submission plumbing. It is gated behind a Clerk sign-in.

Two structural facts drive everything below:

- **The native engine runs in a separate `:benchmark` process** reached over
  AIDL; the main process never loads `libpipette_android.so`.
- **Storage is filesystem JSON.** Structured state is persisted as JSON files;
  settings use DataStore; downloaded-model metadata is a live directory scan.

The UI is programmatic Android Views organized MVVM-style (`MainActivity` is a
thin shell; `MainViewModel` holds state; per-tab `Screen` classes render). The
biggest remaining gap is not architecture but runtime evidence on physical
hardware (live submission, large downloads, thermal behavior).

## Project Layout

```text
android/Pipette/
  app/build.gradle.kts
  build-rust-android.sh
  app/src/main/AndroidManifest.xml          # declares the :benchmark process
  app/src/main/aidl/ai/liquid/pipette/service/
    IBenchmarkService.aidl                   # engine surface (blocking two-way)
    IBenchmarkRunCallback.aidl               # oneway progress channel
    BenchmarkResult.aidl                     # parcelable result
  app/src/main/java/ai/liquid/pipette/
    PipetteApp.kt                            # Application; process entry; Clerk init (main process only)
    AppContainer.kt                          # manual DI container (app-scoped)
    MainActivity.kt                          # thin Activity shell: chrome, auth gate, Pocket Mode
    MainViewModel.kt                         # transient UI state + auth-gate flow
    Screen.kt                                # ScreenContext + Screen base; Tab enum lives in MainViewModel
    SetupScreen.kt  ModelsScreen.kt  JobsScreen.kt  SettingsScreen.kt
    UiKit.kt                                 # programmatic design system
    UiFormatting.kt                          # small formatting helpers
    NewJobWizard.kt                          # job-wizard step model
    ResultsGrid.kt                           # results heatmap intensity
    ClerkAuth.kt  ClerkConfiguration.kt      # auth gate + config
    DataModels.kt                            # job/cell/registration/model/submission models
    LocalStorage.kt                          # filesystem-JSON storage; implements JobStore
    ModelStore.kt                            # live model-directory scan
    ModelCatalog.kt                          # quant-family grouping
    ModelTemplates.kt                        # bundled GGUF presets
    ModelRef.kt                              # plan-type identifiers (mirrors pipette-plan-types)
    AppSettingsStore.kt                      # DataStore preferences
    BenchmarkCatalog.kt  BenchmarkDefinition.kt
    JobController.kt  JobRunner.kt  JobStore.kt  JobQuantFilter.kt
    BenchmarkEngine.kt                       # engine interface (*Sync methods)
    EngineActor.kt                           # single-thread native-call serializer
    LlamaEngine.kt                           # external JNI stubs + NativeLib availability
    Readiness.kt                             # two-tier readiness/thermal gate
    DownloadCoordinator.kt                   # DownloadManager-backed downloads
    ManagementClient.kt                      # ManagementClient + RegistrationService
    Secrets.kt                               # Ed25519 key + HF token (SharedPreferences)
    ResultSubmissionService.kt               # result upload
    CompletedResultsCsvExporter.kt           # CSV string builder
    DeviceInfo.kt                            # device labels / payload metadata
    service/
      PipetteBenchmarkService.kt             # the :benchmark service (IBenchmarkService.Stub)
      RemoteBenchmarkEngine.kt               # UI-process client proxy
      BenchmarkResult.kt                     # result parcelable impl
      ResultSpill.kt                         # large-result file spill

crates/pipette-android/
  build.rs                                   # CMake link markers; stamps LLAMA_CPP_COMMIT
  native_loader.cpp                          # runtime CPU-variant backend loader
  src/lib.rs                                 # JNI entry points
  src/engine.rs                              # concrete Rust LlamaEngine
  src/loader.rs                              # backend loader registry
  src/llama.rs                               # llama.cpp C-FFI wrappers
  src/error.rs                               # PipetteError
```

## Build Configuration

`android/Pipette/app/build.gradle.kts`:

- Plugins: Android application, Kotlin Compose compiler, ktfmt, detekt. (The
  `room`/`ksp` coordinates in `libs.versions.toml` are unused, no applied
  plugin or dependency references them.)
- Java/Kotlin 17 toolchain; `compileSdk = 36` (minor 1), `minSdk = 31`,
  `targetSdk = 36`.
- `ndk { abiFilters += "arm64-v8a" }`: single ABI.
- `buildFeatures`: `compose = true` (only to host Clerk's `AuthView`),
  `viewBinding`, `aidl`, `buildConfig`.
- Key dependencies: AndroidX appcompat / core-ktx / activity-ktx /
  constraintlayout, lifecycle runtime + viewmodel, DataStore preferences,
  Material, coroutines, Sentry, PostHog (`posthog-android`, for
  product analytics), Tink (`tink-android`, for the Ed25519 signing identity),
  and Clerk (`clerk-android-api`, `clerk-android-ui`) + Compose BoM/material3.
  (No `androidx.navigation`.)
- `CLERK_PUBLISHABLE_KEY` is injected into `BuildConfig` from `local.properties`
  or a CI env var (debug build type uses a separate dev key, falling back to the
  release key). `POSTHOG_API_KEY` / `POSTHOG_HOST` are injected the same way,
  both defaulting to the public production project (`phc_…` project keys are
  write-only, so they are baked in rather than treated as secrets).

### Analytics (PostHog)

Product analytics is a hand-picked, boundary-only event set behind the
`Analytics` seam (`Analytics.kt`); the iOS client duplicates the same event and
property names verbatim so both land in one PostHog project. `PostHogAnalytics.create`
returns `NoOpAnalytics` unless `PostHogConfiguration.isComplete`, and it is only
called from the **main process**: the isolated `:benchmark` process must never
load the SDK, since anything resident there inflates the `max_memory_usage` it
exists to measure.

Autocapture, session replay and feature-flag preloading are all off,
because instrumentation overhead corrupts on-device LLM measurements, the same
reason the Sentry Gradle plugin's `tracingInstrumentation` is disabled. One
non-obvious setting carries most of that weight:
`sessionReplayConfig.captureLogcat` defaults to **true** and
`PostHogLogCatIntegration.install()` gates on it *alone*, never consulting
`sessionReplay`, so `sessionReplay = false` by itself still spawns a `logcat`
subprocess plus a lifetime reader thread. `Analytics.kt` disables it explicitly;
re-verify on every SDK bump.

`remoteConfig = false` is a second such setting, and turning the feature flags off
does not imply it: left at its default the SDK GETs `<host>/array/<key>/config` on
every launch, observed firing even while opted out. Nothing here consumes remote
config, so it is off, which is what makes an opted-out launch emit no PostHog
traffic at all. It carries a deprecation warning since 3.58.0 saying remote config
"is now always enabled", and that is wrong as written: `PostHog.setup` still
branches on the flag (`when { config.remoteConfig -> loadRemoteConfigRequest(...);
config.preloadFeatureFlags -> ... }`), so false plus no flag preloading still means
no request. Verified on 3.58.0 in source and on device, where `/batch` was the only
PostHog request of the whole launch. The `@Suppress("DEPRECATION")` sits on the
`config` local because a Kotlin assignment cannot carry an annotation.

`surveys`, `errorTrackingConfig.autoCapture` and (since 3.58.0)
`setDefaultPersonProperties`, `capturePushNotificationOpened` and
`capturePushNotificationSubscriptions` are pinned too. The last three are the ones a
bump can hurt you with: all default **true**, `capturePushNotificationOpened` puts
`PostHogActivityLifecycleCallbackIntegration` back in play on its own (the gate is
now `captureDeepLinks || captureScreenViews || sessionReplay ||
capturePushNotificationOpened`), and `capturePushNotificationSubscriptions` fetches
an FCM token at startup and registers it with PostHog. `setDefaultPersonProperties`
attaches `$app_version` / `$os_name` / `$device_type` and similar to the *person*
profile for local flag evaluation, which nothing here uses.

**On every version bump, diff which integrations `PostHogAndroid.setup` installs and
on which flags.** The 3.19.0 to 3.58.0 jump added the two push options above and
`PostHogTouchActivityIntegration`, which has no flag at all: it is added
unconditionally and its `install()` gates only on API level, so every dispatched
`MotionEvent` now calls `PostHogSessionManager.touchSession()`. That one cannot be
turned off short of forking, and is tolerable only because it is a timestamp write
on a path that is idle while a cell is timed, but it is the same category as
`captureLogcat`, and the next one may not be as cheap.

No event fires inside a measurement window. `job_started` is captured and
flushed *before* the runner begins, so the queue is empty entering the run and
the SDK's periodic flush timer has nothing to send while a cell is timed. The
one event that fires mid-run is `results_submitted`, from the per-cell upload
`JobRunner` performs between cells (PIP-358, matching iOS):
`ResultSubmissionService.submit` flushes immediately after capturing it, so the
next cell still starts on an empty queue. Any capture site added between the
start of a run and its end inherits that obligation. Events carry the
registration `clientId` as the distinct id;
no email, organization, Clerk identity, error message, or feedback body is
ever sent.

Settings carries a "Share anonymous usage analytics" toggle (`SettingsScreen` →
`SettingsIntent.SetAnalyticsOptOut` → `Analytics.setOptedOut`), shown only when
`Analytics.isAvailable`: a toggle over `NoOpAnalytics` would be a control that
does nothing.

`AnalyticsOptOutStore` owns the flag, and `PostHogAnalytics.create` seeds
`PostHogAndroidConfig.optOut` from it **before** `setup`. Do not "simplify" this
to the SDK's own persisted copy. PostHog does persist opt-out and `setup` does
look it back up, but on posthog-android 3.19.0 that lookup did not work: with
`opt-out=true` sitting in the SDK's own `posthog-<key>.xml`, `PostHog.isOptOut()`
read `false` immediately after `setup` returned and the next `app_launched` was
captured and accepted by the ingest endpoint, measured on a Pixel 10 Pro Fold /
Android 17, while `distinctId` from that same file restored fine.

PostHog fixed that in 3.55.1 (the setup-time read consulted an in-memory fallback
store and never honoured a persisted opt-out; the choice is now resolved lazily on
the capture path), and this client is on 3.58.0, so the SDK's own copy would
probably work today. The seeding stays anyway: it is what makes the behaviour ours
rather than the SDK's to get right, and one launch is all it takes to leak an event
from a device whose owner turned analytics off. Re-verified on 3.58.0: opted out,
a cold start queues zero events and issues zero requests of any kind.

The store is `SharedPreferences`, not `AppSettingsStore`, because this is the one
setting that must be readable **synchronously**: `PipetteApp.onCreate` builds the
SDK and captures `app_launched` in a single non-suspending block, and a DataStore
read could not complete in time.

Opting out also makes launch fully network-silent here, because `remoteConfig` is
off (see above). iOS takes the same defensive seeding but cannot match that part:
posthog-ios deprecated `config.remoteConfig` and genuinely ignores it, so an
opted-out iOS device still makes the remote-config GET. Android's copy of the flag
carries the same deprecation text but is still honoured, which is why the two
platforms differ here.
- A Gradle `Exec` task `buildRustAndroidArm64` runs `build-rust-android.sh`; it is
  wired before JNI-lib merge via `tasks.configureEach { if name matches
  merge*JniLibFolders → dependsOn }`, and its staged output dir is added to the
  main source set. `jniLibs { useLegacyPackaging = true }` keeps native libs
  extracted on disk (the CPU-variant loader scans that directory).

`buildRustAndroidArm64` runs `build-rust-android.sh`, which CMake-builds the
vendored llama.cpp/ggml and the Rust bridge for `aarch64-linux-android` and stages
the resulting `.so`s into the generated JNI libs directory. How that native build
works is documented in [build.md](build.md#native-build-internals):
per-CPU-variant backend dispatch, the shipped armv8/armv9 variant set (armv9
carries a known LFM2A-ASR accuracy caveat from an upstream llama.cpp SVE-config
mismatch; see build.md), and the optional KleidiAI kernels.

## Local Storage Model

All structured app data is persisted as filesystem JSON (`org.json`, written with
`writeText(json.toString(2))`). The internal-storage root is `filesDir/Pipette`:

```text
filesDir/Pipette/
  metadata/registration.json
  jobs/<jobId>/manifest.json                       # the cell array lives inside the manifest
  jobs/<jobId>/cells/<cellId>/payload.json
  jobs/<jobId>/cells/<cellId>/metrics.json
  jobs/<jobId>/cells/<cellId>/submission.json
```

Model files live in **external** app storage, repo-bucketed:

```text
<getExternalFilesDir(null)>/models/<hf-org>/<hf-repo>/<filename>.gguf
<getExternalFilesDir(null)>/models/<local-file>.gguf
```

Important storage behavior:

- `LocalStorage` owns it all and implements the narrow `JobStore` interface that
  `JobRunner` consumes (manifest read/write, `clearCellArtifacts`,
  `resolveModelPath`, `writePayload`, `loadRegistration`). Tests use an in-memory
  `FakeJobStore`.
- Cells are embedded in the manifest JSON, not stored as separate files; the
  per-cell directory holds only `payload/metrics/submission.json`.
- `ModelStore` derives `ModelFile` metadata fresh from a live scan of the models
  directory on each call (the filesystem is the source of truth).
- Repo bucketing: `modelRelativePath(repo, filename)` → `repo/filename` (or bare
  `filename` with no repo). Empty `<org>/<repo>` dirs are pruned on delete.
- A legacy internal `Pipette/models` directory is migrated to external storage on
  startup and deleted; it is used as a fallback only if external storage is
  unavailable.
- The stale-path resolver (`resolveModelPath`): existing path → recover the same
  tail under the current models root → unique-filename match → reject if zero or
  ambiguous matches.

## Data Models

`DataModels.kt` defines:

- `JobStatus`, `CellRunStatus`: with a shared `isRerunnable` rule
  (COMPLETED / FAILED / CANCELLED).
- `JobCell`, `JobManifest`: the manifest computes completed/failed/cancelled/
  submitted counts and exposes `recoverInterruptedRunState()` (RUNNING cells →
  CANCELLED; a RUNNING job with leftover work → PAUSED, else COMPLETED).
- `RegistrationData`: clientId, status, serverUrl, organization, contactEmail,
  `registeredAt`, plus optional Clerk link fields (`clerkUserId`,
  `clerkSessionId`, `clerkPrimaryEmail`, `clerkLinkedAt`); `withClerkLink(...)`
  attaches them and stamps `clerkLinkedAt` once.
- `CellSubmissionRecord`: `submitted(serverJobId)` / `failed(errors)`.
- `ModelFile`: `name`, absolute `path`, `sizeBytes`, `hfRepo`, `displayName`,
  `familyId`, with derived `quant` and `isMmproj`. (Computed on each scan.)

`ModelRef.kt` is a separate set of plan-type mirrors (`HfOrg`, `HfRepoName`,
`GgufPath`, `HfRepo`, `Model`, `ClientId`, `BenchmarkId`, …) copied from Rust
`pipette-plan-types`, with validation regexes matching the `nutype` definitions.
These are wire-format identifiers, not the on-disk catalog.

## Registration And Signing

`RegistrationService` is a top-level class defined in `ManagementClient.kt`
(alongside `ManagementClient`), wired in `AppContainer`.

Registration flow (`RegistrationService.register`):

1. Caller supplies server URL, organization, contact email, and optional Clerk
   identity.
2. `secrets.generatePendingSigningKeyPair()` creates an Ed25519 keypair (via Tink:
   the JCA can't supply an exportable one below API 37; see `Secrets`' KDoc),
   persists the raw 32-byte private key as hex under a *pending* slot, and
   returns the public key hex.
3. `POST /clients/register` with `public_key`, `organization`, `contact_email`,
   `client_details` (= `DeviceInfo.modelName()`). This call is **unsigned**.
4. On failure, the pending key is deleted and the error rethrown.
5. On success, `promotePendingPrivateKey()` moves the pending key to the permanent
   slot; `RegistrationData` is built (attaching Clerk link metadata if a
   `clerkUserId` was supplied) and saved to `metadata/registration.json`. Clerk
   data is never sent to the server.

Signed management requests (submissions carry a `clientId`; registration does
not):

- `X-Client-Id` = client id
- `X-Timestamp` = `Instant.now()` ISO-8601
- `X-Nonce` = `generateNonce()`, 16 `SecureRandom` bytes as lowercase hex
- `X-Signature` = Ed25519 signature over the `v1` payload, hex

The `v1` payload is six newline-separated fields: `v1`, the uppercase HTTP
method, the request target the server receives (`URL.requestTarget()`, so the
configured server URL's own path prefix and any query string are included), the
`X-Timestamp` value, the `X-Client-Id` value, and the `X-Nonce` value. Binding
the method and target scopes a captured signature to that method and target, and
the nonce makes it single-use, so a replay inside the 5-minute freshness window
is rejected. The body is still not covered.

Secret storage (`Secrets`): plain `SharedPreferences("pipette_secrets")`; not
EncryptedSharedPreferences or the Keystore. Stores `private_key_hex` (the raw
32-byte Ed25519 seed), transient `pending_private_key_hex`, and `hf_token`. The
public key comes straight off Tink's keypair; there is no X.509 encoding to
unwrap. Installs predating the Tink switch hold `private_key_pkcs8` instead,
converted in place on the first signed request; a leftover
`pending_private_key_pkcs8` is discarded rather than converted, since an
interrupted registration was never completed server-side.

## Clerk Auth Gate

`ClerkAuth.kt` / `ClerkConfiguration.kt` implement the sign-in gate that fronts
the entire app.

- `RealClerkAuth` derives a `ClerkState` (`Loading` / `InitError` / `SignedOut` /
  `SignedIn(userId, email?, sessionId?)`) by combining the Clerk SDK's
  init/user/session flows. It observes only; Clerk's prebuilt Compose `AuthView`
  drives the actual sign-in / sign-up / OAuth calls.
- A pure reducer `reduceAuthGate(clerk, registration, bypass)` maps state to an
  `AuthGate` (`Loading` / `InitError` / `SignedOut` / `Mismatch` / `Ready`). A
  `SignedIn` user whose id differs from the registration's linked `clerkUserId`
  yields `Mismatch`; a debug-only `bypass` short-circuits to `Ready`.
- `MainActivity.render()` checks `vm.authGate` first and draws only the gate until
  it reaches `Ready`. On `SignedIn`, `MainViewModel.linkRegistrationIfNeeded`
  links the Clerk identity onto the local registration (leaving a conflicting
  prior link alone so the gate surfaces the mismatch).
- Clerk is initialized **only in the main process** (`PipetteApp` guards on
  `isMainProcess() && ClerkConfiguration.isComplete`), and references no Clerk
  types directly; init is isolated in `ClerkBootstrap.create`, so `com.clerk.*`
  is class-loaded only there. The `:benchmark` process never loads Clerk.
- The publishable key comes from `BuildConfig.CLERK_PUBLISHABLE_KEY`
  (`local.properties` or env), and no key is baked into the repo. When no key is
  configured the SDK is never initialized and the gate falls back to `Ready` in
  every build type: no key means this build has no auth, not that it is broken.
  That is what lets a fork build and run without a Clerk instance.

## Model Management

The Models tab supports:

- scanning app model storage
- importing local `.gguf` files through Android's document picker
- deleting models and pruning empty Hugging Face repo buckets
- entering manual Hugging Face identifiers or URLs
- searching downloaded models and templates
- downloading from the default model templates
- displaying MMProjector files
- entering and storing a Hugging Face token

Files: `ModelStore.kt`, `ModelTemplates.kt`, `ModelCatalog.kt`,
`DownloadCoordinator.kt`, `LocalStorage.kt`, `ModelsScreen.kt`.

## DownloadManager Integration

`DownloadCoordinator.kt` uses Android `DownloadManager` for manual and template
downloads. Implemented behavior:

- parses Hugging Face shorthand like `LiquidAI/LFM2.5-350M-GGUF:Q4_0` and the
  `org/repo/file.gguf` path form
- parses full Hugging Face URLs; normalizes `blob/main` → `resolve/main`
- validates `.gguf` filenames; URL-encodes path segments
- downloads into `<dest>.part`, then renames into the final repo-bucketed path
- sends `Authorization: Bearer <token>` only for `huggingface.co` hosts;
  always sends `User-Agent: pipette-android`
- prevents duplicates (dest exists / already active); visible notification;
  metered-network allowed
- cancellation via `DownloadManager.remove` (deletes the `.part`)
- persists active-download records in SharedPreferences and re-attaches monitors
  after process restart (the `DownloadManager` job survives process death)
- registers the finished file via `LocalStorage.registerModelFile`

## Model Templates

`ModelTemplates.kt` is the bundled catalog of default templates. Each preset has a
stable id, display name, detail label, HF identifier, parsed repo, filename,
quant, family id, and estimated bytes; the catalog also exposes `repoToName`,
ordered display groups, families, and lookup by family id.

There are 16 families across six groups (LiquidAI, Qwen, Granite, Gemma,
Ministral, and Llama) each carrying Q4_0, Q4_K_M, and Q5_K_M variants. (Ministral
is the one family whose quants span repos: Q4_0 from `unsloth`, Q4_K_M/Q5_K_M from
`mistralai`.)

## Quant Grouping And Job Model Selection

`ModelCatalog.kt` groups downloaded files into model families:

- keys by `familyId` from templates, falling back to a normalized filename stem
  (quant token and a leading `mmproj-` stripped, lowercased, so a model and its
  projector group together)
- summarizes available quant variants per family
- resolves selected family keys through the active quant filters
- detects selected families missing the currently selected quant

`JobQuantFilter.kt` defines the four filters: `ALL`, `Q4_0`, `Q4_K_M`, `Q5_K_M`
(`ALL` is mutually exclusive with the specific filters; the set never empties).
Users select model families, then choose which quant variants are runnable.

## Benchmark Catalog

`BenchmarkCatalog.kt` + `BenchmarkDefinition.kt`. `BenchmarkType` has six kinds
(by rank): end-to-end latency, prefill throughput, decode throughput, max memory
usage, vision-language throughput, and eval accuracy. `BenchmarkDefinition` is a
sealed class mirroring `pipette-ops`; each subclass models its parameters and
serializes to the flat `benchmark_type`-tagged JSON the engine/server consume.

`buildCatalog` generates the four token-ladder kinds over
`[100, 256, 512, 1024, 2048, 4096]` (8192 excluded as too heavy for phones) plus a
fixed set of VL sizes. **Eval accuracy is modeled but not generated into the
bundled catalog**, so it isn't user-selectable on device today.

Helpers: lookup by id, display names, retired-benchmark filtering
(`decode_throughput_100_100`, `prefill_throughput_100`, `max_memory_usage_100` are
kept for replay but not selectable), type ordering, parameter labels, search (id /
type / name / params), per-cell context sizing, and effective job context sizing
(default 4096, VL floored at 8192). VL gating happens in planning, not the catalog.

## Job Orchestration

Ownership is split across three files:

- **`JobController`**: app-scoped owner; constructs the single `JobRunner`,
  adapts its imperative callback into a `StateFlow` for the UI, and forwards
  control calls (`startNewJob`, `resume`, `retryFailed`, `rerunCells`, `cancel`).
  App-scoped so Activity recreation never tears down a running benchmark.
- **`JobRunner`**: plans cells, persists the manifest through `JobStore`, and
  drives the execution loop on a single-thread executor. Talks to the engine
  through the `BenchmarkEngine` interface and gates on `ReadinessGate`; submits via
  `ResultSubmitter`. Does not touch native code directly.
- **`JobStore`**: the narrow persistence slice `JobRunner` needs; implemented by
  `LocalStorage`.

**Planning** (`planCells`): Cartesian product of selected models × benchmarks; for
`VL_THROUGHPUT`, a model is skipped unless VL-compatible (matching MMProjector by
equal `hfRepo` or normalized stem), then one cell per selected MMProjector;
non-VL benchmarks emit one MMProjector-less cell. Each cell records benchmark
id/type, model path/name, optional MMProjector path, run status, optional server
job id, and optional error. `startNewJob` requires ≥1 model, ≥1 benchmark, and ≥1
planned cell, and starts the job in `RUNNING`.

**Execution**:

- single-thread executor; guards against a concurrent job
- runs only `PENDING` cells, sorted by model path then per-cell context size
- `MAX_MEMORY_USAGE` cells use `runFreshSync` (load is part of the measurement);
  others `loadSync` then `runBenchmarkSync`. Note each cell currently (re)loads
  the model rather than reusing a resident handle, because a fresh server is
  started per benchmark
- resolves stale model/MMProjector paths; unresolved → cell FAILED with a
  re-download hint; repeated load failures on the same path skip later cells
- publishes progress (fraction + text) to the UI; cooperative cancel via
  `CancelFlag` (checked in the loop and in progress callbacks); a cancelled cell
  becomes `CANCELLED`, and `RemoteBenchmarkEngine` hard-kills `:benchmark` if a
  mid-decode cancel isn't honored within its grace window
- applies the `ReadinessGate` between cells
- auto-submits a `COMPLETED` job when `contributeResults == true` and a
  registration exists (re-read from the latest manifest, so a mid-run toggle is
  honored)

**Rerun / recovery**: `resume` flips CANCELLED cells back to PENDING; `retryFailed`
reruns FAILED cells; `rerunCells` resets rerunnable cells to PENDING and clears
their artifacts. Cold-launch recovery runs once per process from `AppContainer`
(`recoverInterruptedJobs`) and brings an interrupted job back as PAUSED →
resumable.

## Native Benchmark Engine

The engine runs out-of-process; the boundary design and the *why* live in
[architecture.md](architecture.md#process--runtime-boundary). This section is the
concrete surface: the interface, its two implementations, the IPC types, and the
JNI entry points.

`BenchmarkEngine.kt` is the interface (`loadSync`, `runBenchmarkSync`,
`runFreshSync`, `unloadSync`, `llamaCppCommit`, `cpuBackendDescriptor`,
`isAvailable`, …). Two implementations exist:

- `EngineActor`: the in-process implementation (used by tests and inside the
  service). It serializes every native call onto a single `HandlerThread` behind
  an Empty/Ready/Busy state machine and owns model lifecycle. It calls
  `LlamaEngine`, whose methods are `external` JNI stubs into
  `libpipette_android.so`. (`object NativeLib` is the availability face; there is
  no `NativeBenchmarkEngine` class.)
- `RemoteBenchmarkEngine`: the production implementation wired by `AppContainer`.
  A UI-process client proxy that turns each `*Sync` call into a blocking two-way
  AIDL round-trip to `PipetteBenchmarkService` in the `:benchmark` process. It
  binds lazily, runs the service in the foreground (to survive backgrounding),
  links to death (surfacing service death as a throwable the cell loop handles),
  tears the service down ~5 s after `unloadSync` (so the OS reclaims the native
  heap), and arms a hard-kill watchdog (~12 s) for unresponsive cancels.

IPC surface (`app/src/main/aidl/.../service/`): `IBenchmarkService.aidl`
(blocking two-way `loadModel`/`runBenchmark`/`runBenchmarkFresh`/`unloadModel`/
`llamaCppCommit`/`cpuBackendDescriptor`/`isAvailable`, plus `oneway requestCancel`),
`IBenchmarkRunCallback.aidl` (`oneway onProgress(completed, total, message)`), and
`BenchmarkResult.aidl` (parcelable). `ResultSpill` carries the result JSON across
the boundary: ≤ ~256 KB rides inline in the parcel; larger payloads spill to a
file in the shared cache dir (both processes share a UID) and are deleted after
the proxy resolves them; avoiding `TransactionTooLargeException`.

Mid-run readiness is handled service-side and deliberately not forwarded over
IPC; the UI gate runs only between cells. See
[architecture.md](architecture.md#process--runtime-boundary) and
[Readiness And Pocket Mode](#readiness-and-pocket-mode).

JNI entry points exposed by `crates/pipette-android/src/lib.rs` (on the
`LlamaEngine` class unless noted):

- `nativeLlamaCppCommit`
- `nativeCpuBackendDescriptor`
- `nativeCreate`: load a model, returns a boxed engine pointer
- `nativeDestroy`: free it
- `nativeRunBenchmark`: run against the already-loaded engine
- `nativeRunFresh`: load + measure + unload (the `max_memory_usage` path)
- `Readiness.nativeWaitUntilReady`: the native readiness probe

Rust modules: `engine.rs` (the concrete `LlamaEngine`: one resident model +
context size, `run_benchmark` delegating to the shared kernel, `run_fresh`,
`llama_cpp_commit`), `loader.rs` (a backend loader registry tried in priority
order), `llama.rs` (the in-process llama.cpp/mtmd C-FFI wrappers, error
classification, CPU-backend descriptor, RSS memory polling), and `error.rs`
(`PipetteError`).

If the shared library isn't packaged, `isAvailable` is false and the UI reports
the engine as missing (cells fail with an explicit error); the library is detected
by a classloader path lookup, so "ready" means *packaged*, not *loaded* (the main
process never loads it).

## Readiness And Pocket Mode

`Readiness.kt` defines `ReadinessGate` and a two-tier `Readiness` gate:

- **primary**: the native `pipette_readiness` probe (OS thermal status +
  CPU-cluster die temperature + CPU %busy) via `nativeWaitUntilReady`
- **fallback** (when the native probe can't read its inputs, or the `.so` isn't
  loaded in the UI process): `PowerManager.getThermalHeadroom` (ceiling 0.85),
  falling back to the coarse `getCurrentThermalStatus` enum

It also defines `ThermalStatusProvider` / `AndroidThermalStatusProvider` (status →
labels). It is consulted between cells (UI process) and per measured rep
(service-side), surfacing "waiting for device to cool/settle" status text.

`MainActivity`'s Pocket Mode is a full-screen active-job view shown when
`vm.pocketModeJobId` is set and that job is running. It hides the bottom nav, keeps
the screen on (`FLAG_KEEP_SCREEN_ON`), and updates in place: a serif title, job
summary, progress bar + "N of M cells", current cell, current progress text, a
thermal chip (tinted by severity), an estimated-time-to-complete line, and
slide-to-exit / cancel controls.

## Results

Successful cells write a payload under
`jobs/<jobId>/cells/<cellId>/payload.json` containing the engine result JSON plus
job id, cell id, model metadata, benchmark metadata, runtime metadata, Android
device metadata (from `DeviceInfo`), and context-size / GPU-layer settings.

## CSV Export

`CompletedResultsCsvExporter.kt` builds the CSV **string** (and a suggested
filename); the actual file write goes through Android's create-document (SAF) flow
in `MainActivity`, not the exporter.

The CSV has 32 columns: job id/title/created; cell id; model name/display
name/quant; benchmark id/type/name/parameters; status; primary metric
name/value/unit/display value; submitted-at; server job id; runtime
name/version/flags/cpu-variant; device name/form-factor/os-name/os-version/chip/
ram/battery-level/power-state/power-save-mode; error message.

Primary-metric handling per type: prefill/decode throughput (tok/s from tokens ÷
time), end-to-end latency (ms, lower better), max memory (bytes, lower better), VL
throughput (tok/s from prompt+predicted timing). Rows are ordered model group →
quant → benchmark column, with RFC-style escaping; the heatmap shading uses
direction-aware column normalization (`ResultsGrid.heatmapIntensity`).

## Result Submission

`ResultSubmissionService.kt` implements `ResultSubmitter`:

- batch-submit all completed, unsubmitted cells; or a single cell
- recover previously persisted submission records (back-filling `serverJobId` from
  a `submission.json` marked submitted)
- skip cells without payloads; validate server-response indexes (each result
  `index` must fall within the batch) and flag omitted results
- persist submitted server job ids on the manifest and failed submission records
- a fixed `DEFAULT_BATCH_SIZE = 1000`

## Main UI

The UI is programmatic Android Views in an MVVM arrangement:

- `PipetteApp` (Application) is the process entry point and owns `AppContainer`;
  it initializes Clerk only in the main process.
- `AppContainer` is a manual DI container (no Hilt) wiring the app-scoped
  singletons (`LocalStorage`, `AppSettingsStore`, `Secrets`, `ManagementClient`,
  `RegistrationService`, `DownloadCoordinator`, `ResultSubmissionService`,
  `RemoteBenchmarkEngine`, `JobController`, readiness) and running
  `recoverInterruptedJobs()` once per process.
- `MainViewModel` holds transient UI state (selected tab, search texts,
  selections, run params, `pocketModeJobId`, wizard step) and the `authGate`
  `StateFlow`; it drives an imperative re-render tick.
- `MainActivity` is a thin shell: it draws the header + `BottomNavigationView`,
  owns the document launchers (model import, CSV export), renders the auth gate and
  Pocket Mode, and on each tick delegates the body to the active `Screen`.
- `Screen.kt` defines the `Screen` base + `ScreenContext`; the `Tab` enum
  (`SETUP`, `MODELS`, `JOBS`, `SETTINGS`, default `JOBS`) lives in `MainViewModel`.

Per-tab screens:

- **Setup** (`SetupScreen`): registration form / summary, clear registration.
- **Models** (`ModelsScreen`): downloaded models, local import, HF download,
  templates, MMProjector listing, HF token, search.
- **Jobs** (`JobsScreen`, the largest): job list, the new-job wizard (model-family
  selection, quant filters, benchmark selection, MMProjector selection, context /
  GPU layers, planned-cell review with missing-quant warnings, run), and per-job
  detail (status counts, live progress, auto-submit toggle, resume/retry/rerun,
  CSV export, submit, delete, per-cell detail + the results heatmap).
- **Settings** (`SettingsScreen`): contribute/auto-submit default, account (Clerk
  sign-in/out + debug bypass), thermal state, HF token, reset local data, debug
  info.

Shared UI: `UiKit` (the design system: typography, cards/tiles/rows, buttons,
chips/badges, inputs, progress, heatmap colors, dialogs), `UiFormatting` (plural /
thermal-label helpers), `NewJobWizard` (pure step model: Models → Benchmarks →
Review), `ResultsGrid` (heatmap intensity).

## Tests

Unit tests (`app/src/test`): `AuthGateTest`, `BenchmarkCatalogTest`,
`JobRunnerTest`, `NewJobWizardTest`, `ReadinessTest`, `ResultsGridTest` (plus
`ExampleUnitTest`), backed by fakes (`FakeBenchmarkEngine`, `FakeJobStore`,
`FakeReadinessGate`, `FakeResultSubmitter`). These cover the auth-gate reducer,
benchmark catalog/search/sizing, job planning + pending-only execution ordering +
rerun/recovery, wizard step logic, the readiness fallback tiers, and heatmap
shading.

Instrumented tests (`app/src/androidTest`) exercise the new out-of-process
architecture on device/emulator: `RemoteEngineIsolationTest` (process isolation),
`EngineMemoryBaselineTest` (native-heap reclamation), `KleidiaiBenchTest` (KleidiAI
A/B).

Build and test commands are in [Useful Commands](#useful-commands).

## Known Gaps

The app runs end to end, but is not yet fully proven on physical hardware. The
remaining work is mostly validation:

- validate live registration and result submission against the management server
- validate model/template downloads through `DownloadManager` on device,
  including large files, cancellation, process restart, and HF-token-protected
  downloads
- validate physical-device thermal/readiness behavior (emulators don't throttle
  realistically)
- expand native ABI support if non-arm64 devices matter

## Useful Commands

Build and test (the wrapper resolves `JAVA_HOME`/`ANDROID_HOME`; see
[build.md](build.md)):

```bash
./android/build.sh test
```

Install debug build (extra args forward to `gradlew`):

```bash
./android/build.sh installDebug --console=plain
```

Start ADB server:

```bash
~/Library/Android/sdk/platform-tools/adb start-server
~/Library/Android/sdk/platform-tools/adb devices
```

Launch app on the emulator (the debug build installs as `ai.liquid.pipette.debug`,
but the activity class keeps its original namespace; see
[build.md](build.md#inspecting-devices)):

```bash
~/Library/Android/sdk/platform-tools/adb -s emulator-5554 shell am start -n ai.liquid.pipette.debug/ai.liquid.pipette.MainActivity
```

Inspect crash logs:

```bash
~/Library/Android/sdk/platform-tools/adb -s emulator-5554 logcat -b crash -d
```

Dump app files (debug build; `run-as` needs the installed package id, i.e. the
`.debug` variant):

```bash
~/Library/Android/sdk/platform-tools/adb -s emulator-5554 shell run-as ai.liquid.pipette.debug find files -maxdepth 5 -type f | sort
```
