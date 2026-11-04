# Pipette Android Implementation Ledger

This is the detailed reference for the Android app under `android/Pipette/`. It
tells where each piece lives and how it works: the source layout, the details of
each subsystem, the test suite, and the known gaps. For what the app is and why
it is built this way, see [architecture.md](architecture.md). To build and run
the app, see [build.md](build.md).

## Current Status

The app runs the full workflow end to end: local state, model management,
benchmark planning, native benchmark execution in both engine flavors, result
persistence, CSV export, and submission. The open work is mostly validation on
physical hardware; see [Known Gaps](#known-gaps).

## Project Layout

Key files, by source set. Paths under `java/` are in package `ai.liquid.pipette`.

```text
android/Pipette/
  app/build.gradle.kts                       # flavors, native build tasks, BuildConfig fields
  build-native-android.sh                    # native build for kotlinEngine (C++ only)
  build-rust-android.sh                      # native build for rustEngine (C++ + Rust)
  app/src/main/
    AndroidManifest.xml                      # launcher, :benchmark service + activity, headless activity
    aidl/ai/liquid/pipette/service/
      IBenchmarkService.aidl                 # engine surface
      IBenchmarkRunCallback.aidl             # oneway progress channel, :benchmark → main
      IJobCancelCallback.aidl                # oneway cancel channel, pocket screen → main
      BenchmarkResult.aidl                   # parcelable result
    java/ai/liquid/pipette/
      ComposeMainActivity.kt                 # launcher activity; hosts PipetteAppRoot
      PipetteApp.kt                          # Application; process entry; Clerk and PostHog init (main process only)
      AppContainer.kt                        # hand-written, app-scoped DI container
      compose/
        PipetteAppRoot.kt                    # gates, Navigation 3 display, document launchers
        nav/Route.kt                         # Navigation 3 keys
        shell/                               # ShellViewModel, AuthGateScreen, PocketModeScreen
        jobs/ models/ settings/ setup/       # one screen + one view model per directory
        theme/                               # colors and theme
      ClerkAuth.kt ClerkConfiguration.kt     # auth-gate seam, reducer, key rule
      DataModels.kt                          # job, cell, registration, model, submission models
      LocalStorage.kt JobStore.kt            # filesystem-JSON storage and its JobRunner slice
      ModelStore.kt ModelCatalog.kt          # live model scan; family grouping
      ModelTemplates.kt                      # ModelTemplateCatalog: bundled GGUF presets
      ModelRef.kt ModelUri.kt RuntimeFlagRef.kt SubmissionRef.kt  # plan-types mirrors
      AppSettingsStore.kt                    # DataStore preferences
      BenchmarkSync.kt BenchmarkStore.kt     # catalog sync from the server, and its store
      BenchmarkCatalog.kt BenchmarkDefinition.kt
      JobController.kt JobRunner.kt JobQuantFilter.kt NewJobWizard.kt ResultsGrid.kt
      ModelProvisioner.kt                    # fetches a model= coordinate before a run
      BenchmarkEngine.kt                     # engine interface (*Sync methods)
      NativeLib.kt                           # build id and availability of libpipette_android.so
      BenchmarkActivity.kt BenchmarkProgressBus.kt  # pocket-mode run screen in :benchmark
      CpuAffinityProbe.kt CpusetWatcher.kt   # cpuset demotion guard
      Readiness.kt thermal/Thermal.kt        # readiness gate; thermal telemetry
      DownloadCoordinator.kt DownloadWorker.kt DownloadStore.kt DownloadRegistry.kt
      DownloadLocks.kt DownloadNotifications.kt DownloadActionReceiver.kt
      ManagementClient.kt                    # ManagementClient + RegistrationService
      ProfileRefreshService.kt Capabilities.kt
      Secrets.kt IdentityBundle.kt           # encrypted key + HF token; identity export/import
      ResultSubmissionService.kt CompletedResultsCsvExporter.kt
      Analytics.kt PostHogConfiguration.kt   # PostHog seam (main process only)
      BenchmarkCrashReporter.kt BenchmarkExitReporter.kt CrashEnvelopeForwarder.kt BenchmarkOutbox.kt
      DeviceInfo.kt
      headless/                              # HeadlessRunActivity, verbs, caller guard
      ops/                                   # Measurement, PromptSeed, ThermalSeries (kernel support)
      service/
        PipetteBenchmarkService.kt           # the :benchmark service (IBenchmarkService.Stub)
        RemoteBenchmarkEngine.kt             # main-process client proxy
        JobOrchestratorService.kt            # main-process foreground service during a job
        BenchmarkResult.kt ResultSpill.kt    # result parcelable; large-result file spill
      MainActivity.kt MainViewModel.kt Screen.kt UiKit.kt  # view-based UI; not in the manifest
  app/src/kotlinEngine/java/ai/liquid/pipette/
    EngineActor.kt                           # actor over LlamaBenchmark
    llama/LlamaBenchmark.kt                  # benchmark kernel in Kotlin
    llama/LlamaCpp.kt                        # JNI primitives, LlamaModel
  app/src/rustEngine/java/ai/liquid/pipette/
    EngineActor.kt                           # actor over LlamaEngine
    LlamaEngine.kt                           # external JNI stubs into the Rust crate

android/native/                              # C/C++ for both flavors; see its README.md
  llama_shim.cpp                             # C-ABI layer over llama.cpp and mtmd
  native_loader.cpp                          # runtime CPU-variant backend loader
  jni.cpp jni_crash_reporter.cpp             # kotlinEngine JNI surface
  benchmarks.rs                              # rustEngine benchmark kernel (compiled into the crate)
  crash_reporter.cpp                         # sentry-native crash capture in :benchmark

crates/pipette-android/                      # rustEngine only
  build.rs                                   # CMake link markers; stamps LLAMA_CPP_COMMIT
  src/lib.rs                                 # JNI entry points
  src/engine.rs                              # concrete Rust LlamaEngine
  src/loader.rs                              # backend loader registry
  src/llama.rs                               # llama.cpp C-FFI wrappers
  src/error.rs                               # PipetteError
```

## Build Configuration

`android/Pipette/app/build.gradle.kts`:

- Plugins: Android application, Kotlin Compose compiler, Kotlin serialization,
  ktfmt, and detekt. The Sentry Gradle plugin is applied only when the build has
  a Sentry DSN. The `room`/`ksp` coordinates in `libs.versions.toml` are unused:
  no applied plugin or dependency references them.
- Java/Kotlin 17 toolchain; `compileSdk = 36` (minor 1), `minSdk = 31`,
  `targetSdk = 36`.
- `ndk { abiFilters += "arm64-v8a" }`: single ABI.
- Product flavors on the `engine` dimension: `kotlinEngine` (default) and
  `rustEngine`. Each sets `BuildConfig.ENGINE_FLAVOR`.
- `buildFeatures`: `compose`, `viewBinding`, `aidl`, `buildConfig`.
- Key dependencies: AndroidX appcompat, core-ktx, core-splashscreen,
  activity-ktx and activity-compose, constraintlayout, lifecycle (runtime,
  viewmodel, and their Compose artifacts), DataStore preferences, WorkManager,
  Navigation 3 (runtime and UI), kotlinx-serialization, Material, coroutines,
  Compose BoM with material3, Sentry, PostHog (`posthog-android`, for product
  analytics), Tink (`tink-android`, for the Ed25519 signing identity), and Clerk
  (`clerk-android-api`).
- `CLERK_PUBLISHABLE_KEY` goes into `BuildConfig` from `local.properties` or a CI
  environment variable. The debug build type uses a separate dev key and falls
  back to the release key; see [build.md](build.md#clerk-sign-in).
  `POSTHOG_API_KEY` / `POSTHOG_HOST` are set the same way. Both default to the
  public production project: a `phc_…` project key is write-only, so the build
  contains it and does not treat it as a secret.
- Native build tasks, one per flavor:
  - `buildNativeAndroidArm64` runs `build-native-android.sh` and stages its
    output in `build/generated/kotlinJniLibs`. Every
    `mergeKotlinEngine*JniLibFolders` task depends on it.
  - `buildRustAndroidArm64` runs `build-rust-android.sh` and stages its output
    in `build/generated/rustJniLibs`. Every `mergeRustEngine*JniLibFolders` task
    depends on it.

  Each staged directory is added to its flavor's `jniLibs` source set.
  `jniLibs { useLegacyPackaging = true }` keeps the native libraries extracted
  on disk, because the CPU-variant loader scans that directory.

How the native build works is in [build.md](build.md#native-build-internals):
the per-CPU-variant backend dispatch, the shipped armv8/armv9 variant set (armv9
has a known ASR accuracy caveat from an upstream llama.cpp SVE configuration
mismatch), and the optional KleidiAI kernels.

## Analytics (PostHog)

Product analytics is a small, hand-picked set of events behind the `Analytics`
seam (`Analytics.kt`). The iOS client uses the same event and property names, so
both clients report to one PostHog project. `PostHogAnalytics.create` returns
`NoOpAnalytics` unless `PostHogConfiguration.isComplete`. Only the **main
process** calls it: the isolated `:benchmark` process must never load the SDK,
because anything resident there adds to the `max_memory_usage` it measures.

Autocapture, session replay, and feature-flag preloading are off, because
instrumentation overhead corrupts on-device LLM measurements. The Sentry Gradle
plugin's `tracingInstrumentation` is off for the same reason. Some SDK settings
need explicit values:

- **`sessionReplayConfig.captureLogcat = false`.** It defaults to **true**, and
  `PostHogLogCatIntegration.install()` checks only this setting, not
  `sessionReplay`. With `sessionReplay = false` alone, the SDK still starts a
  `logcat` subprocess and a reader thread for the life of the process.
- **`remoteConfig = false`.** At its default, the SDK sends
  `GET <host>/array/<key>/config` on every launch, even while opted out.
  Nothing in the app uses remote config. Since 3.58.0 the SDK marks this flag
  deprecated, with the text that remote config "is now always enabled". That
  text is wrong for posthog-android: `PostHog.setup` still branches on the flag,
  so `false` with no flag preloading sends no request. This was verified on
  3.58.0 in the source and on a device, where `/batch` was the only PostHog
  request of a launch. The `@Suppress("DEPRECATION")` is on the `config` local,
  because a Kotlin assignment cannot carry an annotation.
- **`surveys`, `errorTrackingConfig.autoCapture`, `setDefaultPersonProperties`,
  `capturePushNotificationOpened`, and `capturePushNotificationSubscriptions`
  are set to false.** The last three default to **true**:
  - `capturePushNotificationOpened` alone enables
    `PostHogActivityLifecycleCallbackIntegration` (its gate is
    `captureDeepLinks || captureScreenViews || sessionReplay ||
    capturePushNotificationOpened`).
  - `capturePushNotificationSubscriptions` fetches an FCM (Firebase Cloud
    Messaging) token at startup and registers it with PostHog.
  - `setDefaultPersonProperties` attaches `$app_version`, `$os_name`,
    `$device_type`, and similar properties to the *person* profile for local
    flag evaluation, which the app does not use.

**On every SDK version bump, compare which integrations `PostHogAndroid.setup`
installs, and under which flags.** `PostHogTouchActivityIntegration` has no
flag: the SDK always adds it, and its `install()` checks only the API level, so
every dispatched `MotionEvent` calls `PostHogSessionManager.touchSession()`. It
cannot be turned off without a fork. It is acceptable only because it is a
timestamp write on a path that is idle while a cell is timed. It is the same
kind of risk as `captureLogcat`, and the next one may cost more.

No event fires inside a measurement window. `job_started` is captured and
flushed *before* the runner starts, so the queue is empty when the run starts,
and the SDK's periodic flush timer has nothing to send while a cell is timed.
The one event that fires during a run is `results_submitted`, from the per-cell
upload that `JobRunner` makes between cells (PIP-358, as on iOS).
`ResultSubmissionService.submit` flushes right after it captures the event, so
the next cell also starts on an empty queue. Any capture site added between the
start and the end of a run must do the same. Events carry the registration
`clientId` as the distinct id. They never carry an email, organization, Clerk
identity, error message, or feedback body.

The Settings screen has a "Share anonymous usage analytics" toggle
(`SettingsScreen` → `SettingsIntent.SetAnalyticsOptOut` →
`Analytics.setOptedOut`). It shows only when `Analytics.isAvailable`, because a
toggle over `NoOpAnalytics` would control nothing.

`AnalyticsOptOutStore` owns the opt-out flag. `PostHogAnalytics.create` sets
`PostHogAndroidConfig.optOut` from it **before** `setup`. Keep this order, and
do not rely on the SDK's own persisted opt-out instead:

- posthog-android 3.19.0 did not apply its persisted opt-out at setup. With
  `opt-out=true` in the SDK's own `posthog-<key>.xml`, `PostHog.isOptOut()` read
  `false` right after `setup`, and the ingest endpoint accepted the next
  `app_launched`. This was measured on a Pixel 10 Pro Fold with Android 17.
- PostHog fixed this in 3.55.1, and the app uses 3.58.0, so the SDK's copy
  would probably work. The app sets the flag itself anyway, so the behavior does
  not depend on the SDK, and one launch is enough to leak an event from a device
  whose owner turned analytics off. On 3.58.0, an opted-out cold start queues
  zero events and sends zero requests of any kind.

The store is `SharedPreferences`, not `AppSettingsStore`, because this setting
must be readable **synchronously**: `PipetteApp.onCreate` builds the SDK and
captures `app_launched` in one non-suspending block, and a DataStore read cannot
complete in time.

With `remoteConfig` off, an opted-out launch makes no network request. iOS sets
the opt-out the same way, but posthog-ios ignores `config.remoteConfig`, so an
opted-out iOS device still sends the remote-config request. Android's flag has
the same deprecation text but still works, which is why the two platforms
differ here.

## Local Storage Model

All structured app data persists as filesystem JSON (`org.json`, written with
`writeText(json.toString(2))`). The internal-storage root is `filesDir/Pipette`:

```text
filesDir/Pipette/
  metadata/registration.json
  benchmarks/                                      # the synced benchmark catalog (FileBenchmarkStore)
  jobs/<jobId>/manifest.json                       # the cell array lives inside the manifest
  jobs/<jobId>/cells/<cellId>/payload.json
  jobs/<jobId>/cells/<cellId>/metrics.json
  jobs/<jobId>/cells/<cellId>/submission.json
```

Model files live in **external** app storage, in one directory per repo:

```text
<getExternalFilesDir(null)>/models/<hf-org>/<hf-repo>/<filename>.gguf
<getExternalFilesDir(null)>/models/<local-file>.gguf
```

Storage behavior:

- `LocalStorage` owns it all and implements the narrow `JobStore` interface that
  `JobRunner` uses (manifest read and write, `clearCellArtifacts`,
  `resolveModelPath`, `writePayload`, `loadRegistration`). Tests use an
  in-memory `FakeJobStore`.
- Cells are embedded in the manifest JSON. The per-cell directory holds only
  `payload.json`, `metrics.json`, and `submission.json`.
- `ModelStore` derives `ModelFile` metadata from a live scan of the models
  directory on each call.
- Repo directories: `modelRelativePath(repo, filename)` → `repo/filename`, or the
  bare `filename` when there is no repo. A delete removes empty `<org>/<repo>`
  directories.
- At startup, a legacy internal `Pipette/models` directory moves to external
  storage and is deleted. Internal storage holds models only when external
  storage is unavailable.
- The stale-path resolver (`resolveModelPath`) tries these in order:
  1. The stored path, if the file exists.
  2. The same path tail under the current models root.
  3. A unique file with the same name anywhere under the models root.

  It returns nothing when no file or more than one file matches.

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
`GgufFilename`, `HfRepo`, `Model`, `ClientId`, `BenchmarkId`, …) copied from the
Rust `pipette-plan-types` crate, with validation regexes that match its `nutype`
definitions. These are wire-format identifiers, separate from the on-disk
catalog.

## Registration And Signing

`RegistrationService` is a top-level class in `ManagementClient.kt` (next to
`ManagementClient`). `AppContainer` wires it.

Registration flow (`RegistrationService.register`):

1. The caller supplies the server URL, organization, contact email, an optional
   pre-auth key, and an optional Clerk identity.
2. `secrets.generatePendingSigningKeyPair()` creates an Ed25519 keypair with
   Tink, stores the raw 32-byte private key as hex in a *pending* slot, and
   returns the public key as hex.
3. `POST /clients/register` sends `public_key`, `organization`,
   `contact_email`, `client_details` (= `DeviceInfo.modelName()`),
   `device_cpu_arch` (omitted when unknown), `device_os_family`, and, when
   supplied, `client_preauth_key`. This call is **unsigned**.
4. On failure, the pending key is deleted and the error is rethrown.
5. On success, `promotePendingPrivateKey()` moves the pending key to the
   permanent slot. `RegistrationData` is built (with Clerk link metadata if a
   `clerkUserId` was supplied) and saved to `metadata/registration.json`. Clerk
   data is never sent to the server.

Every later management call is signed: result submissions, `PATCH /clients/me`,
and `GET /clients/me`. The headers:

- `X-Client-Id` = client id
- `X-Timestamp` = `Instant.now()` ISO-8601
- `X-Nonce` = `generateNonce()`, 16 `SecureRandom` bytes as lowercase hex
- `X-Signature` = Ed25519 signature over the `v1` payload, hex

The `v1` payload is six newline-separated fields:

1. `v1`
2. the uppercase HTTP method
3. the request target that the server receives (`URL.requestTarget()`, so the
   path prefix of the configured server URL and any query string are included)
4. the `X-Timestamp` value
5. the `X-Client-Id` value
6. the `X-Nonce` value

The method and target limit a captured signature to that method and target, and
the nonce makes it single-use, so the server rejects a replay inside the
5-minute freshness window. The signature does not cover the body.

Secret storage (`Secrets`): `SharedPreferences("pipette_secrets")`, with each
value encrypted with AES-256-GCM under an Android Keystore key and stored as
`<slot>_enc`. The slots are `private_key_hex` (the raw 32-byte Ed25519 seed),
the transient `pending_private_key_hex`, and `hf_token`. The public key comes
directly from Tink's keypair. Two in-place migrations run on first use:

- A legacy `private_key_pkcs8` value (base64 PKCS#8) converts to the raw hex
  seed on the first signed request. A leftover `pending_private_key_pkcs8` is
  discarded, because that registration never completed on the server.
- A legacy plaintext slot is encrypted. A plaintext slot stays readable while
  the Keystore is unavailable, so the device keeps its identity through a
  Keystore outage.

If the Keystore is unavailable, `Secrets` writes nothing. `Secrets`' KDoc has
the per-API-level reasons for Tink.

## Clerk Auth Gate

`ClerkAuth.kt` / `ClerkConfiguration.kt` implement the sign-in gate in front of
the app.

- `RealClerkAuth` derives a `ClerkState` (`Loading` / `InitError` / `SignedOut` /
  `SignedIn(userId, email?, sessionId?)`) from the Clerk SDK's init, user, and
  session flows. It only observes.
- `AuthGateScreen` (in `compose/shell/`) draws the sign-in UI, and
  `ShellViewModel` drives it through the Clerk SDK: an email code, a password
  (with reset), OAuth providers, and a second factor.
- A pure reducer `reduceAuthGate(clerk, registration, bypass)` maps the state to
  an `AuthGate` (`Loading` / `InitError` / `SignedOut` / `Mismatch` / `Ready`).
  A `SignedIn` user whose id differs from the registration's linked
  `clerkUserId` yields `Mismatch`. The debug-only `bypass` gives `Ready`.
- `PipetteAppRoot` draws only the gate until it reaches `Ready`. On `SignedIn`,
  `ShellViewModel.linkRegistrationIfNeeded` links the Clerk identity to the
  local registration. It leaves a link to a different user unchanged, so the
  gate shows the mismatch.
- Clerk is initialized **only in the main process**. `PipetteApp` checks
  `ProcessGate.isMainProcess` and `ClerkConfiguration.isComplete`, and it
  references no Clerk types directly. `ClerkBootstrap.create` holds the
  initialization, so only it class-loads `com.clerk.*`.
- The publishable key comes from `BuildConfig.CLERK_PUBLISHABLE_KEY`. When no
  key is configured, the SDK is never initialized and the gate is `Ready` in
  every build type. Key configuration is in
  [build.md](build.md#clerk-sign-in).

## Model Management

The Models tab (`compose/models/`, `ModelsViewModel`) supports:

- listing and searching the downloaded models, grouped by family
- showing MMProjector files
- deleting a model family and removing empty Hugging Face repo directories
- importing a local `.gguf` file through Android's document picker
- **Add models**: searching the bundled templates, filtering by quant, and
  downloading the selected families
- pausing, resuming, and canceling a download

The headless `model=` argument downloads a model by its exact coordinate before
a run (`ModelProvisioner`). The Compose UI has no field for a manual Hugging
Face identifier or for the Hugging Face token. The token comes from an
`auth-import` bundle (see
[architecture.md](architecture.md#carrying-an-identity-to-another-device)).

## Downloads (WorkManager)

`DownloadCoordinator` runs downloads as WorkManager work; `DownloadWorker` does
the transfer. Behavior:

- `DownloadCoordinator.parseDownloadInput` accepts `org/repo-GGUF:quant`
  (for example `LiquidAI/LFM2.5-350M-GGUF:Q4_0`), `org/repo-GGUF:file.gguf`,
  `org/repo/file.gguf`, and a full URL. It normalizes a Hugging Face `blob/`
  URL to `resolve/`, requires a `.gguf` file name, and URL-encodes each path
  segment.
- Each download is a unique work named by its `<repo>/<filename>` key. It writes
  `<dest>.part` and resumes it with an HTTP `Range` request. On success it moves
  the file to its final path and registers it with
  `LocalStorage.registerModelFile`.
- `DownloadWorker` follows redirects itself. It sends
  `Authorization: Bearer <token>` only to `huggingface.co` and its subdomains,
  never to a redirect target such as a CDN. It always sends
  `User-Agent: pipette-android`.
- It refuses a duplicate: a destination that exists, or a download already in
  progress.
- A network error keeps the `.part`. WorkManager retries with exponential
  backoff (30 s base), up to `MAX_NETWORK_ATTEMPTS` (5). The UI shows "Waiting
  for network" meanwhile. The work has no network constraint, because
  WorkManager would stop a running worker when the constraint fails.
- Pause and cancel are both WorkManager cancellations. Pause keeps the `.part`
  and the record; cancel deletes both.
- `DownloadStore` persists a record of each outstanding download in
  `SharedPreferences`. After process death, `DownloadCoordinator` reads the
  records and enqueues again the downloads that should run.
- `DownloadRegistry` is an in-process bus that carries progress from the worker
  to the UI. `DownloadLocks` prevents two workers from writing the same `.part`.
- `DownloadNotifications` posts one ongoing notification per download, with
  pause, resume, and cancel actions that `DownloadActionReceiver` handles.

## Model Templates

`ModelTemplates.kt` holds `ModelTemplateCatalog`, the bundled catalog of default
GGUF presets. Each `PresetModel` has a stable id, a display name, a detail label
(quant and size), and a Hugging Face identifier. The repo, file name, quant,
family id, and estimated size derive from these. The catalog also exposes
`repoToName`, `families`, and `byFamilyId`.

Most families have Q4_0, Q4_K_M, and Q5_K_M. Some have one quant: a family added
from a benchmark run that measured one quant has only that quant. A family's
quants can come from more than one repo: Ministral takes Q4_0 from `unsloth` and
its K-quants from `mistralai`. The catalog itself is the inventory, and
`ci/checks/model-catalog-parity.py` lists the rows that iOS offers and Android
does not.

## Quant Grouping And Job Model Selection

`ModelCatalog.kt` groups downloaded files into model families:

- keys by `familyId` from templates, falling back to a normalized filename stem
  (quant token and a leading `mmproj-` stripped, lowercased, so a model and its
  projector group together)
- summarizes available quant variants per family
- resolves selected family keys through the active quant filters
- detects selected families that have no file for the selected quant

`JobQuantFilter.kt` defines the four filters: `ALL`, `Q4_0`, `Q4_K_M`, `Q5_K_M`.
`ALL` excludes the specific filters, and the set is never empty. Users select
model families, then choose which quant variants run.

## Benchmark Catalog

`BenchmarkSync` pulls the catalog from the management server in two levels, each
conditional on an ETag:

1. `GET /benchmarks` returns the definitions. The sync keeps only entries that
   parse completely.
2. `GET /benchmarks/{id}` returns each kept benchmark with its eval samples.

A failed entry never stops the sync. `FileBenchmarkStore` stores the result
under `filesDir/Pipette/benchmarks/`. `ShellViewModel` runs the sync at launch
and after a registration change. `AppContainer` loads the stored catalog at
startup. The app has no bundled catalog, so the catalog is empty until the first
sync succeeds.

`BenchmarkDefinition.kt` defines `BenchmarkType` with seven kinds (by rank):
end-to-end latency, prefill throughput, decode throughput, max memory usage,
vision-language throughput, eval accuracy, and VLM eval accuracy.
`BenchmarkDefinition` mirrors `pipette-ops`. Each subclass models its parameters
and serializes to the flat, `benchmark_type`-tagged JSON that the engine and the
server use.

`BenchmarkCatalog` holds the synced list (`all`) and the New Job picker's subset
(`selectable`):

- `selectable` keeps the four ladder types (prefill, decode, end-to-end
  latency, max memory) whose required context is under 5000 tokens. VL and eval
  benchmarks stay in `all` and remain resolvable by id, but the picker does not
  offer them.
- Helpers: lookup by id, display names, type ordering, parameter labels, search,
  per-cell context sizing, and job context sizing (default 4096, VL at least
  8192).

VL gating happens in planning, not in the catalog.

## Job Orchestration

Three types share the work:

- **`JobController`**: the app-scoped owner. It constructs the single
  `JobRunner`, adapts its callback into a `StateFlow` for the UI, and forwards
  the control calls (`startNewJob`, `resume`, `retryFailed`, `rerunCells`,
  `cancel`).
- **`JobRunner`**: plans cells, persists the manifest through `JobStore`, and
  runs the execution loop on a single-thread executor. It talks to the engine
  through the `BenchmarkEngine` interface, gates on `ReadinessGate`, and submits
  through `ResultSubmitter`. It never calls native code directly.
- **`JobStore`**: the narrow persistence slice that `JobRunner` needs;
  `LocalStorage` implements it.

**Planning** (`planCells`): the Cartesian product of the selected models and
benchmarks. For `VL_THROUGHPUT`, a model is skipped unless it is VL-compatible
(an MMProjector with the same `hfRepo` or the same normalized stem); then the
plan has one cell per selected MMProjector. A non-VL benchmark gives one cell
with no MMProjector. Each cell records the benchmark id and type, the model path
and name, an optional MMProjector path, the run status, an optional server job
id, and an optional error. `startNewJob` requires at least one model, one
benchmark, and one planned cell, and starts the job in `RUNNING`.

**Execution**:

- runs on a single-thread executor and refuses a second job
- opens pocket mode (`BenchmarkActivity`) at run start, and holds
  `JobOrchestratorService` in the foreground in the main process for the run
- runs only `PENDING` cells, sorted by model path, then by per-cell context size
- `MAX_MEMORY_USAGE` cells use `runFreshSync`, because the load is part of the
  measurement. Other cells use `loadSync`, then `runBenchmarkSync`. Each cell
  loads its model fresh, sized to its benchmark's context. This matches the CLI
  clients, which start a fresh server per benchmark.
- resolves stale model and MMProjector paths. An unresolved path fails the cell
  with a re-download hint. After a load fails, later cells for the same path are
  skipped.
- publishes progress (fraction and text) to the UI. Cancel is cooperative,
  through `CancelFlag`, which the loop and the progress callbacks check. A
  cancelled cell becomes `CANCELLED`. `RemoteBenchmarkEngine` kills `:benchmark`
  if a decode does not honor a cancel within its grace window.
- applies the `ReadinessGate` before each cell
- fails the cell and stops the job when the `:benchmark` process lost the prime
  cores during the measurement (`lastRunInterruption`, checked on both the
  success and the error path). It calls `endJob` from the execution loop's
  `finally`, so the guard's carried baseline is dropped on every exit path. See
  [Cpuset Demotion Guard](#cpuset-demotion-guard).
- uploads finished results after each cell when the job contributes results,
  the device is online, and a registration exists. At the end of a `COMPLETED`
  job it uploads anything still unsent, without the online check.
  `contributeResults` is read from the latest manifest, so a change during the
  run applies.

**Rerun and recovery**: `resume` sets CANCELLED cells back to PENDING;
`retryFailed` reruns FAILED cells; `rerunCells` resets rerunnable cells to
PENDING and clears their artifacts. Cold-launch recovery runs once per process
from `AppContainer` (`recoverInterruptedJobs`) and turns an interrupted job into
a resumable PAUSED job.

## Native Benchmark Engine

The engine runs in the `:benchmark` process; the boundary design and its reasons
are in [architecture.md](architecture.md#process--runtime-boundary). This section
lists the concrete surface.

`BenchmarkEngine.kt` is the interface (`loadSync`, `runBenchmarkSync`,
`runFreshSync`, `unloadSync`, `llamaCppCommit`, `cpuBackendDescriptor`,
`benchmarkProcessCpuAffinity`, `lastRunInterruption`, `endJob`, `isAvailable`,
…). `benchmarkProcessCpuAffinity`, `lastRunInterruption`, and `endJob` serve the
[Cpuset Demotion Guard](#cpuset-demotion-guard). Two implementations exist:

- `EngineActor`, one per flavor, used inside the service and by tests. It runs
  every native call on a single `HandlerThread` behind an Empty/Ready/Busy state
  machine and owns the model lifecycle.
  - `kotlinEngine`: it calls `LlamaBenchmark` and `LlamaCpp`.
  - `rustEngine`: it calls `LlamaEngine`, whose methods are `external` JNI stubs
    into `crates/pipette-android`.
- `RemoteBenchmarkEngine`: the production implementation that `AppContainer`
  wires. It is a main-process client proxy that turns each `*Sync` call into a
  blocking two-way AIDL call to `PipetteBenchmarkService`. It:
  - binds on first use and starts the service in the foreground
  - links to death, so a service death reaches the cell loop as an error
  - stops the service 5 s after `unloadSync`, so the OS reclaims the native heap
  - kills `:benchmark` if a cancel gets no response within 12 s

IPC surface (`app/src/main/aidl/.../service/`):

- `IBenchmarkService.aidl`, blocking two-way: `loadModel`, `runBenchmark`,
  `runBenchmarkFresh`, `unloadModel`, `endJob`, `llamaCppCommit`,
  `cpuBackendDescriptor`, `benchmarkProcessCpuAffinity`, `lastRunInterruption`,
  `isAvailable`. One-way: `requestCancel`, `updateJobProgress`,
  `setJobCancelCallback`.
- `IBenchmarkRunCallback.aidl` (one-way `onProgress(completed, total, message)`).
- `IJobCancelCallback.aidl` (one-way `onCancelRequested`, from the pocket screen).
- `BenchmarkResult.aidl` (parcelable).

`ResultSpill` carries the result JSON across the boundary. A result of 256 KB or
less goes inline in the parcel. A larger result goes to a file in the shared
cache directory (both processes have the same UID), and the proxy deletes the
file after it reads it. This avoids `TransactionTooLargeException`.

The readiness gate between measured reps runs in the service and does not cross
the AIDL boundary. The main-process gate runs only between cells. See
[Readiness And Pocket Mode](#readiness-and-pocket-mode).

**`kotlinEngine` JNI surface** (`android/native/jni.cpp`):

- `NativeLib.nativeLlamaCppCommit`, `nativeLlamaCppTag`,
  `nativeCpuBackendDescriptor`
- `LlamaCpp.native*` primitives: model load and free, thread count, tokenize,
  decode batch, greedy sample, token to text, chat template, end-of-sequence id,
  KV-cache clear, sampler reset, RSS read, `nativeDecodeNGreedyIgnoreEog` (the
  timed decode loop, in C++), and the mtmd calls for vision-language cells
- `BenchmarkCrashReporter.nativeInit` (`android/native/jni_crash_reporter.cpp`)

**`rustEngine` JNI surface** (`crates/pipette-android/src/lib.rs`, on
`LlamaEngine` unless noted):

- `nativeLlamaCppCommit`, `nativeLlamaCppTag`, `nativeCpuBackendDescriptor`
  (also exported for `NativeLib`)
- `nativeCreate`: load a model; returns a boxed engine pointer
- `nativeDestroy`: free it
- `nativeRunBenchmark`: run against the loaded engine
- `nativeRunFresh`: load, measure, and unload (the `max_memory_usage` path)
- `BenchmarkCrashReporter.nativeInit`

Rust modules: `engine.rs` (the concrete `LlamaEngine`: one resident model and its
context size, `run_benchmark` delegating to the shared kernel, `run_fresh`,
`llama_cpp_commit`), `loader.rs` (a backend loader registry tried in priority
order), `llama.rs` (the in-process llama.cpp/mtmd C-FFI wrappers, error
classification, CPU-backend descriptor, RSS memory polling), and `error.rs`
(`PipetteError`).

If the shared library is not packaged, `RemoteBenchmarkEngine.isAvailable` is
false, the Jobs tab shows "Native benchmark engine missing", and cells fail with
an explicit error. The proxy finds the library with a class-loader path lookup
(`findLibrary`), so it checks that the library is packaged without loading it.
`NativeLib.isAvailable` does load the library, and the headless verbs use it in
the main process to report the build id.

## Readiness And Pocket Mode

`Readiness.kt` defines `ReadinessGate` and `Readiness`, a gate on
`PowerManager`:

- **primary**: `PowerManager.getThermalHeadroom`. The device is ready at a
  headroom of 0.85 or less. `AndroidThermalStatusProvider` samples headroom at
  most once per second and keeps the last good value, because the call returns
  `NaN` when polled faster.
- **fallback**, when headroom is unavailable: the coarse
  `PowerManager.getCurrentThermalStatus` value.

The default deadline is 300 s (`COOLDOWN_MAX_MILLIS`). A run can change it with
`RunReadiness` (see the headless `readiness_*` arguments in
[architecture.md](architecture.md#headless--automation-entry-point)).
`Readiness.kt` also defines `ThermalStatusProvider` /
`AndroidThermalStatusProvider` (status → labels). The gate runs between cells in
the main process and before each measured rep in `:benchmark`. While it waits,
the UI shows a cooldown status with the time waited.

Pocket mode is `BenchmarkActivity`, a full-screen run screen in `:benchmark`
(classic Views, so it adds no Compose memory to that process). `JobRunner` opens
it at run start; the Jobs screen's "Open in Pocket Mode" button opens it again.
It shows the job title, overall progress, cells done, the estimated time to
finish, the current cell and status, and a thermal chip. It keeps the screen on
(`FLAG_KEEP_SCREEN_ON`) and ignores Back. Its one exit, "Pause benchmark",
cancels the job and leaves the remaining cells resumable. `BenchmarkProgressBus`
carries the job snapshot from the main process. The Compose `PocketModeScreen`
in `compose/shell/` is a fallback for a job that is not running; no UI path
opens it.

## Cpuset Demotion Guard

A measurement is comparable only if it ran on the cores it expects. The
`:benchmark` process holds the `top-app` cpuset while a run is in flight. The
framework moves it out of `top-app` as soon as the app stops being the visible
foreground: when the screen turns off, and also when the notification shade, a
system dialog, or a biometric prompt takes window focus. Without a check, a
demoted run records a number measured under different scheduling as if it were
normal. On an S25 Ultra the error is large: the same benchmark id took 2.3 s
with the prime cores present and 207 s in `/moderate` without them.

**Detection** (`CpusetWatcher.kt`, running inside `:benchmark`, because cpuset
membership is per process):

- `CpuAffinityProbe` reads `/proc/self/cpuset` and `Cpus_allowed_list` from
  `/proc/self/status` into a `CpuAffinitySnapshot`, and derives the prime-core
  tier from per-core `scaling_max_freq`. An unreadable entry becomes null; it
  does not throw.
- `CpusetWatcher` polls at 1 Hz on a `THREAD_PRIORITY_BACKGROUND` thread and
  exits on the first confirmed demotion. The per-core frequency sweep runs once
  per process and is cached, which saves most of the cost: a poll takes ~430 µs,
  where a full snapshot takes ~1515 µs. That is **0.043% of one core**.
- `CpusetTracker` scores each sample against a baseline. Leaving `top-app` is
  the primary signal; losing a prime core and losing allowed CPUs are the
  others. Two consecutive confirmations filter out a momentary change. Leaving
  `top-app` is the primary signal, not a core count, because OEMs demote
  differently: a Pixel 10 Pro Fold drops to `/foreground` and keeps seven cores,
  while a Samsung S25 Ultra passes through `/foreground-boost` with all eight
  for ~11 s and then lands in `/moderate` with four and neither prime core.
- A third `Reason`, `NEVER_TOP_APP`, covers a run that never reaches `top-app`.
  The other two signals compare a sample with the best state so far, so a run
  that starts demoted and stays demoted takes that state as its baseline,
  compares equal on every poll, and reports a success measured without the
  prime cores. This case occurs with the screen off: a headless run's
  `BenchmarkActivity` resumes normally, but a dozing device grants
  `/foreground`, and the result is ~18% low.
- `NEVER_TOP_APP` needs evidence beyond the cpuset name. A process in the root
  cpuset reports `/`, which is neither null nor `top-app` and says nothing about
  scheduling. The signal fires only when the prime cores are actually denied
  (`excludesTopTier`), which derives from the allowed-CPU list. This leaves one
  deliberate blind spot: a device that denies `top-app` but allows every core,
  where only the *share* of CPU time drops, and that share cannot be measured
  portably.
- `NEVER_TOP_APP` also waits out a grace of three samples, because the
  `top-app` promotion arrives asynchronously after the pocket activity starts,
  so a healthy cell spends its first samples outside it. An unreadable sample
  holds the streak and does not clear it, so a failed `/proc` read cannot
  disable the guard.
- `CpuAffinitySnapshot` carries `sweptCpus` and `totalCpus`, because a
  *partial* `scaling_max_freq` sweep does not look unreadable. An offline core
  has no `cpufreq` directory and drops out silently, so the tier comes from the
  cores that were online. If the fast cores were parked, `topTierCpus` names a
  slower cluster that the process may use, and `excludesTopTier` reads false.
  `totalCpus` counts the `cpuN` directories, so a parked prime core still counts
  against the sweep. Every prime-tier comparison (the never-attained check, the
  `LOST_CPUS` tier clause, and the baseline comparison) requires a complete
  sweep first.
- `CellDemotionWatch` limits a watch to one cell. `beginCell` takes its
  baseline from a probe made before the model load, so the guard covers the load
  too. It carries the best state seen into the next cell, so a process that
  stays demoted cannot take its demoted group as the new normal. It carries the
  never-attained streak the same way, because the condition belongs to the run,
  not to one cell. `closeJob` clears both, so a new job does not inherit them.

**Reaction**: a confirmed demotion sets the run's `CancelFlag`. `JobRunner` then
reads `lastRunInterruption` (a **destructive** read, so one cell consumes one
verdict) and throws `RunInterruptedException`. The cell becomes `FAILED`, not
`CANCELLED`: a cancel means the user chose to stop, while a demotion means the
measurement is not valid. The job then stops, so the remaining cells do not run
under the same demotion. Analytics records a distinct `interrupted` outcome, so
runs that the device stopped are separate from runs that the user cancelled.

**`unloadModel` and `endJob`** are two different meanings of "done", and they
must stay separate. `JobRunner` unloads from its per-cell error path as well as
at the end of a job, so `unloadModel` only stops the watch and keeps the carried
baseline. `endJob` is the AIDL call that drops the baseline. The execution
loop's `finally` makes it once, at the job boundary. It is not `oneway`: a
one-way call could arrive after the next job's load had already set the
tracker's baseline.

**The part of pocket mode**: `BenchmarkActivity` lives in `:benchmark` so that
this process, not the main one, holds `top-app` during a run. It leaves pocket
mode only on a progress snapshot with `running=false`. That snapshot arrives
once, at the end of the run, so a process recreated after the job ended never
receives it. For that reason the activity finishes itself when it is restored
with no run in progress, so it does not hold `:benchmark` at `top-app`
indefinitely.

## Results

A successful cell writes a payload to
`jobs/<jobId>/cells/<cellId>/payload.json`. It contains the engine result JSON,
the job id, the cell id, model metadata, benchmark metadata, runtime metadata,
Android device metadata (from `DeviceInfo`), and the context-size and GPU-layer
settings.

The payload also records where inference ran, as `device_android_cpuset`,
`device_android_cpu_affinity_list`, and
`device_android_cpu_affinity_excludes_top_tier`. These are read fresh for each
cell, because the cpuset group changes with process state. A value cached per
process would report the first cell's group for every cell, which is worse than
no value, because it looks like evidence that the cell ran normally. A demoted
cell normally writes no payload, because the [guard](#cpuset-demotion-guard)
fails it first. These fields are what remains when the guard is not active,
which is the case when the service is not bound.

## CSV Export

`CompletedResultsCsvExporter.kt` builds the CSV **string** and a suggested file
name. `PipetteAppRoot` writes the file through Android's create-document flow of
the Storage Access Framework (SAF).

The CSV has 39 columns (`CompletedResultsCsvExporter.headers`):

- job: id, title, created
- cell id
- model: name, display name, quant
- benchmark: id, type, name, parameters
- status
- primary metric: name, value, unit, display value
- submitted-at, server job id
- runtime: name, version, flags, benchmark flags, CPU variant, thread count
- device: name, form factor, OS name, OS version, OS build, OS security patch,
  chip, RAM, battery level, power state, power-save mode
- cpuset: `device_android_cpuset`, affinity list, excludes-top-tier
- error message

Primary metric per type: prefill and decode throughput (tok/s from tokens ÷
time), end-to-end latency (ms, lower is better), max memory (bytes, lower is
better), and VL throughput (tok/s from prompt and predicted timing). Rows are
ordered by model group, then quant, then benchmark column. A value that
contains a comma, a quote, or a line break is quoted, with inner quotes doubled. The heatmap shading uses a column normalization that knows which
direction is better (`ResultsGrid.heatmapIntensity`).

## Result Submission

`ResultSubmissionService.kt` implements `ResultSubmitter`:

- submits all completed, unsubmitted cells in a batch, or a single cell
- recovers earlier submission records (it fills `serverJobId` from a
  `submission.json` marked submitted)
- skips cells without payloads; checks each result `index` in the server
  response against the batch and flags omitted results
- persists submitted server job ids on the manifest, and records failed
  submissions
- uses a fixed `DEFAULT_BATCH_SIZE = 1000`

## Main UI

The UI is Jetpack Compose with one view model per screen:

- `PipetteApp` (Application) is the process entry point and owns `AppContainer`.
  It initializes Clerk and PostHog only in the main process.
- `AppContainer` is a hand-written DI container that wires the app-scoped
  singletons (`LocalStorage`, `AppSettingsStore`, `Secrets`, `ManagementClient`,
  `RegistrationService`, `DownloadCoordinator`, `ResultSubmissionService`,
  `RemoteBenchmarkEngine`, `JobController`, the readiness gate, the benchmark
  store) and runs `recoverInterruptedJobs()` once per process.
- `ComposeMainActivity` is the launcher. It requests the notification
  permission and sets `PipetteAppRoot` as its content.
- `PipetteAppRoot` shows the auth gate until it is `Ready`, then `SetupScreen`
  until the device has a registration, then the tabbed screens. It also holds
  the document launchers (model import, CSV export) and keeps the screen on
  while a job runs.
- `ShellViewModel` holds the app-wide state: the auth gate and sign-in flow, the
  selected tab, the registration, the runner state, and the catalog sync. The
  screen view models extend `ScreenViewModel` (one `state` flow, one `onIntent`
  entry point, one effects channel) and reach each other only through
  `ShellViewModel`.
- Navigation 3 (`compose/nav/Route.kt`) holds the back stack. The top-level
  routes are `Jobs`, `Models`, and `Settings`, selected from a floating tab bar;
  the default tab is Jobs. `AddModels`, `CreateJob`, `JobDetail`, and
  `CellDetail` are pushed on top. The view models own the state, and
  `PipetteAppRoot` mirrors it onto the back stack.

Screens:

- **Setup** (`SetupScreen`): the registration form, with the collector, the
  organization name, the email, and an optional pre-auth key.
- **Models** (`ModelsScreen`, `AddModelsScreen`): see
  [Model Management](#model-management).
- **Jobs** (`JobsScreen`, `CreateJobScreen`, `JobDetailScreen`,
  `CellDetailScreen`): the job list; the new-job wizard (`NewJobWizard` steps
  Models → Benchmarks → Review, with family and quant selection, MMProjector
  selection, run settings, and a missing-quant warning); and the job detail
  (status counts, live progress, the auto-submit toggle, resume, retry, rerun,
  CSV export, submit, rename, delete, and the results heatmap).
- **Settings** (`SettingsScreen`): account (sign out), the contribute default,
  the thermal state, feedback, open source licenses, analytics opt-out, reset of
  local data, and, in debug builds, the auth-gate bypass and the thermal-gate
  skip.

`MainActivity`, `MainViewModel`, `Screen.kt`, `UiKit.kt`, and the top-level
`*Screen.kt` files are a view-based UI that the app does not use: the manifest
does not declare `MainActivity`, so this code does not run.

## Tests

Unit tests are in `app/src/test` (about 60 classes, shared by both flavors) and
`app/src/testKotlinEngine` (the Kotlin kernel: `LlamaBenchmarkTest`,
`LlamaBenchmarkBatchSizeTest`, `LlamaCppTest`). Fakes and fixtures are in
`app/src/test/.../fakes/` (`FakeBenchmarkEngine`, `FakeJobStore`,
`FakeReadinessGate`, `FakeResultSubmitter`, `FakeClerkAuth`,
`InMemoryBenchmarkStore`, `CpuAffinityFixtures`, …). Representative suites:

- auth and identity: `AuthGateTest`, `EmailAuthFlowTest`, `SecretsTest`,
  `IdentityBundleTest`, `ManagementClientTest`, `RegisterBodyTest`
- catalog and models: `BenchmarkCatalogTest`, `BenchmarkSyncTest`,
  `ModelCatalogResolveTest`, `ModelUriTest`, `DownloadRetryTest`
- jobs: `JobRunnerTest`, `JobRunnerReadinessTest`, `JobRunnerProvisioningTest`,
  `NewJobWizardTest`, `ResultsGridTest`
- headless: `headless/*Test` (command parsing, verbs, caller guard, result codes)
- readiness: `ReadinessTest`, `thermal/ThermalTest`

The [cpuset guard](#cpuset-demotion-guard) is covered by:

- `CpusetTrackerTest` (scoring and confirmations)
- `CpusetWatcherTest` (the poll loop and how it stops)
- `CellDemotionWatchTest` (per-cell scope, the carried baseline and
  never-attained streak, the clear at the job boundary, and the destructive
  verdict read)
- `CpusetDemotionJsonTest` (the AIDL round trip)
- `BenchmarkActivityRestoreTest` (Robolectric; a restored pocket screen
  finishes, a live one does not)
- `JobRunnerDemotionTest` (the cell fails and is not cancelled; the baseline
  drops at the job boundary and not per cell) and `JobRunnerTest` (each payload
  carries its own cpuset reading)

Instrumented tests run on a device or emulator:

- `app/src/androidTest`: `RemoteEngineIsolationTest` (process isolation),
  `EngineMemoryBaselineTest` (native-heap reclamation), `KleidiaiBenchTest`
  (KleidiAI A/B), `CpusetWatcherCostTest` (the guard's per-poll cost),
  `LocalStoragePayloadInstrumentedTest`, `ThermalTelemetrySmokeTest`
- `app/src/androidTestKotlinEngine`: `LlamaBenchmarkInstrumentedTest`,
  `LlamaCppCheckJniTest`

Build and test commands are in [build.md](build.md#how-to-build).

## Known Gaps

The app runs end to end, but it is not yet fully proven on physical hardware.
The remaining work is mostly validation:

- validate live registration and result submission against the management server
- validate model and template downloads through WorkManager on a device,
  including large files, cancellation, process restart, and downloads that need
  a Hugging Face token
- validate thermal and readiness behavior on physical devices (emulators do not
  throttle realistically)
- expand native ABI support if non-arm64 devices matter

## Useful Commands

The build, install, launch, and device-inspection commands are in
[build.md](build.md), with the crash-log and app-file commands under
[Inspecting devices](build.md#inspecting-devices).
