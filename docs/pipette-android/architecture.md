# Pipette Android Architecture

The Android app in `android/Pipette/` is a native Kotlin implementation of the
Pipette mobile client. Behind a Clerk sign-in gate it runs the full local-device
benchmark workflow: device registration, model management, model templates,
benchmark planning, job execution, result persistence, CSV export, and result
submission.

This is the **front door** for the Android client; what the app is, how it's
structured, and why the major pieces are built the way they are. The other two
docs go deeper:

- **Building, installing, and running** the app, including native-build
  internals (per-CPU-variant dispatch, KleidiAI), is in [build.md](build.md).
- The **implementation ledger** is in [implementation.md](implementation.md):
  every file, the full per-subsystem feature list, the test suite, and known
  gaps.

The app is intentionally pragmatic: the UI is programmatic Android Views (not
Compose, apart from one island that hosts Clerk's sign-in screen) driven by a
small design system in `UiKit.kt`. The priority is working behavior over final
visual polish.

## Glossary

The terms below recur throughout these docs; the rest of this page assumes them.

- **Benchmark**: one measurement definition (e.g. prefill throughput at a given
  token count). The bundled set lives in `BenchmarkCatalog`.
- **Cell**: the atomic unit of work, a single *(model × benchmark)* pairing
  (plus one MMProjector for vision-language cells). A job is a set of cells; each
  cell runs once and produces its own result artifacts.
- **Job / manifest**: a job is a planned batch of cells produced by taking the
  Cartesian product of the selected models and benchmarks. The **manifest**
  (`manifest.json`) is the job's persisted record. It embeds the cell array and
  the status counts.
- **GGUF**: the llama.cpp on-disk model file format. Models are downloaded or
  imported as `.gguf` files.
- **Quant / quant family**: the quantization of a GGUF model (`Q4_0`, `Q4_K_M`,
  `Q5_K_M`, …). Downloaded files are grouped into model *families*; job setup
  selects families and then resolves them through the active quant filters.
- **MMProjector (mmproj)**: a multimodal projector file paired with a base model
  to enable vision-language benchmarks. A VL cell exists only when a compatible
  base model and a selected MMProjector are both present.
- **VL (vision-language)**: the multimodal benchmark type (vision-language
  throughput); requires a base model plus an MMProjector.
- **`:benchmark` process**: the isolated Android process that hosts the native
  engine. It is the *only* place `libpipette_android.so` is loaded. See
  [Process & Runtime Boundary](#process--runtime-boundary).
- **AIDL**: Android's IPC interface language. The main process calls into the
  `:benchmark` process over a blocking two-way AIDL interface.
- **Shim**: the C/C++ glue compiled into `libpipette_android.so` that the Rust
  bridge calls to reach llama.cpp (backend init, the CPU-variant loader, the
  shared benchmark kernel).
- **Spill (`ResultSpill`)**: how a benchmark result crosses the AIDL boundary.
  Small JSON rides inline in the parcel; larger results spill to a file in the
  shared cache dir to avoid `TransactionTooLargeException`.
- **Readiness gate**: the thermal/device-readiness check that pauses between
  cells (and between measured reps) until the device is cool and settled enough
  to measure. See [Readiness](#readiness-thermal-cooldown-and-pocket-mode).
- **Plan-types**: wire-format identifiers (`HfRepo`, `Model`, `ClientId`,
  `BenchmarkId`, …) mirrored in `ModelRef.kt` from the Rust `pipette-plan-types`
  crate; distinct from the on-disk model catalog.

## Components

| Component | File(s) | Responsibility |
|---|---|---|
| App shell | `MainActivity.kt` | Thin Activity: header + bottom-nav chrome, document launchers, hosts the auth gate and Pocket Mode; delegates each tab body to a `Screen` |
| UI state (MVVM) | `MainViewModel.kt` | Transient UI state that survives rotation; drives an imperative re-render tick; owns the auth-gate flow |
| DI & process entry | `AppContainer.kt`, `PipetteApp.kt` | Manual (no Hilt) app-scoped wiring; process entry; cold-launch job recovery |
| Screens | `Screen.kt`, `SetupScreen.kt`, `ModelsScreen.kt`, `JobsScreen.kt`, `SettingsScreen.kt` | Per-tab view-controllers (Setup / Models / Jobs / Settings) |
| Shared UI | `UiKit.kt`, `UiFormatting.kt`, `NewJobWizard.kt`, `ResultsGrid.kt` | Programmatic design system, formatting helpers, job-wizard step model, results-heatmap shading |
| Auth gate | `ClerkAuth.kt`, `ClerkConfiguration.kt` | Clerk sign-in gate in front of the whole app; observe-only seam; main-process only |
| Local storage | `LocalStorage.kt`, `DataModels.kt` | Filesystem-JSON jobs/cells/payloads/submissions/registration; the data models |
| Settings store | `AppSettingsStore.kt` | DataStore-backed setup defaults + contribute/auto-submit default |
| Model catalog | `ModelStore.kt`, `ModelCatalog.kt`, `ModelTemplates.kt`, `ModelRef.kt` | Live model-directory scan; quant-family grouping; bundled templates; plan-type identifiers |
| Downloads | `DownloadCoordinator.kt` | `DownloadManager`-backed GGUF downloads |
| Benchmarks | `BenchmarkCatalog.kt`, `BenchmarkDefinition.kt` | Bundled benchmark catalog, benchmark types, context sizing |
| Jobs | `JobController.kt`, `JobRunner.kt`, `JobStore.kt`, `JobQuantFilter.kt` | Lifecycle owner + UI state; planning/execution; persistence seam; quant filters |
| Benchmark engine | `BenchmarkEngine.kt`, `EngineActor.kt`, `LlamaEngine.kt`, `service/*` | Engine interface; out-of-process AIDL engine in `:benchmark`; JNI stubs |
| Registration & secrets | `ManagementClient.kt`, `Secrets.kt` | Registration + signed management requests; Ed25519 key + HF token |
| Submission & export | `ResultSubmissionService.kt`, `CompletedResultsCsvExporter.kt` | Result upload; CSV string builder |
| Readiness | `Readiness.kt` | Two-tier device cooldown/thermal gate + thermal-status labels |
| Device info | `DeviceInfo.kt` | Device labels and payload metadata |
| Native bridge (Rust) | `crates/pipette-android/src/{lib,engine,loader,llama,error}.rs`, `native_loader.cpp` | JNI entry points into the shared llama.cpp benchmark path; CPU-variant loader |

For the full directory layout, see
[implementation.md](implementation.md#project-layout).

## Process & Runtime Boundary

The app runs in **two processes**, and the most important boundary in the
codebase is the line between them:

- The **main process** owns everything Kotlin: UI, the auth gate, local storage,
  job orchestration, registration, downloads, and submission.
- An isolated **`:benchmark` process** owns the native engine: the only place
  `libpipette_android.so` and llama.cpp are ever loaded.

A benchmark call crosses that boundary over **AIDL**:

1. `JobRunner` (main process) calls a blocking `*Sync` method on the
   `BenchmarkEngine` interface. The wired implementation is
   `RemoteBenchmarkEngine`, a client proxy.
2. `RemoteBenchmarkEngine` turns each call into a blocking two-way AIDL
   round-trip to `PipetteBenchmarkService` (declared `android:process=":benchmark"`),
   binding lazily and running it as a foreground service so a run survives
   backgrounding.
3. The service hands the call to `EngineActor`, which serializes every native
   call onto a single worker thread behind an idle/ready/busy state machine.
4. `EngineActor` calls `LlamaEngine` (Kotlin `external` JNI stubs), which enters
   Rust `lib.rs` → `engine.rs` → `llama.rs` → the llama.cpp shim compiled into
   the `.so`.

**Why a separate process** (the reason is memory, not just crash isolation): a
loaded model holds a multi-gigabyte native heap. When a job finishes,
`RemoteBenchmarkEngine` tears the service down and the OS reclaims that process's
memory immediately, rather than leaving it parked in a cached process. The same
teardown bounds cancellation; a hard-kill watchdog kills `:benchmark` if a
running decode ignores cooperative cancel. A side benefit is that the lean
`:benchmark` process never class-loads the Clerk SDK.

Two details that follow from the boundary:

- **Result transport.** Benchmark result JSON returns via `ResultSpill`: small
  payloads ride inline in the parcel; larger ones spill to a file in the shared
  cache dir to avoid `TransactionTooLargeException`. (The exact threshold is in
  [implementation.md](implementation.md#native-benchmark-engine).)
- **Mid-run readiness.** The cooldown/readiness gate between measured reps runs
  *inside* `:benchmark`; the UI process only applies its own readiness gate
  *between cells*. The per-rep callback is intentionally not forwarded over IPC.

The native library is built per ARM CPU-variant and selected at runtime by a
loader: all within the single `arm64-v8a` ABI the app ships, not multiple
architectures; see [build.md](build.md#native-build-internals).

## Design Choices

### Native Kotlin app

A normal Android application; not KMP, not React Native. That keeps it close to
the local-device benchmark model: the app owns local files, local metadata,
download handoff, native code loading, and Android device APIs directly.

### Manual DI + MVVM, programmatic Views

`AppContainer` is a hand-written, app-scoped dependency container (deliberately
no Hilt). It is app-scoped so that Activity recreation never tears down a running
benchmark. `MainViewModel` holds transient UI state across rotation and drives an
imperative full-rebuild render loop: a pragmatic step out of the original
single-Activity monolith rather than a move to Compose. The UI is programmatic
Views; the only Compose in the app is a retained island hosting Clerk's prebuilt
`AuthView`.

### Filesystem JSON + DataStore

Structured records (job manifests, per-cell payloads and submissions,
registration) are persisted as JSON files; simple preferences live in Jetpack
DataStore (`AppSettingsStore`). Downloaded-model metadata is derived on demand by
`ModelStore` from a live scan of the model directory, so the filesystem is the
single source of truth.

### Android DownloadManager for GGUF downloads

GGUF downloads are handed to Android's `DownloadManager` (`DownloadCoordinator`),
the standard system service for long-running, user-visible downloads. The app
still tracks active downloads itself so the UI can show progress, prevent
duplicates, survive process restart, and register the finished model.

### Out-of-process native engine

Covered above under [Process & Runtime Boundary](#process--runtime-boundary):
the engine runs in `:benchmark` behind an AIDL interface, primarily so a model's
native heap is reclaimed by killing the process at job end.

### Clerk auth gate as an observe-only seam

The app gates on Clerk sign-in (see [Auth gate](#auth-gate)). The Clerk SDK is
kept behind a thin seam (`ClerkAuth`) that only *observes* SDK state and is
class-loaded in the main process only, so the gate's reducer is unit-testable
off-device and the `:benchmark` process stays lean.

## Local State

Storage is split into two tiers by size and durability: **structured records**
(job manifests with their embedded cells, per-cell payloads/metrics/submissions,
and registration) persist as JSON in the app's **internal** storage, while
**model files** (multi-gigabyte GGUFs) live in **external** app storage. Simple
preferences use Jetpack DataStore (`AppSettingsStore`). The filesystem is the
single source of truth: `ModelStore` derives model metadata from a live
directory scan rather than persisting it.

`LocalStorage` owns the internal layout but exposes only the narrow `JobStore`
slice that `JobRunner` consumes, so tests can swap an in-memory fake. Because a
manifest can outlive a model's exact path, it resolves a stale model path before
each run: preferring the recorded path, then recovering the same tail under the
current models root, and rejecting ambiguous matches so a job never benchmarks
the wrong repo's bytes.

For the exact on-disk tree, the repo-bucketing scheme, and the legacy-directory
migration, see [implementation.md](implementation.md#local-storage-model).

## Registration And Secrets

Registration is implemented by `RegistrationService` and `ManagementClient` (both
in `ManagementClient.kt`).

- `Secrets.generatePendingSigningKeyPair()` creates an Ed25519 keypair; the
  pending private key is promoted only after the server accepts registration.
  Keygen and signing go through **Tink**, not the JCA: no Android below API 37
  can produce an exportable Ed25519 key through the JCA, which made registration
  impossible on Android 12–16. `Secrets`' KDoc has the per-API-level breakdown.
- `POST /clients/register` is itself **unsigned** (it carries the public key).
  Later management calls (result submissions) are signed with the shared
  Pipette contract: `X-Client-Id`, `X-Timestamp`, `X-Nonce`, and an
  `X-Signature` that is an Ed25519 signature over the `v1` payload. That payload
  is six newline-separated fields: `v1`, the HTTP method, the request target the
  server receives, the timestamp, the client id, and the nonce. Signing them
  scopes the signature to that method and target, and the nonce makes it
  single-use, so a captured signature cannot be replayed. The body is still not
  covered.
- The private key (raw 32-byte seed, hex) and the Hugging Face token are stored
  in plain `SharedPreferences` via `Secrets`: not EncryptedSharedPreferences or
  the Keystore. The hex seed matches what iOS keeps in the Keychain and what the
  Rust CLI writes under `identity/`. Installs predating the Tink switch hold a
  base64 PKCS#8 blob, converted in place on the first signed request: reads
  that only test for a key's presence deliberately leave it untouched.
- `RegistrationData` carries optional Clerk link metadata (`clerkUserId`,
  `clerkSessionId`, `clerkPrimaryEmail`, `clerkLinkedAt`), which is persisted
  locally and never sent to the management server.

## Subsystems

Each subsystem is described here at the design level; implementation.md has the
full feature list for each.

### Auth gate

Clerk sign-in is the outermost gate: `MainActivity.render()` shows the gate and
draws nothing else until the gate reaches `Ready`. `ClerkAuth` observes the SDK
and a pure reducer (`reduceAuthGate`) maps Clerk state to an `AuthGate`
(`Loading` / `InitError` / `SignedOut` / `Mismatch` / `Ready`). Signing in links
the Clerk identity onto the local registration; if the signed-in user differs
from the already-linked one the gate shows `Mismatch`. A debug-only bypass exists.
Clerk is initialized only in the main process, and its publishable key comes from
`BuildConfig` (set from `local.properties` or a CI env var).

### Model management

The Models tab scans app storage, imports local GGUF files through the document
picker, downloads from Hugging Face (manual identifier/URL or default templates),
and deletes models (pruning empty repo buckets). `DownloadCoordinator` drives
downloads through `DownloadManager`: parsing HF shorthand and full URLs,
bucketing destinations by repo, attaching the HF bearer token, and surviving
process restart. See
[implementation.md](implementation.md#downloadmanager-integration).

### Model templates and quant grouping

`ModelTemplates.kt` is the bundled catalog of default GGUF presets (stable id,
display name, quant/size label, HF identifier, repo, family id, estimated bytes).
`ModelCatalog` groups downloaded files into families by `familyId` (falling back
to a normalized stem for sideloaded files), so job setup selects model *families*
and then resolves them through the active quant filters (All, Q4_0, Q4_K_M,
Q5_K_M). The review screen warns when a selected family has no downloaded model
matching the active filter.

### Benchmark catalog

`BenchmarkCatalog` bundles the local benchmark definitions. There are six
benchmark types (end-to-end latency, prefill throughput, decode throughput, max
memory usage, vision-language throughput, and eval accuracy) though the bundled,
user-selectable catalog covers the first five (a token ladder plus a fixed VL
set). It also provides search, retired-benchmark filtering, type ordering, and
per-cell context sizing. Vision-language cells are created only when a compatible
base model and a selected MMProjector exist.

### Job planning and execution

Job orchestration is split: `JobController` is the app-scoped owner that wraps
the runner and exposes its state to the UI as a `StateFlow`; `JobRunner` plans
cells and drives the execution loop on a single-thread executor; `JobStore`
persists. Planning is a Cartesian product of models × benchmarks (plus
MMProjector expansion for VL), with per-benchmark context sizing. Execution runs
only `PENDING` cells (sorted by model path and context size), resolves stale
paths, dispatches each cell to the `:benchmark` process via the engine proxy,
publishes progress, applies the readiness gate between cells, and auto-submits a
completed job when it opted into contribution. It exposes resume / retry / rerun
controls, and cold-launch recovery converts an interrupted `RUNNING`/`PAUSED`
manifest into resumable state. (Cells are reloaded per run rather than reusing a
resident model.)

### Readiness (thermal cooldown) and Pocket Mode

`Readiness` is a two-tier device-readiness gate: it prefers a native
`pipette_readiness` probe (OS thermal status + die temperature + CPU load)
and falls back to `PowerManager` thermal headroom / status when the probe can't
read its inputs. It gates between cells in the UI process and between measured
reps inside `:benchmark`. `MainActivity`'s Pocket Mode is a full-screen active-job
view that keeps the screen on and shows live progress, elapsed/remaining time,
the current cell, and a thermal chip.

### Results and submission

Completed cells write `payload.json` in the per-cell artifact directory.
`ResultSubmissionService` handles batch and single-cell submission, recovery from
existing `submission.json` records, persisted server job IDs, and server-response
index validation. `CompletedResultsCsvExporter` builds the CSV string (job,
model, benchmark, primary-metric, runtime, and device columns); the actual file
write goes through Android's create-document flow in `MainActivity`.
