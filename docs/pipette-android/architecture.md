# Pipette Android Architecture

The Android app in `android/Pipette/` is a native Kotlin client for Pipette. It
runs the local-device benchmark workflow: device registration, model
management, model templates, benchmark planning, job execution, result
persistence, CSV export, and result submission. A Clerk sign-in gate can sit in
front of the app (see [Auth gate](#auth-gate)).

This page tells what the app is, how it is structured, and why. Three other
docs give more detail:

- [build.md](build.md): build, install, and run the app, and the internals of
  the native build (per-CPU-variant dispatch, KleidiAI).
- [implementation.md](implementation.md): where each piece lives and how it
  works. It holds the source layout, the details of each subsystem, the test
  suite, and the known gaps.
- [engine-port.md](engine-port.md): the record of the port from the Rust engine
  to the Kotlin engine, with the benchmark invariants that both engines keep.

## Glossary

The terms below recur throughout these docs; the rest of this page assumes them.

- **Benchmark**: one measurement definition (e.g. prefill throughput at a given
  token count). The catalog comes from the management server, and
  `BenchmarkCatalog` holds the synced copy.
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
- **Main process**: the app's default process. It holds the UI, storage, and job
  orchestration.
- **`:benchmark` process**: the isolated Android process that hosts the native
  engine and the model's memory. See
  [Process & Runtime Boundary](#process--runtime-boundary).
- **AIDL (Android Interface Definition Language)**: Android's IPC interface
  language. The main process calls into the `:benchmark` process over a blocking
  two-way AIDL interface.
- **Engine flavor**: one of the two Gradle product flavors, `kotlinEngine` (the
  default) and `rustEngine`. Each flavor has its own benchmark kernel. See
  [Engine flavors](#engine-flavors).
- **Shim**: `android/native/llama_shim.cpp`, the C-ABI layer over the llama.cpp
  and mtmd C APIs that both engine flavors link. It initializes the backend
  through the CPU-variant loader and holds the performance-core threadpool hooks.
- **Spill (`ResultSpill`)**: how a benchmark result crosses the AIDL boundary.
  Small JSON goes inline in the parcel. A larger result goes to a file in the
  shared cache directory, which avoids `TransactionTooLargeException`.
- **Readiness gate**: the thermal check that pauses before each cell and before
  each measured rep until the device is cool enough to measure. See
  [Readiness](#readiness-thermal-cooldown-and-pocket-mode).
- **Plan-types**: wire-format identifiers (`HfRepo`, `Model`, `ClientId`,
  `BenchmarkId`, …) mirrored in `ModelRef.kt` from the Rust `pipette-plan-types`
  crate. They are separate from the on-disk model catalog.

## Components

| Component | Main types | Responsibility |
|---|---|---|
| App shell | `ComposeMainActivity`, `PipetteAppRoot`, `ShellViewModel` | Launcher activity; shows the auth gate, the setup gate, or the tabbed screens; holds the gate and tab state |
| Screens | `JobsViewModel`, `ModelsViewModel`, `SettingsViewModel`, `SetupViewModel`, `Route` | One Compose view model per screen; a Navigation 3 back stack for the Jobs, Models, and Settings tabs and their detail screens |
| DI and process entry | `AppContainer`, `PipetteApp` | Hand-written, app-scoped dependency container; process entry; cold-launch job recovery |
| Auth gate | `ClerkAuth`, `ClerkConfiguration`, `reduceAuthGate` | Clerk sign-in gate; observe-only seam; main process only |
| Local storage | `LocalStorage`, `JobStore` | Filesystem JSON for jobs, cells, payloads, submissions, registration, and the synced catalog |
| Settings store | `AppSettingsStore` | DataStore preferences |
| Model catalog | `ModelStore`, `ModelCatalog`, `ModelTemplateCatalog`, `ModelRef` | Live scan of the model directory; quant-family grouping; bundled templates; plan-type identifiers |
| Downloads | `DownloadCoordinator`, `DownloadWorker` | Resumable GGUF downloads as WorkManager work |
| Benchmarks | `BenchmarkSync`, `BenchmarkCatalog`, `BenchmarkDefinition` | Catalog sync from the server; the New Job picker filter; context sizing |
| Jobs | `JobController`, `JobRunner`, `JobOrchestratorService` | Lifecycle owner and UI state; planning and execution; a main-process foreground service for the length of a job |
| Benchmark engine | `BenchmarkEngine`, `RemoteBenchmarkEngine`, `PipetteBenchmarkService`, `EngineActor` | Engine interface; AIDL client proxy; the `:benchmark` service; single-thread native-call serializer |
| Engine kernel | `kotlinEngine`: `LlamaBenchmark`, `LlamaCpp`; `rustEngine`: `LlamaEngine`, `crates/pipette-android` | The benchmark kernel of each flavor (see [Engine flavors](#engine-flavors)) |
| Pocket mode | `BenchmarkActivity` | Run screen in `:benchmark` that keeps that process in the `top-app` cpuset |
| Cpuset guard | `CpusetWatcher`, `CellDemotionWatch` | Detects a run that loses the `top-app` cpuset or the prime cores |
| Registration and secrets | `RegistrationService`, `ManagementClient`, `Secrets` | Registration and signed management requests; the Ed25519 key and the Hugging Face token, encrypted at rest |
| Submission and export | `ResultSubmissionService`, `CompletedResultsCsvExporter` | Result upload; CSV string builder |
| Readiness | `Readiness` | Thermal gate and thermal-status labels |
| Headless entry | `HeadlessRunActivity` | Intent-driven runs and verbs for automation |
| Analytics | `Analytics` | PostHog product analytics, main process only |

For the source layout, see
[implementation.md](implementation.md#project-layout).

## Process & Runtime Boundary

The app runs in **two processes**. The line between them is the most important
boundary in the codebase:

- The **main process** holds the UI, the auth gate, local storage, job
  orchestration, registration, downloads, and submission.
  `JobOrchestratorService` keeps it in the foreground for the length of a job.
- The **`:benchmark` process** holds the native engine and the model's memory.
  It also hosts `BenchmarkActivity`, the pocket-mode run screen.

A benchmark call crosses that boundary over **AIDL**:

1. `JobRunner` (main process) calls a blocking `*Sync` method on the
   `BenchmarkEngine` interface. The wired implementation is
   `RemoteBenchmarkEngine`, a client proxy.
2. `RemoteBenchmarkEngine` turns each call into a blocking two-way AIDL call to
   `PipetteBenchmarkService` (declared `android:process=":benchmark"`). It binds
   on first use and starts the service in the foreground, so a run continues
   while the app is in the background.
3. The service passes the call to `EngineActor`, which runs every native call on
   one worker thread behind an Empty/Ready/Busy state machine.
4. `EngineActor` calls the benchmark kernel of the build's engine flavor.

**Why a separate process.** The main reason is memory; crash isolation is a
second reason. A loaded model holds a native heap of several gigabytes. When a
job ends, `RemoteBenchmarkEngine` stops the service, and the OS reclaims that
process's memory at once. A cached process would keep it. The same teardown
bounds cancellation: a watchdog kills `:benchmark` if a running decode ignores a
cooperative cancel. The `:benchmark` process also never loads the Clerk or
PostHog SDKs, so they add nothing to the memory it measures.

Three details follow from the boundary:

- **Result transport.** `ResultSpill` carries the result JSON across the
  boundary (see the glossary). The size threshold is in
  [implementation.md](implementation.md#native-benchmark-engine).
- **Mid-run readiness.** The readiness gate between measured reps runs inside
  `:benchmark`. The main process applies its own readiness gate between cells
  only. The per-rep callback does not cross the AIDL boundary.
- **Native library.** Every benchmark runs in `:benchmark`. The main process can
  also load `libpipette_android.so` to read its build id (`NativeLib`, which the
  headless verbs use), but it never loads a model.

The ggml CPU backend is built once per ARM feature level, and a loader selects
one at runtime. All of these variants are inside the single `arm64-v8a` ABI that
the app ships; see [build.md](build.md#native-build-internals).

## Engine flavors

The app has two Gradle product flavors on the `engine` dimension
(`android/Pipette/app/build.gradle.kts`):

- **`kotlinEngine`** is the default. CI and the release workflow build it.
  `EngineActor` calls `LlamaBenchmark`, the benchmark kernel in Kotlin, and
  `LlamaCpp`, which wraps the JNI primitives in `android/native/jni.cpp`. The
  timed decode loop runs in C++, so it makes no JNI call per token. This flavor
  contains no Rust code.
- **`rustEngine`** keeps the Rust kernel. `EngineActor` calls `LlamaEngine`,
  whose `external` methods enter the Rust crate `crates/pipette-android`
  (`lib.rs` → `engine.rs` → `llama.rs`). That crate runs the kernel in
  `android/native/benchmarks.rs`.

Both flavors link the shim and the CPU-variant loader
(`android/native/native_loader.cpp`), and both produce `libpipette_android.so`.
[engine-port.md](engine-port.md) records the port and the invariants that the
two kernels share.
[docs/client-parity.md](../client-parity.md) tells which code changes reach
which flavor.

## Design Choices

### Native Kotlin app

The app is a standard Android application in Kotlin. It owns its local files,
local metadata, downloads, native code loading, and Android device APIs
directly, which keeps it close to the local-device benchmark model.

### Manual DI, Compose, and one view model per screen

`AppContainer` is a hand-written, app-scoped dependency container. It is
app-scoped so that an Activity recreation never stops a running benchmark.

The UI is Jetpack Compose. `ComposeMainActivity` hosts `PipetteAppRoot`, which
shows the auth gate until the gate is `Ready`, then the setup screen until the
device has a registration, then the tabbed screens. Each screen has one view
model with one state flow and one intent entry point. The screen view models
share `ShellViewModel` for navigation, registration, and the auth gate. A
Navigation 3 back stack (`compose/nav/Route.kt`) holds the Jobs, Models, and
Settings tabs and the detail screens above them.

Pocket mode uses classic Views instead: `BenchmarkActivity` runs in `:benchmark`,
and Compose would add memory to the process that measures the model.

### WorkManager for GGUF downloads

GGUF downloads run as WorkManager work (`DownloadWorker`), one unique work per
file. A download continues after process death, resumes from its partial file,
and shows a notification with pause, resume, and cancel actions. The app keeps
its own record of each download, so the UI can show progress, prevent
duplicates, and register the finished model.

### Clerk auth gate as an observe-only seam

The app can gate on Clerk sign-in (see [Auth gate](#auth-gate)). The Clerk SDK
stays behind a thin seam (`ClerkAuth`) that only *observes* SDK state. Only the
main process class-loads it. This keeps the gate's reducer unit-testable off
device and keeps the SDK out of `:benchmark`.

## Local State

Storage has two tiers, split by size and durability:

- **Structured records** persist as JSON in the app's **internal** storage: job
  manifests with their embedded cells, per-cell payloads, metrics, and
  submissions, the registration, and the synced benchmark catalog.
- **Model files** (multi-gigabyte GGUFs) live in **external** app storage.

Simple preferences use Jetpack DataStore (`AppSettingsStore`). The filesystem is
the single source of truth: `ModelStore` derives model metadata from a live
directory scan and does not persist it.

`LocalStorage` owns the internal layout. It exposes only the narrow `JobStore`
interface to `JobRunner`, so tests can use an in-memory fake. A manifest can
outlive a model's exact path, so `JobRunner` resolves a stale model path before
each run. The resolver refuses an ambiguous match, so a job never benchmarks the
wrong repo's file. The exact on-disk tree and the resolver steps are in
[implementation.md](implementation.md#local-storage-model).

## Registration And Secrets

Registration is implemented by `RegistrationService` and `ManagementClient`
(both in `ManagementClient.kt`). The trust model:

- `POST /clients/register` is **unsigned**. It carries the device's Ed25519 public
  key. The app keeps the private key in a *pending* slot and promotes it only
  after the server accepts the registration.
- Every later management call is **signed** with the client's Ed25519 key. The
  signature covers the method, the request target, a timestamp, the client id,
  and a single-use nonce, so a captured signature cannot be replayed. It does
  not cover the body.
- `Secrets` stores the private key and the Hugging Face token in
  `SharedPreferences`, each encrypted with AES-256-GCM under an Android Keystore
  key. The app can use that key but cannot read it.
- Keys are generated and used through **Tink**. The JCA (Java Cryptography
  Architecture) cannot produce an exportable Ed25519 key on any Android below
  API 37. `Secrets`' KDoc has the per-API-level breakdown.
- `RegistrationData` carries optional Clerk link metadata. The app persists it
  locally and never sends it to the management server.

The header names, the signed payload, and the storage slots are in
[implementation.md](implementation.md#registration-and-signing).

## Subsystems

Each subsystem is described here at the design level. implementation.md has the
details of each one.

### Auth gate

Clerk sign-in is the outermost gate: `PipetteAppRoot` shows only the gate until
it reaches `Ready`. A pure reducer maps the Clerk state and the local
registration to an `AuthGate`. If the signed-in user differs from the user
already linked to the registration, the gate shows `Mismatch`. When the build
has no Clerk publishable key, the gate is `Ready` and the app has no sign-in.
Clerk is initialized only in the main process. The key and the debug bypass are
in [build.md](build.md#clerk-sign-in); the reducer and the sign-in flow are in
[implementation.md](implementation.md#clerk-auth-gate).

### Models and downloads

The Models tab lists the downloaded models. Add models imports a local GGUF file
or downloads one from Hugging Face, from a bundled template or a manual
identifier. `ModelCatalog` groups the files into families, so job setup selects
model *families* and then resolves them through the active quant filters. See
[implementation.md](implementation.md#model-management).

### Benchmark catalog

`BenchmarkSync` pulls the catalog from the management server, and
`BenchmarkCatalog` holds the synced copy. The app has no bundled catalog, so the
list is empty until the first sync. The New Job picker offers only the four
token-ladder types whose required context is under 5000 tokens. See
[implementation.md](implementation.md#benchmark-catalog).

### Job planning and execution

`JobController` is the app-scoped owner. It wraps `JobRunner` and exposes its
state to the UI as a `StateFlow`. `JobRunner` plans the cells as models ×
benchmarks and runs them on one thread. It gates each cell on readiness, sends
each cell to `:benchmark`, and uploads results while the job runs when the job
contributes results. Each cell loads its model fresh, so no benchmark runs on a
model that is warm or sized for another benchmark. See
[implementation.md](implementation.md#job-orchestration).

### Readiness (thermal cooldown) and Pocket Mode

`Readiness` gates on `PowerManager` thermal headroom and falls back to the
coarse thermal status when headroom is unavailable. It runs between cells in the
main process and between measured reps in `:benchmark`.

Pocket mode is `BenchmarkActivity`, a full-screen run screen in `:benchmark`.
`JobRunner` opens it when a run starts. While it has focus, the framework puts
`:benchmark` in the `top-app` cpuset, which on some devices is the only group
with the prime cores. The [cpuset demotion guard](implementation.md#cpuset-demotion-guard)
fails a cell that loses them mid-run.

### Results and submission

A completed cell writes `payload.json` in its artifact directory.
`ResultSubmissionService` uploads results and records the server's job ids.
`CompletedResultsCsvExporter` builds a CSV string, and the Jobs screen writes it
through Android's create-document flow. See
[implementation.md](implementation.md#results).

### Headless / automation entry point

`HeadlessRunActivity` (package `ai.liquid.pipette.headless`) starts a benchmark
from an intent, so automation can drive the shipped APK without the UI. It is
the counterpart to iOS's `HeadlessRunner`. It calls the same
`JobController.startNewJob` that the New Job screen calls, so a headless run and
a run started from the UI measure the same thing.

```
adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity \
  -e match LFM2.5-350M \
  -e benchmark prefill_throughput_512,decode_throughput_512_100 -e submit 0
adb logcat -s pipette-headless
```

The examples on this page use the release package, `ai.liquid.pipette`. For a
debug build, use the component
`ai.liquid.pipette.debug/ai.liquid.pipette.headless.HeadlessRunActivity` (see
[build.md](build.md#inspecting-devices)).

Arguments (`HeadlessCommand`):

| Argument | Meaning | Default |
|---|---|---|
| `match` | Case-insensitive substring of a model's name | every **non-projector** model on device |
| `model` | An exact coordinate URI, fetched first if it is not on disk | none; refused together with `match` |
| `benchmark` / `benchmarks` | Comma-separated catalog ids, e.g. `decode_throughput_512_100` | every selectable benchmark |
| `submit` | `0`/`1` | `1`, but `0` for a `model=` run |
| `batch` | Prefill chunk (llama's `n_ubatch`), clamped to the cell's context | `512` |
| `number_gpu_layers` | Layers on the GPU; `0` is CPU-only | `99` |
| `readiness_max_wait_secs` | Readiness deadline for the run's cells, `1`..`86400` | the client default (300s) |
| `readiness_skip_thermal` | `0`/`1`; waives the thermal criterion for this run only | the Settings toggle |

Every name is the one the CLI and `pipette-plan-types` use for the same concept,
so one runbook covers all three clients. They are spelled with underscores here,
as `number_gpu_layers` already is; iOS spells the same two readiness knobs with
hyphens.

Both readiness knobs belong to the job, not to the caller:

- They reach `JobController.startNewJob` as a `RunReadiness`, which the job's
  manifest stores. `JobRunner` installs the manifest's copy at every run start.
- A job that names neither knob uses the defaults, not the values of the
  previous run.
- A resume, a retry, or a rerun of a job uses the knobs that the job was created
  with. Its later cells thus stay comparable to the cells already submitted.
- `readiness_skip_thermal` applies on top of the Settings toggle for that run.
  It does not change the toggle.

The knobs are flat `-e key value` extras, where iOS uses one `runtime-flags=`
JSON object. An `am start` line passes through a shell and then through `adb`,
and the quoting there damages a JSON argument. The key names are the JSON's
names.

Two departures worth knowing:

- **Two ways to name a model, and they mean different things.** `match=` selects
  by substring from what is already on device; `model=` names an *exact*
  coordinate, is fetched first if it is not here yet (#1224), and is what
  `models-delete` takes too. Passing both is refused rather than resolved, since
  one asks for bytes to obtain and the other filters bytes already present. A
  bare name in `model=` is refused with a pointer at `match=`, which is the right
  advice on a run and not on `models-delete`, where `match=` is not a key.
- **No context argument.** `ctx_size` is refused, as on iOS:
  `JobRunner.cellContextSize` computes the window per cell from the benchmark
  definition and only falls back to the job-level value for a benchmark it
  cannot find, so an authored one would be recorded and then ignored by the
  thing it names.

`batch` is *not* the same case, though it can look like one. It is honored, but
clamped to the cell's context when recorded, on both mobile clients, because
llama.cpp cannot use a micro-batch larger than the context. A cell whose
benchmark sizes the context at 32 records `n_ubatch: 32` however large a `batch`
was asked for; on a 512-token context, `-e batch 128` records `128`. A ceiling
on a knob that works is different from a value that is discarded. `n_ubatch` is
also mobile-only: plan-types declares it just for the two in-process app
engines, and the CLI's llamacpp cells drive stock tools with no micro-batch
flag.

Default model selection skips projector (`mmproj`) files, which are inputs to a
vision model rather than benchmark targets of their own; every projector on
device is still offered for pairing, matching what the New Job screen passes.

An unknown key or an unparseable value is refused rather than defaulted, so a
typo in a CI script fails instead of benchmarking the wrong thing. So is an
unrecognized benchmark id, and so is a key the *selected verb* does not take:
`verb=version batch=256` is an error rather than a version report with the batch
ignored.

### Verbs

`verb=` selects what the invocation does, mirroring iOS's positional verbs with a
hyphen where iOS has a space (`headlessrun benchmarks show` is
`verb=benchmarks-show`). **An invocation with no `verb=` is a run.**

The read-only verbs answer to logcat and start nothing, so none of the run path's
refusals (a job already running, a missing registration) applies to them:

| verb | answers |
|------|---------|
| `version` | the build a submission would carry |
| `status` | registration, whether a job is running, model/job/catalog counts |
| `models` | every model on disk, with repo, quant and size |
| `jobs` | every manifest, with per-cell-status counts |
| `benchmarks` `[type=]` | the synced catalog, optionally narrowed |
| `benchmarks-show` `benchmark=` | one benchmark, or a refusal if the id is unknown |
| `storage-status` | models directory, bytes used, bytes free |
| `runtimes` | the runtime this build links, and its build id |
| `results` | every completed cell whose payload is still on disk |
| `results-show` `result=` | one result by cell id, or a refusal |
| `auth-whoami` | this device's `client_id` and public key, offline |
| `auth-me` | the profile the server holds now, from `GET /clients/me` |
| `auth-export` | the identity as one base64url bundle. **Prints the signing key** |

Each line is a record: a kind token then `key=value` pairs, because what reads
logcat is `grep` rather than a person. A listing ends with its own
`<kind>s count=N` line, since the terminal sentinel says the invocation finished
but not whether logcat dropped rows in the middle of it. A key that would be
empty is omitted rather than emitted blank, so a consumer tests for the key.

**`auth-me` is the one exception to two of those rules, and it is deliberate.**
It emits `tags=` and `capabilities=` empty rather than omitting them, and its
`organization=` and `client_details=` hold free text a person typed, so a value
can contain a space. Both are iOS's choices, and the reason for the exception is
parity. The line matches iOS's `auth me` line field for field and in iOS's
order, so one runbook and one parser read the profile from either mobile client
and from the CLI. A consumer that needs those two fields intact reads them positionally
rather than by splitting on whitespace. Every other verb still follows the rules
above, including the identity verbs beside it, which report only ids and hex.

There is no `storage-status` quota field, because Android has no quota: the model
store is uncapped and nothing evicts (PIP-492 gap 2). Its absence is the accurate
signal, where `quota=unset` would invite a consumer to read it as unlimited.

The local mutations are verbs too (#1243). This component is exported, so the
verbs that destroy data had a decision of their own. The caller gate is
`WRITE_SECURE_SETTINGS`, which adb's shell uid and privileged system apps hold.
Anyone with adb can already `pm uninstall` this app and delete the same data, so
the gate is the real boundary, and a second gate here would add no protection.
The gate has one gap; see the residual surface at the end of this section.

| verb | does |
|------|------|
| `job-rm` `id=` | delete a job and everything under it |
| `job-export` `id=` | write the job's CSV under `getExternalFilesDir(null)/exports/` and print the path |
| `results-delete` `result=` | clear one cell's stored result, leaving the job |
| `models-delete` `model=` | delete the weights a coordinate names, and its projector unless another model still pairs with it |
| `models-rm` `name=` `repo=` | delete weights by exact file name, by repo, or both |
| `auth-import` `bundle=` `force=` | restore an exported identity onto this device |

Two rules are worth carrying into a runbook. **`models-rm` matches a name
exactly**, unlike `run`'s substring `match=`, so `name=LFM2.5-230M` against a
family of quants is refused rather than deleting all of them. And **no delete
runs while a job does**, because the runner writes payloads into the job
directory and loads a model file per cell, so a delete mid-run corrupts a run
that was going to succeed. `job-export` is outside that rule: it is a mutation
because it writes a CSV, but it destroys nothing and reads a manifest a run is
only appending to, so a job in flight makes its output incomplete rather than
wrong.

One verb reaches the network, and it reads: `auth-me` asks the server what
profile it holds for this device, which is how an operator confirms an approval
landed. `HeadlessRunActivity.onCreate` stays synchronous, so `HeadlessFacts`
pays for that round trip by blocking the main thread on an IO dispatcher; the
invocation's worst case is `fetchMe`'s own connect and read timeouts, because a
blocking socket read cannot be cancelled from outside. That call asks for a read
budget sized to the short record it expects, rather than the submission budget,
so a server that accepts the connection and then stalls holds a fleet script for
seconds per device instead of minutes. The other network verbs are not
available: submitting and registering *publish* data, and they need a separate
review. `job run` is also not available, though it is not a network verb; see
[docs/client-parity.md](../client-parity.md), which tracks every verb that iOS
has and Android does not.

### Carrying an identity to another device

`auth-export` and `auth-import` are the pair that moves a `client_id` to
replacement hardware, and `auth-whoami` is how an operator reads the id in the
first place, because approving a client on the server needs it.

```
adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb auth-whoami
adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb auth-export
adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity \
  -e verb auth-import -e bundle "'<token>'"
adb logcat -s pipette-auth       # the identity key events alone
```

Five properties matter for a runbook.

- **`auth-export` prints the Ed25519 signing key in plaintext**, inside the
  bundle on its own second line. This is intentional: the key leaves the device
  no other way. Without it, an uninstall ends the identity for good and leaves a
  server-side client record that no device can sign for. It goes to the `pipette-headless` tag only, and the caller gate is
  `WRITE_SECURE_SETTINGS`.

  **Clear the buffer once the bundle is captured**, because the caller gate does
  not bound who reads the line afterwards:

  ```
  adb logcat -c
  ```

  logcat is a ring buffer. The key stays in it until it is cleared or it ages
  out, `adb bugreport` captures the whole buffer, and any app holding
  `READ_LOGS` can read it. Clearing is the only one of those an operator
  controls.
- **`auth-import` refuses over a signing key that is not the bundle's** unless
  `force=1`, because that overwrite orphans every result the replaced client
  submitted. A key equal to the bundle's is no conflict, so rerunning the same
  bundle finishes a half-written import without `force=1`. No import runs while
  a job does, for the reason no delete does: submissions are signed as cells
  finish, so a key swapped mid-run splits one device's measurements across two
  client ids.
- **`auth-import` also refuses over a key the Keystore cannot open**, and
  `force=1` does not lift that one. An unreadable slot reads exactly like an
  empty one, so neither the caller nor the verb can tell a fresh device from the
  identity an import would destroy. The outage passes and the key returns, so
  retry the verb. A device whose Keystore is broken for good still restores:
  clear the identity from Settings first.
- **`auth-import` refuses when a registration is in flight**, because that
  registration stages a key the server may already hold. Let it finish, then
  rerun the import. `force=1` lifts this refusal, unlike the one above: a
  registration killed between staging a key and promoting it leaves that key
  behind, and no verb here clears the slot.
- **The bundle is written to be wire-compatible with iOS version 1**, field for
  field against `ios/Pipette/Pipette/Identity/Bundle.swift`, so a token
  from either client imports on the other. Android holds a strict subset of what
  iOS holds (no device labels, no client settings, and one Hugging Face token
  rather than one per model), so it writes those three empty and reports any
  per-model tokens an iOS bundle carried as `model_tokens_discarded=`. The
  round trip in each direction is a device check that PIP-547 still lists as
  pending; what the unit tests pin is that Android's decoder accepts the record
  iOS's encoder produces.

An import never clears the existing Hugging Face token, even a forced import. A
bundle that carries `user_token` overwrites it; a bundle without one leaves it
unchanged. The token authenticates against huggingface.co, not against the
management server, so it belongs to the operator and not to the `client_id` that
the import replaces.

iOS does the same. Its import clears the per-model tokens when it replaces
another key. That clear selects Keychain accounts by the `hf_token:` prefix,
which the bare `hf_token` account of the user token does not match. Even
`auth reset force=1` keeps the user token. This client has one token slot, and
that slot is the counterpart of the account that iOS keeps.

Note that **fleet automation does not use this.** The fleet drives Android
hardware by pushing the `pipette` binary and running it over `adb`
(`crates/pipette-ctl/src/transport/adb.rs`). This surface exists for the cases
that path can't cover: CI against the shipped app, a device that can only run
the in-app engine, and reproducing a user-visible bug without driving the UI.

The activity emits `[HEADLESS] …` lines to logcat under the `pipette-headless`
tag and always ends with `BENCH_DONE <code> exit=<code>`, mirroring iOS's console
contract (`0` started or answered, `1` refused or failed, `2` malformed
invocation).

An identity key event goes to the `pipette-auth` tag as well: a performed export,
and a performed or refused import. That stream is the standing record of what has
happened to this device's identity, where `pipette-headless` is one invocation's
transcript. Only the summary line is mirrored, so the bundle, and the key inside
it, reaches one stream and not two.

The **numeric token comes first because that is the contract**, not for looks.
`pipette-ctl`'s `scan_sentinel` splits on the sentinel and parses the next
whitespace token as the status; a token it cannot parse yields "no sentinel
found", and the caller then falls back to the process exit code. `am start`
exits 0 even when this activity refuses, so that fallback would record a refusal
as a completed cell, which is the exact bug iOS's numeric sentinel was introduced
to fix. No transport scans this surface yet; the numeric token makes it ready
for one (PIP-492).

The activity *result* code is a separate scale, because the two vocabularies
collide: iOS's success is `0`, which on Android is `RESULT_CANCELED`. A started
run therefore returns `RESULT_OK`, and the two failures return
`RESULT_FIRST_USER` (refused or failed) and `RESULT_FIRST_USER + 1` (malformed).
Log consumers may key on either `BENCH_DONE ` plus the first token or the
`exit=<code>` spelling, which existing scripts use. Only a caller using `startActivityForResult` or the Activity Result APIs sees the
result code.

**Why it is exported, and what guards it.** `adb shell am start` cannot reach a
non-exported component, so the activity has to be exported, and on Android that
means reachable by any installed app. Since a headless run submits by default,
an unguarded surface would let an arbitrary app upload under this device's
identity.

The guard is `android:permission` on the component, enforced by the framework
before the activity is created. The choice of permission is what makes this
work. A
*custom* `signature` permission does not work: `adb shell` runs as the `shell`
uid, which is not an app and holds no app-defined permissions, so the framework
would reject the automation path this exists for. A *platform* permission that
`shell` already holds does work, and `WRITE_SECURE_SETTINGS` is one:
`com.android.shell` holds it, which is what backs `adb shell settings put
secure`. It is declared `signature|privileged|development`, so no third-party app
*ships* with it. The `development` flag means that someone with adb can grant it
to an app; see the residual surface below and the note in `HeadlessCaller.kt`.

Verified on an API 36 device: `adb shell am start` is admitted, and an ordinary
app calling `startActivity` on the same component is refused by the framework
with `SecurityException: Permission Denial: … requires
android.permission.WRITE_SECURE_SETTINGS`.

The caller's uid is deliberately *not* consulted.
`Activity.getLaunchedFromUid()` cannot see adb: it reports the launching uid
only when the launcher opted in via `ActivityOptions.setShareIdentityEnabled`,
when the launcher shares this app's uid, or when this app is platform-signed.
`am start` does none of those and has no flag to ask for it, so it returns
`Process.INVALID_UID` and a uid-based guard would refuse exactly the caller it
was written to admit. `getLaunchedFromPackage` and `getCallingPackage` are empty
for the same launch, and `getReferrer` is caller-settable via
`Intent.EXTRA_REFERRER`, so none of them is a boundary either.

What `HeadlessCaller` does in code is confirm the manifest guard is still in
force: it reads back the permission the `PackageManager` reports for this
component and refuses if it is not the expected one, so deleting the attribute
turns the surface off rather than opening it.

Residual surface, stated plainly: a privileged system app on the device image
may hold `WRITE_SECURE_SETTINGS`, and anyone with adb can grant it to an app
through the permission's `development` flag. The first already implies authority
over the device well beyond starting a benchmark. The second does not: a
granted app reaches every verb here, including the deletes, without being able to
`pm uninstall` anything. Getting there takes a deliberate `adb shell pm grant`,
which is the same access that could install a different build of this app.
