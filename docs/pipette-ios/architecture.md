# Pipette iOS Architecture

The Pipette iOS app runs its inference engines inside the app. There is no
separately installed runtime. The desktop runners start upstream `llama.cpp`
release binaries as separate processes. iOS runs four engines, and the model
format selects one: llama.cpp, MLX and ONNX Runtime GenAI, which are compiled into
the app, and Apple Foundation, which ships with the OS. Every engine sits behind one benchmark
interface, so the rest of the pipeline (payload → CSV → submission) is identical
regardless of which engine ran a cell. For how the vendored sources are compiled
into the app, see [build.md](build.md).

## Inference engines

Three engines are compiled into the app and the fourth ships with the OS.
llama.cpp and MLX run on the GPU via Metal; ONNX Runtime GenAI runs on the CPU,
and Apple Foundation runs wherever the system puts it.

| Model format | Engine | Implementation | Submission identity |
|--------------|--------|----------------|---------------------|
| `*.gguf` file | **llama.cpp** | `Runtimes/Llama/`: stateless `LlamaCpp` ops over the llama.cpp C API (`import llama`), driven by `LlamaExecute`; `ios/build-llama.sh` builds each vendored pin into a framework that the app embeds. A process loads one framework ([section 2](#2-how-the-ios-app-uses-llamacpp)). No Rust, no FFI bridge | `runtime_descriptor` = the `llamacpp_ios_pipette` `Runtime` of the pin the process loaded, naming its repository and its tag or commit |
| safetensors directory (`config.json` + `*.safetensors`, an mlx-lm export) | **MLX** | `Runtimes/Mlx/`: `MlxRuntime` (scoped load) + `MlxRuntime+Generate` / `MlxExecute` / `MlxRuntime+Probes`, built on the `mlx-swift` / `mlx-swift-lm` Swift packages | `runtime_descriptor` = the `mlx_ios_pipette` `Runtime`, pinned to the resolved `mlx-swift`, `mlx-swift-lm` and `swift-transformers` versions |
| ONNX GenAI package (`genai_config.json` + its graph and sidecar) | **ONNX Runtime GenAI** | `Runtimes/OnnxGenai/`: `OnnxGenaiRuntime` (scoped load) + `OnnxGenaiRuntime+Generate` / `OnnxGenaiExecute`, over the engine's Objective-C surface; both source trees are compiled into the app by `ios/build-onnx-genai.sh` | `runtime_descriptor` = the `onnx_genai_ios_pipette` `Runtime`, pinned to the vendored `onnxruntime` and `onnxruntime-genai` commits |
| none: the weights ship with the OS | **Apple Foundation** | `Runtimes/AFM/`: `AFMRuntime` over the FoundationModels framework | `runtime_descriptor` = the `apple_foundation` `Runtime`; `model_descriptor` is `apple/foundation-text` |

ONNX Runtime GenAI measures four of the ten benchmark kinds. It refuses `eval`
because the tokenizer applies no chat template, and the three `vl_*` and two
`asr_*` kinds have no ONNX variant ([onnx-genai-runtime.md](onnx-genai-runtime.md)).
Apple Foundation runs the cells it can reproduce with the same metric and refuses
the rest, rather than reporting a stand-in ([afm-runtime.md](afm-runtime.md)).

The runtime/model concept lives in `PlanTypes/Runtime/` and `PlanTypes/Model/`, a
hand mirror of the `pipette-plan-types` crate that mgmt owns
([pipette-mgmt crates/pipette-plan-types](https://github.com/Liquid4All/pipette-mgmt/tree/main/crates/pipette-plan-types/src)):
a `Runtime` sum that is build *identity* alone and a loadable `Model` sum, with all
behavior derived via `switch`.

- A cell's load settings are `RuntimeFlags`, carried beside the runtime on the
  `RunRequest`, never inside it.
- `RunCell.prepare` builds the request. It binds the cell's model and names its
  runtime as `Runtime.thisBuild(for:)`: the process runs one build of each engine,
  and the model format selects the engine.
- `RunCell.dispatchRun` routes on the bound `Runtime` alone, as upstream's
  `dispatch_run` does. Each engine's `require*` refuses a model it cannot load.
- The pairing rule is the free `isCompatible(_:_:)` (`PlanTypes/Cell/Cell.swift`,
  the crate's `cell::is_compatible`), applied at claim time.
- The loaded model is assembled **inside the benchmark scope** (`withInference` /
  `withFreshModel`) so the resource never escapes.
- `RuntimeKind` (`Models/RuntimeKind.swift`), the one identity with no
  configuration, is only a UI and CLI selection input: the New Job picker, the
  headless `runtime=` flag and model-list filtering. It is not part of the runtime
  definition, and nothing persists it as a separate tag.

Every engine plugs into one harness. It consumes the typed `RunRequest`, measures
through `BenchmarkMeasurement` (`REPS` reps of prepare → work → sample, with a
readiness gate before the model loads and again before each measured rep), and
takes the same injected `ReadinessGate` and `MeasurementObserver`. It answers with
a typed `BenchmarkResultData` from the *same* `BenchmarkCatalog` definitions. So
the downstream payload → CSV → submission pipeline is identical whichever engine
ran the cell.

### Running MLX

The New Job picker offers every engine with at least one offerable model,
downloaded or in the catalog (`NewJobView.availableRuntimes`).
`MlxFeatureFlag.visibleInUI` decides whether the default catalog merges MLX rows,
so with it off only an MLX bundle on the device makes MLX selectable. MLX is also
exercised two other ways:

- **Headless runner** (`Headless/HeadlessRunner.swift`, with the verbs in `Commands/`):
  launch the app with `headlessrun runtime=mlx …` arguments to run benchmarks with no
  UI taps. The command surface is in [usage.md](usage.md).
- **Parity validation** (the `coherence` metric, in `Commands/BenchmarksCommands.swift`):
  compares MLX's top-5 and greedy output against baked Python `mlx_lm` reference
  token ids to catch routing/correctness regressions. The reference ids are
  LFM2.5-8B's, so the comparison means something only on that model. On llama.cpp
  it only logs the greedy text.

A headless benchmark launches its cells through the New Job screen's
`JobLauncher`, so they run through the same `RunCell.runCell` the UI does and a
number taken headlessly is the number a job would produce.

## Analytics (PostHog)

Product analytics is a hand-picked, boundary-only event set behind the
`Analytics` seam (`Support/Analytics.swift`). The Android client sends the same
event and property names so both land in one PostHog project.
`Analytics.start()` no-ops unless `PostHogConfiguration.isComplete`, leaving the
sink as `NoOpAnalytics`. The `phc_…` project token lives in `Info.plist`
alongside `SentryDSN`. It is write-only and public by design.

Autocapture, session replay, element interactions and feature-flag preloading
are all off, because instrumentation overhead corrupts on-device LLM
measurements, the same reason `SentryConfiguration` disables tracing,
auto-performance instrumentation, the app-hang watchdog and watchdog-termination
tracking (whose per-frame display link also raced `exit` in headless runs). PostHog's own error
tracking (`errorTrackingConfig.autoCapture`) is pinned off as well: the SDK
vendors PLCrashReporter, and a second crash reporter installing signal handlers
would fight Sentry for them.

Some integrations default to on, and none of the switches above covers them, so
`Analytics.start()` turns each one off by name:

- `surveys` and `rageClickConfig.enabled`. Left alone, launch logs
  `PostHogSurveyIntegration installed` and `PostHogRageClickIntegration installed`,
  and the latter hooks touch delivery for the life of the process to spot rapid
  taps. That is background work during a measurement, which the app avoids
  everywhere else.
- `capturePushNotificationSubscriptions` and `capturePushNotificationOpened`. The
  subscription one swizzles the app delegate to register this device's APNs token
  with PostHog. This app sends no push notifications, and a push token is a device
  identifier that neither `PrivacyInfo.xcprivacy` nor the taxonomy accounts for.

With these off, no integration installs at all. Re-check on every SDK bump.

`config.debug` is on in DEBUG builds. Without it the SDK is entirely silent, so a
suppressed event cannot be told from a captured one on a simulator.

No event fires inside a measurement window. `job_started` is captured and
flushed in `JobExecutor.run` *before* the detached run task begins, so the queue
is empty entering the run and the SDK's periodic flush timer has nothing to send
while a cell is timed. The one event that can fire mid-job is
`results_submitted`, from the per-cell upload `JobExecutor` performs between
cells; `ResultUploader` flushes immediately after capturing it, so the next cell
still starts on an empty queue. Events carry the registration `clientId` as the
distinct id; no email, organization, Clerk identity, error message, or feedback
body is ever sent. `PrivacyInfo.xcprivacy` declares the resulting collection
(`ProductInteraction` for analytics, plus an analytics purpose on `DeviceID`).

`SettingsView` carries a "Share anonymous usage analytics" toggle calling
`Analytics.setOptedOut`, shown only when `Analytics.isAvailable`: a toggle over
`NoOpAnalytics` would be a control that does nothing. It sits outside the
`canSubmitResults` gate the other two toggles share: analytics start at launch,
before this device has registered, so the control that stops them has to be
reachable then too. The view's `@State` is a mirror re-seeded in `.onAppear`,
never a second source of truth.

`LocalStorage.analyticsOptOut` owns the flag and `Analytics.start()` seeds
`PostHogConfig.optOut` from it **before** `setup`, rather than relying on the
SDK's own persisted copy: one launch is all it takes to leak an event from a
device whose owner turned analytics off. `UserDefaults` is synchronous, so the
read costs nothing at launch.

Verified on an iPhone 17 Pro simulator: with the flag set, launch logs
`PostHog is in OptOut state.` and captures nothing (no `Queued event`, no batch);
clearing it restores `Queued event 'app_launched'` → `batch sent successfully.`

An opted-out device still logs `Remote config called successfully.`: posthog-ios
deprecated `config.remoteConfig` and ignores it ("is now always enabled"). The
request carries the public project token and no user data.

`AnalyticsEventsTests` pins the wire contract (event names, property keys,
`RunSource` raw values and the outcome mapping), and `PostHogConfigurationTest`
pins the same set on Android. Neither can enforce true parity (each platform
asserts its own copy); they exist because a rename on one platform alone doesn't
break a build, it silently splits a funnel into two unrelated-looking events.

## Components

The vendored engine sources live in the repository's `vendor/` submodules
(`vendor/llama.cpp` is shared with Android). `ios/build-llama.sh` builds a patched
copy of one llama.cpp pin into a framework. It applies `ios/patches/` (or
`ios/patches/llamacpp-<variant>/` for a variant). The run for main also writes
the headers that Swift compiles against to `Generated/Llama/`.
`ios/build-onnx-genai.sh` does the same for ONNX Runtime GenAI with
`ios/patches/onnx-genai/`.

The app target (`ios/Pipette/Pipette/`) is laid out after the Rust crates it
mirrors. The top-level directories:

| Directory | Holds | Mirrors |
|---|---|---|
| `Ops/` | the engine-agnostic half: the rep loop, measurement kinds, eval sample loop, prompt seeding | `pipette-ops` |
| `Readiness/`, `Wait/` | the readiness wait and its iOS platform module | `pipette-readiness`, `pipette-wait` |
| `Runtimes/` | the engines (`Llama/`, `Mlx/`, `OnnxGenai/`, `AFM/`), plus the jetsam pre-flight and footprint probe | each runtime crate's `execute/` |
| `Run/` | `RunCell`: one cell from prepare to dispatch | `pipette-cli`'s `run.rs` |
| `Commands/`, `Headless/` | one file per headless command group; the argv grammar and runner | `pipette-cli`'s `commands/` |
| `Client/` | the management-server flows: claim, registration, worker profile/protocol/timing | `pipette-cli`'s `client/` |
| `Identity/`, `Benchmarks/`, `Results/` | the stores the workspace mints | `pipette-cli`'s modules of the same name |
| `Device/` | device probes and readings, plus the `IMUThermometer` only a phone has | `pipette-device` |
| `PlanTypes/` | the plan vocabulary | `pipette-plan-types` |
| `Artifacts/` | model store, fetch, quota and admission | `pipette-artifacts` |
| `Networking/` | `PipetteHttp/` and `PipetteMgmtClient/`, plus the collector endpoint and reachability | `pipette-http`, `pipette-mgmt-client` |
| `Config/` | build flavor and the Clerk, PostHog and Sentry configuration | none |
| `Native/` | the private-API probes (ObjC): `PipetteThermal.m` for IOHID temperatures, `PipetteClocks.m` for IOReport clock residency; and `PipetteLlama.c`, which loads one llama.cpp framework per process and forwards the C API to it | none |
| `Jobs/`, `Models/`, `Views/`, `Storage/`, `Support/` | iOS-only: the batch noun, the catalog and downloads, SwiftUI screens, app settings, shared helpers | none |
| `Generated/` | build script output; do not edit | none |

### How a file finds its place

A file that ports a crate module is named for that module and sits at the same
depth under the directory that stands for the crate: `pipette-ops`'
`measurement/mod.rs` is `Ops/Measurement.swift`, `pipette-plan-types`'
`benchmark/eval_id.rs` is `PlanTypes/Benchmark/EvalId.swift`, and
`pipette-cli`'s `client/worker/timing.rs` is `Client/Worker/Timing.swift`. Read
the other way, a crate module says where its Swift counterpart is without anyone
having to remember which type ended up where.

Code no crate module accounts for is named for its principal type. A
`Foo+Bar.swift` file holds `extension Foo` only and is never a type's home, and
a file holding only extensions on `Foo` is named `Foo+<Aspect>.swift`, unless it
ports a crate module, which keeps the module's name. Test files are
`<PrincipalType>Tests.swift`; the tests of `Foo+Bar.swift` are
`Foo+BarTests.swift`, holding `struct FooBarTests`.

Two things bend the rule, both by necessity:

- **Basenames are unique across a target.** Xcode names a Swift file's
  `.stringsdata` after its basename alone, so two `Store.swift` in one target
  fail the build with `duplicate output file`. Where the bare module name is
  taken, the file says what it holds instead, usually its principal type. A
  mirrored directory therefore carries both spellings, and neither file
  announces which one it follows:

  | crate module | file | the name it could not have |
  |---|---|---|
  | `benchmarks/store.rs` | `Benchmarks/BenchmarkStore.swift` | `Artifacts/Model/Store.swift` |
  | `results/store.rs` | `Results/ResultsStore.swift` | `Artifacts/Model/Store.swift` |
  | `identity/device.rs` | `Identity/DeviceLabels.swift` | `PlanTypes/Device.swift` |
  | `client/auth.rs` | `Client/ClientAuth.swift` | `PipetteMgmtClient/Auth.swift` |
  | `results/types.rs` | `Results/ResultTypes.swift` | `PipetteMgmtClient/Types.swift` |

  Each crate root is named for its crate (`PipetteMgmtClient.swift`) rather than
  `Lib.swift`, for the same reason.
- **iOS-only code has no module to be named for.** `Views/`, `Jobs/`,
  `Headless/` and the Clerk half of `Identity/` answer to the app, not to a
  crate, and are named for what they do.

Four crate modules are deliberately not mirrored one-to-one:

- `pipette-cli`'s `client/sync.rs` is `Client/Sync.swift` (the submit preflight,
  the catalog pull and the score refresh) less the result push, which stays with the store it drains
  (`Results/ResultSubmissionService.swift`): the app runs that one from its own
  screens as well as from `sync`.
- `pipette-ops`' `fs.rs` has no file. Its JSON read/write helpers are the shared
  coders in `Support/Coding.swift`, which every store the workspace mints
  (`Workspace.swift`) already encodes and decodes through.
- `pipette-mgmt-client`'s `user_api.rs` (user identities, their keys and the
  `/users/*` and `/plans` routes an operator files work with), the types only it
  carries, and `httpsig.rs` (the RFC 9421 signing only a user does) are absent:
  the app only ever authenticates as a device.

One more file breaks the naming rule: `pipette-ops`' `wav.rs` is
`Ops/WavTone.swift`, named for its principal type.

## 1. Core Concepts

### 1.1 Runtime

There is no separately installed runtime on iOS. The engines are compiled into
the app or ship with the OS (see [Inference engines](#inference-engines)). The
llama.cpp and ONNX Runtime GenAI versions are pinned by their `vendor/`
submodules; updating one means bumping the submodule commit and rebuilding the
app. The MLX engine's version tracks its Swift package pins in the Xcode project.
Each reports its release tag when pinned to one and its commit otherwise.

Metal and Accelerate are enabled by `build-llama.sh`, so on-device `llama.cpp`
runs use the GPU through Metal when `n_gpu_layers > 0` and fall back to
Accelerate on CPU otherwise. MLX likewise runs on the GPU through Metal, via
`mlx-swift`.

### 1.2 Model

A model is a GGUF file, an MLX bundle or an ONNX GenAI package stored in the
app's model store (`models/`, below). The app downloads models from Hugging Face
via `DownloadCoordinator`. `LlamaCpp.load` takes an absolute file path and returns a
`LlamaModel` handle (owning the llama.cpp model + context + greedy sampler),
assembled inside the benchmark scope by `withInference` and freed deterministically
when that scope ends.

#### 1.2.1 Local storage

Local job data is versionable JSON under the app's Application Support data
root:

```text
Pipette/
  identity/                 # the identity store's root, as pipette-cli names it
    id_ed25519              # the signing key
    registration.json       # IdentityRegistration, snake_case as the CLI writes it
    settings.json           # ClientSettings: the storage quota
  benchmarks/               # the catalog, as pipette-cli's benchmarks/
    local/<id>.json         # generated by `benchmarks init-local`; never submitted
                            # `<id>` is a BenchmarkId: the store refuses anything else,
                            # so a synced definition the CLI would skip is skipped here
    remote/detail/<id>.json # one synced definition as the server sent it, eval
                            # samples included
    remote/detail/<id>.etag # the SHA-256 of that body, then the ETag it came with
    remote/state/index.json # the GET /benchmarks list as the server sent it, every
                            # row; rows this build cannot convert are filtered on read
    remote/state/sync.json  # server, list ETag
    remote/*.json           # older builds' index, sync state and typed definitions
                            # (which dropped what they did not model): never read;
                            # deleted only with the whole remote half (`auth reset`
                            # or a forced `auth import`)
  state/evals/              # per-run eval checkpoints, so a killed run resumes
  corpora/                  # a cell's dataset, deleted after the cell
  jobs/
    <jobId>/
      manifest.json         # the batch (schemaVersion 2); results are keyed by cell,
                            # not nested here
  results/                  # the results store, as pipette-cli's results/
    local/<cellId>/         # from a generated benchmark; never submitted
    remote/pending/<cellId>/
    remote/synced/<cellId>/ # payload.json, extras.json, metrics.json, submission.json
  models/                   # the model store (below)
```

The pull keeps its bookkeeping in `remote/state/`, apart from older builds' files:

- Older builds keep theirs at `remote/index.json` and `remote/sync.json`, with only
  the rows each build can convert. If two builds shared one file, one build's `304`
  could hand the other a list filtered by a different build. That build's prune
  would then delete the details of the types filtered out.
- Older builds list, read and prune only the `*.json` files directly in `remote/`, so
  they never touch `state/` or `detail/`. A device upgraded from one does one cold pull.
- A pull leaves the older files in place. Deleting them could leave an older build a
  list ETag without its list.

What a pull fetches, and how it revalidates:

- A pull fetches the definition of each listed row whose type this build knows and
  whose id it accepts. Its count is those rows.
- A row leaves out what only the definition carries (a corpus URL, eval samples), so
  the type and id alone decide. A definition that then does not convert is logged
  and not cached.
- A definition is revalidated with its own ETag. Whoever caches it (a pull or a
  claim, below) writes the ETag beside it, with the SHA-256 of the body it came with.
- The ETag is sent only while that hash matches the body read. A crash, or a pull
  and a claim that cache one id at once, costs a full fetch, never a `304` for
  other bytes.
- A pull from a server other than the last pull's sends no ETag, for the list or
  any definition.

#### Model store

Downloaded models live under `models/` as one **entry** per model, addressed by
a flat `ModelStorageKey`: the spec's identity segments sanitized to
`[A-Za-z0-9._-]` and joined with `__`, capped at 32 characters with an 8-hex
SHA-256 fold. It mirrors `pipette-artifacts`' `ModelStorageKey`, so the same
coordinate keys the same string on the CLI and on iOS.

```text
models/
  LiquidAI__LFM2.5-350M-G_26da8d91/   # a key over 32 characters folds to <head>_<hash8>
    manifest.json   # { manifest_version, fetched_at, last_used_at, declared, stored, blobs_bytes, resolved }
    blobs/
      LFM2.5-350M-Q4_0.gguf
  org__vl-GGUF__vl-Q4_0.g_14fbc11f/
    manifest.json
    blobs/
      vl-Q4_0.gguf          # a VL model's weights and projector share one entry
      mmproj-vl-F16.gguf
  LiquidAI__LFM2.5-350M-MLX-4bit/
    manifest.json
    blobs/                  # the MLX bundle: config, weights, tokenizer
  org__multi-mlx__4bit/
    manifest.json
    blobs/
      4bit/                 # a repo subtree keeps its prefix, as the crate lays it out
```

`declared` is the coordinate the model was fetched from. `stored` is the same
model with its paths relative to `models/` (`<key>/blobs/<file>`), which
`ModelArtifactStore` binds under the current `models/` to find the bytes.
Reading a manifest checks its version and that `stored` is still `declared`'s
store form. An entry-relative `stored` an earlier build wrote is re-rooted onto
`models/` and rewritten (`ModelArtifactStore.readManifest`); anything else is
corrupt.

One entry, one manifest, one delete. `models/` and every entry are marked
`URLResourceKey.isExcludedFromBackupKey` because model binaries are large and
re-downloadable. A directory under `models/` is a model only if it carries a
manifest this build can read; everything else there is garbage the sweeper
reclaims.

#### Storage quota

`identity/settings.json` holds `storage_quota_bytes`, defaulting to
`min(16 GiB, 25% of volume capacity)`. Unlike the CLI, iOS admits a whole model
against the quota and the volume's free space *before* the transfer starts
(`Artifacts/Admission.swift`). The mechanism, its pins, and the Settings control
are described in [storage-quota.md](../storage-quota.md).

### 1.3 Benchmark

A benchmark is the same `BenchmarkDefinition` used by the desktop runners. The
catalog has the crate's two halves (`BenchmarkStore`): the `remote/` set synced from
the management server, and the `local/` set this device generates with
`StandardBenchmarks` (the crate's `standard.rs` ladder and smoke entries, id for id).
There is no bundled fallback, so the picker is empty until the first sync or seed.

A run names a benchmark by reference (a bare id or `remote/<id>`, or `local/<id>`). A
plain `benchmarks run` resolves the reference from the half of the store it names, then
from the structured id (the ladder rungs, which spell their own parameters). As in the
crate, only a claim asks the server: the planner worker's claim, and a headless run that
carries `claim=`, as every `pipette-ios-proxy` launch does. A claim resolves a remote
reference through `ensureRemoteBenchmarkCached` (the crate's
`ensure_claim_benchmark_cached`), which the headless path (`JobCell.resolvable`) runs
before the model download. Both requests below are signed with the device's identity
when it holds one, as the crate's are, because the server serves an organization's
private entries only to a signed member.

- **Nothing cached:** `GET /benchmarks/{id}`, cached on the remote half. A proxied
  phone therefore runs a benchmark published after its last `sync` without running
  `sync` first.
- **Cached:** a claim never runs `sync`, so it revalidates the cached definition:
  `GET /benchmarks/{id}` with `If-None-Match` set to its ETag. mgmt's ETag is a hash of
  the whole body, so a `304` means the parameters, corpus pin and eval samples are all
  current, and the cached definition runs. A `200` carries the server's current
  definition, which replaces the cached one and its ETag and runs, so the result
  declares the `benchmark_params` mgmt checks it against. A cached definition without
  an ETag, or a cached file that does not read, is fetched whole. A request that fails
  or is refused, and an answer that does not parse, run the cached definition with a
  warning (`outcome=unchecked`). The claim never writes `state/sync.json`. Each outcome
  is logged with the benchmark id, as `outcome=current`, `changed`, `fetched` (a cached
  definition without an ETag) or `unchecked`.

A `404` on `GET /benchmarks/{id}`, or a definition this build cannot convert, cached or
not, is an unknown benchmark, since mgmt refuses a result whose definition is not the
one it serves: terminal for the worker's claim, a skipped reference for a headless run
(the proxy has already retired a `404` before the launch). With nothing cached, a fetch
that fails in transit is retriable on both. The executor runs the definition the claim
returned (`claimedBenchmarks`), not the synced list's row, which a claim never
refreshes.

One cell runs through `RunCell.runCell`, the peer of the crate's `run_cell`:
prepare (which ensures the model and names every axis), the memory preflight,
the readiness policy the cell resolved to, then `RunCell.dispatchRun`, which routes
on the bound `Runtime` alone to `LlamaExecute`, `MlxExecute`, `OnnxGenaiExecute` or
`AFMRuntime`. The engine takes a typed `RunRequest` and assembles the model fresh
inside its own scope (`withInference` / `withFreshModel`), runs the benchmark, and
frees it. Every cell is an isolated load so memory resets between cells (and
`max_memory_usage` can observe the load itself). It answers with a `RunResponse`
around a typed `BenchmarkResultData`; serialization to the management server's
`BenchmarkSubmissionPayload` happens once the engine returns
(`finishedRunPayload`), not in the engine.

#### 1.3.1 Load mode and benchmark memory semantics

Every llama.cpp timing and memory cell loads with the weights read into anonymous
RAM (`load_mode = "none"`), which is what the desktop `pipette-llamacpp` path pins
for its `llama-bench` and `llama-server` cells (`get_load_mode`). The
benchmark contract is therefore: can this model run without relying on
memory-mapped weights? Eval is upstream's exception and this client's: it leaves
llama.cpp's own mmap default alone, because a run that long amortizes the page
faults.

This matters because llama.cpp's mmap path can map model tensors directly from
the GGUF file, and Metal can wrap those file-backed pages with no-copy buffers.
On iOS, that can let prefill/decode proceed for a model that only fits because
the weights are file-backed and only the touched pages are resident at a given
moment. Those are real executions for an mmap-enabled app mode, but they are not
valid rows for the no-mmap benchmark baseline. If a model loads or runs only with
`load_mode = "mmap"`, the no-mmap benchmark fails and reports no throughput.

This also affects memory reporting. `max_memory_usage` samples the process
physical footprint (`task_vm_info.phys_footprint`, the counter jetsam kills on)
across a fresh load, rather than an allocator-specific high-water mark, so one
figure is comparable across engines ([peak-memory-ios.md](../methodology/peak-memory-ios.md)).
A mapped load would leave the weights out of that counter until they are touched,
which is the other reason the timing and memory cells pin them. A cell that wants
a different mode says so, and the resolved value is reported (on eval, the cell's
own flags), so two rows taken under different modes are told apart in the record
rather than inferred from the client version.

### 1.4 Job

Jobs are the iOS-only concept layered on top of benchmarks. Through the SwiftUI
screens (`NewJobView`, `RunningJobDetailPage`, `JobDetailView`) a user picks a model
and a set of benchmarks, kicks off a run, and watches progress through the
semantic `BenchmarkProgress` enum (`Runtimes/BenchmarkProgress.swift`);
`.attempt` per measured rep and `.sample` per finished eval sample. Cancellation
flows through the readiness gate (`ReadinessCallback` / `ReadinessOutcome`),
which runs before a timing cell's model loads and again before each measured
rep: a `.cancelled` outcome throws `RuntimeError.cancelled`, giving the UI a clean
way to stop an in-flight benchmark. The gate's criteria are in
[end-to-end-latency.md](../methodology/end-to-end-latency.md#system-readiness-control).

### 1.5 Registration and Submission

Management-server networking and request signing live in Swift, in
`Networking/PipetteMgmtClient/`. It is a hand mirror of the `pipette-mgmt-client`
crate: one Swift file per Rust module, each name carried across with nothing
changed but the case (`Auth.swift` ↔ `auth.rs`, `ClientApi.swift` ↔
`client_api.rs`, and so on; the crate root is `PipetteMgmtClient.swift`), the
crate's own names for the types and the endpoints, and its order within
`Types.swift`. The stack every call runs on is `HttpClient` in
`Networking/PipetteHttp/`, the mirror of the `pipette-http` crate. Like the crate,
it neither logs nor prints: every call returns its outcome and the caller decides
what to say about it. Every call is also typed to one error,
`throws(PipetteMgmtClient.Error)`, which states on the signature what the crate's
`Result<T, Error>` states on its own, so `catch` binds the client's error without
a cast and `try` chains calls where the crate writes `?`.

`PipetteMgmtClient` uses Apple frameworks:

- `URLSession` sends registration and result-submission requests.
- `KeychainHelper` generates a CryptoKit signing keypair and stages the private
  key in the Keychain during registration. `ClientRegistrationStore.putPrivateKey`
  writes it to `identity/id_ed25519` and to the Keychain, only after the server
  accepts the public key.
- `CryptoKit.Curve25519.Signing.PrivateKey` produces the `X-Signature`.

The signed headers and payload are the management server's device-signing
contract
([pipette-mgmt docs/authentication.md §2](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#2-request-authentication)).

#### Capability reporting

The planner matches jobs against the device profile and capability set. The app
reports both only when it runs as a worker, as the CLI does: the planner worker
sends `PATCH /clients/me` (`refreshProfileAtStartup`) before its first claim and
waits out `reindex_pending`. The inputs drift between runs (an OS
update, a build from a different llama.cpp commit), and an unchanged resubmit is a
server-side no-op. The first report after registration is always a change, so the
worker's first start waits out one reindex before it claims, as the CLI's does.

Nothing else reports: registration sends no profile, and neither a launch nor a
one-shot `headlessrun` verb does. Whatever claims as the client owns its profile.
On a farm phone that is `pipette-ios-proxy`, which holds a lease as this same
client; a report from the app would replace the proxy's profile and relinquish the
lease. A one-shot launch (every `headlessrun` verb but `worker`) also drains no
stranded results (`HeadlessRunner.isOneShot`).

`installedRuntimeCapabilities` (`Client/Worker/Profile.swift`) builds the runtime
flags: `runtime:llama_cpp`, `runtime:mlx`, `runtime:onnx_genai` and
`runtime:apple_foundation`, each paired with versioned `runtime:<name>:<build>`
flags where a build id exists (`apple_foundation` has none). `Device/Capabilities.swift`
adds the hardware flags. Every level is reported because the planner compares each
flag as a whole, opaque string, so a versioned flag does not imply the general one.
llama.cpp gets a versioned flag for both its commit and its tag, for main's pin
alone: the worker runs every llama.cpp claim on main.

Levels are reported generously rather than minimally: matching is set containment,
so an extra flag only widens what the client matches. MLX and ONNX Runtime GenAI are
each pinned by several independently versioned sources, so each source gets its own
flag, as mgmt recommends for composite runtimes
([pipette-mgmt docs/client-integration.md §3](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/client-integration.md#composite-runtimes-get-a-flag-per-component)):

| Flag | Pins |
|---|---|
| `runtime:mlx` | any MLX build |
| `runtime:mlx:0.32.3` | mlx-swift, bare: the form a spec-generated plan carries, so it matches as-is |
| `runtime:mlx:mlx-swift=0.32.3` | mlx-swift, named |
| `runtime:mlx:mlx-swift-lm=3.32.3` | the MLXLLM model/inference code |
| `runtime:mlx:swift-transformers=1.3.4` | the tokenizer |
| `runtime:onnx_genai` | any ONNX Runtime GenAI build |
| `runtime:onnx_genai:<genai>` | onnxruntime-genai, bare: what `pipette-cli` derives for an `onnx_genai_ios_pipette` cell |
| `runtime:onnx_genai:onnxruntime-genai=<genai>` | the generation loop, named |
| `runtime:onnx_genai:onnxruntime=<ort>` | the graph runtime |

All three MLX packages affect output (swift-transformers changes tokenization, and
so the prompt encoding), which is why mlx-swift alone does not identify a build.
An author pinning an exact build lists every package in `requires`.

In the proxy's pool mode, `tools/pipette-ios-proxy/src/pool/capabilities.rs`
repeats this set. It derives the set from the runtime descriptors it reads off a
build, so a phone reports the same flags whether it claims for itself or the proxy
relays for it, and one plan matches it either way. Two differences remain:

- llama.cpp: a descriptor carries one version (the tag when the build has one, the
  commit otherwise). So in pool mode a tagged build reports no
  `runtime:llama_cpp:<commit>` flag, and a plan that pins that commit matches the
  phone only when it claims for itself.
- Pool mode reports every pin a build carries, where the app reports main's alone.

The proxy has its own implementation because each client owns the flags it reports.
Its tests mirror the ones here case for case, so change the two together. A level
added on one side only makes a phone stop matching a plan that pins it in one of
its two modes, with nothing on the plan side to show why.

The proxy's version mode, which installs a release build per job, reports
`client_version:<version>` flags instead of these `runtime:` flags
([DESIGN.md §9](../../tools/pipette-ios-proxy/DESIGN.md#9-version-mode-the-build-a-job-pins)).

The app itself also reports its build as `client_version:<version>`, for example
`client_version:2026.08.11-0123abcd`. The flag makes the app build visible to job
matching and to a fleet query before any job finishes, and is the backstop for an
app feature that carries no flag of its own: a plan pins the build that introduced
the feature. A build that names no release, which covers a local build and the
TestFlight path, reports nothing in this namespace. `Bundle.releaseVersion` appends
the flavor suffix, so the [private-API build](private-thermal-release-build.md)
reports `client_version:<version>-internal` and matches no bare `<version>`
requirement. That is correct: the private-API build's readiness gate and its
`max_memory_usage` counter both differ, so its results are not comparable.

Flag spelling rules (lowercase, whitespace-free, reserved namespaces) are the
server's
([pipette-mgmt docs/client-integration.md §3](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/client-integration.md#spelling-is-part-of-the-contract)).

#### Planner worker (opt-in)

Settings → **Planner worker** turns the device into a pull client of the
management planner (same protocol as desktop `pipette worker`; the protocol is
[pipette-mgmt docs/client-integration.md §4–5](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/client-integration.md#4-the-work-loop)
and [docs/queue.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/queue.md)).
Nothing saves the toggle: every launch starts with the worker off, so no launch
starts it on its own. That includes the argument-less relaunch Single App Mode makes
after every exit on a farm phone, which would otherwise claim as this client beside
`pipette-ios-proxy`. Headless: `worker` (or its `settings run` spelling) starts the
claim loop on the **app-wide** `JobRunner` and keeps the process alive until that
session ends (launch-as-worker via `devicectl`). While running (and the app is in the
foreground):

1. `PATCH /clients/me` refreshes the device profile and capabilities, then holds
   until the `reindex_pending` gate lifts or `reindexWaitBudget` (5 minutes of
   elapsed time) runs out (`refreshProfileAtStartup`). The worker waits because it
   takes leases: a profile change voids its queue standing, so a claim before the
   gate lifts is answered `204` and idles the worker for a whole idle wait. A
   refresh that fails ends the session, and so does a profile that does not read
   `approved`, as `pipette worker` bails on both. See
   [Capability reporting](#capability-reporting).
2. The app loops on `POST /plans/claim` (`pollClaim`). A `204`
   sleeps the idle wait plus jitter; a transient failure retries after 1 s,
   doubling to a 60 s cap, until the session stops (`claimWithRetry`); any
   other failure ends the session.
3. The claim is the plan-types cell, typed once by `runSpecFromClaim(_:)`
   and checked by `ClientRunSpec.validated`: `runtime` selects the iOS `Runtime`
   variant (`llamacpp_ios_pipette` / `mlx_ios_pipette` / `onnx_genai_ios_pipette` /
   `apple_foundation`); `model` is the `Model`; `runtime_flags` and `model_flags`
   are the flat `…FlagRef` forms, resolved against the cell's own axes by
   `RunCell.prepare`; `benchmark_flags` carries the cell's readiness gate, and the
   policy it resolved to is what the result reports. A desktop runtime fails to
   decode at all, and every `UnrunnableClaimError` is non-retriable. The validated
   cell reaches the manifest whole, never copied field by field.
4. The benchmark body is the cached one once `GET /benchmarks/{id}` revalidates its
   ETag, else the body that answer carries, cached on the remote half
   (`ensureClaimBenchmarkCached` over the shared `ensureRemoteBenchmarkCached`, the
   crate's `ensure_claim_benchmark_cached`; see [Benchmark](#13-benchmark) for each
   answer). A `404`, or a definition this build cannot convert, is a terminal
   failure; a fetch of an uncached definition that fails in transit is retriable.
   The executor runs the definition this returns.
5. The cell runs through the normal `JobExecutor` path (no auto-submit);
   heartbeats run at half `time_window`, or at `heartbeat-secs=` with a warning
   when that is longer. Weights the device does not hold are downloaded there,
   by the `ensureModel` every run path goes through, so a claim is not refused
   for naming a model this device has never seen. The heartbeat covers the
   fetch, and a fetch that fails is a retriable failure. The fetch is admitted
   against the storage quota before it starts ([storage-quota.md](../storage-quota.md)).
6. Success and failure are both submitted with the claim's `job_id` through the
   same backoff (`submitPlanResultWithBackoff`: 1 s doubling to 30 s, 32
   attempts). Exhausting it drops the outcome, as the crate's worker does. A stop
   mid-submit leaves the result in `results/remote/pending/` with no record, so no
   sweep picks it up (one would post it without the claim's `job_id`), and the
   lease running out answers the claim. After the submit, the result is filed
   under `results/remote/synced/` with its submission record and the cell's
   `serverJobId` (`storeCompletedResult`), so no sweep posts it again.

`max-jobs=` counts every claim the worker answered: a refused claim, a lost
lease and a dropped submit included, as the crate counts every `Ok` of
`run_claimed_job`. Only a submit the server refused outright does not count.

The loop stops in the background (same jetsam/timing reasons as local jobs),
and nothing restarts it: the toggle shows off, and a `worker` launch's session
ends there (`outcome=stopped`). An in-flight
cell is cancelled and a **retriable** failure is submitted when possible so the
lease is not left to time out silently. A claim `403`, or an unapproved profile,
disables the toggle and stops until an operator approves the client.

Load knobs are typed in the plan-types `RuntimeFlags` cells for llama.cpp and MLX
(ONNX Runtime GenAI and Apple Foundation have none), checked by
`ClientRunSpec.validated` and resolved onto the run by `RunCell.prepare`.
The model flags `enable_thinking` and `reasoning_effort` are refused rather than
applied: this build drives the bare chat template, so neither would take effect
and the run would measure a different cell.

#### Pre-auth key onboarding

`SetupView` has an optional **Pre-auth key** field. When filled, the token is
sent as `client_preauth_key` on `POST /clients/register` and the device comes up
already `approved`, with no manual `clients approve` step on the management side.
Left blank, the field is omitted from the request body.

The token is **transient**: it is passed to the register request and never
written to the identity store or logs. Because the private key and the
registration persist only *after* the server accepts, a rejected key leaves no
partial identity, and a valid key then works. A rejection surfaces as an
onboarding error. Key minting and the server's answers are in
[pipette-mgmt docs/authentication.md §3.2](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/authentication.md#32-pre-auth-keys).

#### Authentication ([Clerk](https://clerk.com))

[Clerk](https://clerk.com) is a third-party authentication and user-management
service. The app integrates it through the `clerk-ios` Swift package
(`ClerkKit`). At launch `PipetteApp` shows `ClerkAuthGateView`, or a
configuration-error screen if the Clerk settings are absent. Clerk is configured
at build time. See [build.md](build.md#clerk-configuration).

**In an app run, the gate reads the device identity before the Clerk
session.** `reduceAuthGate`
(`Views/AuthGateScreen.swift`) is a pure function over six inputs: whether
this process is a CLI run, the Clerk load state, the signed-in user id, whether
the device holds a registration record, whether it holds a signing key, and the
account the registration is linked to. A device identity is both halves. That
pair is what `ClientRegistrationStore.mgmtSession()` authenticates with, so a
device that holds it can already claim, run, and submit work. The gate admits it
and shows `MainTabView`, with no Clerk session. This is the path a `headlessrun
auth import` device takes.

The two halves are separate inputs because a device can hold one without the
other. A record with no key cannot sign, and setup cannot replace the key,
because `ClientRegistrationStore.ensureEmpty()` refuses over the record the
device already holds. The gate gives that device its own screen instead of
routing it to setup.

The gate reduces to seven screens, in this order:

1. **Headless**: the process was launched with `headlessrun`. The screen shows
   the logo, the words `HEADLESS MODE`, and the device's `clientId`. It carries
   no control, because the operator drives the run from a terminal.
2. **Mismatch**: the device holds an identity linked to one account, and a
   different account is signed in.
3. **The app**: the device holds an identity.
4. **Key missing**: the device holds a registration record and no signing key.
   The screen names the `clientId` and offers the same confirmed "Delete device
   identity" the mismatch screen uses. `auth import` of a bundle exported while
   the key was present is the repair that keeps the `clientId`.
5. **Loading**: Clerk has not loaded yet. The identity arms come first because
   they are local reads, so a registered device does not wait for the network.
6. **Setup**: a user is signed in and the device holds no identity.
7. **Sign-in**: nothing else applies.

**Clerk creates a registration, and links it to an account.** A device with no
identity must sign in, because `SetupView` records the Clerk identity
(`clerkUserId`, `clerkSessionId`, `clerkPrimaryEmail`, `clerkLinkedAt`) on the
record it creates, and prefills the contact-email field from the Clerk email.

A device the gate admitted without a session signs in from **Settings →
Account → Sign in**. That links the registration it already holds, so the
`clientId`, the signing key, and every result submitted under them are kept. The
same screen offers **Delete Account** only while a session is live, because
there is no Clerk account on the device to delete without one.

## 2. How the iOS app uses llama.cpp

The native Swift `LlamaCpp` ops depend on a narrow part of `llama.cpp`, called
through the `llama` module (`import llama`):

- the C API for model loading, tokenization, KV-cache prefill, and greedy
  sampling (`llama_model_load_from_file`, `llama_tokenize`, `llama_decode`,
  `llama_sampler_*`, `llama_memory_clear`, …)
- the Metal and Accelerate backends compiled into each llama.cpp framework

Each llama.cpp pin is a framework, `Frameworks/<pin>.framework`, that the app
embeds but does not link. The [private-API build](private-thermal-release-build.md)
carries every pin, and the App Store app carries only main ([build.md](build.md#llamacpp-variants)). A process that
runs cells loads exactly one pin with `dlopen` (`LlamaPins`): the one that a
headless run's `runtime=` names, or main if the run names none. A loaded
framework cannot be unloaded, so the process refuses a second. Swift compiles
once, against main's headers, and `Native/PipetteLlama.c` forwards each C call
to the table that the loaded framework exports (`ios/native/pipette_llama.h`).

Each framework's `Info.plist` names its repository, its commit, and its tag if
it has one. The app lists its pins from these values without loading them. A
load fails if the identity compiled into the framework differs.
`LlamaPin.submissionVersion` (the tag when the pin has one, the commit otherwise)
is the `runtime_version` a submission carries. There is no concept of a "runtime
release" on iOS the way there is for `pipette-llamacpp`.

Unlike the desktop runners, there is no `llama-bench` or `llama-server` process:
every benchmark type is implemented directly against the C API in
`LlamaExecute`. So the iOS path must reproduce behavior that the upstream binaries
supply, and each piece names the tool it reproduces:

- The rep loop and its reduction are `pipette-ops`' `measurement/mod.rs`.
- Decode feeds tokens with no sampler in the loop, as `llama-bench`'s `test_gen` does.
- End-to-end masks the end-of-generation tokens, as the crate asks `llama-server`
  to, and stops after the last sample, as `llama-server` does.
- Eval applies the chat template and samples at the eval's own temperature.
