# Pipette iOS Architecture

The Pipette iOS app runs its inference engines **in-process**. There is no
separately installed runtime. Unlike the desktop runners, which shell out to
upstream `llama.cpp` release binaries, iOS compiles its engines directly into
the app and runs them on-device via Metal. It runs two engines, selected
automatically by model format:

- **llama.cpp**: stateless Swift ops (`LlamaCpp`) calling the llama.cpp C API
  directly (`import llama`); the llama.cpp sources are compiled into the app.
  **No Rust, no FFI bridge.**
- **MLX**: a Swift-side runtime (`Runtimes/MLX/`) built on the `mlx-swift` /
  `mlx-swift-lm` packages.

Both engines are pure Swift over Metal and sit behind one benchmark interface,
so the rest of the pipeline (payload → CSV → submission) is identical regardless
of which engine ran a cell. See [Inference engines](#inference-engines) for the
split. For how the llama.cpp sources are compiled into the app, see
[build.md](build.md).

> **History.** Through mid-2026 the llama.cpp path was a Rust core
> (`crates/pipette-ios`) exposed to Swift via [uniffi](https://mozilla.github.io/uniffi-rs/).
> That crate has been removed: the path is now native Swift over the llama.cpp C
> API. The shared Rust benchmark kernel (`native/benchmarks.rs`) and the Rust
> FFI live on only in the Android client.

## Inference engines

The app runs benchmarks on **two in-process engines**, selected automatically by
a model's on-disk format. Neither is a separately installed runtime. Both are
compiled into the app and run on-device via Metal.

| Model format | Engine | Implementation | Submission identity |
|--------------|--------|----------------|---------------------|
| `*.gguf` file | **llama.cpp** | `Runtimes/Llama/`: stateless `LlamaCpp` ops over the llama.cpp C API (`import llama`), driven by `LlamaBenchmark`; the sources are compiled into the app by `ios/build-llama.sh` | `runtime_name = ggml-org/llama.cpp:ios`; `runtime_version` = `llamaCppCommit()` (the vendored commit, stamped at build time) |
| safetensors directory (`config.json` + `*.safetensors`, an mlx-lm export) | **MLX** | `Runtimes/MLX/`: `MLXRuntime` (scoped load) + `MLXGenerate` / `MLXBenchmark` / `MLXProbes`, built on the `mlx-swift` / `mlx-swift-lm` Swift packages | `runtime_name = mlx-swift:ios`; `runtime_version` = the linked `mlx-swift` pin |

The runtime/model concept lives in `Runtimes/Runtime.swift`, mirroring the Rust
`pipette-plan-types` crate: a config-carrying `Runtime` sum and a loadable `Model`
sum, with all behavior derived via `switch` and no "kind" projection (the peer of
Rust's `match self` / `match (model, runtime)`). `JobExecutor` builds both
*definitions* at the boundary (`Model.detect` from the on-disk path, then
`Runtime.forModel`; the engine is derived from the model, format → engine 1:1), and
`Engine.run` dispatches by switching on the bound `Runtime` alone, as upstream's
`dispatch_run` does; each engine's `require*` refuses a model it cannot load, and
the pairing rule itself is enforced at claim time; assembling the loaded model
**inside the benchmark scope**
(`withInference` / `MLXRuntime.withFreshModel`) so the resource never escapes. The
one config-less identity, `RuntimeChoice`, is purely a UI/CLI selection input (the
New Job picker, the headless `runtime=` flag, model-list filtering): not part of
the runtime definition, and never persisted as a separate tag (a cell's engine is
derived from its model path).

The two engines are structural twins: each returns a typed `BenchmarkResult` from
the *same* `BenchmarkCatalog` definitions, sharing the `BenchmarkMeasurement` core
(1 warm-up + 5 measured reps, readiness gate before each measured rep), so the
downstream payload → CSV → submission pipeline is identical regardless of which
engine ran the cell.

### MLX is currently UI-gated

`RuntimeChoice.mlxVisibleInUI` is `false`, so the New Job flow only offers
`llama.cpp`; MLX is not user-selectable yet. The path is fully functional,
though: existing MLX jobs and results still run and label correctly (the flag
only gates what's *selectable*), and it is exercised two other ways:

- **Headless runner** (`Headless/HeadlessRunner.swift`, with the verbs in `Commands/`,
  one file per command group as `pipette-cli/src/commands/` has): launch the app with
  `headlessrun runtime=mlx …` arguments to run benchmarks with no UI taps, e.g.

  ```bash
  xcrun devicectl device process launch --device <UDID> --console \
    ai.liquid.liquid-pipette headlessrun runtime=mlx batch=512 \
    metrics=prefill,decode,maxmem offsets=256,512,1024,2048,4096
  ```

- **Parity validation** (the `coherence` metric in `Headless/HeadlessRunner.swift`):
  compares an engine's greedy output against baked Python `mlx_lm` reference
  values to catch routing/correctness regressions.

To re-surface MLX in the New Job flow (the runtime picker and MLX caption text),
flip `mlxVisibleInUI` to `true`.

## Analytics (PostHog)

Product analytics is a hand-picked, boundary-only event set behind the
`Analytics` seam (`Support/Analytics.swift`); the Android client duplicates the
same event and property names verbatim so both land in one PostHog project.
`Analytics.start()` no-ops unless `PostHogConfiguration.isComplete`, leaving the
sink as `NoOpAnalytics`. The `phc_…` project token lives in `Info.plist`
alongside `SentryDSN`. It is write-only and public by design.

Autocapture, session replay, element interactions and feature-flag preloading
are all off, because instrumentation overhead corrupts on-device LLM
measurements, the same reason `SentryConfiguration` disables tracing,
auto-performance instrumentation and the app-hang watchdog. PostHog's own error
tracking (`errorTrackingConfig.autoCapture`) is pinned off as well: the SDK
vendors PLCrashReporter, and a second crash reporter installing signal handlers
would fight Sentry for them.

`surveys` and `rageClickConfig.enabled` need turning off by name: both default to
**true** and neither is implied by the switches above. Left alone, launch logs
`PostHogSurveyIntegration installed` and `PostHogRageClickIntegration installed`;
the latter hooks touch delivery for the life of the process to spot rapid taps,
which is exactly the ambient work this app refuses everywhere else. This is the
iOS twin of Android's `sessionReplayConfig.captureLogcat`: a default-on
integration no obvious flag disables. With both off, no integration installs at
all. Re-check on every SDK bump.

`capturePushNotificationSubscriptions` and `capturePushNotificationOpened` are the
same story one version later: both arrived default-**true** in 3.69.0, and the
subscription one swizzles the app delegate to register this device's APNs token
with PostHog for Workflows delivery. This app sends no push notifications, and a
push token is a device identifier that neither `PrivacyInfo.xcprivacy` nor the
taxonomy accounts for, so both are off. posthog-android 3.58.0 added the identical
pair, and `Analytics.kt` turns them off for the same reasons.

`config.debug` is on in DEBUG builds, matching Android's `debug =
BuildConfig.DEBUG`. Without it the SDK is entirely silent, which is what made the
opt-out unverifiable on a simulator: there was no way to tell a suppressed event
from a captured one.

No event fires inside a measurement window. `job_started` is captured and
flushed in `JobExecutor.run` *before* the detached run task begins, so the queue
is empty entering the run and the SDK's periodic flush timer has nothing to send
while a cell is timed. The one event that can fire mid-job is
`results_submitted`, from the per-cell upload `JobExecutor` performs between
cells (PIP-358); `ResultUploader` flushes immediately after capturing it, so the
next cell still starts on an empty queue. Events carry the registration `clientId` as the distinct
id; no email, organization, Clerk identity, error message, or feedback body is
ever sent. `PrivacyInfo.xcprivacy` declares the resulting collection
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
SDK's own persisted copy, which was caught doing nothing at all on
posthog-android 3.19.0, and fixed upstream in 3.55.1 (see
`docs/pipette-android/implementation.md`). The seeding stays on both platforms
regardless: one launch is all it takes to leak an event from a device whose owner
turned analytics off. `UserDefaults` is synchronous, so the read costs nothing at
launch.

Verified on an iPhone 17 Pro simulator: with the flag set, launch logs
`PostHog is in OptOut state.` and captures nothing (no `Queued event`, no batch);
clearing it restores `Queued event 'app_launched'` → `batch sent successfully.`

One deliberate asymmetry with Android: an opted-out device here still logs
`Remote config called successfully.` posthog-ios deprecated `config.remoteConfig`
and ignores it ("is now always enabled"), whereas Android sets it false and goes
fully silent. posthog-android 3.58.0 copied the deprecation text but not the
behaviour. Its `setup` still branches on the flag, so that platform stays silent.
The request carries the public project token and no user data.

`AnalyticsEventsTests` pins the wire contract (event names, property keys,
`RunSource` raw values and the outcome mapping), and `PostHogConfigurationTest`
pins the same set on Android. Neither can enforce true parity (each platform
asserts its own copy); they exist because a rename on one platform alone doesn't
break a build, it silently splits a funnel into two unrelated-looking events.

## Components

```
vendor/llama.cpp/         llama.cpp sources — the shared submodule (also Android's)
ios/
  build-llama.sh          builds a patched copy of vendor/llama.cpp (Metal +
                          Accelerate) into Generated/Llama/libllama.a + `llama` module
  patches/                NNN-*.patch series applied to the build copy (e.g.
                          001-ggml-metal-oom-nullcheck.patch)

ios/Pipette/Pipette/
  PipetteApp.swift        app entry point
  Ops/                    the engine-agnostic half — one file per `pipette-ops` module
    BenchmarkMeasurement.swift  rep loop / stats / timer (measurement.rs)
    Readiness.swift       ReadinessOutcome / ReadinessCallback / the gates / RepObserver
                          (readiness.rs)
    ThermalSeries.swift   per-rep thermal collector the observer feeds (thermal_series.rs)
    EvalCompletionsStore.swift  per-sample eval resume (eval_completions.rs)
    PromptSeed.swift      shared prompt/token seeding (prompt_seed.rs)
  Runtimes/               the engines themselves, plus the probes only a phone has
    RuntimeSupport.swift  BenchmarkProgress / RuntimeError / bound-model errors
                          + socTemp() / llamaCppCommit() (native, ex-Rust-FFI)
    MemoryGate.swift      jetsam pre-flight; IMUThermometer/ProcessMemory: device probes
    Llama/
      LlamaCpp.swift            stateless ops over the llama.cpp C API + LlamaModel handle
      Inference.swift           the ops-as-data witness + withInference scoped assembly
      LlamaBenchmark.swift      stateless benchmark kernels (warmup+reps, results)
    MLX/
      MLXRuntime.swift          scoped model load (withFreshModel) + run entry
      MLXGenerate.swift         prefill/decode generation primitives
      MLXBenchmark.swift        stateless benchmark kernels over BenchmarkMeasurement
      MLXProbes.swift           coherence/parity probes
  Native/PipetteThermal.m  private SoC die-temp probe (ObjC); Metal allocated-size
                           is read in Swift via MTLDevice.currentAllocatedSize
  Pipette-Bridging-Header.h  exposes the ObjC probe to Swift
  Commands/               one file per headless command group, as pipette-cli has
  Client/                 claim decoding, the worker, profile reporting
  Identity/               IdentityStore, the registration record, client settings,
                          the signing key — as pipette-cli's identity/ has them
  Benchmarks/             BenchmarkStore over both catalog halves, the standard
                          local set, the remote pull — as pipette-cli's benchmarks/
  Results/                ResultsStore over results/, the location ladder, upload
  Storage/                the data root, quota
  Device/                 DeviceProbe — hand mirror of pipette-device
  PlanTypes/              hand mirrors of pipette-plan-types
  Artifacts/              hand mirrors of pipette-artifacts
  Runtimes/               llama.cpp / MLX / Apple Foundation engines
  Jobs/                   the batch noun — iOS-only, no CLI counterpart
  Models/                 DownloadCoordinator + model download/install plumbing
  Contracts/              the remaining iOS-only Codable types
  Networking/             ManagementClient, AuthIdentity, CollectorEndpoint
  Headless/               the argv grammar, the runner, DeepLinkRouter
  Views/                  SwiftUI screens (Setup, Benchmarks, Models, Jobs, …)
  Support/                Analytics, AppLog, Coding, shared string helpers
  Generated/Llama/        build-llama.sh output — do not edit
```

`ios/build-llama.sh` stamps the resolved llama.cpp commit into
`Generated/Llama/LlamaCppBuildInfo.swift`; Swift reads it back through
`llamaCppCommit()` and uses it as `runtime_version` in submission payloads.
there is no concept of a "runtime release" on iOS the way there is for
`pipette-llamacpp`.

## 1. Core Concepts

### 1.1 Runtime

There is no separately installed runtime on iOS. Both inference engines are
compiled into the app (see [Inference engines](#inference-engines)). For the
`llama.cpp` engine, the version is pinned by the `vendor/llama.cpp` submodule;
updating it means bumping the submodule commit and rebuilding the app.
The MLX engine's version tracks the `mlx-swift` package pin in the Xcode project.

Metal and Accelerate are enabled by `build-llama.sh`, so on-device `llama.cpp`
runs use the GPU through Metal when `n_gpu_layers > 0` and fall back to
Accelerate on CPU otherwise. MLX likewise runs on the GPU through Metal, via
`mlx-swift`.

### 1.2 Model

A model is a GGUF file or an MLX bundle stored in the app's local storage. The
Pipette app downloads models from Hugging Face via `DownloadCoordinator` and
tracks them through `LocalStorage`. `LlamaCpp.load` takes an absolute file path and returns a
`LlamaModel` handle (owning the llama.cpp model + context + greedy sampler),
assembled inside the benchmark scope by `withInference` and freed deterministically
when that scope ends.

### 1.2.1 Local storage

Local job data is versionable JSON under the app's Application Support data
root:

```text
Pipette/
  identity/                 # the identity store's root, as pipette-cli names it
    registration.json       # IdentityRegistration, snake_case as the CLI writes it
    settings.json           # ClientSettings; today the storage quota
  benchmarks/               # the catalog, as pipette-cli's benchmarks/
    local/<id>.json         # generated by `benchmarks init-local`; never submitted
    remote/index.json       # the GET /benchmarks list
    remote/sync.json        # list ETag + per-id ETag map
    remote/<id>.json        # one synced definition, eval samples included
  jobs/
    <jobId>/
      manifest.json         # the batch; results are keyed by cell, not nested here
  results/                  # the results store, as pipette-cli's results/
    local/<cellId>/         # from a generated benchmark; never submitted
    remote/pending/<cellId>/
    remote/synced/<cellId>/ # payload.json, extras.json, metrics.json, submission.json
```

#### Model store

Downloaded models live under `models/` as one **entry** per model, addressed by
a flat `ModelStorageKey`: the spec's identity segments sanitized to
`[A-Za-z0-9._-]` and joined with `__`, capped at 32 characters with an 8-hex
SHA-256 fold. It mirrors `pipette-artifacts`' `ModelStorageKey`, so the same
coordinate keys the same string on the CLI and on iOS.

```text
models/
  LiquidAI__LFM2.5-350M-GGUF__LFM2.5-350M-Q4_0.gguf/
    manifest.json   # { manifest_version, declared, fetched_at, last_used_at }
    blobs/
      LFM2.5-350M-Q4_0.gguf
  org__vl-GGUF__vl-Q4_0.gguf__mmproj-vl-F16.gguf/
    manifest.json
    blobs/
      vl-Q4_0.gguf          # a VL model's weights and projector share one entry
      mmproj-vl-F16.gguf
  LiquidAI__LFM2.5-350M-MLX-4bit/
    manifest.json
    blobs/                  # the MLX bundle: config, weights, tokenizer
```

One entry, one manifest, one delete. `models/` and every entry are marked
`URLResourceKey.isExcludedFromBackupKey` because model binaries are large and
re-downloadable.

The manifest is the unit of accounting: a directory under `models/` is a model
only if it carries a manifest this build can read, and everything else there is
garbage the sweeper reclaims. That is also why there is **no migrator**: the
pre-entry `models/<org>/<repo>/<file>.gguf` tree is simply not a valid entry, so
the first sweep reclaims it and those models re-download.

#### Storage quota

`metadata/settings.json` holds `storage_quota_bytes`, defaulting to
`min(16 GiB, 25% of volume capacity)`. Enforcement is fetch → publish → sweep →
return, run inline on the `@MainActor` install-completion path in
`DownloadCoordinator`: the artifact lands first, then `sweepToQuota` reclaims
garbage (manifest-less entries, unreadable manifests, and hub-cache snapshots the
MLX installer left behind) and then evicts models least-recently-used until the
store is back under the cap. Peak disk is therefore the quota plus the newest
artifact.

`last_used_at` is refreshed on every resolve in `JobExecutor`, best-effort. A
failed write never fails a run. Pins (the just-installed entry, every in-flight
download, and every model a running or paused job needs) are assembled by the
coordinator and passed down; running out of unpinned entries while still over
quota warns and continues rather than failing the run. A single artifact larger
than the whole quota is refused before the fetch starts, with an error pointing
at the Settings limit. Settings shows used / quota on a row that is itself the
limit control (a `StorageLimitOption` preset ladder led by the computed default,
so the default is always one tap away) beside a "Free up space" action, and
every eviction is logged through `AppLog.storage`. Choosing a limit only writes
`storage_quota_bytes`: it never evicts, so lowering it below current usage is
allowed and the card discloses the over-limit state until the next download's
sweep or an explicit "Free up space" reclaims.

Sizes are measured by walking the entry at calculation time (`DiskUsage`:
recursive, symlinks not followed, `st_blocks * 512`) and are never persisted.
The policy this implements is `docs/storage-quota.md`.

### 1.3 Benchmark

A benchmark is the same `BenchmarkDefinition` used by the desktop runners. On
iOS, the fixed benchmark catalog is bundled in Swift through `BenchmarkCatalog`
(merged over any server-synced definitions), so first launch does not depend on
management-server catalog sync.

The benchmark entry point is `Engine.run`, which dispatches on the `Runtime` sum:
the llama path runs through `LlamaBenchmark` (the MLX peer is `MLXRuntime.run`).
It takes a typed `BenchmarkDefinition` and assembles the model fresh inside the
scope (`withInference`), runs the benchmark, and frees it. Every cell is an
isolated load so memory resets between cells (and `max_memory_usage` can observe
the load itself). It returns a typed
`BenchmarkResult`; serialization to the management server's `SubmissionPayload`
happens at the persistence boundary, not in the engine.

### 1.3.1 mmap and benchmark memory semantics

iOS intentionally loads GGUF models with `llama_model_params.use_mmap = false`.
That mirrors the desktop `pipette-llamacpp` path, where `llama-bench` is run with
`--mmap 0` unless the operator explicitly overrides it. The benchmark contract is
therefore: can this model run without relying on memory-mapped weights?

This matters because llama.cpp's mmap path can map model tensors directly from
the GGUF file, and Metal can wrap those file-backed pages with no-copy buffers.
On iOS, that can let prefill/decode proceed for a model that only fits because
the weights are file-backed and only the touched pages are resident at a given
moment. Those are real executions for an mmap-enabled app mode, but they are not
valid rows for the no-mmap benchmark baseline. If `mmap=true` is required for the
model to load or run, the no-mmap benchmark should fail instead of reporting
throughput numbers.

This also affects memory reporting. `max_memory_usage` samples Metal's
`currentAllocatedSize` across a fresh model load. With `use_mmap=false`, model
loading takes the stricter allocated-memory path. If iOS re-enables mmap,
`currentAllocatedSize` may undercount no-copy mapped model bytes, so that result
would need a memory breakdown or an explicit second metric before it is
comparable to no-mmap rows.

### 1.4 Job

Jobs are the iOS-only concept layered on top of benchmarks. Through the SwiftUI
screens (`NewJobView`, `RunningJobView`, `JobDetailView`) a user picks a model
and a set of benchmarks, kicks off a run, and watches progress through the
semantic `BenchmarkProgress` enum (`RuntimeSupport.swift`); `.attempt` per
measured rep and `.sample` per finished eval sample. Cancellation flows through
the readiness gate (`ReadinessCallback` / `ReadinessOutcome`), which runs before
each measured rep polling the SoC die temperature via `socTemp()`: a
`.cancelled` outcome throws `RuntimeError.cancelled`, giving the UI a clean way
to stop an in-flight benchmark.

### 1.5 Registration and Submission

Management-server networking and request signing live in Swift. `ManagementClient`
uses Apple frameworks:

- `URLSession` sends registration and result-submission requests.
- `KeychainHelper` generates a CryptoKit signing keypair, stages the private key
  in the Keychain during registration, and promotes it only after the server
  accepts the public key.
- `CryptoKit.Curve25519.Signing.PrivateKey` produces the `X-Signature`.

The client keeps the management-server header contract (`X-Client-Id`,
`X-Timestamp`, `X-Nonce`, `X-Signature`). The signature covers the `v1` payload,
six newline-separated fields: `v1`, the HTTP method, the request target the
server receives (base-URL path prefix and query string included), the timestamp,
the client id, and the nonce. Signing them scopes the signature to that method
and target, and the nonce makes it single-use, so a captured signature cannot be
replayed. The body is still not covered.

#### Capability reporting

The planner matches jobs against the device profile and capability set, so the
app reports both at two moments, independent of whether the planner worker is on;
a device that never claims still has to be matchable:

- **At registration.** `POST /clients/register` carries the profile fields and
  `capabilities` alongside the credentials, so a new client is matchable in one
  request.
- **At every launch.** `ProfileReporter` sends `PATCH /clients/me` from
  `PipetteApp`'s `onAppear`. The inputs drift between runs (an OS update, a build
  from a different llama.cpp commit), and an unchanged resubmit is a server-side
  no-op. Best-effort: a failure is logged and never blocks the UI.

`Capabilities` owns the flag set; `runtime:llama_cpp`, `runtime:mlx`,
`runtime:apple_foundation`, each paired with a versioned `runtime:<name>:<build>`
where a build id exists (`apple_foundation` has none). Every level is reported
because the planner compares each flag as a whole, opaque string, so a versioned
flag does not imply the general one.

Levels are reported generously rather than minimally: matching is set containment
and the server caps neither flag count nor length, so an extra flag only widens
what the client matches. MLX is pinned by a three-package stack, so each package
gets its own flag and a plan pins whichever ones matter:

| Flag | Pins |
|---|---|
| `runtime:mlx` | any MLX build |
| `runtime:mlx:0.31.6` | mlx-swift, bare: the form `runtime_capability_flags` derives for an `mlx_ios_pipette` cell, so a spec-generated plan matches as-is |
| `runtime:mlx:mlx-swift=0.31.6` | mlx-swift, named |
| `runtime:mlx:mlx-swift-lm=f5f18ed9d` | the MLXLLM model/inference code |
| `runtime:mlx:swift-transformers=1.3.3` | the tokenizer |

All three packages affect output (swift-transformers changes tokenization, and so
the prompt encoding), which is why mlx-swift alone does not identify a build. An
author pinning an exact build lists all three in `requires`; short enough to read
and write, unlike a run-together composite. One flag per package deliberately, so
no flag ever runs two pins together: canonical form strips whitespace, which would
turn a composite into the unparseable `mlx-swift=0.31.6mlx-swift-lm=…`.

Flags must be lowercase and whitespace-free or the server rejects the whole
request, and must avoid the reserved `device_*`-derived namespaces the server
owns.

#### Planner worker (opt-in)

Settings → **Planner worker** turns the device into a pull client of the
management planner (same protocol as desktop `pipette worker`). Headless:
`settings set worker=on|off` only flips the preference; `settings run` enables
it, starts the claim loop on the **app-wide** `JobRunner`, and keeps the process
alive (launch-as-worker via `devicectl`). When enabled (and the app is in the
foreground):

1. `PATCH /clients/me` refreshes the device profile and capabilities, then holds
   until the `reindex_pending` gate lifts (at most ~5 minutes) and the client
   reads `approved`. The worker waits because it takes leases: a profile change
   voids its queue standing, so claiming before the gate lifts only burns
   retries. The report itself is not planner-specific. See
   [Capability reporting](#capability-reporting).
2. The app loops on `POST /plans/claim`.
3. The claim is the plan-types cell: `runtime_descriptor` selects the iOS
   `Runtime` variant (`llamacpp_ios_pipette` / `mlx_ios_pipette` /
   `apple_foundation`); `model_descriptor` is the `Model`; claim
   `runtime_flags` (typed plan-types iOS cells or legacy CLI/JSON) and
   `model_flags` become load knobs and eval `enable_thinking` via
   `PlanClaimConfig`. Missing local weights → retriable failure; desktop
   runtimes → non-retriable.
4. The cell runs through the normal `JobExecutor` path (no auto-submit);
   heartbeats run at half `time_window`.
5. Success/failure is submitted with the claim’s `job_id` and echoed
   `model_*` / `runtime_*` fields.

The loop stops in the background (same jetsam/timing reasons as local jobs)
and resumes when the app becomes active if the toggle is still on. An in-flight
cell is cancelled and a **retriable** failure is submitted when possible so the
lease is not left to time out silently. A claim `403` disables the toggle and
stops until an operator approves the client.

**v1 limits:** doomloop is not wired yet; eval samples come from the local
catalog only (no live `GET /benchmarks/{id}` on claim). Load knobs are typed in
plan-types iOS `RuntimeFlags` cells (ngl/ctx/`n_ubatch` for llama, `n_ubatch`
for MLX, empty for AFM) and applied from the claim by `PlanClaimConfig`.

#### Pre-auth key onboarding

`SetupView` has an optional **Pre-auth key** field. When filled, the token is
sent as `preauth_key` on `POST /clients/register` and the device comes up
already `approved`: no manual `clients approve` step on the management side. A
valid key may also seed the client's default tags/org. Leaving the field blank
registers exactly as before (the field is omitted from the request body, so the
keyless wire shape is unchanged).

The token is **transient**: it is passed to the register request and never
written to the Keychain, `registration.json`, or logs. Because the private key
is only promoted and the registration persisted *after* the server accepts, a
rejected key leaves no partial identity: re-entering a valid key just works.
The server rejects a malformed/unknown/expired/already-used key with `401`, and
(when it enforces keys) a missing key with `403`; both surface as a clear
onboarding error. Keys are minted server-side (see mgmt `docs/authentication.md`
§3.2).

#### Authentication ([Clerk](https://clerk.com))

Before any of the above, the app gates the entire UI behind a Clerk sign-in.
[Clerk](https://clerk.com) is a third-party authentication / user-management
service, integrated via the `clerk-ios` Swift package (`ClerkKit`). At launch
`PipetteApp` shows `ClerkAuthGateView` (or a configuration-error screen if the
Clerk settings are absent), so a user must authenticate before reaching the
benchmark UI.

Signing in **links the device registration to a user account**: the Clerk
identity (`clerkUserId`, `clerkSessionId`, `clerkPrimaryEmail`, `clerkLinkedAt`)
is captured in `SetupView` and stored on the registration record, and the
contact-email field is prefilled from the Clerk email. This is the same `clerk*`
metadata the Android client persists; iOS additionally enforces the gate (see
the Android [implementation ledger](../pipette-android/implementation.md) for
its status). Clerk is configured at build time. See
[build.md](build.md#clerk-configuration).

## 2. How the iOS app uses llama.cpp

The native Swift `LlamaCpp` ops depend on a narrow part of `llama.cpp`, called
directly through the `llama` module (`import llama`):

- the C API for model loading, tokenization, KV-cache prefill, and greedy
  sampling (`llama_model_load_from_file`, `llama_tokenize`, `llama_decode`,
  `llama_sampler_*`, `llama_memory_clear`, …)
- the Metal and Accelerate backends compiled into `libllama.a`

Unlike the desktop runners, there is no `llama-bench` or `llama-server` process.
every benchmark type is implemented directly against the C API in
`LlamaBenchmark`. The trade-off is that the iOS path reproduces behavior
the upstream binaries provide for free (the measurement loop, fixed-count
ignore-EOG decode for throughput, chat templating for eval). The benchmark
methodology mirrors the shared Rust kernel (`native/benchmarks.rs`) the Android
client still uses, so iOS and Android numbers stay comparable.
