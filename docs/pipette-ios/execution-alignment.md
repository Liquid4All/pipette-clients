# iOS execution alignment

The iOS run path is the pipeline `pipette-cli` runs: the same words, the same
ordered steps, the same shapes, for everything a phone can do. This document is
the contract for that alignment and the list of places where iOS differs on
purpose.

Related: [iOS usage](usage.md) (the headless verbs) ·
[iOS architecture](architecture.md) (file placement, components) ·
[model store design](model-store-design.md) · [storage quota](../storage-quota.md) ·
[eval checkpoint](../pipette-cli/eval-checkpoint.md)

**The rule: the CLI's behavior is the reference.** Where the two clients
disagree and the platform does not force the difference, iOS changes. Type
names come from the crate, not from this document: where a name here disagrees
with `crates/` or with `pipette-plan-types`, the crate is right and this file needs
a fix.

## Where the reference lives

`pipette-plan-types` lives in pipette-mgmt; every other crate is in
this repo.

| Type or function | Rust |
|---|---|
| `RunnableCell`, `ios_headless_args` | pipette-mgmt `crates/pipette-plan-types/src/plan/runnable_cell.rs` |
| `ClientRunSpec`, `is_compatible` | pipette-mgmt `crates/pipette-plan-types/src/cell/mod.rs` |
| `RunRequest`, `RunResponse`, `DeclaredBound<T>` | pipette-mgmt `crates/pipette-plan-types/src/run.rs` |
| `Model` | pipette-mgmt `crates/pipette-plan-types/src/model/mod.rs` |
| `Runtime` | pipette-mgmt `crates/pipette-plan-types/src/runtime/mod.rs` |
| `RuntimeFlags`, `ModelFlags`, `BenchmarkFlags` | pipette-mgmt `crates/pipette-plan-types/src/cell/runtime_flags.rs`, `cell/model_flags.rs`, `cell/benchmark_flags.rs` |
| `BenchmarkType`, `BenchmarkDefinition`, `SamplingTemperature` | pipette-mgmt `crates/pipette-plan-types/src/benchmark/mod.rs`, `benchmark/sampling.rs` |
| `BenchmarkSubmissionPayload`, `BenchmarkResultData`, `MemoryObservation` | pipette-mgmt `crates/pipette-plan-types/src/submission/mod.rs` |
| `DeviceInfo` | pipette-mgmt `crates/pipette-plan-types/src/device.rs` |
| `RunThermal`, `ThermalReading`, `ThermalTelemetry` | pipette-mgmt `crates/pipette-plan-types/src/thermal.rs` |
| `run_cell`, `prepare`, `dispatch_run` | `crates/pipette-cli/src/run.rs` |
| `parse_model_uri`, `model_to_uri` | `crates/pipette-cli/src/model_uri.rs` |
| `resolve_model_arg` | `crates/pipette-cli/src/artifact_ref.rs` |
| `UnrunnableClaimError` | `crates/pipette-cli/src/client/claim.rs` |
| `ensure_claim_benchmark_cached` | `crates/pipette-cli/src/commands/worker.rs` |
| `ClaimedJob` | `crates/pipette-mgmt-client/src/types.rs` |
| `ReadinessGate` | `crates/pipette-ops/src/readiness.rs` |
| `REPS` | `crates/pipette-ops/src/measurement/mod.rs` |
| `build_prompt_text` | `crates/pipette-ops/src/prompt_seed.rs` |
| `EvalCompletionsStore`, `run_request_digest` | `crates/pipette-ops/src/eval_completions.rs` |
| `MeasurementObserver` | `crates/pipette-ops/src/measurement/mod.rs` |
| `attempt` (`limit`, `LOAD_AND_WARM_UP`, `Overrun`; Swift: `Attempt`), `AttemptTimeoutError` | `crates/pipette-ops/src/attempt.rs` |
| `Clock`, `watched` (Swift: `AttemptClock`, `AttemptWatchdog`) | `crates/pipette-cli/src/attempt_watchdog.rs` |

## The cell

A cell is `(benchmark, model, runtime, flags)`, and it travels as a value from
the claim to the payload. It has four forms, and the transitions between them
are the pipeline:

| Form | Rust | What it adds |
|---|---|---|
| plan-side | `RunnableCell` | routing (`allowed_clients`) |
| wire | `ClientRunSpec` | nothing: the portable cell |
| engine input | `RunRequest` | resolved benchmark body, bound host paths |
| finished | `RunResponse` | metrics, and the flags as the run resolved them |

iOS has three producers of a `ClientRunSpec` (a claim, headless argv, a UI
selection) and one extra layer above `runCell`: a job holds N cells, because a
device-console launch costs seconds and a thirty-cell sweep cannot be thirty
processes. Everything from `prepare` down is the CLI's shape:

```
claim  ·  headless argv  ·  UI selection
        └──────────► ClientRunSpec
                          │
                     job: N cells per launch      iOS only
        for each cell:    ▼
                     RunCell.prepare ──────────► RunRequest (declared + bound, resolved body)
                     RunCell.dispatchRun ──────► RunResponse
                     recordResult              → results/<location>/<cellId>/
                     submit                      at once for a claim, drained otherwise
```

`declared` is what the plan asked for and what gets submitted (`Record.swift`
builds `model_descriptor` and `runtime_descriptor` from it); `bound` is the host
path the engine opens. The readiness policy resolves once per cell, in `runCell`,
and `dispatchRun` hands every engine the same opaque `ReadinessGate`, as the
crate does.

## Names that differ from the CLI

Where the concept is the same, iOS uses the crate's name. These are the
exceptions:

| Concept | CLI | iOS |
|---|---|---|
| Local set of cells | none (`pipette-ctl` holds it on the operator's machine) | `JobManifest`, `JobCell`, `JobId`, `jobs/<jobId>/`; "Jobs" in the UI |
| Server lease | `ClaimedJob.job_id` | `ClaimedJob.jobId` (the CLI's concept; beside `JobManifest`, `job` names two things on iOS) |
| Descriptor encoding | `model_descriptor` / `runtime_descriptor` from `Model` / `Runtime` | `SubmissionRef.model` / `.runtime`, over `Model.encode` / `Runtime.encode` |
| Submission acknowledgement | the result directory moves | the move, plus `CellSubmissionRecord` (`submission.json`), which also records the collector for collector-change resend |
| Headless runtime token | `Runtime::headless_token()` (the plan `type` tag) | `RuntimeType.shortAliases`: `runtime=` refuses a bare type tag because it names an engine without naming the build |
| One sensor snapshot | `ThermalReading` | none; iOS sends no `device_apple_*` series, and `DeviceReadings` is its only snapshot |
| Gate verdict | none (a `ReadinessGate` function) | `thermalOk` returns `Bool` |
| Per-kind measurement | `execute/<type>.rs` | one function per kind in `LlamaExecute` / `MlxExecute` / `OnnxGenaiExecute` / `AFMRuntime` |
| Repetition count | `REPS` | `BenchmarkMeasurement.reps` |
| Prompt construction | `build_prompt_text` | `PromptSeed.buildPromptText` |
| Catalog pull | the first phase of `pipette sync` | `pullRemoteBenchmarks` |
| Claim loop | `pipette worker` | `PlannerWorker`; headless `worker`, alias `settings run` |
| Terminal claim refusal | `UnrunnableClaimError` (4 cases) | `UnrunnableClaimError`: the crate's `noSpec`, `unparseable` and `benchmarkMismatch` by name, its `Invalid` split by cause, plus this build's own limits |
| Device-local refusal | none | `WorkerResolveError`, between a terminal claim refusal and a plain run failure |
| Kill evidence | none | `ActiveCellSentinel` |
| UI selection | none | `DiscoveredModel`, `RuntimeKind` (UI only) |

## Deliberate divergence

Listed so nobody "fixes" a deliberate difference. Most are forced by the
platform; a few are choices, and say why.

**No runtime store, and no runtime `bound` half.** Engines are compiled in. The
CLI's installers (docker, uv, an MLX virtualenv, an archive) have no phone
equivalent, so there is no `runtimes pull/remove/catalog/flavors`, no runtime
phase in the sweep, and capabilities come from build constants. The CLI's
`require_desktop_runtime` is inverted: `Runtime` cannot decode a desktop
runtime, so iOS refuses one at claim time.

**No workspace root and no `init`.** The data root is fixed and created at
launch.

**Both catalog halves.** `benchmarks/local/` is seeded by `StandardBenchmarks`
with the ladder and smoke ids the CLI's `benchmarks/standard.rs` generates,
limited to the kinds a phone can run, and `benchmarks/remote/` is the synced
catalog. A bare id means `remote`, the form plans and claims carry. A cell
records the half its benchmark came from, and `BenchmarkResultLocation(_:)`
puts a `local` result under `results/local/`, which `sync` never submits, as the
crate's `BenchmarkResultLocation::from(BenchmarkSource)` does.

**Submission defaults on for the sweep forms.** `benchmarks run` keeps the
crate's `sync=0` default. `bench` and the bare form take `submit=`, default on; the
reason is in [usage.md](usage.md#the-sweep-forms-bench-and-the-bare-invocation).

**Transfers outlive the process, and sweeps have more triggers.** A background
download survives the app, so staging is addressable and resumable, and the
sweep pins every model a running or paused job's manifest on disk names
(`SweepPins.activeJobs`) rather than one cell's pins passed down a call stack,
so resuming a paused job does not re-download. The running cell is pinned the
same way: its job's manifest is saved as `.running` before `JobExecutor.run`
starts and its `prepare` ensures the model. Nothing else pins that cell, so the
save must stay ahead of the ensure. The mechanism is in
[storage quota](../storage-quota.md) and [model store design](model-store-design.md).

**Kills are invisible to the process.** Jetsam reaches no error path, so
`ActiveCellSentinel` and crash-evidence reconciliation exist with no CLI
counterpart. Backgrounding cancels an in-flight cell and reports a retriable
failure.

**No subprocess.** `RunResponse.command` and `.executable` have no iOS
counterpart; desktop MLX leaves them unfilled too. The llama engine's load log
(`LlamaCpp.capturedLoadLog`) reaches the record and `extras.json` through
`RuntimeOutput`.

**An attempt past its limit cannot be stopped.** The crate's watchdog stops the
run's children, which an in-process engine does not have. `RunCell.dispatchRun`
makes the run's `AttemptClock` as the crate's `dispatch_run` makes its clock,
and the engines mark it through the same `MeasurementObserver` hooks;
`AttemptWatchdog` watches it from a thread of its own. Past the limit a claimed
launch (`claim=`) saves the cell as failed, clears its crash sentinel, logs the
failed cell and `BENCH_DONE 1 failed: …`, and ends with `HeadlessExit.abandon`,
which needs no main actor. Every other run, the resident app's included, keeps
the overrun on the clock and fails the cell with `AttemptTimeoutError` at the
next boundary: a readiness gate, a rep's start or end, a peak-memory attempt,
or the engine's return. So `onAttemptStart` throws, where the crate's
`on_attempt_start` cannot fail. The iOS `vl_spec_decode` cells take no limit,
so a corpus sample is never an attempt here.

**One in-process load, not a process per rep.** `llama-bench` runs a fresh
process per rep; this client loads once and measures `REPS` reps against that
load. So the readiness gate clears before the load as well as before each rep,
and a cold-Metal first-rep penalty is paid by the warm-up. An engine refuses a
kind it does not implement before the gate (for example, MLX refuses the vision and
speech kinds; each engine's support matrix lists its refusals), so a refusal is
never reported as "device not ready".

**A full KV cache is not out of memory.** `llama_decode` returning `1` means the
prompt plus decode outran the context, and llama.h's advice is a larger one. It
is reported as an engine failure, worded as `pipette-android`'s
`decode_error_message` words it, never as `RuntimeError.outOfMemory`, whose hint
offers a smaller context: the opposite of the fix.

**Different memory counters.** The crate samples a child's `phys_footprint`;
this client samples its own process footprint, since the work runs inside it.
That is the counter jetsam kills on.

**The end-to-end clock is the client's.** The crate times a request against a
server it started; this client times tokenize + prefill + decode in process.
Both put tokenization inside the window and both stop after sampling the last
token: `llama-server` releases the slot without decoding it, and
`LlamaCpp.generate` does the same under `.afterLastSample`, which the speech
cells pass. The vision cells pass `.afterLastDecode`: `vl_throughput`'s
`predicted_ms` covers every decode step, and `vl_max_memory` decodes every
token it asks for (one when a cell sets no `parameter_decode_tokens`, as the
text cell's `-pg <N>,1` does), one decode more than the crate's server request.

**The speech cells run the server's request in process.** `asr_throughput` and
`asr_max_memory` follow the vision cells, and differ from the crate's
`llama-server` run where a phone has no server or helper
([asr-throughput.md](../methodology/asr-throughput.md#ios),
[asr-max-memory.md](../methodology/asr-max-memory.md#ios)):

- *The clock window, without the HTTP.* The windows are the server's:
  `total_time_ms` runs from rendering the turns and reading the clip (WAV
  decode, resample, media hash) through the mel spectrogram, prefill and decode
  to the last sample; `asr_prompt_ms` is the encode and prefill up to the first
  sample; `asr_predicted_ms` is the `N - 1` steps after it, scaled to `N` as the
  crate scales the server's. Only the request's HTTP and JSON handling is
  missing from `total_time_ms`.
- *Calibration in process.* The audio token count that sizes the context is
  the audio chunks' tokens from `mtmd_tokenize` on the cell's own projector,
  before the context exists, where the crate runs `llama-mtmd-cli` and caches
  its count. Nothing is cached, and in `asr_max_memory` that tokenization is
  inside the bracket.
- *Confirmations are tokenizations.* The crate's text-only confirmation
  requests read the server's prompt count, which with prompt caching off is
  the token count of the render; the app tokenizes the same renders and sends
  nothing. `asr_max_memory` runs them after its one request, where the crate
  runs them on a warm-up server.
- *Jinja-only rendering.* The speech turns go straight to `common/chat.cpp`
  through `pipette_chat_apply_jinja`, with `enable_thinking` off and the
  template arguments `llama-server` passes, never through
  `llama_chat_apply_template` first (see Deferred for the other cells).
- *The clip goes through a temp file.* The WAV is written once per cell and
  read with `mtmd_helper_bitmap_init_from_file`, which decodes, downmixes and
  resamples as the server's `_from_buf` does, and compiles against every
  vendored llama.cpp variant, which `_from_buf` does not. No base64.

**Credential custody.** The CLI stores no Hugging Face token:
`inject_env_hf_token` folds `PIPETTE_HF_TOKEN` into the in-memory model on every
invocation, and `Model::without_auth_token` clears it wherever a model is written
down. That works because the environment lasts the whole process.

iOS has no such guarantee. A background transfer outlives the app: the OS
relaunches the process to finish it with no claim in memory and no environment.
So a claim's token goes to the Keychain when the claim is decoded, keyed by the
model's own `reference()` (`org/repo[@rev]:file` for a gguf,
`org/repo[@rev][:prefix]` for an mlx bundle). So two cells that pull different
files from one repo keep separate tokens. The fetch
resolves most-specific-first, in one function (`resolveHfToken`) that both
transports call: this run's token, then the one stored for this model, then the
user's own (iOS's analogue of `PIPETTE_HF_TOKEN`). Both transports log which
tier answered (`hf auth: … source=claim`) or that the fetch is anonymous, never
the token itself. Without that line an anonymous fetch of a public repo looks
authenticated until Hugging Face answers 429. Anonymous fetches count against a
per-IP limit, which all the phones on one network share. The `Model`-to-account mapping lives on `KeychainHelper`, not
on `Model`, because a Keychain account has no crate counterpart. `auth reset`
clears every stored model token.

**The auth token is stripped by the encoder, not by each caller.** A divergence
from the CLI, and not a forced one. The crate keeps `Model`'s serialization
faithful and makes every write site call `without_auth_token()`: one obligation
per site. iOS puts the obligation in one place instead, so `Model.encode` drops the
token unconditionally while decode still accepts one, since that is how a claim
delivers it. The price is that a `Model` does not round-trip through disk
unchanged, which is why the credential lives in the Keychain.

**A vault source's `custom_label` is outside equality.** The crate derives `Eq`
and `Hash` over it. Here a `Model` is compared as an identity in many places
(transfer rows, installed-model lookups, a job's failed-model map), so
`CustomLabel` compares equal to any other and hashes nothing, as `HfRepo` does
for its token. The encoder still writes it, so the submitted
`model_descriptor` reports the label the claim carried. Every identity taken
over the encoding clears it first with `Model.withoutCustomLabel`: the model
digest (`model://sha256=`, `models list`) and the eval resume digest, which is
the server's rule.

**Identity custody.** The signing key has two homes. The Keychain copy uses
`kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`, so iCloud Keychain does not
sync it and no restore moves it to other hardware; it does not survive a build
re-signed under a different identity, though the app container does. The
`identity/id_ed25519` file is the second home. `identity/` is excluded from
backup, so a restore onto new hardware carries neither half and setup mints a
fresh identity; a backup that carried them would put one `client_id` on two
devices. `signingIdentity()` needs both halves. `auth export` / `auth import`
move an identity between devices; the bundle is the CLI's, plus this client's
Hugging Face tokens. See [usage.md](usage.md#identity-auth) and
[the CLI's bundle](../pipette-cli/usage.md#moving-an-identity-to-another-machine).
Both verbs refuse a `pipette://` deep link: a URL any web page can open must
never print or replace a key.

**No `runtime_cpu_variant`.** A single static backend, so the field is always
absent.

## Gaps on this side

Not deliberate; these should close.

- **The catalog pull is unsigned.** `pullRemoteBenchmarks` reads the catalog
  without the device's identity, where the crate's `sync` signs whenever the
  workspace holds a key. A pulled store holds the public catalog only. The
  one-id fetch a run makes (`ensureRemoteBenchmarkCached`) is signed, so an
  organization's private entry is filled the first time a run names it.
- **Only headless `sync` refreshes scores.** `refreshScoredJob`
  (`Client/Sync.swift`) runs as `sync`'s third phase; the UI's own drains submit
  without refreshing, so `metrics.json` appears only after a headless `sync`.

## Upstream gaps

The phone could express these; the shared wire format cannot yet.

- **No readiness argument on the iOS transport.** `ios_headless_args` emits
  `runtime=`, `model=`, `benchmarks=`, `runtime-flags=` and `submit=1`, but no
  readiness flags, so a plan's `readiness` block reaches a phone through a claim
  and not through the headless transport. The app accepts
  `readiness-max-wait-secs=` and `readiness-skip-thermal=` on the sweep forms;
  the crate can emit them as `build_argv` does once the phones run a build that
  accepts them. An older build refuses the unknown key and fails the cell.

## Eval checkpoint identity

iOS ports the crate's checkpoint (`Ops/EvalCompletions.swift`, mirroring
`crates/pipette-ops/src/eval_completions.rs`); the file format, rotation and
resume rules are in [eval checkpoint](../pipette-cli/eval-checkpoint.md). The
digest is taken over the portable axes only: the declared runtime, the declared
model with its `custom_label` cleared, the resume-relevant runtime and benchmark
flags, and the benchmark body. Never the bound path: a re-fetched model gets a
new path routinely, and a path-sensitive digest would rotate the checkpoint on
every eviction.

The `model_flags` key is always written as `null`. The claim path refuses every
knob a `ModelFlags` variant can set (`enable_thinking`, `reasoning_effort`, as
`flagNotHonouredHere`), so no run reaching an engine carries one, and writing the
key keeps the crate's shape.

## Contract fixtures

No Rust links into the app, so the mirrored types are held together by shared
fixtures that both languages read. One hand-written JSON file per example,
`crates/pipette-contract/fixtures/<section>/<name>.json`, named for the variants
it holds, joined by dots (`model/gguf_text.huggingface.json`); a trailing
`.<note>` separates two examples of the same variants, so listing a section is
its coverage table.

- **Rust, at test time.** `crates/pipette-contract/tests/contract.rs` runs each
  file as its own case: an example must re-encode to itself and be named for the
  variants it holds. A variant added to any enum fails with the file to add; one
  removed fails its examples' decode. The test keeps `fixtures/variants.json`,
  and `CONTRACT_BLESS=1` rewrites it and the derived files (`params.json`,
  `sampling.json`, `defaults.json`, `runtime-build-flags.json`). `bounds.json` is
  written by hand.
- **Swift, at build time.** `ios/gen-contract-variants.py` renders
  `variants.json` as one exhaustive switch per mirrored enum
  (`ios/Pipette/PipetteTests/Generated/ContractVariants.swift`), skipping the
  variants `ios/contract-refusals.json` lists. A variant either side lacks is a
  compile error. `ci/checks/contract.py` fails when the committed switches are
  stale.
- **Swift, at test time.** `ContractTests` round-trips every example (one holding
  a refused variant must fail to decode) and checks the derived files.
  `ios/contract-differences.json` names the fields iOS reads differently on
  purpose.

## Deferred

Engine differences left for their own change, because each re-baselines rows
already in the warehouse or needs device verification.

**Jinja chat templates, for templates the legacy path knows.** `llama-server`
renders a chat template through `common/chat.cpp` and llama.cpp's Jinja engine.
This client calls `llama_chat_apply_template`, the legacy C path, first, and
renders through the server's code (`ios/native/pipette_chat.{h,cpp}`, a C veneer
over `common/chat.cpp`, linked as part of `llama-common` by `ios/build-llama.sh`)
only when that call returns -1. Gemma-4's template takes the fallback and
renders the server's prompt. A template the legacy path recognizes still renders
through its built-in approximation, so where that differs from the Jinja render,
eval scores and vision prefills differ from the CLI's. The speech cells render
through Jinja only. Moving every template to
Jinja changes the prompt of every cell that already renders, so it belongs with
a re-baselining decision. The `enable_thinking` and `reasoning_effort` model
flags stay refused (`Client/Claim.swift`): neither renderer takes them from the
cell.

**Prompt token content for the throughput cells.** Every engine feeds the same
token count, which is what the cells measure, but not the same tokens:
`llama-bench` feeds random ids, the crate's MLX client a prompt built from the
shared seed passage, iOS llama.cpp the passage tiled, and iOS MLX synthetic ids
(per engine: [prefill throughput](../methodology/prefill-throughput.md#token-control)).
The content is a second-order effect on the cache. The clients should agree on
one answer before any of them moves, because changing it re-baselines every
throughput row.

**The MLX prefill chunk.** `mlx_lm` prefills in 2048-token steps and the crate's
server ignores `runtime_flags`; this client defaults to 512
(`MlxRuntime.defaultPrefillChunk`) and reports what it used. Raising it raises
the activation peak, which is what gets a phone jetsam-killed, so it needs
device verification per model class.

**The MLX prefill shape outside the peak-memory cell.** `mlx_lm` fills the cache
with every prompt token but the last, evaluating only the cache and clearing
MLX's buffer cache after each step, then runs the last token alone through the
output projection. `max_memory_usage` prefills that way
(`MlxRuntime.prefillAsMlxLm`), in the cell's `n_ubatch` windows. On an iPhone 17
Pro this shape gives LFM2.5-350M a kernel peak of 741 MiB at 512 tokens and 904 MiB
at 4096, against 919 and 1426 MiB with per-window logits. The timing cells, eval and the probes still
evaluate each window's logits; moving them re-baselines the MLX prefill and
end-to-end rows, so it needs its own timing run.

**MLX eval `completion_tokens` on an EOS stop.** The crate reads the count off
`mlx_lm.server`'s usage trailer, which includes the EOS token, as
`llama-server`'s `predicted_n` does. `MlxRuntime.generate` stops before it
appends the EOS id, so an `eos` row here reads one lower than the CLI's for the
same answer. Counting the EOS id changes that column alone: `stop_reason`
already comes from the stop itself.
