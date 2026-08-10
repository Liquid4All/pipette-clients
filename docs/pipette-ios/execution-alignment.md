# iOS execution alignment: design

A design record for making the iOS run path the same pipeline `pipette-cli`
runs: same words, same ordered steps, same shapes, same command surface, for
everything a phone can do. Model *storage* is already aligned and is not the
subject here. This document says how execution **uses** the store, not how the
store works.

Related: [model store design](model-store-design.md) · [iOS architecture](architecture.md) ·
[CLI usage](../pipette-cli/usage.md) · [eval checkpoint](../pipette-cli/eval-checkpoint.md) ·
[storage quota](../storage-quota.md)

**How to read the status claims.** This is a design record, not a changelog: unless a
paragraph says otherwise, it describes the target, and the present tense describes what
the code does *today*. Where a piece has landed it is marked inline. Every Rust type named
here was checked against `main` when this document landed; the iOS-side "target" names are
by definition not there yet. If a claim and the code disagree, the code wins and this file
is the bug.

**Confidence.** §1 (naming), §3 (data structures), §5 (the cell) and §6 (the headless
surface) were derived by reading both sides type-by-type and are the load-bearing parts.
§2 (approach and steps) is the least verified: it paraphrases the CLI's `prepare` doc
comment and step order rather than tracing every call, so treat its sequencing as intent
to check against `crates/pipette-cli/src/run.rs` when it is implemented, not as an audited
transcript.

**Where the reference lives.** Line numbers are as of this document landing and will
drift; the paths and names will not.

| Type or function | Rust |
|---|---|
| `RunnableCell`, `ClientRunSpec` | `pipette-plan-types/src/plan.rs:1016`, `:988` |
| `RunRequest`, `RunResponse`, `DeclaredBound<T>` | `pipette-plan-types/src/run.rs:70`, `:135`, `:39` |
| `Model`, `ModelFlags` | `pipette-plan-types/src/model.rs:32`, `:72` |
| `Runtime` | `pipette-plan-types/src/runtime.rs:22` |
| `RuntimeFlags` | `pipette-plan-types/src/runtime_flags.rs:207` |
| `BenchmarkFlags` | `pipette-plan-types/src/benchmark_flags.rs:87` |
| `BenchmarkDefinition`, `Temperature` | `pipette-plan-types/src/benchmark/mod.rs:132`, `:44` |
| `BenchmarkResultData` | `pipette-plan-types/src/result.rs:121` |
| `RunThermal`, `ThermalReading` | `pipette-plan-types/src/thermal.rs:171`, `:149` |
| `run_cell`, `prepare`, `dispatch_run` | `pipette-cli/src/run.rs:45`, `:80`, `:267` |
| `UnrunnableClaim` | `pipette-cli/src/client/claim.rs:45` |
| `ClaimedJob` | `pipette-mgmt-client/src/types.rs:294` |
| `ReadinessGate` | `pipette-ops/src/readiness.rs:18` |
| `REPS`, `build_prompt_text` | `pipette-ops/src/measurement.rs:20`, `prompt_seed.rs:29` |

## Why

The two clients answer the same server, submit into the same warehouse table,
and are compared against each other. Today they diverge in ways that are not
stylistic. Each of these is currently true on device:

**An `eval` or `vl_throughput` claim can never run, and iOS tells the server so
permanently.** Cell construction goes through `BenchmarkDefinition(parsingId:)`
(in `PlannerWorker.makeManifest` on the claim path and in `JobCell.pending` on
the headless path), and that parser returns `nil` for exactly those two kinds.
The claim then produces zero cells, is reported as `unknownBenchmark`, and is
classified **non-retriable**, which retires the job for the whole fleet. The
justification in the code ("the catalog is compiled in") is false:
`BenchmarkCatalog.all` reads the server-synced index and has no bundled
fallback. Meanwhile every engine implements eval, and `JobExecutor` already
resolves definitions through the catalog-aware path that would have found the
body with its samples. The weakest of the two resolvers gates the stronger one.

**A plan-dispatched headless GGUF cell cannot select its own model.**
`RunnableCell::ios_headless_args` emits `model=<Display>`. The receiving matcher
tries three terms; the empty match, the artifact leaf, and `repoIdentifier`
(`org/repo`). An MLX model with no `prefix` renders as exactly `org/repo`, so
those cells match, by accident of which arm `ModelSource::reference()` takes.
`GgufText` renders `{repo}:{path}` and `GgufVision` `{a}+{b}`; neither is a
substring of either arm, so those cells fail with "no model found". A bare
`repo/name` is also ambiguous by construction (the matcher refuses unless
exactly one candidate survives), so a multi-quant repo cannot be addressed this
way even when the string does match. The one input that *is* the CLI's
`--model` grammar, `bench spec=<json>`, is the one the plan runner never emits.

**Two authored flags are accepted and then ignored.** `enable_thinking` is
decoded, resolved, stamped onto the cell's `Model`, persisted, and then
discarded by `ensure`, which returns the on-disk manifest's spec, keyed without
flags. No engine reads it, and the wire `model_flags` is hardcoded `nil`. A
thinking and a non-thinking eval row are indistinguishable in the warehouse.
`ctx_size` is decoded, validated, persisted onto the manifest, and then
overridden unconditionally by `BenchmarkContextSize.perCell`; the wire
`runtime_flags` string is then synthesized from the config and reports the
context size that did **not** run.

**A submitted result describes the binary that ran, not the cell the server
dispatched.** `SubmissionRef.RuntimeRef` regenerates the runtime descriptor from
local build constants; the claim's `repository_version` pin is compared and
logged only. The CLI echoes `runtime.declared` verbatim, which is what makes the
descriptor a join key.

**A jetsam kill mid-eval loses every completion.** The CLI checkpoints eval
samples with a flush per sample on machines that mostly do not die. iOS has no
checkpoint at all, on the platform where `ActiveCellSentinel` exists precisely
because kills are routine. An IFStruct cell is 2000 samples.

**A headless-only device cannot re-pull the catalog, drain every pending
result, or refresh scores.** All three mechanisms exist; the catalog sync is
reachable only from four UI or registration call sites, the drain only per-job,
and score refresh does not exist: `metrics.json` is read and deleted by the
app and written by nobody. There is also no way to list benchmarks, list
results, or look at disk from the console, so operators guess ids and fly blind
on storage.

None of this is a phone limitation. It is one structural difference with a long
tail: **the CLI carries a cell as a value and iOS destructures it**. `PlannerWorker`
takes a typed `ClientRunSpec` and flattens it into a `JobManifest` plus one
`JobCell`, hoisting the load flags to job scope and dropping the rest;
`JobExecutor` then re-derives the benchmark body, the runtime, and the context
size from whatever survived. Every finding above is downstream of that.

## The model iOS is adopting

The CLI's vocabulary, fixed in [`docs/architecture.md`](../architecture.md):
a **benchmark** is the catalog definition of what is measured; a **run** is one
cell's lifecycle; prepare, engine, outcome, record, submit; **`execute/`** is
the engine module of per-kind implementations. "Run this benchmark", never
"execute a benchmark".

A **cell** is `(benchmark, model, runtime, flags)`. It exists in four forms, and
the transitions between them *are* the pipeline:

| Form | Rust | What it adds |
|---|---|---|
| plan-side | `RunnableCell` | routing (`allowed_clients`) |
| wire | `ClientRunSpec` | nothing: the portable cell |
| engine input | `RunRequest` | resolved benchmark body, bound host paths |
| finished | `RunResponse` | metrics, streams, and the flags as the run resolved them |

Two entry paths (argv and a claim) converge on one type (`ClientRunSpec`), and
then on one function (`run_cell`). They diverge again only after the outcome:
the local path prints, records and optionally submits; the worker path submits
with the claim's job id attached and keeps nothing.

iOS expresses the same spine, with one extra layer above it and none below:

```
claim  ·  headless argv  ·  UI selection
        └──────────► ClientRunSpec            one type, three producers
                          │
                     batch expansion          iOS-only: N cells per launch
                          │
        for each cell:    ▼
                     prepare(spec) ─────────► RunRequest
                          │                   declared + bound, resolved body
                     dispatch(request) ─────► RunResponse
                          │                   result data, thermal, resolved flags, log
                     record(request, response) → payload.json (+ extras.json)
                          │
                     submit                   immediate for a claim, drained otherwise
```

The batch is the iOS analogue of `pipette-plan` running on an operator's
machine: a set of cells with per-cell state, held on the device because there is
no operator machine driving the phone. It sits **above** `runCell` and does not
reach inside it. Everything from `prepare` down is the CLI's, verbatim in shape.

What iOS is *not* adopting: a runtime store, a local benchmark catalog half, a
subprocess, or a work directory. Those are dealt with in
[platform-forced divergence](#platform-forced-divergence).

## Dimension by dimension

### 1. Naming

**Status: landed, in full.** The naming pass applied the types, methods, files, and
`job` → `Batch`; stage 5 then landed the four entries it had deferred, because they
change what a persisted cell remembers: `PlanModel` → `Model`, `PlanHfRepo` →
`HFRepo`, `ModelFlags` closed and eval-only, and the `RuntimeFlags` chain collapsed
onto the cell. The "iOS target" column below is therefore the tree as it stands and
"iOS today" is the tree as it was; the dictionary is kept in that shape because the
*reason* for each rename is what a reader needs, not a second copy of the current
names.

The rule: **where the concept is the same, the name is the same.** Where iOS has
a concept the CLI does not, it gets a new word rather than an overloaded one.

Four collisions are load-bearing and are fixed first, because they make correct
reasoning impossible:

- **`Runtime`.** The CLI's `Runtime` is *identity only*: load settings ship
  separately as `RuntimeFlags`, and the submission format says so. The iOS
  `Runtime` fuses identity with `LlamaCppConfig{nGpuLayers, contextSize,
  nUbatch, mmprojPath}` / `MLXConfig{prefillChunk}`. The actual mirror of the
  CLI's type is the private nested `SubmissionRef.RuntimeRef`, which exists only
  to serialize a descriptor. Split: `Runtime` becomes identity (promoted from
  `RuntimeRef`), `RuntimeFlags` carries the load settings.
- **`job`.** Three referents in one file: the local batch (`JobManifest`,
  `JobId`, `jobs/<jobId>/`), the server's lease (`ClaimedJob.jobId`), and a
  synthetic hybrid (`JobId("plan-\(job.jobId)")`). See
  [decisions](#decisions) for the rename and its cost.
- **`ModelFlags`.** *Resolved.* The CLI's is a closed enum keyed on
  `(benchmark, model)`, every variant `Eval…`, so a non-eval cell carrying
  `enable_thinking` is unrepresentable. iOS had an open struct embedded *inside*
  `Model` and serialized into the model JSON. It is now the same closed enum,
  produced by `ModelFlagRef.resolve()` as the crate's `TryFrom<ModelFlagRef>`
  does, and carried by the cell: `Model` emits no `model_flags` at all.
- **`ThermalReading`.** *Resolved.* A sensor snapshot in Rust, a gate verdict in Swift.
  The Swift verdict was a two-case enum, so it is now a `Bool` (`isThermallyReady`).
  the crate has no verdict type to mirror, only a `ReadinessGate` function. The name is
  free for the sensor snapshot, and `RunThermal` can be mirrored as it stands.

The dictionary. `—` means the concept has no iOS name today.

#### Cell and artifacts

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Work payload served to a client | `ClientRunSpec` | `ClientRunSpec` (in `PlanRunSpec.swift`) | same, file renamed `ClientRunSpec.swift` |
| Model coordinate | `Model` | one `Model`: done | — |
| Model kind discriminant | `ModelType` | `ModelType`: done | — |
| Where a model's bytes live | `GgufTextSource` / `GgufVisionSource` / `ModelSource` | same three enums, `huggingFace` case only: done | — |
| A gguf file's repo-relative path | `RepoSubpath` | `RepoSubpath`: done | — |
| Runtime coordinate | `Runtime` | `Runtime` (identity + config), `PlanRuntime`, `SubmissionRef.RuntimeRef` | one `Runtime` (identity) |
| Runtime kind discriminant | `RuntimeType` | `PlanRuntimeType` | `RuntimeType` |
| Headless runtime token | `Runtime::headless_token()` | `BenchRuntime.tag` | `headlessToken`, with a tag-parity test |
| Model/runtime pairing rule | `is_compatible` | one check, `PlanClaimConfig`'s `accepts`, at claim time as upstream calls it | lift it to `PlanTypes` as a free `isCompatible(_:_:)`, where plan-types keeps it |
| Plan coordinate vs launch form | `DeclaredBound<T>` | `DeclaredBound<Runtime>` + `ResolvedModel`: done | — |
| UI selection projection | — | `DiscoveredModel`, `RuntimeKind` | keep, marked UI-only |

#### Flags

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Per-cell load settings | `RuntimeFlags` | `IosRuntimeFlags` → `PlanClaimConfig.{nGpuLayers,contextSize,prefillBatch}` → `JobManifest` fields → `Runtime.LlamaCppConfig` | `RuntimeFlags`, one value, carried on the cell |
| Flat wire form of the above | `RuntimeFlagRef` | `RuntimeFlagRef` | unchanged: best-aligned type in the app |
| Per-cell generation settings | `ModelFlags` (closed, eval-only) | `ModelFlags`, closed and eval-only, on the cell: done | — |
| Flat wire form | `ModelFlagRef` | `ModelFlagRef` | unchanged |
| Run-driving settings | `BenchmarkFlags` | decoded as opaque JSON and refused | refuse until plan-types gains an iOS variant |
| Readiness deadline override | `ReadinessOverrides` | constants in `JobRunner` | `ReadinessOverrides` once expressible |
| Repetition-abort settings | `DoomloopOverrides` | — | deferred; no on-device detector |
| Flag rejection reasons | `RuntimeFlagError`, `ModelFlagError` | `RuntimeFlagResolveError` for both | keep merged; document |

#### Benchmark

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Catalog definition | `BenchmarkDefinition` | `BenchmarkDefinition` | unchanged |
| Kind | `BenchmarkType` | `BenchmarkType` | unchanged |
| Id → kind heuristic | `BenchmarkType::from_id` (marked temporary) | `BenchmarkDefinition(parsingId:)` **and** `PlanClaimConfig.benchmarkType(ofId:)` | one `BenchmarkDefinition(fromStructuredId:)`, fallback only |
| Catalog handle | `BenchmarkStore` | ~~`BenchmarkStore` protocol / `FileBenchmarkStore`~~ | **concrete typed `BenchmarkStore`**: done |
| Catalog origin | `BenchmarkSource` | ~~: ~~ | **`BenchmarkSource`**: done |
| Location-qualified id | `SourcedBenchmarkId` | ~~: bare ids~~ | **`SourcedBenchmarkId`**, bare id = remote: done |
| Loose upstream row | `RemoteBenchmark` | `BenchmarkItem` (typed + `rawJson` bag) | `BenchmarkItem` loses `rawJson` |
| Catalog pull | `benchmark_definition_from_remote` | `BenchmarkSync.keepParseable` | keep |
| Eval identifier | `EvalId` / `KnownEvalId` | `EvalId` / `KnownEvalId` | unchanged, but wire up `samplingTemperature` |
| Sampling temperature | `Temperature` (validated) | — | `Temperature` |

#### Engine contract

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Prepared engine input | `RunRequest` | — six loose parameters plus two ambient globals | `RunRequest` |
| Readiness wait, injected | `ReadinessGate` | `ReadinessCallback` + `BenchmarkReadiness` | `ReadinessGate` |
| What one engine call returns | `RunResponse` | `BenchmarkResult` only; thermal via a task-local | `RunResponse` |
| Measurement payload | `BenchmarkResultData` | `BenchmarkResult` | `BenchmarkResultData` |
| Resolve + ensure step | `prepare` | inline in `JobExecutor` | `prepare` |
| Pick the engine | `dispatch_run` | `RunCell.dispatch` → `Engine.run`: done | — |
| One cell's lifecycle | `run_cell` | inline `for` body | `runCell` |
| Per-kind measurement module | `execute/<type>.rs` | `LlamaBenchmark`, `MLXBenchmark` | `execute` naming inside the engines |
| Repetition count | `REPS = 5` | `measurementRuns`, declared three times | one constant |
| Prompt construction | `build_prompt_text` | `PromptSeed.buildPromptText` | unchanged |
| One sensor snapshot | `ThermalReading` | — (the name is free now) | `ThermalReading` |
| Gate verdict | — (a `ReadinessGate` function, no shape) | `isThermallyReady` returns `Bool`: done | — |
| Per-run thermal series | `RunThermal` | `ThermalSeries`, fed by a `RepObserver`: done | — |

#### Record and submit

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Wire body | `BenchmarkSubmissionPayload` | `BenchmarkSubmissionPayload` | unchanged shape, corrected content |
| Lossless artifact identity | `model_descriptor` / `runtime_descriptor` | `SubmissionRef.model/.runtime` | `Descriptor.model/.runtime` |
| Non-wire sidecar | `BenchmarkResultExtras` / `extras.json` | — | `extras.json` with the engine log |
| Record step | `record_and_maybe_submit_run` | `PayloadBuilder.writeLocal` | `recordResult` |
| Where a result lives | `BenchmarkResultLocation` | implicit | derived `BenchmarkResultLocation` |
| Result lifecycle | `BenchmarkResultState` (`local`/`submitted`/`scored`) | `BenchmarkResultState`: done | keep the attempt record (`CellSubmissionStatus`) |
| Submission acknowledgement | `job_id` marker + directory move | `CellSubmissionRecord` | keep the iOS shape |
| Scores pulled back | `BenchmarkScoredResult`, `BenchmarkJobMetric` | — (`metrics.json` read, never written) | add both, or delete the reader |
| Completion-id pre-flight | `dedupe_completion_ids` | — | add, after checkpointing lands |
| Catalog pull + drain + refresh | `pipette sync` | `BenchmarkSync` (pull only) | `BenchmarkCatalogSync` for the pull; `sync` for all three |

#### Claim and worker

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| Leased job envelope | `ClaimedJob` | `ClaimedJob` | unchanged |
| Opaque spec | `serde_json::Value` | `RawJSONValue` | unchanged |
| Redacted logging | `redacted_spec` | `redactedDescription` | unchanged |
| Claim → spec | `run_spec_from_claim` | `PlanClaimConfig.runSpec(from:)` | `ClientRunSpec.from(claim:)` |
| Terminal claim rejection | `UnrunnableClaim` (4) | `PlanClaimConfig.ParseError` (12) | `UnrunnableClaim`; keep all 12 |
| Run-error disposition | `classify_run_error` | `retriable(_:)`; type-based | keep; the CLI should follow |
| Claim attempt | `ClaimPoll` | inline branches | `ClaimPoll` |
| Heartbeat tick | `LeaseKeepalive` | inline branches | `LeaseKeepalive` |
| Submit attempt | `SubmitDisposition` (4) | `SubmitOutcome` (2) | `SubmitDisposition` (4) |
| Failure body | `FailureSubmission` | `FailureSubmission` | unchanged |
| Claim echo | `attach_claim_to_success_payload` | `attachClaimEcho` | unchanged, but on the typed payload |
| Device-local refusal | — | `WorkerResolveError` | keep. It has no CLI counterpart and should |
| Idle / reindex budgets | named constants | inline literals | named constants |
| The claim loop | `pipette worker` | `PlannerWorker`, verb `settings run` | verb `worker` |

#### Workspace and storage

| Concept | CLI | iOS today | iOS target |
|---|---|---|---|
| The root handle | `PipetteWorkspace` | `Storage` / `FileStorage` | `PipetteWorkspace` / `FilePipetteWorkspace` |
| Model store | `ModelArtifactStore` | `ModelArtifactStore` | unchanged (file renamed) |
| Runtime store | `RuntimeArtifactStore` | — | none, permanently |
| Find-or-fetch | `ensure_model` | `ModelProvisioner.ensure` | unchanged |
| Never-evict set | `SweepPins` | `SweepPins` | unchanged |
| Survey / plan / apply | `survey`, `plan`, `apply_sweep` | `survey`, `SweepPlan`, `applySweep` | unchanged |
| The cap | quota | "limit" in the surface, "quota" in the mechanism | quota, everywhere |
| Eval resume state | `EvalCompletionsStore`, `EvalCompletionSession`, `EvalRunDigest` | — | all three |

### 2. Approach and steps

The target order for one cell, with what each step decides and what it defers.
This is the CLI's `prepare` doc comment, adapted where the platform differs.

| # | Step | Decides | Defers to | iOS today |
|---|---|---|---|---|
| 0 | Parse to `ClientRunSpec` | cell identity, flag legality against a *type* | the body, artifacts, the device | claim path aligned; headless path does not produce a spec at all |
| 1 | Platform admissibility | reject a desktop-only runtime | — | present in the claim path only |
| 2 | Resolve the benchmark body | the concrete `BenchmarkDefinition` and its source | result submittability | **two resolvers, the weaker one gating** |
| 3 | Validate flags against the resolved body | legality of every authored flag | rendering to engine settings | validated against a guessed type only |
| 4 | Device preflight | is this device usable for this cell | — | memory gate, but after the fetch |
| 5 | Ensure the model, pinned | the bound path, disk eviction | launch | present, but called twice on the claim path |
| 6 | Assemble `RunRequest` | everything the engine needs | measurement policy | four separate carriers |
| 7 | Dispatch | which engine, and the readiness deadline, once | when to wait | per-cell gate constructed inline |
| 8 | Measure | rep count, aggregation, thermal series | recording | aligned: 5 reps, gate per rep |
| 9 | Record | payload/extras split, location, result id | the server's job id | payload only; identity partly regenerated |
| 10 | Submit | the job id | metrics | aligned in shape |
| 11 | Refresh scores | `metrics.json` | — | absent |

Two things the CLI validates **twice**, deliberately, and iOS should too: flag
legality (against the peeked type at parse, against the resolved body in
`prepare`) and runtime admissibility (at parse and again at dispatch). Two
things stay single: the benchmark body resolves once, and the readiness deadline
resolves once for every engine.

The delta, in one line each:

- Step 2 must have **one resolver**: the synced catalog first, the structured-id
  parse as a fallback, and (matching `ensure_claim_benchmark_cached`) a
  `GET /benchmarks/{id}` when the catalog misses. Only then is
  `unknownBenchmark` terminal.
- Step 3 must run against the resolved body, which deletes the last prefix
  heuristic from the claim path.
- Step 4's memory gate reads the model file, so it cannot precede the fetch; the
  *disk* preflight already precedes it and stays there. Say so, so a reader who
  knows the CLI does not look for one gate and find two in different places.
- Step 5 happens once. Either `resolveLocalRun` becomes a pure `installed()`
  check that hands pins forward, or it stops resolving and lets the executor own
  ensure; the latter matches the CLI, where the worker resolves nothing.
- Steps 6 and 9 are where the cell must survive as a value; see below. Step 6 becomes
  `prepare` → `RunRequest`, and step 5 resolves with it: `prepare` is the one `ensure`,
  which turns the claim path's second one into a deliberate pre-flight that keeps a
  provisioning failure a claim failure with the lease still held.

### 3. Data structures

Only what a phone runs. The wire types are already mirrored well;
`ClientRunSpec`, `ClaimedJob`, `RuntimeFlagRef`, `ModelFlagRef`,
`BenchmarkDefinition`, `BenchmarkResultData`, `FailureSubmission` are
field-for-field, and in three places iOS is *stricter* than Rust and should stay
that way: `EvalSample` is typed where Rust holds raw JSON,
`BenchmarkEvalCompletion` is a sum type that makes an invalid completion
unrepresentable, and `ParseError` types twelve refusals the CLI still expresses
as strings.

What is missing is the execution half.

**Add `RunRequest`** (`crates/pipette-plan-types/src/run.rs:70`). The things an engine
needs travel today as six parameters on the engine entry point plus the cell's model spec
and `mmprojPath`, the *manifest's* load settings, a task-local telemetry log, and a
process-global catalog. One value:

```swift
nonisolated struct RunRequest: Sendable {
    let runtime: DeclaredBound<Runtime>     // as shipped; the plan may name another build
    let model: ResolvedModel                // already declared-plus-located
    let runtimeFlags: RuntimeFlags?         // authored, plan-shaped; nil = engine default
    let modelFlags: ModelFlags?             // authored, eval-only
    let benchmarkFlags: BenchmarkFlags?     // authored; no iOS variant yet, see upstream gaps
    let benchmark: BenchmarkDefinition      // the resolved body, never an id
}
```

One departure from the crate's shape, and it landed differently than drafted here.
`DeclaredBound<T>` stayed **single**-parameter and wraps the runtime alone; the model is a
bare `ResolvedModel`, which already *is* the declared-plus-located pairing. That is not a
gap to close: plan-types' own module doc blesses it; "the Swift app mirrors this crate,
and independently arrived at the same declared-plus-located pairing in its `ResolvedModel`"
(`pipette-plan-types/src/run.rs:10`). Wrapping it again would be a second implementation of
one idea. And `benchmarkSource` is absent:
labelling a result's catalog origin gates submission of a locally-derived benchmark,
which is a behaviour change of its own, so the field arrives with it.

**No second, resolved settings type.** An earlier draft added a `RuntimeLoadSettings`
alongside the authored flags, on the reasoning that plan-types resolves each
`Option<u32>` by *not* emitting an argv flag and an in-process engine cannot decline to
pass a parameter. The reference answers this: the crate's engines default each absent
setting **at its point of use** (`crates/pipette-llamacpp/src/bench.rs`) rather than
materializing a resolved struct, and report what they derived by round-tripping
`RunRequest::runtime_flags` into `RunResponse::runtime_flags`. iOS does the same, so
`RuntimeFlags?` is the only flags type on the request and the diff between request and
response is what the run decided.

`declared` is what the plan asked for and what gets submitted; `bound` is the
host path the engine opens. Conflating them is why a submitted iOS result
describes the local build rather than the dispatched cell, and why
`enable_thinking` evaporates at `ensure`. The store returns the on-disk
manifest's spec, which is correct for the bound half and wrong for the declared
one.

**Add `RunResponse`** (`crates/pipette-plan-types/src/run.rs:135`). The engine entry point
returns bare result data, so thermal telemetry travels out of band and the engine log is
discarded. The crate's shape, minus the fields nothing here fills; `command`/`executable`
(no shelled-out invocation) and `stdout`/`stderr` (the engines discard the log). An earlier
draft of this sketch carried the two stream fields as permanently `""`; they were removed,
because a mirrored field with no producer is what `RunRequest.modelFlags` was deleted for.
They return when the engines hand the log back:

```swift
nonisolated struct RunResponse: Sendable {
    let resultData: BenchmarkResultData
    var thermal: RunThermal              // the four wire arrays, grouped
    var benchmarkFlags: BenchmarkFlags?  // the readiness policy that actually applied
    var runtimeFlags: RuntimeFlags?      // the request's flags as the run resolved them
}
```

`thermal` and `benchmarkFlags` are `var` and filled by the **caller**, not the engine.
the crate is explicit that the probe, the series and the gating policy belong to the
caller, and an engine is handed an opaque `ReadinessGate` so it never learns what the
policy was. On iOS that caller is `RunCell.dispatch`, which owns the `ThermalSeries` and hands the
engines a `RepObserver` closing over it, so `RunThermal` gets built there; that is what
takes the series off the side channel. The observer replaced an ambient task-local, so a
measurement path that never reports is a missing argument rather than an empty series.

`stdout` is `""` for now: the engines classify their own log internally (llama reads it to
detect OOM) but do not hand it back, and threading it out is the engine signature change
that lands with eval checkpointing. It is **not** platform-forced away: the llama engine
already captures and classifies a log, and the eval path needs somewhere to land a
failed-sample summary.

`command` and `executable` are dropped rather than kept empty: nothing shells out. The
crate types `executable` as `Option<String>` precisely so absence is explicit for a
runtime with no binary, and desktop MLX already leaves both unfilled.

**Add the eval checkpoint.** `EvalCompletionsStore` / `EvalCompletionSession` /
`EvalRunDigest`, digest over the *portable* axes only (declared model, runtime,
the three flag groups, the benchmark body) never the bound path. That exclusion
matters more on iOS than on the desktop: a re-fetched model gets a new path
routinely, and a digest over the path would rotate the checkpoint on every
eviction. `evalSamples` is the single seam both engines share, so the skip-set
check and the per-sample append belong there and the engines cannot drift.

**Add the result-state ladder.** *Done.* `BenchmarkResultLocation` and a concrete
`ResultsStore` over `results/{local,remote/pending,remote/synced}/<cellId>/`, so a
result's directory is its status; the crate's model, including `move_result_dir` on
accept. `BenchmarkResultState` stays three rungs and maps from the location exactly as
`store.rs:168` does: `local` and `remote/pending` are both `recorded`, because "generated
here" and "waiting its turn" are the same distance travelled.

Two consequences the move forced. Results no longer sit under `jobs/<jobId>/`, so
`deleteJob` deletes them explicitly or they orphan: invisible to every listing and still
counted against the quota. And `submission.json` stays, carrying `serverJobId`,
`submittedAt`, `errors` and `collector`; the crate has no such file (its location is the
whole record) but `collector` backs collector-change resend, which the CLI does not have.
The location is the authority on status, the record holds the submission's details.

**Fix the payload.** ~~Drop the five fields the CLI retired
(`model_name`, `model_quant`, `mmproj_quant`, `runtime_name`, `runtime_version`:
superseded by the descriptors)~~ *(done)*, populate `model_flags` from the cell through
an eval-only canonical string, ~~type `device_form_factor` as an enum~~ *(done)*, and stamp
the claim echo on the typed value rather than round-tripping the payload through
an untyped dictionary to add one key.

**Restore the device layer.** *Done.* `device_form_factor` and `device_power_state` were
token strings built at each site, and the twelve device fields were twelve arguments to
the payload initializer. The crate has `DeviceInfo` and `PowerState` as values and
`#[serde(flatten)]`s the first into the submission. `PlanTypes/Device.swift` now holds
`DeviceInfo` + `DeviceFormFactor` and `PlanTypes/Thermal.swift` gained `PowerState` +
`DevicePowerState`, so the payload initializer takes two values instead of fifteen
arguments and `ProfileReporter` maps field-by-field from a snapshot, as
`build_profile_update` does. `device_os_security_patch` is carried too: iOS sources none,
but the crate's flattened struct has it and Android fills it, so it elides rather than
being absent from the type.

The prober kept the name the *payload* has upstream. `pipette_device::probe` reads the
host and `pipette_plan_types::device::DeviceInfo` is what it returns; iOS had one enum
called `DeviceInfo` doing the reading, so the crate's type had nowhere to land. Renamed
to `DeviceProbe` under `Device/`, with the crate's `detect*` verbs: the same move
`ThermalReading` → `ReadinessVerdict` made for the same reason.

**Gather the identity layer.** *Done.* The CLI mints an `IdentityStore` from the
workspace and every command opens with it; iOS had the registration on the `Storage`
protocol as `save`/`load`/`delete`, the signing key on `KeychainHelper`, and the client
settings under `Contracts/` as `DeviceSettings`. `Identity/IdentityStore.swift` now owns
all three behind the crate's verbs (`getRegistration`/`putRegistration`/
`deleteRegistration`/`getSettings`/`putSettings`/`clearRegistrationMaterial`/
`signingIdentity`), derived off `Storage` exactly as `modelStore` is. It composes two
backings where the crate's composes one, because the private key is a Keychain item here
rather than a `0600` file. The root is `identity/`, the crate's name. It was `metadata/`,
shared with the benchmark catalog, until the catalog moved out and a one-shot
`migrateIdentityDirectory()` carried the two files across.

`signingIdentity()` is the part that changes behaviour, not just names. Six call sites
loaded the registration and the key separately and could reach the network having checked
one half; they now take the composed `AuthIdentity` the crate passes to `signed_headers`,
and `ManagementClient` takes that value instead of a `clientId` + `privateKeyHex` pair
threaded through `BatchSubmitter` / `SingleSubmitter` / `CredentialsLoader`.

`registration.json` moved to `snake_case`, which is what every identity file the CLI
writes uses and what `settings.json` here already used. `IdentityRegistration.init(from:)`
falls back to the old camelCase spelling per key, so an installed device keeps its
registration; re-registering would mint a new keypair and orphan every result already
submitted under the old one. The record still carries the registration inputs the CLI
fetches from `auth me` (plus the Clerk fields, which have no CLI counterpart), because the
app renders them without a round trip.

**Collapse the duplicates.** *Done for `Model`, on every path.* One `Model`. The bridge
was lossy: it dropped the auth token, so a gated-repo claim could not be fetched, and
returned `nil` for a nested weight path the Rust `RepoSubpath` permits. `HFRepo` now
carries `auth_token`, kept outside identity (excluded
from `==`, `hash` and both `description` forms) so a token-bearing coordinate is the same
storage entry and the same log line as a bare one.

**This section twice claimed the `Model` half was finished before it was.** It is now:
`PlanModel` and `PlanHfRepo` are deleted and a claim decodes straight into `Model`, so the
revision pin, the strict decode and the redacting `AuthToken` all reach the claim path.

The structure the crate keeps was restored with it. `pipette-plan-types` holds a model's
*location* in its own enum (`GgufText { source: GgufTextSource }`) while iOS had
`repo`, `filename` and `prefix` sitting directly on the format struct, so the crate's four-arm
shape was invisible here. `GgufTextSource`, `GgufVisionSource` and `ModelSource` now exist
with the crate's names, each carrying only the `huggingFace` case: a phone binds no local
path and fetches no bare URL, so `relative_file` / `absolute_file` / `url` are refused at
decode rather than modelled and rejected later. `reference()` and `auth_token()` sit on the
source, as upstream. A one-case enum is behaviourally identical to the struct it
replaced: what it buys is that adding an arm upstream becomes a compile error here.

`Runtime` is still two types: splitting it into identity plus settings, and dissolving
`SubmissionRef.RuntimeRef`, is the remaining half. `PlanRuntime`,
`PlanSourceRepository` and `PlanMlxSwiftStack` survive in `Client/ClientRunSpec.swift`
until it lands.

`revision` **is** carried, and the resolve URL honours it; an earlier draft of this
document had it refused, on the reasoning that a transport which ignores a pin would fetch
the mutable default branch and then submit a descriptor asserting the pin. The transport
now honours the pin instead, which is the better resolution of the same problem: the
revision is part of `HfRepo::reference` (`org/repo@rev`) and therefore part of the storage
key, so two pinned revisions of one repo no longer collide on a single entry. `Sha256`
exists as a validated type; nothing verifies a digest against a downloaded file yet, so a
`sha256` on a claim is carried and not yet enforced.

**Closed.** `auth_token` used to be emitted by `Model.encode`, which wrote a credential
into the on-disk manifest, and submitted descriptors escaped only by accident:
`SubmissionRef.ModelRef`, a second hand-written encoder predating `Model.encode`, happened
to omit the field. That made a duplicate encoder load-bearing. `Model.encode` now drops the
token unconditionally, so collapsing `ModelRef` is safe in either order; decode still
accepts one, since that is how a claim delivers it. Where the credential lives instead is
under [platform-forced divergence](#platform-forced-divergence).

`ModelRef` also omitted the model's `revision`, so the warehouse could not tell two pinned
revisions apart while the crate submits the whole `Model`. It now emits the pin, absent
rather than null when unset.

**Keep the iOS-only types.** `BatchManifest` / `CellRecord` / `BatchStatus` /
`CellRunStatus` are the on-device counterpart of `pipette-plan`'s cell state
file; `ActiveCellSentinel` is the only way a jetsam kill becomes visible;
`CellSubmissionRecord` records the collector, which the CLI cannot express;
`WorkerResolveError` occupies the space between a terminal claim rejection and a
bare run failure. All four earn their place. `JobCell` must **split**: a `Cell` carrying the work axes
(the runtime and both flag groups included) and a `CellRecord` carrying status,
acknowledgement and crash evidence. It is today the CLI's cell *plus* plan state *minus*
the runtime axis *minus* the flags, which is why the load settings end up at batch
scope. Two stored fields go with the split: `modelPath` (a property of this device at
this moment, and one the sweep is allowed to invalidate, `ModelStorageKey.of(model)`
addresses the artifact instead) and `modelName` (derived, so a cell cannot disagree with
itself about which model it names). `ModelStorageKey` has landed; the split has not.

### 4. Disk, quota, storage, execution

The store is aligned. This section is only about how the run path uses it.

**Ensure happens inside `prepare`, once per cell, before the engine exists**,
and the engine receives a bound path, never a store handle. That already holds
on the iOS side except for the double-ensure on the claim path.

**Pins are derived from disk, and this is the one place iOS should not follow
the CLI.** The CLI builds `SweepPins::for_cell` and passes it down a call stack;
the pin set is process-local and covers one cell, so two processes sharing a
workspace can evict each other's in-flight artifact, and nothing pins a queued
cell. iOS re-derives pins at every sweep from the job manifests on disk, plus
in-flight transfers, plus the entry just published. That is durable,
cross-caller, covers the whole batch, and pins a *paused* job so resuming it
does not re-download; none of which the CLI can do. Keep it, and make the
running cell's own pin explicit at the ensure call rather than relying on the
manifest having been written first: the guarantee is real but the ordering
invariant that provides it is not enforced by anything.

The platform reason this must stay disk-derived: iOS has five sweep triggers
(post-publish, launch reconciliation, foreground reconciliation, the reclaim
action, and another model's download completing) where the CLI has two, and four
of the five can overlap a run.

**Enforcement is at collection time only**, on both sides: fetch, publish,
sweep, return; a cache hit publishes nothing and therefore sweeps nothing. Peak
disk is the quota plus the artifact just fetched.

**Where the run path writes.** Artifacts live under
`jobs/<jobId>/cells/<cellId>/`, keyed by the cell rather than by a fresh result
id, and the submitted state is a sidecar record rather than a directory move.
This is better than the CLI's rename (the address is stable across the submit
boundary, and the record carries the collector), and it stays. Three additions:

- `extras.json` beside `payload.json`, carrying the engine log. The immediate
  payload is the eval failed-sample summary that checkpointing will produce.
- `state/evals/<digest16>.jsonl`, a sibling of `jobs/`, for eval resume.
- `metrics.json` gets a writer, or its reader goes away.

**Two unbounded trees, and iOS should bound one of them.** Neither client's
quota covers results: the CLI's `results/` and iOS's `jobs/` both grow without
limit. The CLI at least has `results list --state` and `results delete`; iOS has
neither, and an eval payload with two thousand completions is not small. Report
the size first (the storage card counts models only, so the number is currently
unknown to everyone) then reclaim completed jobs whose every cell carries a
server job id, which is the direct analogue of the CLI's "a result keyed by the
server's id is safe to drop".

**Crash survival.** iOS is ahead on four counts and behind on two. Ahead:
staging is addressable and resumable rather than garbage, a fetch that finished
but did not publish is published on the next launch, a payload written before a
kill is adopted rather than discarded, and a claimed job's result is written to
disk before the planner ever sees it, which the CLI explicitly does not do.
Behind: a kill mid-eval loses every sample, and the submission acknowledgement
is written after the POST rather than before, so a kill in that window
re-submits. The first is the item on the plan; the second is a known, narrow
window that the sidecar partly heals by adopting an acknowledgement the manifest
never absorbed.

**One correctness fix on the side:** the device-data reset removes two paths
that do not exist and misses the synced catalog directory, so a reset leaves
server-derived state behind. The CLI's counterpart clears the remote catalog,
the remote results, and the eval state: the same three, once the eval directory
exists.

### 5. The cell

Everything above converges here. The target is one sentence: **a cell is a value
that survives from the claim to the payload.**

Concretely, five things are lost today between the two and each is restored by
the same change:

| Lost | Restored by |
|---|---|
| `ctx_size`: decoded, persisted, then overridden | `RunRequest.runtimeFlags` overriding the per-cell default, not the reverse |
| `enable_thinking`: decoded, persisted, then dropped at `ensure` and reported `nil` | `DeclaredBound<Model>` keeping the declared half; `model_flags` on the payload |
| the declared runtime: replaced by build constants at submit | `RunRequest.runtime` echoed as the descriptor |
| the claimed engine: dropped after the compatibility check, re-derived from the model | `RunRequest.runtime` carried to dispatch and asserted against the derived engine |
| the error's type: flattened to a string across the manifest | a typed failure on the outcome, so a disposition can ever be terminal |

**Status after stage 5.** The value exists and survives: the claimed engine is carried
to dispatch and asserted there, and the declared model half is kept whole (auth token
included) rather than replaced by what the store handed back. The other three are
*carried but not yet spent*; `ctx_size` is on the cell and still loses to the computed
window, `model_flags` reaches `record` and still reports `nil`, and the declared runtime
reaches `record` while the descriptor is still regenerated from build constants. Each is
now a one-line change, which is the whole point of the carrier; each is stage 6, because
each moves a number in the warehouse. The typed failure is not done at all: a cell's
error is still flattened to a string on the record.

And six facts are currently derived twice, three of which disagree: the
benchmark body (parse vs catalog (the weak one wins), the context size (claim
vs computed) the computed one wins), and the runtime identity for the wire
(claim vs build constants, build constants win, with a warning). The other
three agree only by coincidence.

The change is to build the request once, where the manifest is built today, and
thread it to `Engine.run`. `JobManifest` and `JobCell` remain the persistence
and UI shape; the run path stops reading work axes off them.

On `enable_thinking` specifically: applying it or refusing it are both
acceptable. Silently ignoring it is not, because it mislabels results rather
than failing them, and a wrong-but-plausible number is worse than no number.

Two things the cell must also gain that are not losses but absences: sample-level
resume (the digest is computable the moment `RunRequest` exists) and a
completion-id pre-flight before submission (which only becomes necessary *after*
resume exists, because resume is what can produce a duplicate).

Finally, the hand-mirrored types need a shared fixture. No Rust links into the
app, so `ClientRunSpec`, the flag refs, and the refusal set are mirrored by hand
and nothing fails to compile when plan-types moves. A committed corpus of spec
bodies (one per runnable iOS cell, one per rejected desktop cell, one per
refusal) asserted by both the Rust claim tests and the Swift claim tests, is
the only mechanism that keeps them together.

### 6. The headless surface

The CLI's grammar is `pipette <group> <leaf> --key value`; the headless grammar
is `headlessrun <group> [<leaf>] key=value`. The separator stays `key=value`
because argv over a device console makes it the cheaper token shape and the plan
runner already emits it. **Everything else takes the CLI's name.**

Three parser rules change first, because every added parameter depends on them:
an unknown `key=` is an **error**, not silently ignored; a usage error exits
**2**, distinct from a run failure's 1; and an unrecognized `runtime=` value is
**refused** rather than defaulting to MLX.

Leaf by leaf. `H` = host-only, must not exist on a phone.

| CLI leaf | Headless today | Target |
|---|---|---|
| `init` | — | — (fixed data root, created at launch) |
| `auth register` | `register server= org= email=` | `auth register server= org= email= details= preauth= device-name=`; `org`/`email` required, defaults removed |
| `auth me` | `auth me` | `auth me`: server status, `reindex_pending`, capabilities |
| `auth set-device` | — | `auth set-device device-name= details=` |
| `auth reset` | `auth reset force=1` | `auth reset force=1` |
| `models list` | `models` | `models [format=name\|spec]` |
| `models pull` | `mlxget repo= prefix=`, `ggufget repo= file=\|quant=` | `models pull model=<json\|uri>`; both existing verbs kept as aliases |
| `models delete` | `models rm name=\|repo=` | `models delete model=<json\|uri>`; `rm` kept as an alias |
| `runtimes list` | `runtimes` | `runtimes`: the compiled-in engines and their build ids |
| `runtimes pull/remove/catalog/flavors` | — | **H**. No runtime payload exists on a phone |
| `benchmarks list` | `benchmarks [type=]` | `benchmarks [type=]` |
| `benchmarks show` | `benchmarks show benchmark=` | `benchmarks show benchmark=<id>` |
| `benchmarks init-local` | `benchmarks init-local` | `benchmarks init-local`: done |
| `benchmarks run` | `bench`, the bare form, `afm` | `benchmarks run`. See below |
| `results list` | `results [benchmark=] [type=] [state=] [limit=]` | `results [job=] [type=] [state=] [limit=]` |
| `results show` | `results show result=` | `results show result=<jobId>/<cellId>` |
| `results delete` | `results delete result=` | `results delete result=<jobId>/<cellId>` |
| `sync` | `sync [job=]` | `sync [job=…]`: pull catalog, then drain; one line per phase. Narrows by *job*, not result; see below |
| `worker` | `settings run` | `worker idle-secs= idle-jitter-secs= heartbeat-secs= max-jobs= skip-profile-refresh=` |
| `storage status` | `storage status` | `storage status` |
| `storage gc` | — | `storage gc dry-run=1` |
| — | `jobs`, `job run\|export\|submit\|rm` | keep. The batch noun has no CLI counterpart |
| — | `settings`, `settings set worker=` | keep as preferences; `quota=<preset>` added |
| — | `memseq`, `metrics=coherence\|calibrate\|promptseed`, AFM probes | move under `diag memseq` / `diag probe kind=` |
| — | `pipette://run/…` deep links | keep, with the allow-list |

`benchmarks run`, argument by argument:

| CLI | Headless today | Target |
|---|---|---|
| `--benchmark local/<id>\|remote/<id>` | `benchmarks=<id>,<id>` | `benchmark=<id>[,<id>…]`, bare ids |
| `--model <json\|uri>` | `model=<substring>`, `quant=`, or `spec=<json>` | `model=` accepting the `Model` JSON and the compact URI; substring matching moves to `match=` |
| `--runtime <json\|uri>` | `runtime=<token>`, MLX fallback | `runtime=<headless token>`, unknown refused |
| `--runtime-flags '<json>'` | — (`batch=` only; other settings hardcoded) | `runtime-flags=<json>`, decoded through the same path the claim uses |
| `--model-flags`, `--model-enable-thinking` | — | `model-flags=<json>` |
| `--http-timeout-seconds` | — | `http-timeout-seconds=` |
| `--readiness-max-wait-secs` | — | `readiness-max-wait-secs=` (blocked upstream; see divergence) |
| `--doomloop-*` | — | deferred |
| `--sync` (default off) | `submit=0\|1` (default **on**) | `sync=0\|1`, alias `submit=`, default stays **on** |
| — | `metrics=` × `offsets=` id generation | keep as a convenience; `benchmark=` is the primary form |
| — | `batch=` | subsumed by `runtime-flags`' `n_ubatch` |

**`Display` is not a `model=` spelling.** `model_uri.rs` is explicit that
`{repo}[:{path}]` is the log and warehouse identifier, distinct from the type's
serialization; `resolve_model_arg` accepts a JSON `Model`, the compact URI, or a digest
reference, and nothing else. Mirroring the display form on the client would cement a
spelling the reference rejects, so `model=` takes the two spellings the CLI parses, and
the fix for the emission described in §Why belongs upstream: the transport should emit
canonical `Model` JSON, which is what the desktop `--model` already carries. The digest
reference is the one spelling left unmirrored. It needs a descriptor digest over
installed models that iOS has no equivalent for.

**`sync` narrows by job.** A result is not addressable until `BenchmarkResultLocation` exists, so
`result=` should be refused rather than silently ignored, and score refresh (the CLI's
third phase) should report itself as not run rather than be omitted, since a missing
line reads as "nothing needed refreshing".

Output contract, uniformly: one `[HEADLESS] <group> <key=value>…` line per
record with a `<group> count=N` header before a list. A shape that is more
machine-readable than the CLI's tables and should not be traded for them.
Non-log payloads (result JSON, exported CSV) go to **stdout bare**, mirroring the CLI's
decision to keep `results show` pipeable. It prints the payload with `println!` and its
`--- extras ---` labels with `eprintln!` for exactly that reason.

The `[HEADLESS]` lines move to stderr **only for the commands that carry a payload**, and
an earlier draft of this section said "every" one. That would break the fleet:
`pipette-plan`'s `run_streaming_scanning` scans **stdout alone** for `BENCH_DONE` (stderr
is drained on a background thread and never scanned), so moving the sentinel wholesale
would leave the iOS transport unable to tell a finished run from a hung one. The transport
only ever invokes the bench forms, so `results show` is free to put its stdout to work
while they keep theirs.

Every terminal path emits `BENCH_DONE`, including the early-exit error paths that
originally did not: a console consumer keyed on the sentinel hangs otherwise. A
deliberately resident `worker` emits a readiness line once instead.

### 7. Files and method names

The dictionary above names types. Two things it left out, and both had already
drifted: which file a type lives in, and what a method that does a CLI operation
is called.

**The file rule.** A file is named for the principal type it holds. A
`Foo+Bar.swift` file holds *extensions* on `Foo` and never a type's home: a
reader who knows a type's name should be able to guess its file, and a reader who
opens a file should not find a type the name gave no hint of. The crate follows
this implicitly (`cell.rs` holds the cell, `quota.rs` the quota types), which is
why a reader of `pipette-artifacts` can navigate it without a map.

The drift this corrects. The first two rows have landed; the rest are targets:

| Type | Lives in | Target |
|---|---|---|
| `HFOrg`, `HFRepo`, `Sha256`, … | ~~`HFModelRef.swift`~~ | **`PlanTypes/Primitives.swift`**: done |
| `Model`, `Runtime` | ~~`Contracts/`~~ | **`PlanTypes/`**, beside `Artifacts/` for the on-disk mirrors: done |
| `DeviceInfo`, `DeviceFormFactor` | ~~absent; `Support/DeviceInfo.swift` held the *prober* under that name~~ | **`PlanTypes/Device.swift`**, the prober renamed to `DeviceProbe` in **`Device/`**: done |
| `IdentityStore`, `IdentityRegistration`, `ClientSettings` | ~~`FileStorage+Registration.swift`, `Contracts/DeviceSettings.swift`~~ | **`Identity/`**, one file per type: done |
| `AuthIdentity` | ~~`ManagementClient.private struct Auth`~~ | **`Networking/AuthIdentity.swift`**, as `pipette-mgmt-client/auth.rs`: done |
| `ModelArtifactStore` | `ModelEntryStore.swift` | `ModelArtifactStore.swift` |
| `SweepPlan`, `SweepPins`, `SweepEviction` | `FileStorage+Quota.swift` | `Quota.swift`, as the crate's `quota.rs` |
| `RuntimeRef`, `ModelRef` | nested in `SubmissionRef.swift` | dissolved; the encoding is `Runtime.encode` / `Model.encode` |
| `ClientRunSpec` | `PlanRunSpec.swift` | `ClientRunSpec.swift` |
| the flag refs | `PlanFlagRefs.swift` | `FlagRefs.swift` |
| `ClaimedJob` | `PlanAPITypes.swift` | `ClaimedJob.swift` |
| eight top-level types | `JobRunner.swift` | one file each, by type |
| the headless verbs | `Headless/`, with `runtimes`/`benchmarks`/`storage` merged into one file | `Commands/`, one file per command group: done |

**Directories follow the CLI's groups, not iOS's layers.** `pipette-cli` organises by
feature (`commands/`, `results/`, `benchmarks/`, `client/`, `identity/`) while iOS grew
`Contracts/`, `Services/`, `Persistence/`. A reader who knows one could not navigate the
other, so the app moved to the crate's axis: `Commands/`, `Results/`, `Benchmarks/`,
`Identity/`, `Client/` and `Storage/`, with `Catalog/`, `Services/` and `Persistence/`
dissolved into them and `Contracts/` left holding only iOS-only types. `Run/` now holds
`RunCell`, the counterpart of the crate's `run.rs`, and `Ops/` holds one file per
`pipette-ops` module: the engine-agnostic half the crate keeps behind a crate boundary
and iOS had mixed in beside the engines. `PlanTypes/`, `Artifacts/` and `Runtimes/` already
mirror their crates and stay; `Views/` and `Support/` are iOS-only and have no counterpart
to match.

The `Plan…` prefix goes with them. It marked "this mirrors a plan type", which
stopped being information the moment the mirrors collapsed into the crate's own
shapes: a prefix that says *where a type came from* rather than *what it is* is
a migration artefact, not a name.

**Method names.** Where a method performs an operation the CLI also performs, it
takes the CLI's verb: `Engine.dispatch` (not `run`), `BenchmarkCatalogSync.pull`
(not `sync`, which is the CLI's *result*-upload verb and meant the opposite here),
`recordResult` (not `writeLocal`), `ClientRunSpec.from(claim:)` mirroring
`run_spec_from_claim`. `planSweep`/`applySweep` already matched
`plan_sweep`/`apply_sweep` and were left alone.

**A departure that was reverted.** An interim step gave `Runtime`'s load-settings half its
own `RuntimeLoadSettings` type rather than folding into `RuntimeFlags`, reasoning that
`RuntimeFlags` is the *authored* wire form (every field optional) while the engines need
resolved values. Under the reference rule that does not survive: the crate's engines
default each absent setting at its point of use and report what they derived through
`RunResponse::runtime_flags`, so there is no resolved-settings type to mirror. Moving
default resolution into the engines *is* a behaviour change, which is why it is a staged
step of its own rather than part of a rename pass, but it is the direction, and
`RuntimeLoadSettings` is scaffolding to be removed, not a departure to keep.

## Platform-forced divergence

Short and honest, so nobody "fixes" a deliberate difference.

**No runtime store, and no runtime `bound` half.** Engines are compiled in. The
CLI's installer dispatch is docker, uv, an MLX virtualenv, and an archive
extraction; four mechanisms a phone has none of. Consequently: no `runtimes
pull/remove/catalog/flavors`, no runtime phase in the sweep, capabilities
derived from build constants rather than an installed inventory, and
`require_desktop_runtime` inverted; iOS refuses *desktop* runtimes.

**No workspace root and no `init`.** There is no working directory to choose;
the data root is fixed and created at launch.

**Both catalog halves.** *Done.* The catalog was server-synced only, so there was no
`benchmarks init-local`, no `local/<id>` prefix, and no way to express "these numbers
must not reach the warehouse". `benchmarks/` now holds the crate's two halves:
`local/`, seeded by `StandardBenchmarks` with the same ladder and smoke ids
`standard.rs` generates, and `remote/`, the synced one. `SourcedBenchmarkId` addresses
them, with a bare id meaning `remote` because that is the form plans and claims carry.

The rule the split exists for is enforced, not just named. A cell records the half its
benchmark came from and the submit sweep skips a `local` one, which is the crate's
`BenchmarkResultLocation::from(BenchmarkSource)`. There a `Local` result lands in
`results/local/`, where `sync` never looks. iOS keeps results under `jobs/`, so the gate
is per-cell rather than per-directory; the effect is the same. A manifest predating the
field reads as `remote`, which is what every such cell was.

The store moved with it. It was a `Data`-passing protocol with an in-memory test fake,
justified by a comment citing "the Rust `BenchmarkStore` trait", which does not exist;
the crate's is a concrete handle, deliberately (`grep -rn "trait BenchmarkStore" crates/`
is empty). It is now concrete and typed over `BenchmarkDefinition`, tests point one at a
temporary directory, and the catalog root left `metadata/` for its own top-level
`benchmarks/` directory. Ids are the filename verbatim, as the crate writes them,
instead of percent-encoded.

**N cells per invocation.** A device-console launch costs seconds, so a
thirty-cell sweep cannot be thirty processes. The batch is the iOS noun for
that, and it is a superset of the CLI's nouns rather than a replacement for
`results`.

**Submission defaults on.** The inverse of `--sync`, deliberately: a phone that
measures for an hour and publishes nothing is the failure mode worth designing
against.

**Transfers outlive the process; sweeps have five triggers.** Hence addressable
resumable staging billed to the quota, and disk-derived pins. Both are covered
above; both are places the CLI's design does not transfer.

**Kills are invisible to the process.** Jetsam reaches no error path, so
`ActiveCellSentinel` and crash-evidence reconciliation exist with no CLI
counterpart. Backgrounding cancels an in-flight cell and reports a retriable
failure, likewise.

**No subprocess.** `RunResponse.command` and `.executable` have no iOS counterpart, as
they already go unfilled for desktop MLX. The engine *log* is not covered by this and is
not forced away.

**Credential custody.** The CLI stores no HuggingFace token at all: `inject_env_hf_token`
folds `PIPETTE_HF_TOKEN` into the in-memory model on every invocation, and
`Model::without_auth_token` clears it at each point a model is written down or published.
That works because the environment is present for the whole process.

iOS has no such guarantee. A background transfer outlives the app: the OS kills the
process and relaunches it to finish, with no claim in memory and no environment to read,
so a gated download could not resume from anything. The token therefore goes to the
Keychain when a claim is decoded, keyed by the model's own reference; the crate's
per-source `reference()`, so `org/repo[@rev]:file` for a gguf and `org/repo[@rev][:prefix]`
for an mlx bundle. Keying by repo alone collided: two cells pulling different files out of
one repo shared one entry, so the second claim's token silently replaced the first's. The
fetch resolves most-specific-first: this run's token, then the one stored for this model,
then the user's own, which is iOS's analogue of `PIPETTE_HF_TOKEN`.

That resolution is one function (`resolveHfToken`) and both transports call it: the GGUF
`URLSession` request through `attachHfAuth`, and the MLX pull by handing the resolved token
to `HubApi(hfToken:)`. It was not shared before, which is how this broke: the MLX downloader
read only the user's Keychain slot, which no plan-dispatched run writes, so a plan shipping a token
opened a gated GGUF and 401'd the same repo's MLX bundle, and fetched public MLX weights
anonymously, against HF's per-IP limit that a rack of devices shares. The transport is
handed a credential rather than choosing one, which is also why its test fakes need no
Keychain (a Simulator test host cannot store items).

Both transports log which tier answered (`hf auth: … source=claim`), or that none did and
the fetch is anonymous. Without it an unauthenticated fetch of a public repo is
indistinguishable from an authenticated one until HF answers 429, which is the moment the
answer is needed. The tier's name is logged, never the token: `AuthToken` renders
`<redacted>`, and these lines interpolate only `source`. `AppLog` mirrors to stderr under
`headlessrun`, so a plan run's operator sees them without the lines entering the stdout
contract `pipette-plan` scans.

The `Model`-to-account switch lives on `KeychainHelper`, not on `Model`. A Keychain
account has no crate counterpart at all, so putting it on the mirror would be exactly the
drift this record exists to catch; the per-arm `reference` values it reads are faithful
mirrors of `GgufTextSource::reference` / `ModelSource::reference`.

The lifecycle is consequently iOS's to own, and has no CLI counterpart. `auth reset`
clears every stored repo token, because a command whose purpose is discarding credentials
cannot leave the ones this platform chose to keep.

**No `runtime_cpu_variant`.** A single static backend; the CLI already
documents iOS this way.

**Quota is a preset ladder, not a byte count**, and the sweep has a hub-cache
eviction reason the CLI has no equivalent for.

## Upstream gaps

Not platform-forced. The phone could express these; the shared wire format cannot yet.
They are listed apart from the divergences above so nobody mistakes a gap that should
close for a difference that should persist.

**`BenchmarkFlags` has no iOS variant in plan-types.** Every variant in
`crates/pipette-plan-types/src/benchmark_flags.rs` is a CLI, desktop-MLX or server cell;
none names an iOS runtime. So `readiness` and `http_timeout_seconds` are unexpressible for
a phone cell on *both* sides, and refusing the whole group is the right interim
disposition. But the readiness deadline is fully meaningful on a phone and is a
compile-time constant there today, so this is forced now and drift later. Adding the iOS
variants upstream is the prerequisite; it is not an iOS change.

**No digest verification.** `Sha256` is typed on the iOS side but nothing checks a
downloaded file against it, so a `sha256` on a claim is carried and unenforced.

## Staged plan

Ordered so nothing is built twice. Stages 1–4 are correctness fixes that stand
alone; stage 5 is the structural change everything after it leans on; stages
marked *behaviour-preserving* change no observable behaviour and can land
independently of review load elsewhere.

**Stage 1: unblock the two broken paths.** Route cell construction, on both the
claim path and the headless path, through the catalog-aware resolver with the
structured-id parse as a fallback: the resolver the executor already uses. Make
`unknownBenchmark` retriable unless the catalog is known fresh, and add the
`GET /benchmarks/{id}` fallback on a miss. Fix `model=` to accept the display
form the plan runner emits. Nothing else on this list matters while an eval
claim cannot run and a plan-dispatched cell cannot find its model.

**Stage 2: harden the headless parser.** Unknown `key=` errors, usage errors
exit 2, unknown `runtime=` refused, `BENCH_DONE` on every terminal path,
framing on stderr and payloads on stdout. Do this before adding parameters so
new ones are validated from the first day.

**Stage 3: the missing verbs, over mechanisms that already exist.** `sync`
(three phases, three count lines), `benchmarks` and `benchmarks show`,
`storage status` and `storage gc dry-run=`, `auth me` / `set-device` / `reset`,
and the `register` defaults removed. Each is a handler over an existing
function; the sweep planner is already split into plan and apply for exactly
this.

**Stage 4: payload correctness.** Drop the five retired fields, populate
`model_flags`, type the form factor, and stamp the claim echo on the typed
payload. *Behaviour-preserving* except for the fields, which is the point.

**Stage 5: introduce the cell as a value.** `RunRequest`,
`DeclaredBound<Declared, Bound>`, `RunResponse`; `prepare` and
`runCell` extracted into `Run/`, with `Engine.dispatch` taking the request and
returning the response. `JobCell` splits into `Cell` (the work axes, including the
runtime and both flag groups) and `CellRecord` (status, acknowledgement, crash
evidence); the batch manifest loses its three load-setting fields and goes to schema
2 with no migrator. `PlanModel`/`PlanHfRepo` and `PlanRuntime`/`RuntimeRef`
collapse into `Model` and `Runtime`.

A first pass at this stage was implemented on a branch and abandoned unmerged. It had
grown a second implementation alongside the first, and rebuilding bottom-up from the
primitives proved cheaper than reconciling them. The `Model` half of the collapse has
since landed that way; the rest of this stage has not. What follows is what it costs, not
what it cost.

It is **not** behaviour-preserving in five places, all consequences of deleting a
duplicated carrier rather than of the extraction:

- A `torch`/`local`/`url` model is refused as `ModelError`, which classifies
  **retriable**. It was terminal (`incompatible`, because the bridge produced a type-less
  model that paired with nothing), and a terminal answer retires the job for the whole
  fleet on one phone's say-so. A `revision` pin is *honoured* rather than refused, so it
  costs nothing here; `sha256` is carried unverified.
- An `eval` or `vl_throughput` claim now materializes a cell. The weak
  `BenchmarkDefinition(parsingId:)` gate in `makeManifest` is gone, so the claim no
  longer produces zero cells and reports terminal `unknownBenchmark`; `prepare`
  reports the id retriably instead. This is stage 1's fix arriving early, because
  `Cell.from(spec:)` is the single claim→cell path and it classifies by type.
- A malformed benchmark body no longer poisons its model's other cells: the throw
  moved out of the block that records "this model failed to load".
- MLX quant and filename display come from the coordinate, not from a path whose
  leaf is `blobs`.
- Platform admissibility moved into `Runtime`'s decoder, so an inadmissible runtime
  has no value. The refusals are the same two, with the same dispositions.

Deliberately left regenerated from build constants rather than echoed from
`request.runtime`, even after the stage lands: `runtime_descriptor`, `runtime_name`,
`runtime_version`. And `model_flags` keeps reporting `nil` even once the cell carries it.
Both are stage 6.

**Stage 6: spend the new structure.** Honour `ctx_size` as an override of the
per-cell default. Echo the declared runtime and the effective flags. Apply or
refuse `enable_thinking`, and wire the eval sampling temperature that is defined
and unread: until that lands, iOS eval numbers are not comparable with CLI eval
numbers. Re-validate flags against the resolved body and delete the last prefix
heuristic.

**Stage 7: eval checkpointing.** The store, the session, the portable digest,
the per-sample flush at the shared `evalSamples` seam, and `finalize`'s
delete-on-clean / keep-failed-rows semantics. Teach crash reconciliation that a
killed eval cell is resumable rather than failed. Then add the completion-id
pre-flight, which only becomes reachable once resume can produce a duplicate.

**Stage 8: the result nouns.** `ResultLocation` / `ResultState`,
`results list|show|delete` addressed as `<jobId>/<cellId>`, `extras.json`, and
either the score refresh or the removal of its reader.

**Stage 9: bound the results tree.** Report the size first, then reclaim
completed jobs whose cells all carry a server job id.

**Stage 10. The naming sweep.** *Behaviour-preserving.* Runs **early**, ahead
of stages 1–9 rather than after them, because every later stage reads and writes
the names it fixes: the type/method/file renames, `job` → `Batch` in code with
"Jobs" kept in the surface, the `Runtime` identity/settings split, one
`isCompatible`, one measurement-count constant, and the quota wording. It is
landing in pieces, smallest first; the primitives and the `PlanTypes/` grouping
have gone in; the `Runtime` split and `job` → `Batch` have not. Two items in this
stage are *introductions* rather than renames and each carries a behaviour
question, so they are deferred: the extracted protocol enums (`ClaimPoll`,
`LeaseKeepalive`, the two missing `SubmitDisposition` cases) and the named idle
and reindex constants.

**Stage 11: the cross-language fixture.** The shared claim corpus, asserted
from both sides. Cheap once the shapes are settled, and the only thing that
keeps them settled.

**Upstream, in parallel and not an iOS change:** iOS variants for
`BenchmarkFlags` so a phone cell can carry a readiness override at all, and
`ios_headless_args` extended to emit `runtime-flags=` once stage 3 accepts it.
The eleven iOS `RuntimeFlags` variants are otherwise dead weight over that
transport.

## Decisions

**The governing rule: the CLI's behaviour is the reference.** Where the two
disagree and iOS is not forced by the platform, iOS changes. Anything below that
reads as a preference is settled by that rule, not re-argued.

**`job` is a UI word, not a code word.** The user-facing surface keeps "Jobs".
In code the local batch becomes `Batch` (`BatchManifest`, `BatchId`,
`batches/<batchId>/`), so `job` means what the server means and nothing else. The
synthetic `plan-<jobId>` hybrid goes. The on-disk rename is affordable because
the app is only ever installed fresh.

**The claimed job and the spec come from `pipette-plan-types`.** They are not
mirrored, re-typed, or paraphrased on the iOS side: the claim envelope and the
cell it carries take the crate's names, fields and wire shapes directly. That
collapses `PlanModel`/`Model` and `PlanRuntime`/`RuntimeRef`/`Runtime` rather
than bridging them, and it makes the strict-decode behaviour the crate's, not a
second implementation of it.

**`BenchmarkSource` follows the CLI.** The compiled-in ladder and the
structured-id fallback are labelled `Local`; a `Local` result is not submitted,
exactly as the CLI refuses. The synced catalog is the only `Remote` source. This
is a behaviour change for anyone submitting hand-generated ladder ids and is
called out when it lands.

**`ctx_size`.** The authored value wins with the computed value as its floor,
refused at parse time: a claim rejection rather than a window too small for the
benchmark.

**`enable_thinking`.** Refused in stage 6, applied later. A terminal refusal
teaches the plan author immediately; a half-threaded implementation risks the
same silent mislabelling somewhere new.

**The auth token is stripped by the encoder, not by each caller.** A divergence from the
CLI, and not a forced one. The crate keeps `Model`'s serialization faithful and makes
every write site call `without_auth_token()`; the store, the descriptor digest, the
`models` listing, the submitted `model_descriptor`. That is one obligation per site, and
the obligation has already been missed here: iOS kept tokens out of submitted descriptors
only because `SubmissionRef.ModelRef`, a hand-written encoder predating `Model.encode`,
happened to omit the field, and that encoder is scheduled for deletion.

So `Model.encode` drops the token unconditionally while decode still accepts one. The
asymmetry is the price: a `Model` no longer round-trips through disk unchanged, which is
precisely why the credential needs somewhere else to live.

**Standard deviation.** Bessel-corrected sample stddev, matching the CLI: the
older producer, and the one the warehouse's history is built from.

**Score refresh.** Built, not deleted. One `GET`, the drain is its natural home,
and the detail view already renders the result.

**Cell state.** `JobCell` splits into `Cell` and `CellRecord` during stage 5,
alongside the request extraction. `runCell` is handed the `Cell` alone, so
the run path cannot read a status, adopt an acknowledgement, or write to disk;
`runCellNeedsNoBatch` is the test that keeps that true, since a single app target
gives no compile-time wall.

## Settled by the reference rule

These were open questions until the rule (the CLI's behaviour is the reference)
was applied to them consistently. Two of them overturn a recommendation this
document previously carried.

**The headless runner follows the Rust `pipette` binary.** Not "resembles", not
"covers the same ground": the same verbs, the same argument names, the same
output shape, the same exit codes, with only the host-only commands absent
(docker pulls, uv installs, runtime builds). Where a phone cannot offer a
command, it is missing rather than reinterpreted; where it can, the spelling is
the CLI's. §6's table is the specification, and a divergence found later is a bug
in the iOS side by definition.

**No free-space floor**. See the storage-quota assessment. The CLI consults no
free space, so adding the check on one client only would be the divergence.

**A recorded payload size is allowed.** The CLI records `blobs_bytes` and reads
it in place of a walk; the shared rule now says so, with the walk as the
authority and an absent field falling back to it.

**Type names come from the crate, not from this document.** Four names it prescribed
turned out not to exist upstream: `ReadinessVerdict` (the gate is a function there, not a
shape), and shortened forms of `PipetteWorkspace`, `BenchmarkResultLocation` and
`BenchmarkResultState`. One of them was implemented before the mismatch was caught. When a
row's "iOS target" disagrees with `crates/`, the crate wins and the row is the bug.

## Still to settle

- **File and function naming: the rule is settled; applying it is staged.** The rule:

  > A file is named for its principal type. A file declaring exactly one
  > top-level type takes that type's name. A file declaring a cluster takes the
  > name of the type the cluster exists to serve; any member with independent
  > callers moves to its own file. `Foo+Bar.swift` contains `extension Foo`
  > **only**. It is never a type's home, and a file holding only extensions on
  > `Foo` **must** be named `Foo+<Aspect>.swift`. Test files are
  > `<PrincipalType>Tests.swift`.

  Both directions are violated today (`ModelArtifactStore` in `ModelEntryStore.swift`;
  `MLXBenchmark.swift` holding nothing but `extension MLXRuntime`). The naming pass
  applies the rule across the tree along with the type, method and `job`→`Batch`
  renames; only the primitives and the `PlanTypes/` grouping have moved so far. Two
  file-name topics are additionally unresolved: the eight test files whose names
  describe topics rather than principal types (`ModelParsingAndCatalogTests`,
  `BenchmarkSupportTests`, …), and the SwiftUI subview clusters in `CellDetailView` /
  `ModelsView` / `ModelCatalogUI`.
- **Audit dimension 2.** Per the confidence note at the top, §2's step order is a
  paraphrase of `prepare`'s doc comment rather than a traced reading of
  `crates/pipette-cli/src/run.rs`. Confirming it step-by-step is outstanding, and is
  cheapest to do while implementing stage 5, when both sides are open at once.
