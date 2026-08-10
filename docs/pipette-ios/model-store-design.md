# iOS model store: design

A design record for restructuring iOS model storage along the lines
`crates/pipette-artifacts` already follows. No code yet; this is the shape to
agree on first.

Related: [storage quota](../storage-quota.md)

## Why

Four call sites answer "is this model on the device?" four different ways:

| Call site | How it resolves |
|---|---|
| `AddModelsView` | `startDownload(spec, declaredSizeBytes:)`; fire-and-forget; the only quota gate |
| `HeadlessRunner:403` | scan `availableModels()` for `$0.source == target` → `startDownload()` → poll → re-scan |
| `HeadlessDownloadCommands:54` | bare `startDownload(source)` |
| `PlannerWorker` / `JobExecutor:316` | `storage.resolveModelPath(cell.modelPath)`, fail if absent: never fetches |

Consequences already observed: the oversize refusal is bypassable from three of
the four; a planner job whose model was evicted fails instead of re-fetching;
provenance is *reconstructed* at completion time and can key weights to the
wrong entry; and `DownloadCoordinator` owns transport, layout, install, sweep,
and provenance at once.

`pipette-artifacts` solved this shape already. The point of this document is to
import its strategy rather than invent a second one.

## The strategy being imported

### 1. The store owns *where*; the fetcher owns *how*

`ModelArtifactStore::ensure(declared, fetch)` computes the key, stages a
directory, and hands the fetcher the destination; then publishes and writes the
manifest itself. A fetcher cannot invent layout, because it never sees the store
root.

*iOS today:* `DownloadCoordinator` picks the destination, moves the file, and
writes the manifest, per format, in two installers.

*Change:* `ModelEntryStore.ensure(spec, fetch:)` owns key → stage → publish →
manifest. `fetch` receives a staging directory and puts bytes in it.

### 2. The entry directory is checked against its manifest

The Rust manifest records the model **twice**: `declared` (the plan identity)
and `stored`; the same `Model` with its source arms rewritten relative to the
store root (`to_stored`), rebound via `bind_under(models_dir)` at load. Reading
validates that `stored` still equals a fresh derivation: a drift check.

**iOS should take the check and skip the second field.** The `stored` form earns
its place in Rust because `Model` is also the *loader* input, so the manifest has
to carry a bindable path. iOS has `ResolvedModel` for that role, and (given the
app is only ever installed fresh) the relocation case `stored` would protect
against does not arise: a fresh install has no prior container to hold stale
paths, and models are backup-excluded so they never return from a restore.

*iOS today:* the manifest records only `source`; paths are recomputed from the
spec at each read. `FileStorage.resolveModelPath` carries a fallback for a
changed data-container UUID that looks the filename up under the current
`modelsDir`.

*Change:* validate on read that the entry directory's name equals
`ModelStorageKey.of(manifest.source)`, and treat a mismatch as garbage. That is
the whole benefit of the drift check in three lines, and it closes a real bug
found in review: `deleteModel` recomputes the entry dir from the spec while
discovery lists whatever directory held a readable manifest, so a directory whose
name disagrees with its manifest is listed, occupies quota, and cannot be deleted.

The container-UUID fallback becomes dead code and is deleted outright.

The *other* stale-path case (a job manifest persists an absolute
`PendingCell.modelPath` at creation, and the model is evicted before the job
runs) is not a path problem and `stored` would not have fixed it. The cell also
carries `source`, so `ensure(cell.source)` re-fetches. See §"What this deletes".

### 3. Atomic publish through a staging directory

`install_dir_with_manifest` stages into `.staging/<uuid>`, runs the fetch, writes
the manifest **last** (refusing to clobber a same-named file the payload shipped),
renames into place, and cleans up on every failure path. A reader never sees a
half-written entry; a crash leaves an orphan under `.staging` that the sweep
reclaims.

*iOS today:* `MLXModelInstaller` does `if exists { removeItem }; moveItem`. A
failure between the two leaves the entry destroyed and not replaced.

*Change:* the same engine in Swift, using `FileManager.replaceItemAt` for the
publish step. The staging state lives on disk, not in memory, which is also what
makes a transfer that outlives the process recoverable.

### 4. Strict on the resolve path, lenient in the accountant

`find` / `ensure` treat an unreadable or wrong-version manifest as an error;
corruption surfaces where it matters. `quota::survey` treats the same entry as
garbage, so `storage gc` still works on a store that `find` refuses.

*iOS today:* roughly this by accident (discovery skips, the sweep reclaims); it
should be stated and tested as policy.

### 5. Typed errors per layer

`ModelStoreError` separates `NotStorable`, `UnresolvedPath`, `Corrupt`, `Fetch`,
`Io`, `Parse`. Callers distinguish "this model can never be stored" from "this
fetch failed".

*iOS today:* mostly `DownloadError.io(String)`.

*Change:* a `ModelStoreError` enum mirroring the Rust split.

## Target shape

```
Persistence/ModelEntryStore.swift      // find / ensure / list / remove — owns layout
Persistence/EntryStaging.swift         // stage → prepare → manifest → atomic publish
Contracts/ModelManifest.swift          // + `stored`, bind(under:), drift check
Models/ModelProvisioner.swift          // ensure = preflight → find → fetch → publish → sweep
Models/ModelTransport.swift            // protocol: the URLSession/HubApi seam
Models/DownloadCoordinator.swift       // conforms to ModelTransport; transport + progress only
```

```swift
struct ModelEntryStore {                       // nonisolated; owns layout
    func find(_ spec: Model) throws -> InstalledModel?
    func list() throws -> [InstalledModel]
    func remove(_ spec: Model) throws -> Bool

    // The staging split — see "The downloader".
    func stage(_ spec: Model) throws -> StagedEntry
    func staged() -> [StagedEntry]             // survivors of a previous launch
    func publish(_ staged: StagedEntry) throws -> InstalledModel   // atomic, idempotent
}

@MainActor
final class ModelProvisioner {
    func installed(_ spec: Model) -> ResolvedModel?
    func ensure(_ spec: Model, pinning: SweepPins) async throws -> ResolvedModel
    func cancel(_ spec: Model)                 // cancel the transfer, drop staging
}
```

`ensure`, step for step against `ensure_model`:

```
1. if let model = installed(spec) { touchLastUsed(spec); return model }
2. try preflight(spec)                 // declared size vs quota and vs free space
3. try await registry.join(key) { staged = store.stage(spec)
                                    try await transport.transfer(spec, into: staged.blobs)
                                    store.publish(staged) }
4. sweep(pinning: pins.inserting(key(spec)))
5. return installed(spec) ?? throw .installedButNotDiscoverable
```

Step 5 is a postcondition `HeadlessRunner` already open-codes ("download
finished but the model didn't appear on device"); it belongs here once.

## The downloader

This is the part the Rust design cannot supply. `ModelArtifactStore::ensure`
takes a synchronous `FnMut` fetcher because a CLI process outlives its own
downloads. On iOS the opposite is true: the download outlives the process.

### Publishing is decoupled from awaiting

The Rust store already keeps its in-progress state on disk (`.staging/<uuid>`);
iOS needs that state to be *addressable* so that whoever observes completion can
finish the job. Hence `stage` / `publish` as separate operations:

- `stage(spec)` creates `models/.staging/<key>/` holding `blobs/` and a small
  record of the spec, and returns a `StagedEntry`.
- The transport writes into `staged.blobs`.
- `publish(staged)` writes the manifest and renames the directory onto
  `models/<key>`: atomic, and **idempotent**, so it is safe to call from the
  awaiting task, from a background-session delegate with no awaiting task, or on
  the next launch.

`ensure`'s `await` therefore becomes an optimization for the live-app case
rather than the mechanism. If the app dies mid-transfer, the next `ensure`
finds either a published entry or a `StagedEntry` to resume: not a restart.

### Use the background session, not the pretty API

`try await URLSession.shared.download(from:)` is the idiomatic-looking choice and
the wrong one: async `URLSession` transfers **do not continue when the app is
backgrounded**, which for a multi-GB model means every home-button press restarts
progress. The right tool stays the delegate-driven background
`URLSessionConfiguration.background` session already in `DownloadCoordinator`,
bridged to `async` with a continuation held in the transfer registry.

So the shape is "async on the outside, delegate on the inside":

```swift
protocol ModelTransport: Sendable {
    /// Begin or resume the transfer into `destination`. Returns when the bytes
    /// are in place. Progress is published to the registry, not returned.
    func transfer(_ spec: Model, into destination: URL) async throws
    func pause(_ key: ModelStorageKey)
    func cancel(_ key: ModelStorageKey)
}
```

Two conformances, because the durability genuinely differs and pretending
otherwise would hide it:

| | `GGUFTransport` | `MLXTransport` |
|---|---|---|
| Mechanism | background `URLSession` + delegate | `HubApi` snapshot in a `Task` |
| Survives app suspension | yes | no |
| Survives app termination | yes: delegate wakes the app to publish | no |
| Resumes at | byte offset (resume data) | whole-file granularity, from the hub cache |

### Progress is observed, completion is awaited

Two callers can want the same model (the UI and the planner race today), so
progress cannot be a return value. A stream returned from `ensure` would have to
be duplicated per caller. Keep progress as the existing `@Observable` registry
that SwiftUI already binds to, and let `ensure` be a plain
`async throws -> ResolvedModel`. Shared mutable state for the thing that is
genuinely shared; structured concurrency for the thing that is per-caller.

### Cancellation has two meanings

- **Stop waiting**: the awaiting `Task` is cancelled. The transfer continues;
  other waiters are unaffected. Implemented with `withTaskCancellationHandler`
  decrementing a waiter count, never touching the transfer.
- **Cancel the download**: `provisioner.cancel(spec)`. Stops the transfer,
  drops staging, and fails every waiter.

Conflating these is the most likely bug in this area: today `cancel(key:)`
removes the `downloads` row, and a completion racing it was an
already-fixed TOCTOU. The registry makes the distinction explicit:

```swift
actor TransferRegistry {
    /// Join the in-flight transfer for `key`, or start one. Cancelling the
    /// caller's task leaves the transfer running for other waiters.
    func join(_ key: ModelStorageKey,
              start: @escaping () async throws -> InstalledModel) async throws -> InstalledModel
}
```

## Where iOS must diverge

**Transfers outlive the process**, and **concurrent callers race for the same
model**: both addressed by "The downloader" above (staging/publish split,
transfer registry). They are the two places the Rust design does not transfer
directly.

**No runtimes on device.** iOS compiles its engines in, so only the model half
of the crate has an analogue. There is no runtime store and no runtime phase in
the sweep.

**Main actor.** The Rust store is synchronous and thread-agnostic. `ModelEntryStore`
should be a `nonisolated struct` doing file I/O off the main actor, with only the
observable bits (`ModelStore`, progress) main-actor bound. Today the recursive
`DiskUsage` walk runs on the main actor.

## What this deletes

- `FileStorage.resolveModelPath`'s container-UUID fallback and its filename lookup.
- `DownloadCoordinator.captureProvenance` and the reconstruct-the-spec-at-completion path.
- The discover → start → poll → re-discover dance in `HeadlessRunner`.
- `sweepAfterInstall` as a coordinator concern.
- `startDownload` as public API: replaced by `ensure` everywhere.

Eviction also stops being dangerous: with `ensure` on the job path, a reclaimed
model is re-fetched rather than failing the run, which is what makes the store a
cache rather than something a sweep can break.

## Migration

Incremental, each step shippable:

1. Add `stored` to the manifest with `bind(under:)` and the drift check; delete the path fallback.
2. Add `EntryStaging` and route both installers through it.
3. Add `ModelEntryStore` (`find`/`list`/`remove`/`ensure`), backed by the above.
4. Add `ModelProvisioner.ensure`; convert `AddModelsView` first (it already has the gate).
5. Convert `HeadlessRunner`, `HeadlessDownloadCommands`, then `PlannerWorker`/`JobExecutor`.
6. Delete `startDownload` and the dead paths above.

Steps 1–3 are behaviour-preserving. The quota gate becomes unbypassable at step
5, which is the point of the exercise.

## Settled

- **The app is only ever installed fresh.** No migrators, no compat shims, no
  legacy-layout handling anywhere in this design.
- **No `stored` field**: a key-vs-directory check on read replaces it (§2).
- **Staging is resumable, not garbage.** The staging/publish split exists
  precisely so a killed transfer resumes; only staging with no recoverable
  transfer is reclaimed as garbage.

## Open questions

- `ModelStore` (the `@Observable` UI list) and `ModelEntryStore` need
  distinguishable names.
- Does `MLXTransport` warrant a background-session equivalent, or is
  "restarts on app death, resumes at file granularity" acceptable? It is the
  larger of the two download types, so this decides whether a killed 6 GB MLX
  pull costs minutes or nothing.
- Where the free-space floor from [storage-quota](../storage-quota.md#assessment-2026-07-27)
  belongs: `preflight` is the obvious home, but the check wants
  `reclaimable bytes` from the sweep planner to avoid refusing a download the
  sweep would have made room for.
