# iOS model store: design

How the iOS model store follows the strategy of `crates/pipette-artifacts`, and where
it has to differ. Paths are under `ios/Pipette/Pipette/`, one directory per crate
module (`Artifacts/`, `Artifacts/Model/`; see
[how a file finds its place](architecture.md#how-a-file-finds-its-place)).

Every byte-adding path (the Add Models sheet, the `models` and `benchmarks` headless verbs, a
planner job's `RunCell`) goes through one entry point, `ensureModel` in
`Artifacts/Ensure.swift`, the crate's `ensure_model`. Because a job's cell fetches through it too, a model the
sweep evicted is re-fetched rather than failing the run: that is what makes the store a
cache a sweep cannot break. Admission and the free-space floor are described in
[storage quota](../storage-quota.md).

## The strategy being imported

### 1. The store owns *where*; the fetcher owns *how*

The crate's `ModelArtifactStore::ensure(declared, fetch)` computes the key, hands the
fetcher the destination, then writes the manifest itself. A fetcher cannot invent
layout, because it never sees the store root.

On iOS, `ModelArtifactStore` (`Artifacts/Model/Store.swift`) owns `find` / `list` /
`remove` / `ensure`, and `fetchDestination(for:)` answers where each file lands.
An entry is published only by `publishManifest(for:)` (GGUF) or `publish(_:)` (a
staged directory model).

### 2. The entry directory is checked against its manifest

The manifest records the model twice: `declared` (the plan identity) and `stored`, the
same `Model` with its source arms rewritten relative to `models/` (`to_stored`), bound
back with `bind_under(models_dir)` at load. Reading validates that `stored` still
equals a fresh derivation from `declared`: a drift check.

iOS records `stored` as the crate does and reads it back through the same check
(`ModelArtifactStore.readManifest`). A field the reader re-derives and compares costs
nothing to keep, and one layout rule on both clients is worth more than the lines a
key-vs-directory check would save. Two older shapes are read rather than refused: a
record without `fetched_at` or `stored` is recovered with the missing fields derived,
and a `stored` an earlier build measured from the entry instead of from `models/` is
re-rooted and rewritten.

### 3. Atomic publish through a staging directory

The crate's `install_dir_computing_manifest` stages into `.staging/`, runs the fetch,
writes the manifest **last**, renames into place, and cleans up on every failure path.
A reader never sees a half-written entry; a crash leaves an orphan under `.staging`
that the sweep reclaims.

iOS does the same in `Artifacts/Model/Staging.swift`:

```
models/.staging/<key>/<attempt-id>/{transfer.json, blobs/…}
models/<key>/{manifest.json, blobs/…}
```

`stage` opens an attempt, `publish` checks the payload is complete (and, with a frozen
plan, every file's length and any digest), writes the manifest into staging, then swaps
the directory onto the entry with `FileManager.replaceItemAt` (a plain move for a first
install). The store holds either the old entry or the new one, never neither.

Directory models (MLX, ONNX GenAI, vault archives) stage and publish through
`DirectoryModelInstaller`. A GGUF file uses `.staging` only as scratch and publishes
per file, because a vision model is two transfers into one entry with no single
attempt to publish.

### 4. Strict on the resolve path, lenient in the accountant

On iOS, `find` and `ensure` treat an unreadable or wrong-version manifest as an error, so
corruption surfaces where it matters. The crate's `find` does too, but its `ensure` removes
and refetches a manifest that does not decode, and errors only on one that decodes but
fails a check (a wrong version, or `stored` drift). The quota survey (`Artifacts/Quota.swift`) treats
the same entry as garbage, so the sweep still works on a store that `find` refuses.
Discovery for the UI is lenient too: one unusable manifest does not hide every other
model.

### 5. Typed errors per layer

The crate's `ModelStoreError` separates `NotStorable`, `UnresolvedPath`, `Corrupt`,
`Quota`, `Fetch`, `Io` and `Parse`, so a caller can tell "this model can never be
stored" from "this fetch failed". iOS has its own `ModelStoreError`
(`Artifacts/Model/Store.swift`), narrowed to the cases it can reach (no
`UnresolvedPath` or `Quota`), plus `typeCollision` from `find` and the ways `ensureModel`
can end: `admission`, `notFetchableHere`, `fetchTimedOut`, `fetchPaused`,
`fetchCancelled`, `unresolvedAfterFetch`.
`ModelFetchError` (`Artifacts/Model/Fetch.swift`) is the planner's and the digest
check's.

### 6. The planner decides, the transfer obeys, the digest is checked

The crate's `parts` (`crates/pipette-artifacts/src/model/fetch.rs`) takes the declared
model and the bound destination and returns what the model consists of: one
`Download` per file (where from, how to authenticate, what to verify, where to) and one
part per directory (a Hugging Face snapshot, a vault archive or a local directory). It
is pure, so every decision about a file is made before a byte moves and the loop that
moves bytes makes none.

On iOS, `ModelFetch.planDownloads(_:into:)` plans the GGUF files, with the destination
from `ModelArtifactStore.fetchDestination`. `DownloadCoordinator.startDownload`
enqueues what it answers; a refusal is a typed `ModelFetchError`. Vault files are not
planned as URLs: each goes out as a signed request (`ModelFetch.vaultFiles`). MLX and
ONNX GenAI models are planned by the Hub pull instead: `HubTransferPlanner` freezes the plan before any
payload byte moves, resolving the declared revision to one immutable commit and listing
every file with its expected length (and digest, when the source offers one). The
frozen plan is written to the attempt's `transfer.json` and its total is what admission
reconciles against.

**The digest a plan carries is checked.** For a GGUF file it rides the planned
`Download` through the in-flight row and the persisted `DownloadRecord` to
`GGUFModelInstaller`, which hashes the delivered bytes before moving them into the
entry's `blobs/`. A nil digest verifies nothing, as upstream. On a mismatch the install
throws `ModelFetchError.sha256Mismatch` (destination, expected, actual), nothing is
published, and the row fails with that reason. A model whose plan carries a `sha256`
can therefore fail a download whose bytes arrived; that is the point. A vault file's
pin is checked while it streams instead.

The landing path is asked of the store, not joined from the transfer's filename. A
transfer knows only the leaf, so a repo path naming a subdirectory (`quants/m.gguf`)
would otherwise land at `blobs/m.gguf` while the manifest records
`blobs/quants/m.gguf`. `GGUFModelInstaller.destination` takes the bound path whose leaf
matches, so what lands and what `stored` records come from one rule.

Hashing runs off the main actor. The `URLSession` delegate's temp file is deleted when
the delegate returns, so `adopt` moves it under `.staging` synchronously, and `finish`
hashes and installs it in a detached task. The coordinator holds the background
session's completion handler until pending installs drain. A process that dies
between the two halves leaves an orphan under `.staging` that the sweep reclaims.

## The downloader

This is the part the Rust design cannot supply. The crate's `ensure` takes a
synchronous `FnMut` fetcher because a CLI process outlives its own downloads. On iOS
the opposite is true: the download can outlive the process.

### Publishing is decoupled from awaiting

Whoever observes completion installs the bytes, not the caller awaiting them: the
`URLSession` delegate installs a GGUF file even when no task is waiting (the system can
relaunch the app for it), and a directory pull's task installs through
`DirectoryModelInstaller`, which stages and publishes in one call. `ensureModel`'s wait
polls the store, not a transfer row, so it returns as soon as the entry resolves however
it got there. Nothing resumes a staged attempt in a later launch (`staged()` lists
them); the sweep reclaims any that no live transfer owns.

### Use the background session, not the async API

`URLSession.shared.download(from:)` looks idiomatic and is wrong here: async
`URLSession` transfers do not continue when the app is backgrounded, which for a
multi-GB model means every home-button press restarts progress. GGUF transfers use the
delegate-driven `URLSessionConfiguration.background` session in `DownloadCoordinator`.

The transports differ in durability, and the code does not hide it:

| | GGUF (`DownloadCoordinator`, `URLSession`) | Directory (`HubModelTransport`) | Vault (`startVaultDownload`) |
|---|---|---|---|
| Mechanism | background `URLSession` + delegate | `HubApi` pull in a `Task`, foreground session | signed requests in a `Task` |
| Survives app suspension | yes | no | no |
| Survives app termination | yes: the delegate wakes the app to install | no | no |
| Resumes at | byte offset (resume data) | whole-file granularity: finished files are kept | `Range` within one process; from zero in a new launch |
| Retries a transient failure | 5 attempts; a resume must match the first response's `ETag` | 5 attempts per file; a broken file starts again | 5 attempts per file |

`ModelTransport` (`Models/ModelTransport.swift`) is the seam for the directory column:
one conformance, `HubModelTransport`, serves both MLX and ONNX GenAI, because those
differ in which files they select and how they are validated, not in how bytes move.
Tests inject a fake. The vault pull stays off the background session because each
signed request carries a single-use nonce, and a system retry or relaunch would replay
a stale signature.

A `headlessrun` launch is the exception in the GGUF column. Every launch except
`headlessrun settings run` downloads on a foreground session
(`DownloadCoordinator.usesForegroundSession`), because a background one is throttled
on a warm phone. That includes `headlessrun worker`, the resident claim loop. A
foreground transfer dies with the process. So
before the process exits, normally or on SIGTERM, `keepPartialsForNextLaunch` pauses
in-flight transfers and writes their resume data, and the next `ensureModel` resumes
from that byte offset. A SIGKILL or a crash still loses the partial. `settings run`
keeps the background session, because the planner worker can be suspended between
claims.

`ensureModel` waits until no byte has arrived for 10 minutes (`stallTimeout`), not for
a fixed time. A transfer that keeps moving is never cut off; the farm's per-launch
limit bounds it, and the next launch carries on from the partial.

### Progress is observed, completion is awaited

The UI and a job can want the same model at once, so progress cannot be a return
value: a stream returned from `ensureModel` would have to be duplicated per caller.
Progress stays on the `@Observable` `DownloadCoordinator` that SwiftUI binds to, and
`ensureModel` is a plain `async throws` returning the bound model. Shared state holds
what all callers share, and structured concurrency holds what belongs to one caller.

### Cancellation has two meanings

- **Stop waiting**: the caller's own `Task` is cancelled. The transfer continues and
  other waiters are unaffected (`withTaskCancellationHandler` drops only that waiter).
- **Cancel the download**: `TransferRegistry.cancel(_:)`, reached from
  `DownloadCoordinator.cancelTransfers(for:)`. The transfer stops, its rows go, and
  every waiter fails with `ModelStoreError.fetchCancelled`.

Keep the two apart. If the waiter owned the transfer, a cancelled job would hand back
the byte budget that its admission took while the download kept writing, and the
ledger would offer those same gigabytes to the next admission.

`TransferRegistry` (`Artifacts/TransferRegistry.swift`) puts each transfer in a task of
its own. `join(key, start:)` joins the in-flight transfer for a key or starts one;
`ensureModel` puts the whole byte-adding half (admit, start, wait, release) inside
`start`, so it runs once per transfer however many callers join. A cancelled transfer
stays registered until the coordinator confirms its transport stopped, so a caller
arriving meanwhile gets the same cancellation instead of starting a replacement.

It is a `@MainActor` class rather than an actor because everything `join` serializes
is already main-actor bound, and a second actor would add hops between steps that must
not interleave.

## Where iOS must diverge

**Transfers outlive the process**, and **concurrent callers race for the same
model**: both addressed by "The downloader" above. They are the two places the Rust
design does not transfer directly.

**Admission before the transfer.** The crate sweeps after publishing; iOS refuses a
model that does not fit the quota and free space before any byte moves. See
[storage quota](../storage-quota.md).

**No runtimes on device.** iOS compiles its engines in, so only the model half of the
crate has an analogue. There is no runtime store and no runtime phase in the sweep.

**Main actor.** The Rust store is synchronous and thread-agnostic. `ModelArtifactStore`
is a `nonisolated struct`, with only the observable parts (`ModelStore`, the
coordinator's progress) main-actor bound. UI discovery
(`PipetteWorkspace.availableModels`) still runs the recursive `entrySizeBytes` walk on
the main actor.

## Open question

Should the directory pull get a background-session equivalent, or is "restarts on app
suspension, keeps finished files" acceptable? MLX bundles are the larger downloads, so
this decides whether a killed multi-GB pull costs minutes or nothing.
