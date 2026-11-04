# Storage quota

A cap on the disk a client spends on downloaded artifacts, enforced at the
manifest layer, on the CLI and on iOS. Android has no quota or eviction; see
[client parity, gap 2](client-parity.md#gap-2-no-storage-quota-gc-or-storage-limit-setting-on-android).

## The problem it addresses

Models and runtime installs accumulate with no cap. A handful of multi-GB GGUF
or MLX models plus per-runtime binaries fills a phone or a bench box, and
nothing reclaims the space.

## Where the pieces live

| Concern | CLI | iOS |
|---|---|---|
| Accounting + sweep | `crates/pipette-artifacts/src/quota.rs` | `Artifacts/Quota.swift` |
| Size measurement | `entry_size_bytes` (`crates/pipette-artifacts/src/entry.rs`) | `entrySizeBytes` (`Artifacts/Entry.swift`) |
| Entry identity | `ModelStorageKey` / `RuntimeStorageKey` | `Artifacts/Model/Key.swift` |
| Configured limit | `identity/settings.json` (`crates/pipette-cli/src/storage_quota.rs`) | `identity/settings.json` (`StorageQuota.swift`) |
| Enforcement point | `ensure_model` / `ensure_runtime` (`crates/pipette-artifacts/src/ensure.rs`), after publishing | `Artifacts/Admission.swift`, before any byte moves |
| Free-space floor | none | `ModelAdmission`, 1 GiB of headroom |
| In-flight accounting | none | `ModelAdmission` reservations, persisted in `state/admission.json` |
| User surface | `pipette storage status` / `gc` | Settings → Model storage, headless `storage status` / `gc` |

iOS paths are under `ios/Pipette/Pipette/`.

**The enforcement points diverge on purpose.** The CLI sweeps *after*
publishing: peak disk is the quota plus the newest artifact, and a bench box
absorbs that. A phone does not. iOS admits a whole model *before* the transfer
starts, against both the configured quota and the volume's free space, and
refuses when it does not fit.

## 1. The manifest is the unit of accounting

**An entry counts toward the quota if and only if it carries a manifest this
build understands. Everything else under the store root is garbage, and garbage
is reclaimable.**

Both clients store one directory per artifact, `<key>/{manifest, blobs/}`:
`manifest.toml` on the CLI, `manifest.json` on iOS, the extension naming the
format each already speaks. One entry, one manifest, one delete, which is what
keeps a vision model's mmproj shard and an MLX model's weight shards attached
to their model instead of surviving as orphans.

Garbage is anything else: a `.staging` orphan from a crashed fetch, a directory
with no manifest, a manifest this build cannot parse (including one written at
an older version). Discovery already skipped these; the quota makes them
deletable rather than merely invisible.

On the CLI, a store can be shared by several processes (for example, a host
that runs one `pipette` process per GPU slot over one store), so a sweep can see another process's fetch
in `.staging`. Deleting one mid-download would leave that fetch to publish what
remained of it: an entry with its manifest and part of its shards. So each
fetch holds an exclusive OS file lock (`flock`, `LockFileEx` on Windows) on
`.staging/<name>.lock`, beside its staged dir, until it publishes or cleans up.
The OS releases the lock when the process exits, however it exits. A sweep
counts a staged fetch as garbage only if it can take that lock, and takes it
again before deleting; a fetch whose lock is held is no entry at all. On a
filesystem without locks the fetch runs unlocked and the sweep, unable to take
the lock, leaves it alone. iOS keeps one store per app process and does not
lock; it pins the staging dirs its live transfers write into instead.

Two processes can also miss the same entry at once, and the second publish
would delete the first one's entry, possibly while that process was running
from it. So a store's `ensure`, on a miss, takes an exclusive lock on
`.locks/<key>.lock`, waiting while another process holds it, and looks for the
entry again before fetching. The second process then finds what the first
published; if the first failed, the second fetches it itself. These lock files
are never removed (one empty file per key), and the survey skips `.locks`.

A process running from an entry needs it to stay, too, or another process's
sweep deletes the model or runtime a slot is benchmarking and the run fails on
its next file open. So `ensure_model` / `ensure_runtime` also return a shared
lock on `.locks/<key>.use`, which the run holds until its cell finishes. It is
not the fetch's `.lock`, so a miss never waits for the processes running the
entry. The lock is taken after the resolve and the entry looked up again under
it; one a sweep removed in between is fetched again. A shared lock needs only
read access, so a read-only store locks the file it has; when no lock can be
taken, the run goes on unlocked and logs a warning.

A sweep removes an entry outside `.staging`, garbage included (a build reading
another manifest version may run from it), only after taking its `.use` lock
exclusively without waiting; one it cannot take is reported in use and left.
Under the lock it only renames the entry into `.staging`, then deletes it
there, so a run waits at most for a rename. A resolve stamping `last_used_at`
at that moment fails instead of writing a manifest back over a missing payload,
since that write never creates the entry directory, and its lookup under the
lock finds nothing. On a filesystem without locks no run holds an entry, so the
sweep removes entries without the lock. `pipette models delete` and
`pipette runtimes remove` do not take it and delete even an entry in use. The
`.use` files, like the `.lock` ones, are never removed.

## 2. Size is measured by walking, once, and recorded

The measurement is the same on both clients: recursive, **does not follow
symlinks**, `st_blocks * 512` on unix (matching `du`) with a file-length
fallback. An unreadable child contributes 0 rather than failing the sweep.

The walk runs once, when the payload lands, and the result is recorded on the
entry's manifest as `blobs_bytes`. A survey then totals a store by reading one
small manifest per entry (which it already reads for identity and recency)
instead of traversing every payload file on every publish. An entry without the
field falls back to walking `blobs/`.

That recorded size covers `blobs/` alone, never the entry directory: it is
computed inside the closure that produces the manifest, so it cannot include the
manifest carrying it. A live entry therefore under-reports what removing it
frees, by one manifest: erring low, which is the safe direction for a budget.

Two things the measurement deliberately does not fix:

- **Docker runtime entries** are manifest-only; the image lives in the daemon.
  They measure ~0 and are never eviction candidates, because evicting one frees
  nothing.
- **uv and MLX venvs hardlink** into `~/.cache/uv`. The walk counts those bytes
  but removing the runtime does not reclaim them.

## 3. Eviction order is `last_used_at`

A manifest field, seeded at publish and rewritten (best-effort, via
write-then-rename) on every `ensure` hit: cache hits included, which is the
point of having it. A failed write never fails the resolve.

Ordering on `fetched_at` alone would evict the model you run daily, downloaded
first, ahead of a one-off pulled yesterday, and then re-download it next run.

## 4. The limit is configured per client

| Client | Stored at | Default |
|--------|-----------|---------|
| CLI    | `.pipette/identity/settings.json` | 200 GiB |
| iOS    | `identity/settings.json` | `min(16 GiB, 25% of volume capacity)` |

Both files hold the limit as `{"storage_quota_bytes": <bytes>}`; no CLI verb
writes it, so a persistent CLI limit is set by editing the file. The CLI's
`--storage-quota` flag and its env var override the file (see
`pipette --help`). iOS offers a preset ladder (`8/16/32/64/128 GiB`) led by the
computed default; changing the limit never evicts anything by itself.

## 5. Enforcement: after publish on the CLI, before transfer on iOS

### CLI: sweep after publish

**fetch → publish → sweep → return.** Not a pre-flight reservation: the
artifact lands first, then the store is swept back under the cap. Peak disk is
therefore `quota + size of the newest artifact`. A cache hit does not sweep.

Sweep order: all garbage, then live entries until the total is at or under the
limit:

1. Garbage (always, see below).
2. Models, least-recently-used first.
3. Runtimes, least-recently-used first. (CLI only: iOS compiles its runtimes
   into the app and has none on disk.)

**Pins**, never evicted: the entry just fetched and the in-flight cell's model
and runtime. A sweep that finds a live entry in use pins it and plans again,
so the next least-recently-used entry goes in its place. If the sweep runs out
of unpinned entries while still over, it warns and continues; a run never fails
over disk bookkeeping.

**Oversize artifact**: an artifact larger than the entire limit is refused
before the fetch starts, because otherwise the fetch would evict the whole
store and still not fit. This covers local imports, single-file downloads,
vault files and HuggingFace directory snapshots. A vault archive and every
runtime but docker (sized 0, the image lives in the daemon) are unsized and
fall to the post-publish sweep: a llama.cpp or OVMS release, for instance,
knows only its compressed archive length up front, and the extracted install
is several times that; a uv or MLX venv has no size until it is built.

`pipette-ctl` runs `pipette storage gc` on each box that answers its probe,
after the initial sync, so the box starts a run under quota. A standalone `pipette worker` does not.

### iOS admission

`ModelAdmission` gates every byte-adding path before a transfer starts. It
grants a request of `N` bytes only when both hold, over the configured quota
`Q` and the volume's free space `F`:

```
U + R + N <= Q      what the store holds, plus what is promised, plus this request
R + N + H <= F      the same promise against physical free space, keeping H back
```

`H` is 1 GiB, because iOS starts evicting caches and refusing writes well
before zero free bytes. `U` counts the HubApi cache as well as the store root,
because that tree is real and sits outside the accounted one; the snapshot
behind a pull that is still running is left out, since its own grant covers
it. `R` is each live grant in full for as long as its transfer runs: a transfer
writes its partial where `U` cannot see it, so there is nothing to subtract as
it progresses. The reservations persist, so two concurrent downloads cannot
each be promised the same bytes, and a relaunch does not forget what an
interrupted one was promised.

To fit `U` under the quota, admission evicts in the sweep order above and
logs each eviction with what went and how much it freed. A refusal names the
required, available and protected bytes separately; the protected figure tells
a person whether deleting something would help. `storage status` reports
`reserved_bytes` beside `used_bytes` so a refusal is diagnosable.

The size comes from the catalog's declared size, or, when a caller has none (a
claim, a headless pull), from the vault records or the Hub: `Content-Length`
for a GGUF file, the blob listing for an MLX directory. An artifact nobody can
size is refused rather than attempted.

**Pins** on iOS: every in-flight download, every model a running or paused job
needs, and the staging dirs live transfers write into. iOS also sweeps after
each install, as the CLI does.

### Garbage

Garbage is reclaimed *unconditionally*, whether or not the store is over its
limit, by `storage gc` and by every sweep alike. It is unaccountable by
definition and can never be pinned, so keeping it buys nothing, and a store
stranded by a manifest version bump is usually under quota, where a purely
overage-gated sweep would leave it unrecoverable without hand-deleting
directories.

## What the quota does and does not guarantee

**Does:** bound the bytes each client's artifact store holds, in steady state,
without a background job or a separate index. On iOS, also keep a granted
transfer from filling the volume.

**Does not:**

- On the CLI, prevent the device from running out of disk. The limit caps the
  store, not the volume, and nothing consults free space.
- On the CLI, bound peak usage: the newest artifact lands before the sweep runs.
- Account for everything the client writes: benchmark results and docker images
  sit outside, and `storage status` on iOS reports the store without the hub
  cache.
- Reclaim anything on a schedule. Enforcement is a side effect of collecting an
  artifact, plus the explicit `gc` / "Free up space" (and, on a `pipette-ctl`
  run, the `gc` after initial sync).
