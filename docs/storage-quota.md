# Storage quota

A cap on the disk a client spends on downloaded artifacts, enforced at the
manifest layer. This document describes how it works as built, then assesses
whether the approach is the right one.

## The problem it addresses

Models and runtime installs accumulate with no cap. A handful of multi-GB GGUF
or MLX models plus per-runtime binaries fills a phone or a bench box, and
nothing reclaims the space.

---

# How it works

## Where the pieces live

| Concern | CLI | iOS |
|---|---|---|
| Accounting + sweep | `crates/pipette-artifacts/src/quota.rs` | `Persistence/FileStorage+Quota.swift` |
| Size measurement | `entry_size_bytes` (`src/entry.rs`) | `Persistence/DiskUsage.swift` |
| Entry identity | `ModelStorageKey` / `RuntimeStorageKey` | `Contracts/ModelStorageKey.swift` |
| Configured limit | `identity/settings.json` (`src/storage_quota.rs`) | `metadata/settings.json` (`FileStorage+Settings.swift`) |
| Enforcement point | `ensure_model` / `ensure_runtime` (`src/ensure.rs`) | `DownloadCoordinator.sweepAfterInstall` |
| User surface | `pipette storage status` / `gc` | Settings → Model storage |

## 1. The manifest is the unit of accounting

**An entry counts toward the quota if and only if it carries a manifest this
build understands. Everything else under the store root is garbage, and garbage
is reclaimable.**

Both clients store one directory per artifact, `<key>/{manifest, blobs/}`:
`manifest.toml` on the CLI, `manifest.json` on iOS, the extension naming the
format each already speaks. One entry, one manifest, one delete, which is what
keeps a vision model's
mmproj shard and an MLX model's weight shards attached to their model instead of
surviving as orphans.

Garbage is anything else: a `.staging` orphan from a crashed fetch, a directory
with no manifest, a manifest this build cannot parse (including one written at
an older version). Discovery already skipped these; the quota makes them
deletable rather than merely invisible.

## 2. Size is measured by walking; the CLI records the result

The measurement is the same on both clients: recursive, **does not follow
symlinks**, `st_blocks * 512` on unix (matching `du`) with a file-length
fallback. An unreadable child contributes 0 rather than failing the sweep.

**On the CLI** the walk runs once, when the payload lands, and the result is
recorded on the entry's manifest as `blobs_bytes`. A survey then totals a store
by reading one small manifest per entry (which it already reads for identity
and recency) instead of traversing every payload file on every publish. An
entry written before the field falls back to the walk.

That recorded size covers `blobs/` alone, never the entry directory: it is
computed inside the closure that produces the manifest, so it cannot include the
manifest carrying it. A live entry therefore under-reports what removing it
frees, by one manifest: erring low, which is the safe direction for a budget.

**On iOS** the walk runs at calculation time (`DiskUsage`) and nothing is
recorded. Phone-sized stores hold a handful of entries, so the traversal is
cheap and the extra manifest field would not pay for itself.

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
| iOS    | `metadata/settings.json` | `min(16 GiB, 25% of volume capacity)` |

CLI precedence: `--storage-quota` > `PIPETTE_STORAGE_QUOTA` >
`identity/settings.json` > default. iOS offers a preset ladder
(`8/16/32/64/128 GiB`) led by the computed default; changing the limit never
evicts anything by itself.

## 5. Enforcement happens at collection time

**fetch → publish → sweep → return.** Not a pre-flight reservation: the
artifact lands first, then the store is swept back under the cap. Peak disk is
therefore `quota + size of the newest artifact`.

Sweep order; all garbage, then live entries until the total is at or under the
limit:

1. Garbage (always, see below).
2. Models, least-recently-used first.
3. Runtimes, least-recently-used first. (CLI only: iOS compiles its runtimes
   into the app and has none on disk.)

**Pins**; never evicted: the entry just fetched, everything the in-flight plan
declares, and on iOS every in-flight download plus every model a running or
paused job needs. If the sweep runs out of unpinned entries while still over,
it warns and continues; a run never fails over disk bookkeeping.

**Oversize artifact**; an artifact larger than the entire limit is refused
before the fetch starts, because otherwise the fetch would evict the whole store
and still not fit. On the CLI this covers local imports, single-file downloads,
and HuggingFace directory snapshots (llama.cpp releases are deliberately
unsized: only the compressed archive length is knowable up front, and the
extracted install is several times that).

Garbage is reclaimed *unconditionally*, whether or not the store is over its
limit, by `pipette storage gc` and by the post-publish sweep alike. It is
unaccountable by definition and can never be pinned, so keeping it buys nothing,
and a store stranded by a manifest version bump is usually under quota, where
a purely overage-gated sweep would leave it unrecoverable without hand-deleting
directories.

## What the quota does and does not guarantee

**Does:** bound the bytes each client's artifact store holds, in steady state,
without a background job or a separate index.

**Does not:**

- Prevent the device from running out of disk. The limit caps the store, not the
  volume. Nothing consults free space at any point.
- Bound peak usage: the newest artifact lands before the sweep runs.
- Account for everything the client writes: the iOS MLX hub cache, benchmark
  results, and docker images all sit outside.
- Reclaim anything on a schedule. Enforcement is a side effect of collecting an
  artifact, plus the explicit `gc` / "Free up space".

---

# Assessment (2026-07-27)

Reviewed after implementation, against what the feature is actually for.

## What should stay

- **Manifest as the unit of accounting.** This is the load-bearing idea and it
  paid off: no separate index to keep in sync, artifact grouping falls out for
  free, and "unreadable ⇒ reclaimable" gives a recovery path that needs no
  migration tooling.
- **Recording the walk on the CLI.** A survey reads manifests it already opens
  rather than walking every payload; the drift is bounded to one manifest and
  the cost
  is invisible at realistic store sizes.
- **`last_used_at` over `fetched_at`.** One small write per resolve buys an
  eviction order that doesn't churn the model you use most.
- **Pins passed down from the run layer.** The fetch layer genuinely doesn't
  know what the plan needs; making that explicit is right.

## What I think is wrong, in order

**1. It caps the wrong quantity.** The failure users hit is "the device ran out
of disk", and a store-size cap does not prevent it. Free space is never
consulted: not in the oversize pre-flight, not before a download starts, not in
the sweep. A 32 GiB limit on a phone with 4 GB free permits a 20 GB download
that will fail on the filesystem partway through, and a 200 GiB CLI limit on a
half-full 500 GB bench box does nothing until the disk is already full. The cap
also ignores every other consumer on the volume.

*Recommendation:* keep the configured cap, add a free-space floor. Before a
fetch, refuse when `declared_size > free_space - reserve` (reserve on the order
of a few GB), and let the sweep count reclaimable bytes toward satisfying it.
This is a small change to code that already exists and it addresses the actual
failure.

**2. Auto-eviction is a surprising default on an interactive client.** Starting
a download silently deleting a multi-GB model the user chose to keep is
defensible for an unattended fleet worker and hostile in a phone UI. The CLI
reports every eviction; iOS currently reports none, so models just vanish from
the Models tab.

*Recommendation:* split the policy by client rather than sharing one. Headless
and CLI: evict and report. iOS: report what was evicted, and consider refusing
with "free up space to continue" instead of evicting when the user is present.

**3. "No migration" was cheap to decide and is expensive to live with.** The
call was made before the code existed. Now the concrete cost is that every
existing device and dev box re-downloads its entire model set on upgrade:
tens of GB, sometimes metered. Seeding `last_used_at = fetched_at` for a v1
manifest is roughly ten lines and removes that cost entirely.

*Recommendation:* revisit. The no-migration stance is right for the *layout*
change on iOS (the old bucket tree genuinely can't be reinterpreted) and wrong
for the CLI manifest version bump, where the only new field has an obvious
value.

**4. Enforcement only at collection time leaves obvious gaps.** A store goes
over quota and stays there whenever nothing is being fetched: garbage
accumulates after the last publish, a lowered limit doesn't take effect, and a
long-lived worker never reconciles. `gc` exists but is manual.

*Recommendation:* sweep at one more point (CLI worker startup, iOS app
foreground), which costs one walk and closes the gap without a background job.

**5. The accounting is honest but the number is misleading.** `storage status`
reports store bytes, while docker images (potentially tens of GB) count as zero
and the iOS hub cache is excluded. A user reading "12 GiB used" can be holding
far more.

*Recommendation:* report the excluded consumers as separate lines rather than
silently omitting them.

## Verdict

The mechanism is sound and I would not rebuild it. The design error is one of
framing: it caps *the store* when the thing worth bounding is *the device*.
Recommendations 1 and 2 are worth doing before this ships to anyone's phone;
3 is worth doing before it ships to bench boxes with populated stores; 4 and 5
are follow-ups.
