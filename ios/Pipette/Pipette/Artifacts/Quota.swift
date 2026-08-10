import Foundation

// Disk-cap accounting and the post-publish sweep. Mirrors
// `pipette-artifacts/src/quota.rs`, including its three-phase shape: classify
// (`survey`), decide (`plan`, pure), delete (`applySweep`). Keeping the decision free of
// the filesystem is what makes the eviction order testable without deleting anything.
//
// One store, not two: iOS compiles its engines into the app, so the crate's
// `EntryKind.runtime`, `RuntimeStorageKey` and `StoragePolicy.runtimesDir` have nothing
// to mirror here.

/// What an entry under the store root is, for accounting purposes — the crate's
/// `EntryKind`.
nonisolated enum EntryKind: Sendable, Equatable {
    /// No manifest this build can read: a `.staging` orphan, a manifest-less child, a
    /// legacy schema, a corrupt record, or a husk whose payload never landed. Free to
    /// drop, no policy involved.
    case garbage(reason: String)
    case model(key: ModelStorageKey, lastUsedAt: Date, fetchedAt: Date?)

    /// The eviction timestamp, or `nil` for garbage (which has no policy).
    var lastUsedAt: Date? {
        switch self {
        case .garbage: nil
        case let .model(_, lastUsedAt, _): lastUsedAt
        }
    }

    /// Publish time, breaking a `lastUsedAt` tie. Two entries resolved in the same
    /// second are otherwise ordered by directory-read order, which differs between runs.
    var fetchedAt: Date? {
        switch self {
        case .garbage: nil
        case let .model(_, _, fetchedAt): fetchedAt
        }
    }

    /// Sweep rank: garbage first, then models.
    var rank: Int {
        switch self {
        case .garbage: 0
        case .model: 1
        }
    }
}

/// One thing under the store root that occupies disk — the crate's `StorageEntry`.
nonisolated struct StorageEntry: Sendable, Equatable {
    /// What `applySweep` would remove.
    let path: URL
    /// How the entry reads in a report: the model's key, or the name of a garbage child.
    let label: String
    /// What removing this entry would free.
    let sizeBytes: Int64
    let kind: EntryKind
}

/// Everything under the store root, already in sweep order — the crate's `StorageSurvey`.
nonisolated struct StorageSurvey: Sendable {
    let entries: [StorageEntry]
    let usedBytes: Int64

    /// Order `entries` as the sweep would reclaim them and total their size. Garbage
    /// keeps discovery order; live entries sort least-recently-used first.
    init(entries: [StorageEntry]) {
        self.entries = entries.sorted {
            ($0.kind.rank, $0.kind.lastUsedAt ?? .distantPast, $0.kind.fetchedAt ?? .distantPast)
                < ($1.kind.rank, $1.kind.lastUsedAt ?? .distantPast, $1.kind.fetchedAt ?? .distantPast)
        }
        // Garbage is on disk until a sweep reclaims it, so it counts toward the total.
        self.usedBytes = entries.reduce(0) { $0 + $1.sizeBytes }
    }
}

/// What a sweep would drop to bring the store back under the cap — the crate's
/// `SweepPlan`.
nonisolated struct SweepPlan: Sendable {
    let evictions: [StorageEntry]
    let freedBytes: Int64
    /// What the store held when this was planned. Upstream keeps the survey beside the
    /// plan and reports `used - freed`; carrying it here says the same without a caller
    /// walking the store a second time to find out.
    let usedBytes: Int64
    /// Set when the candidates run out while still over — the caller warns and
    /// continues; a run never fails over disk bookkeeping.
    let stillOverByBytes: Int64?
}

/// What a sweep actually did — the crate's `SweepReport`.
nonisolated struct SweepReport: Sendable {
    var removed: [StorageEntry] = []
    /// Entries the sweep planned but could not remove, with the reason.
    var failed: [(entry: StorageEntry, reason: String)] = []
    var freedBytes: Int64 = 0
}

/// What a sweep must not reclaim. The storage layer knows only its own bytes, so the
/// run layer assembles this and passes it down.
nonisolated struct SweepPins: Sendable {
    /// Model entries, keyed the same way the store names their directories.
    var entries: Set<ModelStorageKey> = []
    /// HF repo slugs whose hub-cache snapshot is backing an in-flight MLX pull.
    var hubRepos: Set<String> = []

    /// Models a resumable job still needs — every cell of a running or paused job.
    /// A paused job is pinned because resuming it must not re-download.
    static func activeJobs(_ manifests: [JobManifest]) -> SweepPins {
        SweepPins(entries: Set(
            manifests
                .filter { $0.status == .running || $0.status == .paused }
                .flatMap(\.cells)
                .compactMap { ModelStorageKey.of($0.source) }
        ))
    }

    mutating func formUnion(_ other: SweepPins) {
        entries.formUnion(other.entries)
        hubRepos.formUnion(other.hubRepos)
    }

    func isPinned(_ kind: EntryKind) -> Bool {
        switch kind {
        // Garbage can never be pinned: it is unaccountable by definition.
        case .garbage: false
        case let .model(key, _, _): entries.contains(key)
        }
    }
}

extension FileStorage {
    /// Bytes the model store occupies right now. Garbage counts: it is on disk until a
    /// sweep reclaims it.
    func storageUsageBytes() -> Int64 {
        survey().usedBytes
    }

    /// Classify every child of the store root, in sweep order — the crate's `survey`.
    ///
    /// Never fails on one bad entry: an unreadable manifest is garbage, which is the
    /// whole point of the manifest-as-unit-of-accounting rule. That is also why this
    /// walks the root itself rather than going through the store's `list`, which skips
    /// what it cannot read — the resolve path keeps its tolerance, the accountant needs
    /// to see the bytes.
    ///
    /// `pins` decides one classification, not the ordering: mid-transfer is exactly when
    /// a payload is legitimately absent, so a pinned entry is never read as a husk.
    func survey(pinning pins: SweepPins = SweepPins()) -> StorageSurvey {
        StorageSurvey(entries: Self.children(of: modelsDir).map { child in
            let isDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
            let name = child.lastPathComponent
            guard isDir else { return Self.garbageEntry(child, reason: "not a directory") }
            guard name != Entry.stagingDirName else {
                return Self.garbageEntry(child, reason: "staging orphan")
            }
            guard let manifest = ModelManifest.forInstalledEntry(atDir: child) else {
                return Self.garbageEntry(child, reason: "no readable manifest")
            }
            guard let key = ModelStorageKey.of(manifest.declared) else {
                return Self.garbageEntry(child, reason: "manifest declares an unstorable model")
            }
            // A manifest alone isn't enough: a terminally failed second transfer (a
            // vision model's weights after its projector landed) leaves the manifest
            // beside a partial payload, which discovery skips — so the UI offers no way
            // to delete it while its install-time `lastUsedAt` sorts it behind every
            // model worth keeping. Husks are garbage, and the bytes come back.
            guard pins.entries.contains(key)
                    || ModelArtifactStore.bound(inEntryDir: child, declared: manifest.declared) != nil
            else {
                return Self.garbageEntry(child, reason: "manifest with no payload")
            }
            return StorageEntry(
                path: child, label: key.value,
                sizeBytes: manifest.blobsBytes ?? DiskUsage.bytes(at: child),
                kind: .model(key: key, lastUsedAt: manifest.lastUsedAt, fetchedAt: manifest.fetchedAt))
        })
    }

    /// What a sweep would reclaim right now — the crate's `quota::plan`, over this store's
    /// current survey and quota. Touches no disk, so `storage gc dry-run=1` can report it.
    func sweepPlan(pinning pins: SweepPins) -> SweepPlan {
        Self.plan(survey(pinning: pins), quotaBytes: storageQuotaBytes, pins: pins)
    }

    /// Reclaim disk until the store is at or under the quota, in the order
    /// `docs/storage-quota.md` fixes: garbage, then models least-recently-used.
    ///
    /// The one-call form over `survey` → `plan` → `applySweep`, plus the hub-cache phase
    /// that has no upstream analogue. Running out of unpinned entries while still over
    /// quota warns and returns normally — a run never fails over disk bookkeeping.
    @discardableResult
    func sweepToQuota(pinning pins: SweepPins) -> SweepReport {
        reclaim(sweepPlan(pinning: pins), pinning: pins)
    }

    /// Carry out a plan already taken — the crate's `quota::apply_sweep` — *and* the
    /// hub-cache phase, which is why this is not simply the static `applySweep`. Separate
    /// from `sweepToQuota` so a caller that planned in order to report one (`storage gc`)
    /// carries out *that* plan rather than re-surveying and executing a second one.
    ///
    /// The hub bytes never reach `freedBytes`: they sit outside the store root and are not
    /// quota-accounted, so counting them would inflate a figure measured against the cap.
    @discardableResult
    func reclaim(_ plan: SweepPlan, pinning pins: SweepPins) -> SweepReport {
        var report = Self.applySweep(plan)
        if let over = plan.stillOverByBytes {
            AppLog.storage.warning(
                "storage over quota by \(ByteFormat.fileSize(over)); every remaining entry is "
                    + "pinned or frees nothing")
        }
        // Bytes outside the store root that only the MLX pull path creates: HubApi keeps
        // a snapshot per repo, `MLXModelInstaller` moves only the model's own directory
        // out, and a cancelled pull leaves its partial behind. Not quota-accounted, but
        // this is the one place iOS accumulates multi-GB orphans.
        report.removed += reclaimHubRemnants(keeping: pins.hubRepos)
        return report
    }

    /// What would be dropped to bring `survey.usedBytes` to or under `quotaBytes`: every
    /// piece of garbage, then live entries least-recently-used first until it fits.
    /// Pure — no filesystem access, which is what makes the order testable.
    ///
    /// Garbage goes unconditionally, whether or not the store is over: it is
    /// unaccountable by definition and can never be pinned, so keeping it buys nothing —
    /// and a store stranded by a manifest version bump is typically *under* quota, where
    /// gating on the overage would leave a hand-deleted directory as the only recovery.
    static func plan(_ survey: StorageSurvey, quotaBytes: Int64, pins: SweepPins) -> SweepPlan {
        var freed: Int64 = 0
        var evictions: [StorageEntry] = []
        for entry in survey.entries {
            let overQuota = survey.usedBytes - freed > quotaBytes
            let isGarbage = entry.kind.rank == 0
            // Garbage sorts ahead of every live entry, so reaching a live one while the
            // store fits means nothing later can qualify either.
            if !overQuota && !isGarbage { break }
            guard !pins.isPinned(entry.kind) else { continue }
            freed += entry.sizeBytes
            evictions.append(entry)
        }
        let remaining = survey.usedBytes - freed
        return SweepPlan(evictions: evictions, freedBytes: freed, usedBytes: survey.usedBytes,
                         stillOverByBytes: remaining > quotaBytes ? remaining - quotaBytes : nil)
    }

    /// Delete the planned entries, reporting each one and skipping any it cannot remove —
    /// a failed unlink is bookkeeping, not a reason to fail a run. No delete is silent.
    static func applySweep(_ plan: SweepPlan) -> SweepReport {
        plan.evictions.reduce(into: SweepReport()) { report, entry in
            do {
                try FileManager.default.removeItem(at: entry.path)
                report.freedBytes += entry.sizeBytes
                report.removed.append(entry)
                AppLog.storage.info(
                    "Reclaimed \(entry.label) (\(ByteFormat.fileSize(entry.sizeBytes)))")
            } catch {
                report.failed.append((entry, error.localizedDescription))
                AppLog.storage.error(
                    "Failed to reclaim \(entry.label): \(error.localizedDescription)")
            }
        }
    }

    private static func garbageEntry(_ url: URL, reason: String) -> StorageEntry {
        StorageEntry(path: url, label: url.lastPathComponent,
                     sizeBytes: DiskUsage.bytes(at: url), kind: .garbage(reason: reason))
    }

    /// Snapshots stranded in the hub cache. These bytes sit outside the store root and
    /// are not quota-accounted, so they are reclaimed unconditionally rather than
    /// planned against the cap.
    private func reclaimHubRemnants(keeping liveRepos: Set<String>) -> [StorageEntry] {
        let root = hubCacheDir.appendingPathComponent("models", isDirectory: true)
        let stranded = Self.children(of: root).flatMap { orgDir in
            Self.children(of: orgDir).filter { repoDir in
                !liveRepos.contains("\(orgDir.lastPathComponent)/\(repoDir.lastPathComponent)")
            }
        }
        // A plan only to reuse the deletion loop — `applySweep` reads `evictions` alone,
        // so the byte fields are inert here rather than claiming an empty store.
        return Self.applySweep(SweepPlan(
            evictions: stranded.map { Self.garbageEntry($0, reason: "hub-cache remnant") },
            freedBytes: 0, usedBytes: 0, stillOverByBytes: nil)).removed
    }

    /// Direct children of `url`, hidden ones included — a dot-prefixed orphan is
    /// exactly what the garbage phase exists to catch.
    private static func children(of url: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: [.isDirectoryKey], options: [])) ?? []
    }
}
