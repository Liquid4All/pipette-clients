import Foundation

// Disk-cap accounting and the sweep, after `crates/pipette-artifacts/src/quota.rs`:
// classify (`survey`), decide (`plan`, pure, so the eviction order is testable), delete
// (`applySweep`). iOS compiles its engines into the app, so there are no runtime entries.

/// The crate's `EntryKind`, without runtimes.
nonisolated enum EntryKind: Sendable, Equatable {
    /// Anything the survey cannot count as a live entry: a `.staging` orphan, a file or
    /// hidden directory, a missing or unreadable manifest, a declared model with no
    /// storage key, or a husk whose payload never landed. Free to drop.
    case garbage(reason: String)
    case model(key: ModelStorageKey, lastUsedAt: Date, fetchedAt: Date)

    var lastUsedAt: Date? {
        switch self {
        case .garbage: nil
        case let .model(_, lastUsedAt, _): lastUsedAt
        }
    }

    /// Breaks a `lastUsedAt` tie, which directory-read order would settle differently
    /// between runs.
    var fetchedAt: Date? {
        switch self {
        case .garbage: nil
        case let .model(_, _, fetchedAt): fetchedAt
        }
    }

    var rank: Int {
        switch self {
        case .garbage: 0
        case .model: 1
        }
    }
}

nonisolated struct StorageEntry: Sendable, Equatable {
    let path: URL
    /// `Model.description` for a model, the store-relative path for garbage.
    let label: String
    /// Garbage measures the whole directory; a live entry measures `blobs/` alone, so it
    /// under-reports by one manifest.
    let sizeBytes: Int64
    let kind: EntryKind
}

/// Already in sweep order.
nonisolated struct StorageSurvey: Sendable {
    let entries: [StorageEntry]
    /// Garbage included: it is on disk until a sweep reclaims it.
    let usedBytes: Int64

    init(entries: [StorageEntry]) {
        self.entries = entries.sorted {
            ($0.kind.rank, $0.kind.lastUsedAt ?? .distantPast, $0.kind.fetchedAt ?? .distantPast)
                < ($1.kind.rank, $1.kind.lastUsedAt ?? .distantPast, $1.kind.fetchedAt ?? .distantPast)
        }
        self.usedBytes = entries.reduce(0) { total, entry in
            let (sum, overflow) = total.addingReportingOverflow(entry.sizeBytes)
            return overflow ? .max : sum
        }
    }
}

nonisolated struct SweepPlan: Sendable {
    let evictions: [StorageEntry]
    let freedBytes: Int64
    /// Set when the candidates run out while still over; the caller warns and continues.
    let stillOverByBytes: Int64?
}

nonisolated struct SweepReport: Sendable {
    var removed: [StorageEntry] = []
    var failed: [(entry: StorageEntry, reason: String)] = []
    var freedBytes: Int64 = 0
}

/// The never-evict set, assembled by the run layer: the storage layer knows only bytes.
nonisolated struct SweepPins: Sendable, Equatable {
    var models: Set<ModelStorageKey> = []
    /// `org/repo` slugs whose hub-cache snapshot backs an in-flight MLX pull (iOS only).
    var hubRepos: Set<String> = []
    /// `.staging` children a live transfer is writing into. Wreckage and live bytes look
    /// identical on disk; only the coordinator knows which is which.
    var stagingAttempts: Set<String> = []

    /// For a caller that picks manifests before decoding them.
    static let activeStatuses: Set<JobStatus> = [.running, .paused]

    /// Every cell of a running or paused job: resuming must not re-download.
    static func activeJobs(_ manifests: [JobManifest]) -> SweepPins {
        manifests
            .filter { activeStatuses.contains($0.status) }
            .flatMap(\.cells)
            .reduce(SweepPins()) { $0.withModel($1.source) }
    }

    func withModel(_ model: Model) -> SweepPins {
        var pins = self
        if let key = ModelStorageKey.of(model) { pins.models.insert(key) }
        return pins
    }

    /// The crate's `merge`. Does not merge `stagingAttempts`.
    mutating func merge(_ other: SweepPins) {
        models.formUnion(other.models)
        hubRepos.formUnion(other.hubRepos)
    }
}

/// Every piece of garbage, then unpinned models least-recently-used first until the
/// store fits. Garbage goes even under quota: a store stranded by a manifest version bump
/// is usually under quota, and would otherwise need deleting by hand.
nonisolated func plan(_ survey: StorageSurvey, quotaBytes: Int64, pins: SweepPins) -> SweepPlan {
    var freed: Int64 = 0
    var evictions: [StorageEntry] = []
    for entry in survey.entries {
        let overQuota = survey.usedBytes - freed > quotaBytes
        let isGarbage = entry.kind.rank == 0
        // Garbage sorts first, so a live entry reached while fitting ends the plan.
        if !overQuota && !isGarbage { break }
        guard isCandidate(entry, pins) else { continue }
        freed += entry.sizeBytes
        evictions.append(entry)
    }
    let remaining = survey.usedBytes - freed
    return SweepPlan(evictions: evictions, freedBytes: freed,
                     stillOverByBytes: remaining > quotaBytes ? remaining - quotaBytes : nil)
}

/// A failed unlink is logged and skipped, never fatal to a run. No delete is silent.
nonisolated func applySweep(_ plan: SweepPlan) -> SweepReport {
    plan.evictions.reduce(into: SweepReport()) { report, entry in
        do {
            try removeEntry(entry.path)
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

private nonisolated func isCandidate(_ entry: StorageEntry, _ pins: SweepPins) -> Bool {
    switch entry.kind {
    case .garbage: true
    case let .model(key, _, _): !pins.models.contains(key)
    }
}

/// A path already gone is what the sweep wanted, not a failure.
private nonisolated func removeEntry(_ url: URL) throws {
    var info = stat()
    guard lstat(url.path, &info) == 0 else {
        let code = errno
        if code == ENOENT { return }
        throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
    }
    try FileManager.default.removeItem(at: url)
}

nonisolated extension PipetteWorkspace {
    func storageUsageBytes() -> Int64 {
        survey().usedBytes
    }

    /// Never fails on one bad entry: an unreadable manifest is garbage. So this walks the
    /// root itself rather than using the store's strict `list`.
    ///
    /// A pinned entry is never read as a husk, since mid-transfer its payload is
    /// legitimately absent.
    func survey(pinning pins: SweepPins = SweepPins()) -> StorageSurvey {
        let root = modelsDir
        return StorageSurvey(entries: Self.storeChildren(root).flatMap { child -> [StorageEntry] in
            guard child.lastPathComponent != Entry.stagingDirName else {
                return Self.stagingOrphans(child, pins: pins)
            }
            return [Self.nonEntry(child) ?? Self.modelEntry(child, under: root, pins: pins)]
        })
    }

    /// `survey`, `plan` and `reclaim` in one call, in the order `docs/storage-quota.md`
    /// fixes. Running out of unpinned entries while over quota only warns.
    @discardableResult
    func sweepToQuota(pinning pins: SweepPins) -> SweepReport {
        reclaim(plan(survey(pinning: pins), quotaBytes: storageQuotaBytes, pins: pins), pinning: pins)
    }

    /// `applySweep` plus the hub-cache phase. Separate from `sweepToQuota` so a caller that
    /// reported a plan (`storage gc`) carries out that plan, not a fresh one. Hub bytes
    /// never reach `freedBytes`: they are outside the quota-accounted root.
    @discardableResult
    func reclaim(_ plan: SweepPlan, pinning pins: SweepPins) -> SweepReport {
        var report = applySweep(plan)
        if let over = plan.stillOverByBytes {
            AppLog.storage.warning(
                "storage over quota by \(ByteFormat.fileSize(over)); every remaining entry is "
                    + "pinned or frees nothing")
        }
        // HubApi keeps a snapshot per repo, the installer moves out only the model's own
        // directory, and a cancelled pull leaves its partial: iOS's one source of
        // multi-GB orphans.
        report.removed += reclaimHubRemnants(keeping: pins.hubRepos)
        report.removed += reclaimSharedHfCache()
        return report
    }

    /// One garbage entry per crashed fetch. `isCandidate` ignores pins for garbage, so a
    /// child a live transfer owns is skipped here: `stagingAttempts` names scratch
    /// directories, and `models` covers attempts named after their entry key.
    private static func stagingOrphans(_ staging: URL, pins: SweepPins) -> [StorageEntry] {
        storeChildren(staging).compactMap { child in
            let name = child.lastPathComponent
            guard !pins.stagingAttempts.contains(name),
                  !pins.models.contains(ModelStorageKey(value: name))
            else { return nil }
            return garbage(child, label: "\(Entry.stagingDirName)/\(name)",
                                reason: "orphaned staging dir")
        }
    }

    /// A file, a hidden directory, or a directory with no manifest; nil otherwise.
    private static func nonEntry(_ child: URL) -> StorageEntry? {
        let isDir = (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
        if !isDir || child.lastPathComponent.hasPrefix(".") {
            return garbage(child, reason: "not a store entry")
        }
        guard FileManager.default.fileExists(atPath: ModelManifest.manifestURL(inEntryDir: child).path) else {
            return garbage(child, reason: "no manifest")
        }
        return nil
    }

    /// Garbage unless the manifest reads and, pins aside, its payload binds.
    private static func modelEntry(_ child: URL, under root: URL, pins: SweepPins) -> StorageEntry {
        let manifest: ModelManifest
        do {
            manifest = try ModelArtifactStore.readManifest(at: ModelManifest.manifestURL(inEntryDir: child))
        } catch {
            return garbage(child, reason: "unreadable manifest: \(error.localizedDescription)")
        }
        guard let key = ModelStorageKey.of(manifest.declared) else {
            return garbage(child, reason: "declared model has no storage key")
        }
        // A vision model whose second transfer failed leaves a manifest beside half a
        // payload. Discovery skips it, so the UI cannot delete it; reclaim it as garbage.
        guard pins.models.contains(key) || ModelArtifactStore.bound(manifest, under: root) != nil else {
            return garbage(child, reason: "manifest without its payload")
        }
        // The crate's `payload_bytes`: the recorded size, else a walk of `blobs/`.
        let blobs = child.appendingPathComponent(Entry.blobsDirName, isDirectory: true)
        return StorageEntry(
            path: child, label: manifest.declared.description,
            sizeBytes: manifest.blobsBytes ?? entrySizeBytes(at: blobs),
            kind: .model(key: key, lastUsedAt: manifest.lastUsedAt, fetchedAt: manifest.fetchedAt))
    }

    private static func garbage(_ url: URL, label: String? = nil, reason: String) -> StorageEntry {
        StorageEntry(path: url, label: label ?? url.lastPathComponent,
                     sizeBytes: entrySizeBytes(at: url), kind: .garbage(reason: reason))
    }

    /// Hub-cache bytes no live grant covers, for admission (`survey` never sees them). A
    /// repo backing an in-flight pull is excluded: its reservation already counts it.
    func hubBytes(excluding liveRepos: Set<String>) -> Int64 {
        let root = hubCacheDir.appendingPathComponent("models", isDirectory: true)
        return Self.storeChildren(root).reduce(into: 0) { total, orgDir in
            for repoDir in Self.storeChildren(orgDir)
            where !liveRepos.contains("\(orgDir.lastPathComponent)/\(repoDir.lastPathComponent)") {
                total += entrySizeBytes(at: repoDir)
            }
        }
    }

    /// Not quota-accounted, so reclaimed unconditionally rather than planned.
    private func reclaimHubRemnants(keeping liveRepos: Set<String>) -> [StorageEntry] {
        let root = hubCacheDir.appendingPathComponent("models", isDirectory: true)
        let stranded = Self.storeChildren(root).flatMap { orgDir in
            Self.storeChildren(orgDir).compactMap { repoDir -> (URL, String)? in
                let repo = "\(orgDir.lastPathComponent)/\(repoDir.lastPathComponent)"
                return liveRepos.contains(repo) ? nil : (repoDir, repo)
            }
        }
        // Only to reuse the deletion loop: `applySweep` reads `evictions` alone.
        return applySweep(SweepPlan(
            evictions: stranded.map { Self.garbage($0, label: $1, reason: "hub-cache remnant") },
            freedBytes: 0, stillOverByBytes: nil)).removed
    }

    /// No pull writes the shared HF cache any more and the app reads only from the store,
    /// so whatever is left there is a duplicate and goes whole, pins or not.
    private func reclaimSharedHfCache() -> [StorageEntry] {
        let dir = sharedHfCacheDir
        guard FileManager.default.fileExists(atPath: dir.path) else { return [] }
        return applySweep(SweepPlan(
            evictions: [Self.garbage(dir, reason: "shared HF cache")],
            freedBytes: 0, stillOverByBytes: nil)).removed
    }

    /// Hidden children included: a dot-prefixed orphan is what the garbage phase catches.
    private static func storeChildren(_ url: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: [.isDirectoryKey], options: [])) ?? []
    }
}
