import Foundation
import Testing

@testable import Pipette

/// The accountant: garbage first, then models least-recently-used, stopping the
/// instant the store is back under the cap. Each test injects its own temporary
/// `FileStorage`, so the suite carries no shared global and runs in parallel.
@MainActor struct StorageQuotaSweepTests {
    private static let epoch = Date(timeIntervalSince1970: 1_700_000_000)

    private func at(_ hoursLater: Double) -> Date {
        Self.epoch.addingTimeInterval(hoursLater * 3600)
    }

    private func entryNames(_ storage: FileStorage) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: storage.modelsDir.path)) ?? [])
    }

    /// What `storage gc dry-run=1` reports: the plan names what would go, and the store is
    /// untouched until a sweep is actually run. Same pins both times, so the second call
    /// reclaims exactly what the first predicted.
    @Test func aPlanReportsWithoutReclaiming() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try installEntry(storage, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                         payloadBytes: 64_000, lastUsedAt: at(0))
        try installEntry(storage, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                         payloadBytes: 64_000, lastUsedAt: at(1))
        try storage.setStorageQuotaBytes(70_000)

        let plan = storage.sweepPlan(pinning: SweepPins())
        #expect(!plan.evictions.isEmpty)
        #expect(entryNames(storage).count == 2, "planning must not delete anything")

        // A prefix, not the whole list: the sweep appends any hub-cache remnants it finds,
        // which are outside the store and so never planned against the cap.
        let removed = storage.sweepToQuota(pinning: SweepPins()).removed
        #expect(removed.prefix(plan.evictions.count).map(\.path) == plan.evictions.map(\.path))
    }

    @Test func underQuotaEvictsNothing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try installEntry(storage, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"), payloadBytes: 64_000)
        try storage.setStorageQuotaBytes(1 << 30)

        #expect(storage.sweepToQuota(pinning: SweepPins()).removed.isEmpty)
        #expect(entryNames(storage).count == 1)
    }

    /// Garbage is free to drop and goes first, so a store that is over quota only
    /// because of unreadable bytes never touches a live entry.
    @Test func reclaimsGarbageBeforeAnyLiveEntry() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let fm = FileManager.default
        let live = try installEntry(storage, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                    payloadBytes: 64_000)

        let manifestLess = storage.modelsDir.appendingPathComponent("org__b__b.gguf", isDirectory: true)
        try fm.createDirectory(at: manifestLess, withIntermediateDirectories: true)
        try Data(repeating: 0x62, count: 64_000).write(to: manifestLess.appendingPathComponent("b.gguf"))

        let strayFile = storage.modelsDir.appendingPathComponent("c.gguf")
        try Data(repeating: 0x63, count: 64_000).write(to: strayFile)

        let staging = storage.modelsDir.appendingPathComponent(".staging", isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try Data(repeating: 0x64, count: 64_000).write(to: staging.appendingPathComponent("partial.part"))

        let stranded = try installEntry(storage, try ggufTextSpec("org/e-GGUF", "e-Q4_0.gguf"))
        try Data(#"{"manifest_version":1,"declared":{"type":"mlx","source":"huggingface","org":"x","repo_name":"y"}}"#.utf8)
            .write(to: ModelManifest.manifestURL(inEntryDir: stranded))

        try storage.setStorageQuotaBytes(DiskUsage.bytes(at: live))
        let report = storage.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.allSatisfy { if case .garbage = $0.kind { true } else { false } })
        #expect(entryNames(storage) == [live.lastPathComponent])
    }

    /// A vision entry whose weights transfer failed terminally: the projector and the
    /// manifest are on disk, the payload the manifest names is not. Discovery skips
    /// it, so nothing in the app could delete it — the sweep has to, at any usage,
    /// or those bytes hold quota forever and push real models out ahead of them.
    @Test func aHuskWhosePayloadNeverLandedIsGarbageEvenUnderQuota() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
        let entry = try installEntry(storage, spec, payloadBytes: 64_000)
        let weights = try #require(storage.modelStore.blobsDir(for: spec))
            .appendingPathComponent("v-Q4_0.gguf")
        try FileManager.default.removeItem(at: weights)
        try storage.setStorageQuotaBytes(1 << 30)

        let report = storage.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.count == 1)
        #expect({ if case .garbage = report.removed[0].kind { true } else { false } }())
        #expect(entryNames(storage).isEmpty)
        #expect(!FileManager.default.fileExists(atPath: entry.path))
    }

    /// The same shape mid-transfer — projector landed, weights still coming — is a
    /// live download, and the pin is what tells the two apart.
    @Test func aPinnedEntryOutlivesItsNotYetArrivedPayload() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
        let entry = try installEntry(storage, spec, payloadBytes: 64_000)
        try FileManager.default.removeItem(
            at: try #require(storage.modelStore.blobsDir(for: spec))
                .appendingPathComponent("v-Q4_0.gguf"))
        try storage.setStorageQuotaBytes(1)

        let pins = SweepPins(entries: [try #require(ModelStorageKey.of(spec))])
        let report = storage.sweepToQuota(pinning: pins)

        #expect(report.removed.isEmpty)
        #expect(entryNames(storage) == [entry.lastPathComponent])
    }

    @Test func evictsTheLeastRecentlyUsedAndStopsAtTheCap() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let oldest = try installEntry(storage, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(0))
        let middle = try installEntry(storage, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(1))
        let newest = try installEntry(storage, try ggufTextSpec("org/c-GGUF", "c-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(2))
        let total = [oldest, middle, newest].reduce(0) { $0 + DiskUsage.bytes(at: $1) }
        try storage.setStorageQuotaBytes(total - DiskUsage.bytes(at: oldest))

        let report = storage.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.map(\.path) == [oldest])
        #expect({ if case .model = report.removed.first?.kind { true } else { false } }())
        #expect(report.removed.first?.sizeBytes ?? 0 > 0)
        // The next-oldest survives: the sweep stops the instant it is under the cap.
        #expect(entryNames(storage) == [middle.lastPathComponent, newest.lastPathComponent])
    }

    @Test func neverEvictsAPinnedEntryEvenWhenItIsOldest() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let pinnedSpec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let pinned = try installEntry(storage, pinnedSpec, payloadBytes: 64_000, lastUsedAt: at(0))
        let newer = try installEntry(storage, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                     payloadBytes: 64_000, lastUsedAt: at(1))
        let total = DiskUsage.bytes(at: pinned) + DiskUsage.bytes(at: newer)
        try storage.setStorageQuotaBytes(total - DiskUsage.bytes(at: pinned))

        let pins = SweepPins(entries: [try #require(ModelStorageKey.of(pinnedSpec))])
        let report = storage.sweepToQuota(pinning: pins)

        #expect(report.removed.map(\.path) == [newer])
        #expect(entryNames(storage) == [pinned.lastPathComponent])
    }

    /// Over quota with nothing left to evict warns and returns normally — a run never
    /// fails over disk bookkeeping.
    @Test func overQuotaWithEverythingPinnedReturnsWithoutEvicting() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let specs = [try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                     try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")]
        for spec in specs { try installEntry(storage, spec, payloadBytes: 64_000) }
        try storage.setStorageQuotaBytes(1)

        let pins = SweepPins(entries: Set(specs.compactMap(ModelStorageKey.of)))
        #expect(storage.sweepToQuota(pinning: pins).removed.isEmpty)
        #expect(entryNames(storage).count == 2)
    }

    /// A model a running job needs is pinned through its job manifest, so a sweep
    /// mid-run cannot pull the weights out from under it.
    @Test func aRunningJobsModelIsPinnedThroughItsManifest() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let running = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let idle = try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")
        let runningEntry = try installEntry(storage, running, payloadBytes: 64_000, lastUsedAt: at(0))
        try installEntry(storage, idle, payloadBytes: 64_000, lastUsedAt: at(1))
        storage.saveJobManifest(jobManifest(status: .running, source: running))
        try storage.setStorageQuotaBytes(DiskUsage.bytes(at: runningEntry))

        let pins = SweepPins.activeJobs(storage.loadAllJobManifests())
        _ = storage.sweepToQuota(pinning: pins)

        #expect(entryNames(storage) == [runningEntry.lastPathComponent])
    }

    /// A vision model's weights and projector live in one entry, so they are reclaimed
    /// together — no orphan projector survives its model.
    @Test func aVisionEntryEvictsBothOfItsFiles() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let entry = try installEntry(
            storage, try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf"),
            payloadBytes: 64_000)
        try storage.setStorageQuotaBytes(1)

        let report = storage.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.map(\.path) == [entry])
        #expect(!FileManager.default.fileExists(atPath: entry.path))
    }

    /// Hub-cache snapshots the installer left behind are reclaimed; the one backing an
    /// in-flight pull is not.
    @Test func reclaimsHubRemnantsExceptAnInFlightPull() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let hubModels = storage.hubCacheDir.appendingPathComponent("models/org", isDirectory: true)
        let stale = hubModels.appendingPathComponent("stale", isDirectory: true)
        let inFlight = hubModels.appendingPathComponent("in-flight", isDirectory: true)
        for dir in [stale, inFlight] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(repeating: 0x68, count: 64_000).write(to: dir.appendingPathComponent("shard.safetensors"))
        }

        let report = storage.sweepToQuota(pinning: SweepPins(hubRepos: ["org/in-flight"]))

        #expect(report.removed.map(\.path) == [stale])
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(FileManager.default.fileExists(atPath: inFlight.path))
    }

    /// The gap a quota-shaped guard leaves: hub-cache orphans sit outside the store root,
    /// so a store comfortably under quota plans no evictions at all. `gc` must still run
    /// the reclaim, or the one place iOS accumulates multi-GB orphans is exactly the case
    /// it declines to clean. Upstream can return early on an empty plan because its survey
    /// already covers everything it reclaims; ours does not.
    @Test func gcReclaimsOrphansWithAnEmptyQuotaPlan() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(1 << 30)
        let stale = storage.hubCacheDir.appendingPathComponent("models/org/stale", isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        try Data(repeating: 0x68, count: 64_000)
            .write(to: stale.appendingPathComponent("shard.safetensors"))

        #expect(storage.sweepPlan(pinning: SweepPins()).evictions.isEmpty, "under quota, nothing planned")
        _ = await StorageCommands.gc(dryRun: false, storage: storage, pins: SweepPins())

        #expect(!FileManager.default.fileExists(atPath: stale.path))
    }

    /// And a dry run reclaims nothing, as upstream asserts of its own.
    @Test func gcDryRunReclaimsNothing() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(1 << 30)
        let stale = storage.hubCacheDir.appendingPathComponent("models/org/stale", isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        try Data(repeating: 0x68, count: 64_000)
            .write(to: stale.appendingPathComponent("shard.safetensors"))

        _ = await StorageCommands.gc(dryRun: true, storage: storage, pins: SweepPins())

        #expect(FileManager.default.fileExists(atPath: stale.path))
    }

    /// Setting the limit is not a reclaim. Lowering it below current usage must leave
    /// every model on disk — otherwise the Settings control would be dangerous to touch
    /// — and the explicit sweep is what actually frees the bytes.
    @Test func loweringTheLimitEvictsNothingByItself() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let first = try installEntry(storage, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                     payloadBytes: 64_000)
        let second = try installEntry(storage, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                      payloadBytes: 64_000)
        let usedBefore = storage.storageUsageBytes()

        try storage.setStorageQuotaBytes(1)

        #expect(FileManager.default.fileExists(atPath: first.path))
        #expect(FileManager.default.fileExists(atPath: second.path))
        #expect(storage.storageUsageBytes() == usedBefore)

        #expect(storage.sweepToQuota(pinning: SweepPins()).removed.count == 2)
        #expect(entryNames(storage).isEmpty)
    }

    private func jobManifest(status: JobStatus, source: Model) -> JobManifest {
        JobManifest(
            jobId: JobId("job-\(UUID().uuidString)"),
            createdAt: "2026-06-05T18:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                JobCell(
                    cellId: "cell-a",
                    benchmarkId: "prefill",
                    benchmarkType: .prefillThroughput,
                    runStatus: .pending,
                    serverJobId: nil,
                    errorMessage: nil,
                    source: source
                )
            ],
            status: status,
            contributeResults: true,
            title: "Quota job"
        )
    }
}
