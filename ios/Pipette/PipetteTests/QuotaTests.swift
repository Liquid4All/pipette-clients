import Foundation
import Testing

@testable import Pipette

/// The accountant: garbage first, then models least-recently-used, stopping the
/// instant the store is back under the cap.
@MainActor struct QuotaTests {
    private static let epoch = Date(timeIntervalSince1970: 1_700_000_000)

    private func at(_ hoursLater: Double) -> Date {
        Self.epoch.addingTimeInterval(hoursLater * 3600)
    }

    private func entryNames(_ ws: PipetteWorkspace) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: ws.modelsDir.path)) ?? [])
    }

    /// What `storage gc dry-run=1` reports. Same pins both times, so the sweep reclaims
    /// exactly what the plan predicted.
    @Test func aPlanReportsWithoutReclaiming() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                         payloadBytes: 64_000, lastUsedAt: at(0))
        try installEntry(ws, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                         payloadBytes: 64_000, lastUsedAt: at(1))
        try ws.setStorageQuotaBytes(70_000)

        let plan = Pipette.plan(ws.survey(), quotaBytes: ws.storageQuotaBytes, pins: SweepPins())
        #expect(!plan.evictions.isEmpty)
        #expect(entryNames(ws).count == 2, "planning must not delete anything")

        // A prefix, not the whole list: the sweep appends any hub-cache remnants it finds,
        // which are outside the store and so never planned against the cap.
        let removed = ws.sweepToQuota(pinning: SweepPins()).removed
        #expect(removed.prefix(plan.evictions.count).map(\.path) == plan.evictions.map(\.path))
    }

    @Test func underQuotaEvictsNothing() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"), payloadBytes: 64_000)
        try ws.setStorageQuotaBytes(1 << 30)

        #expect(ws.sweepToQuota(pinning: SweepPins()).removed.isEmpty)
        #expect(entryNames(ws).count == 1)
    }

    @Test func reclaimsGarbageBeforeAnyLiveEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let fm = FileManager.default
        let live = try installEntry(ws, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                    payloadBytes: 64_000)

        let manifestLess = ws.modelsDir.appendingPathComponent("org__b__b.gguf", isDirectory: true)
        try fm.createDirectory(at: manifestLess, withIntermediateDirectories: true)
        try Data(repeating: 0x62, count: 64_000).write(to: manifestLess.appendingPathComponent("b.gguf"))

        let strayFile = ws.modelsDir.appendingPathComponent("c.gguf")
        try Data(repeating: 0x63, count: 64_000).write(to: strayFile)

        let staging = ws.modelsDir.appendingPathComponent(".staging", isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try Data(repeating: 0x64, count: 64_000).write(to: staging.appendingPathComponent("partial.part"))

        let stranded = try installEntry(ws, try ggufTextSpec("org/e-GGUF", "e-Q4_0.gguf"))
        try Data(#"{"manifest_version":1,"declared":{"type":"mlx","source":"huggingface","org":"x","repo_name":"y"}}"#.utf8)
            .write(to: ModelManifest.manifestURL(inEntryDir: stranded))

        try ws.setStorageQuotaBytes(entrySizeBytes(at: live))
        let report = ws.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.allSatisfy { if case .garbage = $0.kind { true } else { false } })
        // The scratch directory stays: each orphan in it is its own entry, as upstream.
        #expect(entryNames(ws) == [live.lastPathComponent, Entry.stagingDirName])
    }

    /// A vision entry whose weights transfer failed terminally. Discovery skips it, so
    /// nothing else in the app could delete it: the sweep has to, at any usage, or those
    /// bytes hold quota forever.
    @Test func surveyTreatsAManifestWithoutItsPayloadAsGarbage() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
        let entry = try installEntry(ws, spec, payloadBytes: 64_000)
        let weights = try #require(ws.models.blobsDir(for: spec))
            .appendingPathComponent("v-Q4_0.gguf")
        try FileManager.default.removeItem(at: weights)
        try ws.setStorageQuotaBytes(1 << 30)

        let report = ws.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.count == 1)
        #expect({ if case .garbage = report.removed[0].kind { true } else { false } }())
        #expect(entryNames(ws).isEmpty)
        #expect(!FileManager.default.fileExists(atPath: entry.path))
    }

    /// The same shape mid-transfer is a live download, and the pin is what tells the two
    /// apart.
    @Test func aPinnedEntryOutlivesItsNotYetArrivedPayload() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v.gguf")
        let entry = try installEntry(ws, spec, payloadBytes: 64_000)
        try FileManager.default.removeItem(
            at: try #require(ws.models.blobsDir(for: spec))
                .appendingPathComponent("v-Q4_0.gguf"))
        try ws.setStorageQuotaBytes(1)

        let pins = SweepPins().withModel(spec)
        let report = ws.sweepToQuota(pinning: pins)

        #expect(report.removed.isEmpty)
        #expect(entryNames(ws) == [entry.lastPathComponent])
    }

    @Test func evictsTheLeastRecentlyUsedAndStopsAtTheCap() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let oldest = try installEntry(ws, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(0))
        let middle = try installEntry(ws, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(1))
        let newest = try installEntry(ws, try ggufTextSpec("org/c-GGUF", "c-Q4_0.gguf"),
                                      payloadBytes: 64_000, lastUsedAt: at(2))
        let total = [oldest, middle, newest].reduce(0) { $0 + entrySizeBytes(at: $1) }
        try ws.setStorageQuotaBytes(total - entrySizeBytes(at: oldest))

        let report = ws.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.map(\.path) == [oldest])
        #expect({ if case .model = report.removed.first?.kind { true } else { false } }())
        #expect(report.removed.first?.sizeBytes ?? 0 > 0)
        #expect(entryNames(ws) == [middle.lastPathComponent, newest.lastPathComponent])
    }

    @Test func planNeverEvictsAPinnedModel() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let pinnedSpec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let pinned = try installEntry(ws, pinnedSpec, payloadBytes: 64_000, lastUsedAt: at(0))
        let newer = try installEntry(ws, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                     payloadBytes: 64_000, lastUsedAt: at(1))
        let total = entrySizeBytes(at: pinned) + entrySizeBytes(at: newer)
        try ws.setStorageQuotaBytes(total - entrySizeBytes(at: pinned))

        let pins = SweepPins().withModel(pinnedSpec)
        let report = ws.sweepToQuota(pinning: pins)

        #expect(report.removed.map(\.path) == [newer])
        #expect(entryNames(ws) == [pinned.lastPathComponent])
    }

    /// A run never fails over disk bookkeeping.
    @Test func planReportsStillOverWhenOnlyPinnedEntriesRemain() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let specs = [try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                     try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")]
        for spec in specs { try installEntry(ws, spec, payloadBytes: 64_000) }
        try ws.setStorageQuotaBytes(1)

        let pins = specs.reduce(SweepPins()) { $0.withModel($1) }
        #expect(ws.sweepToQuota(pinning: pins).removed.isEmpty)
        #expect(entryNames(ws).count == 2)
    }

    @Test func aRunningJobsModelIsPinnedThroughItsManifest() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let running = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let idle = try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")
        let runningEntry = try installEntry(ws, running, payloadBytes: 64_000, lastUsedAt: at(0))
        try installEntry(ws, idle, payloadBytes: 64_000, lastUsedAt: at(1))
        ws.saveJobManifest(jobManifest(status: .running, source: running))
        try ws.setStorageQuotaBytes(entrySizeBytes(at: runningEntry))

        let pins = SweepPins.activeJobs(ws.loadAllJobManifests())
        _ = ws.sweepToQuota(pinning: pins)

        #expect(entryNames(ws) == [runningEntry.lastPathComponent])
    }

    @Test func aVisionEntryEvictsBothOfItsFiles() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let entry = try installEntry(
            ws, try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-F16.gguf"),
            payloadBytes: 64_000)
        try ws.setStorageQuotaBytes(1)

        let report = ws.sweepToQuota(pinning: SweepPins())

        #expect(report.removed.map(\.path) == [entry])
        #expect(!FileManager.default.fileExists(atPath: entry.path))
    }

    @Test func reclaimsHubRemnantsExceptAnInFlightPull() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let hubModels = ws.hubCacheDir.appendingPathComponent("models/org", isDirectory: true)
        let stale = hubModels.appendingPathComponent("stale", isDirectory: true)
        let inFlight = hubModels.appendingPathComponent("in-flight", isDirectory: true)
        for dir in [stale, inFlight] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(repeating: 0x68, count: 64_000).write(to: dir.appendingPathComponent("shard.safetensors"))
        }

        let report = ws.sweepToQuota(pinning: SweepPins(hubRepos: ["org/in-flight"]))

        #expect(report.removed.map(\.path) == [stale])
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(FileManager.default.fileExists(atPath: inFlight.path))
    }

    /// The shared HF cache earlier MLX pulls filled holds a second copy of every model, so
    /// it goes on any sweep, even under quota with the repo pinned, or evicting the store
    /// entry would free nothing.
    @Test func reclaimsTheSharedHfCache() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)
        let blobs = ws.sharedHfCacheDir
            .appendingPathComponent("models--org--repo/blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try Data(repeating: 0x68, count: 64_000).write(to: blobs.appendingPathComponent("etag"))

        let report = ws.sweepToQuota(pinning: SweepPins(hubRepos: ["org/repo"]))

        #expect(report.removed.map(\.path) == [ws.sharedHfCacheDir])
        #expect(!FileManager.default.fileExists(atPath: ws.sharedHfCacheDir.path))
        #expect(ws.sweepToQuota(pinning: SweepPins()).removed.isEmpty,
                "an absent cache is not a failed reclaim")
    }

    /// Hub-cache orphans sit outside the store root, so a store under quota plans nothing,
    /// yet that is where iOS accumulates multi-GB orphans. Upstream can return early on an
    /// empty plan because its survey covers everything it reclaims; ours does not.
    @Test func gcReclaimsOrphansWithAnEmptyQuotaPlan() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)
        let stale = ws.hubCacheDir.appendingPathComponent("models/org/stale", isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        try Data(repeating: 0x68, count: 64_000)
            .write(to: stale.appendingPathComponent("shard.safetensors"))

        let planned = Pipette.plan(ws.survey(), quotaBytes: ws.storageQuotaBytes,
                                   pins: SweepPins())
        #expect(planned.evictions.isEmpty, "under quota, nothing planned")
        _ = await StorageCommands.gc(dryRun: false, ws: ws, pins: SweepPins())

        #expect(!FileManager.default.fileExists(atPath: stale.path))
    }

    @Test func gcDryRunReclaimsNothing() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)
        let stale = ws.hubCacheDir.appendingPathComponent("models/org/stale", isDirectory: true)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        try Data(repeating: 0x68, count: 64_000)
            .write(to: stale.appendingPathComponent("shard.safetensors"))

        _ = await StorageCommands.gc(dryRun: true, ws: ws, pins: SweepPins())

        #expect(FileManager.default.fileExists(atPath: stale.path))
    }

    /// Only the explicit sweep frees bytes, or the Settings control would be dangerous to
    /// touch.
    @Test func loweringTheLimitEvictsNothingByItself() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let first = try installEntry(ws, try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"),
                                     payloadBytes: 64_000)
        let second = try installEntry(ws, try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf"),
                                      payloadBytes: 64_000)
        let usedBefore = ws.storageUsageBytes()

        try ws.setStorageQuotaBytes(1)

        #expect(FileManager.default.fileExists(atPath: first.path))
        #expect(FileManager.default.fileExists(atPath: second.path))
        #expect(ws.storageUsageBytes() == usedBefore)

        #expect(ws.sweepToQuota(pinning: SweepPins()).removed.count == 2)
        #expect(entryNames(ws).isEmpty)
    }

    // MARK: - survey / plan / applySweep, as the crate's quota.rs tests them

    @Test func surveyIsEmptyForMissingRoots() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let survey = ws.survey()

        #expect(survey.entries.isEmpty)
        #expect(survey.usedBytes == 0)
    }

    @Test func surveyClassifiesEverythingWithoutAReadableManifestAsGarbage() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let fm = FileManager.default
        let root = ws.modelsDir
        let live = try ggufTextSpec("meta/llama", "Q4.gguf")
        try installEntry(ws, live)
        try fm.createDirectory(at: root.appendingPathComponent(".staging/llama.staged-1", isDirectory: true),
                               withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appendingPathComponent("no-manifest", isDirectory: true),
                               withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appendingPathComponent(".hidden", isDirectory: true),
                               withIntermediateDirectories: true)
        try Data("x".utf8).write(to: root.appendingPathComponent("stray.txt"))
        let corrupt = root.appendingPathComponent("meta__zephyr__Q4.gguf", isDirectory: true)
        try fm.createDirectory(at: corrupt, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: ModelManifest.manifestURL(inEntryDir: corrupt))

        let survey = ws.survey()

        let reasons = Dictionary(uniqueKeysWithValues: survey.entries.compactMap { entry -> (String, String)? in
            guard case let .garbage(reason) = entry.kind else { return nil }
            return (entry.label, reason)
        })
        #expect(reasons.count == 5, "\(reasons)")
        #expect(reasons[".staging/llama.staged-1"] == "orphaned staging dir")
        #expect(reasons["no-manifest"] == "no manifest")
        #expect(reasons["stray.txt"] == "not a store entry")
        #expect(reasons[".hidden"] == "not a store entry")
        #expect(reasons["meta__zephyr__Q4.gguf"]?.hasPrefix("unreadable manifest") == true)
        #expect(survey.entries.last?.label == live.description, "garbage sweeps before the live model")
    }

    /// Garbage is evicted at any usage, so a live staging attempt classified as garbage
    /// would be deleted under its download. Covers both names an attempt can carry: a GGUF
    /// adoption's scratch directory and an ONNX stage's entry key.
    @Test func aPinnedStagingAttemptIsNotGarbage() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let fm = FileManager.default
        let staging = ws.modelsDir.appendingPathComponent(
            Entry.stagingDirName, isDirectory: true)
        let key = ModelStorageKey(value: "meta__zephyr__Q4.gguf")
        for name in ["adopt-scratch", key.value, "dead-attempt"] {
            try fm.createDirectory(at: staging.appendingPathComponent(name, isDirectory: true),
                                   withIntermediateDirectories: true)
        }

        var pins = SweepPins()
        pins.stagingAttempts = ["adopt-scratch"]
        pins.models = [key]
        let labels = ws.survey(pinning: pins).entries.compactMap { entry -> String? in
            guard case .garbage = entry.kind else { return nil }
            return entry.label
        }

        #expect(labels == [".staging/dead-attempt"],
                "only the attempt no transfer owns is reclaimable: \(labels)")
    }

    /// Same-second ties fall back to publish time, so the order is stable across runs.
    @Test func anEqualLastUsedTieBreaksOnFetchedAt() throws {
        let older = try modelEntry(try ggufTextSpec("meta/older", "Q4.gguf"),
                                   lastUsedAt: at(10), fetchedAt: at(0))
        let newer = try modelEntry(try ggufTextSpec("meta/newer", "Q4.gguf"),
                                   lastUsedAt: at(10), fetchedAt: at(5))

        // Newest-first, so a stable sort alone could not produce the answer.
        let survey = StorageSurvey(entries: [newer, older])

        #expect(survey.entries.map(\.label) == [older.label, newer.label])
    }

    /// The recorded `blobs_bytes` wins over the bytes on disk, which proves the walk is
    /// skipped.
    @Test func surveyTrustsTheRecordedSizeOverThePayload() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let entry = try installEntry(ws, try ggufTextSpec("meta/llama", "Q4.gguf"))
        try editManifest(entry) { $0["blobs_bytes"] = 777_000 }

        let survey = ws.survey()

        #expect(survey.entries.map(\.sizeBytes) == [777_000])
        #expect(survey.usedBytes == 777_000)
    }

    @Test func surveyFallsBackToTheWalkWithoutARecordedSize() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let entry = try installEntry(ws, try ggufTextSpec("meta/llama", "Q4.gguf"), payloadBytes: 64_000)
        try editManifest(entry) { $0.removeValue(forKey: "blobs_bytes") }

        let size = try #require(ws.survey().entries.first?.sizeBytes)

        #expect(size == entrySizeBytes(at: entry.appendingPathComponent(Entry.blobsDirName)),
                "the fallback measures blobs/, not the entry dir")
        #expect(size > 0)
    }

    @Test func applySweepReportsARemovalItCouldNotPerform() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sweep-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("blocker".utf8).write(to: root.appendingPathComponent("nested"))
        let plan = SweepPlan(
            evictions: [StorageEntry(path: root.appendingPathComponent("nested/entry/child"),
                                     label: "blocked", sizeBytes: 10,
                                     kind: .garbage(reason: "no manifest"))],
            freedBytes: 10, stillOverByBytes: nil)

        let report = applySweep(plan)

        #expect(report.removed.isEmpty)
        #expect(report.freedBytes == 0)
        #expect(report.failed.count == 1)
    }

    /// The crate's `remove_entry`: a planned path that is already gone is the outcome the
    /// sweep wanted, not a failed removal.
    @Test func applySweepCountsAnAlreadyGoneEntryAsReclaimed() {
        let gone = FileManager.default.temporaryDirectory
            .appendingPathComponent("gone-\(UUID().uuidString)", isDirectory: true)
        let plan = SweepPlan(
            evictions: [StorageEntry(path: gone, label: "gone", sizeBytes: 10,
                                     kind: .garbage(reason: "no manifest"))],
            freedBytes: 10, stillOverByBytes: nil)

        let report = applySweep(plan)

        #expect(report.removed.map(\.path) == [gone])
        #expect(report.failed.isEmpty)
    }

    @Test func aCacheHitChangesTheSweepOrder() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let stale = try ggufTextSpec("meta/stale", "Q4.gguf")
        let fresh = try ggufTextSpec("meta/fresh", "Q4.gguf")
        try installEntry(ws, stale, lastUsedAt: at(0), fetchedAt: at(0))
        try installEntry(ws, fresh, lastUsedAt: at(1), fetchedAt: at(1))
        #expect(ws.survey().entries.map(\.label) == [stale.description, fresh.description])

        _ = try await ws.models.ensure(stale) { _ in Issue.record("a cache hit does not re-fetch") }

        #expect(ws.survey().entries.map(\.label) == [fresh.description, stale.description])
    }

    /// `StorageSurvey` and `plan` never touch disk, so their cases need no store.
    private func modelEntry(_ spec: Model, lastUsedAt: Date, fetchedAt: Date) throws -> StorageEntry {
        let key = try #require(ModelStorageKey.of(spec))
        return StorageEntry(path: URL(fileURLWithPath: "/models/\(key.relativeDir)"),
                            label: spec.description, sizeBytes: 100,
                            kind: .model(key: key, lastUsedAt: lastUsedAt, fetchedAt: fetchedAt))
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
