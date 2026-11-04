import Foundation
import Testing

@testable import Pipette

/// Stage, then publish: the atomic install. The property under test throughout is that
/// the store holds either the previous entry or the new one, never neither.
@MainActor struct StagingTests {
    private func mlxSpec(_ repo: String, prefix: String? = nil) throws -> Model {
        .mlx(try HfModelRef.parse(repo: repo, subpath: prefix).asMlx())
    }

    /// Fill a staged attempt with the files an MLX bundle needs, at the layout the
    /// published entry will use.
    @discardableResult
    private func writeBundle(into staged: StagedEntry, spec: Model) throws -> URL {
        let stagedForm = try #require(spec.withoutAuthToken.toStored(base: Entry.blobsDirName))
        let bundle = try #require(stagedForm.underRoot(staged.root)?.boundPaths?.payload)
        let dir = URL(fileURLWithPath: bundle, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in mlxBundleFiles {
            try Data(repeating: 0x67, count: 512).write(to: dir.appendingPathComponent(name))
        }
        return dir
    }

    // MARK: - Layout

    @Test func stagingCreatesAnAttemptOutsideTheEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models

        let staged = try store.stage(spec)

        let fm = FileManager.default
        #expect(fm.fileExists(atPath: staged.blobs.path))
        #expect(fm.fileExists(atPath: staged.root.appendingPathComponent(
            TransferPlan.fileName).path))
        #expect(staged.root.path.contains(Entry.stagingDirName))
        #expect(try store.find(spec) == nil, "staging must not publish anything")
    }

    /// Two attempts at one model get their own directories, so a retry never writes into
    /// the tree an earlier attempt is still filling.
    @Test func twoAttemptsAtOneModelDoNotShareADirectory() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")

        let first = try ws.models.stage(spec)
        let second = try ws.models.stage(spec)

        #expect(first.attemptId != second.attemptId)
        #expect(first.root != second.root)
    }

    /// The staged record carries no credential, so an abandoned attempt cannot leak one.
    @Test func theStagedRecordIsCredentialFree() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var repo = try HfRepo.parseOrgRepo("org/gated-MLX-4bit")
        repo.authToken = try AuthToken("hf_secrettokenvalue")
        let spec = Model.mlx(Mlx(source: .huggingFace(repo: repo, prefix: nil)))

        let staged = try ws.models.stage(spec)

        let raw = try String(contentsOf: staged.root.appendingPathComponent(
            TransferPlan.fileName), encoding: .utf8)
        #expect(!raw.contains("hf_secrettokenvalue"))
        #expect(staged.plan.declared.authToken == nil)
    }

    // MARK: - Publish

    @Test func publishingMakesTheEntryResolvable() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        let staged = try store.stage(spec)
        try writeBundle(into: staged, spec: spec)

        try store.publish(staged)

        #expect(store.bound(spec) != nil, "the published entry must bind")
        #expect(!FileManager.default.fileExists(atPath: staged.root.path),
                "the attempt directory is consumed by the publish")
    }

    /// So an unpinned coordinate can still say which commit and files its installed bytes
    /// came from, without carrying transfer bookkeeping forever.
    @Test func publishingPromotesSnapshotProvenanceAndDropsTransferState() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        var staged = try store.stage(spec)
        _ = try writeBundle(into: staged, spec: spec)
        let resolved = ResolvedTransfer(
            commit: "c0ffee",
            files: mlxBundleFiles.map {
                PlannedFile(path: $0, sizeBytes: 512, etag: "etag-\($0)")
            })
        staged = store.recordResolution(staged, resolved)

        let entry = try store.publish(staged)

        let manifest = try #require(try store.find(spec))
        #expect(manifest.resolved == resolved)
        #expect(manifest.resolved?.files.map(\.path).sorted() == mlxBundleFiles.sorted())
        #expect(!FileManager.default.fileExists(
            atPath: entry.appendingPathComponent(TransferPlan.fileName).path))
        #expect(Set(try FileManager.default.contentsOfDirectory(atPath: entry.path))
            == [Entry.blobsDirName, Entry.manifestName])
    }

    @Test func publishingOverAnExistingEntryReplacesIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        try installEntry(ws, spec, payloadBytes: 16)
        let entry = try #require(store.entryDir(for: spec))
        let stale = entry.appendingPathComponent("\(Entry.blobsDirName)/stale-shard.safetensors")
        try Data(repeating: 0x1, count: 8).write(to: stale)

        let staged = try store.stage(spec)
        try writeBundle(into: staged, spec: spec)
        try store.publish(staged)

        #expect(store.bound(spec) != nil)
        #expect(!FileManager.default.fileExists(atPath: stale.path),
                "the replaced entry's bytes must not survive into the new one")
    }

    @Test func aFailedPublishLeavesThePreviousEntryIntact() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        try installEntry(ws, spec, payloadBytes: 4096)
        let before = try #require(store.bound(spec))

        // Staged, but never filled: the payload the manifest would name is not there.
        let staged = try store.stage(spec)
        #expect(throws: StagingError.self) { try store.publish(staged) }

        #expect(store.bound(spec) == before, "the previous entry must still resolve")
    }

    /// A partial payload is refused before the publish, so it cannot become an entry that
    /// `find` resolves and an engine then fails to load.
    @Test func anIncompletePayloadIsNotPublished() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        let staged = try store.stage(spec)
        // The directory exists but holds none of the files the bundle needs.
        try FileManager.default.createDirectory(at: staged.blobs, withIntermediateDirectories: true)

        do {
            try store.publish(staged)
            Issue.record("an incomplete payload must not publish")
        } catch let error as StagingError {
            guard case .incompletePayload = error else {
                Issue.record("expected incompletePayload, got \(error)"); return
            }
        }
        #expect(try store.find(spec) == nil, "nothing may be published")
    }

    /// One transfer per key publishes once. Answering with whichever entry exists would
    /// report the publish as successful while binding the previous bytes.
    @Test func publishingAConsumedAttemptFailsRatherThanAnsweringWithTheOldEntry() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        let staged = try store.stage(spec)
        try writeBundle(into: staged, spec: spec)
        try store.publish(staged)

        #expect(throws: StagingError.self) { try store.publish(staged) }
        #expect(store.bound(spec) != nil, "the entry the first publish made is untouched")
    }

    /// A repo subtree keeps its prefix from staging through to the published entry, so
    /// what lands and what `stored` records come from one rule.
    @Test func aRepoSubtreeKeepsItsPrefixThroughPublish() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/multi-MLX", prefix: "4bit")
        let store = ws.models
        let staged = try store.stage(spec)
        try writeBundle(into: staged, spec: spec)

        try store.publish(staged)

        let bound = try #require(store.bound(spec)?.boundPaths?.payload)
        #expect(bound.hasSuffix("/\(Entry.blobsDirName)/4bit"))
    }

    // MARK: - Survivors

    /// A crash before the publish leaves an addressable attempt, which is what makes a
    /// transfer that outlived the process recoverable.
    @Test func stagedFindsTheSurvivorsOfAPreviousLaunch() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let staged = try ws.models.stage(spec)

        let survivors = ws.models.staged()

        #expect(survivors.map(\.attemptId) == [staged.attemptId])
        #expect(survivors.first?.plan.declared == spec.withoutAuthToken)
    }

    /// An attempt with no readable record is not a survivor: nothing can be resumed from
    /// a payload whose coordinate is unknown. It stays on disk as garbage for the sweep.
    @Test func anAttemptWithNoReadableRecordIsNotASurvivor() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let staged = try ws.models.stage(try mlxSpec("org/m-MLX-4bit"))
        try FileManager.default.removeItem(
            at: staged.root.appendingPathComponent(TransferPlan.fileName))

        #expect(ws.models.staged().isEmpty)
    }

    /// A file shorter than the plan said refuses the publish. This is the truncated
    /// multi-gigabyte sidecar: every file is present, so a presence check passes it.
    @Test func aFileShorterThanItsPlanIsNotPublished() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        var staged = try ws.models.stage(spec)
        let dir = try writeBundle(into: staged, spec: spec)
        // The plan names one file at twice the length that landed.
        let name = try #require(mlxBundleFiles.first)
        staged = ws.models.recordResolution(staged, ResolvedTransfer(
            commit: "c0ffee", files: [PlannedFile(path: name, sizeBytes: 1024, etag: nil)]))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent(name).path))

        #expect(throws: StagingError.self) { try ws.models.publish(staged) }
    }

    /// A file the plan names and the transfer never wrote refuses the publish, rather
    /// than publishing an entry an engine later fails to load.
    @Test func aFileMissingFromTheAttemptIsNotPublished() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        var staged = try ws.models.stage(spec)
        _ = try writeBundle(into: staged, spec: spec)
        staged = ws.models.recordResolution(staged, ResolvedTransfer(
            commit: "c0ffee",
            files: [PlannedFile(path: "absent.safetensors", sizeBytes: 512, etag: nil)]))

        #expect(throws: StagingError.self) { try ws.models.publish(staged) }
    }

    /// A correct length is not enough when the source supplied a content digest: this is
    /// the same-length corruption that byte-count validation cannot detect.
    @Test func aSameLengthFileWithTheWrongSourceDigestIsNotPublished() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        var staged = try ws.models.stage(spec)
        _ = try writeBundle(into: staged, spec: spec)
        let name = try #require(mlxBundleFiles.first)
        let wrong = try Sha256(String(repeating: "0", count: 64))
        let actual = try ModelFetch.sha256Hex(ofFileAt: staged.blobs.appendingPathComponent(name))
        staged = ws.models.recordResolution(staged, ResolvedTransfer(
            commit: "c0ffee",
            files: [PlannedFile(path: name, sizeBytes: 512, etag: nil, sha256: wrong)]))

        #expect(throws: StagingError.digestMismatch(
            spec.description, path: name, expected: wrong.value, actual: actual)) {
            try ws.models.publish(staged)
        }
    }

    @Test func aFileMatchingItsSourceDigestIsPublished() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        var staged = try ws.models.stage(spec)
        let dir = try writeBundle(into: staged, spec: spec)
        let name = try #require(mlxBundleFiles.first)
        let digest = try Sha256(try ModelFetch.sha256Hex(ofFileAt: dir.appendingPathComponent(name)))
        staged = ws.models.recordResolution(staged, ResolvedTransfer(
            commit: "c0ffee",
            files: [PlannedFile(path: name, sizeBytes: 512, etag: nil, sha256: digest)]))

        try ws.models.publish(staged)

        #expect(ws.models.bound(spec) != nil)
    }

    /// No reference means no speculative hashing requirement: exact size remains the
    /// compatibility fallback for regular Hub files and sources without digests.
    @Test func aFileWithoutASourceDigestFallsBackToSize() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        var staged = try ws.models.stage(spec)
        _ = try writeBundle(into: staged, spec: spec)
        let name = try #require(mlxBundleFiles.first)
        staged = ws.models.recordResolution(staged, ResolvedTransfer(
            commit: "c0ffee",
            files: [PlannedFile(path: name, sizeBytes: 512, etag: "opaque")]))

        try ws.models.publish(staged)

        #expect(ws.models.bound(spec) != nil)
    }

    /// An abandoned attempt is garbage the sweep reclaims, which is what makes it safe
    /// for a crash to leave one behind.
    @Test func anAbandonedAttemptIsReclaimedBySweeping() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let staged = try ws.models.stage(spec)
        try writeBundle(into: staged, spec: spec)

        let survey = ws.survey(pinning: SweepPins())
        #expect(survey.entries.contains { if case .garbage = $0.kind { true } else { false } },
                "an orphaned attempt reads as garbage")

        ws.sweepToQuota(pinning: SweepPins())
        #expect(!FileManager.default.fileExists(atPath: staged.root.path))
    }

    @Test func discardingAnAttemptRemovesIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let staged = try ws.models.stage(try mlxSpec("org/m-MLX-4bit"))

        ws.models.discard(staged)

        #expect(!FileManager.default.fileExists(atPath: staged.root.path))
        #expect(ws.models.staged().isEmpty)
    }

    // MARK: - The frozen plan

    /// The resolution the transport froze is written into the attempt, so a later launch
    /// can tell which commit the bytes on disk belong to, the question an interrupted
    /// multi-gigabyte transfer turns on.
    @Test func theFrozenPlanIsPersistedAndSurvivesARelaunch() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let spec = try mlxSpec("org/m-MLX-4bit")
        let store = ws.models
        let staged = try store.stage(spec)
        #expect(staged.plan.resolved == nil, "staging opens before the transport resolves")

        let resolved = ResolvedTransfer(commit: "c0ffee", files: [
            PlannedFile(path: "config.json", sizeBytes: 100, etag: "\"abc\""),
            PlannedFile(path: "model.safetensors", sizeBytes: 900, etag: nil),
        ])
        let recorded = store.recordResolution(staged, resolved)
        #expect(recorded.plan.resolved == resolved)

        // Re-read from disk, as a relaunch does.
        let survivor = try #require(store.staged().first)
        #expect(survivor.plan.resolved?.commit == "c0ffee")
        #expect(survivor.plan.resolved?.files.map(\.path) == ["config.json", "model.safetensors"])
        #expect(survivor.plan.resolved?.totalBytes == 1000)
    }

    /// The record stays credential-free once the plan is in it: a resolution carries
    /// lengths and ETags, and the auth that fetched them is resolved per request.
    @Test func theFrozenPlanCarriesNoCredential() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var repo = try HfRepo.parseOrgRepo("org/gated-MLX-4bit")
        repo.authToken = try AuthToken("hf_secrettokenvalue")
        let spec = Model.mlx(Mlx(source: .huggingFace(repo: repo, prefix: nil)))
        let store = ws.models

        let staged = store.recordResolution(
            try store.stage(spec),
            ResolvedTransfer(commit: "c0ffee",
                             files: [PlannedFile(path: "a.json", sizeBytes: 1, etag: nil)]))

        let raw = try String(contentsOf: staged.root.appendingPathComponent(
            TransferPlan.fileName), encoding: .utf8)
        #expect(!raw.contains("hf_secrettokenvalue"))
    }
}
