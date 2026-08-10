import Testing
import Foundation
@testable import Pipette

/// Crash-sentinel recovery: a leftover `active-cell.json` at launch means the
/// process died mid-cell (jetsam OOM kill), and recovery must turn that into
/// an explained, non-crash-looping cell state.
@MainActor
struct CrashRecoveryTests {
    @Test func applyCrashEvidenceFailsRunningCellWithExplanation() {
        var manifest = manifest(cells: [
            cell(id: "done", model: "Qwen", status: .completed),
            cell(id: "active", model: "Qwen", status: .running),
            cell(id: "queued", model: "Qwen", status: .pending)
        ])

        let applied = manifest.applyCrashEvidence(sentinel: sentinel(for: "active"), payloadIsFresh: false)
        #expect(applied)

        #expect(manifest.cells[1].runStatus == .failed)
        #expect(manifest.cells[1].crashCount == 1)
        #expect(manifest.cells[1].errorMessage?.contains("terminated the app") == true)
        // First crash leaves siblings alone — only a repeat condemns the model.
        #expect(manifest.cells[2].runStatus == .pending)
    }

    @Test func applyCrashEvidencePromotesCellWhosePayloadLanded() {
        var manifest = manifest(cells: [cell(id: "active", model: "Qwen", status: .running)])

        let applied = manifest.applyCrashEvidence(sentinel: sentinel(for: "active"), payloadIsFresh: true)
        #expect(applied)

        #expect(manifest.cells[0].runStatus == .completed)
        #expect(manifest.cells[0].errorMessage == nil)
        #expect(manifest.cells[0].crashCount == nil)
    }

    @Test func applyCrashEvidenceLeavesTerminalCellsAndUnknownIdsAlone() {
        var manifest = manifest(cells: [cell(id: "done", model: "Qwen", status: .completed)])

        // Kill landed after the terminal save but before the sentinel clear.
        let appliedToDone = manifest.applyCrashEvidence(sentinel: sentinel(for: "done"), payloadIsFresh: false)
        #expect(!(appliedToDone))
        #expect(manifest.cells[0].runStatus == .completed)

        let appliedToMissing = manifest.applyCrashEvidence(sentinel: sentinel(for: "missing"), payloadIsFresh: false)
        #expect(!(appliedToMissing))
    }

    @Test func secondCrashCondemnsRemainingCellsOfTheSameModel() {
        var crashed = cell(id: "active", model: "Qwen", status: .running)
        crashed.crashCount = 1
        var manifest = manifest(cells: [
            crashed,
            cell(id: "queued-same", model: "Qwen", status: .pending),
            cell(id: "cancelled-same", model: "Qwen", status: .cancelled),
            cell(id: "queued-other", model: "Gemma", status: .pending),
            cell(id: "done-same", model: "Qwen", status: .completed)
        ])

        let applied = manifest.applyCrashEvidence(sentinel: sentinel(for: "active"), payloadIsFresh: false)
        #expect(applied)

        #expect(manifest.cells[0].crashCount == 2)
        #expect(manifest.cells[0].runStatus == .failed)
        #expect(manifest.cells[1].runStatus == .failed)
        #expect(manifest.cells[1].errorMessage?.contains("crashed the app 2 times") == true)
        #expect(manifest.cells[2].runStatus == .failed)
        // Other models and already-finished work are untouched.
        #expect(manifest.cells[3].runStatus == .pending)
        #expect(manifest.cells[4].runStatus == .completed)
    }

    @Test func crashEvidenceThenInterruptedRecoveryPausesJobWithFailedCell() {
        var manifest = manifest(cells: [
            cell(id: "active", model: "Qwen", status: .running),
            cell(id: "queued", model: "Gemma", status: .pending)
        ])

        let applied = manifest.applyCrashEvidence(sentinel: sentinel(for: "active"), payloadIsFresh: false)
        #expect(applied)
        let recovered = manifest.recoverInterruptedRunState()
        #expect(recovered)

        // The crashed cell keeps its explained failure (not flattened to
        // .cancelled), the untouched queue becomes resumable, job pauses.
        #expect(manifest.cells.map(\.runStatus) == [.failed, .cancelled])
        #expect(manifest.status == .paused)
    }

    @Test func crashMessageQuantifiesMemoryWhenSentinelHasSnapshot() {
        var manifest = manifest(cells: [cell(id: "active", model: "Qwen", status: .running)])

        let applied = manifest.applyCrashEvidence(
            sentinel: sentinel(for: "active", availableBytes: 2_000_000_000, modelBytes: 1_900_000_000),
            payloadIsFresh: false
        )
        #expect(applied)

        let message = manifest.cells[0].errorMessage ?? ""
        #expect(message.contains(ByteFormat.memory(2_000_000_000)))
        #expect(message.contains(ByteFormat.memory(1_900_000_000)))

        // No snapshot (simulator, old sentinel) → generic message, no numbers.
        var bare = self.manifest(cells: [cell(id: "active", model: "Qwen", status: .running)])
        let bareApplied = bare.applyCrashEvidence(sentinel: sentinel(for: "active"), payloadIsFresh: false)
        #expect(bareApplied)
        #expect(!(bare.cells[0].errorMessage?.contains("of memory available") == true))
    }

    @Test func sentinelDecodesWithoutMemoryFields() throws {
        // A sentinel written by an older build: no memory snapshot, and a `modelPath`
        // this build no longer has a field for. Both are tolerated — an absent key
        // decodes to nil, an unknown one is ignored.
        let legacy = #"{"cellId":"active","modelPath":"/tmp/m.gguf","startedAt":"2026-06-09T10:00:00Z"}"#
        let decoded = try JSONDecoder().decode(ActiveCellSentinel.self, from: Data(legacy.utf8))
        #expect(decoded.availableBytes == nil)
        #expect(decoded.modelBytes == nil)
    }

    @Test func crashCountSurvivesManifestJSONRoundTripAndOldManifestsDecode() throws {
        var crashed = cell(id: "active", model: "Qwen", status: .failed)
        crashed.crashCount = 1
        let manifest = manifest(cells: [crashed])

        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(JobManifest.self, from: data)
        #expect(decoded.cells[0].crashCount == 1)

        // Pre-crashCount manifests must keep decoding (field absent → nil).
        var json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        var cells = try #require(json["cells"] as? [[String: Any]])
        cells[0].removeValue(forKey: "crashCount")
        json["cells"] = cells
        let legacy = try JSONDecoder().decode(
            JobManifest.self,
            from: JSONSerialization.data(withJSONObject: json)
        )
        #expect(legacy.cells[0].crashCount == nil)
    }

    @Test func sentinelRoundTripAndRecoverInterruptedJobsAppliesEvidence() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let jobId = JobId("crash-test-\(UUID().uuidString)")

        let manifest = manifest(jobId: jobId, cells: [
            cell(id: "active", model: "Qwen", status: .running),
            cell(id: "queued", model: "Qwen", status: .pending)
        ])
        storage.saveJobManifest(manifest)

        let sentinel = ActiveCellSentinel(
            cellId: "active",
            startedAt: JobDateFormat.iso8601.string(from: Date())
        )
        storage.saveActiveCellSentinel(sentinel, jobId: jobId)
        #expect(storage.loadActiveCellSentinel(jobId: jobId)?.cellId == "active")

        storage.recoverInterruptedJobs()

        let recovered = try #require(storage.loadJobManifest(jobId: jobId))
        #expect(recovered.cells[0].runStatus == .failed)
        #expect(recovered.cells[0].crashCount == 1)
        #expect(recovered.cells[1].runStatus == .cancelled)
        #expect(recovered.status == .paused)
        // Sentinel is consumed so the next launch doesn't double-count.
        #expect(storage.loadActiveCellSentinel(jobId: jobId) == nil)
    }

    @Test func crashPayloadFreshnessComparesAgainstSentinelStart() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let jobId = JobId("crash-test-\(UUID().uuidString)")

        try storage.results.saveResult(
            .remotePending, "active", payload: Data("{}".utf8), extras: Data("{}".utf8))

        // Payload written now, attempt started 60s ago → this attempt's result.
        let fresh = ActiveCellSentinel(
            cellId: "active",
            startedAt: JobDateFormat.iso8601.string(from: Date().addingTimeInterval(-60))
        )
        #expect(storage.crashPayloadIsFresh(fresh, jobId: jobId))

        // Attempt started after the payload was written → stale leftover from
        // an earlier run; missing payloads are never fresh.
        let stale = ActiveCellSentinel(
            cellId: "active",
            startedAt: JobDateFormat.iso8601.string(from: Date().addingTimeInterval(60))
        )
        #expect(!(storage.crashPayloadIsFresh(stale, jobId: jobId)))
        let noPayload = ActiveCellSentinel(
            cellId: "other",
            startedAt: fresh.startedAt
        )
        #expect(!(storage.crashPayloadIsFresh(noPayload, jobId: jobId)))
    }

    @Test func freshPayloadPromotionEndToEndThroughRecoverInterruptedJobs() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let jobId = JobId("crash-test-\(UUID().uuidString)")

        let manifest = manifest(jobId: jobId, cells: [
            cell(id: "active", model: "Qwen", status: .running)
        ])
        storage.saveJobManifest(manifest)
        try storage.results.saveResult(
            .remotePending, "active", payload: Data("{}".utf8), extras: Data("{}".utf8))
        storage.saveActiveCellSentinel(
            ActiveCellSentinel(
                cellId: "active",
                startedAt: JobDateFormat.iso8601.string(from: Date().addingTimeInterval(-60))
            ),
            jobId: jobId
        )

        storage.recoverInterruptedJobs()

        let recovered = try #require(storage.loadJobManifest(jobId: jobId))
        #expect(recovered.cells[0].runStatus == .completed)
        #expect(recovered.status == .completed)
    }

    private func manifest(jobId: JobId = "job-1", cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: "2026-06-09T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .running
        )
    }

    private func sentinel(
        for cellId: CellId,
        availableBytes: Int64? = nil,
        modelBytes: Int64? = nil
    ) -> ActiveCellSentinel {
        ActiveCellSentinel(
            cellId: cellId,
            startedAt: "2026-06-09T10:00:00Z",
            availableBytes: availableBytes,
            modelBytes: modelBytes
        )
    }

    private func cell(id: CellId, model: String, status: CellRunStatus) -> JobCell {
        JobCell(
            cellId: id,
            benchmarkId: "decode",
            benchmarkType: .decodeThroughput,
            runStatus: status,
            serverJobId: nil,
            errorMessage: nil,
            source: ggufTextFixture("test/\(model)-GGUF", "\(model)-Q4_0.gguf")
        )
    }
}
