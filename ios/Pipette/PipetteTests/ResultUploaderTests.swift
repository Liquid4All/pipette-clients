import Testing
import Foundation
@testable import Pipette

@MainActor struct ResultUploaderTests {
    @Test func drainAllRetriesTransientNetworkFailureWithinPass() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try saveManifest(ws: ws, cells: [cell(id: "cell-1", benchmarkId: "bench-1")], contributeResults: true)
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [
            .fail("network down"),
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])
        let sleeps = SleepRecorder()
        let uploader = try uploader(submitter: submitter, ws: ws, retryDelays: [0.01], sleeps: sleeps)

        let outcomes = await uploader.drainAll()

        #expect(outcomes["job-1"]?.submitted == 1)
        #expect(outcomes["job-1"]?.errors == [])
        #expect(submitter.batches.count == 2)
        #expect(sleeps.recorded == [0.01])
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(ws.results.loadSubmission("cell-1")?.serverJobId == "server-1")
    }

    @Test func networkFailureThenSuccessfulReDrain() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try saveManifest(ws: ws, cells: [cell(id: "cell-1", benchmarkId: "bench-1")], contributeResults: true)
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")

        let failing = ScriptedBatchSubmitter(steps: [.fail("network down")])
        let firstPass = try await uploader(submitter: failing, ws: ws, retryDelays: []).drainAll()

        #expect(firstPass["job-1"]?.submitted == 0)
        #expect(firstPass["job-1"]?.errors.count == 1)
        #expect(ws.results.loadSubmission("cell-1")?.status == .failed)
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == nil)

        let succeeding = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])
        let secondPass = try await uploader(submitter: succeeding, ws: ws, retryDelays: []).drainAll()

        #expect(secondPass["job-1"]?.submitted == 1)
        #expect(secondPass["job-1"]?.errors == [])
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(ws.results.loadSubmission("cell-1")?.status == .submitted)
    }

    @Test func drainAllDoesNotResubmitAlreadySyncedCells() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // cell-2's record was accepted but the manifest never adopted it (the app died
        // between the two writes).
        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try saveManifest(
            ws: ws,
            cells: [
                synced,
                cell(id: "cell-2", benchmarkId: "bench-2"),
                cell(id: "cell-3", benchmarkId: "bench-3")
            ],
            contributeResults: true
        )
        for (cellId, benchmarkId) in [("cell-1", "bench-1"), ("cell-2", "bench-2"), ("cell-3", "bench-3")] {
            try writePayload(ws: ws, cellId: CellId(cellId), benchmarkId: benchmarkId)
        }
        try ws.results.saveSubmission(.submitted(serverJobId: "server-1"), "cell-1")
        try ws.results.saveSubmission(.submitted(serverJobId: "server-2"), "cell-2")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-3"]]))
        ])
        let uploader = try uploader(submitter: submitter, ws: ws)

        let outcomes = await uploader.drainAll()

        // cell-2 heals from its record.
        #expect(submitter.batches == [["bench-3"]])
        #expect(outcomes["job-1"]?.submitted == 2)
        let manifest = try #require(ws.loadJobManifest(jobId: "job-1"))
        #expect(manifest.cells.map(\.serverJobId) == ["server-1", "server-2", "server-3"])

        let secondPass = await uploader.drainAll()
        #expect(secondPass.isEmpty)
        #expect(submitter.batches.count == 1)
    }

    @Test func drainAllSkipsJobsThatNeverIntendedToSubmit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // Neither job opted in, but job-2's failed record proves a submission was attempted.
        try saveManifest(ws: ws, jobId: "job-1", cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        try saveManifest(ws: ws, jobId: "job-2", cells: [cell(id: "cell-2", benchmarkId: "bench-2")])
        try writePayload(ws: ws, cellId: "cell-2", benchmarkId: "bench-2")
        try ws.results.saveSubmission(.failed(["network down"]), "cell-2")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-2"]]))
        ])

        let outcomes = try await uploader(submitter: submitter, ws: ws).drainAll()

        #expect(submitter.batches == [["bench-2"]])
        #expect(outcomes["job-1"] == nil)
        #expect(outcomes["job-2"]?.submitted == 1)
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == nil)
        #expect(ws.loadJobManifest(jobId: "job-2")?.cells[0].serverJobId == "server-2")
    }

    @Test func drainAllSkipsRunningJobs() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try saveManifest(
            ws: ws,
            cells: [cell(id: "cell-1", benchmarkId: "bench-1")],
            status: .running,
            contributeResults: true
        )
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcomes = try await uploader(submitter: submitter, ws: ws).drainAll()

        #expect(outcomes.isEmpty)
        #expect(submitter.batches.isEmpty)
    }

    @Test func drainJobForcesSubmissionWithoutContributeOptIn() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try saveManifest(ws: ws, cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])

        let outcome = try await uploader(submitter: submitter, ws: ws).drainJob(jobId: "job-1")

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == [])
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
    }

    /// The launch drain and a job-end `drainJob` firing together must serialize through
    /// `queueTail`. The submitter's delay makes the passes overlap, so without it both
    /// would see the cells unsubmitted and hit the network twice.
    @Test func concurrentDrainsSerializeAndDoNotDoubleSubmit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try saveManifest(
            ws: ws,
            cells: [
                cell(id: "cell-1", benchmarkId: "bench-1"),
                cell(id: "cell-2", benchmarkId: "bench-2")
            ],
            contributeResults: true
        )
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: "cell-2", benchmarkId: "bench-2")
        let submitter = ScriptedBatchSubmitter(
            steps: [
                .respond(try batchResponse([
                    ["index": 0, "job_id": "server-1"],
                    ["index": 1, "job_id": "server-2"]
                ]))
            ],
            delayNanoseconds: 50_000_000
        )
        let uploader = try uploader(submitter: submitter, ws: ws)

        async let first = uploader.drainAll()
        async let second = uploader.drainJob(jobId: "job-1")
        _ = await (first, second)

        #expect(submitter.batches == [["bench-1", "bench-2"]])
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[1].serverJobId == "server-2")
    }

    @Test func resendMigratesResultsSubmittedToADifferentCollector() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // cell-1 went to another collector and cell-3 recorded none, so both migrate. cell-2
        // is already on the configured collector.
        var acked1 = cell(id: "cell-1", benchmarkId: "bench-1")
        acked1.serverJobId = "old-1"
        var acked2 = cell(id: "cell-2", benchmarkId: "bench-2")
        acked2.serverJobId = "cur-2"
        var acked3 = cell(id: "cell-3", benchmarkId: "bench-3")
        acked3.serverJobId = "old-3"
        try saveManifest(ws: ws, cells: [acked1, acked2, acked3])
        for (cellId, benchmarkId) in [("cell-1", "bench-1"), ("cell-2", "bench-2"), ("cell-3", "bench-3")] {
            try writePayload(ws: ws, cellId: CellId(cellId), benchmarkId: benchmarkId)
        }
        try ws.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: try ServerUrl("https://old.example.com")),
            "cell-1")
        try ws.results.saveSubmission(
            .submitted(serverJobId: "cur-2", collector: try registrationData().serverUrl),
            "cell-2")
        try ws.results.saveSubmission(.submitted(serverJobId: "old-3"), "cell-3")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([
                ["index": 0, "job_id": "server-1"],
                ["index": 1, "job_id": "server-3"]
            ]))
        ])

        let outcome = try await uploader(submitter: submitter, ws: ws)
            .resendForCollectorChange(benchmarkIds: ["bench-1", "bench-2", "bench-3"])

        #expect(outcome.submitted == 2)
        #expect(outcome.errors == [])
        #expect(submitter.batches == [["bench-1", "bench-3"]])
        let manifest = try #require(ws.loadJobManifest(jobId: "job-1"))
        #expect(manifest.cells.map(\.serverJobId) == ["server-1", "cur-2", "server-3"])
        #expect(ws.results.loadSubmission("cell-1")?.collector == (try registrationData().serverUrl))
        #expect(ws.results.loadSubmission("cell-3")?.collector == (try registrationData().serverUrl))
    }

    @Test func resendSkipsBenchmarksNotOfferedByTheCollector() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(ws: ws, cells: [acked])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        try ws.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: try ServerUrl("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcome = try await uploader(submitter: submitter, ws: ws)
            .resendForCollectorChange(benchmarkIds: ["bench-999"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors == [])
        #expect(submitter.batches.isEmpty)
    }

    @Test func resendIgnoresNeverSubmittedResults() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // Never submitted anywhere, so not a migration: the drain triggers handle it.
        try saveManifest(ws: ws, cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcome = try await uploader(submitter: submitter, ws: ws)
            .resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(submitter.batches.isEmpty)
    }

    @Test func resendKeepsPriorRecordWhenMigrationFails() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(ws: ws, cells: [acked])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        try ws.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: try ServerUrl("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [.fail("network down")])

        let outcome = try await uploader(submitter: submitter, ws: ws)
            .resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors.count == 1)
        // The record proves the result already lives on the old collector.
        let record = try #require(ws.results.loadSubmission("cell-1"))
        #expect(record.status == .submitted)
        #expect(record.serverJobId == "old-1")
        #expect(record.collector == (try ServerUrl("https://old.example.com")))
        #expect(ws.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "old-1")
    }

    @Test func resendWithoutRegistrationReportsError() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(ws: ws, cells: [acked])
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")
        try ws.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: try ServerUrl("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [])
        let uploader = ResultUploader(
            submitResultBatch: submitter.submit,
            credentials: { throw IdentityError.privateKeyMissing },
            retryDelays: [],
            ws: ws
        )

        let outcome = await uploader.resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors.count == 1)
        #expect(submitter.batches.isEmpty)
    }

    // MARK: - The stranded counter
    //
    // It must agree with the submit sweep's own gate, or the job page's "Submit N Results"
    // button counts work the sweep will never do.

    private func sourcedCell(_ id: String, source: BenchmarkSource?) -> JobCell {
        JobCell(
            cellId: CellId(id), benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput,
            runStatus: .completed, serverJobId: nil, errorMessage: nil,
            source: ggufTextFixture("test/model-GGUF", "model.gguf"),
            benchmarkSource: source)
    }

    private func manifest(_ cells: [JobCell], contribute: Bool = true) -> JobManifest {
        JobManifest(
            jobId: "job-1", createdAt: "2026-06-24T00:00:00Z",
            nGpuLayers: 99, contextSize: 4096, cells: cells, status: .completed,
            contributeResults: contribute)
    }

    /// A local result is finished, not stranded: counting it would wake the uploader on
    /// every launch and foreground to submit nothing.
    @Test func aLocalResultIsNotStranded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let local = sourcedCell("local-1", source: .local)
        try writePayload(ws: ws, cellId: local.cellId, benchmarkId: local.benchmarkId)

        let uploader = ResultUploader(ws: ws)
        #expect(await uploader.hasStrandedResults(manifest([local])) == false)
    }

    @Test func aRemoteResultIsStillStranded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let remote = sourcedCell("remote-1", source: .remote)
        try writePayload(ws: ws, cellId: remote.cellId, benchmarkId: remote.benchmarkId)

        let uploader = ResultUploader(ws: ws)
        #expect(await uploader.hasStrandedResults(manifest([remote])))
    }

    /// A cell written before the local half existed carries no source and came from the
    /// synced catalog, so it must keep counting, or an upgrade strands every result already
    /// waiting to go up.
    @Test func aCellWithNoRecordedSourceIsStillStranded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let old = sourcedCell("old-1", source: nil)
        try writePayload(ws: ws, cellId: old.cellId, benchmarkId: old.benchmarkId)

        let uploader = ResultUploader(ws: ws)
        #expect(await uploader.hasStrandedResults(manifest([old])))
    }

    @Test func onlySubmittableCellsAreStranded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cells = [sourcedCell("local-1", source: .local), sourcedCell("remote-1", source: .remote)]
        for c in cells {
            try writePayload(ws: ws, cellId: c.cellId, benchmarkId: c.benchmarkId)
        }

        let uploader = ResultUploader(ws: ws)
        #expect(await uploader.hasStrandedResults(manifest(cells)))
        #expect(await uploader.hasStrandedResults(manifest([cells[0]])) == false)
    }

    /// A `claim=` run's leftovers are `pipette-ios-proxy`'s to submit: a launch drain
    /// would echo a lease that may be gone or re-leased, even after a failed attempt.
    @Test func aClaimedJobIsNeverStranded() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let remote = sourcedCell("remote-1", source: .remote)
        try writePayload(ws: ws, cellId: remote.cellId, benchmarkId: remote.benchmarkId)
        try ws.results.saveSubmission(.failed(["offline"]), remote.cellId)
        var claimed = manifest([remote])
        claimed.claimJobId = "job-claimed"

        let uploader = ResultUploader(ws: ws)
        #expect(await uploader.hasStrandedResults(manifest([remote])))
        #expect(await uploader.hasStrandedResults(claimed) == false)
    }

    // MARK: - Helpers

    private func uploader(
        submitter: ScriptedBatchSubmitter,
        ws: PipetteWorkspace,
        retryDelays: [TimeInterval] = [],
        sleeps: SleepRecorder = SleepRecorder()
    ) throws -> ResultUploader {
        // The credentials closure is nonisolated, so it must not call back into
        // MainActor-isolated test helpers.
        let registration = try registrationData()
        let auth = ClientIdentity(
            clientId: registration.clientId, privateKeyHex: try PrivateKey(testPrivateKeyHex))
        return ResultUploader(
            submitResultBatch: submitter.submit,
            credentials: { (registration, auth) },
            retryDelays: retryDelays,
            ws: ws,
            sleep: sleeps.sleep
        )
    }

    private func saveManifest(
        ws: PipetteWorkspace,
        jobId: JobId = "job-1",
        cells: [JobCell],
        status: JobStatus = .completed,
        contributeResults: Bool? = nil
    ) throws {
        ws.saveJobManifest(JobManifest(
            jobId: jobId,
            createdAt: "2026-06-10T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: status,
            contributeResults: contributeResults
        ))
    }

}

/// Explicitly `nonisolated` so `submit` matches the nonisolated `BatchSubmitter` type:
/// under the project's MainActor default isolation, an implicitly isolated method crashes
/// the runner when invoked through that type (seen on CI's Xcode 26.2).
private nonisolated final class ScriptedBatchSubmitter: @unchecked Sendable {
    enum Step {
        case fail(String)
        case respond(BatchSubmitResponse)
    }

    private let lock = NSLock()
    private var steps: [Step]
    private var storedBatches: [[String]] = []
    private let delayNanoseconds: UInt64

    init(steps: [Step], delayNanoseconds: UInt64 = 0) {
        self.steps = steps
        self.delayNanoseconds = delayNanoseconds
    }

    var batches: [[String]] {
        lock.lock()
        defer { lock.unlock() }
        return storedBatches
    }

    func submit(
        server: ServerUrl,
        auth: ClientIdentity,
        payloadsJson: [Data]
    ) async throws -> BatchSubmitResponse {
        if delayNanoseconds > 0 {
            try? await Task.sleep(nanoseconds: delayNanoseconds)
        }
        let payloads = try payloadsJson.map { payload -> [String: Any] in
            guard let object = try JSONSerialization.jsonObject(with: payload) as? [String: Any]
            else { throw UploaderTestError.invalidPayloadJson }
            return object
        }

        let step = record(payloads.map { $0["benchmark_id"] as? String ?? "" })
        switch step {
        case .fail(let message):
            throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        case .respond(let response):
            return response
        case nil:
            throw UploaderTestError.unexpectedNetworkCall
        }
    }

    /// Synchronous because NSLock is unavailable in an async context in Swift 6 mode.
    private func record(_ batch: [String]) -> Step? {
        lock.lock()
        defer { lock.unlock() }
        storedBatches.append(batch)
        return steps.isEmpty ? nil : steps.removeFirst()
    }
}

/// `nonisolated` for the same reason as `ScriptedBatchSubmitter`: `sleep` is passed as
/// the nonisolated `Sleeper` type.
private nonisolated final class SleepRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storedDelays: [TimeInterval] = []

    var recorded: [TimeInterval] {
        lock.lock()
        defer { lock.unlock() }
        return storedDelays
    }

    func sleep(_ seconds: TimeInterval) async {
        record(seconds)
    }

    private func record(_ seconds: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        storedDelays.append(seconds)
    }
}

private enum UploaderTestError: Error {
    case invalidPayloadJson
    case unexpectedNetworkCall
}
