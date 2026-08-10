import Testing
import Foundation
@testable import Pipette

/// Each test injects its own temporary `FileStorage` into the uploader, so the
/// suite carries no shared global and runs in parallel.
@MainActor struct ResultUploaderTests {
    @Test func drainAllRetriesTransientNetworkFailureWithinPass() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try saveManifest(storage: storage, cells: [cell(id: "cell-1", benchmarkId: "bench-1")], contributeResults: true)
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [
            .fail("network down"),
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])
        let sleeps = SleepRecorder()
        let uploader = uploader(submitter: submitter, storage: storage, retryDelays: [0.01], sleeps: sleeps)

        let outcomes = await uploader.drainAll()

        #expect(outcomes["job-1"]?.submitted == 1)
        #expect(outcomes["job-1"]?.errors == [])
        #expect(submitter.batches.count == 2)
        #expect(sleeps.recorded == [0.01])
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(storage.results.loadSubmission("cell-1")?.serverJobId == "server-1")
    }

    @Test func networkFailureThenSuccessfulReDrain() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try saveManifest(storage: storage, cells: [cell(id: "cell-1", benchmarkId: "bench-1")], contributeResults: true)
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")

        // First pass: the network is down and stays down (no retries left),
        // stranding the cell as a failed submission record.
        let failing = ScriptedBatchSubmitter(steps: [.fail("network down")])
        let firstPass = await uploader(submitter: failing, storage: storage, retryDelays: []).drainAll()

        #expect(firstPass["job-1"]?.submitted == 0)
        #expect(firstPass["job-1"]?.errors.count == 1)
        #expect(storage.results.loadSubmission("cell-1")?.status == .failed)
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == nil)

        // Second pass — as fired on the next launch/foreground — re-drives
        // the stranded cell and syncs it.
        let succeeding = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])
        let secondPass = await uploader(submitter: succeeding, storage: storage, retryDelays: []).drainAll()

        #expect(secondPass["job-1"]?.submitted == 1)
        #expect(secondPass["job-1"]?.errors == [])
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(storage.results.loadSubmission("cell-1")?.status == .submitted)
    }

    @Test func drainAllDoesNotResubmitAlreadySyncedCells() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // cell-1 is fully synced; cell-2's record was accepted but the
        // manifest never adopted it (app died between the two writes);
        // cell-3 is genuinely unsubmitted.
        var synced = cell(id: "cell-1", benchmarkId: "bench-1")
        synced.serverJobId = "server-1"
        try saveManifest(
            storage: storage,
            cells: [
                synced,
                cell(id: "cell-2", benchmarkId: "bench-2"),
                cell(id: "cell-3", benchmarkId: "bench-3")
            ],
            contributeResults: true
        )
        for (cellId, benchmarkId) in [("cell-1", "bench-1"), ("cell-2", "bench-2"), ("cell-3", "bench-3")] {
            try writePayload(storage: storage, cellId: CellId(cellId), benchmarkId: benchmarkId)
        }
        try storage.results.saveSubmission(.submitted(serverJobId: "server-1"), "cell-1")
        try storage.results.saveSubmission(.submitted(serverJobId: "server-2"), "cell-2")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-3"]]))
        ])
        let uploader = uploader(submitter: submitter, storage: storage)

        let outcomes = await uploader.drainAll()

        // Only bench-3 ever hits the network; cell-2 heals from its record.
        #expect(submitter.batches == [["bench-3"]])
        #expect(outcomes["job-1"]?.submitted == 2)
        let manifest = try #require(storage.loadJobManifest(jobId: "job-1"))
        #expect(manifest.cells.map(\.serverJobId) == ["server-1", "server-2", "server-3"])

        // A follow-up pass finds nothing stranded and stays off the network.
        let secondPass = await uploader.drainAll()
        #expect(secondPass.isEmpty)
        #expect(submitter.batches.count == 1)
    }

    @Test func drainAllSkipsJobsThatNeverIntendedToSubmit() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // job-1 never opted into contribution and was never submitted
        // manually; job-2 also has no opt-in, but a failed record proves a
        // submission was attempted — only it gets re-driven.
        try saveManifest(storage: storage, jobId: "job-1", cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        try saveManifest(storage: storage, jobId: "job-2", cells: [cell(id: "cell-2", benchmarkId: "bench-2")])
        try writePayload(storage: storage, cellId: "cell-2", benchmarkId: "bench-2")
        try storage.results.saveSubmission(.failed(["network down"]), "cell-2")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-2"]]))
        ])

        let outcomes = await uploader(submitter: submitter, storage: storage).drainAll()

        #expect(submitter.batches == [["bench-2"]])
        #expect(outcomes["job-1"] == nil)
        #expect(outcomes["job-2"]?.submitted == 1)
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == nil)
        #expect(storage.loadJobManifest(jobId: "job-2")?.cells[0].serverJobId == "server-2")
    }

    @Test func drainAllSkipsRunningJobs() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try saveManifest(
            storage: storage,
            cells: [cell(id: "cell-1", benchmarkId: "bench-1")],
            status: .running,
            contributeResults: true
        )
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcomes = await uploader(submitter: submitter, storage: storage).drainAll()

        #expect(outcomes.isEmpty)
        #expect(submitter.batches.isEmpty)
    }

    @Test func drainJobForcesSubmissionWithoutContributeOptIn() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try saveManifest(storage: storage, cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([["index": 0, "job_id": "server-1"]]))
        ])

        let outcome = await uploader(submitter: submitter, storage: storage).drainJob(jobId: "job-1")

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == [])
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
    }

    /// The race PIP-257 fixes: two drains firing together (e.g. the launch
    /// drain and a job-end `drainJob` from the background executor) must
    /// serialize through the actor. The submitter's per-call delay forces the
    /// two passes to overlap in wall-clock time; if the `queueTail` chain were
    /// not serialized they would both see the cell unsubmitted and the network
    /// would be hit twice. We assert exactly one batch per cell.
    @Test func concurrentDrainsSerializeAndDoNotDoubleSubmit() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try saveManifest(
            storage: storage,
            cells: [
                cell(id: "cell-1", benchmarkId: "bench-1"),
                cell(id: "cell-2", benchmarkId: "bench-2")
            ],
            contributeResults: true
        )
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: "cell-2", benchmarkId: "bench-2")
        let submitter = ScriptedBatchSubmitter(
            steps: [
                .respond(try batchResponse([
                    ["index": 0, "job_id": "server-1"],
                    ["index": 1, "job_id": "server-2"]
                ]))
            ],
            delayNanoseconds: 50_000_000
        )
        let uploader = uploader(submitter: submitter, storage: storage)

        // Launch drain (drainAll) and a job-end drain (drainJob) firing
        // together: the second queues behind the first and finds nothing left.
        async let first = uploader.drainAll()
        async let second = uploader.drainJob(jobId: "job-1")
        _ = await (first, second)

        // Each cell submitted exactly once: one network batch, no duplicates.
        #expect(submitter.batches == [["bench-1", "bench-2"]])
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "server-1")
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[1].serverJobId == "server-2")
    }

    @Test func resendMigratesResultsSubmittedToADifferentCollector() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // The configured collector is registrationData().serverUrl. cell-1 was
        // submitted to a different collector and cell-3 to none (a legacy record
        // counts as different) — both must migrate. cell-2 is already on the
        // current collector, so it must be left untouched even though it matches.
        var acked1 = cell(id: "cell-1", benchmarkId: "bench-1")
        acked1.serverJobId = "old-1"
        var acked2 = cell(id: "cell-2", benchmarkId: "bench-2")
        acked2.serverJobId = "cur-2"
        var acked3 = cell(id: "cell-3", benchmarkId: "bench-3")
        acked3.serverJobId = "old-3"
        try saveManifest(storage: storage, cells: [acked1, acked2, acked3])
        for (cellId, benchmarkId) in [("cell-1", "bench-1"), ("cell-2", "bench-2"), ("cell-3", "bench-3")] {
            try writePayload(storage: storage, cellId: CellId(cellId), benchmarkId: benchmarkId)
        }
        try storage.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: ServerURL("https://old.example.com")),
            "cell-1")
        try storage.results.saveSubmission(
            .submitted(serverJobId: "cur-2", collector: registrationData().serverUrl),
            "cell-2")
        // cell-3: a pre-collector-tracking record — no collector recorded.
        try storage.results.saveSubmission(.submitted(serverJobId: "old-3"), "cell-3")
        let submitter = ScriptedBatchSubmitter(steps: [
            .respond(try batchResponse([
                ["index": 0, "job_id": "server-1"],
                ["index": 1, "job_id": "server-3"]
            ]))
        ])

        let outcome = await uploader(submitter: submitter, storage: storage)
            .resendForCollectorChange(benchmarkIds: ["bench-1", "bench-2", "bench-3"])

        #expect(outcome.submitted == 2)
        #expect(outcome.errors == [])
        // Different-collector and legacy results are batched; cell-2 stays put.
        #expect(submitter.batches == [["bench-1", "bench-3"]])
        let manifest = try #require(storage.loadJobManifest(jobId: "job-1"))
        #expect(manifest.cells.map(\.serverJobId) == ["server-1", "cur-2", "server-3"])
        // The migrated results now record the current collector.
        #expect(storage.results.loadSubmission("cell-1")?.collector == registrationData().serverUrl)
        #expect(storage.results.loadSubmission("cell-3")?.collector == registrationData().serverUrl)
    }

    @Test func resendSkipsBenchmarksNotOfferedByTheCollector() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(storage: storage, cells: [acked])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        try storage.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: ServerURL("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcome = await uploader(submitter: submitter, storage: storage)
            .resendForCollectorChange(benchmarkIds: ["bench-999"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors == [])
        #expect(submitter.batches.isEmpty)
    }

    @Test func resendIgnoresNeverSubmittedResults() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // No serverJobId, no submission record: never submitted anywhere, so
        // it isn't a collector migration — the drain triggers handle it.
        try saveManifest(storage: storage, cells: [cell(id: "cell-1", benchmarkId: "bench-1")])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        let submitter = ScriptedBatchSubmitter(steps: [])

        let outcome = await uploader(submitter: submitter, storage: storage)
            .resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(submitter.batches.isEmpty)
    }

    @Test func resendKeepsPriorRecordWhenMigrationFails() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(storage: storage, cells: [acked])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        try storage.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: ServerURL("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [.fail("network down")])

        let outcome = await uploader(submitter: submitter, storage: storage)
            .resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors.count == 1)
        // The failed migration must not clobber the record that proves the
        // result already lives on the old collector — it stays submitted there.
        let record = try #require(storage.results.loadSubmission("cell-1"))
        #expect(record.status == .submitted)
        #expect(record.serverJobId == "old-1")
        #expect(record.collector == ServerURL("https://old.example.com"))
        #expect(storage.loadJobManifest(jobId: "job-1")?.cells[0].serverJobId == "old-1")
    }

    @Test func resendWithoutRegistrationReportsError() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        var acked = cell(id: "cell-1", benchmarkId: "bench-1")
        acked.serverJobId = "old-1"
        try saveManifest(storage: storage, cells: [acked])
        try writePayload(storage: storage, cellId: "cell-1", benchmarkId: "bench-1")
        try storage.results.saveSubmission(
            .submitted(serverJobId: "old-1", collector: ServerURL("https://old.example.com")),
            "cell-1")
        let submitter = ScriptedBatchSubmitter(steps: [])
        let uploader = ResultUploader(
            submitResultBatch: submitter.submit,
            credentials: { throw IdentityError.privateKeyMissing },
            retryDelays: [],
            storage: storage
        )

        let outcome = await uploader.resendForCollectorChange(benchmarkIds: ["bench-1"])

        #expect(outcome.submitted == 0)
        #expect(outcome.errors.count == 1)
        #expect(submitter.batches.isEmpty)
    }

    // MARK: - Helpers

    private func uploader(
        submitter: ScriptedBatchSubmitter,
        storage: Storage,
        retryDelays: [TimeInterval] = [],
        sleeps: SleepRecorder = SleepRecorder()
    ) -> ResultUploader {
        // Capture the registration as a value: the credentials closure is a
        // nonisolated function type, so it must not call back into
        // MainActor-isolated test helpers.
        let registration = registrationData()
        let auth = AuthIdentity(
            clientId: registration.clientId, privateKeyHex: PrivateKeyHex("private-key"))
        return ResultUploader(
            submitResultBatch: submitter.submit,
            credentials: { (registration, auth) },
            retryDelays: retryDelays,
            storage: storage,
            sleep: sleeps.sleep
        )
    }

    private func saveManifest(
        storage: Storage,
        jobId: JobId = "job-1",
        cells: [JobCell],
        status: JobStatus = .completed,
        contributeResults: Bool? = nil
    ) throws {
        storage.saveJobManifest(JobManifest(
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

/// Scripted stand-in for the batch submit network call: plays back one step
/// per call and records the benchmark ids of every batch it saw. Calls past
/// the end of the script fail the test via `unexpectedNetworkCall`.
/// Explicitly `nonisolated` so `submit` matches the nonisolated
/// `BatchSubmitter` function type with no actor hop — under the project's
/// MainActor default isolation, an implicitly isolated method crashes the
/// runner when invoked through that type (seen on CI's Xcode 26.2).
private nonisolated final class ScriptedBatchSubmitter: @unchecked Sendable {
    enum Step {
        case fail(String)
        case respond(String)
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
        serverUrl: ServerURL,
        auth: AuthIdentity,
        payloadsJson: String
    ) async throws -> String {
        if delayNanoseconds > 0 {
            try? await Task.sleep(nanoseconds: delayNanoseconds)
        }
        guard let data = payloadsJson.data(using: .utf8),
              let payloads = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            throw UploaderTestError.invalidPayloadJson
        }

        let step = record(payloads.map { $0["benchmark_id"] as? String ?? "" })
        switch step {
        case .fail(let message):
            throw NSError(domain: "Test", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        case .respond(let json):
            return json
        case nil:
            throw UploaderTestError.unexpectedNetworkCall
        }
    }

    /// Synchronous so the NSLock use stays out of the async context, where
    /// it is unavailable (an error in the Swift 6 language mode).
    private func record(_ batch: [String]) -> Step? {
        lock.lock()
        defer { lock.unlock() }
        storedBatches.append(batch)
        return steps.isEmpty ? nil : steps.removeFirst()
    }
}

/// `nonisolated` for the same reason as `ScriptedBatchSubmitter`: `sleep` is
/// passed as the uploader's nonisolated `Sleeper` function type.
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
