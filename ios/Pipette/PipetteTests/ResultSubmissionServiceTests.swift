import Testing
import Foundation
@testable import Pipette

@MainActor struct ResultSubmissionServiceTests {
    @Test func submitChunksBatchRequestsAndRemapsChunkLocalIndices() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let cells = [
            cell(id: "cell-1", benchmarkId: "bench-1"),
            cell(id: "cell-2", benchmarkId: "bench-2"),
            cell(id: "cell-3", benchmarkId: "bench-3")
        ]
        for cell in cells {
            try writePayload(ws: ws, cellId: cell.cellId, benchmarkId: cell.benchmarkId)
        }

        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 1, "job_id": "server-2"],
                ["index": 0, "job_id": "server-1"]
            ]),
            try batchResponse([
                ["index": 0, "job_id": "server-3"]
            ])
        ])
        let saver = ManifestSaveRecorder()

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: cells),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            ws: ws,
            saveJobManifest: saver.save
        )

        #expect(submitter.batches == [["bench-1", "bench-2"], ["bench-3"]])
        #expect(submitter.serverUrls == ["https://collector.example.com", "https://collector.example.com"])
        #expect(submitter.clientIds == ["client-1", "client-1"])
        #expect(submitter.privateKeys == [testPrivateKeyHex, testPrivateKeyHex])
        #expect(outcome.submitted == 3)
        #expect(outcome.errors.isEmpty)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        // The crate's `move_result_dir`. The store's own test proves `move` works, not that
        // submission calls it.
        for id in ["cell-1", "cell-2", "cell-3"] {
            #expect(ws.results.location(of: CellId(id)) == .remoteSynced,
                    "\(id) did not advance")
        }
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        #expect(outcome.manifest.cells[2].serverJobId == "server-3")
        #expect(saver.manifests.last?.cells[2].serverJobId == "server-3")
        #expect(try submission(cellId: "cell-1", ws: ws).serverJobId == "server-1")
        #expect(try submission(cellId: "cell-2", ws: ws).serverJobId == "server-2")
        #expect(try submission(cellId: "cell-3", ws: ws).serverJobId == "server-3")
    }

    @Test func submitWritesSubmissionRecordForAcceptedResult() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let saver = ManifestSaveRecorder()
        let responseJson = try batchResponse([["index": 0, "job_id": "server-1"]])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [acceptedCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: { _, _, _ in responseJson },
            ws: ws,
            saveJobManifest: saver.save
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        #expect(saver.manifests.last?.cells[0].serverJobId == "server-1")
        #expect(outcome.errors.isEmpty)
        let submission = try submission(cellId: "cell-1", ws: ws)
        #expect(submission.status == .submitted)
        #expect(submission.serverJobId == "server-1")
        #expect(submission.errors.isEmpty)
        // Lets a later sync tell this result from one submitted to another collector.
        #expect(submission.collector == (try registrationData().serverUrl))
    }

    @Test func submitUsesPreviouslyAcceptedSubmissionWithoutNetworkCall() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        try ws.results.saveSubmission(
            .submitted(serverJobId: "server-1"),
            acceptedCell.cellId)
        let saver = ManifestSaveRecorder()

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [acceptedCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: { _, _, _ in
                throw TestRecorderError.unexpectedNetworkCall
            },
            ws: ws,
            saveJobManifest: saver.save
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors.isEmpty)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        #expect(saver.manifests.last?.cells[0].serverJobId == "server-1")
        #expect(try submission(cellId: "cell-1", ws: ws).serverJobId == "server-1")
    }

    @Test func deleteSubmissionClearsRecordSoReRunSubmitsFresh() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let rerunCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: rerunCell.cellId, benchmarkId: rerunCell.benchmarkId)
        try ws.results.saveSubmission(
            .submitted(serverJobId: "server-old"),
            rerunCell.cellId)

        // Otherwise the sweep would resurrect the old upload as this re-run's submission.
        ws.results.deleteSubmission(rerunCell.cellId)
        #expect(ws.results.loadSubmission(rerunCell.cellId) == nil)

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [rerunCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: { _, _, _ in
                try batchResponse([["index": 0, "job_id": "server-new"]])
            },
            ws: ws
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.manifest.cells[0].serverJobId == "server-new")
        #expect(try submission(cellId: "cell-1", ws: ws).serverJobId == "server-new")
    }

    @Test func submitKeepsFailedBatchItemsPending() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let failedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(ws: ws, cellId: failedCell.cellId, benchmarkId: failedCell.benchmarkId)
        try writePayload(ws: ws, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 0, "error": "invalid benchmark"],
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [failedCell, acceptedCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            ws: ws
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: invalid benchmark"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        #expect(
            ws.results.payloadPath(of: "cell-1") != nil
        )
        #expect(try submission(cellId: "cell-1", ws: ws).status == .failed)
        #expect(try submission(cellId: "cell-2", ws: ws).serverJobId == "server-2")
    }

    @Test func submitRecordsFailedForResultOmittedFromBatchResponse() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let omittedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(ws: ws, cellId: omittedCell.cellId, benchmarkId: omittedCell.benchmarkId)
        try writePayload(ws: ws, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [omittedCell, acceptedCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            ws: ws
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: omitted from batch response"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        // `.failed`, not dropped, so `ResultUploader.hasStrandedResults` re-picks it.
        let omittedSubmission = try submission(cellId: "cell-1", ws: ws)
        #expect(omittedSubmission.status == .failed)
        #expect(omittedSubmission.errors == ["omitted from batch response"])
        #expect(try submission(cellId: "cell-2", ws: ws).serverJobId == "server-2")
    }

    @Test func submitRecordsFailedForBatchItemMissingJobId() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let missingCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(ws: ws, cellId: missingCell.cellId, benchmarkId: missingCell.benchmarkId)
        try writePayload(ws: ws, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        // An empty `job_id` counts as missing.
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 0, "job_id": ""],
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [missingCell, acceptedCell]),
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            ws: ws
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: missing job_id in response"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        // `.failed`, not dropped, so `ResultUploader.hasStrandedResults` re-picks it.
        let missingSubmission = try submission(cellId: "cell-1", ws: ws)
        #expect(missingSubmission.status == .failed)
        #expect(missingSubmission.serverJobId == nil)
        #expect(missingSubmission.errors == ["missing job_id in response"])
        #expect(try submission(cellId: "cell-2", ws: ws).serverJobId == "server-2")
    }

    private func submitPending(
        acking submitResult: @escaping ResultSubmissionService.SingleSubmitter
    ) async throws -> (ResultSubmissionService.CellOutcome, CellSubmissionRecord) {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "bench-1")

        let outcome = await ResultSubmissionService.submitPendingResult(
            jobId: "job-1",
            cellId: "cell-1",
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            submitResult: submitResult,
            ws: ws
        )
        return (outcome, try submission(cellId: "cell-1", ws: ws))
    }

    @Test func submitPendingResultRecordsFailedWhenResponseHasEmptyJobId() async throws {
        let (outcome, record) = try await submitPending { _, _, _ in SubmitResponse(jobId: "") }

        guard case .failed = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        #expect(record.status == .failed)
        #expect(record.serverJobId == nil)
        #expect(record.errors == ["missing job_id in response"])
    }

    /// ``SubmitResponse`` requires `job_id`, so the refusal is the decode failing.
    @Test func submitPendingResultRecordsFailedWhenResponseMissingJobId() async throws {
        let (outcome, record) = try await submitPending { _, _, _ in
            try Coding.decoder.decode(
                SubmitResponse.self, from: Data("{\"status\":\"accepted\"}".utf8))
        }

        guard case .failed = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        #expect(record.status == .failed)
        #expect(record.serverJobId == nil)
    }

    /// Refused locally rather than sent for the server to reject silently: the crate's
    /// `require_descriptors_rejects_the_pre_descriptor_shape`.
    @Test(arguments: [
        ["model_name": "org/repo", "runtime_name": "llamacpp_ios_pipette"],
        ["model_descriptor": "", "runtime_descriptor": ""],
        ["model_descriptor": #"{"type":"gguf_text"}"#],
        ["runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#],
    ])
    func requireDescriptorsRejectsThePreDescriptorShape(_ payload: [String: String]) {
        #expect(throws: ResultSubmissionError.recordedBeforeDescriptors(resultId: "r1")) {
            try requireDescriptors("r1", payload)
        }
        let message = ResultSubmissionError.recordedBeforeDescriptors(resultId: "r1")
            .localizedDescription
        #expect(message.contains("descriptor format"), "was: \(message)")
    }

    /// The server never sanctioned a local benchmark, so its result is not published: the
    /// crate routes it to `results/local/`, which `sync` never walks.
    @Test func theSubmitSweepSkipsLocalCells() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var local = cell(id: "local-1", benchmarkId: "prefill_throughput_8")
        local.benchmarkSource = .local
        var remote = cell(id: "remote-1", benchmarkId: "prefill_throughput_512")
        remote.benchmarkSource = .remote
        for c in [local, remote] {
            try writePayload(ws: ws, cellId: c.cellId, benchmarkId: c.benchmarkId)
        }
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([["index": 0, "job_id": "server-1"]])
        ])

        _ = await ResultSubmissionService.submit(
            manifest: manifest(cells: [local, remote]),
            registration: try registrationData(),
            auth: try testIdentity(),
            submitResultBatch: submitter.submit,
            ws: ws)

        #expect(submitter.batches == [["prefill_throughput_512"]])
    }

    private func submission(cellId: CellId, ws: PipetteWorkspace) throws -> CellSubmissionRecord {
        try #require(ws.results.loadSubmission(cellId))
    }

    private func manifest(cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .completed
        )
    }

    /// A `claim=` run's results answer the server job `pipette-ios-proxy` claimed.
    @Test(arguments: [nil, "job-claimed"] as [String?])
    func anUploadEchoesTheJobsClaim(claim: String?) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let claimed = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(ws: ws, cellId: claimed.cellId, benchmarkId: claimed.benchmarkId)
        let sent = SentPayloads()
        let response = try batchResponse([["index": 0, "job_id": claim ?? "server-1"]])
        var job = manifest(cells: [claimed])
        job.claimJobId = claim

        let outcome = await ResultSubmissionService.submit(
            manifest: job,
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            submitResultBatch: { _, _, payloads in
                sent.append(payloads)
                return response
            },
            ws: ws,
            saveJobManifest: ManifestSaveRecorder().save
        )

        #expect(outcome.submitted == 1)
        let bodies = try sent.payloads.map {
            try #require(try JSONSerialization.jsonObject(with: $0) as? [String: Any])
        }
        #expect(bodies.count == 1)
        #expect(bodies.first?["job_id"] as? String == claim)
    }

    /// A collector migration re-sends an old result to a server that never leased the
    /// claim, so that upload carries none.
    @Test func aCollectorMigrationSendsNoClaim() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var claimed = cell(id: "cell-1", benchmarkId: "bench-1")
        claimed.serverJobId = "job-claimed"
        try writePayload(ws: ws, cellId: claimed.cellId, benchmarkId: claimed.benchmarkId,
                         at: .remoteSynced)
        try ws.results.saveSubmission(
            .submitted(serverJobId: "job-claimed", collector: try ServerUrl("https://old.example.com")),
            claimed.cellId)
        let sent = SentPayloads()
        let response = try batchResponse([["index": 0, "job_id": "server-1"]])
        var job = manifest(cells: [claimed])
        job.claimJobId = "job-claimed"

        let outcome = await ResultSubmissionService.submit(
            manifest: job,
            registration: try registrationData(),
            auth: ClientIdentity(clientId: try ClientId("client-1"), privateKeyHex: try PrivateKey(testPrivateKeyHex)),
            resubmitForCollectorChange: true,
            submitResultBatch: { _, _, payloads in
                sent.append(payloads)
                return response
            },
            ws: ws,
            saveJobManifest: ManifestSaveRecorder().save
        )

        #expect(outcome.submitted == 1)
        let bodies = try sent.payloads.map {
            try #require(try JSONSerialization.jsonObject(with: $0) as? [String: Any])
        }
        #expect(bodies.count == 1)
        #expect(bodies.first?["job_id"] == nil)
    }

    // MARK: - The drain tolerates one bad result

    @Test func aCorruptPendingPayloadDoesNotStopTheDrain() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let bad = cell(id: "cell-1-bad", benchmarkId: "bench-1")
        let good = cell(id: "cell-2-good", benchmarkId: "bench-2")
        try ws.results.saveResult(
            .remotePending, bad.cellId, payload: Data("{ not json".utf8), extras: Data("{}".utf8))
        try writePayload(ws: ws, cellId: good.cellId, benchmarkId: good.benchmarkId)
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([["index": 0, "job_id": "server-1"]])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [bad, good]),
            registration: try registrationData(),
            auth: try testIdentity(),
            submitResultBatch: submitter.submit,
            ws: ws)

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: failed to read payload"])
        #expect(submitter.batches == [["bench-2"]])
        #expect(ws.results.location(of: good.cellId) == .remoteSynced)
        #expect(ws.results.location(of: bad.cellId) == .remotePending)
    }

    // MARK: - Completion-id pre-flight

    /// As the crate's `submit_pending_result` repairs them: the last entry for each id wins.
    @Test func duplicateCompletionIdsAreRepairedBeforeSubmit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let eval = cell(id: "cell-1", benchmarkId: "eval_smoke")
        try ws.results.saveResult(.remotePending, eval.cellId, payload: JSONSerialization.data(withJSONObject: [
            "benchmark_id": "eval_smoke",
            "model_descriptor": #"{"type":"gguf_text"}"#,
            "runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#,
            "completions": [
                ["id": "a", "completion": "first"],
                ["id": "b", "completion": "only"],
                ["id": "a", "completion": "second"],
            ],
        ] as [String: Any]), extras: Data("{}".utf8))
        let capture = PayloadCapture()

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [eval]),
            registration: try registrationData(),
            auth: try testIdentity(),
            submitResultBatch: { _, _, payloadsJson in
                await capture.record(payloadsJson)
                return try batchResponse([["index": 0, "job_id": "server-1"]])
            },
            ws: ws)

        #expect(outcome.submitted == 1)
        let sent = try #require(await capture.payloads.first)
        let completions = try #require(sent["completions"] as? [[String: Any]])
        #expect(completions.compactMap { $0["id"] as? String } == ["b", "a"])
        #expect(completions.last?["completion"] as? String == "second")
    }

    @Test func aCompletionWithoutAnIdIsRefusedBeforeSubmit() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let eval = cell(id: "cell-1", benchmarkId: "eval_smoke")
        try ws.results.saveResult(.remotePending, eval.cellId, payload: JSONSerialization.data(withJSONObject: [
            "benchmark_id": "eval_smoke",
            "model_descriptor": #"{"type":"gguf_text"}"#,
            "runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#,
            "completions": [["completion": "no id"]],
        ] as [String: Any]), extras: Data("{}".utf8))
        let capture = PayloadCapture()

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [eval]),
            registration: try registrationData(),
            auth: try testIdentity(),
            submitResultBatch: { _, _, payloadsJson in
                await capture.record(payloadsJson)
                return try batchResponse([])
            },
            ws: ws)

        #expect(outcome.submitted == 0)
        #expect(outcome.errors.first?.contains("pre-flight repair failed for result cell-1") == true,
                "\(outcome.errors)")
        #expect(await capture.calls == 0)
    }
}

private actor PayloadCapture {
    private(set) var payloads: [[String: Any]] = []
    private(set) var calls = 0

    func record(_ payloadsJson: [Data]) {
        calls += 1
        payloads += payloadsJson.compactMap {
            (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any]
        }
    }
}

private final class SentPayloads: @unchecked Sendable {
    nonisolated(unsafe) private(set) var payloads: [Data] = []

    func append(_ batch: [Data]) {
        payloads += batch
    }
}

private final class BatchSubmitRecorder: @unchecked Sendable {
    private let responses: [BatchSubmitResponse]
    nonisolated(unsafe) private var storedBatches: [[String]] = []
    nonisolated(unsafe) private var storedServerUrls: [String] = []
    nonisolated(unsafe) private var storedClientIds: [String] = []
    nonisolated(unsafe) private var storedPrivateKeys: [String] = []

    init(responses: [BatchSubmitResponse]) {
        self.responses = responses
    }

    var batches: [[String]] {
        storedBatches
    }

    var serverUrls: [String] {
        storedServerUrls
    }

    var clientIds: [String] {
        storedClientIds
    }

    var privateKeys: [String] {
        storedPrivateKeys
    }

    func submit(
        server: ServerUrl,
        auth: ClientIdentity,
        payloadsJson: [Data]
    ) async throws -> BatchSubmitResponse {
        let payloads = try payloadsJson.map { payload -> [String: Any] in
            guard let object = try JSONSerialization.jsonObject(with: payload) as? [String: Any]
            else { throw TestRecorderError.invalidPayloadJson }
            return object
        }
        let batch = payloads.map { $0["benchmark_id"] as? String ?? "" }

        storedServerUrls.append(server.value)
        storedClientIds.append(auth.clientId.value)
        storedPrivateKeys.append(auth.privateKeyHex.value)
        storedBatches.append(batch)
        let responseIndex = storedBatches.count - 1

        guard responses.indices.contains(responseIndex) else {
            throw TestRecorderError.missingResponse
        }
        let response = responses[responseIndex]
        return response
    }
}

private enum TestRecorderError: Error {
    case invalidPayloadJson
    case missingResponse
    case unexpectedNetworkCall
}

private final class ManifestSaveRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storedManifests: [JobManifest] = []

    var manifests: [JobManifest] {
        lock.lock()
        defer { lock.unlock() }
        return storedManifests
    }

    func save(_ manifest: JobManifest) {
        lock.lock()
        storedManifests.append(manifest)
        lock.unlock()
    }

}
