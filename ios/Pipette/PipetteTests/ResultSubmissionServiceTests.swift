import Testing
import Foundation
@testable import Pipette

/// Pure codec checks for `CellSubmissionStatus`. The rawValues are the on-disk wire
/// strings for `submission.json`, so these lock the `submitted`/`failed` spellings
/// that keep persisted records decodable.
struct CellSubmissionStatusTests {
    @Test func rawValuesAreTheOnDiskWireStrings() {
        #expect(CellSubmissionStatus.submitted.rawValue == "submitted")
        #expect(CellSubmissionStatus.failed.rawValue == "failed")
        #expect(CellSubmissionStatus(rawValue: "submitted") == .submitted)
        #expect(CellSubmissionStatus(rawValue: "failed") == .failed)
        #expect(CellSubmissionStatus(rawValue: "bogus") == nil)
    }

    @Test func encodeDecodeRoundTrip() throws {
        for status in [CellSubmissionStatus.submitted, .failed] {
            let data = try JSONEncoder().encode(status)
            #expect(try JSONDecoder().decode(CellSubmissionStatus.self, from: data) == status)
        }
        // A bare Codable enum encodes to its JSON-string rawValue.
        #expect(String(decoding: try JSONEncoder().encode(CellSubmissionStatus.submitted), as: UTF8.self) == "\"submitted\"")
    }
}

/// Each test injects its own temporary `FileStorage`, so the suite holds no shared
/// global and runs in parallel.
@MainActor struct ResultSubmissionServiceTests {
    @Test func submitChunksBatchRequestsAndRemapsChunkLocalIndices() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let cells = [
            cell(id: "cell-1", benchmarkId: "bench-1"),
            cell(id: "cell-2", benchmarkId: "bench-2"),
            cell(id: "cell-3", benchmarkId: "bench-3")
        ]
        for cell in cells {
            try writePayload(storage: storage, cellId: cell.cellId, benchmarkId: cell.benchmarkId)
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
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            storage: storage,
            saveJobManifest: saver.save
        )

        #expect(submitter.batches == [["bench-1", "bench-2"], ["bench-3"]])
        #expect(submitter.serverUrls == ["https://collector.example.com", "https://collector.example.com"])
        #expect(submitter.clientIds == ["client-1", "client-1"])
        #expect(submitter.privateKeys == ["private-key", "private-key"])
        #expect(outcome.submitted == 3)
        #expect(outcome.errors.isEmpty)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        // Accepted results advance a location, not just a manifest field — the crate's
        // `move_result_dir`. Asserted through the real sweep, since the store's own test
        // only proves `move` works, not that submission calls it.
        for id in ["cell-1", "cell-2", "cell-3"] {
            #expect(storage.results.location(of: CellId(id)) == .remoteSynced,
                    "\(id) did not advance")
        }
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        #expect(outcome.manifest.cells[2].serverJobId == "server-3")
        #expect(saver.manifests.last?.cells[2].serverJobId == "server-3")
        #expect(try submission(cellId: "cell-1", storage: storage).serverJobId == "server-1")
        #expect(try submission(cellId: "cell-2", storage: storage).serverJobId == "server-2")
        #expect(try submission(cellId: "cell-3", storage: storage).serverJobId == "server-3")
    }

    @Test func submitWritesSubmissionRecordForAcceptedResult() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let saver = ManifestSaveRecorder()
        let responseJson = try batchResponse([["index": 0, "job_id": "server-1"]])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: { _, _, _ in responseJson },
            storage: storage,
            saveJobManifest: saver.save
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        #expect(saver.manifests.last?.cells[0].serverJobId == "server-1")
        #expect(outcome.errors.isEmpty)
        let submission = try submission(cellId: "cell-1", storage: storage)
        #expect(submission.status == .submitted)
        #expect(submission.serverJobId == "server-1")
        #expect(submission.errors.isEmpty)
    }

    @Test func submitUsesPreviouslyAcceptedSubmissionWithoutNetworkCall() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        try storage.results.saveSubmission(
            .submitted(serverJobId: "server-1"),
            acceptedCell.cellId)
        let saver = ManifestSaveRecorder()

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: { _, _, _ in
                throw TestRecorderError.unexpectedNetworkCall
            },
            storage: storage,
            saveJobManifest: saver.save
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors.isEmpty)
        #expect(outcome.manifest.cells[0].serverJobId == "server-1")
        #expect(saver.manifests.last?.cells[0].serverJobId == "server-1")
        #expect(try submission(cellId: "cell-1", storage: storage).serverJobId == "server-1")
    }

    @Test func submitRecordsTheCollectorOnAcceptedResult() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let responseJson = try batchResponse([["index": 0, "job_id": "server-1"]])

        _ = await ResultSubmissionService.submit(
            manifest: manifest(cells: [acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            submitResultBatch: { _, _, _ in responseJson },
            storage: storage
        )

        // The collector stamp is what lets a later sync tell this result apart
        // from one submitted to a different collector.
        #expect(try submission(cellId: "cell-1", storage: storage).collector == registrationData().serverUrl)
    }

    @Test func deleteSubmissionClearsRecordSoReRunSubmitsFresh() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let rerunCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: rerunCell.cellId, benchmarkId: rerunCell.benchmarkId)
        try storage.results.saveSubmission(
            .submitted(serverJobId: "server-old"),
            rerunCell.cellId)

        // The re-run wipe drops the stale record alongside nilling the
        // cell's serverJobId — otherwise the sweep above would resurrect
        // the old upload as this re-run's submission.
        storage.results.deleteSubmission(rerunCell.cellId)
        #expect(storage.results.loadSubmission(rerunCell.cellId) == nil)

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [rerunCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: { _, _, _ in
                try batchResponse([["index": 0, "job_id": "server-new"]])
            },
            storage: storage
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.manifest.cells[0].serverJobId == "server-new")
        #expect(try submission(cellId: "cell-1", storage: storage).serverJobId == "server-new")
    }

    @Test func submitKeepsFailedBatchItemsPending() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let failedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(storage: storage, cellId: failedCell.cellId, benchmarkId: failedCell.benchmarkId)
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 0, "error": "invalid benchmark"],
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [failedCell, acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            storage: storage
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: invalid benchmark"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        #expect(
            storage.results.payloadPath(of: "cell-1") != nil
        )
        #expect(try submission(cellId: "cell-1", storage: storage).status == .failed)
        #expect(try submission(cellId: "cell-2", storage: storage).serverJobId == "server-2")
    }

    @Test func submitRecordsFailedForResultOmittedFromBatchResponse() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let omittedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(storage: storage, cellId: omittedCell.cellId, benchmarkId: omittedCell.benchmarkId)
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        // Server acks index 1 but drops index 0 entirely from the response.
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [omittedCell, acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            storage: storage
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: omitted from batch response"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        // The omitted cell ends `.failed` (not silently dropped) so
        // `ResultUploader.hasStrandedResults` re-picks it next trigger.
        let omittedSubmission = try submission(cellId: "cell-1", storage: storage)
        #expect(omittedSubmission.status == .failed)
        #expect(omittedSubmission.errors == ["omitted from batch response"])
        #expect(try submission(cellId: "cell-2", storage: storage).serverJobId == "server-2")
    }

    @Test func submitRecordsFailedForBatchItemMissingJobId() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let missingCell = cell(id: "cell-1", benchmarkId: "bench-1")
        let acceptedCell = cell(id: "cell-2", benchmarkId: "bench-2")
        try writePayload(storage: storage, cellId: missingCell.cellId, benchmarkId: missingCell.benchmarkId)
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)
        // Server returns an entry for index 0 but with no `job_id`, and an
        // empty-string `job_id` would be no better — the `!isEmpty` guard
        // rejects it. Neither may be treated as accepted.
        let submitter = BatchSubmitRecorder(responses: [
            try batchResponse([
                ["index": 0, "job_id": ""],
                ["index": 1, "job_id": "server-2"]
            ])
        ])

        let outcome = await ResultSubmissionService.submit(
            manifest: manifest(cells: [missingCell, acceptedCell]),
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            batchSize: 2,
            submitResultBatch: submitter.submit,
            storage: storage
        )

        #expect(outcome.submitted == 1)
        #expect(outcome.errors == ["bench-1: missing job_id in response"])
        #expect(outcome.manifest.cells[0].serverJobId == nil)
        #expect(outcome.manifest.cells[1].serverJobId == "server-2")
        // The item ends `.failed` (not submitted, not silently dropped) so
        // `ResultUploader.hasStrandedResults` re-picks it next trigger.
        let missingSubmission = try submission(cellId: "cell-1", storage: storage)
        #expect(missingSubmission.status == .failed)
        #expect(missingSubmission.serverJobId == nil)
        #expect(missingSubmission.errors == ["missing job_id in response"])
        #expect(try submission(cellId: "cell-2", storage: storage).serverJobId == "server-2")
    }

    @Test func submitCellRecordsFailedWhenResponseHasEmptyJobId() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)

        // 2xx body with an empty-string job_id: the `!serverJobId.isEmpty`
        // guard must reject it rather than record an empty serverJobId.
        let outcome = await ResultSubmissionService.submitCell(
            jobId: "job-1",
            cellId: acceptedCell.cellId,
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            submitResult: { _, _, _ in "{\"job_id\":\"\"}" },
            storage: storage
        )

        guard case .failed = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        let record = try submission(cellId: "cell-1", storage: storage)
        #expect(record.status == .failed)
        #expect(record.serverJobId == nil)
        #expect(record.errors == ["missing job_id in response"])
    }

    @Test func submitCellRecordsFailedWhenResponseMissingJobId() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let acceptedCell = cell(id: "cell-1", benchmarkId: "bench-1")
        try writePayload(storage: storage, cellId: acceptedCell.cellId, benchmarkId: acceptedCell.benchmarkId)

        // 2xx body with no job_id: must not fabricate a synthetic serverJobId.
        let outcome = await ResultSubmissionService.submitCell(
            jobId: "job-1",
            cellId: acceptedCell.cellId,
            registration: registrationData(),
            auth: AuthIdentity(clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("private-key")),
            submitResult: { _, _, _ in "{\"status\":\"accepted\"}" },
            storage: storage
        )

        guard case .failed = outcome else {
            Issue.record("expected .failed, got \(outcome)")
            return
        }
        let record = try submission(cellId: "cell-1", storage: storage)
        #expect(record.status == .failed)
        #expect(record.serverJobId == nil)
        #expect(record.errors == ["missing job_id in response"])
    }

    /// A payload without both descriptors is refused locally rather than sent for the
    /// server to reject with nothing said here — the crate's `require_descriptors`.
    @Test(arguments: [
        // Identity fields alone, no descriptors.
        ["model_name": "org/repo", "runtime_name": "llamacpp_ios_pipette"],
        // Present but empty is the same thing.
        ["model_descriptor": "", "runtime_descriptor": ""],
        // One of the two is not enough.
        ["model_descriptor": #"{"type":"gguf_text"}"#],
        ["runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#],
    ])
    func aPayloadWithoutDescriptorsIsRefused(_ payload: [String: String]) {
        let refusal = ResultSubmissionService.descriptorRefusal(payload)

        #expect(refusal?.contains("descriptor format") == true, "was: \(refusal ?? "nil")")
    }

    /// A payload this build writes always carries both, so the guard never fires on one.
    @Test func aFreshPayloadPassesTheDescriptorGuard() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil)),
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let payload = try writtenPayload(cellId: "cell-1", storage: storage)
        #expect(ResultSubmissionService.descriptorRefusal(payload) == nil)
    }

    /// A finished run leaves `extras.json` beside its payload, as the crate's results
    /// store does — a local diagnostic artifact, never submitted. `stderr` carries what
    /// the engine wrote; an in-process run has no argv and no stdout.
    @Test func writingAPayloadLeavesTheExtrasBesideIt() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        var response = RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil))
        response.stderr = "llama_model_loader: loaded meta data\n"

        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: try ggufTextResolved()),
            response: response,
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let url = try #require(storage.results.submittableDir("cell-1"))
            .appendingPathComponent("extras.json")
        let extras = try JSONDecoder().decode(
            BenchmarkResultExtras.self, from: try Data(contentsOf: url))
        #expect(extras.stderr == "llama_model_loader: loaded meta data\n")
        #expect(extras.command.isEmpty)
        #expect(extras.stdout.isEmpty)
        #expect(extras.executable == nil)
        // The submitted payload is unchanged by any of it.
        #expect(try writtenPayload(cellId: "cell-1", storage: storage)["stderr"] == nil)
    }

    @Test func payloadBuilderStampsLlamaRuntimeIdentity() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // The flags the run reported, as `RunCell.dispatch` fills them: the cell's values with the
        // engine's own where it left a knob unset. `runtime_flags` reports these rather
        // than defaulting again, so the string cannot describe a load that did not happen.
        var response = RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil))
        response.runtimeFlags = .decodeLlamacppIosPipetteGgufText(
            numberGpuLayers: 20, ctxSize: 1024, nUbatch: 256, threads: nil, swaFull: nil)

        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: response,
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let payload = try writtenPayload(cellId: "cell-1", storage: storage)
        // The flat wire form with the axes stripped — not an argv string: an iOS engine
        // loads in-process and plan-types gives it no `raw` surface. Compared as an object,
        // since nothing sorts the keys and the server canonicalizes before storing.
        #expect(try flagKnobs(payload["runtime_flags"])
            == ["number_gpu_layers": 20, "ctx_size": 1024, "n_ubatch": 256])
        // model_descriptor is required — always present, never null.
        #expect(payload["model_descriptor"] is String)
        // runtime_descriptor: the plan-types `llamacpp_ios_pipette` descriptor — the
        // upstream repo + the built commit + the iOS flavor. Load knobs live in
        // runtime_flags, not the descriptor.
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "llamacpp_ios_pipette")
        #expect(runtimeObj["repository_url"] as? String == "github.com/ggml-org/llama.cpp")
        #expect(runtimeObj["flavor"] as? String == SubmissionRef.iosFlavor)
        // The built commit, which is where the version is stated.
        #expect(runtimeObj["repository_version"] as? String == LlamaCppBuildInfo.submissionVersion)
    }

    /// The crate refuses to record a response whose flags are not the run's
    /// (`ensure_flags_round_tripped`): a record for a run that did not happen is worse
    /// than no record. Checked on the runtime axis, the only one an iOS variant carries.
    @Test func payloadBuilderRefusesFlagsFromAnotherRuntime() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        var response = RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil))
        response.runtimeFlags = .decodeMlxIosPipetteMlx(nUbatch: 128)

        #expect(throws: RuntimeFlagsAxisMismatch(
            carried: (.decodeThroughput, .mlxIosPipette, .mlx),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            try PayloadBuilder.writeLocal(
                request: payloadRequest(model: try ggufTextResolved(),
                                        benchmarkId: "decode_throughput_512_100"),
                response: response,
                cellId: "cell-1",
                source: .remote,
                storage: storage)
        }
    }

    @Test func payloadBuilderStampsMlxRuntimeIdentity() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // MLX models are safetensors directories, not a `.gguf` file — the quant
        // isn't in the path, so it must come from the typed source.
        var response = RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil))
        response.runtimeFlags = .decodeMlxIosPipetteMlx(nUbatch: 128)

        try PayloadBuilder.writeLocal(
            request: payloadRequest(
                model: .alreadyBound(.mlx(Mlx(source: .huggingFace(
                    repo: try HFRepo.parse("mlx-community/LFM2-350M-4bit"), prefix: nil)))),
                benchmarkId: "decode_throughput_512_100"),
            response: response,
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let payload = try writtenPayload(cellId: "cell-1", storage: storage)
        // An MLX variant carries the prefill chunk alone, and it is reported: the string
        // used to elide because only llama.cpp rendered argv, so the chunk a run used
        // never reached the server. The two llama knobs stay absent.
        #expect(try flagKnobs(payload["runtime_flags"]) == ["n_ubatch": 128])
        // model_descriptor: the plan-types `mlx` directory coordinate for the run.
        let modelObj = try refObject(payload["model_descriptor"])
        #expect(modelObj["type"] as? String == "mlx")
        #expect(modelObj["source"] as? String == "huggingface")
        #expect(modelObj["org"] as? String == "mlx-community")
        #expect(modelObj["repo_name"] as? String == "LFM2-350M-4bit")
        // runtime_descriptor: the plan-types `mlx_ios_pipette` descriptor — the pinned
        // Swift-package stack (each a repo + ref) and the iOS flavor.
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "mlx_ios_pipette")
        #expect(runtimeObj["flavor"] as? String == SubmissionRef.iosFlavor)
        let packages = try #require(runtimeObj["packages"] as? [String: Any])
        let mlxSwift = try #require(packages["mlx_swift"] as? [String: Any])
        #expect(mlxSwift["repository_url"] as? String == MLXBuildInfo.mlxSwiftRepositoryUrl)
        #expect(mlxSwift["repository_version"] as? String == MLXBuildInfo.mlxSwiftVersion)
        let mlxSwiftLM = try #require(packages["mlx_swift_lm"] as? [String: Any])
        #expect(mlxSwiftLM["repository_url"] as? String == MLXBuildInfo.mlxSwiftLMRepositoryUrl)
        // repository_version is the 9-char short revision (see MLXBuildInfo).
        #expect(mlxSwiftLM["repository_version"] as? String == MLXBuildInfo.mlxSwiftLMRevision)
        let swiftTransformers = try #require(packages["swift_transformers"] as? [String: Any])
        #expect(swiftTransformers["repository_url"] as? String == MLXBuildInfo.swiftTransformersRepositoryUrl)
        #expect(swiftTransformers["repository_version"] as? String == MLXBuildInfo.swiftTransformersVersion)
    }

    @Test func payloadBuilderStampsAfmRefs() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: .alreadyBound(.appleFoundationText),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil)),
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let payload = try writtenPayload(cellId: "cell-1", storage: storage)
        // AFM: bare tagged descriptors. model_descriptor uses the *model* tag
        // `apple_foundation_text` (not the runtime tag `apple_foundation`).
        let modelObj = try refObject(payload["model_descriptor"])
        #expect(modelObj["type"] as? String == "apple_foundation_text")
        #expect(modelObj.count == 1)
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "apple_foundation")
        #expect(runtimeObj.count == 1)
    }

    @Test func payloadBuilderStampsGgufModelRefs() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // hf_gguf_text: single-file weight coordinate.
        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil)),
            cellId: "cell-text",
            source: .remote,
            storage: storage)
        let textObj = try refObject(try writtenPayload(cellId: "cell-text", storage: storage)["model_descriptor"])
        #expect(textObj["type"] as? String == "gguf_text")
        #expect(textObj["source"] as? String == "huggingface")
        #expect(textObj["org"] as? String == "LiquidAI")
        #expect(textObj["repo_name"] as? String == "LFM2-350M-GGUF")
        #expect(textObj["path"] as? String == "LFM2-350M-Q4_K_M.gguf")

        // gguf_vision: the projector filename must ride along in the descriptor.
        try PayloadBuilder.writeLocal(
            request: payloadRequest(
                model: .alreadyBound(try ggufVisionSpec("LiquidAI/LFM2.5-VL-450M-GGUF",
                                                        "LFM2.5-VL-450M-Q4_0.gguf",
                                                        "mmproj-f16.gguf")),
                benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil)),
            cellId: "cell-vision",
            source: .remote,
            storage: storage)
        let visionObj = try refObject(try writtenPayload(cellId: "cell-vision", storage: storage)["model_descriptor"])
        #expect(visionObj["type"] as? String == "gguf_vision")
        #expect(visionObj["source"] as? String == "huggingface")
        #expect(visionObj["model"] as? String == "LFM2.5-VL-450M-Q4_0.gguf")
        #expect(visionObj["mmproj"] as? String == "mmproj-f16.gguf")
    }

    /// Parse a `model_descriptor` / `runtime_descriptor` wire value (a JSON *string*) into its object.
    /// A `runtime_flags` string parsed back into its knobs — compared as an object, since
    /// nothing sorts the keys.
    private func flagKnobs(_ value: Any?) throws -> [String: Int] {
        let string = try #require(value as? String, "runtime_flags must be a JSON string")
        return try #require(
            JSONSerialization.jsonObject(with: Data(string.utf8)) as? [String: Int],
            "runtime_flags must decode to an object of numbers")
    }

    private func refObject(_ value: Any?) throws -> [String: Any] {
        let string = try #require(value as? String, "ref must be a JSON string")
        return try #require(
            JSONSerialization.jsonObject(with: Data(string.utf8)) as? [String: Any],
            "descriptor string must decode to an object")
    }

    private func writtenPayload(cellId: CellId, storage: Storage) throws -> [String: Any] {
        let data = try #require(storage.results.loadPayload(cellId))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func submission(cellId: CellId, storage: Storage) throws -> CellSubmissionRecord {
        try #require(storage.results.loadSubmission(cellId))
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
}

private final class BatchSubmitRecorder: @unchecked Sendable {
    private let responses: [String]
    nonisolated(unsafe) private var storedBatches: [[String]] = []
    nonisolated(unsafe) private var storedServerUrls: [String] = []
    nonisolated(unsafe) private var storedClientIds: [String] = []
    nonisolated(unsafe) private var storedPrivateKeys: [String] = []

    init(responses: [String]) {
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
        serverUrl: ServerURL,
        auth: AuthIdentity,
        payloadsJson: String
    ) async throws -> String {
        guard let data = payloadsJson.data(using: .utf8) else {
            throw TestRecorderError.invalidPayloadEncoding
        }
        guard let payloads = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw TestRecorderError.invalidPayloadJson
        }
        let batch = payloads.map { $0["benchmark_id"] as? String ?? "" }

        storedServerUrls.append(serverUrl.value)
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
    case invalidPayloadEncoding
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
