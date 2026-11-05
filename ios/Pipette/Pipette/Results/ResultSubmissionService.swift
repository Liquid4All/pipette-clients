import Foundation

struct ResultSubmissionOutcome {
    let manifest: JobManifest
    let submitted: Int
    let errors: [String]
    /// A batch failed in transit rather than being rejected by the server, so a retry may
    /// succeed.
    var hadTransportFailure: Bool = false
}

enum ResultSubmissionService {
    private nonisolated static let batchSize = 1000
    /// `POST /benchmarks/batch`, injectable so a test drives the sweep without a network.
    typealias BatchSubmitter = (ServerUrl, ClientIdentity, [Data]) async throws
        -> BatchSubmitResponse
    typealias ManifestSaver = @MainActor (JobManifest) -> Void

    enum CellOutcome {
        case submitted(serverJobId: String)
        /// No submittable payload for this cell.
        case skipped
        case failed(String)
    }

    /// Uploads every completed, unacked cell. Idempotent: cells with a `serverJobId` are
    /// skipped. `resubmitForCollectorChange` instead re-sends results recorded against a
    /// different collector than `registration.serverUrl` (no recorded collector counts as
    /// different).
    static func submit(
        manifest input: JobManifest,
        registration: IdentityRegistration,
        auth: ClientIdentity,
        batchSize requestedBatchSize: Int = Self.batchSize,
        resubmitForCollectorChange: Bool = false,
        allowedBenchmarkIds: Set<String>? = nil,
        submitResultBatch: BatchSubmitter? = nil,
        ws: PipetteWorkspace,
        saveJobManifest: ManifestSaver? = nil
    ) async -> ResultSubmissionOutcome {
        let saveJobManifest = saveJobManifest ?? { ws.saveJobManifest($0) }
        guard ResultSubmissionFeatureGate.canSubmitResults(registration: registration) else {
            return ResultSubmissionOutcome(
                manifest: input,
                submitted: 0,
                errors: ["Result submission requires registration."]
            )
        }

        var manifest = input
        let effectiveBatchSize = Swift.max(1, requestedBatchSize)
        let submitBatch: BatchSubmitter = submitResultBatch ?? { server, auth, payloadsJson in
            try await PipetteMgmtClient.ClientApi.submitResultBatch(
                server: server,
                http: .shared,
                auth: auth,
                payloads: payloadsJson
            )
        }
        var submitCount = 0
        var errors: [String] = []
        var hadTransportFailure = false
        var cellIndexes: [Int] = []
        var payloads: [[String: Any]] = []

        // A resend must not overwrite a good prior record on failure: that would erase which
        // collector holds the result. A normal sweep records the failure so
        // `ResultUploader.hasStrandedResults` re-picks the cell.
        func markFailed(_ messages: [String], cellId: CellId) {
            guard !resubmitForCollectorChange else { return }
            try? ws.results.saveSubmission(.failed(messages), cellId)
        }

        for i in manifest.cells.indices {
            let cell = manifest.cells[i]
            guard cell.runStatus == .completed else { continue }
            // A locally generated benchmark was never sanctioned by the server.
            guard cell.isSubmittable else { continue }
            if let allowedBenchmarkIds, !allowedBenchmarkIds.contains(cell.benchmarkId) { continue }
            // A normal sweep skips already-acked cells before any disk I/O;
            // migration mode must still inspect them to compare collectors.
            if !resubmitForCollectorChange, cell.serverJobId != nil { continue }

            guard let resultDir = ws.results.submittableDir(cell.cellId) else {
                continue
            }

            let record = ws.results.loadSubmission(cell.cellId)
            // From the manifest, or a submission record the manifest never adopted.
            let priorServerJobId = cell.serverJobId ?? recordServerJobId(record)

            if resubmitForCollectorChange {
                guard priorServerJobId != nil,
                      !CollectorEndpoint.isSameCollector(record?.collector?.value, as: registration.serverUrl.value)
                else { continue }
            } else if let priorServerJobId {
                // Adopt an accepted record the manifest missed, without re-sending.
                manifest.cells[i].serverJobId = priorServerJobId
                saveJobManifest(manifest)
                submitCount += 1
                continue
            }

            let payloadPath = resultDir.appendingPathComponent("payload.json")
            guard let data = try? Data(contentsOf: payloadPath),
                  var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                errors.append("\(cell.benchmarkId): failed to read payload")
                continue
            }
            do {
                try requireDescriptors(cell.cellId.value, json)
                try repairCompletionIds(cell.cellId.value, &json)
            } catch {
                errors.append("\(cell.benchmarkId): \(error.localizedDescription)")
                continue
            }
            // The claim echo, as the crate worker's `attach_claim_to_success_payload` adds it.
            // Not to another collector: the claim was that server's job.
            if !resubmitForCollectorChange, let claimJobId = manifest.claimJobId {
                json["job_id"] = claimJobId
            }

            cellIndexes.append(i)
            payloads.append(json)
        }

        for batchStart in stride(from: 0, to: payloads.count, by: effectiveBatchSize) {
            let batchEnd = Swift.min(batchStart + effectiveBatchSize, payloads.count)
            let payloadBatch = Array(payloads[batchStart..<batchEnd])

            do {
                let payloadsJson = try payloadBatch.map {
                    try JSONSerialization.data(withJSONObject: $0)
                }
                let response = try await submitBatch(registration.serverUrl, auth, payloadsJson)

                var seenBatchIndices = Set<Int>()
                for result in response.results {
                    let batchIndex = result.index
                    guard payloadBatch.indices.contains(batchIndex) else {
                        errors.append("unknown batch item: invalid response index")
                        continue
                    }
                    seenBatchIndices.insert(batchIndex)

                    let resultIndex = batchStart + batchIndex
                    let cellIndex = cellIndexes[resultIndex]
                    let cell = manifest.cells[cellIndex]

                    if let error = result.error {
                        errors.append("\(cell.benchmarkId): \(error)")
                        markFailed([error], cellId: cell.cellId)
                        continue
                    }
                    guard let serverJobId = result.jobId, !serverJobId.isEmpty else {
                        let message = "missing job_id in response"
                        errors.append("\(cell.benchmarkId): \(message)")
                        markFailed([message], cellId: cell.cellId)
                        continue
                    }

                    do {
                        try ws.results.saveSubmission(
                            .submitted(serverJobId: serverJobId, collector: registration.serverUrl),
                            cell.cellId)
                        try ws.results.move(cell.cellId, to: .remoteSynced)
                    } catch {
                        manifest.cells[cellIndex].serverJobId = serverJobId
                        saveJobManifest(manifest)
                        submitCount += 1
                        errors.append("\(cell.benchmarkId): submitted as \(serverJobId) but recording it locally failed")
                        continue
                    }

                    manifest.cells[cellIndex].serverJobId = serverJobId
                    saveJobManifest(manifest)
                    submitCount += 1
                }

                // A payload the server neither acked nor rejected would otherwise never be
                // re-picked by `ResultUploader.hasStrandedResults`.
                for batchIndex in payloadBatch.indices where !seenBatchIndices.contains(batchIndex) {
                    let cell = manifest.cells[cellIndexes[batchStart + batchIndex]]
                    let message = "omitted from batch response"
                    errors.append("\(cell.benchmarkId): \(message)")
                    markFailed([message], cellId: cell.cellId)
                }
            } catch {
                hadTransportFailure = true
                for resultIndex in batchStart..<batchEnd {
                    let cell = manifest.cells[cellIndexes[resultIndex]]
                    let message = "batch submit failed: \(formatJobError(error, contextSize: 0))"
                    errors.append("\(cell.benchmarkId): \(message)")
                    markFailed([message], cellId: cell.cellId)
                }
            }
        }

        return ResultSubmissionOutcome(
            manifest: manifest,
            submitted: submitCount,
            errors: errors,
            hadTransportFailure: hadTransportFailure
        )
    }

    /// `POST /benchmarks`, injectable like ``BatchSubmitter``.
    typealias SingleSubmitter = (ServerUrl, ClientIdentity, Data) async throws -> SubmitResponse

    /// Independent of `JobManifest` so it can run off the job's run loop: the caller records
    /// the returned `serverJobId`.
    static func submitPendingResult(
        jobId: JobId,
        cellId: CellId,
        registration: IdentityRegistration,
        auth: ClientIdentity,
        submitResult: SingleSubmitter? = nil,
        ws: PipetteWorkspace
    ) async -> CellOutcome {
        guard ResultSubmissionFeatureGate.canSubmitResults(registration: registration) else {
            return .failed("Result submission requires registration.")
        }

        let submit: SingleSubmitter = submitResult ?? { server, auth, payloadJson in
            try await PipetteMgmtClient.ClientApi.submitResult(
                server: server,
                http: .shared,
                auth: auth,
                payloadJson: payloadJson
            )
        }

        guard let resultDir = ws.results.submittableDir(cellId)
        else {
            return .skipped
        }

        if let serverJobId = submittedServerJobId(jobId: jobId, cellId: cellId, ws: ws) {
            return .submitted(serverJobId: serverJobId)
        }

        let payloadPath = resultDir.appendingPathComponent("payload.json")
        guard let data = try? Data(contentsOf: payloadPath),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return .failed("failed to read payload")
        }
        do {
            try requireDescriptors(cellId.value, json)
            try repairCompletionIds(cellId.value, &json)
        } catch {
            let refusal = error.localizedDescription
            try? ws.results.saveSubmission(.failed([refusal]), cellId)
            return .failed(refusal)
        }
        guard let payloadJson = try? JSONSerialization.data(withJSONObject: json) else {
            return .failed("failed to read payload")
        }

        let response: SubmitResponse
        do {
            response = try await submit(registration.serverUrl, auth, payloadJson)
        } catch {
            let message = formatJobError(error, contextSize: 0)
            try? ws.results.saveSubmission(.failed([message]), cellId)
            return .failed(message)
        }

        // A 2xx with an empty job_id is not an ack: record `.failed` so the result is retried.
        let serverJobId = response.jobId
        guard !serverJobId.isEmpty else {
            let message = "missing job_id in response"
            try? ws.results.saveSubmission(.failed([message]), cellId)
            return .failed(message)
        }

        do {
            try ws.results.saveSubmission(
                .submitted(serverJobId: serverJobId, collector: registration.serverUrl),
                cellId
            )
            try ws.results.move(cellId, to: .remoteSynced)
            return .submitted(serverJobId: serverJobId)
        } catch {
            return .failed("submitted as \(serverJobId) but recording it locally failed")
        }
    }

    private static func recordServerJobId(_ record: CellSubmissionRecord?) -> String? {
        guard let record, record.status == .submitted,
              let serverJobId = record.serverJobId, !serverJobId.isEmpty
        else { return nil }
        return serverJobId
    }

    private static func submittedServerJobId(jobId: JobId, cellId: CellId, ws: PipetteWorkspace) -> String? {
        recordServerJobId(ws.results.loadSubmission(cellId))
    }
}

/// Why a stored result could not be offered to the collector.
nonisolated enum ResultSubmissionError: LocalizedError, Equatable {
    /// The crate's `require_descriptors` refusal, worded as it words it.
    case recordedBeforeDescriptors(resultId: String)
    /// The completion-id repair refused the payload (an entry with no id).
    case preflightRepairFailed(resultId: String, reason: String)
    /// The crate's `ensure_descriptor` refusals.
    case missingDescriptor(field: String)
    case invalidDescriptor(field: String, reason: String)
    case clientExtrasTooLarge(bytes: Int)

    var errorDescription: String? {
        switch self {
        case .recordedBeforeDescriptors(let resultId):
            return "result \(resultId) was recorded before the model/runtime descriptor "
                + "format and can't be submitted; re-run the benchmark or discard the "
                + "pending result"
        case let .preflightRepairFailed(resultId, reason):
            return "pre-flight repair failed for result \(resultId): \(reason)"
        case let .missingDescriptor(field):
            return "result is missing `\(field)`; re-run the benchmark (records predating "
                + "the descriptor format can't be submitted)"
        case let .invalidDescriptor(field, reason):
            return "result `\(field)` is not valid JSON: \(reason)"
        case let .clientExtrasTooLarge(bytes):
            return "client_extras is \(bytes) bytes, over the \(maxClientExtrasBytes) the "
                + "server accepts"
        }
    }
}
