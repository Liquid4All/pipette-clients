import Foundation

struct ResultSubmissionOutcome {
    let manifest: JobManifest
    let submitted: Int
    let errors: [String]
    /// True when at least one batch request failed in transit (network or
    /// serialization) rather than being rejected per-result by the server —
    /// the signal that retrying the same payloads may succeed.
    var hadTransportFailure: Bool = false
}

enum ResultSubmissionService {
    private nonisolated static let batchSize = 1000
    typealias BatchSubmitter = (ServerURL, AuthIdentity, String) async throws -> String
    typealias ManifestSaver = @MainActor (JobManifest) -> Void

    /// Outcome of attempting to submit a single cell's payload.
    enum CellOutcome {
        /// Uploaded successfully; carries the server-assigned job id.
        case submitted(serverJobId: String)
        /// Nothing to submit (no local payload for this cell).
        case skipped
        /// The upload was attempted and failed; carries a human-readable reason.
        case failed(String)
    }

    /// Sweep a manifest and upload every completed cell that hasn't been
    /// submitted yet. Used by the end-of-job auto-submit pass. Safe to call
    /// repeatedly: cells already carrying a `serverJobId` are skipped, so it
    /// never double-submits.
    ///
    /// `resubmitForCollectorChange` switches the sweep to migration mode: it
    /// re-sends completed results recorded against a collector that differs from
    /// `registration.serverUrl` — a result with no recorded collector counts as
    /// different, so legacy submissions re-sync — and skips everything already on
    /// the current collector. `allowedBenchmarkIds`, when non-nil, restricts the
    /// sweep to cells whose `benchmarkId` is in the set (the currently-synced
    /// catalog). Every accepted result records the collector it went to.
    static func submit(
        manifest input: JobManifest,
        registration: IdentityRegistration,
        auth: AuthIdentity,
        batchSize requestedBatchSize: Int = Self.batchSize,
        resubmitForCollectorChange: Bool = false,
        allowedBenchmarkIds: Set<String>? = nil,
        submitResultBatch: BatchSubmitter? = nil,
        storage: Storage,
        saveJobManifest: ManifestSaver? = nil
    ) async -> ResultSubmissionOutcome {
        let saveJobManifest = saveJobManifest ?? { storage.saveJobManifest($0) }
        guard ResultSubmissionFeatureGate.canSubmitResults(registration: registration) else {
            return ResultSubmissionOutcome(
                manifest: input,
                submitted: 0,
                errors: ["Result submission requires registration."]
            )
        }

        var manifest = input
        let effectiveBatchSize = Swift.max(1, requestedBatchSize)
        let submitBatch: BatchSubmitter = submitResultBatch ?? { serverUrl, auth, payloadsJson in
            try await ManagementClient.submitResultBatch(
                serverUrl: serverUrl,
                auth: auth,
                payloadsJson: payloadsJson
            )
        }
        var submitCount = 0
        var errors: [String] = []
        var hadTransportFailure = false
        var cellIndexes: [Int] = []
        var payloads: [[String: Any]] = []

        // A collector-change resend must not overwrite a good prior record on
        // failure — that would erase which collector already holds the result,
        // and the next refresh retries anyway. A normal sweep records the
        // failure so `ResultUploader.hasStrandedResults` re-picks the cell.
        func markFailed(_ messages: [String], cellId: CellId) {
            guard !resubmitForCollectorChange else { return }
            try? storage.results.saveSubmission(.failed(messages), cellId)
        }

        for i in manifest.cells.indices {
            let cell = manifest.cells[i]
            guard cell.runStatus == .completed else { continue }
            // A locally-generated benchmark was never sanctioned by the server, so its
            // result is not ours to publish — the crate keeps such results in
            // `results/local/`, which `sync` never walks.
            guard cell.isSubmittable else { continue }
            if let allowedBenchmarkIds, !allowedBenchmarkIds.contains(cell.benchmarkId) { continue }
            // A normal sweep skips already-acked cells before any disk I/O;
            // migration mode must still inspect them to compare collectors.
            if !resubmitForCollectorChange, cell.serverJobId != nil { continue }

            guard let resultDir = storage.results.submittableDir(cell.cellId) else {
                continue
            }

            let record = storage.results.loadSubmission(cell.cellId)
            // The server ack this result already carries, if any — from the
            // manifest, or a submission record the manifest never adopted.
            let priorServerJobId = cell.serverJobId ?? recordServerJobId(record)

            if resubmitForCollectorChange {
                // Migration: re-send results submitted to a collector other than
                // the one we're configured for now. A result with no recorded
                // collector is treated as a different one, so legacy submissions
                // re-sync to the current collector.
                guard priorServerJobId != nil,
                      !CollectorEndpoint.isSameCollector(record?.collector?.value, as: registration.serverUrl.value)
                else { continue }
                // fall through — re-send to the current collector
            } else if let priorServerJobId {
                // Heal a cell whose submission record was accepted but never
                // adopted into the manifest, without re-sending.
                manifest.cells[i].serverJobId = priorServerJobId
                saveJobManifest(manifest)
                submitCount += 1
                continue
            }

            let payloadPath = resultDir.appendingPathComponent("payload.json")
            guard let data = try? Data(contentsOf: payloadPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                errors.append("\(cell.benchmarkId): failed to read payload")
                continue
            }
            if let refusal = descriptorRefusal(json) {
                errors.append("\(cell.benchmarkId): \(refusal)")
                continue
            }

            cellIndexes.append(i)
            payloads.append(json)
        }

        for batchStart in stride(from: 0, to: payloads.count, by: effectiveBatchSize) {
            let batchEnd = Swift.min(batchStart + effectiveBatchSize, payloads.count)
            let payloadBatch = Array(payloads[batchStart..<batchEnd])

            do {
                let payloadData = try JSONSerialization.data(withJSONObject: payloadBatch)
                guard let payloadsJson = String(data: payloadData, encoding: .utf8) else {
                    throw NSError(
                        domain: "Pipette",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "failed to serialize payloads"]
                    )
                }
                let responseJson = try await submitBatch(registration.serverUrl, auth, payloadsJson)
                guard let responseData = responseJson.data(using: .utf8),
                      let response = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
                      let results = response["results"] as? [[String: Any]]
                else {
                    throw NSError(
                        domain: "Pipette",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "invalid batch response"]
                    )
                }

                var seenBatchIndices = Set<Int>()
                for result in results {
                    guard let batchIndex = result["index"] as? Int,
                          payloadBatch.indices.contains(batchIndex)
                    else {
                        errors.append("unknown batch item: invalid response index")
                        continue
                    }
                    seenBatchIndices.insert(batchIndex)

                    let resultIndex = batchStart + batchIndex
                    let cellIndex = cellIndexes[resultIndex]
                    let cell = manifest.cells[cellIndex]

                    if let error = result["error"] as? String {
                        errors.append("\(cell.benchmarkId): \(error)")
                        markFailed([error], cellId: cell.cellId)
                        continue
                    }
                    guard let serverJobId = result["job_id"] as? String, !serverJobId.isEmpty else {
                        let message = "missing job_id in response"
                        errors.append("\(cell.benchmarkId): \(message)")
                        markFailed([message], cellId: cell.cellId)
                        continue
                    }

                    do {
                        try storage.results.saveSubmission(
                            .submitted(serverJobId: serverJobId, collector: registration.serverUrl),
                            cell.cellId)
                        // Accepted, so the result advances a location — the crate's
                        // `move_result_dir`, which is what makes the directory the status.
                        try storage.results.move(cell.cellId, to: .remoteSynced)
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

                // Any submitted payload the server neither acked nor rejected
                // is stranded: without a `.failed` record it gets no
                // serverJobId, `hadTransportFailure` stays false, and the next
                // trigger never re-picks it (see `ResultUploader.hasStrandedResults`).
                // Record it as failed so the cell is retried.
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

    /// Upload a single cell's local payload and record submission state.
    /// This is pure file + network work — the caller owns the manifest and is
    /// responsible for recording the returned serverJobId.
    /// Kept independent of `JobManifest` so it can run off the job's run loop
    /// without sharing mutable state.
    typealias SingleSubmitter = (ServerURL, AuthIdentity, String) async throws -> String

    /// Why a stored payload cannot be submitted, or `nil` when it can — the crate's
    /// `require_descriptors`.
    ///
    /// Both submit paths forward the on-disk payload verbatim, so one without descriptors
    /// would reach the server, be rejected there, and say nothing here. `PayloadBuilder`
    /// types both non-optional, so this refuses a payload it did not write.
    static func descriptorRefusal(_ payload: [String: Any]) -> String? {
        func present(_ key: String) -> Bool {
            guard let value = payload[key] as? String else { return false }
            return !value.isEmpty
        }
        guard present("model_descriptor"), present("runtime_descriptor") else {
            return "recorded before the model/runtime descriptor format and can't be "
                + "submitted; re-run the benchmark or discard the pending result"
        }
        return nil
    }

    static func submitCell(
        jobId: JobId,
        cellId: CellId,
        registration: IdentityRegistration,
        auth: AuthIdentity,
        submitResult: SingleSubmitter? = nil,
        storage: Storage
    ) async -> CellOutcome {
        guard ResultSubmissionFeatureGate.canSubmitResults(registration: registration) else {
            return .failed("Result submission requires registration.")
        }

        let submit: SingleSubmitter = submitResult ?? { serverUrl, auth, payloadJson in
            try await ManagementClient.submitResult(
                serverUrl: serverUrl,
                auth: auth,
                payloadJson: payloadJson
            )
        }

        guard let resultDir = storage.results.submittableDir(cellId)
        else {
            return .skipped
        }

        if let serverJobId = submittedServerJobId(jobId: jobId, cellId: cellId, storage: storage) {
            return .submitted(serverJobId: serverJobId)
        }

        let payloadPath = resultDir.appendingPathComponent("payload.json")
        guard let data = try? Data(contentsOf: payloadPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let payloadData = try? JSONSerialization.data(withJSONObject: json),
              let payloadStr = String(data: payloadData, encoding: .utf8)
        else {
            return .failed("failed to read payload")
        }
        if let refusal = descriptorRefusal(json) {
            try? storage.results.saveSubmission(.failed([refusal]), cellId)
            return .failed(refusal)
        }

        let responseJson: String
        do {
            responseJson = try await submit(registration.serverUrl, auth, payloadStr)
        } catch {
            let message = formatJobError(error, contextSize: 0)
            try? storage.results.saveSubmission(.failed([message]), cellId)
            return .failed(message)
        }

        // A 2xx with no job_id is not a success: defaulting serverJobId to
        // cellId fabricates an ack the server never gave, saves `.submitted`,
        // and the cell is never retried. Mirror the batch path — record
        // `.failed` so the result stays pending for the next attempt.
        guard let responseData = responseJson.data(using: .utf8),
              let response = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let serverJobId = response["job_id"] as? String,
              !serverJobId.isEmpty
        else {
            let message = "missing job_id in response"
            try? storage.results.saveSubmission(.failed([message]), cellId)
            return .failed(message)
        }

        do {
            try storage.results.saveSubmission(
                .submitted(serverJobId: serverJobId, collector: registration.serverUrl),
                cellId
            )
            // Accepted, so the result advances a location, as the batch path does.
            try storage.results.move(cellId, to: .remoteSynced)
            return .submitted(serverJobId: serverJobId)
        } catch {
            return .failed("submitted as \(serverJobId) but recording it locally failed")
        }
    }

    /// The server ack a submission record carries, or nil unless it is an
    /// accepted record with a non-empty `serverJobId`.
    private static func recordServerJobId(_ record: CellSubmissionRecord?) -> String? {
        guard let record, record.status == .submitted,
              let serverJobId = record.serverJobId, !serverJobId.isEmpty
        else { return nil }
        return serverJobId
    }

    private static func submittedServerJobId(jobId: JobId, cellId: CellId, storage: Storage) -> String? {
        recordServerJobId(storage.results.loadSubmission(cellId))
    }
}
