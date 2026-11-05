import Foundation

/// Benchmark results on disk: the crate's `ResultsStore` (`pipette-cli/src/results/store.rs`).
///
/// ```text
/// results/
///   local/<cellId>/{payload,extras}.json            # never submitted
///   remote/pending/<cellId>/{payload,extras}.json   # awaiting the next sweep
///   remote/synced/<cellId>/{payload,extras,metrics,submission}.json
/// ```
///
/// A result is keyed by its cell id; the job is recorded in the manifest, not the path.
/// Unlike the crate, the directory is not renamed to the server job id on submission: that
/// goes in `submission.json`, which has no crate counterpart (it also carries `collector`
/// for collector-change resend). The location is the authority on status.
nonisolated struct ResultsStore: Sendable {
    /// The workspace `results/` directory.
    let root: URL

    private static let payloadName = "payload.json"
    private static let extrasName = "extras.json"
    private static let metricsName = "metrics.json"
    private static let submissionName = "submission.json"

    // MARK: - Paths

    func locationDir(_ location: BenchmarkResultLocation) -> URL {
        switch location {
        case .local: root.appendingPathComponent("local", isDirectory: true)
        case .remotePending: root.appendingPathComponent("remote/pending", isDirectory: true)
        case .remoteSynced: root.appendingPathComponent("remote/synced", isDirectory: true)
        }
    }

    /// `<location>/<id>/`, or nil when `id` is not a safe path component.
    func resultDir(_ location: BenchmarkResultLocation, _ id: CellId) -> URL? {
        let raw = id.value
        guard !raw.isEmpty, !raw.contains("/"), raw != ".", raw != ".." else { return nil }
        return locationDir(location).appendingPathComponent(raw, isDirectory: true)
    }

    func payloadPath(_ location: BenchmarkResultLocation, _ id: CellId) -> URL? {
        resultDir(location, id)?.appendingPathComponent(Self.payloadName)
    }

    func extrasPath(_ location: BenchmarkResultLocation, _ id: CellId) -> URL? {
        resultDir(location, id)?.appendingPathComponent(Self.extrasName)
    }

    func metricsPath(_ location: BenchmarkResultLocation, _ id: CellId) -> URL? {
        resultDir(location, id)?.appendingPathComponent(Self.metricsName)
    }

    /// The crate's `has_valid_metrics`. A corrupt file reads as missing, so a later refresh
    /// can replace it.
    func hasValidMetrics(_ location: BenchmarkResultLocation, _ id: CellId) -> Bool {
        guard let path = metricsPath(location, id),
              FileManager.default.fileExists(atPath: path.path)
        else { return false }
        do {
            _ = try Coding.decoder.decode(BenchmarkScoredResult.self, from: Data(contentsOf: path))
            return true
        } catch {
            AppLog.results.warning("ignoring corrupt metrics for result \(id.value): \(error)")
            return false
        }
    }

    // MARK: - Lookup

    /// Checked newest-first so a result mid-move resolves to its destination.
    func location(of id: CellId) -> BenchmarkResultLocation? {
        [.remoteSynced, .remotePending, .local].first { location in
            guard let path = payloadPath(location, id) else { return false }
            return FileManager.default.fileExists(atPath: path.path)
        }
    }

    /// As the crate's `load_list_entry` derives it. Only a synced result with a metrics file
    /// that parses is scored.
    func state(of id: CellId) -> BenchmarkResultState? {
        guard let location = location(of: id) else { return nil }
        switch location {
        case .remoteSynced: return hasValidMetrics(.remoteSynced, id) ? .scored : .submitted
        case .local, .remotePending: return .local
        }
    }

    /// The sweep's entry point. `local/` never qualifies: that result was never sanctioned.
    func submittableDir(_ id: CellId) -> URL? {
        guard let location = location(of: id), location != .local else { return nil }
        return resultDir(location, id)
    }

    /// What can still go up (the "Submit N Results" count); excludes the `local/` half,
    /// which the sweep skips. For deletion warnings use ``deletableResultCount(_:)``.
    func unsubmittedResultCount(_ manifest: JobManifest) -> Int {
        manifest.cells.filter(isUnsubmitted).count
    }

    /// Queued for upload and never acked. Shared with ``ResultUploader/hasStrandedResults(_:)``.
    func isUnsubmitted(_ cell: JobCell) -> Bool {
        cell.runStatus == .completed
            && cell.isSubmittable
            && cell.serverJobId == nil
            && submittableDir(cell.cellId) != nil
    }

    /// Completed results a device reset would destroy. Includes the `local/` half that
    /// ``unsubmittedResultCount(_:)`` skips, since `resetDeviceData()` removes all of `results/`.
    func deletableResultCount(_ manifest: JobManifest) -> Int {
        manifest.cells.filter {
            $0.runStatus == .completed && $0.serverJobId == nil && payloadPath(of: $0.cellId) != nil
        }.count
    }

    /// Stats a file per cell, so pass the manifests already in hand rather than re-reading.
    func deletableResultCount(across manifests: [JobManifest]) -> Int {
        manifests.reduce(0) { $0 + deletableResultCount($1) }
    }

    func payloadPath(of id: CellId) -> URL? {
        location(of: id).flatMap { payloadPath($0, id) }
    }

    func extrasPath(of id: CellId) -> URL? {
        location(of: id).flatMap { extrasPath($0, id) }
    }

    func metricsPath(of id: CellId) -> URL? {
        location(of: id).flatMap { metricsPath($0, id) }
    }

    // MARK: - Writes

    /// The crate's `save_result`. Not atomic: a failure between the two writes leaves the
    /// payload alone on disk.
    func saveResult(
        _ location: BenchmarkResultLocation, _ id: CellId,
        payload: Data, extras: Data
    ) throws {
        guard let dir = resultDir(location, id) else {
            throw ResultsStoreError.unsafeResultId(id.value)
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try payload.write(to: dir.appendingPathComponent(Self.payloadName), options: .atomic)
        try extras.write(to: dir.appendingPathComponent(Self.extrasName), options: .atomic)
    }

    func loadPayload(_ id: CellId) -> Data? {
        guard let location = location(of: id), let path = payloadPath(location, id) else {
            return nil
        }
        return try? Data(contentsOf: path)
    }

    /// The crate's `load_payload`: throws rather than answering nil, since a caller naming
    /// the location expects a result there.
    func loadPayload(_ location: BenchmarkResultLocation, _ id: CellId) throws -> [String: Any] {
        guard let path = payloadPath(location, id) else {
            throw ResultsStoreError.unsafeResultId(id.value)
        }
        guard let payload = try ResultPayload.readObject(at: path) else {
            throw ResultsStoreError.noResult(id.value)
        }
        return payload
    }

    /// The crate's `save_metrics`.
    func saveMetrics(
        _ location: BenchmarkResultLocation, _ id: CellId, _ scored: BenchmarkScoredResult
    ) throws {
        guard let path = metricsPath(location, id) else {
            throw ResultsStoreError.unsafeResultId(id.value)
        }
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Coding.encoder.encode(scored).write(to: path, options: .atomic)
    }

    /// The crate's `move_result_dir`: a bare rename, so a collision surfaces as an error
    /// rather than a guess about which copy to discard.
    func move(_ id: CellId, to destination: BenchmarkResultLocation) throws {
        guard let from = location(of: id), from != destination else { return }
        guard let source = resultDir(from, id), let target = resultDir(destination, id) else {
            throw ResultsStoreError.unsafeResultId(id.value)
        }
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: source, to: target)
    }

    /// Resets a re-run's progress. No crate counterpart: the CLI mints a result id per run,
    /// but a cell id is stable, so a re-run lands on top of its previous result.
    func clearProgress(_ id: CellId) {
        deleteSubmission(id)
        if let metrics = metricsPath(of: id) {
            try? FileManager.default.removeItem(at: metrics)
        }
    }

    /// Called when its job is deleted: results don't live under `jobs/<jobId>/`, so removing
    /// that tree doesn't take them.
    func delete(_ id: CellId) {
        guard let location = location(of: id), let dir = resultDir(location, id) else { return }
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: - Submission record

    func saveSubmission(_ record: CellSubmissionRecord, _ id: CellId) throws {
        guard let location = location(of: id), let dir = resultDir(location, id) else {
            throw ResultsStoreError.noResult(id.value)
        }
        try Coding.encoder.encode(record)
            .write(to: dir.appendingPathComponent(Self.submissionName), options: .atomic)
    }

    func loadSubmission(_ id: CellId) -> CellSubmissionRecord? {
        guard let location = location(of: id), let dir = resultDir(location, id),
              let data = try? Data(contentsOf: dir.appendingPathComponent(Self.submissionName))
        else { return nil }
        return try? Coding.decoder.decode(CellSubmissionRecord.self, from: data)
    }

    func deleteSubmission(_ id: CellId) {
        guard let location = location(of: id), let dir = resultDir(location, id) else { return }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent(Self.submissionName))
    }

    /// The crate's `list_ids`.
    func listIds(_ location: BenchmarkResultLocation) -> [CellId] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: locationDir(location), includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map { CellId($0.lastPathComponent) }
            .sorted { $0.value < $1.value }
    }
}

nonisolated enum ResultsStoreError: LocalizedError, Equatable {
    case unsafeResultId(String)
    case noResult(String)

    var errorDescription: String? {
        switch self {
        case .unsafeResultId(let id): return "`\(id)` is not usable as a result directory name"
        case .noResult(let id): return "no result recorded for `\(id)`"
        }
    }
}
