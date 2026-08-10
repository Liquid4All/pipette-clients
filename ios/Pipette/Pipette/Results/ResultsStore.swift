import Foundation

/// Benchmark results on disk — the crate's `ResultsStore`
/// (`pipette-cli/src/results/store.rs:29`).
///
/// A concrete filesystem handle over one directory, like ``IdentityStore`` and
/// ``BenchmarkStore``: call sites take the store, and tests point one at scratch space.
/// It replaces nine `Storage` protocol methods that returned URLs and left the IO to
/// their callers.
///
/// ```text
/// results/
///   local/<cellId>/{payload,extras}.json            # never submitted
///   remote/pending/<cellId>/{payload,extras}.json   # awaiting the next sweep
///   remote/synced/<cellId>/{payload,extras,metrics,submission}.json
/// ```
///
/// A result is keyed by its cell id — already a UUID, and already documented on
/// `JobCell` as doubling as the result directory name. The job it belonged to is
/// recorded in the manifest, not in the path, which is what lets the layout match the
/// crate's despite `jobs/` having no counterpart there.
///
/// **Two divergences, both deliberate.** The crate renames a result directory to the
/// server-assigned job id on submission; here the cell id is stable and the server id
/// goes in `submission.json`. And that file has no crate counterpart at all — it carries
/// `serverJobId`, `submittedAt`, `errors` and `collector`, the last of which exists for
/// collector-change resend, a feature the CLI does not have. The *location* is the
/// authority on status; `submission.json` holds the details of the submission event.
nonisolated struct ResultsStore: Sendable {
    /// The workspace `results/` directory.
    let root: URL

    private static let payloadName = "payload.json"
    private static let extrasName = "extras.json"
    private static let metricsName = "metrics.json"
    private static let submissionName = "submission.json"

    // MARK: - Paths

    func locationDir(_ location: BenchmarkResultLocation) -> URL {
        root.appendingPathComponent(location.directory, isDirectory: true)
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

    func metricsPath(_ location: BenchmarkResultLocation, _ id: CellId) -> URL? {
        resultDir(location, id)?.appendingPathComponent(Self.metricsName)
    }

    // MARK: - Lookup

    /// Where this result currently is, or nil when it has produced none. Checked
    /// newest-first so a result mid-move resolves to its destination.
    func location(of id: CellId) -> BenchmarkResultLocation? {
        [.remoteSynced, .remotePending, .local].first { location in
            guard let path = payloadPath(location, id) else { return false }
            return FileManager.default.fileExists(atPath: path.path)
        }
    }

    /// How far this result has travelled — the crate's location-derived ladder. `scored`
    /// outranks `submitted` even without a submission record: a score can only have come
    /// back for something the collector took.
    func state(of id: CellId) -> BenchmarkResultState? {
        guard let location = location(of: id) else { return nil }
        if let metrics = metricsPath(location, id),
           FileManager.default.fileExists(atPath: metrics.path) {
            return .scored
        }
        return location.state
    }

    /// The directory holding a submittable payload, or nil when there is none — the
    /// sweep's entry point. `local/` never qualifies: that result was never sanctioned.
    func submittableDir(_ id: CellId) -> URL? {
        guard let location = location(of: id), location != .local else { return nil }
        return resultDir(location, id)
    }

    /// Cells in `manifest` whose completed result is still on disk and has never been
    /// uploaded.
    ///
    /// Answers "what can still go up", so it backs the job detail's "Submit N Results"
    /// button. A cell from the generated `local/` catalog half is never submitted
    /// (``JobCell/isSubmittable``, and `submittableDir` is nil for `.local`), so counting it
    /// would promise to send work the sweep skips.
    ///
    /// Not the number to quote when warning about deletion — see
    /// ``deletableResultCount(_:)``.
    func unsubmittedResultCount(_ manifest: JobManifest) -> Int {
        manifest.cells.filter(isUnsubmitted).count
    }

    /// Whether this one cell's result is queued for upload and has never been acked. The single
    /// spelling of the four clauses, shared with ``ResultUploader/hasStrandedResults(_:)``, which
    /// needs the cells themselves rather than how many there are.
    func isUnsubmitted(_ cell: JobCell) -> Bool {
        cell.runStatus == .completed
            && cell.isSubmittable
            && cell.serverJobId == nil
            && submittableDir(cell.cellId) != nil
    }

    /// Cells in `manifest` holding a completed result that a device reset would destroy,
    /// whether or not it could ever have been uploaded.
    ///
    /// Deliberately not ``unsubmittedResultCount(_:)``. That count excludes the generated
    /// `local/` half, but `resetDeviceData()` removes the whole `results/` tree, so quoting
    /// it in a deletion warning promises less than the reset takes — a device whose pending
    /// work is all locally generated would be told it loses nothing.
    func deletableResultCount(_ manifest: JobManifest) -> Int {
        manifest.cells.filter {
            $0.runStatus == .completed && $0.serverJobId == nil && payloadPath(of: $0.cellId) != nil
        }.count
    }

    /// Device-wide total of ``deletableResultCount(_:)`` over `manifests`. Stats a file per
    /// cell, so pass the manifests already in hand rather than re-reading the job store.
    func deletableResultCount(across manifests: [JobManifest]) -> Int {
        manifests.reduce(0) { $0 + deletableResultCount($1) }
    }

    /// The payload / metrics file wherever the result currently is. Nil when it has
    /// none — callers that only want to read do not need to know the location.
    func payloadPath(of id: CellId) -> URL? {
        location(of: id).flatMap { payloadPath($0, id) }
    }

    func metricsPath(of id: CellId) -> URL? {
        location(of: id).flatMap { metricsPath($0, id) }
    }

    // MARK: - Writes

    /// Write both halves of a result, as the crate's `save_result` does — one call so a
    /// caller cannot forget the extras. Not atomic: a failure between the two leaves the
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

    /// Move a result to its next location — the crate's `move_result_dir`, which is a
    /// bare rename: no destination-exists special case, so a collision surfaces as an
    /// error rather than a guess about which copy to discard.
    ///
    /// A no-op when the result is already there or has none.
    func move(_ id: CellId, to destination: BenchmarkResultLocation) throws {
        guard let from = location(of: id), from != destination else { return }
        guard let source = resultDir(from, id), let target = resultDir(destination, id) else {
            throw ResultsStoreError.unsafeResultId(id.value)
        }
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: source, to: target)
    }

    /// Drop the submission record and any score, keeping the result — a re-run of the
    /// same cell has to reset how far it has travelled.
    ///
    /// No crate counterpart: the CLI mints a fresh result id per run, so a re-run is a
    /// new directory. A cell id is stable here, so a re-run lands on top of itself.
    func clearProgress(_ id: CellId) {
        deleteSubmission(id)
        if let metrics = metricsPath(of: id) {
            try? FileManager.default.removeItem(at: metrics)
        }
    }

    /// Drop a result wherever it is. Called when its job is deleted — results no longer
    /// live under `jobs/<jobId>/`, so removing that tree no longer takes them with it.
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

    /// Every result id at a location, sorted — the crate's `list_ids`.
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
