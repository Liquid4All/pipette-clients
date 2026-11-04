import Foundation

/// The `results` group, with the CLI's columns and filters. `show` puts the payload alone
/// on stdout so it pipes into `jq`, and its `[HEADLESS]` lines on stderr
/// (`HeadlessRunner.logDiagnostic` says why only payload commands do this).
enum ResultsCommands {
    /// The crate's `ResultRow`, plus the type iOS already prints.
    struct ResultRow: Equatable {
        let resultId: ResultId
        let state: BenchmarkResultState
        let benchmark: String
        let benchmarkType: BenchmarkType?
        let runtime: String
        let createdAt: String
        /// The catalog half, so `benchmark=` filters by the address the crate's
        /// `--benchmark` takes. Not printed: the BENCHMARK column shows the recorded id.
        let source: BenchmarkSource

        /// The address this row answers `benchmark=` on. `nil` when the recorded id is not a
        /// reference, which then matches no filter, as in the crate.
        var reference: SourcedBenchmarkId? {
            (try? SourcedBenchmarkId(parsing: benchmark)).map {
                SourcedBenchmarkId(source: source, id: $0.id)
            }
        }
    }

    /// The crate's `load_list_entry` fallback for a payload that carries no timestamp.
    private static let epoch = "1970-01-01T00:00:00Z"

    /// Every result on disk, most recent first. A cell with no `payload.json` is job state,
    /// which `jobs` reports. The manifest supplies the job half of the address, which the
    /// payload does not carry.
    static func allRows(ws: PipetteWorkspace) -> [ResultRow] {
        ws.loadAllJobManifests()
            .flatMap { manifest in
                manifest.cells.compactMap { cell -> ResultRow? in
                    guard let state = ws.results.state(of: cell.cellId) else { return nil }
                    let payload = readPayload(cell.cellId, ws: ws)
                    return ResultRow(
                        resultId: ResultId(jobId: manifest.jobId, cellId: cell.cellId),
                        state: state,
                        benchmark: payload?["benchmark_id"] as? String ?? cell.benchmarkId,
                        benchmarkType: cell.benchmarkType,
                        runtime: runtimeColumn(payload?["runtime_descriptor"] as? String),
                        createdAt: payload?["submitted_at"] as? String ?? epoch,
                        source: cell.benchmarkSource ?? .remote)
                }
            }
            .sorted { $0.createdAt > $1.createdAt }
    }

    private static func readPayload(_ id: CellId, ws: PipetteWorkspace) -> [String: Any]? {
        do {
            return try ws.results.payloadPath(of: id).flatMap { try ResultPayload.readObject(at: $0) }
        } catch {
            AppLog.results.error("payload parse failed for result \(id.value): \(error)")
            return nil
        }
    }

    /// Falls back to the raw descriptor when it does not decode: a listing must never fail
    /// on one odd record.
    static func runtimeColumn(_ descriptor: String?) -> String {
        guard let descriptor, !descriptor.isEmpty else { return "" }
        guard let runtime = try? Coding.decoder.decode(Runtime.self, from: Data(descriptor.utf8))
        else { return descriptor }
        return runtime.cliRef
    }

    /// Split out so the filtering is testable without capturing stdout. `benchmark=`
    /// compares the parsed address, so `local/x` and `remote/x` are different filters and an
    /// id recorded with the side inline still matches.
    static func rows(benchmark: SourcedBenchmarkId?, benchmarkType: BenchmarkType?,
                     state: BenchmarkResultState?, limit: Int?,
                     ws: PipetteWorkspace) -> [ResultRow] {
        let matched = allRows(ws: ws).filter { row in
            (benchmark.map { row.reference == $0 } ?? true)
                && (benchmarkType.map { row.benchmarkType == $0 } ?? true)
                && (state.map { row.state == $0 } ?? true)
        }
        return limit.map { Array(matched.prefix($0)) } ?? matched
    }

    /// `runtime=` is the last field because an MLX ref carries spaces.
    static func list(benchmark: SourcedBenchmarkId?, benchmarkType: BenchmarkType?,
                     state: BenchmarkResultState?, limit: Int?, ws: PipetteWorkspace) {
        let matched = rows(benchmark: benchmark, benchmarkType: benchmarkType, state: state,
                           limit: limit, ws: ws)
        HeadlessRunner.log("results count=\(matched.count)")
        for row in matched {
            HeadlessRunner.log("result result=\(row.resultId) state=\(row.state.rawValue) "
                + "benchmark=\(row.benchmark) "
                + "type=\(row.benchmarkType?.rawValue ?? "-") "
                + "created_at=\(row.createdAt) runtime=\(row.runtime)")
        }
    }

    struct StoredFile: Equatable {
        /// `nil` for the payload, which prints unlabelled.
        let label: String?
        let contents: String
    }

    /// In the crate's order. Nil when there is no result or its payload won't read.
    static func storedFiles(resultId: ResultId, ws: PipetteWorkspace) -> [StoredFile]? {
        guard let payloadURL = ws.results.payloadPath(of: resultId.cellId),
              let payload = try? String(contentsOf: payloadURL, encoding: .utf8)
        else { return nil }
        var files = [StoredFile(label: nil, contents: payload)]
        for (label, url) in [("--- extras ---", ws.results.extrasPath(of: resultId.cellId)),
                             ("--- metrics ---", ws.results.metricsPath(of: resultId.cellId))] {
            if let url, let contents = try? String(contentsOf: url, encoding: .utf8) {
                files.append(StoredFile(label: label, contents: contents))
            }
        }
        return files
    }

    /// Each label goes to stderr so the concatenated stdout stays parseable.
    static func show(resultId: ResultId, ws: PipetteWorkspace) -> Bool {
        guard ws.results.state(of: resultId.cellId) != nil else {
            HeadlessRunner.logDiagnostic("results show ERROR no result \(resultId)")
            return false
        }
        guard let files = storedFiles(resultId: resultId, ws: ws) else {
            HeadlessRunner.logDiagnostic("results show ERROR cannot read the payload for \(resultId)")
            return false
        }
        for file in files {
            if let label = file.label { HeadlessRunner.logDiagnostic(label) }
            HeadlessRunner.emitPayload(file.contents)
        }
        return true
    }

    /// The cell's record stays: the job still ran that cell, and hiding it would make the
    /// manifest disagree with what happened.
    static func delete(resultId: ResultId, ws: PipetteWorkspace) -> Bool {
        guard let location = ws.results.location(of: resultId.cellId) else {
            HeadlessRunner.log("results delete ERROR no result \(resultId)")
            return false
        }
        ws.results.delete(resultId.cellId)
        HeadlessRunner.log("results delete removed result=\(resultId) location=\(location.label)")
        return true
    }
}
