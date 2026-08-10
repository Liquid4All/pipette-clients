import Foundation

/// The `results` group — `list`, `show`, `delete`.
///
/// The columns and filters are the CLI's (`results list --benchmark --type --state
/// --limit`), and `show` follows its stream split exactly: the payload goes to **stdout
/// bare** so it pipes into `jq`, while the section labels go to stderr.
///
/// `show` is also the one command whose `[HEADLESS]` lines go to stderr. Everywhere else
/// they stay on stdout, because `pipette-plan`'s `run_streaming_scanning` scans *stdout
/// only* for `BENCH_DONE` — moving the sentinel wholesale would leave the iOS transport
/// unable to tell a finished run from a hung one. The transport never invokes `show`, so
/// its stdout is free to be a payload channel.
enum ResultCommands {
    /// One row of `results list`, carrying what the CLI's table shows.
    private struct Row {
        let id: ResultId
        let state: BenchmarkResultState
        let benchmarkId: String
        let benchmarkType: BenchmarkType?
        let runtime: String
        let createdAt: String
    }

    /// Every result on disk, most recent first. A cell with no `payload.json` has not
    /// produced a result yet and is not one — pending and failed cells are job state,
    /// which `jobs` reports.
    private static func rows(storage: Storage) -> [Row] {
        storage.loadAllJobManifests()
            .sorted { $0.createdAt > $1.createdAt }
            .flatMap { manifest in
                manifest.cells.compactMap { cell -> Row? in
                    let id = ResultId(jobId: manifest.jobId, cellId: cell.cellId)
                    guard let state = storage.results.state(of: id.cellId) else { return nil }
                    return Row(id: id, state: state, benchmarkId: cell.benchmarkId,
                               benchmarkType: cell.benchmarkType, runtime: cell.engineLabel,
                               createdAt: manifest.createdAt)
                }
            }
    }

    /// `results [benchmark=] [type=] [state=] [limit=]`.
    static func list(benchmark: String?, type: BenchmarkType?, state: BenchmarkResultState?,
                        limit: Int?, storage: Storage) {
        var matched = rows(storage: storage).filter { row in
            (benchmark.map { row.benchmarkId == $0 } ?? true)
                && (type.map { row.benchmarkType == $0 } ?? true)
                && (state.map { row.state == $0 } ?? true)
        }
        if let limit { matched = Array(matched.prefix(limit)) }
        HeadlessRunner.log("results count=\(matched.count)")
        for row in matched {
            HeadlessRunner.log("result result=\(row.id) state=\(row.state.rawValue) "
                + "benchmark=\(row.benchmarkId) "
                + "type=\(row.benchmarkType?.rawValue ?? "-") "
                + "runtime=\(row.runtime) created_at=\(row.createdAt)")
        }
    }

    /// `results show result=<jobId>/<cellId>`: the stored files, payload first.
    ///
    /// Mirrors the CLI, which prints `payload.json` and then `extras.json` / `metrics.json`
    /// when present, labelling each on stderr so the concatenated stdout stays parseable.
    /// iOS writes no `extras.json` yet, so only the metrics section can appear.
    static func show(id: ResultId, storage: Storage) -> Bool {
        guard storage.results.state(of: id.cellId) != nil else {
            HeadlessRunner.logDiagnostic("results show ERROR no result \(id)")
            return false
        }
        guard let payloadURL = storage.results.payloadPath(of: id.cellId),
              let payload = try? String(contentsOf: payloadURL, encoding: .utf8)
        else {
            HeadlessRunner.logDiagnostic("results show ERROR cannot read the payload for \(id)")
            return false
        }
        HeadlessRunner.emitPayload(payload)

        if let metricsURL = storage.results.metricsPath(of: id.cellId),
           let metrics = try? String(contentsOf: metricsURL, encoding: .utf8) {
            HeadlessRunner.logDiagnostic("--- metrics ---")
            HeadlessRunner.emitPayload(metrics)
        }
        return true
    }

    /// `results delete result=<jobId>/<cellId>`: remove the stored artifacts.
    ///
    /// The cell's own record stays — the job still ran that cell, and rewriting job state
    /// to hide it would make the manifest disagree with what happened.
    static func delete(id: ResultId, storage: Storage) -> Bool {
        guard storage.results.state(of: id.cellId) != nil else {
            HeadlessRunner.log("results delete ERROR no result \(id)")
            return false
        }
        // The store owns the layout; deleting by id finds the result wherever it is.
        storage.results.delete(id.cellId)
        HeadlessRunner.log("results delete removed result=\(id)")
        return true
    }
}
