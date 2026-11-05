import Foundation

// What a finished run prints on stdout, as `run_summary_lines` and `write_record_done` do
// in `crates/pipette-cli/src/commands/benchmarks.rs`. `devicectl --console` does not carry
// `AppLog`, so this is where an operator sees what a cell measured and where it was filed.

/// The crate's `run_summary_lines`, field for field and format for format, so the same cell
/// reads the same whichever client ran it.
///
/// The type is the definition's, never the result data's: `vl_max_memory` reports through
/// the `maxMemoryUsage` variant it shares with the text cell.
nonisolated func runSummaryLines(benchmarkId: String, benchmarkType: BenchmarkType,
                                 data: BenchmarkResultData) -> [String] {
    var lines = ["benchmark: \(benchmarkId)", "type: \(benchmarkType.rawValue)"]
    func push(_ key: String, _ value: Double, _ places: Int) {
        lines.append("\(key): \(fixed(value, places))")
    }
    func corpusLines(_ prefix: String, _ turns: UInt32, _ totals: CorpusTotals) {
        lines.append("\(prefix)_turns: \(turns)")
        push("\(prefix)_prompt_ms", totals.promptMs, 3)
        push("\(prefix)_predicted_ms", totals.predictedMs, 3)
        push("\(prefix)_ttft_ms", totals.ttftMs, 3)
        push("\(prefix)_decode_ms", totals.decodeMs, 3)
        push("\(prefix)_e2e_ms", totals.e2eMs, 3)
        lines.append("\(prefix)_generated_tokens: \(totals.generatedTokens)")
    }
    switch data {
    case let .prefillThroughput(ms, stddev):
        push("prefill_time_ms", ms, 6)
        stddev.map { push("prefill_time_ms_stddev", $0, 6) }
    case let .decodeThroughput(ms, stddev):
        push("decode_time_ms", ms, 6)
        stddev.map { push("decode_time_ms_stddev", $0, 6) }
    case let .endToEndLatency(ms, stddev):
        push("total_time_ms", ms, 6)
        stddev.map { push("total_time_ms_stddev", $0, 6) }
    case let .maxMemoryUsage(host, gpu, npu):
        lines.append("max_host_bytes: \(host)")
        gpu.map { lines.append("max_gpu_bytes: \($0)") }
        npu.map { lines.append("max_npu_bytes: \($0)") }
    case let .eval(completions):
        lines.append("completions: \(completions.count)")
    case let .vlThroughput(promptTokens, imageTokens, promptMs, promptStddev,
                           predictedMs, predictedStddev, totalTimeMs, totalStddev):
        lines.append("prompt_tokens: \(promptTokens)")
        imageTokens.map { lines.append("image_tokens: \($0)") }
        push("prompt_ms", promptMs, 3)
        promptStddev.map { push("prompt_ms_stddev", $0, 3) }
        push("predicted_ms", predictedMs, 3)
        predictedStddev.map { push("predicted_ms_stddev", $0, 3) }
        totalTimeMs.map { push("total_time_ms", $0, 3) }
        totalStddev.map { push("total_time_ms_stddev", $0, 3) }
    // Counters in the wire's names and order: the server derives every metric from them.
    // No speedup or identity rate: those need the other run's row.
    case let .vlSpecDecode(turns, totals, drafted, accepted, acceptLengthSum):
        corpusLines("vl_spec_decode", turns, totals)
        drafted.map { lines.append("vl_spec_decode_drafted_tokens: \($0)") }
        accepted.map { lines.append("vl_spec_decode_accepted_tokens: \($0)") }
        acceptLengthSum.map { push("vl_spec_decode_turn_accept_length_sum", $0, 3) }
    case let .asrThroughput(promptTokens, audioTokens, promptMs, promptStddev,
                            predictedMs, predictedStddev, totalTimeMs, totalStddev):
        lines.append("asr_prompt_tokens: \(promptTokens)")
        lines.append("asr_audio_tokens: \(audioTokens)")
        push("asr_prompt_ms", promptMs, 3)
        promptStddev.map { push("asr_prompt_ms_stddev", $0, 3) }
        push("asr_predicted_ms", predictedMs, 3)
        predictedStddev.map { push("asr_predicted_ms_stddev", $0, 3) }
        totalTimeMs.map { push("total_time_ms", $0, 3) }
        totalStddev.map { push("total_time_ms_stddev", $0, 3) }
    }
    return lines
}

/// Rust's `{:.N}`. The unlocalized `String(format:)`: these lines are read beside the CLI's,
/// and a comma decimal separator would not compare.
private nonisolated func fixed(_ value: Double, _ places: Int) -> String {
    String(format: "%.\(places)f", value)
}

/// One completed cell's filing, read from where the store put it.
nonisolated struct ResultReport: Equatable, Sendable {
    let cellId: CellId
    let benchmarkId: String
    /// Nil for a cell that failed before recording.
    let location: BenchmarkResultLocation?
    let payloadPath: String?
    /// The collector's job id, for a synced result.
    let serverJobId: String?
    /// Both nil for a cell whose payload could not be read back: no summary, but the filing
    /// line still reports where it landed.
    let benchmarkType: BenchmarkType?
    let resultData: BenchmarkResultData?

    init(cellId: CellId, benchmarkId: String, location: BenchmarkResultLocation?,
         payloadPath: String?, serverJobId: String? = nil,
         benchmarkType: BenchmarkType? = nil, resultData: BenchmarkResultData? = nil) {
        self.cellId = cellId
        self.benchmarkId = benchmarkId
        self.location = location
        self.payloadPath = payloadPath
        self.serverJobId = serverJobId
        self.benchmarkType = benchmarkType
        self.resultData = resultData
    }

    /// The location comes from the store, not the cell: the directory is the status
    /// (`ResultsStore`). The data is read back from the filed payload, so the summary
    /// describes what was recorded. The type comes from the cell (see `runSummaryLines`).
    init(cell: JobCell, store: ResultsStore) {
        let location = store.location(of: cell.cellId)
        let payload = location.flatMap { store.payloadPath($0, cell.cellId) }
        self.init(cellId: cell.cellId, benchmarkId: cell.benchmarkId, location: location,
                  payloadPath: payload?.path,
                  serverJobId: location == .remoteSynced
                      ? store.loadSubmission(cell.cellId)?.serverJobId : nil,
                  benchmarkType: cell.benchmarkType,
                  resultData: payload.flatMap(Self.readResult))
    }

    /// Nil for a missing or unparseable payload, reported by its filing line alone rather
    /// than failing the report: the run already happened.
    private static func readResult(at url: URL) -> BenchmarkResultData? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(BenchmarkResultData.self, from: data)
    }

    var summaryLines: [String] {
        guard let benchmarkType, let resultData else { return [] }
        return runSummaryLines(benchmarkId: benchmarkId, benchmarkType: benchmarkType,
                                data: resultData)
    }

    /// The location reads as the crate's `label()` prints it.
    var line: String {
        var parts = ["result cell=\(cellId.value)", "benchmark=\(benchmarkId)",
                     "location=\(location?.label ?? "none")"]
        if let payloadPath { parts.append("payload=\(payloadPath)") }
        if let serverJobId { parts.append("submitted_as_job=\(serverJobId)") }
        return parts.joined(separator: " ")
    }
}

nonisolated enum ResultReporter {
    /// Why a run that asked to submit did not, in the terms of
    /// `JobExecutor.shouldAutoSubmit`. `nil` when nothing stood in the way.
    static func submitBlocker(registered: Bool, online: Bool) -> String? {
        if !registered {
            return "not registered: run `headlessrun auth register organization=… contact-email=…`"
        }
        if !online { return "offline" }
        return nil
    }

    /// Summary before filing, as the crate orders them. A cell left in `remotePending` after
    /// a submitting run is named: otherwise a skipped or refused upload looks like a run
    /// that never asked.
    static func lines(reports: [ResultReport], submitRequested: Bool,
                      blocker: String?, errors: [String]) -> [String] {
        var out = reports.flatMap { $0.summaryLines + [$0.line] }
        guard submitRequested else { return out }
        let synced = reports.filter { $0.location == .remoteSynced }
        if !synced.isEmpty {
            out.append("result submitted cells=\(synced.count)")
        }
        // A local-only result is never offered, so it is not a failure.
        let pending = reports.filter { $0.location == .remotePending }
        guard !pending.isEmpty else { return out }
        if let blocker {
            out.append("result submit SKIPPED \(pending.count) pending: \(blocker)")
        } else if errors.isEmpty {
            out.append("result submit INCOMPLETE \(pending.count) pending")
        } else {
            out.append("result submit FAILED \(pending.count) pending: "
                + errors.joined(separator: "; "))
        }
        return out
    }
}
