import Foundation

// What a finished run answers with on stdout — the counterpart of `print_record_done` in
// `crates/pipette-cli/src/commands/benchmarks.rs`.
//
// Upstream prints the payload path, `recorded <id> (<location>)`, and `submitted as job
// <id>` when the collector accepted it. A headless run printed none of that: a result the
// run never even offered to the collector looked identical to one that was submitted, and
// the reason lived in `AppLog`, which `devicectl --console` does not carry.

/// One completed cell's filing, read from where the store put it.
nonisolated struct ResultReport: Equatable, Sendable {
    let cellId: CellId
    let benchmarkId: String
    /// Nil for a cell that produced no payload — a run that failed before recording.
    let location: BenchmarkResultLocation?
    let payloadPath: String?

    init(cellId: CellId, benchmarkId: String, location: BenchmarkResultLocation?,
         payloadPath: String?) {
        self.cellId = cellId
        self.benchmarkId = benchmarkId
        self.location = location
        self.payloadPath = payloadPath
    }

    /// Takes the location from the store rather than deriving a second answer from the
    /// cell: the directory *is* the status, which is the rule `ResultsStore` is built on.
    init(cell: JobCell, store: ResultsStore) {
        let location = store.location(of: cell.cellId)
        self.init(cellId: cell.cellId, benchmarkId: cell.benchmarkId, location: location,
                  payloadPath: location.flatMap { store.payloadPath($0, cell.cellId)?.path })
    }

    /// `result cell=… benchmark=… location=… [payload=…]`, one line per cell.
    var line: String {
        var parts = ["result cell=\(cellId.value)", "benchmark=\(benchmarkId)",
                     "location=\(location?.rawValue ?? "none")"]
        if let payloadPath { parts.append("payload=\(payloadPath)") }
        return parts.joined(separator: " ")
    }
}

nonisolated enum ResultReporter {
    /// Why a run that asked to submit did not — the terms of `JobExecutor.shouldAutoSubmit`,
    /// named rather than silently skipped. `nil` when nothing stood in the way.
    static func submitBlocker(registered: Bool, online: Bool) -> String? {
        if !registered { return "not registered: run `headlessrun register`" }
        if !online { return "offline" }
        return nil
    }

    /// The lines a finished run prints: one per completed cell, then the submission
    /// outcome when the run asked for one.
    ///
    /// A cell left in `remotePending` after a submitting run is the case worth naming: it
    /// means the upload was skipped or refused, which is otherwise indistinguishable from
    /// a run that never asked.
    static func lines(reports: [ResultReport], submitRequested: Bool,
                      blocker: String?, errors: [String]) -> [String] {
        var out = reports.map(\.line)
        guard submitRequested else { return out }
        let synced = reports.filter { $0.location == .remoteSynced }
        if !synced.isEmpty {
            out.append("result submitted cells=\(synced.count)")
        }
        // A local-only result is not a failure to report: the crate files it under
        // `local/` and never offers it either.
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
