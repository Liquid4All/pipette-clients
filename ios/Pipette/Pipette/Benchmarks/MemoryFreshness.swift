import Foundation

/// Everything a public build says about Max Memory accuracy, in one place. Without the
/// reclaim-ring flush (`ProcessMemory.flushReclaimRing`, a private call) a memory cell is exact
/// only as the first cell since launch, and only closing the app in the App Switcher starts a
/// new process. See docs/methodology/peak-memory-ios.md.
nonisolated enum MemoryFreshness {
    /// Where a run starts: the New Job screen, or a rerun of existing cells.
    enum RunKind { case newJob, rerun }

    struct Notice: Equatable {
        let message: String
        let isWarning: Bool
    }

    static func isMemory(_ type: BenchmarkType?) -> Bool {
        type == .maxMemoryUsage || type == .vlMaxMemory || type == .asrMaxMemory
    }

    /// `JobCell.measuredAfterOtherWork` for a cell about to run; `nil` unless a memory cell on a
    /// public build.
    static func record(
        for type: BenchmarkType?, ranEngineWork: Bool,
        publicBuild: Bool = !BuildFlavor.hasPrivateMemory
    ) -> Bool? {
        publicBuild && isMemory(type) ? ranEngineWork : nil
    }

    private static let freshStart =
        "close Pipette in the App Switcher (swipe up on it), then open it again"

    static let pickerNote = "Accurate only as the first benchmark after a fresh start: \(freshStart)."

    static let alertTitle = "Max Memory can read high"

    /// A rerun stays silent when its lone memory cell will be accurate; a new job says so.
    static func runNotice(
        _ kind: RunKind, publicBuild: Bool, memoryCells: Int, totalCells: Int,
        ranEngineWork: Bool
    ) -> Notice? {
        guard publicBuild, memoryCells > 0 else { return nil }
        // Cells run by model and context size, so with any other cell in the run a memory cell
        // may not be first.
        guard memoryCells == 1, totalCells == 1 else {
            switch kind {
            case .newJob:
                return Notice(
                    message: "Max Memory is accurate only as the first benchmark after a fresh "
                        + "start, and this job includes other benchmarks. For accurate numbers, "
                        + "run each Max Memory test as a job of its own, with no other benchmarks "
                        + "(one model, one quant, one context size), right after a fresh start: "
                        + "\(freshStart).",
                    isWarning: true)
            case .rerun:
                return Notice(
                    message: "Max Memory is accurate only as the first benchmark after a fresh "
                        + "start, and this rerun includes other cells. For accurate numbers, rerun "
                        + "each Max Memory cell on its own (select only that cell), right after a "
                        + "fresh start: \(freshStart).",
                    isWarning: true)
            }
        }
        if ranEngineWork {
            return Notice(
                message: "Pipette has already run a benchmark since it started, so this Max "
                    + "Memory result can read high. For an accurate one, \(freshStart), and run "
                    + "this before anything else.",
                isWarning: true)
        }
        switch kind {
        case .newJob:
            return Notice(
                message: "Max Memory is accurate only as the first benchmark after a fresh start, "
                    + "which this run is if you start it now.",
                isWarning: false)
        case .rerun:
            return nil
        }
    }

    /// `runNotice` for `cells` about to run in this app process, when it warns.
    static func runWarning(_ kind: RunKind, cells: [JobCell]) -> String? {
        let notice = runNotice(
            kind, publicBuild: !BuildFlavor.hasPrivateMemory,
            memoryCells: cells.filter { isMemory($0.benchmarkType) }.count,
            totalCells: cells.count, ranEngineWork: ProcessMemory.hasRunEngineWork)
        guard let notice, notice.isWarning else { return nil }
        return notice.message
    }

    /// `measuredAfterOtherWork` is `nil` on a result recorded before the field existed.
    static func resultNote(
        type: BenchmarkType?, completed: Bool, measuredAfterOtherWork: Bool?, publicBuild: Bool
    ) -> String? {
        guard completed, isMemory(type) else { return nil }
        switch measuredAfterOtherWork {
        case true?:
            return "Measured after other benchmarks since Pipette started, so this number can "
                + "read high. For an accurate one, \(freshStart), and rerun this cell before "
                + "anything else."
        case false?:
            return nil
        case nil:
            return publicBuild
                ? "Accurate only if it was the first benchmark after a fresh start of Pipette."
                : nil
        }
    }

    /// Whether a cell in a run's results table gets the "can read high" marker.
    static func flagsResult(_ cell: JobCell) -> Bool {
        cell.runStatus == .completed && isMemory(cell.benchmarkType)
            && cell.measuredAfterOtherWork == true
    }

    /// The key under a run's results table, when any result there carries the marker.
    static func tableNote(cells: [JobCell]) -> String? {
        cells.contains(where: flagsResult)
            ? "Can read high: measured after other benchmarks since Pipette started. Open the "
                + "cell for details."
            : nil
    }
}
