import Foundation

// MARK: - Job manifest (persisted as JSON in jobs/{jobId}/manifest.json)

nonisolated struct JobManifest: Codable, Sendable {
    let jobId: JobId
    let createdAt: String
    // Stamped on every save; absent means version 1. See `JobManifestSchema` for when to
    // bump it.
    var schemaVersion: Int?
    var nGpuLayers: Int
    var contextSize: Int
    // Prefill batch for every cell: llama's `n_ubatch` and MLX's prefill chunk. `nil`
    // means 512; see `effectivePrefillBatch`.
    var prefillBatch: Int?
    // llama.cpp's `n_threads`/`n_threads_batch`. Absent means the engine derives it from
    // the P-core count and reports what it used, so "unset" stays distinct from a number.
    var threads: Int?
    var cells: [JobCell]
    var status: JobStatus
    // Whether this eligible internal job should upload completed results
    // automatically once all runnable cells finish. Optional for older manifests.
    var contributeResults: Bool?
    // User-chosen name; nil or blank shows the default title.
    var title: String?
    // Why a .paused job paused when the user didn't pause it (auto-pause on backgrounding).
    // Shown on the paused job page; nil for user pauses and cleared on resume/completion.
    var pausedReason: String?
    // The server job a headless `claim=` run answers; every upload of this job's
    // results echoes it as `job_id`. Nil for any other job.
    var claimJobId: String?

    var totalCells: Int { cells.count }
    var completedCells: Int { cells.filter { $0.runStatus == .completed }.count }
    var failedCells: Int { cells.filter { $0.runStatus == .failed }.count }
    var cancelledCells: Int { cells.filter { $0.runStatus == .cancelled }.count }
    var submittedCells: Int { cells.filter { $0.serverJobId != nil }.count }

    var modelNames: [String] {
        Array(Set(cells.map { $0.modelName })).sorted()
    }

    var benchmarkIds: [String] {
        Array(Set(cells.map { $0.benchmarkId })).sorted()
    }

    var runtimeLabel: String {
        Set(cells.map { $0.engineLabel }).sorted().joined(separator: " + ")
    }

    var effectivePrefillBatch: Int { prefillBatch ?? 512 }

    /// User-facing job title. Falls back to a date/model/benchmark summary
    /// when the user hasn't set one (or cleared it back to empty).
    var displayTitle: String {
        if let title, !title.trimmingCharacters(in: .whitespaces).isEmpty {
            return title
        }
        let models = modelNames.count
        let benchmarks = benchmarkIds.count
        return "\(createdDateLabel) · \(models) model\(models == 1 ? "" : "s") · " +
            "\(benchmarks) benchmark\(benchmarks == 1 ? "" : "s")"
    }

    var createdDate: Date? {
        JobDateFormat.iso8601.date(from: createdAt)
    }

    private var createdDateLabel: String {
        guard let createdDate else {
            return String(createdAt.prefix(10))
        }
        return JobDateFormat.shortDate.string(from: createdDate)
    }
}

extension JobManifest {
    /// Apply crash evidence from an active-cell sentinel: the process died while that cell
    /// ran (jetsam, watchdog, force quit; a user cancel clears the sentinel first). Runs at
    /// launch, before `recoverInterruptedRunState()`.
    ///
    /// A fresh payload means the benchmark finished and the kill landed before the
    /// completed status was saved, so the result is promoted rather than re-run. A second
    /// crash on one cell fails every remaining runnable cell of the same model instead of
    /// letting it crash-loop. Returns true when the manifest changed.
    nonisolated mutating func applyCrashEvidence(sentinel: ActiveCellSentinel, payloadIsFresh: Bool) -> Bool {
        guard let index = cells.firstIndex(where: { $0.cellId == sentinel.cellId }) else { return false }
        let status = cells[index].runStatus
        // A terminal status means the cell's outcome was persisted and the
        // kill landed after the save, before the sentinel clear. Nothing to
        // repair.
        guard status == .running || status == .pending else { return false }

        if payloadIsFresh {
            cells[index].runStatus = .completed
            cells[index].errorMessage = nil
            return true
        }

        let crashes = (cells[index].crashCount ?? 0) + 1
        cells[index].crashCount = crashes
        cells[index].runStatus = .failed
        // State the fact, not a diagnosis: jetsam, a watchdog and a force quit look alike
        // from here. A memory snapshot on the sentinel adds the measured numbers.
        var message = "iOS terminated the app while this benchmark was running."
        if let available = sentinel.availableBytes, let model = sentinel.modelBytes,
           available > 0, model > 0 {
            message += " The app had \(ByteFormat.memory(available)) of memory available "
                + "for a \(ByteFormat.memory(model)) model."
        }
        cells[index].errorMessage = message

        if crashes >= 2 {
            for i in cells.indices
            where i != index
                && cells[i].source == cells[index].source
                && (cells[i].runStatus == .pending || cells[i].runStatus == .cancelled) {
                cells[i].runStatus = .failed
                cells[i].errorMessage =
                    "Skipped: \(cells[i].modelName) has crashed the app \(crashes) times."
            }
        }
        return true
    }

    /// Cold launch recovery for a job whose process died mid-run. There is no
    /// live runner after relaunch, so any persisted running/pending work from a
    /// `.running` manifest is resumable work, not active work.
    nonisolated mutating func recoverInterruptedRunState() -> Bool {
        guard status == .running || status == .paused else { return false }

        var changed = false

        for index in cells.indices where cells[index].runStatus == .running {
            cells[index].runStatus = .cancelled
            changed = true
        }

        if status == .running {
            for index in cells.indices where cells[index].runStatus == .pending {
                cells[index].runStatus = .cancelled
                changed = true
            }

            let recoveredStatus: JobStatus = cells.contains { $0.runStatus == .cancelled } ? .paused : .completed
            if status != recoveredStatus {
                status = recoveredStatus
                changed = true
            }
        }

        return changed
    }

    /// End-of-run bookkeeping shared by every executor exit path. Unreached pending cells
    /// become resumable, and an auto-pause records why so the job page can explain it.
    nonisolated mutating func finalizeRunEnd(cancelled: Bool, cancelReason: CancelFlag.Reason?) {
        guard cancelled else {
            status = .completed
            pausedReason = nil
            return
        }
        for i in cells.indices where cells[i].runStatus == .pending {
            cells[i].runStatus = .cancelled
        }
        let hasResumableWork = cells.contains { $0.runStatus == .cancelled }
        status = hasResumableWork ? .paused : .completed
        pausedReason = (cancelReason == .background && hasResumableWork)
            ? "Auto-paused when the app went to the background. A backgrounded benchmark "
              + "reports invalid timings, and iOS is likely to kill the suspended app anyway. "
              + "Resume when ready."
            : nil
    }
}

/// The fields launch recovery and the pin sweep decide on, read without decoding the
/// whole manifest. No schema migration touches them, so a plain decoder reads every
/// version.
nonisolated struct JobManifestPeek: Decodable {
    nonisolated struct Cell: Decodable {
        let cellId: CellId
        let runStatus: CellRunStatus
    }

    let status: JobStatus
    let claimJobId: String?
    let cells: [Cell]

    /// Whether `recoverInterruptedRunState()` would change the manifest.
    var needsRunStateRecovery: Bool {
        status == .running || (status == .paused && cells.contains { $0.runStatus == .running })
    }
}

nonisolated enum JobStatus: String, Codable, Sendable {
    case planned
    case running
    case completed
    case cancelled
    case paused
}

/// Marker for the cell currently executing (`jobs/{jobId}/active-cell.json`), written
/// before a cell starts and removed once its terminal status is saved. One found at launch
/// means the process died mid-cell, which is how a jetsam OOM kill (never reaching any
/// in-process error path) shows. `startedAt` lets recovery tell whether a payload on disk
/// belongs to this attempt.
nonisolated struct ActiveCellSentinel: Codable, Sendable {
    let cellId: CellId
    let startedAt: String
    // Memory snapshot (bytes) taken just before the model load; absent when the budget was
    // unreadable (simulator). Lets crash recovery say "had X free for a Y model".
    var availableBytes: Int64?
    var modelBytes: Int64?
}
