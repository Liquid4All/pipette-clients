import Foundation

// MARK: - Job manifest (persisted as JSON in results/jobs/{jobId}/manifest.json)

nonisolated struct JobManifest: Codable, Sendable {
    let jobId: JobId
    let createdAt: String
    // Manifest schema version, stamped on every save. Absent means version 1
    // (manifests written before versioning existed). Purely additive optional
    // fields don't need a bump; see `JobManifestSchema` for when and how to
    // bump and migrate.
    var schemaVersion: Int?
    var nGpuLayers: Int
    var contextSize: Int
    // Prefill batch size applied to every cell: llama's `n_ubatch` and MLX's
    // prefill chunk. Optional so older manifests decode; `nil` → 512 (the default
    // for both engines). See `effectivePrefillBatch`.
    var prefillBatch: Int?
    // CPU threads for llama.cpp's `n_threads`/`n_threads_batch`. Optional so older
    // manifests decode and so "the cell said nothing" stays distinguishable from a
    // number: absent means the engine derives it from the device's P-core count, which
    // is what it reports back.
    var threads: Int?
    var cells: [JobCell]
    var status: JobStatus
    // Whether this eligible internal job should upload completed results
    // automatically once all runnable cells finish. Optional for older manifests.
    var contributeResults: Bool?
    // User-chosen name. Optional so older manifests keep decoding and so we
    // can distinguish "no name set" (show default) from an explicit empty
    // string (also shows default).
    var title: String?
    // Why a .paused job paused, when the pause wasn't the user's doing
    // (currently: auto-paused on backgrounding). Shown on the paused job
    // page; nil for user pauses and cleared on resume/completion. Optional
    // so older manifests keep decoding.
    var pausedReason: String?

    var totalCells: Int { cells.count }
    var completedCells: Int { cells.filter { $0.runStatus == .completed }.count }
    var failedCells: Int { cells.filter { $0.runStatus == .failed }.count }
    var cancelledCells: Int { cells.filter { $0.runStatus == .cancelled }.count }
    var submittedCells: Int { cells.filter { $0.serverJobId != nil }.count }

    /// Unique model names used in this job.
    var modelNames: [String] {
        Array(Set(cells.map { $0.modelName })).sorted()
    }

    /// Unique benchmark IDs used in this job.
    var benchmarkIds: [String] {
        Array(Set(cells.map { $0.benchmarkId })).sorted()
    }

    /// Runtime(s) this job's cells run on — "llama.cpp", "MLX", or both (joined).
    /// Shown on job screens so the engine is always clear.
    var runtimeLabel: String {
        Set(cells.map { $0.engineLabel }).sorted().joined(separator: " + ")
    }

    /// Prefill batch to apply (llama `n_ubatch` / MLX prefill chunk); 512 default.
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
    /// Apply crash evidence left by an active-cell sentinel: the process died
    /// while the sentinel's cell was executing (jetsam OOM kill, watchdog,
    /// force quit) — a user cancel never gets here because it clears the
    /// sentinel on its way out. Runs at launch, before
    /// `recoverInterruptedRunState()`.
    ///
    /// `payloadIsFresh` reports whether the cell's result payload landed on
    /// disk during this attempt: then the benchmark itself finished and the
    /// kill hit the window before the completed status was saved, so the
    /// finished work is promoted instead of re-run.
    ///
    /// A second crash on the same cell condemns the model: every remaining
    /// runnable cell with the same typed `source` is failed rather than left to
    /// crash-loop a model that doesn't fit this device.
    ///
    /// Returns true when the manifest changed.
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
        // State the fact, not a diagnosis — the kill could be jetsam, a
        // watchdog, or a force quit, and we can't tell them apart. When the
        // run loop stamped a memory snapshot on the sentinel, attach the
        // measured numbers and let them speak for themselves.
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

    /// End-of-run bookkeeping shared by every executor exit path: pending
    /// cells the run never reached become resumable, the job lands on
    /// `.paused` / `.completed`, and an auto-pause records why it paused so
    /// the job page can explain a pause the user didn't ask for.
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

nonisolated struct JobCell: Codable, Sendable, Identifiable {
    var id: CellId { cellId }
    let cellId: CellId           // UUID — doubles as the local result directory name.
    let benchmarkId: String
    // Typed benchmark kind (e.g. `.prefillThroughput`, `.maxMemoryUsage`). The
    // runner uses this to route `max_memory_usage` cells through a fresh model
    // load rather than the reused-handle path. Persists as the bare type string
    // (its `rawValue`) under the unchanged `benchmarkType` key (see the custom
    // Codable below): an unknown/legacy/absent value decodes to `nil`, matching
    // the pre-typed `BenchmarkType(type:)` / `?? ""` / `default` tolerance.
    var benchmarkType: BenchmarkType?
    var runStatus: CellRunStatus
    var serverJobId: String?     // set after submission writes the cell's submission record
    var errorMessage: String?
    // How many times the process died while this cell was executing (counted
    // by launch recovery from the active-cell sentinel). Survives "Retry
    // failed" so a model that keeps killing the app can be condemned instead
    // of crash-looping. Optional so older manifests decode; nil → 0.
    var crashCount: Int?
    /// Which catalog half this cell's benchmark came from. A `local` definition was
    /// generated on this device and the server never sanctioned it, so its result is
    /// not submitted — the crate encodes the same rule as
    /// `BenchmarkResultLocation::from(BenchmarkSource)`, which routes a `Local` result
    /// to `results/local/` where `sync` never looks.
    ///
    /// Optional so manifests written before the local half existed decode; absent means
    /// `remote`, which is what every such cell was.
    var benchmarkSource: BenchmarkSource?

    /// The cell, as the crate describes one: model, runtime, and the authored flag groups.
    /// A claim decodes one; the app authors one from its settings. Carried whole into
    /// `RunCell.prepare` rather than destructured into load settings and rebuilt, which is
    /// what keeps a knob the settings cannot spell (`swa_full`) from being dropped between
    /// authoring and load.
    let spec: ClientRunSpec

    /// The model this cell benchmarks. Reads through the spec, which is where a cell's
    /// identity lives now — the stored `source` it replaced is still decoded, for manifests
    /// written before the spec was.
    var source: Model { spec.model }

    /// The cell's authored `runtime_flags`, or `nil` when it named none.
    var runtimeFlags: RuntimeFlagRef? { spec.runtimeFlags }

    /// Human-facing engine name for this cell — *derived* from the typed `source`
    /// spec by `switch` (GGUF → llama.cpp, MLX → MLX, 1:1), so it can't drift.
    var engineLabel: String { source.engineLabel }

    /// Whether this cell's result may be submitted — the crate's
    /// `BenchmarkResultLocation::from(BenchmarkSource)`, which makes a `Remote` result
    /// `RemotePending` and a `Local` one terminal. Absent reads as `remote`.
    var isSubmittable: Bool { (benchmarkSource ?? .remote) == .remote }

    /// The publishing repo as a slug, derived from `source` rather than stored — every
    /// front-end set it from the same derivation, and the wire never carries it: the
    /// crate's `BenchmarkSubmissionPayload` states a model's identity as
    /// `model_descriptor` alone. A UI label, not a second identity to keep in step.
    var modelName: String { source.repoSlug }

    private enum CodingKeys: String, CodingKey {
        case cellId, benchmarkId, benchmarkType
        case runStatus, serverJobId, errorMessage, crashCount, source
        case benchmarkSource, runtimeFlags, spec
    }

    init(
        cellId: CellId, benchmarkId: String, benchmarkType: BenchmarkType?,
        runStatus: CellRunStatus, serverJobId: String?, errorMessage: String?,
        crashCount: Int? = nil, spec: ClientRunSpec,
        benchmarkSource: BenchmarkSource? = nil
    ) {
        self.cellId = cellId
        self.benchmarkId = benchmarkId
        self.benchmarkType = benchmarkType
        self.runStatus = runStatus
        self.serverJobId = serverJobId
        self.errorMessage = errorMessage
        self.crashCount = crashCount
        self.spec = spec
        self.benchmarkSource = benchmarkSource
    }

    /// A cell authored on this device: the runtime is derived, because this client can only
    /// be what it compiled as. The crate's `into_client_run_spec` fills the same field from
    /// its arguments; here there is only one answer to fill it with.
    init(
        cellId: CellId, benchmarkId: String, benchmarkType: BenchmarkType?,
        runStatus: CellRunStatus, serverJobId: String? = nil, errorMessage: String? = nil,
        crashCount: Int? = nil, source: Model,
        benchmarkSource: BenchmarkSource? = nil,
        runtimeFlags: RuntimeFlagRef? = nil
    ) {
        self.init(
            cellId: cellId, benchmarkId: benchmarkId, benchmarkType: benchmarkType,
            runStatus: runStatus, serverJobId: serverJobId, errorMessage: errorMessage,
            crashCount: crashCount,
            spec: ClientRunSpec(benchmark: benchmarkId, model: source,
                                runtime: Runtime.thisBuild(for: source),
                                runtimeFlags: runtimeFlags),
            benchmarkSource: benchmarkSource)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        cellId = try c.decode(CellId.self, forKey: .cellId)
        benchmarkId = try c.decode(String.self, forKey: .benchmarkId)
        // Lenient: read the bare wire string and map it to the enum. An unknown,
        // legacy, or absent `benchmark_type` decodes to `nil` (never throws),
        // preserving the pre-typed `?? ""` / `default` fallback behaviour so a
        // manifest carrying a type this build doesn't recognise still loads.
        benchmarkType = (try c.decodeIfPresent(String.self, forKey: .benchmarkType))
            .flatMap(BenchmarkType.init(rawValue:))
        runStatus = try c.decode(CellRunStatus.self, forKey: .runStatus)
        serverJobId = try c.decodeIfPresent(String.self, forKey: .serverJobId)
        errorMessage = try c.decodeIfPresent(String.self, forKey: .errorMessage)
        crashCount = try c.decodeIfPresent(Int.self, forKey: .crashCount)
        benchmarkSource = try c.decodeIfPresent(BenchmarkSource.self, forKey: .benchmarkSource)
        // A manifest written before the spec carries the model and flags as separate keys.
        // Reassembled rather than migrated in `JobManifestSchema`: nothing about the older
        // shape is wrong, it is the same cell with its parts spelled out, and the runtime
        // it implies is the one this build derives anyway.
        if let spec = try c.decodeIfPresent(ClientRunSpec.self, forKey: .spec) {
            self.spec = spec
        } else {
            let source = try c.decode(Model.self, forKey: .source)
            self.spec = ClientRunSpec(
                benchmark: benchmarkId, model: source,
                runtime: Runtime.thisBuild(for: source),
                runtimeFlags: try c.decodeIfPresent(RuntimeFlagRef.self, forKey: .runtimeFlags))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(cellId, forKey: .cellId)
        try c.encode(benchmarkId, forKey: .benchmarkId)
        // Write the bare rawValue string (the unchanged wire shape); omit when nil.
        try c.encodeIfPresent(benchmarkType?.rawValue, forKey: .benchmarkType)
        try c.encode(runStatus, forKey: .runStatus)
        try c.encodeIfPresent(serverJobId, forKey: .serverJobId)
        try c.encodeIfPresent(errorMessage, forKey: .errorMessage)
        try c.encodeIfPresent(crashCount, forKey: .crashCount)
        try c.encodeIfPresent(benchmarkSource, forKey: .benchmarkSource)
        try c.encode(spec, forKey: .spec)
        // Also written standalone, for one release: a build predating the spec decodes
        // `source` and *skips* a cell without one, so omitting it would empty a saved job
        // on rollback rather than degrade it. Drop once no such build is in circulation.
        try c.encode(source, forKey: .source)
    }
}

nonisolated enum JobStatus: String, Codable, Sendable {
    case planned
    case running
    case completed
    case cancelled
    case paused
}

/// Marker for the cell currently executing, persisted next to the job's
/// manifest (`jobs/{jobId}/active-cell.json`). Written immediately before a
/// cell starts and removed once its terminal status is saved, so one found at
/// launch means the process died mid-cell — the signature of a jetsam OOM
/// kill, which never reaches any in-process error path. `startedAt` dates the
/// attempt so recovery can tell whether a payload on disk belongs to it.
nonisolated struct ActiveCellSentinel: Codable, Sendable {
    let cellId: CellId
    let startedAt: String
    // Memory snapshot taken just before the model load (bytes); absent when
    // the budget was unreadable (simulator) or on records written by older
    // builds. Lets crash recovery quantify a kill: "had X free for a Y model".
    var availableBytes: Int64?
    var modelBytes: Int64?
}

nonisolated enum CellRunStatus: String, Codable, Sendable {
    case pending
    case running
    case completed
    case failed
    case cancelled

    var label: String {
        switch self {
        case .pending:   return "Pending"
        case .running:   return "Running"
        case .completed: return "Done"
        case .failed:    return "Failed"
        case .cancelled: return "Cancelled"
        }
    }
}
