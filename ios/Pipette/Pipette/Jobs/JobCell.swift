import Foundation

// MARK: - Job cell (one benchmark × model pair of a job, persisted inside the manifest)

nonisolated struct JobCell: Codable, Sendable, Identifiable {
    var id: CellId { cellId }
    let cellId: CellId           // UUID — doubles as the local result directory name.
    let benchmarkId: String
    // Encoded as its bare `rawValue`; an unknown or absent value decodes to `nil` (see the
    // custom Codable below).
    var benchmarkType: BenchmarkType?
    var runStatus: CellRunStatus
    var serverJobId: String?     // set after submission writes the cell's submission record
    var errorMessage: String?
    // How many times the process died while this cell ran (counted at launch from the
    // active-cell sentinel). Survives "Retry failed" so a model that keeps killing the
    // app is condemned instead of crash-looping. nil means 0.
    var crashCount: Int?
    /// Which catalog half this cell's benchmark came from. A `local` definition was
    /// generated on this device and never sanctioned by the server, so its result is not
    /// submitted (the crate's `BenchmarkResultLocation::from(BenchmarkSource)` rule).
    /// Absent means `remote`.
    var benchmarkSource: BenchmarkSource?

    /// For a memory cell a public build ran: whether the app had already run other work
    /// since launch, so the peak can include memory that work freed. `nil` for other
    /// cells, internal builds, and results recorded before this was kept. Never submitted,
    /// though the result it describes still uploads like any other.
    var measuredAfterOtherWork: Bool?

    /// The cell as the crate describes one: model, runtime, and the authored flag groups.
    /// Carried whole into `RunCell.prepare` so a knob the settings cannot spell
    /// (`swa_full`) is not dropped between authoring and load.
    let spec: ClientRunSpec

    /// The model this cell benchmarks, read through the spec.
    var source: Model { spec.model }

    /// The cell's authored `runtime_flags`, or `nil` when it named none.
    var runtimeFlags: RuntimeFlagRef? { spec.runtimeFlags }

    /// Human-facing engine name, derived from the model so it can't drift.
    var engineLabel: String { source.engineLabel }

    /// Whether this cell's result may be submitted; an absent `benchmarkSource` reads as
    /// `remote`.
    var isSubmittable: Bool { (benchmarkSource ?? .remote) == .remote }

    /// The publishing repo slug: a UI label derived from `source`, not a second identity
    /// (the wire states a model as `model_descriptor` alone). A model with no repo (a vault
    /// resource or a local file) uses its artifact name.
    var modelName: String {
        let slug = source.repoSlug
        return slug.isEmpty ? source.artifactName : slug
    }

    private enum CodingKeys: String, CodingKey {
        case cellId, benchmarkId, benchmarkType
        case runStatus, serverJobId, errorMessage, crashCount, source
        case benchmarkSource, runtimeFlags, spec, measuredAfterOtherWork
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

    /// A cell authored on this device: the runtime is the one this build compiled as.
    init(
        cellId: CellId, benchmarkId: String, benchmarkType: BenchmarkType?,
        runStatus: CellRunStatus, serverJobId: String? = nil, errorMessage: String? = nil,
        crashCount: Int? = nil, source: Model,
        benchmarkSource: BenchmarkSource? = nil,
        runtimeFlags: RuntimeFlagRef? = nil,
        benchmarkFlags: BenchmarkFlagRef? = nil
    ) {
        self.init(
            cellId: cellId, benchmarkId: benchmarkId, benchmarkType: benchmarkType,
            runStatus: runStatus, serverJobId: serverJobId, errorMessage: errorMessage,
            crashCount: crashCount,
            spec: ClientRunSpec(benchmark: benchmarkId, model: source,
                                runtime: Runtime.thisBuild(for: source),
                                runtimeFlags: runtimeFlags, benchmarkFlags: benchmarkFlags),
            benchmarkSource: benchmarkSource)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        cellId = try c.decode(CellId.self, forKey: .cellId)
        // A bare id, never a `local/`/`remote/` reference: a manifest that holds one is
        // rewritten by `JobManifestSchema`'s version-2 step before it reaches here, which
        // is what lets `ClientRunSpec` below take it as a `BenchmarkId`.
        benchmarkId = try c.decode(String.self, forKey: .benchmarkId)
        // Lenient: an unknown or absent type decodes to `nil`, so a manifest carrying a
        // type this build doesn't recognise still loads.
        benchmarkType = (try c.decodeIfPresent(String.self, forKey: .benchmarkType))
            .flatMap(BenchmarkType.init(rawValue:))
        runStatus = try c.decode(CellRunStatus.self, forKey: .runStatus)
        serverJobId = try c.decodeIfPresent(String.self, forKey: .serverJobId)
        errorMessage = try c.decodeIfPresent(String.self, forKey: .errorMessage)
        crashCount = try c.decodeIfPresent(Int.self, forKey: .crashCount)
        benchmarkSource = try c.decodeIfPresent(BenchmarkSource.self, forKey: .benchmarkSource)
        measuredAfterOtherWork = try c.decodeIfPresent(Bool.self, forKey: .measuredAfterOtherWork)
        // A manifest written before the spec carries the model and flags as separate keys.
        // Reassembled here rather than migrated in `JobManifestSchema`: it is the same cell
        // spelled out, and the runtime it implies is the one this build derives anyway.
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
        // The bare rawValue string; omitted when nil.
        try c.encodeIfPresent(benchmarkType?.rawValue, forKey: .benchmarkType)
        try c.encode(runStatus, forKey: .runStatus)
        try c.encodeIfPresent(serverJobId, forKey: .serverJobId)
        try c.encodeIfPresent(errorMessage, forKey: .errorMessage)
        try c.encodeIfPresent(crashCount, forKey: .crashCount)
        try c.encodeIfPresent(benchmarkSource, forKey: .benchmarkSource)
        try c.encodeIfPresent(measuredAfterOtherWork, forKey: .measuredAfterOtherWork)
        try c.encode(spec, forKey: .spec)
        // Also written standalone: a build predating the spec decodes `source` and skips a
        // cell without one, so omitting it would empty a saved job on rollback. Drop once
        // no such build is in circulation.
        try c.encode(source, forKey: .source)
    }
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
