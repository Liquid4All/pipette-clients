import Foundation

// Turning a claimed job's opaque `spec` into the cell the runner executes:
// `pipette-cli`'s `client/claim.rs`.

/// A claim no client can run: the crate's `UnrunnableClaimError`. The clients draw from one
/// queue, so a body one retires must not be one the other merely denies.
///
/// Every case is terminal (`classifyRunError`): no earlier job-body revision is in
/// circulation, so even a missing or unreadable `spec` is work no client would do better
/// with. The flag-group cases name what would not read rather than quoting a payload that
/// may hold a gated-repo token; the last four are this build's own limits.
nonisolated enum UnrunnableClaimError: Error, LocalizedError {
    case noSpec
    case unparseable
    case benchmarkMismatch(envelope: String, spec: String)
    case unclassifiableBenchmark(String)
    case unsupportedRuntime(String)
    case flagsNameAnotherCell(group: String, discriminant: String, expected: String)
    case flagAxisMissing(group: String, axis: String)
    case flagFieldUnknown(group: String, field: String)
    case flagValueInvalid(group: String, field: String)
    case flagNotAcceptedByCell(group: String, knob: String, cell: String)
    case noFlagsForCell(group: String, cell: String)
    case incompatible(model: String, runtime: String)
    /// A model kind this build cannot fetch or load. The Rust client runs these cells, and
    /// it is terminal anyway: keeping work off a client is `requires`' job, not a disposition.
    case modelKindNotRunnable(String)
    /// The cell names `private_thermal` and this binary lacks the private SoC read. Only a
    /// different build (a different runtime identity) could run it.
    case buildLacksPrivateThermal
    case buildLacksPrivateMemory
    /// A knob the Rust client honours but this build cannot apply. Terminal for the same
    /// reason as `modelKindNotRunnable`.
    case flagNotHonouredHere(group: String, knob: String)

    /// Prefixed as the crate's `#[error]` strings are, so a `failure_reason` reads the
    /// same whichever client refused the claim.
    var errorDescription: String? { "unrunnable claim: \(detail)" }

    private var detail: String {
        switch self {
        case .noSpec:
            return "carries no `spec`"
        case .unparseable:
            // The crate appends the serde error; a `DecodingError` can quote the value
            // it choked on, and a plan's model may carry a gated-repo token.
            return "spec did not parse"
        case .modelKindNotRunnable(let reason):
            return "claim names a model this build cannot run: \(reason)"
        case .buildLacksPrivateThermal:
            return "claim asks for `private_thermal`, but this build compiled without "
                + "the private SoC-temperature read. Its readiness gate has only "
                + "`thermalState`"
        case .buildLacksPrivateMemory:
            return "claim asks for `private_memory`, but this build compiled without "
                + "the private memory calls. Its `max_memory_usage` cannot flush what "
                + "earlier work freed and has only the 20 ms sample"
        case .benchmarkMismatch(let envelope, let spec):
            // Catalog ids, not payload: safe to name, and naming them is the
            // only way an operator can tell which side was mis-authored.
            return "benchmark_id `\(envelope)` disagrees with spec.benchmark `\(spec)`"
        case .unclassifiableBenchmark(let id):
            return "benchmark `\(id)` names no known type, so its flag groups cannot resolve"
        case .unsupportedRuntime(let type):
            return "runtime type `\(type)` is not runnable on iOS"
        case .flagsNameAnotherCell(let group, let discriminant, let expected):
            return "spec.\(group) names `\(discriminant)`, but the cell runs `\(expected)`"
        case .flagAxisMissing(let group, let axis):
            return "spec.\(group) is missing the required `\(axis)`"
        case .flagFieldUnknown(let group, let field):
            return "spec.\(group) carries unknown field `\(field)`"
        case .flagValueInvalid(let group, let field):
            // The name, never the value: a flag group is plan-supplied.
            return "spec.\(group) field `\(field)` is not a valid value"
        case .flagNotAcceptedByCell(let group, let knob, let cell):
            return "spec.\(group) sets `\(knob)`, which \(cell) does not accept"
        case .noFlagsForCell(let group, let cell):
            return "no \(group) are defined for \(cell)"
        case .flagNotHonouredHere(let group, let knob):
            return "spec.\(group) sets `\(knob)`, which this build cannot apply"
        case .incompatible(let model, let runtime):
            return "model `\(model)` is not compatible with runtime `\(runtime)`"
        }
    }
}

/// Type the claim's payload: the crate's `run_spec_from_claim`, and the only place the raw
/// `spec` is read. The failing payload is never quoted into the error: it reaches the
/// server as `failure_reason`, and a plan may carry a gated-repo token.
func runSpecFromClaim(_ job: ClaimedJob) throws -> ClientRunSpec {
    guard let rawSpec = job.spec else {
        throw UnrunnableClaimError.noSpec
    }
    let spec: ClientRunSpec
    do {
        spec = try JSONDecoder().decode(ClientRunSpec.self, from: rawSpec.data)
    } catch let unknown as UnknownFlagField {
        throw UnrunnableClaimError.flagFieldUnknown(
            group: unknown.group ?? "spec",
            field: unknown.name
        )
    } catch let error as DecodingError {
        throw ClientRunSpec.refusal(for: error)
    } catch let error as RuntimeIdentityError {
        if case let .unsupportedRuntimeType(type) = error {
            throw UnrunnableClaimError.unsupportedRuntime(type)
        }
        throw UnrunnableClaimError.unparseable
    } catch let error as ModelError {
        // Only an unrunnable kind is that refusal; any other rejection (a malformed
        // subpath, an empty `repository_version`) is a payload that did not read.
        guard case .unknownModelType = error else { throw UnrunnableClaimError.unparseable }
        throw UnrunnableClaimError.modelKindNotRunnable(error.localizedDescription)
    } catch {
        throw UnrunnableClaimError.unparseable
    }
    // The server resolves the catalog from the envelope's `benchmark_id`, the cell runs
    // from the payload's. Guessing which was meant would run or file the wrong work.
    guard spec.benchmark.value == job.benchmarkId else {
        throw UnrunnableClaimError.benchmarkMismatch(
            envelope: job.benchmarkId,
            spec: spec.benchmark.value
        )
    }
    return spec
}

extension ClientRunSpec {
    /// Refuse a typed cell this build cannot run as named. A flavor this build lacks
    /// already failed to decode.
    static func validated(_ spec: ClientRunSpec) throws -> ClientRunSpec {
        // `Model` decodes every source arm the crate has, so a CLI body (a `url`, a host
        // path, a store form) parses here and is refused on capability instead.
        guard spec.model.isFetchableHere else {
            throw UnrunnableClaimError.modelKindNotRunnable(
                "\(ModelType.of(spec.model).rawValue) source is not fetchable on iOS")
        }

        // Must happen here: dispatch derives the engine from the model
        // (`Runtime.thisBuild(for:)`), so an incoherent cell would quietly run on the other
        // engine and be filed as the one the claim named.
        guard isCompatible(spec.model, spec.runtime) else {
            throw UnrunnableClaimError.incompatible(
                model: spec.model.description,
                runtime: RuntimeType.of(spec.runtime).rawValue
            )
        }
        // Refused rather than run coarser: an ungated run may start hot, so it answers a
        // different question.
        let wanted = spec.runtime.privateApis, built = PrivateApis.thisBuild
        if wanted.thermal, !built.thermal {
            throw UnrunnableClaimError.buildLacksPrivateThermal
        }
        if wanted.memory, !built.memory {
            throw UnrunnableClaimError.buildLacksPrivateMemory
        }

        // Flag groups resolve against the benchmark type, so an unclassifiable id is
        // refused only when a group needs it; a flagless cell still runs.
        let benchmarkType = BenchmarkType.fromId(spec.benchmark.value)?.rawValue
        if spec.runtimeFlags != nil || spec.modelFlags != nil || spec.benchmarkFlags != nil,
           benchmarkType == nil {
            throw UnrunnableClaimError.unclassifiableBenchmark(spec.benchmark.value)
        }

        let cell = Cell(
            benchmarkType: benchmarkType,
            runtimeType: RuntimeType.of(spec.runtime).rawValue,
            modelType: ModelType.of(spec.model).rawValue
        )
        if let flags = spec.runtimeFlags {
            try Self.checkRuntimeFlags(flags, cell: cell)
        }
        if let flags = spec.modelFlags {
            try Self.checkModelFlags(flags, cell: cell)
        }
        if let flags = spec.benchmarkFlags {
            try Self.checkBenchmarkFlags(flags, cell: cell)
        }
        return spec
    }

    static func validated(job: ClaimedJob) throws -> ClientRunSpec {
        try validated(runSpecFromClaim(job))
    }

    // MARK: - Flag groups

    /// The `(benchmark, runtime, model)` triple a flag group must name, as the
    /// `…FlagRef` axes spell it.
    private struct Cell {
        let benchmarkType: String?
        let runtimeType: String?
        let modelType: String?

        var description: String {
            "\(benchmarkType ?? "?") × \(runtimeType ?? "?") × \(modelType ?? "?")"
        }
    }

    private static func checkRuntimeFlags(_ ref: RuntimeFlagRef, cell: Cell) throws {
        let group = ClientRunSpec.CodingKeys.runtimeFlags.rawValue
        try Self.checkAxis(ref.benchmarkType.rawValue, cell.benchmarkType, group: group)
        try Self.checkAxis(ref.runtimeType.rawValue, cell.runtimeType, group: group)
        try Self.checkAxis(ref.modelType.rawValue, cell.modelType, group: group)

        // Resolved for its refusals, not its value: the entry travels on the spec unchanged.
        _ = try Self.resolve(ref, group: group, cell: cell)
    }

    /// A cell with no readiness variant (see `BenchmarkFlagRef.resolve`) is a `noFlagsForCell`.
    private static func checkBenchmarkFlags(_ ref: BenchmarkFlagRef, cell: Cell) throws {
        let group = ClientRunSpec.CodingKeys.benchmarkFlags.rawValue
        try Self.checkAxis(ref.benchmarkType.rawValue, cell.benchmarkType, group: group)
        try Self.checkAxis(ref.runtimeType.rawValue, cell.runtimeType, group: group)
        try Self.checkAxis(ref.modelType.rawValue, cell.modelType, group: group)
        do {
            _ = try ref.resolve()
        } catch .noSuchCombination {
            throw UnrunnableClaimError.noFlagsForCell(group: group, cell: cell.description)
        }
    }

    private static func checkModelFlags(_ ref: ModelFlagRef, cell: Cell) throws {
        let group = ClientRunSpec.CodingKeys.modelFlags.rawValue
        try Self.checkAxis(ref.benchmarkType.rawValue, cell.benchmarkType, group: group)
        try Self.checkAxis(ref.modelType.rawValue, cell.modelType, group: group)
        // `ModelFlagRef` carries no runtime axis.
        let flags: ModelFlags
        do {
            flags = try ref.resolve()
        } catch .noSuchCombination {
            throw UnrunnableClaimError.noFlagsForCell(group: group, cell: cell.description)
        } catch let .knobNotAllowed(knob, _, _) {
            throw UnrunnableClaimError.flagNotAcceptedByCell(
                group: group, knob: knob, cell: cell.description)
        }
        // Both flags shape generation through the chat template's Jinja context. This
        // build drives the bare template (`llama_chat_apply_template`), using Jinja only as
        // a fallback at the default thinking setting, so neither flag would be applied and
        // the run would measure a different cell.
        if flags.enableThinking != nil {
            throw UnrunnableClaimError.flagNotHonouredHere(group: group, knob: "enable_thinking")
        }
        if flags.reasoningEffort != nil {
            throw UnrunnableClaimError.flagNotHonouredHere(
                group: group, knob: ModelFlagRef.CodingKeys.reasoningEffort.rawValue)
        }
    }

    /// The refusal for a `ClientRunSpec` decode failure: one inside a flag group is reported
    /// against that group, anything else as `unparseable`.
    fileprivate static func refusal(for error: DecodingError) -> UnrunnableClaimError {
        var path: [CodingKey]
        var missingKey: CodingKey?
        switch error {
        case .typeMismatch(_, let context), .valueNotFound(_, let context),
             .dataCorrupted(let context):
            path = context.codingPath
        case .keyNotFound(let key, let context):
            path = context.codingPath
            missingKey = key
        @unknown default:
            path = []
        }
        let flagGroups: [ClientRunSpec.CodingKeys] = [.runtimeFlags, .modelFlags, .benchmarkFlags]
        guard let group = path.first?.stringValue,
              flagGroups.contains(where: { $0.rawValue == group })
        else {
            return .unparseable
        }
        if let missingKey {
            return .flagAxisMissing(group: group, axis: missingKey.stringValue)
        }
        return .flagValueInvalid(group: group, field: path.last?.stringValue ?? "?")
    }

    /// A `…FlagRef` resolves on its own axes, so a group naming another cell would resolve
    /// to that cell's variant unless its axes are checked against this one first.
    private static func checkAxis(
        _ named: String,
        _ expected: String?,
        group: String
    ) throws {
        guard named == expected else {
            throw UnrunnableClaimError.flagsNameAnotherCell(
                group: group,
                discriminant: named,
                expected: expected ?? "?"
            )
        }
    }

    private static func resolve(
        _ ref: RuntimeFlagRef,
        group: String,
        cell: Cell
    ) throws -> RuntimeFlags {
        do {
            return try ref.resolve()
        } catch RuntimeFlagsError.knobNotAllowed(let knob, _, _, _) {
            throw UnrunnableClaimError.flagNotAcceptedByCell(
                group: group,
                knob: knob,
                cell: cell.description
            )
        } catch RuntimeFlagsError.noSuchCombination {
            throw UnrunnableClaimError.noFlagsForCell(group: group, cell: cell.description)
        }
    }
}
