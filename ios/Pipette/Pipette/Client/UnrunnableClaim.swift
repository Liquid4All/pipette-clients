/// Why a claim was refused — the counterpart of the Rust `UnrunnableClaim`, which the
/// clients must agree with: they draw from one queue, so a body one retires must not
/// be one the other merely denies.
///
/// Not variant-for-variant. Upstream carries four, of which `Unparseable` wraps any
/// serde failure; the cases between `unclassifiableBenchmark` and `noFlagsForCell`
/// refine that one so a refusal names what would not read, rather than quoting a
/// payload that may hold a gated-repo token.
///
/// Nor disposition-for-disposition: every `UnrunnableClaim` is terminal upstream —
/// `classify_run_error` answers non-retriable for the whole type. The two retriable
/// cases here are the deliberate exception, and `retriable` says why.
nonisolated enum UnrunnableClaim: Error, LocalizedError {
    case missingSpec
    case unreadableSpec
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
    /// A model whose kind this build cannot fetch or load — a `torch` artifact, or a
    /// source that is not HuggingFace. Distinct from every other refusal because it
    /// is the one that is **not** the plan's fault: the Rust client runs these cells.
    case modelKindNotRunnable(String)
    /// The cell names `private_thermal`, and this binary compiled without the private
    /// SoC read. Terminal like the rest: no re-serving makes this device able to run
    /// it — the fix is a different build, which is a different runtime identity.
    case buildLacksPrivateThermal
    /// A knob the plan may legitimately set, whose variant this cell has, and which
    /// the Rust client honours — but this build cannot apply. Also not the plan's
    /// fault, so it answers the same way `modelKindNotRunnable` does.
    case flagNotHonouredHere(group: String, knob: String)

    var errorDescription: String? {
        switch self {
        case .missingSpec:
            return "claim carries no `spec`"
        case .unreadableSpec:
            return "claim `spec` is not a readable cell"
        case .modelKindNotRunnable(let reason):
            return "claim names a model this build cannot run: \(reason)"
        case .buildLacksPrivateThermal:
            return "claim asks for `private_thermal`, but this build compiled without "
                + "the private SoC-temperature read. Its readiness gate has only "
                + "`thermalState`"
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

    /// Terminal, as upstream: `classify_run_error` answers non-retriable for every
    /// `UnrunnableClaim`, and these are the same refusals. No earlier job-body
    /// revision is in circulation, so a payload that cannot be read describes work no
    /// client will do better with — a missing `spec` included.
    ///
    /// That includes the two this build refuses for its own limits: a `torch` cell and
    /// a knob it cannot apply. Answering them retriably left the job to be re-served,
    /// to this device as readily as to one that could run it. Keeping work away from a
    /// client that cannot do it is a capability in `requires`, not a disposition.
    var retriable: Bool { false }
}

import Foundation

/// How a planner claim configures one iOS run — derived from the plan-types
/// cell carried in the claim's `spec`, not from UI defaults.
///
/// Identity comes from `spec.runtime`. Load settings come from
/// `spec.runtime_flags`, a flat `RuntimeFlagRef` object (`number_gpu_layers` /
/// `ctx_size` / `n_ubatch`); generation settings from `spec.model_flags`
/// (`enable_thinking`).
///
/// `spec.benchmark_flags` is not read, and no plan can author it for an iOS
/// cell: every `BenchmarkFlags` variant belongs to a server runtime and carries
/// server settings (`http_timeout_seconds`, readiness, doomloop). The iOS
/// engines run in-process, so there is nothing to wait on.
extension ClientRunSpec {

    // MARK: - Parse

    /// Type the claim's payload — the iOS counterpart of `run_spec_from_claim`,
    /// and the only place the raw `spec` is read.
    ///
    /// Refusals have to agree with the Rust client's: same checks, same
    /// dispositions. The failing payload is never quoted into the error, because
    /// it reaches the server as `failure_reason` and a plan may carry a
    /// gated-repo token.
    static func runSpec(from job: ClaimedJob) throws -> ClientRunSpec {
        guard let rawSpec = job.spec else {
            throw UnrunnableClaim.missingSpec
        }
        let spec: ClientRunSpec
        do {
            spec = try JSONDecoder().decode(ClientRunSpec.self, from: rawSpec.data)
        } catch let unknown as UnknownFlagField {
            throw UnrunnableClaim.flagFieldUnknown(
                group: unknown.group ?? "spec",
                field: unknown.name
            )
        } catch let error as DecodingError {
            throw refusal(for: error)
        } catch let error as RuntimeIdentityError {
            // A desktop runtime is refused where it is read, as `Model` already is. The
            // spelling survives in the error so the refusal can name what it rejected.
            if case let .unsupportedRuntimeType(type) = error {
                throw UnrunnableClaim.unsupportedRuntime(type)
            }
            throw UnrunnableClaim.unreadableSpec
        } catch let error as ModelError {
            // Only an unrunnable *kind* is that refusal. Every other primitive rejection
            // — a malformed subpath, an empty `repository_version` — is a payload this
            // client cannot read, and reporting it as "model kind not runnable" would
            // name the wrong reason.
            guard case .unknownModelType = error else { throw UnrunnableClaim.unreadableSpec }
            // `Model` refuses a kind this build cannot run at decode time, where the old
            // bridge decoded it into an `unsupported` case and refused later. Same
            // rejection, earlier — but it must not read as an unreadable payload, which
            // is terminal.
            throw UnrunnableClaim.modelKindNotRunnable(error.localizedDescription)
        } catch {
            throw UnrunnableClaim.unreadableSpec
        }
        // `benchmark_id` is duplicated on the wire — the server resolves the
        // catalog from the envelope, the cell runs from the payload. A job whose
        // two halves name different benchmarks is mis-authored, and guessing
        // which was meant would run the wrong work or file it against the wrong
        // id.
        guard spec.benchmark == job.benchmarkId else {
            throw UnrunnableClaim.benchmarkMismatch(
                envelope: job.benchmarkId,
                spec: spec.benchmark
            )
        }
        return spec
    }

    /// Build the run configuration from an already-typed cell: the runtime it named,
    /// then the flag groups the cell's variants carry.
    ///
    /// No flavor check: `LlamacppIosPipetteFlavor` / `MlxIosPipetteFlavor` are typed enums,
    /// so a flavor this build does not have fails to decode before reaching here.
    static func validated(_ spec: ClientRunSpec) throws -> ClientRunSpec {
        // Readable is not runnable. `Model` decodes every source arm the crate has, so a
        // body authored for the CLI — a `url`, a host path, or a store form — parses here
        // and is refused on capability instead. Retriable: the Rust client runs these.
        guard spec.model.isFetchableHere else {
            throw UnrunnableClaim.modelKindNotRunnable(
                "\(ModelType.of(spec.model).rawValue) source is not fetchable on iOS")
        }

        // The cell must cohere before its flags mean anything: a runtime that
        // cannot run this model is mis-authored, and refusing it here rather
        // than discovering it as "no local model matched" keeps that terminal
        // refusal apart from the retriable one (the CLI's `is_compatible`).
        guard spec.runtime.accepts(modelType: ModelType.of(spec.model)) else {
            throw UnrunnableClaim.incompatible(
                model: ModelType.of(spec.model).rawValue,
                runtime: RuntimeType.of(spec.runtime).rawValue
            )
        }
        // A cell that asks for the private thermal gate has to get it. This build either
        // compiled the SoC read in or it did not, and the difference is not a detail of
        // how the number was produced — an ungated run is allowed to start hot, so it
        // answers a different question. Refused rather than run coarser.
        if spec.runtime.privateThermal, !Runtime.privateThermalBuild {
            throw UnrunnableClaim.buildLacksPrivateThermal
        }

        // A flag group resolves against the cell's benchmark *type*, so an id
        // this build cannot classify has no axis to check it against. Only the
        // groups need one, so a flagless cell with an unfamiliar id still runs;
        // refusing by name beats reporting an axis mismatch against `?`.
        let benchmarkType = Self.benchmarkType(ofId: spec.benchmark)
        if spec.runtimeFlags != nil || spec.modelFlags != nil || spec.benchmarkFlags != nil,
           benchmarkType == nil {
            throw UnrunnableClaim.unclassifiableBenchmark(spec.benchmark)
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

    /// Both halves at once, for callers that only need the configuration.
    static func validated(job: ClaimedJob) throws -> ClientRunSpec {
        try validated(runSpec(from: job))
    }

    // MARK: - Flag groups

    /// The `(benchmark, runtime, model)` triple a flag group must name, as the
    /// `…FlagRef` axes spell it.
    private struct Cell {
        let benchmarkType: String?
        let runtimeType: String?
        let modelType: String?

        /// How the triple reads in a refusal — plan-types' own `b × r × m`.
        var description: String {
            "\(benchmarkType ?? "?") × \(runtimeType ?? "?") × \(modelType ?? "?")"
        }
    }

    /// Apply the knobs this cell's resolved variant carries.
    ///
    /// The axes are checked against the cell before resolving: the `…FlagRef`
    /// resolves on its *own* axes, so a group naming another cell would
    /// otherwise resolve happily to that cell's variant and be applied here.
    private static func checkRuntimeFlags(_ ref: RuntimeFlagRef, cell: Cell) throws {
        let group = ClientRunSpec.CodingKeys.runtimeFlags.rawValue
        try Self.checkAxis(ref.benchmarkType.rawValue, cell.benchmarkType, group: group)
        try Self.checkAxis(ref.runtimeType.rawValue, cell.runtimeType, group: group)
        try Self.checkAxis(ref.modelType.rawValue, cell.modelType, group: group)

        // Resolved for its refusals, not its value: `resolve` is what rejects a knob this
        // cell's variant does not carry. The entry itself travels on the spec, unchanged.
        _ = try Self.resolve(ref, group: group, cell: cell)
    }

    /// The readiness gate the cell asks for. Only the timing cells carry one: `eval` and
    /// `vl` name server settings this build cannot apply, and `max_memory_usage` does not
    /// gate — each of those is a `noFlagsForCell`, exactly as it was when no iOS variant
    /// existed at all.
    private static func checkBenchmarkFlags(_ ref: BenchmarkFlagRef, cell: Cell) throws {
        let group = ClientRunSpec.CodingKeys.benchmarkFlags.rawValue
        try Self.checkAxis(ref.benchmarkType.rawValue, cell.benchmarkType, group: group)
        try Self.checkAxis(ref.runtimeType.rawValue, cell.runtimeType, group: group)
        try Self.checkAxis(ref.modelType.rawValue, cell.modelType, group: group)
        do {
            _ = try ref.resolve()
        } catch RuntimeFlagResolveError.noSuchCombination {
            throw UnrunnableClaim.noFlagsForCell(group: group, cell: cell.description)
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
        } catch RuntimeFlagResolveError.noSuchCombination {
            throw UnrunnableClaim.noFlagsForCell(group: group, cell: cell.description)
        }
        // `enable_thinking` shapes generation through the chat template's Jinja context.
        // This build drives the bare template (`llama_chat_apply_template`), which takes
        // no such variable, so the flag has nowhere to go — and a run that quietly ignored
        // it would measure something other than the cell it answered.
        if flags.enableThinking != nil {
            throw UnrunnableClaim.flagNotHonouredHere(group: group, knob: "enable_thinking")
        }
    }

    /// The refusal a `ClientRunSpec` decode failure corresponds to.
    ///
    /// A failure inside a flag group is reported against that group, since the
    /// `…FlagRef` types are where the wire contract lives — required axes,
    /// `deny_unknown_fields`, and `Option<u32>` / `Option<bool>` typing all come
    /// from them. Anything else means the cell itself did not read.
    private static func refusal(for error: DecodingError) -> UnrunnableClaim {
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
        guard let group = path.first?.stringValue,
              group == ClientRunSpec.CodingKeys.runtimeFlags.rawValue
                  || group == ClientRunSpec.CodingKeys.modelFlags.rawValue
        else {
            return .unreadableSpec
        }
        if let missingKey {
            return .flagAxisMissing(group: group, axis: missingKey.stringValue)
        }
        return .flagValueInvalid(group: group, field: path.last?.stringValue ?? "?")
    }

    /// A group resolves on its own axes, so they have to name this cell first.
    private static func checkAxis(
        _ named: String,
        _ expected: String?,
        group: String
    ) throws {
        guard named == expected else {
            throw UnrunnableClaim.flagsNameAnotherCell(
                group: group,
                discriminant: named,
                expected: expected ?? "?"
            )
        }
    }

    /// `TryFrom<RuntimeFlagRef>`, with its two failure modes mapped onto the
    /// refusals: a knob outside the variant, or no variant for the triple.
    private static func resolve(
        _ ref: RuntimeFlagRef,
        group: String,
        cell: Cell
    ) throws -> RuntimeFlags {
        do {
            return try ref.resolve()
        } catch RuntimeFlagResolveError.knobNotAllowed(let knob, _, _, _) {
            throw UnrunnableClaim.flagNotAcceptedByCell(
                group: group,
                knob: knob,
                cell: cell.description
            )
        } catch RuntimeFlagResolveError.noSuchCombination {
            throw UnrunnableClaim.noFlagsForCell(group: group, cell: cell.description)
        }
    }

    /// The `benchmark_type` a benchmark id belongs to. The rule lives on
    /// `BenchmarkType`, whose wire spelling it is about.
    private static func benchmarkType(ofId id: String) -> String? {
        (try? BenchmarkType(benchmarkId: id))?.rawValue
    }

    // MARK: - Apply to app types

}
