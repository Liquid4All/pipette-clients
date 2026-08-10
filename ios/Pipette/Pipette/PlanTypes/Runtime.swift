import Foundation

// `RuntimeKind` and `RuntimeType`. The crate's `Runtime` lives in `RuntimeIdentity.swift`;
// what an engine is handed is `RunRequest`. `RunCell.dispatch` branches by switching on the
// request's bound `Runtime`, like Rust's `match self`.
//
// `RuntimeKind` below is the one config-less, model-less projection — the New Job
// wizard's runtime *selector*, chosen upstream of any model so it can filter which
// models are eligible and which benchmarks are offered. It's a selection concern,
// not a dispatch one; execution still resolves the engine from the model.

/// The inference engine as a bare, config-less choice — what the New Job wizard's
/// runtime step selects *before* any model or config exists. Distinct from `Runtime`
/// (the resolved engine + per-cell config built at execution time): this is the
/// upstream selector that determines which models are eligible and which benchmarks
/// are offered. Format → engine is 1:1, so it maps to and from `Model` losslessly.
enum RuntimeKind: String, CaseIterable, Identifiable, Equatable {
    case llamaCpp
    case mlx
    case afm

    var id: String { rawValue }

    /// The engine a model spec runs on (format → engine, 1:1).
    init(_ model: Model) {
        switch model {
        case .ggufText, .ggufVision: self = .llamaCpp
        case .mlx: self = .mlx
        case .appleFoundationText: self = .afm
        }
    }

    /// Human-facing engine name — matches `Model.engineLabel`.
    var label: String {
        switch self {
        case .llamaCpp: return "llama.cpp"
        case .mlx: return "MLX"
        case .afm: return "Apple Foundation"
        }
    }

    /// One-line description of what this runtime runs, for the picker row.
    var detail: String {
        switch self {
        case .llamaCpp: return "GGUF models"
        case .mlx: return "MLX-format models"
        case .afm: return "Built-in system model · no download"
        }
    }

    /// Compact tag for dense UI (the Jobs-list runtime badge), where the full
    /// `label` ("Apple Foundation") is too wide next to a job title.
    var badgeLabel: String {
        switch self {
        case .llamaCpp: return "llama.cpp"
        case .mlx: return "MLX"
        case .afm: return "AFM"
        }
    }

}

/// plan-types `RuntimeType`. Every variant, not just the iOS ones, so an
/// unrecognized spelling fails to decode here exactly as it does there.
nonisolated enum RuntimeType: String, Decodable, CaseIterable, Hashable {
    case llamacppCliStockTools = "llamacpp_cli_stock_tools"
    case llamacppApkPipette = "llamacpp_apk_pipette"
    case llamacppIosPipette = "llamacpp_ios_pipette"
    case mlxMacosPipette = "mlx_macos_pipette"
    case mlxIosPipette = "mlx_ios_pipette"
    case dockerVllm = "docker_vllm"
    case dockerSglang = "docker_sglang"
    case uvVllm = "uv_vllm"
    case uvSglang = "uv_sglang"
    case uvOpenvino = "uv_openvino"
    case appleFoundation = "apple_foundation"

    /// The type of a concrete `Runtime` — the crate's `RuntimeType::of`. Exhaustive, so
    /// a new `Runtime` case fails to compile until it is classified.
    static func of(_ runtime: Runtime) -> RuntimeType {
        switch runtime {
        case .llamacppIosPipette: .llamacppIosPipette
        case .mlxIosPipette: .mlxIosPipette
        case .appleFoundation: .appleFoundation
        }
    }
}

// MARK: - Headless runtime selection

/// What the headless CLI accepts for `runtime=`, and which models each runtime loads.
///
/// The crate keeps no engine enum: `pipette-cli` matches on `Runtime` directly and gates
/// admissibility with one exhaustive `require_desktop_runtime`. This is that check
/// inverted — on a phone it is the desktop runtimes that are inadmissible — over the same
/// `RuntimeType` the rest of the surface speaks, so there is no second vocabulary to keep
/// in step.
nonisolated extension RuntimeType {
    /// The short tokens this client accepts, which predate the plan `type` tags.
    private static let headlessTokens: [String: RuntimeType] = [
        "afm": .appleFoundation, "llama": .llamacppIosPipette, "mlx": .mlxIosPipette,
    ]

    /// Parse a `runtime=` value, defaulting to MLX when absent. The one place the CLI maps
    /// the runtime token, shared by every verb that takes one.
    ///
    /// An unrecognized value is **refused**. It used to fall through to MLX, so
    /// `runtime=llamacpp_cli_stock_tools` — or a typo — ran an MLX cell and then submitted
    /// a descriptor naming the runtime it did not use.
    ///
    /// The switch is exhaustive, so a runtime added upstream is a compile error rather
    /// than a token that silently reads as a typo.
    static func parseHeadless(_ raw: String?) throws(HeadlessUsageError) -> RuntimeType {
        guard let raw, !raw.isEmpty else { return .mlxIosPipette }
        let value = raw.lowercased()
        if let token = headlessTokens[value] { return token }
        guard let type = RuntimeType(rawValue: value) else {
            throw .invalidValue(key: "runtime", value: raw)
        }
        switch type {
        // A type tag names which runtime, never which build — nothing to check against
        // this binary, and the result would record whatever ran. The plan sends the
        // canonical JSON.
        case .llamacppIosPipette, .mlxIosPipette, .appleFoundation:
            throw .rejected(key: "runtime",
                            reason: "`\(value)` names a runtime type, not a build; pass "
                                + "the JSON `Runtime` (see `headlessrun runtimes`)")
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino:
            throw .hostOnlyRuntime(value)
        }
    }

    /// Whether this runtime runs in-process on a phone. The mirror of the crate's
    /// `require_desktop_runtime`, and the reason a desktop runtime cannot reach a cell.
    var isIosRunnable: Bool {
        switch self {
        case .llamacppIosPipette, .mlxIosPipette, .appleFoundation: return true
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino: return false
        }
    }

    /// Short label for logs and result rows. The runtimes a phone cannot be answer with
    /// their own tag rather than a friendly name — they are unreachable here, and a name
    /// would suggest otherwise.
    var engineLabel: String {
        switch self {
        case .mlxIosPipette: return "MLX"
        case .llamacppIosPipette: return "llama.cpp"
        case .appleFoundation: return "AFM"
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino:
            return rawValue
        }
    }

    /// Whether a discovered model's format is the one this runtime loads. Apple Foundation
    /// pairs only with its built-in model; the file runtimes with their on-disk formats.
    /// The one engine gate the resolve step shares across every front-end.
    func matches(_ source: Model) -> Bool {
        switch (self, source) {
        case (.mlxIosPipette, .mlx), (.llamacppIosPipette, .ggufText),
             (.llamacppIosPipette, .ggufVision), (.appleFoundation, .appleFoundationText):
            return true
        default:
            return false
        }
    }
}
