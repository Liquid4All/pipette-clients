import Foundation

// What the headless CLI accepts for `runtime=`, and which models each runtime loads.
//
// iOS-only, so it lives here rather than in `PlanTypes/Runtime/`. It is `pipette-cli`'s
// `require_desktop_runtime` inverted: on a phone the desktop runtimes are inadmissible.

nonisolated extension RuntimeType {
    /// This client's shorthand for the four runtimes it can be. Not the crate's
    /// `headless_token()` (the `type` tag), which is refused here because it names no build.
    private static let shortAliases: [String: RuntimeType] = [
        "afm": .appleFoundation, "llama": .llamacppIosPipette, "mlx": .mlxIosPipette,
        "onnx": .onnxGenaiIosPipette,
    ]

    /// Parse a `runtime=` value, defaulting to MLX when absent. An unrecognized value is
    /// refused, never defaulted, so a result cannot name a runtime it did not use.
    /// Exhaustive, so a runtime added upstream is a compile error here.
    static func parseHeadless(_ raw: String?) throws(HeadlessUsageError) -> RuntimeType {
        guard let raw, !raw.isEmpty else { return .mlxIosPipette }
        let value = raw.lowercased()
        if let alias = shortAliases[value] { return alias }
        guard let type = RuntimeType(rawValue: value) else {
            throw .invalidValue(key: "runtime", value: raw)
        }
        switch type {
        // A type tag names no build to check against this binary, so the result would
        // record whatever ran.
        case .llamacppIosPipette, .mlxIosPipette, .onnxGenaiIosPipette, .appleFoundation:
            throw .rejected(key: "runtime",
                            reason: "`\(value)` names a runtime type, not a build; pass "
                                + "the JSON `Runtime` (see `headlessrun runtimes`)")
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .coreAiMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino,
             .uvOnnxGenai, .ovms, .trtEdgellm:
            throw .hostOnlyRuntime(value)
        }
    }

    /// Runtimes a phone cannot be answer with their raw tag, so a friendly name does not
    /// suggest they are reachable.
    var engineLabel: String { RuntimeKind(self)?.label ?? rawValue }
}

/// `isCompatible(_:_:)` for selectors that hold a runtime type rather than a `Runtime`.
/// A type this build cannot be pairs with nothing.
nonisolated func isCompatible(_ model: Model, _ runtime: RuntimeType) -> Bool {
    Runtime.thisBuild(for: runtime).map { isCompatible(model, $0) } ?? false
}
