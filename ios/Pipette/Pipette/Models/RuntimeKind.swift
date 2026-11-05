import Foundation

/// The inference engine as a bare choice, what the New Job runtime step selects before
/// any model or config exists. iOS-only: the crate matches on `Runtime`, the build
/// identity a run records.
nonisolated enum RuntimeKind: String, CaseIterable, Identifiable, Equatable {
    case llamaCpp
    case mlx
    case onnxGenai
    case afm

    var id: String { rawValue }

    init(_ model: Model) {
        switch model {
        case .ggufText, .ggufVision, .ggufVisionSpeculative: self = .llamaCpp
        case .mlx: self = .mlx
        case .onnxGenai: self = .onnxGenai
        case .appleFoundationText: self = .afm
        }
    }

    /// The engine a runtime type names, or `nil` for one no phone can be.
    init?(_ type: RuntimeType) {
        switch type {
        case .llamacppIosPipette: self = .llamaCpp
        case .mlxIosPipette: self = .mlx
        case .onnxGenaiIosPipette: self = .onnxGenai
        case .appleFoundation: self = .afm
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .coreAiMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino,
             .uvOnnxGenai, .ovms, .trtEdgellm:
            return nil
        }
    }

    /// The one spelling of an engine's name; `Model.engineLabel` and
    /// `RuntimeType.engineLabel` read it.
    var label: String {
        switch self {
        case .llamaCpp: return "llama.cpp"
        case .mlx: return "MLX"
        case .onnxGenai: return "ONNX Runtime GenAI"
        case .afm: return "Apple Foundation"
        }
    }

    var detail: String {
        switch self {
        case .llamaCpp: return "GGUF models"
        case .mlx: return "MLX-format models"
        case .onnxGenai: return "ONNX Runtime GenAI models · CPU"
        case .afm: return "Built-in system model · no download"
        }
    }

    var badgeLabel: String {
        switch self {
        case .llamaCpp: return "llama.cpp"
        case .mlx: return "MLX"
        case .onnxGenai: return "ONNX"
        case .afm: return "AFM"
        }
    }
}
