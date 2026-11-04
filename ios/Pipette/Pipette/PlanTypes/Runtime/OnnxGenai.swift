import Foundation

// The ONNX Runtime GenAI runtime's iOS half: pipette-mgmt's
// `crates/pipette-plan-types/src/runtime/onnx_genai.rs`.

/// The crate's `OnnxGenaiStack`. onnxruntime-genai links against onnxruntime without
/// vendoring it, so a number depends on both pins.
nonisolated struct OnnxGenaiStack: Hashable, Sendable, Codable, CustomStringConvertible {
    let onnxruntime: SourceRepository
    let onnxruntimeGenai: SourceRepository

    enum CodingKeys: String, CodingKey {
        case onnxruntime
        case onnxruntimeGenai = "onnxruntime_genai"
    }

    /// The crate's `Display`.
    var description: String {
        "onnxruntime=\(onnxruntime) onnxruntime-genai=\(onnxruntimeGenai)"
    }

    /// From the build-time-generated `OnnxGenaiBuildInfo`. Each tree reports its release
    /// tag when the pin carries one, so the descriptor names the release a plan pins by.
    static var thisBuild: OnnxGenaiStack {
        OnnxGenaiStack(
            onnxruntime: SourceRepository(
                repositoryUrl: RepositoryUrl(OnnxGenaiBuildInfo.onnxruntimeRepositoryUrl),
                repositoryVersion: NonEmptyString(
                    validated: OnnxGenaiBuildInfo.onnxruntime.submissionVersion)),
            onnxruntimeGenai: SourceRepository(
                repositoryUrl: RepositoryUrl(OnnxGenaiBuildInfo.onnxruntimeGenaiRepositoryUrl),
                repositoryVersion: NonEmptyString(
                    validated: OnnxGenaiBuildInfo.onnxruntimeGenai.submissionVersion)))
    }
}

/// Build target for the ONNX Runtime GenAI iOS app.
nonisolated enum OnnxGenaiIosPipetteFlavor: String, Sendable, Codable {
    case iosArm64 = "ios-arm64"
}
