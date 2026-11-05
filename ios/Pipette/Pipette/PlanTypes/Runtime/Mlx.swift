import Foundation

// The MLX runtime's iOS half: pipette-mgmt's
// `crates/pipette-plan-types/src/runtime/mlx.rs`.

/// The crate's `MlxSwiftStack`: the Swift packages an iOS MLX build compiled against.
nonisolated struct MlxSwiftStack: Hashable, Sendable, Codable, CustomStringConvertible {
    let mlxSwift: SourceRepository
    let mlxSwiftLm: SourceRepository
    let swiftTransformers: SourceRepository

    enum CodingKeys: String, CodingKey {
        case mlxSwift = "mlx_swift"
        case mlxSwiftLm = "mlx_swift_lm"
        case swiftTransformers = "swift_transformers"
    }

    /// The crate's `Display`.
    var description: String {
        "mlx-swift=\(mlxSwift) mlx-swift-lm=\(mlxSwiftLm) swift-transformers=\(swiftTransformers)"
    }

    /// From the build-time-generated `MLXBuildInfo`. Each package reports its release tag
    /// when the pin carries one, so the descriptor names the release a plan pins by.
    static var thisBuild: MlxSwiftStack {
        MlxSwiftStack(
            mlxSwift: SourceRepository(
                repositoryUrl: RepositoryUrl(MLXBuildInfo.mlxSwiftRepositoryUrl),
                repositoryVersion: NonEmptyString(validated: MLXBuildInfo.mlxSwift.submissionVersion)),
            mlxSwiftLm: SourceRepository(
                repositoryUrl: RepositoryUrl(MLXBuildInfo.mlxSwiftLMRepositoryUrl),
                repositoryVersion: NonEmptyString(validated: MLXBuildInfo.mlxSwiftLM.submissionVersion)),
            swiftTransformers: SourceRepository(
                repositoryUrl: RepositoryUrl(MLXBuildInfo.swiftTransformersRepositoryUrl),
                repositoryVersion: NonEmptyString(
                    validated: MLXBuildInfo.swiftTransformers.submissionVersion)))
    }
}

/// Build target for the MLX iOS app.
nonisolated enum MlxIosPipetteFlavor: String, Sendable, Codable {
    case iosArm64 = "ios-arm64"
}
