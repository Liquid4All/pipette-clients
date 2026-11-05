import Foundation

// The runtime a cell names: pipette-mgmt's
// `crates/pipette-plan-types/src/runtime/mod.rs`. Each engine's stack and flavor is in its
// own file beside this one.

nonisolated let runtimeBuildFlagPrefix = "runtime:descriptor:"

/// For each runtime that the app compiles, the crate's `runtime_build_flag` returns
/// `runtime:descriptor:<digest>`. The full descriptor's `descriptorDigest` supplies
/// `<digest>`. The server's catalog offers a build only to a device that reports this flag.
/// The function returns nil for Apple Foundation, which ships with the OS.
/// `ContractTests` pins the output to the crate's
/// through `crates/pipette-contract/fixtures/runtime-build-flags.json`.
nonisolated func runtimeBuildFlag(_ runtime: Runtime) throws -> String? {
    switch runtime {
    case .llamacppIosPipette, .mlxIosPipette, .onnxGenaiIosPipette:
        try runtimeBuildFlagPrefix + descriptorDigest(runtime)
    case .appleFoundation:
        nil
    }
}

/// The crate's `Runtime`, narrowed to the four a phone can be. Identity only: a cell's
/// load settings are `RuntimeFlags`.
///
/// Decoding a desktop runtime throws `unsupportedRuntimeType`, which the claim path turns
/// into a named terminal refusal.
nonisolated enum Runtime: Hashable, Sendable, Codable, CustomStringConvertible {
    case llamacppIosPipette(source: SourceRepository, flavor: LlamacppIosPipetteFlavor,
                            privateApis: PrivateApis)
    case mlxIosPipette(packages: MlxSwiftStack, flavor: MlxIosPipetteFlavor,
                       privateApis: PrivateApis)
    case onnxGenaiIosPipette(packages: OnnxGenaiStack, flavor: OnnxGenaiIosPipetteFlavor,
                             privateApis: PrivateApis)
    /// Apple Foundation Models. Nothing to pin (`runtime_version` is resolved on-device at
    /// submit), but the readiness gate and memory bracket are the app's, so it carries
    /// `privateApis` like the others.
    case appleFoundation(privateApis: PrivateApis)

    /// The crate's `Display`, for logs and refusals; not an input spelling.
    var description: String {
        switch self {
        case let .llamacppIosPipette(source, flavor, _): "\(source):\(flavor.rawValue)"
        case let .mlxIosPipette(packages, flavor, _): "\(packages):\(flavor.rawValue)"
        case let .onnxGenaiIosPipette(packages, flavor, _): "\(packages):\(flavor.rawValue)"
        case .appleFoundation: RuntimeType.appleFoundation.rawValue
        }
    }

    /// What this binary is, from the build-time-generated build info: the one definition a
    /// descriptor records, `runtimes` advertises and a `--runtime` argument is compared
    /// against.
    static func thisBuild(for type: RuntimeType) -> Runtime? {
        switch type {
        case .llamacppIosPipette: thisLlamacppIosPipette
        case .mlxIosPipette: thisMlxIosPipette
        case .onnxGenaiIosPipette: thisOnnxGenaiIosPipette
        case .appleFoundation: thisAppleFoundation
        case .llamacppCliStockTools, .llamacppApkPipette, .mlxMacosPipette,
             .coreAiMacosPipette,
             .dockerVllm, .dockerSglang, .uvVllm, .uvSglang, .uvOpenvino,
             .uvOnnxGenai, .ovms, .trtEdgellm:
            nil
        }
    }

    /// The build that loads `model`'s format; format to engine is 1:1.
    static func thisBuild(for model: Model) -> Runtime {
        switch model {
        case .ggufText, .ggufVision, .ggufVisionSpeculative: thisLlamacppIosPipette
        case .mlx: thisMlxIosPipette
        case .onnxGenai: thisOnnxGenaiIosPipette
        case .appleFoundationText: thisAppleFoundation
        }
    }

    /// A llama.cpp build the app carries, as a cell names it.
    static func carried(_ pin: LlamaPin) -> Runtime {
        .llamacppIosPipette(source: pin.source, flavor: .iosArm64, privateApis: .thisBuild)
    }

    private static var thisLlamacppIosPipette: Runtime {
        carried(LlamaPins.current)
    }

    private static var thisMlxIosPipette: Runtime {
        .mlxIosPipette(packages: .thisBuild, flavor: .iosArm64, privateApis: .thisBuild)
    }

    private static var thisOnnxGenaiIosPipette: Runtime {
        .onnxGenaiIosPipette(packages: .thisBuild, flavor: .iosArm64, privateApis: .thisBuild)
    }

    private static var thisAppleFoundation: Runtime {
        .appleFoundation(privateApis: .thisBuild)
    }

    var privateApis: PrivateApis {
        switch self {
        case let .llamacppIosPipette(_, _, value), let .mlxIosPipette(_, _, value),
             let .onnxGenaiIosPipette(_, _, value), let .appleFoundation(value):
            value
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type, flavor, packages
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .type)
        switch RuntimeType(rawValue: raw) {
        case .llamacppIosPipette:
            // `SourceRepository` and `PrivateApis` are flattened into the runtime, as the
            // crate's `#[serde(flatten)]` has them.
            self = .llamacppIosPipette(
                source: try SourceRepository(from: decoder),
                flavor: try c.decode(LlamacppIosPipetteFlavor.self, forKey: .flavor),
                privateApis: try PrivateApis(from: decoder))
        case .mlxIosPipette:
            self = .mlxIosPipette(
                packages: try c.decode(MlxSwiftStack.self, forKey: .packages),
                flavor: try c.decode(MlxIosPipetteFlavor.self, forKey: .flavor),
                privateApis: try PrivateApis(from: decoder))
        case .onnxGenaiIosPipette:
            self = .onnxGenaiIosPipette(
                packages: try c.decode(OnnxGenaiStack.self, forKey: .packages),
                flavor: try c.decode(OnnxGenaiIosPipetteFlavor.self, forKey: .flavor),
                privateApis: try PrivateApis(from: decoder))
        case .appleFoundation:
            self = .appleFoundation(privateApis: try PrivateApis(from: decoder))
        default:
            throw RuntimeIdentityError.unsupportedRuntimeType(raw)
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(RuntimeType.of(self).rawValue, forKey: .type)
        switch self {
        case let .llamacppIosPipette(source, flavor, _):
            try source.encode(to: encoder)
            try c.encode(flavor, forKey: .flavor)
        case let .mlxIosPipette(packages, flavor, _):
            try c.encode(packages, forKey: .packages)
            try c.encode(flavor, forKey: .flavor)
        case let .onnxGenaiIosPipette(packages, flavor, _):
            try c.encode(packages, forKey: .packages)
            try c.encode(flavor, forKey: .flavor)
        case .appleFoundation:
            break
        }
        try privateApis.encode(to: encoder)
    }
}

/// The crate's `PrivateApis`: the private Apple APIs an iOS build compiled in. App Review
/// rejects both, so only internal builds carry them, and each changes what a number means.
///
/// `private_memory` defaults to `private_thermal` and is written only when it differs, so
/// a build with both on keeps the descriptor (and digest) builds had before `private_memory`
/// existed. Flattened into the runtime's container.
nonisolated struct PrivateApis: Hashable, Sendable, Codable {
    /// The SoC die temperature through the private IOHID sensors (`PIPETTE_PRIVATE_THERMAL`).
    var thermal: Bool
    /// The reclaim-ring flush and kernel footprint peak (`PIPETTE_PRIVATE_MEMORY`).
    var memory: Bool

    /// The crate's `Default`.
    static let stock = PrivateApis(thermal: false, memory: false)

    /// The thermal read is asked of the C translation unit that owns the `#ifdef`, since
    /// the C and Swift preprocessors take separate flags; the memory calls are Swift-only.
    static var thisBuild: PrivateApis {
        PrivateApis(thermal: pipette_private_thermal_build() != 0,
                    memory: BuildFlavor.hasPrivateMemory)
    }

    private enum CodingKeys: String, CodingKey {
        case privateThermal = "private_thermal"
        case privateMemory = "private_memory"
    }

    init(thermal: Bool, memory: Bool) {
        self.thermal = thermal
        self.memory = memory
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        thermal = try c.decodeIfPresent(Bool.self, forKey: .privateThermal) ?? false
        memory = try c.decodeIfPresent(Bool.self, forKey: .privateMemory) ?? thermal
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        if thermal { try c.encode(true, forKey: .privateThermal) }
        if memory != thermal { try c.encode(memory, forKey: .privateMemory) }
    }
}

nonisolated extension Runtime {
    /// The crate's `Runtime::cli_ref`.
    var cliRef: String {
        func coordinate(_ repository: SourceRepository) -> String {
            "\(repository.repositoryUrl.value)@\(repository.repositoryVersion.value)"
        }
        switch self {
        case let .llamacppIosPipette(source, flavor, _):
            return "\(source.repositoryVersion.value):\(flavor.rawValue)"
        case let .mlxIosPipette(packages, flavor, _):
            return "mlx-swift=\(coordinate(packages.mlxSwift)) "
                + "mlx-swift-lm=\(coordinate(packages.mlxSwiftLm)) "
                + "swift-transformers=\(coordinate(packages.swiftTransformers)):\(flavor.rawValue)"
        case let .onnxGenaiIosPipette(packages, flavor, _):
            return "onnxruntime=\(coordinate(packages.onnxruntime)) "
                + "onnxruntime-genai=\(coordinate(packages.onnxruntimeGenai)):\(flavor.rawValue)"
        case .appleFoundation:
            return "apple_foundation"
        }
    }
}

/// The crate's `RepositoryUrl`: normalized to `<host>/<org>/<repo>` by an infallible
/// sanitizer, so every form GitHub's "Code" dropdown offers stores and compares identically.
nonisolated struct RepositoryUrl: Hashable, Sendable, Codable, CustomStringConvertible {
    let value: String
    var description: String { value }

    init(_ raw: String) { value = Self.normalized(raw) }

    static func normalized(_ raw: String) -> String {
        var s = Substring(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        for scheme in ["https://", "http://", "ssh://", "git://"] where s.hasPrefix(scheme) {
            s = s.dropFirst(scheme.count)
            break
        }
        // Userinfo (`git@…`, `user:token@…`): a `@` before the first path separator.
        if let at = s.firstIndex(of: "@"), !s[s.startIndex..<at].contains("/") {
            s = s[s.index(after: at)...]
        }
        // The scp-like `host:org/repo` separator; a scheme-stripped path is left alone.
        var out = String(s)
        if let colon = s.firstIndex(of: ":"), !s[s.startIndex..<colon].contains("/") {
            out = s[s.startIndex..<colon] + "/" + s[s.index(after: colon)...]
        }
        while out.hasSuffix("/") { out.removeLast() }
        if out.hasSuffix(".git") { out.removeLast(4) }
        while out.hasSuffix("/") { out.removeLast() }
        return out
    }

    /// The crate's `org_repo`: the host dropped.
    var orgRepo: String {
        value.firstIndex(of: "/").map { String(value[value.index(after: $0)...]) } ?? value
    }

    nonisolated init(from decoder: any Decoder) throws {
        self.init(try decoder.singleValueContainer().decode(String.self))
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(value)
    }
}

/// The crate's `SourceRepository`: a `repository_version` (a release tag or a commit)
/// means nothing without its `repository_url`.
nonisolated struct SourceRepository: Hashable, Sendable, Codable, CustomStringConvertible {
    var repositoryUrl: RepositoryUrl
    var repositoryVersion: NonEmptyString

    init(repositoryUrl: RepositoryUrl, repositoryVersion: NonEmptyString) {
        self.repositoryUrl = repositoryUrl
        self.repositoryVersion = repositoryVersion
    }

    /// The crate's `Display`, e.g. `github.com/ggml-org/llama.cpp@b9050`. The repo is kept
    /// so a fork is not mistaken for upstream.
    var description: String { "\(repositoryUrl)@\(repositoryVersion)" }

    enum CodingKeys: String, CodingKey {
        case repositoryUrl = "repository_url"
        case sourceRepo = "source_repo"
        case repositoryVersion = "repository_version"
        case version
    }

    /// Accepts the crate's `#[serde(alias = …)]` spellings; encoding emits the canonical pair.
    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard let url = try c.decodeIfPresent(RepositoryUrl.self, forKey: .repositoryUrl)
            ?? c.decodeIfPresent(RepositoryUrl.self, forKey: .sourceRepo)
        else { throw RuntimeIdentityError.missingRepositoryUrl }
        repositoryUrl = url
        // A descriptor asserting a build it cannot name is worse than a refused claim.
        guard let version = try c.decodeIfPresent(NonEmptyString.self, forKey: .repositoryVersion)
            ?? c.decodeIfPresent(NonEmptyString.self, forKey: .version)
        else { throw RuntimeIdentityError.missingRepositoryVersion }
        repositoryVersion = version
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(repositoryUrl, forKey: .repositoryUrl)
        try c.encode(repositoryVersion, forKey: .repositoryVersion)
    }
}

/// The crate's `RuntimeType`, every variant, so an unrecognized spelling fails to decode
/// here as it does there.
nonisolated enum RuntimeType: String, Decodable, CaseIterable, Hashable {
    case llamacppCliStockTools = "llamacpp_cli_stock_tools"
    case llamacppApkPipette = "llamacpp_apk_pipette"
    case llamacppIosPipette = "llamacpp_ios_pipette"
    case mlxMacosPipette = "mlx_macos_pipette"
    case coreAiMacosPipette = "core_ai_macos_pipette"
    case mlxIosPipette = "mlx_ios_pipette"
    case dockerVllm = "docker_vllm"
    case dockerSglang = "docker_sglang"
    case uvVllm = "uv_vllm"
    case uvSglang = "uv_sglang"
    case uvOpenvino = "uv_openvino"
    case uvOnnxGenai = "uv_onnx_genai"
    case ovms = "ovms"
    case trtEdgellm = "trt_edgellm"
    case onnxGenaiIosPipette = "onnx_genai_ios_pipette"
    case appleFoundation = "apple_foundation"

    /// The crate's `RuntimeType::of`.
    static func of(_ runtime: Runtime) -> RuntimeType {
        switch runtime {
        case .llamacppIosPipette: .llamacppIosPipette
        case .mlxIosPipette: .mlxIosPipette
        case .onnxGenaiIosPipette: .onnxGenaiIosPipette
        case .appleFoundation: .appleFoundation
        }
    }
}

nonisolated enum RuntimeIdentityError: Error, Equatable {
    /// A runtime this device has no engine for, by name for the claim path's refusal.
    case unsupportedRuntimeType(String)
    case missingRepositoryUrl
    case missingRepositoryVersion
}
