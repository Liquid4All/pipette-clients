import Foundation

// The crate's `Runtime` — build identity, and nothing else. A cell's load settings are
// `RuntimeFlags`, and `RunRequest` carries the two together, as upstream does.
// See `pipette-plan-types/src/runtime.rs`.

/// A repository coordinate, normalized to the one host-qualified, scheme-less form
/// `<host>/<org>/<repo>`. Mirrors the crate's `RepositoryUrl` nutype, whose sanitizer is
/// infallible — every pasted form reduces rather than being rejected, so a coordinate
/// copied from GitHub's "Code" dropdown in any form stores and compares identically.
///
/// Without this iOS kept whatever the plan wrote, so `https://github.com/x/y.git` and
/// `github.com/x/y` were two different repositories on this client and one upstream.
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
        // Userinfo (`git@…`, `user:token@…`) — a `@` before the first path separator.
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

    /// The `<org>/<repo>` portion, dropping the leading host — the crate's `org_repo`.
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

/// A source checkout: the repo URL plus the ref built from it. Mirrors the crate's
/// `SourceRepository` — the two travel together because a `repository_version` (a
/// release tag like `b5000`, or a commit) means nothing without its `repository_url`.
nonisolated struct SourceRepository: Hashable, Sendable, Codable {
    var repositoryUrl: RepositoryUrl
    var repositoryVersion: NonEmptyString

    /// The crate's `default_repository_url`, applied when a plan omits the key — the
    /// only field that defaults, because it is the only one the crate defaults.
    static let defaultRepositoryUrl = RepositoryUrl("github.com/ggml-org/llama.cpp")

    init(repositoryUrl: RepositoryUrl = Self.defaultRepositoryUrl, repositoryVersion: NonEmptyString) {
        self.repositoryUrl = repositoryUrl
        self.repositoryVersion = repositoryVersion
    }

    enum CodingKeys: String, CodingKey {
        case repositoryUrl = "repository_url"
        case sourceRepo = "source_repo"
        case repositoryVersion = "repository_version"
        case version
    }

    /// Accepts both spellings of each field, as the crate's `#[serde(alias = …)]` does.
    /// Encoding emits only the canonical pair.
    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repositoryUrl = try c.decodeIfPresent(RepositoryUrl.self, forKey: .repositoryUrl)
            ?? c.decodeIfPresent(RepositoryUrl.self, forKey: .sourceRepo)
            ?? Self.defaultRepositoryUrl
        // `NonEmptyString` rejects an empty pin on the way in: a descriptor asserting a
        // build it cannot name is worse than a refused claim.
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

/// The pinned Swift-package stack an iOS MLX build compiled against, from the app's
/// `Package.resolved`. Mirrors the crate's `MlxSwiftStack`.
nonisolated struct MlxSwiftStack: Hashable, Sendable, Codable {
    let mlxSwift: SourceRepository
    let mlxSwiftLm: SourceRepository
    let swiftTransformers: SourceRepository

    enum CodingKeys: String, CodingKey {
        case mlxSwift = "mlx_swift"
        case mlxSwiftLm = "mlx_swift_lm"
        case swiftTransformers = "swift_transformers"
    }

    /// What this binary actually compiled against, from the build-time-generated
    /// `MLXBuildInfo`. `mlx_swift_lm` is revision-pinned and reports the 9-char short
    /// revision, matching the llama.cpp short-commit convention rather than a full SHA.
    static var thisBuild: MlxSwiftStack {
        MlxSwiftStack(
            mlxSwift: SourceRepository(repositoryUrl: RepositoryUrl(MLXBuildInfo.mlxSwiftRepositoryUrl),
                                       repositoryVersion: NonEmptyString(validated: MLXBuildInfo.mlxSwiftVersion)),
            mlxSwiftLm: SourceRepository(repositoryUrl: RepositoryUrl(MLXBuildInfo.mlxSwiftLMRepositoryUrl),
                                         repositoryVersion: NonEmptyString(validated: MLXBuildInfo.mlxSwiftLMRevision)),
            swiftTransformers: SourceRepository(
                repositoryUrl: RepositoryUrl(MLXBuildInfo.swiftTransformersRepositoryUrl),
                repositoryVersion: NonEmptyString(validated: MLXBuildInfo.swiftTransformersVersion)))
    }
}

/// Build target for the llama.cpp iOS app. One member, as in the crate — its own type
/// so a second ABI is an added case rather than a reinterpreted string.
nonisolated enum LlamacppIosPipetteFlavor: String, Sendable, Codable {
    case iosArm64 = "ios-arm64"
}

/// Build target for the MLX iOS app. One member, as in the crate.
nonisolated enum MlxIosPipetteFlavor: String, Sendable, Codable {
    case iosArm64 = "ios-arm64"
}

nonisolated enum RuntimeIdentityError: Error, Equatable {
    /// A plan named a runtime this device has no engine for. Carries the spelling so a
    /// refusal can name it, which is what the claim path reports back.
    case unsupportedRuntimeType(String)
    case missingRepositoryVersion
}

/// plan-types `Runtime`, narrowed to the three a phone can be.
///
/// Identity only: which engine build this is, never how a cell configures it. The crate
/// draws the same line — `RunRequest` carries `runtime` and `runtime_flags` separately —
/// and iOS had blurred it by calling the engine-plus-settings pair `Runtime`.
///
/// Decoding a desktop runtime throws `unsupportedRuntimeType` rather than modelling an
/// `unsupported` case: a runtime this device cannot be is not a `Runtime`, and the claim
/// path already turns the throw into a named terminal refusal.
nonisolated enum Runtime: Hashable, Sendable, Codable {
    case llamacppIosPipette(source: SourceRepository, flavor: LlamacppIosPipetteFlavor,
                            privateThermal: Bool)
    case mlxIosPipette(packages: MlxSwiftStack, flavor: MlxIosPipetteFlavor,
                       privateThermal: Bool)
    /// Apple Foundation Models. The weights ship with the OS, so there is no repo or ref
    /// to pin (`runtime_version` is resolved on-device at submit) — but the readiness gate
    /// is the app's, so this carries the same build dimension as the other two.
    case appleFoundation(privateThermal: Bool)

    /// The identity of *this* binary for a given engine — the honest record of what
    /// compiled, from the build-time-generated build info. This is what a submitted
    /// descriptor carries, in place of whatever the plan declared.
    /// What this binary *is*, for a runtime type — the one definition of the built-in
    /// identity. A descriptor records it, `runtimes` advertises it, and a `--runtime`
    /// argument is compared against it, so the three cannot disagree.
    static func thisBuild(for type: RuntimeType) -> Runtime? {
        switch type {
        case .llamacppIosPipette:
            .llamacppIosPipette(
                source: SourceRepository(
                    repositoryUrl: SourceRepository.defaultRepositoryUrl,
                    repositoryVersion: NonEmptyString(validated: LlamaCppBuildInfo.submissionVersion)),
                flavor: .iosArm64, privateThermal: Runtime.privateThermalBuild)
        case .mlxIosPipette:
            .mlxIosPipette(packages: .thisBuild, flavor: .iosArm64,
                           privateThermal: Runtime.privateThermalBuild)
        case .appleFoundation:
            .appleFoundation(privateThermal: Runtime.privateThermalBuild)
        // Every other spelling names a runtime this build cannot be.
        default:
            nil
        }
    }

    /// Whether this runtime can run that kind of model at all — the cell's own coherence,
    /// independent of what is installed or downloaded. The crate's
    /// `is_compatible(&model, &runtime)`, reduced to the rows whose runtime this build can
    /// be: the pairings it also lists name desktop runtimes a `Runtime` cannot decode to.
    func accepts(modelType: ModelType) -> Bool {
        switch (self, modelType) {
        case (.llamacppIosPipette, .ggufText), (.llamacppIosPipette, .ggufVision): true
        case (.mlxIosPipette, .mlx): true
        case (.appleFoundation, .appleFoundationText): true
        default: false
        }
    }

    static func thisBuild(for model: Model) -> Runtime {
        switch model {
        case .ggufText, .ggufVision:
            .llamacppIosPipette(
                source: SourceRepository(
                    repositoryUrl: SourceRepository.defaultRepositoryUrl,
                    repositoryVersion: NonEmptyString(validated: LlamaCppBuildInfo.submissionVersion)),
                flavor: .iosArm64, privateThermal: Runtime.privateThermalBuild)
        case .mlx:
            .mlxIosPipette(packages: .thisBuild, flavor: .iosArm64,
                           privateThermal: Runtime.privateThermalBuild)
        case .appleFoundationText:
            .appleFoundation(privateThermal: Runtime.privateThermalBuild)
        }
    }

    /// What this runtime says about the private thermal gate. Apple Foundation carries no
    /// build of ours at all, so it declares nothing and answers `false`.
    var privateThermal: Bool {
        switch self {
        case let .llamacppIosPipette(_, _, value), let .mlxIosPipette(_, _, value),
             let .appleFoundation(value):
            value
        }
    }

    /// Whether this binary compiled in the private SoC-temperature read, asked of the
    /// translation unit that owns the `#ifdef` rather than re-tested with a Swift
    /// condition: the two preprocessors take separate flags, and a build that set only one
    /// would claim a gate it does not have.
    static var privateThermalBuild: Bool { pipette_private_thermal_build() != 0 }

    private enum CodingKeys: String, CodingKey {
        case type, flavor, packages
        case privateThermal = "private_thermal"
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .type)
        switch RuntimeType(rawValue: raw) {
        case .llamacppIosPipette:
            // `SourceRepository` is flattened into the runtime, so it decodes from this
            // same container rather than a nested key.
            self = .llamacppIosPipette(
                source: try SourceRepository(from: decoder),
                flavor: try c.decode(LlamacppIosPipetteFlavor.self, forKey: .flavor),
                // Absent means a stock build, as upstream's `#[serde(default)]` has it, so
                // a plan written before the field keeps meaning what it meant.
                privateThermal: try c.decodeIfPresent(Bool.self, forKey: .privateThermal) ?? false)
        case .mlxIosPipette:
            self = .mlxIosPipette(
                packages: try c.decode(MlxSwiftStack.self, forKey: .packages),
                flavor: try c.decode(MlxIosPipetteFlavor.self, forKey: .flavor),
                privateThermal: try c.decodeIfPresent(Bool.self, forKey: .privateThermal) ?? false)
        case .appleFoundation:
            self = .appleFoundation(
                privateThermal: try c.decodeIfPresent(Bool.self, forKey: .privateThermal) ?? false)
        default:
            throw RuntimeIdentityError.unsupportedRuntimeType(raw)
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(RuntimeType.of(self).rawValue, forKey: .type)
        switch self {
        case let .llamacppIosPipette(source, flavor, privateThermal):
            try source.encode(to: encoder)
            try c.encode(flavor, forKey: .flavor)
            // Omitted when false, matching upstream's `skip_serializing_if`: a stock
            // build's wire form is unchanged by this field existing.
            if privateThermal { try c.encode(true, forKey: .privateThermal) }
        case let .mlxIosPipette(packages, flavor, privateThermal):
            try c.encode(packages, forKey: .packages)
            try c.encode(flavor, forKey: .flavor)
            if privateThermal { try c.encode(true, forKey: .privateThermal) }
        case let .appleFoundation(privateThermal):
            if privateThermal { try c.encode(true, forKey: .privateThermal) }
        }
    }
}
