import Foundation

// pipette-mgmt's `crates/pipette-plan-types/src/model/mod.rs`. The GGUF formats
// are in `Model/Gguf.swift`; what the app derives from a model for its screens is in
// `Models/Model+Catalog.swift`.

/// The on-device subset of the crate's `Model` (no `Torch`/`Openvino`: no on-device engine
/// runs either). Identity only, no flags or resolved paths; a run carries it beside its
/// bound form (source rewritten to an `absolute*` arm) as `DeclaredBound<Model>`.
nonisolated enum Model: Hashable, Sendable, Codable, CustomStringConvertible {
    case ggufText(GgufText)
    case ggufVision(GgufVision)
    case ggufVisionSpeculative(GgufVisionSpeculative)
    case mlx(Mlx)
    case onnxGenai(OnnxGenai)
    /// Apple's on-device system model (FoundationModels). It ships with the OS, so callers
    /// switch on it to skip path resolution, the memory gate and the download, never to
    /// fabricate a path or repo.
    case appleFoundationText

    /// The crate's `Model::auth_token`.
    var authToken: AuthToken? {
        switch self {
        case let .ggufText(m): m.source.authToken
        case let .ggufVision(m): m.source.authToken
        case let .ggufVisionSpeculative(m): m.source.authToken
        case let .mlx(m): m.source.authToken
        case let .onnxGenai(m): m.source.authToken
        case .appleFoundationText: nil
        }
    }

    /// The crate's `Model::without_auth_token`: the form persisted anywhere on disk.
    var withoutAuthToken: Model {
        switch self {
        case let .ggufText(m): .ggufText(.init(source: m.source.withoutAuthToken))
        case let .ggufVision(m): .ggufVision(.init(source: m.source.withoutAuthToken))
        case let .ggufVisionSpeculative(m):
            .ggufVisionSpeculative(.init(source: m.source.withoutAuthToken))
        case let .mlx(m): .mlx(.init(source: m.source.withoutAuthToken))
        case let .onnxGenai(m): .onnxGenai(.init(source: m.source.withoutAuthToken))
        case .appleFoundationText: .appleFoundationText
        }
    }

    /// The form a digest is taken over. The server drops `custom_label` from every object
    /// whose `source` or `type` starts with `vault_`; a model nests none, so this matches.
    var withoutCustomLabel: Model {
        switch self {
        case let .ggufText(m): .ggufText(.init(source: m.source.withoutCustomLabel))
        case let .ggufVision(m): .ggufVision(.init(source: m.source.withoutCustomLabel))
        case let .ggufVisionSpeculative(m):
            .ggufVisionSpeculative(.init(source: m.source.withoutCustomLabel))
        case let .mlx(m): .mlx(.init(source: m.source.withoutCustomLabel))
        case let .onnxGenai(m): .onnxGenai(.init(source: m.source.withoutCustomLabel))
        case .appleFoundationText: .appleFoundationText
        }
    }

    /// The crate's `Display for Model`, for logs, refusals and listings. Not the wire: a
    /// submission carries `model_descriptor`. Apple Foundation reports the fixed slug its
    /// submissions carry.
    var description: String {
        switch self {
        case let .ggufText(m): m.description
        case let .ggufVision(m): m.description
        case let .ggufVisionSpeculative(m): m.description
        case let .mlx(m): m.description
        case let .onnxGenai(m): m.description
        case .appleFoundationText: AFMRuntime.submissionModelName
        }
    }
}

// `Codable` is declared on the (nonisolated) type above so the conformance stays
// nonisolated; stating it on this extension would infer `@MainActor` isolation under the
// module's default-actor-isolation setting and make decode unusable off-main.
extension Model {
    /// The crate's key set: `type` picks the format, `source` where it lives, with the HF
    /// coordinate flattened alongside.
    private enum CodingKeys: String, CodingKey {
        case type, source, org
        case repoName = "repo_name"
        case revision
        case authToken = "auth_token"
        case path, model, mmproj, prefix, dir, url
        case sha256
        case modelSha256 = "model_sha256"
        case mmprojSha256 = "mmproj_sha256"
        case draft
        case draftSha256 = "draft_sha256"
        case principalId = "principal_id"
        case customLabel = "custom_label"
    }

    /// `torch` and `openvino` are recognized only to be refused at parse time rather than
    /// strand a run at load time.
    private enum TypeTag: String {
        case ggufText = "gguf_text", ggufVision = "gguf_vision", mlx, torch, openvino
        case ggufVisionSpeculative = "gguf_vision_speculative"
        case onnxGenai = "onnx_genai"
        case appleFoundationText = "apple_foundation_text"
    }

    /// Every arm the crate has, so a store form round-trips and a body authored for the CLI
    /// is readable; runnability is decided separately.
    private enum SourceTag: String {
        case huggingFace = "huggingface"
        case relativeFile = "relative_file", absoluteFile = "absolute_file"
        case relativeFiles = "relative_files", absoluteFiles = "absolute_files"
        case relativeDir = "relative_dir", absoluteDir = "absolute_dir"
        case url
        case vaultUrl = "vault_url", vaultArchive = "vault_archive"
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .type)
        guard let tag = TypeTag(rawValue: raw) else { throw ModelError.unknownModelType(raw) }
        if tag == .appleFoundationText {
            self = .appleFoundationText
            return
        }
        let rawSource = try c.decode(String.self, forKey: .source)
        guard let sourceTag = SourceTag(rawValue: rawSource) else {
            throw ModelError.unknownModelType("\(raw)/\(rawSource)")
        }
        func hfRepo() throws -> HfRepo {
            HfRepo(
                org: try c.decode(HfOrg.self, forKey: .org),
                repoName: try c.decode(HfRepoName.self, forKey: .repoName),
                revision: try c.decodeIfPresent(HfRevision.self, forKey: .revision),
                authToken: try c.decodeIfPresent(AuthToken.self, forKey: .authToken))
        }
        func mismatch() -> ModelError { .unknownModelType("\(raw)/\(rawSource)") }
        func principalId() throws -> VaultPrincipalId {
            try c.decode(VaultPrincipalId.self, forKey: .principalId)
        }
        // Absent or `null` is no label, as the crate's `Option` reads both.
        func customLabel() throws -> CustomLabel {
            CustomLabel(try c.decodeIfPresent(NonEmptyTrimmedString.self, forKey: .customLabel))
        }

        switch tag {
        case .ggufText:
            switch sourceTag {
            case .huggingFace:
                self = .ggufText(.init(source: .huggingFace(
                    repo: try hfRepo(),
                    path: try c.decode(RepoSubpath.self, forKey: .path),
                    sha256: try c.decodeIfPresent(Sha256.self, forKey: .sha256))))
            case .relativeFile:
                self = .ggufText(.init(source: .relativeFile(
                    path: try c.decode(RelativePath.self, forKey: .path))))
            case .absoluteFile:
                self = .ggufText(.init(source: .absoluteFile(
                    path: try c.decode(AbsolutePath.self, forKey: .path))))
            case .url:
                self = .ggufText(.init(source: .url(
                    url: try c.decode(ResourceUrl.self, forKey: .url),
                    sha256: try c.decodeIfPresent(Sha256.self, forKey: .sha256))))
            case .vaultUrl:
                self = .ggufText(.init(source: .vaultUrl(
                    url: try c.decode(VaultPath.self, forKey: .url),
                    sha256: try c.decodeIfPresent(Sha256.self, forKey: .sha256),
                    principalId: try principalId(), customLabel: try customLabel())))
            default: throw mismatch()
            }
        case .ggufVision:
            switch sourceTag {
            case .huggingFace:
                self = .ggufVision(.init(source: .huggingFace(
                    repo: try hfRepo(),
                    model: try c.decode(RepoSubpath.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(RepoSubpath.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256))))
            case .relativeFiles:
                self = .ggufVision(.init(source: .relativeFiles(
                    model: try c.decode(RelativePath.self, forKey: .model),
                    mmproj: try c.decode(RelativePath.self, forKey: .mmproj))))
            case .absoluteFiles:
                self = .ggufVision(.init(source: .absoluteFiles(
                    model: try c.decode(AbsolutePath.self, forKey: .model),
                    mmproj: try c.decode(AbsolutePath.self, forKey: .mmproj))))
            case .url:
                self = .ggufVision(.init(source: .url(
                    model: try c.decode(ResourceUrl.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(ResourceUrl.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256))))
            case .vaultUrl:
                self = .ggufVision(.init(source: .vaultUrl(
                    model: try c.decode(VaultPath.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(VaultPath.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256),
                    principalId: try principalId(), customLabel: try customLabel())))
            default: throw mismatch()
            }
        case .ggufVisionSpeculative:
            // Both drafter arrangements decode; whether this engine can run one is decided
            // at load.
            switch sourceTag {
            case .huggingFace:
                self = .ggufVisionSpeculative(.init(source: .huggingFace(
                    repo: try hfRepo(),
                    model: try c.decode(RepoSubpath.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(RepoSubpath.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256),
                    draft: try c.decodeIfPresent(RepoSubpath.self, forKey: .draft),
                    draftSha256: try c.decodeIfPresent(Sha256.self, forKey: .draftSha256))))
            case .relativeFiles:
                self = .ggufVisionSpeculative(.init(source: .relativeFiles(
                    model: try c.decode(RelativePath.self, forKey: .model),
                    mmproj: try c.decode(RelativePath.self, forKey: .mmproj),
                    draft: try c.decodeIfPresent(RelativePath.self, forKey: .draft))))
            case .absoluteFiles:
                self = .ggufVisionSpeculative(.init(source: .absoluteFiles(
                    model: try c.decode(AbsolutePath.self, forKey: .model),
                    mmproj: try c.decode(AbsolutePath.self, forKey: .mmproj),
                    draft: try c.decodeIfPresent(AbsolutePath.self, forKey: .draft))))
            case .url:
                self = .ggufVisionSpeculative(.init(source: .url(
                    model: try c.decode(ResourceUrl.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(ResourceUrl.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256),
                    draft: try c.decodeIfPresent(ResourceUrl.self, forKey: .draft),
                    draftSha256: try c.decodeIfPresent(Sha256.self, forKey: .draftSha256))))
            case .vaultUrl:
                self = .ggufVisionSpeculative(.init(source: .vaultUrl(
                    model: try c.decode(VaultPath.self, forKey: .model),
                    modelSha256: try c.decodeIfPresent(Sha256.self, forKey: .modelSha256),
                    mmproj: try c.decode(VaultPath.self, forKey: .mmproj),
                    mmprojSha256: try c.decodeIfPresent(Sha256.self, forKey: .mmprojSha256),
                    draft: try c.decodeIfPresent(VaultPath.self, forKey: .draft),
                    draftSha256: try c.decodeIfPresent(Sha256.self, forKey: .draftSha256),
                    principalId: try principalId(), customLabel: try customLabel())))
            default: throw mismatch()
            }
        case .mlx:
            switch sourceTag {
            case .huggingFace:
                self = .mlx(.init(source: .huggingFace(
                    repo: try hfRepo(),
                    prefix: try c.decodeIfPresent(RepoSubpath.self, forKey: .prefix))))
            case .relativeDir:
                self = .mlx(.init(source: .relativeDir(
                    dir: try c.decode(RelativePath.self, forKey: .dir))))
            case .absoluteDir:
                self = .mlx(.init(source: .absoluteDir(
                    dir: try c.decode(AbsolutePath.self, forKey: .dir))))
            case .vaultArchive:
                self = .mlx(.init(source: .vaultArchive(
                    url: try c.decode(VaultPath.self, forKey: .url),
                    sha256: try c.decodeIfPresent(Sha256.self, forKey: .sha256),
                    principalId: try principalId(), customLabel: try customLabel())))
            default: throw mismatch()
            }
        case .onnxGenai:
            switch sourceTag {
            case .huggingFace:
                self = .onnxGenai(.init(source: .huggingFace(
                    repo: try hfRepo(),
                    prefix: try c.decodeIfPresent(RepoSubpath.self, forKey: .prefix))))
            case .relativeDir:
                self = .onnxGenai(.init(source: .relativeDir(
                    dir: try c.decode(RelativePath.self, forKey: .dir))))
            case .absoluteDir:
                self = .onnxGenai(.init(source: .absoluteDir(
                    dir: try c.decode(AbsolutePath.self, forKey: .dir))))
            case .vaultArchive:
                self = .onnxGenai(.init(source: .vaultArchive(
                    url: try c.decode(VaultPath.self, forKey: .url),
                    sha256: try c.decodeIfPresent(Sha256.self, forKey: .sha256),
                    principalId: try principalId(), customLabel: try customLabel())))
            default: throw mismatch()
            }
        case .torch, .openvino, .appleFoundationText:
            throw ModelError.unknownModelType(raw)
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        if case .appleFoundationText = self {
            try c.encode(TypeTag.appleFoundationText.rawValue, forKey: .type)
            return
        }
        func encodeRepo(_ repo: HfRepo) throws {
            try c.encode(SourceTag.huggingFace.rawValue, forKey: .source)
            try c.encode(repo.org, forKey: .org)
            try c.encode(repo.repoName, forKey: .repoName)
            try c.encodeIfPresent(repo.revision, forKey: .revision)
            // The token is never written; decode still accepts one, as a claim delivers it.
        }
        switch self {
        case let .ggufText(m):
            try c.encode(TypeTag.ggufText.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, path, sha256):
                try encodeRepo(repo)
                try c.encode(path, forKey: .path)
                try c.encodeIfPresent(sha256, forKey: .sha256)
            case let .relativeFile(path):
                try c.encode(SourceTag.relativeFile.rawValue, forKey: .source)
                try c.encode(path, forKey: .path)
            case let .absoluteFile(path):
                try c.encode(SourceTag.absoluteFile.rawValue, forKey: .source)
                try c.encode(path, forKey: .path)
            case let .url(url, sha256):
                try c.encode(SourceTag.url.rawValue, forKey: .source)
                try c.encode(url, forKey: .url)
                try c.encodeIfPresent(sha256, forKey: .sha256)
            case let .vaultUrl(url, sha256, principalId, customLabel):
                try c.encode(SourceTag.vaultUrl.rawValue, forKey: .source)
                try c.encode(url, forKey: .url)
                try c.encodeIfPresent(sha256, forKey: .sha256)
                try c.encode(principalId, forKey: .principalId)
                try c.encodeIfPresent(customLabel.value, forKey: .customLabel)
            }
        case let .ggufVision(m):
            try c.encode(TypeTag.ggufVision.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha):
                try encodeRepo(repo)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
            case let .relativeFiles(model, mmproj):
                try c.encode(SourceTag.relativeFiles.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
            case let .absoluteFiles(model, mmproj):
                try c.encode(SourceTag.absoluteFiles.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
            case let .url(model, modelSha, mmproj, mmprojSha):
                try c.encode(SourceTag.url.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
            case let .vaultUrl(model, modelSha, mmproj, mmprojSha, principalId, customLabel):
                try c.encode(SourceTag.vaultUrl.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
                try c.encode(principalId, forKey: .principalId)
                try c.encodeIfPresent(customLabel.value, forKey: .customLabel)
            }
        case let .ggufVisionSpeculative(m):
            try c.encode(TypeTag.ggufVisionSpeculative.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha, draft, draftSha):
                try encodeRepo(repo)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
                try c.encodeIfPresent(draft, forKey: .draft)
                try c.encodeIfPresent(draftSha, forKey: .draftSha256)
            case let .relativeFiles(model, mmproj, draft):
                try c.encode(SourceTag.relativeFiles.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(draft, forKey: .draft)
            case let .absoluteFiles(model, mmproj, draft):
                try c.encode(SourceTag.absoluteFiles.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(draft, forKey: .draft)
            case let .url(model, modelSha, mmproj, mmprojSha, draft, draftSha):
                try c.encode(SourceTag.url.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
                try c.encodeIfPresent(draft, forKey: .draft)
                try c.encodeIfPresent(draftSha, forKey: .draftSha256)
            case let .vaultUrl(model, modelSha, mmproj, mmprojSha, draft, draftSha, principalId,
                               customLabel):
                try c.encode(SourceTag.vaultUrl.rawValue, forKey: .source)
                try c.encode(model, forKey: .model)
                try c.encode(mmproj, forKey: .mmproj)
                try c.encodeIfPresent(modelSha, forKey: .modelSha256)
                try c.encodeIfPresent(mmprojSha, forKey: .mmprojSha256)
                try c.encodeIfPresent(draft, forKey: .draft)
                try c.encodeIfPresent(draftSha, forKey: .draftSha256)
                try c.encode(principalId, forKey: .principalId)
                try c.encodeIfPresent(customLabel.value, forKey: .customLabel)
            }
        case let .mlx(m):
            try c.encode(TypeTag.mlx.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, prefix):
                try encodeRepo(repo)
                try c.encodeIfPresent(prefix, forKey: .prefix)
            case let .relativeDir(dir):
                try c.encode(SourceTag.relativeDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            case let .absoluteDir(dir):
                try c.encode(SourceTag.absoluteDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            case let .vaultArchive(url, sha256, principalId, customLabel):
                try c.encode(SourceTag.vaultArchive.rawValue, forKey: .source)
                try c.encode(url, forKey: .url)
                try c.encodeIfPresent(sha256, forKey: .sha256)
                try c.encode(principalId, forKey: .principalId)
                try c.encodeIfPresent(customLabel.value, forKey: .customLabel)
            }
        case let .onnxGenai(m):
            try c.encode(TypeTag.onnxGenai.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, prefix):
                try encodeRepo(repo)
                try c.encodeIfPresent(prefix, forKey: .prefix)
            case let .relativeDir(dir):
                try c.encode(SourceTag.relativeDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            case let .absoluteDir(dir):
                try c.encode(SourceTag.absoluteDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            case let .vaultArchive(url, sha256, principalId, customLabel):
                try c.encode(SourceTag.vaultArchive.rawValue, forKey: .source)
                try c.encode(url, forKey: .url)
                try c.encodeIfPresent(sha256, forKey: .sha256)
                try c.encode(principalId, forKey: .principalId)
                try c.encodeIfPresent(customLabel.value, forKey: .customLabel)
            }
        case .appleFoundationText:
            break
        }
    }
}

#if DEBUG
extension Model {
    /// A stock GGUF spec for SwiftUI previews.
    static let previewSample = Model.ggufText(GgufText(source: .huggingFace(
        repo: HfRepo(org: HfOrg(validated: "preview"), repoName: HfRepoName(validated: "Sample-GGUF")),
        path: RepoSubpath(validated: "Sample-Q4_0.gguf"),
        sha256: nil)))
}
#endif

/// A vault source's `custom_label`, the vault record's name: the crate's
/// `Option<NonEmptyTrimmedString>` on each vault arm.
///
/// Display text that names nothing, so any two compare equal and hash alike, like
/// `HfRepo.authToken`. The optional sits inside: `Optional` would tell `nil` from a label.
nonisolated struct CustomLabel: Hashable, Sendable {
    let value: NonEmptyTrimmedString?

    init(_ value: NonEmptyTrimmedString?) { self.value = value }

    static func == (lhs: CustomLabel, rhs: CustomLabel) -> Bool { true }

    func hash(into hasher: inout Hasher) {}
}

/// The crate's `ModelSource`, for directory-style models. `prefix` narrows a repo that
/// bundles several variants to one subdirectory; absent means the repo root.
/// `vaultArchive` is one vault resource holding the whole directory as an archive.
nonisolated enum ModelSource: Hashable, Sendable {
    case huggingFace(repo: HfRepo, prefix: RepoSubpath?)
    /// Wire: `relative_dir`.
    case relativeDir(dir: RelativePath)
    /// Wire: `absolute_dir`.
    case absoluteDir(dir: AbsolutePath)
    /// Wire: `vault_archive`.
    case vaultArchive(url: VaultPath, sha256: Sha256?, principalId: VaultPrincipalId,
                      customLabel: CustomLabel)

    /// The crate's `reference()`.
    var reference: String {
        switch self {
        case let .huggingFace(repo, prefix):
            prefix.map { "\(repo.reference):\($0.value)" } ?? repo.reference
        case let .relativeDir(dir): dir.value
        case let .absoluteDir(dir): dir.value
        case let .vaultArchive(url, _, _, _): url.value
        }
    }

    var authToken: AuthToken? {
        switch self {
        case let .huggingFace(repo, _): repo.authToken
        default: nil
        }
    }

    var withoutAuthToken: ModelSource {
        switch self {
        case let .huggingFace(repo, prefix): .huggingFace(repo: repo.withoutAuthToken, prefix: prefix)
        default: self
        }
    }

    var withoutCustomLabel: ModelSource {
        switch self {
        case let .vaultArchive(url, sha256, principalId, _):
            .vaultArchive(url: url, sha256: sha256, principalId: principalId,
                          customLabel: CustomLabel(nil))
        default: self
        }
    }
}

/// MLX bundle.
nonisolated struct Mlx: Hashable, Sendable, CustomStringConvertible {
    let source: ModelSource

    var description: String { source.reference }
}

/// ONNX Runtime GenAI package. A repo usually bundles several quantizations, so `prefix`
/// selects one.
nonisolated struct OnnxGenai: Hashable, Sendable, CustomStringConvertible {
    let source: ModelSource

    var description: String { source.reference }
}

/// plan-types `ModelType`.
nonisolated enum ModelType: String, Decodable, CaseIterable {
    case ggufText = "gguf_text"
    case ggufVision = "gguf_vision"
    case ggufVisionSpeculative = "gguf_vision_speculative"
    case mlx
    case mlxSpeculative = "mlx_speculative"
    case coreAi = "core_ai"
    case torch
    case torchSpeculative = "torch_speculative"
    case openvino
    case onnxGenai = "onnx_genai"
    case appleFoundationText = "apple_foundation_text"

    /// The crate's `ModelType::of`. No `default:`, so a new `Model` case must be
    /// classified.
    static func of(_ model: Model) -> ModelType {
        switch model {
        case .ggufText: .ggufText
        case .ggufVision: .ggufVision
        case .ggufVisionSpeculative: .ggufVisionSpeculative
        case .mlx: .mlx
        case .onnxGenai: .onnxGenai
        case .appleFoundationText: .appleFoundationText
        }
    }
}
