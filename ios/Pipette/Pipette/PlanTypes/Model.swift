import Foundation

// MARK: - Model flags

/// Per-cell model-generation flags — a closed enum with one variant per
/// `(benchmark, model)` cell that carries flags, mirroring the crate's `ModelFlags`.
/// Generation flags only shape the chat-templated eval path, so every variant is
/// `eval…`; a timing cell, or a model with no generation settings (Apple Foundation,
/// whose weights and template ship in the OS), has no variant.
///
/// These attach to the **cell**, resolved from its `model_flags`, and never to `Model`,
/// which stays identity-only. Authored flat via `ModelFlagRef`.
///
/// `evalTorch` is carried even though no torch engine exists on a phone: the variant set
/// is the crate's, so a claim naming it is refused by the engine check rather than by a
/// gap in this enum.
nonisolated enum ModelFlags: Hashable, Sendable {
    case evalGgufText(enableThinking: Bool?)
    case evalGgufVision(enableThinking: Bool?)
    case evalMlx(enableThinking: Bool?)
    case evalTorch(enableThinking: Bool?)

    /// The model kind this variant covers — the crate's `ModelFlags::model_type`.
    var modelType: ModelType {
        switch self {
        case .evalGgufText: .ggufText
        case .evalGgufVision: .ggufVision
        case .evalMlx: .mlx
        case .evalTorch: .torch
        }
    }

    var enableThinking: Bool? {
        switch self {
        case let .evalGgufText(v), let .evalGgufVision(v), let .evalMlx(v), let .evalTorch(v): v
        }
    }
}

// MARK: - Sources (mirror GgufTextSource / GgufVisionSource / ModelSource)

/// Where a single-file gguf lives — the crate's `GgufTextSource`, all four arms.
///
/// `huggingFace` is what a plan authors and what this client fetches. `relativeFile` is
/// the store form a manifest records (`toStored`); `absoluteFile` is what `bindUnder`
/// hands back. `url` decodes so a body meant for the CLI is *readable* here — running it
/// is a separate question, refused as not-runnable-on-this-client.
nonisolated enum GgufTextSource: Hashable, Sendable {
    case huggingFace(repo: HFRepo, path: RepoSubpath, sha256: Sha256?)
    /// Wire: `relative_file`. Portable, entry-relative.
    case relativeFile(path: RelativePath)
    /// Wire: `absolute_file`. Host path after bind.
    case absoluteFile(path: AbsolutePath)
    case url(url: ResourceUrl, sha256: Sha256?)

    /// `<org>/<repo>[@<revision>]:<path>` for an HF file, else the path or the URL —
    /// the crate's `reference()`.
    var reference: String {
        switch self {
        case let .huggingFace(repo, path, _): "\(repo.reference):\(path.value)"
        case let .relativeFile(path): path.value
        case let .absoluteFile(path): path.value
        case let .url(url, _): url.value
        }
    }

    /// The access token for this source, if any — the crate's `auth_token()`. Only the
    /// HuggingFace arm can carry one.
    var authToken: AuthToken? {
        switch self {
        case let .huggingFace(repo, _, _): repo.authToken
        default: nil
        }
    }

    var withoutAuthToken: GgufTextSource {
        switch self {
        case let .huggingFace(repo, path, sha256):
            .huggingFace(repo: repo.withoutAuthToken, path: path, sha256: sha256)
        default: self
        }
    }
}

/// Where a VL gguf's two files live — the crate's `GgufVisionSource`, all four arms. The
/// two files are always named independently, which is why a store form records both.
nonisolated enum GgufVisionSource: Hashable, Sendable {
    case huggingFace(repo: HFRepo, model: RepoSubpath, modelSha256: Sha256?,
                     mmproj: RepoSubpath, mmprojSha256: Sha256?)
    /// Wire: `relative_files`.
    case relativeFiles(model: RelativePath, mmproj: RelativePath)
    /// Wire: `absolute_files`.
    case absoluteFiles(model: AbsolutePath, mmproj: AbsolutePath)
    case url(model: ResourceUrl, modelSha256: Sha256?, mmproj: ResourceUrl, mmprojSha256: Sha256?)

    /// Identity of the main weights — the crate's `model_reference()`. The projector
    /// shares the repo, and therefore the credential, so the pair needs no second one.
    var modelReference: String {
        switch self {
        case let .huggingFace(repo, model, _, _, _): "\(repo.reference):\(model.value)"
        case let .relativeFiles(model, _): model.value
        case let .absoluteFiles(model, _): model.value
        case let .url(model, _, _, _): model.value
        }
    }

    var authToken: AuthToken? {
        switch self {
        case let .huggingFace(repo, _, _, _, _): repo.authToken
        default: nil
        }
    }

    var withoutAuthToken: GgufVisionSource {
        switch self {
        case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha):
            .huggingFace(repo: repo.withoutAuthToken, model: model, modelSha256: modelSha,
                         mmproj: mmproj, mmprojSha256: mmprojSha)
        default: self
        }
    }
}

/// Where a directory-style model lives — the crate's `ModelSource`, shared by `Mlx`
/// (and, in the crate, `Torch`/`Openvino`). `prefix` narrows a repo that bundles several
/// variants to one subdirectory; absent means the repo root.
nonisolated enum ModelSource: Hashable, Sendable {
    case huggingFace(repo: HFRepo, prefix: RepoSubpath?)
    /// Wire: `relative_dir`.
    case relativeDir(dir: RelativePath)
    /// Wire: `absolute_dir`.
    case absoluteDir(dir: AbsolutePath)

    /// `org/repo[@revision][:prefix]`, else the directory — the crate's `reference()`.
    var reference: String {
        switch self {
        case let .huggingFace(repo, prefix):
            prefix.map { "\(repo.reference):\($0.value)" } ?? repo.reference
        case let .relativeDir(dir): dir.value
        case let .absoluteDir(dir): dir.value
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
}

// MARK: - Per-format model coordinates (mirror GgufText / GgufVision / Mlx)

/// Single-file GGUF text model (llama.cpp). One field, as in the crate: a model is its
/// source. Generation flags belong to the cell, not here.
nonisolated struct GgufText: Hashable, Sendable {
    let source: GgufTextSource
}

/// VL GGUF: main weights + projector file.
nonisolated struct GgufVision: Hashable, Sendable {
    let source: GgufVisionSource
}

/// MLX bundle — a directory-style HF deployment.
nonisolated struct Mlx: Hashable, Sendable {
    let source: ModelSource

    /// The `HFModelRef` the download path uses, or `nil` for an arm with no repo to
    /// fetch from. Swift-only: the crate's fetcher takes the source directly, while iOS's
    /// HubApi wrapper wants repo + subpath as one value.
    var ref: HFModelRef? {
        guard case let .huggingFace(repo, prefix) = source else { return nil }
        return HFModelRef(repo: repo, subpath: prefix)
    }
}

// MARK: - Model spec (mirror the crate's `Model` enum)

/// A model deployment coordinate, tagged by artifact format — the on-device subset of the
/// crate's `Model` (no `Torch`/`Openvino`; there is no on-device engine for either). This
/// is the model *spec* (identity); a run carries it beside its bound form — the same
/// `Model` with its source rewritten to an `absolute*` arm — as `DeclaredBound<Model>`.
///
/// Identity-only, as in the crate: no flags, no resolved paths. `Codable` matches the
/// crate's serde shape — a `type` discriminator, a `source` sub-tag, and the flattened HF
/// coordinate — so a manifest is readable by the same tooling.
nonisolated enum Model: Hashable, Sendable, Codable {
    case ggufText(GgufText)
    case ggufVision(GgufVision)
    case mlx(Mlx)
    /// Apple's on-device system model (FoundationModels). A **bare** case: it carries no
    /// HF coordinate and no on-disk footprint — it ships with the OS — so it has no
    /// associated value. Control flow switches on this case to *skip* the steps that
    /// assume a downloadable model (path resolution, the memory gate, HF download), never
    /// to fabricate a path or repo.
    case appleFoundationText

    /// The access token this model carries for fetching, if any — the crate's
    /// `Model::auth_token`. Only a gated HF source can carry one; AFM never does.
    var authToken: AuthToken? {
        switch self {
        case let .ggufText(m): m.source.authToken
        case let .ggufVision(m): m.source.authToken
        case let .mlx(m): m.source.authToken
        case .appleFoundationText: nil
        }
    }

    /// A copy with every source's auth token cleared — the crate's
    /// `Model::without_auth_token`, and the form persisted anywhere on disk.
    var withoutAuthToken: Model {
        switch self {
        case let .ggufText(m): .ggufText(.init(source: m.source.withoutAuthToken))
        case let .ggufVision(m): .ggufVision(.init(source: m.source.withoutAuthToken))
        case let .mlx(m): .mlx(.init(source: m.source.withoutAuthToken))
        case .appleFoundationText: .appleFoundationText
        }
    }

    /// The HF coordinate, or `nil` for a model with no downloadable coordinate (AFM).
    /// Swift-only: the crate reaches the repo through the source arm it is already
    /// matching on, while iOS has many callers that want only the coordinate.
    var repo: HFRepo? {
        switch self {
        case let .ggufText(m):
            guard case let .huggingFace(repo, _, _) = m.source else { return nil }
            return repo
        case let .ggufVision(m):
            guard case let .huggingFace(repo, _, _, _, _) = m.source else { return nil }
            return repo
        case let .mlx(m):
            guard case let .huggingFace(repo, _) = m.source else { return nil }
            return repo
        case .appleFoundationText:
            return nil
        }
    }

    /// Whether this client can obtain the weights this spec names.
    ///
    /// Swift-only: the crate needs no such predicate because the CLI fetches every arm —
    /// a URL, a host path, a store form. A phone fetches from the Hub only, so a claim
    /// naming any other arm is refused as not-runnable-*here* and stays retriable, which
    /// is what keeps a job the CLI can run from being retired by a phone.
    var isFetchableHere: Bool {
        switch self {
        case let .ggufText(m): if case .huggingFace = m.source { true } else { false }
        case let .ggufVision(m): if case .huggingFace = m.source { true } else { false }
        case let .mlx(m): if case .huggingFace = m.source { true } else { false }
        // Ships with the OS: nothing to fetch, and running it needs no fetch.
        case .appleFoundationText: true
        }
    }

    /// Human-facing engine name (GGUF → llama.cpp, MLX → MLX, AFM → Apple Foundation).
    var engineLabel: String {
        switch self {
        case .ggufText, .ggufVision: "llama.cpp"
        case .mlx: "MLX"
        case .appleFoundationText: "Apple Foundation"
        }
    }

    /// The human identity, `org/repo:path` — the crate's `Display for Model`, and what
    /// `models --format name` renders. Apple Foundation reports the same string the
    /// submitted `model_name` uses, so a plan reference and a warehouse key agree.
    var reference: String {
        switch self {
        case let .ggufText(m): m.source.reference
        case let .ggufVision(m): m.source.modelReference
        case let .mlx(m): m.source.reference
        case .appleFoundationText: AFMRuntime.submissionModelName
        }
    }

    /// The publishing repo as a slug, or `""` for a model addressed by path or URL. AFM
    /// reports its fixed submission slug. Derived here so the discovery row and the job
    /// cell name a model identically; the wire never carries it (the crate's
    /// `BenchmarkSubmissionPayload` states identity as `model_descriptor` alone).
    var repoSlug: String {
        if case .appleFoundationText = self { return AFMRuntime.submissionModelName }
        return repo?.description ?? ""
    }

    /// The artifact's leaf name, as the UI and the headless CLI identify a model: the GGUF
    /// weight file's leaf, or the MLX bundle's directory leaf. Derived from the spec rather
    /// than the on-disk path, because an entry's payload directory is named `blobs` and
    /// names nothing. A `RepoSubpath` may carry directories, so the leaf is taken.
    var artifactName: String {
        switch self {
        case let .ggufText(m): Self.leaf(m.source.reference)
        case let .ggufVision(m): Self.leaf(m.source.modelReference)
        case let .mlx(m): m.ref?.leaf ?? Self.leaf(m.source.reference)
        case .appleFoundationText: "Apple Foundation"
        }
    }

    /// The last path segment of a reference, which is the weights' own name whichever arm
    /// named them — a repo subpath, a store path, or a URL.
    private static func leaf(_ reference: String) -> String {
        reference.split(whereSeparator: { $0 == "/" || $0 == ":" }).last.map(String.init) ?? reference
    }

    /// Quant label: the GGUF K-quant token parsed from the weight file's leaf, or the MLX
    /// bit-width slug segment. The one place either is derived. AFM has no quantization we
    /// can name (a system model, not a file we quantized) → nil.
    var quant: String? {
        switch self {
        case let .ggufText(m): return LocalStorage.parseQuant(from: Self.leaf(m.source.reference))
        case let .ggufVision(m): return LocalStorage.parseQuant(from: Self.leaf(m.source.modelReference))
        case let .mlx(m):
            // Any `<n>bit` leaf segment, not an enumerated set: `familyId` strips this to
            // rejoin an MLX row with its GGUF sibling, so an unrecognized width would
            // silently split the family rather than fail loudly.
            guard case let .huggingFace(repo, _) = m.source,
                  let seg = repo.repoName.value.split(separator: "-").last.map(String.init),
                  seg.hasSuffix("bit") else { return nil }
            let width = seg.dropLast("bit".count)
            return !width.isEmpty && width.allSatisfy(\.isNumber) ? seg : nil
        case .appleFoundationText:
            return nil
        }
    }

    /// Canonical family stem, shared across a model's quants *and* formats: the GGUF weight
    /// file's normalized stem, or the MLX repo leaf with the bit-width quant and `-MLX`
    /// infix stripped — so GGUF and MLX of one model resolve to the same id.
    var familyId: String {
        switch self {
        case let .ggufText(m): return LocalStorage.normalizedModelStem(Self.leaf(m.source.reference))
        case let .ggufVision(m): return LocalStorage.normalizedModelStem(Self.leaf(m.source.modelReference))
        case let .mlx(m):
            // A store or absolute arm has no repo name to strip; its directory leaf is
            // already the family stem.
            guard case let .huggingFace(repo, _) = m.source else {
                return Self.leaf(m.source.reference).lowercased()
            }
            var slug = repo.repoName.value
            if let quant, slug.lowercased().hasSuffix("-" + quant) {
                slug = String(slug.dropLast(quant.count + 1))
            }
            if slug.lowercased().hasSuffix("-mlx") {
                slug = String(slug.dropLast("-mlx".count))
            }
            return slug.lowercased()
        case .appleFoundationText:
            return "apple-foundation"
        }
    }
}

// `Codable` is declared on the (nonisolated) type above so the conformance stays
// nonisolated; stating it on this extension would infer `@MainActor` isolation under the
// module's default-actor-isolation setting and make decode unusable off-main.
extension Model {
    /// The crate's key set. Two discriminators: `type` picks the artifact format and
    /// `source` picks where it lives, with the repo coordinate flattened alongside —
    /// `crates/pipette-plan-types/src/model.rs`.
    private enum CodingKeys: String, CodingKey {
        case type, source, org
        case repoName = "repo_name"
        case revision
        case authToken = "auth_token"
        case path, model, mmproj, prefix, dir, url
        case sha256
        case modelSha256 = "model_sha256"
        case mmprojSha256 = "mmproj_sha256"
    }

    /// `Model`'s `type` tag. `torch` and `openvino` decode and are then refused — there is
    /// no on-device engine for either, and accepting one would strand a run at load time
    /// rather than at parse time.
    private enum TypeTag: String {
        case ggufText = "gguf_text", ggufVision = "gguf_vision", mlx, torch, openvino
        case appleFoundationText = "apple_foundation_text"
    }

    /// The nested `source` tag — every arm the crate has. A phone only *fetches*
    /// `huggingface`; the rest are read so a store form round-trips and a body authored
    /// for the CLI is readable, with runnability decided separately.
    private enum SourceTag: String {
        case huggingFace = "huggingface"
        case relativeFile = "relative_file", absoluteFile = "absolute_file"
        case relativeFiles = "relative_files", absoluteFiles = "absolute_files"
        case relativeDir = "relative_dir", absoluteDir = "absolute_dir"
        case url
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .type)
        guard let tag = TypeTag(rawValue: raw) else { throw ModelError.unknownModelType(raw) }
        // AFM is the bare marker: it ships with the OS, so it names no source and no
        // coordinate.
        if tag == .appleFoundationText {
            self = .appleFoundationText
            return
        }
        let rawSource = try c.decode(String.self, forKey: .source)
        guard let sourceTag = SourceTag(rawValue: rawSource) else {
            throw ModelError.unknownModelType("\(raw)/\(rawSource)")
        }
        // Only the HuggingFace arms carry a repo coordinate; the rest name a path or URL.
        func hfRepo() throws -> HFRepo {
            HFRepo(
                org: try c.decode(HFOrg.self, forKey: .org),
                repoName: try c.decode(HFRepoName.self, forKey: .repoName),
                revision: try c.decodeIfPresent(HFRevision.self, forKey: .revision),
                authToken: try c.decodeIfPresent(AuthToken.self, forKey: .authToken))
        }
        func mismatch() -> ModelError { .unknownModelType("\(raw)/\(rawSource)") }

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
            default: throw mismatch()
            }
        case .torch, .openvino, .appleFoundationText:
            throw ModelError.unknownModelType(raw)
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // AFM is the bare marker: no source, no coordinate.
        if case .appleFoundationText = self {
            try c.encode(TypeTag.appleFoundationText.rawValue, forKey: .type)
            return
        }
        func encodeRepo(_ repo: HFRepo) throws {
            try c.encode(SourceTag.huggingFace.rawValue, forKey: .source)
            try c.encode(repo.org, forKey: .org)
            try c.encode(repo.repoName, forKey: .repoName)
            // Absent rather than null, matching the crate's `skip_serializing_if`.
            try c.encodeIfPresent(repo.revision, forKey: .revision)
            // The credential is never written down — it lives in the Keychain. Decode
            // still accepts one: that is how a claim delivers it.
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
            }
        case let .mlx(m):
            try c.encode(TypeTag.mlx.rawValue, forKey: .type)
            switch m.source {
            case let .huggingFace(repo, prefix):
                try encodeRepo(repo)
                // `prefix` narrows a multi-model repo to one directory; omitted for a
                // root-repo model, matching `skip_serializing_if = "Option::is_none"`.
                try c.encodeIfPresent(prefix, forKey: .prefix)
            case let .relativeDir(dir):
                try c.encode(SourceTag.relativeDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            case let .absoluteDir(dir):
                try c.encode(SourceTag.absoluteDir.rawValue, forKey: .source)
                try c.encode(dir, forKey: .dir)
            }
        case .appleFoundationText:
            break  // handled above (early return)
        }
    }
}

#if DEBUG
extension Model {
    /// A stock GGUF spec for SwiftUI previews, where a cell needs a `source` but the exact
    /// coordinate is irrelevant to what's being previewed.
    static let previewSample = Model.ggufText(GgufText(source: .huggingFace(
        repo: HFRepo(org: HFOrg(validated: "preview"), repoName: HFRepoName(validated: "Sample-GGUF")),
        path: RepoSubpath(validated: "Sample-Q4_0.gguf"),
        sha256: nil)))
}
#endif

/// plan-types `ModelType`.
nonisolated enum ModelType: String, Decodable {
    case ggufText = "gguf_text"
    case ggufVision = "gguf_vision"
    case mlx
    case torch
    case openvino
    case appleFoundationText = "apple_foundation_text"

    /// The kind of a concrete `Model` — the crate's `ModelType::of`. Exhaustive, so a new
    /// `Model` case fails to compile until it is classified rather than silently reading as
    /// some other kind.
    static func of(_ model: Model) -> ModelType {
        switch model {
        case .ggufText: .ggufText
        case .ggufVision: .ggufVision
        case .mlx: .mlx
        case .appleFoundationText: .appleFoundationText
        }
    }
}
