import Foundation

// The GGUF models: pipette-mgmt's `crates/pipette-plan-types/src/model/gguf.rs`.
// Each is one source enum and the struct over it; the crate's `GgufVision*Files` structs
// are folded into the source arms' payloads.

/// The crate's `GgufTextSource`.
///
/// This client fetches `huggingFace` and `vaultUrl` (mgmt's form for a gated model it
/// imported). `relativeFile` is the store form a manifest records (`toStored`);
/// `absoluteFile` is what `bindUnder` hands back. `url` decodes so a body meant for the CLI
/// is readable here, but it is refused as not runnable on this client.
nonisolated enum GgufTextSource: Hashable, Sendable {
    case huggingFace(repo: HfRepo, path: RepoSubpath, sha256: Sha256?)
    /// Wire: `relative_file`. Portable, entry-relative.
    case relativeFile(path: RelativePath)
    /// Wire: `absolute_file`. Host path after bind.
    case absoluteFile(path: AbsolutePath)
    case url(url: ResourceUrl, sha256: Sha256?)
    /// Wire: `vault_url`. Fetched from the registered mgmt with a signed request.
    case vaultUrl(url: VaultPath, sha256: Sha256?, principalId: VaultPrincipalId,
                  customLabel: CustomLabel)

    /// The crate's `reference()`.
    var reference: String {
        switch self {
        case let .huggingFace(repo, path, _): "\(repo.reference):\(path.value)"
        case let .relativeFile(path): path.value
        case let .absoluteFile(path): path.value
        case let .url(url, _): url.value
        case let .vaultUrl(url, _, _, _): url.value
        }
    }

    /// The crate's `auth_token()`.
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

    var withoutCustomLabel: GgufTextSource {
        switch self {
        case let .vaultUrl(url, sha256, principalId, _):
            .vaultUrl(url: url, sha256: sha256, principalId: principalId,
                      customLabel: CustomLabel(nil))
        default: self
        }
    }
}

/// Single-file GGUF text model (llama.cpp). Generation flags belong to the cell.
nonisolated struct GgufText: Hashable, Sendable, CustomStringConvertible {
    let source: GgufTextSource

    var description: String { source.reference }
}

/// The crate's `GgufVisionSource`: the weights and the projector, named independently.
nonisolated enum GgufVisionSource: Hashable, Sendable {
    case huggingFace(repo: HfRepo, model: RepoSubpath, modelSha256: Sha256?,
                     mmproj: RepoSubpath, mmprojSha256: Sha256?)
    /// Wire: `relative_files`.
    case relativeFiles(model: RelativePath, mmproj: RelativePath)
    /// Wire: `absolute_files`.
    case absoluteFiles(model: AbsolutePath, mmproj: AbsolutePath)
    case url(model: ResourceUrl, modelSha256: Sha256?, mmproj: ResourceUrl, mmprojSha256: Sha256?)
    /// Wire: `vault_url`. Both files are vault resources served to one principal.
    case vaultUrl(model: VaultPath, modelSha256: Sha256?, mmproj: VaultPath,
                  mmprojSha256: Sha256?, principalId: VaultPrincipalId,
                  customLabel: CustomLabel)

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

    var withoutCustomLabel: GgufVisionSource {
        switch self {
        case let .vaultUrl(model, modelSha, mmproj, mmprojSha, principalId, _):
            .vaultUrl(model: model, modelSha256: modelSha, mmproj: mmproj,
                      mmprojSha256: mmprojSha, principalId: principalId,
                      customLabel: CustomLabel(nil))
        default: self
        }
    }
}

/// VL GGUF: main weights + projector file.
nonisolated struct GgufVision: Hashable, Sendable, CustomStringConvertible {
    let source: GgufVisionSource

    /// `{weights}+{projector}`: the crate's `Display`.
    var description: String {
        switch source {
        case let .huggingFace(repo, model, _, mmproj, _):
            "\(repo.reference):\(model.value)+\(repo.reference):\(mmproj.value)"
        case let .relativeFiles(model, mmproj): "\(model.value)+\(mmproj.value)"
        case let .absoluteFiles(model, mmproj): "\(model.value)+\(mmproj.value)"
        case let .url(model, _, mmproj, _): "\(model.value)+\(mmproj.value)"
        case let .vaultUrl(model, _, mmproj, _, _, _): "\(model.value)+\(mmproj.value)"
        }
    }
}

/// The crate's `GgufVisionSpeculativeSource`: a `GgufVisionSource`'s files plus the
/// drafter.
///
/// `draft` holds the drafter's weights alone (llama.cpp's `--spec-draft-model`). Absent
/// means the drafter is merged into `model`, as DSpark's release ships it; the baseline arm
/// then loads that same file with speculation off.
///
/// Its own arms rather than a wrapped `GgufVisionSource` because the wire is flat (`draft`
/// beside `model` and `mmproj`); `vision` projects the shared half back out.
nonisolated enum GgufVisionSpeculativeSource: Hashable, Sendable {
    case huggingFace(repo: HfRepo, model: RepoSubpath, modelSha256: Sha256?,
                     mmproj: RepoSubpath, mmprojSha256: Sha256?,
                     draft: RepoSubpath?, draftSha256: Sha256?)
    /// Wire: `relative_files`.
    case relativeFiles(model: RelativePath, mmproj: RelativePath, draft: RelativePath?)
    /// Wire: `absolute_files`.
    case absoluteFiles(model: AbsolutePath, mmproj: AbsolutePath, draft: AbsolutePath?)
    case url(model: ResourceUrl, modelSha256: Sha256?, mmproj: ResourceUrl,
             mmprojSha256: Sha256?, draft: ResourceUrl?, draftSha256: Sha256?)
    /// Wire: `vault_url`. The drafter is a third vault resource for the same principal.
    case vaultUrl(model: VaultPath, modelSha256: Sha256?, mmproj: VaultPath,
                  mmprojSha256: Sha256?, draft: VaultPath?, draftSha256: Sha256?,
                  principalId: VaultPrincipalId, customLabel: CustomLabel)

    /// The crate's `vision()`, so identity, storage key and download reuse `GgufVision`'s.
    var vision: GgufVisionSource {
        switch self {
        case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha, _, _):
            .huggingFace(repo: repo, model: model, modelSha256: modelSha,
                         mmproj: mmproj, mmprojSha256: mmprojSha)
        case let .relativeFiles(model, mmproj, _):
            .relativeFiles(model: model, mmproj: mmproj)
        case let .absoluteFiles(model, mmproj, _):
            .absoluteFiles(model: model, mmproj: mmproj)
        case let .url(model, modelSha, mmproj, mmprojSha, _, _):
            .url(model: model, modelSha256: modelSha, mmproj: mmproj, mmprojSha256: mmprojSha)
        case let .vaultUrl(model, modelSha, mmproj, mmprojSha, _, _, principalId, customLabel):
            .vaultUrl(model: model, modelSha256: modelSha, mmproj: mmproj,
                      mmprojSha256: mmprojSha, principalId: principalId,
                      customLabel: customLabel)
        }
    }

    /// The crate's `draft_reference()`; `nil` when the drafter is merged.
    var draftReference: String? {
        switch self {
        case let .huggingFace(repo, _, _, _, _, draft, _):
            draft.map { "\(repo.reference):\($0.value)" }
        case let .relativeFiles(_, _, draft): draft?.value
        case let .absoluteFiles(_, _, draft): draft?.value
        case let .url(_, _, _, _, draft, _): draft?.value
        case let .vaultUrl(_, _, _, _, draft, _, _, _): draft?.value
        }
    }

    var authToken: AuthToken? {
        switch self {
        case let .huggingFace(repo, _, _, _, _, _, _): repo.authToken
        default: nil
        }
    }

    var withoutAuthToken: GgufVisionSpeculativeSource {
        switch self {
        case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha, draft, draftSha):
            .huggingFace(repo: repo.withoutAuthToken, model: model, modelSha256: modelSha,
                         mmproj: mmproj, mmprojSha256: mmprojSha,
                         draft: draft, draftSha256: draftSha)
        default: self
        }
    }

    var withoutCustomLabel: GgufVisionSpeculativeSource {
        switch self {
        case let .vaultUrl(model, modelSha, mmproj, mmprojSha, draft, draftSha, principalId, _):
            .vaultUrl(model: model, modelSha256: modelSha, mmproj: mmproj,
                      mmprojSha256: mmprojSha, draft: draft, draftSha256: draftSha,
                      principalId: principalId, customLabel: CustomLabel(nil))
        default: self
        }
    }
}

/// Speculative VL GGUF: target weights, projector and drafter (merged or separate).
///
/// A model type of its own rather than a flag on `GgufVision`, as in the crate: the drafter
/// is part of what was measured, so rows that differ by it must not share a model identity.
nonisolated struct GgufVisionSpeculative: Hashable, Sendable, CustomStringConvertible {
    let source: GgufVisionSpeculativeSource

    /// The vision pair's identity, then `+{drafter}` for a separate drafter. A merged one
    /// adds nothing: the model type tells such a row from a plain `gguf_vision` one.
    var description: String {
        let vision = GgufVision(source: source.vision).description
        return source.draftReference.map { "\(vision)+\($0)" } ?? vision
    }
}
