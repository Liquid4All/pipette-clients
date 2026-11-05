import Foundation

// Client-only derivations from a `Model` for the catalog, listing and fetcher.
// `PlanTypes/Model/` mirrors the crate and stays identity-only.

nonisolated extension Model {
    /// The HF coordinate, or `nil` for AFM or a path, URL or vault arm.
    var repo: HfRepo? {
        switch self {
        case let .ggufText(m):
            guard case let .huggingFace(repo, _, _) = m.source else { return nil }
            return repo
        case let .ggufVision(m):
            guard case let .huggingFace(repo, _, _, _, _) = m.source else { return nil }
            return repo
        case let .ggufVisionSpeculative(m):
            guard case let .huggingFace(repo, _, _, _, _, _, _) = m.source else { return nil }
            return repo
        case let .mlx(m):
            guard case let .huggingFace(repo, _) = m.source else { return nil }
            return repo
        case let .onnxGenai(m):
            guard case let .huggingFace(repo, _) = m.source else { return nil }
            return repo
        case .appleFoundationText:
            return nil
        }
    }

    /// Whether this client can obtain the weights: a Hugging Face arm, or a vault arm from
    /// its registered server. The CLI fetches every arm, so the crate has no such
    /// predicate. `ClientRunSpec.validated` refuses a claim naming any other arm, and
    /// `ensureModel` refuses one as `notFetchableHere`.
    var isFetchableHere: Bool {
        switch self {
        case let .ggufText(m):
            switch m.source {
            case .huggingFace, .vaultUrl: true
            case .relativeFile, .absoluteFile, .url: false
            }
        case let .ggufVision(m):
            switch m.source {
            case .huggingFace, .vaultUrl: true
            case .relativeFiles, .absoluteFiles, .url: false
            }
        case let .ggufVisionSpeculative(m):
            switch m.source {
            case .huggingFace, .vaultUrl: true
            case .relativeFiles, .absoluteFiles, .url: false
            }
        case let .mlx(m):
            switch m.source {
            case .huggingFace, .vaultArchive: true
            case .relativeDir, .absoluteDir: false
            }
        case let .onnxGenai(m):
            switch m.source {
            case .huggingFace, .vaultArchive: true
            case .relativeDir, .absoluteDir: false
            }
        // Ships with the OS.
        case .appleFoundationText: true
        }
    }

    var engineLabel: String { RuntimeKind(self).label }

    /// The repo slug, `""` for a model with no HF coordinate, or AFM's submission slug.
    /// Client-side only (labels, projector pairing, `models rm repo=`): the wire states
    /// identity as `model_descriptor`.
    var repoSlug: String {
        if case .appleFoundationText = self { return AFMRuntime.submissionModelName }
        return repo?.description ?? ""
    }

    /// The leaf name the UI and headless CLI show. Derived from the spec, not the on-disk
    /// path, because an entry's payload directory is named `blobs`.
    var artifactName: String {
        switch self {
        case let .ggufText(m): Self.leaf(m.source.reference)
        case let .ggufVision(m): Self.leaf(m.source.weightsReference)
        // The drafter is part of the identity but never the name to show.
        case let .ggufVisionSpeculative(m): Self.leaf(m.source.vision.weightsReference)
        case let .mlx(m): m.ref?.leaf ?? Self.leaf(m.source.reference)
        case let .onnxGenai(m): m.ref?.leaf ?? Self.leaf(m.source.reference)
        case .appleFoundationText: "Apple Foundation"
        }
    }

    /// The last path segment of a reference. Every vault path ends in `content`, so a vault
    /// reference takes its resource id.
    private static func leaf(_ reference: String) -> String {
        if let vault = try? VaultPath(reference) { return vault.resourceId.value }
        return reference.split(whereSeparator: { $0 == "/" || $0 == ":" }).last.map(String.init) ?? reference
    }

    var quant: String? {
        switch self {
        case let .ggufText(m): return LocalStorage.parseQuant(from: Self.leaf(m.source.reference))
        case let .ggufVision(m): return LocalStorage.parseQuant(from: Self.leaf(m.source.weightsReference))
        case let .ggufVisionSpeculative(m):
            return LocalStorage.parseQuant(from: Self.leaf(m.source.vision.weightsReference))
        case let .mlx(m):
            // Any `<n>bit`, not an enumerated set: `familyId` strips this, so an
            // unrecognized width would silently split the family.
            guard case let .huggingFace(repo, _) = m.source,
                  let seg = repo.repoName.value.split(separator: "-").last.map(String.init),
                  seg.hasSuffix("bit") else { return nil }
            let width = seg.dropLast("bit".count)
            return !width.isEmpty && width.allSatisfy(\.isNumber) ? seg : nil
        case let .onnxGenai(m):
            // The quantization is named in the package prefix
            // (`cpu_and_mobile/cpu-int4-rtn-block-32`), not the repo name.
            guard case let .huggingFace(_, prefix?) = m.source else { return nil }
            return prefix.leaf
        case .appleFoundationText:
            return nil
        }
    }

    /// Family stem shared across a model's quants and formats, so GGUF, MLX and ONNX of
    /// one model resolve to the same id.
    var familyId: String {
        switch self {
        case let .ggufText(m): return LocalStorage.normalizedModelStem(Self.leaf(m.source.reference))
        case let .ggufVision(m): return LocalStorage.normalizedModelStem(Self.leaf(m.source.weightsReference))
        case let .ggufVisionSpeculative(m):
            return LocalStorage.normalizedModelStem(
                Self.leaf(m.source.vision.weightsReference))
        case let .mlx(m):
            // A path arm has no repo name to strip; its leaf is already the stem.
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
        case let .onnxGenai(m):
            // The prefix carries the quantization, so the repo name alone is the family,
            // minus its `-onnx` format tag.
            guard case let .huggingFace(repo, _) = m.source else {
                return Self.leaf(m.source.reference).lowercased()
            }
            var slug = repo.repoName.value.lowercased()
            if slug.hasSuffix("-onnx") { slug = String(slug.dropLast("-onnx".count)) }
            return slug
        case .appleFoundationText:
            return "apple-foundation"
        }
    }
}

nonisolated extension GgufVisionSource {
    /// The weights half of the pair: `<repo-ref>:<path>` for HF, else the path, URL or
    /// vault path. The display name, quant and credential key all come from the weights
    /// alone; the projector shares the repo and is never the name to show.
    var weightsReference: String {
        switch self {
        case let .huggingFace(repo, model, _, _, _): "\(repo.reference):\(model.value)"
        case let .relativeFiles(model, _): model.value
        case let .absoluteFiles(model, _): model.value
        case let .url(model, _, _, _): model.value
        case let .vaultUrl(model, _, _, _, _, _): model.value
        }
    }
}

nonisolated extension Mlx {
    var ref: HfModelRef? {
        guard case let .huggingFace(repo, prefix) = source else { return nil }
        return HfModelRef(repo: repo, subpath: prefix)
    }
}

nonisolated extension OnnxGenai {
    var ref: HfModelRef? {
        guard case let .huggingFace(repo, prefix) = source else { return nil }
        return HfModelRef(repo: repo, subpath: prefix)
    }
}
