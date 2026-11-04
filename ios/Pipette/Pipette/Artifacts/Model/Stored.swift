import Foundation

// The store form of a `Model` and the way back, after
// `crates/pipette-artifacts/src/model/stored.rs`. The relative arms of `Model` are the
// stored form, so no separate type has to be kept in sync.

extension VaultPath {
    /// The crate's `vault_filename`. Every vault route ends in `content`, which would
    /// collide across a vision pair.
    nonisolated var storedFilename: String { "\(resourceId.value).gguf" }
}

extension Model {
    /// The crate's `to_stored`. Each path keeps its repo-relative nesting under `base`;
    /// two files with one path are `nil`, as the crate refuses them.
    nonisolated func toStored(base: String) -> Model? {
        func under(_ path: String) -> RelativePath? { try? RelativePath("\(base)/\(path)") }
        switch self {
        case let .ggufText(m):
            let leaf: String
            switch m.source {
            case let .huggingFace(_, path, _): leaf = path.value
            case let .vaultUrl(url, _, _, _): leaf = url.storedFilename
            case .relativeFile, .absoluteFile, .url: return nil
            }
            guard let stored = under(leaf) else { return nil }
            return .ggufText(.init(source: .relativeFile(path: stored)))
        case let .ggufVision(m):
            let leaves: (model: String, mmproj: String)
            switch m.source {
            case let .huggingFace(_, model, _, mmproj, _): leaves = (model.value, mmproj.value)
            case let .vaultUrl(model, _, mmproj, _, _, _):
                leaves = (model.storedFilename, mmproj.storedFilename)
            case .relativeFiles, .absoluteFiles, .url: return nil
            }
            guard leaves.model != leaves.mmproj,
                  let storedModel = under(leaves.model), let storedMmproj = under(leaves.mmproj)
            else { return nil }
            return .ggufVision(.init(source: .relativeFiles(model: storedModel, mmproj: storedMmproj)))
        case let .ggufVisionSpeculative(m):
            // Through `GgufVision`'s own arm, so the two types cannot drift.
            let draftLeaf: String?
            switch m.source {
            case let .huggingFace(_, _, _, _, _, draft, _): draftLeaf = draft?.value
            case let .vaultUrl(_, _, _, _, draft, _, _, _): draftLeaf = draft?.storedFilename
            case .relativeFiles, .absoluteFiles, .url: return nil
            }
            guard case let .ggufVision(vision)? =
                    Model.ggufVision(.init(source: m.source.vision)).toStored(base: base),
                  case let .relativeFiles(storedModel, storedMmproj) = vision.source
            else { return nil }
            let storedDraft: RelativePath?
            if let draftLeaf {
                guard let moved = under(draftLeaf),
                      moved != storedModel, moved != storedMmproj else { return nil }
                storedDraft = moved
            } else {
                storedDraft = nil
            }
            return .ggufVisionSpeculative(.init(source: .relativeFiles(
                model: storedModel, mmproj: storedMmproj, draft: storedDraft)))
        case let .mlx(m):
            // The crate's `dir_to_stored`: a repo subtree keeps its `prefix` under `base`;
            // anything else is `base` itself.
            let dir: RelativePath?
            if case let .huggingFace(_, prefix?) = m.source {
                dir = under(prefix.value)
            } else {
                dir = try? RelativePath(base)
            }
            guard let dir else { return nil }
            return .mlx(.init(source: .relativeDir(dir: dir)))
        case let .onnxGenai(m):
            let dir: RelativePath?
            if case let .huggingFace(_, prefix?) = m.source {
                dir = under(prefix.value)
            } else {
                dir = try? RelativePath(base)
            }
            guard let dir else { return nil }
            return .onnxGenai(.init(source: .relativeDir(dir: dir)))
        case .appleFoundationText:
            return nil
        }
    }

    /// The crate's `under_root`.
    nonisolated func underRoot(_ root: URL) -> Model? {
        func joined(_ path: RelativePath) -> AbsolutePath? {
            try? AbsolutePath(root.appendingPathComponent(path.value).path)
        }
        switch self {
        case let .ggufText(m):
            guard case let .relativeFile(path) = m.source, let bound = joined(path) else { return nil }
            return .ggufText(.init(source: .absoluteFile(path: bound)))
        case let .ggufVision(m):
            guard case let .relativeFiles(model, mmproj) = m.source,
                  let boundModel = joined(model), let boundMmproj = joined(mmproj) else { return nil }
            return .ggufVision(.init(source: .absoluteFiles(model: boundModel, mmproj: boundMmproj)))
        case let .ggufVisionSpeculative(m):
            guard case let .relativeFiles(model, mmproj, draft) = m.source,
                  let boundModel = joined(model), let boundMmproj = joined(mmproj)
            else { return nil }
            // A named drafter that will not join fails rather than binding as merged,
            // which would run a different measurement.
            let boundDraft: AbsolutePath?
            if let draft {
                guard let bound = joined(draft) else { return nil }
                boundDraft = bound
            } else {
                boundDraft = nil
            }
            return .ggufVisionSpeculative(.init(source: .absoluteFiles(
                model: boundModel, mmproj: boundMmproj, draft: boundDraft)))
        case let .mlx(m):
            guard case let .relativeDir(dir) = m.source, let bound = joined(dir) else { return nil }
            return .mlx(.init(source: .absoluteDir(dir: bound)))
        case let .onnxGenai(m):
            guard case let .relativeDir(dir) = m.source, let bound = joined(dir) else { return nil }
            return .onnxGenai(.init(source: .absoluteDir(dir: bound)))
        case .appleFoundationText:
            return nil
        }
    }

    nonisolated var storedRelativePaths: [String] {
        switch self {
        case let .ggufText(m):
            guard case let .relativeFile(path) = m.source else { return [] }
            return [path.value]
        case let .ggufVision(m):
            guard case let .relativeFiles(model, mmproj) = m.source else { return [] }
            return [model.value, mmproj.value]
        case let .ggufVisionSpeculative(m):
            guard case let .relativeFiles(model, mmproj, draft) = m.source else { return [] }
            return [model.value, mmproj.value] + (draft.map { [$0.value] } ?? [])
        case let .mlx(m):
            guard case let .relativeDir(dir) = m.source else { return [] }
            return [dir.value]
        case let .onnxGenai(m):
            guard case let .relativeDir(dir) = m.source else { return [] }
            return [dir.value]
        case .appleFoundationText:
            return []
        }
    }

    /// For re-rooting entry-relative manifests (see
    /// `ModelArtifactStore.upgradedFromEntryRelative`): moves the recorded paths rather
    /// than re-deriving them from `declared`.
    nonisolated func prefixingStoredPaths(with prefix: String) -> Model? {
        func moved(_ path: String) -> RelativePath? { try? RelativePath("\(prefix)/\(path)") }
        switch self {
        case let .ggufText(m):
            guard case let .relativeFile(path) = m.source,
                  let moved = moved(path.value) else { return nil }
            return .ggufText(.init(source: .relativeFile(path: moved)))
        case let .ggufVision(m):
            guard case let .relativeFiles(model, mmproj) = m.source,
                  let movedModel = moved(model.value),
                  let movedMmproj = moved(mmproj.value) else { return nil }
            return .ggufVision(.init(source: .relativeFiles(model: movedModel,
                                                            mmproj: movedMmproj)))
        case let .ggufVisionSpeculative(m):
            guard case let .relativeFiles(model, mmproj, draft) = m.source,
                  let movedModel = moved(model.value),
                  let movedMmproj = moved(mmproj.value) else { return nil }
            let movedDraft: RelativePath?
            if let draft {
                guard let value = moved(draft.value) else { return nil }
                movedDraft = value
            } else {
                movedDraft = nil
            }
            return .ggufVisionSpeculative(.init(source: .relativeFiles(
                model: movedModel, mmproj: movedMmproj, draft: movedDraft)))
        case let .mlx(m):
            guard case let .relativeDir(dir) = m.source,
                  let moved = moved(dir.value) else { return nil }
            return .mlx(.init(source: .relativeDir(dir: moved)))
        case let .onnxGenai(m):
            guard case let .relativeDir(dir) = m.source,
                  let moved = moved(dir.value) else { return nil }
            return .onnxGenai(.init(source: .relativeDir(dir: moved)))
        case .appleFoundationText:
            return nil
        }
    }

    /// `draft` is `nil` both with no drafter and with one merged into the weights; a
    /// caller that must tell them apart asks the source (`draftReference`).
    nonisolated var boundPaths: (payload: String, mmproj: String?, draft: String?)? {
        switch self {
        case let .ggufText(m):
            guard case let .absoluteFile(path) = m.source else { return nil }
            return (path.value, nil, nil)
        case let .ggufVision(m):
            guard case let .absoluteFiles(model, mmproj) = m.source else { return nil }
            return (model.value, mmproj.value, nil)
        case let .ggufVisionSpeculative(m):
            guard case let .absoluteFiles(model, mmproj, draft) = m.source else { return nil }
            return (model.value, mmproj.value, draft?.value)
        case let .mlx(m):
            guard case let .absoluteDir(dir) = m.source else { return nil }
            return (dir.value, nil, nil)
        case let .onnxGenai(m):
            guard case let .absoluteDir(dir) = m.source else { return nil }
            return (dir.value, nil, nil)
        case .appleFoundationText:
            return nil
        }
    }
}
