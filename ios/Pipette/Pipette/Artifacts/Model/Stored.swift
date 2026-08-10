import Foundation

// The store form of a `Model` and the way back — the crate's
// `pipette-artifacts/src/model/stored.rs`.
//
// `toStored` records where a fetch landed, relative to the entry; `bindUnder` prefixes
// those paths with the current root. One `Model` serves both halves, as upstream: the
// relative arms *are* the stored form, so nothing has to be kept in sync by hand.

extension Model {
    /// The store-relative form of this model under `base` — the crate's `to_stored`.
    ///
    /// Each path is joined under `base` independently and keeps its repo-relative
    /// nesting, which is what makes a vision model's two files land where they were
    /// named rather than beside each other. AFM has nothing to store.
    nonisolated func toStored(base: String) -> Model? {
        func under(_ path: String) -> RelativePath? { try? RelativePath("\(base)/\(path)") }
        switch self {
        case let .ggufText(m):
            guard case let .huggingFace(_, path, _) = m.source,
                  let stored = under(path.value) else { return nil }
            return .ggufText(.init(source: .relativeFile(path: stored)))
        case let .ggufVision(m):
            guard case let .huggingFace(_, model, _, mmproj, _) = m.source,
                  let storedModel = under(model.value), let storedMmproj = under(mmproj.value)
            else { return nil }
            return .ggufVision(.init(source: .relativeFiles(model: storedModel, mmproj: storedMmproj)))
        case .mlx:
            // The bundle is `base` itself: the installer moves the downloaded directory
            // there whole. Recorded as it lies, so a later change to keep the repo
            // `prefix` — which the crate does — needs no migration: an entry says where
            // its own bytes are.
            guard let stored = try? RelativePath(base) else { return nil }
            return .mlx(.init(source: .relativeDir(dir: stored)))
        case .appleFoundationText:
            return nil
        }
    }

    /// Prefix every relative path with `root` — the crate's `bind_under`. A relative root
    /// keeps the arms relative; an absolute one makes them absolute, which is the form an
    /// engine is handed.
    nonisolated func bindUnder(_ root: URL) -> Model? {
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
        case let .mlx(m):
            guard case let .relativeDir(dir) = m.source, let bound = joined(dir) else { return nil }
            return .mlx(.init(source: .absoluteDir(dir: bound)))
        case .appleFoundationText:
            return nil
        }
    }

    /// The bound weights and, for a vision model, its projector — what a caller actually
    /// wants out of `bindUnder`.
    nonisolated var boundPaths: (payload: String, mmproj: String?)? {
        switch self {
        case let .ggufText(m):
            guard case let .absoluteFile(path) = m.source else { return nil }
            return (path.value, nil)
        case let .ggufVision(m):
            guard case let .absoluteFiles(model, mmproj) = m.source else { return nil }
            return (model.value, mmproj.value)
        case let .mlx(m):
            guard case let .absoluteDir(dir) = m.source else { return nil }
            return (dir.value, nil)
        case .appleFoundationText:
            return nil
        }
    }
}
