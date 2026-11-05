import Foundation

// The crate's `pipette-llamacpp/src/models.rs`: each type accepts one plan model kind, as its
// `TryFrom<&Model>` does.

nonisolated struct ResolvedGgufTextModel: Equatable, Sendable {
    let path: String

    init(_ model: Model) throws {
        guard case let .ggufText(m) = model, case let .absoluteFile(path) = m.source else {
            throw UnexpectedBoundModel(expected: "GgufText AbsoluteFile", got: model.description)
        }
        self.path = try requireFile(path.value, "GGUF text model")
    }
}

nonisolated struct ResolvedGgufVisionModel: Equatable, Sendable {
    let backbone: String
    let mmproj: String

    init(_ model: Model) throws {
        guard case let .ggufVision(m) = model,
              case let .absoluteFiles(path, mmprojPath) = m.source else {
            throw UnexpectedBoundModel(expected: "GgufVision AbsoluteFiles",
                                       got: model.description)
        }
        try self.init(backbone: path.value, mmproj: mmprojPath.value)
    }

    /// The crate's `vision_files`.
    fileprivate init(backbone: String, mmproj: String) throws {
        self.backbone = try requireFile(backbone, "GGUF vision model")
        self.mmproj = try requireFile(mmproj, "GGUF mmproj")
    }
}

nonisolated struct ResolvedGgufVisionSpeculativeModel: Equatable, Sendable {
    let vision: ResolvedGgufVisionModel
    /// `nil` when the drafter is merged into the backbone: DSpark's release ships one gguf
    /// holding both.
    let draft: String?

    init(_ model: Model) throws {
        guard case let .ggufVisionSpeculative(m) = model,
              case let .absoluteFiles(path, mmprojPath, draftPath) = m.source else {
            throw UnexpectedBoundModel(expected: "GgufVisionSpeculative AbsoluteFiles",
                                       got: model.description)
        }
        vision = try ResolvedGgufVisionModel(backbone: path.value, mmproj: mmprojPath.value)
        draft = try draftPath.map { try requireFile($0.value, "GGUF drafter") }
    }
}

private nonisolated func requireFile(_ path: String, _ what: String) throws -> String {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
          !isDirectory.boolValue
    else { throw BoundPathMissing(what: what, path: path) }
    return path
}

/// The crate's `require_file` bail: a half-completed ensure or bind, named before the engine
/// fails to load it.
nonisolated struct BoundPathMissing: Error, Equatable, LocalizedError {
    let what: String
    let path: String

    var errorDescription: String? {
        "bound \(what) is not a file: \(path) (ensure/bind may be incomplete)"
    }
}
