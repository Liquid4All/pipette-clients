import Foundation

// What a directory-shaped engine checks before loading a bound model: the crate's
// `pipette-ops/src/models.rs`.

/// Bound model directory for a model of kind `expected`, verified to exist and to contain
/// every one of `requiredFiles`: the crate's `require_bound_model_dir`. Name only files the
/// loader always needs: the point is to catch a truncated snapshot, not to second-guess it.
nonisolated func requireBoundModelDir(_ bound: Model, expected: ModelType,
                                      requiredFiles: [String]) throws -> String {
    let actual = ModelType.of(bound)
    guard actual == expected else {
        throw BoundModelDirError.wrongKind(expected: expected, got: actual)
    }
    // Exhaustive, so a new directory-shaped `Model` has to answer here rather than falling
    // into the not-a-directory arm.
    let source: ModelSource
    switch bound {
    case let .mlx(m):
        source = m.source
    case let .onnxGenai(m):
        source = m.source
    case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText:
        throw BoundModelDirError.notDirectoryShaped(expected)
    }
    guard case let .absoluteDir(dir) = source else {
        throw BoundModelDirError.unbound(expected: expected, got: bound.description)
    }

    let path = dir.value
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
          isDirectory.boolValue
    else { throw BoundModelDirError.notADirectory(expected: expected, path: path) }
    let root = URL(fileURLWithPath: path, isDirectory: true)
    if let missing = requiredFiles.first(where: { !isFile(root.appendingPathComponent($0).path) }) {
        throw BoundModelDirError.missingFile(expected: expected, dir: path, file: missing)
    }
    return path
}

private nonisolated func isFile(_ path: String) -> Bool {
    var isDirectory: ObjCBool = false
    return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        && !isDirectory.boolValue
}

/// Why a bound directory is not one an engine can load: the crate's
/// `require_bound_model_dir` bails, with their messages.
nonisolated enum BoundModelDirError: Error, Equatable, LocalizedError {
    /// Another engine's model reached this one: a dispatch bug, named by both kinds so it
    /// does not read as a missing file.
    case wrongKind(expected: ModelType, got: ModelType)
    case notDirectoryShaped(ModelType)
    /// The model was never bound to a host directory: ensure or bind did not run.
    case unbound(expected: ModelType, got: String)
    case notADirectory(expected: ModelType, path: String)
    case missingFile(expected: ModelType, dir: String, file: String)

    var errorDescription: String? {
        switch self {
        case let .wrongKind(expected, got):
            "expected a bound \(expected.rawValue) model, got \(got.rawValue)"
        case let .notDirectoryShaped(expected):
            "\(expected.rawValue) is not a directory-shaped model"
        case let .unbound(expected, got):
            "expected bound \(expected.rawValue) AbsoluteDir (after ensureModel/bindUnder), got \(got)"
        case let .notADirectory(expected, path):
            "bound \(expected.rawValue) model is not a directory: \(path) (ensure/bind may be incomplete)"
        case let .missingFile(expected, dir, file):
            "\(expected.rawValue) model at \(dir) is missing \(file)"
        }
    }
}

/// A file-shaped engine was handed a model of another format: the crate's
/// `ResolvedGgufTextModel::try_from` / `ResolvedGgufVisionModel::try_from` bail, shared
/// with Apple Foundation's pairing check. A dispatch bug, so it names what it got.
nonisolated struct UnexpectedBoundModel: Error, Equatable, LocalizedError {
    let expected: String
    let got: String

    var errorDescription: String? {
        "expected bound \(expected) (after ensureModel/bindUnder), got \(got)"
    }
}
