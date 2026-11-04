import Foundation

// The crate's `pipette-onnx-genai/src/models.rs`.

/// A directory whose `genai_config.json` names a graph on disk, external data included.
///
/// Checked here rather than left to `OGAModel`: every `LiquidAI/*-ONNX` repo is a
/// transformers.js export this engine cannot load, and named here it says which kind of ONNX
/// the directory holds.
///
/// No `check_provider`: this client's runtime has the CPU provider alone.
nonisolated struct ResolvedOnnxGenaiModel: Equatable, Sendable {
    let dir: String

    init(_ model: Model) throws {
        dir = try requireBoundModelDir(model, expected: .onnxGenai,
                                       requiredFiles: OnnxGenaiModels.requiredFiles)
        try OnnxGenaiModels.validatePackage(at: dir)
    }
}

nonisolated enum OnnxGenaiModels {
    /// The crate's `REQUIRED_FILES`. Everything else a package needs is named inside it,
    /// and `validatePackage` checks those.
    static let requiredFiles = ["genai_config.json"]

    /// A graph missing its external-data sidecar loads, then fails at the first inference
    /// as if the engine were at fault. The sidecar is the multi-GB half a truncated transfer
    /// loses.
    static func validatePackage(at dir: String) throws {
        let root = URL(fileURLWithPath: dir, isDirectory: true)
        let configURL = root.appendingPathComponent("genai_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw RuntimeError.engine(
                "`\(dir)` has a genai_config.json that does not parse as JSON")
        }
        guard let model = object["model"] as? [String: Any],
              let decoder = model["decoder"] as? [String: Any],
              let filename = decoder["filename"] as? String
        else {
            throw RuntimeError.engine(
                "`\(dir)`'s genai_config.json names no `model.decoder.filename`, so there "
                    + "is no graph to load")
        }
        let graph = try packageFile(
            filename, relativeTo: root, packageRoot: root, describedAs: "decoder graph")
        guard FileManager.default.fileExists(atPath: graph.path) else {
            throw RuntimeError.engine(
                "`\(dir)` is missing the decoder graph `\(filename)` its genai_config.json "
                    + "names")
        }
        try validateExternalData(of: graph, under: root)
        try validateTokenizer(model: model, under: root)
    }

    /// Resolves an untrusted metadata path without letting it leave the package. A
    /// normalized spelling is required, so `graphs/../model.onnx` is refused rather than
    /// reinterpreted. The second check follows symlinks, since a package-supplied symlink
    /// can escape without any `..`.
    private static func packageFile(
        _ path: String, relativeTo base: URL, packageRoot root: URL, describedAs kind: String
    ) throws -> URL {
        guard RelativePath.validate(path), !path.contains("\\") else {
            throw RuntimeError.engine(
                "`\(root.path)`'s \(kind) path `\(path)` is not a normalized relative "
                    + "path inside the package")
        }

        let packageRoot = root.standardizedFileURL
        let candidate = base.appendingPathComponent(path).standardizedFileURL
        guard contains(candidate, under: packageRoot) else {
            throw RuntimeError.engine(
                "`\(root.path)`'s \(kind) path `\(path)` escapes the package")
        }

        let resolvedRoot = packageRoot.resolvingSymlinksInPath()
        let resolvedCandidate = candidate.resolvingSymlinksInPath()
        guard contains(resolvedCandidate, under: resolvedRoot) else {
            throw RuntimeError.engine(
                "`\(root.path)`'s \(kind) path `\(path)` escapes the package through a "
                    + "symbolic link")
        }
        return candidate
    }

    private static func contains(_ candidate: URL, under root: URL) -> Bool {
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path.hasPrefix(prefix)
    }

    /// The assets ORT Extensions requires in `model.tokenizer_dir`, checked so a completed
    /// download can't become an installed model that fails in `OGATokenizer`.
    private static func validateTokenizer(model: [String: Any], under root: URL) throws {
        let tokenizerDir: URL
        if let raw = model["tokenizer_dir"] {
            guard let path = raw as? String else {
                throw RuntimeError.engine(
                    "`\(root.path)`'s genai_config.json has a non-string "
                        + "`model.tokenizer_dir`")
            }
            tokenizerDir = path.isEmpty
                ? root
                : try packageFile(path, relativeTo: root, packageRoot: root,
                                  describedAs: "tokenizer directory")
        } else {
            tokenizerDir = root
        }

        let config = try packageFile(
            "tokenizer_config.json", relativeTo: tokenizerDir, packageRoot: root,
            describedAs: "tokenizer config")
        guard let data = try? Data(contentsOf: config),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw RuntimeError.engine(
                "`\(root.path)` is missing a valid tokenizer_config.json in its tokenizer "
                    + "directory")
        }

        let vocabulary: URL
        if let raw = object["tiktoken_file"] {
            guard let path = raw as? String, !path.isEmpty else {
                throw RuntimeError.engine(
                    "`\(root.path)`'s tokenizer_config.json has an invalid `tiktoken_file`")
            }
            vocabulary = try packageFile(
                path, relativeTo: tokenizerDir, packageRoot: root,
                describedAs: "tokenizer vocabulary")
        } else {
            vocabulary = try packageFile(
                "tokenizer.json", relativeTo: tokenizerDir, packageRoot: root,
                describedAs: "tokenizer vocabulary")
        }
        guard FileManager.default.fileExists(atPath: vocabulary.path) else {
            throw RuntimeError.engine(
                "`\(root.path)` is missing tokenizer vocabulary `\(vocabulary.lastPathComponent)`")
        }
    }

    /// Walks the memory-mapped protobuf in place; tensor bytes are skipped by range, never
    /// copied.
    private static func validateExternalData(of graph: URL, under root: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: graph, options: .alwaysMapped)
        } catch {
            throw RuntimeError.engine("`\(graph.path)` could not be read: \(error)")
        }
        let names: [String]
        do {
            names = try externalDataLocations(in: data)
        } catch {
            throw RuntimeError.engine("`\(graph.path)` does not parse as an ONNX graph")
        }
        for name in names {
            let sidecar = try packageFile(
                name, relativeTo: graph.deletingLastPathComponent(), packageRoot: root,
                describedAs: "external-data")
            guard FileManager.default.fileExists(atPath: sidecar.path) else {
                throw RuntimeError.engine(
                    "`\(root.path)` is missing `\(name)`, the external-data file its "
                        + "decoder graph references; the download is incomplete")
            }
        }
    }

    private enum ProtobufValue {
        case varint(UInt64)
        case bytes(Range<Int>)
    }

    private struct InvalidProtobuf: Error {}

    /// ONNX field path: ModelProto.graph(7) → GraphProto.initializer(5) →
    /// TensorProto.external_data(13) → StringStringEntryProto key/value(1/2).
    /// Node attributes and sparse initializers can contain tensors too, so those paths are
    /// traversed as well. Unknown fields are skipped according to their wire type.
    private static func externalDataLocations(in data: Data) throws -> [String] {
        var found: Set<String> = []
        var sawGraph = false
        try data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            try fields(in: bytes, range: 0 ..< bytes.count) { number, value in
                if number == 7, case let .bytes(graph) = value {
                    sawGraph = true
                    try scanGraph(graph, in: bytes, depth: 0, found: &found)
                }
            }
        }
        guard sawGraph else { throw InvalidProtobuf() }
        return found.sorted()
    }

    private static func scanGraph(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer, depth: Int,
        found: inout Set<String>
    ) throws {
        guard depth < 64 else { throw InvalidProtobuf() }
        try fields(in: bytes, range: range) { number, value in
            guard case let .bytes(message) = value else { return }
            switch number {
            case 1: try scanNode(message, in: bytes, depth: depth, found: &found)
            case 5: try scanTensor(message, in: bytes, found: &found)
            case 15: try scanSparseTensor(message, in: bytes, found: &found)
            default: break
            }
        }
    }

    private static func scanNode(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer, depth: Int,
        found: inout Set<String>
    ) throws {
        try fields(in: bytes, range: range) { number, value in
            if number == 5, case let .bytes(attribute) = value {
                try scanAttribute(attribute, in: bytes, depth: depth, found: &found)
            }
        }
    }

    private static func scanAttribute(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer, depth: Int,
        found: inout Set<String>
    ) throws {
        try fields(in: bytes, range: range) { number, value in
            guard case let .bytes(message) = value else { return }
            switch number {
            case 5, 10: try scanTensor(message, in: bytes, found: &found)
            case 6, 11: try scanGraph(message, in: bytes, depth: depth + 1, found: &found)
            case 22, 23: try scanSparseTensor(message, in: bytes, found: &found)
            default: break
            }
        }
    }

    private static func scanSparseTensor(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer,
        found: inout Set<String>
    ) throws {
        try fields(in: bytes, range: range) { number, value in
            if number == 1 || number == 2, case let .bytes(tensor) = value {
                try scanTensor(tensor, in: bytes, found: &found)
            }
        }
    }

    private static func scanTensor(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer,
        found: inout Set<String>
    ) throws {
        var external = false
        var location: String?
        try fields(in: bytes, range: range) { number, value in
            if number == 14, case let .varint(raw) = value {
                external = raw == 1
            } else if number == 13, case let .bytes(entry) = value {
                let pair = try stringPair(entry, in: bytes)
                if pair.key == "location" { location = pair.value }
            }
        }
        if let location {
            found.insert(location)
        } else if external {
            throw InvalidProtobuf()
        }
    }

    private static func stringPair(
        _ range: Range<Int>, in bytes: UnsafeRawBufferPointer
    ) throws -> (key: String?, value: String?) {
        var key: String?
        var value: String?
        try fields(in: bytes, range: range) { number, field in
            guard case let .bytes(stringBytes) = field,
                  let string = String(bytes: bytes[stringBytes], encoding: .utf8)
            else { return }
            if number == 1 { key = string }
            if number == 2 { value = string }
        }
        return (key, value)
    }

    private static func fields(
        in bytes: UnsafeRawBufferPointer, range: Range<Int>,
        _ body: (Int, ProtobufValue) throws -> Void
    ) throws {
        var index = range.lowerBound
        while index < range.upperBound {
            let key = try varint(in: bytes, index: &index, end: range.upperBound)
            let rawNumber = key >> 3
            guard rawNumber > 0, rawNumber <= UInt64(Int.max) else {
                throw InvalidProtobuf()
            }
            let number = Int(rawNumber)
            switch key & 0x07 {
            case 0:
                try body(number, .varint(try varint(
                    in: bytes, index: &index, end: range.upperBound)))
            case 1:
                guard index <= range.upperBound - 8 else { throw InvalidProtobuf() }
                index += 8
            case 2:
                let length = try varint(in: bytes, index: &index, end: range.upperBound)
                guard length <= UInt64(range.upperBound - index) else {
                    throw InvalidProtobuf()
                }
                let end = index + Int(length)
                try body(number, .bytes(index ..< end))
                index = end
            case 5:
                guard index <= range.upperBound - 4 else { throw InvalidProtobuf() }
                index += 4
            default:
                throw InvalidProtobuf()
            }
        }
    }

    private static func varint(
        in bytes: UnsafeRawBufferPointer, index: inout Int, end: Int
    ) throws -> UInt64 {
        var value: UInt64 = 0
        for shift in stride(from: 0, through: 63, by: 7) {
            guard index < end else { throw InvalidProtobuf() }
            let byte = bytes[index]
            index += 1
            if shift == 63, byte > 1 { throw InvalidProtobuf() }
            value |= UInt64(byte & 0x7F) << UInt64(shift)
            if byte & 0x80 == 0 { return value }
        }
        throw InvalidProtobuf()
    }
}
