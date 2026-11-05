import CryptoKit
import Foundation

/// The benchmark catalog on disk: the crate's `BenchmarkStore`
/// (`crates/pipette-cli/src/benchmarks/store.rs`). A concrete handle over one directory,
/// not a protocol: tests point one at a temporary directory rather than substituting a fake.
///
/// ```text
/// benchmarks/
///   local/<id>.json          # generated here by StandardBenchmarks; never submitted
///   remote/detail/<id>.json  # one synced definition as the server sent it, eval samples
///                            # included (why: `definitionsDir`)
///   remote/detail/<id>.etag  # the sha256 of that body, then the ETag it came with, if any
///   remote/state/index.json  # the GET /benchmarks list as the server sent it, every row
///   remote/state/sync.json   # RemoteSyncState: server, list ETag
///   remote/*.json            # an older build's index, sync state and definitions: never
///                            # read, deleted only by `clearRemote` (why: `stateDir`)
/// ```
///
/// Ids are written verbatim, as the crate writes them, so the same benchmark is the same
/// filename on both clients: `fileJson` refuses an id that could escape the directory
/// rather than percent-encoding it into a name the crate would not recognize.
///
/// Stateless, so a claim caching a definition while a pull writes or prunes stays safe:
/// each write is atomic, an ETag is honored only for the body it hashes, and a delete that
/// finds the file already gone is not an error.
nonisolated struct BenchmarkStore: Sendable {
    let root: URL

    private func sourceDir(_ source: BenchmarkSource) -> URL {
        root.appendingPathComponent(source.rawValue, isDirectory: true)
    }

    /// Builds before `detail/` cached the typed definition at `remote/<id>.json`, losing any
    /// parameter they did not model (one before `parameter_num_images` reads an 80-image
    /// benchmark back as one image); nothing in those files says what was lost, so nothing
    /// here reads them. A new directory rather than a marker: older builds never look in it,
    /// and with no bookkeeping in it a benchmark named `index` or `sync` lists like any other.
    private func definitionsDir(_ source: BenchmarkSource) -> URL {
        switch source {
        case .local: sourceDir(.local)
        case .remote: sourceDir(.remote).appendingPathComponent("detail", isDirectory: true)
        }
    }

    private func definitionJson(_ source: BenchmarkSource, _ id: String) -> URL? {
        Self.fileJson(in: definitionsDir(source), id)
    }

    private func definitionEtag(_ id: String) -> URL? {
        definitionJson(.remote, id)?.deletingPathExtension().appendingPathExtension("etag")
    }

    /// Only builds with this layout read or write here. Older builds keep their bookkeeping at
    /// `remote/index.json` and `remote/sync.json` with only the rows each can convert, so
    /// sharing it would let a `304` hand one build another's filtered list. They touch only
    /// `remote/*.json`, never a subdirectory.
    private var stateDir: URL {
        sourceDir(.remote).appendingPathComponent("state", isDirectory: true)
    }

    private var indexJson: URL { stateDir.appendingPathComponent("index.json") }
    private var syncJson: URL { stateDir.appendingPathComponent("sync.json") }

    /// Nil when `name` is not a valid `BenchmarkId`. Checked again here because
    /// `BenchmarkId(validated:)` skips the check, and a name that is not an id must not
    /// become a path. The required leading alphanumeric also keeps `.` and `..` out.
    private static func fileJson(in dir: URL, _ name: String) -> URL? {
        guard BenchmarkId.validate(name) else { return nil }
        return dir.appendingPathComponent("\(name).json")
    }

    private static func ensure(_ dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    // MARK: - Definitions

    /// Nil when missing; an error when the file is there but cannot be used, so a corrupt
    /// definition is not mistaken for an absent one. A remote id is read from
    /// `remote/detail/` only, so one an older build cached is nil too.
    func get(_ reference: SourcedBenchmarkId) throws -> BenchmarkDefinition? {
        guard let path = definitionJson(reference.source, reference.id.value) else { return nil }
        return try Self.readDefFile(path, reference.source)
    }

    /// This build generated a local definition, so its type is the whole of it.
    func putLocal(_ definition: BenchmarkDefinition) throws {
        guard let path = definitionJson(.local, definition.benchmarkId.value) else {
            throw BenchmarkStoreError.unsafeBenchmarkId(definition.benchmarkId.value)
        }
        try Self.ensure(definitionsDir(.local))
        try Coding.encoder.encode(definition).write(to: path, options: .atomic)
    }

    /// The `.etag` file is `<sha256 of the body>\n<etag>`, and ``getRemote(id:)`` honors it only
    /// for the body it hashes: a crash, or a pull and a claim caching the same id at once, can
    /// leave the two files from different writes, which then costs a full fetch, never a `304`
    /// to a body the file no longer holds.
    func putRemote(_ remote: ConvertedRemote, etag: EntityTag?) throws {
        let id = remote.definition.benchmarkId.value
        guard let path = definitionJson(.remote, id), let etagPath = definitionEtag(id) else {
            throw BenchmarkStoreError.unsafeBenchmarkId(id)
        }
        try Self.ensure(definitionsDir(.remote))
        try Self.removeIfPresent(etagPath)
        try remote.sent.rawJson.write(to: path, options: .atomic)
        if let etag {
            try Data("\(Self.digest(remote.sent.rawJson))\n\(etag.value)".utf8)
                .write(to: etagPath, options: .atomic)
        }
    }

    /// Nil when nothing is cached; an error when the body does not read. The ETag is nil when
    /// its file is missing, unreadable or records another body (``putRemote(_:etag:)``), so
    /// the next fetch is unconditional.
    func getRemote(id: BenchmarkId) throws -> (definition: BenchmarkDefinition, etag: EntityTag?)? {
        guard let path = definitionJson(.remote, id.value),
              FileManager.default.fileExists(atPath: path.path)
        else { return nil }
        let body = try Data(contentsOf: path)
        let definition = try Self.decodeDef(body, path, .remote)
        let etag = definitionEtag(id.value)
            .flatMap { try? String(contentsOf: $0, encoding: .utf8) }
            .flatMap { recorded -> EntityTag? in
                let parts = recorded.split(separator: "\n", maxSplits: 1)
                guard parts.count == 2, parts[0] == Self.digest(body) else { return nil }
                return EntityTag.fromHeaderValue(String(parts[1]))
            }
        return (definition, etag)
    }

    /// Nil also when the cached definition does not read.
    func remoteDetailEtag(id: BenchmarkId) -> EntityTag? {
        (try? getRemote(id: id))??.etag
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A file that fails to read is logged and skipped, as in the crate, so one corrupt
    /// entry does not hide the catalog.
    func list(_ source: BenchmarkSource) -> [BenchmarkDefinition] {
        guard let files = try? Self.jsonFiles(in: definitionsDir(source)) else { return [] }
        return files
            .compactMap { path in
                do {
                    return try Self.readDefFile(path, source)
                } catch {
                    AppLog.benchmarkSync.warning(
                        "skipping corrupt catalog file \(path.lastPathComponent): \(error)")
                    return nil
                }
            }
            .sorted { $0.benchmarkId < $1.benchmarkId }
    }

    private static func readDefFile(
        _ path: URL, _ source: BenchmarkSource
    ) throws -> BenchmarkDefinition? {
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        return try decodeDef(Data(contentsOf: path), path, source)
    }

    /// A local file in the retired nested-`parameters` shape is refused by name, since the
    /// typed decode would otherwise report it as a missing field. A remote file is decoded as
    /// a fresh body is: one that converted on fetch must never read back as corrupt, or every
    /// claim would refetch it.
    private static func decodeDef(
        _ data: Data, _ path: URL, _ source: BenchmarkSource
    ) throws -> BenchmarkDefinition {
        if source == .local,
           let document = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           document["parameters"] != nil {
            throw BenchmarkStoreError.nestedParameters(path.lastPathComponent)
        }
        do {
            return try Coding.decoder.decode(BenchmarkDefinition.self, from: data)
        } catch {
            throw BenchmarkStoreError.unparseable(path.lastPathComponent, "\(error)")
        }
    }

    // MARK: - Remote sync metadata

    /// Empty when never synced; an error when the file is there but does not read. Rows this
    /// build cannot convert are included, so a reader that lists or resolves synced
    /// benchmarks filters through ``benchmarkDefinitionFromSummary(_:)``.
    func listRemoteIndex() throws -> [PipetteMgmtClient.BenchmarkSummary] {
        guard FileManager.default.fileExists(atPath: indexJson.path) else { return [] }
        return try Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self, from: Data(contentsOf: indexJson))
    }

    /// Each row is written as the bytes the server sent, so a key this client does not model
    /// still round-trips, and no row is dropped: the list ETag validates this file, so a
    /// `304` must hand back the server's list whichever build or version stored it.
    func putRemoteIndex(_ summaries: [PipetteMgmtClient.BenchmarkSummary]) throws {
        try Self.ensure(stateDir)
        var data = Data("[".utf8)
        for (index, summary) in summaries.enumerated() {
            if index > 0 { data.append(Data(",".utf8)) }
            data.append(summary.rawJson)
        }
        data.append(Data("]".utf8))
        try data.write(to: indexJson, options: .atomic)
    }

    /// A file an older build cached at `remote/<id>.json` never counts.
    func hasRemoteDetail(id: BenchmarkId) -> Bool {
        guard let path = definitionJson(.remote, id.value) else { return false }
        return FileManager.default.fileExists(atPath: path.path)
    }

    /// Nil when never pulled; an error when the file is there but does not read.
    func getSyncState() throws -> RemoteSyncState? {
        guard FileManager.default.fileExists(atPath: syncJson.path) else { return nil }
        return try Coding.decoder.decode(RemoteSyncState.self, from: Data(contentsOf: syncJson))
    }

    func putSyncState(_ state: RemoteSyncState) throws {
        try Self.ensure(stateDir)
        try Coding.encoder.encode(state).write(to: syncJson, options: .atomic)
    }

    /// Each pull rewrites the list wholesale but not the detail files and their ETags, so
    /// without this a benchmark dropped server-side stays runnable forever.
    ///
    /// `ids` must be every id of the *list* response, rows this build does not fetch
    /// included: a build that reads one may have cached its detail, and a benchmark whose
    /// detail merely failed this round is still in the catalog.
    func pruneRemoteDetails(keeping ids: [String]) throws {
        let keep = Set(ids)
        for file in try Self.jsonFiles(in: definitionsDir(.remote)) {
            let stem = file.deletingPathExtension().lastPathComponent
            guard !keep.contains(stem) else { continue }
            AppLog.benchmarkSync.debug(
                "pruning remote benchmark detail `\(stem)`: no longer in the catalog")
            try Self.removeIfPresent(file.deletingPathExtension().appendingPathExtension("etag"))
            try Self.removeIfPresent(file)
        }
    }

    /// Takes an older build's `remote/*.json` files too. The local half is untouched.
    func clearRemote() {
        let dir = sourceDir(.remote)
        do {
            try FileManager.default.removeItem(at: dir)
        } catch CocoaError.fileNoSuchFile {
            // Never synced; nothing to drop.
        } catch {
            AppLog.benchmarkSync.error("Failed to clear remote benchmarks at \(dir.path): \(error)")
        }
    }

    // MARK: - Files

    /// A directory that does not exist holds none.
    private static func jsonFiles(in dir: URL) throws -> [URL] {
        guard FileManager.default.fileExists(atPath: dir.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
    }

    private static func removeIfPresent(_ file: URL) throws {
        do {
            try FileManager.default.removeItem(at: file)
        } catch CocoaError.fileNoSuchFile {
            // A concurrent claim or pull got there first.
        }
    }
}

nonisolated enum BenchmarkStoreError: LocalizedError, Equatable {
    case unsafeBenchmarkId(String)
    case nestedParameters(String)
    /// The file name and why it did not decode.
    case unparseable(String, String)

    var errorDescription: String? {
        switch self {
        case .unsafeBenchmarkId(let id):
            return ModelError.invalidBenchmarkId(id).errorDescription
        case .nestedParameters(let file):
            return "benchmark file \(file) uses unsupported nested parameters object"
        case let .unparseable(file, reason):
            return "failed to parse benchmark document in \(file): \(reason)"
        }
    }
}
