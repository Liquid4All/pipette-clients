import Foundation

/// One row of the synced catalog index — the crate's `BenchmarkSummary`
/// (`pipette-mgmt-client/src/types.rs:357`).
///
/// Deliberately loose, as upstream is: the id and type are lifted out, and the whole
/// server row is retained so a `parameter_*` key this client does not model still
/// reaches the readers that index by string. The crate spells that with
/// `#[serde(flatten)] parameters: BTreeMap<String, Value>`; Swift has no flatten for an
/// untyped bag, so the row itself is the bag.
nonisolated struct BenchmarkSummary {
    let benchmarkId: String
    let benchmarkType: String
    /// The full row, `parameter_*` keys included.
    let rawJson: [String: Any]

    init?(row: [String: Any]) {
        guard let benchmarkId = row["benchmark_id"] as? String,
              let benchmarkType = row["benchmark_type"] as? String
        else { return nil }
        self.benchmarkId = benchmarkId
        self.benchmarkType = benchmarkType
        self.rawJson = row
    }
}

/// The benchmark catalog on disk — the crate's `BenchmarkStore`
/// (`pipette-cli/src/benchmarks/store.rs:23`).
///
/// A concrete filesystem handle over one directory, not a protocol: call sites take the
/// store itself, and tests point one at a temporary directory rather than substituting a
/// fake. That is the crate's shape, and the reason it gives — no path-trait indirection
/// — is the same one ``IdentityStore`` states.
///
/// ```text
/// benchmarks/
///   local/<id>.json     # generated here by StandardBenchmarks; never submitted
///   remote/index.json   # the GET /benchmarks list, loose rows
///   remote/sync.json    # list ETag + per-id ETag map
///   remote/<id>.json    # one synced definition, eval samples included
/// ```
///
/// Reads tolerate a missing tree; writes create it. Ids are written verbatim
/// (`<id>.json`) as the crate writes them, so the same benchmark is the same filename on
/// both clients — `entryJson` refuses an id that could escape the directory rather than
/// percent-encoding it into something the crate would not recognize.
///
/// Stateless, so the concurrent per-id writes in the sync's fetch group stay safe: each
/// writes its own file atomically.
nonisolated struct BenchmarkStore: Sendable {
    /// The workspace `benchmarks/` directory.
    let root: URL

    private static let indexName = "index"
    private static let syncName = "sync"
    /// Reserved stems in `remote/` — everything else there is a definition.
    private static let reservedStems: Set<String> = [indexName, syncName]

    private func sourceDir(_ source: BenchmarkSource) -> URL {
        root.appendingPathComponent(source.rawValue, isDirectory: true)
    }

    /// `<source>/<name>.json`, or nil when `name` is not a safe path component.
    private func entryJson(_ source: BenchmarkSource, _ name: String) -> URL? {
        guard !name.isEmpty, !name.contains("/"), name != ".", name != ".." else { return nil }
        return sourceDir(source).appendingPathComponent("\(name).json")
    }

    private func ensure(_ source: BenchmarkSource) throws {
        try FileManager.default.createDirectory(
            at: sourceDir(source), withIntermediateDirectories: true)
    }

    // MARK: - Definitions

    /// Load by qualified address. Nil when missing or unparseable.
    func get(_ reference: SourcedBenchmarkId) -> BenchmarkDefinition? {
        guard let path = entryJson(reference.source, reference.id),
              let data = try? Data(contentsOf: path)
        else { return nil }
        return try? Coding.decoder.decode(BenchmarkDefinition.self, from: data)
    }

    /// Create or replace a definition on one half of the catalog.
    func put(_ source: BenchmarkSource, _ definition: BenchmarkDefinition) throws {
        guard let path = entryJson(source, definition.benchmarkId) else {
            throw BenchmarkStoreError.unsafeBenchmarkId(definition.benchmarkId)
        }
        try ensure(source)
        try Coding.encoder.encode(definition).write(to: path, options: .atomic)
    }

    /// Every definition on one half, sorted by id.
    ///
    /// A file that fails to parse is logged and skipped rather than failing the listing,
    /// as the crate's `list` does — one corrupt entry must not hide the catalog. On the
    /// remote half `index.json` / `sync.json` are skipped: they are sync metadata, not
    /// definitions.
    func list(_ source: BenchmarkSource) -> [BenchmarkDefinition] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: sourceDir(source), includingPropertiesForKeys: nil)
        else { return [] }
        return files
            .filter { $0.pathExtension == "json" }
            .filter {
                source != .remote
                    || !Self.reservedStems.contains($0.deletingPathExtension().lastPathComponent)
            }
            .compactMap { path in
                guard let data = try? Data(contentsOf: path) else { return nil }
                do {
                    return try Coding.decoder.decode(BenchmarkDefinition.self, from: data)
                } catch {
                    AppLog.benchmarkSync.warning(
                        "skipping catalog file \(path.lastPathComponent): \(error)")
                    return nil
                }
            }
            .sorted { $0.benchmarkId < $1.benchmarkId }
    }

    // MARK: - Remote sync metadata

    /// The synced index rows. Empty when never synced.
    func listRemoteIndex() -> [BenchmarkSummary] {
        guard let path = entryJson(.remote, Self.indexName),
              let data = try? Data(contentsOf: path),
              let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }
        return rows.compactMap(BenchmarkSummary.init(row:))
    }

    /// Replace the index with the server's list response, stored verbatim so a row this
    /// client cannot model still round-trips.
    ///
    /// Callers must pass rows that already parsed as definitions (`keepParseable`): the
    /// 304 path in `BenchmarkSync` takes the stored ids without re-filtering, and
    /// `BenchmarkSummary` only checks that `benchmark_id` and `benchmark_type` are
    /// present.
    func putRemoteIndex(_ data: Data) throws {
        guard let path = entryJson(.remote, Self.indexName) else { return }
        try ensure(.remote)
        try data.write(to: path, options: .atomic)
    }

    /// Is a synced definition already on disk? Drives the conditional per-id fetch.
    func hasRemoteDetail(id: String) -> Bool {
        guard let path = entryJson(.remote, id) else { return false }
        return FileManager.default.fileExists(atPath: path.path)
    }

    func getSyncState() -> BenchmarkSync.SyncState? {
        guard let path = entryJson(.remote, Self.syncName),
              let data = try? Data(contentsOf: path)
        else { return nil }
        return try? Coding.decoder.decode(BenchmarkSync.SyncState.self, from: data)
    }

    func putSyncState(_ state: BenchmarkSync.SyncState) throws {
        guard let path = entryJson(.remote, Self.syncName) else { return }
        try ensure(.remote)
        try Coding.encoder.encode(state).write(to: path, options: .atomic)
    }

    /// Drop every synced definition not in `ids` (best-effort). Sync metadata is kept.
    func pruneRemoteDetails(keeping ids: [String]) {
        let keep = Set(ids)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: sourceDir(.remote), includingPropertiesForKeys: nil)
        else { return }
        for file in files where file.pathExtension == "json" {
            let stem = file.deletingPathExtension().lastPathComponent
            guard !keep.contains(stem), !Self.reservedStems.contains(stem) else { continue }
            try? FileManager.default.removeItem(at: file)
            AppLog.benchmarkSync.info("pruned orphan detail \(stem)")
        }
    }

    /// Drop the synced index, details and sync state — the crate's `clear_remote`.
    /// The local half is untouched, which is the whole point of the split.
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
}

nonisolated enum BenchmarkStoreError: LocalizedError, Equatable {
    case unsafeBenchmarkId(String)

    var errorDescription: String? {
        switch self {
        case .unsafeBenchmarkId(let id):
            return "`\(id)` is not usable as a catalog filename"
        }
    }
}
