import Foundation

// The synced half of the catalog: the crate's `benchmarks/remote.rs`, plus its
// `ensure_claim_benchmark_cached` (`commands/worker.rs`). The pull itself is in
// `Client/Sync.swift`.

/// Pull bookkeeping, stored as `remote/state/sync.json`. Its list ETag validates only
/// `state/index.json`; each `detail/` file keeps its own ETag beside it, so a claim that
/// rewrites one never touches this.
nonisolated struct RemoteSyncState: Codable, Equatable {
    var pulledAt: String
    /// The listed rows the pull fetches details for: a `benchmark_type` this build knows and
    /// an id it accepts. Any other row is stored, not counted.
    var benchmarkCount: Int
    /// The server the cached catalog and its ETags came from. `sync server-url=` lets an
    /// unregistered device point the next pull at a different host, where revalidating
    /// would let a 304 keep a catalog that host never served.
    var serverUrl: ServerUrl?
    /// An ``EntityTag`` encodes as the bare header value, as in the crate.
    var benchmarksEtag: EntityTag?

    enum CodingKeys: String, CodingKey {
        case pulledAt = "pulled_at"
        case benchmarkCount = "benchmark_count"
        case serverUrl = "server_url"
        case benchmarksEtag = "benchmarks_etag"
    }

    init(pulledAt: String, benchmarkCount: Int, serverUrl: ServerUrl?,
         benchmarksEtag: EntityTag?) {
        self.pulledAt = pulledAt
        self.benchmarkCount = benchmarkCount
        self.serverUrl = serverUrl
        self.benchmarksEtag = benchmarksEtag
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Older builds wrote none. Such a file records no server either, so the next pull
        // goes cold anyway.
        pulledAt = try c.decodeIfPresent(String.self, forKey: .pulledAt) ?? ""
        benchmarkCount = try c.decode(Int.self, forKey: .benchmarkCount)
        serverUrl = try Self.recordedServerUrl(c)
        benchmarksEtag = try c.decodeIfPresent(EntityTag.self, forKey: .benchmarksEtag)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(pulledAt, forKey: .pulledAt)
        try c.encode(benchmarkCount, forKey: .benchmarkCount)
        try c.encodeIfPresent(serverUrl, forKey: .serverUrl)
        try c.encodeIfPresent(benchmarksEtag, forKey: .benchmarksEtag)
    }

    /// A value that does not parse is an error, not a cold pull: this program wrote the
    /// file, so an unusable URL means it is corrupt. Blank is unset, as older builds wrote it.
    private static func recordedServerUrl(_ c: KeyedDecodingContainer<CodingKeys>) throws -> ServerUrl? {
        guard let raw = try c.decodeIfPresent(String.self, forKey: .serverUrl),
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        do {
            return try ServerUrl(raw)
        } catch {
            throw DecodingError.dataCorruptedError(
                forKey: .serverUrl, in: c,
                debugDescription: "recorded server_url `\(raw)` is not a usable URL")
        }
    }
}

/// The loose value carries the object's own bytes, so the typed decode sees exactly what
/// the server sent.
nonisolated func benchmarkDefinitionFromRemote(_ loose: RemoteBenchmark) throws -> BenchmarkDefinition {
    try Coding.decoder.decode(BenchmarkDefinition.self, from: loose.rawJson)
}

/// Throws when this build cannot convert the row. The list is stored with every row the
/// server sent, so each reader that lists or resolves synced benchmarks filters through this
/// (``BenchmarkCatalog`` decodes the same object the same way).
nonisolated func benchmarkDefinitionFromSummary(
    _ row: PipetteMgmtClient.BenchmarkSummary
) throws -> BenchmarkDefinition {
    try Coding.decoder.decode(BenchmarkDefinition.self, from: row.rawJson)
}

/// ``BenchmarkStore/putRemote(_:etag:)`` takes only this, so a body this build cannot run is
/// never cached, and what is cached is the server's object, keeping the keys this build
/// does not model for a build that does.
nonisolated struct ConvertedRemote: Sendable {
    let definition: BenchmarkDefinition
    let sent: RemoteBenchmark

    init(_ sent: RemoteBenchmark) throws {
        definition = try benchmarkDefinitionFromRemote(sent)
        self.sent = sent
    }
}

/// The server answered `404`, or served a body this build cannot read. A fetch that fails in
/// transit throws the transport's error instead.
nonisolated struct UnknownRemoteBenchmarkError: Error, LocalizedError {
    let message: String

    var errorDescription: String? { message }
}

/// The cached remote definition once `GET /benchmarks/{id}` revalidates it against its
/// stored ETag, else that read's body, cached with its ETag. A claim never syncs, so this is
/// how a change the server made under the same id reaches it: mgmt's ETag is a hash of the
/// body, so a `304` means the cached bytes are the bytes it serves.
///
/// Signed with `auth` because the server serves an organization's private benchmarks only to
/// a signed member; unsigned, one of them is a `404`. Nothing here writes the sync state.
nonisolated func ensureRemoteBenchmarkCached(
    store: BenchmarkStore,
    server: ServerUrl,
    http: HttpClient,
    auth: ClientIdentity?,
    benchmarkId: String
) async throws -> BenchmarkDefinition {
    let log = AppLog.benchmarkSync
    let id = try? BenchmarkId(benchmarkId)
    let cached: BenchmarkDefinition?
    let etag: EntityTag?
    do {
        let stored = try id.flatMap { try store.getRemote(id: $0) }
        cached = stored?.definition
        etag = stored?.etag
    } catch {
        log.warning("benchmark \(benchmarkId): could not read the cached benchmark "
            + "(\(error.localizedDescription)); GET /benchmarks/\(benchmarkId)")
        cached = nil
        etag = nil
    }
    // With a copy cached, an outage runs it unchanged. A `404` does not: mgmt refuses the
    // result of a benchmark it dropped, so it fails the claim as it does uncached.
    func runUnchecked(_ cached: BenchmarkDefinition, _ why: String) -> BenchmarkDefinition {
        log.warning("benchmark \(benchmarkId): outcome=unchecked; could not revalidate the cached "
            + "definition (\(why)), running it")
        return cached
    }
    let response: PipetteMgmtClient.ConditionalResponse<RemoteBenchmark>
    do {
        response = try await PipetteMgmtClient.ClientApi.getBenchmarkConditional(
            server: server, http: http, auth: auth, benchmarkId: benchmarkId,
            ifNoneMatch: etag.map(IfNoneMatch.init))
    } catch {
        if error.httpStatusCode == 404 {
            throw UnknownRemoteBenchmarkError(
                message: "GET /benchmarks/\(benchmarkId): benchmark not found")
        }
        if let cached { return runUnchecked(cached, error.localizedDescription) }
        throw error
    }
    switch response {
    case .notModified:
        // A `304` to a read that named no ETag says nothing about the benchmark, so it is
        // not an unknown one: the claim fails retriably.
        guard let cached else {
            log.warning("benchmark \(benchmarkId): GET /benchmarks/\(benchmarkId) answered 304 "
                + "with nothing cached")
            throw URLError(.badServerResponse)
        }
        log.info("benchmark \(benchmarkId): outcome=current; the server confirmed the cached "
            + "definition")
        return cached
    case let .modified(remote, newEtag):
        let converted: ConvertedRemote
        do {
            converted = try ConvertedRemote(remote)
        } catch {
            // Cached or not: a cache the server changed no longer declares what mgmt checks
            // the result against.
            throw UnknownRemoteBenchmarkError(
                message: "benchmark '\(benchmarkId)' is not a known, valid definition: "
                    + "\(error.localizedDescription)")
        }
        if cached == nil {
            log.info("benchmark \(benchmarkId): not cached; fetched it")
        } else if etag == nil {
            log.info("benchmark \(benchmarkId): outcome=fetched; the cached definition has no "
                + "ETag to revalidate, replacing it")
        } else {
            log.info("benchmark \(benchmarkId): outcome=changed; the server changed it, "
                + "replacing the cached definition")
        }
        do {
            try store.putRemote(converted, etag: newEtag)
            log.debug("benchmark \(benchmarkId): cached on the remote half")
        } catch {
            log.warning("failed to cache remote benchmark \(benchmarkId): \(error.localizedDescription)")
        }
        return converted.definition
    }
}
