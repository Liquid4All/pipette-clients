import Foundation

/// `models/<key>/manifest.json`, the crate's `ModelManifest`. `declared` is the
/// coordinate it was fetched from, so discovery need not guess the type from the layout;
/// `stored` is the same model relative to `models/`, which `bindUnder` turns into paths.
///
/// A record only: the read policy lives in `ModelArtifactStore.readManifest`.
nonisolated struct ModelManifest: Codable, Hashable, Sendable {
    /// The store accepts only this exact version: a bump strands the old store, which
    /// the sweep reclaims as garbage. There is no migrator.
    static let manifestVersion = 2

    var manifestVersion: Int
    /// When the fetch was published; also the tiebreak for equal `lastUsedAt`.
    var fetchedAt: Date
    /// The eviction order. Seeded to `fetchedAt` on publish, rewritten best-effort on
    /// every `ensure` hit.
    var lastUsedAt: Date
    var declared: Model
    /// Read back by `bindUnder`, so the store never re-derives a layout (which has
    /// drifted from the writer before).
    var stored: Model
    /// The Hub snapshot and file inventory behind this entry. Optional so local/GGUF
    /// installs and older version-2 manifests stay readable without a version bump.
    var resolved: ResolvedTransfer?
    /// `blobs/` at publish, so a survey reads manifests instead of walking payloads. Not
    /// the whole entry: the manifest itself changes size as `lastUsedAt` is rewritten.
    /// When absent, the survey walks `blobs/`.
    var blobsBytes: Int64?

    init(fetchedAt: Date, lastUsedAt: Date, declared: Model, stored: Model,
         blobsBytes: Int64?, resolved: ResolvedTransfer? = nil) {
        self.manifestVersion = Self.manifestVersion
        self.fetchedAt = fetchedAt
        self.lastUsedAt = lastUsedAt
        self.declared = declared
        self.stored = stored
        self.blobsBytes = blobsBytes
        self.resolved = resolved
    }

    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version"
        case fetchedAt = "fetched_at"
        case lastUsedAt = "last_used_at"
        case declared
        case stored
        case blobsBytes = "blobs_bytes"
        case resolved
    }

    /// Written credential-free: an `auth_token` belongs to the fetch, not the weights,
    /// and identity ignores it.
    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(manifestVersion, forKey: .manifestVersion)
        try c.encode(fetchedAt, forKey: .fetchedAt)
        try c.encode(lastUsedAt, forKey: .lastUsedAt)
        try c.encode(declared.withoutAuthToken, forKey: .declared)
        try c.encode(stored, forKey: .stored)
        try c.encodeIfPresent(blobsBytes, forKey: .blobsBytes)
        try c.encodeIfPresent(resolved, forKey: .resolved)
    }
}

extension ModelManifest {
    /// The crate's `bind_under`. Nil when `stored` has no store-relative arm.
    nonisolated func bindUnder(_ modelsDir: URL) -> Model? {
        stored.underRoot(modelsDir)
    }

    nonisolated static func manifestURL(inEntryDir dir: URL) -> URL {
        dir.appendingPathComponent(Entry.manifestName)
    }

    nonisolated func encoded() throws -> Data { try Coding.encoder.encode(self) }

    /// For an entry whose payload already landed: failing the install over the record
    /// would lose a completed download.
    nonisolated func writeQuietly(to url: URL) { try? encoded().write(to: url, options: .atomic) }

    /// For a staging attempt, where a failed write must stop the publish: a staged
    /// attempt without a manifest is an orphan the sweep reclaims, but a published one is
    /// a husk that occupies quota and cannot be deleted.
    nonisolated func write(to url: URL) throws {
        try encoded().write(to: url, options: .atomic)
    }
}
