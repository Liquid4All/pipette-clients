import Foundation

/// On-disk provenance at the root of a model's store entry
/// (`models/<key>/manifest.json`), so a downloaded copy is self-describing.
/// `declared` is the crate-tagged `Model` coordinate the model was fetched from;
/// discovery reads it instead of re-deriving the model type from the file layout, and
/// the quota accountant treats a readable manifest as the definition of a live entry.
nonisolated struct ModelManifest: Codable, Hashable, Sendable {
    /// Bump when the on-disk shape changes incompatibly. `read(from:)` accepts only
    /// this exact version: a bump strands the old store, which the sweeper then
    /// reclaims as garbage. There is no migrator.
    static let currentVersion = 2

    var version = currentVersion
    let declared: Model
    /// When the fetch completed. Provenance, and the tiebreak when two entries
    /// share a `lastUsedAt`. Named for the CLI's `fetched_at`, which records the
    /// same thing: a local import is fetched but never downloaded.
    var fetchedAt: Date?
    /// When the model was last resolved for a run — the eviction order. Non-optional
    /// so the synthesized decode requires the key: a v1 manifest lacking it fails to
    /// decode rather than being bridged by a default.
    var lastUsedAt = Date()
    /// Bytes `blobs/` occupied at publish, so a survey can total a store by reading
    /// manifests instead of walking every payload file — the crate's `blobs_bytes`.
    ///
    /// Measures `blobs/` alone, not the entry: `lastUsedAt` is rewritten on every hit,
    /// and a total including the manifest carrying it would drift with it. `nil` on
    /// entries published before this field, where the survey falls back to walking.
    var blobsBytes: Int64?
    /// Where this entry's files actually landed, relative to the entry — the crate's
    /// `stored`. Read back by `bindUnder`, which is how the store answers "where are the
    /// bytes" without re-deriving a layout that has drifted from the writer twice.
    ///
    /// Optional for entries published before it, which fall back to derivation and are
    /// backfilled the next time their manifest is written.
    var stored: Model?

    enum CodingKeys: String, CodingKey {
        case version = "manifest_version"
        case declared
        case fetchedAt = "fetched_at"
        case lastUsedAt = "last_used_at"
        case blobsBytes = "blobs_bytes"
        case stored
    }

    /// Written credential-free, whatever the caller holds. A store entry describes
    /// weights; a plan's `auth_token` is a property of the fetch that produced them, and
    /// identity ignores it, so dropping it costs a reader nothing. Redacting here rather
    /// than at each call site also cleans an entry written by an earlier build, since
    /// `touchLastUsed` reads a manifest and writes it back.
    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(version, forKey: .version)
        try c.encode(declared.withoutAuthToken, forKey: .declared)
        try c.encodeIfPresent(fetchedAt, forKey: .fetchedAt)
        try c.encode(lastUsedAt, forKey: .lastUsedAt)
        try c.encodeIfPresent(blobsBytes, forKey: .blobsBytes)
        try c.encodeIfPresent(stored, forKey: .stored)
    }
}

extension ModelManifest {
    /// The entry's files on this host — the crate's `bind_under`, joining `stored`'s
    /// relative paths onto `entryDir`. `nil` when the entry predates `stored`; the caller
    /// falls back to deriving from `declared`.
    nonisolated func bindUnder(_ entryDir: URL) -> (payload: String, mmproj: String?)? {
        stored?.bindUnder(entryDir)?.boundPaths
    }

    /// Manifest path at the root of a store entry.
    nonisolated static func manifestURL(inEntryDir dir: URL) -> URL {
        dir.appendingPathComponent(Entry.manifestName)
    }

    nonisolated func encoded() throws -> Data { try Coding.encoder.encode(self) }

    /// Best-effort write — a manifest failure must never sink an otherwise-good
    /// install, so callers don't `try`. Atomic so a reader never sees a partial file.
    nonisolated func writeQuietly(to url: URL) { try? encoded().write(to: url, options: .atomic) }

    /// Read the manifest of an installed entry, whatever format it holds.
    nonisolated static func forInstalledEntry(atDir dir: URL) -> ModelManifest? {
        read(from: manifestURL(inEntryDir: dir))
    }

    /// Refresh `last_used_at` on an entry. Best-effort in both directions: an entry
    /// with no readable manifest is left alone (never fabricated), and a failed write
    /// is swallowed — a resolve must not fail over an eviction-order bookkeeping write.
    nonisolated static func touchLastUsed(inEntryDir dir: URL) {
        let url = manifestURL(inEntryDir: dir)
        guard var manifest = read(from: url) else { return }
        manifest.lastUsedAt = Date()
        manifest.writeQuietly(to: url)
    }

    /// Read a manifest, or `nil` if absent, corrupt, or written by any other version.
    nonisolated static func read(from url: URL) -> ModelManifest? {
        guard let data = try? Data(contentsOf: url),
              let m = try? Coding.decoder.decode(ModelManifest.self, from: data),
              m.version == currentVersion
        else { return nil }
        return m
    }
}
