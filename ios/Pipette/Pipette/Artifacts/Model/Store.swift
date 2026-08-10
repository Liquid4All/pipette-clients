import Foundation

/// Why a store operation could not answer — the crate's `ModelStoreError`, narrowed to
/// the cases this client can reach.
nonisolated enum ModelStoreError: Error, Equatable, LocalizedError {
    /// No storage key: Apple Foundation ships with the OS, and a store-relative
    /// coordinate is already inside a store.
    case notStorable(String)
    /// The fetch reported success but the entry still does not resolve — a partial
    /// install, or a projector that never landed.
    case unresolvedAfterFetch(String)
    /// The arm names weights this client cannot fetch — see `Model.isFetchableHere`.
    /// Refused before waiting, since no transfer will ever be registered for it.
    case notFetchableHere(String)
    /// The fetch did not finish within the window.
    case fetchTimedOut(String)
    /// The transfer is paused, so waiting on it would hang the caller indefinitely.
    case fetchPaused(String)
    /// The transfer failed, with the reason it gave.
    case fetchFailed(String, reason: String)

    var errorDescription: String? {
        switch self {
        case let .notStorable(name): "\(name) has no store entry to resolve."
        case let .unresolvedAfterFetch(name): "\(name) finished downloading but its files are missing."
        case let .notFetchableHere(name): "\(name) can't be downloaded on iOS."
        case let .fetchTimedOut(name): "Downloading \(name) took too long."
        case let .fetchPaused(name): "Downloading \(name) is paused."
        case let .fetchFailed(name, reason): "Could not download \(name): \(reason)"
        }
    }
}

/// The model store rooted at `modelsDir` — the Swift mirror of the crate's
/// `ModelArtifactStore` (`pipette-artifacts/src/model/store.rs`).
///
/// One type owns the layout, so callers ask it where an entry is instead of rebuilding
/// `models/<key>/blobs/<subpath>` themselves. Discovery, the installers and the run path
/// each did that separately and agreed only by convention.
///
/// Narrower than the crate's in two documented ways. There is no `ensure` that fetches:
/// downloading is `DownloadCoordinator`'s job — it is long-running, cancellable and
/// reports progress to the UI, none of which a synchronous store call can carry — so
/// `ensure` here is find-or-nil and the fetch stays with the coordinator. And `stored` is
/// its own `StoredModel` rather than a `Model` with relative arms — same information, see
/// that type for why.
///
/// Locations come from the manifest's `stored` (the crate's `bind_under`). An entry
/// published before that field falls back to deriving from `declared`, which is what the
/// per-variant rules below are for; they are the compatibility path, not the design.
nonisolated struct ModelArtifactStore: Sendable {
    let modelsDir: URL

    /// A model's store entry, `models/<key>`. Nil for a spec with no storage key (AFM
    /// ships with the OS). Path only — nothing is created, so a reader cannot leave an
    /// empty manifest-less directory behind for the sweeper to reclaim.
    func entryDir(for declared: Model) -> URL? {
        ModelStorageKey.of(declared).map {
            modelsDir.appendingPathComponent($0.value, isDirectory: true)
        }
    }

    /// Create and backup-exclude an entry, returning it — the install side of the store.
    /// Nil for a spec with no storage key.
    @discardableResult
    func prepareEntryDir(for declared: Model) -> URL? {
        entryDir(for: declared).map {
            try? FileManager.default.createDirectory(at: $0, withIntermediateDirectories: true)
            LocalStorage.markExcludedFromBackup($0)
            return $0
        }
    }

    /// The payload directory inside an entry, `<entry>/blobs`. Path only.
    func blobsDir(for declared: Model) -> URL? {
        entryDir(for: declared)?.appendingPathComponent(Entry.blobsDirName, isDirectory: true)
    }

    /// The manifest of `declared`'s entry, or nil when this build cannot read one —
    /// the crate's `find`. An unreadable manifest is not an entry: that is the rule the
    /// quota accountant's garbage phase depends on.
    func find(_ declared: Model) -> ModelManifest? {
        entryDir(for: declared).flatMap { ModelManifest.forInstalledEntry(atDir: $0) }
    }

    /// Every readable manifest under the root — the crate's `list`. Skips what it cannot
    /// read rather than failing the listing, because the callers are discovery and the
    /// UI, where one bad entry must not hide the rest.
    func list() -> [ModelManifest] {
        Self.entryDirectories(in: modelsDir).compactMap { ModelManifest.forInstalledEntry(atDir: $0) }
    }

    /// Drop `declared`'s entry whole — the crate's `remove`. One entry, one delete: a
    /// vision model's projector and an MLX bundle's shards live inside it. Returns
    /// whether anything was there to remove.
    @discardableResult
    func remove(_ declared: Model) -> Bool {
        guard let dir = entryDir(for: declared),
              FileManager.default.fileExists(atPath: dir.path) else { return false }
        do {
            try FileManager.default.removeItem(at: dir)
            return true
        } catch {
            AppLog.storage.error("could not remove model entry \(dir.lastPathComponent): \(error)")
            return false
        }
    }

    /// The bound `Model` for `declared`, fetching and publishing it first if the store
    /// does not already hold it — the crate's `ensure` (`model/store.rs:177`).
    ///
    /// `fetch` is injected for the same reason it is upstream: the store owns find, touch,
    /// publish and bind, while *how* bytes arrive belongs to the caller. On this client
    /// that is `DownloadCoordinator`, so the closure — and therefore this — is `async`,
    /// where the crate's is blocking.
    ///
    /// A hit stamps `last_used_at` and returns; a miss fetches and then requires the
    /// entry to be resolvable, so "ensured" means the weights are on disk and located,
    /// never merely that a transfer was started.
    func ensure(_ declared: Model, fetch: (Model) async throws -> Void) async throws -> Model {
        if let hit = bound(declared) {
            if let dir = entryDir(for: declared) { ModelManifest.touchLastUsed(inEntryDir: dir) }
            return hit
        }
        guard entryDir(for: declared) != nil else {
            throw ModelStoreError.notStorable(declared.artifactName)
        }
        try await fetch(declared)
        guard let bound = bound(declared) else {
            throw ModelStoreError.unresolvedAfterFetch(declared.artifactName)
        }
        return bound
    }

    /// Write the entry's manifest, measuring `blobs/` as it goes — the publish half of
    /// the crate's `install_dir_computing_manifest`.
    ///
    /// The one place a manifest is created, so `blobsBytes` cannot be forgotten by an
    /// installer. A vision model publishes twice (weights and projector arrive as
    /// separate transfers); the second measurement supersedes the first, which is why
    /// the size is taken here rather than passed in.
    func publishManifest(for declared: Model, fetchedAt: Date = Date()) {
        guard let entry = entryDir(for: declared), let blobs = blobsDir(for: declared) else { return }
        ModelManifest(declared: declared, fetchedAt: fetchedAt,
                      blobsBytes: DiskUsage.bytes(at: blobs),
                      stored: declared.toStored(base: Entry.blobsDirName))
            .writeQuietly(to: ModelManifest.manifestURL(inEntryDir: entry))
    }

    /// `declared` as it exists on this device: the bound `Model`, whose arms are the
    /// `Absolute*` ones — the crate's `bind_under`.
    ///
    /// One lookup for the whole entry, because a vision entry is *two* files that have to
    /// agree. Answering "where are the weights" and "where is the projector" separately is
    /// how they came to disagree in the first place; upstream's consumers destructure one
    /// bound `Model` (`pipette-llamacpp/src/models.rs:36`) for exactly this reason.
    ///
    /// Reads the manifest's `stored`; falls back to deriving from `declared` for an entry
    /// published before that field. Nil when the entry holds no payload — evicted or
    /// half-installed, which is not runnable.
    func bound(_ declared: Model) -> Model? {
        guard let dir = entryDir(for: declared) else { return nil }
        return Self.bound(inEntryDir: dir, declared: declared)
    }

    /// The same, bound under a directory the caller already holds.
    ///
    /// Discovery and the quota survey walk the store root and must answer for the entry
    /// they *found*, not for where the key says it should be — an entry in a misnamed
    /// directory is still readable through its manifest, and recomputing the key would
    /// make it invisible. `bind_under` takes the root it is given for the same reason.
    static func bound(inEntryDir dir: URL, declared: Model) -> Model? {
        let candidate = ModelManifest.forInstalledEntry(atDir: dir)?.stored?.bindUnder(dir)
            ?? derivedBound(inEntryDir: dir, for: declared)
        guard let candidate, let paths = candidate.boundPaths,
              FileManager.default.fileExists(atPath: paths.payload),
              paths.mmproj.map({ FileManager.default.fileExists(atPath: $0) }) ?? true
        else { return nil }
        return candidate
    }

    /// The bound weights path, for callers that need only that.
    func payloadPath(for declared: Model) -> String? { bound(declared)?.boundPaths?.payload }

    /// The bound projector path; nil unless this is a vision entry.
    func mmprojPath(for declared: Model) -> String? { bound(declared)?.boundPaths?.mmproj }

    /// Compatibility for an entry published before `stored`: rebuild the store form from
    /// `declared` — which is exactly what `toStored` records — and bind that.
    ///
    /// Not a second layout rule: it calls `toStored`, so the derivation and the recorded
    /// value cannot drift apart.
    private static func derivedBound(inEntryDir dir: URL, for declared: Model) -> Model? {
        declared.toStored(base: Entry.blobsDirName)?.bindUnder(dir)
    }

    /// One level of the root: each child directory is a candidate entry, recognized only
    /// through its manifest. Nothing is sniffed.
    static func entryDirectories(in root: URL) -> [URL] {
        let children = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [])) ?? []
        return children.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
        }
    }
}
