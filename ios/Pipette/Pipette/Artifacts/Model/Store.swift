import Foundation

/// The crate's `ModelStoreError`, narrowed to the cases this client can reach.
nonisolated enum ModelStoreError: Error, Equatable, LocalizedError {
    /// Apple Foundation ships with the OS; a store-relative coordinate is already in a store.
    case notStorable(String)
    /// An unsupported version, or a `stored` that is not the declared model's store form.
    case corrupt(String)
    case parse(path: String, source: String)
    case io(doing: String, source: String)
    /// A partial install, or a projector that never landed.
    case unresolvedAfterFetch(String)
    /// See `Model.isFetchableHere`. Refused before waiting, since no transfer will start.
    case notFetchableHere(String)
    case admission(AdmissionError)
    /// The key omits the model type, so two directory models of one `org/repo` with no
    /// `prefix` address one entry.
    case typeCollision(key: String, stored: ModelType, requested: ModelType)
    case fetchTimedOut(String)
    /// Paused by a person, or the wait's one resume did not take; waiting would hang.
    case fetchPaused(String)
    case fetch(model: String, source: String)
    /// Fails every waiter. A caller that only stops waiting gets `CancellationError`, and
    /// the transfer carries on for the others.
    case fetchCancelled(String)

    var errorDescription: String? {
        switch self {
        case let .notStorable(name): "\(name) has no store entry to resolve."
        case let .corrupt(detail): "stored model is corrupt: \(detail)"
        case let .parse(path, reason): "parsing the manifest at \(path) failed: \(reason)"
        case let .io(doing, reason): "\(doing): \(reason)"
        case let .unresolvedAfterFetch(name): "\(name) finished downloading but its files are missing."
        case let .notFetchableHere(name): "\(name) can't be downloaded on iOS."
        case let .admission(refusal): refusal.errorDescription
        case let .typeCollision(key, stored, requested):
            "the store entry `\(key)` holds a \(stored.rawValue) model, but this cell names "
                + "a \(requested.rawValue) model; give each of the two a `prefix` so they "
                + "address separate entries, or remove this one"
        case let .fetchTimedOut(name): "Downloading \(name) stopped making progress."
        case let .fetchPaused(name): "Downloading \(name) is paused."
        case let .fetch(model, reason): "Could not download \(model): \(reason)"
        case let .fetchCancelled(name): "Downloading \(name) was cancelled."
        }
    }
}

/// The crate's `ModelArtifactStore` (`crates/pipette-artifacts/src/model/store.rs`). It
/// owns the layout, so callers ask it where an entry is instead of rebuilding
/// `models/<key>/blobs/<subpath>`. Locations come from the manifest's `stored`, never from
/// re-deriving the declared model's layout.
///
/// Directory installs stage and publish (`Staging.swift`); a GGUF install writes into the
/// entry and `publishManifest` makes it one, because a vision model is two transfers.
nonisolated struct ModelArtifactStore: Sendable {
    let modelsDir: URL

    /// Path only: a reader must not leave an empty manifest-less directory behind.
    func entryDir(for declared: Model) -> URL? {
        ModelStorageKey.of(declared).map {
            modelsDir.appendingPathComponent($0.relativeDir, isDirectory: true)
        }
    }

    @discardableResult
    func prepareEntryDir(for declared: Model) -> URL? {
        entryDir(for: declared).map {
            try? FileManager.default.createDirectory(at: $0, withIntermediateDirectories: true)
            LocalStorage.markExcludedFromBackup($0)
            return $0
        }
    }

    func blobsDir(for declared: Model) -> URL? {
        ModelStorageKey.of(declared).map { modelsDir.appendingPathComponent($0.blobsDir, isDirectory: true) }
    }

    /// The store form bound under the entry's `blobs/`: the `into` that
    /// `ModelFetch.planDownloads` takes. Answerable before a byte exists, so the plan can
    /// be made up front. Nil when there is no key or no store form.
    func fetchDestination(for declared: Model) -> Model? {
        guard let key = ModelStorageKey.of(declared) else { return nil }
        return declared.toStored(base: key.blobsDir)?.underRoot(modelsDir)
    }

    /// The crate's `find`: a pure read, no touch. An unusable manifest is an error, not an
    /// absence. Whether the payload is on disk is `bound`'s question.
    ///
    /// Another type's manifest under the same key is `typeCollision` rather than returned,
    /// which would fail far from the cause, at engine load.
    func find(_ declared: Model) throws(ModelStoreError) -> ModelManifest? {
        guard let entry = entryDir(for: declared) else {
            throw .notStorable(declared.artifactName)
        }
        let url = ModelManifest.manifestURL(inEntryDir: entry)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let manifest = try Self.readManifest(at: url)
        let stored = ModelType.of(manifest.declared)
        let requested = ModelType.of(declared)
        guard stored == requested else {
            throw .typeCollision(key: ModelStorageKey.of(declared)?.value ?? entry.lastPathComponent,
                                 stored: stored, requested: requested)
        }
        return manifest
    }

    /// The crate's `list`. Skips dot-directories and children without a manifest. One
    /// unusable manifest fails the whole listing, as in `find`; discovery, which must not
    /// lose the store to one bad entry, walks the root itself.
    func list() throws(ModelStoreError) -> [ModelManifest] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelsDir.path) else { return [] }
        let children: [URL]
        do {
            children = try fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil)
        } catch {
            throw .io(doing: "reading \(modelsDir.path)", source: error.localizedDescription)
        }
        var manifests: [ModelManifest] = []
        for child in children where !child.lastPathComponent.hasPrefix(".") {
            let url = ModelManifest.manifestURL(inEntryDir: child)
            guard fm.fileExists(atPath: url.path) else { continue }
            manifests.append(try Self.readManifest(at: url))
        }
        return manifests.sorted { $0.declared.description < $1.declared.description }
    }

    /// The crate's `remove`. Returns whether anything was there.
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

    /// The crate's `ensure`. `fetch` gets the caller's `declared`, token included, so a
    /// gated download can authenticate; only what the store keeps is credential-free.
    ///
    /// A miss must resolve after the fetch. Unlike the crate, a hit also needs the payload
    /// on disk: a vision model publishes after each of its two transfers, so a manifest
    /// beside half a payload is an ordinary state here. Also unlike the crate, a manifest
    /// that does not decode is an error like any other unusable one, not a re-fetch, so
    /// corruption surfaces instead of being overwritten.
    func ensure(_ declared: Model, fetch: (Model) async throws -> Void) async throws -> Model {
        if let manifest = try find(declared), let hit = Self.bound(manifest, under: modelsDir) {
            touch(manifest)
            return hit
        }
        try await fetch(declared)
        guard let manifest = try find(declared), let bound = Self.bound(manifest, under: modelsDir) else {
            throw ModelStoreError.unresolvedAfterFetch(declared.artifactName)
        }
        return bound
    }

    /// The publish half of the crate's `fetch_and_publish`, for GGUF installs. `fetchedAt`
    /// is stamped here because publish is the first the store hears of a transfer. A
    /// vision model publishes twice, so the size is measured here and the second write wins.
    func publishManifest(for declared: Model) {
        let storable = declared.withoutAuthToken
        guard let key = ModelStorageKey.of(storable),
              let stored = storable.toStored(base: key.blobsDir) else { return }
        let entry = modelsDir.appendingPathComponent(key.relativeDir, isDirectory: true)
        let blobs = modelsDir.appendingPathComponent(key.blobsDir, isDirectory: true)
        let now = Date()
        ModelManifest(fetchedAt: now, lastUsedAt: now, declared: storable, stored: stored,
                      blobsBytes: entrySizeBytes(at: blobs))
            .writeQuietly(to: ModelManifest.manifestURL(inEntryDir: entry))
    }

    /// `find`, then the crate's `bind_under`. One lookup for the whole entry, so a vision
    /// model's weights and projector cannot disagree. Nil when not stored, unusable, or
    /// not all on disk.
    func bound(_ declared: Model) -> Model? {
        (try? find(declared)).flatMap { Self.bound($0, under: modelsDir) }
    }

    /// Nil unless every file it names is on disk (the crate's `payload_paths` rule): a
    /// manifest is not evidence its bytes are still there.
    static func bound(_ manifest: ModelManifest, under modelsDir: URL) -> Model? {
        guard let bound = manifest.bindUnder(modelsDir), let paths = bound.boundPaths,
              FileManager.default.fileExists(atPath: paths.payload),
              paths.mmproj.map({ FileManager.default.fileExists(atPath: $0) }) ?? true
        else { return nil }
        return bound
    }

    func payloadPath(for declared: Model) -> String? { bound(declared)?.boundPaths?.payload }

    func mmprojPath(for declared: Model) -> String? { bound(declared)?.boundPaths?.mmproj }

    /// The crate's `read_manifest`. The token is dropped first, so an in-memory `declared`
    /// never carries a secret. An entry-relative `stored` is upgraded in place
    /// (``upgradedFromEntryRelative(_:key:at:)``) rather than refused.
    static func readManifest(at url: URL) throws(ModelStoreError) -> ModelManifest {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw .io(doing: "reading \(url.path)", source: error.localizedDescription)
        }
        var manifest: ModelManifest
        do {
            manifest = try Coding.decoder.decode(ModelManifest.self, from: data)
        } catch {
            // Older builds wrote records without `fetched_at` and `stored`.
            guard let recovered = legacyManifest(data) else {
                throw .parse(path: url.path, source: "\(error)")
            }
            manifest = recovered
        }
        manifest.declared = manifest.declared.withoutAuthToken
        guard manifest.manifestVersion == ModelManifest.manifestVersion else {
            throw .corrupt("unsupported manifest_version \(manifest.manifestVersion) "
                           + "(expected \(ModelManifest.manifestVersion))")
        }
        guard let key = ModelStorageKey.of(manifest.declared) else {
            throw .notStorable(manifest.declared.artifactName)
        }
        if manifest.stored != manifest.declared.toStored(base: key.blobsDir) {
            guard let upgraded = upgradedFromEntryRelative(manifest, key: key, at: url) else {
                throw .corrupt("stored form is not the declared model's local form")
            }
            manifest = upgraded
        }
        return manifest
    }

    /// Re-root an entry-relative `stored` (`blobs/w.gguf`) onto `models/`
    /// (`<key>/blobs/w.gguf`) and rewrite the file. A version bump would not have helped:
    /// `find` errors on an unsupported version too, and the bytes on disk would never
    /// re-download. Only paths at or under `blobs` qualify; anything else stays corrupt.
    ///
    /// Not re-derived from `declared`: an MLX subtree recorded `blobs` without its
    /// `prefix`, so the derived form would name a directory that never held its files.
    private static func upgradedFromEntryRelative(
        _ manifest: ModelManifest, key: ModelStorageKey, at url: URL
    ) -> ModelManifest? {
        let blobs = Entry.blobsDirName
        let paths = manifest.stored.storedRelativePaths
        guard !paths.isEmpty,
              paths.allSatisfy({ $0 == blobs || $0.hasPrefix("\(blobs)/") }),
              let rerooted = manifest.stored.prefixingStoredPaths(with: key.value)
        else { return nil }
        var upgraded = manifest
        upgraded.stored = rerooted
        upgraded.writeQuietly(to: url)
        AppLog.storage.info(
            "model store: re-rooted \(key.value)'s stored paths onto the models root")
        return upgraded
    }

    /// An older version-2 record with optional `fetched_at` and `stored`. `fetchedAt`
    /// falls back to `lastUsedAt` (seeded from the same publish); a missing `stored` is
    /// derived, as such records were always read. Nil lets the caller report the real
    /// decode failure.
    private static func legacyManifest(_ data: Data) -> ModelManifest? {
        struct Legacy: Decodable {
            let manifestVersion: Int
            let fetchedAt: Date?
            let lastUsedAt: Date
            let declared: Model
            let stored: Model?
            let blobsBytes: Int64?

            enum CodingKeys: String, CodingKey {
                case manifestVersion = "manifest_version"
                case fetchedAt = "fetched_at"
                case lastUsedAt = "last_used_at"
                case declared, stored
                case blobsBytes = "blobs_bytes"
            }
        }
        guard let legacy = try? Coding.decoder.decode(Legacy.self, from: data) else { return nil }
        guard let stored = legacy.stored
            ?? legacy.declared.withoutAuthToken.toStored(base: Entry.blobsDirName)
        else { return nil }
        return ModelManifest(
            fetchedAt: legacy.fetchedAt ?? legacy.lastUsedAt,
            lastUsedAt: legacy.lastUsedAt,
            declared: legacy.declared,
            stored: stored,
            blobsBytes: legacy.blobsBytes)
    }

    /// The crate's `touch`. Best-effort: a read-only store must still resolve models.
    private func touch(_ manifest: ModelManifest) {
        guard let key = ModelStorageKey.of(manifest.declared) else { return }
        var touched = manifest
        touched.lastUsedAt = Date()
        let entry = modelsDir.appendingPathComponent(key.relativeDir, isDirectory: true)
        do {
            try touched.encoded().write(to: ModelManifest.manifestURL(inEntryDir: entry), options: .atomic)
        } catch {
            AppLog.storage.debug("could not record last_used_at for \(key.value): \(error)")
        }
    }

    /// Candidate entries, recognized later only through their manifest. Dot-directories
    /// (`.staging`) are skipped, as the crate's `list` does.
    static func entryDirectories(in root: URL) -> [URL] {
        let children = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [])) ?? []
        return children.filter {
            !$0.lastPathComponent.hasPrefix(".")
                && ((try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false)
        }
    }
}
