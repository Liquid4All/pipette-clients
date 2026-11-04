import Foundation

/// One root that mints a store per subdirectory (`crates/pipette-cli/src/workspace.rs`).
/// A value rather than a global, so a test builds one over a temporary root.
///
/// Under `root`:
///   identity/         signing key, registration, device labels, client settings
///   benchmarks/       generated `local/` and synced `remote/` catalog halves
///   results/          one directory per result, under local/ or remote/{pending,synced}
///   state/evals/      eval checkpoints, so a killed run resumes
///   jobs/<jobId>/     manifest plus per-cell payload and submission artifacts
///   models/<key>/     manifest plus a `blobs/` payload; backup-excluded, quota-capped
///   corpora/          a cell's dataset, deleted after the cell
nonisolated struct PipetteWorkspace: Sendable, RunContext {
    let root: URL
    /// Re-downloadable content, e.g. the MLX hub cache.
    let cacheRoot: URL
    /// Where `registration`'s signing key outlives an uninstall. Off unless named, so a
    /// workspace built for a test or a preview never reaches the device-global Keychain.
    var privateKeyMirror: PrivateKeyMirror = .disabled
    /// The registration record's copy, off for the same reason.
    var registrationMirror: RegistrationMirror = .disabled

    var benchmarks: BenchmarkStore {
        BenchmarkStore(root: root.appendingPathComponent("benchmarks", isDirectory: true))
    }

    var models: ModelArtifactStore { ModelArtifactStore(modelsDir: modelsDir) }

    var registration: ClientRegistrationStore {
        ClientRegistrationStore(root: identityDir, privateKeyMirror: privateKeyMirror,
                                registrationMirror: registrationMirror)
    }

    /// A different lifetime from the registration, so a different store.
    var device: ClientDeviceStore { ClientDeviceStore(root: identityDir) }

    var settings: ClientSettingsStore { ClientSettingsStore(root: identityDir) }

    private var identityDir: URL { root.appendingPathComponent("identity", isDirectory: true) }

    /// All three stores' files, for `auth reset`. Returns whether the key went.
    @discardableResult
    func clearIdentity() -> Bool {
        let keyCleared = registration.clear()
        try? device.clear()
        try? settings.clear()
        return keyCleared
    }

    /// Results live beside the jobs that produced them rather than inside them: the
    /// location, not the job, is what tells you how far a result has travelled.
    var results: ResultsStore {
        ResultsStore(root: root.appendingPathComponent("results", isDirectory: true))
    }

    // MARK: - iOS: roots, production, migrations

    private static let currentStorageDirectoryName = "Pipette"
    private static let legacyStorageDirectoryName = "EdgeEvals"

    /// `Application Support/Pipette` and the system caches; the one workspace that reads the
    /// Keychain.
    static let production = PipetteWorkspace(
        root: applicationSupportRoot.appendingPathComponent(currentStorageDirectoryName, isDirectory: true),
        cacheRoot: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0],
        privateKeyMirror: .keychain,
        registrationMirror: .keychain
    )

    static func ensureDirectory(_ url: URL) -> URL {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static var applicationSupportRoot: URL {
        // Always present in `.userDomainMask` on iOS, so `.first!` can't trap.
        FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
    }

    private var legacyDataRoot: URL? {
        let legacyRoot = root
            .deletingLastPathComponent()
            .appendingPathComponent(Self.legacyStorageDirectoryName, isDirectory: true)
        guard legacyRoot.standardizedFileURL != root.standardizedFileURL else {
            return nil
        }
        return legacyRoot
    }

    private var legacyCachesRoot: URL? {
        let legacyRoot = cacheRoot.appendingPathComponent(Self.legacyStorageDirectoryName, isDirectory: true)
        guard legacyRoot.standardizedFileURL != root.standardizedFileURL else {
            return nil
        }
        return legacyRoot
    }

    @discardableResult
    func cleanupLegacyEdgeEvalsStorage() -> [URL] {
        let fm = FileManager.default
        let roots = [legacyDataRoot, legacyCachesRoot].compactMap { $0 }
        var removed: [URL] = []

        for root in roots where fm.fileExists(atPath: root.path) {
            do {
                try fm.removeItem(at: root)
                removed.append(root)
            } catch {
                AppLog.storage.error("Failed to remove legacy storage at \(root.path): \(error.localizedDescription)")
            }
        }

        return removed
    }

    /// Where the identity files lived before `identity/`. Only `migrateIdentityDirectory()`
    /// may read it.
    var legacyMetadataDir: URL {
        root.appendingPathComponent("metadata", isDirectory: true)
    }

    /// Idempotent: a file already at the destination wins, and the old directory goes once
    /// it is empty. A move rather than reading the old path forever, so there is one
    /// location. Losing the registration here would push a registered device back through
    /// setup with a fresh keypair, orphaning its submitted results.
    @discardableResult
    func migrateIdentityDirectory() -> [String] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: legacyMetadataDir.path) else { return [] }
        let destination = Self.ensureDirectory(identityDir)

        var moved: [String] = []
        for name in ["registration.json", "settings.json"] {
            let from = legacyMetadataDir.appendingPathComponent(name)
            let to = destination.appendingPathComponent(name)
            guard fm.fileExists(atPath: from.path) else { continue }
            guard !fm.fileExists(atPath: to.path) else {
                try? fm.removeItem(at: from)
                continue
            }
            do {
                try fm.moveItem(at: from, to: to)
                moved.append(name)
            } catch {
                AppLog.storage.error("identity migration: \(name) did not move: \(error)")
            }
        }
        // Only when empty: an unexpected file there is someone else's.
        if let rest = try? fm.contentsOfDirectory(atPath: legacyMetadataDir.path), rest.isEmpty {
            try? fm.removeItem(at: legacyMetadataDir)
        }
        if !moved.isEmpty {
            AppLog.storage.info("identity migration: moved \(moved.joined(separator: ", "))")
        }
        return moved
    }

    /// Every identity writer excludes the directory as it creates it; this covers devices
    /// that only read (a record with no key writes nothing) and the directory
    /// ``migrateIdentityDirectory()`` creates without the flag. The record is the half a
    /// restore must not carry on its own, so the flag cannot wait for a write.
    func excludeIdentityFromBackup() {
        guard FileManager.default.fileExists(atPath: identityDir.path) else { return }
        if !LocalStorage.markExcludedFromBackup(identityDir) {
            AppLog.storage.error("identity: the existing directory is not excluded from backups")
        }
    }

    var jobsDir: URL {
        Self.ensureDirectory(root.appendingPathComponent("jobs", isDirectory: true))
    }

    @MainActor
    func resetDeviceData() throws {
        let fm = FileManager.default
        let removableDirectories = [
            root.appendingPathComponent("jobs", isDirectory: true),
            root.appendingPathComponent("results", isDirectory: true),
            root.appendingPathComponent("models", isDirectory: true),
            root.appendingPathComponent("benchmarks", isDirectory: true),
            legacyDataRoot,
            legacyCachesRoot
        ].compactMap { $0 }

        for directory in removableDirectories where fm.fileExists(atPath: directory.path) {
            try fm.removeItem(at: directory)
        }
    }
}

/// The crate's `impl RunContext for PipetteWorkspace`.
nonisolated extension PipetteWorkspace {
    /// Signed as this device, on the route the management server stamped onto the benchmark.
    nonisolated func fetchDataset(_ dataset: Dataset) async throws -> Data {
        guard let path = dataset.url?.value else { throw DatasetError.noDownloadUrl }
        let session = try registration.mgmtSession()
        AppLog.corpus.info("fetching a corpus: \(dataset.bytes) bytes from \(path)")
        return try await PipetteMgmtClient.ClientApi.getDataset(
            server: session.serverUrl, http: .shared, auth: session.auth, path: path)
    }

    /// Outside the model store: a corpus lives for one cell, so the quota sweep must not see it.
    nonisolated var datasetsDir: URL {
        root.appendingPathComponent("corpora", isDirectory: true)
    }

    /// Under `state/`, where the crate keeps it too.
    nonisolated var evalCompletions: EvalCompletionsStore {
        EvalCompletionsStore(
            root: root.appendingPathComponent("state/evals", isDirectory: true))
    }
}
