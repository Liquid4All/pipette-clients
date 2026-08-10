import Foundation

extension FileStorage {
    @discardableResult
    private static func ensureModelDirectory(_ url: URL) -> URL {
        let dir = ensureDirectory(url)
        LocalStorage.markExcludedFromBackup(dir)
        return dir
    }

    /// Root of the model store: one flat entry directory per model.
    var modelsDir: URL {
        Self.ensureModelDirectory(dataRoot.appendingPathComponent("models", isDirectory: true))
    }

    /// Best-effort `last_used_at` refresh, keyed on the spec so a vision model's two
    /// files touch one entry. Silent on every failure: the eviction order is
    /// bookkeeping and must never fail a resolve.
    func touchModelLastUsed(_ spec: Model) {
        guard let dir = modelStore.entryDir(for: spec) else { return }
        ModelManifest.touchLastUsed(inEntryDir: dir)
    }

    /// Stable base for MLX (HubApi) downloads. Re-downloadable content lives in
    /// Caches per iOS convention; kept put so an interrupted MLX pull resumes from
    /// its cache instead of restarting. The finished model is relocated into its
    /// store entry by `MLXModelInstaller`, and what the relocation leaves behind is
    /// reclaimed by `sweepToQuota`.
    var hubCacheDir: URL {
        let url = cacheRoot.appendingPathComponent("Pipette/hub", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Resolve a persisted model path to a currently-valid absolute path.
    ///
    /// Manifests store absolute paths at the time of job creation, but those
    /// paths go stale in two ways: (1) the app's data-container UUID changes
    /// on reinstall, and (2) storage locations themselves have moved
    /// (Documents → Application Support). When the stored path exists we use
    /// it as-is; otherwise we fall back to looking up the filename in the
    /// current `modelsDir`. Returns nil if neither location has the file.
    /// Nil input passes through so callers can use it for optional mmproj paths.
    func resolveModelPath(_ storedPath: String?) -> String? {
        guard let storedPath else { return nil }
        if FileManager.default.fileExists(atPath: storedPath) { return storedPath }
        // The container UUID (or storage location) changed since the manifest
        // was written. Recover the path relative to the current modelsDir.
        return LocalStorage.resolveModelPath(storedPath, modelsRoot: modelsDir)
    }

    /// The models available to run: everything discovered on disk plus Apple's
    /// built-in system model when this device supports it.
    ///
    /// A downloaded model is self-describing — the manifest at its entry root records
    /// the typed `source`, and every display field (name, displayName, familyId,
    /// quant) derives from that spec. So the disk scan alone re-establishes full
    /// provenance on every call; there is no separate persisted copy to keep in sync.
    /// AFM has no file, so it's appended live from an availability check rather than
    /// discovered.
    @MainActor
    func availableModels() -> [DiscoveredModel] {
        let onDiskModels = scanAvailableModelFiles()
        guard AFMRuntime.isAvailable else { return onDiskModels }
        return onDiskModels + [DiscoveredModel.appleFoundation]
    }

    /// One level of `modelsDir`: each child directory is one entry, recognized only
    /// through its manifest. Nothing is sniffed — the manifest's spec names both the
    /// format and the payload to look for.
    @MainActor
    private func scanAvailableModelFiles() -> [DiscoveredModel] {
        var models: [DiscoveredModel] = []
        for entry in ModelArtifactStore.entryDirectories(in: modelsDir) {
            guard let source = ModelManifest.forInstalledEntry(atDir: entry)?.declared else {
                AppLog.storage.debug("Skipping manifest-less model entry: \(entry.lastPathComponent)")
                continue
            }
            // One lookup for the whole entry: a vision model's two files come back
            // together, so discovery cannot pair the wrong projector with the weights.
            guard let paths = ModelArtifactStore.bound(inEntryDir: entry, declared: source)?.boundPaths else {
                AppLog.storage.debug("Skipping model entry with no payload: \(entry.lastPathComponent)")
                continue
            }
            LocalStorage.markExcludedFromBackup(entry)
            models.append(DiscoveredModel(
                source: source, path: paths.payload, sizeBytes: DiskUsage.bytes(at: entry)))
        }
        // Sort by (name, path) so the order is total even for two entries that share a
        // name.
        return models.sorted { ($0.name, $0.path) < ($1.name, $1.path) }
    }

    /// Delete a downloaded model — the store's `remove`.
    @MainActor
    func deleteModel(_ model: DiscoveredModel) {
        // AFM is a built-in system model with no file — there is nothing to delete, so
        // deleting it is a no-op (the UI also hides the delete affordance for it).
        if case .appleFoundationText = model.source { return }
        modelStore.remove(model.source)
    }
}
