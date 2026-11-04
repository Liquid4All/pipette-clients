import Foundation

nonisolated extension PipetteWorkspace {
    @discardableResult
    private static func ensureModelDirectory(_ url: URL) -> URL {
        let dir = ensureDirectory(url)
        LocalStorage.markExcludedFromBackup(dir)
        return dir
    }

    /// One flat entry directory per model.
    var modelsDir: URL {
        Self.ensureModelDirectory(root.appendingPathComponent("models", isDirectory: true))
    }

    /// Stable base for directory (HubApi) pulls, in Caches, so an interrupted pull keeps
    /// the files it finished. `sweepToQuota` reclaims what publishing leaves behind.
    var hubCacheDir: URL {
        let url = cacheRoot.appendingPathComponent("Pipette/hub", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// swift-huggingface's shared `HubCache` in this sandbox. Nothing reads it; the sweep
    /// reclaims blobs older builds left. Named, not created.
    var sharedHfCacheDir: URL {
        cacheRoot.appendingPathComponent("huggingface/hub", isDirectory: true)
    }

    /// GGUF transfer records and resume data, in Caches. Named, not created:
    /// `DownloadStore` creates it on each write, since Caches can be evicted at any time.
    var downloadsDir: URL {
        cacheRoot.appendingPathComponent("Pipette/downloads", isDirectory: true)
    }

    /// Resolve a persisted absolute path, which goes stale when the data-container UUID
    /// changes on reinstall, by falling back to the current `modelsDir`. Nil passes through
    /// for optional mmproj paths.
    func resolveModelPath(_ storedPath: String?) -> String? {
        guard let storedPath else { return nil }
        if FileManager.default.fileExists(atPath: storedPath) { return storedPath }
        return LocalStorage.resolveModelPath(storedPath, modelsRoot: modelsDir)
    }

    /// Everything on disk, plus Apple's built-in model when this device supports it. Each
    /// entry's manifest records its `source`, so the scan alone restores full provenance.
    @MainActor
    func availableModels() -> [DiscoveredModel] {
        let onDiskModels = scanAvailableModelFiles()
        guard AFMRuntime.isAvailable else { return onDiskModels }
        return onDiskModels + [DiscoveredModel.appleFoundation]
    }

    /// Each `modelsDir` child is one entry, recognized only through its manifest. Tolerant
    /// where the store's `list` is strict: one unusable manifest must not hide every other
    /// model, and the sweep reclaims it.
    @MainActor
    private func scanAvailableModelFiles() -> [DiscoveredModel] {
        let root = modelsDir
        var models: [DiscoveredModel] = []
        for entry in ModelArtifactStore.entryDirectories(in: root) {
            guard let manifest = try? ModelArtifactStore.readManifest(
                at: ModelManifest.manifestURL(inEntryDir: entry)) else {
                AppLog.storage.debug("Skipping model entry without a usable manifest: \(entry.lastPathComponent)")
                continue
            }
            // One lookup per entry, so a vision model's projector cannot be mispaired.
            guard let paths = ModelArtifactStore.bound(manifest, under: root)?.boundPaths else {
                AppLog.storage.debug("Skipping model entry with no payload: \(entry.lastPathComponent)")
                continue
            }
            LocalStorage.markExcludedFromBackup(entry)
            models.append(DiscoveredModel(
                source: manifest.declared, path: paths.payload, sizeBytes: entrySizeBytes(at: entry)))
        }
        // Path breaks ties so the order is total.
        return models.sorted { ($0.name, $0.path) < ($1.name, $1.path) }
    }

    @MainActor
    func deleteModel(_ model: DiscoveredModel) {
        // AFM has no file to delete.
        if case .appleFoundationText = model.source { return }
        models.remove(model.source)
    }
}
