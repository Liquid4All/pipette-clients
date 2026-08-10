import Foundation

/// Single source of truth for the downloaded-models list in the UI — the
/// models-domain counterpart of `JobStore`. Views render `models` directly
/// instead of keeping their own `@State` copies of
/// `Storage.availableModels()`, so downloads and deletions update every
/// screen immediately.
///
/// The store starts empty and is loaded by the root view's `onAppear`, which
/// also reloads it whenever `DownloadCoordinator` finishes a download.
@MainActor
@Observable
final class ModelStore {
    private(set) var models: [DiscoveredModel] = []
    private let storage: Storage

    init(storage: Storage) {
        self.storage = storage
    }

    /// Re-run the manifest-driven disk scan that `availableModels()` owns.
    func reload() {
        models = storage.availableModels()
    }

    /// Delete model entries from disk and refresh. A full reload rather than
    /// in-memory removal: the disk scan is the canonical view of the store.
    func delete(_ modelsToDelete: [DiscoveredModel]) {
        modelsToDelete.forEach(storage.deleteModel)
        reload()
    }
}
