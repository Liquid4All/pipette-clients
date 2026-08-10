import Foundation

/// Installs a freshly-downloaded MLX model: validate it's complete, relocate it into
/// its store entry, record provenance. Extracted from `DownloadCoordinator` so the
/// validate → relocate → record pipeline is testable without any download.
@MainActor
struct MLXModelInstaller: ModelInstaller {
    let ref: HFModelRef
    let storage: Storage

    /// Move `downloaded` into the entry's `blobs/` payload and persist provenance.
    /// Throws `.incompleteModel` — and drops the bad download so a retry starts clean —
    /// if it isn't a loadable MLX model; `.io` on a filesystem error.
    func install(from downloaded: URL) throws {
        do {
            try MLXModelLayout.validate(downloaded, ref: ref)
        } catch {
            // A partial/empty download would otherwise install an unusable model;
            // drop it so the next attempt re-fetches rather than resuming garbage.
            try? FileManager.default.removeItem(at: downloaded)
            throw error
        }
        let spec = Model.mlx(ref.asMlx())
        // Creating the entry is the side effect wanted here; the manifest goes in
        // through the store at publish.
        guard storage.modelStore.prepareEntryDir(for: spec) != nil,
              let dest = storage.modelStore.blobsDir(for: spec)
        else { throw DownloadError.io("no model coordinate to store \(ref) under") }

        let fm = FileManager.default
        do {
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: downloaded, to: dest)
        } catch {
            throw DownloadError.io("\(error)")
        }
        // The manifest at the entry root is the sole provenance record; the next disk
        // scan rebuilds the model from it.
        storage.modelStore.publishManifest(for: spec)
    }
}
