import Foundation

/// A model as the UI lists it: on disk, a catalog row not yet downloaded, or the built-in
/// Apple Foundation model. Display fields derive from `source` and the catalog.
///
/// `source` is the declared coordinate, never a bound one: nothing here loads a model.
/// `RunCell.prepare` fetches and binds it when a cell runs.
nonisolated struct DiscoveredModel: Identifiable, Sendable, Hashable {
    let source: Model

    /// The GGUF weight file or the directory model's payload; `""` for AFM and for a row
    /// not yet downloaded.
    let path: String

    /// On-disk size, or the catalog's declared size for a row not yet downloaded; `0` for
    /// AFM.
    let sizeBytes: Int64
    private let catalogQuant: String?

    init(source: Model, path: String, sizeBytes: Int64, catalogQuant: String? = nil) {
        self.source = source
        self.path = path
        self.sizeBytes = sizeBytes
        self.catalogQuant = catalogQuant
    }

    static let appleFoundation = DiscoveredModel(
        source: .appleFoundationText, path: "", sizeBytes: 0)

    var sizeFormatted: String { ByteFormat.fileSize(sizeBytes) }

    var isDownloaded: Bool {
        if case .appleFoundationText = source { return true }   // ships with the OS
        return !path.isEmpty
    }

    /// The path when there is one; a not-yet-downloaded row falls back to the coordinate,
    /// as those rows would all collide on `""`.
    var id: String {
        if case .appleFoundationText = source { return "apple-foundation" }
        return path.isEmpty ? source.description : path
    }

    init(catalog entry: CatalogEntry) {
        self.init(source: entry.source, path: "", sizeBytes: entry.sizeBytes,
                  catalogQuant: entry.quant)
    }

    /// This entry as a run that named `requested` carries it, so a report states the model
    /// that was asked for, vault label included. No token, as a manifest holds none.
    func asRequested(_ requested: Model) -> DiscoveredModel {
        DiscoveredModel(source: requested.withoutAuthToken, path: path, sizeBytes: sizeBytes)
    }

    var name: String { source.artifactName }

    var hfRepo: String { source.repoSlug }

    /// The catalog's model name, or nil for a model absent from the catalog (callers fall
    /// back to the filename stem). AFM reports its engine label.
    var displayName: String? {
        if case .appleFoundationText = source { return source.engineLabel }
        return CatalogEntry.repoToName[hfRepo]
    }

    var modelId: String { source.modelId }

    var quant: String? { source.quant ?? catalogQuant }

    var engineLabel: String { source.engineLabel }
}
