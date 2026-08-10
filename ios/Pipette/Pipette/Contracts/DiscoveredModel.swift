import Foundation

/// A model the app has discovered — either found on disk or the built-in Apple
/// Foundation model — as the UI/discovery layer sees it. Identity, display name, family
/// grouping, quant, and engine label are all *derived* from the spec (and the catalog),
/// never cached on the type.
///
/// A row may be **downloadable rather than downloaded**: the run screen offers the whole
/// catalog so a benchmark can be started against a model that is not on disk yet, and
/// `RunCell.prepare` fetches it when the cell runs. `isDownloaded` is the distinction, and
/// `path` is empty until it is true.
///
/// The spec is the **declared** coordinate, not a bound one: discovery is a browsing
/// surface, and nothing here loads a model. A cell that will run one hands its declared
/// spec to `RunCell.prepare`, which binds it and carries the pair as
/// `DeclaredBound<Model>`. AFM is the bare `.appleFoundationText` spec with no footprint,
/// so there is no faked path and no "reconcile later" mutable field.
nonisolated struct DiscoveredModel: Identifiable, Sendable, Hashable {
    /// The typed spec (identity/provenance) this was discovered as.
    let source: Model

    /// On-disk location: the GGUF weight file, or the MLX directory. `""` for AFM,
    /// which ships with the OS and has no file.
    let path: String

    /// On-disk size in bytes; `0` for AFM.
    let sizeBytes: Int64

    init(source: Model, path: String, sizeBytes: Int64) {
        self.source = source
        self.path = path
        self.sizeBytes = sizeBytes
    }

    /// Apple's built-in system model as a discovered row. It has no file on disk, so it
    /// carries no footprint; the derived `familyId` (`"apple-foundation"`) and `hfRepo`
    /// (`AFMRuntime.submissionModelName`) match the headless path so grouping and
    /// records resolve the same.
    static let appleFoundation = DiscoveredModel(
        source: .appleFoundationText, path: "", sizeBytes: 0)

    var sizeFormatted: String { ByteFormat.fileSize(sizeBytes) }

    /// Whether the weights are on disk. False for a catalog row the run screen is
    /// offering — starting a benchmark against one downloads it first.
    var isDownloaded: Bool {
        if case .appleFoundationText = source { return true }   // ships with the OS
        return !path.isEmpty
    }

    /// SwiftUI list identity. The on-disk path is unique per download, but a
    /// not-yet-downloaded row has none — and every such row would otherwise collide on
    /// `""` — so identity falls back to the coordinate, unique per model and quant.
    var id: String {
        if case .appleFoundationText = source { return "apple-foundation" }
        return path.isEmpty ? source.reference : path
    }

    /// A catalog row the app could fetch. `sizeBytes` is the catalog's declared download
    /// size, shown in place of an on-disk footprint.
    init(catalog entry: CatalogEntry) {
        self.init(source: entry.source, path: "", sizeBytes: entry.sizeBytes)
    }

    /// The artifact leaf — the GGUF filename or the MLX bundle leaf, taken from the
    /// spec. AFM has no file, so it reports its fixed display label.
    var name: String { source.artifactName }

    /// The publishing repo as a slug — the spec's own derivation, so a discovery row and
    /// the cell built from it name the model identically.
    var hfRepo: String { source.repoSlug }

    /// Authored display name from the catalog (repo → name). AFM has no repo, so it
    /// reports its engine label ("Apple Foundation"); models absent from the catalog
    /// (sideloads) return nil and callers fall back to the filename stem.
    var displayName: String? {
        if case .appleFoundationText = source { return source.engineLabel }
        return CatalogEntry.repoToName[hfRepo]
    }

    /// Family identity shared across a model's quants *and* formats — the spec's own
    /// derivation, so discovery, the catalog, and job cells all agree. Total (never
    /// nil): what used to be a mutable "nil until reconciled" field is now computed.
    var familyId: String { source.familyId }

    /// Quant label, derived from the typed spec (the only place an MLX bit-width is
    /// recoverable).
    var quant: String? { source.quant }

    /// Human-facing engine name (GGUF → llama.cpp, MLX → MLX, AFM → Apple Foundation).
    var engineLabel: String { source.engineLabel }
}
