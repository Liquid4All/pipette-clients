import Foundation

/// A downloadable catalog row: the model display `name`, the `Model` coordinate it
/// downloads, and the display metadata the coordinate lacks.
nonisolated struct CatalogEntry: Identifiable, Sendable {
    let name: String
    let source: Model
    /// Total download size in bytes (SI): the GGUF weight file, the MLX repo's tree, or the
    /// ONNX package's subtree. 0 only if unknown.
    let sizeBytes: Int64
    /// The authored quant label, stored rather than parsed so a row shows it even when the
    /// repo slug does not encode it. `Model.quant` derives it for installed models.
    let quant: String?
    /// The group of models that the Add Models list collapses into one section, such as
    /// `LFM2.5`. The catalog gives it; an entry without one takes ``defaultFamily(_:)``.
    let family: String
    /// The SHA-256 digest that names the family's logo, or nil when the catalog gives none.
    let familyLogoDigest: String?
    let storageKey: ModelStorageKey?
    private let selector: String?

    init(name: String, source: Model, sizeBytes: Int64, quant: String?, family: String? = nil,
         familyLogoDigest: String? = nil, selector: String? = nil) {
        self.name = name
        self.source = source
        self.sizeBytes = sizeBytes
        self.quant = quant
        // A blank family from the server counts as none.
        let given = family?.trimmingCharacters(in: .whitespaces)
        self.family = given.flatMap { $0.isEmpty ? nil : $0 } ?? Self.defaultFamily(name)
        self.familyLogoDigest = familyLogoDigest
        self.storageKey = ModelStorageKey.of(source)
        self.selector = selector
    }

    /// The family of an entry with no family from the server: the text of `name` before
    /// its first hyphen. A name with no hyphen, or with nothing before it, is its own family.
    static func defaultFamily(_ name: String) -> String {
        let prefix = name.prefix { $0 != "-" }
        return prefix.isEmpty ? name : String(prefix)
    }

    /// SwiftUI identifies offers separately even when they reference the same download.
    var offerID: String { selector ?? id }

    /// Stable download identity.
    ///
    /// The store key leads because it separates format, repo, pinned revision and prefix.
    /// Repo and quant alone cannot: one ONNX repo carries several packages whose prefixes
    /// can share a leaf. A row with no store key falls back to the repo plus the first of
    /// quant, weight filename or name.
    var id: String {
        if let storageKey { return storageKey.value }
        let fileFallback: String? = switch source {
        case let .ggufText(m): m.source.reference.split(separator: "/").last.map(String.init)
        case let .ggufVision(m): m.source.weightsReference.split(separator: "/").last.map(String.init)
        case let .ggufVisionSpeculative(m):
            m.source.vision.weightsReference.split(separator: "/").last.map(String.init)
        case .mlx, .onnxGenai, .appleFoundationText: nil
        }
        return "\(repoIdentifier)#\(quant ?? fileFallback ?? name)"
    }

    /// The Hugging Face repo slug, or empty for a Vault offer.
    var repoIdentifier: String { source.repo?.description ?? "" }

    var modelId: String { source.modelId }
}

// MARK: - Catalog data

// Read from the main actor and from nonisolated contexts, so the statics are marked
// `nonisolated`: extension members do not inherit the struct's `nonisolated`.
extension CatalogEntry {
    nonisolated static var catalog: [CatalogEntry] { ModelShortList.current(ws: .production) }

    nonisolated static func selectedDownloads(
        _ rows: [CatalogEntry], selected: Set<ModelStorageKey>, unavailable: Set<ModelStorageKey>
    ) -> [CatalogEntry] {
        var downloads: [CatalogEntry] = []
        var indices: [ModelStorageKey: Int] = [:]
        for row in rows {
            guard let key = row.storageKey, selected.contains(key), !unavailable.contains(key)
            else { continue }
            if let index = indices[key] {
                if row.sizeBytes > downloads[index].sizeBytes {
                    downloads[index] = row
                }
            } else {
                indices[key] = downloads.count
                downloads.append(row)
            }
        }
        return downloads
    }

    nonisolated enum DownloadWarning: Equatable {
        case unknownSize
        case largeDownload
    }

    nonisolated static func downloadWarning(
        for rows: [CatalogEntry], largeDownloadThreshold: Int64
    ) -> DownloadWarning? {
        if rows.contains(where: { $0.sizeBytes <= 0 }) { return .unknownSize }
        var remaining = largeDownloadThreshold
        for row in rows {
            if row.sizeBytes > remaining { return .largeDownload }
            remaining -= row.sizeBytes
        }
        return nil
    }

    /// Keeps unknown sizes and compares known download sizes with the storage quota and
    /// physical memory. Current free space can change when a model is deleted.
    nonisolated static func fitting(_ rows: [CatalogEntry], quotaBytes: Int64,
                                    ramBytes: UInt64 = ProcessInfo.processInfo.physicalMemory)
        -> [CatalogEntry] {
        rows.filter { sizeFits($0.sizeBytes, quotaBytes: quotaBytes, ramBytes: ramBytes) }
    }

    nonisolated static func sizeFits(_ bytes: Int64, quotaBytes: Int64,
                                     ramBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> Bool {
        bytes <= 0 || (bytes <= quotaBytes && UInt64(bytes) <= ramBytes)
    }

    nonisolated static var repoToName: [String: String] {
        var dict: [String: String] = [:]
        for entry in catalog where !entry.repoIdentifier.isEmpty {
            dict[entry.repoIdentifier] = entry.name
        }
        return dict
    }

    /// Keyed on `modelId` because a model's quants may span repos and formats; the first
    /// row for a model wins.
    nonisolated static var modelIdToName: [String: String] {
        modelNames(in: catalog)
    }

    nonisolated static func modelNames(in entries: [CatalogEntry]) -> [String: String] {
        firstEntries(in: entries).mapValues(\.name)
    }

    /// The first catalog row of each model, keyed on `modelId`. Each read rebuilds it from
    /// the current catalog.
    nonisolated static var firstEntryByModelId: [String: CatalogEntry] {
        firstEntries(in: catalog)
    }

    nonisolated static func firstEntries(in entries: [CatalogEntry]) -> [String: CatalogEntry] {
        var dict: [String: CatalogEntry] = [:]
        for entry in entries where dict[entry.modelId] == nil { dict[entry.modelId] = entry }
        return dict
    }

    /// Keyed by lowercase family name. The last available digest in catalog order wins.
    static func familyLogoDigests(in entries: [CatalogEntry]) -> [String: String] {
        var digests: [String: String] = [:]
        for entry in entries {
            if let digest = entry.familyLogoDigest {
                digests[entry.family.lowercased()] = digest
            }
        }
        return digests
    }

    /// The family of a model: the family of its catalog row, else the name rule.
    nonisolated static func family(of entry: CatalogEntry?, name: String) -> String {
        entry?.family ?? defaultFamily(name)
    }

    // MARK: Builders
    //
    // Inputs are literals checked by `CatalogEntryTests`, so the newtypes use their
    // unchecked initializers: a malformed literal fails a test, not a runtime throw.

    /// The on-disk filename is `<repo-name minus -GGUF>-<quant>.gguf`, the layout these
    /// repos publish.
    nonisolated static func gguf(_ name: String, _ repo: String, _ quant: String, bytes: Int64,
                                 family: String? = nil) -> CatalogEntry {
        let hf = trustedRepo(repo)
        let filename = RepoSubpath(validated: "\(LocalStorage.repoStem(hf.repoName.value))-\(quant).gguf")
        return CatalogEntry(name: name, source: .ggufText(GgufText(source: .huggingFace(repo: hf, path: filename, sha256: nil))),
                            sizeBytes: bytes, quant: quant, family: family)
    }

    /// `quant` defaults to `4bit`, the common export; pass any other bit-width explicitly.
    nonisolated static func mlx(_ name: String, _ repo: String, bytes: Int64,
                                quant: String = "4bit", family: String? = nil) -> CatalogEntry {
        CatalogEntry(name: name,
                     source: .mlx(Mlx(source: .huggingFace(repo: trustedRepo(repo),
                                                           prefix: nil))),
                     sizeBytes: bytes, quant: quant, family: family)
    }

    nonisolated private static func trustedRepo(_ slug: String) -> HfRepo {
        let parts = slug.split(separator: "/", maxSplits: 1)
        let org = parts.first.map(String.init) ?? slug
        let name = parts.count > 1 ? String(parts[1]) : ""
        return HfRepo(org: HfOrg(validated: org), repoName: HfRepoName(validated: name))
    }

    // MARK: - Fallback

    /// The app uses these LFM models when the current short list has no models.
    /// The sizes come from the Hugging Face file tree and cover each GGUF file or MLX repository.
    /// The names have no hyphen, so each row names its family.
    nonisolated static let fallback: [CatalogEntry] = [
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_0", bytes: 149_080_928, family: "LFM2.5"),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_K_M", bytes: 153_406_304, family: "LFM2.5"),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q5_K_M", bytes: 171_625_312, family: "LFM2.5"),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_0", bytes: 219_309_792, family: "LFM2.5"),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_K_M", bytes: 229_312_224, family: "LFM2.5"),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q5_K_M", bytes: 260_376_288, family: "LFM2.5"),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_0", bytes: 695_751_488, family: "LFM2.5"),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_K_M", bytes: 730_895_168, family: "LFM2.5"),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q5_K_M", bytes: 843_354_944, family: "LFM2.5"),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_0", bytes: 4_844_678_368, family: "LFM2.5"),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_K_M", bytes: 5_155_564_768, family: "LFM2.5"),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q5_K_M", bytes: 6_030_339_296, family: "LFM2.5"),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q4_0", bytes: 1_593_894_912, family: "LFM2.5"),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q4_K_M", bytes: 1_674_455_040, family: "LFM2.5"),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q5_K_M", bytes: 1_939_744_768, family: "LFM2.5"),
        .mlx("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-MLX-4bit", bytes: 150_867_598, family: "LFM2.5"),
        .mlx("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-MLX-4bit", bytes: 226_574_864, family: "LFM2.5"),
        .mlx("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit", bytes: 663_548_128, family: "LFM2.5"),
        .mlx("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", bytes: 4_851_998_389, family: "LFM2.5"),
        .mlx("LFM2 700M", "mlx-community/LFM2-700M-4bit", bytes: 422_679_301, family: "LFM2"),
        .mlx("LFM2 2.6B", "mlx-community/LFM2-2.6B-4bit", bytes: 1_450_530_492, family: "LFM2"),
        .mlx("LFM2 2.6B Exp", "mlx-community/LFM2-2.6B-Exp-4bit", bytes: 1_450_530_508, family: "LFM2"),
    ]
}
