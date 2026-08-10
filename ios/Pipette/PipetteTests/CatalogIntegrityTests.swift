import Foundation
import Testing

@testable import Pipette

/// Guards the built-in `CatalogEntry` catalog. The builders construct validated
/// newtypes via their trusted (unchecked) initializers for compile-time constants,
/// so these tests stand in for the validation that construction skips: every repo,
/// filename, and quant must actually be well-formed, and the row counts must match
/// the authored families (a dropped or malformed literal fails here).
struct CatalogIntegrityTests {
    /// Every authored row, independent of `MLXFeatureFlag.visibleInUI` (which gates the
    /// UI-facing `CatalogEntry.catalog`). Data integrity is a property of the authored
    /// data, so these tests validate it against the arrays directly — MLX coverage holds
    /// whether or not the UI flag is on.
    private let authored = CatalogEntry.defaults + CatalogEntry.mlxDefaults

    /// GGUF families deliberately shipped without an MLX row.
    /// - `Nemotron 3 Nano 4B`: mlx-swift-lm registers `nemotron_h`, but nobody
    ///   publishes an MLX export of it — a packaging gap, so drop this entry once
    ///   one lands on the Hub.
    /// - `Ternary Bonsai 27B`: the MLX 2-bit export is 8.5 GB, past any device
    ///   budget; only the 7.2 GB GGUF is worth offering.
    private static let ggufOnlyFamilies: Set<String> = ["Nemotron 3 Nano 4B", "Ternary Bonsai 27B"]

    @Test func rowCountsMatchAuthoredFamilies() {
        // 21 GGUF families × 3 quants, plus the five Bonsai families, which each
        // publish a single on-device quant (`Q1_0` plain, `Q2_0` ternary).
        #expect(CatalogEntry.defaults.count == 68)
        #expect(CatalogEntry.mlxDefaults.count == 24)
        #expect(authored.count == CatalogEntry.defaults.count + CatalogEntry.mlxDefaults.count)
    }

    @Test func everyEntryHasAValidRepo() throws {
        for entry in authored {
            // Every catalog row is HF-backed, so its source always has a repo.
            let repo = try #require(entry.source.repo, "catalog entry \(entry.repoIdentifier) must have a repo")
            #expect(HFOrg.validate(repo.org.value), "invalid org in \(entry.repoIdentifier)")
            #expect(HFRepoName.validate(repo.repoName.value), "invalid repo name in \(entry.repoIdentifier)")
        }
    }

    @Test func ggufEntriesHaveParsableFilenameAndQuant() {
        for entry in CatalogEntry.defaults {
            // A catalog entry is authored, so it is always a HuggingFace arm — asserting
            // that is the point, not an assumption to reach through.
            guard case let .ggufText(m) = entry.source,
                  case let .huggingFace(_, path, _) = m.source else {
                Issue.record("\(entry.repoIdentifier) should be a HuggingFace ggufText")
                continue
            }
            #expect(path.value.hasSuffix(".gguf"))
            #expect(entry.quant != nil, "\(path.value) should parse a quant")
            #expect(entry.sizeBytes > 0)
        }
    }

    @Test func mlxEntriesAreBareReposWithBitWidthQuant() {
        for entry in CatalogEntry.mlxDefaults {
            guard case .mlx = entry.source else {
                Issue.record("\(entry.repoIdentifier) should be mlx")
                continue
            }
            // The bit-width must be authored, and the repo slug must end in it — that
            // suffix is what `Model.familyId` strips to rejoin an MLX row with its GGUF
            // sibling, so a mismatch silently splits the family in the UI.
            let quant = entry.quant ?? ""
            #expect(quant.hasSuffix("bit"), "\(entry.repoIdentifier) quant=\(entry.quant ?? "nil")")
            #expect(entry.repoIdentifier.lowercased().hasSuffix("-" + quant.lowercased()),
                    "\(entry.repoIdentifier) should end in -\(quant)")
            #expect(!entry.repoIdentifier.contains(":"))
            #expect(entry.sizeBytes > 0, "\(entry.repoIdentifier) should carry a fetched on-disk size")
        }
    }

    /// Every GGUF family has a matching MLX build (same display name) — guards
    /// against a model shipping GGUF-only when an MLX export exists. Families in
    /// `ggufOnlyFamilies` are the audited exceptions.
    @Test func everyGgufFamilyHasAnMlxCounterpart() {
        let ggufNames = Set(CatalogEntry.defaults.map(\.name))
        let mlxNames = Set(CatalogEntry.mlxDefaults.map(\.name))
        let missing = ggufNames.subtracting(mlxNames)
        #expect(missing.subtracting(Self.ggufOnlyFamilies).isEmpty,
                "GGUF families missing MLX: \(missing.subtracting(Self.ggufOnlyFamilies).sorted())")
        // Keep the exception list honest: an entry that no longer applies must go.
        #expect(Self.ggufOnlyFamilies.subtracting(missing).isEmpty,
                "stale ggufOnlyFamilies: \(Self.ggufOnlyFamilies.subtracting(missing).sorted())")
    }

    /// A model's MLX build and its GGUF build must resolve to the *same* `familyId`,
    /// so the downloaded list and job wizard group them as one family regardless of
    /// format — the `-MLX-` infix (LiquidAI) must not split them from the bare stem.
    @Test func mlxAndGgufShareFamilyIdPerModel() {
        let ggufFamilyByName = Dictionary(
            CatalogEntry.defaults.map { ($0.name, $0.familyId) }, uniquingKeysWith: { first, _ in first })
        for mlx in CatalogEntry.mlxDefaults {
            let gguf = ggufFamilyByName[mlx.name]
            #expect(mlx.familyId == gguf,
                    "\(mlx.name): MLX familyId \(mlx.familyId) should equal GGUF \(gguf ?? "nil")")
        }
    }

    @Test func rowIdsAreUnique() {
        let ids = authored.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func repoToNameCoversEveryRow() {
        // Build the repo → name map from the authored data (the production
        // `CatalogEntry.repoToName` is derived from the flag-gated catalog); this catches
        // any two authored rows that collide on a repo identifier with different names.
        var repoToName: [String: String] = [:]
        for entry in authored { repoToName[entry.repoIdentifier] = entry.name }
        for entry in authored {
            #expect(repoToName[entry.repoIdentifier] == entry.name)
        }
    }
}
