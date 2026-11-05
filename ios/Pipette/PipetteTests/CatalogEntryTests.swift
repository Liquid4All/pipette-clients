import Foundation
import Testing

@testable import Pipette

/// The catalog builders use the newtypes' unchecked initializers for literals, so these
/// tests stand in for the validation that construction skips.
@MainActor
struct CatalogEntryTests {
    /// Independent of `MlxFeatureFlag.visibleInUI`, which gates `CatalogEntry.catalog`.
    private let authored = CatalogEntry.defaults + CatalogEntry.mlxDefaults

    /// GGUF families deliberately shipped without an MLX row: the results dashboard
    /// (PIP-511) publishes `llama.cpp` configurations only, so no MLX export of these was
    /// ever benchmarked. Drop an entry when one is.
    private static let ggufOnlyFamilies: Set<String> = [
        "MiniCPM5 1B",
        "LFM 2.5 2.6B",
        "G9v3 3B",
        "Nanbeige 4.2 3B",
        "Olmo 3 7B Think",
        "Ling 3.0 Tiny",
        "Falcon H1R 7B",
        "Ministral 3 8B Instruct 2512",
        "Granite 4.1 8B",
        "Qwen 3.5 9B",
        "Ornith 1.0 9B",
    ]

    /// MLX families with no GGUF sibling: the dashboard publishes nothing for them on the
    /// iPhone 17 Pro (PIP-511), and it publishes no MLX results at all, so it cannot speak
    /// to the MLX catalog. Audited like `ggufOnlyFamilies`, so the GGUF catalog cannot
    /// quietly shed a family.
    private static let mlxOnlyFamilies: Set<String> = [
        "Bonsai 8B",
        "Gemma 3 1B IT",
        "LFM2 700M",
        "LFM2 2.6B",
        "LFM2 2.6B Exp",
        "Ternary Bonsai 1.7B",
        "Ternary Bonsai 8B",
    ]

    @Test func rowCountsMatchAuthoredFamilies() {
        // Exactly the configurations the dashboard publishes for `iPhone 17 Pro /
        // llama.cpp:ios` (PIP-511): 17 families carry three quants and 10 carry `Q4_K_M`
        // alone.
        #expect(CatalogEntry.defaults.count == 61)
        #expect(CatalogEntry.mlxDefaults.count == 23)
        #expect(authored.count == CatalogEntry.defaults.count + CatalogEntry.mlxDefaults.count)
    }

    @Test func everyEntryHasAValidRepo() throws {
        for entry in authored {
            let repo = try #require(entry.source.repo, "catalog entry \(entry.repoIdentifier) must have a repo")
            #expect(HfOrg.validate(repo.org.value), "invalid org in \(entry.repoIdentifier)")
            #expect(HfRepoName.validate(repo.repoName.value), "invalid repo name in \(entry.repoIdentifier)")
        }
    }

    @Test func ggufEntriesHaveParsableFilenameAndQuant() {
        for entry in CatalogEntry.defaults {
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
            // `Model.familyId` strips this suffix to rejoin an MLX row with its GGUF
            // sibling, so a mismatch silently splits the family in the UI.
            let quant = entry.quant ?? ""
            #expect(quant.hasSuffix("bit"), "\(entry.repoIdentifier) quant=\(entry.quant ?? "nil")")
            #expect(entry.repoIdentifier.lowercased().hasSuffix("-" + quant.lowercased()),
                    "\(entry.repoIdentifier) should end in -\(quant)")
            #expect(!entry.repoIdentifier.contains(":"))
            #expect(entry.sizeBytes > 0, "\(entry.repoIdentifier) should carry a fetched on-disk size")
        }
    }

    @Test func everyGgufFamilyHasAnMlxCounterpart() {
        let ggufNames = Set(CatalogEntry.defaults.map(\.name))
        let mlxNames = Set(CatalogEntry.mlxDefaults.map(\.name))
        let missing = ggufNames.subtracting(mlxNames)
        #expect(missing.subtracting(Self.ggufOnlyFamilies).isEmpty,
                "GGUF families missing MLX: \(missing.subtracting(Self.ggufOnlyFamilies).sorted())")
        #expect(Self.ggufOnlyFamilies.subtracting(missing).isEmpty,
                "stale ggufOnlyFamilies: \(Self.ggufOnlyFamilies.subtracting(missing).sorted())")
    }

    /// The downloaded list and job wizard group by `familyId`, so the `-MLX-` infix
    /// (LiquidAI) must not split an MLX build from its GGUF stem.
    @Test func mlxAndGgufShareFamilyIdPerModel() {
        let ggufFamilyByName = Dictionary(
            CatalogEntry.defaults.map { ($0.name, $0.familyId) }, uniquingKeysWith: { first, _ in first })
        for mlx in CatalogEntry.mlxDefaults {
            guard let gguf = ggufFamilyByName[mlx.name] else {
                #expect(Self.mlxOnlyFamilies.contains(mlx.name),
                        "\(mlx.name) has no GGUF sibling and is not declared MLX-only")
                continue
            }
            #expect(mlx.familyId == gguf,
                    "\(mlx.name): MLX familyId \(mlx.familyId) should equal GGUF \(gguf)")
        }
        let mlxNames = Set(CatalogEntry.mlxDefaults.map(\.name))
        let orphans = mlxNames.subtracting(ggufFamilyByName.keys)
        #expect(Self.mlxOnlyFamilies.subtracting(orphans).isEmpty,
                "stale mlxOnlyFamilies: \(Self.mlxOnlyFamilies.subtracting(orphans).sorted())")
    }

    @Test func rowIdsAreUnique() {
        let ids = authored.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func repoToNameCoversEveryRow() {
        // Built from the authored data: the production `CatalogEntry.repoToName` comes from
        // the flag-gated catalog.
        var repoToName: [String: String] = [:]
        for entry in authored { repoToName[entry.repoIdentifier] = entry.name }
        for entry in authored {
            #expect(repoToName[entry.repoIdentifier] == entry.name)
        }
    }

    // MARK: - The gated MLX presets

    @Test func everyMlxPresetIsMlxEngineWithColonlessIdentifier() {
        #expect(!CatalogEntry.mlxDefaults.isEmpty)
        for preset in CatalogEntry.mlxDefaults {
            guard case .mlx = preset.source else {
                Issue.record("\(preset.id) should be .mlx")
                continue
            }
            #expect(!preset.repoIdentifier.contains(":"),
                    "\(preset.id) identifier must be a bare repo slug (no colon)")
        }
    }

    /// The roster is spelled out so a row added or dropped by accident fails here.
    @Test func everyMlxFamilyShipsOneRowAtItsPublishedBitWidth() {
        let expectedQuantByFamily: [String: String] = [
            "LFM 2.5 230M": "4bit",
            "LFM 2.5 350M": "4bit",
            "LFM 2.5 1.2B Instruct": "4bit",
            "LFM 2.5 8B A1B": "4bit",
            "Gemma 3 1B IT": "4bit",
            "LFM2 700M": "4bit",
            "Qwen 3.5 0.8B": "4bit",
            "Granite 4.0 350M": "4bit",
            "Granite 4.0-h 350M": "4bit",
            "Granite 4.0-h 1B": "4bit",
            "Llama 3.2 1B Instruct": "4bit",
            "LFM2 2.6B": "4bit",
            "LFM2 2.6B Exp": "4bit",
            "Qwen 3.5 2B": "4bit",
            "Qwen 3.5 4B": "4bit",
            "Granite 4.0-h Micro": "4bit",
            "Gemma 4 E2B IT": "4bit",
            "Gemma 4 E4B IT": "4bit",
            "Llama 3.2 3B Instruct": "4bit",
            "Ministral 3 3B Instruct 2512": "4bit",
            "Bonsai 8B": "1bit",
            "Ternary Bonsai 1.7B": "2bit",
            "Ternary Bonsai 8B": "2bit",
        ]
        let byName = Dictionary(grouping: CatalogEntry.mlxDefaults, by: \.name)

        #expect(CatalogEntry.mlxDefaults.count == expectedQuantByFamily.count)
        #expect(Set(byName.keys) == Set(expectedQuantByFamily.keys))
        for (name, quant) in expectedQuantByFamily {
            #expect((byName[name] ?? []).map(\.quant) == [quant], "\(name) should be a single \(quant) entry")
        }
        #expect(!CatalogEntry.mlxDefaults.contains { $0.repoIdentifier.hasSuffix("-8bit") })
        #expect(!CatalogEntry.mlxDefaults.contains { $0.name == "Gemma 3 270M IT" })
    }

    @Test func catalogSurfacesMlxRowsWhenVisible() {
        #expect(MlxFeatureFlag.visibleInUI == true)
        #expect(CatalogEntry.catalog.count
            == CatalogEntry.defaults.count + CatalogEntry.mlxDefaults.count
                + CatalogEntry.onnxDefaults.count)
        #expect(CatalogEntry.catalog.contains { if case .mlx = $0.source { true } else { false } })
        #expect(CatalogEntry.catalog.contains {
            if case .onnxGenai = $0.source { true } else { false }
        })
        let ids = Set(CatalogEntry.catalog.map(\.id))
        #expect(CatalogEntry.defaults.allSatisfy { ids.contains($0.id) })
        #expect(CatalogEntry.mlxDefaults.allSatisfy { ids.contains($0.id) })
        #expect(CatalogEntry.onnxDefaults.allSatisfy { ids.contains($0.id) })
    }

    /// The prefix is which package the row means, and the pin keeps a resumed
    /// multi-gigabyte transfer on one tree.
    @Test func everyOnnxRowIsPrefixedAndPinned() throws {
        #expect(!CatalogEntry.onnxDefaults.isEmpty)
        for entry in CatalogEntry.onnxDefaults {
            guard case let .onnxGenai(model) = entry.source,
                  case let .huggingFace(repo, prefix) = model.source else {
                Issue.record("\(entry.name) is not a HuggingFace ONNX row"); continue
            }
            let subpath = try #require(prefix, "\(entry.name) names no prefix")
            #expect(repo.revision != nil, "\(entry.name) is not pinned to a revision")
            #expect(entry.quant == subpath.leaf)
            // The subtree's size, not the repo's, which would budget several times what is
            // fetched.
            #expect(entry.sizeBytes > 0)
        }
    }

    @Test func computedPropsAreMlxSafe() throws {
        let preset = CatalogEntry.mlxDefaults.first { $0.repoIdentifier == "mlx-community/Qwen3.5-0.8B-4bit" }
        let qwen = try #require(preset)

        #expect(qwen.repoIdentifier == "mlx-community/Qwen3.5-0.8B-4bit")
        #expect(qwen.quant == "4bit")
        #expect(qwen.familyId == "qwen3.5-0.8b")

        // A Liquid slug strips both the bit-width and the `-MLX-` infix.
        let lfm = try #require(
            CatalogEntry.mlxDefaults.first { $0.repoIdentifier == "LiquidAI/LFM2.5-350M-MLX-4bit" }
        )
        #expect(lfm.repoIdentifier == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(lfm.quant == "4bit")
        #expect(lfm.familyId == "lfm2.5-350m")
    }
}
