import Testing
import Foundation
@testable import Pipette

/// Correctness guards for the gated MLX preset catalog (`CatalogEntry.mlxDefaults`)
/// added in PIP-263. These assert the catalog's shape (engine, identifier form,
/// per-family variant counts), that the gating keeps MLX rows out of the shown
/// catalog while `MLXFeatureFlag.visibleInUI` is false, and that the
/// `CatalogEntry` computed properties stay MLX-safe for a bare repo slug.
@MainActor
@Suite struct MLXPresetCatalogTests {
    /// Every MLX preset is engine `.mlx` and carries a bare repo slug (a
    /// directory identifier), never a GGUF `repo:quant` identifier.
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

    /// Every family ships exactly one MLX row, at whichever bit-width its publisher
    /// exports — `4bit` for most, `1bit`/`2bit` for the Bonsai line. There are no
    /// `8bit` rows and no retired Gemma 3 270M entry. The roster is spelled out so a
    /// row added or dropped by accident fails here.
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
            "Bonsai 27B": "1bit",
            "Ternary Bonsai 1.7B": "2bit",
            "Ternary Bonsai 8B": "2bit",
        ]
        let byName = Dictionary(grouping: CatalogEntry.mlxDefaults, by: \.name)

        #expect(CatalogEntry.mlxDefaults.count == expectedQuantByFamily.count)
        #expect(Set(byName.keys) == Set(expectedQuantByFamily.keys))
        for (name, quant) in expectedQuantByFamily {
            #expect((byName[name] ?? []).map(\.quant) == [quant], "\(name) should be a single \(quant) entry")
        }
        // Retired: no 8bit rows, no Gemma 3 270M.
        #expect(!CatalogEntry.mlxDefaults.contains { $0.repoIdentifier.hasSuffix("-8bit") })
        #expect(!CatalogEntry.mlxDefaults.contains { $0.name == "Gemma 3 270M IT" })
    }

    /// With `MLXFeatureFlag.visibleInUI == true` the shown `catalog` is `defaults + mlxDefaults`:
    /// the GGUF rows are all still present and the MLX rows are surfaced (they
    /// download as directories via `startMLXDownload`).
    @Test func catalogSurfacesMlxRowsWhenVisible() {
        #expect(MLXFeatureFlag.visibleInUI == true)
        #expect(CatalogEntry.catalog.count == CatalogEntry.defaults.count + CatalogEntry.mlxDefaults.count)
        #expect(CatalogEntry.catalog.contains { if case .mlx = $0.source { true } else { false } })
        // Every GGUF default is still shown.
        let ids = Set(CatalogEntry.catalog.map(\.id))
        #expect(CatalogEntry.defaults.allSatisfy { ids.contains($0.id) })
        #expect(CatalogEntry.mlxDefaults.allSatisfy { ids.contains($0.id) })
    }

    /// The `CatalogEntry` computed props are sane for an MLX slug: no crash, and
    /// they decode the repo, bit-width, family, and (absent) filename correctly.
    @Test func computedPropsAreMlxSafe() throws {
        let preset = CatalogEntry.mlxDefaults.first { $0.repoIdentifier == "mlx-community/Qwen3.5-0.8B-4bit" }
        let qwen = try #require(preset)

        #expect(qwen.repoIdentifier == "mlx-community/Qwen3.5-0.8B-4bit")
        #expect(qwen.quant == "4bit")
        #expect(qwen.familyId == "qwen3.5-0.8b")

        // A Liquid slug resolves its bit-width and strips both it *and* the `-MLX-`
        // infix, so the familyId matches the GGUF sibling (unified family).
        let lfm = try #require(
            CatalogEntry.mlxDefaults.first { $0.repoIdentifier == "LiquidAI/LFM2.5-350M-MLX-4bit" }
        )
        #expect(lfm.repoIdentifier == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(lfm.quant == "4bit")
        #expect(lfm.familyId == "lfm2.5-350m")
    }
}
