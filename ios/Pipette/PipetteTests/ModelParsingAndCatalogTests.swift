import Testing
import Foundation
@testable import Pipette

@MainActor
struct ModelParsingAndCatalogTests {
    @Test func quantParsingSupportsCommonGgufSuffixes() {
        #expect(LocalStorage.parseQuant(from: "LFM2.5-350M-Q4_0.gguf") == "Q4_0")
        #expect(LocalStorage.parseQuant(from: "Qwen-Test-Q4_K_M.GGUF") == "Q4_K_M")
        #expect(LocalStorage.parseQuant(from: "Model-BF16.gguf") == "BF16")
        #expect(LocalStorage.parseQuant(from: "Model-IQ3_XXS.gguf") == "IQ3_XXS")
        #expect(LocalStorage.parseQuant(from: "Model-NoQuant.gguf") == nil)
    }

    @Test func modelStemStripsGgufAndTrailingQuantOnly() {
        #expect(LocalStorage.modelStem(from: "LFM2.5-350M-Q4_0.gguf") == "LFM2.5-350M")
        #expect(LocalStorage.modelStem(from: "Model-F16.gguf") == "Model")
        #expect(LocalStorage.modelStem(from: "Model-Preview.gguf") == "Model-Preview")
    }

    @Test func paramSizePrefersRepoThenFilenameAndRejectsVersionTokens() {
        #expect(
            LocalStorage.parseParamSize(name: "Fallback-350M-Q4_0.gguf", hfRepo: "liquid/LFM2.5-1.5B-GGUF") ==
            "1.5B"
        )
        #expect(
            LocalStorage.parseParamSize(name: "Gemma-3n-350M-Q4_0.gguf", hfRepo: nil) ==
            "350M"
        )
        #expect(LocalStorage.parseParamSize(name: "LFM2.5-Q4_0.gguf", hfRepo: nil) == nil)
    }

    @Test func normalizedModelStemPairsMmprojAndBaseAcrossQuantVariants() {
        #expect(
            LocalStorage.normalizedModelStem("mmproj-Gemma-3n-Q4_K_M.gguf") ==
            "gemma-3n"
        )
        #expect(
            LocalStorage.normalizedModelStem("Gemma-3n-Q4_0.gguf") ==
            "gemma-3n"
        )
        #expect(
            LocalStorage.normalizedModelStem("Gemma-3n.gguf") ==
            "gemma-3n"
        )
    }

    @Test func quantPillsMatchParsedQuantTokens() {
        #expect(QuantPill.all.matches("anything"))
        #expect(!(QuantPill.all.matches(nil)))
        #expect(QuantPill.q1.matches("Q1_0"))
        #expect(QuantPill.q2.matches("q2_0"))
        #expect(QuantPill.q4.matches("q4_0"))
        #expect(QuantPill.q4km.matches("Q4_K_M"))
        #expect(QuantPill.q5km.matches("q5_k_m"))
        #expect(!(QuantPill.q4.matches("Q4_K_M")))
        #expect(!(QuantPill.q2.matches("Q2_K")))
        #expect(!(QuantPill.q5km.matches(nil)))
    }

    @Test func allQuantSelectionIncludesEverySpecificQuantPill() {
        #expect(
            QuantPill.allSelection() ==
            Set(QuantPill.allCases)
        )
    }

    @Test func allQuantSelectionSkipsDisabledSpecificQuantPills() {
        #expect(
            QuantPill.allSelection(disabled: [.q4km]) ==
            Set(QuantPill.allCases).subtracting([.q4km])
        )
    }

    @Test func togglingAllQuantSelectionSelectsAndClearsAllQuantPills() {
        #expect(
            QuantPill.toggledSelection([], toggling: .all) ==
            Set(QuantPill.allCases)
        )
        #expect(
            QuantPill.toggledSelection(Set(QuantPill.allCases), toggling: .all) ==
            []
        )
    }

    @Test func togglingAllQuantOffKeepsSelectionWhenEmptyDisallowed() {
        #expect(
            QuantPill.toggledSelection(Set(QuantPill.allCases), toggling: .all, allowsEmpty: false) ==
            Set(QuantPill.allCases)
        )
        #expect(
            QuantPill.toggledSelection([], toggling: .all, allowsEmpty: false) ==
            Set(QuantPill.allCases)
        )
    }

    @Test func togglingLastSelectedQuantOffEmptiesSelectionWhenEmptyAllowed() {
        #expect(
            QuantPill.toggledSelection([.q4], toggling: .q4) ==
            []
        )
    }

    @Test func togglingLastSelectedQuantOffSelectsAllWhenEmptyDisallowed() {
        #expect(
            QuantPill.toggledSelection([.q4], toggling: .q4, allowsEmpty: false) ==
            Set(QuantPill.allCases)
        )
    }

    @Test func togglingDisabledQuantLeavesSelectionUnchanged() {
        #expect(
            QuantPill.toggledSelection([.q4], toggling: .q4, disabled: [.q4]) ==
            [.q4]
        )
    }

    @Test func quantSelectionNormalizationExpandsAllAndDetectsAllSpecifics() {
        #expect(
            QuantPill.normalizedSelection([.all]) ==
            Set(QuantPill.allCases)
        )
        // Selecting every specific pill collapses back to `.all`.
        #expect(
            QuantPill.normalizedSelection(Set(QuantPill.specificCases)) ==
            Set(QuantPill.allCases)
        )
    }

    @Test func quantSelectionNormalizationCanStayEmptyAfterDisabledSelectionClears() {
        #expect(
            QuantPill.normalizedSelection([], disabled: [.q4], defaultsToAll: false) ==
            []
        )
        #expect(
            QuantPill.normalizedSelection([], disabled: [.q4], defaultsToAll: true) ==
            Set(QuantPill.allCases).subtracting([.q4])
        )
    }

    @Test func modelCatalogGroupsByModelIdentityPreservingFirstSeenOrder() {
        let models = [
            model("Gemma-Test-2B-Q4_0.gguf", repo: "google/gemma-test-2B-GGUF"),
            model("Gemma-Test-2B-Q5_K_M.gguf", repo: "google/gemma-test-2B-GGUF"),
            model("Sideloaded-Model-Q4_0.gguf"),
            model("Sideloaded-Model-Q5_K_M.gguf"),
            model("Qwen-Test-Q4_0.gguf", repo: "qwen/qwen-test-GGUF")
        ]

        let groups = ModelCatalog.groups(from: models)

        // Keyed by normalized model stem (lowercased, quant stripped).
        #expect(groups.map(\.key) == [
            "gemma-test-2b",
            "sideloaded-model",
            "qwen-test"
        ])
        // These repos aren't in the real catalog, so the family name falls back to the
        // filename stem (display names now derive from the catalog, not an injected field).
        #expect(groups[0].name == "Gemma-Test-2B")
        #expect(groups[0].paramLabel == "2B")
        #expect(groups[0].quantCountLabel == "2 quants")
        #expect(groups[1].name == "Sideloaded-Model")
        #expect(groups[1].files.map(\.name) == [
            "Sideloaded-Model-Q4_0.gguf",
            "Sideloaded-Model-Q5_K_M.gguf"
        ])
    }

    @Test func modelCatalogMergesSameModelAcrossReposIntoOneGroup() {
        // Ministral: Q4_0 only exists on unsloth, K-quants on mistralai upstream.
        // They must collapse into one card while keeping per-file provenance.
        let models = [
            model("Ministral-3-3B-Instruct-2512-Q4_0.gguf",
                  repo: "unsloth/Ministral-3-3B-Instruct-2512-GGUF"),
            model("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
                  repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF"),
            model("Ministral-3-3B-Instruct-2512-Q5_K_M.gguf",
                  repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF")
        ]

        let groups = ModelCatalog.groups(from: models)

        #expect(groups.count == 1, "the same model from two repos should be one card")
        // The family name resolves from the catalog (familyId → name), not an injected field.
        #expect(groups[0].name == "Ministral 3 3B Instruct 2512")
        #expect(groups[0].quantCountLabel == "3 quants")
        // Per-file provenance survives the merge.
        #expect(
            Set(groups[0].files.map(\.hfRepo)) ==
            ["unsloth/Ministral-3-3B-Instruct-2512-GGUF",
             "mistralai/Ministral-3-3B-Instruct-2512-GGUF"]
        )
    }

    @Test func catalogDeclaresMinistralAcrossRepos() {
        // (quant, repo) pairs for a family, straight from the catalog.
        func variants(_ familyId: String) -> [(quant: String, repo: String)] {
            CatalogEntry.catalog
                .filter { $0.familyId == familyId }
                .compactMap { entry in entry.quant.map { (quant: $0, repo: entry.repoIdentifier) } }
        }
        let ministral = variants("ministral-3-3b-instruct-2512")
        #expect(!(ministral.isEmpty), "Ministral family missing from the catalog")
        #expect(Set(ministral.map(\.repo)).count > 1, "Ministral is sourced from two repos")
        // Q4_0 only exists on unsloth; the K-quants come from mistralai upstream.
        #expect(
            ministral.first { $0.quant == "Q4_0" }?.repo ==
            "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
        )
        // The GGUF K-quants come from mistralai upstream; with MLX surfaced the
        // 4bit MLX variant also brings in the mlx-community repo.
        #expect(
            Set(ministral.filter { $0.quant != "Q4_0" }.map(\.repo)) ==
            ["mistralai/Ministral-3-3B-Instruct-2512-GGUF",
             "mlx-community/Ministral-3-3B-Instruct-2512-4bit"]
        )
        // GGUF and MLX of one model unify into a single family, so LFM 2.5 350M
        // spans its GGUF repo and its MLX repo.
        #expect(Set(variants("lfm2.5-350m").map(\.repo)).count > 1)
    }

    @Test func groupingUnifiesFamilyAcrossReposAndFlagsRepoSpan() {
        let models = [
            model("Ministral-3-3B-Instruct-2512-Q4_0.gguf",
                  repo: "unsloth/Ministral-3-3B-Instruct-2512-GGUF"),
            model("Ministral-3-3B-Instruct-2512-Q5_K_M.gguf",
                  repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF"),
            // Sideload with a different repo — the same stem rejoins via the shared
            // spec-derived familyId (repo-independent for GGUF).
            model("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf")
        ]

        let groups = ModelCatalog.groups(from: models)

        #expect(groups.count == 1)
        #expect(groups[0].key == "ministral-3-3b-instruct-2512")
        #expect(groups[0].quantCountLabel == "3 quants")
        // The span must reflect the two real catalog repos, not the synthetic
        // per-sideload repo the third (repo-less) file contributes — otherwise
        // `spansRepos` is trivially true even when both catalog repos match.
        #expect(Set(groups[0].files.map(\.hfRepo)).isSuperset(of: [
            "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
            "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
        ]))
        // Name resolves from the declared family.
        #expect(groups[0].name == "Ministral 3 3B Instruct 2512")
    }

    /// Guards the load-bearing invariant: a catalog download and a sideloaded copy of
    /// the same file must land in the *same* group. Grouping keys on the spec's
    /// `familyId`, which for GGUF is the normalized filename stem regardless of repo,
    /// so the two copies share a key even though their (synthetic vs catalog) repos differ.
    @Test func catalogAndSideloadedCopiesShareGroupingKeyForEveryPreset() {
        for preset in CatalogEntry.defaults {
            // Only GGUF presets have a weight filename to sideload; MLX is a directory.
            let filename: String
            switch preset.source {
            case .ggufText, .ggufVision: filename = preset.source.artifactName
            case .mlx, .appleFoundationText: continue
            }
            let groups = ModelCatalog.groups(from: [
                model(filename, repo: preset.repoIdentifier),
                model(filename)  // sideload: different (synthetic) repo, same filename
            ])
            #expect(groups.count == 1,
                    "catalog + sideloaded \(filename) must share one group")
        }
    }

    @Test func modelDisplayNameStripsRepoAndGgufSuffix() {
        #expect(ModelCatalog.displayName(for: "google/gemma-3n-GGUF") == "gemma-3n")
        #expect(ModelCatalog.displayName(for: "Sideloaded-Model") == "Sideloaded-Model")
    }

    @Test func brandDetectionKeysOffModelFamilyNotJustRepoOwner() {
        #expect(ModelBrand.detect(name: "unsloth/gemma-3n", hfRepo: "unsloth/gemma-3n-GGUF") == .google)
        #expect(ModelBrand.detect(name: "Granite Tiny", hfRepo: nil) == .ibm)
        #expect(ModelBrand.detect(name: "Qwen Test", hfRepo: nil) == .qwen)
        #expect(ModelBrand.detect(name: "Meta Llama", hfRepo: nil) == .meta)
        #expect(ModelBrand.detect(name: "Ministral", hfRepo: nil) == .mistral)
        #expect(ModelBrand.detect(name: "Phi 4", hfRepo: nil) == .microsoft)
        #expect(ModelBrand.detect(name: "DeepSeek", hfRepo: nil) == .deepseek)
        #expect(ModelBrand.detect(name: "Unknown", hfRepo: nil) == .unknown)
    }

    private func model(
        _ name: String,
        repo: String? = nil
    ) -> DiscoveredModel {
        // Every discovered model carries a typed manifest `source`, and all display
        // metadata (name, displayName, familyId) is derived from it. For the "sideload"
        // cases (no explicit repo) synthesize a distinct placeholder repo from the
        // filename so grouping still keys off the stem, not the repo.
        let repoSlug = repo ?? "sideload/\(LocalStorage.normalizedModelStem(name))"
        let source = ggufTextFixture(repoSlug, name)
        return DiscoveredModel(source: source, path: "/tmp/\(name)", sizeBytes: 1_000_000)
    }
}
