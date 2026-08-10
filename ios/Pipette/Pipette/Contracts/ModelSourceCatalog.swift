import Foundation

/// A downloadable catalog row: a display `name` (the family it belongs to), the inner
/// `source` spec (the typed `Model` coordinate it downloads from), and a
/// download-size hint. Presentation (grouping, quant filtering) derives everything
/// else from `source`, so the catalog is authored as a list of these — a sequence of
/// `Model` variants wrapped with just the display metadata the coordinate lacks.
nonisolated struct CatalogEntry: Identifiable, Sendable {
    let name: String
    let source: Model
    /// Total download size in bytes (SI): the GGUF weight file, or the MLX repo's
    /// on-disk tree total. 0 only if genuinely unknown.
    let sizeBytes: Int64
    /// The authored quant label (GGUF K-quant like `Q4_0`, or the MLX `4bit`/`8bit`).
    /// Stored explicitly by the builders rather than re-parsed from the artifact name,
    /// so the download flow shows the catalog's declared quant even when a repo slug
    /// doesn't encode it. `Model.quant` still derives it for *installed* models, whose
    /// only provenance is the on-disk manifest.
    let quant: String?

    /// Stable per-row identity: repo + quant is unique across the catalog.
    var id: String {
        // The GGUF weight filename backs the id only when a quant couldn't be parsed;
        // MLX rows are directories with no file, so they fall through to the name.
        // The weights' own name, whatever arm names them — `reference`'s leaf, which is
        // the repo subpath for a HuggingFace source and the path or URL otherwise.
        let fileFallback: String? = switch source {
        case let .ggufText(m): m.source.reference.split(separator: "/").last.map(String.init)
        case let .ggufVision(m): m.source.modelReference.split(separator: "/").last.map(String.init)
        case .mlx, .appleFoundationText: nil
        }
        return "\(repoIdentifier)#\(quant ?? fileFallback ?? name)"
    }

    /// The catalog only ever holds downloadable (HF-backed) rows, so `source.repo` is
    /// always present here; empty is a defensive fallback for the bare (AFM) case.
    var repoIdentifier: String { source.repo?.description ?? "" }

    /// Family identity shared by every quant *and format* of a model — the one
    /// derivation lives on `Model` so the catalog, discovery, and job cells
    /// all agree.
    var familyId: String { source.familyId }
}

// MARK: - Catalog data

// This immutable reference data is read from both the main actor (the catalog UI) and
// nonisolated contexts (`DiscoveredModel`'s derived display name, off-main headless
// listing), so the static members are `nonisolated` — they'd otherwise pick up the
// module's main-actor default from being declared in an extension (the primary
// `nonisolated struct` declaration doesn't cover extension members).
extension CatalogEntry {
    /// The catalog surfaced in the UI: GGUF `defaults` plus `mlxDefaults`, with the
    /// MLX rows merged in only when `MLXFeatureFlag.visibleInUI` is true.
    nonisolated static let catalog: [CatalogEntry] =
        MLXFeatureFlag.visibleInUI ? defaults + mlxDefaults : defaults

    /// Maps HF repo identifier → family display name for all catalog rows.
    nonisolated static let repoToName: [String: String] = {
        var dict: [String: String] = [:]
        for entry in catalog { dict[entry.repoIdentifier] = entry.name }
        return dict
    }()

    /// Maps catalog `familyId` → family display name. A family's quants may span
    /// repos (and formats), so this is keyed on the repo-independent `familyId`; the
    /// first catalog row for a family wins. The single familyId→name lookup — models
    /// absent from the catalog (sideloads) fall back elsewhere.
    nonisolated static let familyIdToName: [String: String] = {
        var dict: [String: String] = [:]
        for entry in catalog where dict[entry.familyId] == nil { dict[entry.familyId] = entry.name }
        return dict
    }()

    // MARK: Builders
    //
    // Inputs are compile-time constants verified by `CatalogIntegrityTests`, so the
    // validated newtypes are built via their trusted (unchecked) initializers — a
    // malformed literal surfaces as a failing catalog assertion, not a runtime throw.

    /// A single-file GGUF text entry. The on-disk filename is
    /// `<repo-name minus -GGUF>-<quant>.gguf`, the layout these repos publish.
    /// The suffix match is case-insensitive because not every publisher uppercases
    /// it (`prism-ml/Bonsai-27B-gguf`), and the file layout is the same either way.
    nonisolated static func gguf(_ name: String, _ repo: String, _ quant: String, bytes: Int64) -> CatalogEntry {
        let hf = trustedRepo(repo)
        let leaf = hf.repoName.value
        let stem = leaf.lowercased().hasSuffix("-gguf") ? String(leaf.dropLast("-GGUF".count)) : leaf
        let filename = RepoSubpath(validated: "\(stem)-\(quant).gguf")
        return CatalogEntry(name: name, source: .ggufText(GgufText(source: .huggingFace(repo: hf, path: filename, sha256: nil))),
                            sizeBytes: bytes, quant: quant)
    }

    /// An MLX bundle entry from a bare `org/name` repo slug (a directory deployment).
    /// `bytes` is the repo's total on-disk size (summed from the HF file tree). `4bit`
    /// is the common export, so `quant` defaults to it; pass it explicitly for any
    /// other bit-width.
    nonisolated static func mlx(_ name: String, _ repo: String, bytes: Int64, quant: String = "4bit") -> CatalogEntry {
        CatalogEntry(name: name, source: .mlx(Mlx(source: .huggingFace(repo: trustedRepo(repo), prefix: nil))),
                     sizeBytes: bytes, quant: quant)
    }

    nonisolated private static func trustedRepo(_ slug: String) -> HFRepo {
        let parts = slug.split(separator: "/", maxSplits: 1)
        let org = parts.first.map(String.init) ?? slug
        let name = parts.count > 1 ? String(parts[1]) : ""
        return HFRepo(org: HFOrg(validated: org), repoName: HFRepoName(validated: name))
    }

    // Sizes are each GGUF file's exact bytes, from the HF file tree.
    nonisolated static let defaults: [CatalogEntry] = [
        // LFM 2.5 230M
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_0", bytes: 149_080_928),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_K_M", bytes: 153_406_304),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q5_K_M", bytes: 171_625_312),
        // LFM 2.5 350M
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_0", bytes: 219_309_792),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_K_M", bytes: 229_312_224),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q5_K_M", bytes: 260_376_288),
        // LFM2 700M
        .gguf("LFM2 700M", "LiquidAI/LFM2-700M-GGUF", "Q4_0", bytes: 446_321_600),
        .gguf("LFM2 700M", "LiquidAI/LFM2-700M-GGUF", "Q4_K_M", bytes: 468_624_320),
        .gguf("LFM2 700M", "LiquidAI/LFM2-700M-GGUF", "Q5_K_M", bytes: 538_026_944),
        // Qwen 3.5 0.8B
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q4_0", bytes: 507_154_688),
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q4_K_M", bytes: 532_517_120),
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q5_K_M", bytes: 590_057_728),
        // Qwen 3.5 2B
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q4_0", bytes: 1_214_873_856),
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q4_K_M", bytes: 1_280_835_840),
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q5_K_M", bytes: 1_435_238_656),
        // Qwen 3.5 4B
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q4_0", bytes: 2_583_221_408),
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q4_K_M", bytes: 2_740_937_888),
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q5_K_M", bytes: 3_143_656_608),
        // Granite 4.0-h 350M
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q4_0", bytes: 216_073_120),
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q4_K_M", bytes: 222_662_560),
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q5_K_M", bytes: 252_331_936),
        // Granite 4.0-h 1B
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q4_0", bytes: 868_316_384),
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q4_K_M", bytes: 901_162_208),
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q5_K_M", bytes: 1_048_556_768),
        // Granite 4.0-h Micro
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q4_0", bytes: 1_855_516_320),
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q4_K_M", bytes: 1_942_564_512),
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q5_K_M", bytes: 2_273_455_776),
        // Granite 4.0 350M
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q4_0", bytes: 228_470_176),
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q4_K_M", bytes: 236_985_760),
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q5_K_M", bytes: 264_052_128),
        // Gemma 3 1B IT
        .gguf("Gemma 3 1B IT", "unsloth/gemma-3-1b-it-GGUF", "Q4_0", bytes: 721_918_496),
        .gguf("Gemma 3 1B IT", "unsloth/gemma-3-1b-it-GGUF", "Q4_K_M", bytes: 806_058_272),
        .gguf("Gemma 3 1B IT", "unsloth/gemma-3-1b-it-GGUF", "Q5_K_M", bytes: 851_345_696),
        // LFM 2.5 1.2B Instruct
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_0", bytes: 695_751_488),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_K_M", bytes: 730_895_168),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q5_K_M", bytes: 843_354_944),
        // LFM2 2.6B
        .gguf("LFM2 2.6B", "LiquidAI/LFM2-2.6B-GGUF", "Q4_0", bytes: 1_483_108_576),
        .gguf("LFM2 2.6B", "LiquidAI/LFM2-2.6B-GGUF", "Q4_K_M", bytes: 1_563_668_704),
        .gguf("LFM2 2.6B", "LiquidAI/LFM2-2.6B-GGUF", "Q5_K_M", bytes: 1_828_958_432),
        // LFM2 2.6B Exp
        .gguf("LFM2 2.6B Exp", "LiquidAI/LFM2-2.6B-Exp-GGUF", "Q4_0", bytes: 1_558_606_176),
        .gguf("LFM2 2.6B Exp", "LiquidAI/LFM2-2.6B-Exp-GGUF", "Q4_K_M", bytes: 1_639_166_304),
        .gguf("LFM2 2.6B Exp", "LiquidAI/LFM2-2.6B-Exp-GGUF", "Q5_K_M", bytes: 1_921_233_248),
        // LFM 2.5 8B A1B
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_0", bytes: 4_844_678_368),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_K_M", bytes: 5_155_564_768),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q5_K_M", bytes: 6_030_339_296),
        // Ministral 3 3B Instruct 2512 (Q4_0 from unsloth, K-quants from mistralai)
        .gguf("Ministral 3 3B Instruct 2512", "unsloth/Ministral-3-3B-Instruct-2512-GGUF", "Q4_0", bytes: 2_046_375_200),
        .gguf("Ministral 3 3B Instruct 2512", "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Q4_K_M", bytes: 2_147_023_008),
        .gguf("Ministral 3 3B Instruct 2512", "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Q5_K_M", bytes: 2_474_178_720),
        // Llama 3.2 1B Instruct
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q4_0", bytes: 773_025_824),
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M", bytes: 807_694_368),
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q5_K_M", bytes: 911_503_392),
        // Llama 3.2 3B Instruct
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q4_0", bytes: 1_921_909_184),
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q4_K_M", bytes: 2_019_377_600),
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q5_K_M", bytes: 2_322_153_920),
        // Gemma 4 E2B IT
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q4_0", bytes: 3_041_376_384),
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q4_K_M", bytes: 3_106_736_256),
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q5_K_M", bytes: 3_356_035_200),
        // Gemma 4 E4B IT
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q4_0", bytes: 4_836_000_928),
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q4_K_M", bytes: 4_977_169_568),
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q5_K_M", bytes: 5_481_796_768),
        // Nemotron 3 Nano 4B — `nemotron_h`, a Mamba-2/attention hybrid. `unsloth`
        // rather than NVIDIA's own repo: the latter publishes one quant under a
        // filename (`NVIDIA-Nemotron3-Nano-4B-…`) that doesn't match its repo stem.
        .gguf("Nemotron 3 Nano 4B", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF", "Q4_0", bytes: 2_528_598_176),
        .gguf("Nemotron 3 Nano 4B", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF", "Q4_K_M", bytes: 2_900_295_712),
        .gguf("Nemotron 3 Nano 4B", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF", "Q5_K_M", bytes: 3_159_586_464),
        // Bonsai — `qwen35`/`qwen3` weights at 1-bit (`Q1_0`), the only quant these
        // repos publish. `PQ2_0` siblings are skipped: the leading `P` isn't a quant
        // token, so they'd land in their own family instead of grouping with `Q2_0`.
        .gguf("Bonsai 8B", "prism-ml/Bonsai-8B-gguf", "Q1_0", bytes: 1_158_654_496),
        .gguf("Bonsai 27B", "prism-ml/Bonsai-27B-gguf", "Q1_0", bytes: 3_803_452_480),
        // Ternary Bonsai — the 2-bit line. 27B is 7.2 GB, past every other row here;
        // it is authored for capable Android devices and expected to be out of budget
        // on iOS.
        .gguf("Ternary Bonsai 1.7B", "prism-ml/Ternary-Bonsai-1.7B-gguf", "Q2_0", bytes: 463_290_464),
        .gguf("Ternary Bonsai 8B", "prism-ml/Ternary-Bonsai-8B-gguf", "Q2_0", bytes: 2_182_184_672),
        .gguf("Ternary Bonsai 27B", "prism-ml/Ternary-Bonsai-27B-gguf", "Q2_0", bytes: 7_165_121_600),
    ]

    // MARK: - MLX preset catalog (gated)
    //
    // Selection criteria: only models the project benchmarks on MLX (`hf_mlx` in the
    // plan files). MLX is 4-bit only — the bit-width is encoded in the repo slug
    // because an MLX model is a repo *directory*. `LiquidAI/*` for Liquid models,
    // `mlx-community/*` otherwise. Gated behind `MLXFeatureFlag.visibleInUI`.
    // Sizes are each repo's total on-disk bytes, summed from the HF file tree.
    nonisolated static let mlxDefaults: [CatalogEntry] = [
        .mlx("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-MLX-4bit", bytes: 150_867_598),
        .mlx("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-MLX-4bit", bytes: 226_574_864),
        .mlx("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit", bytes: 663_548_128),
        .mlx("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", bytes: 4_851_998_389),
        .mlx("Gemma 3 1B IT", "mlx-community/gemma-3-1b-it-4bit", bytes: 771_863_590),
        .mlx("LFM2 700M", "mlx-community/LFM2-700M-4bit", bytes: 422_679_301),
        .mlx("Qwen 3.5 0.8B", "mlx-community/Qwen3.5-0.8B-4bit", bytes: 652_029_391),
        .mlx("Granite 4.0 350M", "mlx-community/granite-4.0-350m-4bit", bytes: 208_127_385),
        .mlx("Granite 4.0-h 350M", "mlx-community/granite-4.0-h-350m-4bit", bytes: 201_778_438),
        .mlx("Granite 4.0-h 1B", "mlx-community/granite-4.0-h-1b-4bit", bytes: 833_197_077),
        .mlx("Llama 3.2 1B Instruct", "mlx-community/Llama-3.2-1B-Instruct-4bit", bytes: 712_593_855),
        .mlx("LFM2 2.6B", "mlx-community/LFM2-2.6B-4bit", bytes: 1_450_530_492),
        .mlx("LFM2 2.6B Exp", "mlx-community/LFM2-2.6B-Exp-4bit", bytes: 1_450_530_508),
        .mlx("Qwen 3.5 2B", "mlx-community/Qwen3.5-2B-4bit", bytes: 1_749_081_927),
        .mlx("Qwen 3.5 4B", "mlx-community/Qwen3.5-4B-4bit", bytes: 3_061_130_647),
        .mlx("Granite 4.0-h Micro", "mlx-community/granite-4.0-h-micro-4bit", bytes: 1_806_622_893),
        .mlx("Gemma 4 E2B IT", "mlx-community/gemma-4-e2b-it-4bit", bytes: 3_613_530_663),
        .mlx("Gemma 4 E4B IT", "mlx-community/gemma-4-e4b-it-4bit", bytes: 5_249_811_602),
        .mlx("Llama 3.2 3B Instruct", "mlx-community/Llama-3.2-3B-Instruct-4bit", bytes: 1_824_825_759),
        .mlx("Ministral 3 3B Instruct 2512", "mlx-community/Ministral-3-3B-Instruct-2512-4bit", bytes: 2_779_152_776),
        .mlx("Bonsai 8B", "prism-ml/Bonsai-8B-mlx-1bit", bytes: 1_296_482_622, quant: "1bit"),
        .mlx("Bonsai 27B", "prism-ml/Bonsai-27B-mlx-1bit", bytes: 5_159_415_841, quant: "1bit"),
        .mlx("Ternary Bonsai 1.7B", "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", bytes: 495_636_905, quant: "2bit"),
        .mlx("Ternary Bonsai 8B", "prism-ml/Ternary-Bonsai-8B-mlx-2bit", bytes: 2_315_264_643, quant: "2bit"),
    ]
}
