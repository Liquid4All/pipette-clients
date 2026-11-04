import Foundation

/// A downloadable catalog row: the family display `name`, the `Model` coordinate it
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

    /// Stable per-row identity.
    ///
    /// The store key leads because it separates format, repo, pinned revision and prefix.
    /// Repo and quant alone cannot: one ONNX repo carries several packages whose prefixes
    /// can share a leaf. A row with no store key falls back to the repo plus the first of
    /// quant, weight filename or name.
    var id: String {
        if let key = ModelStorageKey.of(source) { return key.value }
        let fileFallback: String? = switch source {
        case let .ggufText(m): m.source.reference.split(separator: "/").last.map(String.init)
        case let .ggufVision(m): m.source.weightsReference.split(separator: "/").last.map(String.init)
        case let .ggufVisionSpeculative(m):
            m.source.vision.weightsReference.split(separator: "/").last.map(String.init)
        case .mlx, .onnxGenai, .appleFoundationText: nil
        }
        return "\(repoIdentifier)#\(quant ?? fileFallback ?? name)"
    }

    /// Every catalog row is HF-backed; empty is a defensive fallback.
    var repoIdentifier: String { source.repo?.description ?? "" }

    var familyId: String { source.familyId }
}

// MARK: - Catalog data

// Read from the main actor and from nonisolated contexts, so the statics are marked
// `nonisolated`: extension members do not inherit the struct's `nonisolated`.
extension CatalogEntry {
    /// The ONNX rows are not gated: that runtime ships exposed (PIP-592), so a flag would
    /// be one nobody turns off.
    nonisolated static let catalog: [CatalogEntry] =
        (MlxFeatureFlag.visibleInUI ? defaults + mlxDefaults : defaults) + onnxDefaults

    nonisolated static let repoToName: [String: String] = {
        var dict: [String: String] = [:]
        for entry in catalog { dict[entry.repoIdentifier] = entry.name }
        return dict
    }()

    /// Keyed on `familyId` because a family's quants may span repos and formats; the first
    /// row for a family wins.
    nonisolated static let familyIdToName: [String: String] = {
        var dict: [String: String] = [:]
        for entry in catalog where dict[entry.familyId] == nil { dict[entry.familyId] = entry.name }
        return dict
    }()

    // MARK: Builders
    //
    // Inputs are literals checked by `CatalogEntryTests`, so the newtypes use their
    // unchecked initializers: a malformed literal fails a test, not a runtime throw.

    /// The on-disk filename is `<repo-name minus -GGUF>-<quant>.gguf`, the layout these
    /// repos publish.
    nonisolated static func gguf(_ name: String, _ repo: String, _ quant: String, bytes: Int64) -> CatalogEntry {
        let hf = trustedRepo(repo)
        let filename = RepoSubpath(validated: "\(LocalStorage.repoStem(hf.repoName.value))-\(quant).gguf")
        return CatalogEntry(name: name, source: .ggufText(GgufText(source: .huggingFace(repo: hf, path: filename, sha256: nil))),
                            sizeBytes: bytes, quant: quant)
    }

    /// `quant` defaults to `4bit`, the common export; pass any other bit-width explicitly.
    nonisolated static func mlx(_ name: String, _ repo: String, bytes: Int64,
                                quant: String = "4bit") -> CatalogEntry {
        CatalogEntry(name: name,
                     source: .mlx(Mlx(source: .huggingFace(repo: trustedRepo(repo),
                                                           prefix: nil))),
                     sizeBytes: bytes, quant: quant)
    }

    /// - `prefix`: these repos publish several packages side by side (Phi-3 ships CPU
    ///   int4, CUDA fp16 and DirectML in one repo), so the prefix picks the model, and
    ///   `quant` is its leaf.
    /// - `revision`: a pinned commit, so a resumed multi-gigabyte pull cannot meet a tree
    ///   that moved on `main` (which `HubTransferPlanner` would refuse).
    nonisolated static func onnxGenai(_ name: String, _ repo: String, prefix: String,
                                      revision: String, bytes: Int64) -> CatalogEntry {
        var hf = trustedRepo(repo)
        hf.revision = HfRevision(validated: revision)
        return CatalogEntry(
            name: name,
            source: .onnxGenai(OnnxGenai(source: .huggingFace(repo: hf,
                                                              prefix: trustedPrefix(prefix)))),
            sizeBytes: bytes,
            // Matches what `Model.quant` derives for the installed copy.
            quant: prefix.split(separator: "/").last.map(String.init) ?? prefix)
    }

    nonisolated private static func trustedRepo(_ slug: String) -> HfRepo {
        let parts = slug.split(separator: "/", maxSplits: 1)
        let org = parts.first.map(String.init) ?? slug
        let name = parts.count > 1 ? String(parts[1]) : ""
        return HfRepo(org: HfOrg(validated: org), repoName: HfRepoName(validated: name))
    }

    nonisolated private static func trustedPrefix(_ prefix: String?) -> RepoSubpath? {
        prefix.map { RepoSubpath(validated: $0) }
    }

    // Sizes are each GGUF file's exact bytes, from the HF file tree.
    nonisolated static let defaults: [CatalogEntry] = [
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_0", bytes: 149_080_928),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q4_K_M", bytes: 153_406_304),
        .gguf("LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF", "Q5_K_M", bytes: 171_625_312),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_0", bytes: 219_309_792),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_K_M", bytes: 229_312_224),
        .gguf("LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF", "Q5_K_M", bytes: 260_376_288),
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q4_0", bytes: 507_154_688),
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q4_K_M", bytes: 532_517_120),
        .gguf("Qwen 3.5 0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Q5_K_M", bytes: 590_057_728),
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q4_0", bytes: 1_214_873_856),
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q4_K_M", bytes: 1_280_835_840),
        .gguf("Qwen 3.5 2B", "unsloth/Qwen3.5-2B-GGUF", "Q5_K_M", bytes: 1_435_238_656),
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q4_0", bytes: 2_583_221_408),
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q4_K_M", bytes: 2_740_937_888),
        .gguf("Qwen 3.5 4B", "unsloth/Qwen3.5-4B-GGUF", "Q5_K_M", bytes: 3_143_656_608),
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q4_0", bytes: 216_073_120),
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q4_K_M", bytes: 222_662_560),
        .gguf("Granite 4.0-h 350M", "ibm-granite/granite-4.0-h-350m-GGUF", "Q5_K_M", bytes: 252_331_936),
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q4_0", bytes: 868_316_384),
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q4_K_M", bytes: 901_162_208),
        .gguf("Granite 4.0-h 1B", "ibm-granite/granite-4.0-h-1b-GGUF", "Q5_K_M", bytes: 1_048_556_768),
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q4_0", bytes: 1_855_516_320),
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q4_K_M", bytes: 1_942_564_512),
        .gguf("Granite 4.0-h Micro", "ibm-granite/granite-4.0-h-micro-GGUF", "Q5_K_M", bytes: 2_273_455_776),
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q4_0", bytes: 228_470_176),
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q4_K_M", bytes: 236_985_760),
        .gguf("Granite 4.0 350M", "ibm-granite/granite-4.0-350m-GGUF", "Q5_K_M", bytes: 264_052_128),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_0", bytes: 695_751_488),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q4_K_M", bytes: 730_895_168),
        .gguf("LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "Q5_K_M", bytes: 843_354_944),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_0", bytes: 4_844_678_368),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q4_K_M", bytes: 5_155_564_768),
        .gguf("LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF", "Q5_K_M", bytes: 6_030_339_296),
        .gguf("Ministral 3 3B Instruct 2512", "unsloth/Ministral-3-3B-Instruct-2512-GGUF", "Q4_0", bytes: 2_046_375_200),
        .gguf("Ministral 3 3B Instruct 2512", "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Q4_K_M", bytes: 2_147_023_008),
        .gguf("Ministral 3 3B Instruct 2512", "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Q5_K_M", bytes: 2_474_178_720),
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q4_0", bytes: 773_025_824),
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q4_K_M", bytes: 807_694_368),
        .gguf("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", "Q5_K_M", bytes: 911_503_392),
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q4_0", bytes: 1_921_909_184),
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q4_K_M", bytes: 2_019_377_600),
        .gguf("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Q5_K_M", bytes: 2_322_153_920),
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q4_0", bytes: 3_041_378_400),
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q4_K_M", bytes: 3_106_738_272),
        .gguf("Gemma 4 E2B IT", "unsloth/gemma-4-E2B-it-GGUF", "Q5_K_M", bytes: 3_356_037_216),
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q4_0", bytes: 4_836_002_944),
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q4_K_M", bytes: 4_977_171_584),
        .gguf("Gemma 4 E4B IT", "unsloth/gemma-4-E4B-it-GGUF", "Q5_K_M", bytes: 5_481_798_784),
        // Admission rule (PIP-511), and the whole rule: a GGUF row is offered exactly when
        // the results dashboard publishes it for this phone's llama.cpp engine:
        //
        //   https://pipette.liquid.ai/results?hw=iPhone+17+Pro+%2F+llama.cpp%3Aios
        //
        // Re-derive the list from the link. `ci/checks/model-catalog-parity.py` declares
        // where it differs from Android's. The rows below carry one quant because the
        // dashboard publishes only `Q4_K_M` for them.
        .gguf("MiniCPM5 1B", "openbmb/MiniCPM5-1B-GGUF", "Q4_K_M", bytes: 688_065_920),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q4_0", bytes: 1_593_894_912),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q4_K_M", bytes: 1_674_455_040),
        .gguf("LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF", "Q5_K_M", bytes: 1_939_744_768),
        .gguf("G9v3 3B", "bartowski/ai9stars_G9v3-3B-GGUF", "Q4_K_M", bytes: 1_902_590_528),
        .gguf("Nanbeige 4.2 3B", "bartowski/Nanbeige_Nanbeige4.2-3B-GGUF", "Q4_K_M", bytes: 2_684_023_968),
        .gguf("Olmo 3 7B Think", "unsloth/Olmo-3-7B-Think-GGUF", "Q4_K_M", bytes: 4_471_600_960),
        .gguf("Ling 3.0 Tiny", "bartowski/Ling-3.0-tiny-GGUF", "Q4_K_M", bytes: 4_917_354_688),
        .gguf("Falcon H1R 7B", "tiiuae/Falcon-H1R-7B-GGUF", "Q4_K_M", bytes: 4_598_343_712),
        .gguf("Ministral 3 8B Instruct 2512", "mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Q4_K_M", bytes: 5_198_911_904),
        .gguf("Granite 4.1 8B", "ibm-granite/granite-4.1-8b-GGUF", "Q4_K_M", bytes: 5_347_914_400),
        .gguf("Qwen 3.5 9B", "unsloth/Qwen3.5-9B-GGUF", "Q4_K_M", bytes: 5_680_522_464),
        .gguf("Ornith 1.0 9B", "unsloth/Ornith-1.0-9B-GGUF", "Q4_K_M", bytes: 5_701_067_872),
    ]

    // MARK: - MLX preset catalog (gated)
    //
    // Only models the project benchmarks on MLX (`type = "mlx"` in the plan files). The
    // bit-width is in the repo slug, since an MLX model is a repo directory. Sizes are each
    // repo's total bytes from the HF file tree.
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
        .mlx("Ternary Bonsai 1.7B", "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit", bytes: 495_636_905, quant: "2bit"),
        .mlx("Ternary Bonsai 8B", "prism-ml/Ternary-Bonsai-8B-mlx-2bit", bytes: 2_315_264_643, quant: "2bit"),
    ]

    // MARK: - ONNX Runtime GenAI preset catalog
    //
    // No device results exist yet for `onnx_genai_ios_pipette`, so the rule is PIP-589's
    // for the host: a genuine ONNX Runtime GenAI export (`genai_config.json` beside the
    // graph) that the host runtime has benchmarked. Switch to the dashboard filter once
    // device results exist.
    //
    // No `LiquidAI/*` row: those `-ONNX` repos are transformers.js exports with no
    // `genai_config.json`, which `OnnxGenaiModels` refuses.
    //
    // Sizes are the selected subtree's bytes, not the whole repo's.
    nonisolated static let onnxDefaults: [CatalogEntry] = [
        .onnxGenai("Phi 3 Mini 4K Instruct", "microsoft/Phi-3-mini-4k-instruct-onnx",
                   prefix: "cpu_and_mobile/cpu-int4-rtn-block-32",
                   revision: "5f5f794c1c23c9d5ee142af85df02a6cc52d6945",
                   bytes: 2_725_538_838),
    ]
}
