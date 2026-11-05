package ai.liquid.pipette

import java.net.URL

// Decimal, because that is the scale the size labels are written in. See `estimatedBytes`.
private const val BYTES_PER_KB = 1_000.0
private const val BYTES_PER_MB = 1_000_000.0
private const val BYTES_PER_GB = 1_000_000_000.0

data class PresetModel(val id: String, val name: String, val detail: String, val identifier: String) {
  val parsed: ParsedDownload
    get() = DownloadCoordinator.parseDownloadInput(identifier)

  val filename: String
    get() = URL(parsed.url).path.substringAfterLast('/')

  val repoIdentifier: String?
    get() = parsed.repo

  val quant: String?
    get() = LocalStorage.parseQuant(filename)

  val familyId: String
    get() = LocalStorage.normalizedModelStem(filename)

  /** Typed plan-types coordinate for this preset, when both the repo slug and filename are valid. */
  val ggufModel: HfGgufText?
    get() {
      val repo = repoIdentifier?.let { HfRepo.parseSlugOrNull(it) } ?: return null
      val file = GgufFilename.parseOrNull(filename) ?: return null
      return HfGgufText(repo, file)
    }

  val sizeLabel: String
    get() = detail.substringAfter('·', detail).trim()

  val downloadKey: String
    get() = LocalStorage.modelRelativePath(repoIdentifier, filename)

  /**
   * This preset as something a run can be started against, or null when it has no coordinate to fetch from.
   *
   * [ModelFile] rather than a type of its own, because every consumer between the picker and `JobRunner.planCells` treats a catalog offer and a
   * downloaded file identically. [ModelFile.onDisk] records which one this is, and [ModelFile.path] is the destination the download will write, which
   * is the same relative path `resolveModelPath` will later find it under.
   *
   * Null when no coordinate comes out of [identifier]: without a valid repo slug and filename there is nothing the provisioner could fetch, so
   * offering it would only produce a run that fails at the first cell.
   *
   * The whole read is guarded, not just the null case. [ggufModel] reaches [parsed], and `parseDownloadInput` *throws* on an identifier it cannot
   * parse rather than returning null, so a single malformed catalog row would otherwise take down the run screen that lists it. A bad row should cost
   * that row and nothing else.
   */
  fun asOffer(): ModelFile? {
    val ref = runCatching { ggufModel }.getOrNull() ?: return null
    return ModelFile(
      name = filename,
      path = downloadKey,
      // The label's parsed figure, so the picker can show a size before anything is fetched. An estimate, and marked as one by `onDisk = false`.
      sizeBytes = estimatedBytes,
      hfRepo = repoIdentifier,
      displayName = name,
      familyId = familyId,
      modelRef = Model.GgufText(ref),
      onDisk = false,
    )
  }

  /**
   * The size label read back as bytes, for the download total and the storage pre-check.
   *
   * Decimal units, because that is what the labels are written in: every one is the string `ByteCountFormatter` with `countStyle = .file` produces
   * for that file's real byte count, so the two clients show a model the same size. Parsing them as 1024-based read every row about 7% high.
   */
  val estimatedBytes: Long
    get() {
      val label = sizeLabel.uppercase()
      val value = label.filter { it.isDigit() || it == '.' }.toDoubleOrNull() ?: 0.0
      val unit =
        when {
          label.contains("GB") -> BYTES_PER_GB
          label.contains("MB") -> BYTES_PER_MB
          label.contains("KB") -> BYTES_PER_KB
          else -> 1.0
        }
      return (value * unit).toLong()
    }
}

data class ModelVariant(val quant: String, val repo: String, val sizeLabel: String)

data class ModelFamily(val id: String, val displayName: String, val variants: List<ModelVariant>)

object ModelTemplateCatalog {
  val repoToName: Map<String, String> by lazy { defaults.mapNotNull { preset -> preset.repoIdentifier?.let { it to preset.name } }.toMap() }

  val families: List<ModelFamily> by lazy {
    val order = mutableListOf<String>()
    val seen = mutableSetOf<String>()
    val variantsById = linkedMapOf<String, MutableList<ModelVariant>>()
    val nameById = mutableMapOf<String, String>()

    defaults.forEach { preset ->
      val repo = preset.repoIdentifier ?: return@forEach
      val quant = preset.quant ?: return@forEach
      if (seen.add(preset.familyId)) order += preset.familyId
      variantsById.getOrPut(preset.familyId) { mutableListOf() } += ModelVariant(quant = quant, repo = repo, sizeLabel = preset.sizeLabel)
      nameById.putIfAbsent(preset.familyId, preset.name)
    }

    order.map { id -> ModelFamily(id = id, displayName = nameById[id] ?: id, variants = variantsById[id] ?: emptyList()) }
  }

  val byFamilyId: Map<String, ModelFamily> by lazy { families.associateBy { it.id } }

  val defaults: List<PresetModel> =
    listOf(
      PresetModel("lfm2.5-230m-q4_0", "LFM 2.5 230M", "Q4_0 · 149.1 MB", "LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
      PresetModel("lfm2.5-230m-q4_k_m", "LFM 2.5 230M", "Q4_K_M · 153.4 MB", "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-230m-q5_k_m", "LFM 2.5 230M", "Q5_K_M · 171.6 MB", "LiquidAI/LFM2.5-230M-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-350m-q4_0", "LFM 2.5 350M", "Q4_0 · 219.3 MB", "LiquidAI/LFM2.5-350M-GGUF:Q4_0"),
      PresetModel("lfm2.5-350m-q4_k_m", "LFM 2.5 350M", "Q4_K_M · 229.3 MB", "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-350m-q5_k_m", "LFM 2.5 350M", "Q5_K_M · 260.4 MB", "LiquidAI/LFM2.5-350M-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-0.8b-q4_0", "Qwen 3.5 0.8B", "Q4_0 · 507.2 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q4_0"),
      PresetModel("qwen3.5-0.8b-q4_k_m", "Qwen 3.5 0.8B", "Q4_K_M · 532.5 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-0.8b-q5_k_m", "Qwen 3.5 0.8B", "Q5_K_M · 590.1 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-2b-q4_0", "Qwen 3.5 2B", "Q4_0 · 1.21 GB", "unsloth/Qwen3.5-2B-GGUF:Q4_0"),
      PresetModel("qwen3.5-2b-q4_k_m", "Qwen 3.5 2B", "Q4_K_M · 1.28 GB", "unsloth/Qwen3.5-2B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-2b-q5_k_m", "Qwen 3.5 2B", "Q5_K_M · 1.44 GB", "unsloth/Qwen3.5-2B-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-4b-q4_0", "Qwen 3.5 4B", "Q4_0 · 2.58 GB", "unsloth/Qwen3.5-4B-GGUF:Q4_0"),
      PresetModel("qwen3.5-4b-q4_k_m", "Qwen 3.5 4B", "Q4_K_M · 2.74 GB", "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-4b-q5_k_m", "Qwen 3.5 4B", "Q5_K_M · 3.14 GB", "unsloth/Qwen3.5-4B-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-350m-q4_0", "Granite 4.0-h 350M", "Q4_0 · 216.1 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-350m-q4_k_m", "Granite 4.0-h 350M", "Q4_K_M · 222.7 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-350m-q5_k_m", "Granite 4.0-h 350M", "Q5_K_M · 252.3 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-1b-q4_0", "Granite 4.0-h 1B", "Q4_0 · 868.3 MB", "ibm-granite/granite-4.0-h-1b-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-1b-q4_k_m", "Granite 4.0-h 1B", "Q4_K_M · 901.2 MB", "ibm-granite/granite-4.0-h-1b-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-1b-q5_k_m", "Granite 4.0-h 1B", "Q5_K_M · 1.05 GB", "ibm-granite/granite-4.0-h-1b-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-micro-q4_0", "Granite 4.0-h Micro", "Q4_0 · 1.86 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-micro-q4_k_m", "Granite 4.0-h Micro", "Q4_K_M · 1.94 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-micro-q5_k_m", "Granite 4.0-h Micro", "Q5_K_M · 2.27 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-350m-q4_0", "Granite 4.0 350M", "Q4_0 · 228.5 MB", "ibm-granite/granite-4.0-350m-GGUF:Q4_0"),
      PresetModel("granite-4.0-350m-q4_k_m", "Granite 4.0 350M", "Q4_K_M · 237 MB", "ibm-granite/granite-4.0-350m-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-350m-q5_k_m", "Granite 4.0 350M", "Q5_K_M · 264.1 MB", "ibm-granite/granite-4.0-350m-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-1.2b-instruct-q4_0", "LFM 2.5 1.2B Instruct", "Q4_0 · 695.8 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_0"),
      PresetModel("lfm2.5-1.2b-instruct-q4_k_m", "LFM 2.5 1.2B Instruct", "Q4_K_M · 730.9 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-1.2b-instruct-q5_k_m", "LFM 2.5 1.2B Instruct", "Q5_K_M · 843.4 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-8b-a1b-q4_0", "LFM 2.5 8B A1B", "Q4_0 · 4.84 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_0"),
      PresetModel("lfm2.5-8b-a1b-q4_k_m", "LFM 2.5 8B A1B", "Q4_K_M · 5.16 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-8b-a1b-q5_k_m", "LFM 2.5 8B A1B", "Q5_K_M · 6.03 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q5_K_M"),
      PresetModel(
        "ministral-3-3b-instruct-2512-q4_0",
        "Ministral 3 3B Instruct 2512",
        "Q4_0 · 2.05 GB",
        "unsloth/Ministral-3-3B-Instruct-2512-GGUF:Q4_0",
      ),
      PresetModel(
        "ministral-3-3b-instruct-2512-q4_k_m",
        "Ministral 3 3B Instruct 2512",
        "Q4_K_M · 2.15 GB",
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M",
      ),
      PresetModel(
        "ministral-3-3b-instruct-2512-q5_k_m",
        "Ministral 3 3B Instruct 2512",
        "Q5_K_M · 2.47 GB",
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q5_K_M",
      ),
      PresetModel("llama-3.2-1b-instruct-q4_0", "Llama 3.2 1B Instruct", "Q4_0 · 773 MB", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_0"),
      PresetModel("llama-3.2-1b-instruct-q4_k_m", "Llama 3.2 1B Instruct", "Q4_K_M · 807.7 MB", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"),
      PresetModel("llama-3.2-1b-instruct-q5_k_m", "Llama 3.2 1B Instruct", "Q5_K_M · 911.5 MB", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q5_K_M"),
      PresetModel("llama-3.2-3b-instruct-q4_0", "Llama 3.2 3B Instruct", "Q4_0 · 1.92 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_0"),
      PresetModel("llama-3.2-3b-instruct-q4_k_m", "Llama 3.2 3B Instruct", "Q4_K_M · 2.02 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"),
      PresetModel("llama-3.2-3b-instruct-q5_k_m", "Llama 3.2 3B Instruct", "Q5_K_M · 2.32 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q5_K_M"),
      PresetModel("gemma-4-e2b-it-q4_0", "Gemma 4 E2B IT", "Q4_0 · 3.04 GB", "unsloth/gemma-4-E2B-it-GGUF:Q4_0"),
      PresetModel("gemma-4-e2b-it-q4_k_m", "Gemma 4 E2B IT", "Q4_K_M · 3.11 GB", "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
      PresetModel("gemma-4-e2b-it-q5_k_m", "Gemma 4 E2B IT", "Q5_K_M · 3.36 GB", "unsloth/gemma-4-E2B-it-GGUF:Q5_K_M"),
      PresetModel("gemma-4-e4b-it-q4_0", "Gemma 4 E4B IT", "Q4_0 · 4.84 GB", "unsloth/gemma-4-E4B-it-GGUF:Q4_0"),
      PresetModel("gemma-4-e4b-it-q4_k_m", "Gemma 4 E4B IT", "Q4_K_M · 4.98 GB", "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"),
      PresetModel("gemma-4-e4b-it-q5_k_m", "Gemma 4 E4B IT", "Q5_K_M · 5.48 GB", "unsloth/gemma-4-E4B-it-GGUF:Q5_K_M"),
      // Every row here is a configuration published for this phone on the results
      // dashboard, filtered to `Galaxy S26 Ultra / llama.cpp:android-arm64-v8a` (PIP-511):
      //
      //   https://pipette.liquid.ai/results?hw=Galaxy+S26+Ultra+%2F+llama.cpp%3Aandroid-arm64-v8a
      //
      // That filter is the admission rule, and it is the whole rule: a model absent from it
      // has no reading on this handset and is not offered, and one present is. Re-derive the
      // list by opening the link rather than by reasoning about swap or residency.
      //
      // The iPhone 17 Pro filter is the same rule on the other side, and the two lists differ
      // by exactly one row. `ci/checks/model-catalog-parity.py` names it and fails on any
      // divergence it does not.
      //
      // One quant apiece for the rows below, where the families above carry three: the
      // dashboard publishes `Q4_K_M` alone for them.
      PresetModel("minicpm5-1b-q4_k_m", "MiniCPM5 1B", "Q4_K_M · 688.1 MB", "openbmb/MiniCPM5-1B-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-2.6b-q4_0", "LFM 2.5 2.6B", "Q4_0 · 1.59 GB", "LiquidAI/LFM2.5-2.6B-GGUF:Q4_0"),
      PresetModel("lfm2.5-2.6b-q4_k_m", "LFM 2.5 2.6B", "Q4_K_M · 1.67 GB", "LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-2.6b-q5_k_m", "LFM 2.5 2.6B", "Q5_K_M · 1.94 GB", "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M"),
      PresetModel("g9v3-3b-q4_k_m", "G9v3 3B", "Q4_K_M · 1.9 GB", "bartowski/ai9stars_G9v3-3B-GGUF:Q4_K_M"),
      PresetModel("nanbeige4.2-3b-q4_k_m", "Nanbeige 4.2 3B", "Q4_K_M · 2.68 GB", "bartowski/Nanbeige_Nanbeige4.2-3B-GGUF:Q4_K_M"),
      PresetModel("falcon-h1r-7b-q4_k_m", "Falcon H1R 7B", "Q4_K_M · 4.6 GB", "tiiuae/Falcon-H1R-7B-GGUF:Q4_K_M"),
      PresetModel("ling-3.0-tiny-q4_k_m", "Ling 3.0 Tiny", "Q4_K_M · 4.92 GB", "bartowski/Ling-3.0-tiny-GGUF:Q4_K_M"),
      PresetModel(
        "ministral-3-8b-instruct-2512-q4_k_m",
        "Ministral 3 8B Instruct 2512",
        "Q4_K_M · 5.2 GB",
        "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
      ),
      PresetModel("granite-4.1-8b-q4_k_m", "Granite 4.1 8B", "Q4_K_M · 5.35 GB", "ibm-granite/granite-4.1-8b-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-9b-q4_k_m", "Qwen 3.5 9B", "Q4_K_M · 5.68 GB", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"),
      PresetModel("ornith-1.0-9b-q4_k_m", "Ornith 1.0 9B", "Q4_K_M · 5.7 GB", "unsloth/Ornith-1.0-9B-GGUF:Q4_K_M"),
    )
}
