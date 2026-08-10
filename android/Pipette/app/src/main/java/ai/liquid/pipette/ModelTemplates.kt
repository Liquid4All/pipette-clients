package ai.liquid.pipette

import java.net.URL

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

  val estimatedBytes: Long
    get() {
      val label = sizeLabel.uppercase()
      val value = label.filter { it.isDigit() || it == '.' }.toDoubleOrNull() ?: 0.0
      val unit =
        when {
          label.contains("GB") -> 1024.0 * 1024.0 * 1024.0
          label.contains("MB") -> 1024.0 * 1024.0
          label.contains("KB") -> 1024.0
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
      PresetModel("lfm2.5-230m-q4_0", "LFM 2.5 230M", "Q4_0 · 142.2 MB", "LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
      PresetModel("lfm2.5-230m-q4_k_m", "LFM 2.5 230M", "Q4_K_M · 146.3 MB", "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-230m-q5_k_m", "LFM 2.5 230M", "Q5_K_M · 163.7 MB", "LiquidAI/LFM2.5-230M-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-350m-q4_0", "LFM 2.5 350M", "Q4_0 · 228 MB", "LiquidAI/LFM2.5-350M-GGUF:Q4_0"),
      PresetModel("lfm2.5-350m-q4_k_m", "LFM 2.5 350M", "Q4_K_M · 267 MB", "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-350m-q5_k_m", "LFM 2.5 350M", "Q5_K_M · 279 MB", "LiquidAI/LFM2.5-350M-GGUF:Q5_K_M"),
      PresetModel("lfm2-700m-q4_0", "LFM2 700M", "Q4_0 · 446.3 MB", "LiquidAI/LFM2-700M-GGUF:Q4_0"),
      PresetModel("lfm2-700m-q4_k_m", "LFM2 700M", "Q4_K_M · 468.6 MB", "LiquidAI/LFM2-700M-GGUF:Q4_K_M"),
      PresetModel("lfm2-700m-q5_k_m", "LFM2 700M", "Q5_K_M · 538 MB", "LiquidAI/LFM2-700M-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-0.8b-q4_0", "Qwen 3.5 0.8B", "Q4_0 · 507 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q4_0"),
      PresetModel("qwen3.5-0.8b-q4_k_m", "Qwen 3.5 0.8B", "Q4_K_M · 539 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-0.8b-q5_k_m", "Qwen 3.5 0.8B", "Q5_K_M · 607 MB", "unsloth/Qwen3.5-0.8B-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-2b-q4_0", "Qwen 3.5 2B", "Q4_0 · 1.21 GB", "unsloth/Qwen3.5-2B-GGUF:Q4_0"),
      PresetModel("qwen3.5-2b-q4_k_m", "Qwen 3.5 2B", "Q4_K_M · 1.28 GB", "unsloth/Qwen3.5-2B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-2b-q5_k_m", "Qwen 3.5 2B", "Q5_K_M · 1.44 GB", "unsloth/Qwen3.5-2B-GGUF:Q5_K_M"),
      PresetModel("qwen3.5-4b-q4_0", "Qwen 3.5 4B", "Q4_0 · 2.58 GB", "unsloth/Qwen3.5-4B-GGUF:Q4_0"),
      PresetModel("qwen3.5-4b-q4_k_m", "Qwen 3.5 4B", "Q4_K_M · 2.74 GB", "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"),
      PresetModel("qwen3.5-4b-q5_k_m", "Qwen 3.5 4B", "Q5_K_M · 3.14 GB", "unsloth/Qwen3.5-4B-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-350m-q4_0", "Granite 4.0-h 350M", "Q4_0 · 202 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-350m-q4_k_m", "Granite 4.0-h 350M", "Q4_K_M · 216 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-350m-q5_k_m", "Granite 4.0-h 350M", "Q5_K_M · 243 MB", "ibm-granite/granite-4.0-h-350m-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-1b-q4_0", "Granite 4.0-h 1B", "Q4_0 · 868.3 MB", "ibm-granite/granite-4.0-h-1b-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-1b-q4_k_m", "Granite 4.0-h 1B", "Q4_K_M · 901.2 MB", "ibm-granite/granite-4.0-h-1b-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-1b-q5_k_m", "Granite 4.0-h 1B", "Q5_K_M · 1.05 GB", "ibm-granite/granite-4.0-h-1b-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-h-micro-q4_0", "Granite 4.0-h Micro", "Q4_0 · 1.86 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q4_0"),
      PresetModel("granite-4.0-h-micro-q4_k_m", "Granite 4.0-h Micro", "Q4_K_M · 1.94 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-h-micro-q5_k_m", "Granite 4.0-h Micro", "Q5_K_M · 2.27 GB", "ibm-granite/granite-4.0-h-micro-GGUF:Q5_K_M"),
      PresetModel("granite-4.0-350m-q4_0", "Granite 4.0 350M", "Q4_0 · 221 MB", "ibm-granite/granite-4.0-350m-GGUF:Q4_0"),
      PresetModel("granite-4.0-350m-q4_k_m", "Granite 4.0 350M", "Q4_K_M · 236 MB", "ibm-granite/granite-4.0-350m-GGUF:Q4_K_M"),
      PresetModel("granite-4.0-350m-q5_k_m", "Granite 4.0 350M", "Q5_K_M · 264 MB", "ibm-granite/granite-4.0-350m-GGUF:Q5_K_M"),
      PresetModel("gemma-3-1b-it-q4_0", "Gemma 3 1B IT", "Q4_0 · 620 MB", "unsloth/gemma-3-1b-it-GGUF:Q4_0"),
      PresetModel("gemma-3-1b-it-q4_k_m", "Gemma 3 1B IT", "Q4_K_M · 660 MB", "unsloth/gemma-3-1b-it-GGUF:Q4_K_M"),
      PresetModel("gemma-3-1b-it-q5_k_m", "Gemma 3 1B IT", "Q5_K_M · 733 MB", "unsloth/gemma-3-1b-it-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-1.2b-instruct-q4_0", "LFM 2.5 1.2B Instruct", "Q4_0 · 630 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_0"),
      PresetModel("lfm2.5-1.2b-instruct-q4_k_m", "LFM 2.5 1.2B Instruct", "Q4_K_M · 680 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-1.2b-instruct-q5_k_m", "LFM 2.5 1.2B Instruct", "Q5_K_M · 820 MB", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q5_K_M"),
      PresetModel("lfm2-2.6b-q4_0", "LFM2 2.6B", "Q4_0 · 1.48 GB", "LiquidAI/LFM2-2.6B-GGUF:Q4_0"),
      PresetModel("lfm2-2.6b-q4_k_m", "LFM2 2.6B", "Q4_K_M · 1.56 GB", "LiquidAI/LFM2-2.6B-GGUF:Q4_K_M"),
      PresetModel("lfm2-2.6b-q5_k_m", "LFM2 2.6B", "Q5_K_M · 1.83 GB", "LiquidAI/LFM2-2.6B-GGUF:Q5_K_M"),
      PresetModel("lfm2-2.6b-exp-q4_0", "LFM2 2.6B Exp", "Q4_0 · 1.35 GB", "LiquidAI/LFM2-2.6B-Exp-GGUF:Q4_0"),
      PresetModel("lfm2-2.6b-exp-q4_k_m", "LFM2 2.6B Exp", "Q4_K_M · 1.45 GB", "LiquidAI/LFM2-2.6B-Exp-GGUF:Q4_K_M"),
      PresetModel("lfm2-2.6b-exp-q5_k_m", "LFM2 2.6B Exp", "Q5_K_M · 1.75 GB", "LiquidAI/LFM2-2.6B-Exp-GGUF:Q5_K_M"),
      PresetModel("lfm2.5-8b-a1b-q4_0", "LFM 2.5 8B A1B", "Q4_0 · 4.51 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_0"),
      PresetModel("lfm2.5-8b-a1b-q4_k_m", "LFM 2.5 8B A1B", "Q4_K_M · 4.80 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M"),
      PresetModel("lfm2.5-8b-a1b-q5_k_m", "LFM 2.5 8B A1B", "Q5_K_M · 5.62 GB", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q5_K_M"),
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
      PresetModel("llama-3.2-1b-instruct-q4_k_m", "Llama 3.2 1B Instruct", "Q4_K_M · 808 MB", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"),
      PresetModel("llama-3.2-1b-instruct-q5_k_m", "Llama 3.2 1B Instruct", "Q5_K_M · 912 MB", "unsloth/Llama-3.2-1B-Instruct-GGUF:Q5_K_M"),
      PresetModel("llama-3.2-3b-instruct-q4_0", "Llama 3.2 3B Instruct", "Q4_0 · 1.92 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_0"),
      PresetModel("llama-3.2-3b-instruct-q4_k_m", "Llama 3.2 3B Instruct", "Q4_K_M · 2.02 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"),
      PresetModel("llama-3.2-3b-instruct-q5_k_m", "Llama 3.2 3B Instruct", "Q5_K_M · 2.32 GB", "unsloth/Llama-3.2-3B-Instruct-GGUF:Q5_K_M"),
      PresetModel("gemma-4-e2b-it-q4_0", "Gemma 4 E2B IT", "Q4_0 · 1.1 GB", "unsloth/gemma-4-E2B-it-GGUF:Q4_0"),
      PresetModel("gemma-4-e2b-it-q4_k_m", "Gemma 4 E2B IT", "Q4_K_M · 1.2 GB", "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"),
      PresetModel("gemma-4-e2b-it-q5_k_m", "Gemma 4 E2B IT", "Q5_K_M · 1.4 GB", "unsloth/gemma-4-E2B-it-GGUF:Q5_K_M"),
      PresetModel("gemma-4-e4b-it-q4_0", "Gemma 4 E4B IT", "Q4_0 · 2.2 GB", "unsloth/gemma-4-E4B-it-GGUF:Q4_0"),
      PresetModel("gemma-4-e4b-it-q4_k_m", "Gemma 4 E4B IT", "Q4_K_M · 2.4 GB", "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"),
      PresetModel("gemma-4-e4b-it-q5_k_m", "Gemma 4 E4B IT", "Q5_K_M · 2.9 GB", "unsloth/gemma-4-E4B-it-GGUF:Q5_K_M"),
      // Nemotron 3 Nano 4B — `nemotron_h`, a Mamba-2/attention hybrid. `unsloth` rather
      // than NVIDIA's own repo: the latter publishes one quant under a filename
      // (`NVIDIA-Nemotron3-Nano-4B-…`) that doesn't match its repo stem.
      PresetModel("nemotron-3-nano-4b-q4_0", "Nemotron 3 Nano 4B", "Q4_0 · 2.53 GB", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_0"),
      PresetModel("nemotron-3-nano-4b-q4_k_m", "Nemotron 3 Nano 4B", "Q4_K_M · 2.9 GB", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M"),
      PresetModel("nemotron-3-nano-4b-q5_k_m", "Nemotron 3 Nano 4B", "Q5_K_M · 3.16 GB", "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q5_K_M"),
      // Bonsai — `qwen35`/`qwen3` weights at 1-bit (`Q1_0`), the only quant these repos
      // publish. `PQ2_0` siblings are skipped: the leading `P` isn't a quant token, so
      // they'd land in their own family instead of grouping with `Q2_0`.
      PresetModel("bonsai-8b-q1_0", "Bonsai 8B", "Q1_0 · 1.16 GB", "prism-ml/Bonsai-8B-gguf:Q1_0"),
      PresetModel("bonsai-27b-q1_0", "Bonsai 27B", "Q1_0 · 3.8 GB", "prism-ml/Bonsai-27B-gguf:Q1_0"),
      // Ternary Bonsai — the 2-bit line. 27B is 7.2 GB, past every other row here, and
      // needs a high-memory device.
      PresetModel("ternary-bonsai-1.7b-q2_0", "Ternary Bonsai 1.7B", "Q2_0 · 463 MB", "prism-ml/Ternary-Bonsai-1.7B-gguf:Q2_0"),
      PresetModel("ternary-bonsai-8b-q2_0", "Ternary Bonsai 8B", "Q2_0 · 2.18 GB", "prism-ml/Ternary-Bonsai-8B-gguf:Q2_0"),
      PresetModel("ternary-bonsai-27b-q2_0", "Ternary Bonsai 27B", "Q2_0 · 7.17 GB", "prism-ml/Ternary-Bonsai-27B-gguf:Q2_0"),
    )
}
