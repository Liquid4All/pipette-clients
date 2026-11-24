package ai.liquid.pipette

import java.net.URL

/**
 * One model the device offers for download: a row of the catalog short list ([ModelShortList]) or of [ModelTemplateCatalog.fallback].
 *
 * [identifier] is any input [DownloadCoordinator.parseDownloadInput] accepts. A catalog row uses `<org>/<repo>[@<revision>]/<file>`, so the download
 * fetches the commit the catalog pins. [sizeBytes] is the download size the catalog records, or null when it records none. [variant] tells this row
 * apart from another row of the same entry and quant, such as `bartowski` or `20260922`. It is null when the entry offers the quant once.
 * [quantLabel] is the quant the catalog records for the member, in upper case. A file name such as `Model.Q4_K_M.gguf` or `Model-PQ2_0.gguf` does not
 * give its quant to [LocalStorage.parseQuant]. [family] is the catalog family of the entry, such as `LFM2.5`. [familyLogoDigest] is the SHA-256
 * digest that names the family's logo, or null when the catalog gives none.
 */
data class PresetModel(
  val id: String,
  val name: String,
  val identifier: String,
  val sizeBytes: Long?,
  val variant: String? = null,
  val quantLabel: String? = null,
  val family: String = ModelFamily.defaultFamily(name),
  val familyLogoDigest: String? = null,
) {
  val parsed: ParsedDownload
    get() = DownloadCoordinator.parseDownloadInput(identifier)

  val filename: String
    get() = URL(parsed.url).path.substringAfterLast('/')

  val repoIdentifier: String?
    get() = parsed.repo

  val quant: String?
    get() = quantLabel ?: LocalStorage.parseQuant(filename)

  val modelId: String
    get() = LocalStorage.normalizedModelStem(filename)

  /** Typed plan-types coordinate for this preset, when both the repo slug and filename are valid. */
  val ggufModel: HfGgufText?
    get() {
      val repo = repoIdentifier?.let { HfRepo.parseReferenceOrNull(it) } ?: return null
      val file = GgufFilename.parseOrNull(filename) ?: return null
      return HfGgufText(repo, file)
    }

  /** The download size as the app shows it, or empty when the size is unknown. */
  val sizeLabel: String
    get() = sizeBytes?.let { ByteFormat.fileSize(it) }.orEmpty()

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
      // The declared download size, so the picker can show a size before anything is fetched. Zero when the catalog records none.
      sizeBytes = sizeBytes ?: 0L,
      hfRepo = ref.repo.toString(),
      displayName = name,
      modelId = modelId,
      modelRef = Model.GgufText(ref),
      onDisk = false,
      catalogQuant = quantLabel,
    )
  }
}

data class ModelVariant(val quant: String, val repo: String, val sizeLabel: String)

data class CatalogModel(val id: String, val displayName: String, val variants: List<ModelVariant>)

object ModelTemplateCatalog {
  /** The catalog short list or the bundled rows that fit on this device. */
  val current: List<PresetModel>
    get() = ModelShortList.offered.ifEmpty { deviceFallback }

  /** Display name by bare repo slug, the form [ModelFile.hfRepo] and a cell's `modelName` use. */
  val repoToName: Map<String, String>
    get() = snapshot().repoToName

  val models: List<CatalogModel>
    get() = snapshot().models

  val byModelId: Map<String, CatalogModel>
    get() = snapshot().byModelId

  /**
   * The rows to download for [selected]: one per download key. Two catalog entries can name the same file at the same revision. The row with the
   * larger known size wins, so the download total and its warning do not undercount.
   */
  fun downloads(selected: List<PresetModel>): List<PresetModel> =
    selected.groupBy { it.downloadKey }.values.map { rows -> rows.maxBy { it.sizeBytes ?: 0L } }

  /** Presets and lookups derived from the same catalog list. */
  internal class Snapshot(val presets: List<PresetModel>) {
    val repoToName: Map<String, String> = presets.mapNotNull { preset -> preset.ggufModel?.repo?.toString()?.let { it to preset.name } }.toMap()

    private val logoByFamily: Map<String, String> =
      presets.mapNotNull { preset -> preset.familyLogoDigest?.let { preset.family.lowercase() to it } }.toMap()

    /** Find the logo digest for [family] without regard to case. */
    fun familyLogoDigest(family: String): String? = logoByFamily[family.lowercase()]

    /** The catalog family by [PresetModel.modelId]. */
    val familyByModelId: Map<String, String>
    val models: List<CatalogModel>

    init {
      val variantsById = linkedMapOf<String, MutableList<ModelVariant>>()
      val representativeById = mutableMapOf<String, PresetModel>()
      val fallbackFamilyById = mutableMapOf<String, String>()
      presets.forEach { preset ->
        val id = preset.modelId
        fallbackFamilyById.putIfAbsent(id, preset.family)
        val repo = preset.repoIdentifier ?: return@forEach
        val quant = preset.quant ?: return@forEach
        variantsById.getOrPut(id) { mutableListOf() } += ModelVariant(quant = quant, repo = repo, sizeLabel = preset.sizeLabel)
        representativeById.putIfAbsent(id, preset)
      }
      models = variantsById.map { (id, variants) -> CatalogModel(id = id, displayName = representativeById.getValue(id).name, variants = variants) }
      familyByModelId = fallbackFamilyById + representativeById.mapValues { (_, preset) -> preset.family }
    }

    val byModelId: Map<String, CatalogModel> = models.associateBy { it.id }
  }

  @Volatile private var cachedSnapshot: Snapshot? = null

  internal fun snapshot(): Snapshot {
    val presets = current
    return cachedSnapshot?.takeIf { it.presets === presets } ?: Snapshot(presets).also { cachedSnapshot = it }
  }

  /**
   * The LFM models the app offers when neither the short list nor its cache offers one: the GGUF rows from the `LiquidAI` organization. Sizes are
   * each file's exact bytes from the Hugging Face file tree, the same figures as the iOS fallback list. The names have no hyphen, so each row names
   * its family.
   */
  @Suppress("MagicNumber") // each row's exact download size in bytes
  val fallback: List<PresetModel> =
    listOf(
      PresetModel("lfm2.5-230m-q4_0", "LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF:Q4_0", 149_080_928L, family = "LFM2.5"),
      PresetModel("lfm2.5-230m-q4_k_m", "LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF:Q4_K_M", 153_406_304L, family = "LFM2.5"),
      PresetModel("lfm2.5-230m-q5_k_m", "LFM 2.5 230M", "LiquidAI/LFM2.5-230M-GGUF:Q5_K_M", 171_625_312L, family = "LFM2.5"),
      PresetModel("lfm2.5-350m-q4_0", "LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF:Q4_0", 219_309_792L, family = "LFM2.5"),
      PresetModel("lfm2.5-350m-q4_k_m", "LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M", 229_312_224L, family = "LFM2.5"),
      PresetModel("lfm2.5-350m-q5_k_m", "LFM 2.5 350M", "LiquidAI/LFM2.5-350M-GGUF:Q5_K_M", 260_376_288L, family = "LFM2.5"),
      PresetModel("lfm2.5-1.2b-instruct-q4_0", "LFM 2.5 1.2B Instruct", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_0", 695_751_488L, family = "LFM2.5"),
      PresetModel(
        "lfm2.5-1.2b-instruct-q4_k_m",
        "LFM 2.5 1.2B Instruct",
        "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M",
        730_895_168L,
        family = "LFM2.5",
      ),
      PresetModel(
        "lfm2.5-1.2b-instruct-q5_k_m",
        "LFM 2.5 1.2B Instruct",
        "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q5_K_M",
        843_354_944L,
        family = "LFM2.5",
      ),
      PresetModel("lfm2.5-8b-a1b-q4_0", "LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_0", 4_844_678_368L, family = "LFM2.5"),
      PresetModel("lfm2.5-8b-a1b-q4_k_m", "LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M", 5_155_564_768L, family = "LFM2.5"),
      PresetModel("lfm2.5-8b-a1b-q5_k_m", "LFM 2.5 8B A1B", "LiquidAI/LFM2.5-8B-A1B-GGUF:Q5_K_M", 6_030_339_296L, family = "LFM2.5"),
      PresetModel("lfm2.5-2.6b-q4_0", "LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF:Q4_0", 1_593_894_912L, family = "LFM2.5"),
      PresetModel("lfm2.5-2.6b-q4_k_m", "LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M", 1_674_455_040L, family = "LFM2.5"),
      PresetModel("lfm2.5-2.6b-q5_k_m", "LFM 2.5 2.6B", "LiquidAI/LFM2.5-2.6B-GGUF:Q5_K_M", 1_939_744_768L, family = "LFM2.5"),
    )

  @Volatile private var deviceFallback = fallback

  internal fun limitFallbackTo(ramBytes: Long) {
    deviceFallback = fallback.filter { (it.sizeBytes ?: 0L) <= ramBytes }
  }
}
