// Per-screen VM with the usual derive/intent helper spread (TooManyFunctions) + size-threshold literals (MagicNumber).
@file:Suppress("MagicNumber", "TooManyFunctions", "CyclomaticComplexMethod")

package ai.liquid.pipette.compose.models

import ai.liquid.pipette.ActiveDownload
import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.LocalStorage
import ai.liquid.pipette.ModelCatalog
import ai.liquid.pipette.ModelFamily
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.ModelShortList
import ai.liquid.pipette.ModelTemplateCatalog
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.PresetModel
import ai.liquid.pipette.compose.AddModelUi
import ai.liquid.pipette.compose.AddQuantChipUi
import ai.liquid.pipette.compose.AddQuantSectionUi
import ai.liquid.pipette.compose.DownloadedGroupUi
import ai.liquid.pipette.compose.Effect
import ai.liquid.pipette.compose.ModelFamilyUi
import ai.liquid.pipette.compose.ModelRowUi
import ai.liquid.pipette.compose.ScreenViewModel
import ai.liquid.pipette.compose.matchesSearch
import ai.liquid.pipette.compose.shell.ShellViewModel
import android.app.Application
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Downloaded-model management (grouped by catalog family) + the Add-models download cover (catalog rows by family, each with its quant chips). */
data class ModelsUiState(
  val hasAnyModel: Boolean = false,
  val searchQuery: String = "",
  val matched: Boolean = true,
  val downloadedFamilies: List<ModelFamilyUi<DownloadedGroupUi>> = emptyList(),
  val mmprojs: List<ModelRowUi> = emptyList(),
  val activeDownloads: List<ActiveDownload> = emptyList(),
  val addModelsOpen: Boolean = false,
  val addSearch: String = "",
  val addFamilies: List<ModelFamilyUi<AddModelUi>> = emptyList(),
  val addDownloadCount: Int = 0,
  val addDownloadBytes: Long = 0,
  /** A selected model has no recorded size, so [addDownloadBytes] undercounts and the download asks for confirmation. */
  val addHasUnknownSize: Boolean = false,
  val largeDownloadWarningBytes: Long = LARGE_DOWNLOAD_WARNING_BYTES,
) {
  companion object {
    const val LARGE_DOWNLOAD_WARNING_BYTES = 200L * 1024L * 1024L
  }
}

sealed interface ModelsIntent {
  data class ApplyDownloadedSearch(val query: String) : ModelsIntent

  data object AddLocalModel : ModelsIntent

  data class CancelDownload(val key: String) : ModelsIntent

  data class PauseDownload(val key: String) : ModelsIntent

  data class ResumeDownload(val key: String) : ModelsIntent

  data class DeleteModelGroup(val files: List<ModelFile>) : ModelsIntent

  /** Opens or closes the downloaded-models family [name]. */
  data class ToggleDownloadedFamilyOpen(val name: String) : ModelsIntent

  data object OpenAddModels : ModelsIntent

  data object CloseAddModels : ModelsIntent

  data class ApplyAddSearch(val query: String) : ModelsIntent

  /** Shows or hides the quant chips of the Add models row [name]. */
  data class ToggleAddModelExpanded(val name: String) : ModelsIntent

  /** Select all available quants for [name]. If the user already selected all available quants, clear them. */
  data class ToggleAddModel(val name: String) : ModelsIntent

  /** Selects or clears the quant with the download key [key]. */
  data class ToggleAddQuant(val key: String) : ModelsIntent

  /** Opens or closes the Add models family [name]. */
  data class ToggleAddFamilyOpen(val name: String) : ModelsIntent

  /** Selects the quants of the visible rows of the Add models family [name], or clears them when all are selected. */
  data class ToggleAddFamily(val name: String) : ModelsIntent

  data object DownloadAddModels : ModelsIntent
}

class ModelsViewModel(app: Application, shell: ShellViewModel) : ScreenViewModel(app, shell) {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val secrets = container.secrets
  private val downloadCoordinator = container.downloadCoordinator

  private var downloadedModelSearchText = ""
  // The downloaded-models families closed by hand. Families start open, as the downloaded models are few. A new download reopens its family.
  private val closedDownloadedFamilies = mutableSetOf<String>()
  private var lastActiveDownloadKeys = emptySet<String>()
  private var addSearchText = ""
  private var addModelsOpen = false
  // Download keys (see [PresetModel.downloadKey]) of the selected Add models quants. All offers of one file share a selection.
  private val selectedAddKeys = linkedSetOf<String>()
  // The Add models families opened by hand. A search shows every family with a match open as well.
  private val openAddFamilies = mutableSetOf<String>()
  // The Add models rows whose quant chips show.
  private val expandedAddModels = mutableSetOf<String>()

  // The rows that the Add models search leaves visible, grouped by family. A row is the presets of one catalog entry, in catalog order.
  private fun visibleAddFamilies(catalog: List<PresetModel>): List<ModelFamily<Pair<String, List<PresetModel>>>> {
    val query = addSearchText.trim().lowercase()
    val rows = catalog.groupBy { it.name }.toList().filter { (name, presets) -> query.isEmpty() || matchesAddSearch(name, presets, query) }
    return ModelFamily.grouping(rows, family = { it.second.first().family }, name = { it.first })
  }

  /** Return true if [query] matches the entry name, family, repo, quant, or variant. */
  private fun matchesAddSearch(name: String, presets: List<PresetModel>, query: String): Boolean =
    (listOf(name, presets.first().family) + presets.flatMap { listOfNotNull(it.repoIdentifier, it.quant, it.variant) }).any {
      it.lowercase().contains(query)
    }

  /** The download keys of [presets] that are neither on the device nor downloading. */
  private fun availableAddKeys(presets: List<PresetModel>, unavailable: Set<String>): Set<String> =
    presets.map { it.downloadKey }.filterNot { it in unavailable }.toSet()

  /** The download keys that the device holds or is downloading. */
  private fun unavailableAddKeys(): Set<String> = cachedModels().map { it.storageKey }.toSet() + downloadCoordinator.activeDownloads().map { it.key }

  /** Selects [keys], or clears them when all are selected. Returns true when it selected them. */
  private fun toggleAll(keys: Set<String>): Boolean =
    if (selectedAddKeys.containsAll(keys)) {
      selectedAddKeys.removeAll(keys)
      false
    } else {
      selectedAddKeys.addAll(keys)
      true
    }

  private val _state = MutableStateFlow(ModelsUiState())
  val state: StateFlow<ModelsUiState> = _state.asStateFlow()

  // Single-thread confinement: all intent handling, field mutation, disk I/O, and state
  // building run here — never on Main. Serialized, so the plain sets need no extra locking.
  private val confine = Dispatchers.Default.limitedParallelism(1)

  // Models-dir walk cached across a search-keystroke burst; cleared on every non-search intent
  // (handle()) and on every download callback (refreshFromCallback), so it never serves stale data.
  private var modelsCache: List<ai.liquid.pipette.ModelFile>? = null

  private fun cachedModels(): List<ai.liquid.pipette.ModelFile> = modelsCache ?: storage.availableModels().also { modelsCache = it }

  init {
    viewModelScope.launch(confine) { publish() }
    viewModelScope.launch(confine) {
      shell.dataChanges.collect {
        modelsCache = null
        publish()
      }
    }
    // Re-render when a refresh replaces the offered model list. StateFlow replays the current value, which costs one extra publish at start.
    viewModelScope.launch(confine) { ModelShortList.changes.collect { publish() } }
    // Re-render on ANY download-registry change, incl. progress and downloads resumed by
    // WorkManager after process death (whose session-scoped callbacks no longer exist).
    // Cache-preserving: in-flight progress doesn't change the downloaded-models dir.
    ai.liquid.pipette.DownloadRegistry.onChanged = { republish() }
  }

  override fun onCleared() {
    ai.liquid.pipette.DownloadRegistry.onChanged = null
    super.onCleared()
  }

  fun onIntent(intent: ModelsIntent) {
    viewModelScope.launch(confine) { handle(intent) }
  }

  private fun handle(intent: ModelsIntent) {
    // Search intents reuse the cached models list (instant filtering); others get fresh data.
    if (intent !is ModelsIntent.ApplyDownloadedSearch && intent !is ModelsIntent.ApplyAddSearch) modelsCache = null
    when (intent) {
      is ModelsIntent.ApplyDownloadedSearch -> {
        downloadedModelSearchText = intent.query
        publish()
      }
      ModelsIntent.AddLocalModel -> emit(Effect.PickModel)
      is ModelsIntent.CancelDownload -> {
        downloadCoordinator.cancel(intent.key)
        publish()
      }
      is ModelsIntent.PauseDownload -> {
        downloadCoordinator.pause(intent.key)
        publish()
      }
      is ModelsIntent.ResumeDownload -> {
        downloadCoordinator.resume(intent.key)
        publish()
      }
      is ModelsIntent.DeleteModelGroup -> {
        intent.files.forEach { storage.deleteModel(it) }
        shell.notifyDataChanged()
        publish()
      }
      is ModelsIntent.ToggleDownloadedFamilyOpen -> {
        if (!closedDownloadedFamilies.remove(intent.name)) closedDownloadedFamilies += intent.name
        publish()
      }
      ModelsIntent.OpenAddModels -> {
        addModelsOpen = true
        publish()
      }
      ModelsIntent.CloseAddModels -> {
        selectedAddKeys.clear()
        openAddFamilies.clear()
        expandedAddModels.clear()
        addModelsOpen = false
        publish()
      }
      is ModelsIntent.ApplyAddSearch -> {
        addSearchText = intent.query
        publish()
      }
      is ModelsIntent.ToggleAddModelExpanded -> {
        if (!expandedAddModels.remove(intent.name)) expandedAddModels += intent.name
        publish()
      }
      is ModelsIntent.ToggleAddModel -> {
        val presets = ModelTemplateCatalog.current.filter { it.name == intent.name }
        if (toggleAll(availableAddKeys(presets, unavailableAddKeys()))) expandedAddModels += intent.name
        publish()
      }
      is ModelsIntent.ToggleAddQuant -> {
        if (!selectedAddKeys.remove(intent.key) && intent.key !in unavailableAddKeys()) selectedAddKeys += intent.key
        publish()
      }
      is ModelsIntent.ToggleAddFamilyOpen -> {
        if (addSearchText.isBlank() && !openAddFamilies.remove(intent.name)) openAddFamilies += intent.name
        publish()
      }
      is ModelsIntent.ToggleAddFamily -> {
        // Only the rows the active search leaves visible, so the checkbox can't queue downloads the user can't see.
        val presets = visibleAddFamilies(ModelTemplateCatalog.current).firstOrNull { it.name == intent.name }?.members.orEmpty().flatMap { it.second }
        if (toggleAll(availableAddKeys(presets, unavailableAddKeys())) && addSearchText.isBlank()) openAddFamilies += intent.name
        publish()
      }
      ModelsIntent.DownloadAddModels -> downloadAddModels()
    }
  }

  /** Called by the host after a SAF document pick resolves. Copies the GGUF into app storage off-main, then re-derives. */
  fun onModelUriPicked(uri: android.net.Uri) {
    viewModelScope.launch {
      runCatching {
          withContext(Dispatchers.IO) {
            val name = displayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "model.gguf"
            require(name.endsWith(".gguf", ignoreCase = true)) { "Selected file must be .gguf" }
            storage.copyModelFromUri(uri, name)
          }
        }
        .onSuccess {
          shell.notifyDataChanged()
          refreshFromCallback()
        }
        .onFailure { showError(it.message ?: it.javaClass.simpleName) }
    }
  }

  private fun displayName(uri: android.net.Uri): String? {
    val resolver = getApplication<Application>().contentResolver
    return resolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
      if (cursor.moveToFirst()) cursor.getString(0) else null
    }
  }

  private fun selectedPresets(catalog: List<PresetModel>, unavailable: Set<String>): List<PresetModel> {
    if (selectedAddKeys.isEmpty()) return emptyList()
    return ModelTemplateCatalog.downloads(
      catalog.filter {
        val key = it.downloadKey
        key in selectedAddKeys && key !in unavailable
      }
    )
  }

  private fun downloadAddModels() {
    val selected = selectedPresets(ModelTemplateCatalog.current, unavailableAddKeys())
    if (selected.isEmpty()) {
      showError("Select at least one model")
      return
    }
    // Reset the selection on the confine thread (we're already on it) before kicking off the
    // background enqueue, so the selection set is never mutated off-confine (no CME with publish()).
    selectedAddKeys.clear()
    openAddFamilies.clear()
    expandedAddModels.clear()
    addModelsOpen = false
    publish()
    // Progress/completion surface per-row in the Active downloads list (state.activeDownloads),
    // driven by republish()/notifyDataChanged() below — no shared status message needed.
    runInBackground {
      selected.forEach { preset ->
        downloadCoordinator.enqueueDownload(
          urlString = preset.identifier,
          repo = preset.repoIdentifier,
          modelId = preset.modelId,
          displayName = preset.name,
          onProgress = { republish() },
          onComplete = {
            // notifyDataChanged() re-walks this VM's models dir too (via its dataChanges collector).
            shell.notifyDataChanged()
          },
          onFailure = { error ->
            postError(error)
            republish()
          },
        )
      }
    }
  }

  /** Re-derive, re-walking the models dir (structural change: file added/removed). */
  private fun refreshFromCallback() {
    viewModelScope.launch(confine) {
      modelsCache = null
      publish()
    }
  }

  /** Re-derive without re-walking the models dir (in-flight download progress only). */
  private fun republish() {
    viewModelScope.launch(confine) { publish() }
  }

  private fun publish() {
    val catalog = ModelTemplateCatalog.snapshot()
    val models = cachedModels()
    val query = downloadedModelSearchText.trim()
    val mmprojs = models.filter { it.isMmproj && it.matchesSearch(query) }
    val baseGroups =
      ModelCatalog.groups(models.filterNot { it.isMmproj }, catalog.byModelId).mapNotNull { group ->
        val family = ModelFamily.of(group.key, group.name, catalog.familyByModelId)
        if (family.contains(query, ignoreCase = true)) {
          group
        } else {
          val matchingFiles = group.files.filter { it.matchesSearch(query) }
          group.copy(files = matchingFiles).takeIf { matchingFiles.isNotEmpty() }
        }
      }

    val activeDownloads = downloadCoordinator.activeDownloads()
    activeDownloads
      .filterNot { it.key in lastActiveDownloadKeys }
      .forEach { closedDownloadedFamilies -= ModelFamily.of(LocalStorage.normalizedModelStem(it.filename), it.filename, catalog.familyByModelId) }
    lastActiveDownloadKeys = activeDownloads.map { it.key }.toSet()

    val downloadedFamilies =
      ModelFamily.grouping(baseGroups, family = { ModelFamily.of(it.key, it.name, catalog.familyByModelId) }, name = { it.name }).map { family ->
        ModelFamilyUi(
          name = family.name,
          logoName = family.members.first().name,
          logoDigest = catalog.familyLogoDigest(family.name),
          open = downloadedModelSearchText.isNotBlank() || family.name !in closedDownloadedFamilies,
          members =
            family.members.map { group ->
              DownloadedGroupUi(
                key = group.key,
                name = group.name,
                sizeLabel = ByteFormat.fileSize(group.files.sumOf { it.sizeBytes }),
                quantCount = group.files.size,
                quants = group.files.mapNotNull { it.quant?.uppercase() }.ifEmpty { group.files.map { it.name } },
                files = group.files,
              )
            },
        )
      }

    val searching = addSearchText.isNotBlank()
    val addPresets = if (addModelsOpen) catalog.presets else emptyList()
    val unavailable = (models.map { it.storageKey } + activeDownloads.map { it.key }).toSet()
    val addFamilies =
      visibleAddFamilies(addPresets).map { family ->
        val rows = family.members.map { (name, presets) -> addModelRow(name, presets, unavailable) }
        val available = family.members.flatMap { availableAddKeys(it.second, unavailable) }.toSet()
        val checked = available.isNotEmpty() && selectedAddKeys.containsAll(available)
        ModelFamilyUi(
          name = family.name,
          logoName = rows.first().name,
          logoDigest = catalog.familyLogoDigest(family.name),
          open = searching || family.name in openAddFamilies,
          members = rows,
          checked = checked,
          mixed = !checked && available.any { it in selectedAddKeys },
        )
      }
    val selected = selectedPresets(addPresets, unavailable)

    _state.value =
      ModelsUiState(
        hasAnyModel = models.isNotEmpty(),
        searchQuery = downloadedModelSearchText,
        matched = baseGroups.isNotEmpty() || mmprojs.isNotEmpty(),
        downloadedFamilies = downloadedFamilies,
        mmprojs = mmprojs.map { modelRow(it) },
        activeDownloads = activeDownloads,
        addModelsOpen = addModelsOpen,
        addSearch = addSearchText,
        addFamilies = addFamilies,
        addDownloadCount = selected.size,
        addDownloadBytes = selected.sumOf { it.sizeBytes ?: 0L },
        addHasUnknownSize = selected.any { it.sizeBytes == null },
      )
  }

  /** One Add models row: the entry [name] and the quant chips of its [presets], in one section per repo. */
  private fun addModelRow(name: String, presets: List<PresetModel>, unavailable: Set<String>): AddModelUi {
    val available = availableAddKeys(presets, unavailable)
    val selectedCount = available.count { it in selectedAddKeys }
    val checked = available.isNotEmpty() && selectedCount == available.size
    val sections =
      presets
        .groupBy { it.ggufModel?.repo?.toString() ?: it.repoIdentifier.orEmpty() }
        .map { (repo, rows) ->
          // Two rows of one quant in a repo differ by revision, so their chips name the variant.
          val repeated = rows.groupingBy { it.quant }.eachCount().filterValues { it > 1 }.keys
          AddQuantSectionUi(
            label = repo,
            chips =
              rows.map { preset ->
                val quant = preset.quant ?: preset.filename
                val key = preset.downloadKey
                val label =
                  if (preset.quant in repeated) {
                    val revision = preset.variant ?: preset.ggufModel?.repo?.revision?.value ?: "default"
                    "$quant ($revision)"
                  } else quant
                AddQuantChipUi(
                  key = key,
                  label = label,
                  sizeLabel = preset.sizeLabel,
                  selected = key in selectedAddKeys,
                  onDevice = key in unavailable,
                )
              },
          )
        }
    return AddModelUi(
      name = name,
      subtitle = if (selectedCount > 0) "$selectedCount selected" else "GGUF",
      expanded = name in expandedAddModels,
      checked = checked,
      mixed = !checked && selectedCount > 0,
      sections = sections,
    )
  }

  private fun modelRow(model: ModelFile): ModelRowUi =
    ModelRowUi(model = model, title = model.displayName ?: model.name, subtitle = "${model.hfRepo ?: "sideloaded"} · ${model.sizeFormatted}")
}
