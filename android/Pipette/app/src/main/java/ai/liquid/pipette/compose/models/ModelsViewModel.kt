// Per-screen VM with the usual derive/intent helper spread (TooManyFunctions) + size-threshold literals (MagicNumber).
@file:Suppress("MagicNumber", "TooManyFunctions", "CyclomaticComplexMethod")

package ai.liquid.pipette.compose.models

import ai.liquid.pipette.ActiveDownload
import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.JobQuantFilter
import ai.liquid.pipette.LocalStorage
import ai.liquid.pipette.ModelCatalog
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.ModelTemplateCatalog
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.PresetModel
import ai.liquid.pipette.compose.AddModelGroupUi
import ai.liquid.pipette.compose.DownloadedGroupUi
import ai.liquid.pipette.compose.Effect
import ai.liquid.pipette.compose.ModelRowUi
import ai.liquid.pipette.compose.QuantChipUi
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

/** Downloaded-model management (grouped by family) + the Add-models download cover (family selection + quant pills). */
data class ModelsUiState(
  val hasAnyModel: Boolean = false,
  val searchQuery: String = "",
  val matched: Boolean = true,
  val downloadedGroups: List<DownloadedGroupUi> = emptyList(),
  val mmprojs: List<ModelRowUi> = emptyList(),
  val activeDownloads: List<ActiveDownload> = emptyList(),
  val addModelsOpen: Boolean = false,
  val addSearch: String = "",
  val addGroups: List<AddModelGroupUi> = emptyList(),
  val addAllSelected: Boolean = false,
  val addQuantPills: List<QuantChipUi> = emptyList(),
  val addDownloadCount: Int = 0,
  val addDownloadBytes: Long = 0,
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

  data object OpenAddModels : ModelsIntent

  data object CloseAddModels : ModelsIntent

  data class ApplyAddSearch(val query: String) : ModelsIntent

  data class ToggleAddGroup(val id: String, val checked: Boolean) : ModelsIntent

  data object ToggleAddSelectAll : ModelsIntent

  data class ToggleAddQuant(val filter: JobQuantFilter, val checked: Boolean) : ModelsIntent

  data object DownloadAddModels : ModelsIntent
}

class ModelsViewModel(app: Application, shell: ShellViewModel) : ScreenViewModel(app, shell) {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val secrets = container.secrets
  private val downloadCoordinator = container.downloadCoordinator

  private var downloadedModelSearchText = ""
  private var addSearchText = ""
  private var addModelsOpen = false
  private val selectedFamilies = linkedSetOf<String>()
  private val selectedQuants = linkedSetOf(JobQuantFilter.ALL)

  // Presets grouped by display name, preserving catalog order. Replaces the removed
  // ModelTemplateCatalog.orderedGroups (the catalog now exposes families/variants instead).
  private val orderedFamilies: List<Pair<String, List<PresetModel>>>
    get() = ModelTemplateCatalog.defaults.groupBy { it.name }.entries.map { it.key to it.value }

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
      ModelsIntent.OpenAddModels -> {
        addModelsOpen = true
        publish()
      }
      ModelsIntent.CloseAddModels -> {
        // Drop the family selection so it doesn't linger into the next open (matches legacy
        // close/back behavior); we're on the confine thread, so no CME with publish().
        selectedFamilies.clear()
        addModelsOpen = false
        publish()
      }
      is ModelsIntent.ApplyAddSearch -> {
        addSearchText = intent.query
        publish()
      }
      is ModelsIntent.ToggleAddGroup -> {
        if (intent.checked) selectedFamilies += intent.id else selectedFamilies -= intent.id
        publish()
      }
      ModelsIntent.ToggleAddSelectAll -> {
        // Operate only on families the active search leaves visible, so Select all can't queue
        // downloads for families the user can't currently see.
        val query = addSearchText.trim().lowercase()
        val visible = orderedFamilies.map { it.first }.filter { query.isEmpty() || it.lowercase().contains(query) }
        if (selectedFamilies.containsAll(visible)) selectedFamilies.removeAll(visible.toSet()) else selectedFamilies.addAll(visible)
        publish()
      }
      is ModelsIntent.ToggleAddQuant -> {
        updateQuant(intent.filter, intent.checked)
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

  private fun selectedPresets(): List<PresetModel> {
    val downloadedKeys = cachedModels().map { LocalStorage.modelRelativePath(it.hfRepo, it.name) }.toSet()
    val activeKeys = downloadCoordinator.activeDownloads().map { it.key }.toSet()
    val unavailable = downloadedKeys + activeKeys
    return ModelTemplateCatalog.defaults.filter { preset ->
      preset.name in selectedFamilies && quantMatches(preset.quant) && !unavailable.contains(preset.downloadKey)
    }
  }

  private fun downloadAddModels() {
    val selected = selectedPresets()
    if (selected.isEmpty()) {
      showError("Select at least one model")
      return
    }
    // Reset the selection on the confine thread (we're already on it) before kicking off the
    // background enqueue, so the selection set is never mutated off-confine (no CME with publish()).
    selectedFamilies.clear()
    addModelsOpen = false
    publish()
    // Progress/completion surface per-row in the Active downloads list (state.activeDownloads),
    // driven by republish()/notifyDataChanged() below — no shared status message needed.
    runInBackground {
      selected.forEach { preset ->
        downloadCoordinator.enqueueDownload(
          urlString = preset.identifier,
          repo = preset.repoIdentifier,
          familyId = preset.familyId,
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

  private fun updateQuant(filter: JobQuantFilter, checked: Boolean) {
    if (checked) {
      if (filter == JobQuantFilter.ALL) {
        selectedQuants.clear()
        selectedQuants += JobQuantFilter.ALL
      } else {
        selectedQuants -= JobQuantFilter.ALL
        selectedQuants += filter
      }
    } else {
      selectedQuants -= filter
    }
    if (selectedQuants.isEmpty()) selectedQuants += JobQuantFilter.ALL
  }

  private fun quantMatches(quant: String?): Boolean = selectedQuants.contains(JobQuantFilter.ALL) || selectedQuants.any { it.matches(quant) }

  private fun publish() {
    val models = cachedModels()
    val filtered = models.filter { it.matchesSearch(downloadedModelSearchText) }
    val base = filtered.filterNot { it.isMmproj }
    val mmprojs = filtered.filter { it.isMmproj }

    val downloadedGroups =
      ModelCatalog.groups(base).map { group ->
        DownloadedGroupUi(
          key = group.key,
          name = group.name,
          sizeLabel = ByteFormat.fileSize(group.files.sumOf { it.sizeBytes }),
          quantCount = group.files.size,
          quants = group.files.mapNotNull { it.quant }.ifEmpty { group.files.map { it.name } },
          files = group.files,
        )
      }

    val allFamilies = orderedFamilies
    val addGroups =
      allFamilies
        .filter { (name, _) -> addSearchText.isBlank() || name.lowercase().contains(addSearchText.trim().lowercase()) }
        .map { (name, presets) ->
          AddModelGroupUi(id = name, name = name, sizeLabel = presets.firstOrNull()?.sizeLabel ?: "", checked = selectedFamilies.contains(name))
        }
    val selected = selectedPresets()

    _state.value =
      ModelsUiState(
        hasAnyModel = models.isNotEmpty(),
        searchQuery = downloadedModelSearchText,
        matched = filtered.isNotEmpty(),
        downloadedGroups = downloadedGroups,
        mmprojs = mmprojs.map { modelRow(it) },
        activeDownloads = downloadCoordinator.activeDownloads(),
        addModelsOpen = addModelsOpen,
        addSearch = addSearchText,
        addGroups = addGroups,
        addAllSelected = addGroups.isNotEmpty() && addGroups.all { selectedFamilies.contains(it.id) },
        addQuantPills = JobQuantFilter.entries.map { QuantChipUi(it, it.label, selectedQuants.contains(it)) },
        addDownloadCount = selected.size,
        addDownloadBytes = selected.sumOf { it.estimatedBytes },
      )
  }

  private fun modelRow(model: ModelFile): ModelRowUi =
    ModelRowUi(model = model, title = model.displayName ?: model.name, subtitle = "${model.hfRepo ?: "sideloaded"} · ${model.sizeFormatted}")
}
