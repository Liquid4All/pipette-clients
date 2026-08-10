// The Jobs VM is the largest screen by design: one MVI dispatcher (onIntent) over every job action, plus the verbatim planning/results
// derivation ported from the legacy JobsScreen (which is detekt-baselined the same way). Hence the structural + magic-number suppressions.
@file:Suppress("MagicNumber", "TooManyFunctions", "LargeClass", "LongMethod", "CyclomaticComplexMethod", "ReturnCount")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.BenchmarkCatalog
import ai.liquid.pipette.BenchmarkType
import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.CompletedResultsCsvExporter
import ai.liquid.pipette.CompletedRunMetric
import ai.liquid.pipette.DateFormats
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobQuantFilter
import ai.liquid.pipette.JobRunner
import ai.liquid.pipette.JobStatus
import ai.liquid.pipette.ModelCatalog
import ai.liquid.pipette.ModelGroup
import ai.liquid.pipette.NewJobWizard
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.ResultsGrid
import ai.liquid.pipette.compose.BenchmarkGroupUi
import ai.liquid.pipette.compose.BenchmarkItemUi
import ai.liquid.pipette.compose.CellUi
import ai.liquid.pipette.compose.Effect
import ai.liquid.pipette.compose.JobCardUi
import ai.liquid.pipette.compose.MmprojRowUi
import ai.liquid.pipette.compose.ModelGroupRowUi
import ai.liquid.pipette.compose.QuantChipUi
import ai.liquid.pipette.compose.ResultCellAccent
import ai.liquid.pipette.compose.ResultsCellUi
import ai.liquid.pipette.compose.ResultsGridUi
import ai.liquid.pipette.compose.ResultsRowUi
import ai.liquid.pipette.compose.RunProgress
import ai.liquid.pipette.compose.ScreenViewModel
import ai.liquid.pipette.compose.matchesSearch
import ai.liquid.pipette.compose.shell.ShellViewModel
import ai.liquid.pipette.plural
import ai.liquid.pipette.thermalAccentKind
import ai.liquid.pipette.thermalDescription
import ai.liquid.pipette.thermalHeadroomLabel
import android.app.Application
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** Run setup (the new-job wizard), the job list, and per-job detail (results heatmap + cells). */
sealed interface JobsUiState {
  data class JobList(
    val hasModels: Boolean = false,
    val searchQuery: String = "",
    val matched: Boolean = true,
    val anyJobs: Boolean = false,
    val jobs: List<JobCardUi> = emptyList(),
    // False when libpipette_android.so isn't packaged / the :benchmark engine can't bind: jobs can be
    // planned but every cell will fail, so the list surfaces an upfront warning (parity with legacy).
    val engineAvailable: Boolean = true,
  ) : JobsUiState

  data class Wizard(
    val step: Int,
    val stepTitles: List<String>,
    val modelSearch: String,
    val modelGroups: List<ModelGroupRowUi>,
    val anyModelGroups: Boolean,
    val modelsMatched: Boolean,
    val selectedBaseModelCount: Int,
    val quantFilters: List<QuantChipUi>,
    val benchmarkSearch: String,
    val benchmarkGroups: List<BenchmarkGroupUi>,
    val benchmarksMatched: Boolean,
    val showMmprojCard: Boolean,
    val mmprojs: List<MmprojRowUi>,
    val allMmprojSelected: Boolean,
    val nGpuLayers: Int,
    val contextSize: Int,
    val prefillBatch: Int,
    val contributeResults: Boolean,
    val isRegistered: Boolean,
    val reviewSummary: String,
    val reviewDate: String,
    val reviewSubtitle: String,
    val reviewModels: List<String>,
    val reviewBenchmarks: List<String>,
    val reviewQuants: List<String>,
    val showSkippedWarning: Boolean,
    val skippedWarning: String,
    val canAdvance: Boolean,
    val canRun: Boolean,
    val runLabel: String,
  ) : JobsUiState

  data class CellDetail(val cell: CellUi, val jobId: String, val isRunning: Boolean) : JobsUiState

  data class Detail(
    val manifest: JobManifest,
    val statusAccent: AccentKind,
    val titleDate: String,
    val subtitle: String,
    val modelChips: List<String>,
    val benchmarkChips: List<String>,
    val quantChips: List<String>,
    val gpuLayers: Int,
    val contextSize: Int,
    val createdLine: String,
    val runningHere: Boolean,
    val runProgress: Double,
    val runCellsDone: String,
    val runTimeLeft: String,
    // Live run indicators mirrored from Pocket Mode (only meaningful when runningHere).
    val runCellLabel: String,
    val runProgressText: String,
    val coolingSinceMillis: Long?,
    val thermalLabel: String,
    val thermalAccent: AccentKind,
    val contributeResults: Boolean,
    val isRegistered: Boolean,
    val isRunning: Boolean,
    val canResume: Boolean,
    val failedCells: Int,
    val completedCells: Int,
    val selectedRerunnableCount: Int,
    val unsubmittedCount: Int,
    val isSubmitting: Boolean,
    val resultsGrid: ResultsGridUi?,
    val cells: List<CellUi>,
  ) : JobsUiState
}

sealed interface JobsIntent {
  data object OpenWizard : JobsIntent

  data object CancelWizard : JobsIntent

  data class WizardGoToStep(val step: Int) : JobsIntent

  data class ApplyJobSearch(val query: String) : JobsIntent

  data class OpenJobDetail(val jobId: String) : JobsIntent

  data object BackToJobs : JobsIntent

  data class OpenCellDetail(val cellId: String) : JobsIntent

  data object CloseCellDetail : JobsIntent

  data class ApplyJobModelSearch(val query: String) : JobsIntent

  data class ToggleModelGroup(val key: String, val checked: Boolean) : JobsIntent

  data class ToggleQuantFilter(val filter: JobQuantFilter, val checked: Boolean) : JobsIntent

  data class ApplyBenchmarkSearch(val query: String) : JobsIntent

  data class ToggleBenchmark(val id: String, val checked: Boolean) : JobsIntent

  data class ToggleBenchmarkGroup(val type: String, val select: Boolean) : JobsIntent

  data class ToggleMmproj(val path: String, val checked: Boolean) : JobsIntent

  data class SetAllMmproj(val selected: Boolean) : JobsIntent

  data class SetWizardContribute(val enabled: Boolean) : JobsIntent

  data class SetRunSetup(val nGpuLayers: Int, val contextSize: Int, val prefillBatch: Int) : JobsIntent

  data class RunJob(val nGpuLayers: Int, val contextSize: Int, val prefillBatch: Int) : JobsIntent

  data object CancelRunningJob : JobsIntent

  data class SetJobAutoSubmit(val jobId: String, val enabled: Boolean) : JobsIntent

  data class RenameJob(val jobId: String, val title: String) : JobsIntent

  data class ResumeJob(val jobId: String) : JobsIntent

  data class RetryFailed(val jobId: String) : JobsIntent

  data class ExportCsv(val jobId: String) : JobsIntent

  data class RerunSelectedCells(val jobId: String) : JobsIntent

  data object ClearRerunSelection : JobsIntent

  data class ToggleCellRerun(val cellId: String, val checked: Boolean) : JobsIntent

  data class ToggleCellExpanded(val cellId: String) : JobsIntent

  data class RerunCell(val jobId: String, val cellId: String) : JobsIntent

  data class SubmitCellResult(val jobId: String, val cellId: String) : JobsIntent

  data class SubmitJobResults(val jobId: String) : JobsIntent

  data class DeleteJob(val jobId: String) : JobsIntent

  /** Open the Models tab (when there are no models to benchmark yet). */
  data object GoToModels : JobsIntent

  /** Open the full-screen pocket-mode overlay for a running job (a shell concern). */
  data class OpenPocketMode(val jobId: String) : JobsIntent
}

class JobsViewModel(app: Application, shell: ShellViewModel) : ScreenViewModel(app, shell) {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val submissionService = container.submissionService
  private val runner = container.jobController
  private val thermalProvider = container.thermalStatusProvider

  // Wizard + list + detail nav and selections (encapsulated here, not shared with other screens).
  private var newJobStep: Int? = null
  private var selectedJobId: String? = null
  private var selectedCellId: String? = null
  private val selectedModelKeys = linkedSetOf<String>()
  private val selectedBenchmarkIds = linkedSetOf<String>()
  private val selectedMmprojPaths = linkedSetOf<String>()
  private val selectedJobQuantFilters = linkedSetOf(JobQuantFilter.ALL)
  private val expandedCellIds = linkedSetOf<String>()
  private val selectedRerunCellIds = linkedSetOf<String>()
  // In-flight submit tracking, so the submit buttons show an in-button spinner (iOS parity) and can't
  // be re-tapped mid-submit. Mutated only on confine.
  private var submittingJobResults = false
  private val submittingCellIds = linkedSetOf<String>()
  private var mmprojSelectionInitialized = false
  // Set by wizard selection-changing intents; gates the one selection-reconciliation pass in
  // buildWizard so a background runner tick (pure render) never mutates the user's selections.
  private var wizardSelectionsDirty = true

  // Disk-walk caches reused across a search-keystroke burst so filtering a query doesn't re-walk
  // the models dir / re-parse every manifest per character. Invalidated on every non-search intent
  // and on every runner tick (see handle()/the collector), so they never serve stale data.
  private var manifestsCache: List<JobManifest>? = null
  private var modelsCache: List<ai.liquid.pipette.ModelFile>? = null

  private fun cachedManifests(): List<JobManifest> = manifestsCache ?: storage.loadAllJobManifests().also { manifestsCache = it }

  private fun cachedModels(): List<ai.liquid.pipette.ModelFile> = modelsCache ?: storage.availableModels().also { modelsCache = it }

  private fun invalidateData() {
    manifestsCache = null
    modelsCache = null
  }

  // Built when an ExportCsv effect fires, read back when the SAF create-document result returns.
  // Held here (not in Compose rememberSaveable) so a large CSV never lands in the saved-instance
  // Bundle — that risks TransactionTooLargeException — and it survives an activity recreation while
  // the picker is up. @Volatile: set on confine, consumed on the UI/effect thread.
  @Volatile private var pendingCsvExport: String? = null

  fun consumePendingCsvExport(): String? = pendingCsvExport.also { pendingCsvExport = null }

  private var jobSearchText = ""
  private var benchmarkSearchText = ""
  private var jobModelSearchText = ""
  private var nGpuLayers = 99
  private var contextSize = 4096
  private var prefillBatch = JobManifest.DEFAULT_PREFILL_BATCH
  private var contributeResults = false

  private val _state = MutableStateFlow<JobsUiState>(JobsUiState.JobList())
  val state: StateFlow<JobsUiState> = _state.asStateFlow()

  // Single-thread confinement: all intent handling, field mutation, disk I/O, and state
  // building run here — never on Main. Serialized, so the plain vars/sets above need no
  // extra synchronization and publishes stay ordered. UI only reads _state (thread-safe).
  private val confine = Dispatchers.Default.limitedParallelism(1)

  init {
    selectedBenchmarkIds += BenchmarkCatalog.selectable.map { it.benchmarkId.toString() }
    viewModelScope.launch(confine) { publish() }
    // A progress tick within one running cell changes no manifest on disk, so we must NOT invalidate
    // the whole cache every tick (that re-read + re-parsed every job's manifest). Two things DO change
    // manifests mid-run: (a) the running job advances its own completedCells/cell-status each cell —
    // jobCard reloads just that one manifest fresh; (b) a job starting/finishing/switching rewrites
    // its status — detected here as a runningJobId transition, on which we invalidate ONCE (cheap:
    // once per job start/end, not per tick) so the finished card renders "Completed", not stale.
    // Other structural changes flow through shell.dataChanges below.
    viewModelScope.launch(confine) {
      var lastRunningJobId: String? = null
      runner.state.collect { runnerState ->
        if (runnerState.runningJobId != lastRunningJobId) {
          invalidateData()
          lastRunningJobId = runnerState.runningJobId
        }
        publish()
      }
    }
    viewModelScope.launch(confine) {
      shell.dataChanges.collect {
        invalidateData()
        publish()
      }
    }
    // Re-render (and seed the default selection the first time the catalog lands) when a benchmark sync replaces the catalog. StateFlow replays the
    // current value immediately, so an already-seeded selection is a no-op; a completed sync re-fires it. Runs on confine, so the mutation is safe.
    viewModelScope.launch(confine) {
      BenchmarkCatalog.changes.collect {
        if (selectedBenchmarkIds.isEmpty()) selectedBenchmarkIds += BenchmarkCatalog.selectable.map { it.benchmarkId.toString() }
        publish()
      }
    }
  }

  fun onIntent(intent: JobsIntent) {
    viewModelScope.launch(confine) { handle(intent) }
  }

  private fun handle(intent: JobsIntent) {
    // Search intents reuse the disk caches (instant filtering); everything else gets fresh data.
    if (intent !is JobsIntent.ApplyJobSearch && intent !is JobsIntent.ApplyJobModelSearch) invalidateData()
    when (intent) {
      JobsIntent.OpenWizard -> {
        newJobStep = 0
        wizardSelectionsDirty = true
        contributeResults = storage.isRegistered() && shell.defaultContributeResults.value
        publish()
      }
      JobsIntent.CancelWizard -> {
        newJobStep = null
        publish()
      }
      is JobsIntent.WizardGoToStep -> {
        newJobStep = intent.step.coerceIn(0, NewJobWizard.LAST_STEP)
        publish()
      }
      is JobsIntent.ApplyJobSearch -> {
        jobSearchText = intent.query
        publish()
      }
      is JobsIntent.OpenJobDetail -> {
        selectedJobId = intent.jobId
        selectedCellId = null
        expandedCellIds.clear()
        publish()
      }
      JobsIntent.BackToJobs -> {
        selectedJobId = null
        selectedCellId = null
        expandedCellIds.clear()
        selectedRerunCellIds.clear()
        publish()
      }
      is JobsIntent.OpenCellDetail -> {
        selectedCellId = intent.cellId
        publish()
      }
      JobsIntent.CloseCellDetail -> {
        selectedCellId = null
        publish()
      }
      is JobsIntent.ApplyJobModelSearch -> {
        jobModelSearchText = intent.query
        publish()
      }
      is JobsIntent.ToggleModelGroup -> {
        if (intent.checked) selectedModelKeys += intent.key else selectedModelKeys -= intent.key
        wizardSelectionsDirty = true
        publish()
      }
      is JobsIntent.ToggleQuantFilter -> {
        updateJobQuantFilter(intent.filter, intent.checked)
        wizardSelectionsDirty = true
        publish()
      }
      is JobsIntent.ApplyBenchmarkSearch -> {
        benchmarkSearchText = intent.query
        publish()
      }
      is JobsIntent.ToggleBenchmark -> {
        if (intent.checked) selectedBenchmarkIds += intent.id else selectedBenchmarkIds -= intent.id
        wizardSelectionsDirty = true
        publish()
      }
      is JobsIntent.ToggleBenchmarkGroup -> {
        val items = BenchmarkCatalog.selectable.filter { it.benchmarkType == intent.type }.map { it.benchmarkId.toString() }
        if (intent.select) selectedBenchmarkIds.addAll(items) else selectedBenchmarkIds.removeAll(items.toSet())
        wizardSelectionsDirty = true
        publish()
      }
      is JobsIntent.ToggleMmproj -> {
        if (intent.checked) selectedMmprojPaths += intent.path else selectedMmprojPaths -= intent.path
        wizardSelectionsDirty = true
        publish()
      }
      is JobsIntent.SetAllMmproj -> {
        selectedMmprojPaths.clear()
        if (intent.selected) selectedMmprojPaths += cachedModels().filter { it.isMmproj }.map { it.path }
        wizardSelectionsDirty = true
        publish()
      }

      is JobsIntent.SetWizardContribute -> {
        contributeResults = storage.isRegistered() && intent.enabled
        publish()
      }
      is JobsIntent.SetRunSetup -> {
        nGpuLayers = intent.nGpuLayers
        contextSize = intent.contextSize
        prefillBatch = intent.prefillBatch
        publish()
      }
      is JobsIntent.RunJob -> runJob(intent)
      JobsIntent.CancelRunningJob -> {
        runner.cancel()
        publish()
      }
      is JobsIntent.SetJobAutoSubmit -> {
        val manifest = storage.loadJobManifest(intent.jobId) ?: return
        manifest.contributeResults = storage.isRegistered() && intent.enabled
        storage.saveJobManifest(manifest)
        publish()
      }
      is JobsIntent.RenameJob -> {
        val manifest = storage.loadJobManifest(intent.jobId) ?: return
        manifest.title = intent.title.trim().takeIf { it.isNotEmpty() }
        storage.saveJobManifest(manifest)
        publish()
      }
      is JobsIntent.ResumeJob -> {
        // Pocket mode is entered by BenchmarkActivity (launched by JobRunner in
        // :benchmark), so this only surfaces resume failures.
        runCatching { runner.resume(intent.jobId) }.onFailure { showError(it.message ?: "Resume failed") }
        publish()
      }
      is JobsIntent.RetryFailed -> {
        runCatching { runner.retryFailed(intent.jobId) }.onFailure { showError(it.message ?: "Retry failed") }
        publish()
      }
      is JobsIntent.ExportCsv -> {
        val manifest = storage.loadJobManifest(intent.jobId) ?: return
        runCatching {
            pendingCsvExport = CompletedResultsCsvExporter.csv(storage, manifest)
            emit(Effect.ExportCsv(CompletedResultsCsvExporter.filename(manifest)))
          }
          .onFailure { showError(it.message ?: "Export failed") }
      }
      is JobsIntent.RerunSelectedCells -> {
        runCatching {
            runner.rerunCells(intent.jobId, selectedRerunCellIds.toSet())
            selectedRerunCellIds.clear()
          }
          .onFailure { showError(it.message ?: "Rerun failed") }
        publish()
      }
      JobsIntent.ClearRerunSelection -> {
        selectedRerunCellIds.clear()
        publish()
      }
      is JobsIntent.ToggleCellRerun -> {
        if (intent.checked) selectedRerunCellIds += intent.cellId else selectedRerunCellIds -= intent.cellId
        publish()
      }
      is JobsIntent.ToggleCellExpanded -> {
        if (expandedCellIds.contains(intent.cellId)) expandedCellIds -= intent.cellId else expandedCellIds += intent.cellId
        publish()
      }
      is JobsIntent.RerunCell -> {
        runCatching {
            runner.rerunCells(intent.jobId, setOf(intent.cellId))
            selectedRerunCellIds -= intent.cellId
          }
          .onFailure { showError(it.message ?: "Rerun failed") }
        publish()
      }
      is JobsIntent.SubmitCellResult -> submitCellResult(intent.jobId, intent.cellId)
      is JobsIntent.SubmitJobResults -> submitJobResults(intent.jobId)
      is JobsIntent.DeleteJob -> {
        storage.deleteJob(intent.jobId)
        if (selectedJobId == intent.jobId) selectedJobId = null
        expandedCellIds.clear()
        selectedRerunCellIds.clear()
        publish()
        // A deleted job takes its unsubmitted results with it, and Settings quotes that total in the sign-out warning.
        shell.notifyDataChanged()
      }
      JobsIntent.GoToModels -> shell.navigateTo(ai.liquid.pipette.Tab.MODELS)
      is JobsIntent.OpenPocketMode -> shell.openPocketMode(intent.jobId)
    }
  }

  // ---------------------------------------------------------------------------
  // Side effects
  // ---------------------------------------------------------------------------

  private fun runJob(intent: JobsIntent.RunJob) {
    nGpuLayers = intent.nGpuLayers
    contextSize = intent.contextSize
    prefillBatch = intent.prefillBatch
    val allModels = cachedModels()
    val baseModels = allModels.filterNot { it.isMmproj }
    val groups = ModelCatalog.groups(baseModels)
    val mmprojs = allModels.filter { it.isMmproj }
    val runnableModels = ModelCatalog.resolveSelectedFiles(groups = groups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
    val benchmarks = BenchmarkCatalog.selectable.filter { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
    runCatching {
        val jobId =
          runner.startNewJob(
            models = runnableModels,
            mmprojFiles = mmprojs,
            benchmarks = benchmarks,
            selectedMmprojPaths = selectedMmprojPaths,
            nGpuLayers = nGpuLayers,
            contextSize = contextSize,
            prefillBatch = prefillBatch,
            contributeResults = contributeResults,
          )
        newJobStep = null
        selectedJobId = jobId
        // Pocket mode is entered by BenchmarkActivity, which JobRunner launches in
        // the :benchmark process when the run's first cell loads — that process
        // must be the focused top-app one for the CPU-affinity boost, so it (not
        // the main-process Compose pocket) is the run's pocket screen.
        publish()
      }
      .onFailure { showError(it.message ?: "Failed to start job") }
  }

  private fun submitCellResult(jobId: String, cellId: String) {
    val registration = storage.loadRegistration()
    if (registration == null) {
      showError("Register the device before submitting results")
      return
    }
    submittingCellIds += cellId
    publish()
    viewModelScope.launch {
      runCatching {
          val record =
            withContext(Dispatchers.IO) { submissionService.submitCell(jobId, cellId, registration) }
              ?: error("No payload is available for this cell")
          // Persist the returned serverJobId + re-derive on the confine thread, so this manifest
          // write is serialized with every other VM manifest write (rename, auto-submit) instead of
          // racing them from Dispatchers.IO.
          withContext(confine) {
            if (record.status == "submitted") {
              storage.loadJobManifest(jobId)?.let { manifest ->
                manifest.cells.firstOrNull { it.cellId == cellId }?.serverJobId = record.serverJobId
                storage.saveJobManifest(manifest)
              }
            }
            submittingCellIds -= cellId
            invalidateData()
            publish()
            // Only when the server actually took it: a recorded-but-rejected submission writes no serverJobId, so the result is still unsubmitted and
            // there is nothing for the other tabs to re-derive. One fewer for Settings' sign-out warning to quote when it did land.
            if (record.status == "submitted") shell.notifyDataChanged()
          }
          // A recorded-but-failed submission isn't a thrown error, so surface it as one (iOS shows the
          // submission error) rather than letting it pass silently.
          if (record.status != "submitted") error("Cell submission failed: ${record.errors.joinToString("; ")}")
        }
        .onFailure {
          withContext(confine) {
            submittingCellIds -= cellId
            publish()
          }
          showError(it.message ?: it.javaClass.simpleName)
        }
    }
  }

  private fun submitJobResults(jobId: String) {
    val manifest = storage.loadJobManifest(jobId) ?: return
    val registration = storage.loadRegistration()
    if (registration == null) {
      showError("Register the device before submitting results")
      return
    }
    submittingJobResults = true
    publish()
    viewModelScope.launch {
      runCatching {
          val latest = storage.loadJobManifest(manifest.jobId) ?: manifest
          val outcome = withContext(Dispatchers.IO) { submissionService.submit(latest, registration) }
          // Partial failure isn't thrown; surface it as an error toast (iOS shows submission errors)
          // instead of dropping it now that there's no status bar.
          if (outcome.errors.isNotEmpty()) {
            error("Submitted ${outcome.submitted}; ${outcome.errors.size} error${if (outcome.errors.size == 1) "" else "s"}")
          }
        }
        .onFailure { showError(it.message ?: it.javaClass.simpleName) }
      // Always clear the in-flight flag and re-derive on confine (a non-running job's detail updates).
      withContext(confine) {
        submittingJobResults = false
        invalidateData()
        publish()
        // Accepted results stop counting as unsubmitted, which Settings quotes in its sign-out warning. Announced even when the submission reported
        // errors, since a partial success still moved some of them.
        shell.notifyDataChanged()
      }
    }
  }

  // ---------------------------------------------------------------------------
  // State construction
  // ---------------------------------------------------------------------------

  private fun publish() {
    val runnerState = runner.state.value
    selectedJobId?.let { jobId ->
      val manifest = storage.loadJobManifest(jobId)
      if (manifest != null) {
        val detail = buildJobDetail(manifest, runnerState)
        val cellId = selectedCellId
        val cellUi = if (cellId != null) detail.cells.firstOrNull { it.cell.cellId == cellId } else null
        _state.value = if (cellUi != null) JobsUiState.CellDetail(cellUi, jobId, detail.isRunning) else detail
        return
      }
      selectedJobId = null
      selectedCellId = null
      expandedCellIds.clear()
      selectedRerunCellIds.clear()
    }
    if (newJobStep != null) {
      _state.value = buildWizard(newJobStep!!.coerceIn(0, NewJobWizard.LAST_STEP), runnerState)
      return
    }
    _state.value = buildJobList(runnerState)
  }

  private fun buildJobList(runnerState: ai.liquid.pipette.RunnerState): JobsUiState.JobList {
    val hasModels = cachedModels().any { !it.isMmproj }
    val manifests = cachedManifests()
    val filtered = manifests.filter { jobMatchesSearch(it, jobSearchText) }
    return JobsUiState.JobList(
      hasModels = hasModels,
      searchQuery = jobSearchText,
      matched = filtered.isNotEmpty(),
      anyJobs = manifests.isNotEmpty(),
      jobs = filtered.map { jobCard(it, runnerState) },
      engineAvailable = container.benchmarkEngine.isAvailable,
    )
  }

  private fun jobCard(cached: JobManifest, runnerState: ai.liquid.pipette.RunnerState): JobCardUi {
    val running = runnerState.runningJobId == cached.jobId
    // The running job's manifest advances on disk (completedCells, cell statuses) each cell. The list
    // manifest cache is deliberately NOT invalidated on every runner tick (that re-read + re-parsed
    // every job's manifest), so reload just this one card's manifest when it's the one running.
    val manifest = if (running) storage.loadJobManifest(cached.jobId) ?: cached else cached
    val failed = manifest.cells.firstOrNull { it.runStatus == CellRunStatus.FAILED && it.errorMessage != null }
    // Overall completion so the bar climbs across cells (iOS), not the per-cell fraction that resets each cell.
    val overall = RunProgress.manifestFraction(manifest, runnerState)
    return JobCardUi(
      manifest = manifest,
      statusAccent = jobStatusAccent(manifest.status),
      runningHere = running,
      runProgress = overall,
      countsLine =
        "${manifest.completedCells}/${manifest.totalCells} completed, " +
          "${manifest.failedCells} failed, ${manifest.cancelledCells} cancelled, ${manifest.submittedCells} submitted",
      rowPrimaryMeta =
        if (running) "${manifest.completedCells}/${manifest.totalCells} cells done"
        else "${manifest.totalCells} ${plural("cell", manifest.totalCells)}",
      rowSecondaryMeta =
        when {
          running -> runnerState.currentProgressText.takeIf { it.isNotBlank() }?.let { "In progress · $it" } ?: "In progress"
          manifest.status == JobStatus.COMPLETED -> "Completed"
          manifest.status == JobStatus.CANCELLED -> "Cancelled"
          manifest.status == JobStatus.PAUSED -> "Paused"
          else -> "Created ${DateFormats.shortDate(manifest.createdAt)}"
        },
      firstFailure = failed?.let { "First failure: ${it.benchmarkId} - ${it.errorMessage}" },
      canResume = manifest.status == JobStatus.PAUSED && !runner.isRunning(),
      completedCells = manifest.completedCells,
      unsubmittedCount = storage.unsubmittedResultCount(manifest),
      isRegistered = storage.isRegistered(),
    )
  }

  private fun buildWizard(step: Int, runnerState: ai.liquid.pipette.RunnerState): JobsUiState.Wizard {
    val allModels = cachedModels()
    val baseModels = allModels.filterNot { it.isMmproj }
    val modelGroups = ModelCatalog.groups(baseModels)
    val mmprojs = allModels.filter { it.isMmproj }
    // The groups a quant filter can draw from: the currently-selected models, or all groups when none
    // are selected. Evaluated fresh at each use so it reflects the current selection.
    fun quantSourceGroups(): List<ai.liquid.pipette.ModelGroup> = modelGroups.filter { selectedModelKeys.contains(it.key) }.ifEmpty { modelGroups }
    // Quant filters some model in [sources] offers (ALL is always available). Single-sourced so the
    // dirty-selection reconciliation and the pill list can't diverge on which quants are "available".
    fun availableQuantFiltersFor(sources: List<ai.liquid.pipette.ModelGroup>): List<JobQuantFilter> =
      JobQuantFilter.entries.filter { f -> f == JobQuantFilter.ALL || sources.any { g -> g.files.any { f.matches(it.quant) } } }
    // Reconcile selections to the current catalog only on real selection/data changes — never on a
    // pure render (a background runner tick), so a tick can't silently alter the user's choices.
    if (wizardSelectionsDirty) {
      selectedModelKeys.retainAll(modelGroups.map { it.key }.toSet())
      // Drop any picked quant filter that no currently-selected model offers, so a filter whose pill
      // is no longer shown can't silently empty the runnable set and block Run (fall back to ALL).
      val available = availableQuantFiltersFor(quantSourceGroups())
      selectedJobQuantFilters.retainAll { available.contains(it) }
      if (selectedJobQuantFilters.isEmpty()) selectedJobQuantFilters += JobQuantFilter.ALL
      val runnable = ModelCatalog.resolveSelectedFiles(groups = modelGroups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
      if (runnable.none { JobRunner.isVlCompatible(it, mmprojs) }) pruneVlBenchmarks()
      ensureMmprojSelectionInitialized(mmprojs)
      wizardSelectionsDirty = false
    }
    val filteredModelGroups = modelGroups.filter { modelGroupMatchesSearch(it, jobModelSearchText) }
    val runnableModels = ModelCatalog.resolveSelectedFiles(groups = modelGroups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
    val modelsMissingQuant =
      ModelCatalog.selectedGroupsMissingQuant(groups = modelGroups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
    val selectedBaseModelCount = modelGroups.count { selectedModelKeys.contains(it.key) }
    val hasVlModelSelected = runnableModels.any { JobRunner.isVlCompatible(it, mmprojs) }
    // Quant pills = the UNION of quants present across the selected models (all downloaded models
    // before any selection), so only available quants are offered. A quant the user picks that some
    // models lack is just skipped for those models later (resolveSelectedFiles drops missing files).
    val availableQuantFilters = availableQuantFiltersFor(quantSourceGroups())
    val selectedBenchmarks = BenchmarkCatalog.selectable.filter { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
    val hasVlBenchmarkSelected = selectedBenchmarks.any { it.type == BenchmarkType.VL_THROUGHPUT }
    val plannedCells =
      JobRunner.planCells(models = runnableModels, mmprojFiles = mmprojs, benchmarks = selectedBenchmarks, selectedMmprojPaths = selectedMmprojPaths)

    val benchmarkGroups = if (step == 1) buildBenchmarkGroups(hasVlModelSelected) else emptyList()
    val benchmarksMatched = BenchmarkCatalog.selectable.any { BenchmarkCatalog.matchesSearch(it, benchmarkSearchText) }
    val selectedMmprojCount = mmprojs.count { selectedMmprojPaths.contains(it.path) }
    val running = runnerState.runningJobId != null

    return JobsUiState.Wizard(
      step = step,
      stepTitles = NewJobWizard.STEP_TITLES,
      modelSearch = jobModelSearchText,
      modelGroups =
        filteredModelGroups.map {
          ModelGroupRowUi(
            key = it.key,
            label = modelGroupLabel(it),
            name = it.name,
            sizeLabel = ByteFormat.fileSize(it.files.sumOf { f -> f.sizeBytes }),
            checked = selectedModelKeys.contains(it.key),
          )
        },
      anyModelGroups = modelGroups.isNotEmpty(),
      modelsMatched = filteredModelGroups.isNotEmpty(),
      selectedBaseModelCount = selectedBaseModelCount,
      quantFilters = availableQuantFilters.map { QuantChipUi(it, it.label, selectedJobQuantFilters.contains(it)) },
      benchmarkSearch = benchmarkSearchText,
      benchmarkGroups = benchmarkGroups,
      benchmarksMatched = benchmarksMatched,
      showMmprojCard = hasVlBenchmarkSelected,
      mmprojs = mmprojs.map { MmprojRowUi(it.path, "${it.name} (${it.sizeFormatted})", selectedMmprojPaths.contains(it.path)) },
      allMmprojSelected = mmprojs.isNotEmpty() && selectedMmprojCount == mmprojs.size,
      nGpuLayers = nGpuLayers,
      contextSize = contextSize,
      prefillBatch = prefillBatch,
      contributeResults = contributeResults,
      isRegistered = storage.isRegistered(),
      reviewSummary =
        plannedJobSummary(
          selectedBaseModelCount,
          runnableModels.size,
          selectedBenchmarks.size,
          selectedMmprojCount,
          hasVlBenchmarkSelected,
          plannedCells.size,
        ),
      reviewDate = DateFormats.shortDate(DateFormats.isoNow()),
      reviewSubtitle =
        run {
          val models = modelGroups.count { selectedModelKeys.contains(it.key) }
          val benches = selectedBenchmarks.map { it.benchmarkType }.distinct().size
          "$models ${plural("model", models)} · $benches ${plural("benchmark", benches)}"
        },
      reviewModels = modelGroups.filter { selectedModelKeys.contains(it.key) }.map { it.name },
      reviewBenchmarks = selectedBenchmarks.map { BenchmarkCatalog.displayName(it.benchmarkType) }.distinct(),
      reviewQuants = runnableModels.mapNotNull { it.quant }.distinct(),
      showSkippedWarning = modelsMissingQuant.isNotEmpty(),
      skippedWarning = if (modelsMissingQuant.isEmpty()) "" else missingQuantWarning(modelsMissingQuant),
      canAdvance = NewJobWizard.canAdvance(step, selectedBaseModelCount, selectedBenchmarks.size),
      canRun = NewJobWizard.canRun(plannedCells.size, running),
      runLabel = if (running) "A job is running" else "Run job",
    )
  }

  private fun buildBenchmarkGroups(hasVlModelSelected: Boolean): List<BenchmarkGroupUi> {
    val filtered = BenchmarkCatalog.selectable.filter { BenchmarkCatalog.matchesSearch(it, benchmarkSearchText) }
    return filtered
      .groupBy { it.benchmarkType }
      .toSortedMap(compareBy<String> { BenchmarkCatalog.typeRank(it) })
      .map { (type, items) ->
        val isVlGroupDisabled = type == BenchmarkType.VL_THROUGHPUT.wire && !hasVlModelSelected
        val enabledItems = if (isVlGroupDisabled) emptyList() else items
        val allGroupSelected = enabledItems.isNotEmpty() && enabledItems.all { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
        val anySelected = items.any { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
        BenchmarkGroupUi(
          type = type,
          displayName = BenchmarkCatalog.displayName(type),
          description = benchmarkTypeDescription(type),
          disabled = isVlGroupDisabled,
          allSelected = allGroupSelected,
          someSelected = anySelected && !allGroupSelected,
          toggleLabel = if (allGroupSelected) "Clear ${BenchmarkCatalog.displayName(type)}" else "Select all ${BenchmarkCatalog.displayName(type)}",
          items =
            items
              .sortedBy { it.label }
              .map { item ->
                val itemDisabled = item.type == BenchmarkType.VL_THROUGHPUT && !hasVlModelSelected
                BenchmarkItemUi(
                  id = item.benchmarkId.toString(),
                  label = item.label,
                  enabled = !itemDisabled,
                  checked = selectedBenchmarkIds.contains(item.benchmarkId.toString()) && !itemDisabled,
                  definition = item,
                )
              },
        )
      }
  }

  private fun buildJobDetail(manifest: JobManifest, runnerState: ai.liquid.pipette.RunnerState): JobsUiState.Detail {
    selectedRerunCellIds.retainAll(manifest.cells.map { it.cellId }.toSet())
    val payloads = CompletedResultsCsvExporter.payloadsByCellId(storage, manifest)
    val metrics = CompletedResultsCsvExporter.metricsByCellId(manifest, payloads)
    val selectedRerunnable = selectedRerunCellIds.count { id -> manifest.cells.any { it.cellId == id && it.isRerunnable } }
    val modelNames = manifest.cells.map { it.modelName }.distinct()
    val benchmarkNames =
      manifest.cells.mapNotNull { BenchmarkCatalog.resolve(it.benchmarkId)?.let { b -> BenchmarkCatalog.displayName(b.benchmarkType) } }.distinct()
    val quants = manifest.cells.mapNotNull { CompletedResultsCsvExporter.quantLabel(it).takeIf { q -> q.isNotBlank() } }.distinct()
    val running = runnerState.runningJobId == manifest.jobId
    // Overall progress (for the bar) = completed cells + the running cell's fraction (iOS progressFraction).
    val fraction = RunProgress.manifestFraction(manifest, runnerState)
    val remainingCells = (manifest.totalCells - manifest.completedCells).coerceAtLeast(0)
    // ETA must extrapolate from THIS run's progress, not the whole manifest (a resumed job starts
    // this run at completedInRun=0, so manifest-based math would read "0s left" at once).
    val runFraction = RunProgress.runFraction(runnerState, fraction)
    // Read once for the whole detail: every cell's Submit affordance needs it, and each read re-parses registration.json off disk.
    val isRegistered = storage.isRegistered()
    return JobsUiState.Detail(
      manifest = manifest,
      statusAccent = jobStatusAccent(manifest.status),
      titleDate = DateFormats.shortDate(manifest.createdAt),
      subtitle = "${modelNames.size} ${plural("model", modelNames.size)} · ${benchmarkNames.size} ${plural("benchmark", benchmarkNames.size)}",
      modelChips = modelNames,
      benchmarkChips = benchmarkNames,
      quantChips = quants,
      gpuLayers = manifest.nGpuLayers,
      contextSize = manifest.contextSize,
      createdLine =
        "Created: ${DateFormats.shortDate(manifest.createdAt)}\n" +
          "${manifest.completedCells}/${manifest.totalCells} completed, " +
          "${manifest.failedCells} failed, ${manifest.cancelledCells} cancelled, ${manifest.submittedCells} submitted",
      runningHere = running,
      runProgress = fraction,
      runCellsDone = "${manifest.completedCells}/${manifest.totalCells} cells done",
      runTimeLeft = RunProgress.estimatedTimeLeft(runnerState, runFraction) ?: "$remainingCells ${plural("cell", remainingCells)} left",
      runCellLabel = if (running) runnerState.currentCellLabel else "",
      runProgressText = if (running) runnerState.currentProgressText else "",
      coolingSinceMillis = if (running) runnerState.coolingSinceMillis else null,
      // Only read the thermal signals when running — the row is rendered only in the running branch,
      // and the reads (getThermalHeadroom, a per-process ~1 Hz budget) shouldn't fire on every
      // paused/completed detail render where they'd never be shown.
      thermalLabel = if (running) thermalHeadroomLabel(thermalProvider) else "",
      thermalAccent = if (running) thermalAccentKind(thermalDescription(thermalProvider)) else AccentKind.MUTED,
      contributeResults = manifest.contributeResults == true,
      isRegistered = isRegistered,
      isRunning = runner.isRunning(),
      canResume = manifest.status == JobStatus.PAUSED && !runner.isRunning(),
      failedCells = manifest.failedCells,
      completedCells = manifest.completedCells,
      selectedRerunnableCount = selectedRerunnable,
      unsubmittedCount = storage.unsubmittedResultCount(manifest),
      isSubmitting = submittingJobResults,
      resultsGrid = if (metrics.isEmpty()) null else buildResultsGrid(manifest, metrics),
      cells =
        manifest.cells.mapIndexed { index, cell -> buildCell(manifest, cell, index + 1, payloads[cell.cellId], metrics[cell.cellId], isRegistered) },
    )
  }

  private fun buildResultsGrid(manifest: JobManifest, metrics: Map<String, CompletedRunMetric>): ResultsGridUi {
    val cells = manifest.cells
    val columns = cells.map { it.benchmarkId }.distinct()
    val rowOrder = mutableListOf<String>()
    val rowCells = linkedMapOf<String, MutableMap<String, JobCell>>()
    val rowLabels = linkedMapOf<String, String>()
    val rowModel = linkedMapOf<String, String>()
    val rowQuant = linkedMapOf<String, String>()
    cells.forEach { cell ->
      val key = CompletedResultsCsvExporter.resultModelGroupKey(cell) + "|" + CompletedResultsCsvExporter.quantLabel(cell)
      if (!rowCells.containsKey(key)) {
        rowOrder += key
        rowCells[key] = mutableMapOf()
        rowLabels[key] = "${CompletedResultsCsvExporter.modelDisplayName(cell)}\n${CompletedResultsCsvExporter.quantLabel(cell)}"
        rowModel[key] = CompletedResultsCsvExporter.modelDisplayName(cell)
        rowQuant[key] = CompletedResultsCsvExporter.quantLabel(cell)
      }
      rowCells.getValue(key)[cell.benchmarkId] = cell
    }
    val colRange =
      columns.associateWith { col ->
        val values = rowOrder.mapNotNull { rk -> rowCells[rk]?.get(col)?.cellId?.let { metrics[it]?.numericValue } }
        if (values.isEmpty()) null else (values.min() to values.max())
      }
    val rows =
      rowOrder.map { rk ->
        ResultsRowUi(
          label = rowLabels.getValue(rk),
          modelName = rowModel.getValue(rk),
          quant = rowQuant.getValue(rk),
          cells =
            columns.map { col ->
              val cell = rowCells[rk]?.get(col)
              val metric = cell?.cellId?.let { metrics[it] }
              when {
                cell == null -> ResultsCellUi("—", null, false, null, false)
                metric != null -> {
                  val (lo, hi) = colRange[col] ?: (metric.numericValue to metric.numericValue)
                  val intensity = ResultsGrid.heatmapIntensity(metric.numericValue, lo, hi, metric.higherIsBetter)
                  ResultsCellUi(CompletedResultsCsvExporter.displayMetric(metric), intensity, intensity > 0.55, cell, true)
                }
                // No metric yet: failed/cancelled cells carry a detail page + an accent; pending/running
                // are inert "—" (no result to open), matching iOS (Failed red, Cancelled orange, - inert).
                cell.runStatus == CellRunStatus.FAILED -> ResultsCellUi("Failed", null, false, cell, true, ResultCellAccent.FAILED)
                cell.runStatus == CellRunStatus.CANCELLED -> ResultsCellUi("Cancelled", null, false, cell, true, ResultCellAccent.CANCELLED)
                else -> ResultsCellUi("—", null, false, cell, false)
              }
            },
        )
      }
    return ResultsGridUi(columnLabels = columns.map { columnLabel(cells.first { c -> c.benchmarkId == it }) }, rows = rows)
  }

  private fun columnLabel(cell: JobCell): String {
    val type = CompletedResultsCsvExporter.benchmarkType(cell)
    val short =
      when (type) {
        "prefill_throughput" -> "Prefill"
        "decode_throughput" -> "Decode"
        "end_to_end_latency" -> "E2E"
        "max_memory_usage" -> "Memory"
        "vl_throughput" -> "VL"
        else -> BenchmarkCatalog.displayName(type)
      }
    val params = CompletedResultsCsvExporter.parameterSummary(cell.benchmarkId)
    return if (params != null) "$short\n$params" else short
  }

  private fun buildCell(
    manifest: JobManifest,
    cell: JobCell,
    position: Int,
    payload: JSONObject?,
    metric: CompletedRunMetric?,
    isRegistered: Boolean,
  ): CellUi {
    val benchmarkType = CompletedResultsCsvExporter.benchmarkType(cell)
    val parameter = CompletedResultsCsvExporter.parameterSummary(cell.benchmarkId)
    val benchmarkLabel = buildString {
      append(BenchmarkCatalog.displayName(benchmarkType))
      if (parameter != null) append(" - $parameter")
    }
    val submission = storage.loadSubmission(manifest.jobId, cell.cellId)
    val submissionLine =
      when {
        !cell.serverJobId.isNullOrBlank() -> "Submitted: ${cell.serverJobId}"
        submission?.status == "submitted" -> "Submitted: ${submission.serverJobId ?: "recorded"}"
        submission?.status == "failed" -> "Submission failed: ${submission.errors.joinToString("; ")}"
        else -> null
      }
    val canSubmit = storage.isUnsubmitted(manifest.jobId, cell) && isRegistered
    return CellUi(
      cell = cell,
      position = position,
      title = "Cell $position - ${CompletedResultsCsvExporter.modelDisplayName(cell)}",
      modelName = CompletedResultsCsvExporter.modelDisplayName(cell),
      quant = CompletedResultsCsvExporter.quantLabel(cell),
      benchmarkLabel = benchmarkLabel,
      statusLabel = cell.runStatus.wire,
      statusAccent = cellStatusAccent(cell.runStatus),
      subtitle = "${CompletedResultsCsvExporter.quantLabel(cell)}\n$benchmarkLabel",
      metricLine = metric?.let { "${it.name}: ${CompletedResultsCsvExporter.displayMetric(it)}" },
      errorLine = cell.errorMessage?.takeIf { it.isNotBlank() }?.let { "Error: $it" },
      submissionLine = submissionLine,
      // Gate on !isRunning: rerunCells() clears the cell's artifacts + resets it to PENDING before
      // run() rejects a second concurrent job, so allowing it mid-run would destroy the result with
      // no rerun. Matches RetryFailed / canResume, which are likewise gated.
      rerunSelectable = cell.isRerunnable && !runner.isRunning(),
      rerunSelected = selectedRerunCellIds.contains(cell.cellId),
      expanded = expandedCellIds.contains(cell.cellId),
      canSubmit = canSubmit,
      submitting = submittingCellIds.contains(cell.cellId),
      detailRows = if (payload == null) emptyList() else payloadDetailRows(payload, metric),
      detailModelPath = cell.modelPath,
      detailMmprojPath = cell.mmprojPath,
      hasPayload = payload != null,
    )
  }

  // ---------------------------------------------------------------------------
  // Pure helpers (ported verbatim from JobsScreen)
  // ---------------------------------------------------------------------------

  private fun jobStatusAccent(status: JobStatus): AccentKind =
    when (status) {
      JobStatus.COMPLETED,
      JobStatus.RUNNING -> AccentKind.NOMINAL
      JobStatus.PAUSED -> AccentKind.SERIOUS
      JobStatus.CANCELLED -> AccentKind.CRITICAL
      JobStatus.PLANNED -> AccentKind.MUTED
    }

  private fun cellStatusAccent(status: CellRunStatus): AccentKind =
    when (status) {
      CellRunStatus.COMPLETED,
      CellRunStatus.RUNNING -> AccentKind.NOMINAL
      CellRunStatus.FAILED -> AccentKind.CRITICAL
      CellRunStatus.CANCELLED -> AccentKind.SERIOUS
      CellRunStatus.PENDING -> AccentKind.MUTED
    }

  private fun updateJobQuantFilter(filter: JobQuantFilter, checked: Boolean) {
    if (checked) {
      if (filter == JobQuantFilter.ALL) {
        selectedJobQuantFilters.clear()
        selectedJobQuantFilters += JobQuantFilter.ALL
      } else {
        selectedJobQuantFilters -= JobQuantFilter.ALL
        selectedJobQuantFilters += filter
      }
    } else {
      selectedJobQuantFilters -= filter
    }
    if (selectedJobQuantFilters.isEmpty()) selectedJobQuantFilters += JobQuantFilter.ALL
  }

  private fun jobQuantMatches(quant: String?): Boolean =
    selectedJobQuantFilters.contains(JobQuantFilter.ALL) || selectedJobQuantFilters.any { it.matches(quant) }

  private fun benchmarkTypeDescription(typeWire: String): String =
    when (typeWire) {
      BenchmarkType.END_TO_END_LATENCY.wire -> "The total time to complete a request, from prompt to final token."
      BenchmarkType.PREFILL_THROUGHPUT.wire -> "The rate at which the model processes the prompt, in tok/s."
      BenchmarkType.DECODE_THROUGHPUT.wire -> "The rate at which the model generates output tokens, in tok/s."
      BenchmarkType.MAX_MEMORY_USAGE.wire -> "The maximum memory the model consumes while running a request."
      BenchmarkType.VL_THROUGHPUT.wire -> "Vision-language token throughput, in tok/s."
      else -> ""
    }

  private fun pruneVlBenchmarks() {
    selectedBenchmarkIds.removeAll(
      BenchmarkCatalog.selectable.filter { it.type == BenchmarkType.VL_THROUGHPUT }.map { it.benchmarkId.toString() }.toSet()
    )
  }

  private fun ensureMmprojSelectionInitialized(mmprojs: List<ai.liquid.pipette.ModelFile>) {
    val availablePaths = mmprojs.map { it.path }.toSet()
    selectedMmprojPaths.retainAll(availablePaths)
    if (availablePaths.isEmpty()) {
      mmprojSelectionInitialized = false
      return
    }
    if (!mmprojSelectionInitialized) {
      selectedMmprojPaths += availablePaths
      mmprojSelectionInitialized = true
    }
  }

  private fun modelGroupLabel(group: ModelGroup): String =
    "${group.name}\n${group.quantSummary} - ${ByteFormat.fileSize(group.files.sumOf { it.sizeBytes })}"

  private fun modelGroupMatchesSearch(group: ModelGroup, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    return listOf(group.name, group.key, group.quantSummary).any { it.lowercase().contains(q) } || group.files.any { it.matchesSearch(q) }
  }

  private fun jobMatchesSearch(manifest: JobManifest, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    val benchmarkLabels =
      manifest.cells.mapNotNull { cell ->
        BenchmarkCatalog.resolve(cell.benchmarkId)?.let { item ->
          "${item.benchmarkId} ${BenchmarkCatalog.displayName(item.benchmarkType)} ${item.label}"
        }
      }
    return listOf(
        manifest.displayTitle,
        manifest.jobId,
        manifest.createdAt,
        manifest.cells.joinToString(" ") { it.modelName },
        manifest.cells.joinToString(" ") { it.benchmarkId },
        benchmarkLabels.joinToString(" "),
      )
      .any { it.lowercase().contains(q) }
  }

  private fun quantSummaryForWarning(): String =
    if (selectedJobQuantFilters.contains(JobQuantFilter.ALL)) {
      "selected quant"
    } else {
      JobQuantFilter.entries.filter { it != JobQuantFilter.ALL && selectedJobQuantFilters.contains(it) }.joinToString(", ") { it.label }
    }

  private fun missingQuantWarning(groups: List<ModelGroup>): String {
    val names = groups.joinToString(", ") { it.name }
    val verb = if (groups.size == 1) "has" else "have"
    return "$names $verb no ${quantSummaryForWarning()} build downloaded and will be skipped."
  }

  private fun plannedJobSummary(
    selectedBaseModelCount: Int,
    selectedRunnableModelCount: Int,
    selectedBenchmarkCount: Int,
    selectedMmprojCount: Int,
    hasVlBenchmark: Boolean,
    plannedCellCount: Int,
  ): String {
    val base = buildString {
      append("Planned cells: $plannedCellCount\n")
      append("$selectedRunnableModelCount runnable model ${plural("file", selectedRunnableModelCount)}")
      append(" - $selectedBenchmarkCount selected ${plural("benchmark", selectedBenchmarkCount)}")
      if (hasVlBenchmark) append(" - $selectedMmprojCount selected ${plural("MMProjector", selectedMmprojCount)}")
    }
    val warning =
      when {
        selectedBaseModelCount == 0 -> "Select at least one model."
        selectedRunnableModelCount == 0 -> "Selected models are filtered out by the active quant filter."
        selectedBenchmarkCount == 0 -> "Select at least one benchmark."
        plannedCellCount == 0 && hasVlBenchmark && selectedMmprojCount == 0 -> "Select at least one MMProjector for VL benchmarks."
        plannedCellCount == 0 && hasVlBenchmark -> "No selected model/MMProjector pairing can run the selected VL benchmarks."
        plannedCellCount == 0 -> "No runnable cells for the selected model/benchmark set."
        else -> null
      }
    return warning?.let { "$base\n$it" } ?: base
  }

  private fun payloadDetailRows(payload: JSONObject, metric: CompletedRunMetric?): List<Pair<String, String>> {
    val hiddenKeys =
      setOf(
        "benchmark_flags",
        "benchmark_id",
        "cell_id",
        "completions",
        "device_name",
        "job_id",
        "model_descriptor",
        "model_name",
        "model_quant",
        "runtime_descriptor",
        "runtime_flags",
        "submitted_at",
      )
    val rows = mutableListOf<Pair<String, String>>()
    if (metric != null) rows += metric.name to CompletedResultsCsvExporter.displayMetric(metric)
    payload
      .keys()
      .asSequence()
      .filterNot { hiddenKeys.contains(it) }
      .sorted()
      .forEach { key -> payloadScalarString(payload.opt(key))?.let { value -> rows += humanizedKey(key) to value } }
    return rows
  }

  private fun payloadScalarString(value: Any?): String? {
    if (value == null || value == JSONObject.NULL) return null
    return when (value) {
      is String -> value
      is Boolean -> if (value) "true" else "false"
      is Number -> {
        val double = value.toDouble()
        val rounded = kotlin.math.round(double)
        if (kotlin.math.abs(rounded - double) < 0.0001) rounded.toLong().toString() else String.format(java.util.Locale.US, "%.2f", double)
      }
      else -> null
    }
  }

  private fun humanizedKey(key: String): String =
    key.replace("_ms", " ms").replace("_bytes", " bytes").split("_").joinToString(" ") { word -> word.replaceFirstChar { it.titlecase() } }
}
