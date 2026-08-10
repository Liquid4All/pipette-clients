package ai.liquid.pipette

import ai.liquid.pipette.compose.RunProgress
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class RunnerState(
  val runningJobId: String? = null,
  val currentCellLabel: String = "",
  val currentProgressText: String = "",
  val currentCellFraction: Double = 0.0,
  val startedAtMillis: Long? = null,
  // Run-relative progress (cells finished this run / cells this run will run), seeded at each
  // start/resume/retry. ETA must extrapolate from these, not the whole manifest — a resumed
  // 90%-done job starts this run at 0, so manifest-based math would read "0s left" immediately.
  val completedInRun: Int = 0,
  val totalInRun: Int = 0,
  // Wall-clock instant the CURRENT gate wait began, or null when the gate isn't cooling (loading /
  // measuring). Re-derived on every cooling status from the elapsed the reporting gate hands over,
  // so it always describes the wait in progress: the deadline the UI counts against
  // ([Readiness.COOLDOWN_MAX_MILLIS]) bounds one waitUntilReady call, and there are three call sites
  // that can fire back to back. Cleared the moment cooling ends.
  val coolingSinceMillis: Long? = null,
)

/**
 * Reserved [BenchmarkProgressCallback] `total` value marking a status message as a thermal-cooldown update rather than measured progress. The
 * service-side readiness gate ([service.PipetteBenchmarkService]) rides the AIDL progress channel to reach the UI, so it tags cooldown emits with
 * this sentinel; the cell loop recognizes it, drives the cooling wash, and leaves the cell fraction untouched. A native engine never emits a total
 * this small, so it can't be mistaken for real progress.
 */
internal const val COOLING_PROGRESS_TOTAL = Int.MIN_VALUE

/**
 * How a benchmark run was started, reported as the `source` property on [AnalyticsEvents.JOB_STARTED]. Distinguishing these is what makes the funnel
 * readable: a resume or a retry is not a fresh user intent to run a job, and lumping them together would inflate the "started" count.
 *
 * [wireName] is declared explicitly per entry rather than derived from the identifier (as [JobStatus.wire] and [CellRunStatus.wire] do with
 * `name.lowercase()`). These strings are a cross-platform wire contract: a rename that silently changed one would split a funnel across two spellings
 * without breaking a build, and the derived form gives no protection against that. Spelling them out makes the Kotlin identifier free to change and
 * mirrors iOS, whose `RunSource: String` carries the same raw values. `PostHogConfigurationTest` pins them.
 */
/**
 * iOS declares one case this enum does not: `planner`, for its in-app `PlannerWorker` claim loop (Android's planner worker is CLI-side). If this
 * client ever gains an in-app claim loop, it must reuse that exact wire string rather than folding those runs into [NEW]: a server-claimed cell is
 * not a user starting a job, and counting it as one makes the funnel's "started" figure meaningless on planner-enabled devices.
 */
enum class RunSource(val wireName: String) {
  NEW("new"),
  RESUME("resume"),
  RETRY_FAILED("retry_failed"),
  RERUN("rerun"),
}

/**
 * Which [AnalyticsEvents.OUTCOME] a finished run should report, derived from the persisted manifest rather than from a caught exception: `execute`
 * already absorbs per-cell failures and always writes a terminal status, so a manifest left non-terminal (or unreadable) is exactly the "the run
 * itself died" case that deserves [AnalyticsEvents.OUTCOME_FAILED].
 *
 * A cancel lands as [AnalyticsEvents.OUTCOME_CANCELLED] whether it was observed through the live flag or only through the [JobStatus.PAUSED] the
 * cancel path persisted: a run cancelled after its last cell finishes has nothing left pending and is saved as [JobStatus.COMPLETED], so the flag is
 * what distinguishes it.
 *
 * Top-level (not a `JobRunner` member) to keep that class under detekt's `TooManyFunctions` threshold, which it now sits exactly on.
 */
internal fun jobOutcome(cancelled: Boolean, status: JobStatus?): String =
  when {
    // PAUSED is what this runner's own cancel path persists. CANCELLED is never written here today,
    // but [JobStatus.fromWire] deserializes it, and a manifest from another writer must not be reported
    // as a failure. iOS's `analyticsOutcome` treats both the same way.
    cancelled || status == JobStatus.PAUSED || status == JobStatus.CANCELLED -> AnalyticsEvents.OUTCOME_CANCELLED
    status == JobStatus.COMPLETED -> AnalyticsEvents.OUTCOME_FINISHED
    else -> AnalyticsEvents.OUTCOME_FAILED
  }

/**
 * Emit [AnalyticsEvents.JOB_COMPLETED] for the run that just ended. Called from `run`'s `finally`, so the counters must be passed in BEFORE they are
 * cleared. Never throws: the seam swallows its own failures and the manifest read is wrapped.
 */
private fun captureJobCompleted(
  analytics: Analytics,
  storage: JobStore,
  jobId: String,
  cancelled: Boolean,
  cellCount: Int,
  cellsCompleted: Int,
  startedAtMillis: Long?,
) {
  val status = runCatching { storage.loadJobManifest(jobId) }.getOrNull()?.status
  analytics.capture(
    AnalyticsEvents.JOB_COMPLETED,
    mapOf(
      AnalyticsEvents.JOB_ID to jobId,
      AnalyticsEvents.OUTCOME to jobOutcome(cancelled, status),
      AnalyticsEvents.CELL_COUNT to cellCount,
      AnalyticsEvents.CELLS_COMPLETED to cellsCompleted,
      AnalyticsEvents.DURATION_MS to startedAtMillis?.let { System.currentTimeMillis() - it },
    ),
  )
}

class CancelFlag {
  private val cancelled = AtomicBoolean(false)
  val isCancelled: Boolean
    get() = cancelled.get()

  fun cancel() {
    cancelled.set(true)
  }
}

/**
 * Whether a finished cell's result should be uploaded now. The shared gate for both the per-cell upload and the end-of-run sweep, and the twin of the
 * iOS client's `JobExecutor.shouldAutoSubmit`.
 *
 * Pure, so the rule is unit-testable without a runner, an engine or a server. `online` is required because an offline device would otherwise spend
 * the uploader's retries, and the run's wall-clock, on a request that cannot succeed; the payload is already on disk, so the sweep or the next run
 * sends it instead. A missing registration means there is nothing to submit under.
 *
 * Top-level rather than a [JobRunner] member: that class sits exactly on detekt's `TooManyFunctions` threshold.
 */
internal fun shouldAutoSubmit(manifest: JobManifest, online: Boolean, registration: RegistrationData?): Boolean =
  manifest.contributeResults == true && online && registration != null

/**
 * Upload whatever this job has finished and not yet sent. Used only by the mid-run path; the end-of-run sweep calls [ResultSubmitter.submit]
 * directly.
 *
 * Idempotent by construction: `submit` only picks up cells that are COMPLETED with no `serverJobId`, so calling it after every cell uploads just that
 * cell, and sweeps up any earlier one whose upload failed. That is exactly how the iOS client's per-cell `drainJob` behaves.
 *
 * Returns whether mid-run uploading is still worth attempting. **Everything**, including the [isOnline] probe, happens inside the `runCatching`: this
 * is called from inside the cell loop's `try`, whose `catch` rewrites the cell to FAILED, so a throw escaping here would mark a cell that already
 * finished and wrote its payload as a failure. A false return means an attempt genuinely failed and the caller should stop retrying for the rest of
 * the run rather than paying a connect timeout in every remaining inter-cell gap; the sweep still catches the backlog. Being offline is *not* a
 * failure in that sense (the device may reconnect), so it returns true.
 */
private fun submitFinishedCells(
  submissionService: ResultSubmitter,
  manifest: JobManifest,
  registration: RegistrationData?,
  isOnline: () -> Boolean,
): Boolean =
  runCatching {
      val nothingToDo = registration == null || !hasUnsentResults(manifest) || !shouldAutoSubmit(manifest, isOnline(), registration)
      if (nothingToDo) true else submissionService.submit(manifest, registration!!).errors.isEmpty()
    }
    .getOrDefault(false)

/**
 * Does this job hold a result the server hasn't acknowledged? The same condition [ResultSubmitter.submit] filters on internally, asked up front so a
 * caller can skip a submit that would do nothing, which after the per-cell uploads is the normal case at the end of a run.
 */
internal fun hasUnsentResults(manifest: JobManifest): Boolean =
  manifest.cells.any { it.runStatus == CellRunStatus.COMPLETED && it.serverJobId == null }

/**
 * Should the end-of-run sweep upload anything? The job has to have completed, it has to have opted into contributing, there has to be a registration
 * to submit under, and something has to remain unsent. After the per-cell uploads that last condition is normally false, which keeps a fully
 * successful run from flashing "Submitting results..." for a submit that would do nothing.
 *
 * Deliberately **not** gated on reachability, unlike the per-cell path. The reachability probe is a wall-clock optimization for the inter-cell gap,
 * where a connect timeout would sit between two measurements; at the end of a run there is no measurement left to protect. It also reads false on a
 * firewalled or LAN-only network, exactly the deployments most likely to run an on-prem management server, and before the per-cell upload existed
 * this sweep ran unconditionally. Gating it here would turn that optimization into a silent regression for those devices.
 */
internal fun shouldSweepAtRunEnd(manifest: JobManifest, registration: RegistrationData?): Boolean =
  manifest.status == JobStatus.COMPLETED && manifest.contributeResults == true && registration != null && hasUnsentResults(manifest)

/**
 * The two hooks a run needs into the surrounding Android app. Grouped rather than passed individually because they always travel together, both no-op
 * in tests, and both exist for the same reason: a benchmark run needs the platform to treat two processes a particular way while it is in flight.
 * Grouping also keeps [JobRunner]'s constructor under detekt's parameter ceiling, which per-cell upload and analytics pushed it over from opposite
 * directions.
 *
 * @property launchPocketMode Launches the :benchmark-process pocket-mode Activity (BenchmarkActivity) so that process becomes top-app and inference
 *   regains the prime cores on Samsung. Self-guarding, and a no-op in tests or when unavailable.
 * @property setForegroundHold Holds a foreground service in THIS (main) process while a run is in flight. The pocket screen gives :benchmark the
 *   foreground, which leaves main cached, and the platform kills cached processes for the very binder traffic pushPocketProgress generates. See
 *   JobOrchestratorService.
 */
data class HostHooks(val launchPocketMode: () -> Unit = {}, val setForegroundHold: (Boolean) -> Unit = {})

@Suppress("TooManyFunctions") // the job orchestrator; the pocket-launch helpers pushed it one over
class JobRunner(
  private val storage: JobStore,
  private val engine: BenchmarkEngine,
  private val submissionService: ResultSubmitter,
  private val readiness: ReadinessGate,
  private val analytics: Analytics = NoOpAnalytics,
  /** Injected rather than read from a Context so unit tests stay hermetic; production passes [NetworkReachability.checker]. */
  private val isOnline: () -> Boolean = { true },
  private val executor: Executor = Executors.newSingleThreadExecutor(),
  private val host: HostHooks = HostHooks(),
  private val onStateChanged: (RunnerState) -> Unit,
) {
  @Volatile private var cancelFlag: CancelFlag? = null
  @Volatile private var runningJobId: String? = null
  @Volatile private var startedAtMillis: Long? = null
  @Volatile private var completedInRun: Int = 0

  /**
   * Cells this run actually finished, for [AnalyticsEvents.CELLS_COMPLETED].
   *
   * Deliberately NOT [completedInRun], which is the loop *position* the progress UI wants ("how many cells precede this one") and so is one short at
   * the end of a full run and counts skipped/failed cells too. This one is incremented only where a cell reaches [CellRunStatus.COMPLETED], matching
   * how iOS's `job_completed` counts `runStatus == .completed` cells.
   */
  @Volatile private var cellsCompletedInRun: Int = 0
  @Volatile private var totalInRun: Int = 0
  @Volatile private var coolingSinceMillis: Long? = null
  // The benchmark definitions for the running job, so publishCell can build the rich cell label
  // (display name + param summary) from a cell's typed definition. Set at the top of execute().
  @Volatile private var currentBenchmarkMap: Map<String, BenchmarkDefinition> = emptyMap()
  // Constant-per-run pocket-screen strings (job date + "N models · M benchmarks"), stashed once at
  // the top of execute() so pushPocketProgress can include them without re-deriving from the manifest.
  @Volatile private var currentJobTitle: String = "Benchmark job"
  @Volatile private var currentJobSubtitle: String = ""
  // Whole-job cell counts, refreshed from the manifest as cells finish. The pocket bar and counter
  // report MANIFEST progress like the Compose PocketModeScreen (RunProgress.manifestFraction +
  // manifest.completedCells), so a resumed job continues the bar instead of restarting it at 0.
  @Volatile private var currentJobCompletedCells: Int = 0
  @Volatile private var currentJobTotalCells: Int = 0

  // Last pocket-push dedup key; skips redundant cross-process mirrors (see pushPocketProgress).
  @Volatile private var lastPocketPushKey: String? = null
  // Whether this run has launched the pocket-mode Activity yet (see launchPocketOnce).
  @Volatile private var pocketLaunched: Boolean = false

  // Launch the :benchmark pocket-mode Activity at most once per run, deferred to
  // the first cell that actually loads a model (so the service is bound and the
  // run-ended push can finish the activity). host.launchPocketMode self-guards
  // (AppContainer wraps startActivity in runCatching).
  private fun launchPocketOnce() {
    if (!pocketLaunched) {
      pocketLaunched = true
      host.launchPocketMode()
    }
  }

  fun isRunning(): Boolean = runningJobId != null

  fun cancel() {
    cancelFlag?.cancel()
  }

  fun startNewJob(
    models: List<ModelFile>,
    mmprojFiles: List<ModelFile>,
    benchmarks: List<BenchmarkDefinition>,
    selectedMmprojPaths: Set<String>,
    nGpuLayers: Int,
    contextSize: Int,
    prefillBatch: Int,
    contributeResults: Boolean,
  ): String {
    require(models.isNotEmpty()) { "Select at least one model" }
    require(benchmarks.isNotEmpty()) { "Select at least one benchmark" }
    val cells = planCells(models, mmprojFiles, benchmarks, selectedMmprojPaths)
    require(cells.isNotEmpty()) { "No runnable cells for the selected model/benchmark set" }

    val manifest =
      JobManifest(
        nGpuLayers = nGpuLayers,
        contextSize = BenchmarkContextSize.effective(contextSize, benchmarks),
        cells = cells,
        status = JobStatus.RUNNING,
        contributeResults = contributeResults,
        prefillBatch = prefillBatch,
      )
    storage.saveJobManifest(manifest)
    run(manifest.jobId, source = RunSource.NEW)
    return manifest.jobId
  }

  fun resume(jobId: String) {
    val manifest = storage.loadJobManifest(jobId) ?: error("Job not found")
    manifest.cells.forEach { cell -> if (cell.runStatus == CellRunStatus.CANCELLED) cell.runStatus = CellRunStatus.PENDING }
    manifest.status = JobStatus.RUNNING
    storage.saveJobManifest(manifest)
    run(jobId, source = RunSource.RESUME)
  }

  fun retryFailed(jobId: String) {
    val manifest = storage.loadJobManifest(jobId) ?: error("Job not found")
    val failedIds = manifest.cells.filter { it.runStatus == CellRunStatus.FAILED }.map { it.cellId }.toSet()
    require(failedIds.isNotEmpty()) { "No failed cells to retry" }
    rerunCells(jobId, failedIds, source = RunSource.RETRY_FAILED)
  }

  fun rerunCells(jobId: String, cellIds: Set<String>, source: RunSource = RunSource.RERUN) {
    require(cellIds.isNotEmpty()) { "Select at least one cell to rerun" }
    val manifest = storage.loadJobManifest(jobId) ?: error("Job not found")
    var retryCount = 0
    manifest.cells.forEach { cell ->
      if (cellIds.contains(cell.cellId) && cell.isRerunnable) {
        cell.runStatus = CellRunStatus.PENDING
        cell.errorMessage = null
        cell.serverJobId = null
        storage.clearCellArtifacts(jobId, cell.cellId)
        retryCount += 1
      }
    }
    require(retryCount > 0) { "No matching cells to rerun" }
    manifest.status = JobStatus.RUNNING
    storage.saveJobManifest(manifest)
    run(jobId, source = source)
  }

  private fun run(jobId: String, source: RunSource) {
    check(runningJobId == null) { "A job is already running" }
    val flag = CancelFlag()
    cancelFlag = flag
    runningJobId = jobId
    // Claim the foreground hold BEFORE any work (and before the first progress mirror), while the
    // caller's UI is still foreground and a background-FGS-start restriction can't refuse us.
    host.setForegroundHold(true)
    startedAtMillis = System.currentTimeMillis()
    completedInRun = 0
    cellsCompletedInRun = 0
    coolingSinceMillis = null
    pocketLaunched = false
    // Seed totalInRun BEFORE the first "Starting…" publish (execute() re-derives the same value on
    // the executor thread). Otherwise runFraction sees totalInRun == 0 and falls back to the whole-
    // manifest fraction — for a resumed ~90%-done job that momentarily prints a bogus "0s left".
    totalInRun =
      runCatching {
          val manifest = storage.loadJobManifest(jobId)
          val benchmarkMap = resolveBenchmarks(manifest)
          manifest?.let { pendingExecutionOrder(it, benchmarkMap).size } ?: 0
        }
        .getOrDefault(0)
    publish(RunnerState(runningJobId = jobId, currentProgressText = "Starting...", startedAtMillis = startedAtMillis, totalInRun = totalInRun))

    // Capture-then-flush BEFORE the executor picks the job up, so the analytics queue is empty
    // entering the measurement window and the SDK's periodic flush timer has nothing to send
    // while a cell is being timed. The flush's network call overlaps model load (seconds), not a
    // timed measurement. Nothing else in this class emits events. See [AnalyticsEvents].
    analytics.capture(
      AnalyticsEvents.JOB_STARTED,
      mapOf(AnalyticsEvents.JOB_ID to jobId, AnalyticsEvents.CELL_COUNT to totalInRun, AnalyticsEvents.SOURCE to source.wireName),
    )
    analytics.flush()

    // If the work never gets queued (e.g. a rejected execution), nothing downstream will ever run
    // the finally below, so drop the hold here rather than stranding a foreground service.
    val queued = runCatching {
      executor.execute {
        try {
          execute(jobId, flag)
        } finally {
          // Read the counters BEFORE they are cleared below.
          captureJobCompleted(analytics, storage, jobId, flag.isCancelled, totalInRun, cellsCompletedInRun, startedAtMillis)
          // Clear the counters BEFORE `runningJobId`, which is the `check` guard in `run()`: the moment it goes null another run may start and seed
          // its own counters, and clearing afterwards would zero them out from under it, costing that run its `duration_ms`.
          cancelFlag = null
          startedAtMillis = null
          completedInRun = 0
          cellsCompletedInRun = 0
          totalInRun = 0
          runningJobId = null
          coolingSinceMillis = null
          currentBenchmarkMap = emptyMap()
          publish(RunnerState())
          // Release last: publish() above still mirrors over binder, so the hold must outlive it.
          runCatching { host.setForegroundHold(false) }
        }
      }
    }
    if (queued.isFailure) {
      runningJobId = null
      cancelFlag = null
      runCatching { host.setForegroundHold(false) }
      throw queued.exceptionOrNull() ?: IllegalStateException("Failed to queue job $jobId")
    }
  }

  private fun execute(jobId: String, flag: CancelFlag) {
    var manifest = storage.loadJobManifest(jobId) ?: return
    val benchmarkMap = resolveBenchmarks(manifest)
    currentBenchmarkMap = benchmarkMap
    // Constant pocket strings for the run (the model/benchmark set doesn't change mid-run).
    lastPocketPushKey = null // force the first snapshot of this run through the dedup
    currentJobTitle = DateFormats.shortDate(manifest.createdAt)
    val modelCount = manifest.cells.map { it.modelName }.toSet().size
    val benchmarkCount = manifest.cells.map { it.benchmarkId }.toSet().size
    currentJobSubtitle = "$modelCount ${plural("model", modelCount)} · $benchmarkCount ${plural("benchmark", benchmarkCount)}"
    currentJobCompletedCells = manifest.completedCells
    currentJobTotalCells = manifest.totalCells
    val failedModelPaths = mutableMapOf<String, String>()

    fun saveManifest() {
      val latest = storage.loadJobManifest(manifest.jobId)
      if (latest != null) {
        manifest.contributeResults = latest.contributeResults
        manifest.title = latest.title
        // Preserve serverJobIds a concurrent submit (the UI submitting a completed cell of the
        // running job, or auto-submit) wrote to disk after we loaded: our in-memory manifest never
        // sets them, so a full overwrite would drop them. Only fill blanks — never resurrect an id
        // for a cell being rerun (rerunCells persists serverJobId=null before this run starts).
        val latestServerIds = latest.cells.associate { it.cellId to it.serverJobId }
        manifest.cells.forEach { cell -> if (cell.serverJobId.isNullOrBlank()) cell.serverJobId = latestServerIds[cell.cellId] }
      }
      storage.saveJobManifest(manifest)
    }

    // Read once: the registration cannot change mid-run, and this is a JSON file read on the runner thread that would otherwise repeat after every
    // cell. `perCellUploadHealthy` is the circuit breaker described at the upload site below.
    val registration = storage.loadRegistration()
    var perCellUploadHealthy = true

    val order = pendingExecutionOrder(manifest, benchmarkMap)
    totalInRun = order.size

    for ((position, index) in order.withIndex()) {
      if (flag.isCancelled) break
      completedInRun = position
      val cell = manifest.cells[index]

      failedModelPaths[cell.modelPath]?.let { reason ->
        cell.runStatus = CellRunStatus.FAILED
        cell.errorMessage = reason
        saveManifest()
        publishCell(manifest, cell, "Skipped after model load failure", 0.0)
        return@let
      }
      if (cell.runStatus == CellRunStatus.FAILED) continue

      val item = benchmarkMap[cell.benchmarkId]
      if (item == null) {
        cell.runStatus = CellRunStatus.FAILED
        cell.errorMessage = "Benchmark definition not found"
        saveManifest()
        continue
      }

      // What this cell is, resolved once for the whole iteration. [JobCell.benchmarkType] is nullable (a manifest written before it existed decodes
      // to null) and its stored string may not parse, so every read of it needs the same fallback to [item], the definition the cell is executed
      // from. Resolving per use is how the fresh-load check below came to disagree with the rest of the loop about a typeless cell.
      //
      // [cellContextSize] keeps its own copy of this fallback rather than taking this value, because the ordering comparator calls it before the
      // loop, over cells that have no iteration to hoist out of.
      val cellType = cell.benchmarkType?.let(BenchmarkType::fromWire) ?: item.type

      val resolvedModelPath = storage.resolveModelPath(cell.modelPath)
      if (resolvedModelPath == null) {
        cell.runStatus = CellRunStatus.FAILED
        cell.errorMessage = "Model file not found: ${File(cell.modelPath).name}. Re-download it from the Models tab."
        saveManifest()
        publishCell(manifest, cell, cell.errorMessage ?: "Model file not found", 0.0)
        continue
      }
      val resolvedMmprojPath = storage.resolveModelPath(cell.mmprojPath)
      if (cell.mmprojPath != null && resolvedMmprojPath == null) {
        cell.runStatus = CellRunStatus.FAILED
        cell.errorMessage = "mmproj file not found: ${File(cell.mmprojPath ?: "").name}. Re-download it from the Models tab."
        saveManifest()
        publishCell(manifest, cell, cell.errorMessage ?: "mmproj file not found", 0.0)
        continue
      }

      cell.runStatus = CellRunStatus.RUNNING
      cell.errorMessage = null
      saveManifest()

      val cellContext = cellContextSize(cell, benchmarkMap)
      publishCell(manifest, cell, "Running...", 0.0)
      try {
        // Asked of the resolved type rather than restated here. This was `cell.benchmarkType == MAX_MEMORY_USAGE.wire`, which compares a nullable
        // wire string, is not a type error, and quietly answered false for a typeless cell, routing it down the load-then-measure path below where
        // the allocation it exists to measure has already happened.
        val needsFreshLoad = cellType.requiresFreshLoad
        val resultJson: String
        var latestCellFraction = 0.0
        val progressCallback = BenchmarkProgressCallback { completed, total, message ->
          // The service-side readiness gate rides this channel, tagging cooldown status with the
          // COOLING sentinel; treat it as cooling (keep the fraction) rather than measured progress.
          val cooling = total == COOLING_PROGRESS_TOTAL
          latestCellFraction =
            if (!cooling && total > 0) {
              (completed.toDouble() / total.toDouble()).coerceIn(0.0, 1.0)
            } else {
              latestCellFraction
            }
          // On the cooling sentinel the service reuses `completed` (unused for a cooldown, which has
          // no fraction) to carry its gate's elapsed millis, so the timer anchors to that invocation.
          publishCell(manifest, cell, message, latestCellFraction, cooling = cooling, coolingElapsedMillis = if (cooling) completed.toLong() else 0L)
          !flag.isCancelled
        }
        val cooldownCallback = BenchmarkCooldownCallback {
          val outcome =
            readiness.waitUntilReady(
              cancelFlag = flag,
              onStatus = { text, elapsed -> publishCell(manifest, cell, text, latestCellFraction, cooling = true, coolingElapsedMillis = elapsed) },
            )
          // The gate's own verdict, not `!flag.isCancelled`: a timeout has to reach the native kernel as a timeout so it fails the cell instead of
          // admitting a rep the device was never cool enough for (PIP-143). The kernel turns it into a `PipetteError::Readiness` whose message this
          // runner then records as the cell's `errorMessage`, which is where a readiness failure becomes observable.
          //
          // A cancel that lands during the wait still takes precedence: cancelling is the user's instruction and outranks reporting why the device
          // was too hot, and the resulting cell status is CANCELLED rather than FAILED.
          if (flag.isCancelled) ReadinessOutcome.Cancelled.encode() else outcome.encode()
        }
        // Bring up the pocket-mode Activity in :benchmark the first time a cell
        // is actually about to load — at this point the service binds, so the
        // terminal running=false push (which finishes the activity) is
        // guaranteed to reach it. Launching earlier (at run start) risks
        // orphaning the screen with FLAG_KEEP_SCREEN_ON when every pending cell
        // fails before load (e.g. deleted model files → each cell continues
        // without binding the service).
        launchPocketOnce()
        if (needsFreshLoad) {
          // max_memory_usage must observe the model load itself, so it
          // runs on a fresh load (the actor frees any resident model first).
          publishCell(manifest, cell, "Loading…", 0.0)
          resultJson =
            engine.runFreshSync(
              benchmarkJson = item.rawJson.toString(),
              modelPath = resolvedModelPath,
              nGpuLayers = manifest.nGpuLayers,
              contextSize = cellContext,
              nUbatch = manifest.prefillBatch,
              mmprojPath = resolvedMmprojPath,
              progress = progressCallback,
              cooldown = cooldownCallback,
              // The real sampler is built service-side (PipetteBenchmarkService);
              // a UI-process callback can't be invoked from the native run in :benchmark.
              thermal = null,
            )
        } else {
          // Always load the model fresh, sized to this benchmark's
          // context. The mlx/llama.cpp clients start a fresh server per
          // benchmark, so no benchmark runs on a model warmed or sized
          // for a different one.
          publishCell(manifest, cell, "Loading…", 0.0)
          try {
            engine.loadSync(resolvedModelPath, manifest.nGpuLayers, cellContext, manifest.prefillBatch)
          } catch (error: Throwable) {
            failedModelPaths[cell.modelPath] = formatJobError(error, cellContext)
            throw error
          }
          resultJson =
            engine.runBenchmarkSync(
              benchmarkJson = item.rawJson.toString(),
              nGpuLayers = manifest.nGpuLayers,
              mmprojPath = resolvedMmprojPath,
              progress = progressCallback,
              cooldown = cooldownCallback,
              // Built service-side (see runFreshSync above / RemoteBenchmarkEngine).
              thermal = null,
            )
        }

        // Diagnostic: the :benchmark process's scheduling group (where inference
        // ran), for OEM-cpuset-demotion analysis. The side-by-side logging (main
        // vs :benchmark) lives in RemoteBenchmarkEngine, which runs in the main
        // process and holds both snapshots.
        val benchmarkCpuAffinity = engine.benchmarkProcessCpuAffinity()

        storage.writePayload(
          resultJson = resultJson,
          cellId = cell.cellId,
          jobId = manifest.jobId,
          modelName = cell.modelName,
          modelPath = cell.modelPath,
          mmprojPath = cell.mmprojPath,
          nGpuLayers = manifest.nGpuLayers,
          contextSize = cellContext,
          runtimeVersion = engine.llamaCppCommit(),
          runtimeCpuBackend = engine.cpuBackendDescriptor(),
          // The type resolved once at the top of the iteration, rather than the
          // same fallback spelled out a second time here. #1011 added this site
          // while this fix was in review, and two copies of the rule is the
          // shape the bug came in.
          benchmarkType = cellType,
          // NOTE: this is the between-cell gate, not the per-rep one. Each rep is
          // bracketed by `PipetteBenchmarkService`'s own `Readiness` inside
          // `:benchmark` (RemoteBenchmarkEngine deliberately does not forward the
          // cooldown callback), and that instance's policy never crosses back over
          // AIDL. The two do not disagree today because both read
          // `Readiness.COOLDOWN_MAX_MILLIS`, and the only way either waives the
          // thermal criterion is `PIPETTE_READINESS_SKIP_THERMAL`, which nothing
          // that launches the app sets. PIP-434 adds a real waiver surface and is
          // where the service's resolved policy has to start riding the AIDL
          // channel, or this field will report a gate that did not run.
          readinessPolicy = readiness.policy,
          benchmarkCpuAffinity = benchmarkCpuAffinity,
        )
        cell.runStatus = CellRunStatus.COMPLETED
        cellsCompletedInRun += 1
        saveManifest()
        publishCell(manifest, cell, "Completed", 1.0)
        // Upload this result now, before the next cell, so a crash, a low-memory kill or a pulled charger doesn't strand every completed cell's data
        // until the job ends. Matches the iOS client (PIP-358); the sweep after the loop stays as the catch-all for anything that failed here.
        //
        // Placed before the cooldown wait on purpose: the upload then overlaps thermal recovery instead of adding to the gap between measurements,
        // and it is finished well before the next cell is timed. `submit` is synchronous and runs on this same single-thread executor, so it cannot
        // overlap a measurement.
        //
        // `perCellUploadHealthy` is a one-way circuit breaker: once an attempt fails, a management-server outage would otherwise insert a full
        // connect/read timeout into every remaining inter-cell gap, with the batch growing by one payload each time. The sweep still sends the
        // backlog.
        if (!flag.isCancelled && perCellUploadHealthy) {
          publishCell(manifest, cell, "Submitting result…", 1.0)
          perCellUploadHealthy = submitFinishedCells(submissionService, manifest, registration, isOnline)
          publishCell(manifest, cell, "Completed", 1.0)
        }
        if (position < order.size - 1 && !flag.isCancelled) {
          // Between-cell cooldown. Its outcome is deliberately discarded: a timeout here must not fail the cell that just completed, which
          // passed every gate it was actually measured under. Nor need it pre-emptively fail the next cell, since that cell's own first per-rep
          // gate takes the same reading and reports its own TimedOut if the device is still hot, which is where the failure belongs.
          readiness.waitUntilReady(
            cancelFlag = flag,
            onStatus = { text, elapsed -> publishCell(manifest, cell, text, 1.0, cooling = true, coolingElapsedMillis = elapsed) },
          )
        }
      } catch (error: Throwable) {
        if (flag.isCancelled) {
          cell.runStatus = CellRunStatus.CANCELLED
        } else {
          cell.runStatus = CellRunStatus.FAILED
          cell.errorMessage = formatJobError(error, cellContext)
        }
        runCatching { engine.unloadSync() }
        saveManifest()
        publishCell(manifest, cell, cell.errorMessage ?: "Cancelled", 0.0)
        if (flag.isCancelled) break
      }
    }

    runCatching { engine.unloadSync() }

    manifest = storage.loadJobManifest(jobId) ?: manifest
    if (flag.isCancelled) {
      manifest.cells.forEach { cell -> if (cell.runStatus == CellRunStatus.PENDING) cell.runStatus = CellRunStatus.CANCELLED }
      manifest.status =
        if (manifest.cells.any { it.runStatus == CellRunStatus.CANCELLED }) {
          JobStatus.PAUSED
        } else {
          JobStatus.COMPLETED
        }
    } else {
      manifest.status = JobStatus.COMPLETED
    }
    saveManifest()

    // Final sweep. The cells were already sent inside the loop, so this usually has nothing to do,
    // it exists to catch any whose upload hit a transient error, or that the circuit breaker skipped.
    // Re-read rather than reusing the loop's copy: a device can register mid-run.
    val sweepRegistration = storage.loadRegistration()
    if (shouldSweepAtRunEnd(manifest, sweepRegistration) && sweepRegistration != null) {
      publish(
        RunnerState(
          runningJobId = jobId,
          currentProgressText = "Submitting results...",
          currentCellFraction = 1.0,
          startedAtMillis = startedAtMillis,
          // All cells are done here, so report completed == total (the loop counter is still
          // the last index). That holds the run-relative fraction at 1.0 through submission, so
          // neither the ETA nor the Compose card's run progress snaps back to 0%.
          completedInRun = totalInRun,
          totalInRun = totalInRun,
        )
      )
      // Not wrapped: this is after the last measurement, so a throw here can only reach `run`'s finally, which is where a failed run belongs anyway.
      submissionService.submit(manifest, sweepRegistration)
    }
  }

  private fun publishCell(
    manifest: JobManifest,
    cell: JobCell,
    text: String,
    fraction: Double,
    cooling: Boolean = false,
    coolingElapsedMillis: Long = 0L,
  ) {
    // Derive the cooldown anchor from the elapsed the GATE reports, rather than latching a stamp on
    // the first cooling status. The UI counts against COOLDOWN_MAX_MILLIS, which bounds a single
    // waitUntilReady call; there are three independent gate call sites (before a cell, after a cell,
    // and per rep in :benchmark), so a latched stamp survived across consecutive invocations and the
    // timer read past its own max ("Cooling 3:54 / 3:00 max"). Re-anchoring on every cooling status
    // keeps elapsed describing the wait actually in progress. Cleared the moment a non-cooling
    // status (loading / measuring / completed) arrives.
    coolingSinceMillis = if (cooling) System.currentTimeMillis() - coolingElapsedMillis.coerceAtLeast(0L) else null
    // Keep the pocket's whole-job counts current: the manifest's completedCells advances as cells
    // finish, and this is the one publish path that carries a manifest.
    currentJobCompletedCells = manifest.completedCells
    currentJobTotalCells = manifest.totalCells
    publish(
      RunnerState(
        runningJobId = manifest.jobId,
        currentCellLabel = liveCellLabel(cell, currentBenchmarkMap[cell.benchmarkId]),
        currentProgressText = text,
        currentCellFraction = fraction,
        startedAtMillis = startedAtMillis,
        completedInRun = completedInRun,
        totalInRun = totalInRun,
        coolingSinceMillis = coolingSinceMillis,
      )
    )
  }

  private fun publish(state: RunnerState) {
    onStateChanged(state)
    // Mirror job-level progress to the :benchmark pocket-mode UI (BenchmarkActivity).
    runCatching { pushPocketProgress(state) }
  }

  private fun pushPocketProgress(state: RunnerState) {
    val running = state.runningJobId != null
    val jobTotal = currentJobTotalCells
    // Whole-job fraction for the bar, mirroring RunProgress.manifestFraction on the Compose side.
    val overall =
      if (running && jobTotal > 0) {
        ((currentJobCompletedCells + state.currentCellFraction.coerceIn(0.0, 1.0)) / jobTotal).coerceIn(0.0, 1.0)
      } else {
        0.0
      }
    // The ETA stays RUN-relative (RunProgress.runFraction), like the Compose pocket: elapsed time
    // only covers this run, so extrapolating it over whole-job progress would badly under-report a
    // resumed job's remaining time.
    val runFraction = RunProgress.runFraction(state, overall)
    val snapshot =
      BenchmarkProgressBus.Progress(
        title = currentJobTitle,
        subtitle = currentJobSubtitle,
        cellLabel = state.currentCellLabel,
        statusText = state.currentProgressText,
        completedCells = currentJobCompletedCells,
        totalCells = jobTotal,
        overallPermil = (overall * 1000).toInt(),
        // "calculating" until past 2%, as on the Compose card.
        etaText = RunProgress.estimatedTimeLeft(state, runFraction) ?: "calculating",
        coolingSinceMillis = state.coolingSinceMillis,
        running = running,
      )
    // Coalesce the cross-process (oneway Binder) mirror: RunnerState churns on
    // every native progress tick, but only the discrete pocket-visible fields
    // matter (the bar moves in 0.1% steps; the ETA/status labels are derived).
    // Skipping no-op-key pushes keeps this main→:benchmark hop from firing on
    // pure text/fraction jitter — the reviewer's Binder-frequency concern.
    val key = pocketPushKey(snapshot)
    if (key == lastPocketPushKey) return
    lastPocketPushKey = key
    engine.publishJobProgress(snapshot)
  }

  // Discrete fields that should trigger a pocket refresh — everything the user
  // perceives as a state change (percent step, cell boundary, cooling toggle,
  // run start/stop). Derived labels (etaText/statusText) ride these; they don't
  // warrant their own IPC push.
  private fun pocketPushKey(p: BenchmarkProgressBus.Progress): String =
    "${p.overallPermil}|${p.completedCells}|${p.totalCells}|${p.cellLabel}|${p.cooling}|${p.running}"

  companion object {
    /**
     * Resolve every distinct benchmark id in [manifest]'s cells to its definition — the synced catalog first, else a structurally-parsed id
     * ([BenchmarkCatalog.resolve]) so the four ladder types still resolve their context/params when the catalog is empty (offline / before the first
     * sync). Ids that resolve to nothing (VL/eval when uncatalogued) are simply absent; callers fall back to the cell's stored type / a default.
     */
    internal fun resolveBenchmarks(manifest: JobManifest?): Map<String, BenchmarkDefinition> =
      manifest?.cells.orEmpty().map { it.benchmarkId }.distinct().mapNotNull { id -> BenchmarkCatalog.resolve(id)?.let { id to it } }.toMap()

    internal fun pendingExecutionOrder(manifest: JobManifest, benchmarkMap: Map<String, BenchmarkDefinition>): List<Int> =
      manifest.cells.indices
        .filter { manifest.cells[it].runStatus == CellRunStatus.PENDING }
        .sortedWith(compareBy<Int> { manifest.cells[it].modelPath }.thenBy { cellContextSize(manifest.cells[it], benchmarkMap) })

    internal fun cellContextSize(cell: JobCell, benchmarkMap: Map<String, BenchmarkDefinition>): Int {
      val item = benchmarkMap[cell.benchmarkId] ?: return 4096
      return BenchmarkContextSize.perCell(cell.benchmarkType ?: item.benchmarkType, item.rawJson)
    }

    /**
     * Human-readable "<benchmark> <params> · <model>" label for the running cell, shown live in the progress views (iOS JobExecutor.liveCellLabel).
     * Uses the benchmark's display name plus a compact token summary — so several benchmarks of the same type (e.g. two decode runs at different
     * token counts) are distinguishable — and the model's repo tail rather than the full org/repo. Falls back to the wire type / raw id when the
     * typed definition isn't available.
     */
    internal fun liveCellLabel(cell: JobCell, definition: BenchmarkDefinition?): String {
      val name =
        when {
          // Eval is identified by its dataset, not token params.
          definition is BenchmarkDefinition.Eval -> definition.datasetName.ifBlank { cell.benchmarkId }
          definition != null -> definition.type.displayName.let { base -> benchmarkParamSummary(definition)?.let { "$base · $it" } ?: base }
          cell.benchmarkType != null -> BenchmarkCatalog.displayName(cell.benchmarkType)
          else -> cell.benchmarkId
        }
      val model = cell.modelName.substringAfterLast('/')
      return "$name · $model"
    }

    /**
     * Compact token summary distinguishing same-type benchmarks, e.g. "512→100 tok" or "512 tok". Null when there's nothing sizing-related to show.
     */
    private fun benchmarkParamSummary(definition: BenchmarkDefinition): String? =
      when (definition) {
        is BenchmarkDefinition.PrefillThroughput -> "${definition.prefillTokens} tok"
        is BenchmarkDefinition.MaxMemoryUsage -> "${definition.prefillTokens} tok"
        is BenchmarkDefinition.DecodeThroughput -> "${definition.prefillTokens}→${definition.decodeTokens} tok"
        is BenchmarkDefinition.EndToEndLatency -> "${definition.prefillTokens}→${definition.decodeTokens} tok"
        is BenchmarkDefinition.VlThroughput -> {
          val text = if (definition.textTokens > 0) " · ${definition.textTokens} tok" else ""
          "${definition.imageWidth}×${definition.imageHeight}px$text"
        }
        is BenchmarkDefinition.Eval -> null
      }

    fun planCells(
      models: List<ModelFile>,
      mmprojFiles: List<ModelFile>,
      benchmarks: List<BenchmarkDefinition>,
      selectedMmprojPaths: Set<String>,
    ): MutableList<JobCell> {
      val selectedMmprojs = mmprojFiles.filter { selectedMmprojPaths.contains(it.path) }
      val cells = mutableListOf<JobCell>()
      for (model in models) {
        for (benchmark in benchmarks) {
          if (benchmark.type == BenchmarkType.VL_THROUGHPUT) {
            if (!isVlCompatible(model, mmprojFiles)) continue
            for (mmproj in selectedMmprojs) {
              cells +=
                JobCell(
                  benchmarkId = benchmark.benchmarkId.toString(),
                  benchmarkType = benchmark.benchmarkType,
                  modelPath = model.path,
                  modelName = model.hfRepo ?: model.name,
                  mmprojPath = mmproj.path,
                )
            }
          } else {
            cells +=
              JobCell(
                benchmarkId = benchmark.benchmarkId.toString(),
                benchmarkType = benchmark.benchmarkType,
                modelPath = model.path,
                modelName = model.hfRepo ?: model.name,
              )
          }
        }
      }
      return cells
    }

    fun isVlCompatible(model: ModelFile, mmprojFiles: List<ModelFile>): Boolean {
      val stem = LocalStorage.normalizedModelStem(model.name)
      return mmprojFiles.any { mmproj ->
        (model.hfRepo != null && model.hfRepo == mmproj.hfRepo) || LocalStorage.normalizedModelStem(mmproj.name) == stem
      }
    }

    fun formatJobError(error: Throwable, contextSize: Int): String {
      val message = error.message ?: error.javaClass.simpleName
      return if (message.contains("out of memory", ignoreCase = true)) {
        "Model + context size $contextSize exceeded available device memory. " + "Try a smaller quant or model. Underlying error: $message"
      } else {
        message.removePrefix("java.lang.IllegalStateException: ")
      }
    }
  }
}
