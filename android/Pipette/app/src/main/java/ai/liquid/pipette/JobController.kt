package ai.liquid.pipette

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * App-scoped owner of the [JobRunner]. Wrapping the runner here (rather than in the Activity) is what makes an in-flight benchmark survive Activity
 * recreation — a rotation, dark-mode toggle, or locale change rebuilds the UI but leaves the runner (and the bound `:benchmark` process) untouched.
 * The runner's imperative state callback is adapted into a [StateFlow] the UI observes; emissions are conflated and delivered on the collector's
 * dispatcher, so the old manual `mainHandler.post` marshaling is gone.
 */
class JobController(
  storage: JobStore,
  engine: BenchmarkEngine,
  submissionService: ResultSubmitter,
  readiness: ReadinessGate,
  host: HostHooks = HostHooks(),
  analytics: Analytics = NoOpAnalytics,
  isOnline: () -> Boolean = { true },
) {
  private val _state = MutableStateFlow(RunnerState())
  val state: StateFlow<RunnerState> = _state.asStateFlow()

  private val runner =
    JobRunner(
      storage = storage,
      engine = engine,
      submissionService = submissionService,
      readiness = readiness,
      host = host,
      analytics = analytics,
      isOnline = isOnline,
    ) {
      _state.value = it
    }

  fun isRunning(): Boolean = runner.isRunning()

  fun cancel() = runner.cancel()

  fun startNewJob(
    models: List<ModelFile>,
    mmprojFiles: List<ModelFile>,
    benchmarks: List<BenchmarkDefinition>,
    selectedMmprojPaths: Set<String>,
    nGpuLayers: Int,
    contextSize: Int,
    prefillBatch: Int,
    contributeResults: Boolean,
  ): String =
    runner.startNewJob(
      models = models,
      mmprojFiles = mmprojFiles,
      benchmarks = benchmarks,
      selectedMmprojPaths = selectedMmprojPaths,
      nGpuLayers = nGpuLayers,
      contextSize = contextSize,
      prefillBatch = prefillBatch,
      contributeResults = contributeResults,
    )

  fun resume(jobId: String) = runner.resume(jobId)

  fun retryFailed(jobId: String) = runner.retryFailed(jobId)

  fun rerunCells(jobId: String, cellIds: Set<String>) = runner.rerunCells(jobId, cellIds)
}
