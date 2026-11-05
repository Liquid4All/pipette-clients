package ai.liquid.pipette.fakes

import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkEngine
import ai.liquid.pipette.BenchmarkProgressBus
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.CpuAffinitySnapshot
import ai.liquid.pipette.CpusetDemotion
import ai.liquid.pipette.LlamacppLoadMode

/**
 * In-memory [BenchmarkEngine] for JVM tests. Records every call, returns a canned result JSON, and can be told to fail a load for specific model
 * paths or to run an arbitrary hook mid-run (used to trip cancellation).
 */
class FakeBenchmarkEngine(
  override val isAvailable: Boolean = true,
  private val resultJson: String = "{}",
  private val commit: String = "test-commit",
) : BenchmarkEngine {
  data class LoadCall(val modelPath: String, val nGpuLayers: Int, val contextSize: Int, val nUbatch: Int, val loadMode: LlamacppLoadMode)

  data class RunCall(
    val fresh: Boolean,
    val modelPath: String?,
    val mmprojPath: String?,
    val nUbatch: Int,
    val loadMode: LlamacppLoadMode? = null,
    val nGpuLayers: Int? = null,
    val contextSize: Int? = null,
  )

  val loads = mutableListOf<LoadCall>()
  val runs = mutableListOf<RunCall>()
  var unloadCount = 0
    private set

  var destroyCount = 0
    private set

  /** How many times a run consulted the cooldown gate it was handed. */
  var cooldownCount = 0
    private set

  /** Model paths whose load (or fresh run) should throw, simulating OOM/load failure. */
  var failLoadForPaths: Set<String> = emptySet()

  /** Hook invoked at the start of each run — tests use it to call `runner.cancel()`. */
  var onRun: (() -> Unit)? = null

  /** When set, each run emits this extra progress tuple first — tests use the COOLING sentinel here. */
  var extraProgress: Triple<Int, Int, String>? = null

  override fun llamaCppCommit(): String = commit

  /** Stubbed active CPU backend descriptor; tests override as needed. */
  var cpuBackend: String? = "dotprod,neon"

  override fun cpuBackendDescriptor(): String? = cpuBackend

  /** Stubbed cpuset/affinity snapshot; tests override as needed. Ignored when [cpuAffinityPerCall] is set. */
  var cpuAffinity: CpuAffinitySnapshot? = null

  /**
   * A reading per call, for tests about a process whose cpuset moves between cells. Consumed in order; once exhausted the last entry repeats, so a
   * test states only as many readings as it cares about.
   */
  var cpuAffinityPerCall: List<CpuAffinitySnapshot?>? = null

  /** How many times the affinity was read. A cached reading shows up here as a count below the cell count. */
  var cpuAffinityReads = 0
    private set

  override fun benchmarkProcessCpuAffinity(): CpuAffinitySnapshot? {
    val index = cpuAffinityReads++
    val perCall = cpuAffinityPerCall ?: return cpuAffinity
    // Indexed rather than `getOrNull(...) ?: last()`, so a deliberate null entry (an unreadable
    // sample) stays null instead of falling through to the last reading.
    return if (index < perCall.size) perCall[index] else perCall.lastOrNull()
  }

  /**
   * The demotion the next run reports, or null for an uninterrupted run. Set it and leave it set; this fake models none of the real clearing rules,
   * so a test that needs a later cell to come back clean has to null it out itself.
   *
   * Two divergences from `RemoteBenchmarkEngine` that a test written against this fake cannot see. The real verdict is cleared when the next *cell*
   * opens (a load, or a fresh run), never when a run ends. And the real read is **destructive**: the first call clears it. So a `JobRunner` change
   * that reads twice per cell stays green here and loses the verdict on a device.
   */
  var runInterruption: CpusetDemotion? = null

  override fun lastRunInterruption(): CpusetDemotion? = runInterruption

  override var onJobCancelRequested: (() -> Unit)? = null

  /** Records job-progress pushes so tests can assert the pocket-mode UI is fed. */
  val jobProgressPushes = mutableListOf<String>()

  override fun publishJobProgress(progress: BenchmarkProgressBus.Progress) {
    jobProgressPushes += progress.toJson()
  }

  override fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: LlamacppLoadMode) {
    loads += LoadCall(modelPath, nGpuLayers, contextSize, nUbatch, loadMode)
    if (modelPath in failLoadForPaths) error("fake load failure for $modelPath")
  }

  override fun runBenchmarkSync(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    runs += RunCall(fresh = false, modelPath = null, mmprojPath = mmprojPath, nUbatch = -1)
    return drive(progress, cooldown)
  }

  override fun runFreshSync(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    if (modelPath in failLoadForPaths) error("fake fresh-load failure for $modelPath")
    runs +=
      RunCall(
        fresh = true,
        modelPath = modelPath,
        mmprojPath = mmprojPath,
        nUbatch = nUbatch,
        loadMode = loadMode,
        nGpuLayers = nGpuLayers,
        contextSize = contextSize,
      )
    return drive(progress, cooldown)
  }

  override fun unloadSync() {
    unloadCount++
  }

  /**
   * Counted separately from [unloadCount] so a test can pin the distinction the real implementations depend on: an unload keeps the cpuset baseline
   * and happens per cell, this drops it and must happen only at a job boundary.
   */
  var endJobCount = 0

  override fun endJob() {
    endJobCount++
  }

  override fun destroy() {
    destroyCount++
  }

  // Mirror the native contract: invoke the run hook, report progress, then
  // consult the cooldown gate between steps. A false progress return means
  // cancellation was requested, which the real engine surfaces by aborting the
  // run — modeled here as a thrown error.
  //
  // The cooldown gate reports a `ReadinessOutcome` encoding (null = ready), and
  // a non-null one aborts the run just as `android/native/benchmarks.rs`'s
  // `readiness_gate` does: cancelled and timed-out both fail the cell rather
  // than letting a rep proceed (PIP-143).
  private fun drive(progress: BenchmarkProgressCallback?, cooldown: BenchmarkCooldownCallback?): String {
    onRun?.invoke()
    extraProgress?.let { (completed, total, message) -> progress?.onProgress(completed, total, message) }
    val proceed = progress?.onProgress(1, 1, "running") ?: true
    if (!proceed) error("benchmark cancelled")
    cooldownCount++
    val notReady = cooldown?.waitUntilReady()
    if (notReady != null) error("benchmark aborted before a measured rep: $notReady")
    return resultJson
  }
}
