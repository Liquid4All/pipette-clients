package ai.liquid.pipette

/**
 * The engine operations [JobRunner] needs, exposed as blocking `*Sync` calls.
 *
 * [EngineActor] implements this in-process today. Phase D adds a second implementation — an IPC proxy whose `*Sync` helpers round-trip to a
 * `:benchmark` service that owns the native engine — without [JobRunner] changing. Unit tests substitute a fake.
 *
 * Implementations own all model lifecycle (load / reuse / reload / destroy); callers only ask for the operation they want. Every `*Sync` call blocks
 * the caller until the operation completes and re-throws native failures.
 */
@Suppress("TooManyFunctions") // mirrors the IBenchmarkService AIDL surface, like PipetteBenchmarkService's binder
interface BenchmarkEngine {
  /** Whether the native benchmark library is packaged in this build. */
  val isAvailable: Boolean

  /** The llama.cpp commit the native library was built from. */
  fun llamaCppCommit(): String

  /**
   * The active CPU-backend feature descriptor (e.g. `"dotprod,fp16_va,neon"`) of the runtime-selected variant, or null before a model has been loaded
   * / when the native library is missing. Recorded as `runtime_cpu_variant`.
   */
  fun cpuBackendDescriptor(): String?

  /**
   * Diagnostic snapshot of the cpuset / CPU-affinity the engine actually runs under. For the IPC proxy this is read in the `:benchmark` process, so
   * it captures any OEM cpuset demotion (e.g. Samsung placing a non-`top-app` service process on a throttled core set). Null when the `:benchmark`
   * process isn't bound / the read fails (the in-process implementation always returns a snapshot; individual fields degrade to null).
   */
  fun benchmarkProcessCpuAffinity(): CpuAffinitySnapshot?

  /**
   * Why the last run stopped, when it stopped because the engine's process was descheduled mid-measurement: the screen turning off, the pocket screen
   * losing window focus, another app taking `top-app`. Null when the run ended for any other reason.
   *
   * A user cancel does not clear it: if a demotion was observed this still reports it, and the caller decides precedence. `JobRunner` lets the cancel
   * win for the cell's status, because stopping was the user's instruction.
   *
   * **This is a destructive read on the IPC implementation.** The first call returns the verdict and clears it, so a verdict explains at most one
   * cell; a second call returns null. `JobRunner` reads it up to twice per cell and is correct only because the success path carries the value
   * forward in an exception rather than asking again. The in-process implementations do not consume (they never produce a verdict at all), so a test
   * passing against a fake proves nothing about the double-read.
   *
   * Call after the run returns, and on the success path too: a demotion cancels the run, but the run can still come back having thrown, having
   * returned a partial, or having completed in the window before the cancel landed. All three are invalid measurements.
   *
   * The watch behind it is scoped to a cell rather than a run, so it also covers the model load preceding one; a demotion seen there is reported
   * against the run that follows it.
   */
  fun lastRunInterruption(): CpusetDemotion?

  /**
   * Signal the end of a JOB, so the cpuset watch drops the baseline it carries from cell to cell.
   *
   * Deliberately separate from [unloadSync], which happens per cell as well as at the end of a job. The baseline has to survive a mid-job unload: it
   * is what lets a demotion that persists across a cell boundary still read as a demotion, instead of the next cell sampling the demoted state and
   * adopting it as its own normal. It must NOT survive a job boundary, because the `:benchmark` process outlives a job by a short grace period and a
   * job starting inside that window would be judged against the previous job's `top-app` state.
   *
   * Best-effort and safe to call more than once, including when nothing is bound. No-op for the in-process implementations, which never watch.
   */
  fun endJob()

  /**
   * Push whole-job progress to the `:benchmark` process so [BenchmarkActivity]'s pocket-mode UI (which is `top-app` there, for the cpuset boost) can
   * render it. [overallPermil] is the job fraction × 1000; [running] false signals the run ended (the pocket screen finishes itself). No-op for the
   * in-process fake.
   */
  fun publishJobProgress(progress: BenchmarkProgressBus.Progress)

  /**
   * Invoked when the user taps Cancel in the `:benchmark` pocket-mode UI; the owner ([JobController] wiring) sets it to cancel the *whole* job. The
   * proxy forwards this over a reverse AIDL callback from the `:benchmark` process.
   */
  var onJobCancelRequested: (() -> Unit)?

  /**
   * Load [modelPath] fresh at [contextSize] with prefill micro-batch [nUbatch] (0 → llama.cpp default) and weights loaded per [loadMode], freeing any
   * resident model first.
   */
  fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: LlamacppLoadMode)

  /** Run a benchmark against the already-loaded model (requires a prior [loadSync]). */
  fun runBenchmarkSync(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String

  /**
   * Run a benchmark that loads its own model and unloads when done — the `max_memory_usage` case, where the load is part of the measurement. Any
   * resident model is destroyed first for a clean memory baseline.
   */
  @Suppress("LongParameterList") // one cell's whole load + measure request; grouping it would only move the list behind a type
  fun runFreshSync(
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
  ): String

  /** Free the resident model (if any). */
  fun unloadSync()

  /** Free the model and tear down any backing resources. */
  fun destroy()
}
