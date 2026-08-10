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

  /** Load [modelPath] fresh at [contextSize] with prefill micro-batch [nUbatch] (0 → llama.cpp default), freeing any resident model first. */
  fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int)

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
  fun runFreshSync(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
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
