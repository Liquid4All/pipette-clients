package ai.liquid.pipette

/** Progress sink for a running benchmark. Returning false requests cancellation. */
fun interface BenchmarkProgressCallback {
  fun onProgress(completed: Int, total: Int, message: String): Boolean
}

/**
 * Thermal/charge gate consulted before every measured rep (the gate before a cell's first rep also serves as the between-cell cooldown).
 *
 * The return value is a [ReadinessOutcome] in its JNI encoding. See [ReadinessOutcome.encode], which is the one place the wire format is written
 * down. A `String?` rather than the outcome itself because this signature *is* the JNI contract, read by `JavaReadinessCallback::wait_until_ready`
 * via `jni_sig!`; the native side decodes it and `native/benchmarks.rs`'s `readiness_gate` turns a timeout into `PipetteError::Readiness`, failing
 * the cell.
 *
 * It was a `Boolean` (true to proceed) until PIP-143, which could not express "gave up while the device was still hot" and so recorded throttled
 * numbers as ordinary results.
 */
fun interface BenchmarkCooldownCallback {
  fun waitUntilReady(): String?
}

/**
 * Samples device thermal telemetry for per-rep benchmark cells. The native kernel calls these at each measured rep's gate-pass (`before`) and
 * completion (`after`); the returned series are attached to the result payload (`device_android_thermal_headroom_*`,
 * `device_android_thermal_status_*`). Sampling is never fatal — an unavailable reading maps to a missing sample and never cancels the run.
 * (Per-sensor HAL temperatures, `device_android_thermal_sensors_*`, are read natively from `dumpsys` and are not sourced through this callback.)
 */
interface BenchmarkThermalCallback {
  /** Current thermal headroom, or `Float.NaN` when unavailable. */
  fun sampleHeadroom(): Float

  /**
   * Current `PowerManager.getCurrentThermalStatus()` ordinal (0-6). Any value outside that range is treated as unavailable (dropped to a missing
   * sample) by the native side.
   */
  fun sampleStatus(): Int
}

/**
 * Whether the native `libpipette_android.so` benchmark library is packaged in this build, plus the llama.cpp commit it was built from. The actual
 * engine is created and owned by [EngineActor]; this object is the static, stateless face of the JNI binding used by the UI to report availability.
 */
object NativeLib {
  val isAvailable: Boolean = runCatching { System.loadLibrary("pipette_android") }.isSuccess

  fun llamaCppCommit(): String = if (isAvailable) LlamaEngine.nativeLlamaCppCommit() else "native-unavailable"

  /**
   * The active CPU-backend feature descriptor (e.g. `"dotprod,fp16_va,neon"`) of the runtime-selected `libggml-cpu-*` variant, or null when the
   * native library is missing or no model has been loaded yet (the backend registers lazily on first load).
   */
  fun cpuBackendDescriptor(): String? = if (isAvailable) LlamaEngine.nativeCpuBackendDescriptor() else null
}

/**
 * A loaded model plus the JNI bridge to run benchmarks against it. One engine owns one model in memory; [destroy] frees it. Mirrors the concrete Rust
 * `LlamaEngine`. Lifecycle (create / reuse / destroy) is managed by [EngineActor] — do not call these methods off the actor's worker thread.
 */
class LlamaEngine private constructor(private val enginePtr: Long) {
  /** The context size this engine's model was loaded with. */
  var contextSize: Int = 0
    private set

  /** Run a benchmark against the already-loaded model. */
  fun runBenchmark(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String = nativeRunBenchmark(enginePtr, benchmarkJson, nGpuLayers, mmprojPath, progress, cooldown, thermal)

  /** Free the model. The engine must not be used afterwards. */
  fun destroy() {
    if (enginePtr != 0L) nativeDestroy(enginePtr)
  }

  companion object {
    /** Load [path] and keep the model resident. Throws on load failure. */
    fun create(path: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int): LlamaEngine {
      check(NativeLib.isAvailable) { NATIVE_MISSING }
      val ptr = nativeCreate(path, nGpuLayers, contextSize, nUbatch)
      if (ptr == 0L) error("Native model load returned null")
      return LlamaEngine(ptr).apply { this.contextSize = contextSize }
    }

    /** Run a benchmark that loads its own model and unloads when done — the `max_memory_usage` case, where the load is part of the measurement. */
    fun runFresh(
      benchmarkJson: String,
      modelPath: String,
      nGpuLayers: Int,
      contextSize: Int,
      nUbatch: Int,
      mmprojPath: String?,
      progress: BenchmarkProgressCallback?,
      cooldown: BenchmarkCooldownCallback?,
      thermal: BenchmarkThermalCallback?,
    ): String {
      check(NativeLib.isAvailable) { NATIVE_MISSING }
      return nativeRunFresh(benchmarkJson, modelPath, nGpuLayers, contextSize, nUbatch, mmprojPath, progress, cooldown, thermal)
    }

    private const val NATIVE_MISSING =
      "Native benchmark library libpipette_android.so is not packaged. " +
        "The Android UI, storage, registration, downloads, job planning, and submission " +
        "are available, but real benchmark execution requires the Android native llama.cpp bridge."

    @JvmStatic external fun nativeLlamaCppCommit(): String

    @JvmStatic external fun nativeCpuBackendDescriptor(): String?

    @JvmStatic private external fun nativeCreate(path: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int): Long

    @JvmStatic private external fun nativeDestroy(enginePtr: Long)

    @JvmStatic
    private external fun nativeRunBenchmark(
      enginePtr: Long,
      benchmarkJson: String,
      nGpuLayers: Int,
      mmprojPath: String?,
      progress: BenchmarkProgressCallback?,
      cooldown: BenchmarkCooldownCallback?,
      thermal: BenchmarkThermalCallback?,
    ): String

    @JvmStatic
    private external fun nativeRunFresh(
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
  }
}
