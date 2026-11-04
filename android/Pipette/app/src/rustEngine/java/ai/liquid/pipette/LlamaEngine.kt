package ai.liquid.pipette

/**
 * A loaded model plus the JNI bridge to run benchmarks against it. One engine owns one model in memory; [destroy] frees it. Mirrors the concrete Rust
 * `LlamaEngine`. Lifecycle (create / reuse / destroy) is managed by [EngineActor]: do not call these methods off the actor's worker thread.
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
    fun create(path: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: LlamacppLoadMode): LlamaEngine {
      check(NativeLib.isAvailable) { NATIVE_MISSING }
      val ptr = nativeCreate(path, nGpuLayers, contextSize, nUbatch, loadMode.cValue)
      if (ptr == 0L) error("Native model load returned null")
      return LlamaEngine(ptr).apply { this.contextSize = contextSize }
    }

    /** Run a benchmark that loads its own model and unloads when done: the `max_memory_usage` case, where the load is part of the measurement. */
    @Suppress("LongParameterList") // mirrors the JNI entry point below, which cannot take a Kotlin type
    fun runFresh(
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
      check(NativeLib.isAvailable) { NATIVE_MISSING }
      return nativeRunFresh(benchmarkJson, modelPath, nGpuLayers, contextSize, nUbatch, loadMode.cValue, mmprojPath, progress, cooldown, thermal)
    }

    private const val NATIVE_MISSING =
      "Native benchmark library libpipette_android.so is not packaged. " +
        "The Android UI, storage, registration, downloads, job planning, and submission " +
        "are available, but real benchmark execution requires the Android native llama.cpp bridge."

    @JvmStatic external fun nativeLlamaCppCommit(): String

    @JvmStatic external fun nativeLlamaCppTag(): String

    @JvmStatic external fun nativeCpuBackendDescriptor(): String?

    @JvmStatic private external fun nativeCreate(path: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: Int): Long

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
    @Suppress("LongParameterList") // the JNI signature; every argument crosses as a primitive or a String
    private external fun nativeRunFresh(
      benchmarkJson: String,
      modelPath: String,
      nGpuLayers: Int,
      contextSize: Int,
      nUbatch: Int,
      loadMode: Int,
      mmprojPath: String?,
      progress: BenchmarkProgressCallback?,
      cooldown: BenchmarkCooldownCallback?,
      thermal: BenchmarkThermalCallback?,
    ): String
  }
}
