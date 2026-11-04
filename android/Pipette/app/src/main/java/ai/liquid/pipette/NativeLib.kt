package ai.liquid.pipette

/** Progress sink for a running benchmark. Returning false requests cancellation. */
fun interface BenchmarkProgressCallback {
  fun onProgress(completed: Int, total: Int, message: String): Boolean
}

/** Thermal/charge gate consulted before every measured rep. The return value is a ReadinessOutcome in its JNI encoding. */
fun interface BenchmarkCooldownCallback {
  fun waitUntilReady(): String?
}

/** Samples device thermal telemetry for per-rep benchmark cells. */
interface BenchmarkThermalCallback {
  /** Current thermal headroom, or Float.NaN when unavailable. */
  fun sampleHeadroom(): Float

  /** Current PowerManager.getCurrentThermalStatus() ordinal (0-6). Any value outside that range is treated as unavailable. */
  fun sampleStatus(): Int
}

/** Static availability and metadata face for the native libpipette_android.so benchmark library. */
object NativeLib {
  val isAvailable: Boolean = runCatching { System.loadLibrary("pipette_android") }.isSuccess

  fun llamaCppCommit(): String = if (isAvailable) nativeLlamaCppCommit() else "native-unavailable"

  /**
   * The upstream `bNNNN` tag on the vendored llama.cpp pin, or null when it has none and when the native library is missing.
   *
   * Null rather than a placeholder, so Capabilities (which owns the rule for why the tag is reported at all) has one absent value to test rather than
   * a sentinel that reads like a build id.
   */
  fun llamaCppTag(): String? = if (isAvailable) nativeLlamaCppTag().ifEmpty { null } else null

  fun cpuBackendDescriptor(): String? = if (isAvailable) nativeCpuBackendDescriptor() else null

  @JvmStatic external fun nativeLlamaCppCommit(): String

  @JvmStatic external fun nativeLlamaCppTag(): String

  @JvmStatic external fun nativeCpuBackendDescriptor(): String?
}
