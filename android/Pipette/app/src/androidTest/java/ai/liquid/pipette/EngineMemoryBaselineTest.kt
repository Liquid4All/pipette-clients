package ai.liquid.pipette

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device, single-process memory baseline + end-to-end JNI smoke test.
 *
 * Runs a real `max_memory_usage` benchmark through [EngineActor] against a model sideloaded into the app's external models dir, exercising the
 * `nativeCreate`/`nativeRunFresh` JNI surface (including the new `nUbatch` argument). Records VmRSS/VmHWM from `/proc/self/status` — the methodology
 * from inference_engine's MEMORY_BENCHMARK.md — so the pre-isolation footprint is captured before Phase D moves the engine into a `:benchmark`
 * process.
 *
 * Skips (rather than fails) when the native library isn't packaged or the model hasn't been pushed, so it's safe in CI without the model.
 */
@RunWith(AndroidJUnit4::class)
class EngineMemoryBaselineTest {
  @Test
  fun maxMemoryUsageRunsInProcessAndReportsFootprint() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val model = File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf")
    assumeTrue("native library not packaged", NativeLib.isAvailable)
    assumeTrue("model not sideloaded at ${model.absolutePath}", model.exists())

    val benchmark = BenchmarkCatalog.byId("max_memory_usage_512")!!
    val contextSize = BenchmarkContextSize.perCell(benchmark.benchmarkType, benchmark.rawJson)

    val rssBefore = procStatusKb("VmRSS")
    val engine = EngineActor()
    val result =
      try {
        engine.runFreshSync(
          benchmarkJson = benchmark.rawJson.toString(),
          modelPath = model.absolutePath,
          nGpuLayers = 99,
          contextSize = contextSize,
          nUbatch = 512,
          mmprojPath = null,
          progress = null,
          cooldown = null,
          thermal = null,
        )
      } finally {
        engine.destroy()
      }
    val hwm = procStatusKb("VmHWM")

    Log.i(TAG, "[baseline] single-process VmRSS_before=${rssBefore}kB VmHWM=${hwm}kB " + "ctx=$contextSize result=$result")

    // The JNI round-trip succeeded and produced a result payload.
    assertTrue("empty result payload", result.isNotBlank())
    // The model load left a peak well above the empty-process baseline.
    assertTrue("VmHWM ($hwm kB) not above pre-load RSS ($rssBefore kB)", hwm > rssBefore)
  }

  private fun procStatusKb(field: String): Long =
    File("/proc/self/status").readLines().firstOrNull { it.startsWith("$field:") }?.filter { it.isDigit() }?.toLongOrNull() ?: -1L

  companion object {
    private const val TAG = "pipetteMemBaseline"
  }
}
