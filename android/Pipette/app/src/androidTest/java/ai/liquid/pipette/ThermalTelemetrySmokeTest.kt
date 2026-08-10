package ai.liquid.pipette

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end smoke check for the per-rep Android thermal families (PIP-392). Runs a gated benchmark with a real [BenchmarkThermalCallback] wired (as
 * [PipetteBenchmarkService] does in production) and asserts the result payload carries the status series. Sensors are best-effort: they populate only
 * when `android.permission.DUMP` is granted (`adb shell pm grant ai.liquid.pipette android.permission.DUMP`) and the in-app `dumpsys` exec is
 * permitted — so this only logs them, it does not require them. The existing benchmark androidTests pass `thermal = null`, so this is the only test
 * that exercises the sampler → JNI → native `sample_thermal` → `ThermalTelemetry::from_series` path.
 *
 * Requires the sideloaded LFM2-350M Q4_0 model (same fixture as [KleidiaiBenchTest]); skipped via `assumeTrue` when absent.
 */
@RunWith(AndroidJUnit4::class)
class ThermalTelemetrySmokeTest {
  @Test
  fun gatedBenchmarkEmitsThermalStatusAndSensors() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val internal = File(context.filesDir, "LFM2-350M-Q4_0.gguf")
    val external = File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2-350M-GGUF/LFM2-350M-Q4_0.gguf")
    val model = if (internal.exists()) internal else external
    assumeTrue("native library not packaged", NativeLib.isAvailable)
    assumeTrue("Q4_0 model not found (internal=${internal.exists()} external=${external.exists()})", model.exists())

    val provider = AndroidThermalStatusProvider(context)
    val thermal =
      object : BenchmarkThermalCallback {
        override fun sampleHeadroom(): Float = provider.cachedHeadroom()

        override fun sampleStatus(): Int = provider.currentStatus()
      }

    // Inline definition so the test doesn't depend on the (server-synced,
    // unseeded at test time) BenchmarkCatalog. A gated prefill_throughput cell
    // runs MEASURED_REPS reps, each sampling thermal before/after.
    val benchmarkJson = """{"benchmark_id":"prefill_smoke","benchmark_type":"prefill_throughput","parameter_prefill_tokens":128}"""
    val engine = EngineActor()
    val result =
      try {
        engine.runFreshSync(
          benchmarkJson = benchmarkJson,
          modelPath = model.absolutePath,
          nGpuLayers = 99,
          contextSize = 512,
          nUbatch = 512,
          mmprojPath = null,
          progress = null,
          cooldown = null,
          thermal = thermal,
        )
      } finally {
        engine.destroy()
      }
    Log.i(TAG, "[thermal-smoke] result=$result")

    val json = JSONObject(result)
    // Status is sourced from PowerManager (no permission) and reliably populates
    // (all-or-nothing scalar series; getCurrentThermalStatus reports a valid
    // 0-6 status on supported devices, and the native side drops out-of-range
    // values).
    for (key in listOf("device_android_thermal_status_before", "device_android_thermal_status_after")) {
      val arr = json.optJSONArray(key)
      assertTrue("$key missing/empty in payload", arr != null && arr.length() > 0)
      Log.i(TAG, "[thermal-smoke] $key=$arr")
    }
    // Sensors are DUMP-gated / fail-soft: log presence, do not require.
    for (key in listOf("device_android_thermal_sensors_before", "device_android_thermal_sensors_after")) {
      val arr = json.optJSONArray(key)
      Log.i(TAG, "[thermal-smoke] $key present=${arr != null} count=${arr?.length() ?: 0}")
    }
  }

  companion object {
    private const val TAG = "pipetteThermalSmoke"
  }
}
