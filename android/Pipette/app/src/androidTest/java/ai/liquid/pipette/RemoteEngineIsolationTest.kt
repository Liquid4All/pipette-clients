package ai.liquid.pipette

import ai.liquid.pipette.service.RemoteBenchmarkEngine
import android.app.ActivityManager
import android.content.Context
import android.os.Process
import android.util.Log
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device end-to-end test for Phase D process isolation. Runs a real `max_memory_usage` benchmark through [RemoteBenchmarkEngine], which binds the
 * `:benchmark` service, loads + runs + unloads across the AIDL boundary, and tears the service down. Verifies the engine ran in a *separate* process
 * (the memory win), logs that process's footprint, and confirms teardown kills it so the model heap is reclaimed.
 *
 * A foreground [MainActivity] is kept resumed so the proxy is allowed to `startForegroundService`; it never issues an engine op itself (the proxy
 * binds lazily), so this test's engine is the service's only client and its teardown is unobstructed. Skips when the native lib or model isn't
 * present.
 */
@RunWith(AndroidJUnit4::class)
class RemoteEngineIsolationTest {
  @Test
  fun runsInSeparateProcessAndReclaimsOnTeardown() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val model = File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2-350M-GGUF/LFM2-350M-Q4_K_M.gguf")
    val engine = RemoteBenchmarkEngine(context)
    assumeTrue("native library not packaged", engine.isAvailable)
    assumeTrue("model not sideloaded at ${model.absolutePath}", model.exists())

    val benchmark = BenchmarkCatalog.byId("max_memory_usage_512")!!
    val contextSize = BenchmarkContextSize.perCell(benchmark.benchmarkType, benchmark.rawJson)

    ActivityScenario.launch(MainActivity::class.java).use {
      val result =
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

      // The JNI round-trip crossed Binder and produced a payload.
      assertTrue("empty result payload", result.isNotBlank())
      val json = JSONObject(result)
      assertTrue("result missing max_ram_bytes", json.has("max_ram_bytes"))

      // The engine ran in the :benchmark process, not the UI/test process.
      val benchPid = benchmarkProcessPid(context)
      assertNotNull(":benchmark process should be alive right after a run", benchPid)
      assertNotEquals("engine must not run in the UI/test process", Process.myPid(), benchPid)
      Log.i(
        TAG,
        "[isolation] uiPid=${Process.myPid()} benchPid=$benchPid " +
          "benchVmHWM=${procStatusKb(benchPid!!, "VmHWM")}kB " +
          "engineMaxRam=${json.optLong("max_ram_bytes")}",
      )

      // Teardown kills the process → the model's native heap is reclaimed.
      engine.destroy()
      waitUntilGone(context, TEARDOWN_TIMEOUT_MS)
      assertNull(":benchmark must be killed after teardown", benchmarkProcessPid(context))
    }
  }

  private fun benchmarkProcessPid(context: Context): Int? {
    val am = context.getSystemService(ActivityManager::class.java)
    val want = "${context.packageName}:benchmark"
    return am.runningAppProcesses?.firstOrNull { it.processName == want }?.pid
  }

  private fun waitUntilGone(context: Context, timeoutMs: Long) {
    val deadline = System.currentTimeMillis() + timeoutMs
    while (System.currentTimeMillis() < deadline && benchmarkProcessPid(context) != null) {
      Thread.sleep(200)
    }
  }

  private fun procStatusKb(pid: Int, field: String): Long =
    runCatching { File("/proc/$pid/status").readLines().firstOrNull { it.startsWith("$field:") }?.filter(Char::isDigit)?.toLongOrNull() }.getOrNull()
      ?: -1L

  companion object {
    private const val TAG = "pipetteRemoteIso"
    private const val TEARDOWN_TIMEOUT_MS = 4_000L
  }
}
