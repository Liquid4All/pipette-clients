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
 * On-device prefill + decode throughput on the **Q4_0** model. Serves two A/Bs:
 * - **KleidiAI**: its ggml integration only accelerates `Q4_0`/`Q8_0`/`F32` weights, so a k-quant model wouldn't engage it at all. Run once with the
 *   KleidiAI build and once with a KleidiAI-disabled build.
 * - **CPU-variant**: pin the backend per arm with `adb shell setprop debug.pipette.cpu_variant <tag>` (see `native_loader.cpp`) to compare feature
 *   levels on one build, e.g. `armv9.0_1` (SVE2) against `armv8.6_1` (i8mm only).
 *
 * Either way, compare the `prefill_time_ms` / `decode_time_ms` in the logged result payloads, and confirm which variant registered from the
 * `pipette-cpudispatch` log lines.
 */
@RunWith(AndroidJUnit4::class)
class KleidiaiBenchTest {
  @Test
  fun q40PrefillAndDecodeThroughput() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    // Prefer the app's internal filesDir (push it there with `run-as cat`,
    // which is binary-safe and avoids external scoped-storage perm issues),
    // falling back to the models dir the app downloads into. Resolve that dir
    // the way `LocalStorage` does rather than passing the nullable
    // `getExternalFilesDir(null)` straight to `File(parent, child)`: it returns
    // null whenever external storage is unmounted, and a null parent silently
    // yields a relative path, so the candidate would both miss the real file and
    // print a misleading location in the skip message. Both the LFM2 and LFM2.5
    // 350M Q4_0 builds work; the comparison only needs the two arms to run
    // identical weights.
    val modelsDir = File(context.getExternalFilesDir(null) ?: File(context.filesDir, "Pipette"), "models")
    val candidates =
      listOf(
        File(context.filesDir, "LFM2-350M-Q4_0.gguf"),
        File(context.filesDir, "LFM2.5-350M-Q4_0.gguf"),
        File(modelsDir, "LiquidAI/LFM2-350M-GGUF/LFM2-350M-Q4_0.gguf"),
        File(modelsDir, "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
      )
    val model = candidates.firstOrNull { it.exists() } ?: candidates.first()
    assumeTrue("native library not packaged", NativeLib.isAvailable)
    assumeTrue("Q4_0 model not found; looked in ${candidates.joinToString { it.absolutePath }}", model.exists())
    Log.i(TAG, "[cpu-variant-ab] using model ${model.absolutePath}")

    for (benchId in listOf("prefill_throughput_512", "decode_throughput_512_100")) {
      // `resolve`, not `byId`: `BenchmarkCatalog.all` is populated only from a
      // completed server sync (no bundled fallback), so in an instrumented test
      // it is empty and `byId` returns null. `resolve` falls back to parsing the
      // id structurally, which keeps this test self-contained.
      val benchmark = requireNotNull(BenchmarkCatalog.resolve(benchId)) { "unresolvable benchmark id '$benchId'" }
      val contextSize = BenchmarkContextSize.perCell(benchmark.benchmarkType, benchmark.rawJson)
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
      Log.i(TAG, "[kleidiai-ab] $benchId ctx=$contextSize result=$result")
      assertTrue("empty result payload for $benchId", result.isNotBlank())
    }
  }

  companion object {
    private const val TAG = "pipetteKleidiaiBench"
  }
}
