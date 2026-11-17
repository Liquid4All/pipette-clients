package ai.liquid.pipette.onnx

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests executing benchmarks through [OnnxGenaiBenchmark] on device under kotlinEngine.
 *
 * They need an ONNX Runtime GenAI package on the device, and skip without one. Push the catalog's Phi-3 row to the app's external files directory, or
 * name any package's directory with `-e onnxModelDir <path>`:
 * ```
 * adb push cpu-int4-rtn-block-32 \
 *   /sdcard/Android/data/<applicationId>/files/models/microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/
 * ```
 *
 * `-e onnxPrefillTokens` and `-e onnxDecodeTokens` set the cells' shape (64 and 16 by default, so the tests finish quickly on a slow phone), and each
 * result is logged under the tag `OnnxGenaiInstrumented`.
 */
@RunWith(AndroidJUnit4::class)
class OnnxGenaiBenchmarkInstrumentedTest {
  private val tag = "OnnxGenaiInstrumented"
  private val arguments = InstrumentationRegistry.getArguments()
  private val prefillTokens = arguments.getString("onnxPrefillTokens")?.toInt() ?: DEFAULT_PREFILL_TOKENS
  private val decodeTokens = arguments.getString("onnxDecodeTokens")?.toInt() ?: DEFAULT_DECODE_TOKENS

  private fun findPackage(context: Context): File? {
    val candidates =
      listOfNotNull(
        arguments.getString("onnxModelDir")?.let { File(it) },
        File(context.getExternalFilesDir(null), PHI3_PACKAGE),
        File(context.filesDir, PHI3_PACKAGE),
      )
    for (c in candidates) {
      Log.i(tag, "  ${c.absolutePath}: package=${OnnxGenai.isPackage(c.path)}")
    }
    return candidates.firstOrNull { OnnxGenai.isPackage(it.path) }
  }

  private fun requirePackage(): File {
    assumeTrue("this build has no ONNX Runtime GenAI engine", OnnxGenai.isBuilt)
    val dir = findPackage(InstrumentationRegistry.getInstrumentation().targetContext)
    assumeTrue("ONNX Runtime GenAI package not found", dir != null)
    return checkNotNull(dir)
  }

  private fun run(dir: File, benchmarkId: String, benchmarkType: String, decode: Int? = null): JSONObject {
    val benchmark =
      JSONObject().apply {
        put("benchmark_id", benchmarkId)
        put("benchmark_type", benchmarkType)
        put("parameter_prefill_tokens", prefillTokens)
        decode?.let { put("parameter_decode_tokens", it) }
      }
    val result = JSONObject(OnnxGenaiBenchmark.runFresh(benchmark.toString(), dir.path, null, null, null))
    Log.i(tag, "$benchmarkId: $result")
    assertEquals(benchmarkId, result.getString("benchmark_id"))
    return result
  }

  @Test
  fun testTheBuildNamesItsStack() {
    assumeTrue("this build has no ONNX Runtime GenAI engine", OnnxGenai.isBuilt)
    Log.i(tag, "onnxruntime ${OnnxGenai.onnxRuntimeVersion()}, onnxruntime-genai ${OnnxGenai.genaiCommit()}")
    // What the committed stamp records for each tree: a release tag where the pin is one, else a short commit. The app's BuildConfig reads the
    // same stamp, so the engine and the payload name one stack.
    assertTrue(OnnxGenai.onnxRuntimeVersion().matches(Regex("""v\d+\.\d+\.\d+|[0-9a-f]{9}""")))
    assertTrue(OnnxGenai.genaiCommit().matches(Regex("""v\d+\.\d+\.\d+|[0-9a-f]{9}""")))
    assertEquals(ai.liquid.pipette.BuildConfig.ONNXRUNTIME_VERSION, OnnxGenai.onnxRuntimeVersion())
    assertEquals(ai.liquid.pipette.BuildConfig.ONNXRUNTIME_GENAI_COMMIT, OnnxGenai.genaiCommit())
  }

  /** The prompt is exactly the cell's length under the package's own tokenizer, and a generation is exactly the cell's decode count. */
  @Test
  fun testAGenerationHasTheCellsShape() {
    val dir = requirePackage()
    OnnxGenai.loadModel(dir.path).use { model ->
      val ids = OnnxGenaiBenchmark.promptIds(model, prefillTokens)
      assertEquals(prefillTokens, ids.size)
      val timing = OnnxGenai.generate(model, ids, decodeTokens)
      Log.i(tag, "generation: $timing, end to end ${timing.endToEndMs} ms")
      assertEquals(prefillTokens, timing.promptTokens)
      assertEquals(decodeTokens, timing.generatedTokens)
      assertTrue(timing.ttftMs > 0.0)
      assertTrue(checkNotNull(timing.tpotMs) > 0.0)
    }
  }

  @Test
  fun testTheTimingCells() {
    val dir = requirePackage()
    val prefill = run(dir, "prefill_throughput_$prefillTokens", "prefill_throughput")
    assertTrue(prefill.getDouble("prefill_time_ms") > 0.0)

    val decode = run(dir, "decode_throughput_${prefillTokens}_$decodeTokens", "decode_throughput", decodeTokens)
    assertTrue(decode.getDouble("decode_time_ms") > 0.0)

    val endToEnd = run(dir, "end_to_end_latency_${prefillTokens}_$decodeTokens", "end_to_end_latency", decodeTokens)
    // Rebuilt from the same two intervals, so it is at least the time to the first token.
    assertTrue(endToEnd.getDouble("total_time_ms") >= prefill.getDouble("prefill_time_ms") * LOWER_BOUND)
  }

  /**
   * The peak is a large part of the package's weights, and can be under their size: ORT reads most of the external-data file into its own memory and
   * maps the rest, of which only the pages a generation touched are resident (Phi-3 int4: 2.40 GB against a 2.72 GB file on an emulator).
   */
  @Test
  fun testMaxMemoryUsage() {
    val dir = requirePackage()
    val result = run(dir, "max_memory_usage_$prefillTokens", "max_memory_usage")
    val weights = dir.listFiles().orEmpty().filter { it.name.endsWith(".data") }.sumOf { it.length() }
    Log.i(tag, "max_ram_bytes ${result.getLong("max_ram_bytes")}, external data $weights bytes")
    assertTrue(result.getLong("max_ram_bytes") > weights / 2)
  }

  @Test
  fun testEvalIsRefused() {
    val dir = requirePackage()
    val benchmark = """{"benchmark_id": "eval_smoke", "benchmark_type": "eval", "parameter_max_tokens": 16}"""
    val ex = assertThrows(IllegalStateException::class.java) { OnnxGenaiBenchmark.runFresh(benchmark, dir.path, null, null, null) }
    assertTrue(ex.message!!.contains("chat template"))
  }

  private companion object {
    const val PHI3_PACKAGE = "models/microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32"
    const val DEFAULT_PREFILL_TOKENS = 64
    const val DEFAULT_DECODE_TOKENS = 16
    // A separate run's prefill differs from this run's by noise; the bound only catches a total that lost the prefill altogether.
    const val LOWER_BOUND = 0.5
  }
}
