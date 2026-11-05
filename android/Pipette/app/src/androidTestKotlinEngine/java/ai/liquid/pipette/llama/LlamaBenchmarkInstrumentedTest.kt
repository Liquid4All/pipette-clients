package ai.liquid.pipette.llama

import ai.liquid.pipette.LlamacppLoadMode
import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Instrumented tests executing benchmarks through LlamaBenchmark on device under kotlinEngine. */
@RunWith(AndroidJUnit4::class)
class LlamaBenchmarkInstrumentedTest {
  private val tag = "LlamaBenchmarkInstrumented"

  companion object {
    /**
     * The same 224x224 PNG data-URI fixture `pipette-cli`'s `vlm_eval_smoke` benchmark seeds server-side
     * (`crates/pipette-cli/src/benchmarks/standard.rs`), so a failure here and a failure in that smoke cell should always implicate the same runner
     * behavior, never a fixture mismatch between the two.
     */
    private const val SMOKE_PNG_DATA_URI =
      "data:image/png;base64," +
        "iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAAAAAA/RjU9AAAA3UlEQVR42u3PAQkAAAwDoEVf9MU4HG1g" +
        "+lwEBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUF" +
        "BQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUF" +
        "BQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUF" +
        "BQUFBQUFBQUFBQUFBQUFBe8MJSoFv1MoF1AAAAAASUVORK5CYII="
  }

  private fun findTestModel(context: Context): File? {
    val candidates =
      listOf(
        File(context.filesDir, "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File(context.filesDir, "LFM2.5-350M-Q4_0.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q5_K_M.gguf"),
        File("/sdcard/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File("/storage/emulated/0/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
      )
    Log.i(tag, "Checking candidates for package ${context.packageName}:")
    for (c in candidates) {
      Log.i(tag, "  ${c.absolutePath}: exists=${c.exists()}, canRead=${c.canRead()}")
    }
    return candidates.firstOrNull { it.exists() && it.canRead() }
  }

  /** Same family as [findTestModel]'s LFM2.5-350M, the VL variant referenced by examples/plans/llamacpp-macos-vlm-eval.toml. */
  private fun findTestVisionModel(context: Context): File? {
    val candidates =
      listOf(
        File(context.filesDir, "models/LiquidAI/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q4_K_M.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q4_K_M.gguf"),
        File("/sdcard/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q4_K_M.gguf"),
        File("/storage/emulated/0/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q4_K_M.gguf"),
      )
    return candidates.firstOrNull { it.exists() && it.canRead() }
  }

  private fun findTestMmproj(context: Context): File? {
    val candidates =
      listOf(
        File(context.filesDir, "models/LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"),
        File("/sdcard/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"),
        File("/storage/emulated/0/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"),
      )
    return candidates.firstOrNull { it.exists() && it.canRead() }
  }

  @Test
  fun testModelPaths() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val model = findTestModel(context)
    Log.i(tag, "Selected model: ${model?.absolutePath}")
  }

  @Test
  fun testPrefillAndDecodeThroughputOnModel() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val prefillJson =
      """
      {
        "benchmark_id": "prefill_throughput_64",
        "benchmark_type": "prefill_throughput",
        "parameter_prefill_tokens": 64
      }
      """
        .trimIndent()

    val decodeJson =
      """
      {
        "benchmark_id": "decode_throughput_64_16",
        "benchmark_type": "decode_throughput",
        "parameter_prefill_tokens": 64,
        "parameter_decode_tokens": 16
      }
      """
        .trimIndent()

    val model = LlamaCpp.loadModel(path = modelFile.absolutePath, nGpuLayers = 0, contextSize = 512, nUbatch = 128, loadMode = LlamacppLoadMode.READ)

    try {
      // 1. Run prefill throughput on model
      val prefillResult =
        LlamaBenchmark.runOnModel(
          benchmarkJson = prefillJson,
          model = model,
          nGpuLayers = 0,
          contextSize = 512,
          mmprojPath = null,
          progress = null,
          cooldown = null,
          thermal = null,
        )
      val prefillObj = JSONObject(prefillResult)
      assertEquals("prefill_throughput_64", prefillObj.getString("benchmark_id"))
      assertTrue("prefill_time_ms must be positive", prefillObj.getDouble("prefill_time_ms") > 0.0)
      assertTrue("runtime_thread_count must be positive", prefillObj.getInt("runtime_thread_count") > 0)
      Log.i(tag, "Prefill result: $prefillResult")

      // 2. Run decode throughput on the same model
      val decodeResult =
        LlamaBenchmark.runOnModel(
          benchmarkJson = decodeJson,
          model = model,
          nGpuLayers = 0,
          contextSize = 512,
          mmprojPath = null,
          progress = null,
          cooldown = null,
          thermal = null,
        )
      val decodeObj = JSONObject(decodeResult)
      assertEquals("decode_throughput_64_16", decodeObj.getString("benchmark_id"))
      assertTrue("decode_time_ms must be positive", decodeObj.getDouble("decode_time_ms") > 0.0)
      assertTrue("runtime_thread_count must be positive", decodeObj.getInt("runtime_thread_count") > 0)
      Log.i(tag, "Decode result: $decodeResult")
    } finally {
      model.close()
    }
  }

  @Test
  fun testEvalExecutionEndToEnd() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val evalJson =
      """
      {
        "benchmark_id": "test_eval",
        "benchmark_type": "eval",
        "parameter_max_tokens": 16,
        "samples": [
          {
            "id": "sample_1",
            "messages": [
              {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
            ]
          },
          {
            "id": "sample_2",
            "messages": [
              {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ]
          }
        ]
      }
      """
        .trimIndent()

    val resultJson =
      LlamaBenchmark.runFresh(
        benchmarkJson = evalJson,
        modelPath = modelFile.absolutePath,
        nGpuLayers = 0,
        contextSize = LlamaBenchmark.EVAL_PROMPT_BUDGET + 16,
        nUbatch = 128,
        loadMode = LlamacppLoadMode.READ,
        mmprojPath = null,
        progress = null,
        cooldown = null,
        thermal = null,
      )

    val obj = JSONObject(resultJson)
    assertEquals("test_eval", obj.getString("benchmark_id"))
    val completions = obj.getJSONArray("completions")
    assertEquals(2, completions.length())

    val c1 = completions.getJSONObject(0)
    assertEquals("sample_1", c1.getString("id"))
    assertFalse(c1.getBoolean("failed"))
    assertTrue(c1.getString("completion").isNotEmpty())

    val c2 = completions.getJSONObject(1)
    assertEquals("sample_2", c2.getString("id"))
    assertFalse(c2.getBoolean("failed"))
    assertTrue(c2.getString("completion").isNotEmpty())
    Log.i(tag, "Eval result: $resultJson")
  }

  @Test
  fun testEvalMcqExecution() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val mcqJson =
      """
      {
        "benchmark_id": "test_mcq",
        "benchmark_type": "eval",
        "parameter_max_tokens": 1,
        "parameter_mcq_choices": ["A", "B", "C", "D"],
        "samples": [
          {
            "id": "mcq_1",
            "messages": [
              {"role": "user", "content": "Which is greater? A) 10 B) 20 C) 5 D) 0. Answer with the letter:"}
            ]
          }
        ]
      }
      """
        .trimIndent()

    val resultJson =
      LlamaBenchmark.runFresh(
        benchmarkJson = mcqJson,
        modelPath = modelFile.absolutePath,
        nGpuLayers = 0,
        contextSize = LlamaBenchmark.EVAL_PROMPT_BUDGET + 1,
        nUbatch = 128,
        loadMode = LlamacppLoadMode.READ,
        mmprojPath = null,
        progress = null,
        cooldown = null,
        thermal = null,
      )

    val obj = JSONObject(resultJson)
    assertEquals("test_mcq", obj.getString("benchmark_id"))
    val completions = obj.getJSONArray("completions")
    assertEquals(1, completions.length())
    val c = completions.getJSONObject(0)
    assertEquals("mcq_1", c.getString("id"))
    assertFalse(c.getBoolean("failed"))
    assertTrue(listOf("A", "B", "C", "D").contains(c.getString("completion")))
    Log.i(tag, "MCQ result: $resultJson")
  }

  /** The vlm_eval mirror of [testEvalExecutionEndToEnd]: real image bytes through mtmd instead of a text-only prefill. */
  @Test
  fun testVlmEvalExecutionEndToEnd() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestVisionModel(context)
    val mmprojFile = findTestMmproj(context)
    assumeTrue("Test VL GGUF model not found", modelFile != null)
    assumeTrue("Test mmproj not found", mmprojFile != null)
    checkNotNull(modelFile)
    checkNotNull(mmprojFile)

    val vlmEvalJson =
      """
      {
        "benchmark_id": "test_vlm_eval",
        "benchmark_type": "vlm_eval",
        "parameter_max_tokens": 16,
        "samples": [
          {
            "id": "sample_1",
            "messages": [
              {
                "role": "user",
                "content": [
                  {"type": "text", "text": "What color is this image? Answer in one word."},
                  {"type": "image_url", "image_url": {"url": "$SMOKE_PNG_DATA_URI"}}
                ]
              }
            ]
          }
        ]
      }
      """
        .trimIndent()

    val resultJson =
      LlamaBenchmark.runFresh(
        benchmarkJson = vlmEvalJson,
        modelPath = modelFile.absolutePath,
        nGpuLayers = 0,
        contextSize = LlamaBenchmark.EVAL_PROMPT_BUDGET + 16,
        nUbatch = 128,
        loadMode = LlamacppLoadMode.NONE,
        mmprojPath = mmprojFile.absolutePath,
        progress = null,
        cooldown = null,
        thermal = null,
      )

    val obj = JSONObject(resultJson)
    assertEquals("test_vlm_eval", obj.getString("benchmark_id"))
    val completions = obj.getJSONArray("completions")
    assertEquals(1, completions.length())

    val c1 = completions.getJSONObject(0)
    assertEquals("sample_1", c1.getString("id"))
    assertFalse("failed_reason: ${c1.optString("failed_reason")}", c1.getBoolean("failed"))
    assertTrue(c1.getString("completion").isNotEmpty())
    Log.i(tag, "vlm_eval result: $resultJson")
  }

  /** The vlm_eval mirror of [testEvalMcqExecution]: the MCQ arm (single greedy token) driven from an mtmd-populated KV cache. */
  @Test
  fun testVlmEvalMcqExecution() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestVisionModel(context)
    val mmprojFile = findTestMmproj(context)
    assumeTrue("Test VL GGUF model not found", modelFile != null)
    assumeTrue("Test mmproj not found", mmprojFile != null)
    checkNotNull(modelFile)
    checkNotNull(mmprojFile)

    val mcqJson =
      """
      {
        "benchmark_id": "test_vlm_mcq",
        "benchmark_type": "vlm_eval",
        "parameter_max_tokens": 1,
        "parameter_mcq_choices": ["A", "B", "C", "D"],
        "samples": [
          {
            "id": "mcq_1",
            "messages": [
              {
                "role": "user",
                "content": [
                  {"type": "text", "text": "Is this image mostly A) red B) green C) blue D) white? Answer with the letter:"},
                  {"type": "image_url", "image_url": {"url": "$SMOKE_PNG_DATA_URI"}}
                ]
              }
            ]
          }
        ]
      }
      """
        .trimIndent()

    val resultJson =
      LlamaBenchmark.runFresh(
        benchmarkJson = mcqJson,
        modelPath = modelFile.absolutePath,
        nGpuLayers = 0,
        contextSize = LlamaBenchmark.EVAL_PROMPT_BUDGET + 1,
        nUbatch = 128,
        loadMode = LlamacppLoadMode.NONE,
        mmprojPath = mmprojFile.absolutePath,
        progress = null,
        cooldown = null,
        thermal = null,
      )

    val obj = JSONObject(resultJson)
    assertEquals("test_vlm_mcq", obj.getString("benchmark_id"))
    val completions = obj.getJSONArray("completions")
    assertEquals(1, completions.length())
    val c = completions.getJSONObject(0)
    assertEquals("mcq_1", c.getString("id"))
    assertFalse("failed_reason: ${c.optString("failed_reason")}", c.getBoolean("failed"))
    assertTrue(listOf("A", "B", "C", "D").contains(c.getString("completion")))
    Log.i(tag, "vlm_eval MCQ result: $resultJson")
  }
}
