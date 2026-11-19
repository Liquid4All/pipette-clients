package ai.liquid.pipette.onnx

import ai.liquid.pipette.llama.LlamaBenchmark
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * The parts of the ONNX Runtime GenAI engine that run without its native library: the timing arithmetic, which is iOS's
 * (`OnnxGenaiRuntime.GenerationTiming`), the kinds the kernel refuses, and what counts as a package.
 */
class OnnxGenaiTest {
  @get:Rule val folder = TemporaryFolder()

  /**
   * The time to the first token is the first interval alone; the per-token interval is the second over the tokens after the first; and end to end is
   * rebuilt from the two, which is the iOS engine's definition and the host crate's.
   */
  @Test
  fun aGenerationsIntervalsAreReadAsIosReadsThem() {
    val timing = GenerationTiming.fromNative(promptTokens = PROMPT_TOKENS, values = longArrayOf(TTFT_NANOS, REST_NANOS, GENERATED.toLong()))
    assertEquals(PROMPT_TOKENS, timing.promptTokens)
    assertEquals(GENERATED, timing.generatedTokens)
    assertEquals(TTFT_MS, timing.ttftMs, TOLERANCE)
    assertEquals(TPOT_MS, timing.tpotMs!!, TOLERANCE)
    assertEquals(TTFT_MS + TPOT_MS * (GENERATED - 1), timing.endToEndMs, TOLERANCE)
  }

  /** A one-token generation has no interval after the first token, so its end to end is its time to the first token. */
  @Test
  fun aOneTokenGenerationHasNoPerTokenInterval() {
    val timing = GenerationTiming.fromNative(promptTokens = PROMPT_TOKENS, values = longArrayOf(TTFT_NANOS, 0L, 1L))
    assertNull(timing.tpotMs)
    assertEquals(TTFT_MS, timing.endToEndMs, TOLERANCE)
  }

  @Test
  fun aTimingOfTheWrongShapeIsRefused() {
    assertThrows(IllegalStateException::class.java) { GenerationTiming.fromNative(PROMPT_TOKENS, longArrayOf(1L, 2L)) }
  }

  /** The four cells iOS runs are the four this kernel runs; eval is refused for iOS's reason, and the kinds with no ONNX variant by name. */
  @Test
  fun theKernelRunsTheFourCellsIosRunsAndRefusesTheRest() {
    for (kind in listOf("prefill_throughput", "decode_throughput", "end_to_end_latency", "max_memory_usage")) {
      assertNull(kind, OnnxGenaiBenchmark.unsupportedReason(kind))
    }
    assertTrue(OnnxGenaiBenchmark.unsupportedReason("eval")!!.contains("chat template"))
    for (kind in listOf("vl_throughput", "vlm_eval", "asr_throughput")) {
      assertTrue(kind, OnnxGenaiBenchmark.unsupportedReason(kind)!!.contains(kind))
    }
  }

  /** A batched entry is refused by name, as on llama.cpp: the engine generates one sequence at a time. */
  @Test
  fun aBatchedEntryIsRefused() {
    val batched =
      LlamaBenchmark.BenchmarkDef(benchmarkId = "end_to_end_latency_512_256_bs4", benchmarkType = "end_to_end_latency", parameterBatchSize = 4)
    val ex = assertThrows(IllegalStateException::class.java) { OnnxGenaiBenchmark.checkBatchSize(batched) }
    assertTrue(ex.message!!.contains("end_to_end_latency_512_256_bs4"))
    assertTrue(ex.message!!.contains("one sequence at a time"))
    OnnxGenaiBenchmark.checkBatchSize(batched.copy(parameterBatchSize = 1))
    OnnxGenaiBenchmark.checkBatchSize(batched.copy(parameterBatchSize = null))
  }

  /** A package is the directory that holds a `genai_config.json`; a GGUF file is not one, and neither is a plain ONNX export's directory. */
  @Test
  fun aPackageIsADirectoryWithAGenaiConfig() {
    val packageDir = folder.newFolder("phi3")
    File(packageDir, OnnxGenai.CONFIG_FILE).writeText("{}")
    assertTrue(OnnxGenai.isPackage(packageDir.path))

    val plainExport = folder.newFolder("plain")
    File(plainExport, "model.onnx").writeText("")
    assertFalse(OnnxGenai.isPackage(plainExport.path))

    assertFalse(OnnxGenai.isPackage(folder.newFile("model.gguf").path))
  }

  private companion object {
    const val PROMPT_TOKENS = 512
    const val GENERATED = 100

    // Two seconds to the first token, then 99 tokens at 10 ms each.
    const val TTFT_NANOS = 2_000_000_000L
    const val REST_NANOS = 990_000_000L
    const val TTFT_MS = 2000.0
    const val TPOT_MS = 10.0
    const val TOLERANCE = 1e-9
  }
}
