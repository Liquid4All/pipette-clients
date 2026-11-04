package ai.liquid.pipette.llama

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `parameter_batch_size` on this engine: it decodes one sequence through one `llama_context`, so the only batch size it can run is the reference one.
 * Mirrors the Rust kernel's `check_batch_size` tests in `android/native/benchmarks.rs`.
 */
class LlamaBenchmarkBatchSizeTest {

  /**
   * A batched entry is refused by name, before the weights are read: this engine decodes one sequence, and running the entry serially would file the
   * reference measurement under the batched benchmark id. An entry that omits the field, as every one written before the parameter did, is the
   * reference procedure and runs.
   */
  @Test
  fun aBatchedEntryIsRefusedAndAnAbsentBatchSizeMeansOne() {
    val batched =
      LlamaBenchmark.parseBenchmarkDef(
        """
        {
          "benchmark_id": "end_to_end_latency_512_256_bs4",
          "benchmark_type": "end_to_end_latency",
          "parameter_prefill_tokens": 512,
          "parameter_decode_tokens": 256,
          "parameter_batch_size": 4
        }
        """
          .trimIndent()
      )
    assertEquals(BATCH_SIZE, batched.parameterBatchSize)
    val ex = assertThrows(IllegalStateException::class.java) { LlamaBenchmark.checkBatchSize(batched) }
    assertTrue(ex.message!!.contains("end_to_end_latency_512_256_bs4"))
    assertTrue(ex.message!!.contains("$BATCH_SIZE requests in flight"))
    assertTrue(ex.message!!.contains("one sequence at a time"))

    // Absent means the reference procedure, as the crate's `default_batch_size` does, and it
    // runs; so does an explicit 1, and so does a kind that carries no batch size at all.
    val reference =
      LlamaBenchmark.parseBenchmarkDef(
        """
        {
          "benchmark_id": "decode_throughput_512_256",
          "benchmark_type": "decode_throughput",
          "parameter_prefill_tokens": 512,
          "parameter_decode_tokens": 256
        }
        """
          .trimIndent()
      )
    assertNull(reference.parameterBatchSize)
    LlamaBenchmark.checkBatchSize(reference)
    LlamaBenchmark.checkBatchSize(batched.copy(parameterBatchSize = 1))
    LlamaBenchmark.checkBatchSize(
      LlamaBenchmark.BenchmarkDef(benchmarkId = "max_memory_usage_512", benchmarkType = "max_memory_usage", parameterPrefillTokens = 512)
    )

    // Zero is a catalog mistake rather than a synonym for the reference, and is named as one.
    val zero = assertThrows(IllegalStateException::class.java) { LlamaBenchmark.checkBatchSize(batched.copy(parameterBatchSize = 0)) }
    assertTrue(zero.message!!.contains("at least 1"))
  }

  private companion object {
    /** The batched entry's concurrency: the one the id promises and the engine cannot run. */
    const val BATCH_SIZE = 4
  }
}
