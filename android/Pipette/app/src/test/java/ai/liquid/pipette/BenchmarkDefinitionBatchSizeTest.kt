package ai.liquid.pipette

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `parameter_batch_size` survives the app-side definition. The catalog entry is parsed here and the engine is fed [BenchmarkDefinition.rawJson], so a
 * field dropped on the way through would run a batched entry serially and file it under the batched id, past the engine's own refusal.
 */
class BenchmarkDefinitionBatchSizeTest {

  /** The batched form reaches the engine JSON with the count the id promises. */
  @Test
  fun aBatchedCatalogEntryReachesTheEngineJsonWithItsBatchSize() {
    val def =
      parse(
        """{"benchmark_id":"end_to_end_latency_512_256_bs4","benchmark_type":"end_to_end_latency",
           "parameter_prefill_tokens":512,"parameter_decode_tokens":256,"parameter_batch_size":4}"""
      )
    assertEquals(BenchmarkDefinition.EndToEndLatency(BenchmarkId.parse("end_to_end_latency_512_256_bs4"), 512, 256, batchSize = 4), def)
    assertEquals(4, def.rawJson.getInt("parameter_batch_size"))
    assertTrue(def.label.contains("x4 in flight"))

    val prefill =
      parse(
        """{"benchmark_id":"prefill_throughput_512_bs2","benchmark_type":"prefill_throughput",
           "parameter_prefill_tokens":512,"parameter_batch_size":2}"""
      )
    assertEquals(2, prefill.rawJson.getInt("parameter_batch_size"))
    val decode =
      parse(
        """{"benchmark_id":"decode_throughput_512_100_bs8","benchmark_type":"decode_throughput",
           "parameter_prefill_tokens":512,"parameter_decode_tokens":100,"parameter_batch_size":8}"""
      )
    assertEquals(8, decode.rawJson.getInt("parameter_batch_size"))
  }

  /**
   * An entry that omits the field, or carries it as null, means the reference procedure and is written back as an explicit 1, as the crate and iOS
   * do. A kind that carries no batch writes no key at all.
   */
  @Test
  fun anAbsentBatchSizeMeansOneAndIsWrittenBack() {
    val absent =
      parse(
        """{"benchmark_id":"decode_throughput_512_100","benchmark_type":"decode_throughput",
           "parameter_prefill_tokens":512,"parameter_decode_tokens":100}"""
      )
    assertEquals(1, (absent as BenchmarkDefinition.DecodeThroughput).batchSize)
    assertEquals(1, absent.rawJson.getInt("parameter_batch_size"))
    assertFalse(absent.label.contains("in flight"))

    val explicitNull =
      parse(
        """{"benchmark_id":"prefill_throughput_512","benchmark_type":"prefill_throughput",
           "parameter_prefill_tokens":512,"parameter_batch_size":null}"""
      )
    assertEquals(1, (explicitNull as BenchmarkDefinition.PrefillThroughput).batchSize)

    val peak =
      parse(
        """{"benchmark_id":"max_memory_usage_512","benchmark_type":"max_memory_usage",
           "parameter_prefill_tokens":512}"""
      )
    assertFalse(peak.rawJson.has("parameter_batch_size"))
  }

  private fun parse(json: String): BenchmarkDefinition =
    checkNotNull(BenchmarkDefinition.fromJson(JSONObject(json))) { "expected the entry to parse: $json" }
}
