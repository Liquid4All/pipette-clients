package ai.liquid.pipette

import ai.liquid.pipette.fakes.InMemoryBenchmarkStore
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Android catalog is now driven by the server-synced set (see [BenchmarkSync]); [BenchmarkCatalog.selectable] narrows it exactly like iOS
 * `BenchmarkCatalog.selectable(from:)` — the four ladder types (prefill/decode/e2e/max-memory) whose required context stays under 5000 tokens, with
 * `vl_throughput` and `eval` excluded and the 8192 rung dropped. This guards that parity and the PIP-335 fix (the small variants the old hardcoded
 * ladder retired, plus the server-only `decode_throughput_0_32`, are now offered).
 */
class BenchmarkCatalogTest {
  private val ladder = listOf(100, 256, 512, 1024, 2048, 4096, 8192)

  /** A full server-shaped catalog: every kind across the full ladder, plus a pure-decode variant, a VL size, and an eval. */
  private fun serverCatalog(): List<BenchmarkDefinition> {
    val list = mutableListOf<BenchmarkDefinition>()
    for (t in ladder) {
      list += BenchmarkDefinition.PrefillThroughput(BenchmarkId.parse("prefill_throughput_$t"), t)
      list += BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse("decode_throughput_${t}_100"), t, 100)
      list += BenchmarkDefinition.EndToEndLatency(BenchmarkId.parse("end_to_end_latency_${t}_256"), t, 256)
      list += BenchmarkDefinition.MaxMemoryUsage(BenchmarkId.parse("max_memory_usage_$t"), t)
    }
    // A pure-decode variant the server lists but the old hardcoded ladder never produced — one of the four PIP-335 reported missing on Android.
    list += BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse("decode_throughput_0_32"), 0, 32)
    list += BenchmarkDefinition.VlThroughput(BenchmarkId.parse("vl_throughput_256x256_32_128"), 256, 256, 32, 128)
    list += BenchmarkDefinition.Eval(BenchmarkId.parse("eval_smoke"), "eval_smoke", "local", 4, null, null)
    return list
  }

  @Test
  fun selectableMatchesIosContextCap() {
    val actual = BenchmarkCatalog.selectable(serverCatalog()).map { it.benchmarkId.toString() }.toSet()
    val expected =
      setOf(
        // prefill_throughput: prefill < 5000 → 100..4096 (8192 dropped)
        "prefill_throughput_100",
        "prefill_throughput_256",
        "prefill_throughput_512",
        "prefill_throughput_1024",
        "prefill_throughput_2048",
        "prefill_throughput_4096",
        // decode_throughput: prefill + decode < 5000 → 100..4096, plus the pure-decode 0_32
        "decode_throughput_0_32",
        "decode_throughput_100_100",
        "decode_throughput_256_100",
        "decode_throughput_512_100",
        "decode_throughput_1024_100",
        "decode_throughput_2048_100",
        "decode_throughput_4096_100",
        // end_to_end_latency: prefill + 256 < 5000 → 100..4096
        "end_to_end_latency_100_256",
        "end_to_end_latency_256_256",
        "end_to_end_latency_512_256",
        "end_to_end_latency_1024_256",
        "end_to_end_latency_2048_256",
        "end_to_end_latency_4096_256",
        // max_memory_usage: prefill + 1 < 5000 → 100..4096
        "max_memory_usage_100",
        "max_memory_usage_256",
        "max_memory_usage_512",
        "max_memory_usage_1024",
        "max_memory_usage_2048",
        "max_memory_usage_4096",
      )
    assertEquals(expected, actual)
  }

  @Test
  fun theFourReportedMissingVariantsAreNowSelectable() {
    val ids = BenchmarkCatalog.selectable(serverCatalog()).map { it.benchmarkId.toString() }.toSet()
    // PIP-335: the four the user reported missing on Android.
    listOf("decode_throughput_0_32", "decode_throughput_100_100", "max_memory_usage_100", "prefill_throughput_100").forEach { id ->
      assertTrue("$id must be selectable", ids.contains(id))
    }
  }

  @Test
  fun selectableExcludesHeavyRungAndUnsupportedTypes() {
    val ids = BenchmarkCatalog.selectable(serverCatalog()).map { it.benchmarkId.toString() }
    assertTrue("8192 rung is too heavy for phones", ids.none { it.substringAfterLast('_') == "8192" || it.contains("_8192_") })
    assertTrue("vl_throughput is not offered in the picker", ids.none { it.startsWith("vl_throughput") })
    assertTrue("eval is not offered in the picker", ids.none { it.startsWith("eval") })
  }

  @Test
  fun selectableIsStablySortedById() {
    val ids = BenchmarkCatalog.selectable(serverCatalog()).map { it.benchmarkId.toString() }
    assertEquals(ids.sorted(), ids)
  }

  @Test
  fun emptyCatalogYieldsEmptySelectable() {
    assertTrue(BenchmarkCatalog.selectable(emptyList()).isEmpty())
  }

  @Test
  fun loadPopulatesAllAndResolvesHiddenIdsById() {
    val store = InMemoryBenchmarkStore()
    store.writeIndex(JSONArray(serverCatalog().map { it.toJson() }).toString())
    try {
      BenchmarkCatalog.load(store)
      // `all` keeps everything (VL, eval, 8192) so historical results resolve, even though they aren't selectable.
      assertNotNull("server-only decode variant resolves", BenchmarkCatalog.byId("decode_throughput_0_32"))
      assertNotNull("hidden 8192 rung still resolves via byId", BenchmarkCatalog.byId("prefill_throughput_8192"))
      assertNotNull("VL resolves via byId", BenchmarkCatalog.byId("vl_throughput_256x256_32_128"))
      assertNull(BenchmarkCatalog.byId("not_a_benchmark"))
      val decode = BenchmarkCatalog.byId("decode_throughput_512_100")
      assertTrue(decode is BenchmarkDefinition.DecodeThroughput)
      assertEquals(512, (decode as BenchmarkDefinition.DecodeThroughput).prefillTokens)
      assertEquals(100, decode.decodeTokens)
    } finally {
      // Reset the shared singleton so other tests see an empty catalog.
      BenchmarkCatalog.load(InMemoryBenchmarkStore())
    }
  }

  @Test
  fun typeDisplayNamesAndRanksMatchCanonicalOrder() {
    // Mirrors iOS BenchmarkCatalog.displayName/typeRank exactly.
    val expected =
      listOf(
        Triple("end_to_end_latency", "End-to-End Latency", 0),
        Triple("prefill_throughput", "Prefill Throughput", 1),
        Triple("decode_throughput", "Decode Throughput", 2),
        Triple("max_memory_usage", "Max Memory Usage", 3),
        Triple("vl_throughput", "Vision-Language Throughput", 4),
        Triple("eval", "Eval Accuracy", 5),
      )
    expected.forEach { (wire, name, rank) ->
      assertEquals(name, BenchmarkCatalog.displayName(wire))
      assertEquals(rank, BenchmarkCatalog.typeRank(wire))
    }
  }

  @Test
  fun matchesSearchByIdTypeAndLabel() {
    val decode = BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse("decode_throughput_512_100"), 512, 100)
    assertTrue(BenchmarkCatalog.matchesSearch(decode, ""))
    assertTrue(BenchmarkCatalog.matchesSearch(decode, "decode"))
    assertTrue(BenchmarkCatalog.matchesSearch(decode, "512"))
    assertFalse(BenchmarkCatalog.matchesSearch(decode, "prefill"))
  }

  @Test
  fun onlyMaxMemoryRequiresAFreshLoad() {
    // Asserted over the whole enum rather than the one true case, so a new benchmark type has to answer this question deliberately instead of
    // defaulting to "reuse the resident model". Mirrors iOS `BenchmarkType.requiresFreshLoad`.
    assertEquals(setOf(BenchmarkType.MAX_MEMORY_USAGE), BenchmarkType.entries.filter { it.requiresFreshLoad }.toSet())
  }
}
