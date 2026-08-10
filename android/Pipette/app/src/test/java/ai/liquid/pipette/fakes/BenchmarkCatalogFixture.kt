package ai.liquid.pipette.fakes

import ai.liquid.pipette.BenchmarkCatalog
import ai.liquid.pipette.BenchmarkDefinition
import ai.liquid.pipette.BenchmarkId
import org.json.JSONArray

/**
 * Installs a full server-shaped benchmark catalog into the [BenchmarkCatalog] singleton for tests that resolve benchmarks by id (the post-sync
 * state). Production loads this from a synced [ai.liquid.pipette.BenchmarkStore]; tests that don't drive a sync use [install] / [reset] so
 * `BenchmarkCatalog.byId(...)` returns non-null for the four ladder types plus the VL sizes.
 */
object BenchmarkCatalogFixture {
  private val ladder = listOf(100, 256, 512, 1024, 2048, 4096, 8192)
  private val vlSizes = listOf(256 to 256, 256 to 512, 384 to 512, 512 to 512, 1056 to 1056, 1056 to 2080)

  /** The catalog a full sync would produce — the four ladder kinds over the token ladder plus the fixed VL set. */
  fun fullCatalog(): List<BenchmarkDefinition> {
    val list = mutableListOf<BenchmarkDefinition>()
    for (t in ladder) {
      list += BenchmarkDefinition.PrefillThroughput(BenchmarkId.parse("prefill_throughput_$t"), t)
      list += BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse("decode_throughput_${t}_100"), t, 100)
      list += BenchmarkDefinition.EndToEndLatency(BenchmarkId.parse("end_to_end_latency_${t}_256"), t, 256)
      list += BenchmarkDefinition.MaxMemoryUsage(BenchmarkId.parse("max_memory_usage_$t"), t)
    }
    for ((w, h) in vlSizes) {
      list += BenchmarkDefinition.VlThroughput(BenchmarkId.parse("vl_throughput_${w}x${h}_32_128"), w, h, 32, 128)
    }
    return list
  }

  fun install() {
    val store = InMemoryBenchmarkStore()
    store.writeIndex(JSONArray(fullCatalog().map { it.toJson() }).toString())
    BenchmarkCatalog.load(store)
  }

  /** Reset the singleton to empty so a later test isn't polluted by the installed catalog. */
  fun reset() {
    BenchmarkCatalog.load(InMemoryBenchmarkStore())
  }
}
