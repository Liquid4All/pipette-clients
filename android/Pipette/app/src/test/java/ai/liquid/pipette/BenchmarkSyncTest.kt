package ai.liquid.pipette

import ai.liquid.pipette.fakes.InMemoryBenchmarkStore
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Drives [BenchmarkSync] entirely off-device against an [InMemoryBenchmarkStore] and fake fetchers, mirroring iOS `BenchmarkSyncTests`: the two-level
 * ETag-conditional pull, tolerant parsing, per-id 304/failure handling, and detail pruning.
 */
class BenchmarkSyncTest {
  private val noLog: (String) -> Unit = {}

  private fun def(id: String, prefill: Int, decode: Int) = BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse(id), prefill, decode)

  private fun listBody(vararg entries: JSONObject): String = JSONArray().apply { entries.forEach { put(it) } }.toString()

  private fun ok(body: String, etag: String?) = ManagementClient.ConditionalGet(body, etag)

  private fun notModified(etag: String?) = ManagementClient.ConditionalGet(null, etag)

  /** A detail fetcher that echoes each id back as its own valid definition (prefill/decode derived from the id ordinal is irrelevant). */
  private fun echoDetails(etag: String? = "d1") = BenchmarkSync.DetailFetcher { _, id, _ -> ok(def(id, 1, 1).toJson().toString(), etag) }

  @Test
  fun listModifiedKeepsOnlyParseableAndPersistsIndex() = runBlocking {
    val store = InMemoryBenchmarkStore()
    val unknown = JSONObject().put("benchmark_id", "weird_1").put("benchmark_type", "teleport_latency").put("parameter_prefill_tokens", 1)
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson(), unknown, def("decode_throughput_0_32", 0, 32).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "list-v1") }

    val count = BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), noLog)

    assertEquals(2, count)
    val storedIds = BenchmarkSync.storedDefinitions(store).map { it.benchmarkId.toString() }.toSet()
    assertEquals(setOf("decode_throughput_100_100", "decode_throughput_0_32"), storedIds)
    // Sync state records the list ETag and the per-id detail ETags for the kept ids only.
    val state = BenchmarkSync.SyncState.fromJson(JSONObject(store.readSyncState()!!))
    assertEquals("list-v1", state.benchmarksEtag)
    assertEquals(2, state.benchmarkCount)
    assertEquals(setOf("decode_throughput_100_100", "decode_throughput_0_32"), state.benchmarkEtags.keys)
    assertNotNull(store.readDetail("decode_throughput_0_32"))
  }

  @Test
  fun knownTypeMissingARequiredParameterIsSkippedAndLoggedAsAnError() = runBlocking {
    // The gap PIP-248 closes on Android. These parameters used to be read with `optInt`, so a
    // decode_throughput with no parameter_decode_tokens decoded to zero tokens and was stored and
    // *run*, producing a measurement of nothing that no consumer could tell from a real one. iOS and
    // the Rust client have always rejected it.
    val store = InMemoryBenchmarkStore()
    val infoLines = mutableListOf<String>()
    val errorLines = mutableListOf<String>()
    val incomplete =
      JSONObject().put("benchmark_id", "decode_throughput_128_64").put("benchmark_type", "decode_throughput").put("parameter_prefill_tokens", 128)
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson(), incomplete)
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "list-v1") }

    val count = BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), { infoLines += it }, { errorLines += it })

    assertEquals(1, count)
    assertEquals(setOf("decode_throughput_100_100"), BenchmarkSync.storedDefinitions(store).map { it.benchmarkId.toString() }.toSet())
    // A known type we can't read means server and client disagree about a shape both implement, so
    // it rides the error sink rather than being lost among routine sync chatter.
    assertTrue(
      "expected a schema-mismatch error, got $errorLines",
      errorLines.any { it.contains("decode_throughput_128_64") && it.contains("schema mismatch") },
    )
  }

  @Test
  fun unrecognizedTypeIsSkippedQuietlyRatherThanAsAnError() = runBlocking {
    // The other arm, and why the two are split: a newer server catalog reaching an older client is
    // expected. Reporting it as an error would train readers to ignore the error sink, which is
    // where the schema mismatch above has to be visible.
    val store = InMemoryBenchmarkStore()
    val infoLines = mutableListOf<String>()
    val errorLines = mutableListOf<String>()
    val unknown = JSONObject().put("benchmark_id", "teleport_1").put("benchmark_type", "teleport_latency").put("parameter_prefill_tokens", 1)
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(listBody(def("decode_throughput_100_100", 100, 100).toJson(), unknown), "list-v1") }

    val count = BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), { infoLines += it }, { errorLines += it })

    assertEquals(1, count)
    assertTrue("unknown types must not reach the error sink, got $errorLines", errorLines.isEmpty())
    assertTrue(
      "expected a quiet skip line, got $infoLines",
      infoLines.any { it.contains("teleport_1") && it.contains("unrecognized benchmark_type") },
    )
  }

  @Test
  fun listNotModifiedReusesCachedIndex() = runBlocking {
    val store = InMemoryBenchmarkStore()
    store.writeIndex(listBody(def("decode_throughput_256_100", 256, 100).toJson()))
    val fetchList =
      BenchmarkSync.ListFetcher { _, ifNoneMatch ->
        assertEquals("prev-etag", ifNoneMatch)
        notModified("prev-etag")
      }
    store.writeSyncState(BenchmarkSync.SyncState(1, "prev-etag", emptyMap()).toJson().toString())

    val count = BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), noLog)

    assertEquals(1, count)
    assertNotNull(store.readDetail("decode_throughput_256_100"))
  }

  @Test
  fun listNotModifiedWithNoCacheReturnsZero() = runBlocking {
    val store = InMemoryBenchmarkStore()
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> notModified(null) }
    val count = BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), noLog)
    assertEquals(0, count)
  }

  @Test
  fun staleDetailFilesArePrunedWhenBenchmarkLeavesCatalog() = runBlocking {
    val store = InMemoryBenchmarkStore()
    // Pre-seed a detail for a benchmark that the new list no longer contains.
    store.writeDetail("decode_throughput_999_100", "{}")
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "v1") }

    BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), noLog)

    assertNull("removed benchmark's detail is pruned", store.readDetail("decode_throughput_999_100"))
    assertNotNull(store.readDetail("decode_throughput_100_100"))
  }

  @Test
  fun detailNotModifiedKeepsCachedDetailAndEtag() = runBlocking {
    val store = InMemoryBenchmarkStore()
    store.writeDetail("decode_throughput_100_100", def("decode_throughput_100_100", 100, 100).toJson().toString())
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "v1") }
    val fetchDetail =
      BenchmarkSync.DetailFetcher { _, id, ifNoneMatch ->
        assertEquals("cached-detail-etag", ifNoneMatch)
        notModified("cached-detail-etag")
      }
    store.writeSyncState(BenchmarkSync.SyncState(1, "v0", mapOf("decode_throughput_100_100" to "cached-detail-etag")).toJson().toString())

    BenchmarkSync.sync("https://s", store, fetchList, fetchDetail, noLog)

    assertNotNull(store.readDetail("decode_throughput_100_100"))
    val state = BenchmarkSync.SyncState.fromJson(JSONObject(store.readSyncState()!!))
    assertEquals("cached-detail-etag", state.benchmarkEtags["decode_throughput_100_100"])
  }

  @Test
  fun unparseableDetailIsDroppedWithoutEtag() = runBlocking {
    val store = InMemoryBenchmarkStore()
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "v1") }
    val fetchDetail = BenchmarkSync.DetailFetcher { _, _, _ -> ok("{ not a benchmark }", "bad-etag") }

    val count = BenchmarkSync.sync("https://s", store, fetchList, fetchDetail, noLog)

    // The benchmark still counts (it's in the index), but its detail isn't stored and its ETag is dropped so it re-fetches next time.
    assertEquals(1, count)
    assertNull(store.readDetail("decode_throughput_100_100"))
    val state = BenchmarkSync.SyncState.fromJson(JSONObject(store.readSyncState()!!))
    assertTrue(state.benchmarkEtags.isEmpty())
  }

  @Test
  fun perIdFailureDoesNotAbortTheSync() = runBlocking {
    val store = InMemoryBenchmarkStore()
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson(), def("decode_throughput_256_100", 256, 100).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "v1") }
    val fetchDetail =
      BenchmarkSync.DetailFetcher { _, id, _ ->
        if (id == "decode_throughput_100_100") error("boom") else ok(def(id, 256, 100).toJson().toString(), "d1")
      }

    val count = BenchmarkSync.sync("https://s", store, fetchList, fetchDetail, noLog)

    assertEquals(2, count)
    assertNull("failed id has no detail", store.readDetail("decode_throughput_100_100"))
    assertNotNull("the other id still synced", store.readDetail("decode_throughput_256_100"))
  }

  @Test
  fun malformedListFailsWithoutClobberingTheCachedCatalog() = runBlocking {
    val store = InMemoryBenchmarkStore()
    // A good cached catalog + detail from a prior sync.
    store.writeIndex(listBody(def("decode_throughput_256_100", 256, 100).toJson()))
    store.writeDetail("decode_throughput_256_100", def("decode_throughput_256_100", 256, 100).toJson().toString())
    // Server returns a 200 whose body isn't a JSON array (truncated/garbage).
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok("{ not an array", "v1") }

    val failed = runCatching { BenchmarkSync.sync("https://s", store, fetchList, echoDetails(), noLog) }.isFailure

    assertTrue("malformed list must fail the sync, not return empty", failed)
    // The last good catalog and its detail survive — no empty index written, no prune.
    assertEquals(setOf("decode_throughput_256_100"), BenchmarkSync.storedDefinitions(store).map { it.benchmarkId.toString() }.toSet())
    assertNotNull(store.readDetail("decode_throughput_256_100"))
  }

  @Test
  fun detail304WithoutCachedBodyDropsEtagSoNextSyncRefetches() = runBlocking {
    val store = InMemoryBenchmarkStore()
    val body = listBody(def("decode_throughput_100_100", 100, 100).toJson())
    val fetchList = BenchmarkSync.ListFetcher { _, _ -> ok(body, "v1") }
    // Prior ETag recorded, but the cached detail file is gone (pruned/lost) — a 304 with no body would otherwise strand it forever.
    store.writeSyncState(BenchmarkSync.SyncState(1, "v0", mapOf("decode_throughput_100_100" to "stale-etag")).toJson().toString())
    val fetchDetail = BenchmarkSync.DetailFetcher { _, _, _ -> notModified("stale-etag") }

    BenchmarkSync.sync("https://s", store, fetchList, fetchDetail, noLog)

    val state = BenchmarkSync.SyncState.fromJson(JSONObject(store.readSyncState()!!))
    assertTrue("ETag dropped when no cached detail survives", state.benchmarkEtags.isEmpty())
  }
}
