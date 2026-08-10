package ai.liquid.pipette

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Syncs the benchmark catalog from the management server into on-device storage, the Kotlin port of iOS `BenchmarkSync` (and the Rust client's
 * two-level conditional pull, `pipette-cli::client::sync::pull_remote_benchmarks`).
 *
 * **Two-level, ETag-conditional pull**
 * 1. **List** — `GET /benchmarks` (definitions only) with the list `ETag` as `If-None-Match`. Keep only entries that fully parse; a `304` reuses the
 *    cached index.
 * 2. **Per-id** — `GET /benchmarks/{id}` for each kept benchmark with its stored `ETag`, bounded-concurrent. Only the per-id response carries the
 *    eval `samples`, so each benchmark is fetched individually; a `304` keeps the cached detail.
 *
 * **Tolerant**: stores only definitions [BenchmarkDefinition.parse] fully accepts, and a per-id failure never aborts the sync. The two ways an entry
 * can fail are reported differently, per PIP-248: an unrecognized `benchmark_type` is routine version skew and goes to [log], while a known type
 * carrying parameters the client can't read means server and client disagree about a shape both implement, and goes to [logError]. Neither is
 * surfaced to the user.
 *
 * Storage goes through an injected [BenchmarkStore] (file-backed in production, in-memory in tests); the network goes through injected [ListFetcher]/
 * [DetailFetcher] closures, so the flow is unit-testable without a real server or filesystem.
 */
object BenchmarkSync {
  private const val TAG = "BenchmarkSync"

  /** Max concurrent per-id detail fetches, mirroring the iOS bounded pool. */
  private const val DETAIL_CONCURRENCY = 6

  /** Conditional `GET /benchmarks`. Defaults to [ManagementClient.fetchBenchmarks]; injectable so tests drive the flow without a real network. */
  fun interface ListFetcher {
    fun fetch(serverUrl: String, ifNoneMatch: String?): ManagementClient.ConditionalGet
  }

  /** Conditional `GET /benchmarks/{id}`. Defaults to [ManagementClient.fetchBenchmark]. */
  fun interface DetailFetcher {
    fun fetch(serverUrl: String, benchmarkId: String, ifNoneMatch: String?): ManagementClient.ConditionalGet
  }

  /** Persisted sync state: the list-level ETag plus a per-benchmark ETag map, mirroring the Rust `RemoteSyncState`. */
  data class SyncState(val benchmarkCount: Int, val benchmarksEtag: String?, val benchmarkEtags: Map<String, String>) {
    fun toJson(): JSONObject {
      val etags = JSONObject()
      benchmarkEtags.forEach { (id, etag) -> etags.put(id, etag) }
      return JSONObject().put("benchmark_count", benchmarkCount).put("benchmarks_etag", benchmarksEtag).put("benchmark_etags", etags)
    }

    companion object {
      fun fromJson(obj: JSONObject): SyncState {
        val etagsObj = obj.optJSONObject("benchmark_etags") ?: JSONObject()
        val etags = buildMap { etagsObj.keys().forEach { key -> put(key, etagsObj.getString(key)) } }
        return SyncState(
          benchmarkCount = obj.optInt("benchmark_count"),
          benchmarksEtag = obj.optNullableString("benchmarks_etag"),
          benchmarkEtags = etags,
        )
      }
    }
  }

  /**
   * Pull the catalog and persist it via [client]'s public benchmark endpoints; returns the number of stored (fully-parseable) benchmarks. The
   * production entry point — the trigger sites hold the app's [ManagementClient].
   */
  suspend fun sync(
    serverUrl: String,
    store: BenchmarkStore,
    client: ManagementClient,
    log: (String) -> Unit = { Log.i(TAG, it) },
    logError: (String) -> Unit = { Log.e(TAG, it) },
  ): Int = sync(serverUrl, store, ListFetcher(client::fetchBenchmarks), DetailFetcher(client::fetchBenchmark), log, logError)

  /**
   * Pull the catalog and persist it; returns the number of stored (fully-parseable) benchmarks. Fetchers are injected so tests drive the flow with
   * fakes; [log] is injectable so tests avoid the unmockable `android.util.Log`.
   */
  suspend fun sync(
    serverUrl: String,
    store: BenchmarkStore,
    fetchList: ListFetcher,
    fetchDetail: DetailFetcher,
    log: (String) -> Unit = { Log.i(TAG, it) },
    logError: (String) -> Unit = { Log.e(TAG, it) },
  ): Int =
    withContext(Dispatchers.IO) {
      val prior = loadSyncState(store)

      // Level 1 — list (definitions only), ETag-conditional. Keep only the entries that fully parse; persist that filtered view as the index.
      val list = fetchList.fetch(serverUrl, prior?.benchmarksEtag)

      val ids: List<String>
      val listEtag: String?
      val listBody = list.body
      if (listBody != null) {
        val kept = keepParseable(listBody, log, logError)
        store.writeIndex(kept.entriesJson)
        ids = kept.ids
        listEtag = list.etag
        log("list modified: kept ${ids.size} definitions")
      } else {
        val cached = store.readIndex()
        if (cached != null) {
          ids = keepParseable(cached, log, logError).ids
          listEtag = list.etag ?: prior?.benchmarksEtag
          log("list unchanged (304)")
        } else {
          // 304 with no cached index — nothing to work from.
          log("list 304 but no cached index")
          return@withContext 0
        }
      }

      // Drop detail files for benchmarks no longer in the catalog (the index self-prunes; the stored details don't).
      store.pruneDetails(ids)

      // Level 2 — per-id detail (carries eval samples), ETag-conditional, bounded-concurrent. A 304 keeps the cached detail and its ETag; a detail
      // that doesn't parse (or errors) is skipped without aborting.
      val priorEtags = prior?.benchmarkEtags ?: emptyMap()
      val fetched = fetchDetails(ids, serverUrl, priorEtags, fetchDetail, store, log, logError)

      val newEtags = buildMap { fetched.forEach { (id, etag) -> if (etag != null) put(id, etag) } }
      store.writeSyncState(SyncState(ids.size, listEtag, newEtags).toJson().toString())

      log("synced ${ids.size} benchmarks")
      ids.size
    }

  /** The synced catalog's list entries as typed definitions (from the kept index). Empty if nothing has been synced. Feeds [BenchmarkCatalog]. */
  fun storedDefinitions(store: BenchmarkStore): List<BenchmarkDefinition> {
    val body = store.readIndex() ?: return emptyList()
    return rawArray(body).mapNotNull { BenchmarkDefinition.fromJson(it) }
  }

  /** The full per-id benchmark detail (eval `samples` included), as raw JSON. `null` until that benchmark's detail has been synced. */
  fun storedBenchmark(id: String, store: BenchmarkStore): String? = store.readDetail(id)

  // MARK: - Tolerant filtering

  private data class Kept(val entriesJson: String, val ids: List<String>)

  /**
   * The message for a skipped entry. Which sink it goes to is the caller's, and is the policy: an [BenchmarkDefinition.ParseResult.UnknownType] is
   * routine version skew and rides the info sink, while a [BenchmarkDefinition.ParseResult.SchemaMismatch] means the server and this client disagree
   * about a shape they both implement and rides the error sink.
   */
  private fun skipMessage(parsed: BenchmarkDefinition.ParseResult): String =
    when (parsed) {
      is BenchmarkDefinition.ParseResult.UnknownType -> "skipping '${parsed.benchmarkId}': unrecognized benchmark_type '${parsed.type}'"
      is BenchmarkDefinition.ParseResult.SchemaMismatch -> "skipping '${parsed.benchmarkId}' (${parsed.type}): schema mismatch: ${parsed.detail}"
      is BenchmarkDefinition.ParseResult.Ok -> ""
    }

  /**
   * Keep only the raw list entries that fully parse into a [BenchmarkDefinition], preserving order. Returns the kept raw entries (to persist
   * verbatim) and their ids (to drive the per-id fetch). A skipped entry is logged, never surfaced.
   */
  private fun keepParseable(body: String, log: (String) -> Unit, logError: (String) -> Unit): Kept {
    // Fail on a malformed list rather than treating it as empty: writing an empty index here would clobber the last good catalog and prune every
    // cached detail. A valid empty array (`[]`) is a legitimate "server has no benchmarks" and is kept.
    val array = runCatching { JSONArray(body) }.getOrElse { throw IllegalStateException("benchmark list is not a JSON array", it) }
    val kept = JSONArray()
    val ids = mutableListOf<String>()
    for (i in 0 until array.length()) {
      val entry = array.optJSONObject(i) ?: continue
      when (val parsed = BenchmarkDefinition.parse(entry)) {
        is BenchmarkDefinition.ParseResult.Ok -> {
          kept.put(entry)
          ids += parsed.definition.benchmarkId.toString()
        }
        is BenchmarkDefinition.ParseResult.UnknownType -> log(skipMessage(parsed))
        is BenchmarkDefinition.ParseResult.SchemaMismatch -> logError(skipMessage(parsed))
      }
    }
    return Kept(kept.toString(), ids)
  }

  // MARK: - Per-id fetch

  /**
   * Fetch every kept benchmark's detail with at most [DETAIL_CONCURRENCY] in flight. Each result is `(benchmarkId, etag?)`; a per-id failure returns
   * the prior ETag so the rest of the sync still completes.
   */
  private suspend fun fetchDetails(
    ids: List<String>,
    serverUrl: String,
    priorEtags: Map<String, String>,
    fetch: DetailFetcher,
    store: BenchmarkStore,
    log: (String) -> Unit,
    logError: (String) -> Unit,
  ): List<Pair<String, String?>> = coroutineScope {
    val gate = Semaphore(DETAIL_CONCURRENCY)
    ids.map { id -> async { gate.withPermit { fetchDetail(id, serverUrl, priorEtags[id], fetch, store, log, logError) } } }.awaitAll()
  }

  // Tolerance is the whole point: any per-id failure (network, malformed body, etc.) must keep the prior ETag and let the rest of the sync finish,
  // mirroring iOS — so the broad catch is intentional.
  @Suppress("TooGenericExceptionCaught")
  private fun fetchDetail(
    id: String,
    serverUrl: String,
    priorEtag: String?,
    fetch: DetailFetcher,
    store: BenchmarkStore,
    log: (String) -> Unit,
    logError: (String) -> Unit,
  ): Pair<String, String?> =
    try {
      val response = fetch.fetch(serverUrl, id, priorEtag)
      val body = response.body
      when {
        // 304 — reuse the cached detail, but only keep its ETag if the cached body still exists; otherwise drop it so the next sync sends no
        // If-None-Match and forces a fresh 200 (else a missing detail would never be re-fetched).
        body == null -> id to (response.etag ?: priorEtag).takeIf { store.readDetail(id) != null }
        else ->
          // Store the detail only if it fully parses. A mismatch drops the ETag so a corrected detail is re-fetched next time. The detail is where
          // an eval's `samples` arrive, so this is the arm that catches a sample payload the client can't read.
          when (val parsed = BenchmarkDefinition.parse(runCatching { JSONObject(body) }.getOrNull() ?: JSONObject())) {
            is BenchmarkDefinition.ParseResult.Ok -> {
              store.writeDetail(id, body)
              id to response.etag
            }
            is BenchmarkDefinition.ParseResult.UnknownType -> {
              log("skipping detail '$id': ${skipMessage(parsed)}")
              id to null
            }
            is BenchmarkDefinition.ParseResult.SchemaMismatch -> {
              logError("skipping detail '$id': ${skipMessage(parsed)}")
              id to null
            }
          }
      }
    } catch (error: Throwable) {
      log("detail fetch failed for '$id': ${error.message ?: error.javaClass.simpleName}")
      // Keep the prior ETag only if the cached body survives; else drop it so the next sync refetches from scratch.
      id to priorEtag.takeIf { store.readDetail(id) != null }
    }

  // MARK: - Helpers

  private fun rawArray(body: String): List<JSONObject> {
    val arr = runCatching { JSONArray(body) }.getOrNull() ?: return emptyList()
    return (0 until arr.length()).mapNotNull { arr.optJSONObject(it) }
  }

  private fun loadSyncState(store: BenchmarkStore): SyncState? {
    val data = store.readSyncState() ?: return null
    return runCatching { SyncState.fromJson(JSONObject(data)) }.getOrNull()
  }
}
