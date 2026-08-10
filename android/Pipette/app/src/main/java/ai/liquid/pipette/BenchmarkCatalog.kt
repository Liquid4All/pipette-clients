package ai.liquid.pipette

import kotlin.math.max
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject

object BenchmarkCatalog {
  // The server-synced catalog, loaded from a BenchmarkStore. Empty until the first successful sync (see BenchmarkSync); there is no bundled fallback,
  // matching iOS. Exposed as a StateFlow so view models re-seed default selection and re-render when a sync replaces it.
  private val _catalog = MutableStateFlow<List<BenchmarkDefinition>>(emptyList())

  /** Emits the current catalog and every replacement (startup restore + each completed sync). Collect to react to sync completion. */
  val changes: StateFlow<List<BenchmarkDefinition>> = _catalog.asStateFlow()

  /** Populate [all] from the synced [store] and notify [changes] collectors. Call at startup (restore a prior sync) and after each sync completes. */
  fun load(store: BenchmarkStore) {
    _catalog.value = BenchmarkSync.storedDefinitions(store)
  }

  /** Every benchmark the server-synced catalog currently lists. Empty before the first successful sync, then tracks the server exactly. */
  val all: List<BenchmarkDefinition>
    get() = _catalog.value

  // The benchmark types the New Job picker offers. `eval` and `vl_throughput` are excluded — they don't run on this client's ladder path. Matches
  // iOS `BenchmarkCatalog.selectableTypes` exactly.
  private val SELECTABLE_TYPES =
    setOf(BenchmarkType.PREFILL_THROUGHPUT, BenchmarkType.DECODE_THROUGHPUT, BenchmarkType.END_TO_END_LATENCY, BenchmarkType.MAX_MEMORY_USAGE)

  // Upper bound (tokens) on the context a selectable benchmark may require — keeps the heaviest rung (8192, which jetsam-OOMs on phones) out of the
  // picker. Matches iOS `BenchmarkCatalog.contextLimit`.
  private const val CONTEXT_LIMIT = 5000L

  /**
   * Benchmarks advertised in the New Job picker — the four supported ladder types capped to rungs whose required context stays under [CONTEXT_LIMIT].
   * A UI-visibility filter only; hidden benchmarks stay in [all] and remain resolvable via [byId] and the run path. Mirrors iOS
   * `BenchmarkCatalog.selectable(from:)` for full parity.
   */
  val selectable: List<BenchmarkDefinition>
    get() = selectable(all)

  /**
   * [items] narrowed to the picker's offering:
   * 1. Keep only the four supported ladder types (drop `eval`, `vl_throughput`).
   * 2. Keep each rung whose required context stays under [CONTEXT_LIMIT] (prefill for prefill/max-memory, prefill + decode for decode/e2e).
   *
   * Pure — exposed for testing the filter without the store. Output is stably sorted by benchmark id.
   */
  fun selectable(items: List<BenchmarkDefinition>): List<BenchmarkDefinition> =
    items
      .filter { it.type in SELECTABLE_TYPES }
      .filter {
        val ctx = BenchmarkContextSize.required(it.benchmarkType, it.rawJson)
        ctx != null && ctx < CONTEXT_LIMIT
      }
      .sortedBy { it.benchmarkId.toString() }

  /** Look up a benchmark in the synced catalog by id. Null if the catalog doesn't list it (use [resolve] to also accept a structurally-parsed id). */
  fun byId(id: String): BenchmarkDefinition? = all.firstOrNull { it.benchmarkId.toString() == id }

  /**
   * Resolve an id to a definition: the synced catalog first, else one reconstructed from the structured id ([BenchmarkDefinition.parseId]). This is
   * the catalog-independent resolver the run path and result readers use, so a historical or pre-sync id (the four ladder types) still resolves its
   * type/params even when the catalog is empty. Mirrors iOS `BenchmarkCatalog.item(forId:in:)`.
   */
  fun resolve(id: String): BenchmarkDefinition? = byId(id) ?: BenchmarkDefinition.parseId(id)

  fun displayName(type: String): String =
    BenchmarkType.fromWire(type)?.displayName ?: type.split("_").joinToString(" ") { part -> part.replaceFirstChar { it.titlecase() } }

  fun matchesSearch(item: BenchmarkDefinition, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    return listOf(item.benchmarkId.toString(), item.benchmarkType, displayName(item.benchmarkType), item.label).any { it.lowercase().contains(q) }
  }

  fun typeRank(type: String): Int = BenchmarkType.fromWire(type)?.rank ?: 6
}

object BenchmarkContextSize {
  private const val EVAL_PROMPT_BUDGET = 8192L

  fun required(benchmarkType: String, params: JSONObject): Long? {
    val prefill = params.optLongCompat("parameter_prefill_tokens")
    val decode = params.optLongCompat("parameter_decode_tokens")
    val maxTokens = params.optLongCompat("parameter_max_tokens")
    return when (benchmarkType) {
      "prefill_throughput" -> prefill
      "max_memory_usage" -> saturatingAdd(prefill, 1)
      "decode_throughput",
      "end_to_end_latency" -> saturatingAdd(prefill, decode)
      "vl_throughput" -> {
        val width = params.optLongCompat("parameter_image_width")
        val height = params.optLongCompat("parameter_image_height")
        val imageTokens = saturatingMul(width / 14, height / 14)
        val text = params.optLongCompat("parameter_text_tokens")
        saturatingAdd(saturatingAdd(imageTokens, text), decode)
      }
      "eval" -> saturatingAdd(EVAL_PROMPT_BUDGET, maxTokens)
      else -> null
    }
  }

  fun effective(userPicked: Int, benchmarks: List<BenchmarkDefinition>): Int {
    val required = benchmarks.maxOfOrNull { required(it.benchmarkType, it.rawJson) ?: 0L } ?: 0L
    return max(userPicked, required.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
  }

  fun perCell(benchmarkType: String, params: JSONObject): Int {
    val required = required(benchmarkType, params) ?: return 4096
    val value = if (benchmarkType == "vl_throughput") max(required, 8192L) else required
    return value.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
  }

  private fun saturatingAdd(a: Long, b: Long): Long = if (Long.MAX_VALUE - a < b) Long.MAX_VALUE else a + b

  private fun saturatingMul(a: Long, b: Long): Long = if (a != 0L && Long.MAX_VALUE / a < b) Long.MAX_VALUE else a * b
}

fun JSONObject.optIntOrNull(name: String): Int? {
  if (!has(name) || isNull(name)) return null
  return when (val value = get(name)) {
    is Number -> value.toInt()
    is String -> value.toIntOrNull()
    else -> null
  }
}

fun JSONObject.optLongCompat(name: String): Long {
  if (!has(name) || isNull(name)) return 0
  return when (val value = get(name)) {
    is Number -> value.toLong().coerceAtLeast(0)
    is String -> value.toLongOrNull()?.coerceAtLeast(0) ?: 0
    else -> 0
  }
}
