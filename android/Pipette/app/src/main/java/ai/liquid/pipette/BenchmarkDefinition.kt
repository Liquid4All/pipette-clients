package ai.liquid.pipette

import org.json.JSONArray
import org.json.JSONObject

/**
 * Benchmark kinds, mirroring pipette-ops `BenchmarkType`. `wire` is the `benchmark_type` tag used in the JSON the engine and management server
 * exchange; it matches the Rust serde representation exactly.
 */
enum class BenchmarkType(val wire: String, val displayName: String, val rank: Int) {
  END_TO_END_LATENCY("end_to_end_latency", "End-to-End Latency", 0),
  PREFILL_THROUGHPUT("prefill_throughput", "Prefill Throughput", 1),
  DECODE_THROUGHPUT("decode_throughput", "Decode Throughput", 2),
  MAX_MEMORY_USAGE("max_memory_usage", "Max Memory Usage", 3),
  VL_THROUGHPUT("vl_throughput", "Vision-Language Throughput", 4),
  EVAL("eval", "Eval Accuracy", 5);

  /**
   * Whether this benchmark must observe the model load itself, so the runner has to load fresh inside the measured window rather than reuse a
   * resident handle. Only [MAX_MEMORY_USAGE], whose whole measurement is the peak the load produces.
   *
   * Lives on the type, mirroring iOS `BenchmarkType.requiresFreshLoad`, because a call site that restates the rule is free to restate it wrongly:
   * this began as `cell.benchmarkType == MAX_MEMORY_USAGE.wire` in [JobRunner], which compares a nullable wire string and quietly answered false for
   * a cell whose manifest predates the stored type.
   */
  val requiresFreshLoad: Boolean
    get() = this == MAX_MEMORY_USAGE

  companion object {
    fun fromWire(wire: String): BenchmarkType? = entries.firstOrNull { it.wire == wire }
  }
}

/**
 * Typed benchmark definition, mirroring pipette-ops `BenchmarkDefinition` (a `benchmark_type`-tagged enum with per-kind parameters). [toJson]
 * serializes to the same flat, tagged shape the native engine and the management server consume.
 */
sealed class BenchmarkDefinition {
  abstract val benchmarkId: BenchmarkId
  abstract val type: BenchmarkType

  /** Wire `benchmark_type` tag — convenience for the String-based call sites. */
  val benchmarkType: String
    get() = type.wire

  /** The flat, `benchmark_type`-tagged JSON the engine/server expect. */
  abstract fun toJson(): JSONObject

  /** Alias for [toJson]; the engine bridge is fed this JSON. */
  val rawJson: JSONObject
    get() = toJson()

  /** Short human label for the parameters (used in the picker/CSV). */
  abstract val label: String

  protected fun base(): JSONObject = JSONObject().put("benchmark_id", benchmarkId.toString()).put("benchmark_type", type.wire)

  data class PrefillThroughput(override val benchmarkId: BenchmarkId, val prefillTokens: Int) : BenchmarkDefinition() {
    override val type = BenchmarkType.PREFILL_THROUGHPUT

    override fun toJson(): JSONObject = base().put("parameter_prefill_tokens", prefillTokens)

    override val label: String
      get() = "${prefillTokens}tok in"
  }

  data class DecodeThroughput(override val benchmarkId: BenchmarkId, val prefillTokens: Int, val decodeTokens: Int) : BenchmarkDefinition() {
    override val type = BenchmarkType.DECODE_THROUGHPUT

    override fun toJson(): JSONObject = base().put("parameter_prefill_tokens", prefillTokens).put("parameter_decode_tokens", decodeTokens)

    override val label: String
      get() = "${prefillTokens}tok in - $decodeTokens tok out"
  }

  data class EndToEndLatency(override val benchmarkId: BenchmarkId, val prefillTokens: Int, val decodeTokens: Int) : BenchmarkDefinition() {
    override val type = BenchmarkType.END_TO_END_LATENCY

    override fun toJson(): JSONObject = base().put("parameter_prefill_tokens", prefillTokens).put("parameter_decode_tokens", decodeTokens)

    override val label: String
      get() = "${prefillTokens}tok in - $decodeTokens tok out"
  }

  data class MaxMemoryUsage(override val benchmarkId: BenchmarkId, val prefillTokens: Int) : BenchmarkDefinition() {
    override val type = BenchmarkType.MAX_MEMORY_USAGE

    override fun toJson(): JSONObject = base().put("parameter_prefill_tokens", prefillTokens)

    override val label: String
      get() = "${prefillTokens}tok context"
  }

  data class VlThroughput(
    override val benchmarkId: BenchmarkId,
    val imageWidth: Int,
    val imageHeight: Int,
    val textTokens: Int,
    val decodeTokens: Int,
  ) : BenchmarkDefinition() {
    override val type = BenchmarkType.VL_THROUGHPUT

    override fun toJson(): JSONObject =
      base()
        .put("parameter_image_width", imageWidth)
        .put("parameter_image_height", imageHeight)
        .put("parameter_text_tokens", textTokens)
        .put("parameter_decode_tokens", decodeTokens)

    override val label: String
      get() = "${imageWidth}x$imageHeight - ${textTokens}tok text - $decodeTokens tok out"
  }

  data class Eval(
    override val benchmarkId: BenchmarkId,
    val evalId: String,
    val datasetName: String,
    val maxTokens: Int,
    val mcqChoices: List<String>?,
    val samples: JSONArray?,
  ) : BenchmarkDefinition() {
    override val type = BenchmarkType.EVAL

    override fun toJson(): JSONObject {
      val json = base().put("parameter_eval_id", evalId).put("parameter_dataset_name", datasetName).put("parameter_max_tokens", maxTokens)
      mcqChoices?.let { json.put("parameter_mcq_choices", JSONArray(it)) }
      samples?.let { json.put("samples", it) }
      return json
    }

    override val label: String
      get() = benchmarkId.toString()
  }

  /**
   * The outcome of reading one server catalog entry, split so a consumer can apply the tolerance policy PIP-248 specifies: an unrecognized type is
   * skipped quietly, a known type that fails to decode is skipped and logged as an error. Mirrors what iOS gets from a thrown `DecodingError` plus
   * [BenchmarkType] membership.
   */
  sealed interface ParseResult {
    data class Ok(val definition: BenchmarkDefinition) : ParseResult

    /**
     * `benchmark_type` is not in this client's enum. Quiet on purpose: a newer server catalog reaching an older client is expected, not a fault, and
     * logging it as one trains readers to ignore the log.
     */
    data class UnknownType(val benchmarkId: String, val type: String) : ParseResult

    /**
     * A type we understand, carrying parameters we cannot read. Worth an error: it means the server and this client disagree about a shape they are
     * both supposed to implement, which is a bug on one side rather than a version skew.
     */
    data class SchemaMismatch(val benchmarkId: String, val type: String, val detail: String) : ParseResult
  }

  companion object {
    /** Parse the flat, `benchmark_type`-tagged JSON into a typed definition, or null for any entry this client cannot fully read. */
    fun fromJson(obj: JSONObject): BenchmarkDefinition? = (parse(obj) as? ParseResult.Ok)?.definition

    /**
     * [fromJson] with the reason attached, for the sync's tolerance policy.
     *
     * Strict about required parameters, deliberately. These read with `getInt`/`getString`, which throw on a missing or ill-typed key, rather than
     * the `optInt`/`optString` this used to use, which return `0`/`""`. Defaulting is the wrong kind of tolerance here: a `decode_throughput` whose
     * `parameter_decode_tokens` is absent would decode to zero tokens and then *run*, producing a measurement of nothing that no consumer can tell
     * from a real one. Skipping it is recoverable; a plausible wrong number is not. iOS (`try c.decode(UInt32.self, …)`) and the Rust client
     * (`serde_json::from_value`) have always required these, so this is Android converging on the other two rather than a new rule.
     *
     * `parameter_mcq_choices` and `samples` stay optional, matching iOS's `decodeIfPresent`: only the per-id detail response carries `samples`, so
     * requiring it would reject every entry the list endpoint returns.
     */
    fun parse(obj: JSONObject): ParseResult {
      val rawId = obj.optString("benchmark_id").ifEmpty { "<no id>" }
      val rawType = obj.optString("benchmark_type").ifEmpty { "nil" }
      // Classified by type first so a bad id on a known type reads as a schema mismatch rather than
      // as an unknown type, matching how iOS's logSkip classifies a thrown decode.
      val type = BenchmarkType.fromWire(obj.optString("benchmark_type")) ?: return ParseResult.UnknownType(rawId, rawType)
      return runCatching { ParseResult.Ok(decodeKnown(obj, type)) }
        .getOrElse { error -> ParseResult.SchemaMismatch(rawId, rawType, error.message ?: error.javaClass.simpleName) }
    }

    /** Throws (`JSONException`, or [PlanTypeError] via [BenchmarkId]) when the entry doesn't carry what [type] requires. */
    private fun decodeKnown(obj: JSONObject, type: BenchmarkType): BenchmarkDefinition {
      val id = BenchmarkId.parse(obj.getString("benchmark_id"))
      return when (type) {
        BenchmarkType.PREFILL_THROUGHPUT -> PrefillThroughput(id, obj.getInt("parameter_prefill_tokens"))
        BenchmarkType.DECODE_THROUGHPUT -> DecodeThroughput(id, obj.getInt("parameter_prefill_tokens"), obj.getInt("parameter_decode_tokens"))
        BenchmarkType.END_TO_END_LATENCY -> EndToEndLatency(id, obj.getInt("parameter_prefill_tokens"), obj.getInt("parameter_decode_tokens"))
        BenchmarkType.MAX_MEMORY_USAGE -> MaxMemoryUsage(id, obj.getInt("parameter_prefill_tokens"))
        BenchmarkType.VL_THROUGHPUT ->
          VlThroughput(
            id,
            obj.getInt("parameter_image_width"),
            obj.getInt("parameter_image_height"),
            obj.getInt("parameter_text_tokens"),
            obj.getInt("parameter_decode_tokens"),
          )
        BenchmarkType.EVAL ->
          Eval(
            id,
            obj.getString("parameter_eval_id"),
            obj.getString("parameter_dataset_name"),
            obj.getInt("parameter_max_tokens"),
            obj.optJSONArray("parameter_mcq_choices")?.let { arr -> List(arr.length()) { arr.getString(it) } },
            obj.optJSONArray("samples"),
          )
      }
    }

    /**
     * Reconstruct a definition from a structured benchmark id, for the four ladder types — the resolution fallback when the synced catalog has no
     * entry for an id (e.g. a historical job whose benchmark left the catalog, or before the first sync). Mirrors iOS
     * `BenchmarkDefinition(parsingId:)` and the Rust `BenchmarkType::from_id`: the type is the matched prefix and the workload numbers are parsed
     * straight out of the id.
     * - `prefill_throughput_<P>` → [PrefillThroughput]
     * - `max_memory_usage_<P>` → [MaxMemoryUsage]
     * - `decode_throughput_<P>_<D>` → [DecodeThroughput]
     * - `end_to_end_latency_<P>_<D>` → [EndToEndLatency]
     *
     * Returns null for `eval`, `vl_throughput`, `_smoke`, or anything that doesn't match a known type with the expected trailing integers.
     */
    fun parseId(id: String): BenchmarkDefinition? {
      val bid = BenchmarkId.parseOrNull(id) ?: return null
      // (type prefix, arity, builder over the trailing integers). First match whose id has exactly `arity` integer segments wins.
      val ladder =
        listOf<Triple<String, Int, (List<Int>) -> BenchmarkDefinition>>(
          Triple("prefill_throughput", 1) { PrefillThroughput(bid, it[0]) },
          Triple("max_memory_usage", 1) { MaxMemoryUsage(bid, it[0]) },
          Triple("decode_throughput", 2) { DecodeThroughput(bid, it[0], it[1]) },
          Triple("end_to_end_latency", 2) { EndToEndLatency(bid, it[0], it[1]) },
        )
      return ladder.firstNotNullOfOrNull { (type, arity, build) -> suffixInts(id, type)?.takeIf { it.size == arity }?.let(build) }
    }

    /** The trailing integers after `<type>_` in [id], or null if the prefix doesn't match or any segment isn't an integer. */
    private fun suffixInts(id: String, type: String): List<Int>? {
      val prefix = "${type}_"
      if (!id.startsWith(prefix)) return null
      val parts = id.removePrefix(prefix).split("_")
      val nums = parts.mapNotNull { it.toIntOrNull() }
      return if (parts.isNotEmpty() && nums.size == parts.size) nums else null
    }
  }
}
