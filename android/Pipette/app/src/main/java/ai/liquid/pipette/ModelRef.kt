package ai.liquid.pipette

import org.json.JSONObject

/**
 * Kotlin mirror of the model-identifier types in pipette-plan-types (`HfOrg`, `HfRepoName`, `GgufFilename`, `HfRepo`, `ModelFlags`) plus the GGUF
 * model variants. [HfGgufText] / [HfGgufVision] collapse the reshaped plan-types `GgufText`/`GgufVision` + their `GgufTextSource::HuggingFace` /
 * `GgufVisionSource::HuggingFace` sub-tags into a single HF-sourced Kotlin type — the only source the Android client resolves. The validation regexes
 * are copied verbatim from the Rust `nutype` definitions, so a value accepted here is accepted there and vice-versa. Wire serialization lives in
 * [SubmissionRef], which emits the reshaped tagged shapes (`type` + `source` + coordinate).
 */

/** Raised when a value fails the plan-types validation rules. */
class PlanTypeError(message: String) : IllegalArgumentException(message)

@JvmInline
value class HfOrg private constructor(val value: String) {
  override fun toString(): String = value

  companion object {
    // pipette-plan-types: nutype(validate(regex = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"))
    private val REGEX = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")

    fun parse(raw: String): HfOrg = if (REGEX.matches(raw)) HfOrg(raw) else throw PlanTypeError("invalid HuggingFace org: '$raw'")

    fun parseOrNull(raw: String): HfOrg? = if (REGEX.matches(raw)) HfOrg(raw) else null
  }
}

@JvmInline
value class HfRepoName private constructor(val value: String) {
  override fun toString(): String = value

  companion object {
    // pipette-plan-types: nutype(validate(regex = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"))
    private val REGEX = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")

    fun parse(raw: String): HfRepoName = if (REGEX.matches(raw)) HfRepoName(raw) else throw PlanTypeError("invalid HuggingFace repo name: '$raw'")

    fun parseOrNull(raw: String): HfRepoName? = if (REGEX.matches(raw)) HfRepoName(raw) else null
  }
}

@JvmInline
value class GgufFilename private constructor(val value: String) {
  override fun toString(): String = value

  companion object {
    // pipette-plan-types: nutype(validate(regex = r"^[A-Za-z0-9._-]+\.gguf$"))
    private val REGEX = Regex("^[A-Za-z0-9._-]+\\.gguf$")

    fun parse(raw: String): GgufFilename = if (REGEX.matches(raw)) GgufFilename(raw) else throw PlanTypeError("invalid GGUF filename: '$raw'")

    fun parseOrNull(raw: String): GgufFilename? = if (REGEX.matches(raw)) GgufFilename(raw) else null
  }
}

/** `org/repo_name` coordinate. Mirrors plan-types `HfRepo`. */
data class HfRepo(
  val org: HfOrg,
  val repoName: HfRepoName,
  /** Gated/private repo needing an HF token; not part of the repo identity. */
  val requiresAuth: Boolean = false,
) {
  /** Canonical `org/repo_name` slug — HuggingFace's own convention. */
  override fun toString(): String = "$org/$repoName"

  companion object {
    /** Mirrors `HfRepo::parse_slug` — split on the first `/`. */
    fun parseSlug(slug: String): HfRepo {
      val sep = slug.indexOf('/')
      if (sep < 0) throw PlanTypeError("HF repo slug missing '/': '$slug'")
      return HfRepo(HfOrg.parse(slug.substring(0, sep)), HfRepoName.parse(slug.substring(sep + 1)))
    }

    fun parseSlugOrNull(slug: String): HfRepo? = runCatching { parseSlug(slug) }.getOrNull()
  }
}

/**
 * Per-model flags: what the *generation* was shaped by, as opposed to how the engine was loaded (`runtime_flags`) or how the harness was driven
 * (`benchmark_flags`). Mirrors plan-types `ModelFlags`.
 *
 * **Nothing on Android constructs this with a value yet.** [SubmissionRef] builds every model with the default, so [submissionString] answers null on
 * every submission this client currently produces. The type is not dead, though: it is a field of [HfGgufText] / [HfGgufVision] and keeps the Kotlin
 * mirror shaped like the plan-types one, and the encoder below is the wire rule, so a model that does carry a flag reaches the server correctly.
 *
 * Making it settable needs a source for the value, and the blocker is upstream of this file: `enable_thinking` and `reasoning_effort` only affect
 * eval, and `BenchmarkCatalog.SELECTABLE_TYPES` offers no eval cell to attach them to. Android's kernel does run eval
 * (`android/native/benchmarks.rs`), so this is a picker gap rather than a capability one. See PIP-436.
 *
 * Flat, where plan-types has a variant per cell: plan-types gives `reasoning_effort` to the text cell alone and refuses it on a vision one.
 */
data class ModelFlags(val enableThinking: Boolean? = null, val reasoningEffort: ReasoningEffort? = null) {
  /**
   * Comma-joined `key=value` for set fields, `enable_thinking` first, or null when all default. Matches plan-types `ModelFlags::canonical_string`,
   * whose spelling is load-bearing for row hashing.
   */
  fun canonicalString(): String? {
    val knobs = listOfNotNull(enableThinking?.let { "enable_thinking=$it" }, reasoningEffort?.let { "reasoning_effort=${it.wire}" })
    return knobs.joinToString(",").ifEmpty { null }
  }

  /**
   * The `model_flags` value for a submission, or null when there is nothing to report. Mirrors plan-types
   * `ModelFlags::submission_string(benchmark_type)`.
   *
   * The eval and vlm_eval cells only, and deliberately: `enable_thinking` and `reasoning_effort` change what the model generates, which moves an eval
   * score and nothing else. A throughput/latency/memory row is insensitive to them, so carrying a flag there would split warehouse joins on a value
   * that had no effect on the number being joined.
   */
  fun submissionString(benchmarkType: BenchmarkType): String? =
    when (benchmarkType) {
      BenchmarkType.EVAL -> canonicalString()
      // A vision cell carries `enable_thinking` alone: plan-types refuses `reasoning_effort` there.
      BenchmarkType.VLM_EVAL -> enableThinking?.let { "enable_thinking=$it" }
      BenchmarkType.PREFILL_THROUGHPUT,
      BenchmarkType.DECODE_THROUGHPUT,
      BenchmarkType.END_TO_END_LATENCY,
      BenchmarkType.MAX_MEMORY_USAGE,
      BenchmarkType.VL_THROUGHPUT -> null
    }
}

/**
 * How much a reasoning model is asked to think: the `reasoning_effort` its chat template reads, which gpt-oss renders as `Reasoning: <effort>`.
 * Mirrors plan-types `ReasoningEffort`, closed to the three levels both desktop eval runtimes pass through and spelled lowercase on the wire.
 */
enum class ReasoningEffort(val wire: String) {
  LOW("low"),
  MEDIUM("medium"),
  HIGH("high");

  companion object {
    fun fromWire(wire: String): ReasoningEffort? = entries.firstOrNull { it.wire == wire }
  }
}

/** Single-file GGUF text model. Mirrors plan-types `HfGgufText`. */
data class HfGgufText(val repo: HfRepo, val filename: GgufFilename, val modelFlags: ModelFlags = ModelFlags())

/** VL GGUF: weights + projector in the same repo. Mirrors plan-types `HfGgufVision`. */
data class HfGgufVision(val repo: HfRepo, val filename: GgufFilename, val mmprojFilename: GgufFilename, val modelFlags: ModelFlags = ModelFlags())

/**
 * A model deployment. Mirrors plan-types `Model`, restricted to the GGUF variants the Android (llama.cpp) client runs — the `hf_mlx`/`hf_torch`
 * variants are desktop-only and intentionally omitted.
 */
sealed class Model {
  abstract val modelFlags: ModelFlags
  abstract val requiresAuth: Boolean

  data class GgufText(val model: HfGgufText) : Model() {
    override val modelFlags: ModelFlags
      get() = model.modelFlags

    override val requiresAuth: Boolean
      get() = model.repo.requiresAuth
  }

  data class GgufVision(val model: HfGgufVision) : Model() {
    override val modelFlags: ModelFlags
      get() = model.modelFlags

    override val requiresAuth: Boolean
      get() = model.repo.requiresAuth
  }

  /**
   * The coordinate as a job manifest stores it, so a cell can be re-fetched from what it names rather than from a path that may no longer exist.
   *
   * Deliberately NOT [SubmissionRef]'s wire shape. That one is owned by the submission contract, is encode-only, and answers to the server's schema;
   * this one answers to a file on this device and has to survive being read back by a future build. Keeping them apart means a submission-schema
   * change cannot silently invalidate every stored manifest.
   *
   * `requiresAuth` rides along even though it is not part of the repo identity: without it a gated repo re-fetched from a manifest would be requested
   * unauthenticated and 401, and the cell would fail for a reason that reads like the weights are gone.
   */
  fun toJson(): JSONObject =
    JSONObject().apply {
      when (this@Model) {
        is GgufText -> {
          put("type", WIRE_GGUF_TEXT)
          put("repo", model.repo.toString())
          put("filename", model.filename.toString())
        }
        is GgufVision -> {
          put("type", WIRE_GGUF_VISION)
          put("repo", model.repo.toString())
          put("filename", model.filename.toString())
          put("mmproj_filename", model.mmprojFilename.toString())
        }
      }
      if (requiresAuth) put("requires_auth", true)
      modelFlags.enableThinking?.let { put("enable_thinking", it) }
      modelFlags.reasoningEffort?.let { put("reasoning_effort", it.wire) }
    }

  companion object {
    private const val WIRE_GGUF_TEXT = "hf_gguf_text"
    private const val WIRE_GGUF_VISION = "hf_gguf_vision"

    /**
     * Read back what [toJson] wrote, or null for anything this build cannot turn into a fetchable coordinate.
     *
     * Null rather than throwing, and null for a *malformed* entry as much as an absent one, because the caller is decoding a job manifest: a cell
     * whose reference no longer parses must still list and still report its results. It loses the ability to re-fetch itself, which is exactly where
     * the client was before the reference existed, so the degradation is to the old behaviour rather than to a crash.
     */
    @Suppress("ReturnCount") // one guard per field that can fail to parse; folding them would hide which one did
    fun fromJsonOrNull(json: JSONObject?): Model? {
      if (json == null) return null
      val repo = HfRepo.parseSlugOrNull(json.optString("repo")) ?: return null
      val filename = GgufFilename.parseOrNull(json.optString("filename")) ?: return null
      val authed = repo.copy(requiresAuth = json.optBoolean("requires_auth", false))
      val flags =
        ModelFlags(
          enableThinking = if (json.has("enable_thinking")) json.optBoolean("enable_thinking") else null,
          reasoningEffort = ReasoningEffort.fromWire(json.optString("reasoning_effort")),
        )
      return when (json.optString("type")) {
        WIRE_GGUF_TEXT -> GgufText(HfGgufText(authed, filename, flags))
        WIRE_GGUF_VISION -> {
          val mmproj = GgufFilename.parseOrNull(json.optString("mmproj_filename")) ?: return null
          GgufVision(HfGgufVision(authed, filename, mmproj, flags))
        }
        else -> null
      }
    }
  }
}

/** Non-empty management client id. Mirrors plan-types `ClientId`. */
@JvmInline
value class ClientId private constructor(val value: String) {
  override fun toString(): String = value

  companion object {
    fun parse(raw: String): ClientId = if (raw.isNotEmpty()) ClientId(raw) else throw PlanTypeError("client id must not be empty")

    fun parseOrNull(raw: String): ClientId? = if (raw.isNotEmpty()) ClientId(raw) else null
  }
}

/** Opaque, non-empty benchmark identifier. Mirrors plan-types `BenchmarkId`. */
@JvmInline
value class BenchmarkId private constructor(val value: String) {
  override fun toString(): String = value

  companion object {
    fun parse(raw: String): BenchmarkId = if (raw.isNotEmpty()) BenchmarkId(raw) else throw PlanTypeError("benchmark id must not be empty")

    fun parseOrNull(raw: String): BenchmarkId? = if (raw.isNotEmpty()) BenchmarkId(raw) else null
  }
}
