package ai.liquid.pipette

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
 * Making it settable needs a source for the value, and the blocker is upstream of this file: `enable_thinking` only affects eval, and
 * `BenchmarkCatalog.SELECTABLE_TYPES` offers no eval cell to attach it to. Android's kernel does run eval (`native/benchmarks.rs`), so this is a
 * picker gap rather than a capability one. See PIP-436.
 */
data class ModelFlags(val enableThinking: Boolean? = null) {
  /** Comma-joined `key=value` for set fields, or null when all default. Matches plan-types `ModelFlags::canonical_string`. */
  fun canonicalString(): String? = enableThinking?.let { "enable_thinking=$it" }

  /**
   * The `model_flags` value for a submission, or null when there is nothing to report. Mirrors plan-types
   * `ModelFlags::submission_string(benchmark_type)`.
   *
   * Eval only, and deliberately: `enable_thinking` changes what the model generates, which moves an eval score and nothing else. A
   * throughput/latency/memory row is insensitive to it, so carrying the flag there would split warehouse joins on a value that had no effect on the
   * number being joined.
   */
  fun submissionString(benchmarkType: BenchmarkType): String? =
    when (benchmarkType) {
      BenchmarkType.EVAL -> canonicalString()
      BenchmarkType.PREFILL_THROUGHPUT,
      BenchmarkType.DECODE_THROUGHPUT,
      BenchmarkType.END_TO_END_LATENCY,
      BenchmarkType.MAX_MEMORY_USAGE,
      BenchmarkType.VL_THROUGHPUT -> null
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
