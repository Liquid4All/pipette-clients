package ai.liquid.pipette

/**
 * The compact `model=` URI, mirroring `crates/pipette-cli/src/model_uri.rs` (and iOS's `ModelUri.swift`, which mirrors the same file).
 *
 * The CLI's `--model` accepts a JSON `Model`, this URI, or a digest reference. This client parses the URI only: it is the spelling `models` listings
 * print and the one a plan-supplied coordinate arrives in, so one runbook covers all three clients. `Display` (`{repo}[:{path}]`) is deliberately not
 * an input spelling here, as it is not one in the crate either; it is what logs and warehouse keys use.
 *
 * Grammar, verbatim from the crate:
 * ```text
 * uri    ::= scheme "://" body            ; split on the FIRST "://"
 * scheme ::= "gguf-text" | "gguf-vision" | "mlx" | "torch" | "openvino" | "onnx-genai"
 * body   ::= "" | pair ("&" pair)*        ; keys unordered, each at most once
 * pair   ::= key "=" value                ; first "=" splits; value may contain "="
 * ```
 *
 * Android mirrors the repo-backed GGUF arms only, which is exactly what [Model] can express. Everything else in the vocabulary is refused **by name**
 * rather than left to fail later:
 * - `url=` has no counterpart, because `Model` has no absolute-URL arm.
 * - `mlx`, `torch`, `openvino` and `onnx-genai` name engines this build does not link.
 * - `rev` and `sha256` pin something this client cannot honour.
 *
 * That last one is the important refusal. To accept a pinned digest and ignore it would run bytes the caller did not pin, and then submit the row as
 * though it had. That is worse than a refused invocation.
 *
 * Parse only. There is no `model_to_uri` here because nothing on Android prints a coordinate yet; when a `models` listing verb exists it can grow
 * one, and the shared key constants below are what will keep the two directions from drifting.
 */
object ModelUri {

  /** Why a `model=` value is not a coordinate this client can run. Mirrors the crate's `ModelUriError` for the cases Android can reach. */
  sealed class Failure(val reason: String) {
    /** No `://`, so this was never a URI. Usually a bare name, which is what `match=` is for. */
    data object NotAUri : Failure("expected a `<scheme>://key=value` URI; use `match=` to select by name from the models already on device")

    class UnknownScheme(scheme: String) : Failure("unknown scheme `$scheme`")

    class UnsupportedScheme(scheme: String) : Failure("`$scheme` names an engine this build does not link")

    /** `apple-foundation`/`local`/`file` name real but non-importable sources, which the crate separates from "unknown" for the same reason. */
    class NotImportable(scheme: String) : Failure("`$scheme` names a source that cannot be imported")

    class MalformedPair(pair: String) : Failure("`$pair` is not a `key=value` pair")

    class DuplicateKey(key: String) : Failure("`$key` appears more than once")

    class UnknownKey(key: String) : Failure("no such key `$key`")

    class MissingKey(key: String) : Failure("missing `$key`")

    class NotRepresentable(what: String) : Failure("$what cannot be expressed on this client")

    class InvalidValue(key: String, value: String, why: String) : Failure("`$key=$value` is not valid: $why")
  }

  // Key names, spelled as the crate's KEY_* constants spell them, so a rename there is a one-line change here rather than a hunt.
  private const val KEY_REPO = "repo"
  private const val KEY_PATH = "path"
  private const val KEY_MODEL = "model"
  private const val KEY_MMPROJ = "mmproj"
  private const val KEY_REV = "rev"
  private const val KEY_URL = "url"
  private const val KEY_PREFIX = "prefix"
  private const val KEY_SHA256 = "sha256"
  private const val KEY_MODEL_SHA256 = "model_sha256"
  private const val KEY_MMPROJ_SHA256 = "mmproj_sha256"

  private const val SCHEME_GGUF_TEXT = "gguf-text"
  private const val SCHEME_GGUF_VISION = "gguf-vision"

  /** Engines the crate's vocabulary defines that this build does not link. Named so they are refused as unsupported rather than as unknown. */
  private val UNLINKED_SCHEMES = setOf("mlx", "torch", "openvino", "onnx-genai")

  /** Sources that are real but not importable. The crate gives these their own error rather than calling them unknown schemes. */
  private val NON_IMPORTABLE_SCHEMES = setOf("apple-foundation", "local", "file")

  /** The whole key vocabulary, so an unrecognized key is reported as unknown and a recognized-but-unrepresentable one gets its own refusal. */
  private val KNOWN_KEYS =
    setOf(KEY_REPO, KEY_PATH, KEY_MODEL, KEY_MMPROJ, KEY_REV, KEY_URL, KEY_PREFIX, KEY_SHA256, KEY_MODEL_SHA256, KEY_MMPROJ_SHA256)

  /**
   * Keys this client understands but cannot honour, each with what it would have pinned.
   *
   * Separate from [Failure.UnknownKey] on purpose: a caller who pinned a digest needs to know the pin was rejected, not that the key does not exist.
   */
  private val UNREPRESENTABLE_KEYS =
    mapOf(
      KEY_URL to "an absolute-URL source",
      KEY_REV to "a pinned repo revision",
      KEY_SHA256 to "a pinned file digest",
      KEY_MODEL_SHA256 to "a pinned weights digest",
      KEY_MMPROJ_SHA256 to "a pinned projector digest",
      KEY_PREFIX to "a repo subpath prefix",
    )

  sealed class Result {
    data class Ok(val model: Model) : Result()

    data class Error(val failure: Failure) : Result()
  }

  /** Parse [raw] into a coordinate, or report the first thing wrong with it. */
  @Suppress("ReturnCount") // one return per stage the grammar can fail at, in the order the crate checks them; a single exit would need a sentinel
  fun parse(raw: String): Result {
    val trimmed = raw.trim()
    val separator = trimmed.indexOf("://")
    if (separator <= 0) return Result.Error(Failure.NotAUri)
    val scheme = trimmed.substring(0, separator)
    val body = trimmed.substring(separator + "://".length)

    val schemeFailure = schemeFailure(scheme)
    if (schemeFailure != null) return Result.Error(schemeFailure)

    val pairs = parsePairs(body)
    if (pairs is PairResult.Error) return Result.Error(pairs.failure)
    val keys = (pairs as PairResult.Ok).values

    keys.keys
      .firstOrNull { it !in KNOWN_KEYS }
      ?.let {
        return Result.Error(Failure.UnknownKey(it))
      }
    // Before the required-key check, so a caller who pinned a digest hears about the pin rather than about a key they did present.
    keys.keys
      .mapNotNull { key -> UNREPRESENTABLE_KEYS[key]?.let { key to it } }
      .minByOrNull { it.first }
      ?.let {
        return Result.Error(Failure.NotRepresentable(it.second))
      }

    return when (scheme) {
      SCHEME_GGUF_TEXT -> text(keys)
      else -> vision(keys)
    }
  }

  private fun schemeFailure(scheme: String): Failure? =
    when {
      scheme == SCHEME_GGUF_TEXT || scheme == SCHEME_GGUF_VISION -> null
      scheme in UNLINKED_SCHEMES -> Failure.UnsupportedScheme(scheme)
      scheme in NON_IMPORTABLE_SCHEMES -> Failure.NotImportable(scheme)
      else -> Failure.UnknownScheme(scheme)
    }

  @Suppress("ReturnCount") // one guard per key that can be absent or unparseable; folding them would hide which key the caller got wrong
  private fun text(keys: Map<String, String>): Result {
    val rawRepo = keys[KEY_REPO] ?: return Result.Error(Failure.MissingKey(KEY_REPO))
    val rawPath = keys[KEY_PATH] ?: return Result.Error(Failure.MissingKey(KEY_PATH))
    val repo = HfRepo.parseSlugOrNull(rawRepo) ?: return Result.Error(badRepo(rawRepo))
    val path = GgufFilename.parseOrNull(rawPath) ?: return Result.Error(badFilename(KEY_PATH, rawPath))
    return Result.Ok(Model.GgufText(HfGgufText(repo, path)))
  }

  @Suppress("ReturnCount") // as above, over the three keys a vision coordinate needs
  private fun vision(keys: Map<String, String>): Result {
    val rawRepo = keys[KEY_REPO] ?: return Result.Error(Failure.MissingKey(KEY_REPO))
    val rawModel = keys[KEY_MODEL] ?: return Result.Error(Failure.MissingKey(KEY_MODEL))
    val rawMmproj = keys[KEY_MMPROJ] ?: return Result.Error(Failure.MissingKey(KEY_MMPROJ))
    val repo = HfRepo.parseSlugOrNull(rawRepo) ?: return Result.Error(badRepo(rawRepo))
    val weights = GgufFilename.parseOrNull(rawModel) ?: return Result.Error(badFilename(KEY_MODEL, rawModel))
    val projector = GgufFilename.parseOrNull(rawMmproj) ?: return Result.Error(badFilename(KEY_MMPROJ, rawMmproj))
    return Result.Ok(Model.GgufVision(HfGgufVision(repo, weights, projector)))
  }

  private fun badRepo(raw: String) = Failure.InvalidValue(KEY_REPO, raw, "expected `org/repo_name`")

  private fun badFilename(key: String, raw: String) = Failure.InvalidValue(key, raw, "expected a `.gguf` filename")

  private sealed class PairResult {
    class Ok(val values: Map<String, String>) : PairResult()

    class Error(val failure: Failure) : PairResult()
  }

  /**
   * `key=value` pairs separated by `&`, keys unordered and each at most once.
   *
   * The first `=` splits, so a value may contain further `=`; `&` may not appear in a value, which is the crate's stated limit rather than an
   * omission here. A duplicate key is an error rather than last-wins: the two spellings would usually disagree, and quietly picking one is how a
   * script runs a model it did not name.
   */
  @Suppress("ReturnCount") // the empty body plus the two malformed shapes, each reported where it is detected
  private fun parsePairs(body: String): PairResult {
    if (body.isBlank()) return PairResult.Ok(emptyMap())
    val values = mutableMapOf<String, String>()
    for (pair in body.split("&")) {
      val separator = pair.indexOf('=')
      if (separator <= 0) return PairResult.Error(Failure.MalformedPair(pair))
      val key = pair.substring(0, separator)
      if (values.containsKey(key)) return PairResult.Error(Failure.DuplicateKey(key))
      values[key] = pair.substring(separator + 1)
    }
    return PairResult.Ok(values)
  }
}
