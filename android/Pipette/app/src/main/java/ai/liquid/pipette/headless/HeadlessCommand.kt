package ai.liquid.pipette.headless

import ai.liquid.pipette.Model
import ai.liquid.pipette.ModelUri
import ai.liquid.pipette.RunReadiness

/**
 * Parsing and validation for the headless run surface's arguments, kept free of Android types so the whole grammar is unit-testable without an
 * emulator. [HeadlessRunActivity] supplies the raw key/value pairs an `Intent`'s extras carry and acts on the result.
 *
 * The Android counterpart to iOS `HeadlessCommand.parse` (`ios/Pipette/Pipette/Headless/HeadlessCommand.swift`). Every argument is spelled the way
 * the CLI and `pipette-plan-types` spell the same concept, so one runbook covers all three clients:
 *
 * | This surface               | CLI `pipette benchmarks run`                    | iOS headless                 |
 * |----------------------------|-------------------------------------------------|------------------------------|
 * | `match`                    | (no equivalent)                                 | `match=`                     |
 * | `model`                    | `--model` (the URI spelling)                    | `model=`                     |
 * | `benchmark` / `benchmarks` | `--benchmark`                                   | `benchmark=` / `benchmarks=` |
 * | `number_gpu_layers`        | `--runtime-flags '{"number_gpu_layers":…}'`     | same, via `runtime-flags=`   |
 * | `batch`                    | (no equivalent; see below)                      | `batch=`                     |
 * | `submit`                   | `--sync` (opt-*in* there; see [Request.submit]) | `submit=`                    |
 *
 * The knobs are flat `-e key value` extras rather than iOS's single `runtime-flags=` JSON object: an `am start` line is quoted through a shell and
 * then through `adb`, where a JSON argument is the part that gets mangled. The names are the JSON's names, so the two still read the same.
 *
 * `batch` is the exception, and the difference is real rather than drift: `n_ubatch` is a load-path field of the two in-process app engines, declared
 * in plan-types only for `llamacpp_ios_pipette` and `llamacpp_apk_pipette`. The CLI's llamacpp cells drive the stock tools over argv, which expose no
 * micro-batch flag, so there is no CLI spelling to match. `batch` is iOS's name for it.
 *
 * The value is clamped to the context when recorded, on both mobile clients, because llama.cpp cannot use a micro-batch larger than the context and
 * the payload should say what ran: a cell whose benchmark sizes the context at 32 records `n_ubatch: 32` however large a `batch` was asked for. This
 * is a ceiling on an honoured knob, not the discard that [CTX_REFUSAL] describes, which is why `batch` is accepted and `ctx_size` is not.
 *
 * Refusing a malformed invocation is a guarantee, not a nicety: PIP-428 landed the same rule on iOS after a run silently used a default the caller
 * hadn't asked for. An unknown key is an error rather than ignored, because the failure it catches is a typo in a CI script that would otherwise
 * benchmark the wrong thing and report success.
 */
object HeadlessCommand {

  // The recognized arguments live on [HeadlessVerb], one set per verb, so an extra outside the selected verb's set is a usage error and a typo
  // cannot be silently dropped. Deliberately not mirrored here: two lists of key names is how one of them goes stale.

  /**
   * Why no argument sets the context window, on either mobile client.
   *
   * `JobRunner.cellContextSize` computes it per cell from the benchmark definition (`BenchmarkContextSize.perCell`) and never consults the job-level
   * value except as a fallback for a benchmark id it cannot find. A `ctx_size=` would therefore be accepted, stamped on the manifest, and then
   * ignored by the thing it names, which is worse than refusing it: a caller would have a number in the payload and no measurement behind it.
   *
   * iOS refuses `ctx_size=` for the same reason and in the same words. Honouring it means giving the computed value the role of a floor, which is a
   * change to the run path rather than to this parser.
   */
  private const val CTX_REFUSAL =
    "the per-cell context window is computed from the benchmark (JobRunner.cellContextSize) and would overwrite an authored value; " +
      "iOS refuses it for the same reason"

  /**
   * Keys refused with a specific reason rather than the generic unknown-argument message.
   *
   * Two kinds live here. `ngl` is a spelling a caller plausibly reaches for, and pointing at the replacement costs a line and saves a round trip
   * through the source. `ctx_size` is refused on its merits, not renamed: see below.
   *
   * `model` used to live here too, refused because this surface "cannot resolve" a coordinate. It can now: see [Request.modelUri].
   */
  private val REFUSED_KEYS =
    mapOf("ngl" to "use `number_gpu_layers`, the name the CLI and plan-types use for it", "ctx" to CTX_REFUSAL, "ctx_size" to CTX_REFUSAL)

  /** What a well-formed invocation asked for. Resolution against the on-device model and benchmark catalogs happens in the activity. */
  data class Request(
    /** `match=`: case-insensitive substring matched against a model's name; null runs every non-mmproj model on device. */
    val matchQuery: String?,
    /**
     * `model=`: the exact coordinate to run, fetched first if it is not on disk. Null when the caller selected with [matchQuery] instead.
     *
     * A separate key from `match=`, and refused alongside it, because they ask different questions: this one names bytes to obtain, that one filters
     * bytes already here. iOS learned this the hard way and says so at `HeadlessCommand.swift:447`: substring selection "used to live here, which
     * made `model=` mean two unrelated things and left a plan-supplied coordinate silently matching nothing".
     *
     * Which is also why `match=` was **not** widened to the offerable catalog now that the runner can fetch. An existing runbook's `match=LFM` would
     * have turned into a multi-gigabyte download nobody asked for.
     */
    val modelUri: Model?,
    /** Explicit catalog benchmark ids; empty runs every selectable benchmark. */
    val benchmarkIds: List<String>,
    /**
     * Whether results are contributed.
     *
     * Defaults to **true**, which is a deliberate divergence from the CLI's opt-in `--sync` and matches iOS for the same reason: a phone that
     * measures for an hour and publishes nothing is the failure mode worth designing against. `submit=0` is the opt-out.
     *
     * The exception is a run that named an exact [modelUri], where the default flips to **false**. Nothing sanctioned that model, so an ad-hoc
     * coordinate stays opt-*in* rather than quietly publishing warehouse rows for a build the catalog never validated. iOS draws the same line in the
     * same place (`HeadlessCommand.swift:376`), and a fleet runbook that does want the rows passes `submit=1`.
     */
    val submit: Boolean,
    /** `batch=`: the prefill chunk, llama's `n_ubatch`. */
    val prefillBatch: Int,
    /** `number_gpu_layers=`: the plan-types knob of the same name. */
    val nGpuLayers: Int,
    /**
     * `readiness_max_wait_secs=`: the run's readiness deadline, the `benchmark_flags.readiness.max_wait_secs` of every cell it measures.
     *
     * Null means the caller named none, which is what an absent `max_wait_secs` means everywhere else — the client's own default applies. Not folded
     * into a non-null default here, because the default belongs to [Readiness] and duplicating it would let the two drift.
     */
    val readinessMaxWaitSecs: Long?,
    /** `readiness_skip_thermal=`: waive the thermal criterion for this run only. Null leaves the device's Settings toggle in charge. */
    val readinessSkipThermal: Boolean?,
  )

  /**
   * What a verb other than [HeadlessVerb.RUN] was asked for. One type for the read-only and mutating verbs alike, because between them they take a
   * handful of optional selectors and no verb takes more than two.
   *
   * Kept as one type rather than split by kind because the split that matters is already carried by [HeadlessVerb.kind], and a second one here would
   * mean two places to keep in agreement about which key belongs to which verb.
   */
  data class Query(
    val verb: HeadlessVerb,
    /** `benchmarks type=`: narrow the catalog to one benchmark type. */
    val type: String? = null,
    /** `benchmarks-show benchmark=`: the id to show. Required by that verb, so the parser refuses its absence. */
    val benchmarkId: String? = null,
    /** `results-show result=` and `results-delete result=`: the cell id. Required by both, on the same grounds as [benchmarkId]. */
    val resultId: String? = null,
    /** `job-rm id=` and `job-export id=`: the job id, spelled as iOS spells it. */
    val jobId: String? = null,
    /** `models-delete model=`: the coordinate whose weights to delete, parsed by the same grammar [Request.modelUri] uses. */
    val modelUri: Model? = null,
    /**
     * `model=` exactly as the caller wrote it, carried so a refusal can quote it back.
     *
     * [modelUri] cannot serve: `Model` has no coordinate renderer on Android (`model_to_uri` is a named omission in `docs/client-parity.md`), so
     * interpolating it prints a data-class dump that no caller can copy back into a command, with spaces and parens that break the line format.
     */
    val modelSpelling: String? = null,
    /** `models-rm name=`: a model file name. */
    val name: String? = null,
    /** `models-rm repo=`: a Hugging Face repo, on its own or narrowing [name]. */
    val repo: String? = null,
    /**
     * `auth-import bundle=`: the base64url token `auth-export` printed. Required by that verb, on the same grounds as [benchmarkId].
     *
     * Carried as the raw token rather than a decoded `IdentityBundle`, because decoding is a refusal this parser has no vocabulary for: what is wrong
     * with a bundle is `IdentityBundle.Failure`'s to say, and `HeadlessMutate` is where that refusal is worded.
     */
    val bundle: String? = null,
    /**
     * `auth-import force=`: proceed over a signing key the device already holds.
     *
     * Defaults to false, and the refusal it unlocks is the only one on this surface that guards data the verb does not touch — the results already
     * submitted under the key being replaced.
     */
    val force: Boolean = false,
  )

  sealed interface Parsed {
    /** The run verb, which is also what an invocation with no `verb=` means. */
    data class Ok(val request: Request) : Parsed

    /** A read-only verb: nothing starts, and the answer goes to the log. */
    data class Inspect(val query: Query) : Parsed

    /**
     * A verb that deletes or writes something.
     *
     * Carries the same [Query] an inspection does, and is a separate arm only so the activity cannot route one as the other. What it targets is
     * resolved against the device by [HeadlessMutate], not here: this object never touches disk.
     */
    data class Mutate(val query: Query) : Parsed

    /** The invocation was malformed. [message] is written to the log and the caller's result, and is the whole value of this path. */
    data class UsageError(val message: String) : Parsed
  }

  /**
   * Defaults chosen to match what the New Job screen starts a UI job with (`JobsViewModel`), so a headless run and a hand-driven one measure the same
   * thing unless the caller says otherwise.
   */
  const val DEFAULT_N_GPU_LAYERS = 99
  const val DEFAULT_PREFILL_BATCH = 512

  /**
   * Ceiling on `readiness_max_wait_secs`, refusing anything above it so the caller hears about a typo rather than measuring under it.
   *
   * The value is [RunReadiness.MAX_WAIT_SECS], which documents the choice and applies the same range when the block is read back off a manifest. One
   * definition, because a parse boundary that admitted more than the stored form accepts would drop the difference on the next load.
   */
  const val MAX_READINESS_WAIT_SECS = RunReadiness.MAX_WAIT_SECS

  /**
   * The context size the manifest is stamped with, matching `JobsViewModel`'s. **Not a parse default**: no argument sets it, because nothing would
   * come of one. See [CTX_REFUSAL].
   */
  const val MANIFEST_CONTEXT_SIZE = 4096

  @Suppress("ReturnCount") // the verb refusal, the key-set refusal, and the two well-formed shapes; each is already its own answer
  fun parse(args: Map<String, String>): Parsed {
    val verb =
      when (val raw = args[HeadlessVerb.VERB_KEY]?.trim()) {
        // Absent means run, so every invocation written before verbs existed still parses to exactly what it did.
        null,
        "" -> HeadlessVerb.RUN
        else -> HeadlessVerb.bySpelling(raw) ?: return Parsed.UsageError("unknown verb '$raw'; known: ${HeadlessVerb.spellings().joinToString(",")}")
      }

    keySetError(args, verb)?.let {
      return Parsed.UsageError(it)
    }

    // Stated as "anything that is not a run" rather than as the two kinds it is today, so a fourth `Kind` cannot fall through to the run path by
    // being added to the enum and nowhere else, which is the failure `Kind` exists to prevent.
    if (verb.kind != HeadlessVerb.Kind.RUN) return query(args, verb)

    // Every field is read unconditionally and the reader keeps the first complaint, so this reads as
    // the shape of a Request rather than as a chain of early returns.
    val reader = Reader(args)
    val modelUri = reader.coordinate("model")
    val request =
      Request(
        matchQuery = reader.text("match"),
        modelUri = modelUri,
        benchmarkIds = reader.list(if (args.containsKey("benchmark")) "benchmark" else "benchmarks"),
        // Read against the presence of the key rather than of the parsed coordinate, so a `model=` that failed to parse does not also silently change
        // what submission would have meant. The parse error is what gets reported either way.
        submit = reader.flag("submit", default = !args.containsKey("model")),
        prefillBatch = reader.int("batch", default = DEFAULT_PREFILL_BATCH, min = 1),
        nGpuLayers = reader.int("number_gpu_layers", default = DEFAULT_N_GPU_LAYERS, min = 0),
        // A zero deadline would fail every cell before it measured anything, so the floor is 1, as on the host CLI's `--readiness-max-wait-secs`.
        // The ceiling is a day: past that the caller has mistyped rather than asked for a longer wait, and refusing here is how they find out.
        readinessMaxWaitSecs = reader.longOrNull("readiness_max_wait_secs", min = 1, max = MAX_READINESS_WAIT_SECS),
        readinessSkipThermal = reader.flagOrNull("readiness_skip_thermal"),
      )
    return reader.firstError?.let { Parsed.UsageError(it) } ?: Parsed.Ok(request)
  }

  /**
   * The verbs that name one thing, and the key that names it.
   *
   * Absence is a usage error rather than a listing, and the argument is the same on both sides of the read/mutate line. A caller who wrote
   * `verb=benchmarks-show` and forgot the id asked a question about one benchmark, and answering with all of them would not be that question. A
   * caller who wrote `verb=job-rm` and forgot the id named nothing, and the safe reading of a delete that names nothing is to do nothing.
   *
   * The requirement is declared here, as verb-to-key entries, rather than as a `required` flag on [HeadlessVerb.keys]. A key can be required by one
   * verb and optional on another (`benchmark=` is required by `benchmarks-show` and is a run argument on [HeadlessVerb.RUN]), so the property belongs
   * to the pairing and not to either half.
   */
  private val REQUIRED_KEYS =
    mapOf(
      HeadlessVerb.BENCHMARKS_SHOW to "benchmark",
      HeadlessVerb.RESULTS_SHOW to "result",
      HeadlessVerb.RESULTS_DELETE to "result",
      HeadlessVerb.JOB_RM to "id",
      HeadlessVerb.JOB_EXPORT to "id",
      HeadlessVerb.MODELS_DELETE to "model",
      HeadlessVerb.AUTH_IMPORT to "bundle",
    )

  @Suppress("ReturnCount") // the reader's complaint, the two required-argument checks, and the well-formed query
  private fun query(args: Map<String, String>, verb: HeadlessVerb): Parsed {
    val reader = Reader(args)
    val type = reader.text("type")
    val benchmarkId = reader.text("benchmark")
    val resultId = reader.text("result")
    val jobId = reader.text("id")
    val modelUri = reader.coordinate("model")
    val name = reader.text("name")
    val repo = reader.text("repo")
    // `reader.text` refuses a present-but-blank value, which is what makes `bundle=` with an unexpanded shell variable a usage error rather than a
    // bundle that fails to decode. `force` is strictly `0`/`1`, as every flag on this surface is.
    val bundle = reader.text("bundle")
    val force = reader.flag("force", default = false)
    reader.firstError?.let {
      return Parsed.UsageError(it)
    }
    // The placeholder is per key, not a blanket `<id>`. `model=` takes a coordinate URI and `bundle=` a base64url token, and both are values a caller
    // told they need `<id>` could not write.
    REQUIRED_KEYS[verb]?.let { key ->
      if (reader.text(key) == null) return Parsed.UsageError("${verb.spelling} needs $key=${requiredKeyPlaceholder(key)}")
    }
    // Its own rule rather than an entry above, because either key will do and the map holds exactly one required key per verb. Refusing the empty
    // invocation matters more here than for the `-show` verbs: `models-rm` with neither selector reads as "remove models", and the safe reading of
    // an ambiguous delete is to do nothing.
    if (verb == HeadlessVerb.MODELS_RM && name == null && repo == null) {
      return Parsed.UsageError("models-rm needs name=<file> or repo=<org/repo>, or both; it will not delete every model")
    }
    val query =
      Query(
        verb = verb,
        type = type,
        benchmarkId = benchmarkId,
        resultId = resultId,
        jobId = jobId,
        modelUri = modelUri,
        // Trimmed, like every value `Reader` reads: an unclean shell variable can pad the extra, and the untrimmed text would put spaces inside
        // the quoted value of a line this surface documents as `key=value`.
        modelSpelling = args["model"]?.trim(),
        name = name,
        repo = repo,
        bundle = bundle,
        force = force,
      )
    return if (verb.mutates) Parsed.Mutate(query) else Parsed.Inspect(query)
  }

  /** What the caller has to write after a required key, spelled per key so the usage line describes a value they can actually produce. */
  private fun requiredKeyPlaceholder(key: String): String =
    when (key) {
      "model" -> "<coordinate>"
      "bundle" -> "<base64url token from auth-export>"
      else -> "<id>"
    }

  /**
   * What is wrong with the *set* of keys, before any value is read, or null if nothing is.
   *
   * A single ordered `when` rather than a run of early returns, so the precedence between the complaints is the thing you read. It matters: a renamed
   * key is reported as itself rather than swept into the unknown-argument list, which would hand the caller the accepted names and leave them to work
   * out which one replaced theirs.
   *
   * Judged against [verb]'s own keys rather than a union of every verb's, so `verb=version batch=256` is refused instead of accepting a batch that
   * nothing would read. The message names the keys *that verb* takes, which is the list the caller needs.
   */
  private fun keySetError(args: Map<String, String>, verb: HeadlessVerb): String? {
    val allowed = verb.keys + HeadlessVerb.VERB_KEY
    // Sorted so an invocation that trips more than one still produces a deterministic message.
    val refused = args.keys.filter { it in REFUSED_KEYS }.minOrNull()
    val unknown = args.keys.filterNot { it in allowed || it in REFUSED_KEYS }.sorted()
    return when {
      refused != null -> "$refused is not accepted: ${REFUSED_KEYS.getValue(refused)}"
      unknown.isNotEmpty() ->
        "${verb.spelling} does not take: ${unknown.joinToString(",")}; it takes: " + (verb.keys.sorted().joinToString(",").ifEmpty { "no arguments" })
      // `benchmark=` is the CLI's spelling and `benchmarks=` the plural alias, as on iOS. Both at once is refused rather than merged: they would
      // usually disagree, and picking a winner silently is how a script measures a set it did not ask for.
      args.containsKey("benchmark") && args.containsKey("benchmarks") -> "benchmark and benchmarks are the same argument; pass one"
      // Two different questions, so answering both is a contradiction rather than a narrowing: `model=` names bytes to obtain and `match=` filters
      // bytes already present. iOS refuses the pair for the same reason.
      args.containsKey("model") && args.containsKey("match") ->
        "model and match select differently: model names an exact coordinate to fetch, match filters the models already on device; pass one"
      else -> null
    }
  }

  /**
   * Reads one field at a time, remembering the first thing wrong with the invocation.
   *
   * First rather than all: the messages go to a log a script's author reads one at a time, and a caller who fixes the reported problem re-runs and
   * gets the next. Collecting them all would mean deciding how to render a list, for no gain over that loop.
   */
  private class Reader(private val args: Map<String, String>) {
    var firstError: String? = null
      private set

    private fun fail(message: String) {
      if (firstError == null) firstError = message
    }

    /** A present-but-blank value is an error, not an absent one: `-e match ""` reads as an unexpanded shell variable, not "run everything". */
    fun text(key: String): String? {
      val raw = args[key] ?: return null
      return raw.trim().takeIf { it.isNotEmpty() } ?: null.also { fail("$key must not be blank") }
    }

    /**
     * A `<scheme>://key=value` model coordinate, or null when the key is absent.
     *
     * The whole grammar lives in [ModelUri], which mirrors `crates/pipette-cli/src/model_uri.rs`, so the reason a coordinate is refused is the
     * crate's reason and not a second opinion invented here.
     */
    fun coordinate(key: String): Model? {
      val raw = text(key) ?: return null
      return when (val parsed = ModelUri.parse(raw)) {
        is ModelUri.Result.Ok -> parsed.model
        is ModelUri.Result.Error -> null.also { fail("$key is not a coordinate this client understands: ${parsed.failure.reason}") }
      }
    }

    fun list(key: String): List<String> {
      val raw = args[key] ?: return emptyList()
      val items = raw.split(",").map { it.trim() }.filter { it.isNotEmpty() }
      if (items.isEmpty()) fail("$key must name at least one value")
      return items
    }

    /**
     * Strictly `0` / `1`, unlike `pipette_readiness::skip_thermal_from_str`, whose truthy-string grammar exists to read an env var a human exports.
     * This reads a scripted argument, where accepting `submit=yes` while rejecting `submit=maybe` invites a caller to guess. iOS's `submit=0|1` is
     * the same grammar.
     */
    fun flag(key: String, default: Boolean): Boolean {
      val raw = args[key] ?: return default
      return when (raw.trim()) {
        "0" -> false
        "1" -> true
        else -> default.also { fail("$key must be 0 or 1, got '$raw'") }
      }
    }

    fun int(key: String, default: Int, min: Int): Int {
      val raw = args[key] ?: return default
      val value = raw.trim().toIntOrNull()
      return if (value != null && value >= min) value else default.also { fail("$key must be an integer >= $min, got '$raw'") }
    }

    /** [int]'s nullable sibling, for a knob whose "unset" is meaningful rather than a stand-in for a default this file would have to know. */
    fun longOrNull(key: String, min: Long, max: Long): Long? {
      val raw = args[key] ?: return null
      val value = raw.trim().toLongOrNull()
      return if (value != null && value in min..max) value else null.also { fail("$key must be an integer in $min..$max, got '$raw'") }
    }

    /** [flag]'s nullable sibling, for a waiver whose "unset" defers to a device setting rather than to a fixed default. */
    fun flagOrNull(key: String): Boolean? {
      val raw = args[key] ?: return null
      return when (raw.trim()) {
        "0" -> false
        "1" -> true
        else -> null.also { fail("$key must be 0 or 1, got '$raw'") }
      }
    }
  }
}
