package ai.liquid.pipette.headless

import ai.liquid.pipette.BenchmarkDefinition
import ai.liquid.pipette.IdentityBundle
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.ManagementClient
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.RegistrationData
import java.util.Locale

/**
 * The answers to the read-only verbs, as lines a script can read.
 *
 * Kept free of Android types, and taking its inputs as plain values, so every format below is asserted by a unit test rather than by reading logcat
 * on a device. `HeadlessRunActivity` gathers the inputs and emits what this returns.
 *
 * ## The line format, and why it is `key=value`
 *
 * iOS prints prose to a console a person reads. This surface writes to logcat, and what reads logcat is `grep`, `awk`, or a CI step, so every line is
 * a record: a leading kind token, then `key=value` pairs in a fixed order. That survives a log the platform interleaves with other tags, and a
 * consumer can pull one field without knowing the rest.
 *
 * A list always ends with its own `<kind>s count=N` line. The terminal `BENCH_DONE` says the invocation finished, but not whether logcat dropped rows
 * in the middle of it, and logcat does drop rows under load. The count is how a consumer notices.
 *
 * Values are emitted raw. Nothing here can contain a space: a model filename comes from `GgufFilename`'s regex, a benchmark id from `BenchmarkId`, a
 * repo from `HfRepo`, and a job id is a UUID. A quoting scheme would be dead code that looked load-bearing.
 *
 * That invariant is a constraint on what may be added, not just a description of what is here. It is why a result carries `metric=` as a bare number
 * rather than the Results screen's formatted `1.42 GB`. A field that can hold a space needs either a quoting scheme or a place at the end of the
 * line, and both are decisions to make deliberately rather than by adding a key.
 *
 * **[HeadlessVerb.AUTH_ME] is the one exception, and it is deliberate.** Its line carries `organization=` and `client_details=`, both free text a
 * person typed, so a two-word organization puts a space inside a value. What bought the exception is parity: the line is iOS's `auth me` line field
 * for field, in iOS's order, so one runbook and one parser read the profile off either client and off the CLI. A consumer that needs those two fields
 * intact has to read them positionally rather than by splitting on whitespace. That is the trade, and it is confined to this one verb — the invariant
 * above still governs everything else, including the identity verbs beside it, which report only ids and hex.
 *
 * A space is the whole of that exception. A line break is a separate question, because it ends the record rather than splitting a field, and
 * `HeadlessRunActivity.oneLine` flattens one out of every line this surface emits. Word a value here as if it were already one line.
 */
@Suppress("TooManyFunctions") // one formatter per verb; see the note on [report]
object HeadlessInspect {

  /** Everything a verb needs, gathered by the caller so this object touches no Android service. */
  data class Facts(
    val buildVersion: String,
    val packageName: String,
    val engineFlavor: String = "rust",
    val registered: Boolean,
    val jobRunning: Boolean,
    val models: List<ModelFile>,
    val manifests: List<JobManifest>,
    val benchmarks: List<BenchmarkDefinition>,
    /**
     * The ids the New Job picker would offer, as a set.
     *
     * Passed in rather than read from `BenchmarkCatalog.selectable` inside the reporter. That property is process-global state populated by a sync,
     * so reaching for it here would make the output depend on something no caller of this function passed, and every line would have read
     * `selectable=false` under a unit test while reading correctly on a device.
     */
    val selectableIds: Set<String>,
    val modelsDirPath: String,
    val freeBytes: Long,
    /** What [HeadlessVerb.RUNTIMES] reports. A build-time property, so one value rather than a scan. */
    val runtime: Runtime,
    /**
     * Every completed cell whose payload is still on disk, flattened out of [manifests] by the caller.
     *
     * Flattened there rather than here because deciding membership needs the cell's payload file, which is disk the caller already knows how to reach
     * and this object deliberately cannot. A row can still carry a null metric: see [Result.metricValue] for the difference between a measurement
     * that yields no number and a measurement that is gone.
     */
    val results: List<Result>,
    /**
     * The registration record, gathered only for the verbs [HeadlessVerb.readsIdentity] names.
     *
     * Distinct from [registered], which is the same question reduced to a Boolean for `verb=status`. Null here means either that the device holds no
     * registration or that this verb did not ask for one, and only the auth verbs can tell those apart — which is why they are the only ones that
     * read it.
     */
    val registration: RegistrationData? = null,
    /**
     * The private signing key, gathered only for the verbs [HeadlessVerb.readsIdentity] names.
     *
     * Wrapped rather than a bare `String` because this is a `data class`: its generated `toString` is what a failing assertion prints, and the key is
     * the device's whole identity. Only `auth-export` reads [PrivateKeyHex.value], and printing it is the single thing that verb exists to do.
     */
    val privateKeyHex: PrivateKeyHex? = null,
    /**
     * Whether the device holds a signing key it cannot decrypt, gathered only for the verbs [HeadlessVerb.readsIdentity] names.
     *
     * [privateKeyHex] is null for an empty slot and for a Keystore outage alike, and `auth-import` needs the two apart: it refuses over another
     * client's key, and an outage that reads as an empty slot would turn that refusal off. See [Secrets.hasUnreadablePrivateKey].
     */
    val privateKeyUnreadable: Boolean = false,
    /**
     * The stored Hugging Face token, gathered only for the verbs [HeadlessVerb.readsIdentity] names, so `auth-export` can carry it.
     *
     * A credential, and it is here rather than in the bundle's own gathering because the bundle is built by a pure function. It is never printed on
     * its own: what reaches the log is the token count and the base64url bundle that contains it.
     */
    val hfToken: String? = null,
    /** What `GET /clients/me` returned, for [HeadlessVerb.needsNetwork]. Null when the fetch was not attempted or did not succeed. */
    val profile: ManagementClient.ClientProfile? = null,
    /**
     * Why [profile] is null despite the verb asking for it, or null when the fetch was never attempted.
     *
     * Carried as a reason rather than thrown, so a network failure is a refusal this object words like any other rather than an exception the
     * activity reports in its own generic vocabulary.
     */
    val profileFailure: String? = null,
  )

  /**
   * The one runtime this build links.
   *
   * [available] is false when `libpipette_android.so` failed to load, which is the state in which the app advertises no runtime capability at all and
   * matches no runtime-gated job. Reporting the runtime as present with a placeholder build would hide exactly that condition, so the flag is its own
   * field and [build] carries `NativeLib`'s sentinel unchanged.
   */
  data class Runtime(val id: String, val available: Boolean, val build: String)

  /**
   * One completed cell's measurement.
   *
   * [metricValue] and [metricUnit] are the machine-readable half of what the Results screen renders. The display string is deliberately not carried:
   * it holds a formatted size like `1.42 GB`, whose space would break the line format, and a consumer comparing runs wants the number anyway.
   *
   * Null when the cell's benchmark type has no scalar to report, or its payload lacked the field. The row is still emitted: that the cell completed
   * and submitted is worth knowing even when the metric is not derivable.
   *
   * That is not the same as a cell whose payload is *gone*, which `HeadlessFacts` filters out before building any of these. A metric that cannot be
   * derived is a property of the measurement; an absent payload means there is no measurement, which is the state `results-delete` leaves behind.
   */
  data class Result(
    val jobId: String,
    val cellId: String,
    val benchmarkId: String,
    val benchmarkType: String,
    val modelName: String,
    val quant: String?,
    val serverJobId: String?,
    val metricValue: Double?,
    val metricUnit: String?,
    val createdAt: String,
  ) {
    /**
     * [metricValue] as a plain decimal, or null when there is no measurement to report.
     *
     * **Not** `Double.toString()`, which switches to scientific notation above 10^7: a real `max_memory_usage` row rendered as `2.96300544E8` on
     * device. That is still a number to a JSON parser, but this line is read by `grep`, `awk`, and shell arithmetic, none of which accept an
     * exponent, so a fleet check comparing a byte count against a threshold would have failed on exactly the benchmark whose values run largest.
     *
     * Three places matches `pipette-ops`' `{:.3}` and iOS's `%.3f`, so a measurement logged by any of the three clients compares byte for byte. It
     * costs a trailing `.000` on a byte count, which is the right trade against a second spelling of the same number.
     *
     * [Locale.ROOT] is load-bearing, not decoration. `String.format` without it follows the device locale, so a phone set to German would emit
     * `296300544,000` and split the value across the line's own separator. iOS records the same hazard at `BenchmarkMeasurement.format`.
     *
     * Null unless *both* halves are present, so a caller cannot emit a number with no unit naming it.
     */
    val metricText: String?
      get() = if (metricValue != null && metricUnit != null) String.format(Locale.ROOT, "%.3f", metricValue) else null
  }

  /**
   * The lines answering [query], or null when the verb cannot be answered.
   *
   * Two different nulls, and only one of them is a refusal. The `-show` verbs return null for an id that is not here, a benchmark absent from the
   * catalog or a cell with no stored result, which the activity reports as a refusal. [HeadlessVerb.RUN] and the mutating verbs return null because
   * they are not inspections at all and never reach this call.
   *
   * Every list verb answers an empty world with its count line instead, because "none" is a true answer to a listing and not a failure.
   */
  // One arm per verb and no nesting, so the measured complexity is the number of verbs rather than anything a reader has to hold in their head. The
  // alternative that would satisfy the rule is `else -> null`, which costs the exhaustiveness: a verb added to the enum would then compile here and
  // answer nothing, which is the failure this `when` exists to make impossible. The function count this object carries is the same measurement, one
  // formatter per verb, so `TooManyFunctions` is suppressed on the same grounds.
  @Suppress("CyclomaticComplexMethod")
  fun report(query: HeadlessCommand.Query, facts: Facts): List<String>? =
    when (query.verb) {
      HeadlessVerb.VERSION -> listOf("version build=${facts.buildVersion} package=${facts.packageName} flavor=${facts.engineFlavor}")
      HeadlessVerb.STATUS -> listOf(status(facts))
      HeadlessVerb.MODELS -> withCount("model", facts.models.map(::model))
      // Inlined rather than given a formatter of its own: one line, no branching, and the reason for each field is on [Runtime] already.
      HeadlessVerb.RUNTIMES ->
        withCount("runtime", listOf("runtime id=${facts.runtime.id} available=${facts.runtime.available} build=${facts.runtime.build}"))
      HeadlessVerb.JOBS -> withCount("job", facts.manifests.map(::job))
      HeadlessVerb.RESULTS -> withCount("result", facts.results.map(::result))
      HeadlessVerb.RESULTS_SHOW -> facts.results.firstOrNull { it.cellId == query.resultId }?.let { listOf(result(it)) }
      // Neither a run nor a mutation is answered here: `HeadlessCommand` routes `run` to the run path and the mutating verbs to `Parsed.Mutate`, and
      // the activity hands those to `HeadlessMutate`. Listed rather than swept into an `else` so that adding a verb to the enum still fails this
      // `when` until someone has decided which side of the split it is on, which is the whole point of `HeadlessVerb.kind`.
      HeadlessVerb.RUN,
      HeadlessVerb.JOB_RM,
      HeadlessVerb.JOB_EXPORT,
      HeadlessVerb.RESULTS_DELETE,
      HeadlessVerb.MODELS_DELETE,
      HeadlessVerb.MODELS_RM,
      HeadlessVerb.AUTH_IMPORT -> null
      HeadlessVerb.AUTH_WHOAMI -> whoami(facts)?.let { listOf(it) }
      HeadlessVerb.AUTH_ME -> facts.profile?.let { listOf(profile(it)) }
      HeadlessVerb.AUTH_EXPORT -> export(facts)
      HeadlessVerb.BENCHMARKS -> benchmarks(query.type, facts)
      HeadlessVerb.BENCHMARKS_SHOW ->
        facts.benchmarks.firstOrNull { it.benchmarkId.toString() == query.benchmarkId }?.let { listOf(benchmark(it, facts.selectableIds)) }
      HeadlessVerb.STORAGE_STATUS -> listOf(storage(facts))
    }

  /**
   * Why [report] had no answer for [query], as the reason logged after `REFUSED`.
   *
   * Lives here rather than at the call site because it is wording, and because it is the half of a `-show` verb a caller most needs to be right: the
   * refusal has to say where the ids that *do* exist can be found, or a script author is left guessing whether they mistyped an id or asked a device
   * that has none. Only the verbs named below can reach it, so any other verb gets the generic form rather than a wrong specific one.
   *
   * Takes [facts] as well as the query, because the auth verbs' refusal is about the device rather than about what the caller named: which half of
   * the identity is missing is the whole content of the answer.
   */
  fun notFoundReason(query: HeadlessCommand.Query, facts: Facts): String =
    when (query.verb) {
      HeadlessVerb.RESULTS_SHOW ->
        "no result for cell '${query.resultId}'; a cell has one only once it has completed and written a payload, so list them with verb=results"
      HeadlessVerb.BENCHMARKS_SHOW -> "no benchmark with id '${query.benchmarkId}'; sync the catalog, or list them with verb=benchmarks"
      HeadlessVerb.AUTH_WHOAMI,
      HeadlessVerb.AUTH_EXPORT -> identityReason(facts)
      // The fetch failure wins when there was one, because by then both halves were present: the identity reason would describe a device that is
      // fine.
      HeadlessVerb.AUTH_ME -> facts.profileFailure ?: identityReason(facts)
      else -> "${query.verb.spelling} had no answer"
    }

  /**
   * Which half of the identity is missing, or why the key that is there cannot be used.
   *
   * Registration first, then the key, which is the order iOS's `mgmtSession` composes them in. It is also the more useful order: a device that never
   * registered is the ordinary case and has a one-step fix, while a registration with no key is a fault worth naming as one, since the device reports
   * itself registered and cannot sign anything.
   */
  private fun identityReason(facts: Facts): String =
    when {
      facts.registration == null -> "this device holds no registration; register it from Settings, and there is no client_id until it does"
      facts.privateKeyHex == null ->
        "this device holds a registration for client_id=${facts.registration.clientId} but no signing key, so it can neither sign nor be exported"
      else -> "the stored signing key does not yield a public key, so there is nothing this device could be approved for"
    }

  /**
   * `auth-whoami`: the two fields an operator needs to approve this client, and nothing else.
   *
   * Null when either half of the identity is missing, as `auth-export` is, and for the same reason: an id whose key cannot sign names a client nobody
   * should approve. `public_key` is derived from the private key rather than stored, so it is a value that can be checked against the server's record
   * without trusting anything this device persisted. The key itself is not printed, which is the whole difference between this verb and
   * `auth-export`.
   */
  @Suppress("ReturnCount") // one per half of the identity that can be absent, then the answer; a `let` chain would hide which half is which
  private fun whoami(facts: Facts): String? {
    val registration = facts.registration ?: return null
    val publicKey = facts.privateKeyHex?.let(IdentityBundle::publicKeyHexOf) ?: return null
    return "auth-whoami client_id=${registration.clientId} public_key=$publicKey"
  }

  /**
   * `auth-me`: the profile the server holds, field for field with iOS's `auth me` line and in iOS's order.
   *
   * `tags=` and `capabilities=` are emitted **empty rather than omitted**, which is the other rule this verb breaks (see the class KDoc for the one
   * about spaces). Both are iOS's choice and the reason is the same: a parser reading the profile off either client sees the key either way, so "no
   * tags" and "a field this build does not report" stay distinguishable. Elsewhere on this surface an empty key is dropped, because there the absent
   * key *is* the signal.
   */
  private fun profile(record: ManagementClient.ClientProfile): String =
    "auth-me client_id=${record.clientId} organization=${record.organization} client_details=${record.clientDetails} " +
      "contact_email=${record.contactEmail} status=${record.status} tags=${record.tags.joinToString(",")} " +
      "reindex_pending=${record.reindexPending} capabilities=${record.capabilities.joinToString(",")}"

  /**
   * `auth-export`: a summary line, then the bundle alone.
   *
   * Two lines, as iOS prints them, and the split is what makes the verb usable: the first is for eyeball confirmation against `auth-whoami`, and the
   * second holds nothing but the token, so an operator lifts it with one `sed` and only that line carries a key. It is also what lets the summary go
   * to the audit log while the bundle does not — see `HeadlessRunActivity.logAuthEvent`.
   *
   * `device_labels=no settings=no` are constants here rather than dead fields: this client holds neither, and iOS prints both, so keeping them means
   * one parser reads either client's export. `hf_tokens=` is what this client can restore, which is its one Hugging Face token or none.
   *
   * Null when either half of the identity is missing, or when the stored key cannot produce a public key. Derived before the bundle is printed for
   * iOS's reason: a key that cannot produce a public key cannot sign either, and an export nothing can restore is worth refusing.
   */
  @Suppress("ReturnCount") // one per precondition an export cannot proceed without, then the two lines; see [identityReason] for what each becomes
  private fun export(facts: Facts): List<String>? {
    val registration = facts.registration ?: return null
    val privateKeyHex = facts.privateKeyHex ?: return null
    val publicKey = IdentityBundle.publicKeyHexOf(privateKeyHex) ?: return null
    val bundle = IdentityBundle(privateKeyHex = privateKeyHex, registration = registration, userToken = facts.hfToken)
    val token = bundle.encode()
    return listOf(
      "auth-export client_id=${registration.clientId} server_url=${registration.serverUrl} public_key=$publicKey " +
        "device_labels=no settings=no hf_tokens=${bundle.tokenCount} bytes=${token.length}",
      "auth-export bundle=$token",
    )
  }

  /**
   * What a script checks before deciding whether to start a run.
   *
   * `running` matters most: the surface refuses a second job, so a caller that polls this can wait rather than fail. `benchmarks` is the *synced
   * catalog* count, which is separately worth knowing, because a device with an empty catalog refuses every run and the reason reads like a benchmark
   * id typo.
   */
  private fun status(facts: Facts): String =
    "status registered=${facts.registered} running=${facts.jobRunning} models=${facts.models.size} " +
      "jobs=${facts.manifests.size} benchmarks=${facts.benchmarks.size}"

  /**
   * One model on disk.
   *
   * `quant` and `repo` are absent rather than empty for an imported file that has neither, because `repo=` with nothing after it reads as a repo
   * whose name is the empty string. A consumer tests for the key.
   */
  private fun model(file: ModelFile): String = buildString {
    append("model name=${file.name} bytes=${file.sizeBytes}")
    file.hfRepo?.let { append(" repo=$it") }
    file.quant?.let { append(" quant=$it") }
    if (file.isMmproj) append(" mmproj=true")
  }

  /**
   * One job manifest. `cells` and the per-status counts are what tells a script whether a previous run finished or stalled.
   *
   * Statuses are rendered through `wire`, not the enum name, so a line here reads in the same vocabulary as the manifest on disk and the warehouse.
   */
  private fun job(manifest: JobManifest): String {
    val byStatus = manifest.cells.groupingBy { it.runStatus }.eachCount()
    val counts = byStatus.keys.sorted().joinToString(",") { "${it.wire}=${byStatus[it]}" }
    return buildString {
      append("job id=${manifest.jobId} status=${manifest.status.wire} cells=${manifest.cells.size}")
      if (counts.isNotEmpty()) append(" cell_status=$counts")
      append(" submit=${manifest.contributeResults == true}")
      append(" created=${manifest.createdAt}")
    }
  }

  /**
   * One completed cell's measurement.
   *
   * `submitted` is derived from the server job id rather than carried separately, so the line cannot claim a row reached the warehouse while naming
   * no row. The id itself follows for a consumer that wants to look the row up.
   *
   * `metric` and `unit` are omitted together when the payload had no scalar for this benchmark type, on the same grounds as `model`'s absent `repo`:
   * `metric=` with nothing after it reads as a measurement of zero.
   */
  private fun result(result: Result): String = buildString {
    append("result cell=${result.cellId} job=${result.jobId} benchmark=${result.benchmarkId} type=${result.benchmarkType}")
    append(" model=${result.modelName}")
    result.quant?.let { append(" quant=$it") }
    result.metricText?.let { append(" metric=$it unit=${result.metricUnit}") }
    append(" submitted=${!result.serverJobId.isNullOrBlank()}")
    result.serverJobId?.takeIf { it.isNotBlank() }?.let { append(" server_job=$it") }
    append(" created=${result.createdAt}")
  }

  /**
   * The catalog, optionally narrowed.
   *
   * An unknown `type=` returns an empty list plus its count line rather than a refusal, and the difference from `benchmarks-show` is deliberate: a
   * filter that matches nothing is a true answer to "which benchmarks are of this type", while an id that matches nothing means the caller named
   * something that does not exist.
   */
  private fun benchmarks(type: String?, facts: Facts): List<String> {
    val matching = if (type == null) facts.benchmarks else facts.benchmarks.filter { it.benchmarkType.equals(type, ignoreCase = true) }
    return withCount("benchmark", matching.map { benchmark(it, facts.selectableIds) })
  }

  private fun benchmark(definition: BenchmarkDefinition, selectableIds: Set<String>): String =
    "benchmark id=${definition.benchmarkId} type=${definition.benchmarkType} selectable=${definition.benchmarkId.toString() in selectableIds}"

  /**
   * Where models live and what room is left.
   *
   * No quota line, because Android has none: the store is uncapped and nothing evicts (PIP-492 gap 2). Reporting `quota=unset` would be honest but
   * invites a consumer to treat the field as present-and-unlimited; leaving it out means a script written against iOS's `storage status` fails to
   * find it, which is the accurate signal.
   */
  private fun storage(facts: Facts): String =
    "storage models_dir=${facts.modelsDirPath} models=${facts.models.size} " +
      "bytes=${facts.models.sumOf { it.sizeBytes }} free_bytes=${facts.freeBytes}"

  /** A list plus the count line that lets a consumer notice logcat dropped part of it. */
  private fun withCount(kind: String, lines: List<String>): List<String> = lines + "${kind}s count=${lines.size}"
}
