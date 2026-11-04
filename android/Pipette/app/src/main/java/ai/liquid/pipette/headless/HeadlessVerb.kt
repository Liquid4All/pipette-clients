package ai.liquid.pipette.headless

/**
 * The verbs this surface accepts, and the arguments each one takes.
 *
 * iOS spells its verbs positionally (`headlessrun models`, `headlessrun benchmarks show benchmark=…`). An `am start` line carries only `-e key value`
 * extras, so the verb arrives as `verb=` and a two-word iOS verb becomes one hyphenated token: `benchmarks show` is `verb=benchmarks-show`. The
 * [spelling] of each is the iOS name with a hyphen where iOS has a space, so a runbook translates by eye.
 *
 * **Absent `verb=` means [RUN].** Every invocation written against the surface before verbs existed keeps working unchanged, which matters because
 * the one thing this surface is for is scripts nobody is watching.
 *
 * Each verb declares its own [keys], and `HeadlessCommand` validates against that set rather than a union. A union would accept `verb=version
 * batch=256` and silently ignore the batch, which is the class of quiet misreading the whole grammar exists to refuse: a caller who passes an
 * argument to a verb that cannot use it has misunderstood something, and saying so costs one line.
 *
 * Each verb declares its [kind] rather than having it inferred from the name, and the parameter carries no default, so adding one does not compile
 * until someone has answered whether it destroys anything. The destructive verbs were deliberately absent at first: this component is exported, and a
 * verb that deletes data wanted its own decision rather than arriving as a side effect of adding a listing. That decision has been taken for the
 * local ones, on the grounds that the caller gate is `WRITE_SECURE_SETTINGS`, held by adb's shell uid and by privileged system apps, and anyone with
 * adb can already `pm uninstall` this app and take the same data with it. The gate is the real boundary and a second one here would be ceremony. Not
 * airtight, and `HeadlessCaller` says where it leaks: the permission also carries `development`, so an app granted it over adb reaches these verbs.
 *
 * One verb reaches the network, and it reads: [AUTH_ME] asks the server what profile it holds for this device. The rest of the network family stays
 * absent, because submitting and registering *publish* rather than read, and that is the pass they want. Reaching the network from a synchronous
 * `onCreate` is what [needsNetwork] marks and `HeadlessFacts` pays for.
 */
enum class HeadlessVerb(val spelling: String, val keys: Set<String>, val kind: Kind) {
  /** Start a benchmark run. The original and only behaviour of this surface, and what an invocation with no `verb=` still means. */
  RUN(
    "run",
    setOf("match", "model", "benchmark", "benchmarks", "submit", "batch", "number_gpu_layers", "readiness_max_wait_secs", "readiness_skip_thermal"),
    Kind.RUN,
  ),

  /** The build identity a result would be submitted under. */
  VERSION("version", emptySet(), Kind.READ),

  /** Registration, catalog and model counts, and whether a job is running: what a script checks before deciding to start one. */
  STATUS("status", emptySet(), Kind.READ),

  /** Every model on disk. Not the offerable catalog: this answers "what is here", which is what a storage or provisioning check needs. */
  MODELS("models", emptySet(), Kind.READ),

  /**
   * The inference runtimes this build can run.
   *
   * Always exactly one. llama.cpp is compiled into the APK as `libpipette_android.so` rather than downloaded, so this reports a build-time property
   * and not an on-disk inventory (`Capabilities` says the same thing to the server). iOS lists up to three, since it offers MLX and Apple Foundation
   * alongside llama.cpp (`RuntimeCommands.list`), and the CLI lists whatever is installed. A script that reads a runtime build id off a device
   * therefore gets one line here and several there, which is why the count line is emitted for a list of one.
   */
  RUNTIMES("runtimes", emptySet(), Kind.READ),

  /** Every job manifest, newest first. */
  JOBS("jobs", emptySet(), Kind.READ),

  /**
   * Every completed cell whose payload is still on disk, across every job.
   *
   * A *result* is not a job: one job holds a cell per model × benchmark pair, and only the cells that finished have a payload to report. Listing jobs
   * tells a script whether a run finished; listing results tells it what the run measured and whether the rows reached the warehouse.
   *
   * On disk, not parseable: a payload this client cannot read still lists, with no `metric=`. See `HeadlessFacts.isReportable` for why the rule is
   * existence, and [RESULTS_DELETE] for what depends on it.
   */
  RESULTS("results", emptySet(), Kind.READ),

  /** One result by cell id, with the fields a spot-check wants. A cell id is a UUID and unique across jobs, so it needs no job to disambiguate it. */
  RESULTS_SHOW("results-show", setOf("result"), Kind.READ),

  /** The synced benchmark catalog, optionally narrowed to one `type=`. */
  BENCHMARKS("benchmarks", setOf("type"), Kind.READ),

  /** One benchmark by id, or a refusal. The id-exists check a script wants before naming it in a run. */
  BENCHMARKS_SHOW("benchmarks-show", setOf("benchmark"), Kind.READ),

  /** Where models live, how much they occupy, and how much room is left. */
  STORAGE_STATUS("storage-status", emptySet(), Kind.READ),

  /**
   * This device's `client_id` and public key, read off local storage.
   *
   * **No iOS counterpart.** iOS folds the same two fields into `auth export`'s first line, which also prints the signing key. Approving a client on
   * the management server needs its id, and reading an id must not require printing a key, so this is the verb an operator reaches for. It answers
   * with the network down and the device unapproved, which is exactly the state a device is in while someone is looking its id up.
   */
  AUTH_WHOAMI("auth-whoami", emptySet(), Kind.READ),

  /**
   * The profile the management server holds for this device, live from `GET /clients/me`.
   *
   * The only verb on this surface that reaches the network, and the only one whose answer can change while the device sits still: [STATUS] reports
   * the `status` this device recorded at registration, and this reports the one the server holds now, which is how an operator confirms an approval
   * landed.
   */
  AUTH_ME("auth-me", emptySet(), Kind.READ),

  /**
   * Print everything needed to restore this client on another device, as one base64url bundle.
   *
   * A READ verb that **prints a secret**: the bundle carries the Ed25519 signing key in plaintext, because the Keystore exports it no other way. That
   * is the point rather than an oversight — see the class KDoc on `IdentityBundle` for what an uninstall costs without it — and it is why the caller
   * gate matters here more than for any other verb on the surface.
   */
  AUTH_EXPORT("auth-export", emptySet(), Kind.READ),

  /**
   * Delete a job and everything under it: its manifest, its cells' payloads, its results.
   *
   * `id=` rather than `job=`, matching iOS, which spells this argument `--id` too (`job rm --id`). The id is copied from a listing either way, though
   * the spelling there differs by verb: [JOBS] prints `job id=<id>` as the record's own identifier, while [RESULTS] prints `job=<id>` as one field of
   * a result.
   */
  JOB_RM("job-rm", setOf("id"), Kind.MUTATE),

  /**
   * Write a job's cells to a CSV beside the model store, and print the path.
   *
   * Every cell, not only the completed ones, despite what `CompletedResultsCsvExporter`'s name suggests: it emits a row per cell with a `status`
   * column, so a pending or failed cell is in the file too. That is the more useful export of a partial job, and it is what the Jobs screen's own
   * export button produces, so the two cannot disagree.
   *
   * A mutation because it writes a file, not because it destroys one. Grouped with the destructive verbs rather than the listings for that reason: a
   * reader deciding whether a verb is safe to run twice wants writing and reading told apart, and this one leaves something behind.
   */
  JOB_EXPORT("job-export", setOf("id"), Kind.MUTATE),

  /** Delete one completed cell's stored result, by the cell id [RESULTS] and [RESULTS_SHOW] print. The job and its other cells are untouched. */
  RESULTS_DELETE("results-delete", setOf("result"), Kind.MUTATE),

  /**
   * Delete the weights a coordinate names.
   *
   * Takes the same `model=` URI grammar [RUN] takes, which is iOS's choice too (`models delete --model`), so a coordinate that can be run can be
   * deleted without re-spelling it. `spec=` stays absent for the reason it is absent on [RUN]: no decoder for the plan-types tagged shape.
   */
  MODELS_DELETE("models-delete", setOf("model"), Kind.MUTATE),

  /**
   * Delete weights by file name, by repo, or by both together.
   *
   * The by-name counterpart to [MODELS_DELETE], taking iOS's `models rm --name --repo`. It exists because not every file on disk has a coordinate:
   * the Models tab accepts an imported file the preset catalog has never heard of, and `name=` is the only handle such a file has.
   *
   * One deliberate difference from iOS when both selectors are given: see `HeadlessMutate.deleteByName`.
   */
  MODELS_RM("models-rm", setOf("name", "repo"), Kind.MUTATE),

  /**
   * Restore an exported identity onto this device.
   *
   * A mutation that writes rather than destroys, like [JOB_EXPORT] — except that what it writes is the device's identity, so it is the one verb here
   * that can cost data it never touches: overwriting a signing key orphans every result already submitted under it. That is what `force=` gates, and
   * what makes the refusal unforced. A key equal to the bundle's is no conflict, so a rerun after a half-finished import finishes the restore rather
   * than needing `force=1`. iOS draws both lines in the same places (`AuthCommands.import`).
   */
  AUTH_IMPORT("auth-import", setOf("bundle", "force"), Kind.MUTATE);

  /** What a verb does to the device, declared rather than inferred. */
  enum class Kind {
    /** Starts a benchmark job. */
    RUN,

    /** Answers a question and changes nothing. */
    READ,

    /** Deletes something, or writes something. */
    MUTATE,
  }

  /** Whether this verb changes the device. */
  val mutates: Boolean
    get() = kind == Kind.MUTATE

  /**
   * Whether answering this verb needs the completed cells' payloads read off disk.
   *
   * Gathering those costs a file read per completed cell across every job on the device, so it is done only for the verbs that need them. Reporting
   * is not the only reason to need them: [RESULTS_DELETE] prints no result and is here because the walk is what maps a cell id back to its job.
   * Declared on the verb rather than tested at the call site so the cost stays attached to what incurs it.
   */
  val readsResults: Boolean
    get() = this == RESULTS || this == RESULTS_SHOW || this == RESULTS_DELETE

  /**
   * Whether answering this verb needs the stored signing identity: the registration record and the private key.
   *
   * Declared on the verb, like [readsResults], so the cost stays attached to what incurs it — and here the cost is not a file read but an exposure.
   * The key rides in `HeadlessInspect.Facts` for these verbs and no others, which is why `PrivateKeyHex` wraps it. [AUTH_ME] is included because it
   * signs a request rather than because it prints anything: gathering both halves is what lets the refusal name the one that is missing, as iOS's
   * `mgmtSession` does.
   */
  val readsIdentity: Boolean
    get() = this == AUTH_WHOAMI || this == AUTH_ME || this == AUTH_EXPORT || this == AUTH_IMPORT

  /**
   * Whether answering this verb requires a management-server round trip.
   *
   * Only [AUTH_ME]. Everything else on this surface answers from the device, which is what lets `HeadlessRunActivity.onCreate` stay synchronous;
   * `HeadlessFacts` pays for this one by blocking the main thread on an IO dispatcher rather than by making the whole surface asynchronous.
   */
  val needsNetwork: Boolean
    get() = this == AUTH_ME

  companion object {
    /** The extra that names the verb. Not itself a verb argument, so every verb's key check has to allow it. */
    const val VERB_KEY = "verb"

    fun bySpelling(spelling: String): HeadlessVerb? = entries.firstOrNull { it.spelling == spelling }

    /** Every spelling, for the message an unknown verb gets. */
    fun spellings(): List<String> = entries.map { it.spelling }.sorted()
  }
}
