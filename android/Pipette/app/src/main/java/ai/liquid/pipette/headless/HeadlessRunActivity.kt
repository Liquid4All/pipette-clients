package ai.liquid.pipette.headless

import ai.liquid.pipette.ModelCatalog
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.RunReadiness
import android.app.Activity
import android.os.Bundle
import android.util.Log

/**
 * Intent-driven benchmark entry point (PIP-437), the Android counterpart to iOS's `HeadlessRunner`. Starts a run through the same
 * `JobController.startNewJob` path a New Job screen tap uses, so a headless run and a hand-driven one exercise identical code.
 *
 * ```
 * # select from what is already on device
 * adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity \
 *   -e match LFM2.5-350M -e benchmark prefill_throughput_512,decode_throughput_512_100 -e submit 0
 *
 * # name an exact coordinate, fetched first if it is not here yet (submit=1 because a coordinate defaults to opt-in)
 * adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity \
 *   -e model "'gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf'" \
 *   -e benchmark prefill_throughput_100 -e submit 1
 *
 * # read this device's client_id, with no network and no key printed
 * adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb auth-whoami
 *
 * # carry the identity to a replacement device: export on A, import the printed token on B
 * adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb auth-export
 * adb shell am start -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb auth-import -e bundle "'<token>'"
 *
 * adb logcat -s pipette-headless
 * adb logcat -s pipette-auth        # the identity key events alone
 * ```
 *
 * **Quote the coordinate twice.** `adb shell` joins its arguments into one string and hands it to the *device's* shell, so one layer of quotes is
 * removed locally and the `&` separating the URI's pairs then backgrounds the command. The rest of the URI is lost and the run is refused for a
 * missing key that was in fact supplied. The nested `"'…'"` above survives both shells; `\&` also works. A bundle needs no quoting of its own — it is
 * base64url, so it holds no metacharacter — and it is quoted above because that is the habit the runbook teaches for every value on this surface.
 *
 * **Not a UI.** It finishes immediately, having either started a job or refused; progress goes to logcat under [TAG] and to the normal Jobs screen.
 * The run outlives this activity because [ai.liquid.pipette.JobController] is app-scoped, which is the same property that lets a UI-started run
 * survive rotation.
 *
 * The refusals come in two layers, each logged with a reason and reported through the activity result code. Some apply to every invocation: a surface
 * whose caller guard is missing ([HeadlessCaller]), a malformed invocation ([HeadlessCommand]), a process with no app container, and anything that
 * throws on the way through, which `onCreate` catches. The rest belong to what was asked for. A *run* adds two, mirroring what iOS hardened in #923
 * and #999 (PIP-428): a submitting run on an unregistered device, and a run while another job is going. Submitting is the default, so the
 * registration check is the one most likely to fire; `submit=0` records locally instead.
 *
 * A *mutation* adds its own, which are [HeadlessMutate]'s to make: a target that is not here, and, for every one that destroys something, a job in
 * flight. `job-export` is outside that second one, since it only reads what a run is appending to. `auth-import` adds one more of its own: it refuses
 * over a signing key that is not the bundle's unless `force=1`, because that overwrite orphans results this device already submitted. A missing
 * registration gates none of them, because a delete neither starts a job nor sends anything anywhere.
 *
 * An *unauthorized* caller never reaches this code at all: the manifest guards the component with [HeadlessCaller.REQUIRED_CALLER_PERMISSION] and the
 * framework refuses the launch with a `SecurityException` before the activity is created.
 *
 * `[HEADLESS] …` lines and the terminal [DONE_SENTINEL] mirror iOS's console contract: a consumer keyed on the sentinel would hang without it, so
 * every terminal path emits one, including the early refusals.
 */
class HeadlessRunActivity : Activity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // Caught here rather than only around the act, because the sentinel below is the contract and everything above it can throw: gathering the world
    // walks the job store and stats a payload per completed cell, and resolution reads what that walk returned. A consumer keyed on the sentinel
    // waits forever for a line an escaping throw would never print, so the guarantee has to sit at the one place every path passes through.
    val exit = runCatching { runHeadless() }.getOrElse { EXIT_FAILURE.also { _ -> log("FAILED ${HeadlessAct.reason(it)}") } }
    log(doneLine(exit))
    // The logged `exit=` and the Activity result code are deliberately different numbers. The log carries iOS's 0/1/2, which is the contract a CI
    // consumer greps for; the result code has to live in Android's space, where 0 is RESULT_CANCELED and -1 is RESULT_OK. Setting the exit code
    // directly made a successful run report RESULT_CANCELED to anything using the Activity Result APIs.
    setResult(activityResultFor(exit))
    finish()
  }

  @Suppress("ReturnCount") // one return per refusal, each already logged; a single exit point would only re-derive which arm fired
  private fun runHeadless(): Int {
    // The caller was already vetted by the framework against the manifest's android:permission; this
    // only confirms that gate still exists. See HeadlessCaller for why the check cannot be a uid test.
    when (val verdict = HeadlessCaller.check(declaredCallerPermission())) {
      is HeadlessCaller.Verdict.Refused -> {
        log("REFUSED caller gate: ${verdict.reason}")
        return EXIT_FAILURE
      }
      is HeadlessCaller.Verdict.Allowed -> log("caller holds ${HeadlessCaller.REQUIRED_CALLER_PERMISSION}")
    }

    val parsed = HeadlessCommand.parse(stringExtras())
    if (parsed is HeadlessCommand.Parsed.UsageError) {
      log("USAGE ${parsed.message}")
      return EXIT_USAGE
    }

    // containerOrNull, not container: this activity is exported, so it is reachable in a process
    // where the container was never built, and a lateinit read would throw rather than report.
    val container = (application as? PipetteApp)?.containerOrNull
    if (container == null) {
      log("REFUSED app container unavailable in this process")
      return EXIT_FAILURE
    }

    // Answered before every check below it, because none of them applies: an inspection starts nothing, so a running job or a missing registration is
    // information it should be able to report rather than a reason to refuse it.
    if (parsed is HeadlessCommand.Parsed.Inspect) return inspect(container, parsed.query)
    // Alongside the inspections and above the run checks for the same reason: a delete neither starts a job nor submits anything, so a missing
    // registration is no reason to refuse one. A *running* job is, and that refusal is HeadlessMutate's to make, since it can say which job.
    if (parsed is HeadlessCommand.Parsed.Mutate) return mutate(container, parsed.query)

    val request = (parsed as HeadlessCommand.Parsed.Ok).request

    // Checked before resolving anything else so a CI run fails on the actionable cause rather than
    // on a downstream symptom. Only when submitting: `submit=0` is a legitimate unregistered use.
    if (request.submit && !container.storage.isRegistered()) {
      log("REFUSED device is not registered; register it, or pass submit=0 to record locally only")
      return EXIT_FAILURE
    }

    if (container.jobController.isRunning()) {
      log("REFUSED a job is already running")
      return EXIT_FAILURE
    }

    return startJob(container, request)
  }

  /** The second half: resolve the request against what is actually on device, then hand it to the same entry point the New Job screen calls. */
  @Suppress("ReturnCount") // as above: each arm is its own already-logged refusal
  private fun startJob(container: ai.liquid.pipette.AppContainer, request: HeadlessCommand.Request): Int {
    val allModels = container.storage.availableModels()
    val coordinate = request.modelUri?.let { ModelCatalog.resolveCoordinate(it, allModels) }
    val models = coordinate?.let { listOf(it.weights) } ?: HeadlessRunPlan.selectModels(allModels, request.matchQuery)
    if (models.isEmpty()) {
      val detail = request.matchQuery?.let { "no model on device matches '$it'" } ?: "no models on device"
      log("REFUSED $detail")
      return EXIT_FAILURE
    }

    val benchmarks =
      when (val resolved = HeadlessRunPlan.selectBenchmarks(request.benchmarkIds)) {
        is HeadlessRunPlan.BenchmarkSelection.Unknown -> {
          log("USAGE unknown benchmark id(s): ${resolved.ids.joinToString(",")}")
          return EXIT_USAGE
        }
        is HeadlessRunPlan.BenchmarkSelection.Empty -> {
          log("REFUSED benchmark catalog is empty; sync the catalog before running headless")
          return EXIT_FAILURE
        }
        is HeadlessRunPlan.BenchmarkSelection.Resolved -> resolved.definitions
      }

    // A coordinate brings its own projector or none. Only a `match=` run offers every projector on device, which is what the New Job screen passes
    // when no pairing was made explicitly; naming a coordinate *is* an explicit pairing, so pulling in unrelated projectors would contradict it.
    val mmprojFiles = coordinate?.projector?.let { listOf(it) } ?: allModels.filter { it.isMmproj }
    return runCatching {
        val jobId =
          container.jobController.startNewJob(
            models = models,
            mmprojFiles = mmprojFiles,
            benchmarks = benchmarks,
            // Whatever the line above settled on: the coordinate's own projector, or every one on device for a `match=` run. A text-only model
            // ignores them either way, since only a VL cell reads a projector.
            selectedMmprojPaths = mmprojFiles.map { it.path }.toSet(),
            nGpuLayers = request.nGpuLayers,
            // Not caller-settable; the measurement's context comes from the benchmark. See HeadlessCommand's CTX_REFUSAL.
            contextSize = HeadlessCommand.MANIFEST_CONTEXT_SIZE,
            prefillBatch = request.prefillBatch,
            contributeResults = request.submit,
            readiness = RunReadiness(maxWaitSecs = request.readinessMaxWaitSecs, skipThermal = request.readinessSkipThermal),
          )
        log(
          "STARTED job=$jobId models=${models.size} benchmarks=${benchmarks.size} submit=${request.submit} " +
            "number_gpu_layers=${request.nGpuLayers} batch=${request.prefillBatch}"
        )
        log("models: " + models.joinToString(",") { it.name })
        // Says plainly that bytes will be fetched before the first measurement, so a log reader is not left wondering why a run sat still for
        // minutes.
        val pending = (models + listOfNotNull(coordinate?.projector)).filterNot { it.onDisk }
        if (pending.isNotEmpty()) log("will fetch first: " + pending.joinToString(",") { it.name })
        EXIT_OK
      }
      .getOrElse {
        log("FAILED could not start job: ${HeadlessAct.reason(it)}")
        EXIT_FAILURE
      }
  }

  /**
   * Answer a read-only verb.
   *
   * The gathering lives in [HeadlessFacts] and the wording in [HeadlessInspect], so what remains here is routing: hand the query the world it asks
   * about, then emit what came back or refuse.
   */
  private fun inspect(container: ai.liquid.pipette.AppContainer, query: HeadlessCommand.Query): Int {
    val facts = HeadlessFacts.gather(query, container, packageName, ai.liquid.pipette.BuildConfig.BUILD_VERSION)
    // A `-show` verb for an id that is not here is a refusal rather than silence, so a script that asked whether one exists gets an answer it can
    // branch on. The reason is HeadlessInspect's to word, for the same reason the answers are.
    val lines = HeadlessInspect.report(query, facts) ?: return EXIT_FAILURE.also { log("REFUSED ${HeadlessInspect.notFoundReason(query, facts)}") }
    lines.forEach { log(it) }
    logAuthEvent(query.verb, lines)
    return EXIT_OK
  }

  /**
   * Carry out a mutating verb.
   *
   * The same shape as [inspect], and for the same reason: [HeadlessFacts] gathers, [HeadlessMutate] decides, [HeadlessAct] does, and what is left
   * here is routing. Every line it returns says what happened in the past tense, naming what it touched, because that is the only record the thing
   * happened at all.
   */
  private fun mutate(container: ai.liquid.pipette.AppContainer, query: HeadlessCommand.Query): Int {
    val facts = HeadlessFacts.gather(query, container, packageName, ai.liquid.pipette.BuildConfig.BUILD_VERSION)
    val outcome = HeadlessMutate.resolve(query, facts)
    val report = HeadlessAct.perform(outcome, container.storage, container.secrets, query.verb.spelling, getExternalFilesDir(null))
    report.lines.forEach { log(it) }
    logAuthEvent(query.verb, report.lines)
    return report.exit
  }

  /**
   * Mirror an identity key event into its own log stream.
   *
   * The custody verbs record a durable change to who this device *is*, which is worth reconstructing from the log months later, and [TAG] is the
   * wrong place for it: that stream is one invocation's transcript, and a fleet script filtering it discards everything else. So the event goes to
   * [AUTH_TAG] as well, matching iOS, which writes these two events to `AppLog.auth` on top of the console.
   *
   * **Only the first line**, and that is load-bearing for `auth-export`: its first line is the summary and its second is the bundle, so the key
   * reaches one stream and not two. `HeadlessInspectTest` pins that order. For `auth-import` the first line is the whole event, whether it reads
   * `IMPORTED`, `REFUSED` or `FAILED` — a refused import is as much worth recording as a performed one, since the refusal is what stood between a
   * live identity and an overwrite.
   *
   * `auth-whoami` and `auth-me` are absent: they read and change nothing, so there is nothing to reconstruct.
   */
  private fun logAuthEvent(verb: HeadlessVerb, lines: List<String>) {
    if (verb != HeadlessVerb.AUTH_EXPORT && verb != HeadlessVerb.AUTH_IMPORT) return
    lines.firstOrNull()?.let { Log.i(AUTH_TAG, oneLine(it)) }
  }

  /**
   * The permission the platform has on record as guarding this component, or null if it declares none or the lookup failed. A failed lookup reads as
   * "unguarded" on purpose: [HeadlessCaller.check] then refuses, which is the safe answer to not knowing.
   */
  @Suppress("DEPRECATION") // the ComponentInfoFlags overload is API 33+; minSdk here is 31
  private fun declaredCallerPermission(): String? = runCatching { packageManager.getActivityInfo(componentName, 0).permission }.getOrNull()

  private fun stringExtras(): Map<String, String> {
    val extras = intent?.extras ?: return emptyMap()
    // `am start -e` only ever produces strings, so a non-string extra means a programmatic caller
    // built the Intent by hand. Reading it via getString would silently drop it; rendering it keeps
    // the value in the grammar's hands, where an unparseable one becomes a usage error.
    return extras.keySet().mapNotNull { key -> stringExtra(extras, key)?.let { key to it } }.toMap()
  }

  @Suppress("DEPRECATION") // Bundle.get(String) is the only way to read a value of unknown type; the typed overloads need the type up front
  private fun stringExtra(extras: Bundle, key: String): String? = extras.get(key)?.toString()

  private fun log(message: String) {
    Log.i(TAG, "$LOG_PREFIX ${oneLine(message)}")
  }

  companion object {
    private const val TAG = "pipette-headless"

    /**
     * Where the identity key events go, on top of [TAG].
     *
     * A separate tag rather than a separate prefix, because the two streams answer different questions: [TAG] is one invocation's transcript, and
     * this is the standing record of what happened to this device's identity. `adb logcat -s pipette-auth` is the whole query.
     */
    private const val AUTH_TAG = "pipette-auth"

    private const val LOG_PREFIX = "[HEADLESS]"

    /** Terminal marker every path emits, so a log consumer knows the invocation is over. Mirrors iOS's `BENCH_DONE`. */
    const val DONE_SENTINEL = "BENCH_DONE"

    /**
     * The terminal line: `BENCH_DONE <code> exit=<code>`.
     *
     * The **numeric first token is the contract**, not decoration. `pipette-ctl`'s `scan_sentinel`
     * (`crates/pipette-ctl/src/transport/process.rs:116`) splits on the sentinel and parses the next whitespace token as the status; a token it
     * cannot parse yields `None`, which the caller reads as "no sentinel present" and falls back to the process exit code. For this surface that
     * fallback is worthless, because `am start` exits 0 even when the activity refuses, so a refusal would be recorded as a completed cell. That is
     * the exact bug iOS's numeric sentinel was introduced to fix, and this line previously read `BENCH_DONE exit=2`, which does not parse.
     *
     * `exit=<code>` is kept after it because `docs/pipette-android/architecture.md` told consumers to key on that spelling, and a script already
     * written against it keeps matching. The scanner reads only the first token, so carrying both costs nothing.
     *
     * No transport scans this surface today (the fleet pushes the `pipette` binary over adb instead), so this is alignment ahead of a consumer rather
     * than a live fix. It is worth doing now precisely because the failure mode is silent: nothing would report the mismatch at the moment it started
     * to matter.
     */
    internal fun doneLine(exit: Int): String = "$DONE_SENTINEL $exit exit=$exit"

    /**
     * Flatten [message] to one physical line, so a value inside it cannot forge a record.
     *
     * `Log.i` splits its message on a line break and emits each piece as its own logcat entry under the same tag. Several values on this surface hold
     * text this device did not choose: `auth-me` prints the `organization` and `client_details` a person typed into the server, and `auth-import`
     * prints the `client_id` and `server_url` a supplied bundle carried. A line break in one of those ends the record early and starts another that
     * the consumer reads as this surface's own. A forged `BENCH_DONE` is the one that costs most, because a scanner keyed on the sentinel reads only
     * its first token and would call the invocation finished.
     *
     * [AUTH_TAG] is the sharper case. [LOG_PREFIX] at least marks which entries under [TAG] this surface wrote, and a forged entry lacks it, but the
     * audit stream carries the line alone — so an injected break there writes an unmarked entry into the standing record of who this device is.
     *
     * Applied where the line is emitted rather than where it is worded. Four objects word lines for this surface, and one that forgets would
     * reintroduce this quietly; both streams pass through here.
     *
     * Every ISO control character becomes a space, not just the two that break framing. A field on this surface may already hold a space, which the
     * `auth-me` exception in [HeadlessInspect]'s KDoc spells out, so a space costs a reader nothing that a terminal escape sequence would not cost
     * them more.
     */
    internal fun oneLine(message: String): String = message.map { if (it.isISOControl()) ' ' else it }.joinToString("")

    /** Mirrors iOS `HeadlessExit`: 0 started, 1 the work was refused or failed, 2 the invocation was malformed. Logged, not returned; see below. */
    const val EXIT_OK = 0
    const val EXIT_FAILURE = 1
    const val EXIT_USAGE = 2

    /**
     * The same three outcomes in Android's result-code space, for a caller using `startActivityForResult` or the Activity Result APIs.
     *
     * A separate scale because the two vocabularies collide: iOS's success is `0`, which is [Activity.RESULT_CANCELED], and Android's success is
     * `-1`. `RESULT_FIRST_USER` is where the platform says an app's own codes may start, so the two failures sit there rather than on `1` and `2`,
     * which read as arbitrary to a generic result handler.
     */
    const val RESULT_REFUSED = RESULT_FIRST_USER
    const val RESULT_USAGE = RESULT_FIRST_USER + 1

    internal fun activityResultFor(exit: Int): Int =
      when (exit) {
        EXIT_OK -> RESULT_OK
        EXIT_USAGE -> RESULT_USAGE
        else -> RESULT_REFUSED
      }
  }
}
