package ai.liquid.pipette

import android.os.Process
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

/**
 * The logcat tag every part of the cpuset guard writes under, so `adb logcat -s pipette-cpuset` shows one stream. `RemoteBenchmarkEngine` emits it
 * from the main process and everything else from `:benchmark` (`BenchmarkActivity` included, which runs there for the cpuset boost); the whole point
 * is reading the two side by side, which is why each line names the process it came from.
 */
internal const val CPUSET_TAG = "pipette-cpuset"

/**
 * A confirmed loss of scheduling privilege around a benchmark: the process was moved to a worse scheduling group or onto fewer cores part-way
 * through, or it never reached the good state at all. Either way the measurement ran under conditions that make its numbers incomparable with
 * anything else in the job.
 *
 * Note the first two are not the same thing. Leaving `top-app` is scored on the cpuset alone, and on some devices the allowed-CPU list is unchanged
 * by it; what changed is the share of those cores this process will get.
 */
data class CpusetDemotion(val baseline: CpuAffinitySnapshot, val observed: CpuAffinitySnapshot, val reason: Reason) {
  enum class Reason {
    /** Left the `top-app` cpuset. The ordinary case: the screen turned off, or the pocket screen lost window focus. */
    LEFT_TOP_APP,
    /** The allowed-CPU set shrank, or stopped including the prime tier. The fallback signal, for kernels whose cpuset path is uninformative. */
    LOST_CPUS,

    /**
     * `top-app` was never reached in the first place, so there was no transition for the other two to catch.
     *
     * The case the other reasons structurally cannot see. Both compare a sample against the best state seen so far, and a run that starts demoted and
     * stays demoted adopts that as its own baseline: every sample then compares equal and the guard stays silent while every rep is measured without
     * the prime cores.
     *
     * Found on a Pixel 10 Pro Fold running a **headless run with the screen off**, which is the unattended fleet case the headless surface exists
     * for. `BenchmarkActivity` launches and resumes normally, but a dozing device grants it `/foreground` rather than `/top-app`, so the process runs
     * on `0-6` of `0-7` with the prime core excluded. The cells completed and reported success, roughly 18% below the same cell measured with the
     * screen on, and nothing marked them as different.
     */
    NEVER_TOP_APP,
  }

  /**
   * One line for the cell's `errorMessage`, which is the only place a user ever learns this happened: `JobsScreen` renders it both as the job card's
   * "First failure" and in the per-cell details.
   *
   * So it carries three things, in order of what the reader needs. What went wrong in plain words, since "cpuset" means nothing to most people who
   * will read it. What to do differently, because the cause is almost always an action they can avoid repeating. And the raw scheduling state last,
   * for whoever is actually debugging a device's scheduler: a transition for the two reasons that have one, a single state for the one that does not.
   */
  fun describe(): String =
    // Exhaustive over Reason on purpose, so a fourth one fails the build here rather than silently inheriting another reason's wording. That is
    // also why the two transition reasons pass their own phrasing in instead of `describeTransition` switching on `reason` a second time.
    when (reason) {
      Reason.LEFT_TOP_APP -> describeTransition("the system stopped scheduling this benchmark as the foreground app")
      Reason.LOST_CPUS -> describeTransition("the system reduced the CPUs this benchmark was allowed to run on")
      Reason.NEVER_TOP_APP -> describeNeverTopApp()
    }

  /** The two reasons that describe a *move*, rendered as `before -> after` with the prime cores lost along the way. */
  private fun describeTransition(what: String): String {
    val prime = if (observed.excludesTopTier && !baseline.excludesTopTier) ", losing the prime cores (${baseline.topTierCpus ?: "?"})" else ""
    return "Measurement abandoned: $what, so the result would not be comparable. " +
      "This usually means the screen turned off or another app came to the front; keep the run screen visible for the whole job. " +
      "(cpuset ${baseline.cpusetPath ?: "?"} -> ${observed.cpusetPath ?: "?"}, " +
      "CPUs ${baseline.allowedCpus ?: "?"} -> ${observed.allowedCpus ?: "?"}$prime)"
  }

  /**
   * [Reason.NEVER_TOP_APP], which has no move to render.
   *
   * Both sides hold the same observation, so an arrow between them would read as a bug in the message rather than as the point being made. This
   * states the one state the run stayed in, and names the prime cores it never got rather than ones it lost. The advice differs too: nothing changed
   * part-way, so "the screen turned off" is the wrong diagnosis, and the caller most likely to hit this is an unattended one that never had a screen
   * on to begin with.
   */
  private fun describeNeverTopApp(): String {
    // Belt and braces: `neverTopApp` only builds this reason from a sample with `excludesTopTier` set, so the false branch is unreachable today. Kept
    // because `topTierCpus` names the device's prime cores rather than the ones this process was denied, so a caller that relaxed the firing rule
    // would otherwise silently begin asserting a loss that did not happen. Same shape as the transition wording, unreadable tier included.
    val prime = if (observed.excludesTopTier) ", never reaching the prime cores (${observed.topTierCpus ?: "?"})" else ""
    // "last sample", not "throughout". The streak only requires consecutive samples that were not `top-app` and were denied the prime cores; on a
    // device that moves between two demoted groups those samples can have different paths, and only this one was kept.
    return "Measurement abandoned: the system never scheduled this benchmark as the foreground app, so the result would not be comparable. " +
      "The screen must stay on with the run screen visible for the whole job; an unattended or headless run needs the device kept awake. " +
      "(cpuset ${observed.cpusetPath ?: "?"} at the last sample, CPUs ${observed.allowedCpus ?: "?"}$prime)"
  }

  /** Serialized for the binder hop out of `:benchmark`, the same way [CpuAffinitySnapshot] is. */
  fun toJson(): String =
    JSONObject()
      .apply {
        put("reason", reason.name)
        put("baseline", baseline.toJson())
        put("observed", observed.toJson())
      }
      .toString()

  companion object {
    fun fromJson(json: String): CpusetDemotion? =
      runCatching {
          val o = JSONObject(json)
          CpusetDemotion(
            baseline = CpuAffinitySnapshot.fromJson(o.getString("baseline")) ?: return null,
            observed = CpuAffinitySnapshot.fromJson(o.getString("observed")) ?: return null,
            reason = Reason.valueOf(o.getString("reason")),
          )
        }
        .getOrNull()
  }
}

/**
 * The pure decision core behind [CpusetWatcher]: fed one snapshot at a time, it reports the first *confirmed* demotion and nothing after it. Split
 * out from the polling so the policy is unit-testable off a device, the way [CpuAffinityProbe.buildSnapshot] is.
 *
 * The baseline is the best state seen so far rather than the first sample, because the first sample is taken before the pocket-mode Activity has
 * necessarily been granted `top-app`: `JobRunner.launchPocketOnce` starts it just before the model load, and the promotion lands asynchronously. Were
 * the baseline pinned to the first sample, a cell that began in `/foreground`, was promoted, then dropped back would compare equal and never fire.
 *
 * One instance per cell, seeded from the last cell's [bestSeen]. That carries the baseline forward, which matters because a process still demoted
 * when the next cell starts would otherwise adopt the demotion as its own normal, see no transition, and measure on in silence.
 *
 * A *fresh* instance per cell rather than one shared and reset, because [CellDemotionWatch.stop] cannot guarantee the previous poll thread has
 * finished: its join is bounded, and a thread that outlasts it would go on mutating `streak` and `fired` afterwards. Against a shared tracker that
 * silently disarms the next cell; against a discarded one it harms nothing. Two values cross between cells, [bestSeen] and [neverTopAppSeen], because
 * both describe the run rather than one cell. Both are volatile for the same reason: the thread that reads them out has only a bounded join behind
 * it, so without that they carry no happens-before edge and could be read arbitrarily stale.
 *
 * Transitions *among* non-`top-app` groups are deliberately not ranked. There is no device-independent ordering of `/foreground` against `/moderate`,
 * and inventing one would produce confident nonsense on the OEMs that matter most here; [Reason.LOST_CPUS] covers those cases on the evidence that
 * actually generalizes, which is how many cores the process may run on.
 */
class CpusetTracker(
  private val confirmations: Int = DEFAULT_CONFIRMATIONS,
  seed: CpuAffinitySnapshot? = null,
  /**
   * Consecutive readable, non-`top-app` samples before [CpusetDemotion.Reason.NEVER_TOP_APP] fires.
   *
   * Much longer than [confirmations], and for a different job. That one rides out a blip in a state already established; this one waits for a state
   * to be established at all. `JobRunner.launchPocketOnce` starts the pocket Activity just before the model load and the promotion lands
   * asynchronously, so the first samples of a cell are *expected* to be non-`top-app` on a perfectly healthy run.
   *
   * Three samples is three seconds at the default poll, and the number is bounded from both sides.
   *
   * From below by the promotion latency, which is far smaller than it looks: on a screen-on device the cell-start probe *already* reads `/top-app`,
   * before the first poll has run at all, so three polls is several times any latency observed.
   *
   * From above by how long a cell actually is. The watch covers `loadModel` plus `runBenchmark`, and the poll sleeps before its first sample, so a
   * grace of N needs N seconds of watched time to reach a verdict. A screen-off repro on a Pixel 10 Pro Fold, a 230M model with the weights already
   * in page cache, did not survive five polls: the guard sat one sample short and the cell completed unjudged, which is the very defect this exists
   * to catch.
   *
   * A cell shorter than the grace is still not judged, and that is the residual limitation. It matters less than it reads, because the streak is
   * carried between cells: a *run* of short cells accumulates across them and is refused on a later one, which is the scope the guard is named for.
   */
  private val topAppGrace: Int = DEFAULT_TOP_APP_GRACE,
  /**
   * Streak carried in from the previous cell, for the same reason [seed] is.
   *
   * Without it the check is cell-scoped while the condition it detects is run-scoped: a job of cells shorter than the grace would reset the count on
   * every cell and never reach a verdict, though the device was denying `top-app` for the whole run. Any readable sample that is not the condition
   * still clears it, so a carried streak cannot outlive the condition that produced it.
   *
   * Clamped into `0..topAppGrace - 1` on the way in, which bounds the value rather than changing any verdict. A tracker that fired exports a streak
   * at or above the grace, and carrying that verbatim would let the number grow without limit across a long job while meaning nothing more than "at
   * the grace". Note what the clamp does *not* buy: a seed of `grace - 1` and an unclamped seed both fire on the next judgeable sample, so a cell
   * that opens with a carried streak gets exactly one poll for a promotion to land in either way. What protects the innocent first cell of a *new
   * job* is `CellDemotionWatch.closeJob` clearing this outright.
   */
  neverTopAppSeed: Int = 0,
) {
  init {
    require(topAppGrace >= 1) { "topAppGrace must be at least 1, was $topAppGrace" }
  }

  // The two fields read from outside the poll thread, volatile for the reason given in the class doc
  // above. `streak` and `fired` are never read off-thread and stay plain. Note that for the streak the
  // question is visibility, not atomicity: a 32-bit write cannot tear in the first place.
  @Volatile private var best: CpuAffinitySnapshot? = seed
  @Volatile private var neverTopAppStreak = neverTopAppSeed.coerceIn(0, topAppGrace - 1)

  private var streak = 0
  private var fired = false

  /** The best state observed so far, for seeding the next cell's tracker. Null until a usable sample arrives. */
  val bestSeen: CpuAffinitySnapshot?
    get() = best

  /** The run-scoped streak, for carrying into the next cell's tracker. See the constructor's `neverTopAppSeed`. */
  val neverTopAppSeen: Int
    get() = neverTopAppStreak

  /**
   * Record [sample] as an observation of how good things can be, without scoring it as a demotion.
   *
   * For the sample taken at the very start of a cell, before the model load. Scoring that one would be wrong: with nothing to compare against it can
   * only ever be the baseline, and were it scored against a retained baseline it would spend the fired latch on a demotion nobody is listening for
   * yet. Using it is what puts the baseline at the known-good pre-load state rather than a poll interval into the load.
   */
  fun observe(sample: CpuAffinitySnapshot) {
    if (!sample.isUsable()) return
    val current = best
    if (current == null || isBetter(sample, current)) best = sample
  }

  /** Feed one sample. Returns the demotion on the poll that confirms it, then null forever after. */
  @Suppress(
    "ReturnCount"
  ) // already-fired / unusable-sample / clean-sample / unconfirmed guards, as in RemoteBenchmarkEngine.benchmarkProcessCpuAffinity
  fun offer(sample: CpuAffinitySnapshot): CpusetDemotion? {
    if (fired) return null
    // A sample nothing could be read out of is an absence of evidence, not evidence of demotion. Scoring it would invert the whole guard: a null
    // `cpusetPath` compares unequal to "/top-app" and an unreadable `Cpus_allowed_list` parses to zero CPUs, so two transient read failures would
    // fail a cell that was running perfectly. It also leaves `streak` alone rather than clearing it, so a real demotion straddling one bad read is
    // still confirmed by the samples either side of it.
    if (!sample.isUsable()) return null
    // On the first usable sample this adopts it as the baseline and then compares it against itself, which is trivially not a demotion, so the first
    // poll falls through the clean-sample branch below rather than needing its own.
    val baseline = best ?: sample.also { best = it }
    val reason = demotionReason(baseline, sample)
    if (reason == null) {
      // A single clean sample clears the streak: the point of requiring consecutive
      // confirmations is to ride out a blip, and a recovery is exactly that.
      streak = 0
      if (isBetter(sample, baseline)) best = sample
      // Clean against the baseline is not the same as good. A cell that has been sitting in `/foreground` since it started is clean on every poll,
      // because the baseline it is compared against is `/foreground` too. That is the blind spot this second check exists to cover.
      return neverTopApp(sample)
    }
    streak += 1
    if (streak < confirmations) return null
    fired = true
    return CpusetDemotion(baseline, sample, reason)
  }

  /**
   * Whether [sample] confirms this run never attained `top-app`, having already been judged no worse than its baseline.
   *
   * A sample counts only when it is readable, when neither it nor the best state on record is `top-app`, and when the prime cores are actually
   * denied. Each clause refuses to guess rather than being cautious for its own sake.
   *
   * **No better state on record.** If `top-app` was ever reached this is not a run that never attained it, and a later fall away from it is
   * [CpusetDemotion.Reason.LEFT_TOP_APP]'s to report with the transition it actually saw. Unreachable as the code stands, because `demotionReason`
   * claims every such sample first, and kept only so that reordering it cannot quietly hand those samples to this arm.
   *
   * **The prime cores must actually be denied.** The clause that is not about the cpuset path, and the one that keeps this from being a mass false
   * positive. A path is only evidence if the device spells its groups the way we expect: a process in the root cpuset reports `/`, which is non-null
   * and not `/top-app` yet says nothing at all about scheduling, and an OEM kernel is free to use names we have never seen. Judging on the name alone
   * would fail every cell on exactly the devices whose cpuset path is uninformative, the population [CpusetDemotion.Reason.LOST_CPUS] exists to
   * serve. `excludesTopTier` is derived from the allowed-CPU list rather than from any name, so it is the evidence that generalizes, and it is what
   * the Pixel repro actually showed.
   *
   * That leaves a blind spot, accepted deliberately: a device that denies `top-app` but allows every core is not caught, because there only the
   * *share* of those cores drops, and that is the thing this class says throughout it cannot measure portably.
   *
   * **The streak must survive the grace.** A promotion lands asynchronously after the pocket Activity starts, so a healthy cell spends its first
   * samples here. A readable sample that is not the condition resets the count, so the grace measures consecutive failure to arrive rather than
   * elapsed time; an unreadable one holds it, for the reason given at the branch.
   *
   * The unavoidable cost: a cell that finishes inside the grace window is never judged. Better than the alternative, which is fabricating a verdict
   * from fewer samples than it takes to tell a slow promotion from an absent one, and the cells this protects run for minutes rather than seconds.
   */
  private fun neverTopApp(sample: CpuAffinitySnapshot): CpusetDemotion? {
    // Every input behind the verdict has to be present before this sample says anything. The cpuset names the group; the allowed-CPU list and the
    // frequency tier are what `excludesTopTier` is derived from, and `CpuAffinityProbe.buildSnapshot` reports it as false when either is missing.
    // Read naively, an unreadable input is indistinguishable from good news, which is the inversion this whole class keeps refusing to make.
    //
    // `sweepComplete` is the subtle one, because a partial sweep does not look unreadable at all. A core that is offline has no `cpufreq` directory
    // and drops out silently, so the tier is computed from whatever was online: if the fast cores were parked, `topTierCpus` names a slower tier that
    // this process is free to run on and `excludesTopTier` reads false. That is the same absence of evidence wearing the shape of good news, and it
    // is likeliest in exactly the dozing, screen-off conditions this reason exists to catch.
    val readable = sample.cpusetPath != null && sample.allowedCount > 0 && sample.topTierCpus != null && sample.sweepComplete
    val judgeable = sample.cpusetPath != TOP_APP && best?.cpusetPath != TOP_APP && sample.excludesTopTier
    neverTopAppStreak =
      when {
        !readable -> neverTopAppStreak
        judgeable -> neverTopAppStreak + 1
        else -> 0
      }
    if (neverTopAppStreak < topAppGrace) return null
    fired = true
    // Both sides are the same observation on purpose: there was no transition, and `describeNeverTopApp` renders it without an arrow for that reason.
    return CpusetDemotion(sample, sample, CpusetDemotion.Reason.NEVER_TOP_APP)
  }

  private companion object {
    /** Two consecutive confirmations, so a momentary blip does not discard a cell that may have been running for minutes. */
    const val DEFAULT_CONFIRMATIONS = 2

    /** Three consecutive samples, which is three seconds at the default poll. See [topAppGrace] for how the number was chosen. */
    const val DEFAULT_TOP_APP_GRACE = 3

    const val TOP_APP = "/top-app"

    /** Whether anything at all could be read. Every field of [CpuAffinitySnapshot] degrades to null / zero rather than throwing. */
    fun CpuAffinitySnapshot.isUsable(): Boolean = cpusetPath != null || allowedCount > 0

    /**
     * Each clause is guarded on the evidence it needs actually having been read, so a partial sample (cpuset readable, `/proc/self/status` not, or
     * the reverse) is judged only on the half that is present instead of on the half that is missing.
     */
    fun demotionReason(baseline: CpuAffinitySnapshot, sample: CpuAffinitySnapshot): CpusetDemotion.Reason? =
      when {
        baseline.cpusetPath == TOP_APP && sample.cpusetPath != null && sample.cpusetPath != TOP_APP -> CpusetDemotion.Reason.LEFT_TOP_APP
        // Both remaining guards are needed: a shrunken set is the general signal, but a
        // same-sized set that swapped a prime core for a little one is the specific
        // harm, and neither implies the other.
        sample.allowedCount > 0 && sample.allowedCount < baseline.allowedCount -> CpusetDemotion.Reason.LOST_CPUS
        lostPrimeTier(baseline, sample) -> CpusetDemotion.Reason.LOST_CPUS
        else -> null
      }

    /**
     * Whether [sample] stopped including the prime tier that [baseline] had, judged only when both sides actually know what that tier is.
     *
     * Every clause needs its evidence present on BOTH sides. A snapshot taken while either read failed carries `allowedCount` 0, `topTierCpus` null
     * and `excludesTopTier` false by default, and scoring a complete sample against those defaults reports a demotion that never happened. The `/sys`
     * sweep in particular can fail and recover inside one cell, so empty-then-populated is an expected shape rather than a corner case.
     *
     * `sweepComplete` is required for the same reason the tiers must be non-null, and catches the case that does not look like a failure at all: a
     * tier named after whichever cores happened to be online is not the device's prime tier, so comparing one against the other reports a change in
     * what could be read as though it were a change in scheduling.
     */
    private fun lostPrimeTier(baseline: CpuAffinitySnapshot, sample: CpuAffinitySnapshot): Boolean =
      baseline.allowedCount > 0 &&
        sample.allowedCount > 0 &&
        baseline.topTierCpus != null &&
        sample.topTierCpus != null &&
        baseline.sweepComplete &&
        sample.sweepComplete &&
        sample.excludesTopTier &&
        !baseline.excludesTopTier

    fun isBetter(sample: CpuAffinitySnapshot, baseline: CpuAffinitySnapshot): Boolean =
      when {
        sample.cpusetPath == TOP_APP && baseline.cpusetPath != TOP_APP -> true
        // Ahead of the core-count clauses, and the reason they are not simply an unordered OR: a
        // sample that is not `top-app` must never displace a `top-app` baseline, however many cores
        // it reports. Letting it would retire the LEFT_TOP_APP signal for good and leave only the
        // weaker LOST_CPUS fallback. `offer` is protected from this by ordering (it returns
        // LEFT_TOP_APP before it ever promotes); `observe` has no such check and relies on this one.
        baseline.cpusetPath == TOP_APP && sample.cpusetPath != TOP_APP -> false
        sample.allowedCount > baseline.allowedCount -> true
        // `sample.topTierCpus != null` for the same reason: a sample whose `/sys` sweep failed reports
        // excludesTopTier false by default, and reading that as a recovery would install a baseline
        // carrying no prime-tier evidence, retiring the signal for the rest of the job. `sweepComplete`
        // closes the same hole for a sweep that half succeeded: there the tier is non-null but named
        // after whichever cores were online, so a parked prime cluster reads as a recovery it is not.
        baseline.excludesTopTier && !sample.excludesTopTier && sample.allowedCount > 0 && sample.topTierCpus != null && sample.sweepComplete -> true
        else -> false
      }
  }
}

/**
 * The cell-scoped lifecycle around [CpusetWatcher]: which watch is running, the verdict it produced, and the two values carried between cells, the
 * baseline and the never-attained streak. Both are job-scoped as well, so [closeJob] clears them and a new job cannot inherit either.
 *
 * A **cell** is the unit, not a run, because a cell is `loadModel` followed by `runBenchmark` (or a single `runBenchmarkFresh`), and the model load
 * is exactly where a screen-off most often lands: it is the longest uninterrupted stretch of a cell, and a demotion there means every rep that
 * follows is measured on the reduced core set. Watching only the run would leave that window uncovered.
 *
 * Each cell gets a fresh [CpusetTracker], seeded with the best state seen so far. Fresh, because a tracker fires once, so a resumed job inside
 * `RemoteBenchmarkEngine`'s idle grace would otherwise inherit a spent one that can never fire again, and because a discarded tracker is immune to
 * anything a poll thread that outlasted [stop]'s bounded join goes on to do. Seeded, because a process still demoted when the next cell starts would
 * otherwise adopt the demotion as its own baseline, see no transition, and measure on in silence.
 *
 * Lives here rather than in `PipetteBenchmarkService` so that service keeps one job: bridging AIDL to the engine.
 */
class CellDemotionWatch(
  private val probe: () -> CpuAffinitySnapshot,
  /**
   * The watcher's poll interval. Injectable for the same reason [CpusetWatcher] takes one: so tests run in milliseconds rather than poll intervals.
   */
  private val pollMillis: Long = CpusetWatcher.DEFAULT_POLL_MILLIS,
) {
  // The baseline carried between cells: a volatile reference to an immutable snapshot, which is what
  // makes it safe to read out of a watcher whose poll thread may not have stopped yet. See CpusetTracker.
  @Volatile private var best: CpuAffinitySnapshot? = null

  /** The run-scoped never-attained streak, carried between cells alongside [best]. See `CpusetTracker.neverTopAppSeed`. */
  @Volatile private var neverTopAppStreak: Int = 0
  @Volatile private var watcher: CpusetWatcher? = null

  // Two locks, because one of the two things being protected must never wait on the other.
  //
  // `stateLock` guards the fields a delivering poll thread touches. It is held for microseconds and
  // never across a join, so a delivery can always take it promptly.
  //
  // `lifecycleLock` serializes whole beginCell/stop operations against each other, which stateLock
  // alone cannot: those span a join, and holding stateLock across one would make every late delivery
  // wait out the join. Binder serves loadModel and a proxy teardown on different threads, so the
  // interleaving is reachable rather than theoretical.
  private val stateLock = Any()
  private val lifecycleLock = Any()

  // Which cell is current. A watcher's delivery carries the epoch it was armed for and is dropped if
  // that has moved on, so a thread that wakes late cannot post its verdict into somebody else's cell.
  // Advanced by stop() as well as beginCell: a watch that has been closed has no cell to report into
  // either, and the run it belonged to has already returned.
  private var epoch = 0

  /**
   * The demotion observed during the current cell, or null if none was. Survives [stop], because the caller reads it after a run has already
   * returned; only [beginCell] and [consumeVerdict] clear it.
   */
  @Volatile
  var lastDemotion: CpusetDemotion? = null
    private set

  /**
   * Read the verdict and clear it, so it can explain at most one cell.
   *
   * The watch is deliberately left armed when a run ends, because a demotion confirmed in that run's final moments still has to be readable by the
   * caller about to ask for it. The price of staying armed is that a demotion seen in the gap between cells would otherwise still be sitting there
   * when the *next* cell asks, and that cell may have failed for an unrelated reason, such as never reaching this process at all. Consuming on read
   * means a verdict explains the cell it was recorded for, and nothing else.
   */
  fun consumeVerdict(): CpusetDemotion? = synchronized(stateLock) { lastDemotion.also { lastDemotion = null } }

  /**
   * Whether a cell is currently open, which is not the same as a thread still polling: a watcher that has fired has exited its thread but is left in
   * place, so this stays true until [stop] or the next [beginCell].
   *
   * That distinction is load-bearing for `PipetteBenchmarkService.beginRun`, which gates its belt-and-braces `beginCell` on this. Re-arming after a
   * fire would clear [lastDemotion], discarding the very verdict the run is about to act on.
   */
  val isWatching: Boolean
    get() = watcher != null

  /**
   * Start watching for a cell that is beginning, replacing any watch still running. [onDemotion] runs on the watcher's thread.
   *
   * The whole operation is serialized against [stop] by `lifecycleLock`, while the state a delivery reads moves under `stateLock` in one step. The
   * previous watcher's join happens between the two `stateLock` sections (still holding `lifecycleLock`), so a poll thread delivering at that moment
   * finds the epoch already advanced and drops its verdict instead of waiting on the join.
   */
  fun beginCell(onDemotion: (CpusetDemotion) -> Unit) {
    synchronized(lifecycleLock) {
      // Closing the old cell and opening the new one are two short critical sections with the slow
      // parts (a join, then a probe) deliberately between them, so a delivering poll thread never
      // waits on anything but a few field writes.
      val previous =
        synchronized(stateLock) {
          watcher.also {
            watcher = null
            epoch += 1
            // Cleared with the epoch advance rather than when the new watcher is published: the gap
            // between the two spans a join and a probe, and for that whole window the watch would
            // report no cell open while still holding the last cell's verdict, which is exactly the
            // pair of answers that makes a caller pre-cancel an innocent run.
            lastDemotion = null
          }
        }
      previous?.stop()
      // Both carried values taken under one lock, not two back-to-back ones. They describe the same instant of the same finished cell, so no
      // observer should see the baseline advanced while the streak is still the old cell's. Why the streak is carried is on `neverTopAppSeed`.
      val (seed, neverTopAppSeed) =
        synchronized(stateLock) {
          val carriedBest = previous?.bestSeen?.also { best = it } ?: best
          val carriedStreak = previous?.neverTopAppSeen?.also { neverTopAppStreak = it } ?: neverTopAppStreak
          carriedBest to carriedStreak
        }
      val cellTracker = CpusetTracker(seed = seed, neverTopAppSeed = neverTopAppSeed)
      // One sample at cell start, doing two jobs. It seeds the baseline at the known-good pre-load
      // state instead of leaving the first poll interval of the load unguarded, and it is the
      // diagnostic line a device run reads back with `adb logcat -s pipette-cpuset` (prefixed to
      // match RemoteBenchmarkEngine's side-by-side [main]/[:benchmark] pair under the same tag).
      // Outside stateLock: it is two `/proc` reads, and a `/sys` sweep the first time.
      runCatching { probe() }
        .onSuccess { start ->
          cellTracker.observe(start)
          runCatching { Log.i(CPUSET_TAG, "[:benchmark] ${start.summary()}") }
        }
      val w = CpusetWatcher(probe, pollMillis = pollMillis, tracker = cellTracker)
      synchronized(stateLock) {
        watcher = w
        val cell = epoch
        // Published and started as one step, so nothing can observe a watcher that is not yet running.
        // Safe to hold stateLock across: Thread.start() returns at once, and the new thread sleeps a
        // poll interval before it needs the lock.
        w.start { demotion -> deliver(cell, demotion, onDemotion) }
      }
    }
  }

  /**
   * Accept this verdict, or drop it because its cell has moved on.
   *
   * [CpusetWatcher] already refuses to deliver after its own `stop()`, but that check is claimed before the write rather than with it: a thread
   * descheduled in between, which is precisely what a demotion causes, could wake after this watch had moved on and post cell N's verdict into cell
   * N+1, failing an innocent cell and cancelling its run. The epoch closes that window, and covers a closed watch as well as a replaced one.
   *
   * The two guards are deliberately redundant, and mutation testing says so: with `CellDemotionWatchTest.aVerdictFromAReplacedCellIsDropped` in
   * place, deleting either guard alone keeps the suite green because the other still catches it, and deleting both turns it red. So a future reader
   * who "proves" this check is dead by removing it and watching CI pass has proved only that the other half is still there.
   */
  private fun deliver(cell: Int, demotion: CpusetDemotion, onDemotion: (CpusetDemotion) -> Unit) {
    synchronized(stateLock) {
      if (epoch != cell) return
      // Recorded before the handler runs, because that handler cancels the run and the reason has to
      // be readable by the time the cancelled run returns. A cancel visible without a reason reads as
      // a plain user cancel, and the cell would be filed CANCELLED.
      lastDemotion = demotion
      // Invoked under the same lock as the epoch check, so recording the verdict and cancelling the
      // run cannot be split by a cell change: cancelling a run the verdict was then dropped for would
      // stop an innocent cell for no stated reason. Safe to hold because the handler's contract is to
      // flip a cancel flag and return, doing no I/O and taking no lock of its own.
      onDemotion(demotion)
    }
  }

  /**
   * Stop watching and forget the carried baseline, for the end of a job rather than the end of a cell.
   *
   * The baseline must not survive a job boundary. `RemoteBenchmarkEngine` keeps `:benchmark` alive for a few seconds after a job, so a job started
   * inside that window reuses this process; carrying the old job's `top-app` baseline into it would judge the new job's first cell against a state it
   * has not reached yet. The pocket Activity is relaunched just before that cell's load and its `top-app` promotion is asynchronous, so the honest
   * answer for a brand new job is that it has no baseline until it observes one.
   */
  fun closeJob() {
    // Held across BOTH steps, not just the write. This is the one lifecycle operation whose entire purpose is to guarantee the job-scoped state is
    // gone, and taking the lock only for the write leaves a gap where a `beginCell` between `stop()` returning and that write seeds itself from the
    // finished job's baseline, then has the null land behind it. `synchronized` is reentrant, so the nested `stop()` still takes it.
    synchronized(lifecycleLock) {
      stop()
      // `lastDemotion` is job-scoped too. The watch is deliberately left armed after a run ends, so a demotion confirmed in the tail of the last cell
      // (the payload write, the submit, the cooldown) can still be sitting here unread, because `JobRunner` never asks once a cell is COMPLETED.
      // Carried into the next job it would make `beginRun` skip arming the watch and pre-cancel an innocent first run with the previous job's
      // verdict.
      //
      // The streak is job-scoped for the same reason, and `stop()` above has just written the finished job's count into it. `:benchmark` outlives a
      // job by seconds, so the next job can start in that reuse window with no `top-app` baseline yet and a streak inherited from a run that is over.
      // Its first poll lands inside the asynchronous promotion window, which is precisely what the grace exists to tolerate, and the carried count
      // could spend that grace before the promotion ever had a chance to land.
      synchronized(stateLock) {
        best = null
        lastDemotion = null
        neverTopAppStreak = 0
      }
    }
  }

  /**
   * Stop watching, carrying the cell's baseline and never-attained streak forward. Idempotent.
   *
   * Advances the epoch as it closes the cell, so a poll thread that already won its own delivery race and then wakes past the bounded join is dropped
   * rather than writing a verdict into a watch the caller has been told is finished. The join itself holds no lock, so that thread can take
   * `stateLock`, see the epoch has moved, and exit promptly.
   *
   * Correctness does not depend on the join succeeding: [CpusetWatcher.bestSeen] and [CpusetWatcher.neverTopAppSeen] are both volatile reads, so a
   * thread that outlasted it can at worst leave a value one sample stale, never torn.
   */
  fun stop() {
    synchronized(lifecycleLock) {
      val w =
        synchronized(stateLock) {
          watcher.also {
            watcher = null
            epoch += 1
          }
        } ?: return
      w.stop()
      // One critical section for both, as in `beginCell`. The baseline is written only when the watcher has one, since a cell that never got a
      // usable sample must not erase the run's. The streak is written every time, because zero is a real value: it is what a cell that reached
      // `top-app` has to say.
      synchronized(stateLock) {
        w.bestSeen?.let { best = it }
        neverTopAppStreak = w.neverTopAppSeen
      }
    }
  }

  /** The baseline that would be carried into the next cell. Visible for tests; nothing in the app reads it. */
  val bestSeen: CpuAffinitySnapshot?
    get() = synchronized(stateLock) { watcher?.bestSeen ?: best }

  /** The never-attained streak that would be carried into the next cell. Visible for tests; nothing in the app reads it. */
  val neverTopAppSeen: Int
    get() = synchronized(stateLock) { watcher?.neverTopAppSeen ?: neverTopAppStreak }
}

/**
 * Polls this process's cpuset while a benchmark runs and reports the first confirmed demotion.
 *
 * Runs in whichever process hosts the engine (`:benchmark`), because cpuset membership is per-process and it is the engine's threads whose scheduling
 * is in question. It exists because the alternative, watching for the *causes* of a demotion, cannot be made complete: an `ACTION_SCREEN_OFF`
 * receiver misses the notification shade, a system dialog and a biometric prompt, none of which fire `onPause` even though they take window focus and
 * demote the process just the same. Watching `/proc/self/cpuset` observes the effect, so it covers those, an incoming call, split-screen, another app
 * launching, and whatever the next OEM invents.
 *
 * **Cost.** This samples alongside a measurement, so it must not perturb it. Hoisting the per-core `scaling_max_freq` sweep out of the loop
 * ([CpuAffinityProbe.maxFreqByCpu]) is most of the saving: measured in-process on a Pixel 10 Pro Fold (`CpusetWatcherCostTest`) a poll costs ~430 us
 * against ~1515 us for a full snapshot. At [DEFAULT_POLL_MILLIS] that is **0.043% of one core**, spent on a [Process.THREAD_PRIORITY_BACKGROUND]
 * thread, which makes it very unlikely to take time from inference on a prime core (the priority lowers this thread's share, it does not bar it from
 * one), and which stops sampling the moment it fires.
 *
 * The residual could be cut further, by polling only the cpuset path (the cheaper of the two `/proc` reads) and re-reading `/proc/self/status` when
 * it changes, and deliberately is not: it would cost the [CpusetDemotion.Reason.LOST_CPUS] fallback its per-poll evidence on kernels whose cpuset
 * path is uninformative, which is a real correctness trade against a cost already far below anything the guarded measurement can resolve.
 */
class CpusetWatcher(
  private val sampler: () -> CpuAffinitySnapshot,
  private val pollMillis: Long = DEFAULT_POLL_MILLIS,
  private val tracker: CpusetTracker = CpusetTracker(),
) {
  // Written by start() and by stop(); those can be different binder threads.
  @Volatile private var thread: Thread? = null
  private val stopped = AtomicBoolean(false)

  /** The run-scoped never-attained streak this watcher's tracker holds, for seeding the next cell. Safe to read at any time; see [CpusetTracker]. */
  val neverTopAppSeen: Int
    get() = tracker.neverTopAppSeen

  /** The best state this watcher's tracker has observed, for seeding the next cell. Safe to read at any time; see [CpusetTracker]. */
  val bestSeen: CpuAffinitySnapshot?
    get() = tracker.bestSeen

  /**
   * Begin sampling. [onDemotion] is invoked at most once, on the watcher's own thread, and must be cheap and non-blocking: it flips a cancel flag, it
   * does not do the unwinding itself. [CellDemotionWatch] invokes it while holding a lock, so it must not block or acquire one of its own. Calling
   * [start] again while a thread is alive is a no-op.
   */
  fun start(onDemotion: (CpusetDemotion) -> Unit) {
    if (thread?.isAlive == true) return
    stopped.set(false)
    thread =
      Thread {
          // Guarded like every other framework call on this thread: a background priority is an
          // optimization, and failing to get one is no reason to lose the guard entirely.
          runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND) }
          while (!Thread.currentThread().isInterrupted && !stopped.get()) {
            try {
              Thread.sleep(pollMillis)
            } catch (_: InterruptedException) {
              return@Thread
            }
            // Never let a probe failure take down a run. An unreadable /proc entry degrades to a
            // null field, which CpusetTracker.offer discards rather than scores; anything worse
            // is logged and skipped here.
            val demotion = runCatching { tracker.offer(sampler()) }.onFailure { warn("cpuset sample failed", it) }.getOrNull() ?: continue
            // Re-check under the flag, and claim it atomically. A poll can be in flight when
            // stop() lands, and delivering after that would write a demotion the caller has
            // already read past, so the finding would be silently dropped and then attributed
            // to whatever ran next. Losing it here is the correct trade: the run is ending
            // anyway, and the tracker keeps its baseline for the cell that follows.
            if (!stopped.compareAndSet(false, true)) return@Thread
            warn(demotion.describe(), null)
            runCatching { onDemotion(demotion) }.onFailure { warn("demotion handler failed", it) }
            return@Thread
          }
        }
        .apply {
          name = "pipette-cpuset-watch"
          // Daemon: a watcher must never be the reason the process stays alive,
          // least of all in :benchmark, whose whole purpose is to be reclaimable.
          isDaemon = true
          start()
        }
  }

  /**
   * Stop sampling and wait, briefly, for the poll thread to finish. Idempotent, and safe to call whether or not [start] ran.
   *
   * Two mechanisms, for two different problems. `stopped` closes the delivery window: an interrupt cannot preempt a `/proc` read already underway, so
   * without the flag a sample completing after this returns would still deliver a verdict the caller has already read past. The join then makes the
   * common case tidy, letting a caller treat the watch as finished once this returns.
   *
   * Correctness does not rest on the join succeeding. The tracker this watcher owns is discarded by its caller afterwards, and the two values that
   * outlive it, [bestSeen] and [neverTopAppSeen], are both volatile. A thread that outlasts the join can therefore leave either one sample stale, but
   * can neither tear it nor reach the next cell.
   *
   * The wait is bounded and in practice immediate: the poll thread spends its time in an interruptible `Thread.sleep`, so it observes the interrupt
   * at once. `JOIN_TIMEOUT_MILLIS` caps the pathological case rather than the expected one.
   */
  fun stop() {
    stopped.set(true)
    val t = thread ?: return
    t.interrupt()
    try {
      t.join(JOIN_TIMEOUT_MILLIS)
    } catch (_: InterruptedException) {
      // Restore what we consumed. This runs on a binder thread, and swallowing the flag would hide
      // the interrupt from everything downstream in that call.
      Thread.currentThread().interrupt()
    }
    // Keep the reference if it outlasted the join, so [start] still refuses to spawn a second poll
    // thread alongside it. `stopped` stays set, so the survivor can neither deliver nor loop again.
    if (!t.isAlive) thread = null
  }

  /**
   * Logging that cannot itself end the watch. This runs on a daemon thread, where an uncaught throw is silent, so the guard against losing the whole
   * guard to a logging failure has to be here rather than at the call sites. It also lets the class be exercised under JVM unit tests, where
   * `android.util.Log` is a stub that throws.
   */
  private fun warn(message: String, error: Throwable?) {
    runCatching { if (error == null) Log.w(CPUSET_TAG, message) else Log.w(CPUSET_TAG, message, error) }
  }

  companion object {

    /**
     * 1s. With [CpusetTracker]'s two confirmations that is a ~2s detection latency against cells that run for minutes, which buys a debounce far
     * cheaper than the cost of discarding a long cell over a blip.
     */
    const val DEFAULT_POLL_MILLIS = 1_000L

    /** Bounds [stop]'s wait for the poll thread. Generous, because the expected wait is a scheduler hop, not a poll interval. */
    private const val JOIN_TIMEOUT_MILLIS = 500L
  }
}
