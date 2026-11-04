package ai.liquid.pipette

import android.content.Context
import android.os.PowerManager
import android.os.SystemClock
import java.util.Locale
import kotlin.math.min
import org.json.JSONObject

/**
 * The readiness policy a gate applies, in the shape and units the submission payload records it (`benchmark_flags.readiness`, matching the plan's
 * `readiness.max_wait_secs` / `readiness.skip_thermal`).
 *
 * A gate reports this rather than callers reading [Readiness.COOLDOWN_MAX_MILLIS] themselves: the result has to describe the wait that actually
 * governed the measurement, and a fake gate in a test applies a different one.
 */
data class ReadinessPolicy(val maxWaitSecs: Long, val skipThermal: Boolean)

/**
 * What one run asks of the readiness gate, as `benchmark_flags.readiness` names it.
 *
 * Scoped to a job rather than stored as device state, which is the difference that matters: a deadline is a property of the measurement someone asked
 * for, so a run that names none must be gated on the client default and not on whatever the previous run happened to set. It rides the job's manifest
 * for that reason, and [JobRunner] installs the manifest's copy at every run start — a resume or a retry days later gates the way the job was
 * authored to, or its remaining cells would not be comparable to the ones already submitted.
 *
 * A null field means "not named". For [skipThermal] that is distinct from `false`: it defers to the device's Settings toggle, where `false` overrides
 * the toggle for this run.
 */
data class RunReadiness(val maxWaitSecs: Long? = null, val skipThermal: Boolean? = null) {
  /**
   * The deadline this run puts on the gates, in the units they wait in, falling back to [Readiness.COOLDOWN_MAX_MILLIS] when it names none.
   *
   * Saturates rather than wraps. An overflowing conversion would land a *negative* deadline on the gate, which expires before its first poll and
   * reports a device that never cooled when nothing was ever waited for — the same failure the parser's lower bound exists to prevent, reached from
   * the other end of the range.
   */
  fun maxWaitMillis(): Long =
    when {
      maxWaitSecs == null -> Readiness.COOLDOWN_MAX_MILLIS
      maxWaitSecs > Long.MAX_VALUE / MILLIS_PER_SECOND -> Long.MAX_VALUE
      else -> maxWaitSecs * MILLIS_PER_SECOND
    }

  /**
   * Persisted form, in the wire spellings plan-types uses for the same block (`readiness = { max_wait_secs = 1800 }`), so a manifest and a plan read
   * the same. Only the named knobs appear; a run that named neither writes nothing (see [JobManifest.toJson]).
   */
  fun toJson(): JSONObject =
    JSONObject().apply {
      maxWaitSecs?.let { put("max_wait_secs", it) }
      skipThermal?.let { put("skip_thermal", it) }
    }

  companion object {
    /** What a run that names neither knob asks for: the client's own defaults. */
    val DEFAULT = RunReadiness()

    /**
     * Ceiling on an authored deadline. A day is far past any real cooldown (a typical one is ~30-40s against a 300s default), so a larger value is a
     * typo rather than a request, and it keeps the seconds→millis conversion clear of overflow. `HeadlessCommand` refuses anything outside `1..this`
     * at the parse boundary and [fromJson] applies the same range to a stored value.
     */
    const val MAX_WAIT_SECS = 86_400L

    private const val MILLIS_PER_SECOND = 1_000L

    /**
     * Read a stored block, or [DEFAULT] when the manifest holds none — which is every job written before this field existed, and every job started
     * from the New Job screen, both of which are gated on the client defaults.
     */
    fun fromJson(json: JSONObject?): RunReadiness =
      if (json == null) DEFAULT
      else
        RunReadiness(
          // Bounded like the parse boundary, because this is the other way in. A stored `0` — or a value `optLong` coerced to `0`, which is what a
          // hand-edited manifest holding a non-numeric one yields — would arm a deadline that expires before the gate's first poll and fail every
          // cell of the job for a wait that never happened.
          maxWaitSecs = json.optNullableLong("max_wait_secs")?.takeIf { it in 1..MAX_WAIT_SECS },
          skipThermal = json.optNullableBoolean("skip_thermal"),
        )
  }
}

/**
 * How a readiness wait ended. Named to match the shared native kernel's `ReadinessOutcome` (`crates/pipette-android/src/lib.rs`) and the iOS enum of
 * the same name, because the three of them are one contract: `android/native/benchmarks.rs`'s `readiness_gate` turns [TimedOut] into
 * `PipetteError::Readiness`, which fails the cell rather than recording a measurement taken under unknown thermal conditions (PIP-143).
 *
 * [Ready] and [Cancelled] used to be indistinguishable from [TimedOut] here, because the gate returned `Unit` and its callers reported
 * `!cancelFlag.isCancelled` instead, so a device that never cooled reported "proceed" and its throttled numbers were recorded as a normal result.
 */
sealed interface ReadinessOutcome {
  /** The device settled within the budget: run the rep. */
  data object Ready : ReadinessOutcome

  /** The user cancelled during the wait. Also how a failed JNI hop is reported, since that records nothing and observed no thermal verdict. */
  data object Cancelled : ReadinessOutcome

  /** The budget expired with the device still outside the band. [observed] is the last reading, for the error the cell is recorded with. */
  data class TimedOut(val observed: String) : ReadinessOutcome

  /**
   * The JNI encoding `BenchmarkCooldownCallback.waitUntilReady` carries to the native bridge, decoded by `JavaReadinessCallback::wait_until_ready`:
   * `null` for [Ready], and otherwise a tagged string ([CANCELLED_PREFIX] or [TIMED_OUT_PREFIX]) whose remainder is the detail.
   *
   * A nullable String rather than a richer type because this method's signature *is* the JNI contract (the bridge reads it via `jni_sig!`), using a
   * tagged prefix scheme for cancellation and timeout details.
   *
   * Both non-ready variants are tagged, rather than treating "any untagged string" as a timeout, so an `observed` text that happened to begin with
   * the cancel tag could not be decoded as a cancellation. The bridge still treats an unrecognized tag as a timeout, which fails the cell: the safe
   * direction for a contract mismatch is to reject the rep, not to admit it.
   */
  fun encode(): String? =
    when (this) {
      Ready -> null
      Cancelled -> "${CANCELLED_PREFIX}gate cancelled"
      is TimedOut -> "$TIMED_OUT_PREFIX$observed"
    }

  companion object {
    /** Tag marking an encoded [Cancelled]. Must stay in step with the prefix the native bridge matches on. */
    const val CANCELLED_PREFIX = "cancelled:"

    /** Tag marking an encoded [TimedOut]; the remainder is its `observed`. Must stay in step with the native bridge. */
    const val TIMED_OUT_PREFIX = "timedout:"
  }
}

/**
 * The device-readiness gate [JobRunner] consults between cells (and the benchmark service consults per rep). [Readiness] is the real implementation;
 * unit tests substitute a fake.
 */
interface ReadinessGate {
  /**
   * [onStatus] receives the human-readable status plus the milliseconds elapsed *in this wait*. The elapsed value is what the UI's "Cooling m:ss /
   * max" timer must anchor to: `max` is one invocation's budget ([Readiness.COOLDOWN_MAX_MILLIS]), so a timer anchored to anything longer-lived (e.g.
   * the first cooling status of a run) counts across consecutive gate invocations and sails past `max`.
   *
   * The returned [ReadinessOutcome] is the gate's verdict and callers must not synthesize their own: a timeout has to stay distinguishable from a
   * ready device, or the rep it admitted gets recorded as if the device had cooled.
   */
  fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit): ReadinessOutcome

  /** What this gate waits for, for the result to record. See [ReadinessPolicy]. */
  val policy: ReadinessPolicy
}

/**
 * Device readiness gate.
 *
 * Gates execution on device thermal conditions before and between benchmark runs:
 * 1. [PowerManager.getThermalHeadroom] (or injected [currentHeadroom]) provides a normalized estimate of how close the SoC is to severe throttling
 *    (0.0 = comfortable, 1.0 = at the throttle threshold). Proceeds when headroom is at or below [HEADROOM_READY_MAX] (0.85).
 * 2. [PowerManager.getCurrentThermalStatus] (or injected [currentThermalStatus]) provides the coarse OS thermal status enum, used when headroom is
 *    NaN (unsupported or uninitialized). Proceeds only on [PowerManager.THERMAL_STATUS_NONE].
 *
 * App sandboxing on Android restricts sysfs/procfs and dumpsys access, so this direct thermal gate provides authoritative readiness evaluation across
 * both UI and worker processes without relying on native sysfs probing.
 */
class Readiness(
  private val currentThermalStatus: () -> Int,
  private val currentHeadroom: () -> Float = { Float.NaN },
  private val sleepMillis: (Long) -> Unit = { Thread.sleep(it) },
  private val nowMillis: () -> Long = { System.nanoTime() / 1_000_000L },
  /**
   * Whether the thermal criterion is waived (PIP-434). A supplier rather than a value because the setting lives in DataStore, whose reads are
   * `suspend`, while this is constructed synchronously; callers pass a lambda over an in-memory cached value, so a flipped toggle takes effect at the
   * next gate invocation rather than at the next app launch.
   */
  private val skipThermal: () -> Boolean = { false },
  /**
   * The wait budget one gate invocation gets, from the run's `benchmark_flags.readiness.max_wait_secs`. A supplier for the same reason [skipThermal]
   * is one: it is delivered per run (headless extra, or the start Intent for the `:benchmark` process) after this is constructed.
   *
   * Defaults to [COOLDOWN_MAX_MILLIS], which is what an unset `max_wait_secs` means everywhere else; the plan leaves it out and the client's own
   * default applies.
   */
  private val maxWaitMillis: () -> Long = { COOLDOWN_MAX_MILLIS },
) : ReadinessGate {

  override fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit) =
    waitUntilReady(cancelFlag, onStatus, maxWaitMillis(), DEFAULT_POLL_MILLIS)

  fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit, maxMillis: Long, pollMillis: Long): ReadinessOutcome {
    if (cancelFlag.isCancelled) return ReadinessOutcome.Cancelled
    if (skipThermal()) return ReadinessOutcome.Ready

    val startedAt = nowMillis()
    while (!cancelFlag.isCancelled) {
      val headroom = currentHeadroom()
      val ready: Boolean
      val label: String
      if (!headroom.isNaN()) {
        ready = headroom <= HEADROOM_READY_MAX
        label = "headroom ${String.format(Locale.US, "%.2f", headroom)}"
      } else {
        val status = currentThermalStatus()
        // Proceed only on an explicit NONE; unknown/negative status
        // (e.g. a failed read returning -1) must keep waiting, not be
        // treated as cool because -1 <= 0.
        ready = status == PowerManager.THERMAL_STATUS_NONE
        label = AndroidThermalStatusProvider.statusLabel(status)
      }
      if (ready) return ReadinessOutcome.Ready

      val elapsed = nowMillis() - startedAt
      if (elapsed >= maxMillis) return ReadinessOutcome.TimedOut("$label after ${elapsed / 1000}s")
      onStatus("Waiting for device to cool ($label, ${elapsed / 1000}s)...", elapsed)
      sleepCancellable(pollMillis, cancelFlag)
    }
    return ReadinessOutcome.Cancelled
  }

  /**
   * The deadline this gate is actually applying, plus whether the thermal criterion was waived.
   *
   * Computed off the same [maxWaitMillis] / [skipThermal] suppliers the gate reads, not a snapshot of either, so the recorded value is the one that
   * governed the measurement: [JobRunner] writes a cell's payload right after the between-cell gate that admitted it, and the toggle is a Settings
   * switch a user would have to flip inside that window to desynchronize them.
   */
  override val policy: ReadinessPolicy
    get() = ReadinessPolicy(maxWaitSecs = maxWaitMillis() / 1_000L, skipThermal = skipThermal())

  private fun sleepCancellable(totalMillis: Long, cancelFlag: CancelFlag) {
    var remaining = totalMillis.coerceAtLeast(0L)
    while (remaining > 0 && !cancelFlag.isCancelled) {
      val chunk = min(remaining, SLEEP_CHUNK_MS)
      sleepMillis(chunk)
      remaining -= chunk
    }
  }

  companion object {
    private const val SLEEP_CHUNK_MS = 250L

    /**
     * How long the readiness gate waits for the device to cool before giving up on a rep, and the "allowed to cool" deadline the progress UI counts
     * its live "Cooling m:ss / max" timer against.
     *
     * 300 s is the shared cross-platform default (matching `DEFAULT_MAX_WAIT` across clients; iOS carries it as `readinessTimeoutSeconds`).
     */
    const val COOLDOWN_MAX_MILLIS = 300_000L

    private const val DEFAULT_POLL_MILLIS = 2_000L

    // Thermal-headroom ceiling for entering a rep. Headroom is 0.0 (cool) to
    // 1.0 (severe-throttle threshold). This must sit ABOVE the device's
    // sustained-load equilibrium, or the gate fights normal operation and
    // stalls the benchmark: on a fan-cooled Pixel 10 Pro Fold the SoC settles
    // at ~0.70 headroom under continuous inference (not throttling, but 30% from
    // severe). 0.85 lets the device run at its steady-state load while still holding
    // when it is genuinely close to throttling.
    private const val HEADROOM_READY_MAX = 0.85f
  }
}

/** Current OS thermal status, for display and the [Readiness] fallback. */
interface ThermalStatusProvider {
  fun currentStatus(): Int
}

class AndroidThermalStatusProvider(context: Context, private val elapsedRealtime: () -> Long = { SystemClock.elapsedRealtime() }) :
  ThermalStatusProvider {
  private val appContext = context.applicationContext

  override fun currentStatus(): Int {
    val powerManager = appContext.getSystemService(PowerManager::class.java)
    return powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
  }

  /**
   * Current thermal headroom (0.0 cool to 1.0 at severe-throttle threshold), or `Float.NaN` if unavailable; the API returns NaN when unsupported,
   * called more than once per second, or before the first sample is ready. No permission required (API 30+).
   */
  fun currentHeadroom(): Float {
    val powerManager = appContext.getSystemService(PowerManager::class.java) ?: return Float.NaN
    return runCatching { powerManager.getThermalHeadroom(0) }.getOrDefault(Float.NaN)
  }

  @Volatile private var lastHeadroom = Float.NaN
  @Volatile private var lastHeadroomAt = 0L

  /**
   * [currentHeadroom] throttled to ~1 Hz and holding the last good value. `getThermalHeadroom` returns `NaN` when polled faster than once per second,
   * but the progress UI re-renders on high-frequency runner ticks; so a raw per-render read would flicker between a value and `NaN`. This reads at
   * most once per [HEADROOM_MIN_INTERVAL_MS] and keeps the last non-`NaN` sample, returning `NaN` only until the first real reading arrives. Uses
   * [elapsedRealtime] for monotonic timing.
   */
  fun cachedHeadroom(): Float {
    val now = elapsedRealtime()
    if (now - lastHeadroomAt >= HEADROOM_MIN_INTERVAL_MS) {
      lastHeadroomAt = now
      val fresh = currentHeadroom()
      if (!fresh.isNaN()) lastHeadroom = fresh
    }
    return lastHeadroom
  }

  companion object {
    private const val HEADROOM_MIN_INTERVAL_MS = 1_000L

    fun statusLabel(status: Int): String =
      when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "nominal"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> "unknown"
      }

    fun displayStatusLabel(status: Int): String =
      when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "Normal"
        else -> statusLabel(status).replaceFirstChar { it.titlecase() }
      }
  }
}
