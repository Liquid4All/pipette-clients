package ai.liquid.pipette

import android.content.Context
import android.os.PowerManager
import kotlin.math.min

/**
 * The readiness policy a gate applies, in the shape and units the submission payload records it (`benchmark_flags.readiness`, matching the plan's
 * `readiness.max_wait_secs` / `readiness.skip_thermal`).
 *
 * A gate reports this rather than callers reading [Readiness.COOLDOWN_MAX_MILLIS] themselves: the result has to describe the wait that actually
 * governed the measurement, and a fake gate in a test applies a different one.
 */
data class ReadinessPolicy(val maxWaitSecs: Long, val skipThermal: Boolean)

/**
 * How a readiness wait ended. Named to match the shared native kernel's `ReadinessOutcome` (`crates/pipette-android/src/lib.rs`) and the iOS enum of
 * the same name, because the three of them are one contract: `native/benchmarks.rs`'s `readiness_gate` turns [TimedOut] into
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
   * A nullable String rather than a richer type because this method's signature *is* the JNI contract (the bridge reads it via `jni_sig!`), and it
   * mirrors the null/prefix scheme the sibling `nativeWaitUntilReady` already uses in the other direction.
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
 * Two-tier readiness gate.
 *
 * **Primary:** the native `pipette_readiness` gate (three signals — OS thermal status + CPU-cluster die temperature + instantaneous CPU `%busy`),
 * driven in short slices so cancellation and status reporting stay responsive.
 *
 * **Fallback:** that native gate reads `dumpsys`, `/sys/class/thermal`, and `/proc/stat` — all denied to a sandboxed Android app (no `DUMP`
 * permission; SELinux blocks the sysfs/procfs reads). When a native probe reports its inputs are unreadable, fall back to the thermal signals the app
 * *can* read:
 * 1. [PowerManager.getThermalHeadroom] — a normalized estimate of how close the SoC is to severe throttling (0.0 = comfortable, 1.0 = at the throttle
 *    threshold). This is the granular, no-permission signal that actually tracks die temperature, so it's the primary fallback gate.
 * 2. [PowerManager.getCurrentThermalStatus] — the coarse `NONE…SHUTDOWN` enum, used only when headroom is `NaN` (unsupported, or the first sample
 *    after launch isn't ready yet). It can stay `NONE` while the SoC is ~75 °C, so it's a weak backstop, not the preferred signal.
 *
 * The native gate stays authoritative wherever it can actually read its inputs (the engine CLIs, which run with shell/system access).
 */
class Readiness(
  private val currentThermalStatus: () -> Int,
  private val currentHeadroom: () -> Float = { Float.NaN },
  private val sleepMillis: (Long) -> Unit = { Thread.sleep(it) },
  private val nowMillis: () -> Long = { System.currentTimeMillis() },
  /**
   * Whether the thermal criterion is waived (PIP-434). A supplier rather than a value because the setting lives in DataStore, whose reads are
   * `suspend`, while this is constructed synchronously; callers pass a lambda over an in-memory cached value, so a flipped toggle takes effect at the
   * next gate invocation rather than at the next app launch.
   */
  private val skipThermal: () -> Boolean = { false },
) : ReadinessGate {

  override fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit) =
    waitUntilReady(cancelFlag, onStatus, DEFAULT_MAX_MILLIS, DEFAULT_POLL_MILLIS)

  fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit, maxMillis: Long, pollMillis: Long): ReadinessOutcome {
    if (cancelFlag.isCancelled) return ReadinessOutcome.Cancelled
    val waived = skipThermal()
    val startedAt = nowMillis()
    // The last thing the native gate said it was holding for, so a timeout can report what it saw rather than a placeholder. Only meaningful once a
    // probe has come back NOT_READY; the deadline can also expire on the very first poll, hence the fallback.
    var lastObserved = "no reading from the native gate"
    while (!cancelFlag.isCancelled) {
      when (val probe = nativeProbe(waived)) {
        Probe.Ready -> return ReadinessOutcome.Ready
        Probe.Unavailable -> {
          // Fine-grained native probes can't be read in this process —
          // gate on the app-readable thermal signals for the rest of
          // the wait (headroom, falling back to the status enum).
          // Under the waiver there's nothing left for this tier to wait
          // for: unlike the native gate, which still enforces the load
          // criterion, it only ever gated on thermal. That is a genuine
          // Ready, not an absence of one: the gate it was asked to apply
          // has no criteria left.
          return if (waived) ReadinessOutcome.Ready else waitOnDeviceThermal(cancelFlag, onStatus, startedAt, maxMillis, pollMillis)
        }
        is Probe.NotReady -> lastObserved = probe.reason // readable but still hot, so keep polling the real gate
      }
      val elapsed = nowMillis() - startedAt
      if (elapsed >= maxMillis) return ReadinessOutcome.TimedOut("$lastObserved after ${elapsed / 1000}s")
      onStatus("Waiting for device to settle (${elapsed / 1000}s)...", elapsed)
      sleepCancellable(pollMillis, cancelFlag)
    }
    return ReadinessOutcome.Cancelled
  }

  /**
   * The deadline every production call site uses ([DEFAULT_MAX_MILLIS]), plus whether the thermal criterion was waived.
   *
   * Computed off the same [skipThermal] supplier the gate reads, not a snapshot of it, so the recorded value is the one that governed the
   * measurement: [JobRunner] writes a cell's payload right after the between-cell gate that admitted it, and the toggle is a Settings switch a user
   * would have to flip inside that window to desynchronize them.
   *
   * Waiving is still not the same as the env var being set. `PIPETTE_READINESS_SKIP_THERMAL` can waive the native tier without the app knowing, and
   * that path reports `false` here. It is unreachable for an app started from the launcher, which is what made the toggle necessary, so the two
   * disagree only under an `adb shell` invocation that sets it.
   */
  override val policy: ReadinessPolicy
    get() = ReadinessPolicy(maxWaitSecs = DEFAULT_MAX_MILLIS / 1_000L, skipThermal = skipThermal())

  /** One reading from the native tier. [NotReady] carries the gate's own reason so a [ReadinessOutcome.TimedOut] can quote the last one seen. */
  private sealed interface Probe {
    data object Ready : Probe

    data class NotReady(val reason: String) : Probe

    data object Unavailable : Probe
  }

  // Latched once the native gate is known not to work in this process, so we
  // never invoke it again (each call forks a `dumpsys` subprocess — pure
  // per-rep overhead once we know it'll just report "unavailable"). Two ways it
  // proves unusable here: an `UnsatisfiedLinkError` (UI process — the .so isn't
  // loaded, and we deliberately don't `System.loadLibrary` it, keeping the UI
  // process native-free) or an `"unavailable:"` result (`:benchmark` — the .so
  // is loaded but dumpsys/sysfs/procfs are sandbox-denied). After either, we
  // gate purely on thermal headroom.
  @Volatile private var nativeUnavailable = false

  /**
   * One native readiness reading. The 1 ms budget makes the native gate take a single probe and return (its ready-check runs before the deadline
   * check): `null` → ready; `"unavailable:…"` → probes unreadable here; anything else (`"notready:…"`) → readable but the device is still hot.
   */
  private fun nativeProbe(skipThermal: Boolean): Probe {
    if (nativeUnavailable) return Probe.Unavailable
    val result =
      try {
        nativeWaitUntilReady(1L, skipThermal)
      } catch (_: UnsatisfiedLinkError) {
        nativeUnavailable = true // lib not loaded in this process — never will be
        return Probe.Unavailable
      } catch (_: Throwable) {
        // Any other native failure (e.g. a JNI fault) is treated as a
        // permanent fault and latched too — otherwise we'd re-invoke and
        // re-throw the broken probe on every readiness check instead of
        // falling back cleanly to the thermal-headroom gate.
        nativeUnavailable = true
        return Probe.Unavailable
      }
    return when {
      result == null -> Probe.Ready
      result.startsWith(UNAVAILABLE_PREFIX) -> {
        nativeUnavailable = true // probes denied here — don't keep forking dumpsys
        Probe.Unavailable
      }
      // The prefix is the wire tag, not something to show a user or write into an error message.
      else -> Probe.NotReady(result.removePrefix(NOT_READY_PREFIX))
    }
  }

  /**
   * Fallback gate on the app-readable thermal signals. Prefers thermal headroom (proceed once it's at/below [HEADROOM_READY_MAX], i.e. far enough
   * below the severe-throttle threshold to run a rep without throttling); falls back to the coarse status enum only when headroom is `NaN`
   * (unsupported, or not yet sampled). The headroom value is surfaced in the status text so it can be observed and the threshold tuned per device.
   */
  private fun waitOnDeviceThermal(
    cancelFlag: CancelFlag,
    onStatus: (String, Long) -> Unit,
    startedAt: Long,
    maxMillis: Long,
    pollMillis: Long,
  ): ReadinessOutcome {
    while (!cancelFlag.isCancelled) {
      val headroom = currentHeadroom()
      val ready: Boolean
      val label: String
      if (!headroom.isNaN()) {
        ready = headroom <= HEADROOM_READY_MAX
        label = "headroom ${String.format("%.2f", headroom)}"
      } else {
        val status = currentThermalStatus()
        // Proceed only on an explicit NONE; unknown/negative status
        // (e.g. a failed read returning -1) must keep waiting, not be
        // treated as cool because `-1 <= 0`.
        ready = status == PowerManager.THERMAL_STATUS_NONE
        label = AndroidThermalStatusProvider.statusLabel(status)
      }
      if (ready) return ReadinessOutcome.Ready
      // Measured against the wait's own start, not this tier's: the deadline bounds one gate invocation, and the native tier may have already spent
      // part of it before reporting its probes unreadable.
      val elapsed = nowMillis() - startedAt
      if (elapsed >= maxMillis) return ReadinessOutcome.TimedOut("$label after ${elapsed / 1000}s")
      onStatus("Waiting for device to cool ($label, ${elapsed / 1000}s)...", elapsed)
      sleepCancellable(pollMillis, cancelFlag)
    }
    return ReadinessOutcome.Cancelled
  }

  private fun sleepCancellable(totalMillis: Long, cancelFlag: CancelFlag) {
    var remaining = totalMillis.coerceAtLeast(0L)
    while (remaining > 0 && !cancelFlag.isCancelled) {
      val chunk = min(remaining, SLEEP_CHUNK_MS)
      sleepMillis(chunk)
      remaining -= chunk
    }
  }

  /**
   * Block until ready or [maxWaitMillis] elapses (<= 0 uses the native platform default). Returns null once ready, or a discriminated reason string
   * (`"unavailable:…"` / `"notready:…"`).
   *
   * [skipThermal] waives the thermal criterion only; the native gate still enforces its load criterion, which catches a concurrent benchmark. Passing
   * false leaves the native side at `ThermalGate::Unset` rather than `Enforce`, so `PIPETTE_READINESS_SKIP_THERMAL` still decides when the app has no
   * opinion.
   */
  private external fun nativeWaitUntilReady(maxWaitMillis: Long, skipThermal: Boolean): String?

  companion object {
    private const val SLEEP_CHUNK_MS = 250L

    /**
     * How long the readiness gate waits for the device to cool before giving up on a rep, and the "allowed to cool" deadline the progress UI counts
     * its live "Cooling m:ss / max" timer against.
     *
     * 300 s is the shared cross-platform default (`pipette_readiness::DEFAULT_MAX_WAIT`, which documents the choice; iOS carries it as
     * `readinessTimeoutSeconds`). Kotlin cannot read that constant, so this restates it, and any change here is a divergence from the CLI running on
     * this same phone.
     *
     * It was 180 s while the Android CLI used 600 s, so the same device was held to two deadlines three times apart depending on which binary
     * launched the run (PIP-278). Neither had a measurement behind it. For scale, a typical post-rep cooldown to the entry threshold is ~30-40 s, so
     * this bounds the stuck-hot case rather than normal cooling.
     */
    const val COOLDOWN_MAX_MILLIS = 300_000L

    private const val DEFAULT_MAX_MILLIS = COOLDOWN_MAX_MILLIS
    private const val DEFAULT_POLL_MILLIS = 2_000L
    private const val UNAVAILABLE_PREFIX = "unavailable:"
    private const val NOT_READY_PREFIX = "notready:"

    // Thermal-headroom ceiling for entering a rep. Headroom is 0.0 (cool) →
    // 1.0 (severe-throttle threshold). This must sit ABOVE the device's
    // sustained-load equilibrium, or the gate fights normal operation and
    // stalls the benchmark: on a fan-cooled Pixel 10 Pro Fold the SoC settles
    // at ~0.70 headroom under continuous inference — not throttling (30% from
    // severe), but a 0.70 ceiling held nearly every rep and the run crawled.
    // 0.85 lets the device run at its steady-state load while still holding
    // when it's genuinely close to throttling. Tunable per device — the live
    // value is shown in the cooldown status text.
    private const val HEADROOM_READY_MAX = 0.85f
  }
}

/** Current OS thermal status, for display and the [Readiness] fallback. */
interface ThermalStatusProvider {
  fun currentStatus(): Int
}

class AndroidThermalStatusProvider(context: Context) : ThermalStatusProvider {
  private val appContext = context.applicationContext

  override fun currentStatus(): Int {
    val powerManager = appContext.getSystemService(PowerManager::class.java)
    return powerManager?.currentThermalStatus ?: PowerManager.THERMAL_STATUS_NONE
  }

  /**
   * Current thermal headroom (0.0 cool → 1.0 at severe-throttle threshold), or `Float.NaN` if unavailable — the API returns NaN when unsupported,
   * called more than once per second, or before the first sample is ready. No permission required (API 30+).
   */
  fun currentHeadroom(): Float {
    val powerManager = appContext.getSystemService(PowerManager::class.java) ?: return Float.NaN
    return runCatching { powerManager.getThermalHeadroom(0) }.getOrDefault(Float.NaN)
  }

  @Volatile private var lastHeadroom = Float.NaN
  @Volatile private var lastHeadroomAt = 0L

  /**
   * [currentHeadroom] throttled to ~1 Hz and holding the last good value. `getThermalHeadroom` returns `NaN` when polled faster than ~once/second,
   * but the progress UI re-renders on high-frequency runner ticks — so a raw per-render read would flicker between a value and `NaN`. This reads at
   * most once per [HEADROOM_MIN_INTERVAL_MS] and keeps the last non-`NaN` sample, returning `NaN` only until the first real reading arrives.
   */
  fun cachedHeadroom(): Float {
    val now = System.currentTimeMillis()
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
