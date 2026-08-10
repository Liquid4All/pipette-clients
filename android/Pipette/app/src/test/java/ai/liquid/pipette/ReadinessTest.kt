package ai.liquid.pipette

import android.os.PowerManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Exercises [Readiness]'s PowerManager fallback. On the JVM the native lib isn't loaded, so `nativeProbe()` reports UNAVAILABLE and the gate falls
 * back to the injected thermal-status source — the same path taken in the app sandbox, where the native probes are permission-denied.
 */
class ReadinessTest {
  @Test
  fun proceedsWhenThermalStatusNone() {
    var reads = 0
    val readiness =
      Readiness(
        currentThermalStatus = {
          reads++
          PowerManager.THERMAL_STATUS_NONE
        },
        sleepMillis = {},
      )
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertEquals(1, reads) // one read, NONE → proceed
  }

  @Test
  fun waitsWhileThrottlingThenProceeds() {
    val statuses = ArrayDeque(listOf(PowerManager.THERMAL_STATUS_LIGHT, PowerManager.THERMAL_STATUS_MODERATE, PowerManager.THERMAL_STATUS_NONE))
    var slept = 0
    val readiness = Readiness(currentThermalStatus = { statuses.removeFirst() }, sleepMillis = { slept++ })
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertTrue("polled through to NONE", statuses.isEmpty())
    assertTrue("slept while throttling", slept >= 1)
  }

  @Test
  fun stopsAtMaxMillisWhenNeverCools() {
    var clock = 0L
    val readiness =
      Readiness(
        currentThermalStatus = { PowerManager.THERMAL_STATUS_MODERATE }, // never NONE
        sleepMillis = {},
        nowMillis = {
          clock += 60_000L
          clock
        }, // +60 s per read
      )
    // The acceptance case for PIP-143. Returning via the deadline is not enough: the outcome has to say so, because a gate that reports Ready here
    // admits a rep on a device that never cooled and its throttled numbers are recorded as an ordinary result.
    val outcome = readiness.waitUntilReady(CancelFlag(), { _, _ -> }, maxMillis = 120_000L, pollMillis = 1_000L)
    assertTrue("timed out rather than reporting ready, was $outcome", outcome is ReadinessOutcome.TimedOut)
    // And it has to quote what it last saw, since that text becomes the cell's recorded error message.
    assertTrue("names the signal it gave up on, was $outcome", (outcome as ReadinessOutcome.TimedOut).observed.contains("moderate"))
  }

  @Test
  fun timeoutIsDistinguishableFromCancellation() {
    // These two used to be the same value: the gate returned Unit and callers reported `!isCancelled`, which is
    // exactly how a timeout came to look like a proceed. They must not collapse again, because one records a
    // failed cell and the other a cancelled one.
    var clock = 0L
    val neverCools =
      Readiness(
        currentThermalStatus = { PowerManager.THERMAL_STATUS_MODERATE },
        sleepMillis = {},
        nowMillis = {
          clock += 60_000L
          clock
        },
      )
    val timedOut = neverCools.waitUntilReady(CancelFlag(), { _, _ -> }, maxMillis = 120_000L, pollMillis = 1_000L)
    val cancelled =
      Readiness(currentThermalStatus = { PowerManager.THERMAL_STATUS_MODERATE }).waitUntilReady(CancelFlag().apply { cancel() }) { _, _ -> }
    assertTrue("hot deadline is a timeout, was $timedOut", timedOut is ReadinessOutcome.TimedOut)
    assertEquals(ReadinessOutcome.Cancelled, cancelled)
  }

  @Test
  fun readyEncodesAsNullAndTheOthersDoNot() {
    // The JNI contract the native bridge decodes: null means proceed, and nothing else may. A Cancelled that encoded to null would resume a cancelled
    // run; a TimedOut that did would admit the throttled rep this whole change exists to reject.
    assertEquals(null, ReadinessOutcome.Ready.encode())
    assertTrue("cancelled carries its tag", ReadinessOutcome.Cancelled.encode()!!.startsWith(ReadinessOutcome.CANCELLED_PREFIX))
    assertEquals("${ReadinessOutcome.TIMED_OUT_PREFIX}headroom 0.95 after 180s", ReadinessOutcome.TimedOut("headroom 0.95 after 180s").encode())
    // A timeout whose observed text opens with the cancel tag must still decode as a timeout, which is why both variants are tagged rather than one.
    assertTrue(
      "tagged as a timeout regardless of the detail",
      ReadinessOutcome.TimedOut("${ReadinessOutcome.CANCELLED_PREFIX}odd sensor label").encode()!!.startsWith(ReadinessOutcome.TIMED_OUT_PREFIX),
    )
  }

  @Test
  fun cancelledReturnsWithoutProbing() {
    var reads = 0
    val flag = CancelFlag().apply { cancel() }
    val readiness =
      Readiness(
        currentThermalStatus = {
          reads++
          PowerManager.THERMAL_STATUS_MODERATE
        },
        sleepMillis = {},
      )
    assertEquals(ReadinessOutcome.Cancelled, readiness.waitUntilReady(flag) { _, _ -> })
    assertEquals(0, reads)
  }

  @Test
  fun proceedsWhenHeadroomBelowThreshold() {
    var reads = 0
    val readiness =
      Readiness(
        currentThermalStatus = { error("status enum must not be read when headroom is available") },
        currentHeadroom = {
          reads++
          0.2f
        }, // well under the 0.85 ceiling
        sleepMillis = {},
      )
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertEquals(1, reads)
  }

  @Test
  fun waitsWhileHeadroomHighThenProceeds() {
    val headrooms = ArrayDeque(listOf(0.95f, 0.9f, 0.5f)) // hot, hot, then cool (under the 0.85 ceiling)
    var slept = 0
    val readiness =
      Readiness(currentThermalStatus = { PowerManager.THERMAL_STATUS_NONE }, currentHeadroom = { headrooms.removeFirst() }, sleepMillis = { slept++ })
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertTrue("polled until headroom dropped under the ceiling", headrooms.isEmpty())
    assertTrue("slept while throttled", slept >= 1)
  }

  @Test
  fun fallsBackToStatusEnumWhenHeadroomNaN() {
    var statusReads = 0
    val readiness =
      Readiness(
        currentThermalStatus = {
          statusReads++
          PowerManager.THERMAL_STATUS_NONE
        },
        currentHeadroom = { Float.NaN }, // unsupported / not-yet-sampled
        sleepMillis = {},
      )
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertEquals(1, statusReads) // headroom NaN → consult the status enum
  }

  @Test
  fun reportsThePolicyItWaitsOn() {
    val readiness = Readiness(currentThermalStatus = { PowerManager.THERMAL_STATUS_NONE })
    // Literals, not `COOLDOWN_MAX_MILLIS / 1_000L`: this rides the wire as `benchmark_flags.readiness`, so retuning the deadline must fail here.
    assertEquals(ReadinessPolicy(maxWaitSecs = 300L, skipThermal = false), readiness.policy)
  }

  @Test
  fun waiverProceedsThroughAThermalStateThatWouldOtherwiseHold() {
    // The acceptance case for PIP-434, and the reason the waiver had to reach this tier rather than
    // only the JNI one: a device pinned at MODERATE with headroom above the ceiling waits out the
    // full deadline and cancels the cell. Under the waiver it must not consult either signal.
    var slept = 0
    val readiness =
      Readiness(
        currentThermalStatus = { error("status enum must not be read when the thermal criterion is waived") },
        currentHeadroom = { error("headroom must not be read when the thermal criterion is waived") },
        sleepMillis = { slept++ },
        skipThermal = { true },
      )
    // A waived gate has no criteria left to apply, so its Ready is a real verdict rather than an absent one.
    assertEquals(ReadinessOutcome.Ready, readiness.waitUntilReady(CancelFlag()) { _, _ -> })
    assertEquals("returned without waiting", 0, slept)
  }

  @Test
  fun waiverStillHonoursCancellation() {
    // Skipping the thermal criterion must not turn the gate into an unconditional return: a cancel
    // that arrives while a job is tearing down still has to be observed here.
    val flag = CancelFlag().apply { cancel() }
    val readiness = Readiness(currentThermalStatus = { error("must not probe after cancel") }, skipThermal = { true })
    assertEquals(ReadinessOutcome.Cancelled, readiness.waitUntilReady(flag) { _, _ -> })
  }

  @Test
  fun reportsTheWaiverItApplied() {
    // What makes a waived run distinguishable in the warehouse: the policy is computed off the same
    // supplier the gate reads, so a waiver can't apply without being recorded.
    var waived = false
    val readiness = Readiness(currentThermalStatus = { PowerManager.THERMAL_STATUS_NONE }, skipThermal = { waived })
    assertEquals(ReadinessPolicy(maxWaitSecs = 300L, skipThermal = false), readiness.policy)
    waived = true
    assertEquals(ReadinessPolicy(maxWaitSecs = 300L, skipThermal = true), readiness.policy)
  }
}
