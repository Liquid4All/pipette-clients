package ai.liquid.pipette.fakes

import ai.liquid.pipette.CancelFlag
import ai.liquid.pipette.ReadinessGate
import ai.liquid.pipette.ReadinessOutcome
import ai.liquid.pipette.ReadinessPolicy

/** Counts the cooldown gate calls JobRunner makes between cells. */
class FakeReadinessGate : ReadinessGate {
  var waitCount = 0
    private set

  /** When set, each wait emits this status once — lets tests exercise the cooling-state path. */
  var statusToEmit: String? = null

  /** Elapsed millis reported alongside [statusToEmit], as a real gate would report for its own wait. */
  var elapsedToEmit: Long = 0L

  /** What each wait reports. Defaults to [ReadinessOutcome.Ready]; set it to exercise a gate that gives up on a hot device. */
  var outcomeToReturn: ReadinessOutcome = ReadinessOutcome.Ready

  /** Deliberately not the real gate's deadline, so a test asserting on the recorded policy fails if the payload hard-codes a constant. */
  override val policy: ReadinessPolicy = ReadinessPolicy(maxWaitSecs = 42L, skipThermal = false)

  override fun waitUntilReady(cancelFlag: CancelFlag, onStatus: (String, Long) -> Unit): ReadinessOutcome {
    waitCount++
    statusToEmit?.let { onStatus(it, elapsedToEmit) }
    return outcomeToReturn
  }
}
