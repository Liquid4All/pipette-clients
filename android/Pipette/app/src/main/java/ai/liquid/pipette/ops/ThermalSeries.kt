package ai.liquid.pipette.ops

data class ThermalReading(val thermalStatus: String? = null, val thermalHeadroom: Float? = null)

/** Collects each repetition's two readings, paired. */
class ThermalSeries {
  private val finishedReps = mutableListOf<Pair<ThermalReading, ThermalReading>>()
  private var openRepStart: ThermalReading? = null

  fun start(reading: ThermalReading) {
    check(openRepStart == null) { "a repetition was started while the previous one was still open" }
    openRepStart = reading
  }

  fun finish(reading: ThermalReading) {
    val start = checkNotNull(openRepStart) { "a repetition ended that was never started" }
    openRepStart = null
    finishedReps.add(start to reading)
  }

  fun take(): List<Pair<ThermalReading, ThermalReading>> {
    check(openRepStart == null) { "the run finished with a repetition still open" }
    val copy = ArrayList(finishedReps)
    finishedReps.clear()
    return copy
  }
}
