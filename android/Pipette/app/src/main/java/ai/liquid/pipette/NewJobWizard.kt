package ai.liquid.pipette

/** Pure step model for the new-job wizard (driven by `JobsScreen.renderWizard`). */
object NewJobWizard {
  /** Step titles, in order; also the step count. */
  val STEP_TITLES = listOf("Models", "Benchmarks", "Review")
  val LAST_STEP = STEP_TITLES.lastIndex

  /**
   * Whether the "Next" action is enabled on [step]: step 0 (Models) needs at least one model selected, step 1 (Benchmarks) needs at least one
   * benchmark. The final step (Review) advances via Run, not Next, so it returns false.
   */
  fun canAdvance(step: Int, selectedModelCount: Int, selectedBenchmarkCount: Int): Boolean =
    when (step) {
      0 -> selectedModelCount > 0
      1 -> selectedBenchmarkCount > 0
      else -> false
    }

  /** Whether the Review step's "Run job" action is enabled. */
  fun canRun(plannedCellCount: Int, isRunning: Boolean): Boolean = !isRunning && plannedCellCount > 0
}
