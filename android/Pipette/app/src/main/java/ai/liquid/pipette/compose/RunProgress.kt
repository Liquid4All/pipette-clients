// Time/percent thresholds are inline literals (MagicNumber), as in the screens this was extracted from.
@file:Suppress("MagicNumber", "ReturnCount")

package ai.liquid.pipette.compose

import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.RunnerState

/** Shared run-progress math, so the detail page, the pocket card, and the shell don't each carry a copy. */
object RunProgress {
  /**
   * Overall manifest progress for the bar (iOS progressFraction): (cells completed + the running cell's fraction) / total cells, so the bar climbs
   * across cells instead of resetting each cell. The running-cell fraction only counts when [runnerState] is running THIS manifest.
   */
  fun manifestFraction(manifest: JobManifest, runnerState: RunnerState): Double {
    if (manifest.totalCells <= 0) return 0.0
    val within = if (runnerState.runningJobId == manifest.jobId) runnerState.currentCellFraction.coerceIn(0.0, 1.0) else 0.0
    return ((manifest.completedCells + within) / manifest.totalCells.toDouble()).coerceIn(0.0, 1.0)
  }

  /**
   * Wall-clock estimate from elapsed time and [progressFraction] (which must be run-relative — progress since THIS run started, not the whole
   * manifest). Null until past 2% so the first tick doesn't print a wild number. iOS estimatedTimeLeft.
   */
  fun estimatedTimeLeft(runnerState: RunnerState, progressFraction: Double): String? {
    val startedAt = runnerState.startedAtMillis ?: return null
    if (progressFraction <= 0.02) return null
    val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(1L).toDouble()
    val remainingMs = (elapsedMs / progressFraction - elapsedMs).coerceAtLeast(0.0)
    return if (remainingMs < 60_000) "${(remainingMs / 1000).toInt()}s left" else "${kotlin.math.round(remainingMs / 60_000).toInt()} min left"
  }

  /** Progress of THIS run: (cells finished this run + the running cell's fraction) / cells this run will run. */
  fun runFraction(runnerState: RunnerState, fallback: Double): Double {
    val within = runnerState.currentCellFraction.coerceIn(0.0, 1.0)
    return if (runnerState.totalInRun > 0) {
      ((runnerState.completedInRun + within) / runnerState.totalInRun.toDouble()).coerceIn(0.0, 1.0)
    } else {
      fallback
    }
  }
}
