package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Step-gating logic for the 3-step new-job wizard (Chunk 2). The "Next"/"Run" buttons are validation-gated; these pin that logic so a refactor can't,
 * say, let a job start with no model or no benchmark selected.
 */
class NewJobWizardTest {

  @Test
  fun threeStepsInOrder() {
    assertEquals(listOf("Models", "Benchmarks", "Review"), NewJobWizard.STEP_TITLES)
    assertEquals(2, NewJobWizard.LAST_STEP)
  }

  @Test
  fun modelsStepRequiresAModel() {
    // step 0 Next is gated on model count, regardless of benchmark count.
    assertFalse(NewJobWizard.canAdvance(0, selectedModelCount = 0, selectedBenchmarkCount = 5))
    assertTrue(NewJobWizard.canAdvance(0, selectedModelCount = 1, selectedBenchmarkCount = 0))
  }

  @Test
  fun benchmarksStepRequiresABenchmark() {
    assertFalse(NewJobWizard.canAdvance(1, selectedModelCount = 1, selectedBenchmarkCount = 0))
    assertTrue(NewJobWizard.canAdvance(1, selectedModelCount = 1, selectedBenchmarkCount = 3))
  }

  @Test
  fun reviewStepDoesNotAdvanceViaNext() {
    // The Review step runs the job (canRun); it never has a "Next".
    assertFalse(NewJobWizard.canAdvance(2, selectedModelCount = 1, selectedBenchmarkCount = 3))
  }

  @Test
  fun runRequiresPlannedCellsAndNotAlreadyRunning() {
    assertFalse(NewJobWizard.canRun(plannedCellCount = 0, isRunning = false))
    assertFalse(NewJobWizard.canRun(plannedCellCount = 5, isRunning = true))
    assertTrue(NewJobWizard.canRun(plannedCellCount = 5, isRunning = false))
  }
}
