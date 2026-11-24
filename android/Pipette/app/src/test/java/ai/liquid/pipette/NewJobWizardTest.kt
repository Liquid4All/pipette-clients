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

  private fun model(name: String, onDisk: Boolean) =
    ModelFile(
      name = name,
      path = "/models/$name",
      sizeBytes = 1,
      hfRepo = "org/repo",
      modelId = LocalStorage.normalizedModelStem(name),
      onDisk = onDisk,
    )

  private val q4OnDisk = model("lfm-Q4_0.gguf", onDisk = true)
  private val q8Offer = model("lfm-Q8_0.gguf", onDisk = false)
  private val otherOffer = model("qwen-Q4_0.gguf", onDisk = false)

  @Test
  fun modelsStepListsDownloadedModelsUnlessAskedOrNoneDownloaded() {
    val offered = listOf(q4OnDisk, q8Offer, otherOffer)
    assertEquals(listOf(q4OnDisk), NewJobWizard.listedModels(offered, showsUndownloaded = false))
    assertEquals(offered, NewJobWizard.listedModels(offered, showsUndownloaded = true))
    assertEquals(listOf(q8Offer, otherOffer), NewJobWizard.listedModels(listOf(q8Offer, otherOffer), showsUndownloaded = false))
  }

  @Test
  fun aSelectedModelThatOnlyTheCatalogOffersNeedsTheUndownloadedList() {
    val offered = listOf(q4OnDisk, q8Offer, otherOffer)
    assertTrue(NewJobWizard.selectionNeedsUndownloaded(setOf("qwen"), offered) { true })
    assertTrue(NewJobWizard.selectionNeedsUndownloaded(setOf("lfm"), offered) { it == "Q8_0" })
    assertFalse(NewJobWizard.selectionNeedsUndownloaded(setOf("lfm"), offered) { it == "Q4_0" })
    assertFalse(NewJobWizard.selectionNeedsUndownloaded(setOf("gone"), offered) { true })
  }

  @Test
  fun projectorsDoNotCountAsDownloadedBaseModels() {
    val projector = model("mmproj-lfm-Q4_0.gguf", onDisk = true)
    assertTrue(NewJobWizard.catalogByDefault(listOf(projector, q8Offer)))
    assertEquals(listOf(q8Offer), NewJobWizard.listedModels(listOf(projector, q8Offer), showsUndownloaded = false))
    assertFalse(NewJobWizard.selectionNeedsUndownloaded(setOf("lfm"), listOf(projector.copy(onDisk = false))) { true })
  }

  @Test
  fun downloadedHintNamesTheQuantsOnDisk() {
    val cases =
      listOf(
        listOf(q8Offer) to null,
        listOf(q4OnDisk) to "Downloaded",
        listOf(q4OnDisk, q8Offer) to "Q4_0 downloaded",
        listOf(model("lfm.gguf", onDisk = true), q8Offer) to "1 of 2 downloaded",
      )
    cases.forEach { (files, hint) ->
      assertEquals(files.joinToString { it.name }, hint, NewJobWizard.downloadedHint(ModelGroup("lfm", "LFM", files)))
    }
  }
}
