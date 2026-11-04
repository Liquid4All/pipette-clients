package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** [ModelCatalog.resolve]: a coordinate as something a run can be planned against, whether or not its bytes are here yet. */
class ModelCatalogResolveTest {
  private val repo = HfRepo.parseSlug("LiquidAI/LFM2.5-230M-GGUF")
  private val filename = GgufFilename.parse("LFM2.5-230M-Q4_0.gguf")

  private fun downloaded(name: String, repoSlug: String?, path: String, size: Long) =
    ModelFile(name = name, path = path, sizeBytes = size, hfRepo = repoSlug, modelRef = null)

  @Test
  fun `an absent coordinate resolves to an offer the runner will fetch`() {
    val resolved = ModelCatalog.resolve(repo, filename, emptyList())

    assertFalse("nothing has this on disk", resolved.onDisk)
    assertEquals("LFM2.5-230M-Q4_0.gguf", resolved.name)
    assertEquals("LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf", resolved.path)
    assertEquals("the runner fetches from this", "LiquidAI/LFM2.5-230M-GGUF", (resolved.modelRef as Model.GgufText).model.repo.toString())
  }

  /** Nothing has measured an absent file, and a run summary should not show a guess as though it were a fact. */
  @Test
  fun `an absent coordinate claims no size`() {
    assertEquals(0L, ModelCatalog.resolve(repo, filename, emptyList()).sizeBytes)
  }

  /**
   * Only the downloaded row knows the real size and the absolute path, and planning against the offer would send the runner through a needless fetch.
   */
  @Test
  fun `a downloaded coordinate resolves to the file on disk`() {
    val onDisk =
      downloaded("LFM2.5-230M-Q4_0.gguf", "LiquidAI/LFM2.5-230M-GGUF", "/models/LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf", 149_080_928L)

    val resolved = ModelCatalog.resolve(repo, filename, listOf(onDisk))

    assertTrue(resolved.onDisk)
    assertEquals("/models/LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf", resolved.path)
    assertEquals(149_080_928L, resolved.sizeBytes)
  }

  @Test
  fun `a different quant of the same repo is not mistaken for it`() {
    val other = downloaded("LFM2.5-230M-Q5_K_M.gguf", "LiquidAI/LFM2.5-230M-GGUF", "/models/LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q5_K_M.gguf", 99L)

    assertFalse("a sibling quant is a different coordinate", ModelCatalog.resolve(repo, filename, listOf(other)).onDisk)
  }

  /**
   * Both halves of a vision pair carry a **text** ref, because `JobRunner.visionRef` composes the vision coordinate from the weights' text ref plus
   * the projector it was paired with. A `GgufVision` ref here would leave that composition with nothing to read.
   */
  @Test
  fun `a projector resolves with a text ref so the pairing can be composed`() {
    val projector = GgufFilename.parse("mmproj-LFM2-VL-450M-f16.gguf")
    val vlRepo = HfRepo.parseSlug("LiquidAI/LFM2-VL-450M-GGUF")
    val weights = ModelCatalog.resolve(vlRepo, GgufFilename.parse("LFM2-VL-450M-Q4_0.gguf"), emptyList())

    val resolved = ModelCatalog.resolve(vlRepo, projector, emptyList())

    assertTrue("the projector is recognized as one", resolved.isMmproj)
    assertTrue(resolved.modelRef is Model.GgufText)
    val composed = JobRunner.visionRef(weights, resolved) as? Model.GgufVision
    assertEquals("mmproj-LFM2-VL-450M-f16.gguf", composed?.model?.mmprojFilename?.toString())
    assertEquals("LFM2-VL-450M-Q4_0.gguf", composed?.model?.filename?.toString())
  }
}
