package ai.liquid.pipette

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * That a model discovered on disk carries the coordinate it can be re-fetched from.
 *
 * The case this pins is the one a live run surfaced: [JobCell.modelRef] exists so a cell can survive its file being deleted between planning and
 * execution, and without a ref on discovered files the models with no coordinate would have been exactly the ones on disk at planning time, which is
 * to say exactly the ones that case is about. `ModelCatalog.offerable` also lets a downloaded row overwrite its preset offer wholesale, so the ref
 * has to come from the file's own layout rather than from the preset it happens to match.
 */
class ModelStoreRefTest {
  @get:Rule val temp = TemporaryFolder()

  private fun store(): ModelStore = ModelStore(temp.root)

  private fun put(relative: String): File =
    File(temp.root, relative).apply {
      parentFile?.mkdirs()
      writeText("gguf")
    }

  @Test
  fun `a repo-bucketed file carries its coordinate`() {
    put("LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf")

    val ref = store().availableModels().single().modelRef
    val text = (ref as? Model.GgufText)?.model

    assertEquals("LiquidAI/LFM2.5-230M-GGUF", text?.repo?.toString())
    assertEquals("LFM2.5-230M-Q4_0.gguf", text?.filename?.toString())
  }

  /** The Models tab accepts any `org/repo/file.gguf`, so the ref cannot come from a preset lookup: these are the models no preset describes. */
  @Test
  fun `a repo the preset catalog has never heard of still gets a coordinate`() {
    put("some-org/some-repo-GGUF/whatever-q8_0.gguf")

    val text = (store().availableModels().single().modelRef as? Model.GgufText)?.model

    assertEquals("some-org/some-repo-GGUF", text?.repo?.toString())
    assertTrue("the preset catalog must not be consulted", ModelTemplateCatalog.repoToName["some-org/some-repo-GGUF"] == null)
  }

  /** A loose import has no repo, so there is no coordinate to invent: null is the honest answer, and the cell then behaves as it did before refs. */
  @Test
  fun `a file with no repo bucket has no coordinate`() {
    put("loose.gguf")

    assertNull(store().availableModels().single().modelRef)
  }

  /** The pairing `JobRunner.visionRef` composes needs the weights half to carry a text ref; a projector's own ref is unused. */
  @Test
  fun `a projector beside its weights lets a vision pairing resolve`() {
    put("LiquidAI/LFM2-VL-450M-GGUF/LFM2-VL-450M-Q4_0.gguf")
    put("LiquidAI/LFM2-VL-450M-GGUF/mmproj-LFM2-VL-450M-f16.gguf")

    val models = store().availableModels()
    val weights = models.single { !it.isMmproj }
    val projector = models.single { it.isMmproj }
    val vision = (JobRunner.visionRef(weights, projector) as? Model.GgufVision)?.model

    assertEquals("LiquidAI/LFM2-VL-450M-GGUF", vision?.repo?.toString())
    assertEquals("LFM2-VL-450M-Q4_0.gguf", vision?.filename?.toString())
    assertEquals("mmproj-LFM2-VL-450M-f16.gguf", vision?.mmprojFilename?.toString())
  }

  /** A downloaded row overwrites its preset offer in `offerable`, so the coordinate has to survive that overwrite. */
  @Test
  fun `a downloaded model keeps a coordinate after the preset merge`() {
    put("LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf")

    val offered = ModelCatalog.offerable(store().availableModels())
    val onDisk = offered.single { it.onDisk }

    assertTrue("the downloaded row won the collision", onDisk.path.endsWith("LFM2.5-230M-Q4_0.gguf"))
    assertTrue("and kept a coordinate", onDisk.modelRef != null)
  }
}
