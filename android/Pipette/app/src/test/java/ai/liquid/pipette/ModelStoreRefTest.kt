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

  private fun putPackage(relative: String): File =
    File(temp.root, relative).apply {
      mkdirs()
      File(this, OnnxGenaiPackage.CONFIG_FILE).writeText("{}")
      File(this, "model.onnx.data").writeText("weights")
    }

  /** A package is one model: its directory, the files' size, and the coordinate its place under the models directory spells. */
  @Test
  fun `an ONNX package is listed as one model with its coordinate`() {
    val dir = putPackage("microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32")

    val model = store().availableModels().single()
    val onnx = (model.modelRef as? Model.OnnxGenai)?.model

    assertEquals("cpu-int4-rtn-block-32", model.name)
    assertEquals(dir.absolutePath, model.path)
    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx", model.hfRepo)
    assertEquals("{}weights".length.toLong(), model.sizeBytes)

    // The fetcher's record of the commit is not part of the package: the size stays the package's own, and the coordinate gains the pin.
    File(dir, OnnxPackageFetcher.REVISION_FILE).writeText("5f5f794c1c23c9d5ee142af85df02a6cc52d6945")
    val recorded = ModelStore(temp.root).availableModels().single()
    assertEquals("{}weights".length.toLong(), recorded.sizeBytes)
    assertEquals("5f5f794c1c23c9d5ee142af85df02a6cc52d6945", (recorded.modelRef as Model.OnnxGenai).model.revision.toString())
    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx", onnx?.repo?.toString())
    assertEquals("cpu_and_mobile/cpu-int4-rtn-block-32", onnx?.prefix?.toString())
    assertTrue(model.onDisk)
  }

  /** A package at the root of its repo has no prefix, and two of one repo are two models. */
  @Test
  fun `packages are told apart by their directories`() {
    putPackage("org/root-package")
    putPackage("org/many/int4")
    putPackage("org/many/fp16")

    val refs = store().availableModels().map { (it.modelRef as Model.OnnxGenai).model.relativeDir }.toSet()

    assertEquals(setOf("org/root-package", "org/many/int4", "org/many/fp16"), refs)
  }

  /** A package still being fetched holds a `genai_config.json` too, and is not a model until it is renamed into place. */
  @Test
  fun `a staging directory is not listed`() {
    putPackage("org/repo/.int4.5f5f794c1c23${OnnxPackageFetcher.STAGING_SUFFIX}")

    assertTrue(store().availableModels().isEmpty())
  }

  /** A package outside the repo layout has no repo to go back to, so it is not offered as something a run can name. */
  @Test
  fun `a package with no repo in its path is not listed`() {
    putPackage("loose-package")

    assertTrue(store().availableModels().isEmpty())
  }
}
