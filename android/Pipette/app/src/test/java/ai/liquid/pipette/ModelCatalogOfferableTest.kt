package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [ModelCatalog.offerable]: what the run screen may be started against.
 *
 * The dedup is the part worth pinning. A model that is both in the catalog and on disk exists in two forms that disagree about size and about whether
 * the path exists yet, so a merge that compared values rather than coordinates would list it twice and let a user pick the one that is about to be
 * re-downloaded.
 */
class ModelCatalogOfferableTest {
  private fun preset(repo: String, quant: String, size: String) =
    PresetModel(id = "$repo:$quant", name = repo.substringAfterLast('/'), detail = "text · $size", identifier = "$repo:$quant")

  private fun onDisk(repo: String, filename: String, bytes: Long) =
    ModelFile(name = filename, path = "/data/models/$repo/$filename", sizeBytes = bytes, hfRepo = repo, familyId = null)

  @Test
  fun `presets are offered on a device with nothing downloaded`() {
    val catalogEntry = preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", "149.1 MB")
    val offers = ModelCatalog.offerable(downloaded = emptyList(), presets = listOf(catalogEntry), packages = emptyList())

    assertEquals(1, offers.size)
    assertEquals(false, offers.single().onDisk)
    assertEquals(catalogEntry.sizeLabel, offers.single().sizeFormatted)
    // Offered *because* it can be fetched: the reference is what the runner will act on.
    assertNotNull(offers.single().modelRef)
  }

  @Test
  fun `a downloaded model wins over its catalog entry`() {
    val presets = listOf(preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", "0.7 GB"))
    val local = onDisk("LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", bytes = 742_000_000L)

    val offers = ModelCatalog.offerable(downloaded = listOf(local), presets = presets, packages = emptyList())

    assertEquals(1, offers.size)
    val only = offers.single()
    assertTrue("the on-disk form must win, since only it knows the real size and path", only.onDisk)
    assertEquals(742_000_000L, only.sizeBytes)
    assertEquals(local.path, only.path)
  }

  @Test
  fun `a downloaded model with no catalog entry is still offered`() {
    val local = onDisk("SomeOrg/Custom-GGUF", "custom-q4_k_m.gguf", bytes = 1L)

    val offers = ModelCatalog.offerable(downloaded = listOf(local), presets = emptyList(), packages = emptyList())

    assertEquals(listOf(local), offers)
  }

  @Test
  fun `distinct quants of one family are separate offers`() {
    val presets = listOf(preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", "0.7 GB"), preset("LiquidAI/LFM2-1.2B-GGUF", "Q8_0", "1.3 GB"))

    assertEquals(2, ModelCatalog.offerable(downloaded = emptyList(), presets = presets, packages = emptyList()).size)
  }

  /** A preset whose identifier does not yield a coordinate could not be fetched, so offering it would only produce a run that fails at cell one. */
  @Test
  fun `a preset with no resolvable coordinate is not offered`() {
    val broken = PresetModel(id = "broken", name = "Broken", detail = "text · 1 GB", identifier = "not-a-repo")

    assertTrue(ModelCatalog.offerable(downloaded = emptyList(), presets = listOf(broken), packages = emptyList()).isEmpty())
  }

  @Test
  fun `the real preset catalog is entirely offerable`() {
    val offers = ModelCatalog.offerable(downloaded = emptyList())

    // Every shipped preset should carry a coordinate; one that does not is a catalog typo this catches. The ONNX rows too, in a build whose
    // engine runs them.
    val onnxRows = if (BuildConfig.ONNX_GENAI_ENGINE) ModelTemplateCatalog.onnxDefaults.size else 0
    assertEquals(ModelTemplateCatalog.defaults.size + onnxRows, offers.size)
    assertTrue("every offer needs a reference to fetch from", offers.all { it.modelRef != null })
  }

  @Test
  fun `every preset keeps its declared size when offered`() {
    ModelTemplateCatalog.defaults.forEach { preset ->
      assertEquals("${preset.identifier} changed size scale", preset.sizeLabel, preset.asOffer()?.sizeFormatted)
    }
  }

  /**
   * The ONNX rows are offered as the packages they are: one model whose path is the directory it will have, pinned to the commit the row declares,
   * with the row's size until it is on disk.
   */
  @Test
  fun `an ONNX row is offered as a pinned package`() {
    val row = ModelTemplateCatalog.onnxDefaults.single()

    val offer = ModelCatalog.offerable(downloaded = emptyList(), presets = emptyList(), onnxEngine = true).single()
    val onnx = (offer.modelRef as Model.OnnxGenai).model

    assertEquals("Phi 3 Mini 4K Instruct", offer.displayName)
    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32", offer.path)
    assertEquals(row.bytes, offer.sizeBytes)
    assertEquals(row.revision, onnx.revision.toString())
    assertTrue(!offer.onDisk)
  }

  /** The package on disk wins over its row, as a downloaded file does, and is one entry rather than two: both are keyed on the directory. */
  @Test
  fun `a downloaded package replaces its ONNX row`() {
    val row = ModelTemplateCatalog.onnxDefaults.single()
    val onDisk = ModelCatalog.packageFile(checkNotNull(row.model), path = "/models/${row.repo}/${row.prefix}", sizeBytes = 42L, onDisk = true)

    val offers = ModelCatalog.offerable(downloaded = listOf(onDisk), presets = emptyList(), onnxEngine = true)

    assertEquals(listOf(onDisk), offers)
  }

  /**
   * A build without the engine offers no ONNX model, from the catalog or from disk: started, each would fail at its first cell, and a package named
   * headlessly would be fetched whole before it did.
   */
  @Test
  fun `a build without the engine offers no ONNX model`() {
    val row = ModelTemplateCatalog.onnxDefaults.single()
    val onDisk = ModelCatalog.packageFile(checkNotNull(row.model), path = "/models/${row.repo}/${row.prefix}", sizeBytes = 42L, onDisk = true)

    assertTrue(ModelCatalog.offerable(downloaded = listOf(onDisk), presets = emptyList(), onnxEngine = false).isEmpty())
  }
}
