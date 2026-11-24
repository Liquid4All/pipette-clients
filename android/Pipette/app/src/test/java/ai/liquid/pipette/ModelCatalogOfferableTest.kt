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
  private fun preset(repo: String, quant: String, sizeBytes: Long) =
    PresetModel(id = "$repo:$quant", name = repo.substringAfterLast('/'), identifier = "$repo:$quant", sizeBytes = sizeBytes)

  private fun onDisk(repo: String, filename: String, bytes: Long) =
    ModelFile(name = filename, path = "/data/models/$repo/$filename", sizeBytes = bytes, hfRepo = repo, modelId = null)

  @Test
  fun `presets are offered on a device with nothing downloaded`() {
    val catalogEntry = preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", 149_080_928L)
    val offers = ModelCatalog.offerable(downloaded = emptyList(), presets = listOf(catalogEntry))

    assertEquals(1, offers.size)
    assertEquals(false, offers.single().onDisk)
    assertEquals(catalogEntry.sizeLabel, offers.single().sizeFormatted)
    // Offered *because* it can be fetched: the reference is what the runner will act on.
    assertNotNull(offers.single().modelRef)
  }

  @Test
  fun `a downloaded model wins over its catalog entry`() {
    val presets = listOf(preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", 700_000_000L))
    val local = onDisk("LiquidAI/LFM2-1.2B-GGUF", "LFM2-1.2B-Q4_K_M.gguf", bytes = 742_000_000L)

    val offers = ModelCatalog.offerable(downloaded = listOf(local), presets = presets)

    assertEquals(1, offers.size)
    val only = offers.single()
    assertTrue("the on-disk form must win, since only it knows the real size and path", only.onDisk)
    assertEquals(742_000_000L, only.sizeBytes)
    assertEquals(local.path, only.path)
  }

  @Test
  fun `a downloaded model with no catalog entry is still offered`() {
    val local = onDisk("SomeOrg/Custom-GGUF", "custom-q4_k_m.gguf", bytes = 1L)

    val offers = ModelCatalog.offerable(downloaded = listOf(local), presets = emptyList())

    assertEquals(listOf(local), offers)
  }

  @Test
  fun `catalog quant survives when the weights are downloaded`() {
    val repo = "LiquidAI/Test-GGUF"
    val filename = "Model.Q4_K_M.gguf"
    val preset = PresetModel("test", "Test", "$repo/$filename", sizeBytes = null, quantLabel = "Q4_K_M")
    val installed = onDisk(repo, filename, bytes = 100L)

    assertEquals("Q4_K_M", ModelCatalog.offerable(emptyList(), listOf(preset)).single().quant)
    assertEquals(installed.copy(catalogQuant = "Q4_K_M"), ModelCatalog.offerable(listOf(installed), listOf(preset)).single())
  }

  @Test
  fun `distinct quants of one family are separate offers`() {
    val presets = listOf(preset("LiquidAI/LFM2-1.2B-GGUF", "Q4_K_M", 700_000_000L), preset("LiquidAI/LFM2-1.2B-GGUF", "Q8_0", 1_300_000_000L))

    assertEquals(2, ModelCatalog.offerable(downloaded = emptyList(), presets = presets).size)
  }

  /** A preset whose identifier does not yield a coordinate could not be fetched, so offering it would only produce a run that fails at cell one. */
  @Test
  fun `a preset with no resolvable coordinate is not offered`() {
    val broken = PresetModel(id = "broken", name = "Broken", identifier = "not-a-repo", sizeBytes = 1_000_000_000L)

    assertTrue(ModelCatalog.offerable(downloaded = emptyList(), presets = listOf(broken)).isEmpty())
  }

  @Test
  fun `the fallback list is entirely offerable`() {
    val offers = ModelCatalog.offerable(downloaded = emptyList(), presets = ModelTemplateCatalog.fallback)

    assertEquals(ModelTemplateCatalog.fallback.size, offers.size)
    assertTrue("every offer needs a reference to fetch from", offers.all { it.modelRef != null })
  }

  @Test
  fun `every fallback row keeps its declared size when offered`() {
    ModelTemplateCatalog.fallback.forEach { preset ->
      assertEquals("${preset.identifier} changed size", preset.sizeBytes, preset.asOffer()?.sizeBytes)
    }
  }
}
