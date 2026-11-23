package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Test

class ModelFamilyTest {
  @Test
  fun `a catalog snapshot keeps model and logo lookups from the same catalog`() {
    val oldPreset =
      PresetModel(
        "model",
        "Previous model name",
        "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_0.gguf",
        1,
        family = "Previous family",
        familyLogoDigest = "a".repeat(64),
      )
    val oldCatalog = ModelTemplateCatalog.Snapshot(listOf(oldPreset))
    val newPreset = oldPreset.copy(name = "Current model name", family = "Current family", familyLogoDigest = "b".repeat(64))
    val newCatalog = ModelTemplateCatalog.Snapshot(listOf(newPreset))

    for ((catalog, preset) in listOf(oldCatalog to oldPreset, newCatalog to newPreset)) {
      val offered = ModelCatalog.offerable(emptyList(), catalog.presets)
      val group = ModelCatalog.groups(offered, catalog.byModelId).single()
      val family = ModelFamily.of(group.key, group.name, catalog.familyByModelId)
      assertEquals(preset.name, group.name)
      assertEquals(preset.family, family)
      assertEquals(preset.familyLogoDigest, catalog.familyLogoDigest(family))
    }
  }

  @Test
  fun `a name without a catalog family takes the text before its first hyphen`() {
    val cases =
      listOf(
        "LFM2.5-1.2B-Instruct" to "LFM2.5",
        "Qwen3.5-2B" to "Qwen3.5",
        "Granite" to "Granite",
        "-leading" to "-leading",
        "LFM 2.5 230M" to "LFM 2.5 230M",
      )
    cases.forEach { (name, family) -> assertEquals(name, family, ModelFamily.defaultFamily(name)) }
  }

  @Test
  fun `families sort by name, ignoring case`() {
    val names = listOf("qwen3-4b", "LFM2.5-350M", "granite-3b", "Bonsai-8B")

    val families = ModelFamily.grouping(names, family = { ModelFamily.defaultFamily(it) }, name = { it })

    assertEquals(listOf("Bonsai", "granite", "LFM2.5", "qwen3"), families.map { it.name })
  }

  @Test
  fun `members of one family sort by name, ignoring case`() {
    val names = listOf("LFM2.5-350M", "LFM2.5-1.2B", "LFM2.5-230m")

    val family = ModelFamily.grouping(names, family = { "LFM2.5" }, name = { it }).single()

    assertEquals(listOf("LFM2.5-1.2B", "LFM2.5-230m", "LFM2.5-350M"), family.members)
  }
}
