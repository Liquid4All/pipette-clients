package ai.liquid.pipette.compose

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** [QuantFilter]: the pills come from the quants the models offer, and an empty selection is "All quants". */
class QuantFilterTest {
  @Test
  fun `options are the distinct quants the models offer`() {
    assertEquals(listOf("PQ2_0", "Q4_0", "Q4_K_M", "Q8_0"), QuantFilter.options(listOf("Q4_K_M", "q4_k_m", null, "Q8_0", "pq2_0", "Q4_0")))
  }

  @Test
  fun `all quants matches every model and a selection matches only its quants`() {
    assertTrue(QuantFilter.matches(emptySet(), null))
    assertTrue(QuantFilter.matches(setOf("Q8_0"), "q8_0"))
    assertFalse(QuantFilter.matches(setOf("Q8_0"), "Q4_0"))
    assertFalse(QuantFilter.matches(setOf("Q8_0"), null))
  }

  @Test
  fun `toggling a quant selects it, and all quants clears the selection`() {
    val selection = linkedSetOf<String>()
    QuantFilter.toggle(selection, "Q8_0", checked = true)
    QuantFilter.toggle(selection, "Q4_0", checked = true)
    QuantFilter.toggle(selection, "Q4_0", checked = false)
    assertEquals(setOf("Q8_0"), selection)

    QuantFilter.toggle(selection, null, checked = true)
    assertTrue(selection.isEmpty())
  }

  @Test
  fun `chips lead with all quants and mark the selection`() {
    val chips = QuantFilter.chips(listOf("Q4_0", "Q8_0"), setOf("Q8_0"))

    assertEquals(listOf(null, "Q4_0", "Q8_0"), chips.map { it.quant })
    assertEquals(listOf("All quants", "Q4_0", "Q8_0"), chips.map { it.label })
    assertEquals(listOf(false, false, true), chips.map { it.selected })
  }
}
