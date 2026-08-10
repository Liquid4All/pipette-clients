package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelQuantFilterTest {
  private val all = ModelQuantFilter.allSelection()

  @Test
  fun allSelectionContainsEveryPill() {
    assertEquals(ModelQuantFilter.pills.toSet(), all)
  }

  @Test
  fun matchesSelectionWithAllMatchesEverything() {
    assertTrue(ModelQuantFilter.matchesSelection(all, "Q4_0"))
    assertTrue(ModelQuantFilter.matchesSelection(all, "something-else"))
    assertTrue(ModelQuantFilter.matchesSelection(all, null))
  }

  @Test
  fun matchesSelectionEmptyMatchesNothing() {
    assertFalse(ModelQuantFilter.matchesSelection(emptySet(), "Q4_0"))
    assertFalse(ModelQuantFilter.matchesSelection(emptySet(), null))
  }

  @Test
  fun matchesSelectionSpecificIsCaseInsensitiveAndExclusive() {
    val q4 = setOf(ModelQuantFilter.Q4_0)
    assertTrue(ModelQuantFilter.matchesSelection(q4, "Q4_0"))
    assertTrue(ModelQuantFilter.matchesSelection(q4, "q4_0"))
    assertFalse(ModelQuantFilter.matchesSelection(q4, "Q5_K_M"))
    assertFalse(ModelQuantFilter.matchesSelection(q4, null))
  }

  @Test
  fun togglingAllClearsWhenFullySelected() {
    assertEquals(emptySet<ModelQuantFilter>(), ModelQuantFilter.toggled(all, ModelQuantFilter.ALL))
  }

  @Test
  fun togglingAllFromEmptySelectsEverything() {
    assertEquals(all, ModelQuantFilter.toggled(emptySet(), ModelQuantFilter.ALL))
  }

  @Test
  fun togglingASpecificDropsAllAndDeselectsThatRow() {
    val next = ModelQuantFilter.toggled(all, ModelQuantFilter.Q4_0)
    assertFalse(next.contains(ModelQuantFilter.ALL))
    assertFalse(next.contains(ModelQuantFilter.Q4_0))
    assertTrue(next.contains(ModelQuantFilter.Q4_K_M))
    assertTrue(next.contains(ModelQuantFilter.Q5_K_M))
  }

  @Test
  fun togglingASpecificIntoEmptySelectsOnlyThatRow() {
    assertEquals(setOf(ModelQuantFilter.Q4_0), ModelQuantFilter.toggled(emptySet(), ModelQuantFilter.Q4_0))
  }

  @Test
  fun reselectingTheLastSpecificReaddsAll() {
    // From a full selection, drop one concrete row, then re-add it: ALL should come back so every checkbox reads as checked again.
    val oneShort = ModelQuantFilter.toggled(all, ModelQuantFilter.Q5_K_M)
    assertFalse(oneShort.contains(ModelQuantFilter.ALL))
    val full = ModelQuantFilter.toggled(oneShort, ModelQuantFilter.Q5_K_M)
    assertEquals(all, full)
    assertTrue(full.contains(ModelQuantFilter.ALL))
  }
}
