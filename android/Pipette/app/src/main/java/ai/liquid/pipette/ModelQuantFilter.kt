package ai.liquid.pipette

/**
 * Quantization filter rows on the "Add models" screen — mirrors the iOS client's `QuantPill`. [ALL] ("All quants") is a convenience that mirrors a
 * fully selected concrete set: when it's on, every specific row is on too, and a candidate file matches regardless of quant. The set is fixed rather
 * than derived, so it must be extended whenever the catalog gains a quant level.
 */
enum class ModelQuantFilter(val label: String, private val quant: String?) {
  ALL("All quants", null),
  Q1_0("q1_0", "Q1_0"),
  Q2_0("q2_0", "Q2_0"),
  Q4_0("q4_0", "Q4_0"),
  Q4_K_M("q4_km", "Q4_K_M"),
  Q5_K_M("q5_km", "Q5_K_M");

  fun matches(candidate: String?): Boolean {
    val target = quant ?: return true
    return candidate?.equals(target, ignoreCase = true) == true
  }

  companion object {
    /** All rows in display order (All quants first). */
    val pills: List<ModelQuantFilter> = entries.toList()

    private val specifics: List<ModelQuantFilter> = entries.filter { it != ALL }

    /** A fully-selected set: [ALL] plus every concrete level (so every checkbox reads as checked). */
    fun allSelection(): Set<ModelQuantFilter> = linkedSetOf(ALL).apply { addAll(specifics) }

    /**
     * True when [candidate] passes the current [selection]. [ALL] matches everything; otherwise the candidate must match a selected concrete level.
     */
    fun matchesSelection(selection: Set<ModelQuantFilter>, candidate: String?): Boolean {
      if (selection.contains(ALL)) return true
      return selection.any { it.matches(candidate) }
    }

    /**
     * Toggle one row, keeping [ALL] consistent with the concrete rows (mirrors iOS `toggledSelection`, `allowsEmpty = true`): toggling [ALL]
     * selects-all or clears-all; toggling a concrete row drops [ALL], flips that row, and re-adds [ALL] only when all concrete rows end up selected.
     */
    fun toggled(selection: Set<ModelQuantFilter>, pill: ModelQuantFilter): Set<ModelQuantFilter> {
      if (pill == ALL) {
        return if (selection.contains(ALL)) emptySet() else allSelection()
      }
      val next = linkedSetOf<ModelQuantFilter>().apply { addAll(selection) }
      next.remove(ALL)
      if (!next.remove(pill)) next.add(pill)
      return if (specifics.all { next.contains(it) }) allSelection() else next
    }
  }
}
