package ai.liquid.pipette

/** The models of one catalog family, such as `LFM2.5`. The model lists show a family with more than one member under a header. */
data class ModelFamily<T>(val name: String, val members: List<T>) {
  companion object {
    /**
     * The family of a model with no family from the catalog: the text of [name] before its first hyphen. A name with no hyphen, or with nothing
     * before it, is its own family.
     */
    fun defaultFamily(name: String): String = name.substringBefore('-').ifEmpty { name }

    /** The family of a model: the family of its catalog row, found by [modelId], or else [defaultFamily] of [name]. */
    fun of(modelId: String?, name: String, familyByModelId: Map<String, String>): String = modelId?.let { familyByModelId[it] } ?: defaultFamily(name)

    /**
     * [items] grouped by [family]. Sort families and members by name without regard to case. If names differ only in case, compare the original
     * names.
     */
    fun <T> grouping(items: List<T>, family: (T) -> String, name: (T) -> String): List<ModelFamily<T>> =
      items
        .groupBy(family)
        .map { (familyName, members) -> ModelFamily(familyName, members.sortedWith(compareBy(byName, name))) }
        .sortedWith(compareBy(byName) { it.name })

    private val byName: Comparator<String> = String.CASE_INSENSITIVE_ORDER.thenComparator { a, b -> a.compareTo(b) }
  }
}
