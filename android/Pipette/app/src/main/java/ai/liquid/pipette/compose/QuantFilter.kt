package ai.liquid.pipette.compose

/**
 * The quant pills of a picker, built from the quants that its models offer. A selection holds upper-case quant labels, such as `Q4_K_M` or `PQ2_0`.
 * An empty selection is "All quants", which matches every model, including a model whose quant is unknown.
 */
object QuantFilter {
  private const val ALL_LABEL = "All quants"

  /** One option for each distinct quant in [quants], in upper case and in name order. An unknown quant gives no option. */
  fun options(quants: Iterable<String?>): List<String> = quants.mapNotNull { it?.uppercase() }.distinct().sorted()

  fun matches(selection: Set<String>, quant: String?): Boolean = selection.isEmpty() || quant?.uppercase() in selection

  /** "All quants", then one pill for each option. */
  fun chips(options: List<String>, selection: Set<String>): List<QuantChipUi> =
    listOf(QuantChipUi(quant = null, label = ALL_LABEL, selected = selection.isEmpty())) +
      options.map { QuantChipUi(quant = it, label = it, selected = it in selection) }

  /** Select or clear [quant] in [selection]. A null [quant] is "All quants": selecting it clears the others. */
  fun toggle(selection: MutableSet<String>, quant: String?, checked: Boolean) {
    when {
      quant == null -> if (checked) selection.clear()
      checked -> selection += quant
      else -> selection -= quant
    }
  }
}
