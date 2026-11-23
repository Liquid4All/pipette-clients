package ai.liquid.pipette

/** Pure helpers for the results heatmap grids. */
object ResultsGrid {
  fun rowKey(cell: JobCell): String {
    val key = "${CompletedResultsCsvExporter.resultModelGroupKey(cell)}|${CompletedResultsCsvExporter.quantLabel(cell)}"
    val pinnedRepo = cell.modelRef?.repo?.takeIf { it.revision != null }
    return pinnedRepo?.let { "$key|${it.reference}" } ?: key
  }

  fun modelDisplayName(cell: JobCell): String {
    val name = CompletedResultsCsvExporter.modelDisplayName(cell)
    return cell.modelRef?.repo?.revision?.let { "$name (@$it)" } ?: name
  }

  /**
   * Normalized shading intensity in `[0, 1]` for [value] within its column, given the column's [min]/[max]. Direction-aware: when [higherIsBetter] a
   * larger value maps toward `1` (so "brighter = better"); otherwise a smaller value does. A degenerate column (single value, or all equal) maps to
   * `0.5` since there is no spread to rank. This is the correctness core of the grid: latency/memory are lower-is-better and must invert vs
   * throughput.
   */
  fun heatmapIntensity(value: Double, min: Double, max: Double, higherIsBetter: Boolean): Double {
    if (max <= min) return 0.5
    val fraction = (value - min) / (max - min)
    return (if (higherIsBetter) fraction else 1.0 - fraction).coerceIn(0.0, 1.0)
  }
}
