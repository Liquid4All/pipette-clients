package ai.liquid.pipette

/** Pure step model for the new-job wizard (driven by `JobsScreen.renderWizard`). */
object NewJobWizard {
  /** Step titles, in order; also the step count. */
  val STEP_TITLES = listOf("Models", "Benchmarks", "Review")
  val LAST_STEP = STEP_TITLES.lastIndex

  /**
   * Whether the "Next" action is enabled on [step]: step 0 (Models) needs at least one model selected, step 1 (Benchmarks) needs at least one
   * benchmark. The final step (Review) advances via Run, not Next, so it returns false.
   */
  fun canAdvance(step: Int, selectedModelCount: Int, selectedBenchmarkCount: Int): Boolean =
    when (step) {
      0 -> selectedModelCount > 0
      1 -> selectedBenchmarkCount > 0
      else -> false
    }

  /** Whether the Review step's "Run job" action is enabled. */
  fun canRun(plannedCellCount: Int, isRunning: Boolean): Boolean = !isRunning && plannedCellCount > 0

  fun catalogByDefault(offered: List<ModelFile>): Boolean = offered.none { it.onDisk && !it.isMmproj }

  /** List the base models on disk. Also list catalog offers if [showsUndownloaded] is true or no base model is on disk. */
  fun listedModels(offered: List<ModelFile>, showsUndownloaded: Boolean): List<ModelFile> {
    val includeOffers = showsUndownloaded || catalogByDefault(offered)
    return offered.filter { !it.isMmproj && (includeOffers || it.onDisk) }
  }

  /** Return true if switching to downloaded models would hide a selected base file that matches the current quant filter. */
  fun selectionNeedsUndownloaded(selectedKeys: Set<String>, offered: List<ModelFile>, quantMatches: (String?) -> Boolean): Boolean =
    offered.any { !it.isMmproj && !it.onDisk && ModelCatalog.groupKey(it) in selectedKeys && quantMatches(it.quant) }

  /** Return a label for the quants of [group] on disk, such as `Q4_0 downloaded`. Return null if none are on disk. */
  fun downloadedHint(group: ModelGroup): String? {
    val onDisk = group.files.filter { it.onDisk }
    val quants = onDisk.mapNotNull { it.quant }
    return when {
      onDisk.isEmpty() -> null
      onDisk.size == group.files.size -> "Downloaded"
      quants.size < onDisk.size -> "${onDisk.size} of ${group.files.size} downloaded"
      else -> "${quants.distinct().sorted().joinToString(", ")} downloaded"
    }
  }
}
