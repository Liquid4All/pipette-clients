package ai.liquid.pipette.headless

import ai.liquid.pipette.BenchmarkCatalog
import ai.liquid.pipette.BenchmarkDefinition
import ai.liquid.pipette.ModelFile

/**
 * Turning a parsed run request into the models and benchmarks a job is planned against.
 *
 * Lifted out of [HeadlessRunActivity] so that class is what its own documentation says it is, routing: check the caller, parse, hand off, log. The
 * selection rules here are the part with reasons behind them and no Android types in them, which is also what makes them testable without an
 * emulator.
 */
object HeadlessRunPlan {

  /** Non-mmproj models matching [query] as a case-insensitive substring of the file name, or all of them when null. */
  fun selectModels(available: List<ModelFile>, query: String?): List<ModelFile> {
    val base = available.filterNot { it.isMmproj }
    return if (query == null) base else base.filter { it.name.contains(query, ignoreCase = true) }
  }

  sealed interface BenchmarkSelection {
    data class Resolved(val definitions: List<BenchmarkDefinition>) : BenchmarkSelection

    data class Unknown(val ids: List<String>) : BenchmarkSelection

    data object Empty : BenchmarkSelection
  }

  /**
   * Explicit ids resolve against the catalog and an unrecognized one is a usage error, not a silent drop: a run that quietly benchmarked four of the
   * five cells a CI script asked for would report success.
   */
  fun selectBenchmarks(ids: List<String>): BenchmarkSelection {
    val selectable = BenchmarkCatalog.selectable
    if (ids.isEmpty()) return if (selectable.isEmpty()) BenchmarkSelection.Empty else BenchmarkSelection.Resolved(selectable)
    val resolved = ids.map { it to BenchmarkCatalog.resolve(it) }
    val unknown = resolved.filter { it.second == null }.map { it.first }
    return if (unknown.isNotEmpty()) BenchmarkSelection.Unknown(unknown) else BenchmarkSelection.Resolved(resolved.mapNotNull { it.second })
  }
}
