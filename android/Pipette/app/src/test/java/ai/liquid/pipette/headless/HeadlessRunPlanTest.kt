package ai.liquid.pipette.headless

import ai.liquid.pipette.ModelFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Which models and benchmarks a headless run is planned against.
 *
 * These rules were inside [HeadlessRunActivity] until this branch, where nothing could reach them without an emulator. That was the stated reason for
 * lifting them out, so they are exercised here rather than left as an extraction with the same coverage as before.
 */
class HeadlessRunPlanTest {

  // `isMmproj` is derived from the file name rather than passed, so a projector is spelled by naming it one.
  private fun model(name: String) = ModelFile(name = name, path = "/models/$name", sizeBytes = 1L, hfRepo = "LiquidAI/x", onDisk = true)

  /**
   * A projector is never a benchmark target.
   *
   * It is weights a vision model loads alongside its own, not a model that can be run, so selecting one would plan a cell that cannot execute. The
   * activity pairs projectors separately.
   */
  @Test
  fun projectorsAreNeverSelected() {
    val available = listOf(model("LFM2.5-350M-Q4_0.gguf"), model("mmproj-model-f16.gguf"))

    assertEquals(listOf("LFM2.5-350M-Q4_0.gguf"), HeadlessRunPlan.selectModels(available, query = null).map { it.name })
    assertTrue(HeadlessRunPlan.selectModels(available, query = "mmproj").isEmpty())
  }

  /**
   * `match=` is a case-insensitive substring, which is the looseness `run` wants and `models-rm` deliberately does not have.
   *
   * See `HeadlessMutate.deleteByName`: a loose handle for choosing what to benchmark costs a wrong measurement, and the same looseness for choosing
   * what to delete costs the models.
   */
  @Test
  fun matchIsACaseInsensitiveSubstring() {
    val available = listOf(model("LFM2.5-230M-Q4_0.gguf"), model("LFM2.5-350M-Q4_0.gguf"))

    assertEquals(listOf("LFM2.5-230M-Q4_0.gguf"), HeadlessRunPlan.selectModels(available, query = "230m").map { it.name })
    assertEquals(2, HeadlessRunPlan.selectModels(available, query = "lfm2.5").size)
    assertTrue(HeadlessRunPlan.selectModels(available, query = "nothing-like-this").isEmpty())
  }

  /**
   * An unrecognized benchmark id is a usage error naming every id it could not resolve, not a silent drop.
   *
   * A run that quietly benchmarked four of the five cells a CI script asked for would report success, and the missing measurement would surface as a
   * gap in a dashboard weeks later.
   */
  @Test
  fun unknownBenchmarkIdsAreAllNamed() {
    val selection = HeadlessRunPlan.selectBenchmarks(listOf("not_a_benchmark", "also_not_one"))

    assertTrue("$selection", selection is HeadlessRunPlan.BenchmarkSelection.Unknown)
    assertEquals(listOf("not_a_benchmark", "also_not_one"), (selection as HeadlessRunPlan.BenchmarkSelection.Unknown).ids)
  }

  /**
   * No ids and an empty catalog is [HeadlessRunPlan.BenchmarkSelection.Empty], which the activity refuses with "benchmark catalog is empty; sync the
   * catalog before running headless".
   *
   * The catalog is process-global and server-synced, so it is empty in a JVM test, which is the same state a freshly installed device is in before
   * its first sync. That is the case worth pinning: the run has to say the catalog is empty rather than start a job with no cells.
   */
  @Test
  fun anEmptyCatalogAndNoIdsIsEmptyRatherThanAnEmptyRun() {
    assertTrue(HeadlessRunPlan.selectBenchmarks(emptyList()) is HeadlessRunPlan.BenchmarkSelection.Empty)
  }

  /** Structured ladder benchmark IDs resolve even when the server-synced catalog is empty. */
  @Test
  fun structuredBenchmarkIdsResolveWithoutCatalogSync() {
    val selection = HeadlessRunPlan.selectBenchmarks(listOf("prefill_throughput_512", "decode_throughput_512_128"))

    assertTrue("$selection", selection is HeadlessRunPlan.BenchmarkSelection.Resolved)
    val resolved = (selection as HeadlessRunPlan.BenchmarkSelection.Resolved).definitions
    assertEquals(listOf("prefill_throughput_512", "decode_throughput_512_128"), resolved.map { it.benchmarkId.value })
  }
}
