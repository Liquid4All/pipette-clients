package ai.liquid.pipette

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Correctness guards for the results heatmap grid (Chunk 3). The grid shades each cell by its value's rank within its column, made direction-aware
 * via the metric's `higherIsBetter`. The risk (G3) is coloring lower-is-better metrics (latency, memory) as if higher were better — these tests pin
 * both the normalization and the per-type directions that feed it.
 */
class ResultsGridTest {

  private val eps = 1e-9

  @Test
  fun higherIsBetterMapsLargestToBrightest() {
    // value at the column max → intensity 1 (brightest); min → 0.
    assertEquals(1.0, ResultsGrid.heatmapIntensity(600.0, 300.0, 600.0, higherIsBetter = true), eps)
    assertEquals(0.0, ResultsGrid.heatmapIntensity(300.0, 300.0, 600.0, higherIsBetter = true), eps)
    assertEquals(0.5, ResultsGrid.heatmapIntensity(450.0, 300.0, 600.0, higherIsBetter = true), eps)
  }

  @Test
  fun lowerIsBetterInvertsSoSmallestIsBrightest() {
    // For latency/memory, the SMALLEST value must be brightest.
    assertEquals(1.0, ResultsGrid.heatmapIntensity(3800.0, 3800.0, 6000.0, higherIsBetter = false), eps)
    assertEquals(0.0, ResultsGrid.heatmapIntensity(6000.0, 3800.0, 6000.0, higherIsBetter = false), eps)
  }

  @Test
  fun degenerateColumnMapsToMidIntensity() {
    // Single value / all-equal column has no spread → 0.5, not a divide-by-zero.
    assertEquals(0.5, ResultsGrid.heatmapIntensity(512.0, 512.0, 512.0, higherIsBetter = true), eps)
    assertEquals(0.5, ResultsGrid.heatmapIntensity(512.0, 512.0, 512.0, higherIsBetter = false), eps)
  }

  @Test
  fun intensityIsClampedToUnitRange() {
    // An out-of-range value (shouldn't happen, but be safe) stays within [0,1].
    val below = ResultsGrid.heatmapIntensity(100.0, 300.0, 600.0, higherIsBetter = true)
    val above = ResultsGrid.heatmapIntensity(900.0, 300.0, 600.0, higherIsBetter = true)
    assertTrue(below in 0.0..1.0)
    assertTrue(above in 0.0..1.0)
  }

  // --- The metric directions the grid relies on (CompletedResultsCsvExporter) ---

  @Test
  fun throughputMetricsAreHigherIsBetter() {
    val prefill =
      CompletedResultsCsvExporter.metric(cell("prefill_throughput_1024", "prefill_throughput"), JSONObject().put("prefill_time_ms", 2000.0))
    assertEquals("tok/s", prefill?.unit)
    assertTrue(prefill!!.higherIsBetter)
    // 1024 tokens / 2.0 s = 512 tok/s.
    assertEquals(512.0, prefill.numericValue, 1e-6)

    val decode =
      CompletedResultsCsvExporter.metric(cell("decode_throughput_1024_100", "decode_throughput"), JSONObject().put("decode_time_ms", 1000.0))
    assertTrue(decode!!.higherIsBetter)
    assertEquals(100.0, decode.numericValue, 1e-6)
  }

  @Test
  fun latencyAndMemoryAreLowerIsBetter() {
    val e2e = CompletedResultsCsvExporter.metric(cell("end_to_end_latency_1024_256", "end_to_end_latency"), JSONObject().put("total_time_ms", 5000.0))
    assertEquals("ms", e2e?.unit)
    assertFalse(e2e!!.higherIsBetter)

    val mem = CompletedResultsCsvExporter.metric(cell("max_memory_usage_1024", "max_memory_usage"), JSONObject().put("max_ram_bytes", 545259520.0))
    assertEquals("bytes", mem?.unit)
    assertFalse(mem!!.higherIsBetter)
  }

  private fun cell(benchmarkId: String, type: String) =
    JobCell(
      benchmarkId = benchmarkId,
      benchmarkType = type,
      modelPath = "/models/LFM2.5-350M-Q4_0.gguf",
      modelName = "LFM2.5-350M-Q4_0.gguf",
      runStatus = CellRunStatus.COMPLETED,
    )
}
