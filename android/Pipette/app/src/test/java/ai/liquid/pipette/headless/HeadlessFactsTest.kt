package ai.liquid.pipette.headless

import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.LocalStorage
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Which cells the results walk reports, against a real payload directory.
 *
 * This is the half that broke on device. `results-delete` reported success while `results count=` did not move, because the walk filtered on
 * `CellRunStatus.COMPLETED` alone and clearing a cell's artifacts leaves that status untouched: the cell did complete. The unit tests could not have
 * caught it, and the reason was structural rather than an oversight, since [HeadlessInspectTest] builds `Facts` directly and so covers the wording
 * and never the gathering.
 *
 * So the walk itself is exercised here, with `LocalStorage` writing and removing real payload files, which is the only arrangement in which the
 * regression is reachable off a device.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class HeadlessFactsTest {

  private val storage by lazy { LocalStorage(ApplicationProvider.getApplicationContext()) }

  private fun writePayload(jobId: String, cellId: String, json: String = """{"prefill_throughput":{"value":305.331,"unit":"tok/s"}}""") {
    storage.ensureCellArtifactsDir(jobId, cellId)
    storage.cellPayloadFile(jobId, cellId).writeText(json)
  }

  /** The regression, stated as a test: a cell whose payload has been deleted stops being a result, though its manifest entry still says COMPLETED. */
  @Test
  fun aCellWhosePayloadWasDeletedStopsBeingAResult() {
    val manifest = HeadlessTestFixtures.manifest("job-1", cells = listOf(CellRunStatus.COMPLETED, CellRunStatus.COMPLETED))
    writePayload("job-1", "cell-0")
    writePayload("job-1", "cell-1")

    assertEquals(listOf("cell-0", "cell-1"), HeadlessFacts.results(storage, listOf(manifest)).map { it.cellId })

    storage.clearCellArtifacts("job-1", "cell-1")

    val after = HeadlessFacts.results(storage, listOf(manifest))
    assertEquals("the deleted one is gone and the other is untouched", listOf("cell-0"), after.map { it.cellId })
    assertEquals(CellRunStatus.COMPLETED, manifest.cells.single { it.cellId == "cell-1" }.runStatus)
  }

  /** A completed cell that never wrote a payload was never a result either, which is the same rule seen from the other side. */
  @Test
  fun aCompletedCellWithNoPayloadIsNotAResult() {
    val manifest = HeadlessTestFixtures.manifest("job-2", cells = listOf(CellRunStatus.COMPLETED))

    assertTrue(HeadlessFacts.results(storage, listOf(manifest)).isEmpty())
  }

  /** A cell that never completed is not a result whatever is on disk beside it. */
  @Test
  fun anIncompleteCellIsNotAResultEvenWithAPayload() {
    val manifest = HeadlessTestFixtures.manifest("job-3", cells = listOf(CellRunStatus.FAILED))
    writePayload("job-3", "cell-0")

    assertTrue(HeadlessFacts.results(storage, listOf(manifest)).isEmpty())
  }

  /**
   * A payload that is on disk but unreadable still lists, with no metric.
   *
   * The distinction the predicate turns on, and the reason it tests existence rather than parse-success: `payloadsByCellId` drops a payload it cannot
   * parse, so taking that as the on-disk test would hide this cell from `results` and then leave `results-delete` refusing it as "no stored result",
   * with bytes no verb on this surface could reach.
   */
  @Test
  fun anUnreadablePayloadStillListsSoItsBytesCanBeReclaimed() {
    val manifest = HeadlessTestFixtures.manifest("job-4", cells = listOf(CellRunStatus.COMPLETED))
    writePayload("job-4", "cell-0", json = "{ this is not json")

    val results = HeadlessFacts.results(storage, listOf(manifest))

    assertEquals(listOf("cell-0"), results.map { it.cellId })
    assertEquals("no metric can be derived from it", null, results.single().metricValue)
  }

  /** [HeadlessFacts.isReportable] on its own, which is the rule the walk above applies. */
  @Test
  fun reportableIsCompletedAndOnDisk() {
    val cell = HeadlessTestFixtures.cell("cell-0", CellRunStatus.COMPLETED)

    assertTrue("a payload on disk makes it a result", HeadlessFacts.isReportable(cell, setOf("cell-0")))
    assertFalse("its payload deleted, it is not", HeadlessFacts.isReportable(cell, emptySet()))
    assertFalse(
      "never completed, it never was",
      HeadlessFacts.isReportable(HeadlessTestFixtures.cell("cell-0", CellRunStatus.FAILED), setOf("cell-0")),
    )
  }
}
