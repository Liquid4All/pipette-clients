package ai.liquid.pipette

import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * [LocalStorage.unsubmittedResultCount] and [LocalStorage.unsubmittedResultCountOnDevice], the number quoted to the user in two places: the job
 * detail's "Submit N Results" button and the Settings sign-out warning, which says how many results the reset is about to destroy permanently. A
 * count that drifts high turns that warning into a false alarm; one that drifts low deletes work the user was never told about.
 *
 * Robolectric rather than a pure unit test, because the count is a filesystem question: a cell is only pending upload when its `payload.json` is
 * actually on disk (see [LocalStorage.submittablePayloadDir]).
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class UnsubmittedResultCountTest {
  private val storage by lazy { LocalStorage(ApplicationProvider.getApplicationContext()) }

  @Test
  fun countsOnlyCellsWhoseResultCouldStillGoUp() {
    // Already uploaded: the manifest carries a server job id.
    val synced = cell("cell-1").apply { serverJobId = "server-1" }
    writePayload("job-1", "cell-1")

    // Genuinely pending: completed, no server id, payload on disk.
    val pending = cell("cell-2")
    writePayload("job-1", "cell-2")

    // Completed with nothing on disk: no payload, so there is nothing to lose.
    val payloadless = cell("cell-3")

    // Never ran, so nothing to submit even though a stray payload exists.
    val notRun = cell("cell-4").apply { runStatus = CellRunStatus.PENDING }
    writePayload("job-1", "cell-4")

    // Not saved: [LocalStorage.unsubmittedResultCount] counts the manifest it is handed, not the store. Only the device-wide total below reads disk.
    val manifest = manifest("job-1", listOf(synced, pending, payloadless, notRun))

    assertEquals(1, storage.unsubmittedResultCount(manifest))
  }

  @Test
  fun theDeviceWideTotalSumsEveryStoredJob() {
    val first = manifest("job-1", listOf(cell("cell-1"), cell("cell-2")))
    val second = manifest("job-2", listOf(cell("cell-3")))
    listOf("job-1" to "cell-1", "job-1" to "cell-2", "job-2" to "cell-3").forEach { (jobId, cellId) -> writePayload(jobId, cellId) }
    storage.saveJobManifest(first)
    storage.saveJobManifest(second)

    assertEquals(3, storage.unsubmittedResultCountOnDevice())
  }

  @Test
  fun theDeviceWideTotalIsZeroWithNoJobs() {
    // What drops the warning sentence from the sign-out dialog entirely.
    assertEquals(0, storage.unsubmittedResultCountOnDevice())
  }

  private fun cell(cellId: String) =
    JobCell(
      cellId = cellId,
      benchmarkId = "bench-$cellId",
      benchmarkType = BenchmarkType.DECODE_THROUGHPUT.wire,
      modelPath = "/models/model.gguf",
      modelName = "model",
      runStatus = CellRunStatus.COMPLETED,
    )

  private fun manifest(jobId: String, cells: List<JobCell>) =
    JobManifest(jobId = jobId, nGpuLayers = 99, contextSize = 4096, cells = cells.toMutableList(), status = JobStatus.COMPLETED)

  /** The one condition `submittablePayloadDir` checks: a `payload.json` in the cell's artifacts directory. */
  private fun writePayload(jobId: String, cellId: String) {
    storage.ensureCellArtifactsDir(jobId, cellId)
    storage.cellPayloadFile(jobId, cellId).writeText("{}")
  }
}
