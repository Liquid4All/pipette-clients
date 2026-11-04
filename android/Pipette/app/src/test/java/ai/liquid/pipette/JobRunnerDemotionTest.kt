package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.FOREGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_ID
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_TYPE
import ai.liquid.pipette.fakes.JobRunnerFixtures.cell
import ai.liquid.pipette.fakes.JobRunnerFixtures.runner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * How the [JobRunner] cell loop reacts to the `:benchmark` process being demoted off `top-app` mid-cell, over the same fakes and the same same-thread
 * executor as [JobRunnerTest].
 *
 * Split out of [JobRunnerTest] rather than appended to it only because the combined class outgrew the `LargeClass` budget; these cases are part of
 * the same cell loop and read alongside its cancellation cases.
 */
class JobRunnerDemotionTest {
  /**
   * The distinction the whole demotion path exists to draw: a cell measured across a descheduling is FAILED, not CANCELLED. Compare with
   * [JobRunnerTest.cancellationMarksCellsCancelledAndPausesJob], where the identical job shape produces CANCELLED throughout because the user asked
   * for it. Both end PAUSED; only this one blames the cell.
   */
  @Test
  fun aDemotionFailsTheRunningCellAndPausesTheJob() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE), cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    // The run itself SUCCEEDS. That is the case worth pinning: a demotion cancels the run, but the cancel can land in its last moments and the
    // benchmark still return a perfectly well-formed result, which must not be written as if the measurement were sound.
    engine.runInterruption = demotion()

    jobRunner.resume(manifest.jobId)

    assertEquals(JobStatus.PAUSED, manifest.status)
    assertEquals(CellRunStatus.FAILED, manifest.cells[0].runStatus)
    assertTrue(manifest.cells[0].errorMessage.orEmpty(), manifest.cells[0].errorMessage.orEmpty().contains("/top-app -> /foreground"))
    // The job stops rather than measuring the rest of the cells under whatever demoted the process, and the untouched cell stays resumable.
    assertEquals(CellRunStatus.CANCELLED, manifest.cells[1].runStatus)
    assertEquals(0, manifest.completedCells)
    // The point of the whole change: the measurement must not reach disk, from where the sweep would upload it. Without this the test passes even
    // if the interruption check is moved below storage.writePayload, which is precisely the mistake it exists to catch.
    assertTrue("an invalidated measurement must never be written", store.payloadWrites.isEmpty())
  }

  /** A user cancel outranks a demotion that lands in the same window, matching the readiness gate's precedence. */
  @Test
  fun aUserCancelOutranksASimultaneousDemotion() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    engine.onRun = { jobRunner.cancel() }
    engine.runInterruption = demotion()

    jobRunner.resume(manifest.jobId)

    assertEquals(JobStatus.PAUSED, manifest.status)
    assertEquals(CellRunStatus.CANCELLED, manifest.cells.single().runStatus)
  }

  /**
   * The path a real device takes most often, and the one neither test above covered: the watcher cancels the service-side flag, the native run dies
   * on its way out, and the throw carries the native error rather than the reason. `JobRunner` has to re-read `lastRunInterruption` from the catch to
   * recognize it, instead of filing an ordinary failure.
   */
  @Test
  fun aDemotionIsRecognizedWhenTheRunThrowsRatherThanReturning() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE), cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    engine.runInterruption = demotion()
    engine.onRun = { error("benchmark cancelled") }

    jobRunner.resume(manifest.jobId)

    assertEquals(JobStatus.PAUSED, manifest.status)
    assertEquals(CellRunStatus.FAILED, manifest.cells[0].runStatus)
    // The demotion's explanation, not the native "benchmark cancelled" the throw carried.
    assertTrue(manifest.cells[0].errorMessage.orEmpty(), manifest.cells[0].errorMessage.orEmpty().contains("/top-app -> /foreground"))
    assertEquals(CellRunStatus.CANCELLED, manifest.cells[1].runStatus)
  }

  /**
   * The cpuset baseline may only be dropped at a job boundary, so `endJob` must not ride along with the per-cell unload.
   *
   * `JobRunner` unloads twice for different reasons: once from the cell catch on any failure, and once after the loop. If dropping the baseline were
   * wired to the unload, an ordinary cell failure (a readiness timeout, a load error) would clear it mid-job, and the next cell's first sample would
   * become its own normal. A process already demoted at that moment would then look healthy for the rest of the job and be measured on in silence,
   * which is the exact failure the carry-forward baseline exists to prevent.
   *
   * This job fails its first cell for a reason that is not a demotion, so the run continues, and it pins that the whole job produced exactly one
   * `endJob` against more than one unload.
   */
  @Test
  fun theBaselineIsDroppedOnlyAtTheJobBoundaryAndNotOnAPerCellUnload() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/bad.gguf", DECODE_ID, DECODE_TYPE), cell("/m/good.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    // Fails the first cell WITHOUT a demotion, so the loop keeps going and the per-cell unload runs mid-job.
    engine.failLoadForPaths = setOf("/m/bad.gguf")

    jobRunner.resume(manifest.jobId)

    assertEquals(CellRunStatus.FAILED, manifest.cells[0].runStatus)
    assertEquals("the run must continue past a non-demotion failure", CellRunStatus.COMPLETED, manifest.cells[1].runStatus)
    assertTrue("the per-cell unload must still happen: ${engine.unloadCount}", engine.unloadCount > 1)
    assertEquals("the baseline may be dropped once, at the job boundary", 1, engine.endJobCount)
  }

  /**
   * The baseline must be dropped even when the job does not reach the end of its cell loop.
   *
   * `execute` can throw from the manifest and publish calls that sit outside its per-cell try, and returns early when the manifest will not load. If
   * the drop lived at the tail of `execute` those paths would skip it, `:benchmark` would keep the finished job's `top-app` baseline for its whole
   * grace period, and the next job's first cell would be scored against a state its pocket screen has not been promoted back to yet: an innocent cell
   * failed. Pinning it here is what stops the call migrating back out of `run`'s finally.
   */
  @Test
  fun theBaselineIsDroppedEvenWhenTheJobDiesOutsideTheCellHandling() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    // Armed only now, and only from the save after `resume`'s own status write, so the throw lands inside the run rather than before it starts.
    store.saveManifestError = IllegalStateException("simulated store failure outside the per-cell try")
    store.saveManifestErrorAfter = store.saveCount + 1

    // The throw is the point; whether it surfaces here or is swallowed by the executor is not.
    runCatching { jobRunner.resume(manifest.jobId) }

    assertEquals("the job boundary is reached on every exit, not just the tidy one", 1, engine.endJobCount)
  }

  /**
   * The window every doc in this change calls the important one: the screen goes off during the model load, not during a rep.
   *
   * The load throws, so the demotion arrives through the catch path rather than the success check, and the cell must still be filed as a demotion
   * rather than as an ordinary load failure. `failedModelPaths` would otherwise record the raw native error and skip every later cell using that
   * model, turning one transient screen-off into a whole model's results going missing.
   */
  @Test
  fun aDemotionDuringTheModelLoadFailsTheCellAsADemotion() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE), cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    engine.failLoadForPaths = setOf("/m/model.gguf")
    engine.runInterruption = demotion()

    jobRunner.resume(manifest.jobId)

    assertEquals(CellRunStatus.FAILED, manifest.cells[0].runStatus)
    val message = manifest.cells[0].errorMessage.orEmpty()
    assertTrue(message, message.contains("/top-app -> /foreground"))
    assertTrue("the raw load error must not be what the user sees", !message.contains("fake load failure"))
    // And the job stops rather than carrying on under whatever demoted the process.
    assertEquals(CellRunStatus.CANCELLED, manifest.cells[1].runStatus)
    assertEquals(JobStatus.PAUSED, manifest.status)
  }

  /**
   * A demotion on the last cell leaves nothing to resume, so the job is COMPLETED with that cell FAILED rather than PAUSED.
   *
   * PAUSED would be the friendlier-looking status, and an earlier version wrote it, but a resume runs PENDING cells and revives CANCELLED ones, and
   * the demoted cell is FAILED. The Resume it advertised therefore ran nothing and wrote the job straight back to COMPLETED. The failed cell is what
   * carries the information, via its error message.
   */
  @Test
  fun aDemotionOnTheFinalCellCompletesWithAFailedCellRatherThanAdvertisingAnEmptyResume() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", DECODE_ID, DECODE_TYPE)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    engine.runInterruption = demotion()

    jobRunner.resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertEquals(CellRunStatus.FAILED, manifest.cells.single().runStatus)
    assertTrue(manifest.cells.single().errorMessage.orEmpty(), manifest.cells.single().errorMessage.orEmpty().contains("/top-app -> /foreground"))
  }

  /** A `/top-app` process on all 8 cores, moved to `/foreground` on a set that excludes the prime tier (2-4), so `excludesTopTier` is consistent. */
  private fun demotion(): CpusetDemotion =
    CpusetDemotion(
      baseline = snapshot(TOP_APP, allowed = "0-7", allowedCount = 8),
      observed = snapshot(FOREGROUND, allowed = "0-1,5-7", allowedCount = 5, excludesTopTier = true),
      reason = CpusetDemotion.Reason.LEFT_TOP_APP,
    )
}
