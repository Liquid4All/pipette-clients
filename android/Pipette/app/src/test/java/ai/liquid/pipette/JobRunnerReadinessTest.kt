package ai.liquid.pipette

import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.FakeReadinessGate
import ai.liquid.pipette.fakes.FakeResultSubmitter
import ai.liquid.pipette.fakes.JobRunnerFixtures
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The readiness policy belongs to the job, and [JobRunner] is where that scoping is enforced: the manifest carries it, and every run start installs
 * the manifest's copy.
 *
 * These assert the funnel rather than the gate — [ReadinessTest] already covers a gate honouring a deadline it was handed. What is worth pinning here
 * is that *every* start hands one over, including a resume, because the bug this guards against is one run's override governing another's cells,
 * which measures and publishes under a policy nobody asked for.
 */
class JobRunnerReadinessTest {

  private val applied = mutableListOf<RunReadiness>()
  private val store = FakeJobStore()

  private fun runner() =
    JobRunner(
      storage = store,
      engine = FakeBenchmarkEngine(resultJson = "{\"ok\":true}", commit = "abc123"),
      submissionService = FakeResultSubmitter(),
      readiness = FakeReadinessGate(),
      executor = JobRunnerFixtures.directExecutor,
      host = HostHooks(applyReadiness = { applied += it }),
    ) {}

  /** A job whose one cell runs to completion on the same thread, so the next start is not refused by the running-job check. */
  private fun JobRunner.start(readiness: RunReadiness = RunReadiness.DEFAULT) =
    startNewJob(
      models = listOf(ModelFile(name = "m", path = "/m/model.gguf", sizeBytes = 1, hfRepo = null)),
      mmprojFiles = emptyList(),
      benchmarks = listOf(BenchmarkDefinition.PrefillThroughput(BenchmarkId.parse("prefill_throughput_512"), 512)),
      selectedMmprojPaths = emptySet(),
      nGpuLayers = 99,
      contextSize = 4096,
      prefillBatch = 512,
      contributeResults = false,
      readiness = readiness,
    )

  @Test
  fun aRunCarriesItsOwnReadinessToTheGates() {
    val jobId = runner().start(RunReadiness(maxWaitSecs = 1800, skipThermal = true))

    assertEquals(listOf(RunReadiness(maxWaitSecs = 1800, skipThermal = true)), applied)
    // On the manifest, not just applied in passing: that is what makes the next run of this job gate the same way.
    assertEquals(RunReadiness(maxWaitSecs = 1800, skipThermal = true), store.loadJobManifest(jobId)?.readiness)
  }

  @Test
  fun aRunThatNamesNothingAppliesTheDefaultsRatherThanInheriting() {
    // A headless run sets 1800, then a New Job tap starts a run that named nothing. The second start must apply anyway — with DEFAULT — so the gates
    // are reset. A funnel that applied only when the caller had an opinion would leave 1800 in force.
    val runner = runner()
    runner.start(RunReadiness(maxWaitSecs = 1800))
    runner.start()

    assertEquals(listOf(RunReadiness(maxWaitSecs = 1800), RunReadiness.DEFAULT), applied)
  }

  @Test
  fun aResumeAppliesTheResumedJobsPolicyNotWhicheverRanLast() {
    // The regression: job A is authored with a 60 s budget and paused; an unrelated run then installs its own policy. Resuming A must put A's budget
    // back, or its remaining cells are measured — and their results recorded — under a deadline that job never asked for.
    val runner = runner()
    val paused = runner.start(RunReadiness(maxWaitSecs = 60))
    store.loadJobManifest(paused)!!.let { manifest ->
      manifest.cells.forEach { it.runStatus = CellRunStatus.PENDING }
      manifest.status = JobStatus.PAUSED
      store.saveJobManifest(manifest)
    }
    runner.start(RunReadiness(maxWaitSecs = 1800))
    applied.clear()

    runner.resume(paused)

    assertEquals(listOf(RunReadiness(maxWaitSecs = 60)), applied)
  }

  @Test
  fun theJobsPolicySurvivesTheManifestJson() {
    // What makes a resume days later — a fresh process, reading the file back — gate the way the job was authored to.
    val manifest = manifest(RunReadiness(maxWaitSecs = 60, skipThermal = true))

    assertEquals(RunReadiness(maxWaitSecs = 60, skipThermal = true), JobManifest.fromJson(manifest.toJson()).readiness)
    // The wire spellings plan-types uses for the same block, so a manifest and a plan read the same.
    assertEquals(60L, manifest.toJson().getJSONObject("readiness").getLong("max_wait_secs"))
    assertEquals(true, manifest.toJson().getJSONObject("readiness").getBoolean("skip_thermal"))
    // A waiver on its own: the unnamed deadline has to come back unnamed rather than as a zero the gate would treat as an expired budget.
    val waiverOnly = manifest(RunReadiness(skipThermal = true))
    assertEquals(RunReadiness(skipThermal = true), JobManifest.fromJson(waiverOnly.toJson()).readiness)
    assertEquals(false, waiverOnly.toJson().getJSONObject("readiness").has("max_wait_secs"))
  }

  /**
   * A deadline outside the range the headless parser accepts reads back as unnamed.
   *
   * The parser is not the only way in: this reads a file, and a hand-edited or truncated one can hold a `0`, a negative, or a value `optLong` coerced
   * to `0`. Any of those would arm a budget that expires before the gate's first poll, failing every cell of the job with a readiness timeout for a
   * wait that never happened.
   */
  @Test
  fun anUnusableStoredDeadlineReadsBackAsUnnamed() {
    val stored = { value: Any ->
      JobManifest.fromJson(manifest(RunReadiness.DEFAULT).toJson().put("readiness", JSONObject().put("max_wait_secs", value))).readiness
    }

    assertEquals(RunReadiness.DEFAULT, stored(0))
    assertEquals(RunReadiness.DEFAULT, stored(-1))
    assertEquals(RunReadiness.DEFAULT, stored(RunReadiness.MAX_WAIT_SECS + 1))
    assertEquals(RunReadiness.DEFAULT, stored("not a number"))
    assertEquals(RunReadiness(maxWaitSecs = RunReadiness.MAX_WAIT_SECS), stored(RunReadiness.MAX_WAIT_SECS))
  }

  @Test
  fun aJobThatNamesNoPolicyWritesNoField() {
    // Omitted rather than written as an empty object, so a manifest from before this field round-trips unchanged.
    val manifest = manifest(RunReadiness.DEFAULT)

    assertEquals(false, manifest.toJson().has("readiness"))
    assertEquals(RunReadiness.DEFAULT, JobManifest.fromJson(manifest.toJson()).readiness)
  }

  @Test
  fun aRerunOfSomeCellsAppliesTheJobsPolicyToo() {
    // Same rule for the other two entry points, which reach the runner without a caller stating a policy at all: rerunCells here, and retryFailed,
    // which routes through it.
    val runner = runner()
    val jobId = runner.start(RunReadiness(maxWaitSecs = 60))
    runner.start(RunReadiness(maxWaitSecs = 1800))
    applied.clear()

    runner.rerunCells(jobId, store.loadJobManifest(jobId)!!.cells.map { it.cellId }.toSet())

    assertEquals(listOf(RunReadiness(maxWaitSecs = 60)), applied)
  }

  private fun manifest(readiness: RunReadiness) =
    JobManifest(
      nGpuLayers = 99,
      contextSize = 4096,
      cells = mutableListOf(JobRunnerFixtures.cell("/m/model.gguf", JobRunnerFixtures.DECODE_ID, JobRunnerFixtures.DECODE_TYPE)),
      status = JobStatus.PLANNED,
      readiness = readiness,
    )
}
