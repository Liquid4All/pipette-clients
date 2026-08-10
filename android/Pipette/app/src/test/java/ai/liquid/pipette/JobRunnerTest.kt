package ai.liquid.pipette

import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.FakeReadinessGate
import ai.liquid.pipette.fakes.FakeResultSubmitter
import java.util.concurrent.Executor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Exercises the [JobRunner] cell loop entirely off-device, over the fakes from Phase C. A direct (same-thread) executor makes `resume` block until
 * the job finishes, so the assertions run against the final manifest state.
 */
class JobRunnerTest {
  private val directExecutor = Executor { it.run() }
  private val decodeId = "decode_throughput_512_100"
  private val decodeType = "decode_throughput"

  private fun cell(modelPath: String, benchmarkId: String, benchmarkType: String?): JobCell =
    JobCell(
      benchmarkId = benchmarkId,
      benchmarkType = benchmarkType,
      modelPath = modelPath,
      modelName = modelPath.substringAfterLast('/'),
      runStatus = CellRunStatus.PENDING,
    )

  private fun registration(): RegistrationData =
    RegistrationData(
      clientId = "client-1",
      status = "active",
      serverUrl = "https://example.test",
      organization = "Liquid",
      contactEmail = "someone@example.com",
      registeredAt = "2026-06-08T12:00:00Z",
    )

  private fun runner(
    store: FakeJobStore,
    engine: FakeBenchmarkEngine,
    readiness: FakeReadinessGate = FakeReadinessGate(),
    submitter: FakeResultSubmitter = FakeResultSubmitter(),
  ): JobRunner = JobRunner(storage = store, engine = engine, submissionService = submitter, readiness = readiness, executor = directExecutor) {}

  @Test
  fun runsAllPendingCellsToCompletion() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{\"ok\":true}", commit = "abc123")
    val readiness = FakeReadinessGate()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        // The second cell stores no wire type, as a manifest written before the
        // field existed does, so the payload's `?: item.type` fallback is what
        // has to answer for it.
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType), cell("/m/model.gguf", decodeId, null)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine, readiness).resume(manifest.jobId)

    assertTrue(manifest.cells.all { it.runStatus == CellRunStatus.COMPLETED })
    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertEquals(2, engine.loads.size)
    assertTrue(engine.runs.none { it.fresh })
    assertEquals(2, store.payloadWrites.size)
    assertTrue(store.payloadWrites.all { it.runtimeVersion == "abc123" })
    // The recorded readiness policy comes from the gate this run was handed, not
    // from a constant: the fake reports a deadline the real gate never would, so
    // hard-coding Readiness.COOLDOWN_MAX_MILLIS here would fail.
    assertTrue(store.payloadWrites.all { it.readinessPolicy == readiness.policy })
    // ...and both cells resolve a benchmark type, which is what decides whether
    // benchmark_flags is emitted at all. The second stored none, so this pins
    // that `item.type` supplies the answer for a pre-field manifest.
    assertTrue(store.payloadWrites.all { it.benchmarkType == BenchmarkType.DECODE_THROUGHPUT })
    // Each run consults the cooldown gate it was handed.
    assertEquals(2, engine.cooldownCount)
    // The gate is hit twice mid-run (once per cell) plus once between the
    // two cells; never after the last cell.
    assertEquals(3, readiness.waitCount)
    assertTrue(engine.unloadCount >= 1)
  }

  @Test
  fun prefillBatchFlowsToEngineLoad() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    assertEquals(256, engine.loads.single().nUbatch)
  }

  @Test
  fun maxMemoryUsesFreshLoadPath() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        cells = mutableListOf(cell("/m/model.gguf", "max_memory_usage_512", "max_memory_usage")),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    assertEquals(CellRunStatus.COMPLETED, manifest.cells.single().runStatus)
    assertTrue(engine.loads.isEmpty())
    assertEquals(1, engine.runs.size)
    assertTrue(engine.runs.single().fresh)
    // The fresh-load path must carry prefillBatch through to nUbatch, just
    // like the resident-load path asserted in prefillBatchFlowsToEngineLoad.
    assertEquals(256, engine.runs.single().nUbatch)
  }

  @Test
  fun maxMemoryUsesFreshLoadPathWhenTheCellStoresNoWireType() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        // A manifest written before JobCell carried a wire type: `benchmarkType` decodes to null, and the benchmark id is the only remaining
        // evidence of what this cell is. Reading the type straight off the cell answered "not max-memory" and sent it down the load-then-measure
        // path, where the allocation the benchmark exists to measure has already happened before the measurement starts.
        cells = mutableListOf(cell("/m/model.gguf", "max_memory_usage_512", null)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    assertEquals(CellRunStatus.COMPLETED, manifest.cells.single().runStatus)
    assertTrue("a max-memory cell must never be pre-loaded", engine.loads.isEmpty())
    assertTrue("the load has to happen inside the measured run", engine.runs.single().fresh)
  }

  @Test
  fun cancellationMarksCellsCancelledAndPausesJob() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType), cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)
    val jobRunner = runner(store, engine)
    engine.onRun = { jobRunner.cancel() }

    jobRunner.resume(manifest.jobId)

    assertEquals(JobStatus.PAUSED, manifest.status)
    assertEquals(0, manifest.completedCells)
    assertTrue(manifest.cells.all { it.runStatus == CellRunStatus.CANCELLED })
  }

  @Test
  fun missingModelFileFailsCellWithoutLoading() {
    val store = FakeJobStore(unresolvablePaths = setOf("/m/missing.gguf"))
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/missing.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage?.contains("Model file not found") == true)
    assertTrue(engine.loads.isEmpty())
  }

  @Test
  fun readinessTimeoutFailsTheCellAndWritesNoPayload() {
    // PIP-143 end to end. The per-rep gate reports TimedOut, the engine aborts the run as the native kernel's
    // `readiness_gate` does, and the cell lands FAILED with nothing written. Recording a throttled measurement
    // here is the exact defect: those numbers would look ordinary in the warehouse.
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{\"ok\":true}", commit = "abc123")
    val readiness = FakeReadinessGate().apply { outcomeToReturn = ReadinessOutcome.TimedOut("headroom 0.97 after 180s") }
    val manifest =
      JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)))
    store.saveJobManifest(manifest)

    runner(store, engine, readiness).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    // The observed reading has to survive into the recorded error, or the readiness-failure rate can be counted but never explained.
    assertTrue("error names the reading, was ${failed.errorMessage}", failed.errorMessage?.contains("headroom 0.97") == true)
    assertTrue("no measurement persisted", store.payloadWrites.isEmpty())
  }

  @Test
  fun submitsWhenContributeResultsAndRegistered() {
    val registration =
      RegistrationData(
        clientId = "client-1",
        status = "active",
        serverUrl = "https://example.test",
        organization = "Liquid",
        contactEmail = "user@example.test",
        registeredAt = "2026-06-08T12:00:00Z",
      )
    val store = FakeJobStore(registration = registration)
    val engine = FakeBenchmarkEngine()
    val submitter = FakeResultSubmitter()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = true,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine, submitter = submitter).resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertEquals(1, submitter.submissions.size)
    assertEquals("client-1", submitter.submissions.single().clientId)
  }

  /**
   * The point of the per-cell upload: a crash or a low-memory kill partway through a long job must not strand every finished cell's data. Two cells
   * therefore produce two submits, not one at the end.
   *
   * The count also pins the other half: there is no *third* call. The sweep skips itself once nothing is unsent, so a fully successful run doesn't
   * flash "Submitting results..." for a submit that would do nothing.
   */
  @Test
  fun submitsEachCellAsItFinishesRatherThanOnlyAtTheEnd() {
    val registration =
      RegistrationData(
        clientId = "client-1",
        status = "active",
        serverUrl = "https://example.test",
        organization = "Liquid",
        contactEmail = "someone@example.com",
        registeredAt = "2026-06-08T12:00:00Z",
      )
    val store = FakeJobStore(registration = registration)
    val submitter = FakeResultSubmitter()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = true,
        cells = mutableListOf(cell("/m/a.gguf", decodeId, decodeType), cell("/m/b.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, FakeBenchmarkEngine(), submitter = submitter).resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertEquals(2, submitter.submissions.size)
  }

  /**
   * The sweep's reason to exist: when a per-cell upload fails, the backlog still goes out at the end of the run.
   *
   * Also pins the circuit breaker. With the first call failing, cell 2 must NOT retry mid-run, or a management-server outage inserts a full
   * connect/read timeout into every remaining inter-cell gap. So a 2-cell run makes exactly two calls: the failed per-cell attempt, then the sweep.
   */
  @Test
  fun sweepSendsTheBacklogWhenAPerCellUploadFails() {
    val store = FakeJobStore(registration = registration())
    val submitter = FakeResultSubmitter(failFirst = 1)
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = true,
        cells = mutableListOf(cell("/m/a.gguf", decodeId, decodeType), cell("/m/b.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, FakeBenchmarkEngine(), submitter = submitter).resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertEquals(2, submitter.submissions.size)
    // The sweep succeeded, so nothing is left unsent.
    assertTrue(manifest.cells.all { it.serverJobId != null })
  }

  /** An offline device skips the per-cell upload entirely rather than stalling each measurement behind a connect timeout. */
  @Test
  fun skipsPerCellSubmitWhileOffline() {
    val registration =
      RegistrationData(
        clientId = "client-1",
        status = "active",
        serverUrl = "https://example.test",
        organization = "Liquid",
        contactEmail = "someone@example.com",
        registeredAt = "2026-06-08T12:00:00Z",
      )
    val store = FakeJobStore(registration = registration)
    val submitter = FakeResultSubmitter()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = true,
        cells = mutableListOf(cell("/m/a.gguf", decodeId, decodeType), cell("/m/b.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    JobRunner(
        storage = store,
        engine = FakeBenchmarkEngine(),
        submissionService = submitter,
        readiness = FakeReadinessGate(),
        isOnline = { false },
        executor = directExecutor,
      ) {}
      .resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    // Exactly one call, and it is the end-of-run sweep: neither cell attempted an upload mid-run
    // (that would be three). The sweep runs even offline on purpose: the reachability probe protects
    // the gap between two measurements, and once the run is over there is no measurement left to
    // protect. It also reads false on a firewalled or LAN-only network, where the submit would in
    // fact have succeeded.
    assertEquals(1, submitter.submissions.size)
    assertTrue(manifest.cells.all { it.serverJobId != null })
  }

  @Test
  fun doesNotAutoSubmitWhenContributeResultsDisabled() {
    // Gating parity with iOS: a completed job is NOT auto-submitted unless
    // the manifest opted in via contributeResults, even on a registered device.
    val registration =
      RegistrationData(
        clientId = "client-1",
        status = "active",
        serverUrl = "https://example.test",
        organization = "Liquid",
        contactEmail = "user@example.test",
        registeredAt = "2026-06-08T12:00:00Z",
      )
    val store = FakeJobStore(registration = registration)
    val engine = FakeBenchmarkEngine()
    val submitter = FakeResultSubmitter()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = false,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine, submitter = submitter).resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertTrue(submitter.submissions.isEmpty())
  }

  @Test
  fun doesNotAutoSubmitWhenNotRegistered() {
    // Gating parity with iOS: opted-in but unregistered → no submission
    // (the device has no Ed25519 identity to sign the upload).
    val store = FakeJobStore(registration = null)
    val engine = FakeBenchmarkEngine()
    val submitter = FakeResultSubmitter()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        contributeResults = true,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine, submitter = submitter).resume(manifest.jobId)

    assertEquals(JobStatus.COMPLETED, manifest.status)
    assertTrue(submitter.submissions.isEmpty())
  }

  @Test
  fun liveCellLabelIsParamAwareAndUsesModelTail() {
    fun labelCell(definition: BenchmarkDefinition) =
      JobCell(
        benchmarkId = definition.benchmarkId.toString(),
        benchmarkType = definition.benchmarkType,
        modelPath = "/m/x.gguf",
        modelName = "LiquidAI/LFM2.5-350M",
      )

    val decode = BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse(decodeId), 512, 100)
    assertEquals("Decode Throughput · 512→100 tok · LFM2.5-350M", JobRunner.liveCellLabel(labelCell(decode), decode))

    val prefill = BenchmarkDefinition.PrefillThroughput(BenchmarkId.parse("prefill_throughput_512"), 512)
    assertEquals("Prefill Throughput · 512 tok · LFM2.5-350M", JobRunner.liveCellLabel(labelCell(prefill), prefill))

    val vl = BenchmarkDefinition.VlThroughput(BenchmarkId.parse("vl_throughput_256x512_32_128"), 256, 512, 32, 128)
    assertEquals("Vision-Language Throughput · 256×512px · 32 tok · LFM2.5-350M", JobRunner.liveCellLabel(labelCell(vl), vl))

    // No typed definition: fall back to the wire type's display name (still param-free).
    val unknown = JobCell(benchmarkId = decodeId, benchmarkType = decodeType, modelPath = "/m/x.gguf", modelName = "LiquidAI/LFM2.5-350M")
    assertEquals("Decode Throughput · LFM2.5-350M", JobRunner.liveCellLabel(unknown, null))
  }

  @Test
  fun coolingStateIsStampedWhileGateWaitsAndClearedOnCompletion() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{}")
    val readiness = FakeReadinessGate().apply { statusToEmit = "Waiting for device to cool (headroom 0.90, 4s)..." }
    val states = mutableListOf<RunnerState>()
    val manifest =
      JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)))
    store.saveJobManifest(manifest)

    JobRunner(storage = store, engine = engine, submissionService = FakeResultSubmitter(), readiness = readiness, executor = directExecutor) {
        states += it
      }
      .resume(manifest.jobId)

    // The per-rep gate emits a cooling status → some state carries a cooldown anchor.
    assertTrue("expected a cooling state", states.any { it.coolingSinceMillis != null })
    // The terminal "Completed" emit is not cooling — the wash/timer must switch off.
    assertEquals(null, states.last { it.runningJobId != null && it.currentProgressText == "Completed" }.coolingSinceMillis)
  }

  @Test
  fun coolingSentinelOnProgressChannelDrivesCoolingState() {
    val store = FakeJobStore()
    // The service-side gate rides the progress channel tagged with COOLING_PROGRESS_TOTAL.
    val engine = FakeBenchmarkEngine(resultJson = "{}").apply { extraProgress = Triple(0, COOLING_PROGRESS_TOTAL, "cooling via channel") }
    val states = mutableListOf<RunnerState>()
    val manifest =
      JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)))
    store.saveJobManifest(manifest)

    JobRunner(
        storage = store,
        engine = engine,
        submissionService = FakeResultSubmitter(),
        readiness = FakeReadinessGate(),
        executor = directExecutor,
      ) {
        states += it
      }
      .resume(manifest.jobId)

    val cooling = states.firstOrNull { it.currentProgressText == "cooling via channel" }
    assertTrue("sentinel progress should produce a cooling state", cooling?.coolingSinceMillis != null)
    // The sentinel must not be read as measured progress — the cell fraction stays put (0.0).
    assertEquals(0.0, cooling!!.currentCellFraction, 0.0)
  }

  @Test
  fun coolingAnchorTracksTheCurrentGateWaitNotTheFirstOne() {
    // Regression: the anchor used to be latched on the first cooling status and only cleared by a
    // non-cooling publish, so back-to-back gate invocations shared one stamp and the UI counted past
    // its own max ("Cooling 3:54 / 3:00 max"). Each cooling status must re-anchor to the elapsed the
    // reporting gate hands over.
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{}")
    // A gate already 100s into its own wait, as the between-cell gate reports mid-cooldown.
    val readiness = FakeReadinessGate().apply { statusToEmit = "Waiting for device to cool (headroom 0.90, 100s)..." }
    readiness.elapsedToEmit = 100_000L
    val states = mutableListOf<RunnerState>()
    val manifest =
      JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)))
    store.saveJobManifest(manifest)

    val before = System.currentTimeMillis()
    JobRunner(storage = store, engine = engine, submissionService = FakeResultSubmitter(), readiness = readiness, executor = directExecutor) {
        states += it
      }
      .resume(manifest.jobId)
    val after = System.currentTimeMillis()

    val anchor = states.first { it.coolingSinceMillis != null }.coolingSinceMillis!!
    // Anchored ~100s in the past, i.e. now - reported elapsed, not "now".
    assertTrue("anchor should sit ~100s back, was ${before - anchor}ms", anchor in (before - 100_000L)..(after - 100_000L))
  }

  @Test
  fun foregroundHoldHeldForTheRunAndReleasedAfterTheFinalPublish() {
    // The pocket screen puts :benchmark on top, leaving main cached, and the platform kills cached
    // processes for the binder traffic pushPocketProgress generates. The hold is what keeps main out
    // of that bucket, so it must bracket the whole run, including the terminal publish, which is
    // itself mirrored over binder.
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{}")
    val holds = mutableListOf<Boolean>()
    val states = mutableListOf<RunnerState>()
    val manifest =
      JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType)))
    store.saveJobManifest(manifest)

    JobRunner(
        storage = store,
        engine = engine,
        submissionService = FakeResultSubmitter(),
        readiness = FakeReadinessGate(),
        executor = directExecutor,
        host =
          HostHooks(
            setForegroundHold = { active ->
              holds += active
              // Every state published so far must have happened while the hold was still claimed.
              if (!active) assertTrue("hold released before the run finished publishing", states.isNotEmpty())
            }
          ),
      ) {
        states += it
      }
      .resume(manifest.jobId)

    assertEquals(listOf(true, false), holds)
  }

  @Test
  fun coolingCaptionNeverExceedsItsOwnMax() {
    val max = Readiness.COOLDOWN_MAX_MILLIS
    val now = 10_000_000L
    // An anchor older than the budget (a gate sitting a poll past its deadline) must still render at
    // the cap rather than counting on past it.
    assertEquals("Cooling 5:00 / 5:00 max", coolingCaption(now - max - 30_000L, now))
    assertEquals("Cooling 1:40 / 5:00 max", coolingCaption(now - 100_000L, now))
  }
}
