package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.FOREGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.FakeReadinessGate
import ai.liquid.pipette.fakes.FakeResultSubmitter
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_ID
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_TYPE
import ai.liquid.pipette.fakes.JobRunnerFixtures.cell
import ai.liquid.pipette.fakes.JobRunnerFixtures.directExecutor
import ai.liquid.pipette.fakes.JobRunnerFixtures.runner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Exercises the [JobRunner] cell loop entirely off-device, over the fakes from Phase C. A direct (same-thread) executor makes `resume` block until
 * the job finishes, so the assertions run against the final manifest state.
 */
class JobRunnerTest {
  private val decodeId = DECODE_ID
  private val decodeType = DECODE_TYPE

  private fun registration(): RegistrationData =
    RegistrationData(
      clientId = "client-1",
      status = "active",
      serverUrl = "https://example.test",
      organization = "Liquid",
      contactEmail = "someone@example.com",
      registeredAt = "2026-06-08T12:00:00Z",
    )

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

  /**
   * The invariant the resolved-flags path exists for: the values handed to the engine are the values recorded on the payload. Asserted on both
   * branches, since the resident-load and fresh-load paths pass them separately.
   */
  @Test
  fun theEngineLoadsWithTheValuesTheRecordReports() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val authored = RuntimeFlagRef(numberGpuLayers = 12, ctxSize = 777, nUbatch = 64, loadMode = LlamacppLoadMode.MMAP)
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType, runtimeFlags = authored)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    // The cell's own values reach the load, not the job's.
    val load = engine.loads.single()
    assertEquals(12, load.nGpuLayers)
    assertEquals(777, load.contextSize)
    assertEquals(64, load.nUbatch)
    assertEquals(LlamacppLoadMode.MMAP, load.loadMode)
    // And the payload reports exactly those.
    assertEquals(authored, store.payloadWrites.single().runtimeFlags)
  }

  /** The same invariant on the fresh-load branch, which passes the values separately from the resident one. */
  @Test
  fun theFreshLoadBranchAlsoLoadsWithTheValuesTheRecordReports() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val authored = RuntimeFlagRef(numberGpuLayers = 12, ctxSize = 777, nUbatch = 64, loadMode = LlamacppLoadMode.MMAP)
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        cells = mutableListOf(cell("/m/model.gguf", "max_memory_usage_512", "max_memory_usage", runtimeFlags = authored)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    val run = engine.runs.single()
    assertTrue(run.fresh)
    assertEquals(12, run.nGpuLayers)
    assertEquals(777, run.contextSize)
    assertEquals(64, run.nUbatch)
    assertEquals(LlamacppLoadMode.MMAP, run.loadMode)
    assertEquals(authored, store.payloadWrites.single().runtimeFlags)
  }

  /** A cell that authored nothing still records what the engine answered, rather than leaving the knobs unset. */
  @Test
  fun aCellThatAuthoredNothingRecordsTheEnginesAnswers() {
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

    val reported = store.payloadWrites.single().runtimeFlags
    assertEquals(99, reported?.numberGpuLayers)
    assertEquals(256, reported?.nUbatch)
    assertEquals(LlamacppLoadMode.READ, reported?.loadMode)
    assertEquals(LlamacppLoadMode.READ, engine.loads.single().loadMode)
  }

  /**
   * A knob this build cannot honor fails its cell rather than being ignored. Failing one cell, not the job: the runner's per-cell catch is what turns
   * this into a FAILED status the user can see.
   */
  @Test
  fun aCellAuthoringAnUnsupportedKnobFails() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        prefillBatch = 256,
        cells =
          mutableListOf(
            cell("/m/model.gguf", decodeId, decodeType, runtimeFlags = RuntimeFlagRef(refused = mapOf("threads" to 6))),
            cell("/m/other.gguf", decodeId, decodeType),
          ),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    assertEquals(CellRunStatus.FAILED, manifest.cells.first().runStatus)
    assertTrue(manifest.cells.first().errorMessage!!.contains("threads"))
    // Nothing was loaded for the refused cell, and the job carried on to the next one.
    assertEquals(1, engine.loads.size)
    assertEquals(CellRunStatus.COMPLETED, manifest.cells[1].runStatus)
  }

  /**
   * A cell's authored flags survive the job file, which is the only way they ever reach the runner: nothing in the app composes them today, so a
   * hand-written job file is the entry point for the whole authored-flags path.
   */
  @Test
  fun aCellsAuthoredFlagsRoundTripThroughItsJson() {
    val authored = RuntimeFlagRef(numberGpuLayers = 12, ctxSize = 777, nUbatch = 64, loadMode = LlamacppLoadMode.MMAP)
    val restored = JobCell.fromJson(cell("/m/model.gguf", decodeId, decodeType, runtimeFlags = authored).toJson())
    assertEquals(authored, restored.runtimeFlags)
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
    // And the resolved load mode, which only this branch passes (max_memory_usage is the
    // benchmark whose whole point is observing the load).
    assertEquals(LlamacppLoadMode.READ, engine.runs.single().loadMode)
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

  /**
   * Each payload must carry the cpuset reading taken for its own cell.
   *
   * `RemoteBenchmarkEngine` used to cache the first reading for the life of the `:benchmark` process, on the false premise that a process keeps the
   * cpuset group it was assigned. Every cell then reported the first cell's group, so a run that ended in `/moderate` with the prime cores gone still
   * wrote a clean-looking `/top-app` to its payload. A wrong field here is worse than a missing one, because anything auditing results by it reads it
   * as evidence the cell was fine.
   *
   * The cache itself lived behind the binder and is only reachable on a device, so what is pinned here is the contract it violated: one read per
   * cell, and the reading that cell got is the one that lands in its payload.
   */
  @Test
  fun eachCellsPayloadCarriesItsOwnCpusetReadingRatherThanTheFirstCells() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine(resultJson = "{\"ok\":true}", commit = "abc123")
    // Nothing here trips the guard: the demotion verdict is what fails a cell, and this fake reports
    // none. This is the fail-open path, where the payload's own fields are the only account of where
    // the cell ran, which is exactly where a stale reading does its damage.
    engine.cpuAffinityPerCall =
      listOf(snapshot(cpuset = TOP_APP), snapshot(cpuset = FOREGROUND, allowed = "0-1,4-5", allowedCount = 4, excludesTopTier = true))
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells = mutableListOf(cell("/m/model.gguf", decodeId, decodeType), cell("/m/model.gguf", decodeId, decodeType)),
      )
    store.saveJobManifest(manifest)

    runner(store, engine).resume(manifest.jobId)

    assertEquals(2, store.payloadWrites.size)
    assertEquals(TOP_APP, store.payloadWrites[0].benchmarkCpuAffinity?.cpusetPath)
    assertEquals(FOREGROUND, store.payloadWrites[1].benchmarkCpuAffinity?.cpusetPath)
    // The derived fields have to move with the reading too. `excludes_top_tier` is the one a reader
    // actually filters on, so a stale `false` there is the difference between a discarded result and
    // a trusted one.
    assertEquals(false, store.payloadWrites[0].benchmarkCpuAffinity?.excludesTopTier)
    assertEquals(true, store.payloadWrites[1].benchmarkCpuAffinity?.excludesTopTier)
    assertEquals("one read per cell", 2, engine.cpuAffinityReads)
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
