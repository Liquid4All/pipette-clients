package ai.liquid.pipette

import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.FakeReadinessGate
import ai.liquid.pipette.fakes.FakeResultSubmitter
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_ID
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_TYPE
import ai.liquid.pipette.fakes.JobRunnerFixtures.directExecutor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What [JobRunner] does about a cell whose weights are not on disk.
 *
 * Before [ModelProvisioning] existed there was one answer: fail. Now there are two, and which applies is decided entirely by whether the cell says
 * what its model *is* ([JobCell.modelRef]). Both are pinned here, because the failing one is what every manifest written before this field still
 * gets, and it must not have quietly changed.
 */
class JobRunnerProvisioningTest {
  private val missing = "LiquidAI/LFM2-1.2B-GGUF/lfm2-1.2b-q4_k_m.gguf"
  private val fetched = "/data/models/$missing"
  private val ref = Model.GgufText(HfGgufText(HfRepo.parseSlug("LiquidAI/LFM2-1.2B-GGUF"), GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf")))

  /** Records what it was asked for, and answers with whatever the test wants that attempt to produce. */
  private class FakeProvisioner(private val weights: Result<String>, private val mmproj: Result<String>? = null) : ModelProvisioning {
    val weightsAsked = mutableListOf<Model>()
    val mmprojAsked = mutableListOf<Model>()

    override fun ensureWeights(ref: Model, onStatus: (String) -> Unit): Result<String> {
      weightsAsked += ref
      onStatus("Downloading")
      return weights
    }

    override fun ensureMmproj(ref: Model, onStatus: (String) -> Unit): Result<String>? {
      mmprojAsked += ref
      return mmproj
    }
  }

  private fun runner(store: FakeJobStore, engine: FakeBenchmarkEngine, provisioner: ModelProvisioning?) =
    JobRunner(
      storage = store,
      engine = engine,
      submissionService = FakeResultSubmitter(),
      readiness = FakeReadinessGate(),
      executor = directExecutor,
      provisioner = provisioner,
    ) {}

  private fun manifestWith(cell: JobCell) = JobManifest(nGpuLayers = 99, contextSize = 4096, status = JobStatus.RUNNING, cells = mutableListOf(cell))

  private fun cell(modelRef: Model?) =
    JobCell(benchmarkId = DECODE_ID, benchmarkType = DECODE_TYPE, modelPath = missing, modelName = "LFM2 1.2B", modelRef = modelRef)

  @Test
  fun `a referenced cell whose model is absent is fetched and then loaded`() {
    // Unresolvable at plan time, resolvable once the provisioner reports where it landed: the runner uses the path the fetch returns rather than
    // re-resolving, which is what lets this work without the fake store changing its mind mid-test.
    val store = FakeJobStore(unresolvablePaths = setOf(missing))
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(fetched))
    val manifest = manifestWith(cell(ref))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    assertEquals(listOf(ref), provisioner.weightsAsked)
    assertEquals("the cell must load the file the fetch produced", listOf(fetched), engine.loads.map { it.modelPath })
    assertEquals(CellRunStatus.COMPLETED, manifest.cells.single().runStatus)
  }

  @Test
  fun `a fetch failure fails the cell with the reason the fetch gave`() {
    val store = FakeJobStore(unresolvablePaths = setOf(missing))
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.failure(IllegalStateException("waiting for a network this device does not have")))
    val manifest = manifestWith(cell(ref))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    // The fetch's own reason, not the generic not-found text: the two are different problems and an operator has to be able to tell them apart.
    assertTrue(failed.errorMessage.orEmpty(), failed.errorMessage?.contains("network") == true)
    assertTrue("nothing should have been loaded", engine.loads.isEmpty())
  }

  /** The pre-provisioning behaviour, which every manifest written before `modelRef` existed still gets. */
  @Test
  fun `a cell with no reference still fails with the original message`() {
    val store = FakeJobStore(unresolvablePaths = setOf(missing))
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(fetched))
    val manifest = manifestWith(cell(modelRef = null))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage.orEmpty(), failed.errorMessage?.contains("Model file not found") == true)
    assertTrue("a cell with nothing to fetch from must not reach the provisioner", provisioner.weightsAsked.isEmpty())
    assertTrue(engine.loads.isEmpty())
  }

  /** No provisioner wired (the case a suite that predates it, or the `:benchmark` process, presents) leaves the old behaviour untouched. */
  @Test
  fun `with no provisioner a referenced cell fails as before`() {
    val store = FakeJobStore(unresolvablePaths = setOf(missing))
    val engine = FakeBenchmarkEngine()
    val manifest = manifestWith(cell(ref))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner = null).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage.orEmpty(), failed.errorMessage?.contains("Model file not found") == true)
  }

  /** A model already on disk is never asked for, so an ordinary run pays nothing for this. */
  @Test
  fun `a model already present does not reach the provisioner`() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(fetched))
    val manifest = manifestWith(cell(ref))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    assertTrue("present weights must not be fetched", provisioner.weightsAsked.isEmpty())
    assertEquals(listOf(missing), engine.loads.map { it.modelPath })
  }

  private val packageRef =
    Model.OnnxGenai(
      HfOnnxGenai(HfRepo.parseSlug("org/genai-package"), RepoSubpath.parse("int4"), HfRevision.parse("5f5f794c1c23c9d5ee142af85df02a6cc52d6945"))
    )
  private val packageDir = "/data/models/org/genai-package/int4"

  private fun packageCell(benchmarkId: String = DECODE_ID, benchmarkType: String = DECODE_TYPE) =
    JobCell(benchmarkId = benchmarkId, benchmarkType = benchmarkType, modelPath = packageDir, modelName = "org/genai-package", modelRef = packageRef)

  /**
   * A package is asked of the provisioner even when its directory is there, unlike a file. The path says nothing of which snapshot it holds, and a
   * cell pinned to one would otherwise run, and record, whatever had since replaced it.
   */
  @Test
  fun `a package already present is still asked of the provisioner`() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(packageDir))
    val manifest = manifestWith(packageCell())
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    assertEquals(listOf<Model>(packageRef), provisioner.weightsAsked)
    assertEquals(listOf(packageDir), engine.loads.map { it.modelPath })
    assertEquals(CellRunStatus.COMPLETED, manifest.cells.single().runStatus)
  }

  /** And when the provisioner refuses it, as it does a package from another snapshot, the path on disk is not loaded in its place. */
  @Test
  fun `a package the provisioner refuses is not loaded from its path`() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.failure(IllegalStateException("it is pinned to another snapshot")))
    val manifest = manifestWith(packageCell())
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage.orEmpty(), failed.errorMessage?.contains("another snapshot") == true)
    assertTrue(engine.loads.isEmpty())
  }

  /**
   * A kind the ONNX Runtime GenAI engine does not run is refused before its package is fetched or loaded. Refused by the engine instead, each such
   * cell would cost a multi-gigabyte load and warm the device ahead of the next timed one.
   */
  @Test
  fun `a cell the ONNX engine does not run is refused before its package is fetched`() {
    val store = FakeJobStore()
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(packageDir))
    val manifest = manifestWith(packageCell(benchmarkType = BenchmarkType.EVAL.wire))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    val failed = manifest.cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage.orEmpty(), failed.errorMessage?.contains("chat template") == true)
    assertTrue("nothing is fetched for a cell that cannot run", provisioner.weightsAsked.isEmpty())
    assertTrue(engine.loads.isEmpty())
  }
}
