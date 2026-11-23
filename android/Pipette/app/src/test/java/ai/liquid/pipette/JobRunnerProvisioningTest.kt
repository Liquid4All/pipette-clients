package ai.liquid.pipette

import ai.liquid.pipette.fakes.FakeBenchmarkEngine
import ai.liquid.pipette.fakes.FakeJobStore
import ai.liquid.pipette.fakes.FakeReadinessGate
import ai.liquid.pipette.fakes.FakeResultSubmitter
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_ID
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_TYPE
import ai.liquid.pipette.fakes.JobRunnerFixtures.directExecutor
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * What [JobRunner] does about a cell whose weights are not on disk.
 *
 * Before [ModelProvisioning] existed there was one answer: fail. Now there are two, and which applies is decided entirely by whether the cell says
 * what its model *is* ([JobCell.modelRef]). Both are pinned here, because the failing one is what every manifest written before this field still
 * gets, and it must not have quietly changed.
 */
class JobRunnerProvisioningTest {
  @get:Rule val temp = TemporaryFolder()

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

  private fun runner(store: JobStore, engine: FakeBenchmarkEngine, provisioner: ModelProvisioning?) =
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
  fun `a pinned cell fetches the pin even when its stored path resolves to the default copy`() {
    val pinned = ref.copy(model = ref.model.copy(repo = ref.repo.copy(revision = HfRevision.parse("abc123"))))
    val pinnedKey = LocalStorage.modelRelativePath(pinned.repo.reference, pinned.model.filename.value)
    val pinnedPath = "/data/models/$pinnedKey"
    val store = FakeJobStore(unresolvablePaths = setOf(pinnedKey))
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(pinnedPath))
    val manifest = manifestWith(cell(pinned))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    assertEquals(listOf(pinned), provisioner.weightsAsked)
    assertEquals(listOf(pinnedPath), engine.loads.map { it.modelPath })
  }

  @Test
  fun `an unpinned cell fetches the default copy when its stored path resolves to a pin`() {
    val pinnedPath = "/data/models/${ref.repo}@abc123/${ref.model.filename}"
    val store = FakeJobStore(unresolvablePaths = setOf(missing))
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(fetched))
    val manifest = manifestWith(cell(ref).copy(modelPath = pinnedPath))
    store.saveJobManifest(manifest)

    runner(store, engine, provisioner).resume(manifest.jobId)

    assertEquals(listOf(ref), provisioner.weightsAsked)
    assertEquals(listOf(fetched), engine.loads.map { it.modelPath })
  }

  @Test
  fun `a malformed pin stays blocked after its job is saved and retried`() {
    val pinnedPath = "/data/models/${ref.repo}@abc123/${ref.model.filename}"
    val invalidRef = ref.toJson().put("revision", "refs/pr/3")
    val cellJson = cell(ref).copy(modelPath = pinnedPath).toJson().put("modelRef", invalidRef)
    val manifest = manifestWith(JobCell.fromJson(cellJson))

    val unpinned = File(temp.root, missing)
    unpinned.parentFile?.mkdirs()
    unpinned.writeText("unpinned weights")
    assertEquals(unpinned.absolutePath, LocalStorage.resolveModelPath(pinnedPath, temp.root))

    val persisted = FakeJobStore()
    var pathLookups = 0
    val store =
      object : JobStore by persisted {
        override fun resolveModelPath(storedPath: String?): String? {
          pathLookups++
          return storedPath?.let { LocalStorage.resolveModelPath(it, temp.root) }
        }

        override fun saveJobManifest(manifest: JobManifest) {
          persisted.saveJobManifest(JobManifest.fromJson(JSONObject(manifest.toJson().toString())))
        }
      }
    val engine = FakeBenchmarkEngine()
    val provisioner = FakeProvisioner(weights = Result.success(fetched))
    val runner = runner(store, engine, provisioner)

    store.saveJobManifest(manifest)
    runner.resume(manifest.jobId)
    val failed = checkNotNull(store.loadJobManifest(manifest.jobId)).cells.single()
    assertEquals(CellRunStatus.FAILED, failed.runStatus)
    assertTrue(failed.errorMessage.orEmpty().contains("Create a new job"))
    assertEquals(invalidRef.toString(), failed.toJson().getJSONObject("modelRef").toString())

    runner.retryFailed(manifest.jobId)
    val retried = checkNotNull(store.loadJobManifest(manifest.jobId)).cells.single()
    assertEquals(CellRunStatus.FAILED, retried.runStatus)
    assertEquals(invalidRef.toString(), retried.toJson().getJSONObject("modelRef").toString())
    assertEquals(0, pathLookups)
    assertTrue(engine.loads.isEmpty())
    assertTrue(provisioner.weightsAsked.isEmpty())
    assertTrue(persisted.payloadWrites.isEmpty())
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
}
