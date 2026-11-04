package ai.liquid.pipette.fakes

import ai.liquid.pipette.BenchmarkType
import ai.liquid.pipette.CpuAffinitySnapshot
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStore
import ai.liquid.pipette.ReadinessPolicy
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.RuntimeFlagRef

/**
 * In-memory [JobStore] for JVM tests. Holds manifests by reference (so the cell-status mutations [JobRunner] makes are observable from the test), and
 * records payload writes / artifact clears for assertions.
 */
class FakeJobStore(
  private val registration: RegistrationData? = null,
  /** Stored paths that should fail to resolve (simulating a missing model file). */
  private val unresolvablePaths: Set<String> = emptySet(),
) : JobStore {
  data class PayloadWrite(
    val jobId: String,
    val cellId: String,
    val resultJson: String,
    val runtimeVersion: String,
    val benchmarkType: BenchmarkType,
    val readinessPolicy: ReadinessPolicy,
    val runtimeCpuBackend: String? = null,
    val benchmarkCpuAffinity: CpuAffinitySnapshot? = null,
    val runtimeFlags: RuntimeFlagRef? = null,
  )

  private val manifests = mutableMapOf<String, JobManifest>()
  val payloadWrites = mutableListOf<PayloadWrite>()
  val clearedArtifacts = mutableListOf<Pair<String, String>>()

  /**
   * Make saves start throwing once [saveManifestErrorAfter] of them have succeeded, simulating a job dying somewhere outside `JobRunner`'s per-cell
   * try.
   *
   * That region matters on its own: work in it is not covered by the cell's error handling, so a throw there unwinds `execute` entirely, and anything
   * that has to happen at a job boundary regardless must survive it. Counted rather than a plain flag because the entry points save before they start
   * the run at all (`resume` sets the manifest RUNNING first), so an unconditional throw never reaches the runner.
   */
  var saveManifestError: Throwable? = null
  var saveManifestErrorAfter: Int = 0
  var saveCount = 0
    private set

  override fun saveJobManifest(manifest: JobManifest) {
    saveManifestError?.let { if (saveCount >= saveManifestErrorAfter) throw it }
    saveCount++
    manifests[manifest.jobId] = manifest
  }

  override fun loadJobManifest(jobId: String): JobManifest? = manifests[jobId]

  override fun clearCellArtifacts(jobId: String, cellId: String) {
    clearedArtifacts += jobId to cellId
  }

  override fun resolveModelPath(storedPath: String?): String? =
    when {
      storedPath == null -> null
      storedPath in unresolvablePaths -> null
      else -> storedPath
    }

  override fun writePayload(
    resultJson: String,
    cellId: String,
    jobId: String,
    modelName: String,
    modelPath: String,
    mmprojPath: String?,
    runtimeFlags: RuntimeFlagRef,
    runtimeVersion: String,
    runtimeCpuBackend: String?,
    benchmarkType: BenchmarkType,
    readinessPolicy: ReadinessPolicy,
    benchmarkCpuAffinity: CpuAffinitySnapshot?,
  ) {
    payloadWrites +=
      PayloadWrite(jobId, cellId, resultJson, runtimeVersion, benchmarkType, readinessPolicy, runtimeCpuBackend, benchmarkCpuAffinity, runtimeFlags)
  }

  override fun loadRegistration(): RegistrationData? = registration
}
