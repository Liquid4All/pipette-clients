package ai.liquid.pipette.fakes

import ai.liquid.pipette.BenchmarkType
import ai.liquid.pipette.CpuAffinitySnapshot
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStore
import ai.liquid.pipette.ReadinessPolicy
import ai.liquid.pipette.RegistrationData

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
  )

  private val manifests = mutableMapOf<String, JobManifest>()
  val payloadWrites = mutableListOf<PayloadWrite>()
  val clearedArtifacts = mutableListOf<Pair<String, String>>()

  override fun saveJobManifest(manifest: JobManifest) {
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
    nGpuLayers: Int,
    contextSize: Int,
    runtimeVersion: String,
    runtimeCpuBackend: String?,
    benchmarkType: BenchmarkType,
    readinessPolicy: ReadinessPolicy,
    benchmarkCpuAffinity: CpuAffinitySnapshot?,
  ) {
    payloadWrites += PayloadWrite(jobId, cellId, resultJson, runtimeVersion, benchmarkType, readinessPolicy, runtimeCpuBackend, benchmarkCpuAffinity)
  }

  override fun loadRegistration(): RegistrationData? = registration
}
