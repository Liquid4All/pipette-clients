package ai.liquid.pipette

/**
 * The persistence operations [JobRunner] needs while executing a job: manifest read/write, per-cell artifact cleanup on rerun, model-path resolution,
 * result payload writes, and the registration used to gate submission.
 *
 * [LocalStorage] implements this against the device filesystem. Unit tests substitute an in-memory fake so the cell loop can be exercised off-device.
 * This is intentionally the narrow slice [JobRunner] touches, not the whole of [LocalStorage]'s surface.
 */
interface JobStore {
  fun saveJobManifest(manifest: JobManifest)

  fun loadJobManifest(jobId: String): JobManifest?

  fun clearCellArtifacts(jobId: String, cellId: String)

  fun resolveModelPath(storedPath: String?): String?

  // Intentionally wide: this is the single persistence sink for a completed cell's submission fields — a flat record, not a grouping worth a value
  // object here.
  @Suppress("LongParameterList") // the single flat persistence sink for a completed cell's submission fields
  fun writePayload(
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
    // Not defaulted, unlike the diagnostic below: both are always known at the
    // call site, so a caller that forgets one should fail to compile rather than
    // silently drop `benchmark_flags`. See SubmissionRef.benchmarkFlagsOrNull.
    benchmarkType: BenchmarkType,
    readinessPolicy: ReadinessPolicy,
    benchmarkCpuAffinity: CpuAffinitySnapshot? = null,
  )

  fun loadRegistration(): RegistrationData?
}
