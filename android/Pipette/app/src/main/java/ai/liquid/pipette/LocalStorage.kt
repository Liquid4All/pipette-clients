package ai.liquid.pipette

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import org.json.JSONObject

class LocalStorage(private val context: Context) : JobStore {
  private val root: File = File(context.filesDir, "Pipette").apply { mkdirs() }
  private val metadataDir: File = File(root, "metadata").apply { mkdirs() }
  val jobsDir: File = File(root, "jobs").apply { mkdirs() }
  /** Home for the server-synced benchmark catalog (`index.json`, `detail/`, `sync.json`). See [FileBenchmarkStore]. */
  val benchmarksDir: File = File(root, "benchmarks").apply { mkdirs() }
  private val legacyModelsDir: File = File(root, "models")
  val modelsDir: File = File(context.getExternalFilesDir(null) ?: root, "models").apply { mkdirs() }
  private val modelStore = ModelStore(modelsDir)

  private val registrationFile: File = File(metadataDir, "registration.json")

  init {
    migrateLegacyModelsDir()
    modelStore.availableModels()
  }

  fun isRegistered(): Boolean = loadRegistration() != null

  fun saveRegistration(data: RegistrationData) {
    writeJson(registrationFile, data.toJson())
  }

  override fun loadRegistration(): RegistrationData? = readJson(registrationFile)?.let { runCatching { RegistrationData.fromJson(it) }.getOrNull() }

  fun deleteRegistration() {
    registrationFile.delete()
  }

  fun recoverInterruptedJobs() {
    loadAllJobManifests().forEach { manifest -> if (manifest.recoverInterruptedRunState()) saveJobManifest(manifest) }
  }

  override fun saveJobManifest(manifest: JobManifest) {
    val dir = jobDir(manifest.jobId).apply { mkdirs() }
    File(dir, "cells").mkdirs()
    writeJson(File(dir, "manifest.json"), manifest.toJson())
  }

  override fun loadJobManifest(jobId: String): JobManifest? =
    readJson(File(jobDir(jobId), "manifest.json"))?.let { runCatching { JobManifest.fromJson(it) }.getOrNull() }

  fun loadAllJobManifests(): List<JobManifest> =
    jobsDir
      .listFiles { file -> file.isDirectory }
      ?.mapNotNull { dir -> readJson(File(dir, "manifest.json")) }
      ?.mapNotNull { json -> runCatching { JobManifest.fromJson(json) }.getOrNull() }
      ?.sortedByDescending { it.createdAt } ?: emptyList()

  fun deleteJob(jobId: String) {
    jobDir(jobId).deleteRecursively()
  }

  fun resetDeviceData() {
    jobsDir.deleteRecursively()
    modelsDir.deleteRecursively()
    legacyModelsDir.deleteRecursively()
    modelStore.clear()
    jobsDir.mkdirs()
    modelsDir.mkdirs()
  }

  /**
   * Cells in [manifest] whose completed result is still on disk and has never been uploaded.
   *
   * One definition, because two screens quote this number at the user and they must not disagree: the job detail's "Submit N Results" button reads
   * this, and the Settings sign-out warning reads [unsubmittedResultCountOnDevice] over the whole store.
   */
  fun unsubmittedResultCount(manifest: JobManifest): Int = manifest.cells.count { isUnsubmitted(manifest.jobId, it) }

  /**
   * Whether this one cell's result is on disk and has never been acked. The single spelling of the three clauses, shared with
   * [unsubmittedResultCount] and with the per-cell "Submit" affordances, which add their own registration check on top. The iOS twin is
   * `ResultsStore.isUnsubmitted`.
   */
  fun isUnsubmitted(jobId: String, cell: JobCell): Boolean =
    cell.runStatus == CellRunStatus.COMPLETED && cell.serverJobId.isNullOrBlank() && submittablePayloadDir(jobId, cell.cellId) != null

  /**
   * Device-wide total of [unsubmittedResultCount]. Named apart from it rather than overloading, because by default this one reads the job store
   * instead of a manifest it was handed: it parses every manifest on disk and stats a payload per completed cell, so keep it off the main thread.
   *
   * The nearest iOS call is `ResultsStore.deletableResultCount(across:)`, which takes the manifests already in hand. It is not on this branch yet: it
   * arrives with #1196, which this stack sits under and has not been restacked onto, where it still reads `unsubmittedResultCount(across:)`. Not a
   * twin either way: that change split "what can still be uploaded" from "what a reset would destroy" and pointed the sign-out warning at the second,
   * while this counts the first for both. Worth settling which predicate the warning should quote on each platform, since they are answering the same
   * question for the user.
   *
   * [manifests] is for a caller that has already paid for the walk and would otherwise repeat it, which is the only reason it is a parameter.
   */
  fun unsubmittedResultCountOnDevice(manifests: List<JobManifest> = loadAllJobManifests()): Int = manifests.sumOf { unsubmittedResultCount(it) }

  fun jobDir(jobId: String): File = File(jobsDir, jobId)

  fun cellArtifactsDir(jobId: String, cellId: String): File = File(File(jobDir(jobId), "cells"), cellId)

  fun ensureCellArtifactsDir(jobId: String, cellId: String): File = cellArtifactsDir(jobId, cellId).apply { mkdirs() }

  fun cellPayloadFile(jobId: String, cellId: String): File = File(cellArtifactsDir(jobId, cellId), "payload.json")

  fun cellMetricsFile(jobId: String, cellId: String): File = File(cellArtifactsDir(jobId, cellId), "metrics.json")

  fun cellSubmissionFile(jobId: String, cellId: String): File = File(cellArtifactsDir(jobId, cellId), "submission.json")

  fun submittablePayloadDir(jobId: String, cellId: String): File? {
    val dir = cellArtifactsDir(jobId, cellId)
    return if (File(dir, "payload.json").exists()) dir else null
  }

  fun saveSubmission(record: CellSubmissionRecord, jobId: String, cellId: String) {
    writeJson(cellSubmissionFile(jobId, cellId).also { it.parentFile?.mkdirs() }, record.toJson())
  }

  fun loadSubmission(jobId: String, cellId: String): CellSubmissionRecord? =
    readJson(cellSubmissionFile(jobId, cellId))?.let { runCatching { CellSubmissionRecord.fromJson(it) }.getOrNull() }

  override fun clearCellArtifacts(jobId: String, cellId: String) {
    cellPayloadFile(jobId, cellId).delete()
    cellSubmissionFile(jobId, cellId).delete()
    cellMetricsFile(jobId, cellId).delete()
  }

  fun modelDestFile(repo: String?, filename: String): File {
    val dest = File(modelsDir, modelRelativePath(repo, filename))
    dest.parentFile?.mkdirs()
    return dest
  }

  override fun resolveModelPath(storedPath: String?): String? {
    if (storedPath == null) return null
    if (File(storedPath).exists()) return storedPath
    return resolveModelPath(storedPath, modelsDir)
  }

  fun availableModels(): List<ModelFile> = modelStore.availableModels()

  fun copyModelFromUri(uri: Uri, fallbackName: String, repo: String? = null): ModelFile {
    val name = fallbackName.ifBlank { "model-${System.currentTimeMillis()}.gguf" }
    val dest = modelDestFile(repo, name)
    context.contentResolver.openInputStream(uri).use { input ->
      requireNotNull(input) { "Unable to open selected file" }
      FileOutputStream(dest).use { output -> input.copyTo(output) }
    }
    return registerModelFile(dest, repo)
  }

  fun registerModelFile(file: File, repo: String?, displayName: String? = null, familyId: String? = null): ModelFile {
    require(file.exists()) { "Model file does not exist: ${file.absolutePath}" }
    return modelStore.registerModel(file, repo, displayName, familyId)
  }

  /**
   * Delete a download artifact (e.g. a cancelled `.part`) and prune any now-empty repo directories up to the models root, so cancel leaves nothing
   * behind.
   */
  fun deleteDownloadArtifact(file: File) {
    if (file.exists()) file.delete()
    pruneEmptyModelParents(file.parentFile)
  }

  fun deleteModel(model: ModelFile) {
    val file = File(model.path)
    val rootPath = modelsDir.canonicalPath
    val filePath = file.canonicalPath
    require(filePath == rootPath || filePath.startsWith("$rootPath${File.separator}")) { "Model file is outside app model storage" }
    if (file.exists()) file.delete()
    pruneEmptyModelParents(file.parentFile)
    modelStore.invalidate()
  }

  /**
   * The two payload fields that describe the model: `model_descriptor`, the lossless coordinate the server stores opaquely, and `model_flags`, what
   * shaped the generation. Both are read off one typed model, because building it twice would let them disagree.
   *
   * Both elide rather than emit a placeholder. The descriptor is absent for a source-less cell (an imported file with no HF slug, or a filename that
   * is not a valid `*.gguf`), which the server would reject as an invalid descriptor rather than store. `model_flags` is eval-only by contract and no
   * Android model carries a flag yet, so it is absent on every submission this client currently produces; see [ModelFlags].
   */
  private fun putModelIdentity(json: JSONObject, modelName: String, modelFilename: String, mmprojFilename: String?, benchmarkType: BenchmarkType) {
    val model = SubmissionRef.typedModelOrNull(modelName, modelFilename, mmprojFilename) ?: return
    json.put("model_descriptor", SubmissionRef.model(model))
    model.modelFlags.submissionString(benchmarkType)?.let { json.put("model_flags", it) }
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
    val json = JSONObject(resultJson)
    val modelFile = File(modelPath)
    json.put("model_name", modelName)
    json.put("model_quant", parseQuant(modelFile.name) ?: "unknown")
    // model_descriptor / runtime_descriptor: the lossless typed descriptors the server stores opaquely (mmproj_quant was retired server-side).
    // model_descriptor is elided for a source-less cell (imported file with no HF slug); runtime_descriptor is the engine identity, runtime_flags the
    // load knobs as the CLI-flag string (matching iOS).
    putModelIdentity(json, modelName, modelFile.name, mmprojPath?.let { File(it).name }, benchmarkType)
    json.put("submitted_at", DateFormats.isoNow())
    // This app build — the harness, not the engine it drove (runtime_version).
    // The server stores it opaquely as a grouping key, so a shift in the numbers
    // can be attributed to an app change rather than to the device. VERSION_NAME
    // alone is the whole identity here: CI folds the version code into it and
    // debug builds carry a `-debug` suffix, so unlike iOS there is no separate
    // build number worth appending.
    json.put("client_version", BuildConfig.VERSION_NAME)
    json.put("runtime_name", "llamacpp_apk_pipette")
    json.put("runtime_version", runtimeVersion)
    json.put("runtime_descriptor", SubmissionRef.runtime(runtimeVersion))
    json.put("runtime_flags", SubmissionRef.runtimeFlags(nGpuLayers, contextSize))
    // What the harness ran under, next to the engine's load knobs above: the
    // readiness policy this cell was gated by. Omitted (not null) for a cell the
    // flag schema reports no readiness for. See SubmissionRef.benchmarkFlagsOrNull.
    SubmissionRef.benchmarkFlagsOrNull(benchmarkType, readinessPolicy)?.let { json.put("benchmark_flags", it) }
    // Active CPU backend variant the loader selected for this device (the
    // ggml feature descriptor, e.g. "dotprod,fp16_va,neon"); omitted when
    // not yet known. Lets result analysis detect a backend change.
    runtimeCpuBackend?.let { json.put("runtime_cpu_variant", it) }
    json.put("device_name", DeviceInfo.modelName())
    json.put("device_form_factor", DeviceInfo.formFactor(context))
    json.put("device_os_name", "Android")
    json.put("device_os_version", DeviceInfo.osVersion())
    // Finer-grained OS identity; omitted (not null) when the platform reports
    // nothing, matching the optional device fields below.
    DeviceInfo.osBuild()?.let { json.put("device_os_build", it) }
    DeviceInfo.osSecurityPatch()?.let { json.put("device_os_security_patch", it) }
    json.put("device_chip_model", DeviceInfo.chipModel())
    json.put("device_ram_bytes", DeviceInfo.ramBytes(context))
    // Run-environment power state (captured at completion ≈ during the run),
    // recorded locally (CSV export) and on the submission. Optional fields
    // are omitted (not null) when the platform won't report them.
    DeviceInfo.batteryLevel(context)?.let { json.put("device_battery_level", it) }
    DeviceInfo.powerState(context)?.let { json.put("device_power_state", it) }
    json.put("device_power_save_mode", DeviceInfo.isPowerSaveMode(context))
    // Diagnostic: the cpuset group + CPU affinity the :benchmark process ran
    // under, to surface OEM demotion (e.g. Samsung throttling a non-top-app
    // service process off the prime cores). Omitted when unreadable / the
    // service wasn't bound. These keys ride the submission payload but pipette-
    // mgmt has no typed field for them yet, so the server currently drops them
    // on ingest — the CSV export is the only consumer until mgmt adds them (see
    // PIP follow-up). Forward-compatible: harmless once the server fields land.
    benchmarkCpuAffinity?.let { affinity ->
      affinity.cpusetPath?.let { json.put("device_android_cpuset", it) }
      affinity.allowedCpus?.let { json.put("device_android_cpu_affinity_list", it) }
      json.put("device_android_cpu_affinity_excludes_top_tier", affinity.excludesTopTier)
    }
    // Wire shape mirrors pipette-ops `BenchmarkSubmissionPayload` /
    // `DeviceInfo`: optional GPU/NPU fields are omitted (not null) when
    // absent, and no `job_id` is sent (the server assigns it on submit). The
    // projector coordinate now rides inside `model_descriptor` (VL variant), so
    // the standalone `mmproj_quant` field is no longer emitted.

    val dir = ensureCellArtifactsDir(jobId, cellId)
    writeJson(File(dir, "payload.json"), json)
    cellSubmissionFile(jobId, cellId).delete()
    cellMetricsFile(jobId, cellId).delete()
  }

  private fun migrateLegacyModelsDir() {
    if (!legacyModelsDir.exists()) return
    if (legacyModelsDir.canonicalPath == modelsDir.canonicalPath) return
    val files = legacyModelsDir.walkTopDown().filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }.toList()
    files.forEach { file ->
      val relative = file.relativeTo(legacyModelsDir).invariantSeparatorsPath
      val dest = File(modelsDir, relative)
      if (!dest.exists()) {
        dest.parentFile?.mkdirs()
        if (!file.renameTo(dest)) file.copyTo(dest, overwrite = false)
      }
    }
    legacyModelsDir.deleteRecursively()
  }

  private fun pruneEmptyModelParents(start: File?) {
    var current = start
    val rootPath = modelsDir.canonicalPath
    while (current != null && current.exists() && current.canonicalPath != rootPath) {
      val children = current.listFiles()
      if (children == null || children.isNotEmpty()) break
      if (!current.delete()) break
      current = current.parentFile
    }
  }

  private fun readJson(file: File): JSONObject? = if (!file.exists()) null else runCatching { JSONObject(file.readText()) }.getOrNull()

  private fun writeJson(file: File, json: JSONObject) {
    file.parentFile?.mkdirs()
    file.writeText(json.toString(2))
  }

  companion object {
    fun modelRelativePath(repo: String?, filename: String): String = if (repo.isNullOrBlank()) filename else "${repo.trim('/')}/$filename"

    fun isQuantToken(value: String): Boolean {
      val upper = value.uppercase()
      return upper.startsWith("Q") || upper.startsWith("F") || upper.startsWith("BF") || upper.startsWith("IQ")
    }

    fun parseQuant(filename: String): String? {
      val stem = filename.replace(Regex("\\.gguf$", RegexOption.IGNORE_CASE), "")
      return stem.split("-").asReversed().firstOrNull { isQuantToken(it) }
    }

    fun modelStem(filename: String): String {
      val stem = filename.replace(Regex("\\.gguf$", RegexOption.IGNORE_CASE), "")
      val parts = stem.split("-")
      return if (parts.isNotEmpty() && isQuantToken(parts.last())) {
        parts.dropLast(1).joinToString("-")
      } else {
        stem
      }
    }

    fun normalizedModelStem(filename: String): String {
      var base = filename.replace(Regex("\\.gguf$", RegexOption.IGNORE_CASE), "")
      if (base.lowercase().startsWith("mmproj-")) base = base.drop("mmproj-".length)
      val parts = base.split("-")
      return if (parts.isNotEmpty() && isQuantToken(parts.last())) {
        parts.dropLast(1).joinToString("-").lowercase()
      } else {
        base.lowercase()
      }
    }

    fun resolveModelPath(storedPath: String, modelsRoot: File): String? {
      val components = storedPath.split(File.separatorChar, '/').filter { it.isNotBlank() }
      val filename = components.lastOrNull()?.takeIf { it.isNotBlank() } ?: return null
      val modelsIndex = components.indexOfLast { it == "models" }
      if (modelsIndex >= 0 && modelsIndex + 1 < components.size) {
        val tail = components.drop(modelsIndex + 1).joinToString(File.separator)
        val candidate = File(modelsRoot, tail)
        if (candidate.exists()) return candidate.absolutePath
      }

      val matches = modelsNamed(filename, modelsRoot)
      return if (matches.size == 1) matches.single().absolutePath else null
    }

    private fun modelsNamed(filename: String, root: File): List<File> {
      if (!root.exists()) return emptyList()
      return root.walkTopDown().filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) && it.name == filename }.toList()
    }
  }
}
