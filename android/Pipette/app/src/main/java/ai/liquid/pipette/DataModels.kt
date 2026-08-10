package ai.liquid.pipette

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID
import org.json.JSONArray
import org.json.JSONObject

enum class JobStatus {
  PLANNED,
  RUNNING,
  COMPLETED,
  CANCELLED,
  PAUSED;

  val wire: String
    get() = name.lowercase()

  companion object {
    fun fromWire(value: String?): JobStatus = entries.firstOrNull { it.wire == value } ?: PLANNED
  }
}

enum class CellRunStatus {
  PENDING,
  RUNNING,
  COMPLETED,
  FAILED,
  CANCELLED;

  val wire: String
    get() = name.lowercase()

  val isRerunnable: Boolean
    get() = this == COMPLETED || this == FAILED || this == CANCELLED

  companion object {
    fun fromWire(value: String?): CellRunStatus = entries.firstOrNull { it.wire == value } ?: PENDING
  }
}

data class JobCell(
  val cellId: String = UUID.randomUUID().toString(),
  val benchmarkId: String,
  val benchmarkType: String?,
  val modelPath: String,
  val modelName: String,
  val mmprojPath: String? = null,
  var runStatus: CellRunStatus = CellRunStatus.PENDING,
  var serverJobId: String? = null,
  var errorMessage: String? = null,
) {
  val isRerunnable: Boolean
    get() = runStatus.isRerunnable

  fun toJson(): JSONObject =
    JSONObject()
      .put("cellId", cellId)
      .put("benchmarkId", benchmarkId)
      .putOptString("benchmarkType", benchmarkType)
      .put("modelPath", modelPath)
      .put("modelName", modelName)
      .putOptString("mmprojPath", mmprojPath)
      .put("runStatus", runStatus.wire)
      .putOptString("serverJobId", serverJobId)
      .putOptString("errorMessage", errorMessage)

  companion object {
    fun fromJson(json: JSONObject): JobCell =
      JobCell(
        cellId = json.getString("cellId"),
        benchmarkId = json.getString("benchmarkId"),
        benchmarkType = json.optNullableString("benchmarkType"),
        modelPath = json.getString("modelPath"),
        modelName = json.getString("modelName"),
        mmprojPath = json.optNullableString("mmprojPath"),
        runStatus = CellRunStatus.fromWire(json.optString("runStatus")),
        serverJobId = json.optNullableString("serverJobId"),
        errorMessage = json.optNullableString("errorMessage"),
      )
  }
}

data class JobManifest(
  val jobId: String = UUID.randomUUID().toString(),
  val createdAt: String = DateFormats.isoNow(),
  var nGpuLayers: Int,
  var contextSize: Int,
  var cells: MutableList<JobCell>,
  var status: JobStatus,
  var contributeResults: Boolean? = null,
  var title: String? = null,
  /** Prefill micro-batch applied to every cell (llama.cpp `n_ubatch`); 512 default. */
  var prefillBatch: Int = DEFAULT_PREFILL_BATCH,
) {
  val totalCells: Int
    get() = cells.size

  val completedCells: Int
    get() = cells.count { it.runStatus == CellRunStatus.COMPLETED }

  val failedCells: Int
    get() = cells.count { it.runStatus == CellRunStatus.FAILED }

  val cancelledCells: Int
    get() = cells.count { it.runStatus == CellRunStatus.CANCELLED }

  val submittedCells: Int
    get() = cells.count { !it.serverJobId.isNullOrBlank() }

  val displayTitle: String
    get() {
      val trimmed = title?.trim()
      if (!trimmed.isNullOrEmpty()) return trimmed
      val models = cells.map { it.modelName }.toSet().size
      val benchmarks = cells.map { it.benchmarkId }.toSet().size
      return "${DateFormats.shortDate(createdAt)} - $models model${if (models == 1) "" else "s"} - " +
        "$benchmarks benchmark${if (benchmarks == 1) "" else "s"}"
    }

  fun recoverInterruptedRunState(): Boolean {
    if (status != JobStatus.RUNNING && status != JobStatus.PAUSED) return false
    var changed = false
    cells.forEach { cell ->
      if (cell.runStatus == CellRunStatus.RUNNING) {
        cell.runStatus = CellRunStatus.CANCELLED
        changed = true
      }
    }
    if (status == JobStatus.RUNNING) {
      cells.forEach { cell ->
        if (cell.runStatus == CellRunStatus.PENDING) {
          cell.runStatus = CellRunStatus.CANCELLED
          changed = true
        }
      }
      val recovered =
        if (cells.any { it.runStatus == CellRunStatus.CANCELLED }) {
          JobStatus.PAUSED
        } else {
          JobStatus.COMPLETED
        }
      if (status != recovered) {
        status = recovered
        changed = true
      }
    }
    return changed
  }

  fun toJson(): JSONObject =
    JSONObject()
      .put("jobId", jobId)
      .put("createdAt", createdAt)
      .put("nGpuLayers", nGpuLayers)
      .put("contextSize", contextSize)
      .put("cells", JSONArray().also { array -> cells.forEach { array.put(it.toJson()) } })
      .put("status", status.wire)
      .putOptBool("contributeResults", contributeResults)
      .putOptString("title", title)
      .put("prefillBatch", prefillBatch)

  companion object {
    const val DEFAULT_PREFILL_BATCH = 512

    fun fromJson(json: JSONObject): JobManifest {
      val cellArray = json.getJSONArray("cells")
      val parsedCells = MutableList(cellArray.length()) { index -> JobCell.fromJson(cellArray.getJSONObject(index)) }
      return JobManifest(
        jobId = json.getString("jobId"),
        createdAt = json.getString("createdAt"),
        nGpuLayers = json.optInt("nGpuLayers", 99),
        contextSize = json.optInt("contextSize", 4096),
        cells = parsedCells,
        status = JobStatus.fromWire(json.optString("status")),
        contributeResults = json.optNullableBoolean("contributeResults"),
        title = json.optNullableString("title"),
        prefillBatch = json.optInt("prefillBatch", DEFAULT_PREFILL_BATCH),
      )
    }
  }
}

data class RegistrationData(
  val clientId: String,
  val status: String,
  val serverUrl: String,
  val organization: String,
  val contactEmail: String,
  val registeredAt: String,
  val clerkUserId: String? = null,
  val clerkSessionId: String? = null,
  val clerkPrimaryEmail: String? = null,
  val clerkLinkedAt: String? = null,
) {
  fun toJson(): JSONObject =
    JSONObject()
      .put("clientId", clientId)
      .put("status", status)
      .put("serverUrl", serverUrl)
      .put("organization", organization)
      .put("contactEmail", contactEmail)
      .put("registeredAt", registeredAt)
      .putOptString("clerkUserId", clerkUserId)
      .putOptString("clerkSessionId", clerkSessionId)
      .putOptString("clerkPrimaryEmail", clerkPrimaryEmail)
      .putOptString("clerkLinkedAt", clerkLinkedAt)

  fun withClerkLink(userId: String, sessionId: String?, primaryEmail: String?): RegistrationData =
    copy(clerkUserId = userId, clerkSessionId = sessionId, clerkPrimaryEmail = primaryEmail, clerkLinkedAt = clerkLinkedAt ?: DateFormats.isoNow())

  /**
   * Drop the Clerk link, keeping [clientId], the registration inputs, and (elsewhere) the signing key. Applied on an explicit sign-out: the link is a
   * purely local guard, since the Clerk identity is never sent to the management server, so un-pinning the device costs nothing server-side and lets
   * the next account link instead of colliding with this one.
   *
   * [clerkLinkedAt] goes too. It records when the *current* link was made, and [withClerkLink] preserves an existing value, so leaving it would date
   * the next account's link from the previous one's.
   */
  fun withoutClerkLink(): RegistrationData = copy(clerkUserId = null, clerkSessionId = null, clerkPrimaryEmail = null, clerkLinkedAt = null)

  companion object {
    fun fromJson(json: JSONObject): RegistrationData =
      RegistrationData(
        clientId = json.getString("clientId"),
        status = json.getString("status"),
        serverUrl = json.getString("serverUrl"),
        organization = json.getString("organization"),
        contactEmail = json.getString("contactEmail"),
        registeredAt = json.getString("registeredAt"),
        clerkUserId = json.optNullableString("clerkUserId"),
        clerkSessionId = json.optNullableString("clerkSessionId"),
        clerkPrimaryEmail = json.optNullableString("clerkPrimaryEmail"),
        clerkLinkedAt = json.optNullableString("clerkLinkedAt"),
      )
  }
}

data class CellSubmissionRecord(val status: String, val serverJobId: String?, val submittedAt: String?, val errors: List<String>) {
  fun toJson(): JSONObject =
    JSONObject()
      .put("status", status)
      .putOptString("server_job_id", serverJobId)
      .putOptString("submitted_at", submittedAt)
      .put("errors", JSONArray().also { array -> errors.forEach { array.put(it) } })

  companion object {
    fun submitted(serverJobId: String): CellSubmissionRecord = CellSubmissionRecord("submitted", serverJobId, DateFormats.isoNow(), emptyList())

    fun failed(errors: List<String>): CellSubmissionRecord = CellSubmissionRecord("failed", null, DateFormats.isoNow(), errors)

    fun fromJson(json: JSONObject): CellSubmissionRecord {
      val errorsJson = json.optJSONArray("errors") ?: JSONArray()
      return CellSubmissionRecord(
        status = json.optString("status"),
        serverJobId = json.optNullableString("server_job_id"),
        submittedAt = json.optNullableString("submitted_at"),
        errors = List(errorsJson.length()) { errorsJson.optString(it) },
      )
    }
  }
}

data class ModelFile(
  val name: String,
  val path: String,
  val sizeBytes: Long,
  val hfRepo: String?,
  val displayName: String? = null,
  val familyId: String? = null,
) {
  val sizeFormatted: String
    get() = ByteFormat.fileSize(sizeBytes)

  val quant: String?
    get() = LocalStorage.parseQuant(name)

  val isMmproj: Boolean
    get() = name.contains("mmproj", ignoreCase = true)
}

object DateFormats {
  private val shortFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd").withZone(ZoneId.systemDefault())

  fun isoNow(): String = DateTimeFormatter.ISO_INSTANT.format(Instant.now())

  fun shortDate(iso: String): String = runCatching { shortFormatter.format(Instant.parse(iso)) }.getOrElse { iso.take(10) }
}

object ByteFormat {
  fun fileSize(bytes: Long): String {
    if (bytes < 1024L * 1024L) return "${bytes / 1024L} KB"
    val mb = bytes / (1024.0 * 1024.0)
    // Fixed locale so the decimal separator is "." everywhere (these strings show in download progress and notifications).
    if (mb < 1024.0) return String.format(java.util.Locale.US, "%.1f MB", mb)
    return String.format(java.util.Locale.US, "%.2f GB", mb / 1024.0)
  }
}

fun JSONObject.optNullableString(name: String): String? {
  if (!has(name) || isNull(name)) return null
  return optString(name).takeIf { it.isNotEmpty() }
}

fun JSONObject.optNullableBoolean(name: String): Boolean? {
  if (!has(name) || isNull(name)) return null
  return optBoolean(name)
}

fun JSONObject.putOptString(name: String, value: String?): JSONObject = if (value == null) put(name, JSONObject.NULL) else put(name, value)

fun JSONObject.putOptBool(name: String, value: Boolean?): JSONObject = if (value == null) put(name, JSONObject.NULL) else put(name, value)
