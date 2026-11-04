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
  /**
   * The cell's authored load knobs, or null when it named none and inherits the job's. The Android counterpart of a plan cell's `runtime_flags`
   * entry: absent for every cell the in-app UI composes today, present for one authored against the typed surface.
   */
  val runtimeFlags: RuntimeFlagRef? = null,
  /**
   * What this cell's model *is*, as opposed to where it currently sits ([modelPath]). Lets [JobRunner] fetch the weights when the path resolves to
   * nothing, which is what allows a run against a catalog model that was never downloaded, and what lets a run survive the model being deleted
   * between planning and execution.
   *
   * Null for a cell this build cannot name a coordinate for: a manifest written before this field existed, and a locally imported file that has no
   * HuggingFace coordinate to go back to. Such a cell keeps the old behaviour exactly, failing with "Model file not found", because there is nothing
   * to fetch it from. iOS carries the same thing as `JobCell.source`, where it is non-optional because that client never had a path-only form.
   */
  val modelRef: Model? = null,
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
      // Omitted entirely when the cell authored none, so a job file written before this
      // field existed round-trips byte-identically through a load/save cycle. Same rule for
      // modelRef below, for the same reason.
      .also { obj -> runtimeFlags?.let { obj.put("runtimeFlags", it.toJson()) } }
      .also { obj -> modelRef?.let { obj.put("modelRef", it.toJson()) } }

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
        runtimeFlags = RuntimeFlagRef.fromJsonOrNull(json.optJSONObject("runtimeFlags")),
        modelRef = Model.fromJsonOrNull(json.optJSONObject("modelRef")),
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
  /**
   * What this job asks of the readiness gate, from the `benchmark_flags.readiness` its cells were authored with. [RunReadiness.DEFAULT] for every job
   * the New Job screen composes, which is what leaves them on the client defaults.
   *
   * Persisted rather than held by whoever started the run: a job is resumed, retried and re-run long after that caller is gone, and every one of
   * those measures the same job. [JobRunner] installs this at each run start.
   */
  var readiness: RunReadiness = RunReadiness.DEFAULT,
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
      // Omitted entirely when the job named neither knob, so a manifest written before this field existed round-trips byte-identically through a
      // load/save cycle — the rule JobCell.runtimeFlags follows, for the same reason.
      .also { obj -> if (readiness != RunReadiness.DEFAULT) obj.put("readiness", readiness.toJson()) }

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
        readiness = RunReadiness.fromJson(json.optJSONObject("readiness")),
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
  /**
   * The coordinate to fetch this model from, when there is one. Carried so a run planned against it can put it on the cell ([JobCell.modelRef]); null
   * for a locally imported file, which has nowhere to go back to.
   */
  val modelRef: Model? = null,
  /**
   * Whether these bytes are on disk. False for a catalog entry the run screen offers but nothing has downloaded, where [path] is the destination the
   * download *will* write and [sizeBytes] is the catalog's declared figure rather than a measured one.
   *
   * One type for both cases rather than a parallel "offer" type, because every consumer between the picker and [JobRunner.planCells] treats them
   * identically: the difference is the runner's to resolve, and it resolves it by path, not by this flag. iOS reached the same arrangement with
   * `DiscoveredModel`, which also carries the catalog and downloaded forms in one value.
   */
  val onDisk: Boolean = true,
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

  /**
   * An ISO-8601 instant as a local date and time, for a row a person reads.
   *
   * Minutes and no seconds: this dates a registration, where the day and the hour are what an operator correlates against a server-side record, and
   * the second is noise. Falls back to the stored string, as [shortDate] does, so a value written by some other build still renders as itself rather
   * than as an error.
   */
  fun dateTime(iso: String): String = runCatching { dateTimeFormatter.format(Instant.parse(iso)) }.getOrElse { iso }

  private val dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm").withZone(ZoneId.systemDefault())
}

object ByteFormat {
  private const val BYTES_PER_KB = 1_000L
  private const val BYTES_PER_MB = 1_000_000L
  private const val BYTES_PER_GB = 1_000_000_000L

  fun fileSize(bytes: Long): String {
    if (bytes < BYTES_PER_MB) return "${bytes / BYTES_PER_KB} KB"
    val mb = bytes / BYTES_PER_MB.toDouble()
    // Fixed locale so the decimal separator is "." everywhere (these strings show in download progress and notifications).
    if (bytes < BYTES_PER_GB) return "${formattedDecimal(mb, 1)} MB"
    return "${formattedDecimal(bytes / BYTES_PER_GB.toDouble(), 2)} GB"
  }

  private fun formattedDecimal(value: Double, decimalPlaces: Int): String =
    String.format(java.util.Locale.US, "%.${decimalPlaces}f", value).trimEnd('0').trimEnd('.')
}

fun JSONObject.optNullableString(name: String): String? {
  if (!has(name) || isNull(name)) return null
  return optString(name).takeIf { it.isNotEmpty() }
}

fun JSONObject.optNullableInt(name: String): Int? {
  if (!has(name) || isNull(name)) return null
  return optInt(name)
}

fun JSONObject.optNullableLong(name: String): Long? {
  if (!has(name) || isNull(name)) return null
  return optLong(name)
}

fun JSONObject.optNullableBoolean(name: String): Boolean? {
  if (!has(name) || isNull(name)) return null
  return optBoolean(name)
}

fun JSONObject.putOptString(name: String, value: String?): JSONObject = if (value == null) put(name, JSONObject.NULL) else put(name, value)

fun JSONObject.putOptBool(name: String, value: Boolean?): JSONObject = if (value == null) put(name, JSONObject.NULL) else put(name, value)
