package ai.liquid.pipette

import java.io.File
import org.json.JSONArray
import org.json.JSONObject

data class ResultSubmissionOutcome(val manifest: JobManifest, val submitted: Int, val errors: List<String>)

/**
 * Submits a completed job's results. [JobRunner] depends on this narrow interface so tests can substitute a fake; [ResultSubmissionService] is the
 * real, batching implementation.
 */
interface ResultSubmitter {
  fun submit(manifest: JobManifest, registration: RegistrationData): ResultSubmissionOutcome
}

class ResultSubmissionService(
  private val storage: LocalStorage,
  private val managementClient: ManagementClient,
  private val analytics: Analytics = NoOpAnalytics,
) : ResultSubmitter {
  /**
   * Persist [manifest] without clobbering the fields the user can edit while a submit is in flight.
   *
   * This service owns `serverJobId` and nothing else, but it saves the whole manifest, and a submit spans an HTTP round trip (15s connect, 120s
   * read), so a rename or a contribute-toggle made during that window would be silently reverted by a blind overwrite. Re-adopting both from disk
   * immediately before the write keeps the last writer of *those* fields the UI, which is the only writer of them. The same discipline
   * `JobRunner.saveManifest` applies for the same reason; it matters more here now that a run submits after every cell rather than once at the end.
   */
  private fun saveManifestPreservingUserEdits(manifest: JobManifest) {
    storage.loadJobManifest(manifest.jobId)?.let { latest ->
      manifest.title = latest.title
      manifest.contributeResults = latest.contributeResults
    }
    storage.saveJobManifest(manifest)
  }

  override fun submit(manifest: JobManifest, registration: RegistrationData): ResultSubmissionOutcome {
    var submitted = 0
    val errors = mutableListOf<String>()
    val cellIndexes = mutableListOf<Int>()
    val payloads = mutableListOf<JSONObject>()

    manifest.cells.forEachIndexed { index, cell ->
      if (cell.runStatus != CellRunStatus.COMPLETED || cell.serverJobId != null) return@forEachIndexed

      val existing = submittedServerJobId(manifest.jobId, cell.cellId)
      if (existing != null) {
        cell.serverJobId = existing
        saveManifestPreservingUserEdits(manifest)
        submitted += 1
        return@forEachIndexed
      }

      val dir = storage.submittablePayloadDir(manifest.jobId, cell.cellId) ?: return@forEachIndexed
      val payloadFile = File(dir, "payload.json")
      val payload = runCatching { JSONObject(payloadFile.readText()) }.getOrNull()
      if (payload == null) {
        errors += "${cell.benchmarkId}: failed to read payload"
      } else {
        cellIndexes += index
        payloads += payload
      }
    }

    val effectiveBatchSize = DEFAULT_BATCH_SIZE
    var start = 0
    while (start < payloads.size) {
      val end = minOf(start + effectiveBatchSize, payloads.size)
      val batch = JSONArray()
      for (i in start until end) batch.put(payloads[i])
      try {
        val responseText = managementClient.submitResultBatch(registration.serverUrl, registration.clientId, batch)
        val results = JSONObject(responseText).getJSONArray("results")
        for (i in 0 until results.length()) {
          val result = results.getJSONObject(i)
          val batchIndex = result.optInt("index", -1)
          if (batchIndex !in 0 until batch.length()) {
            errors += "unknown batch item: invalid response index"
            continue
          }
          val resultIndex = start + batchIndex
          val cell = manifest.cells[cellIndexes[resultIndex]]
          val error = result.optNullableString("error")
          if (error != null) {
            errors += "${cell.benchmarkId}: $error"
            storage.saveSubmission(CellSubmissionRecord.failed(listOf(error)), manifest.jobId, cell.cellId)
            continue
          }
          val serverJobId = result.optNullableString("job_id")
          if (serverJobId == null) {
            errors += "${cell.benchmarkId}: missing job_id in response"
            continue
          }
          storage.saveSubmission(CellSubmissionRecord.submitted(serverJobId), manifest.jobId, cell.cellId)
          cell.serverJobId = serverJobId
          saveManifestPreservingUserEdits(manifest)
          submitted += 1
        }
        if (results.length() < batch.length()) {
          errors += "Batch response omitted ${batch.length() - results.length()} result(s)"
        }
      } catch (error: Throwable) {
        for (resultIndex in start until end) {
          val cell = manifest.cells[cellIndexes[resultIndex]]
          val message = "batch submit failed: ${error.message ?: error.javaClass.simpleName}"
          errors += "${cell.benchmarkId}: $message"
          storage.saveSubmission(CellSubmissionRecord.failed(listOf(message)), manifest.jobId, cell.cellId)
        }
      }
      start = end
    }
    // Only when this call actually did something. Every cell here may already carry a serverJobId (a
    // re-submit of an already-uploaded job, or a run whose cells all failed), and capturing those
    // no-op passes would swamp the event with rows describing no user-visible action. Error TEXT is
    // never sent: `errors` holds server messages that embed benchmark ids and the server URL, only
    // whether any occurred.
    if (submitted > 0 || errors.isNotEmpty()) {
      analytics.capture(
        AnalyticsEvents.RESULTS_SUBMITTED,
        mapOf(AnalyticsEvents.JOB_ID to manifest.jobId, AnalyticsEvents.SUBMITTED_COUNT to submitted, AnalyticsEvents.OK to errors.isEmpty()),
      )
      // Flush immediately, because this is now the one event that fires *inside* a run. Before the
      // per-cell upload, `submit` ran once after the whole cell loop and the queue could safely be
      // left for the SDK's periodic timer; now it runs between cells, and a row left queued is a row
      // that timer can POST while the next cell is being timed. That is the exact perturbation the
      // taxonomy is built to avoid, and [AnalyticsEvents] states the obligation this discharges.
      // The flush costs nothing extra here: the upload above just did network I/O on this thread, so
      // we are already off the measurement path.
      analytics.flush()
    }
    return ResultSubmissionOutcome(manifest, submitted, errors)
  }

  fun submitCell(jobId: String, cellId: String, registration: RegistrationData): CellSubmissionRecord? {
    submittedServerJobId(jobId, cellId)?.let {
      return CellSubmissionRecord.submitted(it)
    }
    val dir = storage.submittablePayloadDir(jobId, cellId) ?: return null
    val payload = File(dir, "payload.json").readText()
    return try {
      val response = JSONObject(managementClient.submitResult(registration.serverUrl, registration.clientId, payload))
      val serverJobId = response.optNullableString("job_id") ?: cellId
      CellSubmissionRecord.submitted(serverJobId).also { storage.saveSubmission(it, jobId, cellId) }
    } catch (error: Throwable) {
      CellSubmissionRecord.failed(listOf(error.message ?: error.javaClass.simpleName)).also { storage.saveSubmission(it, jobId, cellId) }
    }
  }

  private fun submittedServerJobId(jobId: String, cellId: String): String? {
    val submission = storage.loadSubmission(jobId, cellId) ?: return null
    if (submission.status != "submitted") return null
    return submission.serverJobId?.takeIf { it.isNotBlank() }
  }

  companion object {
    private const val DEFAULT_BATCH_SIZE = 1000
  }
}
