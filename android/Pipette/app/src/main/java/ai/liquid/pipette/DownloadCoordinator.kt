package ai.liquid.pipette

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.io.File
import java.net.URL
import java.util.concurrent.TimeUnit

data class DownloadProgress(val bytesRead: Long, val totalBytes: Long, val message: String)

data class ActiveDownload(
  val key: String,
  val filename: String,
  val repo: String?,
  val bytesRead: Long,
  val totalBytes: Long,
  val message: String,
  val state: String,
) {
  val isPaused: Boolean
    get() = state == DownloadWorker.STATE_PAUSED

  val isFailed: Boolean
    get() = state == DownloadWorker.STATE_FAILED

  /**
   * Stopped by a transient network error, not by the user and not terminally. WorkManager re-runs it on its own once there is a network; this exists
   * so the UI can say "waiting" instead of showing it as a failure.
   */
  val isWaitingForNetwork: Boolean
    get() = state == DownloadWorker.STATE_WAITING_NETWORK

  /**
   * Paused/failed downloads can be resumed (re-enqueued); running/queued ones can be paused.
   *
   * A network-interrupted download is deliberately NOT resumable: it is still active, waiting on the worker's own backoff to re-run it, so offering
   * "Resume" for something already trying to resume would be noise. It is pausable instead (that is how the user stops the retry loop), and Cancel
   * stays available either way.
   */
  val canResume: Boolean
    get() = isPaused || isFailed

  val canPause: Boolean
    get() = state == DownloadWorker.STATE_RUNNING || state == DownloadWorker.STATE_QUEUED || isWaitingForNetwork

  /**
   * Human-readable form of [state] for a status badge.
   *
   * Lives here rather than in a screen because there are two of them (the Compose Models tab and the legacy Views one) and they were rendering the
   * same state differently: one mapped it, the other printed the raw string. That is fine while every state is a single lowercase word, and it starts
   * leaking identifiers ("waiting_network") the moment one isn't.
   */
  val displayLabel: String
    get() =
      when (state) {
        DownloadWorker.STATE_QUEUED -> "Queued"
        DownloadWorker.STATE_RUNNING -> "Downloading"
        DownloadWorker.STATE_PAUSED -> "Paused"
        DownloadWorker.STATE_FAILED -> "Failed"
        DownloadWorker.STATE_WAITING_NETWORK -> "Waiting for network"
        else -> state.replaceFirstChar { it.uppercase() }
      }
}

/**
 * Orchestrates resumable model downloads on top of WorkManager (the transfer itself lives in [DownloadWorker]). Each download is a unique work named
 * by its `<repo>/<filename>` key, so it survives process death and resumes from its `.part`. Pause/cancel are both WorkManager cancellation; this
 * class owns the distinction — pause keeps the `.part` and the persisted record so [resume] can re-enqueue, cancel deletes both.
 *
 * The public surface (enqueue + callbacks, [activeDownloads], [cancel]) is unchanged from the old DownloadManager implementation; [pause]/[resume]
 * are new. Live progress and the per-download callbacks flow through [DownloadRegistry], which the worker updates from the same process.
 */
class DownloadCoordinator(context: Context, private val storage: LocalStorage, private val analytics: Analytics = NoOpAnalytics) {
  private val appContext = context.applicationContext
  private val workManager = WorkManager.getInstance(appContext)

  init {
    DownloadNotifications.ensureChannel(appContext)
    // Off the main thread: restore() reads SharedPreferences + stats .part files, and this runs in PipetteApp.onCreate on every cold start.
    Thread({ restore() }, "download-restore").start()
  }

  fun enqueueDownload(
    urlString: String,
    repo: String?,
    familyId: String? = null,
    displayName: String? = null,
    onProgress: (DownloadProgress) -> Unit = {},
    onComplete: (ModelFile) -> Unit = {},
    onFailure: (Throwable) -> Unit = {},
  ): ActiveDownload {
    val record = buildRecord(urlString, repo, familyId, displayName)
    require(!File(record.destPath).exists()) { "${record.filename} already exists" }
    require(DownloadRegistry.get(record.key) == null) { "${record.filename} is already downloading" }

    DownloadStore.saveRecord(appContext, record)
    DownloadRegistry.setCallbacks(record.key, DownloadRegistry.Callbacks(onProgress, onComplete, onFailure))
    val queued = ActiveDownload(record.key, record.filename, record.repo, 0, -1, "Queued…", DownloadWorker.STATE_QUEUED)
    DownloadRegistry.put(queued)
    enqueueWork(record, ExistingWorkPolicy.KEEP)
    // Only a genuinely new transfer reaches here: the requires above reject an already-present
    // file and an in-flight duplicate, so this can't double-count. The matching terminal events
    // are captured in DownloadWorker, which also sees transfers that finish after a restart.
    analytics.capture(AnalyticsEvents.MODEL_DOWNLOAD_STARTED, mapOf(AnalyticsEvents.MODEL_ID to record.filename))
    return queued
  }

  fun activeDownloads(): List<ActiveDownload> = DownloadRegistry.all()

  fun pause(key: String) {
    val current = DownloadRegistry.get(key) ?: return
    // Persist + publish PAUSED before cancelling the work, so an in-flight worker emit() sees the paused state (via putIfActive) and can't flip the
    // row back to running.
    DownloadStore.updateState(appContext, key, DownloadWorker.STATE_PAUSED)
    DownloadRegistry.put(current.copy(message = "Paused", state = DownloadWorker.STATE_PAUSED))
    workManager.cancelUniqueWork(key)
    DownloadNotifications.update(appContext, key, current.filename, current.bytesRead, current.totalBytes, paused = true)
  }

  fun resume(key: String) {
    val record = DownloadStore.records(appContext).firstOrNull { it.key == key } ?: return
    DownloadStore.updateState(appContext, key, DownloadWorker.STATE_QUEUED)
    val existing = File(record.partialPath).let { if (it.exists()) it.length() else 0L }
    DownloadRegistry.put(ActiveDownload(key, record.filename, record.repo, existing, -1, "Resuming…", DownloadWorker.STATE_QUEUED))
    // REPLACE restarts the unique work; the per-key mutex in DownloadWorker keeps the resuming worker from writing the .part until the outgoing one
    // has stopped.
    enqueueWork(record.copy(state = DownloadWorker.STATE_QUEUED), ExistingWorkPolicy.REPLACE)
  }

  /**
   * Cancel every download this device knows about, live or paused, and drop their partial files.
   *
   * For the sign-out reset (PIP-459), which deletes `models/`: a transfer left running would recreate that tree and land a model on a device that has
   * just been wiped. [DownloadStore] is unioned in rather than trusting the registry alone, because [restore] repopulates the registry from those
   * records asynchronously at startup: a record whose row has not been restored yet would otherwise survive the cancel and resume later.
   *
   * Deliberately a loop over the single-key [cancel] rather than a bulk path that snapshots the records once. That would save a re-read and a rewrite
   * per key, both against a memory-backed [android.content.SharedPreferences] and both trivial at the count involved (live plus paused downloads, so
   * single digits), and it would cost a second spelling of what cancelling one download means: registry entry, work request, partial file,
   * notification, stored record. A key missed by a divergent copy resumes a multi-GB transfer onto a device that has just been wiped.
   */
  fun cancelAll() {
    val keys = (DownloadRegistry.all().map { it.key } + DownloadStore.records(appContext).map { it.key }).distinct()
    keys.forEach { cancel(it) }
  }

  fun cancel(key: String) {
    // Remove the registry entry first: a still-running worker's onStopped() reads the missing entry as "cancelled" and deletes any .part it recreated
    // (and its own emit() bails via putIfActive), so no ghost row/notification or orphaned partial survives the async work cancellation.
    DownloadRegistry.remove(key)
    workManager.cancelUniqueWork(key)
    DownloadStore.records(appContext).firstOrNull { it.key == key }?.let { storage.deleteDownloadArtifact(File(it.partialPath)) }
    DownloadStore.clearRecord(appContext, key)
    DownloadNotifications.cancel(appContext, key)
  }

  private fun enqueueWork(record: DownloadStore.Record, policy: ExistingWorkPolicy) {
    val data =
      Data.Builder()
        .putString(DownloadWorker.KEY_KEY, record.key)
        .putString(DownloadWorker.KEY_URL, record.urlString)
        .putString(DownloadWorker.KEY_REPO, record.repo)
        .putString(DownloadWorker.KEY_FILENAME, record.filename)
        .putString(DownloadWorker.KEY_FAMILY, record.familyId)
        .putString(DownloadWorker.KEY_DISPLAY, record.displayName)
        .putString(DownloadWorker.KEY_DEST, record.destPath)
        .putString(DownloadWorker.KEY_PART, record.partialPath)
        .build()
    val request =
      OneTimeWorkRequestBuilder<DownloadWorker>()
        .setInputData(data)
        .addTag(WORK_TAG)
        // Backoff, but deliberately NO network constraint. A CONNECTED constraint looks like the obvious way to auto-resume, and it is a trap here:
        // WorkManager stops a *running* worker whose constraints stop being met, so the moment Wi-Fi actually drops (the case this whole retry path
        // exists for) the worker is cancelled mid-transfer instead of reaching its own error handling. It would take the isStopped branch, leave the
        // row at STATE_RUNNING with a frozen byte count, and never mark it waiting. Letting the worker own its failure keeps the .part, the state and
        // the retry decision in one place; the backoff then re-runs it, and an attempt made while still offline fails fast and cheaply.
        .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, RETRY_BACKOFF_SECONDS, TimeUnit.SECONDS)
        .build()
    workManager.enqueueUniqueWork(record.key, policy, request)
  }

  private fun buildRecord(urlString: String, repo: String?, familyId: String?, displayName: String?): DownloadStore.Record {
    val parsed = parseDownloadInput(urlString)
    val url = URL(parsed.url)
    val filename = url.path.substringAfterLast('/').ifBlank { "model-${System.currentTimeMillis()}.gguf" }
    require(filename.endsWith(".gguf", ignoreCase = true)) { "Only .gguf model files are supported" }
    val resolvedRepo = repo?.takeIf { it.isNotBlank() } ?: parsed.repo
    val dest = storage.modelDestFile(resolvedRepo, filename)
    val partial = File(dest.parentFile, "${dest.name}.part")
    val key = LocalStorage.modelRelativePath(resolvedRepo, filename)
    return DownloadStore.Record(
      key = key,
      filename = filename,
      urlString = parsed.url,
      repo = resolvedRepo,
      familyId = familyId,
      displayName = displayName,
      destPath = dest.absolutePath,
      partialPath = partial.absolutePath,
    )
  }

  // Rebuild the in-memory list from persisted records after a (re)launch, keyed off each record's own lifecycle state. Paused/failed rows stay
  // resumable (the user taps Resume); queued rows are re-enqueued with KEEP so a record can never become a row with no backing worker — WorkManager
  // keeps an already-pending work as-is and (re)starts a missing one from its .part. A failed record persists as FAILED (not QUEUED) so it surfaces a
  // Resume affordance instead of a permanently stuck "Reconnecting…" row that no terminally-failed work would ever re-run.
  //
  // A WAITING_NETWORK record falls through to that same re-enqueue branch on purpose: it was never a failure, and re-enqueueing is exactly what
  // should
  // happen to it. The fresh worker starts a new attempt sequence; if the device is still offline that attempt fails fast, republishes
  // WAITING_NETWORK and backs off, so the row simply reads "Reconnecting…" until connectivity returns.
  private fun restore() {
    DownloadStore.records(appContext).forEach { record ->
      val existing = File(record.partialPath).let { if (it.exists()) it.length() else 0L }
      when (record.state) {
        DownloadWorker.STATE_PAUSED ->
          DownloadRegistry.put(ActiveDownload(record.key, record.filename, record.repo, existing, -1, "Paused", DownloadWorker.STATE_PAUSED))
        DownloadWorker.STATE_FAILED ->
          DownloadRegistry.put(ActiveDownload(record.key, record.filename, record.repo, existing, -1, "Download failed", DownloadWorker.STATE_FAILED))
        else -> {
          DownloadRegistry.put(ActiveDownload(record.key, record.filename, record.repo, existing, -1, "Reconnecting…", DownloadWorker.STATE_QUEUED))
          enqueueWork(record, ExistingWorkPolicy.KEEP)
        }
      }
    }
  }

  companion object {
    private const val WORK_TAG = "pipette-download"

    /**
     * First delay before re-running a download that hit a transient network error; WorkManager doubles it per attempt (30s, 1m, 2m, 4m…).
     *
     * Paired with [DownloadWorker.MAX_NETWORK_ATTEMPTS], which is what actually bounds the sequence. WorkManager never gives up on its own, and its
     * backoff would otherwise keep growing toward a 5-hour ceiling.
     */
    private const val RETRY_BACKOFF_SECONDS = 30L

    fun parseDownloadInput(input: String): ParsedDownload {
      val trimmed = input.trim()
      require(trimmed.isNotBlank()) { "Enter a Hugging Face identifier or URL" }
      if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
        return ParsedDownload(normalizeHuggingFaceFileUrl(trimmed), huggingFaceRepoFromUrl(trimmed))
      }

      val colonIndex = trimmed.indexOf(':')
      if (colonIndex > 0) {
        val repo = trimmed.substring(0, colonIndex).trim()
        val suffix = trimmed.substring(colonIndex + 1).trim()
        require(repo.isNotBlank() && suffix.isNotBlank()) { "Use org/repo-GGUF:quant, org/repo-GGUF:file.gguf, org/repo/file.gguf, or a full URL" }
        val repoName = repo.substringAfterLast('/')
        // Case-insensitive: not every publisher uppercases the suffix
        // (`prism-ml/Bonsai-27B-gguf`), and the file layout is the same either way.
        val stem = if (repoName.endsWith("-GGUF", ignoreCase = true)) repoName.dropLast("-GGUF".length) else repoName
        val filename = if (suffix.endsWith(".gguf", ignoreCase = true)) suffix else "$stem-$suffix.gguf"
        val parsedRepo = HfRepo.parseSlug(repo)
        GgufFilename.parse(filename)
        return ParsedDownload(huggingFaceResolveUrl(parsedRepo.toString(), filename), parsedRepo.toString())
      }

      val parts = trimmed.split("/")
      require(parts.size >= 3) { "Use org/repo-GGUF:quant, org/repo-GGUF:file.gguf, org/repo/file.gguf, or a full URL" }
      val parsedRepo = HfRepo.parseSlug("${parts[0]}/${parts[1]}")
      val filename = parts.drop(2).joinToString("/")
      GgufFilename.parse(filename.substringAfterLast('/'))
      return ParsedDownload(huggingFaceResolveUrl(parsedRepo.toString(), filename), parsedRepo.toString())
    }

    private fun huggingFaceResolveUrl(repo: String, filename: String): String {
      val encodedFile =
        filename.split("/").joinToString("/") { segment -> java.net.URLEncoder.encode(segment, Charsets.UTF_8.name()).replace("+", "%20") }
      return "https://huggingface.co/$repo/resolve/main/$encodedFile"
    }

    private fun huggingFaceRepoFromUrl(url: String): String? {
      if (!url.contains("huggingface.co/")) return null
      val afterHost = url.substringAfter("huggingface.co/").substringBefore("?").substringBefore("#")
      val parts = afterHost.split("/")
      return if (parts.size >= 2 && parts[0].isNotBlank() && parts[1].isNotBlank()) "${parts[0]}/${parts[1]}" else null
    }

    private fun normalizeHuggingFaceFileUrl(url: String): String {
      if (!url.contains("huggingface.co/")) return url
      val beforeHost = url.substringBefore("huggingface.co/")
      val afterHost = url.substringAfter("huggingface.co/")
      val queryIndex = afterHost.indexOf('?')
      val query = if (queryIndex >= 0) afterHost.substring(queryIndex) else ""
      val path = afterHost.substringBefore("?")
      val parts = path.split("/")
      if (parts.size >= 5 && parts[2] == "blob") {
        val normalizedPath = (parts.take(2) + "resolve" + parts.drop(3)).joinToString("/")
        return "${beforeHost}huggingface.co/$normalizedPath$query"
      }
      return url
    }
  }
}

data class ParsedDownload(val url: String, val repo: String?)
