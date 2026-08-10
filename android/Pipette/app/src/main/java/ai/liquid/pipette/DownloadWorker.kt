package ai.liquid.pipette

import android.app.ForegroundServiceStartNotAllowedException
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.SystemClock
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import java.io.EOFException
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.ProtocolException
import java.net.SocketException
import java.net.URL
import java.net.UnknownHostException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * A response that ended before delivering the bytes it promised: a dropped connection rather than a server saying no.
 *
 * A named type rather than a bare [IOException] with a formatted message so [isRecoverableNetworkError] can classify it without matching on text. iOS
 * sees this same case as `URLError.networkConnectionLost` and treats it as recoverable; so do we.
 */
class IncompleteDownloadException(bytesRead: Long, total: Long) : IOException("Incomplete download: $bytesRead of $total bytes")

/**
 * Whether a download failure is transient connectivity worth keeping the download resumable for, rather than a terminal error no retry fixes.
 *
 * The Android twin of the iOS client's `isRecoverableNetworkError` / `downloadURLErrorInfo`, and deliberately the same set: a timeout, a DNS failure,
 * an unreachable host, a dropped connection, or a truncated response. Everything else (an HTTP status, a 416, too many redirects, a TLS failure) is
 * terminal on both platforms, because no amount of retrying fixes it.
 *
 * `SocketException` covers the connect/reset family (`ConnectException`, `NoRouteToHostException`) and `InterruptedIOException` covers the timeouts
 * (`SocketTimeoutException`); they are listed as the base types deliberately, since the subtype set differs across OEM stacks. The three EOF shapes
 * matter more than they look: this app streams over `HttpURLConnection`, which on Android is OkHttp-backed, so a connection dropped mid-body usually
 * arrives as `ProtocolException("unexpected end of stream")` or `EOFException`, **not** as anything under `SocketException`. Only a clean early
 * `read() == -1` reaches [IncompleteDownloadException]. Missing those two would send the most common real-world drop straight to a terminal failure.
 *
 * Walks `cause` because the streaming stack wraps: an `IOException` with a `SocketException` cause is the same interrupted transfer.
 *
 * Top-level rather than a [DownloadWorker] member so it can be unit-tested without constructing a worker.
 */
fun isRecoverableNetworkError(error: Throwable?): Boolean =
  when (error) {
    null -> false
    is UnknownHostException,
    is SocketException,
    is java.io.InterruptedIOException,
    is ProtocolException,
    is EOFException,
    is IncompleteDownloadException -> true
    // A foreground-service start refused because the app is in the background (API 31+). Every auto-retry re-enters setForeground() minutes later,
    // typically backgrounded, so treating this as terminal would let the very mechanism built to survive a blip kill the download instead.
    is ForegroundServiceStartNotAllowedException -> true
    else -> isRecoverableNetworkError(error.cause)
  }

/**
 * Resumable model download as a WorkManager foreground worker (mirrors leap-android-sdk's downloader). Streams the file over HttpURLConnection with
 * an HTTP `Range` header so an interrupted `.part` resumes from where it stopped, shows an ongoing notification with pause/cancel actions, and on
 * success moves the file into place and registers it. WorkManager re-runs an unfinished unique work after process death, so downloads survive being
 * killed.
 *
 * Pause and cancel both arrive as WorkManager cancellation (a [CancellationException]); the coordinator owns the distinction by what it leaves in the
 * stores — on pause it keeps a PAUSED [DownloadRegistry] entry and the [DownloadStore] record, on cancel it removes them. The worker reads that on
 * stop ([onStopped]): a missing registry entry means cancel, so it drops any `.part` it may still have open; a PAUSED entry means keep it for resume.
 */
class DownloadWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {

  override suspend fun getForegroundInfo(): ForegroundInfo {
    val key = inputData.getString(KEY_KEY).orEmpty()
    val filename = inputData.getString(KEY_FILENAME).orEmpty()
    val live = DownloadRegistry.get(key)
    val notification = DownloadNotifications.build(applicationContext, key, filename, live?.bytesRead ?: 0, live?.totalBytes ?: 0, paused = false)
    return ForegroundInfo(DownloadNotifications.notificationId(key), notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
  }

  @Suppress("TooGenericExceptionCaught") // a download worker must turn any failure into a "failed" state for the UI/notification.
  override suspend fun doWork(): Result {
    val inputs = parseInputs() ?: return Result.failure()
    return try {
      // Inside the try: starting the foreground service can throw (e.g. a background-restricted FGS start on Android 12+), and that must run the
      // same failure cleanup as any other error rather than escaping doWork() with a stale row left behind.
      setForeground(getForegroundInfo())
      download(inputs)
      completeDownload(inputs)
      Result.success()
    } catch (cancel: CancellationException) {
      onStopped(inputs) // pause keeps the .part; cancel drops it
      throw cancel
    } catch (error: Exception) {
      // A cancel can surface as an IOException (closed socket) rather than CancellationException; don't record it as a failure.
      if (isStopped) {
        onStopped(inputs)
        throw CancellationException("Download stopped")
      }
      // A flaky connection must not turn a multi-gigabyte transfer into a failed row the user has to notice and resume by hand: keep the .part and
      // hand the work back to WorkManager, which re-runs it on an exponential backoff.
      //
      // Bounded by [MAX_NETWORK_ATTEMPTS]. WorkManager imposes no attempt limit of its own, so without this a permanently broken host (a typo'd repo,
      // a decommissioned CDN) throws UnknownHostException forever and the row sits at "Waiting for network…" indefinitely, with no failure text and
      // no Resume affordance. Falling through to fail() after a few attempts restores the old behaviour for that case while still absorbing a blip.
      if (isRecoverableNetworkError(error) && runAttemptCount < MAX_NETWORK_ATTEMPTS - 1) {
        waitForNetwork(inputs)
        Result.retry()
      } else {
        fail(inputs, error)
        Result.failure()
      }
    }
  }

  private suspend fun download(inputs: Inputs) =
    // Serialize writers for this key: WorkManager's REPLACE starts the resuming worker before the outgoing one has stopped, so without this two
    // workers could append to the same .part concurrently and corrupt it.
    DownloadLocks.forKey(inputs.key).withLock {
      withContext(Dispatchers.IO) {
        inputs.partial.parentFile?.mkdirs()
        val existing = if (inputs.partial.exists()) inputs.partial.length() else 0L
        val connection = openConnection(inputs.url, existing)
        try {
          val transfer = resolveTransfer(connection, existing) ?: return@withContext // server says the .part is already complete
          if (!transfer.append && inputs.partial.exists()) inputs.partial.delete()
          stream(connection, inputs, transfer)
        } finally {
          runCatching { connection.errorStream?.close() }
          connection.disconnect() // always release the socket, including the error/416 paths
        }
      }
    }

  /** Open the URL, following redirects manually so the HF bearer token is never replayed to a non-`huggingface.co` redirect target (e.g. a CDN). */
  private fun openConnection(urlString: String, existing: Long): HttpURLConnection {
    var target = URL(urlString)
    var redirects = 0
    while (true) {
      val connection = buildConnection(target, existing)
      if (connection.responseCode !in REDIRECT_CODES) return connection
      val location = connection.getHeaderField("Location")
      connection.disconnect()
      if (location.isNullOrBlank() || redirects++ >= MAX_REDIRECTS) throw IOException("Too many redirects for $urlString")
      target = URL(target, location) // resolve relative redirects against the current URL
    }
  }

  private fun buildConnection(url: URL, existing: Long): HttpURLConnection =
    (url.openConnection() as HttpURLConnection).apply {
      requestMethod = "GET"
      connectTimeout = TIMEOUT_MS
      readTimeout = TIMEOUT_MS
      instanceFollowRedirects = false
      setRequestProperty("User-Agent", USER_AGENT)
      if (isHuggingFaceHost(url.host)) {
        Secrets(applicationContext).loadHfToken()?.let { setRequestProperty("Authorization", "Bearer $it") }
      }
      if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
      connect()
    }

  /** Decide how to write the body from the response, or null when the server reports the `.part` is already complete (416). */
  private fun resolveTransfer(connection: HttpURLConnection, existing: Long): Transfer? =
    when (connection.responseCode) {
      HttpURLConnection.HTTP_PARTIAL -> {
        val rangeTotal = connection.getHeaderField("Content-Range")?.substringAfterLast('/')?.trim()?.toLongOrNull()
        val contentLength = connection.contentLengthLong
        // Total = the range's grand total when given, else existing + this chunk — but only when the chunk length is known; never `existing - 1`
        // from a -1 (unknown/chunked) length, which would be a bogus positive total that defeats the short-read guard below.
        val total = rangeTotal ?: if (contentLength >= 0) existing + contentLength else UNKNOWN_TOTAL
        Transfer(append = true, total = total, startBytes = existing)
      }
      HttpURLConnection.HTTP_OK -> Transfer(append = false, total = connection.contentLengthLong, startBytes = 0L)
      // 416 means "range not satisfiable" — only treat it as "already complete" when the .part really is the full size.
      HTTP_RANGE_NOT_SATISFIABLE -> {
        val total = connection.getHeaderField("Content-Range")?.substringAfterLast('/')?.trim()?.toLongOrNull()
        if (total != null && existing >= total) null else throw IOException("HTTP 416 (range not satisfiable)")
      }
      else -> throw IOException("HTTP ${connection.responseCode} ${connection.responseMessage.orEmpty()}".trim())
    }

  private suspend fun stream(connection: HttpURLConnection, inputs: Inputs, transfer: Transfer) =
    withContext(Dispatchers.IO) {
      val startedAt = SystemClock.elapsedRealtime()
      var bytesRead = transfer.startBytes
      var lastEmit = 0L
      connection.inputStream.use { input ->
        FileOutputStream(inputs.partial, transfer.append).use { output ->
          val buffer = ByteArray(BUFFER_SIZE)
          while (true) {
            if (isStopped) throw CancellationException("Download stopped")
            val read = input.read(buffer)
            if (read < 0) break
            output.write(buffer, 0, read)
            bytesRead += read
            val now = SystemClock.elapsedRealtime()
            if (now - lastEmit >= PROGRESS_INTERVAL_MS) {
              emit(inputs, bytesRead, transfer, startedAt)
              lastEmit = now
            }
          }
          output.flush()
        }
      }
      // A stop closes the stream and surfaces as an early EOF; treat it as cancellation, not a finished file.
      if (isStopped) throw CancellationException("Download stopped")
      // Never finalize a short read (cancel, dropped connection, truncated response) as a complete model — keep the .part for a resume.
      if (transfer.total > 0 && bytesRead < transfer.total) {
        throw IncompleteDownloadException(bytesRead, transfer.total)
      }
      emit(inputs, bytesRead, transfer, startedAt)
    }

  private fun emit(inputs: Inputs, bytesRead: Long, transfer: Transfer, startedAt: Long) {
    if (isStopped) return
    val elapsed = (SystemClock.elapsedRealtime() - startedAt).coerceAtLeast(1L)
    val bytesPerSecond = ((bytesRead - transfer.startBytes) * MILLIS_PER_SECOND) / elapsed
    val speed = if (bytesPerSecond > 0) " @ ${ByteFormat.fileSize(bytesPerSecond)}/s" else ""
    val message =
      if (transfer.total > 0) "${ByteFormat.fileSize(bytesRead)} / ${ByteFormat.fileSize(transfer.total)}$speed"
      else "${ByteFormat.fileSize(bytesRead)}$speed"
    val running = ActiveDownload(inputs.key, inputs.filename, inputs.repo, bytesRead, transfer.total, message, STATE_RUNNING)
    // Bail if the coordinator paused/cancelled this download since the last tick — don't resurrect a running row/notification it just cleared.
    if (!DownloadRegistry.putIfActive(running)) return
    DownloadRegistry.callbacks(inputs.key)?.onProgress?.invoke(DownloadProgress(bytesRead, transfer.total, message))
    DownloadNotifications.update(applicationContext, inputs.key, inputs.filename, bytesRead, transfer.total, paused = false)
  }

  private fun completeDownload(inputs: Inputs) {
    val model = moveIntoPlace(inputs)
    // Clear the persisted record so a finished download isn't re-queued by DownloadCoordinator.restore() on the next launch.
    DownloadStore.clearRecord(applicationContext, inputs.key)
    DownloadRegistry.callbacks(inputs.key)?.onComplete?.invoke(model)
    DownloadRegistry.remove(inputs.key)
    DownloadNotifications.cancel(applicationContext, inputs.key)
    analytics()
      .capture(
        AnalyticsEvents.MODEL_DOWNLOAD_COMPLETED,
        mapOf(AnalyticsEvents.MODEL_ID to inputs.filename, AnalyticsEvents.SIZE_BYTES to model.sizeBytes),
      )
  }

  /** Called when the worker is stopped: cancel (registry entry gone) drops any `.part` we may have re-created; pause/system-stop keeps it. */
  private fun onStopped(inputs: Inputs) {
    if (DownloadRegistry.get(inputs.key) == null) {
      storage().deleteDownloadArtifact(inputs.partial) // drop the .part and prune the now-empty repo dir
      DownloadStore.clearRecord(applicationContext, inputs.key)
      DownloadNotifications.cancel(applicationContext, inputs.key)
    }
  }

  /**
   * A transient connectivity error stopped the transfer. Unlike [fail], keep the download resumable: leave the `.part` alone, persist and publish
   * [STATE_WAITING_NETWORK], and say nothing alarming: a blip on the way to a 4 GB file is not a failure the user needs to act on.
   *
   * Deliberately NOT [STATE_PAUSED]: [DownloadRegistry.putIfActive] refuses to write over a paused or failed row (so a straggling `emit()` can't
   * resurrect a download the user just paused), which would leave the retrying worker unable to publish any progress and the row stuck at "Paused"
   * while bytes moved underneath it. A distinct state also keeps this honestly separate from a user-initiated pause, the same distinction the iOS
   * client draws with its `interruptedByNetwork` flag.
   *
   * No `onFailure` callback for the same reason: nothing has failed yet.
   */
  private fun waitForNetwork(inputs: Inputs) {
    val live = DownloadRegistry.get(inputs.key)
    // putIfActive, not put: `pause()` and `cancel()` both run on another thread and are only best-effort ordered against the `isStopped` check above.
    // A blind write would either resurrect a row the user just cancelled (leaving a "Waiting for network…" entry with no record and no .part, the
    // ghost row putIfActive exists to prevent) or overwrite a PAUSED row whose work is already cancelled, so no retry would ever fire and, since a
    // waiting row is deliberately not resumable, the download would be stuck until the next launch. Losing the race simply means the user's action
    // wins, which is correct.
    val published =
      DownloadRegistry.putIfActive(
        ActiveDownload(
          inputs.key,
          inputs.filename,
          inputs.repo,
          live?.bytesRead ?: 0,
          live?.totalBytes ?: -1,
          "Waiting for network…",
          STATE_WAITING_NETWORK,
        )
      )
    if (!published) return
    DownloadStore.updateState(applicationContext, inputs.key, STATE_WAITING_NETWORK)
    // Drop the notification rather than re-posting it. WorkManager tears down this worker's foreground service on return and cancels the notification
    // id with it, so anything posted here would vanish anyway; and posting it as `paused = true`, the only "not running" variant available, renders
    // "Paused" plus a Resume action, which is both untrue and an affordance the in-app row deliberately withholds.
    DownloadNotifications.cancel(applicationContext, inputs.key)
  }

  private fun fail(inputs: Inputs, error: Throwable) {
    // Persist FAILED so a relaunch restores a resumable row (a record left as QUEUED would never re-run a terminally-failed work).
    DownloadStore.updateState(applicationContext, inputs.key, STATE_FAILED)
    val live = DownloadRegistry.get(inputs.key)
    DownloadRegistry.put(
      ActiveDownload(
        inputs.key,
        inputs.filename,
        inputs.repo,
        live?.bytesRead ?: 0,
        live?.totalBytes ?: -1,
        error.message ?: "Download failed",
        STATE_FAILED,
      )
    )
    DownloadRegistry.callbacks(inputs.key)?.onFailure?.invoke(error)
    DownloadNotifications.cancel(applicationContext, inputs.key)
    // Error KIND only, never error.message: a download failure message embeds the source URL.
    analytics()
      .capture(
        AnalyticsEvents.MODEL_DOWNLOAD_FAILED,
        mapOf(AnalyticsEvents.MODEL_ID to inputs.filename, AnalyticsEvents.ERROR_KIND to AnalyticsEvents.errorKind(error)),
      )
  }

  private fun moveIntoPlace(inputs: Inputs): ModelFile {
    if (!inputs.partial.exists()) throw IOException("Download finished, but the file was not found")
    inputs.dest.parentFile?.mkdirs()
    if (inputs.dest.exists()) inputs.dest.delete()
    if (!inputs.partial.renameTo(inputs.dest)) {
      inputs.partial.copyTo(inputs.dest, overwrite = true)
      inputs.partial.delete()
    }
    return storage()
      .registerModelFile(
        file = inputs.dest,
        repo = inputs.repo,
        displayName = inputs.displayName ?: inputs.repo?.let { ModelTemplateCatalog.repoToName[it] },
        familyId = inputs.familyId,
      )
  }

  /**
   * The app-scoped storage (same [ModelStore] the UI reads) so a completed download invalidates the cached catalog; falls back outside the app
   * process.
   */
  private fun storage(): LocalStorage = (applicationContext as? PipetteApp)?.containerOrNull?.storage ?: LocalStorage(applicationContext)

  /**
   * The app's analytics seam, or [NoOpAnalytics] when the container isn't available.
   *
   * Terminal download outcomes are captured here in the worker rather than around [DownloadCoordinator.enqueueDownload]'s callbacks, because those
   * callbacks only exist for downloads enqueued in this process session: a transfer that WorkManager finishes after a process restart has no
   * registered callback, so instrumenting there would systematically under-count completions against starts.
   */
  private fun analytics(): Analytics = (applicationContext as? PipetteApp)?.containerOrNull?.analytics ?: NoOpAnalytics

  @Suppress("ReturnCount") // guard clauses for required WorkManager inputs read clearer than one compound condition.
  private fun parseInputs(): Inputs? {
    val key = inputData.getString(KEY_KEY) ?: return null
    val url = inputData.getString(KEY_URL) ?: return null
    val filename = inputData.getString(KEY_FILENAME) ?: return null
    val dest = inputData.getString(KEY_DEST) ?: return null
    val part = inputData.getString(KEY_PART) ?: return null
    return Inputs(
      key = key,
      url = url,
      filename = filename,
      repo = inputData.getString(KEY_REPO),
      familyId = inputData.getString(KEY_FAMILY),
      displayName = inputData.getString(KEY_DISPLAY),
      dest = File(dest),
      partial = File(part),
    )
  }

  private data class Inputs(
    val key: String,
    val url: String,
    val filename: String,
    val repo: String?,
    val familyId: String?,
    val displayName: String?,
    val dest: File,
    val partial: File,
  )

  private data class Transfer(val append: Boolean, val total: Long, val startBytes: Long)

  companion object {
    const val KEY_KEY = "key"
    const val KEY_URL = "url"
    const val KEY_REPO = "repo"
    const val KEY_FILENAME = "filename"
    const val KEY_FAMILY = "familyId"
    const val KEY_DISPLAY = "displayName"
    const val KEY_DEST = "dest"
    const val KEY_PART = "part"

    const val STATE_QUEUED = "queued"
    const val STATE_RUNNING = "running"
    const val STATE_PAUSED = "paused"
    const val STATE_FAILED = "failed"

    /** Stopped by a transient network error and waiting for WorkManager's backoff to re-run it. See [waitForNetwork]. */
    const val STATE_WAITING_NETWORK = "waiting_network"

    /**
     * How many times a download may run before a recoverable error is treated as terminal.
     *
     * WorkManager itself never gives up, so this is the only thing standing between a permanently unreachable host and a row that says "Waiting for
     * network…" forever. With the 30s exponential backoff, five attempts spans roughly eight minutes: long enough to ride out a tunnel or a Wi-Fi
     * handover, short enough that a genuine dead link still surfaces as a failure the user can read.
     */
    const val MAX_NETWORK_ATTEMPTS = 5

    // Exact host or a real subdomain — so the HF token isn't sent to look-alikes like "evil-huggingface.co".
    private const val HF_HOST = "huggingface.co"

    fun isHuggingFaceHost(host: String?): Boolean = host == HF_HOST || (host != null && host.endsWith(".$HF_HOST"))

    private const val USER_AGENT = "pipette-android"
    private const val TIMEOUT_MS = 30_000
    private const val BUFFER_SIZE = 64 * 1024
    private const val PROGRESS_INTERVAL_MS = 500L
    private const val MILLIS_PER_SECOND = 1000L
    private const val UNKNOWN_TOTAL = -1L
    private const val HTTP_RANGE_NOT_SATISFIABLE = 416
    private const val HTTP_TEMP_REDIRECT = 307
    private const val HTTP_PERM_REDIRECT = 308
    private const val MAX_REDIRECTS = 5
    private val REDIRECT_CODES =
      setOf(
        HttpURLConnection.HTTP_MOVED_PERM,
        HttpURLConnection.HTTP_MOVED_TEMP,
        HttpURLConnection.HTTP_SEE_OTHER,
        HTTP_TEMP_REDIRECT,
        HTTP_PERM_REDIRECT,
      )
  }
}
