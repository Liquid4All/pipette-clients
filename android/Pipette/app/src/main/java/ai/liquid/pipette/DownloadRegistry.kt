package ai.liquid.pipette

/**
 * In-process bus shared between [DownloadWorker] (which runs the actual transfer) and [DownloadCoordinator] (which the UI reads). The worker pushes
 * live [ActiveDownload] snapshots here and invokes the per-download callbacks; the coordinator reads [all] for the UI and registers callbacks on
 * enqueue. Both live in the main app process, so a plain synchronized singleton is enough — no IPC.
 *
 * Callbacks are session-scoped: they're registered when the user starts a download and are lost if the process is recreated. A worker resumed by
 * WorkManager after process death still updates [live] and registers the finished model; it just can't call a callback that no longer exists.
 */
object DownloadRegistry {
  data class Callbacks(val onProgress: (DownloadProgress) -> Unit, val onComplete: (ModelFile) -> Unit, val onFailure: (Throwable) -> Unit)

  private val lock = Any()
  private val live = linkedMapOf<String, ActiveDownload>()
  private val callbacks = mutableMapOf<String, Callbacks>()

  /**
   * Fired (off the main thread) whenever the live set changes. The UI sets this to bump its render tick, so a download resumed by WorkManager after
   * process death — which repopulates [live] but can't recreate the session-scoped [callbacks] — still drives a re-render instead of showing frozen
   * progress until some other event triggers one.
   */
  @Volatile var onChanged: (() -> Unit)? = null

  fun put(download: ActiveDownload) {
    synchronized(lock) { live[download.key] = download }
    onChanged?.invoke()
  }

  /**
   * Publish a live progress snapshot only if the coordinator hasn't taken the download over in the meantime — returns false (and writes nothing) when
   * the entry was removed (cancel) or moved to paused/failed. The check-and-set is atomic under [lock], so a worker's in-flight `emit()` can't
   * resurrect a download the user just paused or cancelled (the worker stops on its next `isStopped` check).
   */
  fun putIfActive(download: ActiveDownload): Boolean {
    val written =
      synchronized(lock) {
        val current = live[download.key]
        if (current == null || current.state == DownloadWorker.STATE_PAUSED || current.state == DownloadWorker.STATE_FAILED) {
          false
        } else {
          live[download.key] = download
          true
        }
      }
    if (written) onChanged?.invoke()
    return written
  }

  fun get(key: String): ActiveDownload? = synchronized(lock) { live[key] }

  fun all(): List<ActiveDownload> = synchronized(lock) { live.values.toList() }

  fun remove(key: String) {
    synchronized(lock) {
      live.remove(key)
      callbacks.remove(key)
    }
    onChanged?.invoke()
  }

  fun setCallbacks(key: String, value: Callbacks) = synchronized(lock) { callbacks[key] = value }

  fun callbacks(key: String): Callbacks? = synchronized(lock) { callbacks[key] }
}
