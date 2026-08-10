package ai.liquid.pipette

import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.sync.Mutex

/**
 * Per-download-key mutex so at most one [DownloadWorker] writes a given `.part` at a time. WorkManager's `REPLACE` policy starts the new worker
 * before the old (cancelled) one has cooperatively stopped, so a quick pause→resume could otherwise have two workers appending to the same file. A
 * coroutine [Mutex] (not a thread-bound lock) is used because the download holds it across `withContext` hops; the replacing worker waits until the
 * outgoing one releases on its next cancellation check.
 */
object DownloadLocks {
  private val locks = ConcurrentHashMap<String, Mutex>()

  fun forKey(key: String): Mutex = locks.getOrPut(key) { Mutex() }
}
