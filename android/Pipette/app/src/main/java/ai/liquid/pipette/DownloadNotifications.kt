package ai.liquid.pipette

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Foreground-service notification for model downloads (mirrors the leap-android-sdk downloader): a low-importance, ongoing progress notification with
 * Pause/Resume + Cancel actions wired to [DownloadActionReceiver]. One notification per download, keyed by a stable id derived from the download key.
 */
object DownloadNotifications {
  const val CHANNEL_ID = "ai.liquid.pipette.downloads"
  private const val CHANNEL_NAME = "Model downloads"

  fun ensureChannel(context: Context) {
    val manager = context.getSystemService(NotificationManager::class.java)
    if (manager.getNotificationChannel(CHANNEL_ID) == null) {
      manager.createNotificationChannel(
        NotificationChannel(CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_LOW).apply {
          description = "Progress for Hugging Face model downloads"
          setShowBadge(false)
        }
      )
    }
  }

  // Assign a stable, unique id per download key. key.hashCode() could collide across two concurrent downloads, letting one worker's foreground
  // notification overwrite another's (and one Cancel dismiss both); a monotonic counter avoids that.
  private val ids = ConcurrentHashMap<String, Int>()
  private val nextId = AtomicInteger(NOTIFICATION_ID_BASE)

  fun notificationId(key: String): Int = ids.getOrPut(key) { nextId.getAndIncrement() }

  fun build(context: Context, key: String, title: String, bytesRead: Long, totalBytes: Long, paused: Boolean): Notification {
    ensureChannel(context)
    val percent = if (totalBytes > 0) ((bytesRead * PERCENT_MAX) / totalBytes).toInt() else 0
    val sizeText = if (totalBytes > 0) "${ByteFormat.fileSize(bytesRead)} / ${ByteFormat.fileSize(totalBytes)}" else ByteFormat.fileSize(bytesRead)
    val builder =
      NotificationCompat.Builder(context, CHANNEL_ID)
        .setSmallIcon(android.R.drawable.stat_sys_download)
        .setContentTitle(title)
        .setContentText(if (paused) "Paused · $sizeText" else sizeText)
        .setOngoing(!paused)
        .setOnlyAlertOnce(true)
        .setCategory(NotificationCompat.CATEGORY_PROGRESS)
        .setProgress(PERCENT_MAX, percent, totalBytes <= 0 && !paused)
        .setContentIntent(contentIntent(context))
    if (paused) {
      builder.addAction(0, "Resume", actionIntent(context, DownloadActionReceiver.ACTION_RESUME, key))
    } else {
      builder.addAction(0, "Pause", actionIntent(context, DownloadActionReceiver.ACTION_PAUSE, key))
    }
    builder.addAction(0, "Cancel", actionIntent(context, DownloadActionReceiver.ACTION_CANCEL, key))
    return builder.build()
  }

  /** Update an already-shown download notification (foreground notifications can be refreshed via notify with the same id). */
  fun update(context: Context, key: String, title: String, bytesRead: Long, totalBytes: Long, paused: Boolean) {
    if (NotificationManagerCompat.from(context).areNotificationsEnabled()) {
      NotificationManagerCompat.from(context).notify(notificationId(key), build(context, key, title, bytesRead, totalBytes, paused))
    }
  }

  // Cancel only if this key actually had a notification, and drop its id so the map doesn't grow unbounded (a completed/cancelled download is done
  // with its id). Calling cancel on an unknown key is a no-op rather than allocating a fresh id.
  fun cancel(context: Context, key: String) {
    ids.remove(key)?.let { NotificationManagerCompat.from(context).cancel(it) }
  }

  private fun contentIntent(context: Context): PendingIntent {
    val intent = Intent(context, ComposeMainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
    return PendingIntent.getActivity(context, 0, intent, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
  }

  private fun actionIntent(context: Context, action: String, key: String): PendingIntent {
    val intent = Intent(context, DownloadActionReceiver::class.java).setAction(action).putExtra(DownloadActionReceiver.EXTRA_KEY, key)
    return PendingIntent.getBroadcast(context, (action + key).hashCode(), intent, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
  }

  private const val PERCENT_MAX = 100
  private const val NOTIFICATION_ID_BASE = 4200
}
