package ai.liquid.pipette.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Keeps the MAIN process out of the cached bucket for the length of a benchmark run.
 *
 * `JobRunner` orchestrates a job from the main process, but the pocket screen ([ai.liquid.pipette.BenchmarkActivity]) that the run depends on lives
 * in `:benchmark` and is deliberately the top activity: that is how `:benchmark` claims `top-app` and keeps the prime cores. The side effect is that
 * main is left with no foreground component at all, so the framework buckets it as cached (`adj 900`). Meanwhile `JobRunner` keeps mirroring progress
 * to `:benchmark` over oneway Binder, and the platform kills cached processes that do sustained binder traffic:
 * ```
 * ActivityManager: Killing 14504:ai.liquid.pipette.debug/u0a652 (adj 900): excessive binder traffic during cached
 * ```
 *
 * That killed the orchestrator mid-run and left `:benchmark` alive showing a frozen pocket screen, with the job silently dead. Holding a foreground
 * service here for the run's duration keeps main out of `cached`, so the mirror traffic is no longer "during cached".
 *
 * This does NOT compete with `:benchmark` for `top-app`: a foreground *service* does not make its process the top activity, so the pocket screen
 * keeps the cpuset group it was launched to get.
 *
 * Started/stopped by `AppContainer` off `JobRunner`'s run lifecycle. Deliberately not bound and not sticky: the job manifest drives recovery if the
 * process dies anyway.
 */
class JobOrchestratorService : Service() {

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    // Started with startForegroundService(), so startForeground() must follow promptly or the OS
    // kills the app for not honoring the contract. If the promotion is refused (background-start
    // limits), drop the started-service reference rather than sitting here as a plain background
    // service that satisfies nothing: the run continues, just without the cached-bucket protection.
    val promoted =
      runCatching {
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
          } else {
            startForeground(NOTIFICATION_ID, buildNotification())
          }
        }
        .onFailure { Log.w(TAG, "startForeground refused; main process stays cache-eligible for this run", it) }
        .isSuccess
    if (!promoted) stopSelf(startId)
    return START_NOT_STICKY
  }

  private fun buildNotification(): Notification {
    val manager = getSystemService(NotificationManager::class.java)
    if (manager.getNotificationChannel(CHANNEL_ID) == null) {
      manager.createNotificationChannel(
        NotificationChannel(CHANNEL_ID, "Benchmark job", NotificationManager.IMPORTANCE_LOW).apply {
          description = "Keeps a benchmark job's coordinator running"
          setShowBadge(false)
        }
      )
    }
    return Notification.Builder(this, CHANNEL_ID)
      .setContentTitle("Pipette")
      .setContentText("Benchmark job in progress")
      .setSmallIcon(android.R.drawable.stat_sys_download)
      .setOngoing(true)
      .build()
  }

  private companion object {
    const val TAG = "pipetteJobFgs"
    const val CHANNEL_ID = "pipette_job"
    // Distinct from PipetteBenchmarkService's 0x4243 ('BC'): both are posted by the same package, so
    // a shared id would have one service's startForeground replace the other's notification.
    const val NOTIFICATION_ID = 0x4A43 // 'JC'
  }
}
