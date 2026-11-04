package ai.liquid.pipette.adbconnector

import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.util.Log

/**
 * Periodically re-enables wireless debugging when something has switched it off mid-session.
 *
 * `AdbDebuggingManager` writes `adb_wifi_enabled` back to 0 whenever it finds no wireless network, so a wifi blip can turn wireless debugging off on
 * a device that never rebooted, and nothing in the platform turns it back on. After a roam to an unfamiliar access point the manager zeroes the
 * setting again until its confirmation dialog is answered, so there this job only supplies the retry and [WifiDebuggingDialogService] the answer.
 *
 * Safe on a timer because [AdbWifi.enable] writes only when the setting reads 0, when adbd is not listening and there is no host connection to drop.
 * It never takes the destructive 0 -> 1 flip of [AdbWifi.forceRearm].
 */
class AdbWatchdogJob : JobService() {
  override fun onStartJob(params: JobParameters): Boolean {
    // Inline and finished on return: a settings read plus at most one write is cheap binder IPC,
    // and a background thread would outlive onStopJob. Caught because a crash loop on an
    // unattended device is worse than a missed check.
    try {
      AdbWifi.enable(this)
    } catch (e: RuntimeException) {
      Log.e(AdbWifi.TAG, "watchdog check failed", e)
    }
    return false
  }

  // Unreachable in practice: onStartJob never leaves work in flight.
  override fun onStopJob(params: JobParameters): Boolean = false

  companion object {
    private const val JOB_ID = 1

    /**
     * (Re)registers the periodic check. Idempotent: the same id replaces the existing job.
     *
     * Without a battery exemption Doze stretches the period well past the platform floor on an unplugged device; `dumpsys deviceidle whitelist
     * +ai.liquid.pipette.adbconnector` tightens it.
     */
    fun schedule(context: Context) {
      val scheduler = context.getSystemService(JobScheduler::class.java)
      if (scheduler == null) {
        Log.e(AdbWifi.TAG, "no JobScheduler; watchdog not scheduled")
        return
      }
      val job =
        JobInfo.Builder(JOB_ID, ComponentName(context, AdbWatchdogJob::class.java))
          // Sooner is strictly better on a lab fleet, and anything shorter is clamped anyway.
          .setPeriodic(JobInfo.getMinPeriodMillis())
          // Belt-and-braces: the boot receiver reschedules too.
          .setPersisted(true)
          // Not UNMETERED: lab wifi marked metered (a user toggle, or a detected hotspot) would
          // never satisfy it, leaving the job silently inert while the UI says it is on.
          .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
          .build()
      // Never fatal: callers are MainActivity.onCreate (the launch that clears FLAG_STOPPED) and
      // the boot receiver, neither of which may die over a missing watchdog.
      val result =
        try {
          scheduler.schedule(job)
        } catch (e: RuntimeException) {
          Log.e(AdbWifi.TAG, "watchdog not scheduled", e)
          return
        }
      if (result == JobScheduler.RESULT_SUCCESS) {
        Log.i(AdbWifi.TAG, "watchdog scheduled every ${JobInfo.getMinPeriodMillis()}ms")
      } else {
        Log.e(AdbWifi.TAG, "failed to schedule watchdog (result=$result)")
      }
    }

    fun isScheduled(context: Context): Boolean = context.getSystemService(JobScheduler::class.java)?.getPendingJob(JOB_ID) != null
  }
}
