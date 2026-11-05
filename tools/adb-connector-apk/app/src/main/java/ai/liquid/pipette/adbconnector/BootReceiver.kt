package ai.liquid.pipette.adbconnector

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Re-enables wireless debugging after a reboot.
 *
 * `AdbDebuggingManager` restores `adb_wifi_enabled` during boot, finds wifi not yet associated, writes the setting back to 0, and nothing retries,
 * leaving the phone pingable with no adb listener. This receiver supplies that retry.
 */
class BootReceiver : BroadcastReceiver() {
  override fun onReceive(context: Context, intent: Intent) {
    // The receiver is exported (it has to be, to hear the system), so any local app can send it
    // an arbitrary intent.
    if (intent.action !in HANDLED_ACTIONS) {
      Log.w(AdbWifi.TAG, "ignoring unexpected action ${intent.action}")
      return
    }

    // Gated on uptime, not on the action: QUICKBOOT_POWERON is unprotected, so any app can send
    // it. Off the boot window the destructive flip degrades to the safe enable, so an arbitrary
    // caller cannot kill a live benchmark transport.
    val sinceBoot = SystemClock.elapsedRealtime()
    val nearBoot = sinceBoot < BOOT_WINDOW_MS
    Log.i(AdbWifi.TAG, "${intent.action} received ${sinceBoot}ms after boot (nearBoot=$nearBoot)")

    // forceRearm waits between two writes, which onReceive's thread must not do. goAsync keeps
    // the process alive until finish().
    val pending = goAsync()
    CoroutineScope(SupervisorJob() + Dispatchers.Default).launch {
      try {
        // No withTimeout: forceRearm sits at 0 for AdbWifi.HOLDER_GRACE_MS or longer by design,
        // and a timeout there would trade a hang that cannot happen for a failure that can.
        if (nearBoot) {
          AdbWifi.forceRearm(context)
        } else {
          AdbWifi.enable(context)
        }
        AdbWatchdogJob.schedule(context)
        // Re-asserted every boot: the enabled-services list is shared with every other
        // accessibility service, so anything that rewrites it can drop this entry.
        DialogAccepter.enable(context)
      } catch (e: Exception) {
        // Swallowed: a crash during boot leaves the phone as unreachable as doing nothing.
        // PermissionDenied (no `pm grant` yet) arrives here too.
        Log.e(AdbWifi.TAG, "boot rearm failed", e)
      } finally {
        pending.finish()
      }
    }
  }

  private companion object {
    /** How long after boot the destructive flip stays available. Generous next to the ~21s BOOT_COMPLETED takes on the fleet. */
    const val BOOT_WINDOW_MS = 5 * 60 * 1000L

    val HANDLED_ACTIONS = setOf(Intent.ACTION_BOOT_COMPLETED, "android.intent.action.QUICKBOOT_POWERON")
  }
}
