package ai.liquid.pipette.adbconnector

import android.content.Context
import android.provider.Settings
import android.util.Log
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * Turns wireless debugging back on by writing the global setting the Developer-options toggle writes. `AdbService`'s settings observer in
 * system_server reacts to the change and starts adbd's TLS listener on a fresh random port, which the host still has to discover (`host-connect.sh`).
 *
 * ## Concurrency
 *
 * The UI buttons, the boot receiver, the watchdog job, [WifiRejoinMonitor] and the dialog service's post-accept check can overlap, most of all when
 * wifi associates around BOOT_COMPLETED. Without [rearmInProgress] the watchdog could write 1 under a re-arm's 0 and cut its grace short, and so
 * could a second [forceRearm]. A flag rather than a lock: a re-arm suspends for [HOLDER_GRACE_MS] or longer, and a lock held that long would block
 * the UI thread's next [enable] into an ANR. Concurrent [enable] calls need no exclusion since both only write 1.
 */
internal object AdbWifi {
  const val TAG = "pipette-adb-connector"

  /** `Settings.Global.ADB_WIFI_ENABLED` is @hide, so the key is spelled out. */
  private const val ADB_WIFI_ENABLED = "adb_wifi_enabled"

  /**
   * How long [forceRearm] holds the setting at 0 before it writes 1.
   *
   * Long enough for the previous session's processes to exit. adbd's mDNS socket has no CLOEXEC, so every process adbd forks after registering holds
   * a copy, and mdnsd withdraws the record only once every copy has closed. A re-arm while one lives announces `<name> (2)` and leaves the old name
   * on a dead port. The pipette worker exits within ~10s of its session closing.
   */
  const val HOLDER_GRACE_MS = 15_000L

  @Volatile private var rearmInProgress = false

  /** Whether wireless debugging is currently on, or null if the value cannot be read. */
  fun isEnabled(context: Context): Boolean? =
    try {
      Settings.Global.getInt(context.contentResolver, ADB_WIFI_ENABLED) == 1
    } catch (e: Settings.SettingNotFoundException) {
      // Unset until wireless debugging is first used after a factory reset. Unknown, not false.
      Log.w(TAG, "$ADB_WIFI_ENABLED is not set", e)
      null
    }

  /**
   * Enables wireless debugging, returning whether it reads as on afterwards.
   *
   * Throws [PermissionDenied] when the WRITE_SECURE_SETTINGS grant is missing, a normal state for this app rather than a bug. The UI shows the `pm
   * grant` line; background callers log and carry on, because a lab tool that dies on boot is harder to diagnose than a log line.
   */
  fun enable(context: Context): Boolean {
    if (rearmInProgress) {
      // Writing 1 now would announce the port before the old record's holders exit.
      Log.i(TAG, "rearm in progress; leaving it to finish")
      return false
    }
    if (isEnabled(context) == true) {
      // No flip here: these callers may have a live host connection, and a stop/start would drop
      // it and move the port. [forceRearm] is for callers that can afford the flip.
      Log.i(TAG, "wireless debugging already enabled; nothing to do")
      return true
    }
    write(context, 1)
    return isEnabled(context) == true
  }

  /**
   * Enables wireless debugging, forcing a 0->1 transition when the setting already reads 1.
   *
   * The observer fires only on a change, so writing 1 over a stale 1 with no listener bound arms nothing and strands the device. The flip also makes
   * adbd register its `_adb-tls-connect` record again. Either path writes 1 only after [HOLDER_GRACE_MS] at 0, so the old record is gone first.
   *
   * ## Precondition, which the caller owns
   *
   * No host transport may be live: the flip moves adbd's port and kills any transport on the old one. [BootReceiver] gates on uptime,
   * [WifiRejoinMonitor] on the rejoin edge, and [MainActivity]'s force button is explicit intent from a person at the device. Nothing here re-checks
   * it.
   */
  suspend fun forceRearm(context: Context): Boolean =
    // NonCancellable on purpose: a cancellation after the 0 would leave the phone reachable only
    // over USB. The window can run past HOLDER_GRACE_MS: writing 0 drops the app's own adb
    // connection and the OEM freezes the process mid-delay (~11s measured).
    withContext(NonCancellable) {
      if (rearmInProgress) {
        Log.i(TAG, "re-arm already in progress; not starting a second one")
        return@withContext false
      }
      rearmInProgress = true
      try {
        if (isEnabled(context) == true) {
          Log.w(TAG, "setting already reads 1; forcing a 0->1 flip to arm the listener")
          write(context, 0)
        }
        // Also keeps a flip's two edges from collapsing into one no-op.
        delay(HOLDER_GRACE_MS)
        write(context, 1)
      } catch (e: Throwable) {
        // Even an Error: the device sits at 0 with no listener, so a best-effort re-write is the
        // difference between a working phone and a USB trip.
        Log.e(TAG, "re-arm failed with the setting at 0; restoring $ADB_WIFI_ENABLED=1", e)
        runCatching { write(context, 1) }
        throw e
      } finally {
        rearmInProgress = false
      }
      isEnabled(context) == true
    }

  /** Writes the setting, returning whether the provider call succeeded. Not evidence the value moved: callers read it back with [isEnabled]. */
  private fun write(context: Context, value: Int): Boolean =
    try {
      val accepted = Settings.Global.putInt(context.contentResolver, ADB_WIFI_ENABLED, value)
      Log.i(TAG, "wrote $ADB_WIFI_ENABLED=$value: provider accepted=$accepted")
      accepted
    } catch (e: SecurityException) {
      throw PermissionDenied(
        "cannot write $ADB_WIFI_ENABLED. Grant it with: adb shell pm grant " + "${context.packageName} android.permission.WRITE_SECURE_SETTINGS",
        e,
      )
    }

  /** The WRITE_SECURE_SETTINGS grant is missing. Its own type so the UI can tell someone at the device to run `pm grant`. */
  class PermissionDenied(message: String, cause: SecurityException) : RuntimeException(message, cause)
}
