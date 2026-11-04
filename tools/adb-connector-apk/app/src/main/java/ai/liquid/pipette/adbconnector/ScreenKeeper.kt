package ai.liquid.pipette.adbconnector

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Turns the screen back on after it goes off, while the device is plugged in and the global setting [KEEP_SCREEN_ON] reads 1.
 *
 * `stay_on_while_plugged_in` only stops the idle timeout. A side-key press while someone handles the rack still sleeps the handset, and a sleeping
 * handset suspends between samples and benchmarks slow. Hosted by [WifiDebuggingDialogService] because SCREEN_OFF only reaches a runtime receiver,
 * and that service is the app's one long-lived process.
 */
internal class ScreenKeeper(private val context: Context, private val scope: CoroutineScope) {

  private var registered = false

  /** The pending re-wake, so a burst of SCREEN_OFF broadcasts wakes once. Touched on the main thread only. */
  private var pending: Job? = null

  private val receiver =
    object : BroadcastReceiver() {
      override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_SCREEN_OFF) return
        // Caught: a throwable here would take the dialog accepter's process down too.
        try {
          onScreenOff()
        } catch (e: Exception) {
          Log.e(AdbWifi.TAG, "screen keeper failed on screen off", e)
        }
      }
    }

  fun start() {
    val filter = IntentFilter(Intent.ACTION_SCREEN_OFF)
    runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
          context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
          context.registerReceiver(receiver, filter)
        }
      }
      .onSuccess {
        registered = true
        Log.i(AdbWifi.TAG, "screen keeper registered (enabled=${isEnabled(context)})")
      }
      .onFailure { Log.e(AdbWifi.TAG, "could not register the screen keeper", it) }
  }

  fun stop() {
    pending?.cancel()
    pending = null
    if (!registered) return
    registered = false
    runCatching { context.unregisterReceiver(receiver) }
      .onSuccess { Log.i(AdbWifi.TAG, "screen keeper unregistered") }
      .onFailure { Log.w(AdbWifi.TAG, "could not unregister the screen keeper", it) }
  }

  private fun onScreenOff() {
    if (!isEnabled(context)) return
    if (!pluggedIn()) {
      Log.i(AdbWifi.TAG, "screen off on battery; leaving it off")
      return
    }
    val power = context.getSystemService(PowerManager::class.java) ?: return
    // Keeps a screen-off device from suspending through the delay. One per attempt, so cancelling
    // a superseded attempt releases only its own.
    val cpu = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG).apply { setReferenceCounted(false) }
    cpu.acquire(WAKE_LOCK_TIMEOUT_MS)
    pending?.cancel()
    pending =
      scope.launch {
        try {
          delay(WAKE_DELAY_MS)
          // Re-checked: the switch or the cable can change during the delay.
          if (power.isInteractive || !isEnabled(context) || !pluggedIn()) return@launch
          wake()
          delay(VERIFY_DELAY_MS)
          if (power.isInteractive) {
            Log.i(AdbWifi.TAG, "screen is back on")
          } else {
            Log.w(AdbWifi.TAG, "screen still off after the wake activity")
          }
        } catch (e: CancellationException) {
          throw e
        } catch (e: Exception) {
          Log.e(AdbWifi.TAG, "screen keeper re-wake failed", e)
        } finally {
          if (cpu.isHeld) cpu.release()
        }
      }
  }

  private fun wake() {
    val intent = Intent(context, WakeActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_NO_ANIMATION)
    runCatching { context.startActivity(intent) }
      .onSuccess { Log.i(AdbWifi.TAG, "screen went off on power; started the wake activity") }
      .onFailure { Log.e(AdbWifi.TAG, "could not start the wake activity", it) }
  }

  // EXTRA_PLUGGED, not charging status: a handset held at its battery-protection limit reads
  // "not charging" while on power.
  private fun pluggedIn(): Boolean {
    val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return false
    return battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
  }

  companion object {
    /** Global setting the farm writes; 1 turns the keeper on. */
    private const val KEEP_SCREEN_ON = "pipette_keep_screen_on"

    private const val WAKE_LOCK_TAG = "pipette:screen-keeper"

    /** Lets a burst of screen transitions settle before acting. */
    private const val WAKE_DELAY_MS = 2_000L

    /** Time for the wake activity to resume before reading back whether the screen came on. */
    private const val VERIFY_DELAY_MS = 1_000L

    /** Backstop release, in case the attempt never reaches its finally. */
    private const val WAKE_LOCK_TIMEOUT_MS = 10_000L

    fun isEnabled(context: Context): Boolean =
      runCatching { Settings.Global.getInt(context.contentResolver, KEEP_SCREEN_ON, 0) == 1 }.getOrDefault(false)
  }
}
