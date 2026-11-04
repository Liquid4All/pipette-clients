package ai.liquid.pipette.adbconnector

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * One screen showing whether wireless debugging is on, with buttons to turn it on.
 *
 * Its load-bearing job is to be launchable: a package never started is excluded from broadcasts, so BootReceiver fires only after one launch. It is
 * also the only surface when the host cannot connect, so it reports why a write failed.
 */
class MainActivity : Activity() {
  // No lifecycleScope without AndroidX. Settings and JobScheduler calls are IPC, so they run on
  // Dispatchers.Default and only the resulting text touches the main thread.
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

  private lateinit var status: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // The launch that clears FLAG_STOPPED also registers the watchdog, so one launch is the setup.
    AdbWatchdogJob.schedule(this)

    status = TextView(this)

    val enableButton =
      Button(this).apply {
        text = getString(R.string.enable_button)
        setOnClickListener { performWrite { AdbWifi.enable(this@MainActivity) } }
      }

    // Destructive (drops adbd's listener and moves the port), so separate from Enable, which
    // no-ops on a stale 1 whose listener died. A tap is explicit intent from someone at the device.
    val forceButton =
      Button(this).apply {
        text = getString(R.string.force_button)
        setOnClickListener { performWrite { AdbWifi.forceRearm(this@MainActivity) } }
      }

    val padding = (24 * resources.displayMetrics.density).toInt()
    val root =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER
        setPadding(padding, padding, padding, padding)
        addView(status)
        addView(enableButton)
        addView(forceButton)
        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
      }
    setContentView(root)

    // After setContentView because it refreshes the status view. A failure is only logged: a
    // device with no accepter still works on every access point it has already approved.
    scope.launch {
      try {
        withContext(Dispatchers.Default) { DialogAccepter.enable(this@MainActivity) }
      } catch (e: RuntimeException) {
        Log.e(AdbWifi.TAG, "could not register the dialog accepter", e)
      }
      refresh()
    }
  }

  // Picks up a toggle flipped in Developer options or over adb while in the background.
  override fun onResume() {
    super.onResume()
    refresh()
  }

  override fun onDestroy() {
    scope.cancel()
    super.onDestroy()
  }

  /**
   * Applies a write off the main thread, then reports the outcome on screen.
   *
   * A missing WRITE_SECURE_SETTINGS grant (a clean reinstall drops it) leaves the status unchanged, and the only other diagnostic is logcat, which
   * needs the adb connection that is already broken.
   */
  private fun performWrite(write: suspend () -> Unit) {
    scope.launch {
      val error =
        try {
          withContext(Dispatchers.Default) { write() }
          null
        } catch (e: AdbWifi.PermissionDenied) {
          getString(R.string.error_no_permission)
        } catch (e: Exception) {
          getString(R.string.error_generic, e.javaClass.simpleName)
        }
      refresh(error)
    }
  }

  private fun refresh(error: String? = null) {
    scope.launch { status.text = withContext(Dispatchers.Default) { statusText(error) } }
  }

  private fun statusText(error: String?): String {
    val state =
      when (AdbWifi.isEnabled(this)) {
        true -> getString(R.string.status_enabled)
        false -> getString(R.string.status_disabled)
        null -> getString(R.string.status_unknown)
      }
    val watchdog = getString(if (AdbWatchdogJob.isScheduled(this)) R.string.watchdog_on else R.string.watchdog_off)
    val running = DialogAccepter.running
    val accepter =
      getString(
        when {
          running -> R.string.accepter_on
          DialogAccepter.isEnabled(this) -> R.string.accepter_not_running
          else -> R.string.accepter_off
        }
      )
    // The keeper runs inside the accessibility service, so the switch alone does not make it live.
    val keeper = getString(if (running && ScreenKeeper.isEnabled(this)) R.string.keeper_on else R.string.keeper_off)
    return getString(R.string.status_format, state, "$watchdog\n\n$accepter\n\n$keeper", error.orEmpty())
  }
}
