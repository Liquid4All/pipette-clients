package ai.liquid.pipette.adbconnector

import android.accessibilityservice.AccessibilityService
import android.os.SystemClock
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Answers the "Allow wireless debugging on this network?" dialog, so a device that roams to an unfamiliar access point re-arms itself.
 *
 * ## Why writing the setting is not enough
 *
 * `AdbDebuggingManager` trusts wireless debugging per BSSID, not per SSID, and one benchmark SSID spans several BSSIDs. When the setting goes to 1 on
 * an untrusted BSSID, the manager raises a confirmation dialog and writes `adb_wifi_enabled` back to 0 within about 25ms (measured on the fleet). No
 * write schedule wins that race: the unanswered dialog means "no". Ticking "always allow" adds the BSSID to the trusted set for good.
 *
 * ## Why an AccessibilityService
 *
 * `IAdbManager.allowWirelessDebugging`, the call the dialog makes, needs signature-level MANAGE_DEBUGGING, which `pm grant` cannot reach, so tapping
 * the dialog is the only route left. `packageNames` in the service config restricts it to SystemUI.
 *
 * ## How the dialog is recognised
 *
 * By its contents: its window-state event reports `className` as `android.app.Dialog` on the fleet's firmware, which also covers the shade, the
 * volume panel and the power menu. [isWifiDebuggingDialog] requires the always-allow checkbox, the positive button, and a message naming an SSID. The
 * USB debugging dialog carries the first two, and accepting it would hand adb to whatever host is plugged in.
 *
 * This service is also the app's only long-lived process, so it hosts [WifiRejoinMonitor] and [ScreenKeeper].
 */
class WifiDebuggingDialogService : AccessibilityService() {

  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

  /**
   * Elapsed-realtime stamp of the last accept attempt, or 0 before the first.
   *
   * The dialog emits a burst of window events, and a tick plus a click emits more. Without a floor the service re-enters on its own output and clicks
   * the ticked checkbox back off.
   */
  @Volatile private var lastAttemptAt = 0L

  private val rejoinMonitor = WifiRejoinMonitor(this, scope)
  private val screenKeeper = ScreenKeeper(this, scope)

  override fun onServiceConnected() {
    // The one log line that tells "listed in settings" from "actually bound".
    Log.i(AdbWifi.TAG, "dialog accepter connected; watching SystemUI for the wireless-debugging dialog")
    DialogAccepter.running = true

    // Here rather than onCreate: only the binding keeps this process alive.
    rejoinMonitor.start()
    screenKeeper.start()
  }

  override fun onAccessibilityEvent(event: AccessibilityEvent) {
    if (event.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) return

    // rootInActiveWindow, not event.source: the checkbox and button are siblings of the emitting
    // node. Null while the window is composed; the rest of the event burst retries.
    val root = rootInActiveWindow ?: return
    if (!isWifiDebuggingDialog(root)) return

    // Stamped only once identified, so an unrelated window (the shade) cannot swallow a real dialog.
    val now = SystemClock.elapsedRealtime()
    if (now - lastAttemptAt < REATTEMPT_FLOOR_MS) return
    lastAttemptAt = now

    accept(root)
  }

  /**
   * Ticks "always allow" and presses the positive button.
   *
   * A failure is logged, not retried: the next [AdbWatchdogJob] write puts the dialog back up, while a tight retry would spin against a layout this
   * does not understand.
   */
  private fun accept(root: AccessibilityNodeInfo) {
    val alwaysAllowed = tickAlwaysAllow(root)

    val allow = findAllowButton(root)
    if (allow == null) {
      Log.e(AdbWifi.TAG, "dialog matched but its positive button vanished; left on screen")
      return
    }
    if (!allow.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
      Log.e(AdbWifi.TAG, "positive button refused ACTION_CLICK; dialog left on screen")
      return
    }

    // alwaysAllowed decides whether this access point is fixed for good or only until the next roam.
    Log.i(AdbWifi.TAG, "accepted wireless debugging for this network (alwaysAllowed=$alwaysAllowed)")

    confirmEnabled()
  }

  /**
   * Ticks the "always allow" checkbox, returning whether it ends up ticked.
   *
   * False is not a failure to accept: the positive button still restores adb for this association, but this access point will ask again.
   */
  private fun tickAlwaysAllow(root: AccessibilityNodeInfo): Boolean {
    val checkbox = findAlwaysAllowCheckbox(root)
    if (checkbox == null) {
      Log.w(AdbWifi.TAG, "no always-allow checkbox in the dialog; accepting this association only")
      return false
    }
    if (checkbox.isTicked()) {
      // Already ticked, so clicking would clear it. Reached when a burst outruns the floor.
      Log.i(AdbWifi.TAG, "always-allow already ticked")
      return true
    }
    if (!checkbox.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
      Log.w(AdbWifi.TAG, "always-allow refused ACTION_CLICK; accepting this association only")
      return false
    }
    return true
  }

  /**
   * Reads the setting back after the platform has had time to act on the accept, and re-enables it if it is still off.
   *
   * Normally the accept already armed adbd and this does nothing. It covers an accept recorded without the enable following, which would otherwise
   * wait out a watchdog period.
   */
  private fun confirmEnabled() {
    scope.launch {
      delay(ACCEPT_SETTLE_MS)
      val enabled =
        try {
          AdbWifi.enable(this@WifiDebuggingDialogService)
        } catch (e: RuntimeException) {
          // Includes PermissionDenied before `pm grant`. Never rethrown: it would crash the
          // accepter's process, and the accept itself has already landed.
          Log.e(AdbWifi.TAG, "post-accept check failed", e)
          return@launch
        }
      Log.i(AdbWifi.TAG, "wireless debugging after accept: enabled=$enabled")
    }
  }

  // The service gives no feedback, so there is nothing to interrupt.
  override fun onInterrupt() = Unit

  override fun onUnbind(intent: android.content.Intent?): Boolean {
    // Before the cancel, so no callback launches into a cancelled scope.
    rejoinMonitor.stop()
    screenKeeper.stop()
    scope.cancel()
    DialogAccepter.running = false
    return super.onUnbind(intent)
  }

  private companion object {
    const val MESSAGE_VIEW_ID = "android:id/message"

    /** Separates this dialog from the USB debugging consent. See [isWifiDebuggingDialog]. */
    const val SSID_MARKER = "SSID"

    /** Shortest gap between two accept attempts: covers one dialog's event burst without dropping a genuine second dialog from another roam. */
    const val REATTEMPT_FLOOR_MS = 1_500L

    const val ACCEPT_SETTLE_MS = 2_000L

    /**
     * A fast path only: `android:id/alwaysUse` is the AOSP id, but an OEM's SystemUI may re-declare it, so the class-name search in
     * [findAlwaysAllowCheckbox] is what has to hold.
     */
    val CHECKBOX_VIEW_IDS = listOf("android:id/alwaysUse", "com.android.systemui:id/alwaysallow", "com.android.systemui:id/alwaysUse")

    /**
     * The framework alert layout's positive button, so one id rather than a list. No label-based fallback: telling Allow from Deny would need
     * translated text, and clicking the wrong one is worse than leaving the dialog up.
     */
    const val ALLOW_BUTTON_VIEW_ID = "android:id/button1"

    const val CHECKBOX_CLASS = "android.widget.CheckBox"
  }

  /**
   * Whether the window on screen is the wireless-debugging dialog.
   *
   * The SSID in the message is what separates it from the USB debugging consent, which shares the rest of its layout. The check reads the acronym,
   * which the fleet's locales keep untranslated, rather than prose tied to one language. A locale that translates it leaves the dialog on screen.
   */
  private fun isWifiDebuggingDialog(root: AccessibilityNodeInfo): Boolean =
    findAlwaysAllowCheckbox(root) != null && findAllowButton(root) != null && messageNamesAnSsid(root)

  private fun messageNamesAnSsid(root: AccessibilityNodeInfo): Boolean =
    root.findAccessibilityNodeInfosByViewId(MESSAGE_VIEW_ID).any { it.text?.contains(SSID_MARKER) == true }

  private fun findAlwaysAllowCheckbox(root: AccessibilityNodeInfo): AccessibilityNodeInfo? =
    CHECKBOX_VIEW_IDS.firstNotNullOfOrNull { id -> root.findAccessibilityNodeInfosByViewId(id).firstOrNull() }
      ?: firstNode(root) { it.className == CHECKBOX_CLASS }

  private fun findAllowButton(root: AccessibilityNodeInfo): AccessibilityNodeInfo? =
    root.findAccessibilityNodeInfosByViewId(ALLOW_BUTTON_VIEW_ID).firstOrNull { it.isEnabled && it.isClickable }

  /**
   * `isChecked` is deprecated for `getChecked` (API 36; minSdk is 30). Not version-gated because the two agree on a two-state checkbox: `getChecked`
   * only adds the partial state.
   */
  @Suppress("DEPRECATION") private fun AccessibilityNodeInfo.isTicked(): Boolean = isChecked

  /** Breadth-first, so a shallow match wins over a deep one. */
  private fun firstNode(root: AccessibilityNodeInfo, predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
    val queue = ArrayDeque(listOf(root))
    while (queue.isNotEmpty()) {
      val node = queue.removeFirst()
      if (predicate(node)) return node
      for (i in 0 until node.childCount) {
        // Null children are normal while a window is being composed.
        node.getChild(i)?.let(queue::addLast)
      }
    }
    return null
  }
}
