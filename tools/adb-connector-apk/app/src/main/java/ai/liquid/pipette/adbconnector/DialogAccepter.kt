package ai.liquid.pipette.adbconnector

import android.content.ComponentName
import android.content.Context
import android.provider.Settings
import android.util.Log

/**
 * Registers [WifiDebuggingDialogService] with the accessibility framework, using the WRITE_SECURE_SETTINGS grant the app already holds.
 *
 * Otherwise a human would have to enable it in Settings (on Android 13 and up, after unblocking a sideloaded app's restricted settings) after every
 * reinstall on every device, a manual step that, like `pm grant`, fails silently until the next reboot. Called from [MainActivity] on launch and from
 * [BootReceiver], so install plus one launch stays the whole setup and a lost registration comes back on the next boot.
 */
internal object DialogAccepter {

  /** Both keys are @hide, so they are spelled out. */
  private const val ENABLED_ACCESSIBILITY_SERVICES = "enabled_accessibility_services"

  private const val ACCESSIBILITY_ENABLED = "accessibility_enabled"

  private const val SERVICE_SEPARATOR = ':'

  private fun component(context: Context) = ComponentName(context, WifiDebuggingDialogService::class.java)

  /** Whether the system has bound the service. Differs from [isEnabled]: the system can decline to bind a listed service. */
  @Volatile var running = false

  /** Whether the service is listed as enabled, which is what the app asked for. */
  fun isEnabled(context: Context): Boolean {
    val self = component(context)
    return enabledServices(context).any { it == self }
  }

  /**
   * Adds the service to the enabled list, returning whether it reads as enabled afterwards.
   *
   * Additive: the list is shared with every other accessibility service, so overwriting it would disable a real accessibility tool on a device
   * someone also uses by hand.
   *
   * Throws [AdbWifi.PermissionDenied] when the WRITE_SECURE_SETTINGS grant is missing, matching [AdbWifi.enable] so callers need one catch.
   */
  fun enable(context: Context): Boolean {
    val self = component(context)
    val existing = enabledServices(context)
    if (existing.any { it == self }) {
      Log.i(AdbWifi.TAG, "dialog accepter already enabled; nothing to do")
      // The list can survive while the master switch reads 0, and then no service gets events.
      writeMasterSwitch(context)
      return true
    }

    val updated = (existing + self).joinToString(SERVICE_SEPARATOR.toString()) { it.flattenToString() }
    write(context, ENABLED_ACCESSIBILITY_SERVICES, updated)
    writeMasterSwitch(context)
    val enabled = isEnabled(context)
    Log.i(AdbWifi.TAG, "dialog accepter enabled=$enabled")
    return enabled
  }

  /** The master accessibility switch, which gates delivery to every service regardless of the enabled list. */
  private fun writeMasterSwitch(context: Context) = write(context, ACCESSIBILITY_ENABLED, "1")

  /**
   * Parsed rather than compared as strings: the framework accepts both `pkg/pkg.Cls` and `pkg/.Cls` for one component, so a string comparison would
   * add a duplicate. Unparseable entries are dropped rather than refusing to register.
   */
  private fun enabledServices(context: Context): List<ComponentName> =
    (Settings.Secure.getString(context.contentResolver, ENABLED_ACCESSIBILITY_SERVICES) ?: "")
      .split(SERVICE_SEPARATOR)
      .filter { it.isNotBlank() }
      .mapNotNull(ComponentName::unflattenFromString)

  private fun write(context: Context, key: String, value: String) {
    try {
      val accepted = Settings.Secure.putString(context.contentResolver, key, value)
      Log.i(AdbWifi.TAG, "wrote secure $key: provider accepted=$accepted")
    } catch (e: SecurityException) {
      throw AdbWifi.PermissionDenied(
        "cannot write secure $key. Grant it with: adb shell pm grant ${context.packageName} android.permission.WRITE_SECURE_SETTINGS",
        e,
      )
    }
  }
}
