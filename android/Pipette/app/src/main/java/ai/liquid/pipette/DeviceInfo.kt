package ai.liquid.pipette

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager

object DeviceInfo {
  fun modelName(): String =
    listOf(Build.MANUFACTURER, Build.MODEL).filter { it.isNotBlank() }.joinToString(" ").ifBlank { Build.DEVICE ?: "Android device" }

  fun chipModel(): String =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      Build.SOC_MODEL.ifBlank { Build.HARDWARE ?: Build.BOARD ?: "unknown" }
    } else {
      Build.HARDWARE ?: Build.BOARD ?: "unknown"
    }

  fun formFactor(context: Context): String = if (context.resources.configuration.smallestScreenWidthDp >= 600) "tablet" else "phone"

  fun osVersion(): String = Build.VERSION.RELEASE ?: Build.VERSION.SDK_INT.toString()

  /**
   * Precise OS build id (finer than [osVersion]), from `Build.VERSION.INCREMENTAL` — the internal source-control build number (OEM-specific, e.g.
   * `12621605`). Matches the Rust client's `ro.build.version.incremental`. Null if blank.
   */
  fun osBuild(): String? = Build.VERSION.INCREMENTAL?.ifBlank { null }

  /**
   * OS security-patch level (e.g. `2025-06-01`), or null. `Build.VERSION.SECURITY_PATCH` (added in API 23) is always available at our minSdk 31, but
   * its value can be blank on some emulators/custom ROMs, hence blank -> null.
   */
  fun osSecurityPatch(): String? = Build.VERSION.SECURITY_PATCH?.ifBlank { null }

  fun ramBytes(context: Context): Long {
    val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo()
    manager.getMemoryInfo(info)
    return info.totalMem
  }

  // Run-environment power state, captured with each result so on-battery runs
  // (where the PMIC can cap CPU clocks to avoid voltage sag — distinct from
  // thermal throttling) can be filtered/flagged after the fact. No permission
  // required for any of these.

  /** Battery charge percent (0–100), or null if the platform won't report it. */
  fun batteryLevel(context: Context): Int? {
    val manager = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return null
    // BATTERY_PROPERTY_CAPACITY returns Integer.MIN_VALUE when unsupported.
    return manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY).takeIf { it in 0..100 }
  }

  /**
   * Run-environment power state, or null if the platform won't report it. One of `charging` / `not_charging` / `plugged_in_not_charging` — the
   * `device_power_state` enum the management server expects. Distinguishes "plugged in and topping up" from "plugged in but holding" (full /
   * charge-limited), which a plain charging boolean can't: both remove the battery current-limiting that can throttle the SoC.
   */
  fun powerState(context: Context): String? {
    val manager = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager ?: return null
    return when (manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_STATUS)) {
      BatteryManager.BATTERY_STATUS_CHARGING -> "charging"
      BatteryManager.BATTERY_STATUS_DISCHARGING -> "not_charging"
      // FULL/NOT_CHARGING are reported while on external power but not
      // adding charge (battery full or charge-limited / maintenance).
      BatteryManager.BATTERY_STATUS_FULL,
      BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "plugged_in_not_charging"
      // UNKNOWN or an unsupported gauge — report nothing rather than guess.
      else -> null
    }
  }

  /** Whether OS battery-saver / low-power mode is active (can lower CPU clocks). */
  fun isPowerSaveMode(context: Context): Boolean {
    val manager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return false
    return manager.isPowerSaveMode
  }
}
