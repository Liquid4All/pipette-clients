package ai.liquid.pipette

import android.content.res.Resources

/**
 * dp → px using the device density, so programmatic-View layout code can read `8.dp` instead of `dp(8)` or raw pixel math. Density comes from the
 * system resources (it's a device property, independent of any Activity), so the extension needs no Context. detekt's `MagicNumber` rule ignores
 * literals used as extension receivers (`ignoreExtensionFunctions`), which keeps `8.dp` from tripping the gate the way a bare `8` would.
 */
val Int.dp: Int
  get() = (this * Resources.getSystem().displayMetrics.density).toInt()
