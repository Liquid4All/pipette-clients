package ai.liquid.pipette.adbconnector

import android.app.Activity

/**
 * Turns the screen on and finishes. Translucent; the turn-screen-on and show-when-locked attributes are in the manifest.
 *
 * An activity rather than a wake lock: ACQUIRE_CAUSES_WAKEUP is deprecated and gated, while an app whose accessibility service is bound may start an
 * activity from the background.
 */
class WakeActivity : Activity() {

  override fun onResume() {
    super.onResume()
    // Not finished in onCreate: the screen turns on only once the activity resumes.
    window.decorView.postDelayed({ finish() }, FINISH_DELAY_MS)
  }

  private companion object {
    /** Long enough for the window to be shown and the screen to come on before it goes. */
    const val FINISH_DELAY_MS = 500L
  }
}
