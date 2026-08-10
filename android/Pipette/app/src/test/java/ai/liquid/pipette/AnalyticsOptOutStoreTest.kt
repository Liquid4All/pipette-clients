package ai.liquid.pipette

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Drives [AnalyticsOptOutStore] against Robolectric's SharedPreferences.
 *
 * What makes this worth a test rather than a glance: the store exists **because** the PostHog SDK's own persisted opt-out does not survive a launch
 * (see the store's KDoc for the measurement), and the whole fix rests on this value being readable *synchronously* by the time `PipetteApp.onCreate`
 * seeds `PostHogAndroidConfig.optOut`. A regression that made the read async, or that lost the default, would put analytics back on for a user who
 * turned them off, silently and only on a real cold start, which is exactly the failure the device test caught the first time.
 */
// A stock Application, not the manifest PipetteApp, whose onCreate wires up WorkManager/Clerk/etc. that have no place in this unit test.
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class AnalyticsOptOutStoreTest {
  private val context: Context = ApplicationProvider.getApplicationContext()

  /** A fresh install collects: the absent key must read as opted **in**, matching the SDK's own default. */
  @Test
  fun defaultsToCollectingWhenNothingHasBeenWritten() {
    assertFalse(AnalyticsOptOutStore.read(context))
  }

  /** The round trip the seeding depends on. Reads go through a plain synchronous `getBoolean`, so a written value is visible immediately. */
  @Test
  fun readsBackWhatWasWritten() {
    AnalyticsOptOutStore.write(context, true)
    assertTrue(AnalyticsOptOutStore.read(context))

    AnalyticsOptOutStore.write(context, false)
    assertFalse(AnalyticsOptOutStore.read(context))
  }
}
