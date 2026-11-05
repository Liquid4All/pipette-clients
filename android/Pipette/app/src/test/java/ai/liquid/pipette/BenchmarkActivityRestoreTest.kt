package ai.liquid.pipette

import android.os.Bundle
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The pocket screen must not outlive the run it exists to show.
 *
 * [BenchmarkActivity] leaves pocket mode only from `render`, on a `running=false` snapshot pushed from the main process. That is edge-triggered, so a
 * screen recreated in a process with nobody left to push to it never sees the edge. The platform recreates exactly that way after a job ends, because
 * ending a job tears `:benchmark` down while this activity is still the top of its task; unlocking the device later restores the task into a fresh
 * process. Before this was handled the screen sat on "Benchmarking in progress" indefinitely, holding `:benchmark` at `top-app` with nothing running.
 *
 * Both directions are pinned here, because the failure mode of an over-eager fix is worse than the bug: finishing a live pocket screen hands the
 * process straight back to a demoted cpuset mid-measurement, which is the exact corruption the cpuset guard exists to catch.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class BenchmarkActivityRestoreTest {
  @Test
  fun aRestoredPocketScreenFinishesInsteadOfClaimingARunIsInFlight() {
    val activity = Robolectric.buildActivity(BenchmarkActivity::class.java).create(Bundle()).get()

    assertTrue("a pocket screen restored with no run behind it must finish itself", activity.isFinishing)
  }

  @Test
  fun aFreshlyLaunchedPocketScreenStaysUp() {
    // No saved state: this is JobRunner launching the pocket for a run that is starting.
    val activity = Robolectric.buildActivity(BenchmarkActivity::class.java).create().get()

    assertFalse("a pocket screen launched for a live run must not finish itself", activity.isFinishing)
  }

  /** Resuming a restored instance must not fault against the views it never built, nor claim a top-app promotion on its way out. */
  @Test
  fun aRestoredPocketScreenSurvivesTheRestOfItsLifecycle() {
    val controller = Robolectric.buildActivity(BenchmarkActivity::class.java).create(Bundle())

    controller.start().resume().pause().stop().destroy()

    assertTrue(controller.get().isFinishing)
  }
}
