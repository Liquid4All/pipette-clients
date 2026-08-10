package ai.liquid.pipette

import android.os.Looper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

/**
 * Covers the snapshot-retention contract [BenchmarkProgressBus] relies on to survive a `:benchmark` process that outlives a single job: a terminal
 * snapshot must stay retained for a late observer, but must not leak into the next run.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class BenchmarkProgressBusTest {

  private fun progress(running: Boolean, cellLabel: String = "cell", coolingSinceMillis: Long? = null) =
    BenchmarkProgressBus.Progress(
      title = "job",
      subtitle = "1 model · 1 benchmark",
      cellLabel = cellLabel,
      statusText = "measuring",
      completedCells = 1,
      totalCells = 4,
      overallPermil = 250,
      etaText = "3 min left",
      coolingSinceMillis = coolingSinceMillis,
      running = running,
    )

  /** Drains the main looper so the bus's posted deliveries actually run. */
  private fun drain() = shadowOf(Looper.getMainLooper()).idle()

  private fun observeOnce(): List<BenchmarkProgressBus.Progress> {
    val seen = mutableListOf<BenchmarkProgressBus.Progress>()
    BenchmarkProgressBus.observe { seen += it }
    drain()
    BenchmarkProgressBus.stopObserving()
    return seen
  }

  @Test
  fun observe_redeliversRetainedTerminalSnapshot() {
    BenchmarkProgressBus.publish(progress(running = false))
    drain()

    // An activity attaching after the terminal was delivered still sees it, so it can finish()
    // instead of hanging on "Benchmarking in progress".
    val seen = observeOnce()

    assertEquals(1, seen.size)
    assertFalse(seen.single().running)
  }

  @Test
  fun markStarting_clearsStaleTerminalSnapshot() {
    BenchmarkProgressBus.publish(progress(running = false, cellLabel = "prior-run-cell"))
    drain()

    BenchmarkProgressBus.markStarting()
    val seen = observeOnce()

    // A new run in a reused :benchmark process must not observe the prior run's terminal.
    val snapshot = seen.single()
    assertTrue(snapshot.running)
    assertEquals("", snapshot.cellLabel)
    assertNull(snapshot.coolingSinceMillis)
  }

  @Test
  fun markStarting_keepsLiveSnapshotForMidRunReopen() {
    BenchmarkProgressBus.publish(progress(running = true, cellLabel = "live-cell"))
    drain()

    BenchmarkProgressBus.markStarting()
    val seen = observeOnce()

    // Reopening the pocket mid-run keeps showing real progress rather than resetting to "Starting".
    val snapshot = seen.single()
    assertTrue(snapshot.running)
    assertEquals("live-cell", snapshot.cellLabel)
    assertEquals(250, snapshot.overallPermil)
  }

  @Test
  fun stopObserving_suppressesDeliveryPostedBeforePause() {
    val seen = mutableListOf<BenchmarkProgressBus.Progress>()
    BenchmarkProgressBus.observe { seen += it }
    drain()
    seen.clear()

    // publish() posts to the main thread; a stopObserving() landing before the post runs (onPause)
    // must cancel the delivery rather than push into a backgrounded activity.
    BenchmarkProgressBus.publish(progress(running = true))
    BenchmarkProgressBus.stopObserving()
    drain()

    assertTrue(seen.isEmpty())
  }

  @Test
  fun latestSnapshot_readsRetainedStateWithoutDisturbingObservers() {
    val delivered = mutableListOf<BenchmarkProgressBus.Progress>()
    BenchmarkProgressBus.observe { delivered += it }
    drain()
    delivered.clear()

    BenchmarkProgressBus.publish(progress(running = true, cellLabel = "live-cell"))
    drain()
    delivered.clear()

    // A fold/rotate re-inflate reads the retained snapshot to repopulate its new views; doing so
    // must not re-register or re-deliver to the existing observer.
    assertEquals("live-cell", BenchmarkProgressBus.latestSnapshot().cellLabel)
    drain()
    assertTrue("reading the snapshot should not deliver anything", delivered.isEmpty())

    BenchmarkProgressBus.stopObserving()
  }

  @Test
  fun progressJson_roundTripsAcrossTheAidlHop() {
    val cooling = progress(running = true, coolingSinceMillis = 1_700_000_000_000L)
    assertEquals(cooling, BenchmarkProgressBus.Progress.fromJson(cooling.toJson()))

    // Null cooling must survive as null, not collapse to 0 (which would render a bogus countdown).
    val notCooling = progress(running = true, coolingSinceMillis = null)
    val decoded = BenchmarkProgressBus.Progress.fromJson(notCooling.toJson())
    assertEquals(notCooling, decoded)
    assertNull(decoded?.coolingSinceMillis)
  }

  @Test
  fun progressJson_returnsNullOnMalformedPayload() {
    assertNull(BenchmarkProgressBus.Progress.fromJson("not json"))
  }
}
