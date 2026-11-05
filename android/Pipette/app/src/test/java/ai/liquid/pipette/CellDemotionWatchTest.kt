package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.FOREGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.demoted
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The lifecycle `PipetteBenchmarkService` drives and the AIDL contract depends on: when a verdict is cleared, how long it stays readable, and that
 * the baseline survives from one cell to the next.
 *
 * These are the invariants the service's callers rely on without being able to see them. `lastRunInterruption` is read *after* a run has returned, so
 * a verdict cleared one step too early silently disarms the guard, and one cleared too late files an innocent cell as demoted. `consumeVerdict` is
 * the read the service ships; `lastDemotion` is the non-destructive peek `beginRun` gates on.
 */
class CellDemotionWatchTest {
  @Test
  fun recordsAVerdictAndKeepsItReadableAfterTheWatchStops() {
    val samples = queue(snapshot(TOP_APP), snapshot(FOREGROUND))
    val watch = CellDemotionWatch(samples, pollMillis = POLL)
    val latch = CountDownLatch(1)

    watch.beginCell { latch.countDown() }
    assertTrue(latch.await(5, TimeUnit.SECONDS))
    watch.stop()

    // Deliberately still readable: the caller reads this after the run it describes has returned.
    assertNotNull("stop() must not discard the verdict", watch.lastDemotion)
  }

  @Test
  fun aSecondCellOnTheSameWatchStartsWithNoVerdict() {
    // The sampler recovers to TOP_APP after the first cell has fired, so the second cell has nothing
    // to legitimately re-fire on. Without that the assertion below would be racing the poll interval:
    // a watcher left on a demoted sample re-fires two polls later, and any stall on a loaded machine
    // would make this fail for a reason that is not the one under test.
    // Three entries, not two: beginCell's own cell-start probe takes the first, then the watcher
    // needs two consecutive demoted polls to confirm.
    val samples = ArrayDeque(listOf(snapshot(TOP_APP), snapshot(FOREGROUND), snapshot(FOREGROUND)))
    val watch = CellDemotionWatch({ samples.removeFirstOrNull() ?: snapshot(TOP_APP) }, pollMillis = POLL)
    val latch = CountDownLatch(1)
    watch.beginCell { latch.countDown() }
    assertTrue(latch.await(5, TimeUnit.SECONDS))
    assertNotNull(watch.lastDemotion)

    watch.beginCell {}
    Thread.sleep(POLL * 10)

    assertNull("beginCell must clear the previous cell's verdict", watch.lastDemotion)
    watch.stop()
  }

  /**
   * The reason the baseline is carried between cells rather than re-learned. A process still demoted when the next cell opens must not adopt the
   * demotion as its own baseline; the first sample of the second cell is already demoted here, and is still recognized as such.
   */
  @Test
  fun carriesTheBaselineIntoTheNextCell() {
    val samples = queue(snapshot(TOP_APP), snapshot(FOREGROUND), snapshot(FOREGROUND), snapshot(FOREGROUND))
    val watch = CellDemotionWatch(samples, pollMillis = POLL)
    val first = CountDownLatch(1)
    watch.beginCell { first.countDown() }
    assertTrue(first.await(5, TimeUnit.SECONDS))
    assertEquals(TOP_APP, watch.bestSeen?.cpusetPath)

    val second = AtomicReference<CpusetDemotion?>(null)
    val latch = CountDownLatch(1)
    watch.beginCell {
      second.set(it)
      latch.countDown()
    }

    assertTrue("a persistent demotion must still fire in the next cell", latch.await(5, TimeUnit.SECONDS))
    assertEquals(TOP_APP, second.get()?.baseline?.cpusetPath)
    watch.stop()
  }

  @Test
  fun isWatchingTracksTheLifecycle() {
    val watch = CellDemotionWatch({ snapshot(TOP_APP) }, pollMillis = POLL)
    // `PipetteBenchmarkService.beginRun` gates its belt-and-braces path on this.
    assertFalse(watch.isWatching)
    watch.beginCell {}
    assertTrue(watch.isWatching)
    watch.stop()
    assertFalse(watch.isWatching)
  }

  /** `PipetteBenchmarkService.unloadModel` calls stop() without knowing whether a cell was ever opened. */
  @Test
  fun stopIsIdempotentAndSafeBeforeAnyCell() {
    val watch = CellDemotionWatch({ snapshot(TOP_APP) }, pollMillis = POLL)
    watch.stop()
    watch.stop()
    // The property under test is that neither call throws. Assert something a broken stop() could
    // actually get wrong: the watch must still be usable afterwards.
    watch.beginCell {}
    assertTrue("a watch must still be armable after redundant stops", watch.isWatching)
    watch.stop()
    assertFalse(watch.isWatching)
  }

  /**
   * The epoch's whole reason for existing: a verdict from a cell that has since been replaced must be dropped, not posted into the cell that replaced
   * it. Without it, cell N's demotion fails cell N+1's innocent run and cancels it.
   *
   * The sampler parks mid-poll so the delivery lands after `beginCell` has moved on, which is exactly what a descheduled poll thread does on a real
   * device: the thing being detected is also what delays the detection.
   *
   * Two mechanisms in `CpusetWatcher`/`CellDemotionWatch` can each stop this delivery on their own, so removing either one leaves this test green;
   * removing both turns it red. Do not read a passing suite after deleting one as evidence that it was unnecessary.
   */
  @Test
  fun aVerdictFromAReplacedCellIsDropped() {
    val entered = CountDownLatch(1)
    val release = CountDownLatch(1)
    val delivered = AtomicInteger(0)
    val polls = AtomicInteger(0)
    val watch =
      CellDemotionWatch(
        {
          // 1 is beginCell's own baseline probe, so it must be healthy or nothing below is a
          // demotion at all. 2 and 3 are the two consecutive demoted polls a default tracker needs,
          // and 3 parks until the test has opened a new cell. Anything later belongs to that new
          // cell and stays healthy, so only the parked verdict is in play.
          when (polls.incrementAndGet()) {
            1 -> snapshot(TOP_APP)
            2 -> snapshot(FOREGROUND)
            3 -> {
              entered.countDown()
              var released = false
              while (!released) {
                released =
                  try {
                    release.await(5, TimeUnit.MILLISECONDS)
                  } catch (_: InterruptedException) {
                    false
                  }
              }
              snapshot(FOREGROUND)
            }
            else -> snapshot(TOP_APP)
          }
        },
        pollMillis = POLL,
      )

    watch.beginCell { delivered.incrementAndGet() }
    assertTrue("the sampler never reached the parked poll", entered.await(5, TimeUnit.SECONDS))
    // The cell this verdict belongs to is gone before the verdict arrives.
    watch.beginCell {}
    release.countDown()
    Thread.sleep(POLL * 20)

    assertEquals("a verdict from a replaced cell must not be delivered", 0, delivered.get())
    assertNull("nor recorded against the cell that replaced it", watch.lastDemotion)
    watch.stop()
  }

  /**
   * The handler cancels the run, and `RemoteBenchmarkEngine` reads the reason off the back of that cancelled run. If the verdict were recorded after
   * the handler instead of before it, the cancel would be visible without a reason, read as a plain user cancel, and file the cell CANCELLED rather
   * than FAILED. Nothing else pins that ordering.
   */
  @Test
  fun theVerdictIsReadableByTheTimeTheHandlerRuns() {
    val watch = CellDemotionWatch(queue(snapshot(TOP_APP), snapshot(FOREGROUND)), pollMillis = POLL)
    val seenByHandler = AtomicReference<CpusetDemotion?>(null)
    val latch = CountDownLatch(1)

    watch.beginCell {
      seenByHandler.set(watch.lastDemotion)
      latch.countDown()
    }

    assertTrue(latch.await(5, TimeUnit.SECONDS))
    assertNotNull("lastDemotion must already be set when the handler runs", seenByHandler.get())
    watch.stop()
  }

  /**
   * The read the service actually ships. It is destructive so that a verdict explains at most one cell: the watch stays armed between cells, so a
   * demotion seen in that gap would otherwise still be sitting here when a later cell asks, and that cell may have failed for an unrelated reason.
   */
  @Test
  fun consumeVerdictReturnsTheVerdictOnceAndThenNull() {
    val watch = CellDemotionWatch(queue(snapshot(TOP_APP), snapshot(FOREGROUND)), pollMillis = POLL)
    val latch = CountDownLatch(1)
    watch.beginCell { latch.countDown() }
    assertTrue(latch.await(5, TimeUnit.SECONDS))

    assertNotNull("the first read must return the verdict", watch.consumeVerdict())
    assertNull("a second read must not hand the same verdict to another cell", watch.consumeVerdict())
    assertNull("and the property must agree", watch.lastDemotion)
    watch.stop()
  }

  @Test
  fun consumeVerdictIsNullWhenNothingFired() {
    val watch = CellDemotionWatch({ snapshot(TOP_APP) }, pollMillis = POLL)
    assertNull("before any cell", watch.consumeVerdict())
    watch.beginCell {}
    Thread.sleep(POLL * 10)
    assertNull("and while a healthy cell is running", watch.consumeVerdict())
    watch.stop()
  }

  /**
   * A job boundary must not carry a baseline. `:benchmark` outlives a job by a few seconds, so a job started inside that window reuses this process;
   * keeping the old job's `top-app` baseline would judge the new job's first cell against a state it has not reached yet, and the pocket screen's
   * `top-app` promotion is asynchronous. That produces a wrong verdict, which is worse than a missed one.
   */
  @Test
  fun closeJobForgetsTheBaselineSoTheNextJobDoesNotInheritIt() {
    val watch = CellDemotionWatch({ snapshot(TOP_APP) }, pollMillis = POLL)
    watch.beginCell {}
    Thread.sleep(POLL * 5)
    assertEquals("the baseline is learned within a job", TOP_APP, watch.bestSeen?.cpusetPath)

    watch.closeJob()

    assertNull("a new job must start with no baseline at all", watch.bestSeen)
  }

  /**
   * The verdict is job-scoped too, and for a sharper reason than the baseline.
   *
   * The watch is deliberately left armed when a run ends, so a demotion confirmed in the tail of the last cell can still be sitting unread:
   * `JobRunner` stops asking once a cell is COMPLETED. Carried into the next job, that verdict makes `PipetteBenchmarkService.beginRun` take its
   * `!isWatching && lastDemotion == null` gate as already-armed, skip arming the watch, and then pre-cancel an innocent first run with the previous
   * job's reason.
   */
  @Test
  fun closeJobDropsAnUncollectedVerdictSoItCannotFailTheNextJobsFirstCell() {
    val samples = ArrayDeque(listOf(snapshot(TOP_APP), snapshot(FOREGROUND), snapshot(FOREGROUND), snapshot(FOREGROUND)))
    val watch = CellDemotionWatch({ samples.removeFirstOrNull() ?: snapshot(FOREGROUND) }, pollMillis = POLL)
    watch.beginCell {}
    Thread.sleep(POLL * 6)
    assertNotNull("the demotion must be confirmed before this proves anything", watch.lastDemotion)

    watch.closeJob()

    assertNull("an uncollected verdict must not outlive the job it belongs to", watch.lastDemotion)
  }

  /** By contrast, a cell boundary keeps it: that is what catches a process still demoted when the next cell opens. */
  @Test
  fun stopKeepsTheBaselineForTheNextCell() {
    val watch = CellDemotionWatch({ snapshot(TOP_APP) }, pollMillis = POLL)
    watch.beginCell {}
    Thread.sleep(POLL * 5)
    watch.stop()
    assertEquals(TOP_APP, watch.bestSeen?.cpusetPath)
  }

  /** The never-attained streak crosses a cell boundary the same way the baseline does, because the condition it counts belongs to the run. */
  @Test
  fun stopKeepsTheNeverAttainedStreakForTheNextCell() {
    val watch = CellDemotionWatch({ demoted() }, pollMillis = POLL)
    watch.beginCell {}
    awaitStreak(watch)
    watch.stop()

    // Opening the next cell is the point: the seed reaches the new tracker rather than merely surviving on the watch.
    watch.beginCell {}
    // Read on the very next line, before the new cell's first poll can land: `CpusetWatcher` sleeps a whole interval before its first sample, so what
    // this observes is the seed and nothing the new cell earned for itself. Asserted as non-zero rather than against the count from before `stop()`,
    // because that count depends on how many polls the first cell got, and once it reaches the grace the carry is clamped to one below it: comparing
    // the two would fail on scheduling rather than on the invariant. `CpusetTrackerTest.theStreakCarriesIntoTheNextCell` pins the exact arithmetic
    // without any timing at all; this pins that the watch actually wires it up.
    val seeded = watch.neverTopAppSeen

    assertTrue("the next cell must not start from zero: $seeded", seeded > 0)
    watch.stop()
  }

  /**
   * The streak is job-scoped for the same reason the baseline and the verdict are.
   *
   * `stop()` writes the finished job's count forward, and `:benchmark` outlives a job by seconds. A job starting in that reuse window has no
   * `top-app` baseline yet, so an inherited count could be spent on the asynchronous promotion window the grace exists to tolerate, failing an
   * innocent first cell.
   */
  @Test
  fun closeJobForgetsTheNeverAttainedStreakSoTheNextJobDoesNotInheritIt() {
    val watch = CellDemotionWatch({ demoted() }, pollMillis = POLL)
    watch.beginCell {}
    awaitStreak(watch)

    watch.closeJob()

    assertEquals("a new job must start with no accumulated streak at all", 0, watch.neverTopAppSeen)
  }

  /**
   * Waits for the never-attained streak to become non-zero rather than sleeping a fixed span.
   *
   * The sibling baseline tests are satisfied by `beginCell`'s synchronous cell-start probe, but a streak needs a completed poll, so a fixed sleep
   * would be racing the scheduler on a loaded CI machine.
   */
  private fun awaitStreak(watch: CellDemotionWatch) {
    val deadline = System.currentTimeMillis() + STREAK_TIMEOUT_MILLIS
    while (System.currentTimeMillis() < deadline) {
      if (watch.neverTopAppSeen > 0) return
      Thread.sleep(1)
    }
    throw AssertionError("the streak never became non-zero, so nothing downstream of it proves anything")
  }

  /** A sampler queue that holds its last value, so a watch outliving the script keeps seeing the final state rather than restarting it. */
  private fun queue(vararg samples: CpuAffinitySnapshot): () -> CpuAffinitySnapshot {
    val remaining = samples.toMutableList()
    return { if (remaining.size > 1) remaining.removeAt(0) else remaining.first() }
  }

  private companion object {
    /** Fast enough to keep the suite quick; the watcher default is a full second. */
    const val POLL = 10L

    /** Generous, because it only bounds a failure: a healthy watch satisfies `awaitStreak` within a poll or two. */
    const val STREAK_TIMEOUT_MILLIS = 5_000L
  }
}
