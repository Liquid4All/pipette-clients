package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.FOREGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The concurrent half of the guard, which [CpusetTrackerTest] deliberately does not touch. Both seams the class was built with are used here: the
 * sampler is a lambda, so no `/proc` is read, and [CpusetWatcher] takes an explicit poll interval, so these run in milliseconds.
 *
 * `android.os.Process.setThreadPriority` and `android.util.Log` are not no-ops on the JVM, they are stubs that *throw*. The watcher thread runs here
 * unmodified only because it guards both, which is deliberate: it is a daemon thread whose death would be silent, so a logging failure must not be
 * able to end the watch. These tests would fail if that guarding were removed.
 */
class CpusetWatcherTest {
  @Test
  fun reportsADemotionOnceTheStreakIsConfirmed() {
    val fired = AtomicReference<CpusetDemotion?>(null)
    val latch = CountDownLatch(1)
    val samples = sequenceOf(snapshot(TOP_APP), snapshot(FOREGROUND), snapshot(FOREGROUND)).iterator()
    val watcher = CpusetWatcher({ if (samples.hasNext()) samples.next() else snapshot(FOREGROUND) }, pollMillis = POLL)

    watcher.start {
      fired.set(it)
      latch.countDown()
    }
    assertTrue("watcher never reported the demotion", latch.await(5, TimeUnit.SECONDS))
    watcher.stop()

    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, fired.get()?.reason)
  }

  @Test
  fun deliversAtMostOneDemotionAndThenStopsSampling() {
    val calls = AtomicInteger(0)
    val latch = CountDownLatch(1)
    val polls = AtomicInteger(0)
    val watcher =
      CpusetWatcher(
        {
          polls.incrementAndGet()
          if (polls.get() == 1) snapshot(TOP_APP) else snapshot(FOREGROUND)
        },
        pollMillis = POLL,
      )

    watcher.start {
      calls.incrementAndGet()
      latch.countDown()
    }
    assertTrue(latch.await(5, TimeUnit.SECONDS))
    val pollsAtFire = polls.get()
    // Long enough that a thread which kept running would poll many more times.
    Thread.sleep(POLL * 20)
    watcher.stop()

    assertEquals("onDemotion must be invoked exactly once", 1, calls.get())
    assertEquals("the thread must stop sampling after it fires", pollsAtFire, polls.get())
  }

  /**
   * The race `stop()` exists to close. A sample already in flight cannot be preempted by an interrupt, so without the stopped flag it would deliver
   * after the caller has read past it: the finding would be dropped silently and then attributed to whatever ran next.
   *
   * The sampler swallows the interrupt on purpose. An earlier version of this test let `stop()`'s interrupt break the await, so the sampler never
   * returned a demoted snapshot, nothing was ever confirmed, and the test passed against an implementation with no guard at all.
   */
  @Test
  fun doesNotDeliverADemotionDetectedAfterStop() {
    val entered = CountDownLatch(1)
    val release = CountDownLatch(1)
    val delivered = AtomicInteger(0)
    val polls = AtomicInteger(0)
    val watcher =
      CpusetWatcher(
        {
          if (polls.incrementAndGet() >= 2) {
            entered.countDown()
            // Genuinely uninterruptible, standing in for a /proc read already underway when stop()
            // lands. `await` alone is NOT enough: it throws on interrupt, so an earlier version of
            // this test never reached the demotion at all and passed against a watcher with no
            // guard whatsoever. Swallowing the interrupt here is what makes the test able to fail.
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
          } else {
            snapshot(TOP_APP)
          }
        },
        pollMillis = POLL,
        tracker = CpusetTracker(confirmations = 1),
      )

    watcher.start { delivered.incrementAndGet() }
    assertTrue(entered.await(5, TimeUnit.SECONDS))
    watcher.stop()
    release.countDown()
    Thread.sleep(POLL * 20)

    // The sample completes and confirms a demotion; the guard is what stops it being handed on.
    assertEquals("a demotion confirmed after stop() must not be delivered", 0, delivered.get())
  }

  @Test
  fun startIsRefusedWhileAThreadIsAliveAndStopIsIdempotent() {
    // Counting distinct sampling threads, not samples: a poll-count assertion cannot tell one thread
    // from two, so it would stay green with the guard deleted, which is the bug it is meant to pin.
    val threads = ConcurrentHashMap.newKeySet<Thread>()
    val polls = AtomicInteger(0)
    val watcher =
      CpusetWatcher(
        {
          // Identity, not name: every watcher thread is named "pipette-cpuset-watch", so counting
          // names would report 1 for two threads and defeat the assertion below.
          threads.add(Thread.currentThread())
          polls.incrementAndGet()
          snapshot(TOP_APP)
        },
        pollMillis = POLL,
      )
    watcher.stop()
    watcher.stop()

    watcher.start {}
    watcher.start {}
    Thread.sleep(POLL * 10)
    watcher.stop()
    val settled = polls.get()
    Thread.sleep(POLL * 10)

    assertTrue("some sampling must have happened", settled > 0)
    assertEquals("a second start must not add a second sampling thread", 1, threads.size)
    assertEquals("stop() must leave nothing polling", settled, polls.get())
    watcher.stop()
  }

  /**
   * A handler that throws must not escape the poll thread. Asserted through the default uncaught-exception handler, because an exception on a
   * non-main thread does not fail a JUnit run on its own: without this, the test passes with the `runCatching` around `onDemotion` deleted.
   */
  @Test
  fun survivesAThrowingHandler() {
    val escaped = AtomicReference<Throwable?>(null)
    val previous = Thread.getDefaultUncaughtExceptionHandler()
    Thread.setDefaultUncaughtExceptionHandler { _, error -> escaped.set(error) }
    try {
      val latch = CountDownLatch(1)
      val samples = sequenceOf(snapshot(TOP_APP)).iterator()
      val watcher =
        CpusetWatcher(
          { if (samples.hasNext()) samples.next() else snapshot(FOREGROUND) },
          pollMillis = POLL,
          tracker = CpusetTracker(confirmations = 1),
        )

      watcher.start {
        latch.countDown()
        error("handler blew up")
      }
      assertTrue(latch.await(5, TimeUnit.SECONDS))
      watcher.stop()
      Thread.sleep(POLL * 5)

      assertNull("the handler's exception must not escape the poll thread", escaped.get())
    } finally {
      Thread.setDefaultUncaughtExceptionHandler(previous)
    }
  }

  /** A sampler that throws is logged and skipped, never fatal: a probe failure must not be able to end a run. */
  @Test
  fun survivesAThrowingSampler() {
    val polls = AtomicInteger(0)
    val fired = AtomicReference<CpusetDemotion?>(null)
    val latch = CountDownLatch(1)
    val watcher =
      CpusetWatcher(
        {
          when (polls.incrementAndGet()) {
            1 -> snapshot(TOP_APP)
            2 -> error("/proc read exploded")
            else -> snapshot(FOREGROUND)
          }
        },
        pollMillis = POLL,
        tracker = CpusetTracker(confirmations = 1),
      )

    watcher.start {
      fired.set(it)
      latch.countDown()
    }
    assertTrue("a throwing sample must not kill the watcher", latch.await(5, TimeUnit.SECONDS))
    watcher.stop()
    assertNotNull(fired.get())
  }

  private companion object {
    /** Fast enough to keep the suite quick, slow enough that ordering assertions are not racing the scheduler. */
    const val POLL = 10L
  }
}
