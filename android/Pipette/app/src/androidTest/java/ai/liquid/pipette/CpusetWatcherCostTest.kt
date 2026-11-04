package ai.liquid.pipette

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device cost and viability of [CpusetWatcher]'s per-poll sampling, measured in the app's own sandbox.
 *
 * Two questions a JVM test cannot answer. First, whether the reads are permitted at all: everything the watcher depends on was validated through `adb
 * shell`, which runs as the shell uid with different access than an app. A `/proc` entry the app cannot read degrades to a null field, which
 * `CpusetTracker.offer` discards as an absence of evidence, so an app-uid denial would disarm the guard silently rather than failing loudly. Second,
 * what a poll actually costs in ART in-process, rather than through a shell interpreter whose own loop overhead dominates the reads it is timing.
 *
 * The costs are reported via logcat rather than asserted against an absolute threshold, which would be flaky on shared hardware; the only timing
 * assertion is the relative one (a cached-sweep poll must beat a full snapshot), where the margin is ~3.5x. `adb logcat -s pipette-cpuset-cost`.
 */
@RunWith(AndroidJUnit4::class)
class CpusetWatcherCostTest {
  @Test
  fun theAppSandboxCanReadEverythingTheWatcherDependsOn() {
    val snapshot = CpuAffinityProbe.snapshot()
    Log.i(TAG, "app-sandbox snapshot: ${snapshot.summary()}")
    // The primary signal. Null here means /proc/self/cpuset was unreadable from
    // the app uid, which would leave the guard resting entirely on its
    // allowed-CPU fallback.
    assertNotNull("app cannot read /proc/self/cpuset; the LEFT_TOP_APP signal would never fire", snapshot.cpusetPath)
    // The fallback signal, from /proc/self/status.
    assertNotNull("app cannot read Cpus_allowed_list; the LOST_CPUS fallback would never fire", snapshot.allowedCpus)
    assertTrue("no CPUs parsed out of Cpus_allowed_list", snapshot.allowedCount > 0)
    // The per-core sweep, which decides which cores count as the prime tier.
    assertNotNull("no per-core scaling_max_freq readable; excludesTopTier would be meaningless", snapshot.topTierCpus)
  }

  @Test
  fun measurePerPollCost() {
    val cached = CpuAffinityProbe.freqSweep()
    // Warm up: first-touch page faults and ART's interpreter-to-JIT transition
    // would otherwise land entirely in the measured window.
    repeat(WARMUP) { CpuAffinityProbe.sample(cached) }
    repeat(WARMUP) { CpuAffinityProbe.snapshot() }

    val pollNanos = timeNanos(ITERATIONS) { CpuAffinityProbe.sample(cached) }
    val fullNanos = timeNanos(ITERATIONS) { CpuAffinityProbe.snapshot() }
    val sweepNanos = timeNanos(ITERATIONS) { CpuAffinityProbe.freqSweep() }

    // What the watcher actually pays per second at its 1s poll interval, as a
    // fraction of one core, the number that answers "does this perturb the
    // measurement it is guarding".
    val dutyPercent = pollNanos / 1_000_000_000.0 * 100.0
    Log.i(TAG, "watcher poll (cached sweep): ${pollNanos / 1000.0} us")
    Log.i(TAG, "full snapshot (re-sweeps):   ${fullNanos / 1000.0} us")
    Log.i(TAG, "sweep alone:                 ${sweepNanos / 1000.0} us")
    Log.i(TAG, "duty cycle at 1 Hz:          $dutyPercent % of one core")

    assertTrue("a poll must cost less than a full snapshot; the cached sweep is the whole point", pollNanos < fullNanos)
  }

  private fun timeNanos(iterations: Int, block: () -> Unit): Double {
    val started = System.nanoTime()
    repeat(iterations) { block() }
    return (System.nanoTime() - started).toDouble() / iterations
  }

  private companion object {
    const val TAG = "pipette-cpuset-cost"
    const val WARMUP = 50
    const val ITERATIONS = 500
  }
}
