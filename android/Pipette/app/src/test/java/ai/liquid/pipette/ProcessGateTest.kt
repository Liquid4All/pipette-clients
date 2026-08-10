package ai.liquid.pipette

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the process classification [PipetteApp] relies on to decide whether to build [AppContainer]. Regression guard for the "Could not connect to
 * :benchmark service" crash: building the container in the isolated `:benchmark` process calls `WorkManager.getInstance()`, which is uninitialized
 * there and throws. The container must be built ONLY in the main process, so `:benchmark` (and any other `:suffix` process) must classify as
 * not-main.
 *
 * The multi-process crash itself only reproduces across real processes (covered on-device by `RemoteEngineIsolationTest`, which CI does not run);
 * this unit test locks the pure predicate the guard depends on so a "simplification" to a prefix/substring check can't silently reintroduce the bug.
 */
class ProcessGateTest {
  private val pkg = "ai.liquid.pipette"

  @Test
  fun mainProcessNameIsMain() {
    assertTrue(ProcessGate.isMainProcess(pkg, pkg))
  }

  @Test
  fun benchmarkProcessIsNotMain() {
    // The manifest declares the benchmark service as android:process=":benchmark",
    // which the platform reports as "<packageName>:benchmark".
    assertFalse(ProcessGate.isMainProcess("$pkg:benchmark", pkg))
  }

  @Test
  fun otherSuffixProcessIsNotMain() {
    // Any private `:suffix` process must be treated as non-main, so a prefix or
    // substring match can't creep back in and reintroduce the WorkManager crash.
    assertFalse(ProcessGate.isMainProcess("$pkg:other", pkg))
  }

  @Test
  fun nullProcessNameIsNotMain() {
    assertFalse(ProcessGate.isMainProcess(null, pkg))
  }

  @Test
  fun unrelatedProcessNameIsNotMain() {
    assertFalse(ProcessGate.isMainProcess("com.example.other", pkg))
  }
}
