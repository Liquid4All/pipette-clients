package ai.liquid.pipette.headless

import android.app.Activity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * The headless surface reports its outcome twice, on two different scales, and they must not be confused for one another.
 *
 * The log line `BENCH_DONE <n> exit=<n>` carries iOS's `HeadlessExit` numbering, which a CI consumer greps for. The Activity result code has to live
 * in Android's space, where success is `RESULT_OK` (-1) and `0` already means `RESULT_CANCELED`. Returning the exit code directly made a successful
 * headless run report "cancelled" to anything using `startActivityForResult` or the Activity Result APIs.
 */
class HeadlessResultCodeTest {

  @Test
  fun successIsResultOkRatherThanZero() {
    assertEquals(Activity.RESULT_OK, HeadlessRunActivity.activityResultFor(HeadlessRunActivity.EXIT_OK))
    // The regression this exists for: RESULT_CANCELED is 0, which is also iOS's success code.
    assertNotEquals(
      "a started run must not report RESULT_CANCELED",
      Activity.RESULT_CANCELED,
      HeadlessRunActivity.activityResultFor(HeadlessRunActivity.EXIT_OK),
    )
  }

  @Test
  fun theTwoFailuresAreDistinguishableAndSitInTheAppsOwnRange() {
    val refused = HeadlessRunActivity.activityResultFor(HeadlessRunActivity.EXIT_FAILURE)
    val usage = HeadlessRunActivity.activityResultFor(HeadlessRunActivity.EXIT_USAGE)

    assertNotEquals("a refusal and a malformed invocation must be tellable apart", refused, usage)
    // RESULT_FIRST_USER is where the platform says an app's own codes may start; below it the values are the platform's.
    for (code in listOf(refused, usage)) {
      assertEquals("$code must be an app-defined code", true, code >= Activity.RESULT_FIRST_USER)
    }
  }

  /**
   * The token straight after the sentinel must parse as an integer.
   *
   * `pipette-ctl`'s `scan_sentinel` (`crates/pipette-ctl/src/transport/process.rs:116`) splits on `BENCH_DONE` and parses the next whitespace token;
   * one it cannot parse yields `None`, which reads as "no sentinel" and falls back to the process exit code. `am start` exits 0 even when this
   * activity refuses, so that fallback would record a refusal as a completed cell. This line used to read `BENCH_DONE exit=2`, whose first token is
   * `exit=2`.
   */
  @Test
  fun theSentinelsFirstTokenIsAParsableStatus() {
    for (exit in listOf(HeadlessRunActivity.EXIT_OK, HeadlessRunActivity.EXIT_FAILURE, HeadlessRunActivity.EXIT_USAGE)) {
      val line = HeadlessRunActivity.doneLine(exit)
      val afterSentinel = line.substringAfter(HeadlessRunActivity.DONE_SENTINEL).trim().substringBefore(' ')

      assertEquals("the first token after the sentinel must be the status", exit, afterSentinel.toIntOrNull())
    }
  }

  /** The `exit=` spelling the Android docs told consumers to key on is kept, since the scanner only ever reads the first token. */
  @Test
  fun theHumanReadableExitSpellingIsKept() {
    assertEquals(true, HeadlessRunActivity.doneLine(HeadlessRunActivity.EXIT_USAGE).contains("exit=2"))
  }

  /**
   * A value carrying a line break cannot end the record and start another.
   *
   * `Log.i` emits each line of its message as its own logcat entry, and several fields here hold text this device did not choose: `auth-me` prints
   * the `organization` a person typed into the server, and `auth-import` prints the `client_id` a supplied bundle carried.
   */
  @Test
  fun aLineBreakInAValueCannotForgeARecord() {
    val forged = HeadlessRunActivity.oneLine("auth-me client_id=ev1_a organization=Acme\n${HeadlessRunActivity.doneLine(0)}")

    assertEquals("the record stays one line", 1, forged.lines().size)
    assertEquals("nothing is dropped, so a reader still sees the injected text", true, forged.contains("BENCH_DONE"))
  }

  /**
   * A carriage return and the other ISO control characters go too, not just the newline.
   *
   * A lone `\r` rewrites a terminal line in place, and an escape sequence can repaint one, so a consumer reading the log by eye sees something the
   * surface did not write. A space costs nothing here: a field on this surface may already hold one.
   */
  @Test
  fun everyControlCharacterBecomesASpace() {
    for (control in listOf("\u000A", "\u000D", "\u0000", "\u001B", "\u0009")) {
      assertEquals("U+%04X must not survive".format(control.first().code), "a b", HeadlessRunActivity.oneLine("a${control}b"))
    }
    assertEquals("an ordinary value is untouched", "organization=Acme Corp", HeadlessRunActivity.oneLine("organization=Acme Corp"))
  }
}
