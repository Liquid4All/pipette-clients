package ai.liquid.pipette

import android.app.ApplicationExitInfo
import io.sentry.SentryLevel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the Sentry event/attachments built for an abnormal `:benchmark` exit ([BenchmarkExitReporter.buildEvent] / [buildAttachments]). The
 * system-side read of [ApplicationExitInfo] can't be unit-tested (no public constructor, and the stub throws), so the reporting logic is fed a plain
 * [BenchmarkExit] — the decoupling that makes this testable. `REASON_*` are compile-time `static final int` constants, so they inline without
 * touching the Android stub.
 */
class BenchmarkExitReporterTest {
  private fun exit(reason: Int, description: String? = "killed", trace: String? = null) =
    BenchmarkExit(
      reason = reason,
      timestampMs = 1_700_000_000_000L,
      description = description,
      importance = 100,
      status = 0,
      pssKb = 512_000L,
      rssKb = 640_000L,
      pid = 4242,
      traceBytes = trace?.toByteArray(),
    )

  @Test
  fun lowMemoryIsFatalWithReasonTagAndFingerprint() {
    val event = BenchmarkExitReporter.buildEvent(exit(ApplicationExitInfo.REASON_LOW_MEMORY, description = "low mem"))
    assertEquals(SentryLevel.FATAL, event.level)
    assertEquals("java", event.platform)
    assertEquals("benchmark", event.tags?.get("process"))
    assertEquals("LOW_MEMORY", event.tags?.get("exit.reason"))
    assertEquals(1_700_000_000_000L, event.timestamp.time)
    assertTrue(event.message?.formatted?.contains("LOW_MEMORY") == true)
    assertTrue(event.message?.formatted?.contains("low mem") == true) // description included
    assertEquals(listOf("benchmark-exit", "LOW_MEMORY"), event.fingerprints)
    assertEquals(512_000L, event.extras?.get("exit.pss_kb"))
    assertEquals(4242, event.extras?.get("exit.pid"))
  }

  @Test
  fun anrIsErrorLevel() {
    val event = BenchmarkExitReporter.buildEvent(exit(ApplicationExitInfo.REASON_ANR))
    assertEquals(SentryLevel.ERROR, event.level)
    assertEquals("ANR", event.tags?.get("exit.reason"))
    assertEquals(listOf("benchmark-exit", "ANR"), event.fingerprints)
  }

  @Test
  fun unknownReasonIsLabelledWithItsCode() {
    val event = BenchmarkExitReporter.buildEvent(exit(reason = 999))
    assertEquals("OTHER(999)", event.tags?.get("exit.reason"))
    assertEquals(SentryLevel.FATAL, event.level) // non-ANR → fatal
  }

  @Test
  fun nullDescriptionOmittedFromMessage() {
    val event = BenchmarkExitReporter.buildEvent(exit(ApplicationExitInfo.REASON_LOW_MEMORY, description = null))
    val formatted = event.message?.formatted!!
    assertTrue(formatted.contains("LOW_MEMORY"))
    assertTrue("no dangling parenthetical when description is null", !formatted.contains("("))
  }

  @Test
  fun traceBecomesAttachment() {
    val attachments = BenchmarkExitReporter.buildAttachments(exit(ApplicationExitInfo.REASON_ANR, trace = "main thread stuck here"))
    assertEquals(1, attachments.size)
    assertEquals("benchmark-exit-trace.txt", attachments[0].filename)
    assertEquals("text/plain", attachments[0].contentType)
    assertEquals("main thread stuck here", attachments[0].bytes!!.decodeToString())
  }

  @Test
  fun noTraceMeansNoAttachment() {
    assertTrue(BenchmarkExitReporter.buildAttachments(exit(ApplicationExitInfo.REASON_LOW_MEMORY, trace = null)).isEmpty())
    assertTrue(BenchmarkExitReporter.buildAttachments(exit(ApplicationExitInfo.REASON_ANR, trace = "")).isEmpty()) // empty ⇒ skipped
  }

  @Test
  fun extrasNullSafeForMissingDescription() {
    val event = BenchmarkExitReporter.buildEvent(exit(ApplicationExitInfo.REASON_LOW_MEMORY, description = null))
    assertNull(event.extras?.get("exit.description"))
  }
}
