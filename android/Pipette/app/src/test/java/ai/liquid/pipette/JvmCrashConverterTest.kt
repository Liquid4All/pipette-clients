package ai.liquid.pipette

import io.sentry.SentryLevel
import io.sentry.protocol.SentryId
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Locks the reconstruction of a Sentry event from the JSON a `:benchmark` JVM crash is serialized to ([BenchmarkCrashReporter] writes it,
 * [JvmCrashConverter] reads it). The tricky, regression-prone parts are the two order reversals (Sentry wants cause chain oldest-first and stack
 * frames crashing-frame-last, both opposite to the raw JVM order) and the type/module split — so those are pinned explicitly.
 */
class JvmCrashConverterTest {
  private fun frame(cls: String, method: String, file: String?, line: Int, native: Boolean = false): JSONObject =
    JSONObject().apply {
      put("class", cls)
      put("method", method)
      put("file", file)
      put("line", line)
      put("native", native)
    }

  private fun exception(type: String, message: String?, frames: List<JSONObject>): JSONObject =
    JSONObject().apply {
      put("type", type)
      put("message", message)
      put("frames", JSONArray(frames))
    }

  private fun crashJson(thread: String, timestampMs: Long, exceptions: List<JSONObject>): String =
    JSONObject()
      .apply {
        put("thread", thread)
        put("timestamp_ms", timestampMs)
        put("exceptions", JSONArray(exceptions))
      }
      .toString()

  @Test
  fun malformedOrEmptyReturnsNull() {
    assertNull(JvmCrashConverter.toEvent("not json"))
    assertNull(JvmCrashConverter.toEvent(JSONObject().toString())) // no exceptions key
    assertNull(JvmCrashConverter.toEvent(crashJson("main", 1L, emptyList()))) // empty exceptions
  }

  @Test
  fun fatalLevelPlatformAndProcessTag() {
    val json = crashJson("main", 42L, listOf(exception("java.lang.IllegalStateException", "boom", emptyList())))
    val event = JvmCrashConverter.toEvent(json)!!
    assertEquals(SentryLevel.FATAL, event.level)
    assertEquals("java", event.platform)
    assertEquals("benchmark", event.tags?.get("process"))
    assertEquals(42L, event.timestamp.time)
  }

  @Test
  fun eventIdReusedFromJsonSoReForwardsDeDupServerSide() {
    // The writer persists the event_id it minted at capture; the converter must reuse it verbatim (a re-drained file would otherwise mint a fresh id
    // and surface as a duplicate Sentry issue). SentryId renders as 32 lowercase hex chars.
    val hex = "0123456789abcdef0123456789abcdef"
    val json =
      JSONObject()
        .apply {
          put("event_id", hex)
          put("thread", "main")
          put("timestamp_ms", 1L)
          put("exceptions", JSONArray(listOf(exception("java.lang.IllegalStateException", "boom", emptyList()))))
        }
        .toString()
    val event = JvmCrashConverter.toEvent(json)!!
    assertEquals(hex, event.eventId.toString())
  }

  @Test
  fun missingOrMalformedEventIdFallsBackToAGeneratedId() {
    // Files that predate the event_id field (or carry a bad value) must still convert — with some non-empty id, just not a stable one.
    val noId = JvmCrashConverter.toEvent(crashJson("main", 1L, listOf(exception("E", "m", emptyList()))))!!
    assertNotEquals(SentryId.EMPTY_ID, noId.eventId)
    val badId =
      JSONObject()
        .apply {
          put("event_id", "not-a-uuid")
          put("exceptions", JSONArray(listOf(exception("E", "m", emptyList()))))
        }
        .toString()
    assertNotEquals(SentryId.EMPTY_ID, JvmCrashConverter.toEvent(badId)!!.eventId)
  }

  @Test
  fun typeAndModuleSplitFromFqcn() {
    val event = JvmCrashConverter.toEvent(crashJson("main", 1L, listOf(exception("java.lang.IllegalStateException", "boom", emptyList()))))!!
    val exc = event.exceptions!!.single()
    assertEquals("IllegalStateException", exc.type)
    assertEquals("java.lang", exc.module)
    assertEquals("boom", exc.value)
  }

  @Test
  fun causeChainReversedToOldestFirstWithMechanismOnThrown() {
    // Raw JSON is thrown-first: [thrown = IllegalState, cause = IOException]. Sentry wants oldest-first: [IOException, IllegalState].
    val json =
      crashJson(
        "worker-1",
        1L,
        listOf(exception("java.lang.IllegalStateException", "outer", emptyList()), exception("java.io.IOException", "root cause", emptyList())),
      )
    val event = JvmCrashConverter.toEvent(json)!!
    val exceptions = event.exceptions!!
    assertEquals(2, exceptions.size)
    assertEquals("IOException", exceptions[0].type) // oldest (root cause) first
    assertEquals("IllegalStateException", exceptions[1].type) // thrown last
    // The unhandled mechanism marks the crash; it belongs on the thrown exception (now last).
    assertNull(exceptions[0].mechanism)
    assertEquals("UncaughtExceptionHandler", exceptions[1].mechanism?.type)
    assertEquals(false, exceptions[1].mechanism?.isHandled)
  }

  @Test
  fun framesReversedSoCrashingFrameIsLastAndInAppFlagged() {
    // Raw frames are crash-site-first: [inner ai.liquid.pipette, framework android.os]. Sentry renders crashing frame last → reversed.
    val json =
      crashJson(
        "main",
        1L,
        listOf(
          exception(
            "java.lang.RuntimeException",
            "x",
            listOf(frame("ai.liquid.pipette.Foo", "crashHere", "Foo.kt", 10), frame("android.os.Handler", "dispatchMessage", "Handler.java", 99)),
          )
        ),
      )
    val frames = JvmCrashConverter.toEvent(json)!!.exceptions!!.single().stacktrace!!.frames!!
    assertEquals(2, frames.size)
    // Reversed: framework frame first, crashing (in-app) frame last.
    assertEquals("android.os.Handler", frames[0].module)
    assertEquals("ai.liquid.pipette.Foo", frames[1].module)
    assertEquals("crashHere", frames[1].function)
    assertEquals(10, frames[1].lineno)
    assertEquals(true, frames[1].isInApp) // ai.liquid.pipette frame is in-app
    assertFalse(frames[0].isInApp == true) // framework frame is not
  }

  @Test
  fun missingOptionalFieldsAreNull() {
    // A frame with no file, negative line (unknown), and an exception with no message.
    val json = crashJson("main", 1L, listOf(exception("SomeException", null, listOf(frame("Foo", "bar", null, -1)))))
    val exc = JvmCrashConverter.toEvent(json)!!.exceptions!!.single()
    assertNull(exc.value)
    assertEquals("SomeException", exc.type) // no package → type is the whole string
    assertNull(exc.module)
    val f = exc.stacktrace!!.frames!!.single()
    assertNull(f.filename)
    assertNull(f.lineno) // negative sentinel dropped
  }
}
