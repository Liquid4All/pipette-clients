package ai.liquid.pipette

import io.sentry.SentryEvent
import io.sentry.SentryLevel
import io.sentry.protocol.Mechanism
import io.sentry.protocol.SentryException
import io.sentry.protocol.SentryId
import io.sentry.protocol.SentryStackFrame
import io.sentry.protocol.SentryStackTrace
import io.sentry.protocol.SentryThread
import java.util.Date
import org.json.JSONObject

/**
 * Reconstructs a Sentry [SentryEvent] from the JSON a `:benchmark`-process JVM crash was serialized to (see [BenchmarkCrashReporter]). Pure and
 * dependency-light (org.json + the Sentry protocol types) so it can be unit-tested off-device.
 *
 * The main process delivers the returned event via [BenchmarkOutbox] (an outbox envelope, NOT `Sentry.captureEvent`, so it can't mutate the main
 * process's release-health session). BenchmarkOutbox stamps `release`/`environment`/`dist` from the SDK options; this converter fills in what's
 * crash-specific: the exception chain, stack frames, the fatal level, and a `process:benchmark` tag (mirroring the native path).
 */
internal object JvmCrashConverter {
  private const val IN_APP_PACKAGE = "ai.liquid.pipette"

  /** Returns null if the JSON is malformed or carries no exception. */
  fun toEvent(json: String): SentryEvent? {
    val root = runCatching { JSONObject(json) }.getOrNull() ?: return null
    return buildEvent(root)
  }

  private fun buildEvent(root: JSONObject): SentryEvent? {
    val rawExceptions = root.optJSONArray("exceptions")?.takeIf { it.length() > 0 } ?: return null

    val event = SentryEvent(Date(root.optLong("timestamp_ms", System.currentTimeMillis())))
    // Reuse the id the writer minted at capture time so a re-forwarded crash keeps the same event_id and Sentry de-dups it server-side (see
    // BenchmarkCrashReporter.writeJvmCrash). Fall back to the constructor's random id for older files that predate the field or carry a malformed
    // one.
    root.optString("event_id").takeIf { it.isNotEmpty() }?.let { id -> runCatching { event.eventId = SentryId(id) } }
    event.level = SentryLevel.FATAL
    event.platform = "java"
    event.setTag("process", "benchmark")

    // The crashing thread's id links the exception's stacktrace to the thread in the Sentry UI. Absent (0) if the writer didn't record one.
    val threadId = root.optLong("thread_id", 0L).takeIf { it > 0L }

    // JSON is thrown-first (index 0 = the thrown exception); Sentry wants oldest-first (root cause first, thrown last) → reverse. The mechanism that
    // marks the crash as unhandled belongs on the thrown exception (index 0).
    val built = ArrayList<SentryException>(rawExceptions.length())
    for (i in 0 until rawExceptions.length()) {
      built.add(buildException(rawExceptions.getJSONObject(i), isThrown = i == 0, threadId = threadId))
    }
    event.exceptions = built.asReversed().toList()

    root
      .optString("thread")
      .takeIf { it.isNotEmpty() }
      ?.let { name ->
        event.threads =
          listOf(
            SentryThread().apply {
              this.name = name
              id = threadId
              isCrashed = true
              isCurrent = true
            }
          )
      }
    return event
  }

  private fun buildException(obj: JSONObject, isThrown: Boolean, threadId: Long?): SentryException {
    val fqcn = obj.optString("type").ifEmpty { "UnknownException" }
    return SentryException().apply {
      // Split the FQCN the way the Sentry SDK does for real throwables: simple name as `type`, package as `module`.
      type = fqcn.substringAfterLast('.')
      module = fqcn.substringBeforeLast('.', missingDelimiterValue = "").ifEmpty { null }
      value = obj.optString("message").ifEmpty { null }
      this.threadId = threadId // ties this exception's stacktrace to the crashed thread
      obj.optJSONArray("frames")?.let { stacktrace = buildStackTrace(it) }
      if (isThrown) {
        mechanism =
          Mechanism().apply {
            type = "UncaughtExceptionHandler"
            isHandled = false
          }
      }
    }
  }

  private fun buildStackTrace(rawFrames: org.json.JSONArray): SentryStackTrace {
    val frames = ArrayList<SentryStackFrame>(rawFrames.length())
    for (i in 0 until rawFrames.length()) {
      val f = rawFrames.getJSONObject(i)
      frames.add(
        SentryStackFrame().apply {
          module = f.optString("class").ifEmpty { null }
          function = f.optString("method").ifEmpty { null }
          filename = f.optString("file").ifEmpty { null }
          f.optInt("line", -1).takeIf { it >= 0 }?.let { lineno = it }
          if (f.optBoolean("native", false)) isNative = true
          isInApp = module?.startsWith(IN_APP_PACKAGE) == true
        }
      )
    }
    // StackTraceElement order is crash-site-first; Sentry renders the crashing frame last → reverse.
    return SentryStackTrace(frames.asReversed().toList())
  }
}
