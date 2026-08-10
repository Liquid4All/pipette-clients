package ai.liquid.pipette

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.util.Log
import io.sentry.Attachment
import io.sentry.Sentry
import io.sentry.SentryEvent
import io.sentry.SentryLevel
import io.sentry.protocol.Message
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.Date
import java.util.concurrent.Executors

/**
 * Reports the `:benchmark` crash classes that leave **no in-process exception to catch** — OOM kills and ANRs — from the main process, by reading
 * [ActivityManager.getHistoricalProcessExitReasons] (minSdk 31) and forwarding each abnormal, otherwise-uncaptured exit as a Sentry event.
 *
 * Native SIGSEGV (M1, sentry-native) and JVM uncaught exceptions (M3, [BenchmarkCrashReporter]) are caught inside `:benchmark`; this covers what the
 * OS reaps silently. Events go out via the [BenchmarkOutbox] (not `Sentry.captureEvent`) so a subprocess death never mutates the main process's
 * session.
 *
 * **Reported reasons:** ANR, LOW_MEMORY, and EXCESSIVE_RESOURCE_USAGE only. Excluded: REASON_CRASH (JVM, owned by M3), REASON_CRASH_NATIVE (native,
 * owned by M1), normal idle-teardown exits (EXIT_SELF/OTHER), and **REASON_SIGNALED** — the service self-terminates via `Process.killProcess`
 * (SIGKILL) on every normal teardown (`PipetteBenchmarkService.onDestroy`), which is indistinguishable from an LMK OOM SIGKILL, so reporting SIGNALED
 * would flag every finished run as a crash. A low-memory kill that surfaces only as SIGNALED is therefore missed — accepted, to avoid that
 * false-positive flood.
 *
 * **De-dup:** each exit has a stable key (`timestamp-pid-reason`); a persisted set of already-reported keys gates what's new — robust to out-of-order
 * or same-millisecond records (a scalar high-water mark is not). A key is added only after the event is handed off, so a failed forward is retried;
 * the set is pruned to the records still in the system buffer so it stays bounded, and persisted synchronously with `commit()` once after the loop (a
 * crash in the window between forwarding exits and that commit can re-report the exits handed off in that pass — at-least-once, accepted). The
 * **first** run only records the baseline (all current keys), so pre-existing history isn't back-reported on rollout — and because the baseline is
 * seeded even when the buffer is empty, the first genuinely-new exit on a fresh install IS reported.
 */
internal object BenchmarkExitReporter {
  private const val TAG = "pipetteExitReport"
  private const val PREFS = "benchmark-exit-reporter"
  private const val KEY_REPORTED = "reported_keys"
  private const val MAX_TRACE_BYTES = 2 * 1024 * 1024
  private const val READ_CHUNK = 8192

  private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "benchmark-exit-report").apply { isDaemon = true } }

  /** Enqueue a scan of the `:benchmark` exit history. No-op when Sentry is disabled or there's nothing new. */
  fun reportNewExits(context: Context) {
    val appContext = context.applicationContext
    executor.execute { runCatching { reportNow(appContext) }.onFailure { Log.w(TAG, "exit report failed", it) } }
  }

  private fun reportNow(context: Context) {
    val am = if (Sentry.isEnabled()) context.getSystemService(ActivityManager::class.java) else null
    am ?: return
    val benchProc = ProcessGate.benchmarkProcessName(context.packageName)
    val exits = am.getHistoricalProcessExitReasons(context.packageName, 0, 0).filter { it.processName == benchProc }
    processExits(context, exits)
  }

  private fun processExits(context: Context, exits: List<ApplicationExitInfo>) {
    val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    val currentKeys = exits.mapTo(HashSet()) { it.dedupKey() }
    if (!prefs.contains(KEY_REPORTED)) {
      // First run with this feature: treat everything currently in the buffer as already-known (don't back-report pre-existing history). Seeding
      // happens even when the buffer is empty, so the first genuinely-new exit later is reported rather than mistaken for baseline.
      prefs.edit().putStringSet(KEY_REPORTED, currentKeys).commit()
      return
    }
    val stored = prefs.getStringSet(KEY_REPORTED, emptySet()).orEmpty()
    val reported = HashSet(stored)
    var reportedCount = 0
    for (info in exits.sortedBy { it.timestamp }) {
      val key = info.dedupKey()
      if (key in reported || !isReportable(info.reason)) continue
      val exit = info.toBenchmarkExit()
      val handed =
        runCatching { BenchmarkOutbox.writeEvent(buildEvent(exit), buildAttachments(exit)) }
          .getOrElse { e ->
            Log.w(TAG, "failed to forward exit", e)
            false
          }
      if (handed) {
        reported.add(key)
        reportedCount++
      }
    }
    // Prune to the still-visible records so the set stays bounded; persist synchronously so a process death can't drop it and re-report. Skip the
    // write when the pruned set is unchanged (the steady state on a reconnect: no new exits, nothing aged out of the buffer) so a multi-cell run
    // doesn't force a redundant synchronous SharedPreferences commit per reconnect.
    val pruned = reported.intersect(currentKeys)
    if (pruned != stored) prefs.edit().putStringSet(KEY_REPORTED, pruned).commit()
    if (reportedCount > 0) Log.i(TAG, "forwarded $reportedCount :benchmark abnormal exit(s) to Sentry outbox")
  }

  private fun isReportable(reason: Int): Boolean =
    reason == ApplicationExitInfo.REASON_ANR ||
      reason == ApplicationExitInfo.REASON_LOW_MEMORY ||
      reason == ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE

  private fun reasonName(reason: Int): String =
    when (reason) {
      ApplicationExitInfo.REASON_ANR -> "ANR"
      ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
      ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
      else -> "OTHER($reason)"
    }

  /** Build the Sentry event for an abnormal `:benchmark` exit. Pure (no Android system types) so it's unit-testable. */
  fun buildEvent(exit: BenchmarkExit): SentryEvent {
    val reasonName = reasonName(exit.reason)
    return SentryEvent(Date(exit.timestampMs)).apply {
      level = if (exit.reason == ApplicationExitInfo.REASON_ANR) SentryLevel.ERROR else SentryLevel.FATAL
      platform = "java"
      setTag("process", "benchmark")
      setTag("exit.reason", reasonName)
      message = Message().apply { formatted = ":benchmark exited abnormally: $reasonName" + (exit.description?.let { " ($it)" } ?: "") }
      // Group all exits of the same reason together, independent of the varying description/timestamp.
      fingerprints = listOf("benchmark-exit", reasonName)
      setExtra("exit.description", exit.description)
      setExtra("exit.importance", exit.importance)
      setExtra("exit.status", exit.status)
      setExtra("exit.pss_kb", exit.pssKb)
      setExtra("exit.rss_kb", exit.rssKb)
      setExtra("exit.pid", exit.pid)
    }
  }

  /** The ANR/tombstone trace as a Sentry attachment, when the record carried one. Pure/unit-testable. */
  fun buildAttachments(exit: BenchmarkExit): List<Attachment> =
    exit.traceBytes?.takeIf { it.isNotEmpty() }?.let { listOf(Attachment(it, "benchmark-exit-trace.txt", "text/plain")) } ?: emptyList()

  private fun ApplicationExitInfo.dedupKey(): String = "$timestamp-$pid-$reason"

  private fun ApplicationExitInfo.toBenchmarkExit(): BenchmarkExit =
    BenchmarkExit(
      reason = reason,
      timestampMs = timestamp,
      description = description,
      importance = importance,
      status = status,
      pssKb = pss,
      rssKb = rss,
      pid = pid,
      // Present for ANR (the trace) and native crash (tombstone); null otherwise. Read capped so a pathological multi-MB trace can't balloon memory,
      // and kept as raw bytes (no String round-trip that could corrupt non-UTF8 content).
      traceBytes = runCatching { traceInputStream?.use { readCapped(it, MAX_TRACE_BYTES) } }.getOrNull(),
    )

  private fun readCapped(stream: InputStream, cap: Int): ByteArray {
    val out = ByteArrayOutputStream()
    val chunk = ByteArray(READ_CHUNK)
    var total = 0
    while (total < cap) {
      val n = stream.read(chunk, 0, minOf(chunk.size, cap - total))
      if (n < 0) break
      out.write(chunk, 0, n)
      total += n
    }
    return out.toByteArray()
  }
}

/**
 * Plain holder decoupling exit-reporting logic from the un-mockable [ApplicationExitInfo] system type (so [BenchmarkExitReporter.buildEvent] is
 * testable).
 */
@Suppress("ArrayInDataClass") // used only as a field carrier; instances are never compared/hashed
internal data class BenchmarkExit(
  val reason: Int,
  val timestampMs: Long,
  val description: String?,
  val importance: Int,
  val status: Int,
  val pssKb: Long,
  val rssKb: Long,
  val pid: Int,
  val traceBytes: ByteArray?,
)
