package ai.liquid.pipette

import android.content.Context
import android.util.Log
import io.sentry.ScopesAdapter
import io.sentry.Sentry
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

/**
 * Main-process drain for crash reports the `:benchmark` process leaves in `filesDir/pending-crashes/` (see [BenchmarkCrashReporter]). Runs only in
 * the main process, where the JVM Sentry SDK is initialized. Two on-disk formats, each delivered via the SDK **outbox** (never `Sentry.captureEvent`,
 * so a subprocess crash can't mutate the main process's release-health session — see [BenchmarkOutbox]):
 * - **Native** (`.envelope`): a ready-to-send Sentry envelope from sentry-native — moved (renamed) straight into the outbox.
 * - **JVM** (`.jvm.json`): a serialized throwable — reconstructed into a [io.sentry.SentryEvent] by [JvmCrashConverter] and written to the outbox.
 *
 * Both formats reach the outbox by an atomic rename, so `SendCachedEnvelopeIntegration` sends them at the next app launch (the live `FileObserver`
 * only fires on in-place writes, not renames — see [BenchmarkOutbox]); this is the standard "reported on next launch" crash-delivery model. De-dup is
 * structural: a native envelope is consumed by an atomic MOVE (never copied — a copy + failed-delete would double-report); a JVM file is deleted only
 * after it is safely handed off, and per-file error isolation means one bad file can't starve the rest.
 *
 * Timing (native): sentry-native flushes a persisted crash on the next `sentry_init` (the respawn's `onCreate`), before the main process observes
 * `onServiceConnected` — so the drain triggers are process launch and `:benchmark` (re)connect. A crash from a session's last run waits until the
 * next run flushes it. JVM crashes are written eagerly, so they're available at the next drain.
 */
internal object CrashEnvelopeForwarder {
  private const val TAG = "pipetteCrashForward"
  private const val ENVELOPE_SUFFIX = BenchmarkCrashReporter.ENVELOPE_SUFFIX
  private const val JVM_SUFFIX = BenchmarkCrashReporter.JVM_SUFFIX
  private const val TMP_SUFFIX = ".tmp"

  // A temp file left by a writer killed mid-write is swept once it's older than this (comfortably longer than any write), so orphans can't
  // accumulate.
  private const val STALE_TMP_MS = 60L * 60L * 1000L

  // Serializes drains onto one background thread: keeps disk I/O off the caller (main thread / binder thread) and makes concurrent drain requests
  // (launch + connect) race-free without extra locking — the second run just finds an empty dir.
  private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "crash-forward").apply { isDaemon = true } }

  /** Enqueue a drain of `pending-crashes/`. Cheap no-op when Sentry is disabled or the dir is empty. */
  fun drain(context: Context) {
    val appContext = context.applicationContext
    executor.execute { runCatching { drainNow(appContext) }.onFailure { Log.w(TAG, "crash drain failed", it) } }
  }

  private fun drainNow(context: Context) {
    if (!Sentry.isEnabled()) return // skips local/no-DSN builds
    val pendingDir = BenchmarkCrashReporter.pendingCrashDir(context)
    if (!pendingDir.isDirectory) return
    forwardNativeEnvelopes(pendingDir)
    forwardJvmCrashes(pendingDir)
    sweepStaleTmp(pendingDir)
  }

  private fun forwardNativeEnvelopes(pendingDir: File) {
    val outboxPath = ScopesAdapter.getInstance().options.outboxPath
    if (outboxPath.isNullOrBlank()) return
    val envelopes = pendingDir.listFiles { f -> f.isFile && f.name.endsWith(ENVELOPE_SUFFIX) }
    if (envelopes.isNullOrEmpty()) return

    val outboxDir = File(outboxPath).apply { mkdirs() }
    var forwarded = 0
    for (src in envelopes) {
      // Move (not copy) into the outbox under a fresh name: outbox + pending share the app-private filesystem, so rename is atomic and reliable, and
      // consuming the file this way IS the de-dup. A rename failure leaves the file for the next drain — we never fall back to copy, since a
      // copy + failed-delete would leave the source behind and re-forward a duplicate.
      val dest = File(outboxDir, "${UUID.randomUUID()}$ENVELOPE_SUFFIX")
      if (src.renameTo(dest)) forwarded++ else Log.w(TAG, "could not move ${src.name} to outbox; will retry next drain")
    }
    if (forwarded > 0) Log.i(TAG, "forwarded $forwarded native crash envelope(s) to Sentry outbox")
  }

  private fun forwardJvmCrashes(pendingDir: File) {
    val crashes = pendingDir.listFiles { f -> f.isFile && f.name.endsWith(JVM_SUFFIX) }
    if (crashes.isNullOrEmpty()) return
    var sent = 0
    for (src in crashes) {
      val event = runCatching { JvmCrashConverter.toEvent(src.readText()) }.getOrNull()
      if (event == null) {
        // Corrupt/unparseable — delete so it can't wedge the drain on every pass.
        Log.w(TAG, "discarding unparseable JVM crash ${src.name}")
        if (!src.delete()) Log.w(TAG, "could not delete unparseable ${src.name}")
        continue
      }
      // Hand off via the outbox (never Sentry.captureEvent — see BenchmarkOutbox). Per-file runCatching isolates a failure so it can't starve the
      // rest of the batch (no poison pill); delete only once handed off, and keep the file for retry otherwise.
      val handed =
        runCatching { BenchmarkOutbox.writeEvent(event) }
          .getOrElse { e ->
            Log.w(TAG, "failed to forward ${src.name}; will retry", e)
            false
          }
      // Consume the source so it can't be re-forwarded (a fresh envelope is already in the outbox). If delete fails — rare on app-private storage —
      // rename it out of the way to a `.tmp` name, which the JVM-suffix filter ignores and sweepStaleTmp later reaps, so de-dup stays structural.
      if (handed && (src.delete() || src.renameTo(File("${src.path}$TMP_SUFFIX")))) sent++
      else if (handed) Log.w(TAG, "forwarded but could not consume ${src.name}; may re-send")
    }
    if (sent > 0) Log.i(TAG, "forwarded $sent :benchmark JVM crash(es) to Sentry outbox")
  }

  private fun sweepStaleTmp(pendingDir: File) {
    val now = System.currentTimeMillis()
    // Two writers leave temp files: the native/JVM writers in pending-crashes/, and BenchmarkOutbox in the outbox's PARENT dir (the SDK's own cache
    // dir). An abrupt kill mid-write can orphan one in either, so reap both once older than any plausible write — matching ONLY our own temp shapes
    // (`*.envelope.tmp` / `*.jvm.json.tmp`) so a blanket `*.tmp` sweep can't reap a file the SDK owns.
    val ourTmp = { f: File -> f.isFile && (f.name.endsWith("$ENVELOPE_SUFFIX$TMP_SUFFIX") || f.name.endsWith("$JVM_SUFFIX$TMP_SUFFIX")) }
    val outboxParent = ScopesAdapter.getInstance().options.outboxPath?.let { File(it).parentFile }
    listOfNotNull(pendingDir, outboxParent).forEach { dir ->
      dir.listFiles(ourTmp)?.forEach { tmp -> if (now - tmp.lastModified() > STALE_TMP_MS) tmp.delete() }
    }
  }
}
