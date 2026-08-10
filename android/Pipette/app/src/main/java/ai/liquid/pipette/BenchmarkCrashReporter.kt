package ai.liquid.pipette

import android.content.Context
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.util.Log
import java.io.File
import java.util.UUID
import org.json.JSONArray
import org.json.JSONObject

/**
 * Crash capture for the isolated `:benchmark` process.
 *
 * `:benchmark` runs the llama.cpp model and exists to measure model memory cleanly, so it deliberately carries **no JVM Sentry SDK** — resident SDK
 * threads/allocations skew `max_memory_usage` (measured on PR #527: a consistent +7.8 MiB on the reported metric). Both crash classes are captured
 * without it and written as files under [pendingCrashDir]; the **main** process (which does have the JVM Sentry SDK) drains that directory and
 * uploads them (see [CrashEnvelopeForwarder]).
 * - **Native** (SIGSEGV/abort inside llama.cpp): the engine's own native lib (`libpipette_android.so`) links **sentry-native** with the `inproc`
 *   signal-handler backend and a custom transport that writes a ready-to-send Sentry `.envelope` — no network and no transport/logs/metrics/session
 *   threads (all disabled at init); only the inproc backend's idle signal-handler thread stays resident.
 * - **JVM uncaught exceptions** (the #525 class — e.g. `WorkManager.getInstance()` in `:benchmark`): a dependency-free
 *   [Thread.setDefaultUncaughtExceptionHandler] shim that serializes the throwable to a `.jvm.json` file (atomic temp+rename) and then chains the
 *   previous handler so the platform still logs a tombstone and terminates the process. The main process reconstructs a Sentry event from that JSON.
 *
 * OOM/ANR (no in-process exception to catch) are handled separately via main-process `ApplicationExitInfo`.
 */
internal object BenchmarkCrashReporter {
  private const val LIB = "pipette_android"
  private const val TAG = "pipette-crash"

  /** Cap the cause chain we serialize — defensive against a pathological/looping `cause` graph. */
  private const val MAX_CAUSE_DEPTH = 20

  /**
   * Cap the frames serialized per exception. A StackOverflowError (or deep recursion) can carry thousands of frames; serializing them all would write
   * a multi-MB JSON from an already-degraded crashing process and force the main process to rebuild that many frames. Keep the frames nearest the
   * crash site (the raw stack is crash-site-first), which are the ones that matter.
   */
  private const val MAX_FRAMES_PER_EXCEPTION = 250

  /** Exit code used only in the (unusual) no-previous-handler fallback, so the process still dies on an uncaught exception. */
  private const val UNCAUGHT_EXIT_CODE = 10

  // Shared crash-file protocol — one definition so the writer (here + native/crash_reporter.cpp) and the reader ([CrashEnvelopeForwarder]) can't
  // drift. The dir is passed to the native side via nativeInit's envelope_dir arg; the native writer builds `<pid>-<process-token>-<seq>.envelope`,
  // so keep ENVELOPE_SUFFIX in sync with the C writer's suffix.
  const val PENDING_DIR_NAME = "pending-crashes"
  const val ENVELOPE_SUFFIX = ".envelope"
  const val JVM_SUFFIX = ".jvm.json"

  /** Where crash envelopes are written for the main process to drain + upload. Shared `filesDir` (same package UID as main). */
  fun pendingCrashDir(context: Context): File = File(context.filesDir, PENDING_DIR_NAME).apply { mkdirs() }

  /** sentry-native's private working/database dir (persists a crash across process death until the next init flushes it through the transport). */
  private fun sentryDbDir(context: Context): File = File(context.filesDir, "sentry-native").apply { mkdirs() }

  // The `int`-flag PackageManager overloads are deprecated on API 33+ (compileSdk 36); use the typed *Flags overloads there and fall back on 31/32.
  @Suppress("DEPRECATION")
  private fun PackageManager.applicationMetaData(pkg: String): Bundle? =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
      getApplicationInfo(pkg, PackageManager.ApplicationInfoFlags.of(PackageManager.GET_META_DATA.toLong())).metaData
    } else {
      getApplicationInfo(pkg, PackageManager.GET_META_DATA).metaData
    }

  @Suppress("DEPRECATION")
  private fun PackageManager.packageInfoCompat(pkg: String): PackageInfo =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
      getPackageInfo(pkg, PackageManager.PackageInfoFlags.of(0L))
    } else {
      getPackageInfo(pkg, 0)
    }

  /**
   * Initialize native crash capture in the current (`:benchmark`) process. Reads the DSN + environment from the same manifest meta-data the
   * main-process auto-init uses, so there's a single source of truth. A no-op when no DSN is configured (local builds without Sentry).
   */
  fun init(context: Context) {
    val meta = runCatching { context.packageManager.applicationMetaData(context.packageName) }.getOrNull() ?: return
    val dsn = meta.getString("io.sentry.dsn").orEmpty()
    if (dsn.isBlank()) return
    val environment = meta.getString("io.sentry.environment").orEmpty()
    // Match the release the JVM Sentry SDK derives by default in the main process — "<applicationId>@<versionName>+<versionCode>" — so native
    // :benchmark crashes land under the SAME release as every other event from this build (not a "+versionCode"-less variant that would split
    // release-health and orphan symbols).
    val release =
      runCatching {
          val pkgInfo = context.packageManager.packageInfoCompat(context.packageName)
          "${context.packageName}@${pkgInfo.versionName}+${pkgInfo.longVersionCode}"
        }
        .getOrDefault(context.packageName)

    val pending = pendingCrashDir(context)
    // JVM uncaught-exception capture first: it's pure JVM, so it works even if the native lib below fails to load.
    installJvmCrashHandler(pending)

    // Guard the native load/init: a failure here (missing/mis-staged .so for an ABI, link error, unsupported device) must NOT crash `:benchmark`'s
    // onCreate — that would take down the process the reporter exists to observe and break all benchmarking. Degrade to "native capture off" (the JVM
    // handler above is already installed), exactly as the main-process engine load does via NativeLib.isAvailable.
    runCatching {
        System.loadLibrary(LIB)
        nativeInit(dsn, environment, release, sentryDbDir(context).absolutePath, pending.absolutePath)
      }
      .onFailure { Log.w(TAG, "native crash reporter init failed; JVM crash capture remains active", it) }
  }

  /**
   * Install a default uncaught-exception handler that persists the throwable as JSON for the main process to send, then chains the previous handler
   * so the platform's default crash behavior (tombstone + process kill) still runs. Serialization is best-effort — any failure is swallowed so it can
   * never prevent the chained handler from terminating the process.
   */
  private fun installJvmCrashHandler(pendingDir: File) {
    val previous = Thread.getDefaultUncaughtExceptionHandler()
    Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
      runCatching { writeJvmCrash(pendingDir, thread, throwable) }.onFailure { Log.w(TAG, "failed to persist JVM crash", it) }
      if (previous != null) {
        previous.uncaughtException(thread, throwable)
      } else {
        // No prior handler (unusual on Android): don't swallow the crash — terminate like the default would.
        Process.killProcess(Process.myPid())
        @Suppress("ExitOutsideMain") System.exit(UNCAUGHT_EXIT_CODE)
      }
    }
  }

  /** Serialize the throwable + its cause chain to `<pendingDir>/jvm-<pid>-<ts>-<eventId>.jvm.json`, written atomically (temp + rename). */
  @Suppress("DEPRECATION") // Thread.id: the non-deprecated Thread.threadId() is API 34+, but minSdk is 31
  private fun writeJvmCrash(pendingDir: File, thread: Thread, throwable: Throwable) {
    // Mint the Sentry event id HERE, at capture, and persist it — the reader ([JvmCrashConverter]) reuses it as the event's id. This keeps the id
    // stable across re-forwards: if a handed-off `.jvm.json` can't be deleted and is drained again, Sentry ingests the same event_id and de-dups it
    // server-side (a fresh random id per conversion would surface as a duplicate). It also makes the filename globally unique — no pid-reuse /
    // wall-clock-step collision that a pid+timestamp name could hit — so a new incarnation can't clobber an earlier still-undrained crash.
    val eventId = UUID.randomUUID().toString().replace("-", "") // Sentry's 32-hex-char SentryId form

    val root = JSONObject()
    root.put("event_id", eventId)
    root.put("timestamp_ms", System.currentTimeMillis())
    root.put("thread", thread.name)
    root.put("thread_id", thread.id) // links the crashed thread to the exception's stacktrace on the Sentry side

    // Thrown-first order (thrown exception, then its cause, ...); the main process reverses to Sentry's oldest-first convention.
    val exceptions = JSONArray()
    var current: Throwable? = throwable
    var depth = 0
    while (current != null && depth < MAX_CAUSE_DEPTH) {
      val exc = JSONObject()
      exc.put("type", current.javaClass.name)
      exc.put("message", current.message) // org.json drops the key when the value is null

      // Cap frames per exception (crash-site-first order → keep the frames nearest the crash). Bounds the on-disk JSON and the reader's rebuild for a
      // StackOverflowError / deeply recursive stack that would otherwise carry thousands of frames.
      val stack = current.stackTrace
      val frames = JSONArray()
      for (i in 0 until minOf(stack.size, MAX_FRAMES_PER_EXCEPTION)) {
        val frame = stack[i]
        frames.put(
          JSONObject().apply {
            put("class", frame.className)
            put("method", frame.methodName)
            put("file", frame.fileName)
            put("line", frame.lineNumber)
            put("native", frame.isNativeMethod)
          }
        )
      }
      exc.put("frames", frames)
      exceptions.put(exc)

      val next = current.cause
      if (next === current) break // self-cause guard
      current = next
      depth++
    }
    root.put("exceptions", exceptions)

    val name = "jvm-${Process.myPid()}-${System.currentTimeMillis()}-$eventId"
    val tmp = File(pendingDir, "$name$JVM_SUFFIX.tmp")
    val dest = File(pendingDir, "$name$JVM_SUFFIX")
    tmp.writeText(root.toString())
    if (tmp.renameTo(dest)) {
      Log.i(TAG, "wrote JVM crash ${dest.name}")
    } else {
      tmp.delete()
    }
  }

  private external fun nativeInit(dsn: String, environment: String, release: String, dbPath: String, envelopeDir: String)
}
