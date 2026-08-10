package ai.liquid.pipette

import android.content.Context
import com.posthog.PostHog
import com.posthog.android.PostHogAndroid
import com.posthog.android.PostHogAndroidConfig

/**
 * Product-analytics seam. Everything in the app captures through this interface rather than calling [PostHog] directly, which keeps the SDK at one
 * edge of the codebase: tests inject [NoOpAnalytics] and stay hermetic (no disk queue, no network, no static init), and swapping or removing the
 * vendor is a one-file change.
 *
 * Implementations must be safe to call from any thread and must never throw: analytics is advisory telemetry and may never take down a benchmark run
 * or a registration attempt.
 */
interface Analytics {
  /**
   * Whether this sink sends anything at all. False for [NoOpAnalytics] (an unconfigured build or a test), which is what the Settings screen gates its
   * opt-out control on, the same way the feedback card is gated on `FeedbackDialog.isAvailable()`: a toggle over a sink that collects nothing would
   * be a control that does nothing.
   */
  val isAvailable: Boolean

  /** Record [event] (use an [AnalyticsEvents] constant). Null-valued properties are dropped; PostHog rejects null property values. */
  fun capture(event: String, properties: Map<String, Any?> = emptyMap())

  /**
   * Bind subsequent events to the server-assigned device identity. See [AnalyticsEvents] for why `clientId` is the distinct id.
   *
   * No person-properties overload on purpose: this app sends PostHog an opaque device id and nothing else about who is holding the device, and the
   * iOS `AnalyticsSink` has the same single-argument shape.
   */
  fun identify(distinctId: String)

  /**
   * Send anything queued now, rather than on the SDK's periodic timer.
   *
   * Called immediately before a benchmark run begins so the queue is empty entering the measurement window. See [AnalyticsEvents] for the full
   * rationale.
   */
  fun flush()

  /**
   * Whether the user has turned analytics collection off in Settings. While true, [capture] and [identify] do nothing.
   *
   * Backed by [AnalyticsOptOutStore], **not** by the SDK's own persisted copy. See that store for the measurement showing why the SDK's is not
   * trustworthy across a launch.
   */
  fun isOptedOut(): Boolean

  /**
   * Turn collection off ([optedOut] true) or back on. Persisted by [AnalyticsOptOutStore] and re-applied at the next launch, so this is the whole of
   * the setting; see [isOptedOut].
   *
   * Events already queued when this flips to true are not recalled: PostHog's only queue-dropping call is `reset()`, which also destroys the distinct
   * id, and the queue can hold at most what was captured while the user was still opted in.
   */
  fun setOptedOut(optedOut: Boolean)
}

/** Analytics sink for builds with no PostHog project configured, and for unit tests. Does nothing, on purpose. */
object NoOpAnalytics : Analytics {
  override val isAvailable = false

  override fun capture(event: String, properties: Map<String, Any?>) = Unit

  override fun identify(distinctId: String) = Unit

  override fun flush() = Unit

  /** True because it describes what this sink does (nothing is collected), not because a user chose it. [isAvailable] is what tells the two apart. */
  override fun isOptedOut() = true

  override fun setOptedOut(optedOut: Boolean) = Unit
}

/**
 * The shared Android + iOS event taxonomy. **Every name and property key here is duplicated verbatim in the iOS client's `AnalyticsEvents`** so both
 * platforms land in one PostHog project as a single funnel rather than two dialects that have to be unioned in every query. Changing a name here
 * means changing it there.
 *
 * ### Why events fire at boundaries, never inside a measurement
 *
 * This app measures on-device LLM throughput, and background work perturbs those measurements, the same reason Sentry runs with tracing,
 * auto-performance instrumentation and app-hang tracking all disabled (see `SentryConfiguration` on iOS and the `sentry {}` block in
 * `build.gradle.kts`). So the taxonomy deliberately contains **no per-cell or per-rep measurement event**, and [JOB_STARTED] is captured and flushed
 * *before* the runner begins, which leaves the SDK's queue empty entering the measurement window. That flush lands during model load, seconds long,
 * not during a timed measurement.
 *
 * The one event that fires while a run is in flight is [RESULTS_SUBMITTED], from the per-cell upload `JobRunner` performs between cells (PIP-358,
 * matching iOS). It sits next to an HTTP upload that path is doing anyway, so it adds no network work that wasn't already there, but it does leave a
 * queued event the SDK's periodic timer could POST during the next cell. `ResultSubmissionService.submit` therefore flushes immediately after
 * capturing it, restoring the empty-queue invariant before the next measurement. **Any capture site added between the start of a run and its end
 * inherits that obligation.**
 *
 * This is why there is still no event-buffering machinery here: nothing is produced during a timed measurement itself, so there is nothing to buffer.
 */
object AnalyticsEvents {
  /** App reached a usable state. Carries the device/OS descriptors that let funnels be split by hardware. */
  const val APP_LAUNCHED = "app_launched"

  /** Device registration with the management server succeeded; the client id becomes the distinct id from here on. */
  const val DEVICE_REGISTERED = "device_registered"

  /** Device registration failed. Carries a coarse error kind only, never the server's message, which can contain the contact email. */
  const val DEVICE_REGISTRATION_FAILED = "device_registration_failed"

  const val MODEL_DOWNLOAD_STARTED = "model_download_started"
  const val MODEL_DOWNLOAD_COMPLETED = "model_download_completed"
  const val MODEL_DOWNLOAD_FAILED = "model_download_failed"

  /** A benchmark run began (new job, resume, retry-failed, or re-run of selected cells; [SOURCE] distinguishes them). */
  const val JOB_STARTED = "job_started"

  /** A benchmark run ended, whether it finished its cells, was cancelled, or died. [OUTCOME] says which. */
  const val JOB_COMPLETED = "job_completed"

  /** Results for a job were pushed to the management server. */
  const val RESULTS_SUBMITTED = "results_submitted"

  /** In-app feedback was sent (the Sentry user-feedback flow). Carries the category only, never the free-text body. */
  const val FEEDBACK_SUBMITTED = "feedback_submitted"

  // Property keys. Shared with iOS for the same reason the event names are.
  const val PLATFORM = "platform"
  const val APP_VERSION = "app_version"
  const val APP_ENVIRONMENT = "app_environment"
  const val OS_VERSION = "os_version"
  const val DEVICE_MODEL = "device_model"
  const val CHIP = "chip"
  const val FORM_FACTOR = "form_factor"
  const val JOB_ID = "job_id"

  /** The model's file name (e.g. `LFM2-1.2B-Q4_K_M.gguf`), a catalog identifier, not user-authored text. */
  const val MODEL_ID = "model_id"

  const val SIZE_BYTES = "size_bytes"

  /** Cells a run was scheduled to execute. Run-relative: a resume reports the cells it picked up, not the whole manifest. */
  const val CELL_COUNT = "cell_count"

  const val CELLS_COMPLETED = "cells_completed"

  /**
   * Results accepted by the management server in one [RESULTS_SUBMITTED] pass.
   *
   * A key of its own rather than [CELL_COUNT]: that one means "cells this run scheduled", and reusing it here would give a single property two
   * meanings, making any aggregate across the two events wrong.
   */
  const val SUBMITTED_COUNT = "submitted_count"
  const val DURATION_MS = "duration_ms"
  const val OUTCOME = "outcome"
  const val SOURCE = "source"
  const val ERROR_KIND = "error_kind"
  const val CATEGORY = "category"
  const val STATUS = "status"
  const val OK = "ok"

  /** [OUTCOME] values for [JOB_COMPLETED]. */
  const val OUTCOME_FINISHED = "finished"
  const val OUTCOME_CANCELLED = "cancelled"
  const val OUTCOME_FAILED = "failed"

  /**
   * Coarse, non-identifying label for a failure. Deliberately the exception's simple class name and nothing else: messages from this app's network
   * and registration paths embed the management-server URL and the contact email, and neither belongs in analytics.
   */
  fun errorKind(error: Throwable): String = error.javaClass.simpleName.ifBlank { "Throwable" }
}

/**
 * The analytics opt-out flag, and the authority on it.
 *
 * ### Why this exists rather than reusing the SDK's own persisted copy
 *
 * PostHog persists opt-out itself and `PostHog.setup` looks it back up (key `opt-out`), but on posthog-android 3.19.0 that lookup **does not work**.
 * Measured on a Pixel 10 Pro Fold / Android 17: with `opt-out=true` sitting in the SDK's own `posthog-<key>.xml`, `PostHog.isOptOut()` reads `false`
 * immediately after `setup` returns, and the next `app_launched` is captured and accepted by the ingest endpoint. Its `getValue(OPT_OUT, default)`
 * hands back the default instead of the stored value, so the restore is a no-op, while `distinctId` from that same file restores correctly.
 *
 * A value assigned to `PostHogAndroidConfig.optOut` *before* `setup`, on the other hand, survives it and suppresses every capture in the process
 * (verified: zero events queued). So this store owns the flag and seeds the config, which makes the behaviour ours rather than the SDK's to get
 * right.
 *
 * ### Why [SharedPreferences] rather than [AppSettingsStore]
 *
 * This is the one setting that must be readable **synchronously**: `PipetteApp.onCreate` builds the SDK and captures `app_launched` in a single
 * non-suspending block, so a DataStore read (suspend-only) could not complete in time and an opted-out device would leak one event per cold start.
 * `SharedPreferences.getBoolean` blocks on the file's first load and is safe here for that reason.
 */
internal object AnalyticsOptOutStore {
  private const val PREFS_NAME = "pipette_analytics"
  private const val KEY_OPT_OUT = "analytics_opt_out"

  /** False (collecting) unless the user has turned analytics off, matching the SDK's own default. */
  fun read(context: Context): Boolean = runCatching { prefs(context).getBoolean(KEY_OPT_OUT, false) }.getOrDefault(false)

  fun write(context: Context, optedOut: Boolean) {
    runCatching { prefs(context).edit().putBoolean(KEY_OPT_OUT, optedOut).apply() }
  }

  private fun prefs(context: Context) = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
}

/**
 * PostHog-backed [Analytics].
 *
 * Build it with [create], which returns [NoOpAnalytics] unless a real project is configured, so an unconfigured build (or the `:benchmark` process)
 * never initializes the SDK.
 */
class PostHogAnalytics private constructor(private val appContext: Context) : Analytics {
  override val isAvailable = true

  override fun capture(event: String, properties: Map<String, Any?>) {
    // Never let telemetry break a run: PostHog queues to disk and can throw on a full/corrupt queue.
    runCatching { PostHog.capture(event, properties = properties.dropNullValues()) }
  }

  override fun identify(distinctId: String) {
    runCatching { PostHog.identify(distinctId) }
  }

  override fun flush() {
    runCatching { PostHog.flush() }
  }

  /** Reads [AnalyticsOptOutStore], not `PostHog.isOptOut()`; see that store for why the SDK's own copy is not trustworthy across a launch. */
  override fun isOptedOut(): Boolean = AnalyticsOptOutStore.read(appContext)

  override fun setOptedOut(optedOut: Boolean) {
    // Persist first: this is what the next launch seeds the config from, and it must be durable
    // even if the SDK call below throws. Then tell the live SDK so the current process stops (or
    // resumes) immediately rather than at the next launch.
    AnalyticsOptOutStore.write(appContext, optedOut)
    runCatching { if (optedOut) PostHog.optOut() else PostHog.optIn() }
  }

  companion object {
    /**
     * Initialize PostHog and return the seam, or [NoOpAnalytics] when no project is configured.
     *
     * **Main process only.** `PipetteApp` calls this behind its main-process guard: the isolated `:benchmark` process must not load the SDK, because
     * anything resident there inflates the `max_memory_usage` that process exists to measure.
     */
    fun create(context: Context): Analytics {
      if (!PostHogConfiguration.isComplete) return NoOpAnalytics

      // DEPRECATION: `remoteConfig`, below. The annotation has to sit here because a Kotlin assignment
      // is a statement, not an expression, so it can't carry one of its own.
      @Suppress("DEPRECATION")
      val config =
        PostHogAndroidConfig(apiKey = PostHogConfiguration.apiKey, host = PostHogConfiguration.host).apply {
          // Manual events only. Turning these off ALSO prevents PostHogActivityLifecycleCallbackIntegration
          // from being installed at all: it is gated on `captureDeepLinks || captureScreenViews ||
          // sessionReplay || capturePushNotificationOpened`, and that last term is new in 3.58.0.
          captureScreenViews = false
          captureDeepLinks = false
          captureApplicationLifecycleEvents = false
          sessionReplay = false
          // Both new in 3.58.0 and both default TRUE. `capturePushNotificationOpened` is what puts the
          // activity lifecycle integration above back in play, so the four flags before it are not
          // sufficient on their own any more. `capturePushNotificationSubscriptions` fetches an FCM token
          // on a background executor at startup and registers it with PostHog when Firebase Messaging is
          // on the classpath: this app sends no push notifications, and a push token is a device
          // identifier that neither the privacy declaration nor the taxonomy accounts for.
          capturePushNotificationOpened = false
          capturePushNotificationSubscriptions = false
          // NOT redundant with `sessionReplay = false`. PostHogLogCatIntegration.install() gates on
          // sessionReplayConfig.captureLogcat (which defaults to TRUE) and never consults `sessionReplay`,
          // so leaving it alone spawns a `logcat` subprocess plus a reader thread that live for the whole
          // process, exactly the background work that would perturb a benchmark. Still true in
          // posthog-android 3.58.0; re-check on every version bump.
          sessionReplayConfig.captureLogcat = false
          // Default is already false on Android (unlike posthog-ios, where it is true), but pinned
          // because this is the flag that installs PostHogSurveysIntegration, and a survey rendering
          // itself over a running benchmark is exactly the kind of surprise this config exists to rule
          // out. The iOS client sets the same line.
          surveys = false
          // Sentry is the only crash handler in this app. Already the default, pinned because the
          // failure mode is bad and silent: PostHogErrorTrackingAutoCaptureIntegration links itself into
          // the Thread.UncaughtExceptionHandler chain when this is on, and two reporters fighting over
          // that chain is how crash reports go missing. Same reasoning as iOS's errorTrackingConfig.
          errorTrackingConfig.autoCapture = false
          // Feature flags are unused, so don't evaluate them or tag events with them.
          preloadFeatureFlags = false
          sendFeatureFlagEvent = false
          // New in 3.58.0, default true: attaches $app_version / $app_build / $app_namespace / $os_name /
          // $os_version / $device_type / $lib / $lib_version to the *person* profile so feature flags can
          // be evaluated against them locally. Nothing here reads flags, so this is only extra data on a
          // person record, and this app's whole position is that PostHog gets an opaque device id and the
          // properties the taxonomy names, not a growing profile assembled by the SDK.
          setDefaultPersonProperties = false
          // Separate switch, and turning the flags off does NOT imply it: with `remoteConfig` left
          // at its default the SDK GETs `<host>/array/<key>/config` on every launch, which was
          // observed happening *even while opted out*. A privacy toggle that still phones home is
          // not much of a toggle. Nothing here consumes remote config (no flags, no surveys, no
          // session replay), so switching it off costs nothing and removes a launch network call.
          //
          // Deprecated since 3.58.0 as "now always enabled", which is wrong as written: `PostHog.setup`
          // still branches on it (`when { config.remoteConfig -> loadRemoteConfigRequest(...);
          // config.preloadFeatureFlags -> ... }`), so false plus no flag preloading still means no
          // request. Verified in the 3.58.0 source and on device. Suppressed rather than dropped because
          // dropping it would silently reinstate a network call on every launch, including opted-out
          // ones. If a later version really does remove the branch, Android inherits the iOS behaviour
          // and the docs need updating to match.
          remoteConfig = false
          debug = BuildConfig.DEBUG
          // Seed the opt-out BEFORE setup, from our own store. See [AnalyticsOptOutStore] for why
          // the SDK's persisted copy can't be relied on. A value set here survives `setup` and
          // suppresses every capture in the process, including the `app_launched` that
          // `PipetteApp.onCreate` fires moments later.
          optOut = AnalyticsOptOutStore.read(context)
        }
      // `setup` creates the on-disk event queue, so it is the call here most able to throw (full or
      // unwritable storage), and it runs from `PipetteApp.onCreate`, where an escape would take the
      // app down at launch. Analytics may never do that, so a failed init degrades to NoOpAnalytics.
      val started = runCatching { PostHogAndroid.setup(context.applicationContext, config) }.isSuccess
      if (started) {
        // Super properties, attached to every event, so funnels can be split by platform and dev
        // traffic excluded. The PostHog analogue of the per-build-type `io.sentry.environment`.
        runCatching {
          PostHog.register(AnalyticsEvents.PLATFORM, "android")
          PostHog.register(AnalyticsEvents.APP_ENVIRONMENT, if (BuildConfig.DEBUG) "debug" else "production")
          PostHog.register(AnalyticsEvents.APP_VERSION, BuildConfig.VERSION_NAME)
        }
      }
      return if (started) PostHogAnalytics(context.applicationContext) else NoOpAnalytics
    }
  }
}

/** PostHog's `capture` takes `Map<String, Any>?`; drop nulls rather than making every call site pre-filter its optional properties. */
private fun Map<String, Any?>.dropNullValues(): Map<String, Any> = mapNotNull { (key, value) -> value?.let { key to it } }.toMap()
