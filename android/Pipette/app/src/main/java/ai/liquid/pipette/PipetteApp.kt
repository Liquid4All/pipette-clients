package ai.liquid.pipette

import android.app.Application

/**
 * Application entry point that owns the app-scoped [AppContainer]. The container (and therefore the benchmark engine + running job) outlives any
 * single Activity instance, so configuration changes don't disturb an in-flight run.
 *
 * The container is built **only in the main process**. `Application.onCreate` runs in every process — including the isolated `:benchmark` process —
 * but [AppContainer]'s construction eagerly builds `DownloadCoordinator`, whose init calls `WorkManager.getInstance()`. WorkManager auto-initializes
 * through `androidx.startup`'s `InitializationProvider`, and that provider (declared with no `android:process`) runs **only in the main process** —
 * so in `:benchmark` WorkManager is never initialized and `getInstance()` throws, crashing the process during `onCreate` before it can serve the
 * benchmark AIDL binder (surfacing to the UI as "Could not connect to :benchmark service"). The benchmark service builds its own engine and never
 * reads [container], so skipping it in `:benchmark` is correct and keeps that process lean (its `max_memory_usage` reflects only the model).
 *
 * Guarding here also keeps the Clerk SDK (plus its Compose / OkHttp / Credential-Manager stack) out of `:benchmark`. This class references **no Clerk
 * types directly** — the actual init lives in [ClerkBootstrap], only class-loaded when [ClerkBootstrap.create] is invoked (main process only). So in
 * `:benchmark` the `com.clerk.*` classes are never loaded/verified, not just never initialized.
 */
class PipetteApp : Application() {
  // @Volatile: written on the main thread in onCreate, read off-main via containerOrNull (e.g. a WorkManager worker). Makes that publication correct
  // by construction rather than relying on incidental framework happens-before.
  @Volatile
  lateinit var container: AppContainer
    private set

  /**
   * The app-scoped container, or null in a process where it was never built — i.e. any non-main process (see [onCreate]).
   *
   * Read this (not [container]) from anything that can run off the main process, e.g. a WorkManager worker or a broadcast receiver: [container] is a
   * `lateinit var`, so a safe-cast chain like `(ctx as? PipetteApp)?.container` still throws [UninitializedPropertyAccessException] there — the `?.`
   * guards the failed cast, not the uninitialized property. `containerOrNull` makes such a `?.` chain genuinely null-safe.
   */
  val containerOrNull: AppContainer?
    get() = if (::container.isInitialized) container else null

  override fun onCreate() {
    super.onCreate()

    // Application.onCreate fires in every process. Only the main (UI) process
    // may build the container — see the class doc: AppContainer -> WorkManager
    // would crash the `:benchmark` process, which needs neither the container
    // nor Clerk. `container` stays unset there (nothing in `:benchmark` reads it).
    if (!isMainProcess()) {
      // Native crash capture for `:benchmark` (sentry-native inproc + disk
      // transport in the engine's own .so). No JVM Sentry SDK here — it would
      // skew `max_memory_usage`. See [BenchmarkCrashReporter].
      BenchmarkCrashReporter.init(this)
      return
    }

    // Build the auth seam only when a real publishable key is configured;
    // otherwise clerkAuth stays null and ClerkBootstrap/Clerk are never loaded.
    val clerkAuth: ClerkAuth? = if (ClerkConfiguration.isComplete) ClerkBootstrap.create(this) else null

    // Product analytics (main process only; see the class doc: the SDK must stay out of
    // `:benchmark`, whose whole purpose is an uncontaminated memory/latency measurement).
    // Returns NoOpAnalytics when no PostHog project is configured, so nothing is loaded.
    val analytics = PostHogAnalytics.create(this)

    container = AppContainer(this, clerkAuth, analytics)

    // Bind analytics to the device identity as early as possible so events from this launch
    // attribute to the right device rather than to a fresh anonymous id. Only meaningful once
    // the device has registered; before that the SDK's anonymous id applies.
    container.storage.loadRegistration()?.let { analytics.identify(it.clientId) }

    analytics.capture(
      AnalyticsEvents.APP_LAUNCHED,
      mapOf(
        AnalyticsEvents.OS_VERSION to DeviceInfo.osVersion(),
        AnalyticsEvents.DEVICE_MODEL to DeviceInfo.modelName(),
        AnalyticsEvents.CHIP to DeviceInfo.chipModel(),
        AnalyticsEvents.FORM_FACTOR to DeviceInfo.formFactor(this),
      ),
    )

    // Force DownloadCoordinator construction now (main process only) so its resume-on-launch
    // restore() runs at startup, as it did when the field was eager — before a download-notification
    // action can cold-start the process and access the coordinator first, racing the restore.
    container.downloadCoordinator

    // Ship any crash reports the `:benchmark` process left behind (a prior
    // session's native/JVM crash flushed to disk but not yet uploaded). Sentry
    // auto-inits via its ContentProvider before Application.onCreate, so its
    // outbox is ready here. Runs off-thread; no-op when the dir is empty.
    CrashEnvelopeForwarder.drain(this)
    // Report any OOM/ANR the system recorded for `:benchmark` since last launch
    // (no in-process handler can catch those). Off-thread, de-duped internally.
    BenchmarkExitReporter.reportNewExits(this)
  }

  /** True in the default (UI) process; false in `:benchmark`. minSdk 31 → [getProcessName] always available. */
  private fun isMainProcess(): Boolean = ProcessGate.isMainProcess(getProcessName(), packageName)
}
