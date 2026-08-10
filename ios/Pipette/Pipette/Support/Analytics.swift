import Foundation
import PostHog

/// The shared Android + iOS event taxonomy.
///
/// **Every name and property key here is duplicated verbatim in the Android client's
/// `AnalyticsEvents`** so both platforms land in one PostHog project as a single funnel rather than
/// two dialects that have to be unioned in every query. Changing a name here means changing it
/// there.
///
/// ### Why events fire at boundaries, never inside a measurement
///
/// This app measures on-device LLM throughput, and background work perturbs those measurements,
/// the same reason `SentryConfiguration` disables tracing, auto-performance instrumentation and the
/// app-hang watchdog. So the taxonomy deliberately contains **no per-cell or per-rep measurement
/// event**, and `jobStarted` is captured and flushed *before* the runner begins, which leaves the
/// SDK's queue empty entering the measurement window. That flush lands during model load, seconds
/// long, not during a timed measurement.
///
/// The one event that can fire while a job is in flight is ``resultsSubmitted``, from the per-cell
/// upload `JobExecutor` performs between cells (PIP-358). It sits next to an HTTP upload that path
/// is doing anyway, so it adds no network work that wasn't already there, but it does leave a
/// queued event the SDK's periodic timer could flush during the next cell. Capture sites in that
/// position must flush immediately, restoring the empty-queue invariant before the next measurement.
///
/// This is why there is no event-buffering machinery here: no event is produced during a timed
/// measurement, so there is nothing to buffer.
nonisolated enum AnalyticsEvents {
    /// App reached a usable state. Carries the device/OS descriptors that let funnels be split by
    /// hardware.
    static let appLaunched = "app_launched"

    /// Device registration with the management server succeeded; the client id becomes the distinct
    /// id from here on.
    static let deviceRegistered = "device_registered"

    /// Device registration failed. Carries a coarse error kind only, never the server's message,
    /// which can contain the contact email.
    static let deviceRegistrationFailed = "device_registration_failed"

    static let modelDownloadStarted = "model_download_started"
    static let modelDownloadCompleted = "model_download_completed"
    static let modelDownloadFailed = "model_download_failed"

    /// A benchmark run began (new job, resume, retry-failed, or re-run of selected cells; `source`
    /// distinguishes them).
    static let jobStarted = "job_started"

    /// A benchmark run ended, whether it finished its cells, was cancelled, or died. `outcome` says
    /// which.
    static let jobCompleted = "job_completed"

    /// Results for a job were pushed to the management server.
    static let resultsSubmitted = "results_submitted"

    /// In-app feedback was sent (the Sentry user-feedback flow). Carries the category only, never
    /// the free-text body.
    static let feedbackSubmitted = "feedback_submitted"

    // Property keys. Shared with Android for the same reason the event names are.
    static let platform = "platform"
    static let appVersion = "app_version"
    static let appEnvironment = "app_environment"
    static let osVersion = "os_version"
    static let deviceModel = "device_model"
    static let chip = "chip"
    static let formFactor = "form_factor"
    static let jobId = "job_id"

    /// The model's file name (e.g. `LFM2-1.2B-Q4_K_M.gguf`), a catalog identifier, not
    /// user-authored text.
    static let modelId = "model_id"

    static let sizeBytes = "size_bytes"

    /// Cells a run was scheduled to execute. Run-relative: a resume reports the cells it picked up,
    /// not the whole manifest.
    static let cellCount = "cell_count"

    static let cellsCompleted = "cells_completed"

    /// Results accepted by the management server in one ``resultsSubmitted`` pass.
    ///
    /// A key of its own rather than ``cellCount``: that one means "cells this run scheduled", and
    /// reusing it here would give a single property two meanings, making any aggregate across the
    /// two events wrong.
    static let submittedCount = "submitted_count"
    static let durationMs = "duration_ms"
    static let outcome = "outcome"
    static let source = "source"
    static let errorKind = "error_kind"
    static let category = "category"
    static let status = "status"
    static let ok = "ok"

    /// `outcome` values for ``jobCompleted``.
    static let outcomeFinished = "finished"
    static let outcomeCancelled = "cancelled"
    static let outcomeFailed = "failed"

    /// Coarse, non-identifying label for a failure. Deliberately the error's type and nothing else:
    /// messages from this app's network and registration paths embed the management-server URL and
    /// the contact email, and neither belongs in analytics.
    ///
    /// Note this must never be `error.localizedDescription`, which is exactly that message.
    static func errorKind(_ error: Error) -> String {
        String(describing: type(of: error))
    }
}

/// How a benchmark run was started, reported as the `source` property on
/// ``AnalyticsEvents/jobStarted``. Distinguishing these is what makes the funnel readable: a resume
/// or a retry is not a fresh user intent to run a job, and lumping them together would inflate the
/// "started" count.
///
/// The raw values are the wire contract shared with Android's `RunSource.wireName`.
///
/// ``planner`` has no Android counterpart today: only this client has an in-app `PlannerWorker`
/// claim loop (Android's planner worker is CLI-side). It is declared here rather than folded into
/// ``new`` because a server-claimed cell is not a user starting a job, and counting it as one would
/// make the funnel's "started" figure meaningless on planner-enabled devices. If Android ever gains
/// an in-app claim loop it must reuse this exact string.
nonisolated enum RunSource: String {
    case new
    case resume
    case retryFailed = "retry_failed"
    case rerun
    case planner

    /// The source implied by re-running every cell currently in `target`.
    ///
    /// Used by `JobLauncher.rerun`, which backs both `job run --scope …` and the deep-link resume:
    /// without this it reported a flat `rerun`, so `--scope cancelled` filed under a different
    /// intent than the identical action from `JobDetailView`. That view keeps its own mapping (its
    /// scope enum carries a hand-picked `.selected(ids)` case that has no `CellRunStatus` to map
    /// from), but the two agree case for case, and
    /// `AnalyticsEventsTests.testRunSourceDerivedFromResetTargetMatchesJobDetailView` pins that they do.
    static func resettingCells(withStatus target: CellRunStatus) -> RunSource {
        switch target {
        case .cancelled: .resume
        case .failed: .retryFailed
        default: .rerun
        }
    }
}

/// Which ``AnalyticsEvents/outcome`` a finished run should report. The twin of Android's
/// `jobOutcome`, and it must stay in step with it.
///
/// A cancel lands as ``AnalyticsEvents/outcomeCancelled`` whether it was observed through the live
/// flag or only through the persisted `.paused`/`.cancelled` status: a run cancelled after its last
/// cell finishes has nothing left pending and is saved `.completed`, so the flag is what
/// distinguishes it. A manifest left non-terminal means the run itself died.
nonisolated func analyticsOutcome(cancelled: Bool, status: JobStatus) -> String {
    if cancelled || status == .paused || status == .cancelled {
        return AnalyticsEvents.outcomeCancelled
    }
    return status == .completed ? AnalyticsEvents.outcomeFinished : AnalyticsEvents.outcomeFailed
}

/// Product-analytics seam. Everything in the app captures through this rather than calling
/// `PostHogSDK` directly, which keeps the SDK at one edge of the codebase and lets tests and
/// unconfigured builds use a sink that touches no disk queue and no network.
///
/// Every method is safe to call from any thread and never throws: analytics is advisory telemetry
/// and may never take down a benchmark run or a registration attempt.
nonisolated protocol AnalyticsSink: Sendable {
    /// Whether this sink sends anything at all. False for ``NoOpAnalytics`` (an unconfigured build
    /// or a test), which is what `SettingsView` gates its opt-out control on, the same way the
    /// feedback card is gated on `SentryConfiguration.isEnabled`: a toggle over a sink that collects
    /// nothing would be a control that does nothing.
    var isAvailable: Bool { get }

    func capture(_ event: String, _ properties: [String: Any?])
    func identify(_ distinctId: String)
    /// Send anything queued now, rather than on the SDK's periodic timer. Called immediately before
    /// a benchmark run begins; see ``AnalyticsEvents`` for the rationale.
    func flush()

    /// Whether the user has turned analytics collection off in Settings. While true, ``capture(_:_:)``
    /// and ``identify(_:)`` do nothing.
    ///
    /// Backed by ``LocalStorage/analyticsOptOut``, **not** by the SDK's own persisted copy, which
    /// `Analytics.start()` seeds `PostHogConfig.optOut` from before `setup`. See that property for
    /// why the SDK's restore is not trusted.
    var isOptedOut: Bool { get }

    /// Turn collection off (`optedOut` true) or back on. Persisted by ``LocalStorage/analyticsOptOut``
    /// and re-applied at the next launch, so this is the whole of the setting; see ``isOptedOut``.
    ///
    /// Events already queued when this flips to true are not recalled: PostHog's only queue-dropping
    /// call is `reset()`, which also destroys the distinct id, and the queue can hold at most what
    /// was captured while the user was still opted in.
    func setOptedOut(_ optedOut: Bool)
}

/// Analytics sink for builds with no PostHog project configured, and for tests. Does nothing, on
/// purpose.
nonisolated struct NoOpAnalytics: AnalyticsSink {
    var isAvailable: Bool { false }
    func capture(_: String, _: [String: Any?]) {}
    func identify(_: String) {}
    func flush() {}
    /// True because it describes what this sink does (nothing is collected), not because a user
    /// chose it. ``isAvailable`` is what tells the two apart.
    var isOptedOut: Bool { true }
    func setOptedOut(_: Bool) {}
}

/// PostHog-backed ``AnalyticsSink``.
nonisolated struct PostHogAnalytics: AnalyticsSink {
    var isAvailable: Bool { true }

    func capture(_ event: String, _ properties: [String: Any?]) {
        PostHogSDK.shared.capture(event, properties: properties.droppingNilValues())
    }

    func identify(_ distinctId: String) {
        PostHogSDK.shared.identify(distinctId)
    }

    func flush() {
        PostHogSDK.shared.flush()
    }

    /// Reads ``LocalStorage/analyticsOptOut``, not `PostHogSDK.isOptOut()`; see that property for
    /// why the SDK's own persisted copy is not trusted across a launch.
    var isOptedOut: Bool { LocalStorage.analyticsOptOut }

    func setOptedOut(_ optedOut: Bool) {
        // Persist first: this is what the next launch seeds the config from, and it has to be
        // durable independently of the SDK. Then tell the live SDK so the current session stops
        // (or resumes) immediately rather than at the next launch.
        LocalStorage.analyticsOptOut = optedOut
        if optedOut {
            PostHogSDK.shared.optOut()
        } else {
            PostHogSDK.shared.optIn()
        }
    }
}

/// App-wide analytics entry point, the iOS analogue of the `analytics` field Android hangs off its
/// `AppContainer`. This client has no equivalent container (its stores are injected through the
/// SwiftUI environment, which service-layer types like `RegistrationService` can't reach), so the
/// seam is resolved through this one accessor instead of being threaded through every initializer.
nonisolated enum Analytics {
    private static let sinkLock = NSLock()
    nonisolated(unsafe) private static var _sink: AnalyticsSink = NoOpAnalytics()

    /// The live sink. ``NoOpAnalytics`` until ``start()`` swaps in the real one, so anything that
    /// captures before (or without) initialization is a no-op rather than a crash.
    ///
    /// Lock-guarded rather than `nonisolated(unsafe)`: the write happens on the main actor during
    /// launch while the reads come from arbitrary threads (a run finishes on a background executor,
    /// downloads complete on the URLSession delegate queue). In practice the single write precedes
    /// every read, but that is an argument rather than a guarantee, and `nonisolated(unsafe)` would
    /// simply switch the compiler's checking off. The lock is uncontended and analytics is not a hot
    /// path, so it costs nothing worth measuring.
    static var sink: AnalyticsSink {
        sinkLock.withLock { _sink }
    }

    /// Whether a real sink is wired. See ``AnalyticsSink/isAvailable``.
    static var isAvailable: Bool {
        sink.isAvailable
    }

    /// The user's Settings choice. See ``AnalyticsSink/isOptedOut``.
    static var isOptedOut: Bool {
        sink.isOptedOut
    }

    static func setOptedOut(_ optedOut: Bool) {
        sink.setOptedOut(optedOut)
    }

    static func capture(_ event: String, _ properties: [String: Any?] = [:]) {
        sink.capture(event, properties)
    }

    static func identify(_ distinctId: String) {
        sink.identify(distinctId)
    }

    static func flush() {
        sink.flush()
    }

    /// Start the SDK. Call once, as early as possible in app launch, after Sentry. No-op without a
    /// configured project, in which case ``sink`` stays ``NoOpAnalytics``.
    ///
    /// `@MainActor` while the rest of this type is `nonisolated`: setup reads `PostHogConfiguration`
    /// and the bundle version, which are main-actor isolated like the rest of the app, whereas
    /// ``capture(_:_:)`` must stay callable from the background executor a benchmark run finishes on.
    @MainActor
    static func start() {
        guard PostHogConfiguration.isComplete,
              let projectToken = PostHogConfiguration.projectToken,
              let host = PostHogConfiguration.host
        else { return }

        let config = PostHogConfig(projectToken: projectToken, host: host)
        // Manual events only. See AnalyticsEvents for why instrumentation overhead is
        // unacceptable in this app.
        config.captureScreenViews = false
        config.captureApplicationLifecycleEvents = false
        config.captureElementInteractions = false
        config.sessionReplay = false
        // Both default to TRUE and neither is implied by the switches above: with them left alone
        // the console shows `PostHogSurveyIntegration installed` and `PostHogRageClickIntegration
        // installed` at every launch. Rage-click detection hooks touch delivery for the life of the
        // process to watch for rapid taps, which is precisely the ambient background work this app
        // refuses elsewhere; surveys add a fetch and a presentation path nothing here uses. This is
        // the iOS twin of Android's `sessionReplayConfig.captureLogcat`: a default-on integration
        // that no obvious flag turns off. Re-check on every SDK bump.
        config.surveys = false
        config.rageClickConfig.enabled = false
        // Already the SDK default, pinned explicitly because the consequence of it flipping is
        // severe and silent: PostHog vendors PLCrashReporter, and a second crash reporter
        // installing signal handlers would fight Sentry for them. SentryConfiguration must stay
        // the single handler on this client.
        config.errorTrackingConfig.autoCapture = false
        // Both new in 3.69.0 and both default TRUE, the exact twins of the two lines added to the
        // Android config in the same bump. `capturePushNotificationSubscriptions` swizzles the app
        // delegate to register this device's APNs token with PostHog for Workflows delivery; this app
        // sends no push notifications, and a push token is a device identifier that neither
        // PrivacyInfo.xcprivacy nor the event taxonomy accounts for. `capturePushNotificationOpened`
        // captures a `$push_notification_opened` event that could never fire here, and is off for the
        // same reason the other autocapture switches are.
        config.capturePushNotificationSubscriptions = false
        config.capturePushNotificationOpened = false
        // Feature flags are unused; preloading them would fire a network request at launch.
        config.preloadFeatureFlags = false
        config.sendFeatureFlagEvent = false
        // Parity with Android's `debug = BuildConfig.DEBUG`. Without it the SDK is completely
        // silent, which is what made verifying the opt-out on a simulator impossible until now:
        // there was no way to see whether an event had been captured or suppressed.
        #if DEBUG
            config.debug = true
        #endif
        // Seed the opt-out BEFORE setup, from our own store. A value set here holds through setup
        // and suppresses every capture in the session, including the ``AnalyticsEvents/appLaunched``
        // that `PipetteApp` fires moments later, whereas the SDK's own restore of its persisted
        // copy was measured failing outright on Android. See ``LocalStorage/analyticsOptOut``.
        config.optOut = LocalStorage.analyticsOptOut
        // Note: unlike Android, this client cannot suppress the SDK's launch-time remote-config
        // GET: posthog-ios deprecated `config.remoteConfig` and ignores it ("is now always
        // enabled"). So an opted-out iOS device still makes that one request, carrying the public
        // project token and no user data. Android sets `remoteConfig = false` and is fully silent.
        PostHogSDK.shared.setup(config)

        // Super properties, attached to every event, so funnels can be split by platform and dev
        // traffic excluded. The PostHog analogue of Sentry's per-build `environment`.
        PostHogSDK.shared.register([
            AnalyticsEvents.platform: "ios",
            AnalyticsEvents.appEnvironment: environmentName,
            AnalyticsEvents.appVersion: Bundle.main.appVersionAnalyticsString,
        ])
        sinkLock.withLock { _sink = PostHogAnalytics() }
    }

    private static var environmentName: String {
        #if DEBUG
            return "debug"
        #else
            return "production"
        #endif
    }
}

nonisolated private extension [String: Any?] {
    /// PostHog's `capture` takes `[String: Any]`; drop nils rather than making every call site
    /// pre-filter its optional properties.
    func droppingNilValues() -> [String: Any] {
        compactMapValues { $0 }
    }
}
