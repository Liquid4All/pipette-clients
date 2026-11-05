import Foundation
import PostHog

/// The event taxonomy shared with Android.
///
/// Every name and property key is duplicated verbatim in Android's `AnalyticsEvents`, so both
/// platforms form one PostHog funnel. Change a name in both places.
///
/// Background work perturbs the throughput this app measures, so no event fires inside a timed
/// measurement: `jobStarted` is captured and flushed before the runner begins (the POST lands
/// during model load), and ``resultsSubmitted`` (the per-cell upload between cells, PIP-358)
/// flushes at once so the queue is empty before the next cell. A capture between cells must
/// flush the same way.
nonisolated enum AnalyticsEvents {
    /// Carries the device and OS descriptors that let funnels be split by hardware.
    static let appLaunched = "app_launched"

    /// The client id is the distinct id from here on.
    static let deviceRegistered = "device_registered"

    /// Carries a coarse error kind only, never the server's message, which can contain the contact
    /// email.
    static let deviceRegistrationFailed = "device_registration_failed"

    static let modelDownloadStarted = "model_download_started"
    static let modelDownloadCompleted = "model_download_completed"
    static let modelDownloadFailed = "model_download_failed"

    /// `source` says how the run was started (see ``RunSource``).
    static let jobStarted = "job_started"

    /// `outcome` says whether the run finished, was cancelled, or died.
    static let jobCompleted = "job_completed"

    static let resultsSubmitted = "results_submitted"

    /// Carries the category only, never the free-text body.
    static let feedbackSubmitted = "feedback_submitted"

    // Property keys, shared with Android like the event names.
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

    /// Results accepted by the management server in one ``resultsSubmitted`` pass. Not
    /// ``cellCount``: one property with two meanings would break any aggregate across the events.
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

    /// The error's type and nothing else: the messages (`localizedDescription`) on the network and
    /// registration paths embed the management-server URL and the contact email.
    static func errorKind(_ error: Error) -> String {
        String(describing: type(of: error))
    }
}

/// The `source` property on ``AnalyticsEvents/jobStarted``, so a resume or retry does not inflate
/// the "started" count. Raw values are a wire contract shared with Android's `RunSource.wireName`.
///
/// ``planner`` (the in-app `PlannerWorker` claim loop) has no Android counterpart yet; it is not
/// ``new`` because a server-claimed cell is not a user starting a job.
nonisolated enum RunSource: String {
    case new
    case resume
    case retryFailed = "retry_failed"
    case rerun
    case planner

    /// The source implied by re-running every cell currently in `target`. `JobDetailView` keeps
    /// its own mapping (it also has a hand-picked selection); a test pins that the two agree.
    static func resettingCells(withStatus target: CellRunStatus) -> RunSource {
        switch target {
        case .cancelled: .resume
        case .failed: .retryFailed
        default: .rerun
        }
    }
}

/// Which ``AnalyticsEvents/outcome`` a finished run reports; mirrors Android's `jobOutcome`
/// (which also has an `interrupted` outcome).
///
/// `cancelled` is needed because a run cancelled after its last cell is saved `.completed`. A
/// non-terminal manifest means the run itself died.
nonisolated func analyticsOutcome(cancelled: Bool, status: JobStatus) -> String {
    if cancelled || status == .paused || status == .cancelled {
        return AnalyticsEvents.outcomeCancelled
    }
    return status == .completed ? AnalyticsEvents.outcomeFinished : AnalyticsEvents.outcomeFailed
}

/// The only path to `PostHogSDK`, so tests and unconfigured builds get a sink with no disk queue
/// and no network. Callable from any thread and never throws: analytics must never take down a
/// benchmark run or a registration.
nonisolated protocol AnalyticsSink: Sendable {
    /// False for ``NoOpAnalytics``. `SettingsView` hides the opt-out toggle then, since it would
    /// control nothing.
    var isAvailable: Bool { get }

    func capture(_ event: String, _ properties: [String: Any?])
    func identify(_ distinctId: String)
    /// Send the queue now rather than on the SDK's timer; see ``AnalyticsEvents``.
    func flush()

    /// The user's Settings opt-out, backed by ``LocalStorage/analyticsOptOut`` (see there for why
    /// not the SDK's own persisted copy).
    var isOptedOut: Bool { get }

    /// Events already queued when this turns on are not recalled: PostHog's only queue-dropping
    /// call, `reset()`, also destroys the distinct id.
    func setOptedOut(_ optedOut: Bool)
}

/// For builds with no PostHog project configured, and for tests.
nonisolated struct NoOpAnalytics: AnalyticsSink {
    var isAvailable: Bool { false }
    func capture(_: String, _: [String: Any?]) {}
    func identify(_: String) {}
    func flush() {}
    /// True because nothing is collected, not because a user chose it; ``isAvailable`` tells the
    /// two apart.
    var isOptedOut: Bool { true }
    func setOptedOut(_: Bool) {}
}

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

    var isOptedOut: Bool { LocalStorage.analyticsOptOut }

    func setOptedOut(_ optedOut: Bool) {
        // Persisted for the next launch's config seed; the live SDK call applies it to this session.
        LocalStorage.analyticsOptOut = optedOut
        if optedOut {
            PostHogSDK.shared.optOut()
        } else {
            PostHogSDK.shared.optIn()
        }
    }
}

/// App-wide analytics entry point, the analogue of Android's `AppContainer.analytics`. A static
/// accessor because service-layer code like `register` cannot reach the SwiftUI environment.
nonisolated enum Analytics {
    private static let sinkLock = NSLock()
    nonisolated(unsafe) private static var _sink: AnalyticsSink = NoOpAnalytics()

    /// ``NoOpAnalytics`` until ``start()`` swaps in the real one.
    ///
    /// Locked because the one write is on the main actor at launch while reads come from any
    /// thread; that the write precedes every read is an argument, not a guarantee.
    static var sink: AnalyticsSink {
        sinkLock.withLock { _sink }
    }

    static var isAvailable: Bool {
        sink.isAvailable
    }

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

    /// Call once, early in launch and after Sentry. No-op without a configured project.
    /// `@MainActor` because it reads main-actor `PostHogConfiguration`; the rest stays
    /// `nonisolated` for the background executor a run finishes on.
    @MainActor
    static func start() {
        guard PostHogConfiguration.isComplete,
              let projectToken = PostHogConfiguration.projectToken,
              let host = PostHogConfiguration.host
        else { return }

        let config = PostHogConfig(projectToken: projectToken, host: host)
        // Manual events only; see AnalyticsEvents for why instrumentation is unacceptable here.
        config.captureScreenViews = false
        config.captureApplicationLifecycleEvents = false
        config.captureElementInteractions = false
        config.sessionReplay = false
        // Both default true and neither is implied by the switches above. Rage-click detection
        // hooks touch delivery for the whole process, the ambient work this app refuses; surveys
        // add an unused fetch. Re-check on every SDK bump.
        config.surveys = false
        config.rageClickConfig.enabled = false
        // Already the default, pinned because PostHog vendors PLCrashReporter and a second crash
        // reporter would fight Sentry for the signal handlers.
        config.errorTrackingConfig.autoCapture = false
        // Both default true since 3.69.0. Subscriptions swizzles the app delegate to send the APNs
        // token, a device identifier PrivacyInfo.xcprivacy does not declare; this app sends no pushes.
        config.capturePushNotificationSubscriptions = false
        config.capturePushNotificationOpened = false
        // Feature flags are unused; preloading them would fire a network request at launch.
        config.preloadFeatureFlags = false
        config.sendFeatureFlagEvent = false
        // Without it the SDK logs nothing, so a capture cannot be told from a suppressed one.
        #if DEBUG
            config.debug = true
        #endif
        // Seeded before setup so it already suppresses the `appLaunched` `PipetteApp` fires next.
        config.optOut = LocalStorage.analyticsOptOut
        // posthog-ios ignores `config.remoteConfig` (always enabled), so even an opted-out device
        // makes one launch GET carrying the public project token and no user data.
        PostHogSDK.shared.setup(config)

        // Attached to every event, so funnels split by platform and exclude dev traffic.
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
    /// PostHog's `capture` takes `[String: Any]`, so call sites can pass optionals.
    func droppingNilValues() -> [String: Any] {
        compactMapValues { $0 }
    }
}
