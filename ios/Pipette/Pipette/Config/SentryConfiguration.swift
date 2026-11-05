import Foundation
import Sentry

/// Sentry, for crash and error reporting only: tracing and auto-instrumentation are off
/// because any background work would skew on-device benchmarks (Android disables the
/// plugin's `tracingInstrumentation` for the same reason). Without a `SentryDSN`, Sentry
/// stays off and the in-app feedback entry hides itself.
enum SentryConfiguration {
    static let dsn: String? = Bundle.main.normalizedInfoString("SentryDSN")

    /// The SDK's running state, not "a DSN is present", so the feedback entry hides when
    /// `start()` failed.
    static var isEnabled: Bool {
        SentrySDK.isEnabled
    }

    /// Call once, as early as possible in app launch.
    static func start() {
        guard let dsn else { return }
        SentrySDK.start { options in configure(options, dsn: dsn) }
        // Sentry captures device and OS in `contexts`; as tags they are filterable and
        // match Android's feedback tags.
        SentrySDK.configureScope { scope in
            scope.setTag(value: detectDeviceName(), key: "device")
            scope.setTag(value: detectChipModel(), key: "chip")
            scope.setTag(value: "\(detectOsName()) \(detectOsVersion())", key: "os")
            scope.setTag(value: Bundle.main.appVersionDisplayString, key: "app_version")
        }
    }

    /// Separate so a test can start the SDK with exactly these settings.
    static func configure(_ options: Sentry.Options, dsn: String) {
        options.dsn = dsn
        options.tracesSampleRate = 0.0
        options.enableAutoPerformanceTracing = false
        options.enableAppHangTracking = false
        // Watchdog-termination tracking's display link runs even with the two above off, and
        // raced `exit` in headless runs (PIP-662). Off, Sentry reports no jetsam or watchdog kills.
        options.enableWatchdogTerminationTracking = false
        // Off: a screenshot could carry the signed-in email Settings shows. Explicit though
        // it is the SDK default, so an upgrade cannot turn a privacy control back on.
        options.attachScreenshot = false
        // The view hierarchy carries no text. Its one free-form field, the accessibility
        // identifier, is off so one interpolated from data cannot leak. Same on Android.
        options.attachViewHierarchy = true
        options.reportAccessibilityIdentifier = false
        // Keeps non-release events off production dashboards, as Android's
        // per-build-type `io.sentry.environment` does.
        #if DEBUG
            options.environment = "debug"
        #else
            options.environment = "production"
        #endif
    }
}
