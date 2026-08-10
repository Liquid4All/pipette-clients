import Foundation
import Sentry

/// Sentry setup for the iOS client — the analogue of the Android app's Gradle/manifest
/// wiring. Crash/error reporting only: performance tracing is off (`tracesSampleRate = 0`)
/// AND the auto-instrumentation (swizzling-based perf tracing + the app-hang watchdog) is
/// disabled, because this app runs on-device LLM benchmarks and any background work or
/// instrumentation overhead would skew the measurements. This mirrors the Android config,
/// which disables the Sentry Gradle plugin's `tracingInstrumentation`.
///
/// The DSN is read from the `SentryDSN` Info.plist key (resolved like the Clerk keys). If
/// it's missing or still an unresolved `$(…)` build-setting placeholder, Sentry stays off
/// and the in-app feedback entry hides itself — the same DSN-gating idea as the web app's
/// `FEEDBACK_ENABLED` and the Android `Sentry.isEnabled()` check.
enum SentryConfiguration {
    /// Resolved once — the DSN is fixed for the lifetime of the process, so there's no
    /// reason to re-read and re-validate the Info.plist on every access.
    static let dsn: String? = Bundle.main.normalizedInfoString("SentryDSN")

    /// Whether crash reporting + the feedback feature are active. Reflects the SDK's actual
    /// running state (not merely "a DSN string is present") so the feedback entry can't show
    /// when `start()` failed to bring the SDK up — matching the Android `Sentry.isEnabled()`
    /// gate. Cheap: a bool read, no Info.plist lookup per call.
    static var isEnabled: Bool {
        SentrySDK.isEnabled
    }

    /// Start the SDK. Call once, as early as possible in app launch. No-op without a DSN.
    static func start() {
        guard let dsn else { return }
        SentrySDK.start { options in
            options.dsn = dsn
            // Crash/error reporting only — see the type doc for why tracing is disabled.
            options.tracesSampleRate = 0.0
            // Turn off the auto-instrumentation so no swizzling-based performance spans and
            // no main-thread app-hang watchdog run during a benchmark (the iOS analogue of
            // Android disabling the plugin's tracingInstrumentation). Crash + manual
            // error/feedback reporting is unaffected.
            options.enableAutoPerformanceTracing = false
            options.enableAppHangTracking = false
            // Screenshot attachment is deliberately OFF: Settings renders the signed-in
            // account's email and the feedback sheet pre-fills it, so a crash screenshot
            // could carry an address nobody chose to send. Assigned explicitly even though
            // false is the current SDK default, so an upgrade cannot quietly turn a privacy
            // control back on.
            options.attachScreenshot = false
            // The view hierarchy stays on: it carries no text, so it cannot leak the same
            // address. Per node Sentry emits the rendering system, the view's class name,
            // frame, alpha, visibility, the enclosing view controller's class name, and
            // children. Its one free-form field is `identifier`, copied from
            // accessibilityIdentifier, so that is switched off too: nothing sets an
            // identifier today, but one added for a UI test (or interpolated from data,
            // say "model_\(name)") would otherwise start riding along in every crash
            // report. Same split on Android.
            options.attachViewHierarchy = true
            options.reportAccessibilityIdentifier = false
            // Segregate non-release events from production dashboards/alerts. Explicit on
            // both branches so it matches the Android per-build-type `io.sentry.environment`.
            #if DEBUG
                options.environment = "debug"
            #else
                options.environment = "production"
            #endif
        }
        // Constant per-install context, set once so it rides along with every event
        // (crashes and feedback alike). Sentry already captures device/OS in its `contexts`,
        // but surfacing them as tags makes them filterable and matches the Android feedback
        // tag taxonomy (device / chip / os / app_version).
        SentrySDK.configureScope { scope in
            scope.setTag(value: DeviceProbe.detectDeviceName(), key: "device")
            scope.setTag(value: DeviceProbe.detectChipModel(), key: "chip")
            scope.setTag(value: "\(DeviceProbe.detectOsName()) \(DeviceProbe.detectOsVersion())", key: "os")
            scope.setTag(value: Bundle.main.appVersionDisplayString, key: "app_version")
        }
    }
}
