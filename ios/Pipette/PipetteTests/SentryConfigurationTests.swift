import Sentry
import Testing

@testable import Pipette

/// Sentry stays crash reporting only. The behaviour test is the one that matters across
/// SDK upgrades: the per-frame display link raced `exit` in headless runs (PIP-662), and
/// sentry-cocoa started it from an option this app never asked for.
@Suite(.serialized)
struct SentryConfigurationTests {
    @Test func configureTurnsTracingAndPrivacyFieldsOff() {
        let options = Sentry.Options()
        SentryConfiguration.configure(options, dsn: "https://key@sentry.invalid/0")

        #expect(!options.isTracingEnabled)
        #expect(!options.enableAutoPerformanceTracing)
        #expect(!options.enableAppHangTracking)
        #expect(!options.enableWatchdogTerminationTracking)
        #expect(!options.attachScreenshot)
        #expect(!options.reportAccessibilityIdentifier)
    }

    /// `.invalid` never resolves, so nothing leaves the simulator. On the main actor the
    /// SDK installs its integrations before `start` returns.
    @Test func aStartedSDKRunsNoFramesTracker() {
        let options = Sentry.Options()
        SentryConfiguration.configure(options, dsn: "https://key@sentry.invalid/0")
        SentrySDK.start(options: options)
        defer { SentrySDK.close() }

        #expect(SentrySDK.isEnabled)
        #expect(!PrivateSentrySDKOnly.isFramesTrackingRunning)
    }
}
