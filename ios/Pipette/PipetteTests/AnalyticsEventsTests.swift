import XCTest
@testable import Pipette

final class AnalyticsEventsTests: XCTestCase {
    /// Tripwire pinning the analytics wire contract. The same names are hand-duplicated in the
    /// Android `AnalyticsEvents`, and both platforms report into one PostHog project, so a rename
    /// on one side alone doesn't break a build, it silently splits a funnel into two events that
    /// look unrelated in every query. Like `FeedbackCategoryTests`, this can't enforce true parity
    /// (each platform pins its own copy); it catches an accidental change here, and makes an
    /// intentional one the reminder to update Android.
    func testEventNamesMatchCrossPlatformContract() {
        XCTAssertEqual(AnalyticsEvents.appLaunched, "app_launched")
        XCTAssertEqual(AnalyticsEvents.deviceRegistered, "device_registered")
        XCTAssertEqual(AnalyticsEvents.deviceRegistrationFailed, "device_registration_failed")
        XCTAssertEqual(AnalyticsEvents.modelDownloadStarted, "model_download_started")
        XCTAssertEqual(AnalyticsEvents.modelDownloadCompleted, "model_download_completed")
        XCTAssertEqual(AnalyticsEvents.modelDownloadFailed, "model_download_failed")
        XCTAssertEqual(AnalyticsEvents.jobStarted, "job_started")
        XCTAssertEqual(AnalyticsEvents.jobCompleted, "job_completed")
        XCTAssertEqual(AnalyticsEvents.resultsSubmitted, "results_submitted")
        XCTAssertEqual(AnalyticsEvents.feedbackSubmitted, "feedback_submitted")
    }

    /// Property keys split a funnel just as event names do. `submitted_count` is deliberately not
    /// `cell_count`: that one means "cells this run scheduled" on `job_started`/`job_completed`, and
    /// overloading it on `results_submitted` would make any aggregate across the two events wrong.
    func testPropertyKeysMatchCrossPlatformContract() {
        XCTAssertEqual(AnalyticsEvents.cellCount, "cell_count")
        XCTAssertEqual(AnalyticsEvents.cellsCompleted, "cells_completed")
        XCTAssertEqual(AnalyticsEvents.submittedCount, "submitted_count")
        XCTAssertEqual(AnalyticsEvents.durationMs, "duration_ms")
        XCTAssertEqual(AnalyticsEvents.sizeBytes, "size_bytes")
        XCTAssertEqual(AnalyticsEvents.modelId, "model_id")
        XCTAssertEqual(AnalyticsEvents.errorKind, "error_kind")
        XCTAssertEqual(AnalyticsEvents.osVersion, "os_version")
        XCTAssertEqual(AnalyticsEvents.appVersion, "app_version")
    }

    /// The run-source values travel as the `source` property. `planner` is intentionally iOS-only
    /// today (Android's planner worker is CLI-side, not in-app); the rest must match Android's
    /// `RunSource.wireName`.
    func testRunSourceRawValuesMatchCrossPlatformContract() {
        XCTAssertEqual(RunSource.new.rawValue, "new")
        XCTAssertEqual(RunSource.resume.rawValue, "resume")
        XCTAssertEqual(RunSource.retryFailed.rawValue, "retry_failed")
        XCTAssertEqual(RunSource.rerun.rawValue, "rerun")
        XCTAssertEqual(RunSource.planner.rawValue, "planner")
    }

    /// `JobLauncher.rerun` backs both `job run --scope …` and the deep-link resume, and used to
    /// report a flat `.rerun` for all of them, so `--scope cancelled` filed under a different
    /// intent than the identical action taken from `JobDetailView`. These are the pairings that
    /// view's own switch uses; the two mappings have to agree case for case.
    func testRunSourceDerivedFromResetTargetMatchesJobDetailView() {
        XCTAssertEqual(RunSource.resettingCells(withStatus: .cancelled), .resume)
        XCTAssertEqual(RunSource.resettingCells(withStatus: .failed), .retryFailed)
        // No `.selected` equivalent exists as a status: a hand-picked set is the only true rerun,
        // and JobDetailView maps that case itself.
        XCTAssertEqual(RunSource.resettingCells(withStatus: .completed), .rerun)
    }

    /// A cancel that arrives after the last cell finishes leaves the manifest `.completed`
    /// (nothing was left to abandon), so only the live flag can distinguish it, which is why the
    /// flag wins over the status. A non-terminal manifest means the run itself died. Mirrors
    /// Android's `jobOutcome` test.
    func testJobOutcomeReflectsCancellationThenManifestStatus() {
        XCTAssertEqual(analyticsOutcome(cancelled: false, status: .completed), "finished")
        XCTAssertEqual(analyticsOutcome(cancelled: true, status: .paused), "cancelled")
        XCTAssertEqual(analyticsOutcome(cancelled: false, status: .paused), "cancelled")
        XCTAssertEqual(analyticsOutcome(cancelled: false, status: .cancelled), "cancelled")
        XCTAssertEqual(analyticsOutcome(cancelled: true, status: .completed), "cancelled")
        XCTAssertEqual(analyticsOutcome(cancelled: false, status: .running), "failed")
        XCTAssertEqual(analyticsOutcome(cancelled: false, status: .planned), "failed")
    }

    /// `error_kind` must stay a coarse, non-identifying label: registration and download failure
    /// messages in this app embed the management-server URL and the contact email, so anything
    /// richer than the error's type, in particular `localizedDescription`, would leak them.
    func testErrorKindIsTheTypeAndNeverTheMessage() {
        struct RegistrationRejected: Error {
            let detail = "failed to reach https://mgmt.example.internal for someone@example.com"
        }
        XCTAssertEqual(AnalyticsEvents.errorKind(RegistrationRejected()), "RegistrationRejected")
    }

    /// The gate that decides whether the SDK starts at all. The `phc_` check is the
    /// security-relevant one: a PostHog *personal* key (`phx_…`) is a read-write credential that
    /// must never ship in a client, so a mix-up disables analytics rather than leaking a secret.
    func testConfigGateRequiresProjectKeyAndHttpsHost() {
        XCTAssertTrue(isCompletePostHogConfig("phc_abc123", "https://us.i.posthog.com"))
        XCTAssertFalse(isCompletePostHogConfig(nil, "https://us.i.posthog.com"))
        XCTAssertFalse(isCompletePostHogConfig("phc_abc123", nil))
        XCTAssertFalse(isCompletePostHogConfig("phx_personal_api_key", "https://us.i.posthog.com"))
        XCTAssertFalse(isCompletePostHogConfig("phc_abc123", "http://us.i.posthog.com"))
    }

    /// The Settings opt-out row is drawn from these two properties, and they mean different things:
    /// ``NoOpAnalytics`` reports opted-out because it collects nothing, which is not a user's
    /// choice, so the row is hidden on `isAvailable`, not on the opt-out flag. Collapsing the two
    /// would put a dead toggle in front of every user of an unconfigured build, reading "off" as
    /// though they had turned analytics off themselves. Android pins the same pair in
    /// `PostHogConfigurationTest`.
    ///
    /// Only the no-op sink is asserted: `PostHogAnalytics` delegates both to the SDK, and
    /// initializing that in a unit test would create a disk queue.
    func testNoOpSinkReportsUnavailableAndCollectsNothing() {
        let sink = NoOpAnalytics()
        XCTAssertFalse(sink.isAvailable)
        XCTAssertTrue(sink.isOptedOut)
        // Setting it either way changes nothing: there is no state to change.
        sink.setOptedOut(false)
        XCTAssertTrue(sink.isOptedOut)
    }

    /// The opt-out's persistence contract, the twin of Android's `AnalyticsOptOutStoreTest`.
    ///
    /// Worth pinning rather than eyeballing: `Analytics.start()` seeds `PostHogConfig.optOut` from
    /// this value *before* `setup`, and that seeding is the entire mechanism: the SDK's own restore
    /// of its persisted copy was measured doing nothing on posthog-android 3.19.0. A regression that
    /// lost the default here would turn analytics back on for someone who switched them off, and
    /// only on a real cold start.
    func testAnalyticsOptOutDefaultsToCollectingAndRoundTrips() {
        let original = LocalStorage.analyticsOptOut
        defer { LocalStorage.analyticsOptOut = original }

        UserDefaults.standard.removeObject(forKey: LocalStorage.analyticsOptOutKey)
        XCTAssertFalse(LocalStorage.analyticsOptOut, "absent must read as opted in, matching the SDK default and Android's store")

        LocalStorage.analyticsOptOut = true
        XCTAssertTrue(LocalStorage.analyticsOptOut)

        LocalStorage.analyticsOptOut = false
        XCTAssertFalse(LocalStorage.analyticsOptOut)
    }
}
