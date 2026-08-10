package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the gate that decides whether PostHog initializes at all ([isCompletePostHogConfig]) plus the property helpers whose output ships to the
 * analytics backend. The seam itself ([PostHogAnalytics]) isn't exercised here: it's a thin forwarder to a static SDK, and initializing that SDK in a
 * unit test would create a disk queue.
 */
class PostHogConfigurationTest {
  private val realKey = "phc_tGKa98yNH3WzFs7C2SKKisXYoKU2rLU8VkB3aYMRnyao"
  private val realHost = "https://us.i.posthog.com"

  @Test
  fun configRuleAcceptsRealProjectKeyAndHost() {
    assertTrue(isCompletePostHogConfig(realKey, realHost))
    assertTrue(isCompletePostHogConfig("phc_test123", "https://eu.i.posthog.com"))
    assertTrue(isCompletePostHogConfig("phc_test123", "https://posthog.internal.example"))
  }

  @Test
  fun configRuleRejectsMissingOrUnsubstitutedValues() {
    // A build with no PostHog configured must run analytics-free rather than push at an endpoint
    // that will reject it.
    assertFalse(isCompletePostHogConfig("", realHost))
    assertFalse(isCompletePostHogConfig("   ", realHost))
    assertFalse(isCompletePostHogConfig(realKey, ""))
    // Unsubstituted build-config placeholders are not real values.
    assertFalse(isCompletePostHogConfig("\$(POSTHOG_API_KEY)", realHost))
    assertFalse(isCompletePostHogConfig(realKey, "\$(POSTHOG_HOST)"))
  }

  /**
   * The security-relevant case. A PostHog **personal** API key (`phx_…`) is a read-write credential; only a `phc_…` project key is write-only and
   * safe to ship in a client. Refusing to initialize with anything else means a mix-up disables analytics instead of shipping a live credential in an
   * APK, so this assertion is the reason the prefix check exists.
   */
  @Test
  fun configRuleRejectsNonProjectKeys() {
    assertFalse(isCompletePostHogConfig("phx_personal_api_key", realHost))
    assertFalse(isCompletePostHogConfig("sk_live_something", realHost))
    assertFalse(isCompletePostHogConfig("tGKa98yNH3WzFs7C2SKKisXYoKU2rLU8VkB3aYMRnyao", realHost))
  }

  /** Plain http would send events in the clear; the gate requires TLS. */
  @Test
  fun configRuleRequiresHttpsHost() {
    assertFalse(isCompletePostHogConfig(realKey, "http://us.i.posthog.com"))
    assertFalse(isCompletePostHogConfig(realKey, "us.i.posthog.com"))
  }

  /**
   * `error_kind` must stay a coarse, non-identifying label. Registration and download failure messages in this app embed the management-server URL
   * and the contact email, so anything richer than the exception type would leak them into analytics.
   */
  @Test
  fun errorKindIsTheExceptionTypeAndNeverTheMessage() {
    val error = IllegalStateException("failed to reach https://mgmt.example.internal for someone@example.com")
    assertEquals("IllegalStateException", AnalyticsEvents.errorKind(error))

    // Anonymous subclasses have a blank simple name; fall back rather than emitting "".
    val anonymous = object : RuntimeException("boom") {}
    assertEquals("Throwable", AnalyticsEvents.errorKind(anonymous))
  }

  /**
   * The reported outcome of a finished run. A cancel that arrives after the last cell has already finished leaves the manifest [JobStatus.COMPLETED]
   * (nothing was left to abandon), so only the live flag can distinguish it, which is why the flag wins over the status. A manifest left non-terminal
   * (or unreadable) means the run itself died rather than ending normally.
   */
  @Test
  fun jobOutcomeReflectsCancellationThenManifestStatus() {
    assertEquals(AnalyticsEvents.OUTCOME_FINISHED, jobOutcome(cancelled = false, status = JobStatus.COMPLETED))
    assertEquals(AnalyticsEvents.OUTCOME_CANCELLED, jobOutcome(cancelled = true, status = JobStatus.PAUSED))
    assertEquals(AnalyticsEvents.OUTCOME_CANCELLED, jobOutcome(cancelled = false, status = JobStatus.PAUSED))
    // Never written by this runner, but `JobStatus.fromWire` deserializes it, and a manifest from
    // another writer must not be reported as a failure. iOS treats `.cancelled` the same way.
    assertEquals(AnalyticsEvents.OUTCOME_CANCELLED, jobOutcome(cancelled = false, status = JobStatus.CANCELLED))
    // Cancelled on the final cell: the manifest is COMPLETED, but the run did not finish on its own.
    assertEquals(AnalyticsEvents.OUTCOME_CANCELLED, jobOutcome(cancelled = true, status = JobStatus.COMPLETED))
    // Non-terminal or unreadable manifest: the run died.
    assertEquals(AnalyticsEvents.OUTCOME_FAILED, jobOutcome(cancelled = false, status = JobStatus.RUNNING))
    assertEquals(AnalyticsEvents.OUTCOME_FAILED, jobOutcome(cancelled = false, status = null))
  }

  /**
   * The counterpart of iOS's `testEventNamesMatchCrossPlatformContract`.
   *
   * Event names are the highest-value half of the contract: a rename on one platform alone doesn't break a build, it silently splits every funnel
   * query into two events that look unrelated. Like `FeedbackCategoryTest`, this can't enforce true parity (each platform pins its own copy), but it
   * catches an accidental edit and makes a deliberate one the reminder to change iOS too.
   */
  @Test
  fun eventNamesMatchCrossPlatformContract() {
    assertEquals("app_launched", AnalyticsEvents.APP_LAUNCHED)
    assertEquals("device_registered", AnalyticsEvents.DEVICE_REGISTERED)
    assertEquals("device_registration_failed", AnalyticsEvents.DEVICE_REGISTRATION_FAILED)
    assertEquals("model_download_started", AnalyticsEvents.MODEL_DOWNLOAD_STARTED)
    assertEquals("model_download_completed", AnalyticsEvents.MODEL_DOWNLOAD_COMPLETED)
    assertEquals("model_download_failed", AnalyticsEvents.MODEL_DOWNLOAD_FAILED)
    assertEquals("job_started", AnalyticsEvents.JOB_STARTED)
    assertEquals("job_completed", AnalyticsEvents.JOB_COMPLETED)
    assertEquals("results_submitted", AnalyticsEvents.RESULTS_SUBMITTED)
    assertEquals("feedback_submitted", AnalyticsEvents.FEEDBACK_SUBMITTED)
  }

  /**
   * Property keys split a funnel just as event names do, and these are hand-duplicated on iOS. [AnalyticsEvents.SUBMITTED_COUNT] is deliberately not
   * [AnalyticsEvents.CELL_COUNT]: that one means "cells this run scheduled" on `job_started`/`job_completed`, and overloading it on
   * `results_submitted` would give one property two meanings and make any aggregate across the two events wrong.
   */
  @Test
  fun propertyKeysMatchCrossPlatformContract() {
    assertEquals("cell_count", AnalyticsEvents.CELL_COUNT)
    assertEquals("cells_completed", AnalyticsEvents.CELLS_COMPLETED)
    assertEquals("submitted_count", AnalyticsEvents.SUBMITTED_COUNT)
    assertEquals("duration_ms", AnalyticsEvents.DURATION_MS)
    assertEquals("size_bytes", AnalyticsEvents.SIZE_BYTES)
    assertEquals("model_id", AnalyticsEvents.MODEL_ID)
    assertEquals("error_kind", AnalyticsEvents.ERROR_KIND)
    assertEquals("os_version", AnalyticsEvents.OS_VERSION)
    assertEquals("app_version", AnalyticsEvents.APP_VERSION)
  }

  /**
   * The run-source names travel to PostHog as event property values, so they are part of the wire contract: renaming the Kotlin constant must not
   * silently rename the property value and split a funnel across two spellings. [RunSource] declares each string explicitly for that reason, and this
   * test is what catches an edit to those literals; the same values are pinned on iOS in `AnalyticsEventsTests`.
   */
  @Test
  fun runSourceWireNamesAreStable() {
    assertEquals("new", RunSource.NEW.wireName)
    assertEquals("resume", RunSource.RESUME.wireName)
    assertEquals("retry_failed", RunSource.RETRY_FAILED.wireName)
    assertEquals("rerun", RunSource.RERUN.wireName)
  }

  /**
   * The Settings opt-out row is drawn from these two properties, and they mean different things: [NoOpAnalytics] reports opted-out because it
   * collects nothing, which is not a user's choice, so the row is hidden on [Analytics.isAvailable], not on the opt-out flag. Collapsing the two
   * would put a dead toggle in front of every user of an unconfigured build, reading "off" as though they had turned analytics off themselves.
   *
   * Only the no-op sink is asserted here: `PostHogAnalytics` delegates both to the SDK, which a JVM unit test can't initialize.
   */
  @Test
  fun noOpSinkReportsUnavailableAndCollectsNothing() {
    assertFalse(NoOpAnalytics.isAvailable)
    assertTrue(NoOpAnalytics.isOptedOut())
    // Setting it either way changes nothing: there is no state to change.
    NoOpAnalytics.setOptedOut(false)
    assertTrue(NoOpAnalytics.isOptedOut())
  }
}
