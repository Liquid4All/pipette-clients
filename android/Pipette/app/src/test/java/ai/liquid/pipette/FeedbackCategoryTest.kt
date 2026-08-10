package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Tripwire pinning the Android feedback category ids and their order. The same id set is hand-duplicated in the iOS `FeedbackCategory` enum and
 * pipette-dashboard's `FEEDBACK_CATEGORIES`. This can't enforce true cross-platform parity (each platform pins its own copy), but it catches an
 * *accidental* change here — and when the change is intentional, updating this list is the reminder to update the other platforms so the Sentry
 * `category` tag keeps meaning the same thing across web, Android, and iOS.
 */
class FeedbackCategoryTest {
  @Test
  fun categoryIdsMatchCrossPlatformContract() {
    val expected = listOf("report_bug", "report_incorrect_data", "request_model", "request_runtime", "request_hardware", "request_eval", "other")
    assertEquals(expected, FeedbackDialog.CATEGORY_IDS)
  }
}
