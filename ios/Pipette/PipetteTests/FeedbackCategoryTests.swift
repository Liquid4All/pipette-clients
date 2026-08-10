import XCTest
@testable import Pipette

final class FeedbackCategoryTests: XCTestCase {
    /// Tripwire pinning the iOS feedback category ids and their order. The same set is
    /// hand-duplicated in the Android `FeedbackDialog.CATEGORY_IDS` and pipette-dashboard's
    /// `FEEDBACK_CATEGORIES`. This can't enforce true cross-platform parity (each platform pins
    /// its own copy), but it catches an *accidental* change here — and when the change is
    /// intentional, updating this list is the reminder to update the other platforms so the
    /// Sentry `category` tag keeps meaning the same thing across web, Android, and iOS.
    func testCategoryIdsMatchCrossPlatformContract() {
        let expected = [
            "report_bug",
            "report_incorrect_data",
            "request_model",
            "request_runtime",
            "request_hardware",
            "request_eval",
            "other"
        ]
        XCTAssertEqual(FeedbackCategory.allCases.map(\.id), expected)
    }
}
