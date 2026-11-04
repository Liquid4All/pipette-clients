import XCTest
@testable import Pipette

final class FeedbackCategoryTests: XCTestCase {
    /// The same ids, in order, are hand-copied in Android's `FeedbackDialog.CATEGORY_IDS`
    /// and pipette-dashboard's `FEEDBACK_CATEGORIES`, so the Sentry `category` tag means
    /// the same on every platform. Changing this list means changing those too.
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
