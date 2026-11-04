import Foundation
import Testing

@testable import Pipette

/// The `pipette-http` crate's `range.rs` tests.
struct RangeTests {
    @Test(arguments: [
        ("\"abc\"", "\"abc\""),
        ("W/\"abc\"", nil),
    ] as [(String, String?)])
    func onlyAStrongEtagIsReturned(value: String, expected: String?) {
        #expect(strongEtagValue(value) == expected)
    }

    @Test(arguments: [
        ("bytes 4-9/10", ContentRange(first: 4, last: 9, total: 10)),
        ("bytes 4-9/*", ContentRange(first: 4, last: 9, total: nil)),
        ("bytes */10", nil),
        ("bytes 9-4/10", nil),
        ("items 4-9/10", nil),
    ] as [(String, ContentRange?)])
    func aContentRangeNamesOneByteOrMore(value: String, expected: ContentRange?) {
        #expect(ContentRange(headerValue: value) == expected)
    }

    @Test func anAbsentHeaderGivesNothing() {
        let response = PipetteMgmtClient.HttpResponse(status: 206, headers: [], body: Data())
        #expect(response.strongEtag == nil)
        #expect(response.contentRange == nil)
    }
}
