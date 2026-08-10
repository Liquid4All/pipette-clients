import Testing
import XCTest
@testable import Pipette

/// Collector-identity comparison used to detect results that belong to a
/// different collector and must be re-sent after a collector change.
struct CollectorIdentityTests {
    @Test func sameCollectorIgnoresTrailingSlashAndScheme() {
        #expect(CollectorEndpoint.isSameCollector(
            "https://collector.example.com/", as: "https://collector.example.com"))
        #expect(CollectorEndpoint.isSameCollector(
            "collector.example.com", as: "https://collector.example.com"))
    }

    @Test func sameCollectorIgnoresHostCase() {
        #expect(CollectorEndpoint.isSameCollector(
            "https://Collector.Example.com", as: "https://collector.example.com"))
    }

    @Test func differentHostsAreDifferentCollectors() {
        #expect(!CollectorEndpoint.isSameCollector(
            "https://old.example.com", as: "https://collector.example.com"))
    }

    @Test func missingOrBlankStoredCollectorCountsAsDifferent() {
        #expect(!CollectorEndpoint.isSameCollector(nil, as: "https://collector.example.com"))
        #expect(!CollectorEndpoint.isSameCollector("  ", as: "https://collector.example.com"))
    }
}

final class CollectorEndpointTests: XCTestCase {
    func testEndpointOptionsUseExpectedLabels() {
        XCTAssertEqual(CollectorEndpointOption.production.title, "Liquid AI")
        XCTAssertEqual(CollectorEndpointOption.custom.title, "Custom")
    }

    func testProductionOptionUsesHttpsCollectorURL() {
        XCTAssertEqual(
            CollectorEndpointOption.production.serverURL(customURL: ""),
            "https://collector.pipette.liquid.ai"
        )
    }

    func testCustomOptionAddsHttpsToBareHost() {
        XCTAssertEqual(
            CollectorEndpointOption.custom.serverURL(customURL: "collector.example.com"),
            "https://collector.example.com"
        )
    }

    func testCustomOptionPreservesHttpsSchemeAndPathPrefix() {
        XCTAssertEqual(
            CollectorEndpointOption.custom.serverURL(customURL: "https://collector.example.com/pipette/"),
            "https://collector.example.com/pipette"
        )
    }

    func testCustomOptionRejectsHttpUnsupportedSchemesAndQueries() {
        XCTAssertNil(CollectorEndpointOption.custom.serverURL(customURL: "http://collector.example.com"))
        XCTAssertNil(CollectorEndpointOption.custom.serverURL(customURL: "ftp://collector.example.com"))
        XCTAssertNil(CollectorEndpointOption.custom.serverURL(customURL: "https://collector.example.com?token=1"))
    }
}
