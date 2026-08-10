import XCTest
@testable import Pipette

final class AcknowledgementsTests: XCTestCase {
    func testBundledLicensesDecodeAndAreNonEmpty() {
        XCTAssertFalse(Acknowledgements.all.isEmpty, "ThirdPartyLicenses.json missing from bundle or malformed")
    }

    func testEntryIdsAreUnique() {
        let ids = Acknowledgements.all.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "duplicate entry names break List identity")
    }

    func testVendoredLlamaCppIsAttributed() {
        let llama = Acknowledgements.all.first { $0.name == "llama.cpp" }
        XCTAssertNotNil(llama)
        XCTAssertTrue(llama?.text.contains("ggml authors") ?? false)
    }

    func testEntriesHaveNonEmptyLicenseTexts() {
        for entry in Acknowledgements.all {
            XCTAssertFalse(entry.license.isEmpty, "\(entry.name) has no license label")
            XCTAssertFalse(entry.text.isEmpty, "\(entry.name) has no license text")
        }
    }
}
