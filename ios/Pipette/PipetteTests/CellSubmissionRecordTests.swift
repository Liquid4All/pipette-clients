import Foundation
import Testing

@testable import Pipette

/// The raw values are `submission.json`'s on-disk strings: changing one strands every
/// persisted record.
struct CellSubmissionRecordTests {
    @Test func rawValuesAreTheOnDiskWireStrings() {
        #expect(CellSubmissionStatus.submitted.rawValue == "submitted")
        #expect(CellSubmissionStatus.failed.rawValue == "failed")
        #expect(CellSubmissionStatus(rawValue: "submitted") == .submitted)
        #expect(CellSubmissionStatus(rawValue: "failed") == .failed)
        #expect(CellSubmissionStatus(rawValue: "bogus") == nil)
    }

    @Test func encodeDecodeRoundTrip() throws {
        for status in [CellSubmissionStatus.submitted, .failed] {
            let data = try JSONEncoder().encode(status)
            #expect(try JSONDecoder().decode(CellSubmissionStatus.self, from: data) == status)
        }
        #expect(String(decoding: try JSONEncoder().encode(CellSubmissionStatus.submitted), as: UTF8.self) == "\"submitted\"")
    }
}
