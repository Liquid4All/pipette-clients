import Foundation
import Testing

@testable import Pipette

/// `keepLastElements` in `PlanTypes/Submission.swift`: the crate's tests of the same names in
/// pipette-mgmt's `crates/pipette-plan-types/src/submission/mod.rs`.
struct KeepLastElementsTests {
    @Test(arguments: [
        // fits as sent
        (#"[ {"i": 0}, {"i": 1} ]"#, 22, #"[ {"i": 0}, {"i": 1} ]"#),
        // drops the first
        (#"[ {"i": 0}, {"i": 1}, {"i": 2} ]"#, 19, #"[{"i": 1},{"i": 2}]"#),
        // keeps nested arrays whole
        ("[[1,2],[3,4]]", 12, "[[3,4]]"),
        // nothing fits
        (#"[{"i":0}]"#, 5, "[]"),
        // last too long
        (#"[0,{"i":"long"}]"#, 10, "[]"),
        // a string holding the delimiters is one element
        (#"["a,]}\"[",1]"#, 5, "[1]"),
    ])
    func keepLastElementsDropsWholeElementsFromTheFront(
        _ s: String, _ maxBytes: Int, _ expected: String
    ) throws {
        #expect(try keepLastElements(s, maxBytes: maxBytes) == expected)
    }

    @Test(arguments: [#"{"i":0}"#, #""[]""#, "throttled=0x0", "", "[1,", #"…"i":2}]"#],
          [2, maxReadinessGateWaitsBytes])
    func keepLastElementsRefusesAnythingButAJsonArray(_ s: String, _ maxBytes: Int) {
        #expect(throws: (any Error).self, "\(s)") {
            try keepLastElements(s, maxBytes: maxBytes)
        }
    }
}
