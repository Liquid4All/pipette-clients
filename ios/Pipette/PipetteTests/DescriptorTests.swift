import Foundation
import Testing

@testable import Pipette

/// `Descriptor` — the canonical form and the digest that addresses it, mirroring
/// `crates/pipette-plan-types/src/descriptor.rs`.
///
/// The rule is shared with `pipette-mgmt`'s `canonical_json`, so a digest computed here
/// addresses the same artifact the warehouse groups by. Drift would split one identity in
/// two, which is why the vectors are pinned rather than assumed.
struct DescriptorTests {
    /// The fixed vectors `descriptor.rs` shares with `pipette-mgmt`'s `canonical_json`.
    /// If either side changes its rule, one of these fails instead of the repos quietly
    /// disagreeing about what a descriptor's id is.
    @Test func canonicalizationMatchesTheSharedVectors() {
        let cases: [([String: Any], String)] = [
            ([:], "{}"),
            (["b": 1, "a": 2], #"{"a":2,"b":1}"#),
            (["a": [1, 2]], #"{"a":[1,2]}"#),
            (["a": NSNull()], #"{"a":null}"#),
            (["a": "x y"], #"{"a":"x y"}"#),
        ]

        for (value, expected) in cases {
            #expect(Descriptor.canonicalize(value) == expected)
        }
    }

    /// Anchored on the hash too, not only the canonical form, as the crate anchors it.
    @Test func theDigestIsSha256OfTheCanonicalForm() throws {
        let value: [String: Any] = ["b": 1, "a": 2]

        #expect(try Descriptor.digest(["a": 2, "b": 1])
            == Descriptor.sha256Hex(Descriptor.canonicalize(value)))
    }

    /// Nested objects sort at every level; arrays keep their order because position
    /// carries meaning. The crate's own case, verbatim.
    @Test func canonicalFormSortsDeeplyAndPreservesArrays() {
        let value: [String: Any] = [
            "b": ["z": 1, "a": 2],
            "a": ["second", "first"],
        ]

        #expect(Descriptor.canonicalize(value)
            == #"{"a":["second","first"],"b":{"a":2,"z":1}}"#)
    }

    /// A bool is not a number. `JSONSerialization` hands both back as `NSNumber`, and only
    /// the ObjC type tells them apart — writing `true` as `1` would hash differently from
    /// the same descriptor on the Rust side.
    @Test func boolsStayBools() {
        #expect(Descriptor.canonicalize(["t": true, "n": 1]) == #"{"n":1,"t":true}"#)
    }

    /// The same model formatted two ways reduces to one digest — the property the whole
    /// scheme rests on.
    @Test func fieldOrderDoesNotChangeTheDigest() throws {
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let json = try JSONEncoder().encode(model)
        let object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
        // Re-serialize with sorted keys: different bytes, same value.
        let sorted = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])

        #expect(Descriptor.canonicalize(object)
            == Descriptor.canonicalize(try #require(
                JSONSerialization.jsonObject(with: sorted) as? [String: Any])))
        #expect(try Descriptor.digest(model).count == 64)
    }

    /// Two different models do not share a digest, and the display form is a prefix of the
    /// full one.
    @Test func distinctModelsGetDistinctDigests() throws {
        let one = try Descriptor.digest(try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"))
        let two = try Descriptor.digest(try ggufTextSpec("org/a-GGUF", "a-Q5_K_M.gguf"))

        #expect(one != two)
        #expect(one.hasPrefix(Descriptor.shortDigest(one)))
        #expect(Descriptor.shortDigest(one).count == Descriptor.digestDisplayLength)
    }
}
