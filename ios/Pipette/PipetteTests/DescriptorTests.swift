import Foundation
import Testing

@testable import Pipette

/// The canonical form and the digest that addresses it, mirroring `canonical_json` in
/// pipette-mgmt's `crates/pipette-plan-types/src/descriptor.rs`. The warehouse groups
/// by that digest, so drift would split one identity in two.
struct DescriptorTests {
    /// The fixed vectors from mgmt's `descriptor.rs`.
    @Test func canonicalizationMatchesTheSharedVectors() {
        let cases: [([String: Any], String)] = [
            ([:], "{}"),
            (["b": 1, "a": 2], #"{"a":2,"b":1}"#),
            (["a": [1, 2]], #"{"a":[1,2]}"#),
            (["a": NSNull()], #"{"a":null}"#),
            (["a": "x y"], #"{"a":"x y"}"#),
        ]

        for (value, expected) in cases {
            #expect(canonicalJson(value) == expected)
        }
    }

    /// Anchored on the hash too, not only the canonical form, as the crate anchors it.
    @Test func theDigestIsSha256OfTheCanonicalForm() throws {
        let value: [String: Any] = ["b": 1, "a": 2]

        #expect(try descriptorDigest(["a": 2, "b": 1])
            == sha256Hex(canonicalJson(value)))
    }

    /// Nested objects sort at every level; arrays keep their order because position
    /// carries meaning. The crate's own case, verbatim.
    @Test func canonicalFormSortsDeeplyAndPreservesArrays() {
        let value: [String: Any] = [
            "b": ["z": 1, "a": 2],
            "a": ["second", "first"],
        ]

        #expect(canonicalJson(value)
            == #"{"a":["second","first"],"b":{"a":2,"z":1}}"#)
    }

    /// `JSONSerialization` hands bools and numbers back as `NSNumber`, and only the ObjC type
    /// tells them apart; writing `true` as `1` would hash differently from the Rust side.
    @Test func boolsStayBools() {
        #expect(canonicalJson(["t": true, "n": 1]) == #"{"n":1,"t":true}"#)
    }

    /// `1` and `1.0` stay distinct, as `serde_json` renders them: an integral float keeps
    /// its fraction, and an exponent is spelled without a sign or padding.
    @Test func anIntegralFloatKeepsItsFraction() throws {
        let parsed = try #require(try JSONSerialization.jsonObject(
            with: Data(#"{"f":1.0,"i":1,"h":0.5}"#.utf8)) as? [String: Any])
        #expect(canonicalJson(parsed) == #"{"f":1.0,"h":0.5,"i":1}"#)
        #expect(canonicalJson(["big": 1e20, "small": 1e-7]) == #"{"big":1e20,"small":1e-7}"#)
    }

    @Test func fieldOrderDoesNotChangeTheDigest() throws {
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let json = try JSONEncoder().encode(model)
        let object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
        // Re-serialize with sorted keys: different bytes, same value.
        let sorted = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])

        #expect(canonicalJson(object)
            == canonicalJson(try #require(
                JSONSerialization.jsonObject(with: sorted) as? [String: Any])))
        #expect(try descriptorDigest(model).count == 64)
    }

    /// Two different models do not share a digest, and the display form is a prefix of the
    /// full one.
    @Test func distinctModelsGetDistinctDigests() throws {
        let one = try descriptorDigest(try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"))
        let two = try descriptorDigest(try ggufTextSpec("org/a-GGUF", "a-Q5_K_M.gguf"))

        #expect(one != two)
        #expect(one.hasPrefix(shortDigest(one)))
        #expect(shortDigest(one).count == digestDisplayLen)
    }

    /// `short_digest_truncates_without_panicking_on_short_input`.
    @Test func shortDigestTruncatesWithoutTrappingOnShortInput() {
        #expect(shortDigest("abcdef0123456789abcdef") == "abcdef012345")
        #expect(shortDigest("abc") == "abc")
    }
}
