import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// `Ops/VlSpecDecode.swift`, the peer of `crates/pipette-ops/src/vl_spec_decode.rs`'s own
/// tests: the per-turn checks, the reduction and the output digest, against hand-built
/// `TurnRecord`s.
struct VlSpecDecodeTests {
    /// One forward per committed token but the prefill's, plus the prefill's own: the
    /// shape an engine that accepted nothing reports.
    static func turn(_ generated: UInt64, _ drafted: UInt64, _ accepted: UInt64,
                     _ text: String = "x") -> TurnRecord {
        TurnRecord(predictedMs: 1, e2eMs: 1, generatedTokens: generated,
                   draftedTokens: drafted, acceptedTokens: accepted,
                   targetForwards: generated, text: text)
    }

    // MARK: - Turns

    /// A turn a truthful engine can report, including one whose drafter ran past the
    /// stop token: more accepted drafts than the reply has tokens is a short reply, not
    /// a contradiction, as long as the engine committed every one it counted.
    @Test(arguments: [
        (generated: UInt64(6), drafted: UInt64(8), accepted: UInt64(2), forwards: UInt64(4)),
        (generated: UInt64(9), drafted: UInt64(16), accepted: UInt64(6), forwards: UInt64(3)),
    ])
    func aTruthfulTurnIsCountable(
        generated: UInt64, drafted: UInt64, accepted: UInt64, forwards: UInt64
    ) throws {
        try TurnRecord(predictedMs: 1, e2eMs: 1, generatedTokens: generated,
                           draftedTokens: drafted, acceptedTokens: accepted,
                           targetForwards: forwards, text: "x").validate()
    }

    /// The accept length is the decode loop's own tokens over its own forwards, the
    /// prefill out of both, so a one-token reply that ran no decode loop reads zero.
    @Test(arguments: [
        (generated: UInt64(6), drafted: UInt64(8), forwards: UInt64(4), expected: 5.0 / 3.0),
        (generated: UInt64(2), drafted: UInt64(8), forwards: UInt64(2), expected: 1.0),
        (generated: UInt64(1), drafted: UInt64(0), forwards: UInt64(1), expected: 0.0),
    ])
    func aTurnDividesByTheForwardsTheRuntimeCounted(
        generated: UInt64, drafted: UInt64, forwards: UInt64, expected: Double
    ) throws {
        let turn = TurnRecord(predictedMs: 1, e2eMs: 1, generatedTokens: generated,
                                  draftedTokens: drafted, acceptedTokens: 0,
                                  targetForwards: forwards, text: "x")
        try turn.validate()
        #expect(abs(turn.acceptLength - expected) < 1e-9)
    }

    /// One way an engine can contradict its own counters, and the words the refusal uses.
    struct RefusalCase {
        let name: String
        let generated: UInt64
        let drafted: UInt64
        let accepted: UInt64
        let forwards: UInt64
        let expected: String
    }

    /// An engine that contradicts its own counters fails the turn rather than being clamped
    /// into one: a turn clamped to zero forwards drops out of the pooled denominator, which
    /// *raises* the accept length the server derives.
    ///
    /// The last two are what the counted forwards buy. A drafting turn with no forward
    /// count has no denominator at all, and more forwards than decode tokens means the
    /// count is not this turn's own, since every forward commits a token.
    @Test(arguments: [
        RefusalCase(name: "accepted past drafted", generated: 9, drafted: 2, accepted: 5,
                    forwards: 4, expected: "5 accepted of 2 drafted"),
        RefusalCase(name: "accepted past generated", generated: 4, drafted: 16, accepted: 4,
                    forwards: 2, expected: "counted a draft it never committed"),
        RefusalCase(name: "emitted nothing", generated: 0, drafted: 0, accepted: 0,
                    forwards: 0, expected: "measured nothing"),
        RefusalCase(name: "drafted with no forward count", generated: 6, drafted: 8,
                    accepted: 2, forwards: 1, expected: "counted no decode forward"),
        RefusalCase(name: "more forwards than tokens", generated: 6, drafted: 8, accepted: 2,
                    forwards: 7, expected: "not this turn's own"),
    ])
    func aContradictoryTurnIsRefused(testCase: RefusalCase) {
        let turn = TurnRecord(predictedMs: 1, e2eMs: 1,
                                  generatedTokens: testCase.generated,
                                  draftedTokens: testCase.drafted,
                                  acceptedTokens: testCase.accepted,
                                  targetForwards: testCase.forwards, text: "x")
        let error = #expect(throws: RuntimeError.self) { try turn.validate() }
        #expect(error?.localizedDescription.contains(testCase.expected) == true,
                "\(testCase.name): \(error?.localizedDescription ?? "no error")")
    }

    // MARK: - Counters

    /// Refused rather than reported: a speculative row that never drafted, a baseline row
    /// that drafted, and a traversal with no turns or no client clock.
    @Test(arguments: [
        (VlSpecDecode.Speculation.on, [VlSpecDecodeTests.turn(4, 0, 0)]),
        (.off, [VlSpecDecodeTests.turn(4, 4, 2)]),
        (.off, []),
        (.off, [TurnRecord(predictedMs: 1, e2eMs: 0, generatedTokens: 4, text: "x")]),
    ] as [(VlSpecDecode.Speculation, [TurnRecord])])
    func aTraversalThatMeasuredNothingIsRefused(speculation: VlSpecDecode.Speculation,
                                                turns: [TurnRecord]) {
        #expect(throws: (any Error).self) {
            try VlSpecDecode.counters(speculation: speculation, turns: turns)
        }
    }

    // MARK: - The output digest

    /// The digest follows text and length per turn, and ignores what drafting did: two
    /// rows that said the same things at the same lengths share one, which is all the join
    /// reads to decide greedy identity.
    @Test func theOutputDigestFollowsTextAndLengthPerTurn() {
        let same = VlSpecDecode.outputDigest([Self.turn(12, 0, 0, "a"), Self.turn(9, 0, 0, "b")])
        #expect(same == VlSpecDecode.outputDigest(
            [Self.turn(12, 16, 8, "a"), Self.turn(9, 8, 8, "b")]))
        #expect(same != VlSpecDecode.outputDigest(
            [Self.turn(12, 0, 0, "a"), Self.turn(9, 0, 0, "c")]))
        #expect(same != VlSpecDecode.outputDigest(
            [Self.turn(12, 0, 0, "a"), Self.turn(10, 0, 0, "b")]))
        #expect(VlSpecDecode.outputDigest([Self.turn(12, 0, 0, "ab"), Self.turn(9, 0, 0, "")])
                != VlSpecDecode.outputDigest([Self.turn(12, 0, 0, "a"), Self.turn(9, 0, 0, "b")]),
                "turn boundaries are part of the digest")
    }

    /// Byte-for-byte the crate's `output_digest`, so a phone's row and a laptop's over the
    /// same corpus share a digest exactly when they said the same things. The expected
    /// value is sha256 over `sha256("a")\t12\nsha256("b")\t9\n`, computed independently.
    @Test func theOutputDigestMatchesTheCratesConstruction() {
        let a = Data(SHA256.hash(data: Data("a".utf8))).hexEncodedString()
        let b = Data(SHA256.hash(data: Data("b".utf8))).hexEncodedString()
        let expected = Data(SHA256.hash(data: Data("\(a)\t12\n\(b)\t9\n".utf8)))
            .hexEncodedString()
        #expect(VlSpecDecode.outputDigest([Self.turn(12, 0, 0, "a"), Self.turn(9, 0, 0, "b")])
                == expected)
    }
}
