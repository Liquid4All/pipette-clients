import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// `Benchmarks/Turn.swift`: the `vl_spec_decode` traversal driven against a fake
/// `VisionChat`, with no model and no drafter. The counting it reduces to is
/// `VlSpecDecodeTests`. What is *not* covered here is the decoding loop, which is
/// llama.cpp's own (`ios/native/pipette_vlspec.cpp`) and needs a device.
@Suite
struct TurnTests {
    // MARK: - Fixtures

    /// A corpus on disk with `samples` conversations, so the traversal can resolve real
    /// media paths.
    static func corpus(_ samples: [(id: String, turns: Int)]) throws -> VlSpecDecodeCorpus {
        let archive = VlSpecDecodeCorpusTests.archive(samples: samples)
        let (corpus, _) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive,
            expected: try Sha256(Data(SHA256.hash(data: archive)).hexEncodedString()),
            dest: try VlSpecDecodeCorpusTests.scratch())
        return corpus
    }

    /// A witness that answers every turn with `reply`, reporting the counters a load with
    /// `accepted` accepted draft tokens per block would.
    ///
    /// `turns` records what it was asked, so a test can assert the traversal's *order*,
    /// which is part of the measurement, rather than only its totals.
    final class FakeChat {
        var conversations: [String] = []
        var turns: [(text: String, images: [String], maxTokens: Int)] = []
        var reply: String
        var drafted: UInt64
        var accepted: UInt64
        var hasDrafter: Bool
        var failOnTurn: Int?
        var generated: UInt64 = 6
        var endedOnEos = false
        var endedOnLength = false

        init(reply: String = "an answer", drafted: UInt64 = 0, accepted: UInt64 = 0,
             hasDrafter: Bool = false, failOnTurn: Int? = nil) {
            self.reply = reply
            self.drafted = drafted
            self.accepted = accepted
            self.hasDrafter = hasDrafter
            self.failOnTurn = failOnTurn
        }

        var chat: VisionChat {
            VisionChat(
                begin: { self.conversations.append($0) },
                turn: { text, images, maxTokens in
                    self.turns.append((text, images, maxTokens))
                    if self.turns.count == self.failOnTurn {
                        throw RuntimeError.engine("the fake refused this turn")
                    }
                    // Six committed tokens per turn by default: the prefill's, then two
                    // verified blocks. The forward count is counted, not derived from
                    // `accepted`, which a drafter can run past the stop token.
                    return TurnRecord(
                        promptMs: 30, predictedMs: 40, ttftMs: 25, decodeMs: 15, e2eMs: 0,
                        generatedTokens: self.generated, draftedTokens: self.drafted,
                        acceptedTokens: self.accepted,
                        targetForwards: 3, endedOnEos: self.endedOnEos,
                        endedOnLength: self.endedOnLength, text: self.reply)
                },
                hasDrafter: hasDrafter)
        }
    }

    static func run(speculation: VlSpecDecode.Speculation, corpus: VlSpecDecodeCorpus, chat: FakeChat)
        async throws -> (data: BenchmarkResultData, traversal: VlSpecDecodeCorpusTurn.Traversal) {
        try await VlSpecDecodeCorpusTurn.run(
            speculation: speculation, corpus: corpus, draftLength: 8, maxTokens: 256,
            readinessGate: {}, observer: noObserver()
        ) { (chat.chat, {}) }
    }

    /// `traverse` alone, for the tests about order and gating.
    static func traverse(speculation: VlSpecDecode.Speculation, corpus: VlSpecDecodeCorpus, chat: FakeChat,
                         readinessGate: ReadinessGate = {}) async throws
        -> VlSpecDecodeCorpusTurn.Traversal {
        try await VlSpecDecodeCorpusTurn.traverse(
            speculation: speculation, corpus: corpus, maxTokens: 256, readinessGate: readinessGate,
            observer: noObserver(),
            isCancelled: { false }, progress: { _ in }) { (chat.chat, {}) }
    }

    // MARK: - Prompt shaping

    /// Each image becomes a media marker in place, so the prompt reads as the corpus
    /// composed it and the markers match the bitmaps one for one.
    @Test func promptPutsAMarkerWhereEachImageSits() throws {
        let (text, images) = try VlSpecDecodeCorpusTurn.prompt(for: [
            .text("what is this? "),
            .image(path: "media/aa.png", mime: "image/png", bytes: 4),
            .text(" be brief."),
        ])
        #expect(text == "what is this? \(LlamaMtmd.defaultMediaMarker) be brief.")
        #expect(images == ["media/aa.png"])
    }

    @Test func aTurnWithNoContentIsRefused() {
        #expect(throws: (any Error).self) { try VlSpecDecodeCorpusTurn.prompt(for: []) }
    }

    // MARK: - The traversal

    /// A length stop short of the 256 `maxTokens` a turn was given is refused at the warm-up,
    /// naming both counts; at the budget, or on the model's own stop, every turn is counted.
    @Test(arguments: [
        (generated: UInt64(256), endedOnEos: false, endedOnLength: true, refused: false),
        (generated: UInt64(12), endedOnEos: false, endedOnLength: true, refused: true),
        (generated: UInt64(12), endedOnEos: true, endedOnLength: false, refused: false),
    ])
    func aLengthStopShortOfMaxTokensIsRefused(
        generated: UInt64, endedOnEos: Bool, endedOnLength: Bool, refused: Bool
    ) async throws {
        let corpus = try Self.corpus([("s1", 2)])
        let chat = FakeChat(drafted: 8, accepted: 2, hasDrafter: true)
        chat.generated = generated
        chat.endedOnEos = endedOnEos
        chat.endedOnLength = endedOnLength
        do {
            let traversal = try await Self.traverse(speculation: .on, corpus: corpus, chat: chat)
            #expect(!refused, "a cut reply was counted")
            #expect(traversal.turns.count == 2)
        } catch {
            #expect(refused, "\(error.localizedDescription)")
            #expect(error.localizedDescription.contains(
                "at \(generated) tokens, short of the 256 max_tokens"),
                "\(error.localizedDescription)")
            #expect(chat.turns.count == 1)
        }
    }

    /// Order is part of the measurement: the warm-up, then the samples in the file's order,
    /// each its own conversation.
    @Test func traversesEverySampleInOrderAfterTheWarmUp() async throws {
        let corpus = try Self.corpus([("s1", 2), ("s2", 1)])
        let chat = FakeChat(drafted: 8, accepted: 2, hasDrafter: true)
        let traversal = try await Self.traverse(speculation: .on, corpus: corpus, chat: chat)

        // Warm-up, then three measured turns.
        #expect(chat.turns.count == 4)
        // Warm-up, one per sample.
        #expect(chat.conversations.count == 3)
        // The measured turns: the first of each sample carries the image, the second not.
        #expect(chat.turns.allSatisfy { $0.maxTokens == 256 })
        #expect(chat.turns[1].images.count == 1)
        #expect(chat.turns[2].images.isEmpty)
        #expect(chat.turns[1].images[0].hasSuffix("media/aa.png"))
        #expect(traversal.turns.count == 3)
        #expect(traversal.observations.count == 3)
    }

    /// Each condition is refused the other's load: a speculative run whose drafter never
    /// attached would submit its own baseline under the speculative name.
    @Test func aSpeculativeRunWithNoDrafterIsRefused() async throws {
        let corpus = try Self.corpus([("s1", 1)])
        await #expect(throws: (any Error).self) {
            _ = try await Self.traverse(speculation: .on, corpus: corpus,
                                        chat: FakeChat(hasDrafter: false))
        }
    }

    /// …and a run with speculation off that loaded a drafter would submit speculation as
    /// the thing a speedup is divided by.
    @Test func aRunWithSpeculationOffAndADrafterIsRefused() async throws {
        let corpus = try Self.corpus([("s1", 1)])
        await #expect(throws: (any Error).self) {
            _ = try await Self.traverse(speculation: .off, corpus: corpus,
                                        chat: FakeChat(hasDrafter: true))
        }
    }

    /// A failed sample names itself: a multi-hour traversal that died on sample 417 is only
    /// attributable if the error says so.
    @Test func aFailedSampleNamesItself() async throws {
        let corpus = try Self.corpus([("s1", 1), ("s2", 1)])
        // Turn 1 is the warm-up, so failing on turn 3 fails the second sample.
        let chat = FakeChat(drafted: 8, accepted: 2, hasDrafter: true, failOnTurn: 3)
        do {
            _ = try await Self.traverse(speculation: .on, corpus: corpus, chat: chat)
            Issue.record("the traversal should have failed")
        } catch {
            #expect("\(error)".contains("s2"))
            #expect("\(error)".contains("vl_spec_decode"))
        }
    }

    /// The gate falls before the first sample and before every sample after the first, never
    /// between the turns of one: a multi-turn sample is one conversation, and a wait inside
    /// it would split what the corpus defines as a unit.
    @Test func theGateFallsOncePerSample() async throws {
        let corpus = try Self.corpus([("s1", 3), ("s2", 2), ("s3", 1)])
        let chat = FakeChat(drafted: 8, accepted: 2, hasDrafter: true)
        var gates = 0
        _ = try await Self.traverse(speculation: .on, corpus: corpus, chat: chat,
                                    readinessGate: { gates += 1 })
        // One per sample; per turn would be six.
        #expect(gates == 3)
    }

    // MARK: - Counters

    @Test func aSpeculativeTraversalReducesToItsCounters() async throws {
        let corpus = try Self.corpus([("s1", 2), ("s2", 1)])
        let (data, _) = try await Self.run(
            speculation: .on, corpus: corpus,
            chat: FakeChat(drafted: 8, accepted: 2, hasDrafter: true))
        guard case let .vlSpecDecode(turns, totals, drafted, accepted, acceptLengthSum) = data
        else {
            Issue.record("expected a vl_spec_decode row, got \(data)")
            return
        }
        #expect(turns == 3)
        #expect(totals.generatedTokens == 18)
        #expect(drafted == 24)
        #expect(accepted == 6)
        // Per request, each turn reads its five decode tokens over its two decode
        // forwards; summed over three turns.
        #expect(acceptLengthSum == 7.5)
        // The engine's own clocks are summed; the client clock is stamped per turn.
        #expect(totals.promptMs == 90)
        #expect(totals.predictedMs == 120)
        #expect(totals.ttftMs == 75)
        #expect(totals.e2eMs > 0)
    }

    /// With speculation off, the same sums and the drafter counters absent rather than
    /// zero: a zero would put a `0 / 0` in the server's acceptance rate.
    @Test func aTraversalWithSpeculationOffCarriesNoDrafterCounters() async throws {
        let corpus = try Self.corpus([("s1", 2), ("s2", 1)])
        let (data, _) = try await Self.run(speculation: .off, corpus: corpus, chat: FakeChat())
        guard case let .vlSpecDecode(turns, totals, drafted, accepted, acceptLengthSum) = data
        else {
            Issue.record("expected a vl_spec_decode row, got \(data)")
            return
        }
        #expect(turns == 3)
        #expect(totals.generatedTokens == 18)
        #expect(totals.predictedMs == 120)
        #expect(drafted == nil)
        #expect(accepted == nil)
        #expect(acceptLengthSum == nil)
    }

    // MARK: - The observations

    /// The per-turn record carries the reply's hash, never the reply.
    @Test func theTurnRecordHoldsNoText() async throws {
        let corpus = try Self.corpus([("s1", 1)])
        let (_, traversal) = try await Self.run(speculation: .off, corpus: corpus,
                                                chat: FakeChat(reply: "a secret reply"))
        let encoded = String(decoding: try JSONEncoder().encode(traversal.observations),
                             as: UTF8.self)
        #expect(!encoded.contains("a secret reply"))
        #expect(traversal.observations[0].textSha256
                == VlSpecDecode.textSha256("a secret reply"))
    }

    /// The kept record carries the turn's decode forwards, the prefill's out, and how it
    /// stopped: the crate's `TurnObservation.counted_forwards` and `ended_on_eos`.
    @Test func theObservationKeepsTheDecodeForwardsAndTheStop() throws {
        let turn = TurnRecord(predictedMs: 1, e2eMs: 1, generatedTokens: 6,
                                  draftedTokens: 8, acceptedTokens: 3, targetForwards: 4,
                                  endedOnEos: true, text: "x")
        let observation = TurnObservation(sampleId: "s1", turn: 0, record: turn)
        #expect(observation.countedForwards == 3)
        #expect(observation.endedOnEos)

        let json = try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(observation)) as? [String: Any])
        #expect(json["counted_forwards"] as? Int == 3)
        #expect(json["ended_on_eos"] as? Bool == true)
    }

    // MARK: - The parameters

    @Test func aZeroDraftLengthIsRefusedWithSpeculationOn() async throws {
        let corpus = try Self.corpus([("s1", 1)])
        await #expect(throws: (any Error).self) {
            _ = try await VlSpecDecodeCorpusTurn.run(
                speculation: .on, corpus: corpus, draftLength: 0, maxTokens: 256,
                readinessGate: {}, observer: noObserver()
            ) { (FakeChat(hasDrafter: true).chat, {}) }
        }
    }
}
