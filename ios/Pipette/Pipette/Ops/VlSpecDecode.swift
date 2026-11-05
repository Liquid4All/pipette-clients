import CryptoKit
import Foundation

// What a `vl_spec_decode` turn counts as, what a traversal sums to, and which traversals
// are refused: `crates/pipette-ops/src/vl_spec_decode.rs`. One model, one traversal, one
// row; comparing two rows is the management server's join.

nonisolated enum VlSpecDecode {
    /// A run with speculation off that drafted anything, and one with it on that drafted
    /// nothing, are both refused: each is the other condition's measurement under the wrong
    /// name.
    nonisolated enum Speculation: Equatable {
        case off, on

        static func fromEnabled(_ enabled: Bool) -> Self { enabled ? .on : .off }

        var enabled: Bool { self == .on }

        var label: String {
            switch self {
            case .off: "vl_spec_decode (speculation off)"
            case .on: "vl_spec_decode (speculation on)"
            }
        }
    }
}

/// What one turn produced, in the terms the wire counts.
///
/// Two clocks, as upstream carries two. `predictedMs` is the engine's own, which the server
/// turns into tok/s and nothing else. `ttftMs`, `decodeMs` and `e2eMs` are the caller's,
/// which the latency metrics report; `e2eMs` is measured around the whole call rather than
/// added up from the other two, because the server rejects a value below the sum of its own
/// phases as a reconstruction.
nonisolated struct TurnRecord: Equatable, Sendable {
    /// The engine's prefill time, vision tower included. Not on the wire (no metric reads
    /// it), but per turn it says whether prefill drifted as the device heat-soaked.
    var promptMs: Double = 0
    var predictedMs: Double = 0
    var ttftMs: Double = 0
    var decodeMs: Double = 0
    var e2eMs: Double = 0
    var generatedTokens: UInt64 = 0
    var draftedTokens: UInt64 = 0
    var acceptedTokens: UInt64 = 0
    /// Target-model forward passes the engine counted: the prefill's, then one per
    /// verified block. Never submitted: it is the accept length's denominator, which the
    /// management server derives from the wire's counters instead.
    var targetForwards: UInt64 = 0
    /// Whether the reply stopped on an end-of-generation token rather than the budget.
    var endedOnEos = false
    /// Whether the reply stopped at its length limit, the veneer's `STOP_LIMIT`.
    var endedOnLength = false
    /// The reply as the model emitted it, for the greedy-identity comparison. Never
    /// submitted: the comparison is computed on the device and reported as a turn count.
    var text: String = ""

    /// What the engine must have told the truth about for this turn to be countable.
    ///
    /// `generated - accepted` is no forward count (a drafter proposes past the stop token),
    /// so a drafting turn must carry the engine's own and is refused without one.
    ///
    /// Refused, never clamped: a turn clamped to zero forwards would drop out of the pooled
    /// denominator, which raises the accept length the server derives.
    func validate() throws {
        guard generatedTokens > 0 else {
            throw RuntimeError.engine("the turn emitted no tokens, so it measured nothing")
        }
        guard acceptedTokens <= draftedTokens else {
            throw RuntimeError.engine(
                "the engine reported \(acceptedTokens) accepted of \(draftedTokens) "
                    + "drafted tokens")
        }
        // The engine counts a draft as accepted only once it commits it, and the
        // prefill's token is never a draft, so a turn that accepted everything it
        // generated is an accounting bug in the loop rather than a short reply. It is
        // also what keeps the denominator the server derives from the wire positive.
        guard acceptedTokens < generatedTokens else {
            throw RuntimeError.engine(
                "the engine reported \(acceptedTokens) accepted drafts against "
                    + "\(generatedTokens) generated tokens: it counted a draft it never "
                    + "committed")
        }
        guard draftedTokens == 0 || countedForwards > 0 else {
            throw RuntimeError.engine(
                "the engine drafted \(draftedTokens) tokens and counted no decode forward: "
                    + "the accept length has no denominator")
        }
        // Every forward commits a token, so more forwards than tokens means the count
        // is not this turn's own.
        guard countedForwards <= decodeTokens else {
            throw RuntimeError.engine(
                "the engine counted \(countedForwards) decode forwards for \(decodeTokens) "
                    + "decode tokens: the count is not this turn's own")
        }
    }

    /// A length stop short of the budget is a truncated reply. The veneer reports a
    /// context cut as its own `STOP_CONTEXT`, so this only checks that `STOP_LIMIT`
    /// reached the budget.
    func checkLength(maxTokens: Int) throws {
        guard !endedOnLength || generatedTokens >= UInt64(clamping: maxTokens) else {
            throw RuntimeError.engine(
                "the engine stopped the reply for its length at \(generatedTokens) tokens, "
                    + "short of the \(maxTokens) max_tokens it was sent with")
        }
    }

    /// Tokens committed per decode forward, the prefill out of both terms. A one-token reply
    /// ran no decode loop and reads zero. Only meaningful after `validate()`.
    var acceptLength: Double {
        countedForwards == 0 ? 0 : Double(decodeTokens) / Double(countedForwards)
    }

    /// Every token but the first, which the prefill sampled before any draft existed.
    var decodeTokens: UInt64 { generatedTokens == 0 ? 0 : generatedTokens - 1 }

    /// Every target forward but the prefill's: the crate's `counted_forwards`, which its
    /// runtimes report with the prefill's forward already out.
    var countedForwards: UInt64 { targetForwards == 0 ? 0 : targetForwards - 1 }
}

nonisolated extension VlSpecDecode {
    /// One traversal, summed.
    private struct Totals {
        var totals = CorpusTotals()
        var turns: UInt64 = 0
        var draftedTokens: UInt64 = 0
        var acceptedTokens: UInt64 = 0

        /// Only sums, no branches, as upstream's `Totals`: this client and the reference one
        /// can disagree about what they observed, never about what they did with it.
        mutating func add(_ turn: TurnRecord) {
            turns += 1
            totals.promptMs += turn.promptMs
            totals.predictedMs += turn.predictedMs
            totals.ttftMs += turn.ttftMs
            totals.decodeMs += turn.decodeMs
            totals.e2eMs += turn.e2eMs
            totals.generatedTokens += turn.generatedTokens
            draftedTokens += turn.draftedTokens
            acceptedTokens += turn.acceptedTokens
        }
    }

    /// Reduce one traversal to the counters the wire carries, refusing anything the server
    /// would reject or divide by, and anything that says the row is not the cell it claims.
    ///
    /// Checked here, as upstream does, rather than left to the gateway: a multi-hour run
    /// that ends in a `400` has thrown away the measurement.
    static func counters(speculation: Speculation,
                         turns: [TurnRecord]) throws -> BenchmarkResultData {
        var sum = Totals()
        turns.forEach { sum.add($0) }

        guard sum.turns > 0 else { throw RuntimeError.engine("the traversal covered no turns") }
        guard sum.totals.predictedMs > 0 else {
            throw RuntimeError.engine(
                "the traversal reported no generation time (\(sum.totals.predictedMs)ms)")
        }
        guard sum.totals.generatedTokens > 0 else {
            throw RuntimeError.engine("the traversal emitted no tokens")
        }
        guard sum.totals.e2eMs > 0 else {
            throw RuntimeError.engine(
                "the traversal took no time on the client clock (\(sum.totals.e2eMs)ms)")
        }
        let tokS = Double(sum.totals.generatedTokens) / sum.totals.predictedMs * 1000

        switch speculation {
        case .off:
            guard sum.draftedTokens == 0, sum.acceptedTokens == 0 else {
                throw RuntimeError.engine(
                    "the run drafted \(sum.draftedTokens) tokens (\(sum.acceptedTokens) "
                        + "accepted) with speculation off: a speculative measurement under "
                        + "the baseline's name")
            }
            AppLog.jobRun.info(
                "\(speculation.label): \(sum.turns) turns, \(sum.totals.generatedTokens) "
                    + "tokens, \(tokS) tok/s")
            return .vlSpecDecode(turns: UInt32(sum.turns), totals: sum.totals,
                                   draftedTokens: nil, acceptedTokens: nil,
                                   turnAcceptLengthSum: nil)
        case .on:
            // Drafting is a property of the traversal, not of a turn: one turn may
            // legitimately draft nothing, a whole traversal may not.
            // `TurnRecord.validate` has already checked everything judged per turn.
            guard sum.draftedTokens > 0 else {
                throw RuntimeError.engine(
                    "the traversal drafted no tokens: the drafter is not wired up")
            }
            // Per turn, then summed: over `turns` the request-mean convention, which the
            // pooled ratio below is not.
            let turnAcceptLengthSum = try turns.reduce(0.0) { total, turn in
                try turn.validate()
                return total + turn.acceptLength
            }
            // The server's pooled figure counts the prefill and the drafts the stop token
            // discarded; this one is over the decode loop alone, and the two disagree.
            // `validate` kept both denominators positive, per turn.
            let decodeTokens = turns.reduce(UInt64(0)) { $0 + $1.decodeTokens }
            let countedForwards = turns.reduce(UInt64(0)) { $0 + $1.countedForwards }
            let acceptLength = Double(decodeTokens) / Double(countedForwards)
            AppLog.jobRun.info(
                "\(speculation.label): \(sum.turns) turns, \(sum.totals.generatedTokens) "
                    + "tokens over \(countedForwards) decode forwards, "
                    + "accept_length \(acceptLength) pooled, "
                    + "\(turnAcceptLengthSum / Double(sum.turns)) per request, \(tokS) tok/s")
            return .vlSpecDecode(
                turns: UInt32(sum.turns), totals: sum.totals,
                draftedTokens: sum.draftedTokens, acceptedTokens: sum.acceptedTokens,
                turnAcceptLengthSum: turnAcceptLengthSum)
        }
    }

    /// sha256 over the traversal's replies, in order: per turn, the reply's own sha256, a
    /// tab, its token count, a newline. Upstream's byte for byte, so a phone's row and a
    /// laptop's over the same corpus share a digest exactly when they said the same things.
    static func outputDigest(_ turns: [TurnRecord]) -> String {
        var lines = Data()
        for turn in turns {
            lines.append(Data(textSha256(turn.text).utf8))
            lines.append(Data("\t\(turn.generatedTokens)\n".utf8))
        }
        return Data(SHA256.hash(data: lines)).hexEncodedString()
    }

    static func textSha256(_ text: String) -> String {
        Data(SHA256.hash(data: Data(text.utf8))).hexEncodedString()
    }

    /// Every hundredth sample, so a multi-hour run says how it is going without writing a
    /// line per turn.
    static func logProgress(speculation: Speculation, done: Int, total: Int,
                            records: [TurnRecord]) {
        guard done % 100 == 0 || done == total else { return }
        let generated = records.reduce(UInt64(0)) { $0 + $1.generatedTokens }
        guard speculation.enabled else {
            AppLog.jobRun.info(
                "\(speculation.label): \(done)/\(total) samples, \(records.count) turns, "
                    + "\(generated) tokens")
            return
        }
        let decodeTokens = records.reduce(UInt64(0)) { $0 + $1.decodeTokens }
        let countedForwards = records.reduce(UInt64(0)) { $0 + $1.countedForwards }
        let acceptLength = countedForwards > 0
            ? Double(decodeTokens) / Double(countedForwards) : nil
        AppLog.jobRun.info(
            "\(speculation.label): \(done)/\(total) samples, \(records.count) turns, "
                + "accept_length so far \(acceptLength.map { "\($0)" } ?? "n/a")")
    }
}
