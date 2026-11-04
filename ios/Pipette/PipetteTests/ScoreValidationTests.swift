import Foundation
import Testing

@testable import Pipette

/// The completion-id pre-flight, mirroring `pipette-cli`'s `score_validation.rs` test for
/// test.
struct ScoreValidationTests {
    private func payload(_ benchmarkId: String, _ ids: [String]) -> [String: Any] {
        ["benchmark_id": benchmarkId,
         "completions": ids.map { ["id": $0, "completion": "c-\($0)"] }]
    }

    private func ids(in payload: [String: Any]) -> [String] {
        (payload["completions"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? String }
    }

    private func completion(for id: String, in payload: [String: Any]) -> String? {
        (payload["completions"] as? [[String: Any]] ?? [])
            .first { $0["id"] as? String == id }?["completion"] as? String
    }

    @Test func aUniquePayloadPassesThroughUnchanged() throws {
        var value = payload("eval_x", ["a", "b", "c"])
        let before = NSDictionary(dictionary: value)
        #expect(try dedupeCompletionIds(&value) == nil)
        #expect(NSDictionary(dictionary: value) == before)
    }

    /// A non-eval payload (prefill, say) has no `completions`.
    @Test func aPayloadWithoutCompletionsPassesThrough() throws {
        var value: [String: Any] = ["benchmark_id": "prefill_throughput_512", "prefill_time_ms": 12.3]
        let before = NSDictionary(dictionary: value)
        #expect(try dedupeCompletionIds(&value) == nil)
        #expect(NSDictionary(dictionary: value) == before)
    }

    /// Last write wins, matching the checkpoint's append-then-load, and the survivors keep
    /// their original positions.
    @Test func theLastOccurrenceIsKept() throws {
        var value: [String: Any] = [
            "benchmark_id": "eval_x",
            "completions": [
                ["id": "a", "completion": "first-a"],
                ["id": "b", "completion": "only-b"],
                ["id": "a", "completion": "second-a"],
                ["id": "c", "completion": "only-c"],
            ],
        ]
        let report = try #require(try dedupeCompletionIds(&value))
        #expect(report.before == 4)
        #expect(report.after == 3)
        #expect(report.distinctDuplicates == 1)
        #expect(ids(in: value) == ["b", "a", "c"])
        #expect(completion(for: "a", in: value) == "second-a")
    }

    @Test func theSummaryNamesTheBenchmarkAndTheListedIds() throws {
        var value = payload("eval_ifstruct_original", ["a", "b", "a", "c", "b"])
        let summary = try #require(try dedupeCompletionIds(&value)).summary
        #expect(summary.contains("eval_ifstruct_original"), "\(summary)")
        #expect(summary.contains("[a, b]"), "\(summary)")
        // Two distinct duplicates, so no truncation note.
        #expect(!summary.contains("(and "), "\(summary)")
    }

    @Test func theListedIdsAreTruncatedAboveTen() throws {
        let unique = (0..<12).map { String(format: "id%02d", $0) }
        var value = payload("eval_ifstruct_original", unique + unique)
        let report = try #require(try dedupeCompletionIds(&value))
        #expect(report.distinctDuplicates == 12)
        #expect(report.listedDuplicateIds.count == maxListedDuplicates)
        #expect(report.summary.contains("(and 2 more)"))
    }

    /// The production shape: every id twice. After the repair the payload satisfies the
    /// scorer's uniqueness rule.
    @Test func aDoubledIFStructPayloadIsDedupedToUniqueIds() throws {
        let unique = (0..<2000).map { String(format: "sample%04d", $0) }
        var value = payload("eval_ifstruct_original", unique + unique)
        let report = try #require(try dedupeCompletionIds(&value))
        #expect(report.before == 4000)
        #expect(report.after == 2000)
        #expect(report.distinctDuplicates == 2000)
        let final = ids(in: value)
        #expect(final.count == Set(final).count)
        #expect(final.count == 2000)
    }

    /// An entry the server cannot key to a sample is a bug to report, not a row to drop.
    @Test func aCompletionWithoutAnIdIsRefusedWithTheBenchmarkNamed() {
        var value: [String: Any] = [
            "benchmark_id": "eval_x", "completions": [["completion": "no id here"]],
        ]
        #expect(throws: ScoreValidationError.completionMissingId(benchmarkId: "eval_x")) {
            try dedupeCompletionIds(&value)
        }
        let message = ScoreValidationError.completionMissingId(benchmarkId: "eval_x")
            .localizedDescription
        #expect(message.contains("missing string `id`"))
    }

    @Test func aFullSizeUniquePayloadIsANoOp() throws {
        var value = payload("eval_ifstruct_original", (0..<2000).map { String(format: "sample%04d", $0) })
        #expect(try dedupeCompletionIds(&value) == nil)
        #expect(ids(in: value).count == 2000)
    }
}
