import Testing
import Foundation
@testable import Pipette

/// Unit tests for the stateless `LlamaBenchmark` core — driven through `runOn`
/// against a fake `Inference` witness (no model, no `llama` linked), so the
/// per-type dispatch, rep counts, and `BenchmarkResult` construction are verified
/// deterministically. `max_memory_usage`'s peak sampling is integration-tested on
/// device; its on-model work (`driveMaxMemory`) and the sampler are unit-tested here.
@Suite struct LlamaBenchmarkTests {

    /// Records calls and returns deterministic, content-free results so the
    /// benchmark control flow can be asserted without real inference.
    final class FakeInference {
        enum Boom: Error { case sample }
        var tokenizeCount = 0
        var prefillCount = 0
        var decodeCount = 0
        var lastDecodeCount = 0
        var chatCount = 0
        var lastChatMaxTokens = 0
        var lastChatMcq: [String]?
        var chatResult = "ok"
        /// Throw from `chatCompletion` when the messages JSON contains this marker.
        var failChatMarker: String?

        func make() -> Inference {
            Inference(
                tokenize: { text, addSpecial in
                    self.tokenizeCount += 1
                    let n = text.count + (addSpecial ? 1 : 0)
                    return (0 ..< n).map { Int32($0 % 1000) }
                },
                resetContext: {},
                resetSampler: {},
                prefill: { _ in self.prefillCount += 1 },
                decodeIgnoringEoG: { count in self.decodeCount += 1; self.lastDecodeCount = count },
                chatCompletion: { json, maxTokens, mcq in
                    self.chatCount += 1; self.lastChatMaxTokens = maxTokens; self.lastChatMcq = mcq
                    if let marker = self.failChatMarker, json.contains(marker) { throw Boom.sample }
                    return EvalGeneration(text: self.chatResult, stopReason: .unknown, stopDetail: nil, completionTokens: nil)
                })
        }
    }

    /// A gate that always proceeds (the `.ready` outcome).
    private let ready: () throws -> Void = {}

    // MARK: - Throughput dispatch (1 warmup + 5 reps)

    @Test func prefillRunsWarmupPlusReps() async throws {
        let fake = FakeInference()
        let result = try await LlamaBenchmark.runOn(
            .prefillThroughput(benchmarkId: "p_512", prefillTokens: 512), fake.make(), gate: ready, observer: .ignore)
        guard case .prefillThroughput = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)   // 1 warmup + 5 reps
        #expect(fake.decodeCount == 0)    // prefill-only never decodes
    }

    @Test func decodeTimesFixedCountWithWarmup() async throws {
        let fake = FakeInference()
        let result = try await LlamaBenchmark.runOn(
            .decodeThroughput(benchmarkId: "d_512_100", prefillTokens: 512, decodeTokens: 100),
            fake.make(), gate: ready, observer: .ignore)
        guard case .decodeThroughput = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)        // warmup + 5 reps each prefill (untimed)
        #expect(fake.decodeCount == 6)         // warmup + 5 reps each decode
        #expect(fake.lastDecodeCount == 100)   // rep decode uses the cell's decode count
    }

    @Test func endToEndRunsPrefillAndDecode() async throws {
        let fake = FakeInference()
        let result = try await LlamaBenchmark.runOn(
            .endToEndLatency(benchmarkId: "e_512_256", prefillTokens: 512, decodeTokens: 256),
            fake.make(), gate: ready, observer: .ignore)
        guard case .endToEndLatency = result else { Issue.record("got \(result)"); return }
        #expect(fake.prefillCount == 6)
        #expect(fake.decodeCount == 6)
        #expect(fake.lastDecodeCount == 256)
        // e2e tokenizes inside each measured rep (CLI-parity "true" e2e), not once
        // up front like the throughput benches — so at least one tokenize per rep.
        #expect(fake.tokenizeCount >= 5)
    }

    // MARK: - max_memory_usage on-model work

    @Test func driveMaxMemorySinglePrefillAndDecode() throws {
        let fake = FakeInference()
        try LlamaBenchmark.driveMaxMemory(fake.make(), prefillTokens: 512)
        #expect(fake.prefillCount == 1)     // no warmup, no rep loop
        #expect(fake.decodeCount == 1)
        #expect(fake.lastDecodeCount == 1)  // single decode to touch the KV path
    }

    // MARK: - eval (per-sample, no rep loop)

    @Test func evalCompletesEachSample() async throws {
        let fake = FakeInference()
        fake.chatResult = "answer"
        let samples = [
            EvalSample(id: "q1", messages: [["role": "user", "content": "hi"]]),
            EvalSample(id: "q2", messages: [["role": "user", "content": "there"]]),
        ]
        let result = try await LlamaBenchmark.runOn(
            .eval(benchmarkId: "eval", evalId: .known(.gpqaDiamond), datasetName: "gpqa",
                  maxTokens: 256, mcqChoices: nil, samples: samples), fake.make(), gate: ready, observer: .ignore)
        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        #expect(completions.count == 2)
        #expect(completions.map(\.id) == ["q1", "q2"])
        #expect(completions.allSatisfy { if case .completed(_, "answer", _, _, _) = $0 { true } else { false } })
        #expect(fake.lastChatMaxTokens == 256)   // free-form keeps the cell's budget
    }

    @Test func evalSampleFailureBecomesFailedCompletion() async throws {
        let fake = FakeInference()
        fake.failChatMarker = "BOOM"
        let samples = [
            EvalSample(id: "q1", messages: [["role": "user", "content": "fine"]]),
            EvalSample(id: "q2", messages: [["role": "user", "content": "BOOM"]]),
        ]
        let result = try await LlamaBenchmark.runOn(
            .eval(benchmarkId: "eval", evalId: .known(.gpqaDiamond), datasetName: "gpqa",
                  maxTokens: 256, mcqChoices: nil, samples: samples), fake.make(), gate: ready, observer: .ignore)
        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        guard case .completed = completions[0] else { Issue.record("q1 should have completed"); return }
        guard case .failed(_, let reason) = completions[1] else { Issue.record("q2 should have failed"); return }
        #expect(!reason.isEmpty)
    }

    @Test func evalMcqConstrainsToSingleToken() async throws {
        let fake = FakeInference()
        let samples = [EvalSample(id: "q1", messages: [["role": "user", "content": "pick"]])]
        _ = try await LlamaBenchmark.runOn(
            .eval(benchmarkId: "eval", evalId: .known(.gpqaDiamond), datasetName: "gpqa",
                  maxTokens: 256, mcqChoices: ["A", "B", "C", "D"], samples: samples),
            fake.make(), gate: ready, observer: .ignore)
        #expect(fake.lastChatMaxTokens == 1)             // MCQ → one choice token
        #expect(fake.lastChatMcq == ["A", "B", "C", "D"])
    }

    // MARK: - Gating & unsupported types

    @Test func cancelledGateThrowsBeforeReps() async {
        let fake = FakeInference()
        await #expect(throws: RuntimeError.cancelled) {
            try await LlamaBenchmark.runOn(
                .prefillThroughput(benchmarkId: "p_512", prefillTokens: 512),
                fake.make(), gate: { throw RuntimeError.cancelled }, observer: .ignore)
        }
        // Warm-up ran (ungated) but the gate fired before the first measured rep.
        #expect(fake.prefillCount == 1)
    }

    @Test func timedOutGateThrowsReadiness() async {
        let fake = FakeInference()
        await #expect(throws: RuntimeError.readiness("hot")) {
            try await LlamaBenchmark.runOn(
                .prefillThroughput(benchmarkId: "p_512", prefillTokens: 512),
                fake.make(), gate: { throw RuntimeError.readiness("hot") }, observer: .ignore)
        }
    }

    @Test func vlThroughputIsUnsupported() async {
        let fake = FakeInference()
        await #expect(throws: RuntimeError.unsupported("vl_1")) {
            try await LlamaBenchmark.runOn(
                .vlThroughput(benchmarkId: "vl_1", imageWidth: 64, imageHeight: 64,
                              textTokens: 32, decodeTokens: 16), fake.make(), gate: ready, observer: .ignore)
        }
    }

    // MARK: - Semantic progress events

    @Test func evalReportsPerSampleProgress() async throws {
        let fake = FakeInference()
        var events: [BenchmarkProgress] = []
        let samples = (1 ... 3).map { EvalSample(id: "q\($0)", messages: [["role": "user", "content": "hi"]]) }
        _ = try await LlamaBenchmark.runOn(
            .eval(benchmarkId: "eval", evalId: .known(.gpqaDiamond), datasetName: "gpqa",
                  maxTokens: 8, mcqChoices: nil, samples: samples),
            fake.make(), gate: ready, observer: .ignore, progress: { events.append($0) })
        #expect(events == [.sample(completed: 1, total: 3),
                           .sample(completed: 2, total: 3),
                           .sample(completed: 3, total: 3)])
    }

    @Test func throughputReportsPerAttemptProgress() async throws {
        let fake = FakeInference()
        var events: [BenchmarkProgress] = []
        _ = try await LlamaBenchmark.runOn(
            .decodeThroughput(benchmarkId: "d", prefillTokens: 8, decodeTokens: 4),
            fake.make(), gate: ready, observer: .ignore, progress: { events.append($0) })
        #expect(events == [.attempt(completed: 1, total: 5),
                           .attempt(completed: 2, total: 5),
                           .attempt(completed: 3, total: 5),
                           .attempt(completed: 4, total: 5),
                           .attempt(completed: 5, total: 5)])
    }

    // MARK: - Prompt-seed undersized guard

    @Test func seedThrowsWhenTokenizerCannotReachTarget() async {
        // A degenerate tokenizer that always returns a fixed small array regardless
        // of input — the bounded grow loop can never reach `target`, so seeding must
        // fail loudly rather than silently emit a short (wrong-shape) prompt.
        let inf = Inference(
            tokenize: { _, _ in [Int32](repeating: 7, count: 4) },
            resetContext: {}, resetSampler: {}, prefill: { _ in },
            decodeIgnoringEoG: { _ in },
            chatCompletion: { _, _, _ in EvalGeneration(text: "", stopReason: .unknown, stopDetail: nil, completionTokens: nil) })
        // prefill_throughput drives seedTokens(target: 512); 4 < 512 → throw.
        await #expect(throws: RuntimeError.self) {
            try await LlamaBenchmark.runOn(
                .prefillThroughput(benchmarkId: "p_512", prefillTokens: 512), inf, gate: ready, observer: .ignore)
        }
    }

    @Test func seedThrowsWhenCorpusTokenizesToEmpty() async {
        // A tokenizer that always returns an empty array — the corpus tokenizes to
        // zero tokens, which can't tile to any positive target. The empty-unit guard
        // must route to the same loud failure rather than letting an empty seed escape.
        let inf = Inference(
            tokenize: { _, _ in [] },
            resetContext: {}, resetSampler: {}, prefill: { _ in },
            decodeIgnoringEoG: { _ in },
            chatCompletion: { _, _, _ in EvalGeneration(text: "", stopReason: .unknown, stopDetail: nil, completionTokens: nil) })
        await #expect(throws: RuntimeError.self) {
            try await LlamaBenchmark.runOn(
                .prefillThroughput(benchmarkId: "p_512", prefillTokens: 512), inf, gate: ready, observer: .ignore)
        }
    }

    // MARK: - Memory peak sampler

    @Test func memoryPeakSamplerCapturesHighWaterMark() {
        let lock = NSLock()
        var scripted: [UInt64] = [100, 500, 300, 900, 200]   // 900 is the high-water mark
        var callCount = 0
        let sampler = MemoryPeakSampler(source: {
            lock.lock(); defer { lock.unlock() }
            callCount += 1
            return scripted.isEmpty ? 200 : scripted.removeFirst()
        }, intervalMs: 1)
        sampler.start()
        // Wait until the poller has actually drained every scripted value, not
        // a fixed interval: a 1ms timer coalesces to much coarser ticks on a
        // loaded CI runner, so a fixed sleep races the high-water value (900).
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            lock.lock(); let drained = scripted.isEmpty; lock.unlock()
            if drained { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
        let peak = sampler.stop()
        #expect(peak >= 900)                  // captured the high-water mark

        // stop() joins the poll thread, so no source() call may happen after it
        // returns. Snapshot the count, wait well past the poll interval, re-read.
        lock.lock(); let afterStop = callCount; lock.unlock()
        Thread.sleep(forTimeInterval: 0.1)
        lock.lock(); let settled = callCount; lock.unlock()
        #expect(settled == afterStop)         // the polling thread was joined
    }
}
