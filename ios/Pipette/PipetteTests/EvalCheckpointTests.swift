import Foundation
import Testing

@testable import Pipette

/// Per-sample eval resume — the crate's `EvalCompletionsStore`.
///
/// The behaviour that matters is the one a phone hits: a run killed part-way through
/// must not start again from sample zero. These drive the real `evalSamples` seam rather
/// than the store alone, since the seam is what both engines call.
struct EvalCheckpointTests {

    private func store() -> EvalCompletionsStore {
        EvalCompletionsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteEvals-\(UUID().uuidString)", isDirectory: true))
    }

    private func samples(_ ids: [String]) -> [EvalSample] {
        ids.map { EvalSample(id: $0, messages: [["role": "user", "content": "hi"]]) }
    }

    /// A `RunRequest` for an eval, varying only what a test is about.
    private func evalRequest(
        benchmarkId: String = "eval_smoke",
        path: String = "/tmp/model.gguf"
    ) throws -> RunRequest {
        payloadRequest(
            model: try ggufTextResolved(path: path),
            benchmark: .eval(benchmarkId: benchmarkId, evalId: EvalId("eval_smoke"),
                             datasetName: "local", maxTokens: 4,
                             mcqChoices: nil, samples: nil))
    }

    private func generation(_ text: String) -> EvalGeneration {
        EvalGeneration(text: text, stopReason: .eos, stopDetail: nil, completionTokens: 1)
    }

    /// The headline: a run killed after two of four samples resumes at the third, and the
    /// completions it already produced survive into the final set.
    @Test func aKilledRunResumesWhereItStopped() throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()
        let all = samples(["a", "b", "c", "d"])

        // First attempt, killed after "b". A jetsam kill ends the process mid-run, so it
        // is simulated by the run only reaching the first two samples — not by throwing,
        // which `evalSamples` deliberately turns into a failed completion instead.
        let first = try store.open(request: request)
        _ = try evalSamples(Array(all.prefix(2)), resuming: first, progress: { _ in }) { sample in
            generation("answer-\(sample.id)")
        }
        first.close()

        // Second attempt: only the unfinished samples reach the engine.
        var asked: [String] = []
        let second = try store.open(request: request)
        let completions = try evalSamples(all, resuming: second, progress: { _ in }) { sample in
            asked.append(sample.id)
            return generation("answer-\(sample.id)")
        }

        #expect(asked == ["c", "d"])
        #expect(completions.map(\.id) == ["a", "b", "c", "d"])
        // The resumed rows carry the first attempt's text, not a re-run's.
        guard case let .completed(_, text, _, _, _) = completions[0] else {
            Issue.record("expected a completed sample"); return
        }
        #expect(text == "answer-a")
    }

    /// Without a checkpoint the seam behaves exactly as before — every sample runs, and
    /// a per-sample failure is still a failed completion rather than an aborted run.
    @Test func withoutACheckpointEverySampleRuns() throws {
        var asked: [String] = []
        let completions = try evalSamples(samples(["a", "b"]), progress: { _ in }) { sample in
            asked.append(sample.id)
            if sample.id == "b" { throw NSError(domain: "t", code: 1) }
            return generation("x")
        }
        #expect(asked == ["a", "b"])
        #expect(completions.count == 2)
        guard case .failed = completions[1] else {
            Issue.record("a per-sample failure should be a failed completion"); return
        }
    }

    /// A failed sample counts as done: re-running it on resume would spend the run
    /// retrying something that already had its turn.
    @Test func aFailedSampleIsNotRetriedOnResume() throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()

        let first = try store.open(request: request)
        _ = try evalSamples(samples(["a"]), resuming: first, progress: { _ in }) { _ in
            throw NSError(domain: "t", code: 1)
        }
        first.close()

        var asked: [String] = []
        let second = try store.open(request: request)
        let completions = try evalSamples(samples(["a"]), resuming: second, progress: { _ in }) {
            asked.append($0.id)
            return generation("x")
        }
        #expect(asked.isEmpty)
        #expect(completions.count == 1)
    }

    /// The digest covers the plan coordinates and nothing else. A model re-fetched to a
    /// different path is the same run — a path-sensitive digest would rotate the
    /// checkpoint on every eviction, which is the failure this design avoids.
    @Test func theDigestIgnoresBoundPaths() throws {
        let a = try evalRequest(path: "/models/one/model.gguf")
        let b = try evalRequest(path: "/models/two/model.gguf")
        #expect(try EvalCompletionsStore.digest(of: a) == EvalCompletionsStore.digest(of: b))
    }

    /// The digest has to be stable across encodings of the same request. It was not:
    /// descriptor encoding does not sort keys, so two calls could differ and every resume
    /// rotated its own checkpoint and started from zero.
    @Test func theDigestIsStableAcrossCalls() throws {
        let request = try evalRequest()
        let first = try EvalCompletionsStore.digest(of: request)
        for _ in 0..<20 {
            #expect(try EvalCompletionsStore.digest(of: request) == first)
        }
    }

    /// A different benchmark is a different run, so it gets its own checkpoint.
    @Test func adifferentBenchmarkGetsItsOwnCheckpoint() throws {
        let a = try evalRequest(benchmarkId: "eval_smoke")
        let b = try evalRequest(benchmarkId: "eval_other")
        #expect(try EvalCompletionsStore.digest(of: a) != EvalCompletionsStore.digest(of: b))
    }

    /// A checkpoint whose header names another run is set aside, not appended to — mixing
    /// two runs' rows into one file would corrupt both.
    @Test func aForeignCheckpointIsRotatedRatherThanAppendedTo() throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()
        let digest = try EvalCompletionsStore.digest(of: request)
        let path = store.root.appendingPathComponent("\(digest.prefix).jsonl")
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        try Data(#"{"digest":"someone-else","meta":{"benchmark_id":"x","runtime":"r","model":"m"}}"#.utf8)
            .write(to: path)

        let session = try store.open(request: request)
        defer { session.close() }
        #expect(session.completions.isEmpty)
        let rotated = try FileManager.default.contentsOfDirectory(atPath: store.root.path)
            .filter { $0.contains("stale-") }
        #expect(rotated.count == 1)
    }

    // MARK: - finalize

    /// A clean run leaves nothing behind: the checkpoint is resume state, and keeping it
    /// after the result is recorded would resume a run that already finished.
    @Test func finalizeDropsTheCheckpointWhenEverySampleSucceeded() throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()
        let session = try store.open(request: request)
        _ = try evalSamples(samples(["a", "b"]), resuming: session, progress: { _ in }) { _ in
            generation("x")
        }

        let returned = session.finalize()

        #expect(returned.count == 2)
        #expect(try FileManager.default.contentsOfDirectory(atPath: store.root.path).isEmpty)
    }

    /// A failure is kept. A sample that kills the engine would otherwise be re-hit by
    /// every fresh run of the same cell; retaining it means the next run skips it.
    @Test func finalizeKeepsFailuresSoAFreshRunSkipsThem() throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()
        let first = try store.open(request: request)
        _ = try evalSamples(samples(["ok", "poison"]), resuming: first, progress: { _ in }) {
            if $0.id == "poison" { throw NSError(domain: "engine", code: 9) }
            return generation("x")
        }
        first.finalize()

        // A fresh run of the same cell re-runs the success and skips the poison sample.
        var asked: [String] = []
        let second = try store.open(request: request)
        defer { second.close() }
        _ = try evalSamples(samples(["ok", "poison"]), resuming: second, progress: { _ in }) {
            asked.append($0.id)
            return generation("x")
        }
        #expect(asked == ["ok"])
    }

    /// The seam the runtimes gained: a session opened from the store reaches the engine, so
    /// a sample an earlier attempt finished is served from the checkpoint instead of being
    /// generated again. Driven through `LlamaBenchmark.runOn` rather than `evalSamples`,
    /// because the store→engine wiring is the part that was absent — the store existed and
    /// nothing in a run opened it.
    @Test func aRuntimeResumesFromTheStoreRatherThanRegenerating() async throws {
        let store = store()
        defer { store.clear() }
        let request = try evalRequest()

        let prior = try store.open(request: request)
        try prior.append(.completed(id: "a", text: "from-checkpoint", stopReason: .eos,
                                    stopDetail: nil, completionTokens: 1))
        prior.close()

        let fake = LlamaBenchmarkTests.FakeInference()
        fake.chatResult = "freshly-generated"
        let result = try await LlamaBenchmark.runOn(
            .eval(benchmarkId: "eval_smoke", evalId: EvalId("eval_smoke"), datasetName: "local",
                  maxTokens: 4, mcqChoices: nil, samples: samples(["a", "b"])),
            fake.make(), gate: {}, observer: .ignore,
            openCheckpoint: { try store.open(request: request) })

        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        #expect(fake.chatCount == 1)                       // only "b" reached the engine
        #expect(completions.map(\.id) == ["a", "b"])
        #expect(completions.contains { if case .completed(_, "from-checkpoint", _, _, _) = $0 { true } else { false } })
    }

    /// A throughput cell leaves no checkpoint behind: `openCheckpoint` is a closure so the
    /// session is minted on the eval arm alone, not once per dispatched cell.
    @Test func aThroughputCellNeverOpensASession() async throws {
        let store = store()
        defer { store.clear() }
        var opened = 0

        _ = try await LlamaBenchmark.runOn(
            .prefillThroughput(benchmarkId: "prefill_throughput_512", prefillTokens: 512),
            LlamaBenchmarkTests.FakeInference().make(), gate: {}, observer: .ignore,
            openCheckpoint: {
                opened += 1
                return try store.open(request: try evalRequest())
            })

        #expect(opened == 0)
    }
}
