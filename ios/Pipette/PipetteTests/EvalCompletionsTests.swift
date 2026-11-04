import Foundation
import Testing

@testable import Pipette

/// Per-sample eval resume: the crate's `eval_completions.rs` tests, by name, plus this
/// client's torn-line repair. The sample loop is `EvalMeasurementTests`'.
struct EvalCompletionsTests {

    private func store() -> EvalCompletionsStore {
        EvalCompletionsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteEvals-\(UUID().uuidString)", isDirectory: true))
    }

    private func samples(_ ids: [String]) -> [EvalSample] {
        ids.map { EvalSample(id: $0, messages: [["role": "user", "content": "hi"]]) }
    }

    private func evalRequest(
        benchmarkId: String = "eval_smoke",
        path: String = "/tmp/model.gguf"
    ) throws -> RunRequest {
        payloadRequest(
            model: try ggufTextResolved(path: path),
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: benchmarkId),
                parameterEvalId: EvalId("eval_smoke"),
                parameterDatasetName: "local",
                parameterMaxTokens: 4,
                parameterMcqChoices: nil,
                samples: nil)))
    }

    private func completion(_ id: String) -> BenchmarkEvalCompletion {
        .completed(id: id, completion: "ans-\(id)", stopReason: .eos, stopDetail: nil,
                   completionTokens: 1)
    }

    /// The checkpoint files under `store`, stale rotations aside.
    private func checkpoints(_ store: EvalCompletionsStore) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: store.root.path)
            .filter { $0.hasSuffix(".jsonl") }
    }

    /// A kill between `write` and `synchronize` leaves a line with no newline. Resume
    /// skips it, but appending onto it would splice two records into one unparseable line
    /// and lose the *good* one too, so the next resume would re-run a second sample.
    /// Closing the torn line first keeps one interruption to one lost sample.
    @Test func aTornLastLineDoesNotSwallowTheNextRecord() async throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()

        let first = try store.open(request)
        try first.append(completion("a"))
        first.close()

        let path = try #require(FileManager.default
            .enumerator(at: store.root, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }
            .first { $0.pathExtension == "jsonl" })
        let torn = try String(contentsOf: path, encoding: .utf8) + #"{"id":"b","text":"half"#
        try torn.write(to: path, atomically: true, encoding: .utf8)

        let second = try store.open(request)
        try second.append(completion("c"))
        second.close()

        let third = try store.open(request)
        defer { third.close() }
        #expect(third.doneIds == ["a", "c"])
    }

    /// A model re-fetched to a different path is the same run: a path-sensitive digest
    /// would rotate the checkpoint on every eviction.
    @Test func digestIgnoresBoundPathsAndSource() throws {
        let a = try evalRequest(path: "/models/one/model.gguf")
        let b = try evalRequest(path: "/models/two/model.gguf")
        #expect(try runRequestDigest(a) == runRequestDigest(b))
    }

    /// The digest hashes sorted-key JSON: an unstable digest would make every resume
    /// rotate its own checkpoint and start from zero.
    @Test func digestStableForSameDeclaredRequest() throws {
        let request = try evalRequest()
        let first = try runRequestDigest(request)
        for _ in 0..<20 {
            #expect(try runRequestDigest(request) == first)
        }
    }

    /// Appending another run's rows to the file would corrupt both runs.
    @Test func digestMismatchRotatesAndStartsFresh() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let digest = try runRequestDigest(request)
        let path = store.root.appendingPathComponent("\(digestPrefix(digest)).jsonl")
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        try Data(#"{"digest":"someone-else","meta":{"benchmark_id":"x","runtime":"r","model":"m"}}"#.utf8)
            .write(to: path)

        let session = try store.open(request)
        defer { session.close() }
        #expect(session.completions.isEmpty)
        let rotated = try FileManager.default.contentsOfDirectory(atPath: store.root.path)
            .filter { $0.contains("stale-") }
        #expect(rotated.count == 1)
    }

    /// Appends resume, a clean run's finalize removes the file, and clearing is idempotent.
    @Test func openAppendResumeFinalizeAndClear() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()

        let first = try store.open(request)
        try first.append(completion("s1"))
        #expect(first.contains("s1") && !first.contains("s2"))
        first.close()
        let second = try store.open(request)
        #expect(second.contains("s1"))
        try second.append(completion("s2"))
        #expect(second.intoCompletions().count == 2)

        try store.finalize(try runRequestDigest(request))
        #expect(try checkpoints(store).isEmpty)

        let third = try store.open(request)
        try third.append(completion("s1"))
        third.close()
        try store.clear()
        #expect(!FileManager.default.fileExists(atPath: store.root.path))
        try store.clear()
    }

    /// Finalize rewrites the file to its failed rows, so a reopen skips the completed ids and
    /// sees the failures again, with the counts the progress line and the summary read.
    @Test func finalizeKeepsFailedOnlyAndReopenSkipsCompleted() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let session = try store.open(request)
        try session.append(completion("done"))
        try session.append(.failed(id: "bad", failedReason: "crash"))
        #expect(session.failedIds == ["bad"])
        #expect(session.intoCompletions().count == 2)
        try store.finalize(try runRequestDigest(request))

        let reopened = try store.open(request)
        defer { reopened.close() }
        #expect(!reopened.contains("done"))
        #expect(reopened.contains("bad"))
        #expect(reopened.failedCount == 1)
    }

    /// Finalizing twice, or with no run at all, does nothing.
    @Test func aCheckpointOutlivesItsSessionUntilFinalized() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let digest = try runRequestDigest(request)
        try store.finalize(digest)
        let session = try store.open(request)
        for id in ["s1", "s2"] {
            try session.append(completion(id))
        }
        #expect(session.intoCompletions().count == 2)

        let resumed = try store.open(request)
        #expect(resumed.contains("s1") && resumed.contains("s2"))
        resumed.close()

        try store.finalize(digest)
        try store.finalize(digest)
        let reopened = try store.open(request)
        defer { reopened.close() }
        #expect(!reopened.contains("s1"))
    }

    /// The eval entry appends from several tasks behind a lock: the counters the progress
    /// tick and the summary read stay exact, and a resumed open targets exactly the failed ids.
    @Test func concurrentAppendsUnderMutexKeepCountsConsistent() async throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let session = try store.open(request)
        let lock = NSLock()
        let (workers, perWorker) = (8, 25)
        let id = { (n: Int) in String(format: "s%03d", n) }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0 ..< workers {
                group.addTask {
                    for i in 0 ..< perWorker {
                        let n = worker * perWorker + i
                        let row: BenchmarkEvalCompletion = n.isMultiple(of: 5)
                            ? .failed(id: id(n), failedReason: "boom")
                            : .completed(id: id(n), completion: "ans-\(n)", stopReason: .eos,
                                         stopDetail: nil, completionTokens: 1)
                        try lock.withLock { try session.append(row) }
                    }
                }
            }
            try await group.waitForAll()
        }

        let total = workers * perWorker
        #expect(session.completions.count == total)
        #expect(session.failedCount == total / 5)
        #expect((0 ..< total).allSatisfy { session.contains(id($0)) })

        _ = session.intoCompletions()
        try store.finalize(try runRequestDigest(request))
        let resumed = try store.open(request)
        defer { resumed.close() }
        #expect(resumed.failedCount == total / 5)
        #expect((0 ..< total).allSatisfy { resumed.contains(id($0)) == $0.isMultiple(of: 5) })
    }

    /// Set aside as a resume would; nothing is deleted.
    @Test func finalizeSetsAnotherRunsCheckpointAside() throws {
        let store = store()
        defer { try? store.clear() }
        let digest = try runRequestDigest(try evalRequest())
        let path = store.root.appendingPathComponent("\(digestPrefix(digest)).jsonl")
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        let foreign = Data(#"{"digest":"someone-else","meta":{"benchmark_id":"x","runtime":"r","model":"m"}}"#.utf8)
        try foreign.write(to: path)

        try store.finalize(digest)

        let files = try FileManager.default.contentsOfDirectory(atPath: store.root.path)
        #expect(files.count == 1)
        let rotated = try #require(files.first { $0.contains("stale-") })
        #expect(try Data(contentsOf: store.root.appendingPathComponent(rotated)) == foreign)
    }

    /// It stays on disk, and the caller hears why.
    @Test func anUnreadableCheckpointIsNotFinalized() throws {
        let store = store()
        defer { try? store.clear() }
        let digest = try runRequestDigest(try evalRequest())
        let path = store.root.appendingPathComponent("\(digestPrefix(digest)).jsonl")
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        try Data("not json\n".utf8).write(to: path)

        #expect(throws: EvalCompletionsError.unreadable(
            path: path.path, reason: "failed to parse completion header")) {
            try store.finalize(digest)
        }
        #expect(FileManager.default.fileExists(atPath: path.path))
    }

    /// `LlamaExecute.run` opens the store the same way; this drives the kernel directly
    /// because `run` needs real weights.
    @Test func aRuntimeResumesFromTheStoreRatherThanRegenerating() async throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()

        let prior = try store.open(request)
        try prior.append(.completed(id: "a", completion: "from-checkpoint", stopReason: .eos,
                                    stopDetail: nil, completionTokens: 1))
        prior.close()

        let fake = LlamaExecuteTests.FakeInference()
        fake.chatResult = "freshly-generated"
        let benchmark = EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_smoke"),
            parameterEvalId: EvalId("eval_smoke"),
            parameterDatasetName: "local",
            parameterMaxTokens: 4,
            samples: samples(["a", "b"]))
        let result = try await LlamaExecute.eval(
            benchmark, cell: try EvalCell(benchmark), fake.make(),
            checkpoint: try store.open(request), isCancelled: { false },
            progress: { _ in })

        guard case .eval(let completions) = result else { Issue.record("got \(result)"); return }
        #expect(fake.chatCount == 1)                       // only "b" reached the engine
        #expect(completions.map(\.id) == ["a", "b"])
        #expect(completions.contains { if case .completed(_, "from-checkpoint", _, _, _) = $0 { true } else { false } })
    }

    /// A run leaves its checkpoint for the caller to finalize, so running the cell again
    /// before then generates nothing.
    @Test func anEvalRunLeavesItsCheckpointForTheCaller() async throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let fake = LlamaExecuteTests.FakeInference()
        let benchmark = EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_smoke"),
            parameterEvalId: EvalId("eval_smoke"),
            parameterDatasetName: "local",
            parameterMaxTokens: 4,
            samples: samples(["a", "b"]))
        func run() async throws -> [String] {
            let result = try await LlamaExecute.eval(
                benchmark, cell: try EvalCell(benchmark), fake.make(),
                checkpoint: try store.open(request), isCancelled: { false },
                progress: { _ in })
            guard case .eval(let completions) = result else { return [] }
            return completions.map(\.id)
        }

        #expect(try await run() == ["a", "b"])
        #expect(try await run() == ["a", "b"])
        #expect(fake.chatCount == 2, "the second run generated again")
    }

    // MARK: - The crate's resume-identity cases

    /// No eval variant of the benchmark flags exists, so tests pass a timing one.
    private func evalRequest(flags: RuntimeFlags?,
                             benchmarkFlags: BenchmarkFlags? = nil) throws -> RunRequest {
        let base = try evalRequest()
        return RunRequest(runtime: base.runtime, model: base.model, runtimeFlags: flags,
                          benchmarkFlags: benchmarkFlags, benchmark: base.benchmark)
    }

    private func evalFlags(ctxSize: UInt32? = nil, numberGpuLayers: UInt32? = nil) -> RuntimeFlags {
        .evalLlamacppIosPipetteGgufText(numberGpuLayers: numberGpuLayers, ctxSize: ctxSize,
                                        nUbatch: nil, threads: nil, swaFull: nil, loadMode: nil)
    }

    /// Retuning the context must not orphan a checkpoint.
    @Test func perfOnlyFlagsDoNotChangeTheResumeIdentity() throws {
        let baseline = try runRequestDigest(evalRequest(flags: evalFlags()))
        for ctx: UInt32 in [2048, 8448, 589_824] {
            #expect(try runRequestDigest(evalRequest(flags: evalFlags(ctxSize: ctx)))
                    == baseline, "ctx_size=\(ctx) should resume the same checkpoint")
        }
        // The crate's `a_timeout_does_not_change_the_resume_identity`: with no timeout declared
        // here, what is left is flags with no knob set.
        let bare = try evalRequest(flags: evalFlags(),
                                   benchmarkFlags: .prefillThroughputLlamacppIosPipetteGgufText(readiness: nil))
        #expect(try runRequestDigest(bare) == baseline,
                "benchmark flags with no knob set should resume the same checkpoint")
    }

    /// Otherwise a resume would mix incompatible runs: a device knob, a longer budget,
    /// another benchmark.
    @Test func aSemanticChangeStillChangesTheResumeIdentity() throws {
        let baseline = try runRequestDigest(evalRequest(flags: evalFlags()))
        #expect(try runRequestDigest(evalRequest(flags: evalFlags(numberGpuLayers: 0)))
                != baseline, "a knob outside the throughput-only set must still change the digest")
        let base = try evalRequest(flags: evalFlags())
        guard case .eval(var longer) = base.benchmark else { Issue.record("expected an eval"); return }
        longer.parameterMaxTokens = 8
        let changed = RunRequest(runtime: base.runtime, model: base.model,
                                 runtimeFlags: base.runtimeFlags, benchmarkFlags: nil,
                                 benchmark: .eval(longer))
        #expect(try runRequestDigest(changed) != baseline,
                "a changed benchmark definition must still change the digest")
        #expect(try runRequestDigest(evalRequest(benchmarkId: "eval_smoke"))
                != runRequestDigest(evalRequest(benchmarkId: "eval_other")))
    }

    /// Absent flags stay a distinct cell from flags whose only knob was cleared.
    @Test func noFlagsAtAllIsStillItsOwnIdentity() throws {
        #expect(try runRequestDigest(evalRequest(flags: evalFlags(ctxSize: 4096)))
                != runRequestDigest(evalRequest(flags: nil)))
    }

    @Test func otherBenchmarkFlagsStayInTheResumeIdentity() throws {
        let gated = try evalRequest(flags: nil, benchmarkFlags: .prefillThroughputLlamacppIosPipetteGgufText(
            readiness: ReadinessOverrides(maxWaitSecs: 1800)))
        #expect(try runRequestDigest(gated) != runRequestDigest(evalRequest(flags: nil)),
                "readiness must stay in the identity")
    }

    /// A vault source's `custom_label` names nothing, so a labelled claim of a cell resumes
    /// the checkpoint an unlabelled one left.
    @Test func aCustomLabelDoesNotChangeTheResumeIdentity() throws {
        let base = try evalRequest()
        func digest(_ model: Model) throws -> EvalRunDigest {
            try runRequestDigest(RunRequest(
                runtime: base.runtime,
                model: DeclaredBound(declared: model, bound: base.model.bound),
                runtimeFlags: nil, benchmarkFlags: nil, benchmark: base.benchmark))
        }
        for (plain, labelled) in try vaultSpecPairs() {
            #expect(try digest(labelled) == digest(plain), "\(ModelType.of(plain))")
        }
    }

    /// Resume compatibility: a checkpoint written by any earlier build has to keep resuming,
    /// so the digest of a fixed request is a wire constant. The runtime is spelled out rather
    /// than taken from `thisBuild`, whose llama.cpp commit moves with every submodule bump.
    @Test func digestIsPinnedAcrossBuilds() throws {
        let digest = try runRequestDigest(pinnedRequest()).value
        #expect(digest == "5636e3a78666ef75861f199d9b0aff212a9ec804e51b9408d3c481afe1c7ea19",
                "unless the request changed, resume just broke for every checkpoint on disk")
    }

    private func pinnedRequest() throws -> RunRequest {
        let runtime = Runtime.llamacppIosPipette(
            source: SourceRepository(repositoryUrl: RepositoryUrl(LlamaPins.main.repositoryUrl),
                                     repositoryVersion: NonEmptyString(validated: "b1234")),
            flavor: .iosArm64, privateApis: .stock)
        return RunRequest(
            runtime: .alreadyBound(runtime),
            model: .alreadyBound(try ggufTextSpec("org/model-GGUF", "model.gguf")),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "bench-a"),
                parameterEvalId: EvalId("math_500"),
                parameterDatasetName: "math_500",
                parameterMaxTokens: 16,
                parameterMcqChoices: nil,
                samples: [EvalSample(id: "s1", messages: [["role": "user", "content": "hi"]])])))
    }

    /// A crash-collateral failure overwritten by a success stops counting as failed.
    /// Otherwise the sample would be scored as a failure and skipped on every resume despite
    /// succeeding.
    @Test func aLaterSuccessClearsAnEarlierFailureForTheSameId() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try evalRequest()
        let session = try store.open(request)
        try session.append(.failed(id: "collateral", failedReason: "engine crashed mid-completion"))
        #expect(session.failedCount == 1)
        #expect(session.contains("collateral"))

        try session.append(.completed(id: "collateral", completion: "ok", stopReason: .eos,
                                      stopDetail: nil, completionTokens: 1))
        #expect(session.failedCount == 0)
        #expect(session.completions.count == 1, "the row is replaced, not doubled")

        // Nothing failed, so finalize deletes the file and a resume starts clean: the
        // append-only tail must not resurrect the failure.
        let all = session.intoCompletions()
        #expect(all.count == 1)
        guard case .completed = all[0] else { Issue.record("expected the success to win"); return }
        try store.finalize(try runRequestDigest(request))
        let reopened = try store.open(request)
        defer { reopened.close() }
        #expect(!reopened.contains("collateral"))
    }

    /// `head -1` names the run by its declared coordinates' `Display`, as the crate writes
    /// the header, rather than by the descriptor JSON.
    @Test func theHeaderNamesTheDeclaredCoordinates() throws {
        let store = store()
        defer { try? store.clear() }
        let request = try pinnedRequest()
        try store.open(request).close()

        let digest = try runRequestDigest(request)
        let text = try String(contentsOf: store.root.appendingPathComponent("\(digestPrefix(digest)).jsonl"),
                              encoding: .utf8)
        let header = try #require(text.split(separator: "\n").first)
        let object = try JSONSerialization.jsonObject(with: Data(header.utf8)) as? [String: Any]
        let meta = try #require(object?["meta"] as? [String: String])
        #expect(meta["benchmark_id"] == "bench-a")
        #expect(meta["runtime"] == "github.com/ggml-org/llama.cpp@b1234:ios-arm64")
        #expect(meta["model"] == "org/model-GGUF:model.gguf")
    }
}
