import Foundation
import Testing

@testable import Pipette

/// Ports of pipette-ops' `measurement/eval.rs` tests. `a_panicking_attempt_reaches_the_caller`
/// has no port: a Swift trap ends the process.
@Suite struct EvalMeasurementTests {
    /// An eval cell over `ids`, and the store its checkpoint opens in.
    private func cell(_ ids: [String]) throws -> (EvalCompletionsStore, RunRequest, EvalBenchmark) {
        let store = EvalCompletionsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteEvalEntry-\(UUID().uuidString)", isDirectory: true))
        let benchmark = withSamples(ids.map {
            EvalSample(id: $0, messages: [["role": "user", "content": $0]])
        })
        return (store, payloadRequest(model: try ggufTextResolved(), benchmark: .eval(benchmark)),
                benchmark)
    }

    private func withSamples(_ samples: [EvalSample]?) -> EvalBenchmark {
        EvalBenchmark(benchmarkId: BenchmarkId(validated: "eval_entry"),
                      parameterEvalId: EvalId("math_500"), parameterDatasetName: "d",
                      parameterMaxTokens: 8, samples: samples)
    }

    private func completed(_ id: String) -> EvalMeasurementReading {
        .completed(completion: "answer to \(id)", stopReason: .eos, stopDetail: nil,
                   completionTokens: 3)
    }

    private func answerAll(_ samples: [EvalMeasurementSample],
                           _ record: EvalMeasurementRecord) throws {
        for (at, sample) in samples.enumerated() {
            try record(at, completed(sample.id))
        }
    }

    private func rows(_ data: BenchmarkResultData) throws -> [BenchmarkEvalCompletion] {
        guard case let .eval(completions) = data else {
            throw EvalTestFailure("expected eval result data, got \(data)")
        }
        return completions
    }

    private func sortedIds(_ rows: [BenchmarkEvalCompletion]) -> [String] {
        rows.map(\.id).sorted()
    }

    /// Whatever the mode, a run sends each sample the checkpoint lacks once and reports every
    /// row, the resumed ones included.
    @Test(arguments: [EvalMeasurementInFlight.batch(1), .batch(3), .parallel(1), .parallel(3)])
    func everySampleRunsOnceAndResumeSkipsTheDone(_ inFlight: EvalMeasurementInFlight) async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2", "s3", "s4", "s5", "s6"])
        defer { try? store.clear() }
        let earlier = try store.open(request)
        for id in ["s1", "s4"] {
            try earlier.append(.completed(id: id, completion: "earlier", stopReason: .eos,
                                          stopDetail: nil, completionTokens: nil))
        }
        earlier.close()

        let sent = EvalTestLog<String>()
        let data = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: inFlight, checkpoint: try store.open(request), ctx: ()
        ) { _, samples, record in
            sent.append(samples.map(\.id))
            try answerAll(samples, record)
        }

        #expect(sent.items.sorted() == ["s0", "s2", "s3", "s5", "s6"])
        #expect(sortedIds(try rows(data)) == ["s0", "s1", "s2", "s3", "s4", "s5", "s6"])
    }

    @Test func aBatchCarriesUpToItsSizeInDatasetOrder() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2", "s3", "s4", "s5", "s6"])
        defer { try? store.clear() }
        let groups = EvalTestLog<[String]>()
        _ = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .batch(3), checkpoint: try store.open(request), ctx: ()
        ) { _, samples, record in
            groups.append([samples.map(\.id)])
            try answerAll(samples, record)
        }
        #expect(groups.items == [["s0", "s1", "s2"], ["s3", "s4", "s5"], ["s6"]])
    }

    /// A row is on disk the moment its sample is recorded, before the attempt carrying it
    /// returns, so a batch that dies halfway keeps the samples it finished.
    @Test func aBatchKeepsWhatItRecordedBeforeItFailed() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2", "s3"])
        defer { try? store.clear() }
        await #expect(throws: ServerWentAway.self) {
            _ = try await BenchmarkMeasurement.eval(
                benchmark, inFlight: .batch(4), checkpoint: try store.open(request),
                ctx: ()
            ) { _, _, record in
                try record(2, completed("s2"))
                let onDisk = try store.open(request)
                defer { onDisk.close() }
                guard onDisk.contains("s2") else { throw EvalTestFailure("s2 is not on disk yet") }
                try record(0, completed("s0"))
                throw ServerWentAway()
            }
        }

        let resumed = try store.open(request)
        defer { resumed.close() }
        #expect(resumed.contains("s0") && resumed.contains("s2"))
        #expect(!resumed.contains("s1") && !resumed.contains("s3"))
    }

    /// A sample that fails on its own is a row, not an abort: the cell goes on, and the row
    /// says why, in the shape the scorer and later runs read.
    @Test func aFailedSampleIsARowAndTheCellGoesOn() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2"])
        defer { try? store.clear() }
        let data = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(1), checkpoint: try store.open(request),
            ctx: ()
        ) { _, samples, record in
            for (at, sample) in samples.enumerated() {
                try record(at, sample.id == "s1"
                    ? .failed(reason: "context overflow") : completed(sample.id))
            }
        }

        let rows = try rows(data)
        #expect(sortedIds(rows) == ["s0", "s1", "s2"])
        #expect(rows.first(where: { $0.id == "s1" }) == .failed(id: "s1", failedReason: "context overflow"))
        #expect(rows.first(where: { $0.id == "s0" }) == .completed(
            id: "s0", completion: "answer to s0", stopReason: .eos, stopDetail: nil, completionTokens: 3))
    }

    /// A crash takes down every sample in flight with it, so the samples lost beside others
    /// are sent again, each alone once nothing else runs: the one that crashes the engine
    /// again is recorded as failed, the others as what they answer. Its peers cost no row,
    /// and it is sent twice, not three times.
    @Test(arguments: [(EvalMeasurementInFlight.batch(3), false), (.parallel(3), true)])
    func samplesLostBesideOthersRunAgainAlone(_ inFlight: EvalMeasurementInFlight,
                                              _ pooled: Bool) async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2", "s3", "s4", "s5"])
        defer { try? store.clear() }
        let engine = CrashingEngine()
        let (started, crashed) = (EvalTestLatch(3), EvalTestLatch(3))

        let data = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: inFlight, checkpoint: try store.open(request), ctx: ()
        ) { _, samples, record in
            let firstRound = pooled && engine.nextAttempt() < 3
            engine.start(samples)
            if firstRound { try await started.wait() }
            if samples.contains(where: { $0.id == "s1" }) { engine.crash() }
            if firstRound { try await crashed.wait() }
            let lost = engine.finish()
            for (at, sample) in samples.enumerated() {
                try record(at, lost ? .lost(reason: "server crashed") : completed(sample.id))
            }
        }

        let rows = try rows(data)
        #expect(sortedIds(rows) == ["s0", "s1", "s2", "s3", "s4", "s5"])
        for row in rows {
            let failed = if case .failed = row { true } else { false }
            #expect(failed == (row.id == "s1"), "\(row)")
        }
        #expect(rows.first(where: { $0.id == "s1" }) == .failed(id: "s1", failedReason: "server crashed"))

        let sends = engine.sends
        let times = { (id: String) in sends.filter { $0.id == id }.count }
        #expect(times("s1") == 2, "\(sends)")
        #expect((times("s0"), times("s2")) == (2, 2), "\(sends)")
        #expect((times("s3"), times("s4"), times("s5")) == (1, 1, 1), "\(sends)")
        // The resends are the three sends after the first three.
        try #require(sends.count >= 6, "\(sends)")
        for send in sends[3 ..< 6] {
            #expect(["s0", "s1", "s2"].contains(send.id) && send.size == 1 && send.beside == 1,
                    "\(send.id) was not sent again alone: \(sends)")
        }
    }

    /// A sample lost while it already ran alone is what the engine went away on: recorded as
    /// failed at once rather than sent again, and the cell goes on.
    @Test(arguments: [EvalMeasurementInFlight.batch(1), .parallel(1)])
    func aSampleLostAloneIsFailedAtOnce(_ inFlight: EvalMeasurementInFlight) async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2"])
        defer { try? store.clear() }
        let sent = EvalTestLog<String>()
        let data = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: inFlight, checkpoint: try store.open(request), ctx: ()
        ) { _, samples, record in
            sent.append(samples.map(\.id))
            for (at, sample) in samples.enumerated() {
                try record(at, sample.id == "s1"
                    ? .lost(reason: "server crashed") : completed(sample.id))
            }
        }
        #expect(sent.items == ["s0", "s1", "s2"])
        let failed = try rows(data).filter { if case .failed = $0 { true } else { false } }
        #expect(failed.map(\.id) == ["s1"])
    }

    /// A pool refills a slot the moment its sample lands: the others run while one sample is
    /// still generating, instead of waiting for it as a batch would.
    @Test(.timeLimit(.minutes(1)))
    func aPoolRefillsWhileALongSampleRuns() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2", "s3", "s4", "s5"])
        defer { try? store.clear() }
        let everyone = EvalTestLatch(6)
        _ = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(2), checkpoint: try store.open(request),
            ctx: ()
        ) { _, samples, record in
            if samples.contains(where: { $0.id == "s0" }) {
                try await everyone.wait()
            } else {
                everyone.arrive()
            }
            try answerAll(samples, record)
        }
    }

    /// A throw ends the cell: it reaches the caller unwrapped, the rows recorded before it
    /// stay for the next run, and what the attempts kept in `ctx` is the caller's to read.
    @Test func anAttemptErrorEndsTheCellAndKeepsWhatCameBefore() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2"])
        defer { try? store.clear() }
        let seen = EvalTestLog<String>()
        await #expect(throws: ServerWentAway.self) {
            _ = try await BenchmarkMeasurement.eval(
                benchmark, inFlight: .parallel(1), checkpoint: try store.open(request),
                ctx: seen
            ) { seen, samples, record in
                seen.append(samples.map(\.id))
                if samples.contains(where: { $0.id == "s1" }) { throw ServerWentAway() }
                try answerAll(samples, record)
            }
        }

        #expect(seen.items == ["s0", "s1"])
        let resumed = try store.open(request)
        defer { resumed.close() }
        #expect(resumed.contains("s0") && !resumed.contains("s1") && !resumed.contains("s2"))
    }

    @Test(arguments: [
        ([0, 0], EvalMeasurementError.recordedTwice(id: "s0")),
        ([], .unrecorded(id: "s0")),
        ([5], .outOfRange(at: 5, count: 1)),
    ] as [([Int], EvalMeasurementError)])
    func aRecordingTheEntryCannotUseFailsTheCell(_ positions: [Int],
                                                 _ expected: EvalMeasurementError) async throws {
        let (store, request, benchmark) = try cell(["s0"])
        defer { try? store.clear() }
        await #expect(throws: expected) {
            _ = try await BenchmarkMeasurement.eval(
                benchmark, inFlight: .parallel(1), checkpoint: try store.open(request),
                ctx: ()
            ) { _, _, record in
                for at in positions { try record(at, completed("s0")) }
            }
        }
    }

    /// A dataset the entry cannot read fails before any sample is sent. A row without an id
    /// or messages does not decode here, so a missing list is the one case left.
    @Test func anUnreadableDatasetFailsBeforeAnythingRuns() async throws {
        let (store, request, _) = try cell([])
        defer { try? store.clear() }
        let sent = EvalTestLog<String>()
        await #expect(throws: EvalMeasurementError.missingSamples) {
            _ = try await BenchmarkMeasurement.eval(
                withSamples(nil), inFlight: .parallel(1),
                checkpoint: try store.open(request), ctx: ()
            ) { _, samples, record in
                sent.append(samples.map(\.id))
                try answerAll(samples, record)
            }
        }
        #expect(sent.items.isEmpty)
    }

    /// An id listed twice is one row, so it runs once.
    @Test func anIdListedTwiceRunsOnce() async throws {
        let (store, request, benchmark) = try cell(["a", "b", "a"])
        defer { try? store.clear() }
        let sent = EvalTestLog<String>()
        let data = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(2), checkpoint: try store.open(request),
            ctx: ()
        ) { _, samples, record in
            sent.append(samples.map(\.id))
            try answerAll(samples, record)
        }
        #expect(sent.items.sorted() == ["a", "b"])
        #expect(sortedIds(try rows(data)) == ["a", "b"])
    }

    /// The heartbeat exits once told, and beats only when an interval has passed. The
    /// injected sleep raises `stop` after `polls` wake-ups.
    @Test(arguments: [(0, Duration.zero, 0), (3, .zero, 3), (3, .seconds(30), 0)]
          as [(Int, Duration, Int)])
    func heartbeatStopsWhenTold(_ polls: Int, _ interval: Duration,
                                _ expectedBeats: Int) async {
        var stop = polls == 0
        var slept = 0
        var beats = 0
        await beatUntilStopped(
            stop: { stop }, poll: .milliseconds(200), interval: interval,
            sleep: { _ in
                slept += 1
                if slept == polls { stop = true }
            }
        ) { _ in beats += 1 }
        #expect((slept, beats) == (polls, expectedBeats))
    }

    /// A failed row is a skip: the next run leaves it, and the phone's progress starts at the
    /// rows a resume already holds, then counts every row as it lands. The crate only logs the
    /// progress.
    @Test func progressCountsTheResumedRowsFirst() async throws {
        let (store, request, benchmark) = try cell(["s0", "s1", "s2"])
        defer { try? store.clear() }
        let earlier = try store.open(request)
        try earlier.append(.failed(id: "s1", failedReason: "crash"))
        earlier.close()

        let sent = EvalTestLog<String>()
        let events = EvalTestLog<BenchmarkProgress>()
        _ = try await BenchmarkMeasurement.eval(
            benchmark, inFlight: .parallel(1), checkpoint: try store.open(request),
            ctx: (), progress: { events.append([$0]) }
        ) { _, samples, record in
            sent.append(samples.map(\.id))
            try answerAll(samples, record)
        }
        #expect(sent.items == ["s0", "s2"])
        #expect(events.items == [.sample(completed: 1, total: 3), .sample(completed: 2, total: 3),
                                 .sample(completed: 3, total: 3)])
    }

    /// The stop mapping the scorer reads, shared so it means the same thing on every engine:
    /// an end-of-generation token is a model that stopped where it meant to, and anything else
    /// is the token cap binding.
    @Test func theStopReasonFollowsTheEnginesStopSignal() {
        #expect(evalStopReason(stoppedOnEnd: true) == .eos)
        #expect(evalStopReason(stoppedOnEnd: false) == .truncated)
    }
}

/// What a test's attempts appended, in order, from any task.
private nonisolated final class EvalTestLog<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [T] = []

    func append(_ items: [T]) { lock.withLock { stored += items } }
    var items: [T] { lock.withLock { stored } }
}

/// Arrivals counted up to `n`. `wait` returns once all have arrived, and fails after a bound,
/// so a pool that does not run them together fails a test instead of hanging it.
private nonisolated final class EvalTestLatch: @unchecked Sendable {
    private let lock = NSLock()
    private let n: Int
    private var arrived = 0

    init(_ n: Int) { self.n = n }

    func arrive() { lock.withLock { arrived += 1 } }

    func wait() async throws {
        arrive()
        let deadline = ContinuousClock.now + .seconds(10)
        while lock.withLock({ arrived < n }) {
            guard ContinuousClock.now < deadline else {
                throw EvalTestFailure("only \(lock.withLock { arrived }) of \(n) arrived")
            }
            try await Task.sleep(for: .milliseconds(1))
        }
    }
}

/// An engine whose samples crash it: every attempt in flight when one does loses its samples.
private nonisolated final class CrashingEngine: @unchecked Sendable {
    struct Send: CustomStringConvertible {
        let id: String
        let size: Int
        /// How many attempts ran when this one started.
        let beside: Int
        var description: String { "\(id)/\(size)/\(beside)" }
    }

    private let lock = NSLock()
    private var attempts = 0
    private var running = 0
    private var down = false
    private var recorded: [Send] = []

    func nextAttempt() -> Int {
        lock.withLock {
            defer { attempts += 1 }
            return attempts
        }
    }

    /// The engine brings back what an earlier attempt lost.
    func start(_ samples: [EvalMeasurementSample]) {
        lock.withLock {
            running += 1
            recorded += samples.map { Send(id: $0.id, size: samples.count, beside: running) }
            down = false
        }
    }

    func crash() { lock.withLock { down = true } }

    /// Whether the engine went down under this attempt.
    func finish() -> Bool {
        lock.withLock {
            running -= 1
            return down
        }
    }

    var sends: [Send] { lock.withLock { recorded } }
}

private nonisolated struct EvalTestFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
