import Foundation

// Eval: every sample of a dataset generated once, each completion appended to the checkpoint as
// it lands: the crate's `pipette-ops/src/measurement/eval.rs` (`docs/methodology/evals.md`).
//
// Nothing is timed, so there is no gate, observer or warm-up: an eval's result is its
// completions. A Swift trap ends the process, so the crate's panic guard has no port.

/// How often a cell with several samples in flight says it is alive: long enough not to be
/// noise across a multi-hour eval, short enough that silence reads as a hang.
private nonisolated let heartbeatInterval = Duration.seconds(30)
/// How often the heartbeat wakes to check whether the run is over.
private nonisolated let heartbeatPoll = Duration.milliseconds(200)

/// How an eval cell keeps samples in flight.
nonisolated enum EvalMeasurementInFlight: Equatable, Sendable {
    /// One attempt at a time carrying up to N samples, for an engine that sends N prompts in one
    /// request. A batch lasts as long as its longest sample.
    case batch(Int)
    /// N one-sample attempts at once, the next started as each returns.
    case parallel(Int)
}

/// One dataset row, as an attempt receives it.
nonisolated struct EvalMeasurementSample: Equatable, Sendable {
    /// How the management server joins the completion back to its dataset row.
    let id: String
    let messages: [[String: String]]
    /// `eval sample <n>/<total>: id=<id>`, which every log line about it carries.
    let position: String
}

/// What an attempt records for one of its samples.
nonisolated enum EvalMeasurementReading: Equatable, Sendable {
    /// The sample generated. `stopReason` is `unknown` on an engine that does not classify its
    /// stops, with `stopDetail` saying so.
    case completed(completion: String, stopReason: BenchmarkEvalCompletionStopReason,
                   stopDetail: String?, completionTokens: UInt64?)
    /// The sample failed on its own: recorded as failed, and the cell goes on. Later runs of the
    /// cell skip it.
    case failed(reason: String)
    /// The sample did not finish, for a reason it shared with the samples in flight beside it.
    /// Sent again alone; lost alone, it is recorded as failed.
    case lost(reason: String)
}

/// Records a reading the moment its sample finishes, by the sample's position in the slice the
/// attempt was sent. A throw means the row could not be written; the attempt rethrows it.
typealias EvalMeasurementRecord = (Int, EvalMeasurementReading) throws -> Void

/// A dataset or a recording an eval cell cannot use. `EvalSample` refuses a row without an id
/// or messages when it decodes, so the crate's `MissingId` and `MissingMessages` never arise.
nonisolated enum EvalMeasurementError: Error, Equatable, LocalizedError {
    case missingSamples
    case outOfRange(at: Int, count: Int)
    case recordedTwice(id: String)
    case unrecorded(id: String)

    var errorDescription: String? {
        switch self {
        case .missingSamples: "eval benchmark missing samples"
        case let .outOfRange(at, count): "engine recorded sample \(at) of an attempt carrying \(count)"
        case let .recordedTwice(id): "engine recorded sample \(id) twice"
        case let .unrecorded(id): "engine returned without recording sample \(id)"
        }
    }
}

nonisolated extension BenchmarkMeasurement {
    /// Run an eval cell: each sample the checkpoint does not carry, once and in dataset order,
    /// its row appended the moment the attempt records it; the result is every row of the
    /// checkpoint. An id listed twice runs once.
    ///
    /// A `lost` reading writes no row, and once one is recorded no more samples are claimed. The
    /// attempts in flight return; then each lost sample is sent again alone and, lost again,
    /// recorded as failed. A sample lost while it already ran alone is recorded as failed at
    /// once. The engine brings back what it lost when its next attempt starts.
    ///
    /// A throw from an attempt ends the cell, unwrapped, and the rows appended before it stay
    /// for the next run to resume from. The checkpoint stays on disk too: the caller finalizes
    /// it once the result is safe (`EvalCompletionsStore.finalize`).
    ///
    /// `ctx` is lent to every attempt, for what the engine keeps across them; shared, as
    /// parallel attempts run at once. `progress` hears of every row, in order and under the
    /// lock that appends it; the crate only logs them.
    static func eval<C>(
        _ benchmark: EvalBenchmark,
        inFlight: EvalMeasurementInFlight,
        checkpoint: EvalCompletionSession,
        ctx: C,
        progress: @escaping (BenchmarkProgress) -> Void = { _ in },
        attempt: @escaping (C, [EvalMeasurementSample], EvalMeasurementRecord) async throws -> Void
    ) async throws -> BenchmarkResultData {
        let marker = benchmark.benchmarkId.value

        let samples = try dataset(benchmark)
        // Ids, not rows: an id listed twice is one row in the checkpoint.
        let total = Set(samples.map(\.id)).count
        let pending = pending(samples, checkpoint)
        if pending.count < total {
            AppLog.jobRun.info("resuming eval from its checkpoint cell=\(marker) "
                + "completed=\(total - pending.count) total=\(total)")
            progress(.sample(completed: total - pending.count, total: total))
        }

        let sender = Sender(marker: marker, total: total, checkpoint: checkpoint, ctx: ctx,
                            attempt: attempt, progress: progress)
        let slots = switch inFlight {
        case let .batch(size): size
        case let .parallel(workers): workers
        }
        precondition(slots > 0, "an eval cell keeps at least one sample in flight")
        // With several samples in flight a run can go quiet for as long as its slowest sample
        // takes, which reads the same as a wedged engine.
        let heartbeat = slots > 1 ? Task.detached { await sender.heartbeat() } : nil
        defer { heartbeat?.cancel() }
        switch inFlight {
        case let .batch(size):
            if size > 1 {
                AppLog.jobRun.info("samples per attempt cell=\(marker) samples=\(size)")
            }
            try await sender.batches(pending, size: size)
        case let .parallel(workers):
            if workers > 1 {
                AppLog.jobRun.info("samples in flight at once; completion order will not match "
                    + "dataset order cell=\(marker) workers=\(workers)")
            }
            try await sender.pool(pending, workers: workers)
        }

        sender.logSummary()
        return .eval(completions: checkpoint.intoCompletions())
    }
}

/// What every attempt of one cell shares. The checkpoint's lock is held only to append a row,
/// never across an attempt.
private nonisolated final class Sender<C>: @unchecked Sendable {
    private let marker: String
    /// Distinct ids in the dataset, for the progress line.
    private let total: Int
    private let checkpoint: EvalCompletionSession
    private let ctx: C
    private let attempt: (C, [EvalMeasurementSample], EvalMeasurementRecord) async throws -> Void
    private let progress: (BenchmarkProgress) -> Void
    private let lock = NSLock()
    /// Samples inside an attempt right now, for the heartbeat.
    private var inFlight = 0

    init(marker: String, total: Int, checkpoint: EvalCompletionSession, ctx: C,
         attempt: @escaping (C, [EvalMeasurementSample], EvalMeasurementRecord) async throws -> Void,
         progress: @escaping (BenchmarkProgress) -> Void) {
        self.marker = marker
        self.total = total
        self.checkpoint = checkpoint
        self.ctx = ctx
        self.attempt = attempt
        self.progress = progress
    }

    func heartbeat() async {
        await beatUntilStopped(
            stop: { Task.isCancelled }, poll: heartbeatPoll, interval: heartbeatInterval,
            sleep: { try? await Task.sleep(for: $0) }
        ) { elapsed in
            let (recorded, inFlight) = lock.withLock { (checkpoint.doneIds.count, self.inFlight) }
            AppLog.jobRun.info("eval running benchmark_id=\(marker) recorded=\(recorded) "
                + "total=\(total) in_flight=\(inFlight) elapsed=\(elapsed)")
        }
    }

    /// `size` samples per attempt, one attempt at a time.
    func batches(_ pending: [EvalMeasurementSample], size: Int) async throws {
        for start in stride(from: 0, to: pending.count, by: size) {
            let group = Array(pending[start ..< min(start + size, pending.count)])
            let lost = try await send(group, alone: group.count == 1)
            try await settle(lost)
        }
    }

    /// `workers` one-sample attempts at once. A `lost` or a throw stops the claiming and the
    /// attempts in flight return; after a `lost`, the lost samples are settled and the rest
    /// start over.
    func pool(_ pending: [EvalMeasurementSample], workers: Int) async throws {
        var rest = pending
        while !rest.isEmpty {
            let workers = min(workers, rest.count)
            let alone = workers == 1
            let claims = EvalClaims(rest)
            let worker: () async throws -> Void = {
                while let sample = claims.next() {
                    let dropped: [EvalMeasurementSample]
                    do {
                        dropped = try await self.send([sample], alone: alone)
                    } catch {
                        claims.stop()
                        throw error
                    }
                    if !dropped.isEmpty { claims.lose(dropped) }
                }
            }
            if alone {
                try await worker()
            } else {
                // Siblings are not cancelled: each runs to its own end, and the first worker's
                // error surfaces once they all have.
                let failures = await withTaskGroup(of: (Int, (any Error)?).self) { group in
                    for index in 0 ..< workers {
                        group.addTask {
                            do {
                                try await worker()
                                return (index, nil)
                            } catch {
                                return (index, error)
                            }
                        }
                    }
                    var failures: [(Int, any Error)] = []
                    for await (index, error) in group {
                        if let error { failures.append((index, error)) }
                    }
                    return failures
                }
                if let first = failures.min(by: { $0.0 < $1.0 }) { throw first.1 }
            }
            let (claimed, lost) = claims.settled()
            rest = Array(rest.dropFirst(claimed))
            try await settle(lost)
        }
    }

    /// Send each sample lost beside others again, alone.
    private func settle(_ lost: [EvalMeasurementSample]) async throws {
        guard !lost.isEmpty else { return }
        AppLog.jobRun.info("sending lost samples again, one at a time cell=\(marker) "
            + "samples=\(lost.count)")
        for sample in lost {
            _ = try await send([sample], alone: true)
        }
    }

    /// One attempt carrying `group`. Each row is appended as the attempt records it; the samples
    /// lost beside others come back. `alone` says nothing ran beside the attempt, and it carries
    /// one sample.
    private func send(_ group: [EvalMeasurementSample],
                      alone: Bool) async throws -> [EvalMeasurementSample] {
        let recorded = EvalRecorded(count: group.count)
        let record: EvalMeasurementRecord = { at, reading in
            guard group.indices.contains(at) else {
                throw EvalMeasurementError.outOfRange(at: at, count: group.count)
            }
            let sample = group[at]
            guard recorded.mark(at) else {
                throw EvalMeasurementError.recordedTwice(id: sample.id)
            }
            let row: BenchmarkEvalCompletion
            switch reading {
            case let .lost(reason) where !alone:
                AppLog.jobRun.warning("sample lost; to be sent again alone "
                    + "sample=\(sample.position) sample_id=\(sample.id) reason=\(reason)")
                recorded.lose(sample)
                return
            case let .completed(completion, stopReason, stopDetail, completionTokens):
                row = .completed(id: sample.id, completion: completion, stopReason: stopReason,
                                 stopDetail: stopDetail, completionTokens: completionTokens)
            case let .failed(reason):
                row = .failed(id: sample.id, failedReason: reason)
            case let .lost(reason):
                AppLog.jobRun.warning("sample lost alone; recorded as failed "
                    + "sample=\(sample.position) sample_id=\(sample.id) reason=\(reason)")
                row = .failed(id: sample.id, failedReason: reason)
            }
            try self.append(row)
        }

        let label = Self.label(group)
        lock.withLock { inFlight += group.count }
        defer { lock.withLock { inFlight -= group.count } }
        do {
            try await attempt(ctx, group, record)
        } catch {
            // Unwrapped, so a caller still matches its type; the label goes to the log instead.
            AppLog.jobRun.warning("eval attempt failed cell=\(marker) attempt=\(label): \(error)")
            throw error
        }
        if let missed = recorded.firstUnrecorded(in: group) {
            AppLog.jobRun.warning("eval attempt failed cell=\(marker) attempt=\(label)")
            throw EvalMeasurementError.unrecorded(id: missed.id)
        }
        return recorded.lost
    }

    /// Append one row, then report it: the log every `total.clamp(1, 20)` rows, the phone's
    /// progress on every one.
    private func append(_ row: BenchmarkEvalCompletion) throws {
        try lock.withLock {
            try checkpoint.append(row)
            let processed = checkpoint.doneIds.count
            logProgress(processed: processed, failed: checkpoint.failedCount)
            progress(.sample(completed: processed, total: total))
        }
    }

    /// `processed` counts completed and failed rows so the end-of-cell tick still fires when
    /// samples failed; the displayed `completed` stays success-only.
    private func logProgress(processed: Int, failed: Int) {
        let every = min(max(total, 1), 20)
        guard processed == 1 || processed == total || processed.isMultiple(of: every) else {
            return
        }
        AppLog.jobRun.info("eval progress benchmark_id=\(marker) completed=\(processed - failed) "
            + "total=\(total)")
    }

    /// The final summary an operator looks for: completed against failed, the failed ids
    /// inlined, and how many hit the token ceiling.
    func logSummary() {
        let (completions, failedIds) = lock.withLock { (checkpoint.completions, checkpoint.failedIds) }
        let completed = completions.count - failedIds.count
        if failedIds.isEmpty {
            AppLog.jobRun.info("eval finished benchmark_id=\(marker) completed=\(completed) failed=0")
        } else {
            AppLog.jobRun.warning("eval finished benchmark_id=\(marker) completed=\(completed) "
                + "failed=\(failedIds.count) failed_ids=\(failedIds)")
        }
        // A truncated completion is scored like any other, so `failed=0` can hide a run where
        // the token budget decided most answers.
        let truncated = completions.filter {
            if case .completed(_, _, .truncated, _, _) = $0 { true } else { false }
        }.count
        guard truncated > 0 else { return }
        let percent = completed > 0 ? truncated * 100 / completed : 0
        AppLog.jobRun.warning("eval completions hit max_tokens rather than stopping on their own; "
            + "those scores reflect the token budget as much as the model "
            + "benchmark_id=\(marker) truncated=\(truncated) scored=\(completed) percent=\(percent)")
    }

    /// Names an attempt in the log: its one sample, or the first of its batch.
    private static func label(_ group: [EvalMeasurementSample]) -> String {
        guard let first = group.first else { return "" }
        return group.count == 1 ? first.position : "batch of \(group.count) from \(first.position)"
    }
}

/// The pool's claims on its samples. How many are claimed and whether claiming has stopped share
/// one lock, so no claim slips past a loss or an error.
private nonisolated final class EvalClaims: @unchecked Sendable {
    private let samples: [EvalMeasurementSample]
    private let lock = NSLock()
    private var claimed = 0
    private var stopped = false
    private var lost: [EvalMeasurementSample] = []

    init(_ samples: [EvalMeasurementSample]) { self.samples = samples }

    func next() -> EvalMeasurementSample? {
        lock.withLock {
            guard !stopped, claimed < samples.count else { return nil }
            claimed += 1
            return samples[claimed - 1]
        }
    }

    func stop() { lock.withLock { stopped = true } }

    func lose(_ dropped: [EvalMeasurementSample]) {
        lock.withLock {
            stopped = true
            lost += dropped
        }
    }

    func settled() -> (claimed: Int, lost: [EvalMeasurementSample]) {
        lock.withLock { (claimed, lost) }
    }
}

/// Which of an attempt's samples it recorded, and which it lost beside others.
private nonisolated final class EvalRecorded: @unchecked Sendable {
    private let lock = NSLock()
    private var done: [Bool]
    private var dropped: [EvalMeasurementSample] = []

    init(count: Int) { done = Array(repeating: false, count: count) }

    /// Whether `at` was unrecorded until now.
    func mark(_ at: Int) -> Bool {
        lock.withLock {
            defer { done[at] = true }
            return !done[at]
        }
    }

    func lose(_ sample: EvalMeasurementSample) { lock.withLock { dropped.append(sample) } }

    var lost: [EvalMeasurementSample] { lock.withLock { dropped } }

    func firstUnrecorded(in group: [EvalMeasurementSample]) -> EvalMeasurementSample? {
        lock.withLock { zip(group, done).first { !$0.1 }?.0 }
    }
}

/// The benchmark's rows, numbered for the log, before anything runs.
private nonisolated func dataset(_ benchmark: EvalBenchmark) throws -> [EvalMeasurementSample] {
    guard let rows = benchmark.samples else { throw EvalMeasurementError.missingSamples }
    return rows.enumerated().map { index, row in
        EvalMeasurementSample(id: row.id, messages: row.messages,
                              position: "eval sample \(index + 1)/\(rows.count): id=\(row.id)")
    }
}

/// The samples to send: those the checkpoint does not carry, each id once, in dataset order.
private nonisolated func pending(_ samples: [EvalMeasurementSample],
                                 _ checkpoint: EvalCompletionSession) -> [EvalMeasurementSample] {
    var seen = Set<String>()
    return samples.filter { !checkpoint.contains($0.id) && seen.insert($0.id).inserted }
}

/// Call `beat` with the time since start once per elapsed `interval`, waking every `poll` to
/// check `stop`. Over an injected sleep, so the stop is testable without a run.
nonisolated func beatUntilStopped(
    stop: () -> Bool, poll: Duration, interval: Duration,
    sleep: (Duration) async -> Void, beat: (Duration) -> Void
) async {
    let clock = ContinuousClock()
    let started = clock.now
    var last = started
    while !stop() {
        await sleep(poll)
        guard clock.now - last >= interval else { continue }
        last = clock.now
        beat(clock.now - started)
    }
}

/// One sample as an in-process engine generated it, before the attempt records it.
nonisolated struct EvalGeneration {
    let text: String
    var stopReason: BenchmarkEvalCompletionStopReason
    var stopDetail: String?
    var completionTokens: UInt64?

    var reading: EvalMeasurementReading {
        .completed(completion: text, stopReason: stopReason, stopDetail: stopDetail,
                   completionTokens: completionTokens)
    }
}

/// How a completion ended, from the engine's own stop signal: an end-of-generation token means
/// the model stopped where it meant to (`eos`), anything else that the token cap bound it
/// (`truncated`). The crate reads this off its server's stream; an in-process engine knows it.
nonisolated func evalStopReason(stoppedOnEnd: Bool) -> BenchmarkEvalCompletionStopReason {
    stoppedOnEnd ? .eos : .truncated
}
