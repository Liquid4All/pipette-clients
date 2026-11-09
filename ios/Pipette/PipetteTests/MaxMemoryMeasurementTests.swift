import Foundation
import Testing

@testable import Pipette

/// `BenchmarkMeasurement.maxMemory` and `vlMaxMemory`, the crate's
/// `pipette-ops/src/measurement/max_memory.rs` and `vl_max_memory.rs`, which are copies of
/// one another and tested the same way.
struct MaxMemoryMeasurementTests {
    private let body = MaxMemoryUsageBenchmark(
        benchmarkId: BenchmarkId(validated: "max_memory_usage_512"), parameterPrefillTokens: 512)
    private let vlBody = VlMaxMemoryBenchmark(
        benchmarkId: BenchmarkId(validated: "vl_max_memory_512x512"), parameterImageWidth: 512,
        parameterImageHeight: 512)

    /// Both entries' readings of the same host, GPU and NPU peaks.
    private func readings(_ peaks: (UInt64, UInt64?, UInt64?))
        -> (MaxMemoryMeasurementReading, VlMaxMemoryMeasurementReading) {
        (MaxMemoryMeasurementReading(maxHostBytes: peaks.0, maxGpuBytes: peaks.1,
                                     maxNpuBytes: peaks.2),
         VlMaxMemoryMeasurementReading(maxHostBytes: peaks.0, maxGpuBytes: peaks.1,
                                       maxNpuBytes: peaks.2))
    }

    /// One past the largest byte count the server accepts.
    nonisolated static let tooLarge: UInt64 = 1 << 63

    /// The peaks pass through as read; a zero peak is no reading.
    @Test(arguments: [
        ((8, nil, nil), "ok"),
        ((8, 4, nil), "ok"),
        ((8, nil, 2), "ok"),
        ((0, nil, nil), "max_host_bytes"),
        ((8, 0, nil), "max_gpu_bytes"),
        ((8, nil, 0), "max_npu_bytes"),
    ] as [((UInt64, UInt64?, UInt64?), String)])
    func aReadingBecomesThePeak(_ peaks: (UInt64, UInt64?, UInt64?), _ expected: String)
        async {
        let (reading, vlReading) = readings(peaks)
        let got = await outcome {
            try await BenchmarkMeasurement.maxMemory(body, observer: noObserver(), warmup: .cold,
                                                     ctx: ()) { _ in reading }
        }
        let vlGot = await outcome {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .cold, ctx: ()
            ) { _ in vlReading }
        }
        let peak = BenchmarkResultData.maxMemoryUsage(
            maxHostBytes: peaks.0, maxGpuBytes: peaks.1, maxNpuBytes: peaks.2)
        switch (got, vlGot, expected) {
        case let (.success(resultData), .success(vlResultData), "ok"):
            #expect(resultData == peak)
            #expect(vlResultData == peak)
        case let (.failure(error), .failure(vlError), field):
            #expect(error as? MaxMemoryMeasurementError == .zeroPeak(field: field))
            #expect(vlError as? VlMaxMemoryMeasurementError == .zeroPeak(field: field))
        default:
            Issue.record("expected \(expected), got \(got) and \(vlGot)")
        }
    }

    /// The largest byte count passes; one more is refused, naming its field.
    @Test(arguments: [
        ((tooLarge, nil, nil), "max_host_bytes"),
        ((8, tooLarge, nil), "max_gpu_bytes"),
        ((8, nil, tooLarge), "max_npu_bytes"),
    ] as [((UInt64, UInt64?, UInt64?), String)])
    func aPeakPastTheWireRangeIsRefused(_ peaks: (UInt64, UInt64?, UInt64?), _ field: String)
        async {
        let (fits, vlFits) = readings((Self.tooLarge - 1, Self.tooLarge - 1, Self.tooLarge - 1))
        await #expect(throws: Never.self) {
            try await BenchmarkMeasurement.maxMemory(body, observer: noObserver(), warmup: .cold,
                                                     ctx: ()) { _ in fits }
        }
        await #expect(throws: Never.self) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .cold, ctx: ()
            ) { _ in vlFits }
        }

        let (refused, vlRefused) = readings(peaks)
        await #expect(throws: MaxMemoryMeasurementError.tooLarge(field: field,
                                                                 bytes: Self.tooLarge)) {
            try await BenchmarkMeasurement.maxMemory(body, observer: noObserver(), warmup: .cold,
                                                     ctx: ()) { _ in refused }
        }
        await #expect(throws: VlMaxMemoryMeasurementError.tooLarge(field: field,
                                                                   bytes: Self.tooLarge)) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .cold, ctx: ()
            ) { _ in vlRefused }
        }
    }

    /// A failed attempt is the cell's error, and what it recorded before failing is still
    /// the caller's; the error keeps its type.
    @Test func aFailedAttemptIsNamedByTheMarker() async throws {
        let recorded = Recorded()
        await #expect(throws: ServerWentAway.self) {
            try await BenchmarkMeasurement.maxMemory(
                body, observer: noObserver(), warmup: .cold, ctx: recorded
            ) { recorded in
                recorded.attempts.append("loaded")
                throw ServerWentAway()
            }
        }
        await #expect(throws: ServerWentAway.self) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .cold, ctx: recorded
            ) { recorded in
                recorded.attempts.append("loaded")
                throw ServerWentAway()
            }
        }
        #expect(recorded.attempts == ["loaded", "loaded"])
    }

    /// A warm-up runs the attempt first; its reading is checked, then dropped for the
    /// measured one's.
    @Test(arguments: [
        (MemoryMeasurementWarmup.cold, [8]),
        (.once, [16, 8]),
    ] as [(MemoryMeasurementWarmup, [UInt64])])
    func aWarmUpRunsTheAttemptFirst(_ warmup: MemoryMeasurementWarmup, _ expected: [UInt64])
        async throws {
        var peaks = expected.makeIterator()
        var vlPeaks = expected.makeIterator()
        var read: [UInt64] = []
        var vlRead: [UInt64] = []
        let resultData = try await BenchmarkMeasurement.maxMemory(
            body, observer: noObserver(), warmup: warmup, ctx: ()
        ) { _ in
            let next = peaks.next()
            let peak = try #require(next)
            read.append(peak)
            return readings((peak, nil, nil)).0
        }
        let vlResultData = try await BenchmarkMeasurement.vlMaxMemory(
            vlBody, observer: noObserver(), warmup: warmup, ctx: ()
        ) { _ in
            let next = vlPeaks.next()
            let peak = try #require(next)
            vlRead.append(peak)
            return readings((peak, nil, nil)).1
        }
        #expect(read == expected)
        #expect(vlRead == expected)
        let peak = BenchmarkResultData.maxMemoryUsage(maxHostBytes: 8, maxGpuBytes: nil,
                                                      maxNpuBytes: nil)
        #expect(resultData == peak)
        #expect(vlResultData == peak)
    }

    /// Each attempt starts on the attempt clock first, under its name; a refusal there, an
    /// earlier attempt past the limit, runs nothing more.
    @Test func eachAttemptStartsOnTheClockFirst() async throws {
        let recorded = Recorded()
        let observer = MeasurementObserver(onBeforeRepStart: {}, onAfterRepEnd: {},
                                           onAttemptStart: { recorded.attempts.append($0) })
        _ = try await BenchmarkMeasurement.maxMemory(
            body, observer: observer, warmup: .once, ctx: recorded
        ) { recorded in
            recorded.attempts.append("ran")
            return readings((8, nil, nil)).0
        }
        #expect(recorded.attempts == [BenchmarkMeasurement.peakMemoryWarmUp, "ran",
                                      BenchmarkMeasurement.peakMemoryMeasured, "ran"])

        let refused = MeasurementObserver(onBeforeRepStart: {}, onAfterRepEnd: {},
                                          onAttemptStart: { throw StepFailed(event: $0) })
        await #expect(throws: StepFailed(event: BenchmarkMeasurement.peakMemoryMeasured)) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: refused, warmup: .cold, ctx: ()
            ) { _ in
                Issue.record("an attempt ran past a refused start")
                return readings((8, nil, nil)).1
            }
        }
    }

    /// A failed or refused warm-up is the cell's error, and the measured attempt never runs.
    @Test func aFailedWarmUpSkipsTheMeasuredAttempt() async {
        let recorded = Recorded()
        await #expect(throws: ServerWentAway.self) {
            try await BenchmarkMeasurement.maxMemory(
                body, observer: noObserver(), warmup: .once, ctx: recorded
            ) { recorded in
                recorded.attempts.append("failed")
                throw ServerWentAway()
            }
        }
        await #expect(throws: MaxMemoryMeasurementError.zeroPeak(field: "max_host_bytes")) {
            try await BenchmarkMeasurement.maxMemory(
                body, observer: noObserver(), warmup: .once, ctx: recorded
            ) { recorded in
                recorded.attempts.append("refused")
                return readings((0, nil, nil)).0
            }
        }
        await #expect(throws: ServerWentAway.self) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .once, ctx: recorded
            ) { recorded in
                recorded.attempts.append("failed")
                throw ServerWentAway()
            }
        }
        await #expect(throws: VlMaxMemoryMeasurementError.zeroPeak(field: "max_host_bytes")) {
            try await BenchmarkMeasurement.vlMaxMemory(
                vlBody, observer: noObserver(), warmup: .once, ctx: recorded
            ) { recorded in
                recorded.attempts.append("refused")
                return readings((0, nil, nil)).1
            }
        }
        #expect(recorded.attempts == ["failed", "refused", "failed", "refused"])
    }
}
