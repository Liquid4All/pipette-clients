import Testing

@testable import Pipette

/// `isBenchmarkSupported(_:on:)` — the single source of truth for which
/// `(runtime, benchmark)` pairs this client runs. Pure (config-independent), so no
/// device is needed.
struct BenchmarkSupportTests {

    /// Sample `Runtime` values covering each engine; the config knobs don't affect
    /// support, so any values do.
    private let afm = RuntimeKind.afm
    private let mlx = RuntimeKind.mlx
    private let llama = RuntimeKind.llamaCpp

    // MARK: - The support matrix

    @Test func afmSupportsOnlyEvalDecodeAndE2E() {
        #expect(isBenchmarkSupported(.eval, on: afm))
        #expect(isBenchmarkSupported(.decodeThroughput, on: afm))
        #expect(isBenchmarkSupported(.endToEndLatency, on: afm))
        #expect(!isBenchmarkSupported(.prefillThroughput, on: afm))
        #expect(!isBenchmarkSupported(.maxMemoryUsage, on: afm))
        #expect(!isBenchmarkSupported(.vlThroughput, on: afm))
    }

    @Test func mlxSupportsEverythingExceptVL() {
        for bt in BenchmarkType.allCases {
            #expect(isBenchmarkSupported(bt, on: mlx) == (bt != .vlThroughput),
                    "mlx \(bt.rawValue)")
        }
    }

    @Test func llamaSupportsEveryBenchmarkType() {
        for bt in BenchmarkType.allCases {
            #expect(isBenchmarkSupported(bt, on: llama), "llama \(bt.rawValue)")
        }
    }

    // MARK: - Drift guard: AFM's accepted set == isBenchmarkSupported(_, on: .afm)

    /// `AFMRuntime.run` throws `.unsupported` for exactly the benchmark types
    /// `isBenchmarkSupported(_, on: .afm)` rejects (its guard is the authority). This
    /// pins the *intended* accepted set so a change to either side that lets them
    /// diverge fails here rather than at runtime on-device.
    @Test func afmAcceptedSetMatchesSupportPredicate() {
        let intendedAccepted: Set<BenchmarkType> = [.eval, .decodeThroughput, .endToEndLatency]
        let predicateAccepted = Set(BenchmarkType.allCases.filter { isBenchmarkSupported($0, on: .afm) })
        #expect(predicateAccepted == intendedAccepted)
    }
}
