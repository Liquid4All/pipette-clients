import Foundation
import MLX

/// The crate's `ONE_AT_A_TIME`.
private nonisolated let oneAtATime = "the in-process MLX engine decodes one sequence at a time"

/// The crate's `pipette-mlx/src/execute/mod.rs`, one file per benchmark module. Where the
/// crate drives an `mlx_lm` server, this runs the model in process.
nonisolated enum MlxExecute {
    static func run(_ request: RunRequest, runContext: any RunContext,
                    readinessGate: ReadinessGate, observer: MeasurementObserver,
                    isCancelled: @escaping () -> Bool = { false },
                    progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> RunResponse {
        let definition = request.benchmark
        // One overlay, both run with and reported, so a result records what ran.
        let flags = try MlxRuntimeFlags.forRun(request)
        let chunk = Int(flags.nUbatch ?? MlxRuntime.defaultPrefillChunk)

        let resultData: BenchmarkResultData
        switch definition {
        case let .prefillThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await prefillThroughput(
                benchmark, model: try ResolvedMlxModel(request.model.bound), chunk: chunk,
                readinessGate: readinessGate, observer: observer)
        case let .decodeThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await decodeThroughput(
                benchmark, model: try ResolvedMlxModel(request.model.bound), chunk: chunk,
                readinessGate: readinessGate, observer: observer)
        case let .endToEndLatency(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await endToEndLatency(
                benchmark, model: try ResolvedMlxModel(request.model.bound), chunk: chunk,
                readinessGate: readinessGate, observer: observer)
        case let .maxMemoryUsage(benchmark):
            resultData = try await maxMemoryUsage(
                benchmark, model: try ResolvedMlxModel(request.model.bound), chunk: chunk,
                observer: observer)
        case let .eval(benchmark):
            // Before the load: a body without rows has nothing to run.
            let cell = try EvalCell(benchmark)
            let model = try ResolvedMlxModel(request.model.bound)
            resultData = try await eval(
                benchmark, cell: cell,
                // Opened once the model resolved: it writes the checkpoint.
                checkpoint: try runContext.evalCompletions.open(request),
                model: model, chunk: chunk, isCancelled: isCancelled, progress: progress)
        case .vlThroughput, .vlMaxMemory, .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            // plan-types gives these kinds no `mlx_ios_pipette` variant.
            throw RuntimeError.unsupported(definition.benchmarkId.value)
        }
        var response = RunResponse(resultData: resultData, runtimeOutput: RuntimeOutput())
        response.runtimeFlags = flags
        return response
    }

    /// Re-baselines `mlxPeak` for this cell and clears transient buffers after it, so the
    /// footprint returns to ~weights instead of creeping up to an OOM.
    static func measuringCell<T>(_ id: BenchmarkId, chunk: Int,
                                 _ body: () async throws -> T) async rethrows -> T {
        defer { MLX.Memory.clearCache() }
        MLX.Memory.peakMemory = 0
        MlxRuntime.logMem("bench \(id) chunk=\(chunk)")
        return try await body()
    }
}
