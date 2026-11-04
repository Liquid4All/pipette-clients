import Foundation

/// The crate's `ONE_AT_A_TIME`.
private nonisolated let oneAtATime =
    "the in-process ONNX Runtime GenAI engine generates one sequence at a time"

/// The crate's `pipette-onnx-genai/src/execute/mod.rs`. The engine has no prefill-only entry
/// point, so every cell runs one generation and reads its intervals.
nonisolated enum OnnxGenaiExecute {
    static func run(_ request: RunRequest, readinessGate: ReadinessGate,
                    observer: MeasurementObserver) async throws -> RunResponse {
        let definition = request.benchmark
        let resultData: BenchmarkResultData
        switch definition {
        case let .prefillThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await prefillThroughput(
                benchmark, model: try ResolvedOnnxGenaiModel(request.model.bound),
                readinessGate: readinessGate, observer: observer)
        case let .decodeThroughput(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await decodeThroughput(
                benchmark, model: try ResolvedOnnxGenaiModel(request.model.bound),
                readinessGate: readinessGate, observer: observer)
        case let .endToEndLatency(benchmark):
            try BenchmarkMeasurement.requireOneInFlight(definition, engine: oneAtATime)
            resultData = try await endToEndLatency(
                benchmark, model: try ResolvedOnnxGenaiModel(request.model.bound),
                readinessGate: readinessGate, observer: observer)
        case let .maxMemoryUsage(benchmark):
            resultData = try await maxMemoryUsage(
                benchmark, model: try ResolvedOnnxGenaiModel(request.model.bound))
        case .eval:
            // `OGATokenizer` exposes no chat template, and a borrowed renderer would score a
            // prompt no other runtime ran.
            throw RuntimeError.unsupported(definition.benchmarkId.value)
        case .vlThroughput, .vlMaxMemory, .vlSpecDecode, .asrThroughput, .asrMaxMemory:
            // plan-types gives these kinds no `onnx_genai_ios_pipette` variant.
            throw RuntimeError.unsupported(definition.benchmarkId.value)
        }
        return RunResponse(resultData: resultData, runtimeOutput: RuntimeOutput())
    }
}
