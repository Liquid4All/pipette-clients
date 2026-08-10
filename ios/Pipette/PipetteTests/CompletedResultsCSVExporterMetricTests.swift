import Testing
import Foundation
@testable import Pipette

@MainActor
struct CompletedResultsCSVExporterMetricTests {
    @Test func metricCalculatesPrefillThroughput() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "prefill", benchmarkId: "prefill_throughput_512", type: "prefill_throughput"),
            payload: ["prefill_time_ms": 250]
        ))

        #expect(metric.name == "Prefill throughput")
        #expect(metric.unit == "tok/s")
        #expect(abs(metric.numericValue - 2048) <= 0.001)
        #expect(metric.displayValue == "2048")
        #expect(metric.higherIsBetter)
    }

    @Test func metricCalculatesDecodeThroughput() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "decode", benchmarkId: "decode_throughput_100_100", type: "decode_throughput"),
            payload: ["decode_time_ms": "500"]
        ))

        #expect(metric.name == "Decode throughput")
        #expect(metric.unit == "tok/s")
        #expect(abs(metric.numericValue - 200) <= 0.001)
        #expect(metric.displayValue == "200")
        #expect(metric.higherIsBetter)
    }

    @Test func metricCalculatesEndToEndLatency() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "e2e", benchmarkId: "end_to_end_latency_100_256", type: "end_to_end_latency"),
            payload: ["total_time_ms": 99.4]
        ))

        #expect(metric.name == "E2E latency")
        #expect(metric.unit == "ms")
        #expect(abs(metric.numericValue - 99.4) <= 0.001)
        #expect(metric.displayValue == "99.4 ms")
        #expect(!(metric.higherIsBetter))
    }

    @Test func metricCalculatesMaxMemoryUsage() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "memory", benchmarkId: "max_memory_usage_512", type: "max_memory_usage"),
            payload: ["max_ram_bytes": 1_073_741_824]
        ))

        #expect(metric.name == "Max memory")
        #expect(metric.unit == "bytes")
        #expect(abs(metric.numericValue - 1_073_741_824) <= 0.001)
        #expect(!(metric.displayValue.isEmpty))
        #expect(!(metric.higherIsBetter))
    }

    @Test func metricCalculatesVisionLanguageThroughputWithPromptTokensFromPayload() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "vl", benchmarkId: "vl_throughput_256x256_32_128", type: "vl_throughput"),
            payload: [
                "prompt_ms": 250,
                "predicted_ms": 750,
                "prompt_tokens": 72
            ]
        ))

        #expect(metric.name == "VL throughput")
        #expect(metric.unit == "tok/s")
        #expect(abs(metric.numericValue - 200) <= 0.001)
        #expect(metric.displayValue == "200")
        #expect(metric.higherIsBetter)
    }

    @Test func metricReturnsNilForMissingOrInvalidPayloads() {
        #expect(CompletedResultsCSVExporter.metric(
            for: cell(id: "decode", benchmarkId: "decode_throughput_100_100", type: "decode_throughput"),
            payload: nil
        ) == nil)
        #expect(CompletedResultsCSVExporter.metric(
            for: cell(id: "decode", benchmarkId: "decode_throughput_100_100", type: "decode_throughput"),
            payload: ["decode_time_ms": 0]
        ) == nil)
        #expect(CompletedResultsCSVExporter.metric(
            for: cell(id: "unknown", benchmarkId: "unknown", type: "unknown"),
            payload: ["value": 1]
        ) == nil)
    }

    @Test func parameterSummaryAndQuantLabels() {
        #expect(CompletedResultsCSVExporter.parameterSummary("prefill_throughput_512") == "512")
        #expect(CompletedResultsCSVExporter.parameterSummary("decode_throughput_100_100") == "100-100")
        #expect(CompletedResultsCSVExporter.parameterSummary("end_to_end_latency_100_256") == "100-256")
        #expect(CompletedResultsCSVExporter.parameterSummary("vl_throughput_256x512_32_128") == "256x512")
        #expect(CompletedResultsCSVExporter.parameterSummary("missing") == nil)

        #expect(
            CompletedResultsCSVExporter.quantLabel(for: cell(
                id: "q",
                benchmarkId: "decode_throughput_100_100",
                type: "decode_throughput",
                weightsFile: "Qwen-Test-Q4_K_M.gguf"
            )) == "q4_km"
        )
    }

    @Test func resultColumnsUseFirstSeenBenchmarkOrderAndCatalogLabels() {
        let manifest = manifest(cells: [
            cell(id: "decode", benchmarkId: "decode_throughput_100_100", type: "decode_throughput"),
            cell(id: "prefill", benchmarkId: "prefill_throughput_512", type: "prefill_throughput"),
            cell(id: "decode-2", benchmarkId: "decode_throughput_100_100", type: "decode_throughput")
        ])

        let columns = CompletedResultsCSVExporter.resultColumns(for: manifest)

        #expect(columns.map(\.id) == ["decode_throughput_100_100", "prefill_throughput_512"])
        #expect(columns.map(\.title) == ["Decode Throughput", "Prefill Throughput"])
        #expect(columns.map(\.subtitle) == ["100-100", "512"])
    }

    @Test func metricsByCellIdSkipsCellsWithoutMetrics() throws {
        let completed = cell(id: "decode", benchmarkId: "decode_throughput_100_100", type: "decode_throughput")
        let failed = cell(id: "failed", benchmarkId: "prefill_throughput_512", type: "prefill_throughput", status: .failed)
        let metrics = CompletedResultsCSVExporter.metricsByCellId(
            for: manifest(cells: [completed, failed]),
            payloadsByCellId: [completed.cellId: ["decode_time_ms": 500]]
        )

        #expect(Set(metrics.keys) == Set([completed.cellId]))
        let metric = try #require(metrics[completed.cellId])
        #expect(abs(metric.numericValue - 200) <= 0.001)
    }

    @Test func resultGroupsPreserveModelQuantOrderAndMarkFailedMissingValues() {
        let qwenDecode = cell(
            id: "qwen-decode",
            benchmarkId: "decode_throughput_100_100",
            type: "decode_throughput",
            modelName: "qwen/qwen-test",
            weightsFile: "Qwen-Test-Q4_K_M.gguf"
        )
        let qwenFailed = cell(
            id: "qwen-failed",
            benchmarkId: "prefill_throughput_512",
            type: "prefill_throughput",
            modelName: "qwen/qwen-test",
            weightsFile: "Qwen-Test-Q4_K_M.gguf",
            status: .failed
        )
        let gemmaDecode = cell(
            id: "gemma-decode",
            benchmarkId: "decode_throughput_100_100",
            type: "decode_throughput",
            modelName: "google/gemma-test",
            weightsFile: "Gemma-Test-Q4_0.gguf"
        )
        let groups = CompletedResultsCSVExporter.resultGroups(
            for: manifest(cells: [qwenDecode, qwenFailed, gemmaDecode]),
            payloadsByCellId: [
                qwenDecode.cellId: ["decode_time_ms": 500],
                gemmaDecode.cellId: ["decode_time_ms": 250]
            ],
            store: makeTemporaryBenchmarkStore()
        )

        #expect(groups.map(\.modelName) == ["qwen-test", "gemma-test"])
        #expect(groups[0].brand == .qwen)
        #expect(groups[0].rows.map(\.quant) == ["q4_km"])
        #expect(groups[0].rows[0].values[0].displayValue == "200")
        #expect(groups[0].rows[0].values[1].displayValue == nil)
        #expect(groups[0].rows[0].values[1].isFailed)
        #expect(groups[1].rows[0].values[0].displayValue == "400")
        #expect(!(groups[1].rows[0].values[0].isFailed))
    }

    @Test func resultGroupsCollapseMinistralQuantsAcrossRepoBuckets() {
        let q40 = cell(
            id: "ministral-q40",
            benchmarkId: "decode_throughput_100_100",
            type: "decode_throughput",
            modelName: "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
            weightsFile: "Ministral-3-3B-Instruct-2512-Q4_0.gguf"
        )
        let q4km = cell(
            id: "ministral-q4km",
            benchmarkId: "prefill_throughput_512",
            type: "prefill_throughput",
            modelName: "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            weightsFile: "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
        )
        let q5km = cell(
            id: "ministral-q5km",
            benchmarkId: "decode_throughput_100_100",
            type: "decode_throughput",
            modelName: "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            weightsFile: "Ministral-3-3B-Instruct-2512-Q5_K_M.gguf"
        )

        let groups = CompletedResultsCSVExporter.resultGroups(
            for: manifest(cells: [q40, q4km, q5km]),
            payloadsByCellId: [
                q40.cellId: ["decode_time_ms": 500],
                q4km.cellId: ["prefill_time_ms": 250],
                q5km.cellId: ["decode_time_ms": 400]
            ],
            store: makeTemporaryBenchmarkStore()
        )

        #expect(groups.count == 1)
        #expect(groups[0].id == "ministral-3-3b-instruct-2512")
        #expect(groups[0].modelName == "Ministral 3 3B Instruct 2512")
        #expect(groups[0].brand == .mistral)
        #expect(groups[0].rows.map(\.quant) == ["q4_0", "q4_km", "q5_km"])
        #expect(groups[0].rows[0].values[0].displayValue == "200")
        #expect(groups[0].rows[1].values[1].displayValue == "2048")
        #expect(groups[0].rows[2].values[0].displayValue == "250")
    }

    private func manifest(cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: "job-results",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .completed
        )
    }

    private func cell(
        id: CellId,
        benchmarkId: String,
        type: String,
        modelName: String = "google/gemma-test",
        weightsFile: String = "Gemma-Test-Q4_0.gguf",
        status: CellRunStatus = .completed,
        source: Model? = nil
    ) -> JobCell {
        // Default the typed source to a GGUF coordinate matching the cell's identity:
        // repo from `modelName` (org/name), weights leaf from `weightsFile`.
        let resolvedSource = source ?? ggufTextFixture(modelName, weightsFile)
        return JobCell(
            cellId: id,
            benchmarkId: benchmarkId,
            // `type` may be an unknown string ("unknown") in the not-a-benchmark
            // test — map leniently so it decodes to nil, as a real cell would.
            benchmarkType: BenchmarkType(rawValue: type),
            runStatus: status,
            serverJobId: nil,
            errorMessage: status == .failed ? "failed" : nil,
            source: resolvedSource
        )
    }

    /// An MLX model is a directory, not a quant-named GGUF file, so a filename parse
    /// could only ever yield "unknown". The typed `source` resolves both correctly.
    @Test func mlxQuantAndGroupKeyComeFromTypedSource() throws {
        let mlx = cell(
            id: "mlx", benchmarkId: "decode_throughput_100_100", type: "decode_throughput",
            modelName: "LiquidAI/LFM2.5-350M-MLX-4bit",
            source: .mlx(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))))
        #expect(CompletedResultsCSVExporter.quantLabel(for: mlx) == "4bit")
        #expect(CompletedResultsCSVExporter.resultModelGroupKey(for: mlx) == "lfm2.5-350m")
    }
}
