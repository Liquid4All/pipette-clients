import Testing
import Foundation
@testable import Pipette

@MainActor
struct CompletedResultsCSVExporterTests {
    @Test func csvIncludesHeadersMetricsFailedRowsAndEscapedFields() throws {
        let completed = makeCell(
            id: "cell-completed",
            benchmarkId: "decode_throughput_100_100",
            benchmarkType: .decodeThroughput,
            modelName: "google/gemma-3n",
            weightsFile: "Gemma-3n-Q4_0.gguf",
            status: .completed,
            serverJobId: "server-123"
        )
        let failed = makeCell(
            id: "cell-failed",
            benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput,
            modelName: "google/gemma-3n",
            weightsFile: "Gemma-3n-Q4_0.gguf",
            status: .failed,
            errorMessage: "Model \"missing\", retry"
        )
        let manifest = makeManifest(
            title: "Nightly, \"quoted\"",
            cells: [completed, failed]
        )
        let payloads = [
            completed.cellId: [
                "benchmark_id": completed.benchmarkId,
                "decode_time_ms": 500,
                "submitted_at": "2026-05-28T16:45:00Z",
                "runtime_descriptor": #"""
                    {"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp",                    "repository_version":"abc123","flavor":"ios-arm64"}
                    """#,
                "runtime_flags": #"{"number_gpu_layers":99,"ctx_size":4056}"#,
                "model_descriptor": #"{"type":"gguf_text","repo":"google/gemma-3n"}"#,
                "runtime_cpu_variant": "neon",
                "device_name": "Alex \"Phone\", Pro",
                "device_form_factor": "phone",
                "device_os_name": "iOS",
                "device_os_version": "26.4",
                "device_os_build": "22F76",
                "device_chip_model": "A20",
                "device_ram_bytes": 17_179_869_184,
                "device_battery_level": 42,
                "device_power_state": "not_charging",
                "device_power_save_mode": true
            ]
        ]

        let csv = CompletedResultsCSVExporter.csv(
            for: manifest,
            payloadsByCellId: payloads,
            store: makeTemporaryBenchmarkStore())
        let rows = try parseCSV(csv)
        #expect(rows.count == 3)
        #expect(rows[0] == CompletedResultsCSVExporter.headers)

        let completedRow = try dictionary(header: rows[0], row: rows[1])
        #expect(completedRow["job_id"] == "job-abcdef1234")
        #expect(completedRow["job_title"] == "Nightly, \"quoted\"")
        #expect(completedRow["cell_id"] == "cell-completed")
        #expect(completedRow["benchmark_parameters"] == "100-100")
        #expect(completedRow["status"] == "completed")
        #expect(completedRow["metric_name"] == "Decode throughput")
        #expect(completedRow["metric_value"] == "200")
        #expect(completedRow["metric_unit"] == "tok/s")
        #expect(completedRow["metric_display_value"] == "200")
        #expect(completedRow["submitted_at"] == "2026-05-28T16:45:00Z")
        // The descriptor verbatim, like every other passthrough column.
        #expect(completedRow["runtime_descriptor"]?.contains("llamacpp_ios_pipette") == true)
        #expect(completedRow["runtime_descriptor"]?.contains("abc123") == true)
        #expect(completedRow["model_descriptor"]?.contains("gguf_text") == true)
        #expect(completedRow["model_descriptor"]?.contains("google/gemma-3n") == true)
        #expect(completedRow["runtime_flags"] == #"{"number_gpu_layers":99,"ctx_size":4056}"#)
        #expect(completedRow["server_job_id"] == "server-123")
        #expect(completedRow["device_name"] == "Alex \"Phone\", Pro")
        #expect(completedRow["device_os_build"] == "22F76")
        #expect(completedRow["device_ram_bytes"] == "17179869184")
        #expect(completedRow["device_battery_level"] == "42")
        #expect(completedRow["device_power_state"] == "not_charging")
        #expect(completedRow["device_power_save_mode"] == "true")
        #expect(completedRow["runtime_cpu_variant"] == "neon")

        let failedRow = try dictionary(header: rows[0], row: rows[2])
        #expect(failedRow["cell_id"] == "cell-failed")
        #expect(failedRow["benchmark_parameters"] == "512")
        #expect(failedRow["status"] == "failed")
        #expect(failedRow["metric_name"] == "")
        #expect(failedRow["metric_value"] == "")
        #expect(failedRow["metric_unit"] == "")
        #expect(failedRow["metric_display_value"] == "")
        #expect(failedRow["error_message"] == "Model \"missing\", retry")

        #expect(csv.contains("\"Nightly, \"\"quoted\"\"\""))
        #expect(csv.contains("\"Alex \"\"Phone\"\", Pro\""))
    }

    @Test func csvOrdersRowsByPageModelQuantAndBenchmarkOrder() throws {
        let bDecode = makeCell(
            id: "b-decode",
            benchmarkId: "decode_throughput_100_100",
            benchmarkType: .decodeThroughput,
            modelName: "qwen/qwen-test",
            weightsFile: "Qwen-Test-Q4_K_M.gguf",
            status: .completed
        )
        let aDecode = makeCell(
            id: "a-decode",
            benchmarkId: "decode_throughput_100_100",
            benchmarkType: .decodeThroughput,
            modelName: "google/gemma-test",
            weightsFile: "Gemma-Test-Q4_0.gguf",
            status: .completed
        )
        let bPrefill = makeCell(
            id: "b-prefill",
            benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput,
            modelName: "qwen/qwen-test",
            weightsFile: "Qwen-Test-Q4_K_M.gguf",
            status: .completed
        )
        let manifest = makeManifest(cells: [bDecode, aDecode, bPrefill])

        let csv = CompletedResultsCSVExporter.csv(for: manifest, payloadsByCellId: [:], store: makeTemporaryBenchmarkStore())
        let rows = try parseCSV(csv)
        let cellIds = try rows.dropFirst().map {
            try dictionary(header: rows[0], row: $0)["cell_id"]
        }

        #expect(cellIds == ["b-decode", "b-prefill", "a-decode"])
    }

    @Test func filenameUsesDateTitleAndJobPrefix() {
        let manifest = makeManifest(cells: [])

        #expect(
            CompletedResultsCSVExporter.filename(for: manifest, dateTitle: "2026-05-28") ==
            "pipette-results-2026-05-28-job-abcd.csv"
        )
    }

    private func makeManifest(
        title: String? = nil,
        cells: [JobCell]
    ) -> JobManifest {
        JobManifest(
            jobId: "job-abcdef1234",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4056,
            cells: cells,
            status: .completed,
            title: title
        )
    }

    /// A quant parsed from the weights' filename would only guess what `model_descriptor`
    /// states exactly.
    @Test func theExportCarriesNoDerivedQuantColumn() {
        #expect(!CompletedResultsCSVExporter.headers.contains("model_quant"))
        #expect(CompletedResultsCSVExporter.headers.contains("model_descriptor"))
    }

    private func makeCell(
        id: CellId,
        benchmarkId: String,
        benchmarkType: BenchmarkType?,
        modelName: String,
        weightsFile: String,
        status: CellRunStatus,
        serverJobId: String? = nil,
        errorMessage: String? = nil,
        source: Model? = nil
    ) -> JobCell {
        // The weights leaf carries the quant token the derived quant and family read.
        let resolvedSource = source ?? ggufTextFixture(modelName, weightsFile)
        return JobCell(
            cellId: id,
            benchmarkId: benchmarkId,
            benchmarkType: benchmarkType,
            runStatus: status,
            serverJobId: serverJobId,
            errorMessage: errorMessage,
            source: resolvedSource
        )
    }

    private func dictionary(header: [String], row: [String]) throws -> [String: String] {
        #expect(header.count == row.count)
        return Dictionary(uniqueKeysWithValues: zip(header, row))
    }

    private func parseCSV(_ csv: String) throws -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var isQuoted = false
        var index = csv.startIndex

        while index < csv.endIndex {
            let char = csv[index]
            let next = csv.index(after: index)

            if isQuoted {
                if char == "\"" {
                    if next < csv.endIndex, csv[next] == "\"" {
                        field.append("\"")
                        index = csv.index(after: next)
                        continue
                    }
                    isQuoted = false
                } else {
                    field.append(char)
                }
            } else {
                switch char {
                case "\"":
                    isQuoted = true
                case ",":
                    row.append(field)
                    field = ""
                case "\n":
                    row.append(field)
                    rows.append(row)
                    row = []
                    field = ""
                case "\r":
                    break
                default:
                    field.append(char)
                }
            }

            index = next
        }

        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            rows.append(row)
        }

        return rows
    }

    // MARK: - Metric

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

    /// `ttft` is `prompt_ms` itself, the first of the four metrics the scorer derives for
    /// the type.
    @Test func metricExportsTimeToFirstTokenForAVisionCell() throws {
        let metric = try #require(CompletedResultsCSVExporter.metric(
            for: cell(id: "vl", benchmarkId: "vl_throughput_256x256_32_128", type: "vl_throughput"),
            payload: [
                "prompt_ms": 250,
                "predicted_ms": 750,
                "prompt_tokens": 72
            ]
        ))

        #expect(metric.name == "TTFT")
        #expect(metric.unit == "ms")
        #expect(abs(metric.numericValue - 250) <= 0.001)
        #expect(!metric.higherIsBetter)
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
        let resolvedSource = source ?? ggufTextFixture(modelName, weightsFile)
        return JobCell(
            cellId: id,
            benchmarkId: benchmarkId,
            // An unknown `type` decodes to nil, as a real cell's would.
            benchmarkType: BenchmarkType(rawValue: type),
            runStatus: status,
            serverJobId: nil,
            errorMessage: status == .failed ? "failed" : nil,
            source: resolvedSource
        )
    }

    /// An MLX model is a directory, not a quant-named file, so a filename parse could only
    /// yield "unknown".
    @Test func mlxQuantAndGroupKeyComeFromTypedSource() throws {
        let mlx = cell(
            id: "mlx", benchmarkId: "decode_throughput_100_100", type: "decode_throughput",
            modelName: "LiquidAI/LFM2.5-350M-MLX-4bit",
            source: .mlx(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))))
        #expect(CompletedResultsCSVExporter.quantLabel(for: mlx) == "4bit")
        #expect(CompletedResultsCSVExporter.resultModelGroupKey(for: mlx) == "lfm2.5-350m")
    }
}
