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
        // The export carries the identity the payload carries: the descriptor itself,
        // verbatim, like every other passthrough column.
        #expect(completedRow["runtime_descriptor"]?.contains("llamacpp_ios_pipette") == true)
        #expect(completedRow["runtime_descriptor"]?.contains("abc123") == true)
        #expect(completedRow["model_descriptor"]?.contains("gguf_text") == true)
        #expect(completedRow["model_descriptor"]?.contains("google/gemma-3n") == true)
        // A passthrough of the payload's own field.
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

    /// The quant was parsed out of the weights' filename — a guess at an identity
    /// `model_descriptor` states exactly.
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
        // Default the typed source to a GGUF coordinate matching the cell's identity:
        // repo from `modelName` (org/name), weights leaf from `weightsFile`. The leaf
        // carries the quant token, which is where the derived quant/family come from.
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
}
