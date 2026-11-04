import Foundation
import Testing

@testable import Pipette

/// `BenchmarkFlagRef` resolution to an iOS `BenchmarkFlags` variant, mirroring the iOS cells
/// of mgmt's `pipette-plan-types` `benchmark_flags.rs`: in-process timing cells that carry
/// only the readiness gate.
struct BenchmarkFlagsTests {
    private func group(benchmark: String, runtime: String, model: String,
                       _ fields: [String: Any] = [:]) throws -> Data {
        var group: [String: Any] = [
            "benchmark_type": benchmark, "runtime_type": runtime, "model_type": model,
        ]
        for (key, value) in fields { group[key] = value }
        return try JSONSerialization.data(withJSONObject: group)
    }

    private func parse(benchmark: String, runtime: String, model: String,
                       _ fields: [String: Any] = [:]) throws -> BenchmarkFlags {
        try JSONDecoder().decode(
            BenchmarkFlagRef.self,
            from: try group(benchmark: benchmark, runtime: runtime, model: model, fields))
        .resolve()
    }

    private func object(_ json: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    /// A name per variant; exhaustive, so a new variant fails to compile here first.
    private func variant(_ flags: BenchmarkFlags) -> String {
        switch flags {
        case .prefillThroughputLlamacppIosPipetteGgufText: "prefillThroughputLlamacppIosPipetteGgufText"
        case .decodeThroughputLlamacppIosPipetteGgufText: "decodeThroughputLlamacppIosPipetteGgufText"
        case .endToEndLatencyLlamacppIosPipetteGgufText: "endToEndLatencyLlamacppIosPipetteGgufText"
        case .vlThroughputLlamacppIosPipetteGgufVision: "vlThroughputLlamacppIosPipetteGgufVision"
        case .asrThroughputLlamacppIosPipetteGgufVision: "asrThroughputLlamacppIosPipetteGgufVision"
        case .prefillThroughputMlxIosPipetteMlx: "prefillThroughputMlxIosPipetteMlx"
        case .decodeThroughputMlxIosPipetteMlx: "decodeThroughputMlxIosPipetteMlx"
        case .endToEndLatencyMlxIosPipetteMlx: "endToEndLatencyMlxIosPipetteMlx"
        case .prefillThroughputOnnxGenaiIosPipetteOnnxGenai: "prefillThroughputOnnxGenaiIosPipetteOnnxGenai"
        case .decodeThroughputOnnxGenaiIosPipetteOnnxGenai: "decodeThroughputOnnxGenaiIosPipetteOnnxGenai"
        case .endToEndLatencyOnnxGenaiIosPipetteOnnxGenai: "endToEndLatencyOnnxGenaiIosPipetteOnnxGenai"
        case .vlSpecDecodeLlamacppIosPipetteGgufVision:
            "vlSpecDecodeLlamacppIosPipetteGgufVision"
        case .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative:
            "vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative"
        }
    }

    /// The crate's `CELLS` rows for the iOS runtimes: in-process, so the gate is the only
    /// thing a plan can drive.
    private static let cells: [(benchmark: String, runtime: String, model: String,
                                accepted: [String])] = [
        ("prefill_throughput", "llamacpp_ios_pipette", "gguf_text", ["readiness"]),
        ("decode_throughput", "llamacpp_ios_pipette", "gguf_text", ["readiness"]),
        ("end_to_end_latency", "llamacpp_ios_pipette", "gguf_text", ["readiness"]),
        ("vl_throughput", "llamacpp_ios_pipette", "gguf_vision", ["readiness"]),
        ("asr_throughput", "llamacpp_ios_pipette", "gguf_vision", ["readiness"]),
        ("prefill_throughput", "mlx_ios_pipette", "mlx", ["readiness"]),
        ("decode_throughput", "mlx_ios_pipette", "mlx", ["readiness"]),
        ("end_to_end_latency", "mlx_ios_pipette", "mlx", ["readiness"]),
        ("prefill_throughput", "onnx_genai_ios_pipette", "onnx_genai", ["readiness"]),
        ("decode_throughput", "onnx_genai_ios_pipette", "onnx_genai", ["readiness"]),
        ("end_to_end_latency", "onnx_genai_ios_pipette", "onnx_genai", ["readiness"]),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision", ["readiness"]),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision_speculative", ["readiness"]),
    ]

    /// Every settable field on the crate's `BenchmarkFlagRef` but the absorbed `doomloop`;
    /// only `readiness` is an iOS field, and the rest are refused as unknown.
    private static var probes: [(field: String, value: Any)] {
        [
            ("http_timeout_seconds", 600), ("bench_timeout_seconds", 1800),
            ("readiness", ["max_wait_secs": 120]),
        ]
    }

    @Test func aDoomloopKeyIsAbsorbedAndDropped() throws {
        for cell in Self.cells {
            let flags = try parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                  model: cell.model,
                                  ["doomloop": ["exact_repeat": ["required": 5, "window": 8192]]])
            #expect(flags.readiness == nil)
            let wire = try JSONEncoder().encode(BenchmarkFlagRef(flags))
            let json = try #require(try JSONSerialization.jsonObject(with: wire) as? [String: Any])
            #expect(json["doomloop"] == nil, "\(cell.benchmark)/\(cell.runtime)")
            #expect(try object(try flags.submissionValue())["doomloop"] == nil)
        }
    }

    /// The result carries that identity in its own columns.
    @Test func theCellCoordinatesAreNotReported() throws {
        let reported = try object(try BenchmarkFlags.prefillThroughputLlamacppIosPipetteGgufText(readiness: nil)
            .submissionValue())
        for key in BenchmarkFlagRef.axisKeys { #expect(reported[key] == nil, "\(key)") }
    }

    @Test func aResolvedReadinessBlockReportsBothFields() throws {
        let flags = BenchmarkFlags.prefillThroughputMlxIosPipetteMlx(
            readiness: ReadinessOverrides(maxWaitSecs: 300, skipThermal: true))
        let readiness = try #require(try object(try flags.submissionValue())["readiness"] as? [String: Any])

        #expect(readiness["max_wait_secs"] as? Int == 300)
        #expect(readiness["skip_thermal"] as? Bool == true)
    }

    /// The *string*, as `RuntimeFlagsTests` asserts for its own family: the nested
    /// readiness block is rendered here too, so both levels have to sort.
    @Test func theSubmissionValueRendersItsKeysSortedAsTheCrateDoes() throws {
        let flags = BenchmarkFlags.prefillThroughputMlxIosPipetteMlx(
            readiness: ReadinessOverrides(maxWaitSecs: 300, skipThermal: true))

        #expect(try flags.submissionValue()
            == #"{"readiness":{"max_wait_secs":300,"skip_thermal":true}}"#)
    }

    /// Over the iOS timing cells: the override survives resolution and the flat-wire round
    /// trip, carrying the runtime axis.
    @Test func readinessOnATimingCellRoundTrips() throws {
        for cell in Self.cells {
            let flags = try parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                  model: cell.model,
                                  ["readiness": ["max_wait_secs": 1800, "skip_thermal": true]])
            #expect(flags.readiness == ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true))

            let wire = try JSONEncoder().encode(BenchmarkFlagRef(flags))
            let json = try #require(try JSONSerialization.jsonObject(with: wire) as? [String: Any])
            #expect(json["runtime_type"] as? String == cell.runtime)
            #expect(try JSONDecoder().decode(BenchmarkFlagRef.self, from: wire).resolve() == flags)
        }
    }

    /// `cell_table_covers_every_variant`.
    @Test func theCellTableCoversEveryVariant() throws {
        let seen = try Self.cells.map {
            variant(try parse(benchmark: $0.benchmark, runtime: $0.runtime, model: $0.model))
        }
        #expect(Set(seen).count == seen.count)
        #expect(seen.count == 13, "the crate declares thirteen iOS variants")
    }

    /// `every_cell_accepts_exactly_its_declared_fields_and_keeps_their_values`.
    @Test func everyCellAcceptsExactlyItsDeclaredFields() throws {
        for cell in Self.cells {
            for probe in Self.probes {
                let parsed = try? parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                        model: cell.model, [probe.field: probe.value])
                #expect((parsed != nil) == cell.accepted.contains(probe.field),
                        "\(cell.benchmark)/\(cell.runtime): `\(probe.field)`")
                guard let parsed else { continue }
                switch probe.field {
                case "readiness": #expect(parsed.readiness?.maxWaitSecs == 120)
                default: Issue.record("\(probe.field) was accepted with no assertion")
                }
            }
        }
    }

    /// `unsupported_cells_have_no_variant` and `readiness_on_eval_is_rejected`, iOS rows:
    /// eval and the memory kinds do not gate here, and Apple Foundation has none.
    @Test(arguments: [
        ("eval", "llamacpp_ios_pipette", "gguf_text"),
        ("max_memory_usage", "llamacpp_ios_pipette", "gguf_text"),
        ("max_memory_usage", "mlx_ios_pipette", "mlx"),
        ("eval", "onnx_genai_ios_pipette", "onnx_genai"),
        ("max_memory_usage", "onnx_genai_ios_pipette", "onnx_genai"),
        ("vl_max_memory", "llamacpp_ios_pipette", "gguf_vision"),
        ("asr_max_memory", "llamacpp_ios_pipette", "gguf_vision"),
        ("eval", "apple_foundation", "apple_foundation_text"),
    ])
    func anUngatedCellHasNoVariant(benchmark: String, runtime: String, model: String) throws {
        let ref = try JSONDecoder().decode(BenchmarkFlagRef.self, from: try group(
            benchmark: benchmark, runtime: runtime, model: model,
            ["readiness": ["max_wait_secs": 60]]))

        #expect(throws: BenchmarkFlagsError.noSuchCombination(
            benchmark: ref.benchmarkType, runtime: ref.runtimeType, model: ref.modelType)) {
            try ref.resolve()
        }
    }

    /// `unset_knobs_omitted_from_wire`.
    @Test func anUnsetGateIsOmittedFromTheWire() throws {
        let wire = try JSONEncoder().encode(BenchmarkFlagRef(.decodeThroughputMlxIosPipetteMlx(readiness: nil)))
        #expect(!String(decoding: wire, as: UTF8.self).contains("readiness"))
    }

    /// `ReadinessOverrides` is `deny_unknown_fields` too, so a typo inside the block is
    /// refused rather than read as "no override".
    @Test func aReadinessBlockRejectsAnUnknownField() {
        #expect(throws: UnknownFlagField.self) {
            try parse(benchmark: "prefill_throughput", runtime: "mlx_ios_pipette", model: "mlx",
                      ["readiness": ["max_wait": 60]])
        }
    }

    /// The crate's `matches`.
    @Test func benchmarkFlagsMatchTheirCellAlone() throws {
        let flags = BenchmarkFlags.decodeThroughputMlxIosPipetteMlx(readiness: nil)
        let mlx = try mlxSpec("org/m-MLX-4bit")
        #expect(flags.matches(benchmark: .decodeThroughput, runtime: Runtime.thisBuild(for: mlx),
                              model: mlx))
        #expect(!flags.matches(benchmark: .endToEndLatency, runtime: Runtime.thisBuild(for: mlx),
                               model: mlx))
    }
}
