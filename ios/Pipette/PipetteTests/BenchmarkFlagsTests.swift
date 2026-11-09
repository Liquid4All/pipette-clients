import Foundation
import Testing

@testable import Pipette

/// `BenchmarkFlagRef` resolution to an iOS `BenchmarkFlags` variant, mirroring the iOS cells
/// of mgmt's `pipette-plan-types` `benchmark_flags.rs`: in-process cells, so the attempt limit
/// and the readiness gate are all a plan can drive.
struct BenchmarkFlagsTests {
    private func group(benchmark: String, runtime: String, model: String,
                       _ fields: [String: Any] = [:]) throws -> Data {
        var group: [String: Any] = [
            "benchmark_type": benchmark, "runtime_type": runtime, "model_type": model,
        ]
        for (key, value) in fields { group[key] = value }
        return try JSONSerialization.data(withJSONObject: group)
    }

    private func decode(benchmark: String, runtime: String, model: String,
                        _ fields: [String: Any] = [:]) throws -> BenchmarkFlagRef {
        try JSONDecoder().decode(
            BenchmarkFlagRef.self,
            from: try group(benchmark: benchmark, runtime: runtime, model: model, fields))
    }

    private func parse(benchmark: String, runtime: String, model: String,
                       _ fields: [String: Any] = [:]) throws -> BenchmarkFlags {
        try decode(benchmark: benchmark, runtime: runtime, model: model, fields).resolve()
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
        case .maxMemoryUsageLlamacppIosPipetteGgufText: "maxMemoryUsageLlamacppIosPipetteGgufText"
        case .maxMemoryUsageMlxIosPipetteMlx: "maxMemoryUsageMlxIosPipetteMlx"
        case .maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai: "maxMemoryUsageOnnxGenaiIosPipetteOnnxGenai"
        case .vlMaxMemoryLlamacppIosPipetteGgufVision: "vlMaxMemoryLlamacppIosPipetteGgufVision"
        case .asrMaxMemoryLlamacppIosPipetteGgufVision: "asrMaxMemoryLlamacppIosPipetteGgufVision"
        case .decodeThroughputAppleFoundationAppleFoundationText:
            "decodeThroughputAppleFoundationAppleFoundationText"
        case .endToEndLatencyAppleFoundationAppleFoundationText:
            "endToEndLatencyAppleFoundationAppleFoundationText"
        }
    }

    private static let timed = ["attempt_timeout_seconds", "readiness"]

    /// The crate's `CELLS` rows for the iOS runtimes: a timing cell takes the attempt limit
    /// and the gate, a memory cell the limit alone, a corpus cell the gate alone.
    private static let cells: [(benchmark: String, runtime: String, model: String,
                                accepted: [String])] = [
        ("prefill_throughput", "llamacpp_ios_pipette", "gguf_text", timed),
        ("decode_throughput", "llamacpp_ios_pipette", "gguf_text", timed),
        ("end_to_end_latency", "llamacpp_ios_pipette", "gguf_text", timed),
        ("vl_throughput", "llamacpp_ios_pipette", "gguf_vision", timed),
        ("asr_throughput", "llamacpp_ios_pipette", "gguf_vision", timed),
        ("prefill_throughput", "mlx_ios_pipette", "mlx", timed),
        ("decode_throughput", "mlx_ios_pipette", "mlx", timed),
        ("end_to_end_latency", "mlx_ios_pipette", "mlx", timed),
        ("prefill_throughput", "onnx_genai_ios_pipette", "onnx_genai", timed),
        ("decode_throughput", "onnx_genai_ios_pipette", "onnx_genai", timed),
        ("end_to_end_latency", "onnx_genai_ios_pipette", "onnx_genai", timed),
        ("decode_throughput", "apple_foundation", "apple_foundation_text", timed),
        ("end_to_end_latency", "apple_foundation", "apple_foundation_text", timed),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision", ["readiness"]),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision_speculative", ["readiness"]),
        ("max_memory_usage", "llamacpp_ios_pipette", "gguf_text", ["attempt_timeout_seconds"]),
        ("max_memory_usage", "mlx_ios_pipette", "mlx", ["attempt_timeout_seconds"]),
        ("max_memory_usage", "onnx_genai_ios_pipette", "onnx_genai", ["attempt_timeout_seconds"]),
        ("vl_max_memory", "llamacpp_ios_pipette", "gguf_vision", ["attempt_timeout_seconds"]),
        ("asr_max_memory", "llamacpp_ios_pipette", "gguf_vision", ["attempt_timeout_seconds"]),
    ]

    /// Every settable field on the crate's `BenchmarkFlagRef` but the absorbed ones. An iOS
    /// ref has no `http_timeout_seconds`, so it is refused as unknown on every cell.
    private static var probes: [(field: String, value: Any)] {
        [
            ("http_timeout_seconds", 600), ("attempt_timeout_seconds", 600),
            ("readiness", ["max_wait_secs": 120]),
        ]
    }

    /// The crate's obsolete keys, absorbed unread: a queued spec or a recorded result still
    /// carries them.
    @Test func anObsoleteKeyIsAbsorbedAndDropped() throws {
        let obsolete: [String: Any] = [
            "doomloop": ["exact_repeat": ["required": 5, "window": 8192]],
            "bench_timeout_seconds": 1800,
        ]
        for cell in Self.cells {
            for (key, value) in obsolete {
                let flags = try parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                      model: cell.model, [key: value])
                #expect(flags.readiness == nil)
                #expect(flags.attemptTimeout == nil)
                let wire = try JSONEncoder().encode(BenchmarkFlagRef(flags))
                let json = try #require(try JSONSerialization.jsonObject(with: wire) as? [String: Any])
                #expect(json[key] == nil, "\(cell.benchmark)/\(cell.runtime)")
                #expect(try object(try flags.submissionValue())[key] == nil)
            }
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
            attemptTimeoutSeconds: 600,
            readiness: ReadinessOverrides(maxWaitSecs: 300, skipThermal: true))

        #expect(try flags.submissionValue()
            == #"{"attempt_timeout_seconds":600,"readiness":{"max_wait_secs":300,"skip_thermal":true}}"#)
    }

    /// A memory cell reports the limit its plan set, and nothing when it set none.
    @Test func aMemoryCellReportsItsAuthoredLimitAlone() throws {
        #expect(try BenchmarkFlags.maxMemoryUsageMlxIosPipetteMlx(attemptTimeoutSeconds: 2400)
            .submissionValue() == #"{"attempt_timeout_seconds":2400}"#)
        #expect(try BenchmarkFlags.vlMaxMemoryLlamacppIosPipetteGgufVision(attemptTimeoutSeconds: nil)
            .submissionValue() == "{}")
    }

    /// Over every iOS cell: each field it takes survives resolution and the flat-wire round
    /// trip, carrying the runtime axis.
    @Test func everyAcceptedFieldRoundTrips() throws {
        for cell in Self.cells {
            let gated = cell.accepted.contains("readiness")
            let limited = cell.accepted.contains("attempt_timeout_seconds")
            var fields: [String: Any] = [:]
            if gated { fields["readiness"] = ["max_wait_secs": 1800, "skip_thermal": true] }
            if limited { fields["attempt_timeout_seconds"] = 900 }
            let flags = try parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                  model: cell.model, fields)
            #expect(flags.readiness
                == (gated ? ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true) : nil))
            #expect(flags.attemptTimeout == (limited ? UInt64(900) : nil))

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
        #expect(seen.count == 20, "the crate declares twenty iOS variants")
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
                case "attempt_timeout_seconds": #expect(parsed.attemptTimeout == 600)
                default: Issue.record("\(probe.field) was accepted with no assertion")
                }
            }
        }
    }

    /// A knob the cell's variant has no field for is refused by name, as the crate's
    /// `deny` refuses it: the gate on a memory cell, the limit on an iOS corpus cell.
    @Test(arguments: [
        ("max_memory_usage", "llamacpp_ios_pipette", "gguf_text", "readiness"),
        ("max_memory_usage", "mlx_ios_pipette", "mlx", "readiness"),
        ("max_memory_usage", "onnx_genai_ios_pipette", "onnx_genai", "readiness"),
        ("vl_max_memory", "llamacpp_ios_pipette", "gguf_vision", "readiness"),
        ("asr_max_memory", "llamacpp_ios_pipette", "gguf_vision", "readiness"),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision", "attempt_timeout_seconds"),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision_speculative",
         "attempt_timeout_seconds"),
    ])
    func aKnobTheVariantLacksIsRefusedByName(benchmark: String, runtime: String, model: String,
                                             knob: String) throws {
        let value: Any = knob == "readiness" ? ["max_wait_secs": 60] : 600
        let ref = try decode(benchmark: benchmark, runtime: runtime, model: model, [knob: value])

        #expect(throws: BenchmarkFlagsError.knobNotAllowed(
            knob: knob, benchmark: ref.benchmarkType, runtime: ref.runtimeType,
            model: ref.modelType)) {
            try ref.resolve()
        }
    }

    /// `unsupported_cells_have_no_variant`, iOS rows: eval carries server settings an
    /// in-process engine cannot apply, and Apple Foundation runs no prefill.
    @Test(arguments: [
        ("eval", "llamacpp_ios_pipette", "gguf_text"),
        ("eval", "mlx_ios_pipette", "mlx"),
        ("eval", "onnx_genai_ios_pipette", "onnx_genai"),
        ("eval", "apple_foundation", "apple_foundation_text"),
        ("prefill_throughput", "apple_foundation", "apple_foundation_text"),
    ])
    func aCellWithNoVariantIsRefused(benchmark: String, runtime: String, model: String) throws {
        let ref = try decode(benchmark: benchmark, runtime: runtime, model: model)

        #expect(throws: BenchmarkFlagsError.noSuchCombination(
            benchmark: ref.benchmarkType, runtime: ref.runtimeType, model: ref.modelType)) {
            try ref.resolve()
        }
    }

    /// The crate's `PositiveSeconds`: a zero limit would stop every attempt as it starts.
    @Test func aZeroAttemptLimitIsRefused() {
        #expect(throws: DecodingError.self) {
            try parse(benchmark: "prefill_throughput", runtime: "mlx_ios_pipette", model: "mlx",
                      ["attempt_timeout_seconds": 0])
        }
    }

    /// `unset_knobs_omitted_from_wire`.
    @Test func unsetKnobsAreOmittedFromTheWire() throws {
        let wire = String(decoding: try JSONEncoder().encode(
            BenchmarkFlagRef(.decodeThroughputMlxIosPipetteMlx(readiness: nil))), as: UTF8.self)
        #expect(!wire.contains("readiness"))
        #expect(!wire.contains("attempt_timeout_seconds"))
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
