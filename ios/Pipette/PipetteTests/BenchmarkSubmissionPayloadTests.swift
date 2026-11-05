import XCTest
@testable import Pipette

/// Pins the Swift types to the canonical Rust schema (`pipette-plan-types::submission` in
/// pipette-mgmt), mirroring its `result_data_untagged_variant_round_trips` test.
final class BenchmarkSubmissionPayloadTests: XCTestCase {
    private func encodeToObject(_ payload: BenchmarkSubmissionPayload) throws -> [String: Any] {
        let data = try JSONEncoder().encode(payload)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return try XCTUnwrap(obj, "encoded payload was not a JSON object")
    }

    private func decodeResult(_ json: String) throws -> BenchmarkResultData {
        try JSONDecoder().decode(BenchmarkResultData.self, from: Data(json.utf8))
    }

    /// `clientVersion` defaults to a fixed literal rather than the bundle default, so shape
    /// assertions don't depend on what the test host's Info.plist carries.
    private func phonePayload(
        benchmarkId: String,
        result: BenchmarkResultData,
        clientVersion: String = "1.2 (34)"
    ) -> BenchmarkSubmissionPayload {
        BenchmarkSubmissionPayload(
            benchmarkId: benchmarkId,
            device: deviceInfoFixture(),
            power: powerStateFixture(),
            modelFlags: nil,
            modelDescriptor: SubmissionFixtures.mlxModelDescriptor,
            runtimeFlags: nil,
            benchmarkFlags: nil,
            runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
            runtimeCpuVariant: nil,
            clientVersion: clientVersion,
            submittedAt: "2026-06-24T00:00:00Z",
            result: result
        )
    }

    // MARK: - Variant resolution (untagged decode)

    /// The field-set dispatch is unambiguous (the ordering hazard mgmt's
    /// `submission/mod.rs` warns about).
    func testEachVariantResolvesToItsCase() throws {
        XCTAssertEqual(
            try decodeResult(#"{"prefill_time_ms":12.0,"prefill_time_ms_stddev":0.3}"#),
            .prefillThroughput(prefillTimeMs: 12.0, prefillTimeMsStddev: 0.3))
        XCTAssertEqual(
            try decodeResult(#"{"decode_time_ms":50.0}"#),
            .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil))
        XCTAssertEqual(
            try decodeResult(#"{"total_time_ms":120.0,"total_time_ms_stddev":4.0}"#),
            .endToEndLatency(totalTimeMs: 120.0, totalTimeMsStddev: 4.0))
        XCTAssertEqual(
            try decodeResult(#"{"max_ram_bytes":5368709120}"#),
            .maxMemoryUsage(maxHostBytes: 5_368_709_120, maxGpuBytes: nil, maxNpuBytes: nil))
        XCTAssertEqual(
            try decodeResult(#"{"completions":[]}"#),
            .eval(completions: []))
        XCTAssertEqual(
            try decodeResult(#"{"prompt_tokens":100,"prompt_ms":8.0,"predicted_ms":40.0}"#),
            .vlThroughput(promptTokens: 100, promptMs: 8.0, promptMsStddev: nil,
                          predictedMs: 40.0, predictedMsStddev: nil))
    }

    /// `vlThroughput` resolves ahead of `endToEndLatency`, as the crate orders them, so a VL
    /// body carrying `total_time_ms` decodes as VL, and an e2e body, which has no
    /// `prompt_tokens`, still decodes as e2e.
    func testVlTotalTimeResolvesAheadOfEndToEnd() throws {
        XCTAssertEqual(
            try decodeResult(
                #"{"prompt_tokens":100,"prompt_ms":8.0,"predicted_ms":40.0,"total_time_ms":52.0,"total_time_ms_stddev":1.5}"#),
            .vlThroughput(promptTokens: 100, promptMs: 8.0, promptMsStddev: nil,
                          predictedMs: 40.0, predictedMsStddev: nil, totalTimeMs: 52.0,
                          totalTimeMsStddev: 1.5))
        XCTAssertEqual(
            try decodeResult(#"{"total_time_ms":120.0,"total_time_ms_stddev":4.0}"#),
            .endToEndLatency(totalTimeMs: 120.0, totalTimeMsStddev: 4.0))
    }

    /// There is no empty variant, so a body with no recognized metric is not a member.
    func testMetriclessResultThrows() throws {
        XCTAssertThrowsError(try decodeResult(#"{"benchmark_id":"prefill_throughput_512"}"#))
        XCTAssertThrowsError(try decodeResult(#"{}"#))
        // Stddev alone is meaningless without its mean → not a member either.
        XCTAssertThrowsError(try decodeResult(#"{"prefill_time_ms_stddev":0.3}"#))
    }

    /// An incomplete variant (vl missing required `prompt_ms`/`predicted_ms`)
    /// throws rather than producing a payload the server can't deserialize.
    func testIncompleteVariantThrows() throws {
        XCTAssertThrowsError(try decodeResult(#"{"prompt_tokens":100}"#))
    }

    func testRunDecodesBenchmarkIdAndResult() throws {
        let run = try JSONDecoder().decode(
            BenchmarkRun.self,
            from: Data(#"{"benchmark_id":"decode_throughput_512_100","decode_time_ms":50.0,"decode_time_ms_stddev":2.1}"#.utf8))
        XCTAssertEqual(run.benchmarkId, "decode_throughput_512_100")
        XCTAssertEqual(run.result, .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: 2.1))
    }

    // MARK: - Encoded wire shape (flatten + omit-nil)

    /// A decode-throughput payload as MLX on a phone produces it: on battery,
    /// no GPU/NPU/flags. Pins the exact key set and the omit-nil behavior.
    func testDecodeThroughputPhonePayloadShape() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "decode_throughput_512_100",
            result: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: 2.1)))

        XCTAssertEqual(Set(obj.keys), [
            "benchmark_id",
            "device_name", "device_form_factor", "device_os_name", "device_os_version",
            "device_os_build", "device_cpu_arch", "device_os_family",
            "device_chip_model", "device_ram_bytes",
            "device_battery_level", "device_power_state", "device_power_save_mode",
            "model_descriptor", "runtime_descriptor",
            "client_version",
            "submitted_at",
            "decode_time_ms", "decode_time_ms_stddev",
        ])
        XCTAssertEqual(obj["decode_time_ms"] as? Double, 50.0)
        XCTAssertEqual(obj["device_battery_level"] as? Int, 82)
        XCTAssertEqual(obj["device_power_state"] as? String, "not_charging")
        // Nil fields elide rather than serialize as null. `job_id`, the `device_apple_*`
        // series (the readings carry them) and the model/runtime name, quant and version
        // fields are not properties at all.
        for absent in ["device_gpu_model", "device_npu_model", "runtime_flags",
                       "benchmark_flags", "device_hardware_readings",
                       "device_os_throttling_readings", "readiness_gate_waits",
                       "runtime_cpu_variant", "model_flags", "job_id", "device_time_ms",
                       "device_apple_thermal_state_before", "device_apple_soc_temp_c_before",
                       "model_name", "model_quant", "mmproj_quant",
                       "runtime_name", "runtime_version"] {
            XCTAssertNil(obj[absent], "\(absent) must be absent, not present/null")
        }
    }

    /// `max_memory_usage` keeps the historical wire name `max_ram_bytes`, never the
    /// methodology field name `max_host_bytes`.
    func testMaxMemoryWireName() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "max_memory_usage_512",
            result: .maxMemoryUsage(maxHostBytes: 5_368_709_120, maxGpuBytes: nil, maxNpuBytes: nil)))
        XCTAssertEqual(obj["max_ram_bytes"] as? UInt64, 5_368_709_120)
        XCTAssertNil(obj["max_host_bytes"])
    }

    /// A VL row's whole-request time goes out as `total_time_ms`, the text latency's name;
    /// a row without it carries neither key.
    func testVlThroughputTotalWireNames() throws {
        let result = BenchmarkResultData.vlThroughput(
            promptTokens: 300, imageTokens: 256, promptMs: 120, promptMsStddev: 2,
            predictedMs: 40, predictedMsStddev: 1, totalTimeMs: 170, totalTimeMsStddev: 3)
        let payload = phonePayload(benchmarkId: "vl_throughput_512x512", result: result)
        let timed = try encodeToObject(payload)
        XCTAssertEqual(timed["total_time_ms"] as? Double, 170)
        XCTAssertEqual(timed["total_time_ms_stddev"] as? Double, 3)
        let untimed = try encodeToObject(phonePayload(
            benchmarkId: "vl_throughput_512x512",
            result: .vlThroughput(promptTokens: 300, imageTokens: 256, promptMs: 120,
                                  promptMsStddev: 2, predictedMs: 40, predictedMsStddev: 1,
                                  totalTimeMs: nil, totalTimeMsStddev: nil)))
        XCTAssertNil(untimed["total_time_ms"])
        XCTAssertNil(untimed["total_time_ms_stddev"])
    }

    /// It can carry `endToEndLatency`'s only required field too.
    func testAsrResolvesAheadOfEndToEnd() throws {
        XCTAssertEqual(
            try decodeResult(
                #"{"asr_prompt_tokens":137,"asr_audio_tokens":118,"asr_prompt_ms":83.3,"asr_predicted_ms":129.4,"total_time_ms":250.0}"#),
            .asrThroughput(promptTokens: 137, audioTokens: 118, promptMs: 83.3,
                           promptMsStddev: nil, predictedMs: 129.4, predictedMsStddev: nil,
                           totalTimeMs: 250, totalTimeMsStddev: nil))
        XCTAssertThrowsError(try decodeResult(#"{"asr_prompt_tokens":137,"asr_prompt_ms":83.3}"#))
    }

    func testAsrThroughputWireNamesAndClientExtras() throws {
        let extras = #"{"asr_audio_tokens_source":"calibrated","asr_beyond_model_audio_limit":false,"asr_context_sizing":"exact"}"#
        let obj = try encodeToObject(BenchmarkSubmissionPayload(
            benchmarkId: "asr_throughput_15000ms_text0_decode256",
            device: deviceInfoFixture(), power: powerStateFixture(), modelFlags: nil,
            modelDescriptor: SubmissionFixtures.mlxModelDescriptor, runtimeFlags: nil,
            benchmarkFlags: nil, runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
            runtimeCpuVariant: nil, clientVersion: "1.2 (34)", clientExtras: extras,
            submittedAt: "2026-06-24T00:00:00Z",
            result: .asrThroughput(promptTokens: 137, audioTokens: 118, promptMs: 83.3,
                                   promptMsStddev: 0.9, predictedMs: 129.4,
                                   predictedMsStddev: 1.1, totalTimeMs: 250,
                                   totalTimeMsStddev: 2)))
        XCTAssertEqual(obj["asr_prompt_tokens"] as? Int, 137)
        XCTAssertEqual(obj["asr_audio_tokens"] as? Int, 118)
        XCTAssertEqual(obj["asr_prompt_ms"] as? Double, 83.3)
        XCTAssertEqual(obj["asr_prompt_ms_stddev"] as? Double, 0.9)
        XCTAssertEqual(obj["asr_predicted_ms"] as? Double, 129.4)
        XCTAssertEqual(obj["asr_predicted_ms_stddev"] as? Double, 1.1)
        XCTAssertEqual(obj["total_time_ms"] as? Double, 250)
        XCTAssertEqual(obj["total_time_ms_stddev"] as? Double, 2)
        XCTAssertEqual(obj["client_extras"] as? String, extras)
        for absent in ["prompt_tokens", "prompt_ms", "predicted_ms", "image_tokens"] {
            XCTAssertNil(obj[absent], absent)
        }
    }

    /// The CLI's spelling and key order (a `BTreeMap`): mgmt types these at intake.
    func testAsrClientExtrasAreTheClisObject() throws {
        XCTAssertEqual(
            try AsrClientExtras(contextSizing: .exact, audioTokensSource: .calibrated,
                                beyondModelAudioLimit: false).json(),
            #"{"asr_audio_tokens_source":"calibrated","asr_beyond_model_audio_limit":false,"asr_context_sizing":"exact"}"#)
        XCTAssertEqual(
            try AsrClientExtras(contextSizing: .upperBound, audioTokensSource: .measured,
                                beyondModelAudioLimit: true).json(),
            #"{"asr_audio_tokens_source":"measured","asr_beyond_model_audio_limit":true,"asr_context_sizing":"upper_bound"}"#)
        // `asr_max_memory` sends no source.
        XCTAssertEqual(
            try AsrClientExtras(contextSizing: .pinned, beyondModelAudioLimit: false).json(),
            #"{"asr_beyond_model_audio_limit":false,"asr_context_sizing":"pinned"}"#)
        XCTAssertEqual(AsrContextSizing(pinned: true, calibrated: false), .pinned)
        XCTAssertEqual(AsrContextSizing(pinned: false, calibrated: true), .exact)
        XCTAssertEqual(AsrContextSizing(pinned: false, calibrated: false), .upperBound)
    }

    /// Not cut: cut JSON is no longer JSON.
    func testOversizedClientExtrasAreRefused() throws {
        XCTAssertNil(try checkedClientExtras(nil))
        let atCap = "{\"k\":\"" + String(repeating: "x", count: maxClientExtrasBytes - 8) + "\"}"
        XCTAssertEqual(atCap.utf8.count, maxClientExtrasBytes)
        XCTAssertEqual(try checkedClientExtras(atCap), atCap)
        XCTAssertThrowsError(try checkedClientExtras(atCap + " ")) {
            XCTAssertEqual($0 as? ResultSubmissionError,
                           .clientExtrasTooLarge(bytes: maxClientExtrasBytes + 1))
        }
    }

    /// Eval completions survive decode → flatten → encode, including a failed one
    /// (and the successful one omits the `failed` flag).
    func testEvalCompletionsRoundTrip() throws {
        let run = try JSONDecoder().decode(BenchmarkRun.self, from: Data(#"""
        {"benchmark_id":"eval","completions":[
            {"id":"q1","completion":"A"},
            {"id":"q2","completion":"","failed":true,"failed_reason":"timeout"},
            {"id":"q3","completion":"B","stop_reason":"truncated","completion_tokens":8192},
            {"id":"q4","completion":"P","stop_reason":"unknown","stop_detail":"stream dropped"}
        ]}
        """#.utf8))
        let obj = try encodeToObject(phonePayload(benchmarkId: run.benchmarkId, result: run.result))
        let completions = try XCTUnwrap(obj["completions"] as? [[String: Any]])
        XCTAssertEqual(completions.count, 4)
        XCTAssertEqual(completions[0]["id"] as? String, "q1")
        XCTAssertNil(completions[0]["failed"])
        // A payload omitting stop_reason decodes to `unknown` and is re-emitted: it is
        // required on the client.
        XCTAssertEqual(completions[0]["stop_reason"] as? String, "unknown")
        XCTAssertNil(completions[0]["completion_tokens"])   // still elides when absent
        XCTAssertNil(completions[0]["stop_detail"])          // and so does stop_detail
        // A failed sample carries stop_reason=failure + the detail dual-written to
        // both failed_reason and stop_detail.
        XCTAssertEqual(completions[1]["failed"] as? Bool, true)
        XCTAssertEqual(completions[1]["failed_reason"] as? String, "timeout")
        XCTAssertEqual(completions[1]["stop_reason"] as? String, "failure")
        XCTAssertEqual(completions[1]["stop_detail"] as? String, "timeout")
        XCTAssertEqual(completions[2]["stop_reason"] as? String, "truncated")
        XCTAssertEqual(completions[2]["completion_tokens"] as? Int, 8192)
        XCTAssertEqual(completions[3]["stop_reason"] as? String, "unknown")
        XCTAssertEqual(completions[3]["stop_detail"] as? String, "stream dropped")
    }

    func testStopReasonWireTokens() throws {
        // The enum must serialize to the exact snake_case tokens the mgmt
        // receiver (the canonical owner) expects.
        let pairs: [(BenchmarkEvalCompletionStopReason, String)] = [
            (.eos, "eos"), (.truncated, "truncated"),
            (.failure, "failure"), (.unknown, "unknown"),
        ]
        for (reason, token) in pairs {
            let data = try JSONEncoder().encode(reason)
            XCTAssertEqual(String(decoding: data, as: UTF8.self), "\"\(token)\"")
            XCTAssertEqual(try JSONDecoder().decode(BenchmarkEvalCompletionStopReason.self, from: data), reason)
        }
    }

    /// Asserting against `Bundle.main` here would compare the code to itself and pass just
    /// as happily on a bundle that reports nothing.
    func testClientVersionReachesTheWireVerbatim() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "prefill_throughput_256",
            result: .prefillThroughput(prefillTimeMs: 12.0, prefillTimeMsStddev: nil),
            clientVersion: "9.9.9 (1234)"))
        XCTAssertEqual(obj["client_version"] as? String, "9.9.9 (1234)")
    }

    /// `"Unknown"` is `appVersionDisplayString`'s fallback when the Info.plist keys are
    /// missing. The server accepts it (it is non-blank), so nothing downstream would catch
    /// a build shipping it, and every row would group under a value that identifies nothing.
    func testBundleVersionDefaultIsNotThePlaceholder() {
        let version = Bundle.main.appVersionDisplayString
        XCTAssertFalse(version.isEmpty)
        XCTAssertFalse(
            version.contains("Unknown"),
            "bundle is missing CFBundleShortVersionString / CFBundleVersion: \(version)")
    }

    /// The marker tells a collector the run used a private capability (the die-temperature
    /// gate or the memory reads), so its numbers are not comparable with a stock build's.
    /// Asserted against `isInternal`, not either capability, so the rule holds whichever
    /// is compiled in.
    func testVersionCarriesTheInternalMarker() {
        let version = Bundle.main.appVersionDisplayString
        XCTAssertEqual(version.contains("-internal"), BuildFlavor.isInternal, version)
    }

}
