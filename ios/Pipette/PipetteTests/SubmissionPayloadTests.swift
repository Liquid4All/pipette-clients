import XCTest
@testable import Pipette

/// Pins the `BenchmarkResult` variant resolution and the encoded
/// `BenchmarkSubmissionPayload` wire shape so the Swift types can't drift from
/// the canonical Rust schema (`pipette-plan-types::benchmark`). Mirrors Rust's
/// `result_data_round_trip_*` tests, which guard the untagged variant ordering.
final class SubmissionPayloadTests: XCTestCase {
    private func encodeToObject(_ payload: BenchmarkSubmissionPayload) throws -> [String: Any] {
        let data = try JSONEncoder().encode(payload)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return try XCTUnwrap(obj, "encoded payload was not a JSON object")
    }

    private func decodeResult(_ json: String) throws -> BenchmarkResult {
        try JSONDecoder().decode(BenchmarkResult.self, from: Data(json.utf8))
    }

    /// Build a payload with phone metadata around a given result, for shape checks.
    ///
    /// `clientVersion` defaults to a fixed literal rather than the payload's own
    /// bundle default, so shape assertions don't depend on what the test host's
    /// Info.plist happens to carry.
    private func phonePayload(
        benchmarkId: String,
        result: BenchmarkResult,
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

    /// Each variant's wire JSON resolves to the matching case — the field-set
    /// dispatch is unambiguous (the ordering hazard Rust's `types.rs` warns about).
    func testEachVariantResolvesToItsCase() throws {
        XCTAssertEqual(
            try decodeResult(#"{"prefill_time_ms":12.0,"prefill_time_ms_stddev":0.3}"#),
            .prefillThroughput(timeMs: 12.0, stddev: 0.3))
        XCTAssertEqual(
            try decodeResult(#"{"decode_time_ms":50.0}"#),
            .decodeThroughput(timeMs: 50.0, stddev: nil))
        XCTAssertEqual(
            try decodeResult(#"{"total_time_ms":120.0,"total_time_ms_stddev":4.0}"#),
            .endToEndLatency(timeMs: 120.0, stddev: 4.0))
        XCTAssertEqual(
            try decodeResult(#"{"max_ram_bytes":5368709120}"#),
            .maxMemoryUsage(hostBytes: 5_368_709_120, gpuBytes: nil, npuBytes: nil))
        XCTAssertEqual(
            try decodeResult(#"{"completions":[]}"#),
            .eval(completions: []))
        XCTAssertEqual(
            try decodeResult(#"{"prompt_tokens":100,"prompt_ms":8.0,"predicted_ms":40.0}"#),
            .vlThroughput(promptTokens: 100, promptMs: 8.0, promptMsStddev: nil,
                          predictedMs: 40.0, predictedMsStddev: nil))
    }

    /// A result with no recognized metric field is not a member of the union —
    /// decoding throws (this is the coherence guard; there is no empty variant).
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

    /// `BenchmarkRun` decodes the `benchmark_id` sibling alongside the flattened result.
    func testRunDecodesBenchmarkIdAndResult() throws {
        let run = try JSONDecoder().decode(
            BenchmarkRun.self,
            from: Data(#"{"benchmark_id":"decode_throughput_512_100","decode_time_ms":50.0,"decode_time_ms_stddev":2.1}"#.utf8))
        XCTAssertEqual(run.benchmarkId, "decode_throughput_512_100")
        XCTAssertEqual(run.result, .decodeThroughput(timeMs: 50.0, stddev: 2.1))
    }

    // MARK: - Encoded wire shape (flatten + omit-nil)

    /// A decode-throughput payload as MLX on a phone produces it: on battery,
    /// no GPU/NPU/flags. Pins the exact key set and the omit-nil behavior.
    func testDecodeThroughputPhonePayloadShape() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "decode_throughput_512_100",
            result: .decodeThroughput(timeMs: 50.0, stddev: 2.1)))

        XCTAssertEqual(Set(obj.keys), [
            "benchmark_id",
            "device_name", "device_form_factor", "device_os_name", "device_os_version",
            "device_os_build", "device_chip_model", "device_ram_bytes",
            "device_battery_level", "device_power_state", "device_power_save_mode",
            "model_descriptor", "runtime_descriptor",
            "client_version",
            "submitted_at",
            "decode_time_ms", "decode_time_ms_stddev",
        ])
        XCTAssertEqual(obj["decode_time_ms"] as? Double, 50.0)
        XCTAssertEqual(obj["device_battery_level"] as? Int, 82)
        XCTAssertEqual(obj["device_power_state"] as? String, "not_charging")
        // Absent (nil) fields elide rather than serialize as null; `job_id` is
        // never part of the schema.
        // `runtime_flags` / `benchmark_flags` are declared and elide when unset; the rest
        // are absent by construction — the struct declares no such property.
        for absent in ["device_gpu_model", "device_npu_model", "runtime_flags",
                       "benchmark_flags",
                       "runtime_cpu_variant", "model_flags", "job_id",
                       "model_name", "model_quant", "mmproj_quant",
                       "runtime_name", "runtime_version"] {
            XCTAssertNil(obj[absent], "\(absent) must be absent, not present/null")
        }
    }

    /// `max_memory_usage` keeps the historical wire name `max_ram_bytes` — never
    /// the methodology field name `max_host_bytes`.
    func testMaxMemoryWireName() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "max_memory_usage_512",
            result: .maxMemoryUsage(hostBytes: 5_368_709_120, gpuBytes: nil, npuBytes: nil)))
        XCTAssertEqual(obj["max_ram_bytes"] as? UInt64, 5_368_709_120)
        XCTAssertNil(obj["max_host_bytes"])
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
        // A pre-feature payload omitting stop_reason decodes to `unknown` and is
        // re-emitted — stop_reason is required on the client, always present.
        XCTAssertEqual(completions[0]["stop_reason"] as? String, "unknown")
        XCTAssertNil(completions[0]["completion_tokens"])   // still elides when absent
        XCTAssertNil(completions[0]["stop_detail"])          // and so does stop_detail
        // A failed sample carries stop_reason=failure + the detail dual-written to
        // both failed_reason and stop_detail.
        XCTAssertEqual(completions[1]["failed"] as? Bool, true)
        XCTAssertEqual(completions[1]["failed_reason"] as? String, "timeout")
        XCTAssertEqual(completions[1]["stop_reason"] as? String, "failure")
        XCTAssertEqual(completions[1]["stop_detail"] as? String, "timeout")
        // stop_reason / completion_tokens round-trip through decode → encode.
        XCTAssertEqual(completions[2]["stop_reason"] as? String, "truncated")
        XCTAssertEqual(completions[2]["completion_tokens"] as? Int, 8192)
        // stop_detail round-trips for an unknown stop.
        XCTAssertEqual(completions[3]["stop_reason"] as? String, "unknown")
        XCTAssertEqual(completions[3]["stop_detail"] as? String, "stream dropped")
    }

    func testStopReasonWireTokens() throws {
        // The enum must serialize to the exact snake_case tokens the mgmt
        // receiver (the canonical owner) expects.
        let pairs: [(BenchmarkEvalCompletionStopReason, String)] = [
            (.eos, "eos"), (.truncated, "truncated"), (.doomLoop, "doom_loop"),
            (.failure, "failure"), (.unknown, "unknown"),
        ]
        for (reason, token) in pairs {
            let data = try JSONEncoder().encode(reason)
            XCTAssertEqual(String(decoding: data, as: UTF8.self), "\"\(token)\"")
            XCTAssertEqual(try JSONDecoder().decode(BenchmarkEvalCompletionStopReason.self, from: data), reason)
        }
    }

    /// The app build reaches the wire under the key the server stores as
    /// `client_version`, carrying whatever it was given — not a re-derivation of
    /// it. Asserting against `Bundle.main` here would compare the code to
    /// itself and pass just as happily on a bundle that reports nothing.
    func testClientVersionReachesTheWireVerbatim() throws {
        let obj = try encodeToObject(phonePayload(
            benchmarkId: "prefill_throughput_256",
            result: .prefillThroughput(timeMs: 12.0, stddev: nil),
            clientVersion: "9.9.9 (1234)"))
        XCTAssertEqual(obj["client_version"] as? String, "9.9.9 (1234)")
    }

    /// The bundle default is a real version, not the `"Unknown"` placeholder
    /// `appVersionDisplayString` falls back to when the Info.plist keys are
    /// missing. The server accepts that placeholder — it is non-blank — so
    /// nothing downstream would catch a build shipping it, and every row would
    /// group under a value that identifies nothing.
    func testBundleVersionDefaultIsNotThePlaceholder() {
        let version = Bundle.main.appVersionDisplayString
        XCTAssertFalse(version.isEmpty)
        XCTAssertFalse(
            version.contains("Unknown"),
            "bundle is missing CFBundleShortVersionString / CFBundleVersion: \(version)")
    }

    /// The `-internal` marker rides on the reported version exactly when the build must
    /// not ship — which is what lets a collector tell a run gated by a real die
    /// temperature from one gated by `thermalState`. Asserted against `isInternal`, not
    /// the capability behind it, so the rule survives a second internal-only capability.
    func testVersionCarriesTheInternalMarker() {
        let version = Bundle.main.appVersionDisplayString
        XCTAssertEqual(version.contains("-internal"), BuildFlavor.isInternal, version)
    }

}
