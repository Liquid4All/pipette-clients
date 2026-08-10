import Testing
import Foundation
@testable import Pipette

/// Pins the wire shape of the per-iteration Apple thermal fields on
/// `BenchmarkSubmissionPayload`: the `device_apple_thermal_state_*`
/// keys, their `[String]` array form, and the omit-when-nil behavior that keeps
/// non-throughput runs from emitting empty lists. Complements the XCTest
/// `SubmissionPayloadTests` drift guard for the rest of the payload.
@Suite struct ThermalTelemetryPayloadTests {
    private func encodeToObject(_ payload: BenchmarkSubmissionPayload) throws -> [String: Any] {
        let data = try JSONEncoder().encode(payload)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// A payload carrying a per-rep Apple thermal series encodes to the
    /// snake_case list columns the mgmt schema expects.
    @Test func appleThermalStateSeriesEncodesToWireKeys() throws {
        let payload = BenchmarkSubmissionPayload(
            benchmarkId: "decode_throughput_512_100",
            device: deviceInfoFixture(osBuild: nil),
            power: powerStateFixture(),
            deviceAppleThermalStateBefore: ["nominal", "nominal", "nominal"],
            deviceAppleThermalStateAfter: ["nominal", "fair", "serious"],
            modelFlags: nil,
            modelDescriptor: SubmissionFixtures.mlxModelDescriptor,
            runtimeFlags: nil,
            benchmarkFlags: nil,
            runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
            runtimeCpuVariant: nil,
            submittedAt: "2026-06-24T00:00:00Z",
            result: .decodeThroughput(timeMs: 50.0, stddev: 2.1))
        let obj = try encodeToObject(payload)
        #expect(obj["device_apple_thermal_state_before"] as? [String] == ["nominal", "nominal", "nominal"])
        #expect(obj["device_apple_thermal_state_after"] as? [String] == ["nominal", "fair", "serious"])
    }

    /// The gated PIPETTE_PRIVATE_THERMAL SoC die-temp series encodes to its
    /// float list columns alongside the state series.
    @Test func appleSocTempSeriesEncodesToWireKeys() throws {
        let payload = BenchmarkSubmissionPayload(
            benchmarkId: "decode_throughput_512_100",
            device: deviceInfoFixture(osBuild: nil),
            power: powerStateFixture(),
            deviceAppleThermalStateBefore: ["nominal", "nominal"],
            deviceAppleThermalStateAfter: ["nominal", "fair"],
            deviceAppleSocTempCBefore: [38, 41],
            deviceAppleSocTempCAfter: [40, 45],
            modelFlags: nil,
            modelDescriptor: SubmissionFixtures.mlxModelDescriptor,
            runtimeFlags: nil,
            benchmarkFlags: nil,
            runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
            runtimeCpuVariant: nil,
            submittedAt: "2026-06-24T00:00:00Z",
            result: .decodeThroughput(timeMs: 50.0, stddev: 2.1))
        let obj = try encodeToObject(payload)
        #expect(obj["device_apple_soc_temp_c_before"] as? [Double] == [38, 41])
        #expect(obj["device_apple_soc_temp_c_after"] as? [Double] == [40, 45])
    }

    /// A `PIPETTE_PRIVATE_THERMAL` SoC series with any missing per-rep reading is
    /// elided whole: `[Float?]` array position is the iteration index, so a
    /// partial series can't be shipped without misaligning it. Driven through
    /// `PayloadBuilder.writeLocal` (where the fold lives), the `before` series
    /// carrying a `nil` rep drops out entirely while the fully-populated `after`
    /// series survives.
    @Test func partialNilSocSeriesElidesWholeSeries() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        var response = RunResponse(resultData: .decodeThroughput(timeMs: 50.0, stddev: nil))
        response.thermal = RunThermal(
            before: RunThermal.series(states: [], socTemps: [38, nil, 41]),
            after: RunThermal.series(states: [], socTemps: [40, 43, 44]))
        try PayloadBuilder.writeLocal(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: response,
            cellId: "cell-1",
            source: .remote,
            storage: storage)

        let data = try #require(storage.results.loadPayload("cell-1"))
        let obj = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        // A single missing `before` reading elides the whole series...
        #expect(obj["device_apple_soc_temp_c_before"] == nil)
        // ...while the fully-populated `after` series is emitted verbatim.
        #expect(obj["device_apple_soc_temp_c_after"] as? [Double] == [40, 43, 44])
    }

    /// A payload with no thermal series (the default) omits both keys rather than
    /// serializing null or an empty array.
    @Test func nilThermalSeriesElide() throws {
        let payload = BenchmarkSubmissionPayload(
            benchmarkId: "eval",
            device: deviceInfoFixture(osBuild: nil),
            power: powerStateFixture(batteryLevel: nil, powerState: nil),
            modelFlags: nil,
            modelDescriptor: SubmissionFixtures.mlxModelDescriptor,
            runtimeFlags: nil,
            benchmarkFlags: nil,
            runtimeDescriptor: SubmissionFixtures.mlxRuntimeDescriptor,
            runtimeCpuVariant: nil,
            submittedAt: "2026-06-24T00:00:00Z",
            result: .eval(completions: []))
        let obj = try encodeToObject(payload)
        #expect(obj["device_apple_thermal_state_before"] == nil)
        #expect(obj["device_apple_thermal_state_after"] == nil)
        #expect(obj["device_apple_soc_temp_c_before"] == nil)
        #expect(obj["device_apple_soc_temp_c_after"] == nil)
    }
}
