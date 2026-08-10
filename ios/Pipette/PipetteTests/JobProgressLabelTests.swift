import Testing
import Foundation
@testable import Pipette

/// Covers the pure helpers behind the live job-progress UI: the cell label +
/// param summary, the cooldown caption, and the readiness debug line's -1 hiding.
@Suite @MainActor struct JobProgressLabelTests {
    /// `modelName` names the repo, since the label derives it from the cell's source.
    private func cell(benchmarkId: String, type: BenchmarkType?, modelName: String) -> JobCell {
        JobCell(
            cellId: "c",
            benchmarkId: benchmarkId,
            benchmarkType: type,
            runStatus: .running,
            serverJobId: nil,
            errorMessage: nil,
            source: ggufTextFixture(modelName, "weights-Q4_0.gguf")
        )
    }

    @Test func decodeLabelShowsPrefillAndDecodeTokens() {
        let c = cell(benchmarkId: "decode_throughput_512_100", type: .decodeThroughput,
                     modelName: "LiquidAI/LFM2.5-350M-GGUF")
        let label = JobExecutor.liveCellLabel(c, params: [
            "parameter_prefill_tokens": 512,
            "parameter_decode_tokens": 100
        ])
        #expect(label == "Decode Throughput · 512→100 tok · LFM2.5-350M-GGUF")
    }

    @Test func prefillLabelShowsSingleTokenCount() {
        let c = cell(benchmarkId: "prefill_throughput_512", type: .prefillThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_prefill_tokens": 512]) == "Prefill Throughput · 512 tok · Foo")
    }

    @Test func labelWithoutParamsOmitsSummary() {
        let c = cell(benchmarkId: "decode_throughput", type: .decodeThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: nil) == "Decode Throughput · Foo")
    }

    @Test func evalLabelUsesDatasetId() {
        // Eval benchmarks differ by dataset, not token params, so the id is shown.
        let c = cell(benchmarkId: "ifbench", type: .eval, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_max_tokens": 512]) == "ifbench · Foo")
    }

    @Test func vlLabelShowsImageDimsAndTokens() {
        let c = cell(benchmarkId: "vl_throughput_336_336", type: .vlThroughput, modelName: "org/Foo")
        let label = JobExecutor.liveCellLabel(c, params: [
            "parameter_image_width": 336,
            "parameter_image_height": 336,
            "parameter_text_tokens": 64
        ])
        #expect(label == "Vision-Language Throughput · 336×336px · 64 tok · Foo")
    }

    @Test func decodeLabelFallsBackToDecodeTokensWhenPrefillMissing() {
        let c = cell(benchmarkId: "decode_throughput_100", type: .decodeThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_decode_tokens": 100]) == "Decode Throughput · 100 tok · Foo")
    }

    @Test func coolingCaptionFormatsElapsedAndDeadline() {
        let start = Date(timeIntervalSinceReferenceDate: 1_000)
        let cooling = JobCoolingState(since: start, deadline: 300, targetC: 36)
        #expect(cooling.caption(at: start.addingTimeInterval(20)) == "Cooling 0:20 / 5:00 max")
        #expect(cooling.caption(at: start.addingTimeInterval(125)) == "Cooling 2:05 / 5:00 max")
    }

    @Test func coolingCaptionClampsNegativeElapsed() {
        let start = Date(timeIntervalSinceReferenceDate: 1_000)
        let cooling = JobCoolingState(since: start, deadline: 300, targetC: 36)
        #expect(cooling.caption(at: start.addingTimeInterval(-5)) == "Cooling 0:00 / 5:00 max")
    }

    @Test func readinessDescriptionTagsMissingTemperatureUnreadable() {
        let status = ReadinessStatus(phase: .waiting, temperatureC: nil, thermalStateLabel: "nominal",
                                     thresholdC: 36, elapsedSeconds: 10, maxSeconds: 300, action: "waiting")
        // The -1 sentinel lives only in this debug line, tagged unreadable — never the UI.
        #expect(status.description.contains("temp:-1.0°C(unreadable,<36)"))
        #expect(status.isWaiting)
    }

    @Test func readinessDescriptionTagsHotVsOk() {
        func desc(_ temp: Double) -> String {
            ReadinessStatus(phase: .waiting, temperatureC: temp, thermalStateLabel: "fair",
                            thresholdC: 36, elapsedSeconds: 0, maxSeconds: 300, action: "waiting").description
        }
        #expect(desc(42).contains("(hot,<36)"))
        #expect(desc(30).contains("(ok,<36)"))
    }
}
