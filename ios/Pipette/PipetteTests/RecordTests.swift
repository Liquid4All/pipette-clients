import Foundation
import Testing

@testable import Pipette

/// The record step (`pipette-cli`'s `results/record.rs`): what a finished run files as its
/// payload and extras, and the refusals that keep a record honest.
@MainActor struct RecordTests {
    @Test func aFreshPayloadPassesTheDescriptorGuard() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try await record(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                  runtimeOutput: RuntimeOutput()),
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let payload = try writtenPayload(cellId: "cell-1", ws: ws)
        #expect(throws: Never.self) {
            try requireDescriptors("cell-1", payload)
        }
    }

    /// So the server can refuse a run its catalog disagrees with.
    @Test func aFinishedRunDeclaresTheBenchmarkItRan() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try await record(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                  runtimeOutput: RuntimeOutput()),
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let payload = try writtenPayload(cellId: "cell-1", ws: ws)
        #expect(payload["benchmark_type"] as? String == "decode_throughput")
        #expect(payload["benchmark_params"] as? NSDictionary
            == ["prefill_tokens": 512, "decode_tokens": 100, "batch_size": 1] as NSDictionary)
    }

    /// `extras.json` is a local diagnostic, never submitted. An in-process run has no argv
    /// and no stdout, stored empty.
    @Test func writingAPayloadLeavesTheExtrasBesideIt() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var response = RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                   runtimeOutput: RuntimeOutput())
        response.runtimeOutput.runtimeStderr = "llama_model_loader: loaded meta data\n"

        try await record(
            request: payloadRequest(model: try ggufTextResolved()),
            response: response,
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let url = try #require(ws.results.submittableDir("cell-1"))
            .appendingPathComponent("extras.json")
        let extras = try JSONDecoder().decode(
            BenchmarkResultExtras.self, from: try Data(contentsOf: url))
        #expect(extras.stderr == "llama_model_loader: loaded meta data\n")
        #expect(extras.command.isEmpty)
        #expect(extras.stdout.isEmpty)
        #expect(extras.executable == nil)
        #expect(try writtenPayload(cellId: "cell-1", ws: ws)["stderr"] == nil)
    }

    /// Each field rides as one JSON array in a string, and a run with none sends no key.
    @Test func deviceTelemetryRidesAsJsonArrays() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let sensors: JsonValue = .array([.object(["Product": .string("PMU tdie1"),
                                                  "TemperatureLevel": .double(45.5)])])
        let reading = ReadingsElement(iteration: 0, samplePoint: .beforeRep,
                                      facilities: ["apple_iohid_temperature_sensors": sensors])
        let wait = GateWaitElement(iteration: 0, pollCount: 14, platform: "ios",
                                   phase: "proceeding", elapsedMs: 87_400, observed: "ok")
        let cells: [(CellId, RepTelemetry)] = [
            ("cell-gated", RepTelemetry(hardwareReadings: [reading], gateWaits: [wait])),
            ("cell-ungated", RepTelemetry()),
        ]
        for (cellId, telemetry) in cells {
            try await record(
                request: payloadRequest(model: try ggufTextResolved(),
                                        benchmarkId: "decode_throughput_512_100"),
                response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                      runtimeOutput: RuntimeOutput()),
                telemetry: telemetry,
                cellId: cellId,
                ws: ws)
        }

        let gated = try writtenPayload(cellId: "cell-gated", ws: ws)
        let parsed = { (field: String) throws -> NSArray? in
            let sent = try #require(gated[field] as? String, "\(field) should be sent")
            return try JSONSerialization.jsonObject(with: Data(sent.utf8)) as? NSArray
        }
        #expect(try parsed("device_hardware_readings")
            == [["iteration": 0, "sample_point": "before_rep",
                 "apple_iohid_temperature_sensors": [["Product": "PMU tdie1",
                                                      "TemperatureLevel": 45.5]]]] as NSArray)
        #expect(try parsed("readiness_gate_waits")
            == [["iteration": 0, "poll_count": 14, "platform": "ios", "phase": "proceeding",
                 "elapsed_ms": 87_400, "observed": "ok"]] as NSArray)
        #expect(gated["device_os_throttling_readings"] == nil)
        let ungated = try writtenPayload(cellId: "cell-ungated", ws: ws)
        for field in ["device_hardware_readings", "device_os_throttling_readings",
                      "readiness_gate_waits"] {
            #expect(ungated[field] == nil, "\(field)")
        }
    }

    /// Over its cap a field keeps its last elements, as the server would, and the result
    /// still goes out.
    @Test func deviceTelemetryOverTheCapKeepsItsLastElements() throws {
        let wait = { (iteration: Int, fill: String) in
            GateWaitElement(iteration: iteration, pollCount: 1, platform: "ios",
                            phase: "proceeding", elapsedMs: 5, observed: fill)
        }
        let half = maxReadinessGateWaitsBytes / 2
        let sent = try #require(try telemetryField(
            "readiness_gate_waits",
            [wait(0, String(repeating: "a", count: half)),
             wait(1, String(repeating: "b", count: half)),
             wait(2, "last")],
            maxBytes: maxReadinessGateWaitsBytes))

        let kept = try #require(
            try JSONSerialization.jsonObject(with: Data(sent.utf8)) as? [[String: Any]])
        #expect(kept.map { $0["iteration"] as? Int } == [1, 2])
    }

    @Test func recordStampsLlamaRuntimeIdentity() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // As `RunCell.dispatchRun` fills them: the cell's values, with the engine's own where
        // a knob was unset. Reported rather than defaulted again, so the string cannot
        // describe a load that did not happen.
        var response = RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                   runtimeOutput: RuntimeOutput())
        response.runtimeFlags = .decodeThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: 20, ctxSize: 1024, nUbatch: 256, threads: nil, swaFull: nil,
            // Spelled out: in an optional position `.none` is `Optional.none`, i.e. nil.
            loadMode: LlamacppLoadMode.read)

        try await record(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: response,
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let payload = try writtenPayload(cellId: "cell-1", ws: ws)
        // The flat wire form with the axes stripped, not an argv string: an iOS engine
        // loads in-process and plan-types gives it no `raw` surface.
        #expect(try flagKnobs(payload["runtime_flags"])
            == ["number_gpu_layers": 20, "ctx_size": 1024, "n_ubatch": 256,
                "load_mode": "none"] as NSDictionary)
        #expect(payload["model_descriptor"] is String)
        // Load knobs live in runtime_flags, not the descriptor.
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "llamacpp_ios_pipette")
        #expect(runtimeObj["repository_url"] as? String == "github.com/ggml-org/llama.cpp")
        #expect(runtimeObj["flavor"] as? String == LlamacppIosPipetteFlavor.iosArm64.rawValue)
        #expect(runtimeObj["repository_version"] as? String == LlamaPins.main.submissionVersion)
    }

    /// As the crate's `ensure_flags_round_tripped`: a record for a run that did not happen
    /// is worse than no record. Checked on the runtime axis, the only one an iOS variant carries.
    @Test func flagsForAnotherCellAreRefused() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var response = RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                   runtimeOutput: RuntimeOutput())
        response.runtimeFlags = .decodeThroughputMlxIosPipetteMlx(nUbatch: 128)

        await #expect(throws: RunRequestError.runtimeFlagsForAnotherCell(
            flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 128),
            cell: (.decodeThroughput, .llamacppIosPipette, .ggufText))) {
            try await record(
                request: payloadRequest(model: try ggufTextResolved(),
                                        benchmarkId: "decode_throughput_512_100"),
                response: response,
                telemetry: RepTelemetry(),
                cellId: "cell-1",
                ws: ws)
        }
    }

    @Test func recordStampsMlxRuntimeIdentity() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        var response = RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                   runtimeOutput: RuntimeOutput())
        response.runtimeFlags = .decodeThroughputMlxIosPipetteMlx(nUbatch: 128)

        try await record(
            request: payloadRequest(
                model: .alreadyBound(.mlx(Mlx(source: .huggingFace(
                    repo: try HfRepo.parseOrgRepo("mlx-community/LFM2-350M-4bit"), prefix: nil)))),
                benchmarkId: "decode_throughput_512_100"),
            response: response,
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let payload = try writtenPayload(cellId: "cell-1", ws: ws)
        // An MLX variant carries the prefill chunk alone; the llama knobs stay absent.
        #expect(try flagKnobs(payload["runtime_flags"]) == ["n_ubatch": 128] as NSDictionary)
        let modelObj = try refObject(payload["model_descriptor"])
        #expect(modelObj["type"] as? String == "mlx")
        #expect(modelObj["source"] as? String == "huggingface")
        #expect(modelObj["org"] as? String == "mlx-community")
        #expect(modelObj["repo_name"] as? String == "LFM2-350M-4bit")
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "mlx_ios_pipette")
        #expect(runtimeObj["flavor"] as? String == MlxIosPipetteFlavor.iosArm64.rawValue)
        let packages = try #require(runtimeObj["packages"] as? [String: Any])
        let mlxSwift = try #require(packages["mlx_swift"] as? [String: Any])
        #expect(mlxSwift["repository_url"] as? String == MLXBuildInfo.mlxSwiftRepositoryUrl)
        #expect(mlxSwift["repository_version"] as? String == MLXBuildInfo.mlxSwift.submissionVersion)
        let mlxSwiftLM = try #require(packages["mlx_swift_lm"] as? [String: Any])
        #expect(mlxSwiftLM["repository_url"] as? String == MLXBuildInfo.mlxSwiftLMRepositoryUrl)
        #expect(mlxSwiftLM["repository_version"] as? String == MLXBuildInfo.mlxSwiftLM.submissionVersion)
        let swiftTransformers = try #require(packages["swift_transformers"] as? [String: Any])
        #expect(swiftTransformers["repository_url"] as? String == MLXBuildInfo.swiftTransformersRepositoryUrl)
        #expect(swiftTransformers["repository_version"] as? String == MLXBuildInfo.swiftTransformers.submissionVersion)
    }

    @Test func recordStampsAfmRefs() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try await record(
            request: payloadRequest(model: .alreadyBound(.appleFoundationText),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                  runtimeOutput: RuntimeOutput()),
            telemetry: RepTelemetry(),
            cellId: "cell-1",
            ws: ws)

        let payload = try writtenPayload(cellId: "cell-1", ws: ws)
        let modelObj = try refObject(payload["model_descriptor"])
        #expect(modelObj["type"] as? String == "apple_foundation_text")
        #expect(modelObj.count == 1)
        let runtimeObj = try refObject(payload["runtime_descriptor"])
        #expect(runtimeObj["type"] as? String == "apple_foundation")
        #expect(runtimeObj.count == 1)
    }

    @Test func recordStampsGgufModelRefs() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try await record(
            request: payloadRequest(model: try ggufTextResolved(),
                                    benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                  runtimeOutput: RuntimeOutput()),
            telemetry: RepTelemetry(),
            cellId: "cell-text",
            ws: ws)
        let textObj = try refObject(try writtenPayload(cellId: "cell-text", ws: ws)["model_descriptor"])
        #expect(textObj["type"] as? String == "gguf_text")
        #expect(textObj["source"] as? String == "huggingface")
        #expect(textObj["org"] as? String == "LiquidAI")
        #expect(textObj["repo_name"] as? String == "LFM2-350M-GGUF")
        #expect(textObj["path"] as? String == "LFM2-350M-Q4_K_M.gguf")

        try await record(
            request: payloadRequest(
                model: .alreadyBound(try ggufVisionSpec("LiquidAI/LFM2.5-VL-450M-GGUF",
                                                        "LFM2.5-VL-450M-Q4_0.gguf",
                                                        "mmproj-f16.gguf")),
                benchmarkId: "decode_throughput_512_100"),
            response: RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                                  runtimeOutput: RuntimeOutput()),
            telemetry: RepTelemetry(),
            cellId: "cell-vision",
            ws: ws)
        let visionObj = try refObject(try writtenPayload(cellId: "cell-vision", ws: ws)["model_descriptor"])
        #expect(visionObj["type"] as? String == "gguf_vision")
        #expect(visionObj["source"] as? String == "huggingface")
        #expect(visionObj["model"] as? String == "LFM2.5-VL-450M-Q4_0.gguf")
        #expect(visionObj["mmproj"] as? String == "mmproj-f16.gguf")
    }

    /// A vault model's `custom_label` is reported as the claim carried it, and a claim
    /// without one gains no key: downstream tells entries apart by descriptor text.
    @Test func recordStampsAVaultModelsCustomLabel() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let bound = try ggufTextResolved().bound

        for (cellId, label) in [("cell-labelled", "acme LFM3 q4"), ("cell-plain", nil)] as [(CellId, String?)] {
            try await record(
                request: payloadRequest(
                    model: DeclaredBound(declared: try vaultTextSpec("vault_1", label: label),
                                         bound: bound),
                    benchmarkId: "decode_throughput_512_100"),
                response: Self.response(),
                telemetry: RepTelemetry(),
                cellId: cellId,
                ws: ws)
            let modelObj = try refObject(try writtenPayload(cellId: cellId, ws: ws)["model_descriptor"])
            #expect(modelObj["source"] as? String == "vault_url")
            #expect(modelObj["custom_label"] as? String == label)
        }
    }

    @Test func emptyDescriptorIsRejected() {
        #expect(throws: ResultSubmissionError.missingDescriptor(field: "model_descriptor")) {
            try ensureDescriptor(field: "model_descriptor", value: "")
        }
    }

    @Test func aFinishedRunCarriesTheWorkspaceDeviceLabels() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        _ = try ws.device.update(deviceName: "bench-box", deviceFormFactor: nil)
        var response = Self.response()
        response.runtimeOutput.runtimeStdout = "out"

        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: payloadRequest(model: try ggufTextResolved()),
            outcome: response, telemetry: RepTelemetry())

        #expect(record.payload.deviceName == "bench-box")
        #expect(record.payload.benchmarkId == "decode_throughput_512_100")
        #expect(record.extras.stdout == "out")
    }

    @Test func aLocalResultIsRecordedAndNeverSubmitted() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: payloadRequest(model: try ggufTextResolved()),
            outcome: Self.response(), telemetry: RepTelemetry())

        try recordResult(results: ws.results, payload: record.payload, extras: record.extras,
                         location: .local, resultId: "cell-1")

        #expect(ws.results.listIds(.local) == ["cell-1"])
        #expect(ws.results.listIds(.remotePending).isEmpty)
        #expect(ws.results.submittableDir("cell-1") == nil)
    }

    /// The time since the claim, in whole milliseconds; a run no claim started files none.
    @Test(arguments: [true, false])
    func aClaimedRunFilesItsDeviceTime(claimed: Bool) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: payloadRequest(model: try ggufTextResolved()),
            outcome: Self.response(), telemetry: RepTelemetry())
        let claimedAt = ContinuousClock.now
        let payload = timedFromClaim(
            record.payload, claimedAt: claimed ? claimedAt : nil,
            now: claimedAt + .milliseconds(1250) + .microseconds(900))

        try recordResult(results: ws.results, payload: payload, extras: record.extras,
                         location: .remotePending, resultId: "cell-1")

        let dir = try #require(ws.results.submittableDir("cell-1"))
        let filed = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: dir.appendingPathComponent("payload.json"))) as? [String: Any])
        #expect(filed["device_time_ms"] as? Int == (claimed ? 1250 : nil))
    }

    /// Left in place, the synced copy shadows the new result and the sweep re-sends the old
    /// payload.
    @Test func aReRunOfASyncedCellReplacesItsSyncedResult() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writePayload(ws: ws, cellId: "cell-1", benchmarkId: "old", at: .remoteSynced)
        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: payloadRequest(model: try ggufTextResolved()),
            outcome: Self.response(), telemetry: RepTelemetry())

        try recordResult(results: ws.results, payload: record.payload, extras: record.extras,
                         location: .remotePending, resultId: "cell-1")

        #expect(ws.results.location(of: "cell-1") == .remotePending)
        let filed = try #require(ws.results.loadPayload("cell-1"))
        let object = try #require(JSONSerialization.jsonObject(with: filed) as? [String: Any])
        #expect(object["benchmark_id"] as? String == "decode_throughput_512_100")
    }

    private static func response() -> RunResponse {
        RunResponse(resultData: .decodeThroughput(decodeTimeMs: 50.0, decodeTimeMsStddev: nil),
                    runtimeOutput: RuntimeOutput())
    }

    /// Saves one run as a finished cell does: filed remote-pending.
    private func record(
        request: RunRequest, response: RunResponse, telemetry: RepTelemetry, cellId: CellId,
        ws: PipetteWorkspace
    ) async throws {
        let record = try await finishedRunPayload(
            deviceStore: ws.device, req: request, outcome: response, telemetry: telemetry)
        try recordResult(results: ws.results, payload: record.payload, extras: record.extras,
                         location: .remotePending, resultId: cellId)
    }

    /// Compared as an object because nothing sorts the keys. `NSDictionary` keeps each
    /// knob's JSON type, so `{"ctx_size":"1024"}` cannot pass for `{"ctx_size":1024}`.
    private func flagKnobs(_ value: Any?) throws -> NSDictionary {
        let string = try #require(value as? String, "runtime_flags must be a JSON string")
        return try #require(
            JSONSerialization.jsonObject(with: Data(string.utf8)) as? NSDictionary,
            "runtime_flags must decode to an object")
    }

    /// Parse a `model_descriptor` / `runtime_descriptor` wire value (a JSON *string*) into its object.
    private func refObject(_ value: Any?) throws -> [String: Any] {
        let string = try #require(value as? String, "ref must be a JSON string")
        return try #require(
            JSONSerialization.jsonObject(with: Data(string.utf8)) as? [String: Any],
            "descriptor string must decode to an object")
    }

    private func writtenPayload(cellId: CellId, ws: PipetteWorkspace) throws -> [String: Any] {
        let data = try #require(ws.results.loadPayload(cellId))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
