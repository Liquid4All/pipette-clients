import Foundation
import Testing

@testable import Pipette

/// `RunCell.prepare`, the one place a `RunRequest` is assembled.
///
/// `.serialized`: these share the coordinator's process-global background session.
@Suite(.serialized) @MainActor struct RunCellTests {

    private func spec(_ model: Model, _ benchmark: BenchmarkDefinition,
                      numberGpuLayers: UInt32? = 99, ctxSize: UInt32? = 4096,
                      nUbatch: UInt32? = 512, threads: UInt32? = nil,
                      swaFull: Bool? = nil) -> ClientRunSpec {
        ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: model,
            runtime: Runtime.thisBuild(for: model),
            runtimeFlags: RuntimeFlagRef(
                benchmarkType: benchmark.benchmarkType,
                runtimeType: RuntimeType.of(Runtime.thisBuild(for: model)),
                modelType: ModelType.of(model),
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull))
    }

    /// Both runtime halves are this build: a cell declares no runtime this client can be,
    /// so there is no second value to carry.
    @Test func anInstalledModelBindsToItsStorePathsAndThisBuild() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(ws, model, payloadBytes: 4096)
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark), benchmark: benchmark,
            ws: ws, coordinator: DownloadCoordinator(ws: ws))

        #expect(request.model.declared == model)
        #expect(request.model.bound.boundPaths?.payload
                == ws.models.payloadPath(for: model))
        #expect(request.runtime.declared == request.runtime.bound)
        #expect(request.runtime.bound == Runtime.thisBuild(for: request.model.declared))
    }

    /// `runtimeFlagsRef` would refuse anything else at run time.
    @Test func theFlagsAreKeyedOnTheCellsAxes() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")
        try installEntry(ws, model)
        let benchmark = BenchmarkDefinition.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100))

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark), benchmark: benchmark,
            ws: ws, coordinator: DownloadCoordinator(ws: ws))

        let flags = try #require(request.runtimeFlags)
        #expect(flags.axes == (.decodeThroughput, .llamacppIosPipette, .ggufText))
        let flat = RuntimeFlagRef(flags)
        #expect(flat.numberGpuLayers == 99)
        #expect(flat.ctxSize == 4096)
        #expect(flat.nUbatch == 512)
    }

    /// Apple Foundation ships with the OS: nothing to ensure, no path, and no runtime flags
    /// variant upstream, so the cell carries no entry at all.
    @Test func appleFoundationSkipsTheStoreEntirely() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let coordinator = DownloadCoordinator(ws: ws)
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))

        let request = try await RunCell.prepare(
            spec: ClientRunSpec(benchmark: benchmark.benchmarkId, model: .appleFoundationText,
                                runtime: Runtime.thisBuild(for: .appleFoundationText)),
            benchmark: benchmark,
            ws: ws, coordinator: coordinator)

        #expect(request.model.declared == .appleFoundationText)
        #expect(request.model.bound == .appleFoundationText)
        #expect(request.runtime.bound == .appleFoundation(privateApis: .thisBuild))
        #expect(request.runtimeFlags == nil)
        #expect(!coordinator.hasTransfer(for: .appleFoundationText))
    }

    /// `authored` states what the app's own settings apply to, so it leaves Apple
    /// Foundation alone: an entry for a cell with no variant is refused below.
    @Test func anAuthoredAppleFoundationCellCarriesNoFlags() {
        let spec = ClientRunSpec.authored(
            benchmarkId: "prefill_throughput_512", benchmarkType: .prefillThroughput,
            model: .appleFoundationText, numberGpuLayers: 99, nUbatch: 512)

        #expect(spec.runtimeFlags == nil)
    }

    /// The spec names a benchmark and the caller resolves the body, so the two have to
    /// agree: a cell whose halves disagree would be recorded under an id that never ran.
    @Test func aSpecNamingAnotherBenchmarkIsRefused() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try ggufTextSpec("org/e-GGUF", "e-Q4_0.gguf")
        try installEntry(ws, model)
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        let other = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
            parameterPrefillTokens: 256))

        await #expect(throws: RunCellError.benchmarkMismatch(
            spec: "prefill_throughput_256", body: "prefill_throughput_512")) {
            _ = try await RunCell.prepare(
                spec: spec(model, other), benchmark: benchmark,
                ws: ws, coordinator: DownloadCoordinator(ws: ws))
        }
    }

    /// A flag group is checked against the *resolved* body rather than the type an id
    /// suggested: an entry for another cell would otherwise resolve to that cell's variant
    /// and be applied here.
    @Test func flagsForAnotherCellAreRefused() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try ggufTextSpec("org/f-GGUF", "f-Q4_0.gguf")
        try installEntry(ws, model)
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        // The cell is a prefill one; the entry names eval.
        let mismatched = ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: model,
            runtime: Runtime.thisBuild(for: model),
            runtimeFlags: RuntimeFlagRef(benchmarkType: .eval, runtimeType: .llamacppIosPipette,
                                         modelType: .ggufText, ctxSize: 4096))

        await #expect(throws: RunCellError.flagsAreNotForThisCell(
            group: "runtime_flags",
            carried: (.eval, .llamacppIosPipette, .ggufText),
            cell: (.prefillThroughput, .llamacppIosPipette, .ggufText))) {
            _ = try await RunCell.prepare(
                spec: mismatched, benchmark: benchmark,
                ws: ws, coordinator: DownloadCoordinator(ws: ws))
        }
    }

    /// A triple plan-types defines no variant for is refused rather than dropped: running
    /// on engine defaults and reporting those as what the cell asked for is the failure
    /// the flags exist to prevent.
    @Test func anEntryWithNoVariantForThisCellIsRefused() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))
        let afm = ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: .appleFoundationText,
            runtime: Runtime.thisBuild(for: .appleFoundationText),
            runtimeFlags: RuntimeFlagRef(benchmarkType: .prefillThroughput,
                                         runtimeType: .appleFoundation,
                                         modelType: .appleFoundationText, nUbatch: 512))

        await #expect(throws: RuntimeFlagsError.noSuchCombination(
            benchmark: .prefillThroughput, runtime: .appleFoundation,
            model: .appleFoundationText)) {
            _ = try await RunCell.prepare(
                spec: afm, benchmark: benchmark,
                ws: ws, coordinator: DownloadCoordinator(ws: ws))
        }
    }

    /// Unchanged, including a knob no job setting spells: flags rebuilt from an argument
    /// per knob would lose it.
    @Test func theAuthoredEntryReachesTheRequest() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try ggufTextSpec("org/c-GGUF", "c-Q4_0.gguf")
        try installEntry(ws, model)
        let benchmark = BenchmarkDefinition.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512))

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark, ctxSize: 2048, threads: 3, swaFull: false),
            benchmark: benchmark,
            ws: ws, coordinator: DownloadCoordinator(ws: ws))

        let flat = RuntimeFlagRef(try #require(request.runtimeFlags))
        #expect(flat.ctxSize == 2048)
        #expect(flat.threads == 3)
        #expect(flat.swaFull == false)
    }

    /// Degrading it to "no flags" would run on engine defaults and report those as what
    /// the cell asked for: a result that looks valid and answers a different question.
    @Test func anEntryNamingAForeignKnobFailsTheCell() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try mlxSpec("org/m-MLX-4bit")
        try installEntry(ws, model)
        let benchmark = BenchmarkDefinition.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100))
        // `ctx_size` is a llama knob; an MLX cell's variant carries the prefill chunk alone.
        let foreign = ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: model,
            runtime: Runtime.thisBuild(for: model),
            runtimeFlags: RuntimeFlagRef(
                benchmarkType: benchmark.benchmarkType, runtimeType: .mlxIosPipette, modelType: .mlx,
                ctxSize: 4096))

        await #expect(throws: RuntimeFlagsError.self) {
            _ = try await RunCell.prepare(
                spec: foreign, benchmark: benchmark,
                ws: ws, coordinator: DownloadCoordinator(ws: ws))
        }
    }

    /// Substituting a cell's load settings must not cost it the groups it named. The
    /// executor stands in for a pre-spec cell's `runtime_flags`; a rebuild there would drop
    /// the readiness gate from the record while the gate still held the run.
    @Test func replacingTheLoadSettingsKeepsTheOtherGroups() throws {
        let model = try ggufTextSpec("org/d-GGUF", "d-Q4_0.gguf")
        let gate = BenchmarkFlagRef(
            benchmarkType: .prefillThroughput, runtimeType: .llamacppIosPipette,
            modelType: .ggufText,
            readiness: ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true))
        let original = ClientRunSpec(
            benchmark: "prefill_throughput_512", model: model,
            runtime: Runtime.thisBuild(for: model), benchmarkFlags: gate)

        let replaced = original.replacingRuntimeFlags(
            RuntimeFlagRef(benchmarkType: .prefillThroughput, runtimeType: .llamacppIosPipette,
                           modelType: .ggufText, numberGpuLayers: 99))

        #expect(replaced.runtimeFlags?.numberGpuLayers == 99)
        #expect(replaced.benchmarkFlags?.readiness?.maxWaitSecs == 1800)
        #expect(replaced.benchmarkFlags?.readiness?.skipThermal == true)
        #expect(replaced.model == original.model)
    }

    @Test func anUnfetchableModelIsRefusedBeforeAnythingLoads() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let url = try ResourceUrl("https://example.com/weights.gguf")
        let model = Model.ggufText(GgufText(source: .url(url: url, sha256: nil)))
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
            parameterPrefillTokens: 512))

        await #expect(throws: ModelStoreError.notFetchableHere(model.artifactName)) {
            _ = try await RunCell.prepare(
                spec: spec(model, benchmark), benchmark: benchmark,
                ws: ws, coordinator: DownloadCoordinator(ws: ws))
        }
    }

    // MARK: - The policy a cell asks for: `readiness_max_wait` / `readiness_thermal_gate`

    private func request(flags: BenchmarkFlags?) throws -> RunRequest {
        let model = try ggufTextSpec("org/g-GGUF", "g-Q4_0.gguf")
        return RunRequest(runtime: .alreadyBound(Runtime.thisBuild(for: model)),
                          model: .alreadyBound(model), runtimeFlags: nil, benchmarkFlags: flags,
                          benchmark: .prefillThroughput(PrefillThroughputBenchmark(
                              benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
                              parameterPrefillTokens: 256)))
    }

    /// The crate's `the_deadline_comes_from_the_cells_flags`.
    @Test(arguments: [
        (nil, nil),
        (readinessFlags(nil, true), nil),
        (readinessFlags(1800, nil), Duration.seconds(1800)),
    ] as [(BenchmarkFlags?, Duration?)])
    func theDeadlineComesFromTheCellsFlags(flags: BenchmarkFlags?, expected: Duration?) throws {
        #expect(RunCell.readinessMaxWait(try request(flags: flags)) == expected)
    }

    /// Three answers, not two: an absent knob is `unset`, while an authored `false`
    /// enforces. The crate's `the_thermal_gate_keeps_unset_apart_from_false`.
    @Test(arguments: [
        (nil, ThermalGate.unset),
        (readinessFlags(60, nil), .unset),
        (readinessFlags(nil, true), .skip),
        (readinessFlags(nil, false), .enforce),
    ] as [(BenchmarkFlags?, ThermalGate)])
    func theThermalGateKeepsUnsetApartFromFalse(
        flags: BenchmarkFlags?, expected: ThermalGate
    ) throws {
        #expect(RunCell.readinessThermalGate(try request(flags: flags)) == expected)
    }

    // MARK: - The policy a result reports: the crate's `resolved_flags`

    /// A cell running `benchmark` with no authored benchmark flags, so what the record
    /// carries can only have come from resolution.
    private func request(_ benchmark: BenchmarkDefinition) throws -> RunRequest {
        let model = try ggufTextSpec("org/g-GGUF", "g-Q4_0.gguf")
        return RunRequest(runtime: .alreadyBound(Runtime.thisBuild(for: model)),
                          model: .alreadyBound(model), runtimeFlags: nil, benchmarkFlags: nil,
                          benchmark: benchmark)
    }

    /// The knobs a submitted `benchmark_flags` carries, as a dictionary: nothing sorts the
    /// keys, so the field order is the encoder's business, and the crate asserts the same
    /// way against a `serde_json::Value`.
    private func submitted(_ flags: BenchmarkFlags) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(flags.submissionValue().utf8))
        return try #require(object as? [String: Any])
    }

    /// A cell that gates reports what it gated on, even though it authored nothing: the
    /// resolved policy is the whole point of the field. The authored block cannot describe
    /// the run, since an absent deadline becomes the platform's.
    @Test func aGatedCellReportsThePolicyItApplied() throws {
        let benchmark = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
            parameterPrefillTokens: 256))

        let flags = try #require(RunCell.resolvedFlags(
            try request(benchmark),
            readiness: resolveReadiness(maxWait: .fromSecs(300), thermal: .enforce)))

        #expect(flags.readiness == ReadinessOverrides(maxWaitSecs: 300, skipThermal: false))
        let readiness = try #require(try submitted(flags)["readiness"] as? [String: Any])
        #expect(readiness["max_wait_secs"] as? Int == 300)
        #expect(readiness["skip_thermal"] as? Bool == false)
    }

    @Test func aVisionTimingCellReportsThePolicyItApplied() throws {
        let model = try ggufVisionSpec("org/v-GGUF", "v-Q4_0.gguf", "mmproj-v-F16.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)), model: .alreadyBound(model),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .vlThroughput(VlThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_throughput_512x512_text0_decode64_img1"),
                parameterImageWidth: 512, parameterImageHeight: 512, parameterTextTokens: 0,
                parameterDecodeTokens: 64)))

        let flags = try #require(try RunCell.resolvedFlags(
            request, readiness: resolveReadiness(maxWait: .fromSecs(300), thermal: .enforce)))

        #expect(flags == .vlThroughputLlamacppIosPipetteGgufVision(
            readiness: ReadinessOverrides(maxWaitSecs: 300, skipThermal: false)))
    }

    /// A cell run with the temperature criterion waived is not comparable to a gated one,
    /// and the record is the only place that difference survives.
    @Test func aWaivedTemperatureCriterionIsReported() throws {
        let benchmark = BenchmarkDefinition.decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100))

        let flags = try #require(RunCell.resolvedFlags(
            try request(benchmark),
            readiness: resolveReadiness(maxWait: .fromSecs(1800), thermal: .skip)))

        let readiness = try #require(try submitted(flags)["readiness"] as? [String: Any])
        #expect(readiness["max_wait_secs"] as? Int == 1800)
        #expect(readiness["skip_thermal"] as? Bool == true)
    }

    /// Eval never calls the gate, so reporting a readiness block would claim a wait that
    /// never happened. The crate's eval variant carries no readiness field; iOS declares no
    /// eval variant at all, so the answer is nothing either way.
    @Test func anEvalCellReportsNoReadiness() throws {
        let benchmark = BenchmarkDefinition.eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "ifbench"),
            parameterEvalId: EvalId("ifbench"),
            parameterDatasetName: "ifbench",
            parameterMaxTokens: 64,
            parameterMcqChoices: nil,
            samples: nil))

        #expect(try RunCell.resolvedFlags(try request(benchmark),
                                          readiness: ResolvedReadiness()) == nil)
    }

    /// A cell that gates on nothing and authored none of its knobs reports nothing:
    /// submitting `{}` is what the server NULLs anyway, so `nil` says the same thing while
    /// being honest locally.
    @Test func aCellWithNothingSetReportsNothing() throws {
        let benchmark = BenchmarkDefinition.maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
            parameterPrefillTokens: 512))

        #expect(try RunCell.resolvedFlags(try request(benchmark),
                                          readiness: ResolvedReadiness()) == nil)
    }

    /// The limit a plan set is what the cell ran under, so the record carries it beside the
    /// gate; a defaulted one is not reported.
    @Test func onlyAnAuthoredAttemptLimitIsReported() throws {
        let gate = resolveReadiness(maxWait: .fromSecs(300), thermal: .enforce)
        let authored = try #require(try RunCell.resolvedFlags(
            try request(flags: .prefillThroughputLlamacppIosPipetteGgufText(
                attemptTimeoutSeconds: 600, readiness: nil)),
            readiness: gate))
        #expect(try submitted(authored)["attempt_timeout_seconds"] as? Int == 600)

        let defaulted = try #require(try RunCell.resolvedFlags(try request(flags: nil),
                                                               readiness: gate))
        #expect(try submitted(defaulted)["attempt_timeout_seconds"] == nil)
        #expect(defaulted.readiness == ReadinessOverrides(maxWaitSecs: 300, skipThermal: false))
    }

    /// A memory cell has no gate to report, so its authored limit is the whole record.
    @Test func aMemoryCellReportsItsAuthoredLimit() throws {
        let model = try ggufTextSpec("org/g-GGUF", "g-Q4_0.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)), model: .alreadyBound(model),
            runtimeFlags: nil,
            benchmarkFlags: .maxMemoryUsageLlamacppIosPipetteGgufText(attemptTimeoutSeconds: 2400),
            benchmark: .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
                parameterPrefillTokens: 512)))

        let flags = try #require(try RunCell.resolvedFlags(request, readiness: ResolvedReadiness()))
        #expect(try submitted(flags) as NSDictionary == ["attempt_timeout_seconds": 2400] as NSDictionary)
    }

    /// Apple Foundation's timing cells gate as the others do, and now have a variant to say so.
    @Test func anAppleFoundationTimingCellReportsThePolicyItApplied() throws {
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: .appleFoundationText)),
            model: .alreadyBound(.appleFoundationText), runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_256_100"),
                parameterPrefillTokens: 256, parameterDecodeTokens: 100)))

        let flags = try #require(try RunCell.resolvedFlags(
            request, readiness: resolveReadiness(maxWait: .fromSecs(300), thermal: .enforce)))
        #expect(flags == .decodeThroughputAppleFoundationAppleFoundationText(
            readiness: ReadinessOverrides(maxWaitSecs: 300, skipThermal: false)))
    }

    /// Charging is refused with discharging: a fast-charging battery throttles the GPU
    /// even on AC. An unreadable state warns rather than refusing, so the cell stays
    /// runnable on a device this client cannot probe.
    @Test(arguments: [
        (DevicePowerState.pluggedInNotCharging, RunCell.PowerVerdict.stable),
        (.charging, .moving),
        (.notCharging, .moving),
        (nil, .unknown),
    ] as [(DevicePowerState?, RunCell.PowerVerdict)])
    func thePowerGateAdmitsOnlyAStateThatHoldsStill(_ state: DevicePowerState?,
                                                    _ expected: RunCell.PowerVerdict) {
        #expect(RunCell.powerVerdict(state) == expected)
    }

    /// Each measured rep is one `.attempt`, the screen's "Measurement n/5"; the corpus
    /// traversal, which has no rep total, reports none.
    @Test func eachMeasuredRepReportsAnAttempt() {
        var events: [BenchmarkProgress] = []
        let measured = RepProgress(total: 5) { events.append($0) }
        let traversal = RepProgress(total: nil) { events.append($0) }
        (1 ... 5).forEach { _ in
            measured.finished()
            traversal.finished()
        }
        #expect(events == (1 ... 5).map { .attempt(completed: $0, total: 5) })
    }
}

/// A prefill cell's flags carrying `readiness`. File-scope so a nonisolated `@Test(arguments:)`
/// can build them.
private nonisolated func readinessFlags(_ maxWaitSecs: UInt64?, _ skipThermal: Bool?)
    -> BenchmarkFlags {
    .prefillThroughputLlamacppIosPipetteGgufText(
        readiness: ReadinessOverrides(maxWaitSecs: maxWaitSecs, skipThermal: skipThermal))
}
