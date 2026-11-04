import Foundation
import Testing

@testable import Pipette

/// The iOS subset of plan-types' `cell/runtime_flags.rs` (pipette-mgmt), whose
/// test names the cases below cite. A knob no iOS variant declares is refused while
/// decoding; one declared but wrong for this cell's variant is refused by `resolve()`.
struct RuntimeFlagsTests {

    private func group(
        benchmark: String = "prefill_throughput",
        runtime: String = "llamacpp_ios_pipette",
        model: String = "gguf_text",
        _ knobs: [String: Any] = [:]
    ) throws -> Data {
        var group: [String: Any] = [
            "benchmark_type": benchmark, "runtime_type": runtime, "model_type": model,
        ]
        for (key, value) in knobs { group[key] = value }
        return try JSONSerialization.data(withJSONObject: group)
    }

    private func decode(_ data: Data) throws -> RuntimeFlagRef {
        try JSONDecoder().decode(RuntimeFlagRef.self, from: data)
    }

    private func parse(benchmark: String, runtime: String, model: String,
                       _ knobs: [String: Any] = [:]) throws -> RuntimeFlags {
        try decode(try group(benchmark: benchmark, runtime: runtime, model: model, knobs)).resolve()
    }

    private func object(_ json: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    /// The axes come off the ref, so an expectation stays about the refusal.
    private func noVariant(_ ref: RuntimeFlagRef) -> RuntimeFlagsError {
        .noSuchCombination(benchmark: ref.benchmarkType, runtime: ref.runtimeType,
                           model: ref.modelType)
    }

    private func knobRefused(_ knob: String, _ ref: RuntimeFlagRef) -> RuntimeFlagsError {
        .knobNotAllowed(knob: knob, benchmark: ref.benchmarkType,
                        runtime: ref.runtimeType, model: ref.modelType)
    }

    /// Exhaustive, so a new `RuntimeFlags` variant fails to compile here until the sweeps
    /// account for it.
    private func variant(_ flags: RuntimeFlags) -> String {
        switch flags {
        case .prefillThroughputLlamacppIosPipetteGgufText: "prefillThroughputLlamacppIosPipetteGgufText"
        case .decodeThroughputLlamacppIosPipetteGgufText: "decodeThroughputLlamacppIosPipetteGgufText"
        case .maxMemoryUsageLlamacppIosPipetteGgufText: "maxMemoryUsageLlamacppIosPipetteGgufText"
        case .endToEndLatencyLlamacppIosPipetteGgufText: "endToEndLatencyLlamacppIosPipetteGgufText"
        case .evalLlamacppIosPipetteGgufText: "evalLlamacppIosPipetteGgufText"
        case .vlThroughputLlamacppIosPipetteGgufVision: "vlThroughputLlamacppIosPipetteGgufVision"
        case .vlMaxMemoryLlamacppIosPipetteGgufVision: "vlMaxMemoryLlamacppIosPipetteGgufVision"
        case .vlSpecDecodeLlamacppIosPipetteGgufVision:
            "vlSpecDecodeLlamacppIosPipetteGgufVision"
        case .vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative:
            "vlSpecDecodeLlamacppIosPipetteGgufVisionSpeculative"
        case .asrThroughputLlamacppIosPipetteGgufVision: "asrThroughputLlamacppIosPipetteGgufVision"
        case .asrMaxMemoryLlamacppIosPipetteGgufVision: "asrMaxMemoryLlamacppIosPipetteGgufVision"
        case .prefillThroughputMlxIosPipetteMlx: "prefillThroughputMlxIosPipetteMlx"
        case .decodeThroughputMlxIosPipetteMlx: "decodeThroughputMlxIosPipetteMlx"
        case .maxMemoryUsageMlxIosPipetteMlx: "maxMemoryUsageMlxIosPipetteMlx"
        case .endToEndLatencyMlxIosPipetteMlx: "endToEndLatencyMlxIosPipetteMlx"
        case .evalMlxIosPipetteMlx: "evalMlxIosPipetteMlx"
        }
    }

    private static let llamaFields = ["number_gpu_layers", "ctx_size", "n_ubatch", "threads",
                                      "swa_full", "load_mode"]

    /// The crate's `CELLS` rows for `llamacpp_ios_pipette` and `mlx_ios_pipette`, transcribed
    /// rather than derived, so a field added to or dropped from a variant fails here.
    private static let cells: [(benchmark: String, runtime: String, model: String, accepted: [String])] = [
        ("prefill_throughput", "llamacpp_ios_pipette", "gguf_text", llamaFields),
        ("decode_throughput", "llamacpp_ios_pipette", "gguf_text", llamaFields),
        ("max_memory_usage", "llamacpp_ios_pipette", "gguf_text", llamaFields),
        ("end_to_end_latency", "llamacpp_ios_pipette", "gguf_text", llamaFields),
        ("eval", "llamacpp_ios_pipette", "gguf_text", llamaFields),
        ("vl_throughput", "llamacpp_ios_pipette", "gguf_vision", llamaFields),
        ("vl_max_memory", "llamacpp_ios_pipette", "gguf_vision", llamaFields),
        // Only `speculative_enabled` beyond the load knobs: the engine is this process, so
        // there is no environment to set and no argv to append.
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision",
         llamaFields + ["speculative_enabled"]),
        ("vl_spec_decode", "llamacpp_ios_pipette", "gguf_vision_speculative",
         llamaFields + ["speculative_enabled"]),
        ("asr_throughput", "llamacpp_ios_pipette", "gguf_vision", llamaFields),
        ("asr_max_memory", "llamacpp_ios_pipette", "gguf_vision", llamaFields),
        ("prefill_throughput", "mlx_ios_pipette", "mlx", ["n_ubatch"]),
        ("decode_throughput", "mlx_ios_pipette", "mlx", ["n_ubatch"]),
        ("max_memory_usage", "mlx_ios_pipette", "mlx", ["n_ubatch"]),
        ("end_to_end_latency", "mlx_ios_pipette", "mlx", ["n_ubatch"]),
        ("eval", "mlx_ios_pipette", "mlx", ["n_ubatch"]),
    ]

    /// The crate's `PROBES`. The fields no iOS variant declares are refused as unknown, so
    /// one sweep covers both refusal mechanisms.
    private static var probes: [(field: String, value: Any)] {
        [
            ("threads", 4), ("number_gpu_layers", 99), ("mmap", true),
            ("load_mode", "mmap_mlock"), ("flash_attention", "auto"), ("ctx_size", 8192),
            ("n_ubatch", 512), ("swa_full", true), ("no_cache", true), ("parallel", 4),
            ("repetition_detection", ["min_pattern_size": 2, "max_pattern_size": 128, "min_count": 10]),
            ("tensor_parallel_size", 2), ("dtype", "bfloat16"), ("max_model_len", 8192),
            ("device", "gpu"), ("max_prompt_len", 4096), ("min_response_len", 256),
            ("generate_hint", "best-perf"), ("prefix_caching", true), ("gpus", "all"),
            ("shm_size", "16g"), ("ipc", "host"), ("envs", ["HF_TOKEN=x"]),
            ("raw", ["--pipette-field-probe"]),
            // `false`: the one value both corpus cells accept, since `true` needs a drafter.
            ("speculative_enabled", false),
        ]
    }

    // MARK: - Routing

    /// `variants_are_named_for_their_cell`, iOS rows.
    @Test func everyIosTripleRoutesToAVariantThatReportsIt() throws {
        for cell in Self.cells {
            let flags = try parse(benchmark: cell.benchmark, runtime: cell.runtime, model: cell.model)
            #expect(flags.axes.benchmark.rawValue == cell.benchmark)
            #expect(flags.axes.runtime.rawValue == cell.runtime)
            #expect(flags.axes.model.rawValue == cell.model)
            #expect(try RuntimeFlagRef(flags).resolve() == flags)
        }
    }

    /// `cell_table_covers_every_variant`.
    @Test func theCellTableCoversEveryVariant() throws {
        let seen = try Self.cells.map {
            variant(try parse(benchmark: $0.benchmark, runtime: $0.runtime, model: $0.model))
        }
        #expect(Set(seen).count == seen.count, "two rows share a variant")
        #expect(seen.count == 16, "the crate declares sixteen iOS variants")
    }

    /// A missing `deny` silently drops an author's flag, a stray one rejects a legitimate
    /// cell, and an arm that accepts a field but forgets to read it drops the value. The
    /// value is compared against the authored wire, since a dropped field is absent from
    /// both sides of a round trip.
    @Test func everyCellAcceptsExactlyItsDeclaredFieldsAndKeepsTheirValues() throws {
        for cell in Self.cells {
            for probe in Self.probes {
                let expected = cell.accepted.contains(probe.field)
                let parsed = try? parse(benchmark: cell.benchmark, runtime: cell.runtime,
                                        model: cell.model, [probe.field: probe.value])
                let label = "\(cell.benchmark)/\(cell.runtime)/\(cell.model): `\(probe.field)`"
                #expect((parsed != nil) == expected, "\(label) accepted = \(parsed != nil)")
                guard let parsed else { continue }
                let emitted = try JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(RuntimeFlagRef(parsed))) as? [String: Any]
                #expect((emitted?[probe.field] as? NSObject) == (probe.value as? NSObject),
                        "\(label) came back as \(String(describing: emitted?[probe.field]))")
            }
        }
    }

    @Test func anIosLlamaCellCarriesEveryDeclaredKnob() throws {
        let ref = try decode(try group([
            "number_gpu_layers": 8, "ctx_size": 4096, "n_ubatch": 512, "threads": 6,
            "swa_full": false, "load_mode": "mmap",
        ]))

        #expect(try ref.resolve() == .prefillThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: 8, ctxSize: 4096, nUbatch: 512, threads: 6, swaFull: false,
            loadMode: .mmap))
    }

    /// `ios_llama_knobs_round_trip`.
    @Test func theIosLlamaKnobsRoundTrip() throws {
        let flags = try parse(benchmark: "eval", runtime: "llamacpp_ios_pipette", model: "gguf_text", [
            "number_gpu_layers": 99, "ctx_size": 8192, "n_ubatch": 512, "threads": 6,
            "swa_full": false, "load_mode": "mmap",
        ])
        #expect(flags == .evalLlamacppIosPipetteGgufText(
            numberGpuLayers: 99, ctxSize: 8192, nUbatch: 512, threads: 6, swaFull: false,
            loadMode: .mmap))
        let wire = try JSONEncoder().encode(RuntimeFlagRef(flags))
        #expect(try JSONDecoder().decode(RuntimeFlagRef.self, from: wire).resolve() == flags)
    }

    /// `load_mode_spellings`.
    @Test(arguments: LlamacppLoadMode.allCases)
    func everyLoadModeSurvivesThePlanWire(mode: LlamacppLoadMode) throws {
        let flags = try parse(benchmark: "eval", runtime: "llamacpp_ios_pipette",
                              model: "gguf_text", ["load_mode": mode.rawValue])
        #expect(flags.loadMode == mode)
        let wire = try JSONEncoder().encode(RuntimeFlagRef(flags))
        #expect(try JSONDecoder().decode(RuntimeFlagRef.self, from: wire).resolve() == flags)
    }

    @Test func aGroupWithNoKnobsResolves() throws {
        #expect(try decode(try group()).resolve() == .prefillThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: nil, swaFull: nil,
            loadMode: nil))
    }

    @Test func anIosMlxCellCarriesThePrefillChunkOnly() throws {
        let ref = try decode(try group(runtime: "mlx_ios_pipette", model: "mlx", ["n_ubatch": 256]))

        #expect(try ref.resolve() == .prefillThroughputMlxIosPipetteMlx(nUbatch: 256))
    }

    /// `matches` — all three axes.
    @Test func flagsMatchTheirCellAlone() throws {
        let flags = RuntimeFlags.decodeThroughputMlxIosPipetteMlx(nUbatch: nil)
        let mlx = try mlxSpec("org/m-MLX-4bit")
        let runtime = Runtime.thisBuild(for: mlx)
        #expect(flags.matches(benchmark: .decodeThroughput, runtime: runtime, model: mlx))
        #expect(!flags.matches(benchmark: .prefillThroughput, runtime: runtime, model: mlx))
        let gguf = try ggufTextSpec("org/a-GGUF", "a.gguf")
        #expect(!flags.matches(benchmark: .decodeThroughput, runtime: Runtime.thisBuild(for: gguf),
                               model: gguf))
    }

    // MARK: - Refused while decoding: not declared at all

    /// `ios_llama_takes_threads_but_not_the_argv_knobs` and `ios_rejects_raw`.
    @Test(arguments: [
        ("dtype", "float16" as Any),
        ("tensor_parallel_size", 2 as Any),
        ("gpus", "all" as Any),
        ("shm_size", "1g" as Any),
        ("ipc", "host" as Any),
        ("mmap", true as Any),
        ("flash_attention", "on" as Any),
        ("no_cache", true as Any),
        ("envs", ["A=1"] as Any),
        ("raw", ["--verbose"] as Any),
    ])
    func aKnobNoIosVariantCarriesIsRefusedAsUnknown(_ name: String, _ value: Any) throws {
        let data = try group([name: value])

        #expect(throws: UnknownFlagField.self) { _ = try decode(data) }
    }

    // MARK: - Refused while resolving: declared, wrong variant

    /// `ios_mlx_rejects_ngl`, and the other llama settings an MLX variant has no home for.
    @Test(arguments: ["number_gpu_layers", "ctx_size", "threads"])
    func aLlamaKnobOnAnMlxCellIsRefusedByName(_ knob: String) throws {
        let ref = try decode(try group(runtime: "mlx_ios_pipette", model: "mlx", [knob: 8]))

        #expect(throws: knobRefused(knob, ref)) { _ = try ref.resolve() }
    }

    @Test func theLoadModeOnAnMlxCellIsRefusedByName() throws {
        let ref = try decode(try group(runtime: "mlx_ios_pipette", model: "mlx", ["load_mode": "mmap"]))

        #expect(throws: knobRefused("load_mode", ref)) { _ = try ref.resolve() }
    }

    @Test func theSwaPolicyOnAnMlxCellIsRefusedByName() throws {
        let ref = try decode(try group(runtime: "mlx_ios_pipette", model: "mlx", ["swa_full": true]))

        #expect(throws: knobRefused("swa_full", ref)) { _ = try ref.resolve() }
    }

    /// One knob is dropped per step, so each position in the crate's order is pinned.
    @Test func anMlxCellRefusesLlamaKnobsInTheCratesOrder() throws {
        var knobs: [String: Any] = [
            "swa_full": true, "threads": 4, "number_gpu_layers": 1, "load_mode": "mmap",
            "ctx_size": 512,
        ]
        for knob in ["swa_full", "threads", "number_gpu_layers", "load_mode", "ctx_size"] {
            let ref = try decode(try group(runtime: "mlx_ios_pipette", model: "mlx", knobs))
            #expect(throws: knobRefused(knob, ref)) { _ = try ref.resolve() }
            knobs.removeValue(forKey: knob)
        }
    }

    /// `swa_full` is the iOS llama.cpp engine's alone, so the crate checks it before routing.
    @Test func theSwaPolicyIsRefusedBeforeAMissingVariant() throws {
        let ref = try decode(try group(runtime: "apple_foundation", model: "apple_foundation_text",
                                       ["swa_full": false]))

        #expect(throws: knobRefused("swa_full", ref)) { _ = try ref.resolve() }
    }

    // MARK: - No variant at all

    @Test func anAppleFoundationCellHasNoVariant() throws {
        let ref = try decode(try group(runtime: "apple_foundation", model: "apple_foundation_text"))

        #expect(throws: noVariant(ref)) { _ = try ref.resolve() }
    }

    @Test func theVisionKindsResolveOnlyWhereTheCrateDeclaresThem() throws {
        for benchmark in ["vl_throughput", "vl_max_memory", "asr_throughput", "asr_max_memory"] {
            let vision = try decode(try group(benchmark: benchmark, model: "gguf_vision"))
            #expect((try? vision.resolve()) != nil, "\(benchmark)")

            let text = try decode(try group(benchmark: benchmark, model: "gguf_text"))
            #expect(throws: noVariant(text)) { _ = try text.resolve() }
        }
    }

    /// Refused as a cell this client cannot run, rather than as an unreadable payload.
    @Test func aDesktopRuntimeDecodesAndThenHasNoVariant() throws {
        let ref = try decode(try group(runtime: "docker_vllm", model: "torch"))

        #expect(throws: noVariant(ref)) { _ = try ref.resolve() }
    }

    // MARK: - Submission value

    /// `submission_value_is_empty_when_the_cell_configures_nothing`: the axis keys are the
    /// payload's job.
    @Test func theSubmissionValueIsEmptyWhenTheCellConfiguresNothing() throws {
        let flags = try parse(benchmark: "eval", runtime: "llamacpp_ios_pipette", model: "gguf_text")
        #expect(try flags.submissionValue() == "{}")
    }

    /// `submission_value_reports_the_cells_settings`.
    @Test func theSubmissionValueReportsTheCellsSettings() throws {
        let flags = try parse(benchmark: "eval", runtime: "llamacpp_ios_pipette", model: "gguf_text",
                              ["number_gpu_layers": 7, "threads": 2, "load_mode": "direct_io"])
        let reported = try object(try flags.submissionValue())

        #expect(reported["number_gpu_layers"] as? Int == 7)
        #expect(reported["threads"] as? Int == 2)
        #expect(reported["load_mode"] as? String == "direct_io")
        #expect(reported.count == 3)
        for key in RuntimeFlagRef.axisKeys { #expect(reported[key] == nil) }
    }

    /// Upstream renders a `BTreeMap`, so its keys arrive sorted. Asserted on the string,
    /// because parsing it back into a dictionary cannot see the ordering go wrong.
    @Test func theSubmissionValueRendersItsKeysSortedAsTheCrateDoes() throws {
        let flags = try parse(benchmark: "eval", runtime: "llamacpp_ios_pipette", model: "gguf_text",
                              ["number_gpu_layers": 7, "threads": 2, "load_mode": "direct_io"])

        #expect(try flags.submissionValue()
            == #"{"load_mode":"direct_io","number_gpu_layers":7,"threads":2}"#)
    }

    // MARK: - fromCellJson

    /// Checked before routing, so even a cell with no variant: the crate's `from_cell_json`
    /// returns `None` first.
    @Test func anEmptyKnobSetIsNoFlags() throws {
        #expect(try RuntimeFlags.fromCellJson(
            Data("{}".utf8), runtimeType: .appleFoundation, modelType: .appleFoundationText,
            benchmarkType: .eval) == nil)
    }

    @Test func theKnobsResolveAgainstTheCellTheCallerNamed() throws {
        let flags = try RuntimeFlags.fromCellJson(
            Data(#"{"threads":4}"#.utf8), runtimeType: .llamacppIosPipette, modelType: .ggufText,
            benchmarkType: .decodeThroughput)
        #expect(flags == .decodeThroughputLlamacppIosPipetteGgufText(
            numberGpuLayers: nil, ctxSize: nil, nUbatch: nil, threads: 4, swaFull: nil,
            loadMode: nil))
    }

    /// An array is the wire before the axes were derived.
    @Test(arguments: [#"[{"threads":4}]"#, "4", #""threads""#])
    func aValueThatIsNotAnObjectIsRefusedAsSuch(json: String) {
        #expect(throws: RuntimeFlagsError.notAnObject) {
            try RuntimeFlags.fromCellJson(Data(json.utf8), runtimeType: .mlxIosPipette,
                                          modelType: .mlx, benchmarkType: .prefillThroughput)
        }
    }

    /// The cell comes from the arguments, so naming an axis is refused rather than checked.
    @Test(arguments: ["runtime_type", "model_type", "benchmark_type"])
    func anAxisKeyIsNotTheCallersToSupply(key: String) {
        let json = #"{"\#(key)":"x","n_ubatch":256}"#
        #expect(throws: RuntimeFlagsError.axisNotAccepted(key: key)) {
            try RuntimeFlags.fromCellJson(Data(json.utf8), runtimeType: .mlxIosPipette,
                                          modelType: .mlx, benchmarkType: .prefillThroughput)
        }
    }

    @Test func aKnobTheCellDoesNotCarryIsRefused() {
        #expect(throws: RuntimeFlagsError.knobNotAllowed(
            knob: "number_gpu_layers", benchmark: .prefillThroughput, runtime: .mlxIosPipette,
            model: .mlx)) {
            try RuntimeFlags.fromCellJson(Data(#"{"number_gpu_layers":10}"#.utf8),
                                          runtimeType: .mlxIosPipette, modelType: .mlx,
                                          benchmarkType: .prefillThroughput)
        }
    }

    @Test func anUndeclaredFieldPropagatesFromTheDecoder() {
        #expect(throws: UnknownFlagField.self) {
            try RuntimeFlags.fromCellJson(Data(#"{"tensor_parallel_size":2}"#.utf8),
                                          runtimeType: .mlxIosPipette, modelType: .mlx,
                                          benchmarkType: .prefillThroughput)
        }
    }

    // MARK: - speculative_enabled

    /// The crate's `speculation_resolves_from_the_flag_and_the_model`.
    @Test(arguments: [
        (ModelType.ggufVisionSpeculative, nil as Bool?, true as Bool?),
        (.ggufVisionSpeculative, false, false),
        (.mlxSpeculative, true, true),
        (.torchSpeculative, nil, true),
        (.ggufVision, nil, false),
        (.ggufVision, false, false),
        (.ggufVision, true, nil),
    ])
    func speculationResolvesFromTheFlagAndTheModel(
        model: ModelType, flag: Bool?, expected: Bool?
    ) {
        #expect((try? resolveSpeculation(model, flag)) == expected)
    }

    @Test func theFlagIsRefusedWhereItCannotApply() {
        #expect(throws: (any Error).self) {
            _ = try parse(benchmark: "vl_spec_decode", runtime: "llamacpp_ios_pipette",
                          model: "gguf_vision", ["speculative_enabled": true])
        }
        #expect(throws: (any Error).self) {
            _ = try parse(benchmark: "vl_throughput", runtime: "llamacpp_ios_pipette",
                          model: "gguf_vision", ["speculative_enabled": false])
        }
    }
}
