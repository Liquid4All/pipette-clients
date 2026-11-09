import Foundation
import Testing

@testable import Pipette

/// Turning a claimed job into a cell, and the refusals: `pipette-cli`'s `client/claim.rs`
/// tests, plus the iOS-only refusals.
struct ClaimTests {
    @Test func claimedJobDecodesClaimResponse() throws {
        let json = Data("""
        {
          "job_id": "job-abc",
          "benchmark_id": "prefill_throughput_256",
          "time_window": "PT10M",
          "expires_at": null,
          "model_name": "m",
          "spec": {
            "benchmark": "prefill_throughput_256",
            "model": {"type": "gguf_text", "org": "o", "repo_name": "r", "path": "m.gguf"},
            "runtime": {"type": "llamacpp_ios_pipette",
                        "repository_url": "github.com/ggml-org/llama.cpp",
                        "repository_version": "b9050"}
          },
          "future_field": 1
        }
        """.utf8)
        let job = try JSONDecoder().decode(ClaimedJob.self, from: json)
        #expect(job.jobId == "job-abc")
        #expect(job.benchmarkId == "prefill_throughput_256")
        #expect(job.timeWindow == "PT10M")
        let spec = try #require(job.spec?.object)
        #expect((spec["model"] as? [String: Any])?["type"] as? String == "gguf_text")
        #expect((spec["runtime"] as? [String: Any])?["type"] as? String == "llamacpp_ios_pipette")
    }

    /// The envelope names a leased job, and reporting it is the only way it stops being
    /// re-served.
    @Test func claimedJobWithoutSpecStillDecodes() throws {
        let json = Data("""
        {
          "job_id": "job-abc",
          "benchmark_id": "prefill_throughput_256",
          "time_window": "PT10M",
          "model_descriptor": "{\\"type\\":\\"gguf_text\\"}"
        }
        """.utf8)
        let job = try JSONDecoder().decode(ClaimedJob.self, from: json)
        #expect(job.jobId == "job-abc")
        #expect(job.spec == nil)
    }

    /// A plan carries the token for a gated repo inside the model spec, so the
    /// payload an operator reads must not contain it.
    @Test func specDescriptionRedactsPlanSuppliedTokens() throws {
        let json = Data("""
        {
          "job_id": "job-abc",
          "benchmark_id": "b",
          "time_window": "PT10M",
          "spec": {
            "model": {"type": "gguf_text", "org": "o", "repo_name": "r",
                      "auth_token": "hf_tokenthatmustnotescape"}
          }
        }
        """.utf8)
        let job = try JSONDecoder().decode(ClaimedJob.self, from: json)
        let described = try #require(job.spec).redactedDescription
        #expect(!described.contains("hf_tokenthatmustnotescape"))
        #expect(described.contains("<redacted>"))
    }

    /// Claims are built by decoding, not by a memberwise init: `spec` holds the
    /// bytes as they arrived, so the tests exercise the same path the wire does.
    static func claimedJob(
        benchmarkId: String = "prefill_throughput_256",
        spec: [String: Any]?
    ) throws -> ClaimedJob {
        var body: [String: Any] = [
            "job_id": "job-1",
            "benchmark_id": benchmarkId,
            "time_window": "PT10M",
        ]
        if let spec { body["spec"] = spec }
        let data = try JSONSerialization.data(withJSONObject: body)
        return try JSONDecoder().decode(ClaimedJob.self, from: data)
    }

    // MARK: - ClientRunSpec validation (claim → run construction)

    /// `benchmark` sets both halves, so they agree by default; `specBenchmark` overrides
    /// only the payload's.
    private func claim(
        runtime: [String: Any],
        model: [String: Any]? = ClaimTests.ggufModel,
        runtimeFlags: [String: Any]? = nil,
        modelFlags: [String: Any]? = nil,
        benchmarkFlags: [String: Any]? = nil,
        benchmark: String = "prefill_throughput_256",
        specBenchmark: String? = nil
    ) throws -> ClaimedJob {
        var spec: [String: Any] = [
            "benchmark": specBenchmark ?? benchmark,
            "runtime": runtime,
        ]
        if let model { spec["model"] = model }
        if let runtimeFlags { spec["runtime_flags"] = runtimeFlags }
        if let modelFlags { spec["model_flags"] = modelFlags }
        if let benchmarkFlags { spec["benchmark_flags"] = benchmarkFlags }
        return try Self.claimedJob(benchmarkId: benchmark, spec: spec)
    }

    private func refusal(
        _ job: ClaimedJob,
        _ comment: Comment
    ) throws -> UnrunnableClaimError? {
        do {
            _ = try ClientRunSpec.validated(job: job)
            Issue.record(comment)
            return nil
        } catch let err as UnrunnableClaimError {
            return err
        }
    }

    /// Every `…FlagRef` requires its axes, so a group without them never deserializes.
    private func flags(
        benchmark: String = "prefill_throughput",
        runtime: String? = "llamacpp_ios_pipette",
        model: String = "gguf_text",
        _ knobs: [String: Any] = [:]
    ) -> [String: Any] {
        var group: [String: Any] = ["benchmark_type": benchmark, "model_type": model]
        if let runtime { group["runtime_type"] = runtime }
        for (key, value) in knobs { group[key] = value }
        return group
    }

    /// Runtimes carrying every field plan-types requires of them, so a fixture
    /// this suite accepts is a body the Rust client would also have accepted.
    /// `flavor` is required on both; `mlx_ios_pipette` also pins its Swift stack.
    private static let llamaRuntime: [String: Any] = [
        "type": "llamacpp_ios_pipette", "repository_url": "github.com/ggml-org/llama.cpp",
        "repository_version": "b1", "flavor": "ios-arm64",
    ]
    private static let mlxRuntime: [String: Any] = [
        "type": "mlx_ios_pipette", "flavor": "ios-arm64",
        "packages": [
            "mlx_swift": ["repository_url": "github.com/ml-explore/mlx-swift",
                          "repository_version": "0.31.6"],
            "mlx_swift_lm": ["repository_url": "github.com/ml-explore/mlx-swift-examples",
                             "repository_version": "2.29.4"],
            "swift_transformers": ["repository_url": "github.com/huggingface/swift-transformers",
                                   "repository_version": "0.1.24"],
        ],
    ]

    /// `source` is the required tag selecting the arm, as in plan-types.
    private static let ggufModel: [String: Any] = [
        "type": "gguf_text", "source": "huggingface",
        "org": "o", "repo_name": "r", "path": "m.gguf",
    ]
    private static let mlxModel: [String: Any] = [
        "type": "mlx", "source": "huggingface", "org": "o", "repo_name": "r",
    ]
    private static let afmModel: [String: Any] = ["type": "apple_foundation_text"]

    @Test func planConfigParsesLlamacppIosRuntimeAndFlags() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: [
                    "type": "llamacpp_ios_pipette",
                    "repository_url": "github.com/ggml-org/llama.cpp",
                    "repository_version": "b9050",
                    "flavor": "ios-arm64",
                ],
                model: Self.ggufModel,
                runtimeFlags: flags([
                    "number_gpu_layers": 33, "ctx_size": 2048, "n_ubatch": 256, "threads": 4,
                ])
            )
        )
        guard case let .llamacppIosPipette(source, flavor, _) = cfg.runtime else {
            Issue.record("expected llamacppIosPipette"); return
        }
        #expect(source.repositoryVersion.value == "b9050")
        #expect(flavor == .iosArm64)
        #expect(cfg.runtimeFlags?.numberGpuLayers == 33)
        #expect(cfg.runtimeFlags?.ctxSize == 2048)
        #expect(cfg.runtimeFlags?.nUbatch == 256)
        #expect(cfg.runtimeFlags?.threads == 4)
    }

    /// A claim naming no thread count leaves it unset, which is what makes the engine
    /// derive one from the device rather than run on a number the claim never asked for.
    @Test func planConfigLeavesTheThreadCountUnsetWhenTheClaimOmitsIt() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: [
                    "type": "llamacpp_ios_pipette",
                    "repository_url": "github.com/ggml-org/llama.cpp",
                    "repository_version": "b9050",
                    "flavor": "ios-arm64",
                ],
                model: Self.ggufModel,
                runtimeFlags: flags(["ctx_size": 2048])
            )
        )
        #expect(cfg.runtimeFlags?.threads == nil)
    }

    /// An `mlx_ios_pipette` cell declares `n_ubatch` alone. The llama knobs are absent
    /// rather than defaulted: a value here would claim the cell asked for one.
    @Test func theEntryCarriesOnlyTheVariantsKnobs() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                runtimeFlags: flags(runtime: "mlx_ios_pipette", model: "mlx", ["n_ubatch": 64])
            )
        )
        #expect(cfg.runtimeFlags?.nUbatch == 64)
        #expect(cfg.runtimeFlags?.numberGpuLayers == nil)
        #expect(cfg.runtimeFlags?.ctxSize == nil)
    }

    /// `mlx_ios_pipette × mlx` types `n_ubatch` alone. A knob the variant does
    /// not carry is `KnobNotAllowed` in plan-types, so it must be refused here
    /// rather than quietly applied to a run that cannot honor it.
    @Test func planConfigRetiresAKnobTheCellDoesNotAccept() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                runtimeFlags: flags(
                    runtime: "mlx_ios_pipette", model: "mlx",
                    ["n_ubatch": 64, "number_gpu_layers": 99]
                )
            ),
            "expected a knob outside the cell's variant to be refused"
        )
        #expect(err.map(classifyRunError) == false)
        #expect(err?.localizedDescription.contains("number_gpu_layers") == true)
    }

    /// `apple_foundation` has no `RuntimeFlags` variant on any benchmark, so a
    /// group naming that cell resolves to nothing at all.
    @Test func planConfigRetiresRuntimeFlagsOnAnAppleFoundationCell() throws {
        let err = try refusal(
            claim(
                runtime: ["type": "apple_foundation"],
                model: Self.afmModel,
                runtimeFlags: flags(runtime: "apple_foundation", model: "apple_foundation_text")
            ),
            "expected runtime_flags on an AFM cell to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// `deny_unknown_fields`: a misspelled setting fails the parse instead of
    /// being dropped, which would run the cell at a default nobody asked for.
    @Test func planConfigRetiresAnUnknownFlagField() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(["n_gpu_layers": 33])
            ),
            "expected an unknown field to be refused"
        )
        #expect(err.map(classifyRunError) == false)
        #expect(err?.localizedDescription.contains("n_gpu_layers") == true)
    }

    /// The axes are required fields of the `…FlagRef`, not optional hints.
    @Test func planConfigRetiresAFlagGroupMissingAnAxis() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: ["number_gpu_layers": 33]
            ),
            "expected a group with no axes to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// The knobs are `Option<u32>`. A quoted digit string is not a JSON number
    /// and does not deserialize on the Rust side either.
    @Test func planConfigRetiresANonNumericKnob() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(["ctx_size": "2048"])
            ),
            "expected a string knob to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// `u32`, so a negative value has no representation.
    @Test func planConfigRetiresANegativeKnob() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(["ctx_size": -1])
            ),
            "expected a negative knob to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    @Test func planConfigParsesModelFlagsEnableThinking() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                benchmark: "eval_ifbench"
            )
        )
        guard case .mlxIosPipette = cfg.runtime else {
            Issue.record("expected mlxIosPipette"); return
        }
    }

    /// `ModelFlags` exists only for `eval`: a timing cell has no generation to shape.
    @Test func planConfigRetiresModelFlagsOnATimingCell() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                modelFlags: flags(benchmark: "prefill_throughput", runtime: nil, model: "mlx",
                                  ["enable_thinking": true])
            ),
            "expected model_flags on a timing cell to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// `enable_thinking` is `Option<bool>`; `1` is a number, not a boolean.
    @Test func planConfigRetiresANonBooleanEnableThinking() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                modelFlags: flags(benchmark: "eval", runtime: nil, model: "mlx",
                                  ["enable_thinking": 1]),
                benchmark: "eval_ifbench"
            ),
            "expected a numeric enable_thinking to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// A claim carrying no `spec` cannot run and no other client will do better
    /// with it: matching already keeps revisions this build cannot parse away
    /// from it, so one that arrives anyway is mis-authored.
    @Test func planConfigRetiresASpeclessClaim() throws {
        let err = try refusal(
            Self.claimedJob(spec: nil),
            "expected a claim with no spec to be refused"
        )
        guard case .noSpec = try #require(err) else {
            Issue.record("expected noSpec, got \(String(describing: err))"); return
        }
        #expect(err.map(classifyRunError) == false)
    }

    /// A `spec` this revision should have read is terminal too.
    @Test func planConfigRetiresASpecItCannotRead() throws {
        let err = try refusal(
            Self.claimedJob(spec: ["benchmark": "prefill_throughput_256"]),
            "expected a spec with no runtime to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// A `spec` present but not an object must still leave the envelope
    /// decodable: the envelope names the leased job, and reporting it is the
    /// only way it stops being re-served.
    @Test func planConfigRefusesANonObjectSpecWithoutLosingTheClaim() throws {
        let body = Data("""
        {
          "job_id": "job-1",
          "benchmark_id": "prefill_throughput_256",
          "time_window": "PT10M",
          "spec": "not-a-cell"
        }
        """.utf8)
        let job = try JSONDecoder().decode(ClaimedJob.self, from: body)
        #expect(job.jobId == "job-1")
        let err = try refusal(job, "expected a non-object spec to be refused")
        // Unreadable, not absent: the refusal has to say which.
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
        #expect(err.map(classifyRunError) == false)
    }

    /// Guessing which half was meant would run the wrong work or file the result against
    /// the wrong id (Rust `UnrunnableClaimError::BenchmarkMismatch`).
    @Test func planConfigRetiresAClaimWhoseHalvesNameDifferentBenchmarks() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                specBenchmark: "decode_throughput_512"
            ),
            "expected a claim naming two benchmarks to be refused"
        )
        #expect(err.map(classifyRunError) == false)
        let described = try #require(err?.localizedDescription)
        #expect(described.contains("prefill_throughput_256"))
        #expect(described.contains("decode_throughput_512"))
    }

    /// Applying settings meant for another runtime would run the wrong cell. Rust refuses
    /// these on arrival too.
    @Test func planConfigRetiresRuntimeFlagsNamingAnotherRuntime() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(runtime: "mlx_macos_pipette", ["number_gpu_layers": 99])
            ),
            "expected flags naming another runtime to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    @Test func planConfigRetiresModelFlagsNamingAnotherModel() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                modelFlags: flags(benchmark: "eval", runtime: nil, model: "gguf_text",
                                  ["enable_thinking": true]),
                benchmark: "eval_ifbench"
            ),
            "expected flags naming another model to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    /// The benchmark axis comes from the id itself: `prefill_throughput_256` is
    /// a `prefill_throughput` cell, so an `eval` group does not belong to it.
    @Test func planConfigRetiresFlagsNamingAnotherBenchmarkType() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                runtimeFlags: flags(benchmark: "eval", runtime: "mlx_ios_pipette",
                                    model: "mlx", ["n_ubatch": 64])
            ),
            "expected flags naming another benchmark type to be refused"
        )
        #expect(err.map(classifyRunError) == false)
    }

    @Test func planConfigAcceptsFlagsNamingThisCell() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(benchmark: "eval", ["number_gpu_layers": 40]),
                benchmark: "eval_ifbench"
            )
        )
        #expect(cfg.runtimeFlags?.numberGpuLayers == 40)
    }

    /// `enable_thinking` shapes generation through the chat template's Jinja context,
    /// which this build does not drive. The claim is refused rather than run as something
    /// other than what it asked for. Terminally, although the Rust client applies the
    /// knob: `requires` is what keeps such a cell away from a build that cannot.
    @Test(arguments: [true, false])
    func planConfigRefusesThinkingItCannotApply(_ thinking: Bool) throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                modelFlags: flags(benchmark: "eval", runtime: nil,
                                  ["enable_thinking": thinking]),
                benchmark: "eval_ifbench"
            ),
            "expected a knob this build cannot apply to be refused"
        )
        guard case let .flagNotHonouredHere(_, knob) = try #require(err) else {
            Issue.record("expected flagNotHonouredHere, got \(String(describing: err))")
            return
        }
        #expect(knob == "enable_thinking")
        #expect(err.map(classifyRunError) == false, "every claim refusal is terminal, as upstream")
    }

    /// `reasoning_effort` reaches the model through the same Jinja context as
    /// `enable_thinking`, so the cell that carries it refuses it the same way.
    @Test func planConfigRefusesAReasoningEffortItCannotApply() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                modelFlags: flags(benchmark: "eval", runtime: nil,
                                  ["reasoning_effort": "low"]),
                benchmark: "eval_ifbench"
            ),
            "expected a knob this build cannot apply to be refused"
        )
        guard case let .flagNotHonouredHere(_, knob) = try #require(err) else {
            Issue.record("expected flagNotHonouredHere, got \(String(describing: err))")
            return
        }
        #expect(knob == "reasoning_effort")
        #expect(err.map(classifyRunError) == false, "every claim refusal is terminal, as upstream")
    }

    /// The crate gives the MLX cell no `reasoning_effort`, so one there is a knob the cell
    /// does not accept, named as such.
    @Test func planConfigRefusesAReasoningEffortOnACellWithoutOne() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                modelFlags: flags(benchmark: "eval", runtime: nil, model: "mlx",
                                  ["reasoning_effort": "low"]),
                benchmark: "eval_ifbench"
            ),
            "expected a reasoning_effort on an MLX cell to be refused"
        )
        guard case let .flagNotAcceptedByCell(_, knob, _) = try #require(err) else {
            Issue.record("expected flagNotAcceptedByCell, got \(String(describing: err))")
            return
        }
        #expect(knob == "reasoning_effort")
        #expect(err.map(classifyRunError) == false)
    }

    @Test func planConfigRejectsDesktopRuntime() throws {
        do {
            _ = try ClientRunSpec.validated(job: claim(runtime: [
                "type": "llamacpp_cli_stock_tools", "repository_url": "github.com/ggml-org/llama.cpp",
                "repository_version": "b1", "flavor": "macos-arm64",
            ]))
            Issue.record("expected unsupported runtime")
        } catch let err as UnrunnableClaimError {
            #expect(classifyRunError(err) == false)
        }
    }

    @Test func planConfigRejectsWrongFlavor() throws {
        do {
            _ = try ClientRunSpec.validated(job: claim(runtime: [
                "type": "llamacpp_ios_pipette", "repository_url": "github.com/ggml-org/llama.cpp",
                "repository_version": "b1", "flavor": "android-arm64-v8",
            ]))
            Issue.record("expected invalid flavor")
        } catch let err as UnrunnableClaimError {
            #expect(!classifyRunError(err))
        }
    }

    /// `flavor` is required on both iOS runtimes: it must not fall back to this device's own.
    @Test func planConfigRetiresARuntimeWithNoFlavor() throws {
        let err = try refusal(
            claim(runtime: ["type": "llamacpp_ios_pipette",
                            "repository_url": "github.com/ggml-org/llama.cpp",
                            "repository_version": "b1"]),
            "expected a runtime with no flavor to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
    }

    /// `repository_url` has no default, so a body omitting it does not deserialize.
    @Test func planConfigRetiresARuntimeWithNoRepositoryUrl() throws {
        let err = try refusal(
            claim(runtime: ["type": "llamacpp_ios_pipette", "repository_version": "b1",
                            "flavor": "ios-arm64"]),
            "expected a runtime with no repository_url to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
    }

    /// `mlx_ios_pipette` pins its Swift-package stack. This build never reads
    /// it, but a body omitting it is not a valid cell.
    @Test func planConfigRetiresAnMlxRuntimeWithNoPackages() throws {
        let err = try refusal(
            claim(
                runtime: ["type": "mlx_ios_pipette", "flavor": "ios-arm64"],
                model: Self.mlxModel
            ),
            "expected an MLX runtime with no packages to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
    }

    /// `repository_version` is `NonEmptyString`, and `version` is its alias.
    @Test func planConfigAcceptsTheVersionAliasAndRejectsAnEmptyPin() throws {
        let aliased = try ClientRunSpec.validated(
            job: claim(runtime: [
                "type": "llamacpp_ios_pipette", "repository_url": "github.com/ggml-org/llama.cpp",
                "version": "b77", "flavor": "ios-arm64",
            ])
        )
        guard case let .llamacppIosPipette(source, _, _) = aliased.runtime else {
            Issue.record("expected llamacppIosPipette"); return
        }
        #expect(source.repositoryVersion.value == "b77")

        let err = try refusal(
            claim(runtime: [
                "type": "llamacpp_ios_pipette", "repository_url": "github.com/ggml-org/llama.cpp",
                "repository_version": "", "flavor": "ios-arm64",
            ]),
            "expected an empty repository_version to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
    }

    /// Flag groups resolve against the benchmark *type*, so an id this build
    /// cannot classify is refused by name rather than as an axis mismatch. A
    /// flagless cell with the same id is left alone.
    @Test func planConfigRetiresFlagsOnAnUnclassifiableBenchmark() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                runtimeFlags: flags(["number_gpu_layers": 8]),
                benchmark: "smoke"
            ),
            "expected flags on an unclassifiable benchmark to be refused"
        )
        guard case .unclassifiableBenchmark = try #require(err) else {
            Issue.record("expected unclassifiableBenchmark, got \(String(describing: err))")
            return
        }

        // Without flags there is no axis to resolve, so the cell parses with no entry.
        let cfg = try ClientRunSpec.validated(
            job: claim(runtime: Self.llamaRuntime, benchmark: "smoke")
        )
        #expect(cfg.runtimeFlags == nil)
    }

    /// The crate keeps `Model` identity-only, so a generation flag has nowhere to land on
    /// it, wire form included.
    @Test func thinkingStaysOnTheCellAndNeverOnTheModel() throws {
        let text = Model.ggufText(.init(source: .huggingFace(
            repo: try HfRepo.parseOrgRepo("org/repo"),
            path: RepoSubpath(validated: "m-Q4_0.gguf"), sha256: nil)))
        let json = String(decoding: try JSONEncoder().encode(text), as: UTF8.self)
        #expect(!json.contains("model_flags"))
    }

    // MARK: - Cell coherence and failure classification

    /// A runtime that cannot run this kind of model is mis-authored, and dies at the
    /// claim (plan-types' `is_compatible`): dispatch would otherwise run it on the model's
    /// engine.
    @Test func planConfigRetiresAModelItsRuntimeCannotRun() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.ggufModel
            ),
            "expected a gguf model on an MLX runtime to be refused"
        )
        #expect(err.map(classifyRunError) == false)
        #expect(err?.localizedDescription.contains("not compatible") == true)
    }

    /// `source` is the tag selecting the arm; a model without one does not
    /// deserialize on either side.
    @Test func planConfigRetiresAModelWithNoSource() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: ["type": "gguf_text", "org": "o", "repo_name": "r", "path": "m.gguf"]
            ),
            "expected a model with no source to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
    }

    /// A `local` source is well-formed but names nothing this device can fetch. Refused
    /// terminally, as every claim refusal is: keeping a cell away from a client that
    /// cannot run it is a capability in `requires`, not a disposition on the answer.
    @Test func planConfigRefusesAModelSourceItCannotFetch() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: ["type": "gguf_text", "source": "local", "dir": "/tmp/m"]
            ),
            "expected a local model source to be refused"
        )
        guard case .modelKindNotRunnable = try #require(err) else {
            Issue.record("expected modelKindNotRunnable, got \(String(describing: err))"); return
        }
        #expect(err.map(classifyRunError) == false, "every claim refusal is terminal, as upstream")
    }

    @Test func planConfigAcceptsAVaultArchive() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: Self.mlxRuntime,
                model: [
                    "type": "mlx", "source": "vault_archive",
                    "url": "/vault/resources/vault_1/content", "principal_id": "org_1",
                ]
            )
        )
        guard case let .mlx(m) = cfg.model, case .vaultArchive = m.source else {
            Issue.record("expected an mlx vault archive, got \(cfg.model)"); return
        }
    }

    /// A vault model's `custom_label` rides the claim into the cell and out again in the
    /// descriptor it submits. A blank one fails the parse, as the crate's does.
    @Test func planConfigKeepsAVaultCustomLabel() throws {
        var model: [String: Any] = [
            "type": "mlx", "source": "vault_archive",
            "url": "/vault/resources/vault_1/content", "principal_id": "org_1",
            "custom_label": " acme LFM3 q4 ",
        ]
        let cfg = try ClientRunSpec.validated(job: claim(runtime: Self.mlxRuntime, model: model))
        let submitted = try JSONSerialization.jsonObject(
            with: Data(try SubmissionRef.model(cfg.model).utf8)) as? [String: Any]
        #expect(submitted?["custom_label"] as? String == "acme LFM3 q4")

        model["custom_label"] = "  "
        let err = try refusal(
            claim(runtime: Self.mlxRuntime, model: model),
            "expected a blank custom_label to be refused"
        )
        guard case .unparseable = try #require(err) else {
            Issue.record("expected unparseable, got \(String(describing: err))"); return
        }
        #expect(err.map(classifyRunError) == false)
    }

    @Test func planConfigCarriesTheReadinessGate() throws {
        let cfg = try ClientRunSpec.validated(
            job: claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                benchmarkFlags: [
                    "benchmark_type": "prefill_throughput",
                    "runtime_type": "llamacpp_ios_pipette",
                    "model_type": "gguf_text",
                    "readiness": ["max_wait_secs": 1800, "skip_thermal": true],
                ]
            )
        )
        #expect(cfg.benchmarkFlags?.readiness?.maxWaitSecs == 1800)
        #expect(cfg.benchmarkFlags?.readiness?.skipThermal == true)
    }

    /// The server-only knobs are not declared here, so a body carrying one is refused as an
    /// unknown field rather than accepted and dropped: an in-process engine has no HTTP
    /// client to bound.
    @Test func planConfigRetiresAServerBenchmarkKnob() throws {
        let job = try Self.claimedJob(spec: [
            "benchmark": "prefill_throughput_256",
            "runtime": Self.llamaRuntime,
            "model": Self.ggufModel,
            "benchmark_flags": [
                "benchmark_type": "prefill_throughput",
                "runtime_type": "llamacpp_ios_pipette",
                "model_type": "gguf_text",
                "http_timeout_seconds": 600,
            ],
        ])
        let err = try refusal(job, "expected http_timeout_seconds to be refused")
        #expect(err.map(classifyRunError) == false)
    }

    /// `max_memory_usage` does not gate, so its variant has no readiness field.
    @Test func planConfigRetiresBenchmarkFlagsOnAnUngatedCell() throws {
        let job = try Self.claimedJob(benchmarkId: "max_memory_usage_512", spec: [
            "benchmark": "max_memory_usage_512",
            "runtime": Self.llamaRuntime,
            "model": Self.ggufModel,
            "benchmark_flags": [
                "benchmark_type": "max_memory_usage",
                "runtime_type": "llamacpp_ios_pipette",
                "model_type": "gguf_text",
                "readiness": ["skip_thermal": true],
            ],
        ])
        let err = try refusal(job, "expected an ungated cell to refuse a readiness block")
        guard case .flagNotAcceptedByCell(_, "readiness", _)? = err else {
            Issue.record("expected flagNotAcceptedByCell, got \(String(describing: err))"); return
        }
        #expect(err.map(classifyRunError) == false)
    }

    /// The attempt limit, and the retired `bench_timeout_seconds` a queued spec may still
    /// carry, which is absorbed unread.
    @Test func planConfigCarriesTheAttemptLimit() throws {
        let cfg = try ClientRunSpec.validated(job: Self.claimedJob(
            benchmarkId: "max_memory_usage_512", spec: [
                "benchmark": "max_memory_usage_512",
                "runtime": Self.llamaRuntime,
                "model": Self.ggufModel,
                "benchmark_flags": [
                    "benchmark_type": "max_memory_usage",
                    "runtime_type": "llamacpp_ios_pipette",
                    "model_type": "gguf_text",
                    "attempt_timeout_seconds": 2400,
                    "bench_timeout_seconds": 2400,
                ],
            ]))
        #expect(cfg.benchmarkFlags?.attemptTimeoutSeconds == 2400)
    }

    // MARK: - Which claims a device will take

    /// A `torch` model cannot be built as a `Model`, so it can never match a local one.
    @Test func aTorchClaimIsRefusedAsARunnableKind() throws {
        let job = try Self.claimedJob(benchmarkId: "b", spec: [
            "benchmark": "b",
            "model": ["type": "torch", "source": "huggingface", "org": "o", "repo_name": "r"],
            "runtime": ["type": "apple_foundation"],
        ])
        do {
            _ = try runSpecFromClaim(job)
            Issue.record("a torch model should not decode")
        } catch let error as UnrunnableClaimError {
            guard case .modelKindNotRunnable = error else {
                Issue.record("expected modelKindNotRunnable, got \(error)")
                return
            }
            // Terminal, as every claim refusal is upstream.
            #expect(!classifyRunError(error))
        }
    }

    /// The two builds produce different numbers for one cell, because an ungated run may
    /// start hot. Terminal: no re-serving makes this device able to run it.
    @Test func aClaimRequiringPrivateThermalIsRefusedByAStockBuild() throws {
        let job = try Self.claimedJob(benchmarkId: "prefill_throughput_512", spec: [
            "benchmark": "prefill_throughput_512",
            "model": ["type": "gguf_text", "source": "huggingface",
                      "org": "LiquidAI", "repo_name": "r", "path": "m.gguf"],
            "runtime": ["type": "llamacpp_ios_pipette",
                        "repository_url": "github.com/ggml-org/llama.cpp",
                        "repository_version": "b10216", "flavor": "ios-arm64",
                        "private_thermal": true],
        ])

        do {
            // The capability refusals live in `validated`; `runSpecFromClaim` only types
            // the payload.
            _ = try ClientRunSpec.validated( try runSpecFromClaim(job))
            // A build compiled *with* the read accepts it, so the refusal is asserted only on
            // one that lacks it.
            #expect(PrivateApis.thisBuild.thermal, "a stock build should have refused")
        } catch let error as UnrunnableClaimError {
            guard case .buildLacksPrivateThermal = error else {
                Issue.record("expected buildLacksPrivateThermal, got \(error)")
                return
            }
            #expect(!classifyRunError(error))
        }
    }

    /// As `aClaimRequiringPrivateThermalIsRefusedByAStockBuild`, for the memory calls.
    @Test func aClaimRequiringPrivateMemoryIsRefusedByAStockBuild() throws {
        let job = try Self.claimedJob(benchmarkId: "prefill_throughput_512", spec: [
            "benchmark": "prefill_throughput_512",
            "model": ["type": "gguf_text", "source": "huggingface",
                      "org": "LiquidAI", "repo_name": "r", "path": "m.gguf"],
            "runtime": ["type": "llamacpp_ios_pipette",
                        "repository_url": "github.com/ggml-org/llama.cpp",
                        "repository_version": "b10216", "flavor": "ios-arm64",
                        "private_memory": true],
        ])

        do {
            _ = try ClientRunSpec.validated(try runSpecFromClaim(job))
            #expect(PrivateApis.thisBuild.memory, "a stock build should have refused")
        } catch let error as UnrunnableClaimError {
            guard case .buildLacksPrivateMemory = error else {
                Issue.record("expected buildLacksPrivateMemory, got \(error)")
                return
            }
            #expect(!classifyRunError(error))
        }
    }

    /// The field defaults to the stock meaning, so a cell that omits it runs on either build.
    @Test func aClaimSayingNothingAboutThermalRuns() throws {
        let job = try Self.claimedJob(benchmarkId: "prefill_throughput_512", spec: [
            "benchmark": "prefill_throughput_512",
            "model": ["type": "gguf_text", "source": "huggingface",
                      "org": "LiquidAI", "repo_name": "r", "path": "m.gguf"],
            "runtime": ["type": "llamacpp_ios_pipette",
                        "repository_url": "github.com/ggml-org/llama.cpp",
                        "repository_version": "b10216", "flavor": "ios-arm64"],
        ])

        let spec = try runSpecFromClaim(job)
        _ = try ClientRunSpec.validated( spec)

        #expect(spec.runtime.privateApis == .stock)
    }

    // MARK: - The refusal text

    /// A `failure_reason` reads the same whichever client refused the claim: the crate's
    /// four `#[error]` strings, prefix included.
    @Test func aRefusalReadsAsTheCratesDoes() {
        #expect(UnrunnableClaimError.noSpec.localizedDescription
            == "unrunnable claim: carries no `spec`")
        #expect(UnrunnableClaimError.unparseable.localizedDescription
            == "unrunnable claim: spec did not parse")
        #expect(UnrunnableClaimError.benchmarkMismatch(envelope: "a", spec: "b").localizedDescription
            == "unrunnable claim: benchmark_id `a` disagrees with spec.benchmark `b`")
        #expect(UnrunnableClaimError.incompatible(model: "m", runtime: "r").localizedDescription
            == "unrunnable claim: model `m` is not compatible with runtime `r`")
    }

    /// The refusals the crate does not need carry the same prefix, so a reader can tell
    /// a claim refusal from a run failure without knowing every case.
    @Test func everyIosOnlyRefusalCarriesThePrefixToo() {
        let refusals: [UnrunnableClaimError] = [
            .unclassifiableBenchmark("smoke"), .unsupportedRuntime("docker_vllm"),
            .flagAxisMissing(group: "runtime_flags", axis: "model_type"),
            .modelKindNotRunnable("torch"), .buildLacksPrivateThermal, .buildLacksPrivateMemory,
            .flagNotHonouredHere(group: "model_flags", knob: "enable_thinking"),
        ]
        for refusal in refusals {
            #expect(refusal.localizedDescription.hasPrefix("unrunnable claim: "))
        }
    }

    /// The model is named as the crate's `Display` renders it, so an operator can find
    /// the cell from the failure reason alone.
    @Test func anIncompatibleCellNamesTheModel() throws {
        let err = try refusal(
            claim(runtime: Self.mlxRuntime, model: Self.ggufModel),
            "expected a gguf model on an MLX runtime to be refused")
        #expect(err?.localizedDescription.contains("`o/r:m.gguf`") == true)
        #expect(err?.localizedDescription.contains("`mlx_ios_pipette`") == true)
    }

    /// A plan carries the access token for a gated repo inside the model spec, and the
    /// rejection reason is submitted to the server as `failure_reason` and stored there.
    /// Quoting the payload into that message would publish the token.
    @Test func aRejectionNeverCarriesAPlanSuppliedToken() throws {
        let token = "hf_tokenthatmustnotescape"
        var gated = Self.ggufModel
        gated["auth_token"] = token
        // A cell that cannot resolve, so the payload is what fails.
        let job = try claim(
            runtime: Self.llamaRuntime,
            model: gated,
            runtimeFlags: flags(benchmark: "eval", runtime: "mlx_macos_pipette",
                                ["number_gpu_layers": 99]),
            benchmark: "eval_ifbench")

        let err = try refusal(job, "expected an unresolvable cell to be refused")
        let reported = try #require(err?.localizedDescription)
        #expect(!reported.contains(token), "token leaked into: \(reported)")
        // The same holds for the payload an operator sees in the logs.
        let logged = try #require(job.spec).redactedDescription
        #expect(!logged.contains(token), "token leaked into: \(logged)")
    }

    /// An unreadable payload must survive decoding: the job it names still has to be failed.
    @Test(arguments: [false, true])
    func anUnreadableSpecIsAnUnrunnableClaim(_ carriesSpec: Bool) throws {
        let job = try Self.claimedJob(
            spec: carriesSpec
                ? ["benchmark": "prefill_throughput_256", "model": "not-a-model"] : nil)
        do {
            _ = try runSpecFromClaim(job)
            Issue.record("an unreadable spec cannot produce a cell")
        } catch {
            #expect(error is UnrunnableClaimError)
            #expect(!classifyRunError(error), "must be terminal, not retried")
        }
    }
}
