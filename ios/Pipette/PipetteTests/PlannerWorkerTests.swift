import Foundation
import Testing

@testable import Pipette

struct PlannerWorkerTests {
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
            "runtime": {"type": "llamacpp_ios_pipette", "repository_version": "b9050"}
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

    /// A body carrying no `spec` must still decode —
    /// the envelope names a leased job, and reporting it is the only way it
    /// stops being re-served.
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

    @Test func clientProfileDefaultsMissingOptionalFields() throws {
        let json = Data("""
        {
          "client_id": "ev1_x",
          "organization": "o",
          "client_details": "d",
          "contact_email": "a@b.com",
          "status": "approved"
        }
        """.utf8)
        let profile = try JSONDecoder().decode(ClientProfile.self, from: json)
        #expect(profile.tags.isEmpty)
        #expect(profile.reindexPending == false)
        #expect(profile.capabilities.isEmpty)
        #expect(profile.status == "approved")
    }

    @Test func iso8601DurationParsesCommonWindows() {
        #expect(Iso8601Duration.seconds(from: "PT10M") == 600)
        #expect(Iso8601Duration.seconds(from: "PT5M") == 300)
        #expect(Iso8601Duration.seconds(from: "PT1H2M3S") == 3723)
        #expect(Iso8601Duration.seconds(from: "pt30s") == 30)
        #expect(Iso8601Duration.seconds(from: "10M") == nil)
        #expect(Iso8601Duration.seconds(from: "PT10") == nil)
    }

    @Test func heartbeatIntervalDefaultsToHalfWindow() {
        #expect(Iso8601Duration.heartbeatInterval(timeWindow: "PT10M") == 300)
        #expect(Iso8601Duration.heartbeatInterval(timeWindow: "bogus") == 300) // 600/2 default
        #expect(Iso8601Duration.heartbeatInterval(timeWindow: "PT10M", overrideSeconds: 60) == 60)
        #expect(Iso8601Duration.heartbeatInterval(timeWindow: "PT10M", overrideSeconds: 0) == 1)
    }

    @Test func failureSubmissionIdentifiesTheJob() throws {
        let job = try Self.claimedJob(
            benchmarkId: "eval_x",
            spec: ["runtime": ["type": "mlx_ios_pipette"]]
        )
        let failure = FailureSubmission.fromClaim(
            job, reason: "[ts] oom", retriable: true, clientVersion: "9.9.9 (1234)")
        let data = try JSONEncoder().encode(failure)
        let obj = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(obj["message_type"] as? String == "failure")
        #expect(obj["job_id"] as? String == "job-1")
        #expect(obj["benchmark_id"] as? String == "eval_x")
        #expect(obj["retriable"] as? Bool == true)
        #expect(obj["failure_reason"] as? String == "[ts] oom")
        // The server cannot recover this from the job body — a failure that
        // omitted it would leave "which build reported this" unanswerable.
        #expect(obj["client_version"] as? String == "9.9.9 (1234)")
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

    @Test func deviceProfileUpdateOmitsNilFields() throws {
        let update = DeviceProfileUpdate(
            clientDetails: "iPhone",
            deviceName: "iPhone 16",
            deviceFormFactor: .phone,
            deviceOsName: "iOS",
            deviceOsVersion: "18.0",
            deviceChipModel: "A18",
            deviceRamBytes: 8_000_000_000,
            capabilities: ["runtime:llama_cpp"]
        )
        let data = try JSONEncoder().encode(update)
        let obj = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(obj["device_name"] as? String == "iPhone 16")
        #expect(obj["capabilities"] as? [String] == ["runtime:llama_cpp"])
        #expect(obj["device_gpu_model"] == nil)
    }

    // MARK: - ClientRunSpec validation (claim → run construction)

    /// A claim whose `spec` names `runtime`, optionally with the model and the
    /// typed flag groups a cell would carry.
    ///
    /// `benchmark` sets both halves, so they agree by default; `specBenchmark`
    /// overrides only the payload's, for the case that sets out to break that.
    private func claim(
        runtime: [String: Any],
        model: [String: Any]? = PlannerWorkerTests.ggufModel,
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

    /// The `ParseError` a refused claim carries, or a recorded failure.
    private func refusal(
        _ job: ClaimedJob,
        _ comment: Comment
    ) throws -> UnrunnableClaim? {
        do {
            _ = try ClientRunSpec.validated(job: job)
            Issue.record(comment)
            return nil
        } catch let err as UnrunnableClaim {
            return err
        }
    }

    /// A flag group with its three axes filled in — every `…FlagRef` requires
    /// them, so a group without them never deserializes.
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
        "type": "llamacpp_ios_pipette", "repository_version": "b1",
        "flavor": "ios-arm64",
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

    /// The entry carries what the cell's variant declares and nothing else — an
    /// `mlx_ios_pipette` cell's is `n_ubatch` alone. The llama knobs are absent rather than
    /// defaulted: a value here would claim the cell asked for one.
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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

    /// `ModelFlags` exists only for `eval` — a timing cell has no generation to
    /// shape, so a group on one names a variant that does not exist.
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
    }

    /// A claim carrying no `spec` cannot run and no other client will do better
    /// with it: matching already keeps revisions this build cannot parse away
    /// from it, so one that arrives anyway is mis-authored.
    @Test func planConfigRetiresASpeclessClaim() throws {
        let err = try refusal(
            Self.claimedJob(spec: nil),
            "expected a claim with no spec to be refused"
        )
        guard case .missingSpec = try #require(err) else {
            Issue.record("expected missingSpec, got \(String(describing: err))"); return
        }
        #expect(err?.retriable == false)
    }

    /// A `spec` this revision should have read is terminal too.
    @Test func planConfigRetiresASpecItCannotRead() throws {
        let err = try refusal(
            Self.claimedJob(spec: ["benchmark": "prefill_throughput_256"]),
            "expected a spec with no runtime to be refused"
        )
        #expect(err?.retriable == false)
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
        // Unreadable, not absent — the refusal has to say which.
        guard case .unreadableSpec = try #require(err) else {
            Issue.record("expected unreadableSpec, got \(String(describing: err))"); return
        }
        #expect(err?.retriable == false)
    }

    /// `benchmark_id` is duplicated on the wire and the two halves must agree —
    /// guessing which was meant would run the wrong work or file the result
    /// against the wrong id (Rust `UnrunnableClaim::BenchmarkMismatch`).
    @Test func planConfigRetiresAClaimWhoseHalvesNameDifferentBenchmarks() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.mlxModel,
                specBenchmark: "decode_throughput_512"
            ),
            "expected a claim naming two benchmarks to be refused"
        )
        #expect(err?.retriable == false)
        let described = try #require(err?.localizedDescription)
        #expect(described.contains("prefill_throughput_256"))
        #expect(described.contains("decode_throughput_512"))
    }

    /// Flags naming a cell other than the one being run are mis-authored, and
    /// silently applying settings meant for another runtime is the outcome the
    /// check rules out. Rust refuses these on arrival; so must this.
    @Test func planConfigRetiresRuntimeFlagsNamingAnotherRuntime() throws {
        let err = try refusal(
            claim(
                runtime: Self.llamaRuntime,
                model: Self.ggufModel,
                runtimeFlags: flags(runtime: "mlx_macos_pipette", ["number_gpu_layers": 99])
            ),
            "expected flags naming another runtime to be refused"
        )
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
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
        #expect(err?.retriable == false)
    }

    /// A correctly authored cell: both groups name it on every axis, and each
    /// sets only knobs its own variant carries.
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
    /// other than what it asked for — and refused *retriably*, because the Rust client
    /// applies the knob and would run the cell correctly.
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
        #expect(err?.retriable == false, "every claim refusal is terminal, as upstream")
    }

    @Test func planConfigRejectsDesktopRuntime() throws {
        do {
            _ = try ClientRunSpec.validated(job: claim(runtime: [
                "type": "llamacpp_cli_stock_tools", "repository_version": "b1",
                "flavor": "macos-arm64",
            ]))
            Issue.record("expected unsupported runtime")
        } catch let err as UnrunnableClaim {
            #expect(err.retriable == false)
        }
    }

    @Test func planConfigRejectsWrongFlavor() throws {
        do {
            _ = try ClientRunSpec.validated(job: claim(runtime: [
                "type": "llamacpp_ios_pipette", "repository_version": "b1",
                "flavor": "android-arm64-v8",
            ]))
            Issue.record("expected invalid flavor")
        } catch let err as UnrunnableClaim {
            #expect(!err.retriable)
        }
    }

    /// `flavor` is required on both iOS runtimes, so a body omitting it does not
    /// deserialize — it must not fall back to this device's own flavor.
    @Test func planConfigRetiresARuntimeWithNoFlavor() throws {
        let err = try refusal(
            claim(runtime: ["type": "llamacpp_ios_pipette", "repository_version": "b1"]),
            "expected a runtime with no flavor to be refused"
        )
        guard case .unreadableSpec = try #require(err) else {
            Issue.record("expected unreadableSpec, got \(String(describing: err))"); return
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
        guard case .unreadableSpec = try #require(err) else {
            Issue.record("expected unreadableSpec, got \(String(describing: err))"); return
        }
    }

    /// `repository_version` is `NonEmptyString`, and `version` is its alias.
    @Test func planConfigAcceptsTheVersionAliasAndRejectsAnEmptyPin() throws {
        let aliased = try ClientRunSpec.validated(
            job: claim(runtime: [
                "type": "llamacpp_ios_pipette", "version": "b77", "flavor": "ios-arm64",
            ])
        )
        guard case let .llamacppIosPipette(source, _, _) = aliased.runtime else {
            Issue.record("expected llamacppIosPipette"); return
        }
        #expect(source.repositoryVersion.value == "b77")

        let err = try refusal(
            claim(runtime: [
                "type": "llamacpp_ios_pipette", "repository_version": "", "flavor": "ios-arm64",
            ]),
            "expected an empty repository_version to be refused"
        )
        guard case .unreadableSpec = try #require(err) else {
            Issue.record("expected unreadableSpec, got \(String(describing: err))"); return
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

        // Without flags there is no axis to resolve, so the cell still parses — and it
        // carries no entry, rather than one full of values it never named.
        let cfg = try ClientRunSpec.validated(
            job: claim(runtime: Self.llamaRuntime, benchmark: "smoke")
        )
        #expect(cfg.runtimeFlags == nil)
    }

    /// Generation flags belong to the cell, never to the model. The crate is explicit that
    /// `Model` stays identity-only, so there is nowhere on a `Model` for a generation flag
    /// to land — including in its wire form.
    @Test func thinkingStaysOnTheCellAndNeverOnTheModel() throws {
        let text = Model.ggufText(.init(source: .huggingFace(
            repo: try HFRepo.parse("org/repo"),
            path: RepoSubpath(validated: "m-Q4_0.gguf"), sha256: nil)))
        let json = String(decoding: try JSONEncoder().encode(text), as: UTF8.self)
        #expect(!json.contains("model_flags"))
    }

    // MARK: - Cell coherence and failure classification

    /// A runtime that cannot run this kind of model is mis-authored, and dies at
    /// the claim rather than surfacing later as "no local model matched" — the
    /// CLI's `is_compatible` gate.
    @Test func planConfigRetiresAModelItsRuntimeCannotRun() throws {
        let err = try refusal(
            claim(
                runtime: Self.mlxRuntime,
                model: Self.ggufModel
            ),
            "expected a gguf model on an MLX runtime to be refused"
        )
        #expect(err?.retriable == false)
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
        guard case .unreadableSpec = try #require(err) else {
            Issue.record("expected unreadableSpec, got \(String(describing: err))"); return
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
        #expect(err?.retriable == false, "every claim refusal is terminal, as upstream")
    }

    /// A timing cell carries the readiness gate, and it reaches the config typed.
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
        #expect(err?.retriable == false)
    }

    /// `max_memory_usage` does not gate, so it has no variant to carry one — the same
    /// refusal every cell got before iOS declared any.
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
        #expect(err?.retriable == false)
    }

    /// The disposition policy lives in one classifier, so it can be pinned here
    /// rather than inferred from eleven scattered call sites.
    @Test func failureClassificationDefaultsToRetriable() {
        // Terminal: unreadable by construction, and a benchmark no build knows.
        #expect(!PlannerWorker.retriable(UnrunnableClaim.missingSpec))
        #expect(!PlannerWorker.retriable(UnrunnableClaim.unreadableSpec))
        #expect(!PlannerWorker.retriable(PlannerWorker.WorkerResolveError.unknownBenchmark("b")))

        // Retriable: specific to this device, so another may still succeed.
        #expect(PlannerWorker.retriable(PlannerWorker.WorkerResolveError.noLocalModel("m")))
        #expect(PlannerWorker.retriable(PlannerWorker.WorkerResolveError.deviceUnavailable("busy")))
        // Anything unrecognized stays retriable: a wrongly-terminal failure is
        // discarded for the whole fleet, a wrongly-retriable one just expires.
        #expect(PlannerWorker.retriable(URLError(.timedOut)))
    }

    // MARK: - Local-model matching

    private static func mlxOnDisk(org: String, repo: String, subpath: String?) throws -> DiscoveredModel {
        DiscoveredModel(
            source: .mlx(.init(source: .huggingFace(repo: try HFRepo.parse("\(org)/\(repo)"), prefix: subpath.map { RepoSubpath(validated: $0) }))),
            path: "/models/\(repo)",
            sizeBytes: 1
        )
    }

    /// A `torch` cell can no longer be built as a `Model` at all — the kind is refused
    /// when the claim decodes, where the old bridge decoded it into an `unsupported` case
    /// and matched it against the local inventory. This is the guarantee that replaces
    /// "a torch coordinate must not match a local MLX model".
    @Test func aTorchClaimIsRefusedAsARunnableKind() throws {
        let job = try Self.claimedJob(benchmarkId: "b", spec: [
            "benchmark": "b",
            "model": ["type": "torch", "source": "huggingface", "org": "o", "repo_name": "r"],
            "runtime": ["type": "apple_foundation"],
        ])
        do {
            _ = try ClientRunSpec.runSpec(from: job)
            Issue.record("a torch model should not decode")
        } catch let error as UnrunnableClaim {
            guard case .modelKindNotRunnable = error else {
                Issue.record("expected modelKindNotRunnable, got \(error)")
                return
            }
            // Terminal, as every claim refusal is upstream.
            #expect(!error.retriable)
        }
    }

    /// A cell that asks for the private thermal gate is refused by a build without it: the
    /// two produce different numbers for one cell, because an ungated run is allowed to
    /// start hot. Terminal — no re-serving makes this device able to run it.
    @Test func aClaimRequiringPrivateThermalIsRefusedByAStockBuild() throws {
        let job = try Self.claimedJob(benchmarkId: "prefill_throughput_512", spec: [
            "benchmark": "prefill_throughput_512",
            "model": ["type": "gguf_text", "source": "huggingface",
                      "org": "LiquidAI", "repo_name": "r", "path": "m.gguf"],
            "runtime": ["type": "llamacpp_ios_pipette", "repository_version": "b10216",
                        "flavor": "ios-arm64", "private_thermal": true],
        ])

        do {
            // The capability refusals live in `parse`, where the engine identity is read;
            // `runSpec` only types the payload.
            _ = try ClientRunSpec.validated( try ClientRunSpec.runSpec(from: job))
            // A build compiled *with* the read accepts it, so only assert the refusal on
            // the build that lacks it — which is what the test target is.
            #expect(Runtime.privateThermalBuild, "a stock build should have refused")
        } catch let error as UnrunnableClaim {
            guard case .buildLacksPrivateThermal = error else {
                Issue.record("expected buildLacksPrivateThermal, got \(error)")
                return
            }
            #expect(!error.retriable)
        }
    }

    /// A cell that says nothing about it runs on either build — the field defaults to the
    /// stock meaning, so plans written before it keep working.
    @Test func aClaimSayingNothingAboutThermalRuns() throws {
        let job = try Self.claimedJob(benchmarkId: "prefill_throughput_512", spec: [
            "benchmark": "prefill_throughput_512",
            "model": ["type": "gguf_text", "source": "huggingface",
                      "org": "LiquidAI", "repo_name": "r", "path": "m.gguf"],
            "runtime": ["type": "llamacpp_ios_pipette", "repository_version": "b10216",
                        "flavor": "ios-arm64"],
        ])

        let spec = try ClientRunSpec.runSpec(from: job)
        _ = try ClientRunSpec.validated( spec)

        #expect(!spec.runtime.privateThermal)
    }

    /// A subdirectory pin is exact in both directions: a cell naming one does
    /// not take a bare checkout, and a cell naming none does not take a variant
    /// out of a repo that bundles several.
    @Test func anMlxSubdirectoryPinMatchesExactly() throws {
        let repo = try HFRepo.parse("o/r")
        let bare = try Self.mlxOnDisk(org: "o", repo: "r", subpath: nil)
        let nested = try Self.mlxOnDisk(org: "o", repo: "r", subpath: "4bit")
        func cell(_ prefix: String?) throws -> Model {
            .mlx(Mlx(source: .huggingFace(repo: repo, prefix: try prefix.map { try RepoSubpath($0) })))
        }

        #expect(!PlannerWorker.matchesModel(bare, coord: .init(try cell("4bit"))))
        #expect(PlannerWorker.matchesModel(nested, coord: .init(try cell("4bit"))))
        #expect(!PlannerWorker.matchesModel(nested, coord: .init(try cell(nil))))
    }

    /// A different repo never matches, whatever the type agreement.
    @Test func aDifferentRepoDoesNotMatch() throws {
        let onDisk = try Self.mlxOnDisk(org: "o", repo: "r", subpath: nil)
        let elsewhere = Model.mlx(Mlx(source: .huggingFace(repo: try HFRepo.parse("other/r"), prefix: nil)))
        #expect(!PlannerWorker.matchesModel(onDisk, coord: .init(elsewhere)))
    }
}
