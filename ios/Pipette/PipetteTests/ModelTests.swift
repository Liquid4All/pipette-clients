import Foundation
import Testing

@testable import Pipette

/// `Model` against plan-types' `model/mod.rs` (mgmt repo): wire shape, identity string,
/// submitted `model_descriptor`, auth token custody and `ModelFlags`.
struct ModelTests {
    private func roundTrip(_ m: Model) throws -> Model {
        try JSONDecoder().decode(Model.self, from: try JSONEncoder().encode(m))
    }

    private func decode(_ json: String) throws -> Model {
        try JSONDecoder().decode(Model.self, from: Data(json.utf8))
    }

    private func descriptor(_ model: Model) throws -> [String: Any] {
        let json = try SubmissionRef.model(model)
        return try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    private func encoded(_ model: Model) throws -> [String: Any] {
        try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(model)) as? [String: Any])
    }

    private func pinned(_ slug: String, _ revision: String) throws -> HfRepo {
        var repo = try HfRepo.parseOrgRepo(slug)
        repo.revision = try HfRevision(revision)
        return repo
    }

    private func tokenedModel() throws -> Model {
        var repo = try pinned("LiquidAI/Gated-GGUF", "v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        return .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil)))
    }

    // MARK: - Wire shape

    @Test func ggufTextMatchesCrateShapeAndRoundTrips() throws {
        let model = Model.ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-GGUF"), path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil)))
        let data = try JSONEncoder().encode(model)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["type"] as? String == "gguf_text")
        #expect(json["org"] as? String == "LiquidAI")
        #expect(json["repo_name"] as? String == "LFM2.5-350M-GGUF")
        #expect(json["path"] as? String == "LFM2.5-350M-Q4_K_M.gguf")
        #expect(try roundTrip(model) == model)
    }

    @Test func visionRoundTripsWithMmproj() throws {
        let model = Model.ggufVision(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-VL-3B-GGUF"), model: try RepoSubpath("model-Q4_K_M.gguf"), modelSha256: nil, mmproj: try RepoSubpath("mmproj-f16.gguf"), mmprojSha256: nil)))
        #expect(try roundTrip(model) == model)
    }

    @Test func mlxRootAndSubpathRoundTrip() throws {
        let root = Model.mlx(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil)))
        #expect(try roundTrip(root) == root)
        #expect(root.repo?.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        // A root MLX manifest omits `prefix` (matches the crate's bare HfMlx).
        let rootJSON = try #require(try JSONSerialization.jsonObject(with: JSONEncoder().encode(root)) as? [String: Any])
        #expect(rootJSON["prefix"] == nil)
        #expect(rootJSON["type"] as? String == "mlx")

        let sub = Model.mlx(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("org/multi"), prefix: try RepoSubpath("4bit"))))
        #expect(try roundTrip(sub) == sub)
        guard case let .mlx(mlx) = sub else { Issue.record("expected mlx"); return }
        #expect(mlx.ref?.leaf == "4bit")
    }

    @Test func decodesCrateTaggedJson() throws {
        let model = try decode(#"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#)
        #expect(model == .mlx(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))))
    }

    @Test func rejectsUnknownType() {
        #expect(throws: ModelError.unknownModelType("hf_torch")) {
            try decode(#"{"type":"hf_torch","org":"x","repo_name":"y"}"#)
        }
    }

    @Test func digestsRoundTripForEveryFetchingArm() throws {
        let repo = try HfRepo.parseOrgRepo("org/repo")
        let text = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("w.gguf"),
            sha256: try Sha256(String(repeating: "a", count: 64)))))
        let vision = Model.ggufVision(GgufVision(source: .huggingFace(
            repo: repo,
            model: try RepoSubpath("w.gguf"),
            modelSha256: try Sha256(String(repeating: "b", count: 64)),
            mmproj: try RepoSubpath("mm.gguf"),
            mmprojSha256: try Sha256(String(repeating: "c", count: 64)))))

        for model in [text, vision] {
            #expect(try roundTrip(model) == model)
        }
    }

    // MARK: - Apple Foundation (the bare case: no coordinate, no disk footprint)

    @Test func appleFoundationEncodesBareDiscriminatorAndRoundTrips() throws {
        let model = Model.appleFoundationText
        let data = try JSONEncoder().encode(model)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["type"] as? String == "apple_foundation_text")
        #expect(json.count == 1)
        #expect(try roundTrip(model) == model)
        #expect(try decode(#"{"type":"apple_foundation_text"}"#) == .appleFoundationText)
    }

    @Test func appleFoundationHasNoCoordinateOrQuant() {
        let model = Model.appleFoundationText
        #expect(model.repo == nil)
        #expect(model.quant == nil)
        #expect(model.familyId == "apple-foundation")
        #expect(model.engineLabel == "Apple Foundation")
        #expect(model.authToken == nil)
    }

    /// AFM has no path to bind, so it needs no separate located form.
    @Test func appleFoundationIsAlreadyBound() {
        let pair = DeclaredBound.alreadyBound(Model.appleFoundationText)
        #expect(pair.declared == .appleFoundationText)
        #expect(pair.bound == .appleFoundationText)
    }

    // MARK: - Identity: the crate's `Display`

    /// `gguf_text_from_hf_parses_and_references`, `…_from_local_file_…`, `…_from_url_…`.
    @Test func aTextModelIsNamedByItsSource() throws {
        #expect(try decode(#"{"type":"gguf_text","source":"huggingface","org":"org","repo_name":"repo","path":"Q4_K_M.gguf"}"#).description
            == "org/repo:Q4_K_M.gguf")
        #expect(try decode(#"{"type":"gguf_text","source":"absolute_file","path":"/models/m.gguf"}"#).description
            == "/models/m.gguf")
        #expect(try decode(#"{"type":"gguf_text","source":"url","url":"https://cdn.example/m.gguf"}"#).description
            == "https://cdn.example/m.gguf")
    }

    /// `gguf_vision_from_hf_references_model_and_mmproj` and
    /// `gguf_vision_file_arms_name_both_files`: weights then projector.
    @Test func aVisionModelNamesBothFiles() throws {
        #expect(try decode(#"{"type":"gguf_vision","source":"huggingface","org":"org","repo_name":"repo","model":"a.gguf","mmproj":"mmproj.gguf"}"#).description
            == "org/repo:a.gguf+org/repo:mmproj.gguf")
        #expect(try decode(#"{"type":"gguf_vision","source":"absolute_files","model":"/m/a.gguf","mmproj":"/m/mmproj.gguf"}"#).description
            == "/m/a.gguf+/m/mmproj.gguf")
        #expect(try decode(#"{"type":"gguf_vision","source":"url","model":"https://cdn.example/m.gguf","mmproj":"https://cdn.example/mmproj.gguf"}"#).description
            == "https://cdn.example/m.gguf+https://cdn.example/mmproj.gguf")
    }

    /// `mlx_pins_revision_and_folds_it_into_identity`, `mlx_hf_prefix_folds_into_identity`,
    /// `mlx_from_local_dir_parses_needs_no_auth`.
    @Test func anMlxModelFoldsRevisionAndPrefixIntoItsIdentity() throws {
        #expect(try decode(#"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit","revision":"v2"}"#).description
            == "LiquidAI/LFM2.5-350M-MLX-4bit@v2")
        #expect(try decode(#"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#).description
            == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(try decode(#"{"type":"mlx","source":"huggingface","org":"mlx-community","repo_name":"bundle","prefix":"4bit"}"#).description
            == "mlx-community/bundle:4bit")
        let local = try decode(#"{"type":"mlx","source":"absolute_dir","dir":"/models/LFM2.5-350M-MLX-4bit"}"#)
        #expect(local.description == "/models/LFM2.5-350M-MLX-4bit")
        #expect(local.authToken == nil)
    }

    /// `apple_foundation_text_variant_round_trips`: the string the submitted `model_name`
    /// uses, so a plan reference and a warehouse key agree.
    @Test func appleFoundationIsNamedAsItIsSubmitted() {
        #expect(Model.appleFoundationText.description == AFMRuntime.submissionModelName)
        #expect(Model.appleFoundationText.description == "apple/foundation-text")
    }

    // MARK: - The submitted descriptor

    /// Two pins of one repo must stay distinct upstream, as they are in the storage key.
    @Test func aPinnedRevisionReachesTheDescriptor() throws {
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: try pinned("org/repo", "v1.2.3"),
            path: try RepoSubpath("w.gguf"), sha256: nil)))

        #expect(try descriptor(model)["revision"] as? String == "v1.2.3")
    }

    /// Absent, not null, matching the crate's `skip_serializing_if`.
    @Test func anUnpinnedModelOmitsTheKeyEntirely() throws {
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: try HfRepo.parseOrgRepo("org/repo"),
            path: try RepoSubpath("w.gguf"), sha256: nil)))

        #expect(!(try descriptor(model)).keys.contains("revision"))
    }

    @Test func everyFetchableArmCarriesTheCoordinate() throws {
        let repo = try pinned("org/repo", "v1")
        let models: [Model] = [
            .ggufText(GgufText(source: .huggingFace(
                repo: repo, path: try RepoSubpath("w.gguf"), sha256: nil))),
            .ggufVision(GgufVision(source: .huggingFace(
                repo: repo, model: try RepoSubpath("w.gguf"), modelSha256: nil,
                mmproj: try RepoSubpath("mm.gguf"), mmprojSha256: nil))),
            .mlx(Mlx(source: .huggingFace(repo: repo, prefix: try RepoSubpath("4bit")))),
        ]

        for model in models {
            let json = try descriptor(model)
            #expect(json["org"] as? String == "org", "\(ModelType.of(model))")
            #expect(json["repo_name"] as? String == "repo", "\(ModelType.of(model))")
            #expect(json["revision"] as? String == "v1", "\(ModelType.of(model))")
            #expect(json["source"] as? String == "huggingface", "\(ModelType.of(model))")
        }
    }

    /// AFM ships with the OS: a bare tag, no coordinate to pin.
    @Test func appleFoundationStaysABareTag() throws {
        let json = try descriptor(.appleFoundationText)

        #expect(json["type"] as? String == "apple_foundation_text")
        #expect(json.count == 1)
    }

    // MARK: - Token custody

    /// `Model.encode` drops the token unconditionally, so nothing a `Model` is persisted
    /// in can carry it (`docs/pipette-ios/execution-alignment.md`).
    @Test func encodingAModelNeverWritesTheToken() throws {
        let encoded = try JSONEncoder().encode(tokenedModel())
        let json = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(json["auth_token"] == nil)
        #expect(!String(decoding: encoded, as: UTF8.self).contains("hf_secret"))
        // The revision is identity and feeds the storage key, so it must survive.
        #expect(json["revision"] as? String == "v1.2.3")
    }

    /// Decode still accepts one: that is how a claim delivers a credential in the first
    /// place. Strip on the way out only.
    @Test func decodingAModelStillAcceptsAToken() throws {
        let model = try decode(#"""
        {"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r",
         "path":"m.gguf","auth_token":"hf_secret"}
        """#)
        #expect(model.repo?.authToken?.value == "hf_secret")
    }

    @Test func aRoundTripKeepsIdentityAndDropsTheCredential() throws {
        let original = try tokenedModel()
        let restored = try roundTrip(original)
        #expect(restored.repo?.authToken == nil)
        #expect(restored == original)  // identity ignores the token
        #expect(restored.repo?.reference == "LiquidAI/Gated-GGUF@v1.2.3")
    }

    /// The identity string never names the token either.
    @Test func noModelKindSubmitsItsToken() throws {
        var repo = try pinned("LiquidAI/Gated", "v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        let models: [Model] = [
            .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil))),
            .ggufVision(.init(source: .huggingFace(repo: repo, model: try RepoSubpath("m.gguf"), modelSha256: nil, mmproj: try RepoSubpath("mmproj.gguf"), mmprojSha256: nil))),
            .mlx(.init(source: .huggingFace(repo: repo, prefix: try RepoSubpath("4bit")))),
        ]
        for model in models {
            let descriptor = try SubmissionRef.model(model)
            #expect(!descriptor.contains("hf_secret"), "\(ModelType.of(model)) leaked its token")
            #expect(!descriptor.contains("auth_token"))
            #expect(!model.description.contains("hf_secret"))
        }
    }

    /// Two files from one repo may be granted different credentials, so custody is per
    /// model. Asserts the derivation: a Simulator test host cannot store Keychain items.
    @Test func twoModelsInOneRepoKeySeparately() throws {
        let repo = try HfRepo.parseOrgRepo("LiquidAI/Gated-GGUF")
        let q4 = Model.ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil)))
        let q8 = Model.ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q8_0.gguf"), sha256: nil)))
        let account = try #require(KeychainHelper.modelTokenAccount(q4))

        #expect(account != KeychainHelper.modelTokenAccount(q8))
        #expect(account.hasSuffix("LiquidAI/Gated-GGUF:m-Q4_K_M.gguf"))
    }

    @Test func aPinnedModelKeysSeparatelyFromAnUnpinnedOne() throws {
        var repo = try HfRepo.parseOrgRepo("LiquidAI/Gated-GGUF")
        let filename = try RepoSubpath("m-Q4_K_M.gguf")
        let unpinned = KeychainHelper.modelTokenAccount(.ggufText(.init(source: .huggingFace(repo: repo, path: filename, sha256: nil))))
        repo.revision = try HfRevision("v1.2.3")
        let pinned = KeychainHelper.modelTokenAccount(.ggufText(.init(source: .huggingFace(repo: repo, path: filename, sha256: nil))))

        #expect(pinned != unpinned)
        #expect(pinned?.hasSuffix("LiquidAI/Gated-GGUF@v1.2.3:m-Q4_K_M.gguf") == true)
    }

    /// The stored token is looked up from the persisted copy, which has been stripped. If
    /// stripping moved the account, the whole stored tier would be write-only.
    @Test func strippingTheTokenDoesNotMoveTheAccount() throws {
        let model = try tokenedModel()
        #expect(KeychainHelper.modelTokenAccount(model)
            == KeychainHelper.modelTokenAccount(model.withoutAuthToken))
    }

    @Test func appleFoundationHasNoTokenAccount() {
        #expect(KeychainHelper.modelTokenAccount(.appleFoundationText) == nil)
    }

    /// No CLI equivalent: the CLI stores no model tokens for `auth reset` to clear.
    @MainActor
    @Test func resetClearsEveryStoredModelToken() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var cleared = 0

        #expect(AuthCommands.reset(force: true, ws: ws,
                                   clearModelTokens: { cleared += 1; return 2 }))
        #expect(cleared == 1)
    }

    /// `auth reset` refuses only over results awaiting submission, so the test seeds one.
    @MainActor
    @Test func aRefusedResetClearsNothing() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let pending = try #require(ws.results.resultDir(.remotePending, CellId("r-1")))
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)
        var touched = 0

        #expect(!AuthCommands.reset(force: false, ws: ws,
                                    clearModelTokens: { touched += 1; return 0 }))
        #expect(touched == 0)
    }

    // MARK: - ModelFlags

    /// `model_flag_ref_routes_eval_pair_to_its_variant`.
    @Test(arguments: [
        (ModelType.ggufText, ModelFlags.evalGgufText(enableThinking: true)),
        (.ggufVision, .evalGgufVision(enableThinking: true)),
        (.mlx, .evalMlx(enableThinking: true)),
        (.torch, .evalTorch(enableThinking: true)),
    ])
    func aModelFlagRefRoutesAnEvalPairToItsVariant(model: ModelType, expected: ModelFlags) throws {
        let flags = try ModelFlagRef(benchmarkType: .eval, modelType: model, enableThinking: true)
            .resolve()
        #expect(flags == expected)
        #expect(flags.axes.benchmark == .eval)
        #expect(flags.axes.model == model)
    }

    /// `routing_cases_cover_every_variant`: every variant is some pair's answer, and no two
    /// pairs share one.
    @Test func routingCoversEveryVariant() throws {
        let routed = try [ModelType.ggufText, .ggufVision, .mlx, .torch].map {
            try ModelFlagRef(benchmarkType: .eval, modelType: $0).resolve()
        }
        #expect(routed.count == ModelFlags.allCases.count)
        #expect(Set(routed.map { $0.axes.model.rawValue }).count == routed.count)
    }

    /// `model_flag_ref_rejects_unsupported_pair`: flags are eval-only, and Apple Foundation
    /// carries no generation settings.
    @Test(arguments: [
        (ModelType.ggufText, BenchmarkType.prefillThroughput),
        (.ggufText, .decodeThroughput),
        (.ggufText, .endToEndLatency),
        (.appleFoundationText, .eval),
    ])
    func aModelFlagRefRejectsAnUnsupportedPair(model: ModelType, benchmark: BenchmarkType) {
        #expect(throws: ModelFlagsError.noSuchCombination(benchmark: benchmark, model: model)) {
            try ModelFlagRef(benchmarkType: benchmark, modelType: model, enableThinking: true)
                .resolve()
        }
    }

    /// `reasoning_effort_routes_only_to_the_cells_that_take_it`: the llama.cpp text and
    /// torch cells carry it; the vision and MLX cells have no field for it and refuse it
    /// by name.
    @Test(arguments: [(ModelType.ggufText, true), (.ggufVision, false), (.mlx, false),
                      (.torch, true)])
    func aReasoningEffortRoutesOnlyToTheCellsThatTakeIt(model: ModelType, takesIt: Bool) {
        for effort in ReasoningEffort.allCases {
            let ref = ModelFlagRef(benchmarkType: .eval, modelType: model, enableThinking: true,
                                   reasoningEffort: effort)
            if takesIt {
                let flags = try? ref.resolve()
                #expect(flags?.enableThinking == true, "\(model) \(effort)")
                #expect(flags?.reasoningEffort == effort, "\(model) \(effort)")
            } else {
                #expect(throws: ModelFlagsError.knobNotAllowed(
                    knob: "reasoning_effort", benchmark: .eval, model: model)) {
                    try ref.resolve()
                }
            }
        }
    }

    /// `reasoning_effort_round_trips_through_flat_wire` and
    /// `reasoning_effort_has_one_spelling`: the lowercase word on the wire, read back to
    /// the same variant.
    @Test(arguments: [(ReasoningEffort.low, "low"), (.medium, "medium"), (.high, "high")])
    func aReasoningEffortRoundTripsThroughTheFlatWire(effort: ReasoningEffort,
                                                      word: String) throws {
        let flags = ModelFlags.evalGgufText(enableThinking: nil, reasoningEffort: effort)
        let wire = try JSONEncoder().encode(ModelFlagRef(flags))
        let json = try #require(try JSONSerialization.jsonObject(with: wire) as? [String: Any])
        #expect(json["reasoning_effort"] as? String == word)
        #expect(json["enable_thinking"] == nil)
        #expect(try JSONDecoder().decode(ModelFlagRef.self, from: wire).resolve() == flags)
    }

    /// `model_flags_refuse_a_reasoning_effort_they_cannot_carry`: the set is closed, so
    /// `none` and any other spelling fail to decode.
    @Test(arguments: ["none", "minimal", "Low"])
    func aReasoningEffortOutsideTheSetIsRefused(word: String) {
        let json = #"{"model_type":"torch","benchmark_type":"eval","reasoning_effort":"\#(word)"}"#
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(ModelFlagRef.self, from: Data(json.utf8))
        }
    }

    /// `model_flags_round_trips_through_flat_wire`.
    @Test func modelFlagsRoundTripThroughTheFlatWire() throws {
        let flags = ModelFlags.evalTorch(enableThinking: false)
        let wire = try JSONEncoder().encode(ModelFlagRef(flags))
        let json = try #require(try JSONSerialization.jsonObject(with: wire) as? [String: Any])
        #expect(json["model_type"] as? String == "torch")
        #expect(json["benchmark_type"] as? String == "eval")
        #expect(try JSONDecoder().decode(ModelFlagRef.self, from: wire).resolve() == flags)
    }

    /// `model_flag_ref_rejects_unknown_field`: a typo like `enabel_thinking` is refused.
    @Test func aModelFlagRefRejectsAnUnknownField() {
        let json = #"{"model_type":"gguf_text","benchmark_type":"eval","enabel_thinking":true}"#
        #expect(throws: UnknownFlagField.self) {
            try JSONDecoder().decode(ModelFlagRef.self, from: Data(json.utf8))
        }
    }

    /// `model_flags_enable_thinking_omitted_from_wire_when_unset`.
    @Test func anUnsetEnableThinkingIsOmittedFromTheWire() throws {
        let wire = try JSONEncoder().encode(ModelFlagRef(.evalMlx(enableThinking: nil)))
        #expect(!String(decoding: wire, as: UTF8.self).contains("enable_thinking"))
    }

    /// `model_flags_canonical_string_reflects_enable_thinking` and
    /// `model_flags_submission_string_eval_keeps_canonical_form`.
    @Test(arguments: [(Bool?.some(true), "enable_thinking=true"),
                      (false, "enable_thinking=false"),
                      (nil, nil)] as [(Bool?, String?)])
    func theCanonicalStringReflectsEnableThinking(enableThinking: Bool?, expected: String?) {
        let flags = ModelFlags.evalGgufText(enableThinking: enableThinking)
        #expect(flags.canonicalString == expected)
        #expect(flags.submissionString(.eval) == expected)
    }

    /// `model_flags_canonical_string_appends_reasoning_effort`: the effort follows the
    /// thinking flag, and stands alone when no thinking flag is set.
    @Test(arguments: [(Bool?.some(true), ReasoningEffort.low,
                       "enable_thinking=true,reasoning_effort=low"),
                      (false, .high, "enable_thinking=false,reasoning_effort=high"),
                      (nil, .medium, "reasoning_effort=medium")] as [(Bool?, ReasoningEffort, String)])
    func theCanonicalStringAppendsTheReasoningEffort(enableThinking: Bool?,
                                                      effort: ReasoningEffort,
                                                      expected: String) {
        let flags = ModelFlags.evalTorch(enableThinking: enableThinking, reasoningEffort: effort)
        #expect(flags.canonicalString == expected)
        #expect(flags.submissionString(.eval) == expected)
    }

    /// `model_flags_submission_string_non_eval_reports_nothing`: a throughput or memory row
    /// is insensitive to the flag, so carrying it would split warehouse joins.
    @Test func aNonEvalSubmissionReportsNoModelFlags() {
        for benchmark in BenchmarkType.allCases where benchmark != .eval {
            for enableThinking in [true, false, nil] as [Bool?] {
                #expect(ModelFlags.evalGgufText(enableThinking: enableThinking)
                    .submissionString(benchmark) == nil, "\(benchmark)")
            }
        }
    }

    @Test func modelFlagsMatchTheirCellAlone() throws {
        let flags = ModelFlags.evalMlx(enableThinking: nil)
        let mlx = try mlxSpec("org/m-MLX-4bit")
        #expect(flags.matches(benchmark: .eval, model: mlx))
        #expect(!flags.matches(benchmark: .prefillThroughput, model: mlx))
        #expect(!flags.matches(benchmark: .eval, model: try ggufTextSpec("org/a-GGUF", "a.gguf")))
    }

    // MARK: - Completeness against the crate
    //
    // The axis spells every plan-types variant, so a recognized one this client cannot run
    // is refused by variant instead of failing to decode.

    @Test(arguments: ["gguf_text", "gguf_vision", "mlx", "torch", "openvino", "onnx_genai",
                      "apple_foundation_text"])
    func modelTypeSpellsEveryCrateVariant(raw: String) {
        #expect(ModelType(rawValue: raw) != nil)
    }

    /// A spelling neither side knows still fails, which is what makes the above a
    /// completeness claim rather than a permissive decoder.
    @Test func anInventedModelTypeStillFails() {
        #expect(ModelType(rawValue: "gguf_quantum") == nil)
    }

    // MARK: - Vault arms

    /// The crate's `a_vault_url_carries_its_principal_and_its_own_tag` and its twins: each
    /// vault arm reads, and writes back the JSON it read, principal and pins included.
    @Test(arguments: [
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","principal_id":"org_01936d"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"user_01936d"}"#,
        #"{"type":"gguf_vision","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","mmproj_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","principal_id":"org_1"}"#,
        #"{"type":"gguf_vision_speculative","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","draft":"/vault/resources/vault_3/content","draft_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","principal_id":"org_1"}"#,
        #"{"type":"mlx","source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#,
        #"{"type":"onnx_genai","source":"vault_archive","url":"/vault/resources/vault_1/content","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","principal_id":"org_1"}"#,
    ])
    func aVaultArmWritesBackTheJsonItRead(json: String) throws {
        let model = try decode(json)
        let written = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(model))
        let read = try JSONSerialization.jsonObject(with: Data(json.utf8))
        #expect((written as? NSDictionary) == (read as? NSDictionary))
        #expect(try roundTrip(model) == model)
    }

    /// The crate's `a_vault_model_keeps_its_custom_label`, on every vault arm: a label reads
    /// trimmed and is written back, none writes no key, and a blank one fails the decode.
    /// Read off the JSON, because `==` ignores the label.
    @Test(arguments: [
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_01936d"}"#,
        #"{"type":"gguf_vision","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","principal_id":"org_01936d"}"#,
        #"{"type":"gguf_vision_speculative","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","draft":"/vault/resources/vault_3/content","principal_id":"org_01936d"}"#,
        #"{"type":"mlx","source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"org_01936d"}"#,
        #"{"type":"onnx_genai","source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"org_01936d"}"#,
    ])
    func aVaultModelKeepsItsCustomLabel(json: String) throws {
        func labelled(_ label: String) -> String {
            String(json.dropLast()) + #","custom_label":\#(label)}"#
        }
        #expect(try encoded(decode(json))["custom_label"] == nil)
        #expect(try encoded(decode(labelled("null")))["custom_label"] == nil)

        let model = try decode(labelled(#"" acme LFM3 q4 ""#))
        #expect(try encoded(model)["custom_label"] as? String == "acme LFM3 q4")
        #expect(try encoded(roundTrip(model))["custom_label"] as? String == "acme LFM3 q4")

        for blank in [#""""#, #""  ""#] {
            #expect(throws: ModelError.emptyValue("")) { try decode(labelled(blank)) }
        }
        #expect(throws: DecodingError.self) { try decode(labelled("7")) }
    }

    /// A label names nothing: the labelled spec is the unlabelled one to `==`, to a hashed
    /// collection and to every derived name, and clearing it leaves the unlabelled spec.
    @Test func aCustomLabelIsNoPartOfIdentity() throws {
        for (plain, labelled) in try vaultSpecPairs() {
            #expect(labelled == plain)
            #expect(labelled.hashValue == plain.hashValue)
            #expect(Set([plain, labelled]).count == 1)
            #expect(labelled.description == plain.description)
            #expect(labelled.artifactName == plain.artifactName)
            #expect(labelled.familyId == plain.familyId)
            // Guards the premise: the two differ on the wire, and only by the label.
            #expect(try encoded(labelled)["custom_label"] as? String == "acme LFM3 q4")
            #expect(try encoded(plain)["custom_label"] == nil)
            #expect(try encoded(labelled.withoutCustomLabel) as NSDictionary
                == encoded(plain) as NSDictionary)
        }
    }

    /// The crate's `vision()` clones the pair's files whole, so the projected half of a
    /// speculative model still carries the label.
    @Test func theVisionHalfOfASpeculativeModelKeepsTheLabel() throws {
        let model = try vaultSpeculativeSpec("vault_a", "vault_b", draft: nil, label: "acme LFM3 q4")
        guard case let .ggufVisionSpeculative(m) = model else {
            Issue.record("expected a speculative model"); return
        }
        #expect(try encoded(.ggufVision(GgufVision(source: m.source.vision)))["custom_label"]
            as? String == "acme LFM3 q4")
    }

    /// The submitted `model_descriptor` is the claimed model re-encoded, so it reports the
    /// label a claim carried and gains no key when the claim carried none.
    @Test func theSubmittedDescriptorCarriesTheClaimedLabel() throws {
        for (plain, labelled) in try vaultSpecPairs() {
            #expect(try descriptor(labelled)["custom_label"] as? String == "acme LFM3 q4")
            #expect(try descriptor(plain)["custom_label"] == nil)
        }
    }

    /// A public arm has no label: the crate drops the key there, and so does this client.
    @Test func aPublicArmDropsACustomLabel() throws {
        let model = try decode(#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"m.gguf","custom_label":"acme LFM3 q4"}"#)
        #expect(try encoded(model)["custom_label"] == nil)
    }

    /// The crate's `gguf_text_vault`, `gguf_vision_vault` and `mlx_vault` identity cases:
    /// a vault arm is named by its path, a vision pair by both.
    @Test func aVaultArmIsNamedByItsPath() throws {
        #expect(try decode(#"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#).description
            == "/vault/resources/vault_1/content")
        #expect(try decode(#"{"type":"gguf_vision","source":"vault_url","model":"/vault/resources/vault_1/content","mmproj":"/vault/resources/vault_2/content","principal_id":"org_1"}"#).description
            == "/vault/resources/vault_1/content+/vault/resources/vault_2/content")
        #expect(try decode(#"{"type":"mlx","source":"vault_archive","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#).description
            == "/vault/resources/vault_1/content")
    }

    /// Every vault path ends in `content`, so a vault model takes its resource id as its
    /// name and its family. Two vault models stay apart, and a job cell is not unnamed.
    @Test func aVaultModelIsLabelledByItsResourceId() throws {
        let text = try decode(#"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#)
        let vision = try decode(#"{"type":"gguf_vision","source":"vault_url","model":"/vault/resources/vault_2/content","mmproj":"/vault/resources/vault_3/content","principal_id":"org_1"}"#)
        #expect(text.artifactName == "vault_1")
        #expect(text.familyId == "vault_1")
        #expect(vision.artifactName == "vault_2")
        #expect(vision.familyId == "vault_2")

        let cell = JobCell(
            cellId: CellId("cell-vault"), benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput, runStatus: .pending,
            serverJobId: nil, errorMessage: nil, source: text)
        #expect(cell.modelName == "vault_1")
    }

    /// The crate's `VaultPath` and `VaultPrincipalId` refusals: an absolute URL, another
    /// route, an id that is not a vault id, an unknown principal kind, and a vault arm
    /// with no principal.
    @Test(arguments: [
        #"{"type":"gguf_text","source":"vault_url","url":"https://mgmt.example/vault/resources/vault_1/content","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/datasets/vault_1/content","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/org_1/content","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_a/b/content","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources//content","principal_id":"org_1"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"acme"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_has space"}"#,
        #"{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_1/content"}"#,
        #"{"type":"mlx","source":"vault_url","url":"/vault/resources/vault_1/content","principal_id":"org_1"}"#,
    ])
    func aVaultBodyTheCrateRefusesIsRefused(json: String) {
        #expect(throws: (any Error).self) { try decode(json) }
    }

    /// Padding around the path or its id reads back without it, as the crate trims both.
    @Test func aPaddedVaultPathReadsBackCanonical() throws {
        #expect(try VaultPath("  /vault/resources/ vault_1 /content ").value
            == "/vault/resources/vault_1/content")
        #expect(try VaultPath("/vault/resources/vault_1/content").resourceId.value == "vault_1")
        #expect(try VaultPrincipalId(" user_01936d ").value == "user_01936d")
    }
}
