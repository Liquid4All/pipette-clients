import Foundation
import Testing

@testable import Pipette

/// `Runtime` against plan-types' `runtime/` and `cell/mod.rs` `is_compatible` (mgmt repo):
/// the submitted `runtime_descriptor`, what a claim may declare, the identity string, and
/// which models each runtime loads.
struct RuntimeTests {
    private func object(_ runtime: Runtime) throws -> [String: Any] {
        let json = try SubmissionRef.runtime(runtime)
        return try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    private func llama(_ version: String, privateApis: PrivateApis = .stock) throws -> Runtime {
        .llamacppIosPipette(
            source: SourceRepository(repositoryUrl: RepositoryUrl(LlamaPins.main.repositoryUrl),
                                     repositoryVersion: try NonEmptyString(version)),
            flavor: .iosArm64, privateApis: privateApis)
    }

    // MARK: - Descriptor

    /// Flattened, not nested: the crate's `#[serde(flatten)]`.
    @Test func theLlamaCppDescriptorFlattensItsSourceRepository() throws {
        let json = try object(try llama("b77"))

        #expect(json["type"] as? String == "llamacpp_ios_pipette")
        #expect(json["repository_url"] as? String == "github.com/ggml-org/llama.cpp")
        #expect(json["repository_version"] as? String == "b77")
        #expect(json["flavor"] as? String == "ios-arm64")
        #expect(json["packages"] == nil)
    }

    /// MLX pins a stack rather than one repo, so its version is three named repos.
    @Test func theMlxDescriptorCarriesTheSwiftStack() throws {
        let json = try object(.mlxIosPipette(packages: .thisBuild, flavor: .iosArm64,
                                             privateApis: .stock))
        let packages = try #require(json["packages"] as? [String: Any])

        #expect(json["type"] as? String == "mlx_ios_pipette")
        for key in ["mlx_swift", "mlx_swift_lm", "swift_transformers"] {
            let repo = try #require(packages[key] as? [String: Any], "\(key) missing")
            #expect((repo["repository_url"] as? String)?.isEmpty == false)
            #expect((repo["repository_version"] as? String)?.isEmpty == false)
        }
    }

    /// ONNX Runtime GenAI pins two source trees, so its version is two named repos.
    @Test func theOnnxGenaiDescriptorCarriesBothSourceTrees() throws {
        let json = try object(.onnxGenaiIosPipette(packages: .thisBuild, flavor: .iosArm64,
                                                   privateApis: .stock))
        let packages = try #require(json["packages"] as? [String: Any])

        #expect(json["type"] as? String == "onnx_genai_ios_pipette")
        for key in ["onnxruntime", "onnxruntime_genai"] {
            let repo = try #require(packages[key] as? [String: Any], "\(key) missing")
            #expect((repo["repository_url"] as? String)?.isEmpty == false)
            #expect((repo["repository_version"] as? String)?.isEmpty == false)
        }
    }

    /// Duplicated verbatim from plan-types' `descriptor.rs`
    /// `onnx_genai_ios_descriptor_matches_the_swift_mirror` (mgmt repo). Two clients
    /// submitting different bytes for one build would key it under two ids, so the
    /// literals are repeated rather than derived.
    @Test func theOnnxGenaiDescriptorMatchesTheCrate() throws {
        let canonical = #"{"flavor":"ios-arm64","packages":{"onnxruntime":{"repository_url":"#
            + #""github.com/microsoft/onnxruntime","repository_version":"v1.26.0"},"#
            + #""onnxruntime_genai":{"repository_url":"github.com/microsoft/onnxruntime-genai","#
            + #""repository_version":"v0.15.2"}},"type":"onnx_genai_ios_pipette"}"#
        let digest = "529c5b33969fd906c9cae6a75bc714486be88834cc946421724df20000821653"

        let runtime = Runtime.onnxGenaiIosPipette(
            packages: OnnxGenaiStack(
                onnxruntime: SourceRepository(
                    repositoryUrl: RepositoryUrl("github.com/microsoft/onnxruntime"),
                    repositoryVersion: try NonEmptyString("v1.26.0")),
                onnxruntimeGenai: SourceRepository(
                    repositoryUrl: RepositoryUrl("github.com/microsoft/onnxruntime-genai"),
                    repositoryVersion: try NonEmptyString("v0.15.2"))),
            flavor: .iosArm64, privateApis: .stock)

        // The digest is the contract, not the raw bytes: `SubmissionRef.runtime` emits each
        // type's own field order, and the server canonicalizes before storing.
        let encoded = try JSONSerialization.jsonObject(
            with: Data(try SubmissionRef.runtime(runtime).utf8), options: [.fragmentsAllowed])
        #expect(canonicalJson(encoded) == canonical)
        #expect(try descriptorDigest(runtime) == digest)
    }

    /// AFM ships with the OS: no repo, no ref, so the descriptor is the tag alone.
    @Test func appleFoundationIsABareTag() throws {
        let json = try object(.appleFoundation(privateApis: .stock))

        #expect(json["type"] as? String == "apple_foundation")
        #expect(json.count == 1)
    }

    /// AFM's encoder case is empty; the build flags still reach its descriptor.
    @Test func appleFoundationCarriesBothBuildFlags() throws {
        let json = try object(.appleFoundation(privateApis: PrivateApis(thermal: true, memory: false)))

        #expect(json["private_thermal"] as? Bool == true)
        #expect(json["private_memory"] as? Bool == false)
        #expect(json.count == 3)
    }

    @Test func aDescriptorRoundTrips() throws {
        let runtime = Runtime.llamacppIosPipette(
            source: SourceRepository(repositoryUrl: RepositoryUrl("github.com/fork/llama.cpp"),
                                     repositoryVersion: try NonEmptyString("b91")),
            flavor: .iosArm64, privateApis: .stock)
        let restored = try JSONDecoder().decode(
            Runtime.self, from: JSONEncoder().encode(runtime))

        #expect(restored == runtime)
    }

    /// `private_thermal_separates_two_otherwise_identical_ios_runtimes`: the gate's
    /// fidelity is part of the identity, absent means stock, and a stock build's wire form
    /// is unchanged by the field existing.
    @Test func privateThermalSeparatesTwoOtherwiseIdenticalRuntimes() throws {
        let stock = try llama("b10216")
        let gated = try llama("b10216", privateApis: PrivateApis(thermal: true, memory: false))
        #expect(stock != gated)

        let absent = try JSONDecoder().decode(Runtime.self, from: Data(
            #"{"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10216","flavor":"ios-arm64"}"#.utf8))
        #expect(absent == stock)
        #expect(try object(stock)["private_thermal"] == nil)
        #expect(try object(gated)["private_thermal"] as? Bool == true)
    }

    /// `private_memory_defaults_to_private_thermal`: written only when it differs, so a
    /// build with both switches keeps the `private_thermal: true` it had before the field.
    @Test(arguments: [
        (#"{"type":"apple_foundation"}"#, PrivateApis.stock),
        (#"{"private_thermal":true,"type":"apple_foundation"}"#,
         PrivateApis(thermal: true, memory: true)),
        (#"{"private_memory":false,"private_thermal":true,"type":"apple_foundation"}"#,
         PrivateApis(thermal: true, memory: false)),
        (#"{"private_memory":true,"type":"apple_foundation"}"#,
         PrivateApis(thermal: false, memory: true)),
    ])
    func privateMemoryDefaultsToPrivateThermal(wire: String, apis: PrivateApis) throws {
        let runtime = Runtime.appleFoundation(privateApis: apis)

        #expect(try JSONDecoder().decode(Runtime.self, from: Data(wire.utf8)) == runtime)
        let written = try JSONSerialization.jsonObject(with: Data(wire.utf8)) as? NSDictionary
        #expect(try object(runtime) as NSDictionary == written)
    }

    /// `a_redundant_private_memory_is_dropped`.
    @Test(arguments: [
        #"{"type":"apple_foundation","private_thermal":true,"private_memory":true}"#,
        #"{"type":"apple_foundation","private_memory":false}"#,
    ])
    func aRedundantPrivateMemoryIsDropped(wire: String) throws {
        let runtime = try JSONDecoder().decode(Runtime.self, from: Data(wire.utf8))

        #expect(try object(runtime)["private_memory"] == nil)
    }

    /// A desktop runtime is refused where it is read, and the error names the spelling so
    /// the claim path can report what it rejected rather than "unreadable".
    @Test(arguments: ["uv_vllm", "docker_sglang", "llamacpp_cli_stock_tools", "nonsense"])
    func aRuntimeThisDeviceCannotBeIsRefusedByName(type: String) throws {
        let json = Data(#"{"type":"\#(type)","flavor":"ios-arm64"}"#.utf8)

        #expect(throws: RuntimeIdentityError.unsupportedRuntimeType(type)) {
            try JSONDecoder().decode(Runtime.self, from: json)
        }
    }

    /// `source_repo` for `repository_url` and `version` for `repository_version`, as the
    /// crate's serde aliases accept.
    @Test func theAliasesFollowTheCrate() throws {
        let json = Data(#"{"type":"llamacpp_ios_pipette","source_repo":"github.com/fork/llama.cpp","version":"b77","flavor":"ios-arm64"}"#.utf8)
        let runtime = try JSONDecoder().decode(Runtime.self, from: json)

        guard case let .llamacppIosPipette(source, _, _) = runtime else {
            Issue.record("expected llamacppIosPipette"); return
        }
        #expect(source.repositoryVersion.value == "b77")
        #expect(source.repositoryUrl.value == "github.com/fork/llama.cpp")
    }

    /// No repo has a default, however the `SourceRepository` is reached.
    @Test(arguments: [
        #"{"type":"llamacpp_ios_pipette","repository_version":"b1","flavor":"ios-arm64"}"#,
        #"""
        {"type":"mlx_ios_pipette","flavor":"ios-arm64","packages":{
         "mlx_swift":{"repository_version":"1"},
         "mlx_swift_lm":{"repository_url":"github.com/ml-explore/mlx-swift-lm","repository_version":"1"},
         "swift_transformers":{"repository_url":"github.com/huggingface/swift-transformers","repository_version":"1"}}}
        """#,
        #"""
        {"type":"onnx_genai_ios_pipette","flavor":"ios-arm64","packages":{
         "onnxruntime":{"repository_version":"v1.23.0"},
         "onnxruntime_genai":{"repository_url":"github.com/microsoft/onnxruntime-genai","repository_version":"v0.15.2"}}}
        """#,
    ])
    func aSourceRepositoryWithoutAUrlIsRefused(json: String) {
        #expect(throws: RuntimeIdentityError.missingRepositoryUrl) {
            try JSONDecoder().decode(Runtime.self, from: Data(json.utf8))
        }
    }

    /// The crate types the pin `NonEmptyString`; an empty one would submit a descriptor
    /// asserting a build it cannot name. The rejection comes from the type, so it fires
    /// wherever a `SourceRepository` is built, not only in this decoder.
    @Test func anEmptyPinIsRefused() {
        let json = Data(#"{"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"","flavor":"ios-arm64"}"#.utf8)

        #expect(throws: ModelError.emptyValue("")) {
            try JSONDecoder().decode(Runtime.self, from: json)
        }
    }

    /// Reduced as the crate's `strip_url_scheme` does, or this client and the CLI would
    /// disagree about which repository a descriptor names.
    @Test(arguments: [
        "https://github.com/ggml-org/llama.cpp",
        "http://github.com/ggml-org/llama.cpp/",
        "https://github.com/ggml-org/llama.cpp.git",
        "git@github.com:ggml-org/llama.cpp.git",
        "ssh://git@github.com/ggml-org/llama.cpp",
        "github.com/ggml-org/llama.cpp",
    ])
    func everyPastedFormNormalizesToOne(raw: String) {
        #expect(RepositoryUrl(raw).value == "github.com/ggml-org/llama.cpp")
    }

    @Test func orgRepoDropsTheHost() {
        #expect(RepositoryUrl("github.com/ggml-org/llama.cpp").orgRepo == "ggml-org/llama.cpp")
    }

    /// Refused at decode, so the claim path needs no flavor refusal of its own.
    @Test func anUnknownFlavorFailsToDecode() {
        let json = Data(#"{"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","version":"b77","flavor":"ios-x86"}"#.utf8)

        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(Runtime.self, from: json)
        }
    }

    // MARK: - Identity string: the crate's `Display`

    /// `{source}:{flavor}`, where the source keeps the repo as well as the ref, so a fork
    /// is not mistaken for upstream.
    @Test func aRuntimeIsNamedAsTheCrateDisplaysIt() throws {
        #expect(SourceRepository(repositoryUrl: RepositoryUrl(LlamaPins.main.repositoryUrl),
                                 repositoryVersion: try NonEmptyString("b9050")).description
            == "github.com/ggml-org/llama.cpp@b9050")
        #expect(try llama("b9050").description == "github.com/ggml-org/llama.cpp@b9050:ios-arm64")
        #expect(Runtime.appleFoundation(privateApis: .stock).description
            == "apple_foundation")
    }

    /// The MLX stack names each package as its `repo@version`, space-separated, in field
    /// order.
    @Test func anMlxStackNamesEachPackage() throws {
        let stack = MlxSwiftStack(
            mlxSwift: SourceRepository(repositoryUrl: RepositoryUrl("github.com/ml-explore/mlx-swift"),
                                       repositoryVersion: try NonEmptyString("0.31.3")),
            mlxSwiftLm: SourceRepository(repositoryUrl: RepositoryUrl("github.com/ml-explore/mlx-swift-lm"),
                                         repositoryVersion: try NonEmptyString("abc123def")),
            swiftTransformers: SourceRepository(
                repositoryUrl: RepositoryUrl("github.com/huggingface/swift-transformers"),
                repositoryVersion: try NonEmptyString("1.0.0")))

        #expect(stack.description
            == "mlx-swift=github.com/ml-explore/mlx-swift@0.31.3 "
            + "mlx-swift-lm=github.com/ml-explore/mlx-swift-lm@abc123def "
            + "swift-transformers=github.com/huggingface/swift-transformers@1.0.0")
        #expect(Runtime.mlxIosPipette(packages: stack, flavor: .iosArm64, privateApis: .stock)
            .description == "\(stack.description):ios-arm64")
    }

    // MARK: - This build

    /// A runtime type names this build only when it is one of the four a phone can be,
    /// and the model-keyed form answers with the same four identities.
    @Test func thisBuildAnswersForThePhoneRuntimesAlone() throws {
        let runnable = RuntimeType.allCases.filter { Runtime.thisBuild(for: $0) != nil }
        #expect(Set(runnable) == [.llamacppIosPipette, .mlxIosPipette, .onnxGenaiIosPipette,
                                  .appleFoundation])

        for model in [try ggufTextSpec("org/a-GGUF", "a.gguf"), try mlxSpec("org/m-MLX-4bit"),
                      try onnxGenaiSpec("org/m-onnx", subpath: "cpu-int4"),
                      .appleFoundationText] {
            let built = Runtime.thisBuild(for: model)
            #expect(Runtime.thisBuild(for: RuntimeType.of(built)) == built)
        }
    }

    // MARK: - Compatibility: the crate's `is_compatible`

    /// Each row is one accepted pair; everything else is refused, AFM included in both
    /// directions (`apple_foundation_compatibility`).
    @Test func eachModelPairsWithTheRuntimeThatLoadsItsFormat() throws {
        let text = try ggufTextSpec("org/a-GGUF", "a.gguf")
        let vision = try ggufVisionSpec("org/vl-GGUF", "vl.gguf", "mmproj.gguf")
        let mlx = try mlxSpec("org/m-MLX-4bit")
        let onnx = try onnxGenaiSpec("org/m-onnx", subpath: "cpu-int4")
        let llamaRuntime = Runtime.thisBuild(for: text)
        let mlxRuntime = Runtime.thisBuild(for: mlx)
        let onnxRuntime = Runtime.thisBuild(for: onnx)
        let afm = Runtime.appleFoundation(privateApis: .stock)

        #expect(isCompatible(text, llamaRuntime))
        #expect(isCompatible(vision, llamaRuntime))
        #expect(isCompatible(mlx, mlxRuntime))
        #expect(isCompatible(onnx, onnxRuntime))
        #expect(isCompatible(.appleFoundationText, afm))

        #expect(!isCompatible(text, mlxRuntime))
        #expect(!isCompatible(mlx, llamaRuntime))
        #expect(!isCompatible(.appleFoundationText, llamaRuntime))
        #expect(!isCompatible(text, afm))
        #expect(!isCompatible(mlx, afm))
        // The two directory-shaped formats are the pair worth pinning: both are a tree of
        // weights fetched from one repo, so a wrong row would look plausible.
        #expect(!isCompatible(onnx, mlxRuntime))
        #expect(!isCompatible(mlx, onnxRuntime))
        #expect(!isCompatible(text, onnxRuntime))
        #expect(!isCompatible(onnx, afm))
    }

    /// `isCompatible`'s `default` fails closed, so a missing row is silent. Driving from
    /// `ModelType.allCases` makes a new case fail here instead.
    @Test func everyModelTypeNamesTheEngineThatLoadsIt() throws {
        let llamaRuntime = Runtime.thisBuild(for: try ggufTextSpec("org/a-GGUF", "a.gguf"))
        let mlxRuntime = Runtime.thisBuild(for: try mlxSpec("org/m-MLX-4bit"))
        let afm = Runtime.appleFoundation(privateApis: .stock)

        for type in ModelType.allCases {
            // Skipped: the desktop-only formats have no `Model` case here, and ONNX GenAI's
            // pairing is pinned in `eachModelPairsWithTheRuntimeThatLoadsItsFormat`.
            let model: Model? = switch type {
            case .ggufText: try ggufTextSpec("org/a-GGUF", "a.gguf")
            case .ggufVision: try ggufVisionSpec("org/vl-GGUF", "vl.gguf", "mmproj.gguf")
            case .ggufVisionSpeculative:
                try ggufVisionSpeculativeSpec("org/vl-GGUF", "vl.gguf", "mmproj.gguf",
                                              draft: "draft.gguf")
            case .mlx: try mlxSpec("org/m-MLX-4bit")
            case .appleFoundationText: .appleFoundationText
            case .mlxSpeculative, .coreAi, .torch, .torchSpeculative, .openvino, .onnxGenai: nil
            }
            guard let model else { continue }
            let expected: Runtime = switch type {
            case .ggufText, .ggufVision, .ggufVisionSpeculative: llamaRuntime
            case .mlx: mlxRuntime
            case .appleFoundationText: afm
            case .mlxSpeculative, .coreAi, .torch, .torchSpeculative, .openvino, .onnxGenai:
                afm  // unreachable, filtered above
            }
            #expect(isCompatible(model, expected), "\(type.rawValue) pairs with its engine")
            for other in [llamaRuntime, mlxRuntime, afm] where other != expected {
                #expect(!isCompatible(model, other),
                        "\(type.rawValue) must not pair with \(RuntimeType.of(other).rawValue)")
            }
        }
    }

    /// A merged drafter pairs the same way: whether *that* arrangement can be attached is
    /// the engine's call at load, not the table's.
    @Test func aMergedDrafterPairsWithTheLlamaEngineToo() throws {
        let merged = try ggufVisionSpeculativeSpec("org/vl-GGUF", "vl.gguf", "mmproj.gguf")
        #expect(isCompatible(merged, Runtime.thisBuild(for: merged)))
    }

    /// The headless selectors hold a runtime type; they reach the same table through the
    /// build this binary is, and a type this build cannot be pairs with nothing.
    @Test func aRuntimeTypeReachesTheSameTable() throws {
        let text = try ggufTextSpec("org/a-GGUF", "a.gguf")
        #expect(isCompatible(text, RuntimeType.llamacppIosPipette))
        #expect(!isCompatible(text, RuntimeType.mlxIosPipette))
        #expect(!isCompatible(text, RuntimeType.llamacppCliStockTools))
    }

    /// One spelling per engine: a model, a runtime type and the wizard's selector all read
    /// `RuntimeKind.label`.
    @Test func anEngineIsSpelledOneWay() throws {
        for model in [try ggufTextSpec("org/a-GGUF", "a.gguf"), try mlxSpec("org/m-MLX-4bit"),
                      try onnxGenaiSpec("org/m-onnx", subpath: "cpu-int4"),
                      .appleFoundationText] {
            let type = RuntimeType.of(Runtime.thisBuild(for: model))
            #expect(model.engineLabel == RuntimeKind(model).label)
            #expect(type.engineLabel == model.engineLabel)
        }
        #expect(RuntimeType.dockerVllm.engineLabel == "docker_vllm")
    }

    // MARK: - Completeness against the crate
    //
    // The axis spells every plan-types variant, so a recognized one this client cannot run
    // is refused by variant instead of failing to decode.

    @Test(arguments: ["llamacpp_cli_stock_tools", "llamacpp_apk_pipette",
                      "llamacpp_ios_pipette", "mlx_macos_pipette", "mlx_ios_pipette",
                      "docker_vllm", "docker_sglang", "uv_vllm", "uv_sglang",
                      "uv_openvino", "uv_onnx_genai", "ovms", "trt_edgellm",
                      "onnx_genai_ios_pipette", "apple_foundation"])
    func runtimeTypeSpellsEveryCrateVariant(raw: String) {
        #expect(RuntimeType(rawValue: raw) != nil)
    }

    /// A spelling neither side knows still fails, which is what makes the above a
    /// completeness claim rather than a permissive decoder.
    @Test func anInventedRuntimeTypeStillFails() {
        #expect(RuntimeType(rawValue: "uv_banana") == nil)
    }
}
