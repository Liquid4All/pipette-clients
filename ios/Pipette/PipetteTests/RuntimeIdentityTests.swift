import Foundation
import Testing

@testable import Pipette

/// `Runtime` as build identity: what a submitted `runtime_descriptor` says this binary
/// is, and what a claim is allowed to declare.
///
/// This encoder replaced `SubmissionRef.RuntimeRef`, a hand-written mirror, so the wire
/// shape is pinned here rather than left to the type's synthesis.
struct RuntimeIdentityTests {
    private func object(_ runtime: Runtime) throws -> [String: Any] {
        let json = try SubmissionRef.runtime(runtime)
        return try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    /// `SourceRepository` is flattened into the runtime, not nested — the crate's
    /// `#[serde(flatten)]`.
    @Test func theLlamaCppDescriptorFlattensItsSourceRepository() throws {
        let json = try object(.llamacppIosPipette(
            source: SourceRepository(repositoryVersion: try NonEmptyString("b77")),
            flavor: .iosArm64, privateThermal: false))

        #expect(json["type"] as? String == "llamacpp_ios_pipette")
        #expect(json["repository_url"] as? String == "github.com/ggml-org/llama.cpp")
        #expect(json["repository_version"] as? String == "b77")
        #expect(json["flavor"] as? String == "ios-arm64")
        #expect(json["packages"] == nil)
    }

    /// MLX pins a stack rather than one repo, so its version is three named repos.
    @Test func theMlxDescriptorCarriesTheSwiftStack() throws {
        let json = try object(.mlxIosPipette(packages: .thisBuild, flavor: .iosArm64, privateThermal: false))
        let packages = try #require(json["packages"] as? [String: Any])

        #expect(json["type"] as? String == "mlx_ios_pipette")
        for key in ["mlx_swift", "mlx_swift_lm", "swift_transformers"] {
            let repo = try #require(packages[key] as? [String: Any], "\(key) missing")
            #expect((repo["repository_url"] as? String)?.isEmpty == false)
            #expect((repo["repository_version"] as? String)?.isEmpty == false)
        }
    }

    /// AFM ships with the OS: no repo, no ref, so the descriptor is the tag alone.
    @Test func appleFoundationIsABareTag() throws {
        let json = try object(.appleFoundation(privateThermal: false))

        #expect(json["type"] as? String == "apple_foundation")
        #expect(json.count == 1)
    }

    @Test func aDescriptorRoundTrips() throws {
        let runtime = Runtime.llamacppIosPipette(
            source: SourceRepository(repositoryUrl: RepositoryUrl("github.com/fork/llama.cpp"),
                                     repositoryVersion: try NonEmptyString("b91")),
            flavor: .iosArm64, privateThermal: false)
        let restored = try JSONDecoder().decode(
            Runtime.self, from: JSONEncoder().encode(runtime))

        #expect(restored == runtime)
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

    /// `version` for `repository_version`, as the crate's serde alias accepts, and the
    /// URL defaults the way `default_repository_url` does.
    @Test func theVersionAliasAndUrlDefaultFollowTheCrate() throws {
        let json = Data(#"{"type":"llamacpp_ios_pipette","version":"b77","flavor":"ios-arm64"}"#.utf8)
        let runtime = try JSONDecoder().decode(Runtime.self, from: json)

        guard case let .llamacppIosPipette(source, _, _) = runtime else {
            Issue.record("expected llamacppIosPipette"); return
        }
        #expect(source.repositoryVersion.value == "b77")
        #expect(source.repositoryUrl.value == "github.com/ggml-org/llama.cpp")
    }

    /// The crate types the pin `NonEmptyString`; an empty one would submit a descriptor
    /// asserting a build it cannot name. The rejection comes from the type, so it fires
    /// wherever a `SourceRepository` is built, not only in this decoder.
    @Test func anEmptyPinIsRefused() {
        let json = Data(#"{"type":"llamacpp_ios_pipette","repository_version":"","flavor":"ios-arm64"}"#.utf8)

        #expect(throws: ModelError.emptyValue("")) {
            try JSONDecoder().decode(Runtime.self, from: json)
        }
    }

    /// Any pasted coordinate reduces to `<host>/<org>/<repo>`, as the crate's
    /// `strip_url_scheme` does — otherwise this client and the CLI would disagree about
    /// which repository a descriptor names.
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

    /// A flavor this build does not have fails to decode, which is why the claim path no
    /// longer carries an `invalidFlavor` refusal of its own.
    @Test func anUnknownFlavorFailsToDecode() {
        let json = Data(#"{"type":"llamacpp_ios_pipette","version":"b77","flavor":"ios-x86"}"#.utf8)

        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(Runtime.self, from: json)
        }
    }
}
