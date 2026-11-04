import Foundation
import Testing

@testable import Pipette

/// A syntactically valid 64-hex digest for the `sha256` value slots.
private let sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

/// `parseModelUri`, `modelToUri` and `parseModelArg`: the crate's `model_uri.rs`. The property that
/// matters most is the round trip: a rendered URI that does not parse back to the model
/// it came from would print a reference nothing can use.
struct ModelUriTests {

    // MARK: - Parse

    @Test func aTextHfUriNeedsOnlyRepoAndPath() throws {
        let model = try parseModelUri("gguf-text://repo=org/repo&path=Q4_K_M.gguf")
        guard case let .ggufText(m) = model, case let .huggingFace(repo, path, sha256) = m.source else {
            Issue.record("expected gguf-text HuggingFace, got \(model)"); return
        }
        #expect(path.value == "Q4_K_M.gguf")
        #expect(sha256 == nil)
        #expect(repo.revision == nil)
        #expect(model.description == "org/repo:Q4_K_M.gguf")
    }

    @Test func aTextHfUriCarriesARevisionAndADigest() throws {
        let model = try parseModelUri("gguf-text://repo=org/repo&path=Q4_K_M.gguf&rev=v2&sha256=\(sha)")
        guard case let .ggufText(m) = model, case let .huggingFace(repo, _, sha256) = m.source else {
            Issue.record("expected gguf-text HuggingFace, got \(model)"); return
        }
        #expect(repo.revision?.value == "v2")
        #expect(sha256?.value == sha)
    }

    @Test func aTextUrlUriIsADirectDownload() throws {
        for uri in ["gguf-text://url=https://cdn.example/m.gguf",
                    "gguf-text://url=https://cdn.example/m.gguf&sha256=\(sha)"] {
            let model = try parseModelUri(uri)
            guard case let .ggufText(m) = model, case let .url(url, _) = m.source else {
                Issue.record("expected gguf-text Url, got \(model)"); continue
            }
            #expect(url.value == "https://cdn.example/m.gguf")
        }
    }

    @Test func aVisionHfUriNamesBothFiles() throws {
        let model = try parseModelUri("gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf")
        #expect(model.description == "org/repo:a.gguf+org/repo:mm.gguf")

        let full = try parseModelUri(
            "gguf-vision://repo=org/repo&model=a.gguf&mmproj=mm.gguf&rev=v2"
                + "&model_sha256=\(sha)&mmproj_sha256=\(sha)")
        guard case let .ggufVision(m) = full,
              case let .huggingFace(repo, _, modelSha, _, mmprojSha) = m.source else {
            Issue.record("expected gguf-vision HuggingFace, got \(full)"); return
        }
        #expect(repo.revision != nil)
        #expect(modelSha != nil)
        #expect(mmprojSha != nil)
    }

    @Test func aVisionUriWithoutARepoNamesTwoUrls() throws {
        let model = try parseModelUri("gguf-vision://model=https://x/a.gguf&mmproj=https://x/mm.gguf")
        guard case let .ggufVision(m) = model, case .url = m.source else {
            Issue.record("expected gguf-vision Url, got \(model)"); return
        }
        #expect(model.description == "https://x/a.gguf+https://x/mm.gguf")
    }

    /// `dir_scheme_parses_to_variant_and_identity`, the `mlx` rows: the revision pins the
    /// repo (`@v2`) and the prefix names the subdirectory (`:4bit`).
    @Test(arguments: [("mlx://repo=org/repo", "org/repo"),
                      ("mlx://repo=org/repo&prefix=4bit&rev=v2", "org/repo@v2:4bit")])
    func anMlxUriFoldsPrefixAndRevisionIntoItsIdentity(uri: String, identity: String) throws {
        let model = try parseModelUri(uri)
        #expect(ModelType.of(model) == .mlx)
        #expect(model.description == identity)
    }

    /// The same directory grammar under the `onnx-genai` scheme, where the prefix is
    /// load-bearing: it is what selects one package out of a repo that bundles several.
    @Test(arguments: [("onnx-genai://repo=org/repo", "org/repo"),
                      ("onnx-genai://repo=org/repo&prefix=cpu_and_mobile/cpu-int4&rev=v2",
                       "org/repo@v2:cpu_and_mobile/cpu-int4")])
    func anOnnxGenaiUriFoldsPrefixAndRevisionIntoItsIdentity(uri: String,
                                                             identity: String) throws {
        let model = try parseModelUri(uri)
        #expect(ModelType.of(model) == .onnxGenai)
        #expect(model.description == identity)
    }

    /// `dispatch_json_object` / `dispatch_uri`: a `{`-leading argument is the JSON form,
    /// anything else the URI, and the surrounding whitespace is trimmed either way.
    @Test(arguments: [
        #"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"Q4_K_M.gguf"}"#,
        #"  {"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"Q4_K_M.gguf"}  "#,
    ])
    func aBraceLeadingArgumentIsTheJsonForm(arg: String) throws {
        guard case .ggufText = try parseModelArg(arg) else { Issue.record("expected gguf-text"); return }
    }

    @Test(arguments: ["mlx://repo=o/r", "   mlx://repo=o/r   "])
    func anyOtherArgumentIsTheUri(arg: String) throws {
        guard case .mlx = try parseModelArg(arg) else { Issue.record("expected mlx"); return }
    }

    // MARK: - Refused

    @Test func anUnknownSchemeIsNamedBeforeItsKeys() {
        #expect(throws: ModelUriError.unknownScheme("foo")) { try parseModelUri("foo://repo=o/r") }
        #expect(throws: ModelUriError.unknownScheme("banana")) { try parseModelUri("banana://url=x") }
    }

    /// Real sources that are not importable are told apart from a typo, a `file://` value
    /// included, since the scheme is what precedes the first `://`.
    @Test(arguments: [("apple-foundation://", "apple-foundation"), ("local://dir=/x", "local"),
                      ("file:///models/m.gguf", "file")])
    func aSourceThatIsNotImportableIsRefusedAsSuch(uri: String, source: String) {
        #expect(throws: ModelUriError.notImportable(source)) { try parseModelUri(uri) }
    }

    /// `torch` and `openvino` are the crate's schemes, but name kinds this build has no
    /// `Model` case for, so they are refused by name rather than as unknown.
    @Test(arguments: ["torch", "openvino"])
    func aKindWithNoEngineHereIsRefusedByName(scheme: String) {
        #expect(throws: ModelUriError.unsupportedScheme(scheme)) {
            try parseModelUri("\(scheme)://repo=org/repo")
        }
    }

    /// `missing_scheme_separator`: `Display` (`{repo}[:{path}]`) is not an input spelling.
    @Test(arguments: ["mlx:repo=o/r", "just-text", "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_K_M.gguf"])
    func aStringWithoutTheSeparatorIsNotAUri(uri: String) {
        #expect(throws: ModelUriError.missingSchemeSeparator) { try parseModelUri(uri) }
    }

    @Test(arguments: [("gguf-text://repo=o/r", "gguf-text", "path"),
                      ("gguf-vision://repo=o/r&model=a.gguf", "gguf-vision", "mmproj"),
                      ("mlx://", "mlx", "repo")])
    func aMissingRequiredKeyIsNamed(uri: String, scheme: String, key: String) {
        #expect(throws: ModelUriError.missingKey(scheme: scheme, key: key)) { try parseModelUri(uri) }
    }

    /// `gguf_text_repo_url_exclusive`: exactly one of the two sources.
    @Test(arguments: ["gguf-text://sha256=\(sha)", "gguf-text://repo=o/r&url=https://x"])
    func aTextUriNamesExactlyOneSource(uri: String) {
        #expect(throws: ModelUriError.mutuallyExclusive(scheme: "gguf-text",
                                                        expected: "`repo`+`path` or `url`")) {
            try parseModelUri(uri)
        }
    }

    @Test func aKeyTwiceIsADuplicate() {
        #expect(throws: ModelUriError.duplicateKey("repo")) { try parseModelUri("mlx://repo=a/b&repo=c/d") }
    }

    /// Unknown whether the key is real elsewhere (`path` on an `mlx://`), exists nowhere,
    /// or sits beside a URL source.
    @Test(arguments: [("mlx://repo=o/r&foo=bar", "mlx", "foo"),
                      ("mlx://repo=o/r&path=x", "mlx", "path"),
                      ("gguf-text://url=https://x/m.gguf&foo=bar", "gguf-text", "foo")])
    func aKeyTheSchemeDoesNotClaimIsUnknown(uri: String, scheme: String, key: String) {
        #expect(throws: ModelUriError.unknownKey(scheme: scheme, key: key)) { try parseModelUri(uri) }
    }

    /// A segment with no `=` is a malformed pair, not a key, a trailing `&` included.
    @Test(arguments: [("mlx://repo", "repo"), ("mlx://repo=org/r&garbage", "garbage"),
                      ("mlx://repo=org/r&", "")])
    func aSegmentWithoutAnEqualsIsAMalformedPair(uri: String, pair: String) {
        #expect(throws: ModelUriError.malformedPair(pair)) { try parseModelUri(uri) }
    }

    @Test func anEmptyKeyIsRefused() {
        #expect(throws: ModelUriError.emptyKey) { try parseModelUri("mlx://=abc") }
    }

    /// `query_url_rejected`: the second case is the silent-truncation trap, where the
    /// `&sha256=` would fracture off as a real digest for a cut URL.
    @Test(arguments: [("gguf-text://url=https://cdn.example/m.gguf?sig=a&exp=b", "url"),
                      ("gguf-text://url=https://host/m.gguf?token=x&sha256=\(sha)", "url"),
                      ("gguf-vision://model=https://x/a?a=1&b=2&mmproj=https://x/mm.gguf", "model")])
    func aUrlWithAQueryStringIsRefused(uri: String, key: String) {
        #expect(throws: ModelUriError.queryUrl(key: key)) { try parseModelUri(uri) }
    }

    /// One bad value per primitive, reported against its key.
    @Test(arguments: [("mlx://repo=noslash", "repo"),
                      ("gguf-text://url=https://x&sha256=nothex", "sha256"),
                      ("mlx://repo=o/r&prefix=../escape", "prefix"),
                      ("mlx://repo=o/r&rev=-bad", "rev")])
    func anInvalidValueNamesItsKey(uri: String, key: String) {
        do {
            _ = try parseModelUri(uri)
            Issue.record("expected `\(uri)` to be refused")
        } catch let .invalidValue(got, _) {
            #expect(got == key)
        } catch {
            Issue.record("expected invalidValue, got \(error)")
        }
    }

    /// The refusals read as the crate's do, so a message copied from either client finds
    /// the other's cause.
    @Test func theRefusalsSpeakTheCratesWords() {
        #expect(ModelUriError.missingSchemeSeparator.localizedDescription
            == "model URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`")
        #expect(ModelUriError.unknownKey(scheme: "mlx", key: "foo").localizedDescription
            == "unknown key `foo` for scheme `mlx`")
        #expect(ModelUriError.missingKey(scheme: "gguf-text", key: "path").localizedDescription
            == "missing required key `path` for scheme `gguf-text`")
        #expect(ModelUriError.notImportable("local").localizedDescription
            == "model source `local` is not importable")
    }

    // MARK: - Render

    /// `round_trips`, for the schemes this client has: parse → render is byte-identical on a
    /// canonical URI, and re-parse recovers the model.
    @Test(arguments: [
        "gguf-text://repo=org/repo&path=Q4_K_M.gguf",
        "gguf-text://repo=org/repo&path=Q4_K_M.gguf&rev=v2&sha256=\(sha)",
        "gguf-text://url=https://cdn.example/m.gguf",
        "gguf-text://url=https://cdn.example/m.gguf&sha256=\(sha)",
        "gguf-vision://repo=org/repo&model=m.gguf&mmproj=p.gguf",
        "gguf-vision://model=https://x/m.gguf&mmproj=https://x/p.gguf",
        "gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf",
        "gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf&draft=d.gguf",
        "gguf-vision-speculative://repo=org/repo&model=m.gguf&mmproj=p.gguf&draft=d.gguf"
            + "&rev=v2&draft_sha256=\(sha)",
        "gguf-vision-speculative://model=https://x/m.gguf&mmproj=https://x/p.gguf",
        "gguf-vision-speculative://model=https://x/m.gguf&mmproj=https://x/p.gguf"
            + "&draft=https://x/d.gguf",
        "mlx://repo=org/repo",
        "mlx://repo=org/repo&prefix=4bit&rev=v2",
        "onnx-genai://repo=org/repo",
        "onnx-genai://repo=org/repo&prefix=cpu_and_mobile/cpu-int4&rev=v2",
    ])
    func aCanonicalUriRoundTripsByteForByte(uri: String) throws {
        let model = try parseModelUri(uri)
        let rendered = try modelToUri(model)

        #expect(rendered == uri)
        #expect(try parseModelUri(rendered) == model)
    }

    @Test func everyImportableArmRoundTrips() throws {
        let models: [Model] = [
            try ggufTextSpec("LiquidAI/LFM2-350M-GGUF", "LFM2-350M-Q4_K_M.gguf"),
            try ggufVisionSpec("LiquidAI/LFM2.5-VL-450M-GGUF", "vl-Q4_0.gguf", "mmproj-f16.gguf"),
            try mlxSpec("mlx-community/LFM2-350M-4bit"),
            try onnxGenaiSpec("microsoft/Phi-3-mini-4k-instruct-onnx",
                              subpath: "cpu_and_mobile/cpu-int4-rtn-block-32"),
        ]

        for model in models {
            #expect(try parseModelUri(try modelToUri(model)) == model, "\(model)")
        }
    }

    @Test func aPinnedRevisionSurvives() throws {
        var repo = try HfRepo.parseOrgRepo("LiquidAI/LFM2-350M-GGUF")
        repo.revision = try HfRevision("v1.2.3")
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("m-Q4_0.gguf"), sha256: nil)))

        let uri = try modelToUri(model)
        #expect(uri.contains("rev=v1.2.3"))
        #expect(try parseModelUri(uri) == model)
    }

    /// One model renders one string: `model` before `mmproj`, the order
    /// `gguf_vision_to_uri` writes them, so both clients render it identically.
    @Test func renderingIsDeterministic() throws {
        let model = try ggufVisionSpec("org/vl-GGUF", "vl.gguf", "proj.gguf")

        #expect(try modelToUri(model) == "gguf-vision://repo=org/vl-GGUF&model=vl.gguf&mmproj=proj.gguf")
    }

    /// `render_rejects_a_repo_with_an_auth_token`: a URI cannot hold a secret, and dropping
    /// it silently would print a reference that fetches without the credential.
    @Test func aRepoCarryingATokenHasNoUriForm() throws {
        var repo = try HfRepo.parseOrgRepo("o/r")
        repo.authToken = try AuthToken("hf_secret")
        let model = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("Q4_K_M.gguf"), sha256: nil)))

        #expect(throws: ModelUriError.secretNotRepresentable) { try modelToUri(model) }
        #expect(try modelToUri(model.withoutAuthToken) == "gguf-text://repo=o/r&path=Q4_K_M.gguf")
    }

    /// The crate's `VaultNotRepresentable`: the grammar has no key for a principal.
    @Test func aVaultSourceHasNoUriForm() throws {
        let archive = try vaultArchiveSpec("vault_1")
        for model in [try vaultTextSpec("vault_1"), try vaultVisionSpec("vault_1", "vault_2"), archive] {
            #expect(throws: ModelUriError.vaultNotRepresentable) { try modelToUri(model) }
        }
    }

    /// `render_rejects_apple_foundation`, `…_a_local_source`, `…_a_query_url`: what no URI
    /// can name is refused with the parser's own errors.
    @Test func whatNoUriCanNameIsRefused() throws {
        let stored = try #require(try ggufTextSpec("org/r", "m.gguf").toStored(base: "blobs"))
        let queryUrl = Model.ggufText(GgufText(source: .url(
            url: try ResourceUrl("https://x/m.gguf?token=y"), sha256: nil)))

        #expect(throws: ModelUriError.notImportable("local")) { try modelToUri(stored) }
        #expect(throws: ModelUriError.notImportable("apple-foundation")) {
            try modelToUri(.appleFoundationText)
        }
        #expect(throws: ModelUriError.queryUrl(key: "url")) { try modelToUri(queryUrl) }
    }
}
