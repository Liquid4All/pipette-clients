import Foundation
import Testing

@testable import Pipette

/// The validated scalars themselves, tested directly rather than through a coordinate
/// that happens to build one. `HfModelRefTests` covers what `HfModelRef.parse` does
/// with them.
struct PrimitivesTests {

    /// Decode re-runs validation, so an invalid value cannot enter through the back door
    /// that construction closes.
    @Test func encodesAsBareSlugString() throws {
        let org = try HfOrg("LiquidAI")
        let data = try JSONEncoder().encode(org)
        #expect(String(decoding: data, as: UTF8.self) == "\"LiquidAI\"")
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(HfOrg.self, from: Data("\".bad\"".utf8))
        }
    }

    /// `.` and `..` match the segment character class, so the exclusion is by name and
    /// a reader could mistake it for redundant. Asserted on the validator directly, not
    /// only through `HfModelRef.parse`, so the rule survives a caller changing.
    @Test(arguments: ["..", "../escape", "dir/../seg", "./seg", "dir/."])
    func repoSubpathRejectsADotSegment(_ candidate: String) {
        #expect(!RepoSubpath.validate(candidate))
    }

    @Test(arguments: ["4bit", "variants/4bit", "a.b-c_d"])
    func repoSubpathAcceptsOrdinarySegments(_ candidate: String) {
        #expect(RepoSubpath.validate(candidate))
    }

    /// `/` is allowed, as in the crate's regex, so a pin like `refs/pr/1` is representable.
    @Test(arguments: ["main", "v1.2.3", "refs/pr/1", "a1b2c3d"])
    func revisionAcceptsTagsBranchesAndRefs(_ candidate: String) {
        #expect(HfRevision.validate(candidate))
    }

    @Test(arguments: ["", "-leading", "/leading", "has space"])
    func revisionRejectsMalformed(_ candidate: String) {
        #expect(!HfRevision.validate(candidate))
    }

    // The path pairs below carry the crate's own case tables for `is_relative_fs_path`
    // and `is_absolute_path`. They are the two validators whose verdicts move files: a
    // relative path addresses a store entry, and an absolute one is what a sweep may
    // delete, so a rule that drifts from the crate's either strands a model the CLI can
    // read or rejects a path it writes.

    /// `blobs/a..b.gguf` guards the boundary between the two rules: `..` is rejected as a
    /// whole *segment*, not wherever it appears in a filename.
    @Test(arguments: [
        "models/m", "entry/blobs/Q4.gguf", "weights.safetensors", "blobs", "blobs/a..b.gguf",
    ])
    func relativePathAcceptsAStoreCoordinate(_ candidate: String) {
        #expect(RelativePath.validate(candidate))
    }

    @Test(arguments: [
        "/models/m", "C:/models/m", #"C:\models\m"#, #"\\server\share\m"#, "//server/share",
        "~/models/m", "../sibling/m", "a/../b", "a/./b", "", "a//b", #"a\..\b"#, #"a\\b"#, #"a\"#,
    ])
    func relativePathRejectsAbsoluteOrTraversing(_ candidate: String) {
        #expect(!RelativePath.validate(candidate))
    }

    /// A leading `./` is stripped before validation, so `./local/m` is the same
    /// coordinate as `local/m` rather than a second spelling of it.
    @Test func relativePathNormalizesALeadingCurrentDir() throws {
        #expect(try RelativePath("./local/m").value == "local/m")
        #expect(try RelativePath("local/m").value == "local/m")
    }

    @Test(arguments: [
        "/models/m", "C:/models/m", #"C:\models\m"#, #"\\server\share\m"#, "//server/share/m",
        "/",
    ])
    func absolutePathAcceptsEveryHostForm(_ candidate: String) {
        #expect(AbsolutePath.validate(candidate))
    }

    @Test(arguments: [
        "models/m", "~/models/m", "/models/../x", "", "C:models/m", "/a//b", #"/a\..\b"#,
        #"\\\share\m"#,
    ])
    func absolutePathRejectsRelativeOrTraversing(_ candidate: String) {
        #expect(!AbsolutePath.validate(candidate))
    }

    /// Sanitizing before validating, the crate's order, makes `./models/m` a rejection
    /// rather than an accidental accept.
    @Test func absolutePathRejectsWhatSanitizingMakesRelative() {
        #expect(throws: ModelError.invalidAbsolutePath("models/m")) {
            _ = try AbsolutePath("./models/m")
        }
    }

    @Test(arguments: [
        ("https://example.com/a.gguf", true), ("http://example.com/a.gguf", true),
        ("file:///models/a.gguf", true), ("ftp://example.com/a.gguf", false),
        ("example.com/a.gguf", false), ("", false), ("https://", false),
    ])
    func resourceUrlAcceptsOnlyFetchableSchemes(_ candidate: String, _ valid: Bool) {
        #expect(ResourceUrl.validate(candidate) == valid)
    }

    @Test func sha256AcceptsOnlyLowercaseHexOfExactlySixtyFour() {
        let valid = String(repeating: "a1b2c3d4", count: 8)
        #expect(valid.count == 64 && Sha256.validate(valid))
        #expect(!Sha256.validate(String(repeating: "A1B2C3D4", count: 8)), "uppercase")
        #expect(!Sha256.validate(String(repeating: "ab", count: 31)), "62 chars")
        #expect(!Sha256.validate(String(repeating: "ab", count: 33)), "66 chars")
    }

    /// The token must not reach a log or an error, so neither rendering exposes it and
    /// the failure case carries no payload.
    @Test func authTokenNeverRendersItsSecret() throws {
        let token = try AuthToken("hf_averysecretvalue")
        #expect(token.description == "<redacted>")
        #expect(token.debugDescription == "AuthToken(<redacted>)")
        #expect("\(token)" == "<redacted>")
        #expect(!"\(token)".contains("secret"))
        #expect(token.value == "hf_averysecretvalue", "`value` stays the one door")
        #expect(throws: ModelError.invalidAuthToken) { _ = try AuthToken("") }
    }

    /// A revision is identity; a credential is not. Two references to the same weights
    /// must compare equal whether or not one carries a token, or a claim that ships one
    /// would fail to match the model already on disk.
    @Test func repoIdentityCountsTheRevisionAndIgnoresTheToken() throws {
        let bare = try HfRepo.parseOrgRepo("org/repo")
        var tokened = bare
        tokened.authToken = try AuthToken("hf_secret")
        var pinned = bare
        pinned.revision = try HfRevision("v1.2.3")

        #expect(bare == tokened)
        #expect(bare.hashValue == tokened.hashValue)
        #expect(bare != pinned)
        #expect(bare.reference == "org/repo")
        #expect(pinned.reference == "org/repo@v1.2.3")
    }

    /// An absent revision or token is omitted, not written as `null`, matching the
    /// crate's `skip_serializing_if`.
    @Test func repoOmitsAbsentRevisionAndToken() throws {
        let json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(try HfRepo.parseOrgRepo("org/repo"))) as? [String: Any]
        let keys = Set(json?.keys ?? [:].keys)
        #expect(keys == ["org", "repo_name"])
    }

    /// Redaction drops the credential and keeps everything that is identity, so a
    /// redacted coordinate still addresses the same storage entry.
    @Test func redactionDropsOnlyTheToken() throws {
        var repo = try HfRepo.parseOrgRepo("org/repo")
        repo.revision = try HfRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")

        let redacted = repo.withoutAuthToken
        #expect(redacted.authToken == nil)
        #expect(redacted.reference == "org/repo@v1.2.3")
        #expect(redacted == repo)  // identity ignores the token, so this must still hold
    }

    /// `withoutAuthToken` reaches the repo inside every file-backed `Model` case. A
    /// case added without a redacting arm would leak, and nothing else would fail.
    @Test func modelRedactionReachesEveryCase() throws {
        var repo = try HfRepo.parseOrgRepo("org/repo")
        repo.authToken = try AuthToken("hf_secret")
        let models: [Model] = [
            .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("m-Q4_0.gguf"), sha256: nil))),
            .ggufVision(.init(source: .huggingFace(repo: repo, model: try RepoSubpath("m-Q4_0.gguf"), modelSha256: nil, mmproj: try RepoSubpath("mmproj-f16.gguf"), mmprojSha256: nil))),
            .mlx(.init(source: .huggingFace(repo: repo, prefix: nil))),
            .appleFoundationText,
        ]
        for model in models {
            #expect(model.withoutAuthToken.repo?.authToken == nil)
        }
        // Guards the premise: these specs really did carry a token before redaction.
        #expect(models.compactMap(\.repo?.authToken).count == 3)
    }

    // MARK: - HfRepo.parseOrgRepo

    /// The crate's `parse_org_repo`: a nested `/` is refused as a repo name, not as a
    /// missing separator.
    @Test func parseOrgRepoSplitsOnTheFirstSlash() throws {
        let repo = try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-GGUF")
        #expect(repo.org.value == "LiquidAI")
        #expect(repo.repoName.value == "LFM2.5-350M-GGUF")
        #expect(throws: ModelError.invalidRepoName("b/c")) { try HfRepo.parseOrgRepo("a/b/c") }
        #expect(throws: ModelError.repoMissingSeparator("noslash")) {
            try HfRepo.parseOrgRepo("noslash")
        }
        #expect(ModelError.repoMissingSeparator("x").localizedDescription
            == "HF repo must be in the form `org/repo_name`")
    }

    // MARK: - BenchmarkId

    /// An id, never an address: a catalog side and whitespace are refused, and so is
    /// anything outside the server's grammar, which it interpolates into its keys.
    @Test(arguments: [("prefill_throughput_256", true), ("eval_ifbench_2026.06.1", true),
                      ("vl_max_memory_224x224_t0_f1", true), ("", false),
                      ("local/eval_smoke", false), ("remote/x", false), ("has space", false),
                      ("tab\there", false), (".hidden", false), ("-flag", false),
                      ("a:b", false), (".", false), ("..", false)])
    func benchmarkIdIsABareId(_ candidate: String, _ valid: Bool) {
        #expect(BenchmarkId.validate(candidate) == valid)
    }

    /// The crate's `ClientId`: trimmed, then held to the server's `[A-Za-z0-9_-]`.
    @Test(arguments: [("ev1_0a1b2c", true), ("macbook-pro-2", true), ("  ev1_a  ", true),
                      ("", false), ("   ", false), ("a.b", false), ("a/b", false),
                      ("ev1:a", false), ("\u{200B}ev1_a", false), ("ev1_a\nb", false)])
    func clientIdHoldsTheServerCharset(_ candidate: String, _ valid: Bool) {
        #expect(((try? ClientId(candidate)) != nil) == valid)
    }

    /// The refused value is shown escaped, so a server-sent id cannot forge a log line
    /// or hide a zero-width character.
    @Test func aRefusedIdIsRenderedEscaped() {
        for refusal in [ModelError.invalidClientId("ev1_a\nb\u{200B}"),
                        .invalidBenchmarkId("ev1_a\nb\u{200B}")] {
            let message = refusal.errorDescription ?? ""
            #expect(!message.contains("\n"))
            #expect(message.contains(#""ev1_a\nb\u{200B}""#))
        }
    }

    /// Validated on decode too, so a claim naming a catalog address is refused at the
    /// boundary.
    @Test func aBenchmarkIdIsValidatedOnDecode() {
        #expect(throws: ModelError.invalidBenchmarkId("local/eval_smoke")) {
            _ = try JSONDecoder().decode(BenchmarkId.self, from: Data(#""local/eval_smoke""#.utf8))
        }
    }

    /// The crate's `NonEmptyTrimmedString`: trimmed as Rust's `str::trim` trims, refused
    /// when nothing is left, a bare string on the wire, and held to both on decode.
    @Test func aNonEmptyTrimmedStringTrimsAndRefusesBlank() throws {
        #expect(try NonEmptyTrimmedString("  llama-3.2-1b\t").value == "llama-3.2-1b")
        #expect(try NonEmptyTrimmedString("\u{00A0}acme LFM3 q4\n").value == "acme LFM3 q4")
        // Rust's `trim` leaves a zero-width space, so the crate accepts one alone.
        #expect(try NonEmptyTrimmedString("\u{200B}").value == "\u{200B}")
        for blank in ["", "   \t\n  "] {
            #expect(throws: ModelError.emptyValue("")) { _ = try NonEmptyTrimmedString(blank) }
        }

        let padded = try JSONDecoder().decode(
            NonEmptyTrimmedString.self, from: Data(#""  acme LFM3 q4 ""#.utf8))
        #expect(String(decoding: try JSONEncoder().encode(padded), as: UTF8.self)
            == #""acme LFM3 q4""#)
        #expect(throws: ModelError.emptyValue("")) {
            _ = try JSONDecoder().decode(NonEmptyTrimmedString.self, from: Data(#""  ""#.utf8))
        }
    }

    // MARK: - CapabilityFlag

    /// `capability_flag_canonical_form`: non-empty, no whitespace, already lowercase.
    @Test(arguments: [("os:ios", true), ("os_version:26.1", true),
                      ("runtime:llama_cpp:b9999", true), ("ram_bytes:17179869184", true),
                      ("job_retry", true), ("device:iphone17pro", true), ("", false),
                      ("os:iOS", false), ("OS:ios", false), ("os: ios", false),
                      ("os:ios ", false), (" os:ios", false), ("os:i\tos", false)])
    func capabilityFlagIsCanonical(_ candidate: String, _ valid: Bool) {
        #expect(CapabilityFlag.validate(candidate) == valid)
    }

    /// `positive_bounds`, the count cases: past the max a value is refused rather than
    /// wrapped, since a count lands in an `Int32` warehouse column.
    @Test(arguments: [
        ("1", true, true), ("2147483647", true, true), ("0", false, true),
        ("-1", false, false), ("2147483648", false, false),
    ])
    func positiveBounds(json: String, positive: Bool, nonNegative: Bool) {
        let data = Data(json.utf8)
        #expect(((try? JSONDecoder().decode(PositiveCount.self, from: data)) != nil) == positive)
        #expect(((try? JSONDecoder().decode(NonNegativeCount.self, from: data)) != nil)
            == nonNegative)
    }
}
