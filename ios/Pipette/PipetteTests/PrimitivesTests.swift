import Foundation
import Testing

@testable import Pipette

/// The validated scalars themselves, tested directly rather than through a coordinate
/// that happens to build one. `HFModelRefTests` covers what `HFModelRef.parse` does
/// with them.
struct PrimitivesTests {

    /// A `ValidatedString` is transparent on the wire — a bare string, not a wrapper
    /// object — and decode re-runs validation, so an invalid value cannot enter through
    /// the back door that construction closes.
    @Test func encodesAsBareSlugString() throws {
        let org = try HFOrg("LiquidAI")
        let data = try JSONEncoder().encode(org)
        #expect(String(decoding: data, as: UTF8.self) == "\"LiquidAI\"")
        #expect(throws: (any Error).self) {
            _ = try JSONDecoder().decode(HFOrg.self, from: Data("\".bad\"".utf8))
        }
    }

    /// `.` and `..` match the segment character class, so the exclusion is by name and
    /// a reader could mistake it for redundant. Asserted on the validator directly, not
    /// only through `HFModelRef.parse`, so the rule survives a caller changing.
    @Test(arguments: ["..", "../escape", "dir/../seg", "./seg", "dir/."])
    func repoSubpathRejectsADotSegment(_ candidate: String) {
        #expect(!RepoSubpath.validate(candidate))
    }

    @Test(arguments: ["4bit", "variants/4bit", "a.b-c_d"])
    func repoSubpathAcceptsOrdinarySegments(_ candidate: String) {
        #expect(RepoSubpath.validate(candidate))
    }

    /// `/` is allowed so a ref like `refs/pr/1` validates — the crate's regex does the
    /// same, and rejecting it would make a legitimate pin unrepresentable.
    @Test(arguments: ["main", "v1.2.3", "refs/pr/1", "a1b2c3d"])
    func revisionAcceptsTagsBranchesAndRefs(_ candidate: String) {
        #expect(HFRevision.validate(candidate))
    }

    @Test(arguments: ["", "-leading", "/leading", "has space"])
    func revisionRejectsMalformed(_ candidate: String) {
        #expect(!HFRevision.validate(candidate))
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

    /// Sanitizing before validating is what makes `./models/m` a *rejection* rather than
    /// an accidental accept — the crate's ordering, and the reason the hook is shared.
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
        let bare = try HFRepo.parse("org/repo")
        var tokened = bare
        tokened.authToken = try AuthToken("hf_secret")
        var pinned = bare
        pinned.revision = try HFRevision("v1.2.3")

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
            with: try JSONEncoder().encode(try HFRepo.parse("org/repo"))) as? [String: Any]
        let keys = Set(json?.keys ?? [:].keys)
        #expect(keys == ["org", "repo_name"])
    }

    /// Redaction drops the credential and keeps everything that is identity, so a
    /// redacted coordinate still addresses the same storage entry.
    @Test func redactionDropsOnlyTheToken() throws {
        var repo = try HFRepo.parse("org/repo")
        repo.revision = try HFRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")

        let redacted = repo.withoutAuthToken
        #expect(redacted.authToken == nil)
        #expect(redacted.reference == "org/repo@v1.2.3")
        #expect(redacted == repo)  // identity ignores the token, so this must still hold
    }

    /// `withoutAuthToken` reaches the repo inside every file-backed `Model` case. A
    /// case added without a redacting arm would leak, and nothing else would fail.
    @Test func modelRedactionReachesEveryCase() throws {
        var repo = try HFRepo.parse("org/repo")
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
}
