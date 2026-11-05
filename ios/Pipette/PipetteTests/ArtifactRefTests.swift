import Foundation
import Testing

@testable import Pipette

/// Addressing an installed artifact by a prefix of its descriptor digest, mirroring
/// `pipette-cli`'s `artifact_ref.rs`.
struct ArtifactRefTests {
    /// `malformed_prefixes_are_rejected`: too short or not hex can never match.
    @Test(arguments: [("abc", ArtifactRefError.tooShort("abc")),
                      ("zzzzzzzzzz", .notHex("zzzzzzzzzz")),
                      ("abc123", .tooShort("abc123"))])
    func malformedPrefixesAreRejected(prefix: String, expected: ArtifactRefError) {
        #expect(throws: expected) { try validatedPrefix(prefix) }
    }

    /// `prefixes_are_normalized_and_trimmed`: digests are hex, so case carries nothing.
    @Test func prefixesAreNormalizedAndTrimmed() throws {
        #expect(try validatedPrefix("  ABCDEF01  ") == "abcdef01")
    }

    @Test func aSchemeOnlyClaimsItsOwnDigests() throws {
        #expect(try digestPrefix("runtime://sha256=abcdef01", scheme: runtimeScheme)
            == "abcdef01")
        #expect(try digestPrefix("model://sha256=abcdef01", scheme: runtimeScheme)
            == nil)
        #expect(try digestPrefix("  model://sha256=ABCDEF01 ", scheme: modelScheme)
            == "abcdef01")
    }

    /// The shape a bare digest pasted after the scheme takes gets its own refusal, not
    /// "unknown scheme".
    @Test func theSchemeWithoutADigestKeyExplainsItself() {
        #expect(throws: ArtifactRefError.digestKeyMissing(scheme: "runtime://")) {
            try digestPrefix("runtime://abcdef0123", scheme: runtimeScheme)
        }
        #expect(ArtifactRefError.digestKeyMissing(scheme: "model://").localizedDescription
            .contains("takes a digest"))
    }

    @Test func ordinaryUrisAreNotTreatedAsDigests() throws {
        #expect(try digestPrefix("gguf-text://repo=a/b&path=c.gguf",
                                             scheme: modelScheme) == nil)
        #expect(try digestPrefix(#"{"type":"mlx"}"#, scheme: modelScheme) == nil)
    }

    /// `misses_and_ambiguity_are_both_errors`: the second names the candidates so the
    /// operator can lengthen the prefix.
    @Test func missesAndAmbiguityAreBothErrors() {
        let none: [(digest: String, value: String)] = []
        #expect(throws: ArtifactRefError.noMatch(kind: "runtime", prefix: "abcdef01")) {
            try uniqueMatch("abcdef01", none, kind: "runtime", render: { $0 })
        }
        #expect(ArtifactRefError.noMatch(kind: "runtime", prefix: "abcdef01").localizedDescription
            .contains("no installed runtime"))

        let two: [(digest: String, value: String)] = [("abcdef0111", "first"), ("abcdef0122", "second")]
        do {
            _ = try uniqueMatch("abcdef01", two, kind: "runtime", render: { $0 })
            Issue.record("two matches should be ambiguous")
        } catch {
            #expect(error == .ambiguous(kind: "runtime", prefix: "abcdef01",
                                        candidates: ["first", "second"]))
            #expect(error.localizedDescription.contains("ambiguous"))
            #expect(error.localizedDescription.contains("first, second"))
        }
    }

    /// The address is the digest of the token-stripped descriptor a submission carries.
    @Test func aModelIsAddressedByItsStrippedDescriptor() throws {
        var repo = try HfRepo.parseOrgRepo("org/gated")
        repo.authToken = try AuthToken("hf_secret")
        let tokened = Model.ggufText(GgufText(source: .huggingFace(
            repo: repo, path: try RepoSubpath("m.gguf"), sha256: nil)))

        #expect(try modelDescriptorDigest(tokened) == descriptorDigest(tokened.withoutAuthToken))
        #expect(try modelDescriptorDigest(tokened) == modelDescriptorDigest(tokened.withoutAuthToken))
    }

    /// The server takes `model_descriptor_sha256` without the vault `custom_label`, so the
    /// unlabelled prefix finds a labelled model installed.
    @Test func aCustomLabelDoesNotMoveTheAddress() throws {
        for (plain, labelled) in try vaultSpecPairs() {
            let digest = try descriptorDigest(plain)
            #expect(try modelDescriptorDigest(plain) == digest)
            #expect(try modelDescriptorDigest(labelled) == digest)
            // Guards the premise: the labelled descriptor as submitted hashes apart.
            #expect(try descriptorDigest(labelled) != digest)
            let prefix = String(digest.prefix(digestDisplayLen))
            let resolved = try resolveModel(digestPrefix: prefix, installed: [labelled])
            #expect(resolved == plain)
            #expect(try reportedLabel(resolved) == nil)
        }
        // The hash of the canonical unlabelled text, as the server computes the id.
        #expect(try modelDescriptorDigest(vaultTextSpec("vault_1", label: "acme LFM3 q4"))
            == sha256Hex(#"{"principal_id":"org_1","source":"vault_url","type":"gguf_text","url":"/vault/resources/vault_1/content"}"#))
    }

    @Test func aListedDigestResolvesTheInstalledArtifact() throws {
        let a = try ggufTextSpec("org/a-GGUF", "a.gguf")
        let b = try mlxSpec("org/b-MLX-4bit")
        let prefix = String(try modelDescriptorDigest(a).prefix(digestDisplayLen))

        #expect(try resolveModel(digestPrefix: prefix, installed: [a, b]) == a)
        #expect(throws: ArtifactRefError.noMatch(kind: "model", prefix: prefix)) {
            try resolveModel(digestPrefix: prefix, installed: [b])
        }
    }
}
