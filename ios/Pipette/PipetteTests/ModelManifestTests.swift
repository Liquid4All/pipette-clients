import Foundation
import Testing

@testable import Pipette

/// `ModelManifest` write/read round-trips, embeds the crate-tagged `Model`
/// under a `declared` key, and accepts only the exact current version — a bump strands
/// the store on purpose, so there is nothing to bridge in either direction.
struct ModelManifestTests {
    private func tempFile() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-\(UUID().uuidString).json")
    }

    @Test func roundTripsThroughDisk() throws {
        let url = tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let lastUsedAt = Date(timeIntervalSince1970: 1_700_000_000)
        let manifest = ModelManifest(
            declared: .mlx(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))),
            lastUsedAt: lastUsedAt)
        manifest.writeQuietly(to: url)
        let read = try #require(ModelManifest.read(from: url))
        #expect(read.declared == manifest.declared)
        #expect(read.version == ModelManifest.currentVersion)
        #expect(read.lastUsedAt == lastUsedAt)
    }

    /// A store entry describes weights, never a credential. The redaction is in
    /// `encode`, so a caller holding a claim's token cannot write one out by accident.
    @Test func neverWritesTheAuthToken() throws {
        var repo = try HFRepo.parse("LiquidAI/Gated-GGUF")
        repo.revision = try HFRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        let manifest = ModelManifest(
            declared: .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil))))

        let encoded = try manifest.encoded()
        let json = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let declared = try #require(json["declared"] as? [String: Any])
        #expect(declared["auth_token"] == nil)
        // The revision is identity and must survive — this is a redaction, not a narrowing.
        #expect(declared["revision"] as? String == "v1.2.3")
        #expect(!String(decoding: encoded, as: UTF8.self).contains("hf_secret"))
    }

    /// An entry written by an earlier build can hold a token; `touchLastUsed` reads a
    /// manifest and writes it back, which is the opportunity to clean it.
    @Test func rewritingAnEntryStripsAnAlreadyStoredToken() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("entry-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = ModelManifest.manifestURL(inEntryDir: dir)

        // Stage the pre-fix shape by injecting the key into a real manifest's JSON: the
        // encoder redacts, so a tainted file cannot be produced by encoding one.
        let manifest = ModelManifest(
            declared: .ggufText(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/Gated-GGUF"), path: try RepoSubpath("m-Q4_K_M.gguf"), sha256: nil))))
        var json = try #require(try JSONSerialization.jsonObject(with: try manifest.encoded()) as? [String: Any])
        var declared = try #require(json["declared"] as? [String: Any])
        declared["auth_token"] = "hf_secret"
        json["declared"] = declared
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        let staged = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        #expect(staged.contains("hf_secret"))

        ModelManifest.touchLastUsed(inEntryDir: dir)
        let rewritten = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        #expect(!rewritten.contains("hf_secret"))
    }

    @Test func embedsDeclaredUnderDeclaredKey() throws {
        let manifest = ModelManifest(
            declared: .ggufText(.init(source: .huggingFace(repo: try HFRepo.parse("LiquidAI/LFM2.5-350M-GGUF"), path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil))))
        let json = try #require(try JSONSerialization.jsonObject(with: manifest.encoded()) as? [String: Any])
        #expect(json["manifest_version"] as? Int == ModelManifest.currentVersion)
        let declared = try #require(json["declared"] as? [String: Any])
        #expect(declared["type"] as? String == "gguf_text")
        #expect(declared["repo_name"] as? String == "LFM2.5-350M-GGUF")
    }

    @Test func manifestLivesAtTheEntryRoot() throws {
        let entry = URL(fileURLWithPath: "/models/LiquidAI__LFM2.5-350M-MLX-4bit")
        #expect(ModelManifest.manifestURL(inEntryDir: entry).path
            == "/models/LiquidAI__LFM2.5-350M-MLX-4bit/manifest.json")
    }

    @Test func rejectsNewerVersion() throws {
        #expect(readingJSON(
            #"{"manifest_version":999,"declared":{"type":"mlx","source":"huggingface","org":"x","repo_name":"y"},"last_used_at":"2026-07-01T00:00:00Z"}"#
        ) == nil)
    }

    @Test func rejectsOlderVersion() throws {
        #expect(readingJSON(
            #"{"manifest_version":1,"declared":{"type":"mlx","source":"huggingface","org":"x","repo_name":"y"}}"#
        ) == nil)
    }

    /// No serde default bridges the new field: a v2 manifest without `last_used_at`
    /// simply doesn't decode.
    @Test func requiresLastUsedAt() throws {
        #expect(readingJSON(
            #"{"manifest_version":2,"declared":{"type":"mlx","source":"huggingface","org":"x","repo_name":"y"}}"#
        ) == nil)
    }

    private func readingJSON(_ json: String) -> ModelManifest? {
        let url = tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        try? Data(json.utf8).write(to: url)
        return ModelManifest.read(from: url)
    }
}
