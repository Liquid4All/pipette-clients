import Foundation
import Testing

@testable import Pipette

/// `ModelManifest` round-trips, embeds the crate-tagged `Model` under `declared`, and reads
/// back only at the exact current version: a bump strands the store on purpose.
struct ModelManifestTests {
    private func tempFile() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-\(UUID().uuidString).json")
    }

    /// The crate's `serde_round_trips_fields`: every field survives, and the two
    /// timestamps are carried independently.
    @Test func roundTripsThroughDisk() throws {
        let url = tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let manifest = try manifestFixture(
            .mlx(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-MLX-4bit"), prefix: nil))),
            fetchedAt: Date(timeIntervalSince1970: 1_700_000_000),
            lastUsedAt: Date(timeIntervalSince1970: 1_700_086_400),
            blobsBytes: 4_096)
        manifest.writeQuietly(to: url)

        let read = try ModelArtifactStore.readManifest(at: url)

        #expect(read == manifest)
        #expect(read.manifestVersion == ModelManifest.manifestVersion)
        #expect(read.lastUsedAt != read.fetchedAt)
    }

    /// A store entry describes weights, never a credential. The redaction is in
    /// `encode`, so a caller holding a claim's token cannot write one out by accident.
    @Test func neverWritesTheAuthToken() throws {
        var repo = try HfRepo.parseOrgRepo("LiquidAI/Gated-GGUF")
        repo.revision = try HfRevision("v1.2.3")
        repo.authToken = try AuthToken("hf_secret")
        let manifest = try manifestFixture(
            .ggufText(.init(source: .huggingFace(repo: repo, path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil))))

        let encoded = try manifest.encoded()
        let json = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let declared = try #require(json["declared"] as? [String: Any])
        #expect(declared["auth_token"] == nil)
        // The revision is identity and must survive: a redaction, not a narrowing.
        #expect(declared["revision"] as? String == "v1.2.3")
        #expect(!String(decoding: encoded, as: UTF8.self).contains("hf_secret"))
    }

    @Test func embedsDeclaredAndStoredUnderTheCratesKeys() throws {
        let manifest = try manifestFixture(
            .ggufText(.init(source: .huggingFace(repo: try HfRepo.parseOrgRepo("LiquidAI/LFM2.5-350M-GGUF"), path: try RepoSubpath("LFM2.5-350M-Q4_K_M.gguf"), sha256: nil))))
        let json = try #require(try JSONSerialization.jsonObject(with: manifest.encoded()) as? [String: Any])

        #expect(json["manifest_version"] as? Int == ModelManifest.manifestVersion)
        #expect(json["fetched_at"] is String)
        #expect(json["last_used_at"] is String)
        let declared = try #require(json["declared"] as? [String: Any])
        #expect(declared["type"] as? String == "gguf_text")
        #expect(declared["repo_name"] as? String == "LFM2.5-350M-GGUF")
        let stored = try #require(json["stored"] as? [String: Any])
        #expect(stored["source"] as? String == "relative_file")
    }

    @Test func manifestLivesAtTheEntryRoot() throws {
        let entry = URL(fileURLWithPath: "/models/LiquidAI__LFM2.5-350M-MLX-4bit")
        #expect(ModelManifest.manifestURL(inEntryDir: entry).path
            == "/models/LiquidAI__LFM2.5-350M-MLX-4bit/manifest.json")
    }

    /// A version bump strands the store on purpose: any version but this one is corrupt.
    @Test(arguments: [1, 3, 999])
    func rejectsAnyOtherVersion(_ version: Int) throws {
        let error = try reading { $0["manifest_version"] = version }
        #expect({ if case .corrupt = error { true } else { false } }(), "\(String(describing: error))")
    }

    /// No default bridges the two fields every build has always written: a manifest
    /// missing either does not decode.
    @Test(arguments: ["last_used_at", "declared"])
    func requiresEveryRecordedField(_ field: String) throws {
        let error = try reading { $0.removeValue(forKey: field) }
        #expect({ if case .parse = error { true } else { false } }(), "\(String(describing: error))")
    }

    /// Older builds left `fetched_at` and `stored` out, so such a record is read, not
    /// refused: `fetchedAt` falls back to `lastUsedAt` and `stored` is derived.
    @Test(arguments: ["fetched_at", "stored"])
    func recoversTheFieldsAnOlderBuildLeftOut(_ field: String) throws {
        #expect(try reading { $0.removeValue(forKey: field) } == nil)
    }

    /// Why a manifest with `edit` applied does not read, or nil if it does.
    private func reading(_ edit: (inout [String: Any]) -> Void) throws -> ModelStoreError? {
        let url = tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let manifest = try manifestFixture(try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf"))
        var json = try #require(try JSONSerialization.jsonObject(with: manifest.encoded()) as? [String: Any])
        edit(&json)
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        do {
            _ = try ModelArtifactStore.readManifest(at: url)
            return nil
        } catch {
            return error
        }
    }
}
