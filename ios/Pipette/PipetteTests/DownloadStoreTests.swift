import Foundation
import Testing

@testable import Pipette

/// `DownloadStore`'s isolation by directory, which keeps a test's persisted transfers out
/// of the real caches directory, and where the production one sits.
@MainActor struct DownloadStoreTests {
    @Test func downloadStoresAreIsolatedByDirectory() throws {
        let a = makeTemporaryWorkspace()
        let b = makeTemporaryWorkspace()
        defer { removeWorkspace(a); removeWorkspace(b) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let repo = "org/a-GGUF"
        let filename = "a-Q4_0.gguf"

        DownloadStore(dir: a.downloadsDir).saveRecord(DownloadRecord(
            filename: filename,
            urlString: "https://huggingface.co/\(repo)/resolve/main/\(filename)",
            repo: repo,
            familyId: nil,
            source: spec
        ))

        #expect(DownloadStore(dir: a.downloadsDir).allRecords().count == 1)
        #expect(DownloadStore(dir: b.downloadsDir).allRecords().isEmpty)
        // Production resolves under the system Caches root, where bytes that can be fetched
        // again belong. Asserted whole rather than by suffix: the same suffix hangs off
        // `root` (Application Support, which is backed up and never evicted), and that
        // is the plausible way for this to drift.
        let caches = try #require(
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first)
        #expect(PipetteWorkspace.production.downloadsDir.path
            == caches.appendingPathComponent("Pipette/downloads").path)
    }
}
