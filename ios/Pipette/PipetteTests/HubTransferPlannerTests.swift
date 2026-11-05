import Foundation
import Testing

@testable import Pipette

/// Freezing a directory pull: one commit, every file, every length, and a refusal for each
/// way that cannot be established. The metadata source is stated, so no case needs a network.
struct HubTransferPlannerTests {
    /// A stated Hub. `commits` supplies a commit per file so a moving branch can be
    /// spelled as two files disagreeing.
    private struct StatedHub: HubMetadataSource {
        var names: [String]
        var sizes: [String: Int64] = [:]
        var commits: [String: String] = [:]
        var etags: [String: String] = [:]
        var sha256s: [String: Sha256] = [:]
        /// Paths the server answers without a size, and without a commit. Stated as sets
        /// rather than as a nil value, which in a dictionary of optionals is a nested
        /// optional nobody reads correctly.
        var unsized: Set<String> = []
        var uncommitted: Set<String> = []
        var listingError: (any Error)?
        /// What each call was asked, so a test can pin the revision that was used.
        let recorder = Recorder()

        final class Recorder: @unchecked Sendable {
            private let lock = NSLock()
            private var seen: [String] = []
            func record(_ entry: String) { lock.lock(); seen.append(entry); lock.unlock() }
            var entries: [String] { lock.lock(); defer { lock.unlock() }; return seen }
        }

        func filenames(repo: HfRepo, revision: String,
                       matching globs: [String]) async throws -> [String] {
            if let listingError { throw listingError }
            recorder.record("list@\(revision)")
            return names
        }

        func metadata(repo: HfRepo, revision: String, path: String) async throws -> HubFileMetadata {
            recorder.record("head@\(revision):\(path)")
            return HubFileMetadata(sizeBytes: unsized.contains(path) ? nil : (sizes[path] ?? 1024),
                                   commitHash: uncommitted.contains(path)
                                       ? nil : (commits[path] ?? "c0ffee"),
                                   etag: etags[path], sha256: sha256s[path])
        }
    }

    private func ref(_ repo: String, prefix: String? = nil) throws -> HfModelRef {
        try HfModelRef.parse(repo: repo, subpath: prefix)
    }

    @Test func aFrozenPlanCarriesTheCommitEveryFileAndEveryLength() async throws {
        let hub = StatedHub(names: ["b.safetensors", "a.json"],
                            sizes: ["a.json": 100, "b.safetensors": 900])

        let plan = try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)

        #expect(plan.commit == "c0ffee")
        // Sorted, so one coordinate freezes to a byte-identical plan whatever order the
        // listing arrived in.
        #expect(plan.files.map(\.path) == ["a.json", "b.safetensors"])
        #expect(plan.totalBytes == 1000)
    }

    /// The listing is read at the declared revision, not at the default. `HubApi`'s own
    /// `getFileMetadata(from:revision:matching:)` gets this wrong, which is why the
    /// planner composes the two calls itself.
    @Test func theListingIsReadAtTheDeclaredRevision() async throws {
        var repo = try HfRepo.parseOrgRepo("org/m-MLX-4bit")
        repo.revision = try HfRevision("v1.2.3")
        let hub = StatedHub(names: ["a.json"])

        _ = try await HubTransferPlanner.resolve(
            HfModelRef(repo: repo, subpath: nil), globs: ["*"], using: hub)

        #expect(hub.recorder.entries.first == "list@v1.2.3")
        #expect(hub.recorder.entries.contains("head@v1.2.3:a.json"))
    }

    /// An unpinned coordinate resolves against `main` and freezes to whatever commit that
    /// names right now, which is the whole point: the pull addresses the commit from here
    /// on, not the branch.
    @Test func anUnpinnedCoordinateFreezesToTheBranchTip() async throws {
        let hub = StatedHub(names: ["a.json"], commits: ["a.json": "deadbeef"])

        let plan = try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)

        #expect(hub.recorder.entries.first == "list@main")
        #expect(plan.commit == "deadbeef")
    }

    /// **The moving-branch case.** A branch that advances between two HEADs shows up as
    /// two commits, and planning across that boundary would mix trees.
    @Test func filesDisagreeingOnACommitAreRefused() async throws {
        let hub = StatedHub(names: ["a.json", "b.safetensors"],
                            commits: ["a.json": "aaa111", "b.safetensors": "bbb222"])

        await #expect(throws: TransferPlanningError.unresolvedCommit("org/m-MLX-4bit")) {
            try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)
        }
    }

    /// A server that sends no commit at all leaves nothing immutable to address, so the
    /// pull is refused rather than made against a branch.
    @Test func aMissingCommitIsRefused() async throws {
        var hub = StatedHub(names: ["a.json"])
        hub.uncommitted = ["a.json"]

        await #expect(throws: TransferPlanningError.unresolvedCommit("org/m-MLX-4bit")) {
            try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)
        }
    }

    /// A file the server will not size refuses the whole pull: admission cannot budget an
    /// unsized transfer, and guessing is what overruns the quota.
    @Test func aFileWithNoSizeRefusesThePull() async throws {
        var hub = StatedHub(names: ["a.json", "big.safetensors"])
        hub.unsized = ["big.safetensors"]

        await #expect(throws: TransferPlanningError.unknownSize(path: "big.safetensors")) {
            try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)
        }
    }

    /// An empty selection is "no such subtree", not an empty model: publishing it would
    /// produce an entry bound to an empty directory.
    @Test func anEmptySelectionIsRefused() async throws {
        let hub = StatedHub(names: [])

        await #expect(throws: TransferPlanningError.emptySelection("org/multi-MLX/4bit")) {
            try await HubTransferPlanner.resolve(
                try ref("org/multi-MLX", prefix: "4bit"), globs: ["4bit/*"], using: hub)
        }
    }

    @Test func anUnreadableListingIsReportedNotTreatedAsEmpty() async throws {
        var hub = StatedHub(names: [])
        hub.listingError = URLError(.notConnectedToInternet)

        await #expect(throws: TransferPlanningError.self) {
            try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)
        }
    }

    /// An ETag rides the plan for change detection and is never presented as a digest: for
    /// a small inline file it is not a content hash at all.
    @Test func anEtagIsCarriedButNotTreatedAsADigest() async throws {
        var hub = StatedHub(names: ["a.json"])
        hub.etags = ["a.json": "\"abc123\""]

        let plan = try await HubTransferPlanner.resolve(try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)

        #expect(plan.files.first?.etag == "\"abc123\"")
        #expect(plan.files.first?.sha256 == nil)
    }

    /// The Hub adapter recognizes only the Hub's 64-hex LFS validator as a SHA-256
    /// reference. A generic ETag remains useful for identity but never triggers hashing.
    @Test func onlyAHubSha256ReferenceBecomesADigest() throws {
        let digest = String(repeating: "a", count: 64)

        #expect(HubApiMetadataSource.sha256Reference(fromHubETag: digest)?.value == digest)
        #expect(HubApiMetadataSource.sha256Reference(fromHubETag: "abc123") == nil)
        #expect(HubApiMetadataSource.sha256Reference(fromHubETag: nil) == nil)
    }

    /// Once a metadata source offers a trusted reference, the frozen plan retains it for
    /// publication instead of silently reducing verification to length.
    @Test func aSourceDigestIsCarriedIntoTheFrozenPlan() async throws {
        let digest = try Sha256(String(repeating: "b", count: 64))
        var hub = StatedHub(names: ["model.safetensors"])
        hub.sha256s = ["model.safetensors": digest]

        let plan = try await HubTransferPlanner.resolve(
            try ref("org/m-MLX-4bit"), globs: ["*"], using: hub)

        #expect(plan.files.first?.sha256 == digest)
    }

    /// A listed path that leaves the payload is refused before any byte moves. Joined
    /// beneath the entry, each of these writes somewhere the caller did not ask for.
    @Test(arguments: ["../escape.bin", "a/../../escape.bin", "/etc/passwd", ""])
    func anEscapingPathIsRefused(path: String) throws {
        #expect(throws: TransferPlanningError.self) {
            try HubTransferPlanner.checkDestinations(of: ["config.json", path])
        }
    }

    /// Two paths that reach one destination are refused. The volume is case-insensitive,
    /// so a pair differing only by case is one file, and the inventory would not match.
    @Test func pathsCollidingOnTheDestinationAreRefused() throws {
        #expect(throws: TransferPlanningError.self) {
            try HubTransferPlanner.checkDestinations(of: ["Config.json", "config.json"])
        }
    }

    @Test func repeatedBasenamesInDifferentDirectoriesAreAccepted() throws {
        try HubTransferPlanner.checkDestinations(
            of: ["a/config.json", "b/config.json", "model.onnx"])
    }
}
