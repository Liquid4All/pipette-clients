import Foundation
import Hub
import HuggingFace
import Testing

@testable import Pipette

/// `resetStaging`, the part of the MLX downloader that owns disk state: a staged model
/// directory survives only a pull at the revision it was fetched at. No network.
struct HubModelTransportTests {
    private static let weights = "model.safetensors"

    private func stage(_ dir: URL, revision: String?) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: dir.appending(path: Self.weights))
        if let revision {
            try Data(revision.utf8).write(to: dir.appending(path: HubModelTransport.revisionMarker))
        }
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "staging-\(UUID().uuidString)")
    }

    @Test func freshPullCreatesMarkedDirectory() throws {
        let dir = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: dir) }

        try HubModelTransport.resetStaging(dir, revision: "main")

        let marker = dir.appending(path: HubModelTransport.revisionMarker)
        #expect(try String(contentsOf: marker, encoding: .utf8) == "main")
    }

    /// The resume: an interrupted pull at the same revision keeps what it fetched.
    @Test func sameRevisionKeepsStagedFiles() throws {
        let dir = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: dir) }
        try stage(dir, revision: "main")

        try HubModelTransport.resetStaging(dir, revision: "main")

        #expect(FileManager.default.fileExists(atPath: dir.appending(path: Self.weights).path))
    }

    /// An interrupted `main` pull, then a pinned one: main's files must not reach the
    /// pin's install. An unmarked tree's revision is unknown, so it is dropped too.
    @Test(arguments: ["main", nil] as [String?])
    func otherOrUnknownRevisionDropsStagedFiles(_ staged: String?) throws {
        let dir = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: dir) }
        try stage(dir, revision: staged)

        try HubModelTransport.resetStaging(dir, revision: "0123abcd")

        #expect(!FileManager.default.fileExists(atPath: dir.appending(path: Self.weights).path))
        let marker = dir.appending(path: HubModelTransport.revisionMarker)
        #expect(try String(contentsOf: marker, encoding: .utf8) == "0123abcd")
    }

    @Test func siblingSubpathIsLeftAlone() throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try stage(root.appending(path: "a"), revision: "main")
        try stage(root.appending(path: "b"), revision: "main")

        try HubModelTransport.resetStaging(root.appending(path: "a"), revision: "0123abcd")

        #expect(FileManager.default.fileExists(atPath: root.appending(path: "b/\(Self.weights)").path))
    }
}

/// The retries of a directory pull: each file of the frozen plan retries on its own, and a
/// `HubApi` failure is transient or permanent by what it says about the next attempt. No
/// network: a script supplies each file's fetch.
@MainActor struct HubModelTransportRetryTests {
    private static let files = [
        PlannedFile(path: "config.json", sizeBytes: 100, etag: nil),
        PlannedFile(path: "model.safetensors", sizeBytes: 800, etag: nil),
        PlannedFile(path: "tokenizer.json", sizeBytes: 100, etag: nil),
    ]

    /// What one `fetchEach` run did.
    @MainActor final class Run {
        var fetches: [String] = []
        var events: [RetryEvent] = []
    }

    /// Fetch the plan, with the file at `path` failing `times` times with `failure`.
    private func fetch(failing path: String, times: Int, with failure: any Error,
                       run: Run) async throws {
        try await HubModelTransport.fetchEach(
            Self.files, totalBytes: 1000, url: { "https://huggingface.co/org/m/resolve/c/\($0.path)" },
            policy: .download,
            hooks: RetryHooks(sleep: { _ in }, report: { run.events.append($0) }),
            progress: { _ in }
        ) { file, _ in
            run.fetches.append(file.path)
            if file.path == path, run.fetches.filter({ $0 == path }).count <= times {
                throw failure
            }
        }
    }

    /// A file that fails part-way is fetched again, and the files around it once each.
    @Test(arguments: [1, 2, 4])
    func aFileThatFailsIsRetriedAndTheOthersAreNot(times: Int) async throws {
        let run = Run()
        try await fetch(failing: "model.safetensors", times: times,
                        with: URLError(.networkConnectionLost), run: run)

        #expect(run.fetches.filter { $0 == "config.json" }.count == 1)
        #expect(run.fetches.filter { $0 == "model.safetensors" }.count == times + 1)
        #expect(run.fetches.filter { $0 == "tokenizer.json" }.count == 1)
        let url = "https://huggingface.co/org/m/resolve/c/model.safetensors"
        #expect(run.events.count == times + 1, "one warning per failure, and the recovery")
        #expect(run.events.last == .recovered(url: url, attempts: times + 1))
    }

    /// Five failures exhaust the file's attempts, and the pull stops there.
    @Test func aFileThatKeepsFailingStopsThePull() async {
        let run = Run()
        await #expect(throws: URLError.self) {
            try await fetch(failing: "model.safetensors", times: 5,
                            with: URLError(.timedOut), run: run)
        }
        #expect(run.fetches == ["config.json"] + Array(repeating: "model.safetensors", count: 5))
    }

    @Test func aMissingFileGetsOneAttempt() async {
        let run = Run()
        await #expect(throws: Hub.HubClientError.self) {
            try await fetch(failing: "model.safetensors", times: 1,
                            with: Hub.HubClientError.fileNotFound("model.safetensors"), run: run)
        }
        #expect(run.fetches == ["config.json", "model.safetensors"])
        #expect(run.events.isEmpty)
    }

    /// A cancellation that the fetch returns through stops the loop after that file.
    @Test func cancellationStopsTheLoop() async {
        var fetches: [String] = []
        let task = Task { @MainActor in
            try await HubModelTransport.fetchEach(
                Self.files, totalBytes: 1000, url: { $0.path },
                policy: .download, hooks: RetryHooks(sleep: { _ in }, report: { _ in }),
                progress: { _ in }
            ) { file, _ in
                fetches.append(file.path)
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(fetches == ["config.json"])
    }

    /// The fraction is of the whole plan, by size: halfway through the large file is
    /// halfway through the pull.
    @Test func progressIsTheFractionOfThePlanBySize() async throws {
        let reported = Reported()
        try await HubModelTransport.fetchEach(
            Self.files, totalBytes: 1000, url: { $0.path },
            policy: .download, hooks: RetryHooks(sleep: { _ in }, report: { _ in }),
            progress: { reported.append($0) }
        ) { file, progress in
            if file.path == "model.safetensors" { progress(0.5) }
            progress(1)
        }
        #expect(reported.values == [0.1, 0.5, 0.9, 1.0])
    }

    /// Progress arrives from any thread, so the test collects it behind a lock.
    final class Reported: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [Double] = []
        func append(_ value: Double) { lock.withLock { stored.append(value) } }
        var values: [Double] { lock.withLock { stored } }
    }

    @Test(arguments: [
        ("model.safetensors", "model.safetensors"),
        ("weights[0].bin", "weights\\[0\\].bin"),
        ("a*b?.json", "a\\*b\\?.json"),
        ("a[\u{301}b.json", "a\\[\u{301}b.json"),
    ])
    func anExactGlobEscapesThePatternCharacters(path: String, glob: String) {
        #expect(HubModelTransport.exactGlob(path) == glob)
    }

    /// Each shape a `HubApi` failure arrives in, and what it says about the next attempt.
    @Test func aHubFailureIsTransientOnlyWhenItCanPass() throws {
        let url = try #require(URL(string: "https://cdn.example.test/model.safetensors"))
        let response = { (status: Int, headers: [String: String]) in
            HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                            headerFields: headers)
        }
        let rateLimited = try #require(response(429, ["Retry-After": "7"]))
        let missing = try #require(response(404, [:]))
        let cases: [(any Error, Disposition)] = [
            (URLError(.networkConnectionLost), .transient(retryAfter: nil)),
            (URLError(.badServerResponse), .permanent),
            (HTTPClientError.responseError(response: rateLimited, detail: ""),
             .transient(retryAfter: 7)),
            (HTTPClientError.responseError(response: missing, detail: ""), .permanent),
            (Hub.HubClientError.httpStatusCode(503), .transient(retryAfter: nil)),
            (Hub.HubClientError.httpStatusCode(400), .permanent),
            (Hub.HubClientError.networkError(URLError(.timedOut)), .transient(retryAfter: nil)),
            (Hub.HubClientError.downloadError("The network connection was lost."),
             .transient(retryAfter: nil)),
            (Hub.HubClientError.authorizationRequired, .permanent),
            (Hub.HubClientError.fileNotFound("model.safetensors"), .permanent),
            (CocoaError(.fileWriteOutOfSpace), .permanent),
        ]
        for (error, expected) in cases {
            #expect(Disposition(ofHub: error) == expected, "\(error)")
        }
    }
}
