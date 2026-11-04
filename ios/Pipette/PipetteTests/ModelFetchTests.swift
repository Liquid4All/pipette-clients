import Foundation
import Testing

@testable import Pipette

/// `ModelFetch.declaredSizeBytes`, the size a fetch is refused on before it starts, asserted
/// against a stubbed wire as the crate's `declared_size_bytes_*` tests use `httpmock`.
///
/// `.serialized`: `StubURLProtocol.answer` is process-global.
@Suite(.serialized) @MainActor struct ModelFetchTests {

    /// A client whose every request is answered by `reply`, with the requests it made.
    private func stub(
        _ reply: @escaping @Sendable (URLRequest) -> StubResponse
    ) -> (http: HttpClient, seen: StubRecorder) {
        let recorder = StubRecorder()
        StubURLProtocol.answer = { request in
            recorder.record(request)
            return reply(request)
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return (HttpClient(userAgent: "pipette-test/0", configuration: configuration), recorder)
    }

    /// A HEAD answer: the length a GGUF file is sized by, and no body.
    nonisolated private static func length(_ bytes: Int) -> StubResponse {
        StubResponse(status: 200, body: Data(), headers: ["Content-Length": "\(bytes)"])
    }

    nonisolated private static let notFound = StubResponse(status: 404, body: Data(), headers: [:])

    /// A `?blobs=true` listing. A `nil` size omits the key, as the Hub does for a file it
    /// has not sized.
    nonisolated private static func listing(_ files: [(String, Int?)]) -> StubResponse {
        let siblings = files.map { name, size in
            size.map { #"{"rfilename":"\#(name)","size":\#($0)}"# } ?? #"{"rfilename":"\#(name)"}"#
        }
        let body = #"{"siblings":[\#(siblings.joined(separator: ","))]}"#
        return StubResponse(status: 200, body: Data(body.utf8), headers: [:])
    }

    /// A GGUF file is sized by a HEAD on the URL its transfer downloads from, pin and
    /// credential included, through the shared client's `User-Agent`.
    @Test func aGgufFileIsSizedByAHeadOnTheURLItDownloadsFrom() async throws {
        var repo = try HfRepo.parseOrgRepo("org/a-GGUF")
        repo.revision = try HfRevision("v1")
        repo.authToken = try AuthToken("hf_claim")
        let path = try RepoSubpath("a-Q4_0.gguf")
        let model = Model.ggufText(GgufText(source: .huggingFace(repo: repo, path: path, sha256: nil)))
        let (http, seen) = stub { _ in Self.length(4096) }

        #expect(await ModelFetch.declaredSizeBytes(model, http: http) == 4096)

        let request = try #require(seen.requests.first)
        #expect(request.httpMethod == "HEAD")
        #expect(request.url == URL(string: ModelFetch.hfResolveUrl(repo: repo, path: path.value)))
        #expect(request.url?.path == "/org/a-GGUF/resolve/v1/a-Q4_0.gguf")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer hf_claim")
        #expect(request.value(forHTTPHeaderField: "User-Agent") == "pipette-test/0")
    }

    /// A vision model is its weights plus its projector, and unsized when either cannot be
    /// sized: a total missing one file would pass an artifact that cannot fit.
    @Test func aVisionModelIsBothFilesOrUnsized() async throws {
        let model = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-f16.gguf")

        let (http, seen) = stub { request in
            request.url?.lastPathComponent == "vl-Q4_0.gguf" ? Self.length(3000) : Self.length(500)
        }
        #expect(await ModelFetch.declaredSizeBytes(model, http: http) == 3500)
        #expect(seen.requests.count == 2)

        let (partial, _) = stub { request in
            request.url?.lastPathComponent == "vl-Q4_0.gguf" ? Self.length(3000) : Self.notFound
        }
        #expect(await ModelFetch.declaredSizeBytes(model, http: partial) == nil)
    }

    /// An MLX root is sized from the files its globs pull, not the whole repo: a README and
    /// a GGUF quant sit beside the bundle and never land.
    @Test func anMlxRootIsSizedFromTheFilesItsGlobsPull() async throws {
        let model = try mlxSpec("org/m-MLX")
        let (http, seen) = stub { _ in
            Self.listing([
                ("README.md", 10), ("config.json", 20), ("model.safetensors", 1000),
                ("tokenizer.json", 30), ("m-Q4_0.gguf", 5000),
            ])
        }

        #expect(await ModelFetch.declaredSizeBytes(model, http: http) == 1050)

        let url = try #require(seen.requests.first?.url)
        #expect(url.path == "/api/models/org/m-MLX/revision/main")
        #expect(url.query == "blobs=true")
    }

    @Test func anMlxSubdirectoryIsSizedFromItsSubtree() async throws {
        let model = try mlxSpec("org/multi-MLX", subpath: "4bit")
        let (http, _) = stub { _ in
            Self.listing([
                ("config.json", 5), ("4bit/config.json", 20),
                ("4bit/model.safetensors", 1000), ("8bit/model.safetensors", 2000),
            ])
        }

        #expect(await ModelFetch.declaredSizeBytes(model, http: http) == 1020)
    }

    /// A listing that cannot bound the pull is no size at all: a file the Hub did not size,
    /// or a subtree that matches nothing ("no such subtree", not a zero-byte one).
    @Test func anMlxListingThatCannotBoundThePullIsUnsized() async throws {
        let model = try mlxSpec("org/m-MLX")

        let (unsized, _) = stub { _ in
            Self.listing([("config.json", 20), ("model.safetensors", nil)])
        }
        #expect(await ModelFetch.declaredSizeBytes(model, http: unsized) == nil)

        let (empty, _) = stub { _ in Self.listing([("README.md", 10)]) }
        #expect(await ModelFetch.declaredSizeBytes(model, http: empty) == nil)
    }

    /// Sizes are the server's, so a sum past `Int64.max` saturates rather than trapping,
    /// and `.max` fails any quota.
    @Test func aSizeThatWouldOverflowSaturates() async throws {
        let (mlx, _) = stub { _ in
            Self.listing([("model.safetensors", Int.max), ("config.json", Int.max)])
        }
        #expect(await ModelFetch.declaredSizeBytes(try mlxSpec("org/m-MLX"), http: mlx) == .max)

        let (vision, _) = stub { _ in Self.length(Int.max) }
        let model = try ggufVisionSpec("org/vl-GGUF", "vl-Q4_0.gguf", "mmproj-vl-f16.gguf")
        #expect(await ModelFetch.declaredSizeBytes(model, http: vision) == .max)
    }

    /// A pinned revision is one path component, `/` included, as HubApi encodes it.
    @Test func anMlxPinIsOnePathComponent() async throws {
        var repo = try HfRepo.parseOrgRepo("org/m-MLX")
        repo.revision = try HfRevision("refs/pr/1")
        let model = Model.mlx(Mlx(source: .huggingFace(repo: repo, prefix: nil)))
        let (http, seen) = stub { _ in Self.listing([("model.safetensors", 1000)]) }

        #expect(await ModelFetch.declaredSizeBytes(model, http: http) == 1000)

        let url = try #require(seen.requests.first?.url)
        #expect(url.absoluteString.contains("/revision/refs%2Fpr%2F1?blobs=true"))
    }
}
