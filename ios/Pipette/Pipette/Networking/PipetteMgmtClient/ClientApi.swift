import Foundation

// Endpoints a device calls, signing with its ``ClientIdentity``: the crate's
// `client_api.rs`.
//
// No endpoint defaults its ``HttpClient``: a default would let a call site reach the
// process-shared client unseen, which is how a test ends up on the real network.
//
// The catalog reads are public (the mgmt repo's `docs/authentication.md` §4); a
// signature adds the entries private to the signer's organization. The catalog pull
// reads unsigned, unlike the crate's `sync`; a claim's one-id read signs, and fills a
// private entry the first time a run names it.

nonisolated extension PipetteMgmtClient {
    enum ClientApi {
        /// `POST /clients/register`. **Unauthenticated.**
        static func register(
            server: ServerUrl,
            http: HttpClient,
            request: RegisterRequest
        ) async throws(PipetteMgmtClient.Error) -> RegisterResponse {
            try await Call.sendJson(
                server: server, http: http, method: .post, path: "/clients/register", auth: nil,
                body: request)
        }

        /// `GET /clients/me`
        static func me(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity
        ) async throws(PipetteMgmtClient.Error) -> ClientProfile {
            try await Call.getJson(server: server, http: http, path: "/clients/me", auth: auth)
        }

        /// `PATCH /clients/me`
        static func updateMe(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            request: UpdateClientRequest
        ) async throws(PipetteMgmtClient.Error) -> ClientProfile {
            try await Call.sendJson(
                server: server, http: http, method: .patch, path: "/clients/me", auth: auth,
                body: request)
        }

        /// `GET /benchmarks`: definitions only, no eval samples. A `nil`
        /// `ifNoneMatch` is the unconditional request.
        static func listBenchmarksConditional(
            server: ServerUrl,
            http: HttpClient,
            ifNoneMatch: IfNoneMatch?
        ) async throws(PipetteMgmtClient.Error) -> ConditionalResponse<[BenchmarkSummary]> {
            try await Call.getJsonConditional(
                server: server, http: http, path: "/benchmarks", auth: nil,
                ifNoneMatch: ifNoneMatch)
        }

        /// `GET /benchmarks/{benchmark_id}`, with the eval `samples` the list
        /// omits. With `auth` nil, an id private to the device's organization is a `404`.
        static func getBenchmark(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity?,
            benchmarkId: String
        ) async throws(PipetteMgmtClient.Error) -> RemoteBenchmark {
            try await Call.getJson(
                server: server, http: http, path: "/benchmarks/\(pathSegment(benchmarkId))",
                auth: auth)
        }

        /// `GET /benchmarks/{benchmark_id}`
        static func getBenchmarkConditional(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity?,
            benchmarkId: String,
            ifNoneMatch: IfNoneMatch?
        ) async throws(PipetteMgmtClient.Error) -> ConditionalResponse<RemoteBenchmark> {
            try await Call.getJsonConditional(
                server: server, http: http, path: "/benchmarks/\(pathSegment(benchmarkId))",
                auth: auth, ifNoneMatch: ifNoneMatch)
        }

        /// `GET {path}`: a benchmark's corpus as a packed tar, at the route the
        /// catalog's `dataset.url` carries. The server owns that spelling.
        ///
        /// Returns the bytes unverified: a digest mismatch is a fact about the run,
        /// not a reason to refuse one (`Benchmarks/VlSpecDecodeCorpus.swift`).
        ///
        /// Signed, unlike the catalog reads: a corpus is work input, admitted to an
        /// approved client holding `perform_work`. Every refusal comes before the first
        /// byte as `.httpStatus`: `503` while the artifact is unbuilt or its upstream
        /// check is stale or unreachable, `502` once the upstream no longer backs the pin.
        ///
        /// A `503`'s `Retry-After` is not surfaced: the client already holds the
        /// job, so the job fails and is rescheduled rather than slept through.
        static func getDataset(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            path: String
        ) async throws(PipetteMgmtClient.Error) -> Data {
            let dispatched = try await Call.send(
                server: server, http: http, method: .get, path: path, auth: auth)
            guard (200..<300).contains(dispatched.response.status) else {
                throw Call.statusError(dispatched)
            }
            return dispatched.response.body
        }

        /// `GET /vault/resources/{resource_id}`: the pinned `sha256` and size, read
        /// before the bytes so a refusal or a disagreeing pin costs no download.
        static func getVaultResource(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            resourceId: VaultResourceId
        ) async throws(PipetteMgmtClient.Error) -> VaultResource {
            try await Call.getJson(
                server: server, http: http, path: "/vault/resources/\(resourceId.value)",
                auth: auth)
        }

        /// `GET /vault/resources/{resource_id}/content`, streamed to `sink` because an
        /// artifact can be many gigabytes. Unverified: the caller hashes what it wrote
        /// against the record's `sha256`. Every refusal comes before the first byte.
        ///
        /// With `resume`: a `206` if `If-Range` matches, a `200` with the whole object
        /// if it changed, and a `416` if the offset is at or past the end.
        ///
        /// `Range` and `If-Range` are outside the signature, which is safe: a range
        /// only narrows bytes the device can already fetch.
        ///
        /// The id goes on the path raw: `VaultResourceId` admits only `[A-Za-z0-9_-]`.
        static func getVaultContent(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            resourceId: VaultResourceId,
            resume: ResumeFrom? = nil,
            into sink: any ByteSink
        ) async throws(PipetteMgmtClient.Error) -> HttpResponse {
            try await Call.getStream(
                server: server, http: http,
                path: "/vault/resources/\(resourceId.value)/content", auth: auth,
                extra: resume?.headers ?? [], into: sink)
        }

        /// `POST /benchmarks`: one result, or a plan-attached failure.
        static func submitResult<P: Encodable>(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            payload: P
        ) async throws(PipetteMgmtClient.Error) -> SubmitResponse {
            try await Call.sendJson(
                server: server, http: http, method: .post, path: "/benchmarks", auth: auth,
                body: payload)
        }

        /// `POST /benchmarks` for a payload that is already JSON.
        static func submitResult(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            payloadJson: Data
        ) async throws(PipetteMgmtClient.Error) -> SubmitResponse {
            try await Call.sendJson(
                server: server, http: http, method: .post, path: "/benchmarks", auth: auth,
                bodyJson: payloadJson)
        }

        /// `POST /benchmarks/batch` as `{"submissions": [...]}`, assembled with
        /// `JSONSerialization`: string concatenation would silently make a malformed
        /// body from an empty or non-object payload.
        static func submitResultBatch(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            payloads: [Data]
        ) async throws(PipetteMgmtClient.Error) -> BatchSubmitResponse {
            let body: Data
            do {
                let submissions = try payloads.map {
                    try JSONSerialization.jsonObject(with: $0)
                }
                body = try JSONSerialization.data(withJSONObject: ["submissions": submissions])
            } catch {
                throw .encodeJson(error)
            }
            return try await Call.sendJson(
                server: server, http: http, method: .post, path: "/benchmarks/batch", auth: auth,
                bodyJson: body)
        }

        /// `GET /jobs/{job_id}`. The id comes off a stored receipt, so it is
        /// encoded (see ``pathSegment(_:)``).
        static func getJob(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            jobId: String
        ) async throws(PipetteMgmtClient.Error) -> JobResponse {
            try await Call.getJson(
                server: server, http: http, path: "/jobs/\(pathSegment(jobId))", auth: auth)
        }

        /// `POST /plans/claim`: the next eligible job, or `nil` on `204`.
        static func claim(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity
        ) async throws(PipetteMgmtClient.Error) -> ClaimedJob? {
            let dispatched = try await Call.send(
                server: server, http: http, method: .post, path: "/plans/claim", auth: auth)
            if dispatched.response.status == 204 { return nil }
            return try Call.decodeSuccess(dispatched)
        }

        /// `PUT /plans/{job_id}/heartbeat`. The id goes on the path raw: it is a
        /// lease handle the same server minted moments ago, as the crate echoes it.
        static func heartbeat(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            jobId: String
        ) async throws(PipetteMgmtClient.Error) {
            try await Call.sendEmpty(
                server: server, http: http, method: .put, path: "/plans/\(jobId)/heartbeat",
                auth: auth)
        }

        /// `POST /plans/{job_id}/reclaim`. The handle travels raw, as
        /// ``heartbeat(server:http:auth:jobId:)``'s does.
        static func reclaim(
            server: ServerUrl,
            http: HttpClient,
            auth: ClientIdentity,
            jobId: String
        ) async throws(PipetteMgmtClient.Error) {
            try await Call.sendEmpty(
                server: server, http: http, method: .post, path: "/plans/\(jobId)/reclaim",
                auth: auth)
        }

        /// An id encoded as one path segment, unlike the crate, which interpolates it
        /// raw: a legal id carrying `?` or `#` would otherwise become a query or a
        /// fragment.
        private static func pathSegment(_ raw: String) -> String {
            raw.addingPercentEncoding(
                withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "/?#"))
            ) ?? raw
        }
    }
}
