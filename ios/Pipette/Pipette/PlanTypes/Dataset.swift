import Foundation

// A benchmark's corpus: pipette-mgmt's `crates/pipette-plan-types/src/dataset.rs`.
//
// A client only reads this: it downloads `Dataset.url`, checks the bytes against
// `Dataset.digest`, and traverses what unpacks (`Benchmarks/VlSpecDecodeCorpus.swift`).
// The whole upstream pin is kept, not just the download fields, because the catalog entry
// is what says which corpus a result measured.

/// A git object id: 40 lowercase hex characters. A tag moves and an abbreviation is
/// ambiguous, so either would make the corpus unreproducible.
nonisolated struct GitOid: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /[0-9a-f]{40}/) != nil
    }
    static let rejection = ModelError.invalidGitOid
}

/// A scheme-less `host/path` repository coordinate; an `http(s)://` prefix is stripped
/// before validation (the crate's `strip_http_scheme`) so one upstream has one spelling.
///
/// Kept apart from `RepositoryUrl`, as the crate keeps them: a runtime's coordinate
/// sanitizes infallibly, while a corpus upstream's validates, since a coordinate that is
/// not `host/path` makes the pin unresolvable.
nonisolated struct DatasetRepositoryUrl: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    /// Other schemes are left intact so validation can reject them.
    static func sanitize(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        for scheme in ["https://", "http://"] where trimmed.hasPrefix(scheme) {
            return String(trimmed.dropFirst(scheme.count))
        }
        return trimmed
    }
    static func validate(_ value: String) -> Bool {
        !value.isEmpty && !value.contains("://") && !value.hasPrefix("/")
            && !value.hasPrefix(".") && !value.hasPrefix("-") && value.contains("/")
            && !value.contains(where: \.isWhitespace)
    }
    static let rejection = ModelError.invalidRepositoryUrl
}

/// A path relative to a repository root: the crate's `is_dataset_path`, the same predicate
/// as `RepoSubpath` but a separate type, as in the crate, so a corpus path and a model-repo
/// path cannot be swapped.
nonisolated struct DatasetPath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func sanitize(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    static func validate(_ value: String) -> Bool {
        !value.isEmpty && !value.hasPrefix("/")
            && value.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
                    && $0.wholeMatch(of: /[A-Za-z0-9._-]+/) != nil
            }
    }
    static let rejection = ModelError.invalidDatasetPath
}

/// The layout a corpus is repacked into. Decoded strictly: a client that cannot read the
/// layout must fail at the catalog rather than half-way through a multi-hour traversal.
nonisolated enum CorpusFormat: String, Codable, Hashable, Sendable, CaseIterable {
    /// An uncompressed tar of `manifest.json`, `samples.jsonl` and content-addressed
    /// `media/`.
    case vlSpecDecodeCorpusV1 = "vl-spec-decode-corpus/1"
}

/// The server-side reader of one upstream's layout. Opaque to a client, carried so a
/// result's provenance is whole.
nonisolated enum DatasetAdapter: String, Codable, Hashable, Sendable, CaseIterable {
    /// MMSpec's `mmspec.jsonl` + `images/`.
    case mmspecV1 = "mmspec@1"
}

/// Where a corpus lives upstream, pinned. Tagged by `source`.
nonisolated enum DatasetUpstream: Hashable, Sendable, Codable {
    /// A git repository at one commit, narrowed to `subpath`.
    case git(url: DatasetRepositoryUrl, revision: GitOid, subpath: DatasetPath?,
             identity: GitOid, digest: Sha256, license: String?)

    /// `url@revision[:subpath]`: the crate's `reference()`.
    var reference: String {
        switch self {
        case let .git(url, revision, subpath, _, _, _):
            if let subpath {
                "\(url.value)@\(revision.value):\(subpath.value)"
            } else {
                "\(url.value)@\(revision.value)"
            }
        }
    }

    /// Recorded, not interpreted.
    var license: String? {
        switch self {
        case let .git(_, _, _, _, _, license): license
        }
    }

    private enum CodingKeys: String, CodingKey {
        case source, url, revision, subpath, identity, digest, license
    }

    private enum SourceTag: String { case git }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .source)
        guard let tag = SourceTag(rawValue: raw) else {
            throw DatasetError.unknownUpstreamSource(raw)
        }
        switch tag {
        case .git:
            self = .git(
                url: try c.decode(DatasetRepositoryUrl.self, forKey: .url),
                revision: try c.decode(GitOid.self, forKey: .revision),
                subpath: try c.decodeIfPresent(DatasetPath.self, forKey: .subpath),
                identity: try c.decode(GitOid.self, forKey: .identity),
                digest: try c.decode(Sha256.self, forKey: .digest),
                license: try c.decodeIfPresent(String.self, forKey: .license))
        }
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .git(url, revision, subpath, identity, digest, license):
            try c.encode(SourceTag.git.rawValue, forKey: .source)
            try c.encode(url, forKey: .url)
            try c.encode(revision, forKey: .revision)
            // Absent rather than null, matching the crate's `skip_serializing_if`.
            try c.encodeIfPresent(subpath, forKey: .subpath)
            try c.encode(identity, forKey: .identity)
            try c.encode(digest, forKey: .digest)
            try c.encodeIfPresent(license, forKey: .license)
        }
    }
}

/// A benchmark's corpus, as the catalog pins it and the server serves it.
///
/// Unknown fields are ignored, as in the crate: a field a newer server adds must not make
/// the definition unreadable to a client that would otherwise run it correctly.
nonisolated struct Dataset: Hashable, Sendable, Codable {
    let format: CorpusFormat
    let adapter: DatasetAdapter
    /// sha256 of the packed tar: what a client verifies and submits back as
    /// `observation_dataset_digest`.
    let digest: Sha256
    let bytes: UInt64
    /// The one subset of the upstream this corpus keeps, as the adapter names it (`GQA`,
    /// `CharXiv`, …); absent = the whole upstream.
    let subset: String?
    let upstream: DatasetUpstream
    /// A server-relative download path (`/datasets/{benchmark_id}`) that
    /// `GET /benchmarks/{id}` stamps on the way out; the catalog entry itself has none. A
    /// definition without one (a hand-authored `local/` entry) fails at run time, not parse.
    let url: NonEmptyString?

    init(format: CorpusFormat, adapter: DatasetAdapter, digest: Sha256, bytes: UInt64,
         subset: String? = nil, upstream: DatasetUpstream, url: NonEmptyString? = nil) {
        self.format = format
        self.adapter = adapter
        self.digest = digest
        self.bytes = bytes
        self.subset = subset
        self.upstream = upstream
        self.url = url
    }

    private enum CodingKeys: String, CodingKey {
        case format, adapter, digest, bytes, subset, upstream, url
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        format = try c.decode(CorpusFormat.self, forKey: .format)
        adapter = try c.decode(DatasetAdapter.self, forKey: .adapter)
        digest = try c.decode(Sha256.self, forKey: .digest)
        bytes = try c.decode(UInt64.self, forKey: .bytes)
        subset = try c.decodeIfPresent(String.self, forKey: .subset)
        upstream = try c.decode(DatasetUpstream.self, forKey: .upstream)
        url = try c.decodeIfPresent(NonEmptyString.self, forKey: .url)
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(format, forKey: .format)
        try c.encode(adapter, forKey: .adapter)
        try c.encode(digest, forKey: .digest)
        try c.encode(bytes, forKey: .bytes)
        try c.encodeIfPresent(subset, forKey: .subset)
        try c.encode(upstream, forKey: .upstream)
        try c.encodeIfPresent(url, forKey: .url)
    }
}

/// A catalog corpus entry this client cannot use. `VlSpecDecodeCorpusError` covers the
/// bytes.
nonisolated enum DatasetError: Error, Equatable, LocalizedError {
    case unknownUpstreamSource(String)
    case noDownloadUrl

    var errorDescription: String? {
        switch self {
        case let .unknownUpstreamSource(s): "unknown corpus upstream source: \(s)"
        case .noDownloadUrl:
            "the dataset names no download URL, so there is nothing to fetch; a corpus is "
                + "served by the management server alone"
        }
    }
}
