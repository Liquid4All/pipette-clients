import Foundation

// The compact `model=` URI, ported from `crates/pipette-cli/src/model_uri.rs`:
// `parseModelUri`, its inverse `modelToUri`, and the `model=` dispatch `parseModelArg`.
// The grammar and keys are in docs/pipette-cli/models-and-runtimes.md.
//
// `&` separates pairs, so no value may contain `&`, and a URL value may not contain `?`
// (use the JSON form). Whether a phone can fetch what a URI names is decided later
// (`Model.isFetchableHere`).

/// Shared by the parser and `modelToUri`, so a rename cannot desync them.
private nonisolated enum Key {
    static let repo = "repo"
    static let rev = "rev"
    static let path = "path"
    static let url = "url"
    static let sha256 = "sha256"
    static let model = "model"
    static let mmproj = "mmproj"
    static let modelSha256 = "model_sha256"
    static let mmprojSha256 = "mmproj_sha256"
    static let draft = "draft"
    static let draftSha256 = "draft_sha256"
    static let prefix = "prefix"
}

/// Parsing and rendering both route through it, so neither direction can silently drop a kind.
private nonisolated enum Scheme: String {
    case ggufText = "gguf-text"
    case ggufVision = "gguf-vision"
    case ggufVisionSpeculative = "gguf-vision-speculative"
    case mlx
    case torch
    case openvino
    case onnxGenai = "onnx-genai"

    /// `apple-foundation`, `local` and `file` name real sources that are not importable, so
    /// they are told apart from a typo. A `file://` value is unaffected: the scheme is what
    /// precedes the first `://`.
    static func parse(_ s: String) throws(ModelUriError) -> Scheme {
        if let scheme = Scheme(rawValue: s) { return scheme }
        switch s {
        case "apple-foundation", "local", "file": throw .notImportable(s)
        default: throw .unknownScheme(s)
        }
    }
}

nonisolated enum ModelUriError: Error, Equatable, LocalizedError {
    case missingSchemeSeparator
    case unknownScheme(String)
    /// iOS-only: `torch` and `openvino` are the crate's schemes but have no `Model` case here.
    case unsupportedScheme(String)
    case notImportable(String)
    case malformedPair(String)
    case emptyKey
    case duplicateKey(String)
    case unknownKey(scheme: String, key: String)
    case missingKey(scheme: String, key: String)
    case mutuallyExclusive(scheme: String, expected: String)
    case queryUrl(key: String)
    case secretNotRepresentable
    case vaultNotRepresentable
    case invalidValue(key: String, message: String)

    var errorDescription: String? {
        switch self {
        case .missingSchemeSeparator:
            "model URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`"
        case let .unknownScheme(scheme):
            "unknown model URI scheme `\(scheme)` (expected `gguf-text`, `gguf-vision`, `mlx`, "
                + "`torch`, `openvino`, or `onnx-genai`)"
        case let .unsupportedScheme(scheme):
            "model URI scheme `\(scheme)` names a model kind this build has no engine for"
        case let .notImportable(source):
            "model source `\(source)` is not importable"
        case let .malformedPair(pair):
            "malformed `key=value` pair `\(pair)` in model URI"
        case .emptyKey:
            "empty key in model URI"
        case let .duplicateKey(key):
            "duplicate key `\(key)` in model URI"
        case let .unknownKey(scheme, key):
            "unknown key `\(key)` for scheme `\(scheme)`"
        case let .missingKey(scheme, key):
            "missing required key `\(key)` for scheme `\(scheme)`"
        case let .mutuallyExclusive(scheme, expected):
            "scheme `\(scheme)` requires exactly one of \(expected)"
        case let .queryUrl(key):
            "query-string URL in key `\(key)` is unsupported in the URI form (the `&` separator "
                + "can't carry query params). Pass a JSON `model=` object instead"
        case .secretNotRepresentable:
            "a model with an auth token has no URI form (a URI can't carry a secret). "
                + "Pass a JSON `model=` object instead"
        case .vaultNotRepresentable:
            "a vault source has no URI form (the grammar has no key for its principal). "
                + "Pass a JSON `model=` object instead"
        case let .invalidValue(key, message):
            "invalid value for key `\(key)`: \(message)"
        }
    }
}

/// The first `=` in each segment splits, so a value keeps any later `=`. URL values may not
/// carry a query string, so every `&` is a real pair boundary and a leftover key is a typo.
private nonisolated struct Pairs {
    let scheme: Scheme
    private var items: [(key: String, value: String)] = []

    init(scheme: Scheme, body: String) throws(ModelUriError) {
        self.scheme = scheme
        // An empty body carries no pairs; the scheme's `require` calls then report a
        // precise missing key rather than a malformed empty pair.
        guard !body.isEmpty else { return }
        for segment in body.split(separator: "&", omittingEmptySubsequences: false) {
            guard let eq = segment.firstIndex(of: "=") else {
                throw .malformedPair(String(segment))
            }
            let key = String(segment[..<eq])
            guard !key.isEmpty else { throw .emptyKey }
            guard !items.contains(where: { $0.key == key }) else { throw .duplicateKey(key) }
            items.append((key, String(segment[segment.index(after: eq)...])))
        }
    }

    mutating func take(_ key: String) -> String? {
        items.firstIndex { $0.key == key }.map { items.remove(at: $0).value }
    }

    mutating func require(_ key: String) throws(ModelUriError) -> String {
        guard let value = take(key) else { throw .missingKey(scheme: scheme.rawValue, key: key) }
        return value
    }

    func finish() throws(ModelUriError) {
        if let leftover = items.first {
            throw .unknownKey(scheme: scheme.rawValue, key: leftover.key)
        }
    }
}

private nonisolated func message(_ error: any Error) -> String {
    (error as? LocalizedError)?.errorDescription ?? String(describing: error)
}

private nonisolated func hfRepo(_ orgRepo: String, rev: String?) throws(ModelUriError) -> HfRepo {
    var repo: HfRepo
    do {
        repo = try HfRepo.parseOrgRepo(orgRepo)
    } catch {
        throw .invalidValue(key: Key.repo, message: message(error))
    }
    if let rev {
        do {
            repo.revision = try HfRevision(rev)
        } catch {
            throw .invalidValue(key: Key.rev, message: message(error))
        }
    }
    return repo
}

private nonisolated func repoSubpath(_ key: String, _ value: String) throws(ModelUriError) -> RepoSubpath {
    do { return try RepoSubpath(value) } catch { throw .invalidValue(key: key, message: message(error)) }
}

/// A query string cannot survive the `&` split: `url=…?a=x&b=y` fractures into `url=…?a=x`
/// plus a bogus `b=y` pair, which, if `b` is a consumed key like `sha256`, would silently
/// truncate the URL and lift a fake value. Refused up front.
private nonisolated func resourceUrl(_ key: String, _ value: String) throws(ModelUriError) -> ResourceUrl {
    guard !value.contains("?") else { throw .queryUrl(key: key) }
    do { return try ResourceUrl(value) } catch { throw .invalidValue(key: key, message: message(error)) }
}

private nonisolated func optSha256(_ key: String, _ value: String?) throws(ModelUriError) -> Sha256? {
    guard let value else { return nil }
    do { return try Sha256(value) } catch { throw .invalidValue(key: key, message: message(error)) }
}

private nonisolated func parseGgufText(_ pairs: Pairs) throws(ModelUriError) -> Model {
    var p = pairs
    let source: GgufTextSource
    switch (p.take(Key.repo), p.take(Key.url)) {
    case let (repo?, nil):
        let rev = p.take(Key.rev)
        let path = try p.require(Key.path)
        let sha256 = p.take(Key.sha256)
        source = .huggingFace(repo: try hfRepo(repo, rev: rev),
                              path: try repoSubpath(Key.path, path),
                              sha256: try optSha256(Key.sha256, sha256))
    // `rev` has no meaning for a raw URL, so `finish` refuses it.
    case let (nil, url?):
        let sha256 = p.take(Key.sha256)
        source = .url(url: try resourceUrl(Key.url, url),
                      sha256: try optSha256(Key.sha256, sha256))
    default:
        throw .mutuallyExclusive(scheme: p.scheme.rawValue, expected: "`repo`+`path` or `url`")
    }
    try p.finish()
    return .ggufText(GgufText(source: source))
}

private nonisolated func parseGgufVision(_ pairs: Pairs) throws(ModelUriError) -> Model {
    var p = pairs
    let source = try parseGgufVisionFiles(&p)
    try p.finish()
    return .ggufVision(GgufVision(source: source))
}

/// The drafter is resolved against the same source as the weights (a repo-relative path
/// under `repo`, else a URL) because it is part of the same model.
private nonisolated func parseGgufVisionSpeculative(
    _ pairs: Pairs
) throws(ModelUriError) -> Model {
    var p = pairs
    let files = try parseGgufVisionFiles(&p)
    let draft = p.take(Key.draft)
    let draftSha = try optSha256(Key.draftSha256, p.take(Key.draftSha256))
    let source: GgufVisionSpeculativeSource
    switch files {
    case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha):
        source = .huggingFace(
            repo: repo, model: model, modelSha256: modelSha,
            mmproj: mmproj, mmprojSha256: mmprojSha,
            draft: try draft.map { (value) throws(ModelUriError) in
                try repoSubpath(Key.draft, value)
            },
            draftSha256: draftSha)
    case let .url(model, modelSha, mmproj, mmprojSha):
        source = .url(
            model: model, modelSha256: modelSha, mmproj: mmproj, mmprojSha256: mmprojSha,
            draft: try draft.map { (value) throws(ModelUriError) in
                try resourceUrl(Key.draft, value)
            },
            draftSha256: draftSha)
    // `parseGgufVisionFiles` builds only the two importable arms.
    case .relativeFiles, .absoluteFiles:
        throw .notImportable("local")
    case .vaultUrl:
        throw .vaultNotRepresentable
    }
    try p.finish()
    return .ggufVisionSpeculative(GgufVisionSpeculative(source: source))
}

/// Leaves `p` open so a caller can take the keys its own scheme adds.
private nonisolated func parseGgufVisionFiles(
    _ p: inout Pairs
) throws(ModelUriError) -> GgufVisionSource {
    let source: GgufVisionSource
    if let repo = p.take(Key.repo) {
        let rev = p.take(Key.rev)
        let model = try p.require(Key.model)
        let mmproj = try p.require(Key.mmproj)
        let modelSha = p.take(Key.modelSha256)
        let mmprojSha = p.take(Key.mmprojSha256)
        source = .huggingFace(
            repo: try hfRepo(repo, rev: rev),
            model: try repoSubpath(Key.model, model),
            modelSha256: try optSha256(Key.modelSha256, modelSha),
            mmproj: try repoSubpath(Key.mmproj, mmproj),
            mmprojSha256: try optSha256(Key.mmprojSha256, mmprojSha))
    } else {
        let model = try p.require(Key.model)
        let mmproj = try p.require(Key.mmproj)
        let modelSha = p.take(Key.modelSha256)
        let mmprojSha = p.take(Key.mmprojSha256)
        source = .url(
            model: try resourceUrl(Key.model, model),
            modelSha256: try optSha256(Key.modelSha256, modelSha),
            mmproj: try resourceUrl(Key.mmproj, mmproj),
            mmprojSha256: try optSha256(Key.mmprojSha256, mmprojSha))
    }
    return source
}

/// The crate's `parse_dir_source`, shared by `mlx` and `onnx-genai`.
private nonisolated func parseDirSource(_ pairs: Pairs) throws(ModelUriError) -> ModelSource {
    var p = pairs
    let repo = try p.require(Key.repo)
    let prefix = p.take(Key.prefix)
    let rev = p.take(Key.rev)
    let source = ModelSource.huggingFace(
        repo: try hfRepo(repo, rev: rev),
        prefix: try prefix.map { (value) throws(ModelUriError) in try repoSubpath(Key.prefix, value) })
    try p.finish()
    return source
}

/// Only importable sources have a scheme; on-disk and Apple Foundation sources are refused.
nonisolated func parseModelUri(_ input: String) throws(ModelUriError) -> Model {
    guard let split = input.range(of: "://") else { throw .missingSchemeSeparator }
    let scheme = try Scheme.parse(String(input[..<split.lowerBound]))
    let pairs = try Pairs(scheme: scheme, body: String(input[split.upperBound...]))
    switch scheme {
    case .ggufText: return try parseGgufText(pairs)
    case .ggufVision: return try parseGgufVision(pairs)
    case .ggufVisionSpeculative: return try parseGgufVisionSpeculative(pairs)
    case .mlx: return .mlx(Mlx(source: try parseDirSource(pairs)))
    case .onnxGenai: return .onnxGenai(OnnxGenai(source: try parseDirSource(pairs)))
    case .torch, .openvino: throw .unsupportedScheme(scheme.rawValue)
    }
}

private nonisolated struct Body {
    let scheme: Scheme
    private var pairs: [String] = []

    init(_ scheme: Scheme) { self.scheme = scheme }

    mutating func push(_ key: String, _ value: String) { pairs.append("\(key)=\(value)") }

    /// A repo carrying a token is refused rather than rendered without it: a URI cannot hold
    /// a secret, and dropping it would print a reference that fetches without its credential.
    mutating func repo(_ repo: HfRepo) throws(ModelUriError) {
        guard repo.authToken == nil else { throw .secretNotRepresentable }
        push(Key.repo, repo.description)
    }

    mutating func rev(_ repo: HfRepo) {
        if let revision = repo.revision { push(Key.rev, revision.value) }
    }

    mutating func sha(_ key: String, _ sha: Sha256?) {
        if let sha { push(key, sha.value) }
    }

    /// A URL with an `&` or a `?` would fracture on re-parse, so it is refused like the
    /// parser refuses it.
    mutating func url(_ key: String, _ url: ResourceUrl) throws(ModelUriError) {
        guard !url.value.contains("&"), !url.value.contains("?") else { throw .queryUrl(key: key) }
        push(key, url.value)
    }

    func finish() -> String { "\(scheme.rawValue)://" + pairs.joined(separator: "&") }
}

/// The key order matches the crate's so one model renders one string whichever client
/// writes it. What no URI can name is refused with the parser's own errors.
nonisolated func modelToUri(_ model: Model) throws(ModelUriError) -> String {
    switch model {
    case let .ggufText(m):
        var body = Body(.ggufText)
        switch m.source {
        case let .huggingFace(repo, path, sha256):
            try body.repo(repo)
            body.push(Key.path, path.value)
            body.rev(repo)
            body.sha(Key.sha256, sha256)
        case let .url(url, sha256):
            try body.url(Key.url, url)
            body.sha(Key.sha256, sha256)
        case .vaultUrl:
            throw .vaultNotRepresentable
        case .relativeFile, .absoluteFile:
            throw .notImportable("local")
        }
        return body.finish()
    case let .ggufVision(m):
        var body = Body(.ggufVision)
        switch m.source {
        case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha):
            try body.repo(repo)
            body.push(Key.model, model.value)
            body.push(Key.mmproj, mmproj.value)
            body.rev(repo)
            body.sha(Key.modelSha256, modelSha)
            body.sha(Key.mmprojSha256, mmprojSha)
        case let .url(model, modelSha, mmproj, mmprojSha):
            try body.url(Key.model, model)
            try body.url(Key.mmproj, mmproj)
            body.sha(Key.modelSha256, modelSha)
            body.sha(Key.mmprojSha256, mmprojSha)
        case .vaultUrl:
            throw .vaultNotRepresentable
        case .relativeFiles, .absoluteFiles:
            throw .notImportable("local")
        }
        return body.finish()
    case let .ggufVisionSpeculative(m):
        var body = Body(.ggufVisionSpeculative)
        switch m.source {
        case let .huggingFace(repo, model, modelSha, mmproj, mmprojSha, draft, draftSha):
            try body.repo(repo)
            body.push(Key.model, model.value)
            body.push(Key.mmproj, mmproj.value)
            if let draft { body.push(Key.draft, draft.value) }
            body.rev(repo)
            body.sha(Key.modelSha256, modelSha)
            body.sha(Key.mmprojSha256, mmprojSha)
            body.sha(Key.draftSha256, draftSha)
        case let .url(model, modelSha, mmproj, mmprojSha, draft, draftSha):
            try body.url(Key.model, model)
            try body.url(Key.mmproj, mmproj)
            if let draft { try body.url(Key.draft, draft) }
            body.sha(Key.modelSha256, modelSha)
            body.sha(Key.mmprojSha256, mmprojSha)
            body.sha(Key.draftSha256, draftSha)
        case .vaultUrl:
            throw .vaultNotRepresentable
        case .relativeFiles, .absoluteFiles:
            throw .notImportable("local")
        }
        return body.finish()
    case let .mlx(m):
        if case .vaultArchive = m.source { throw .vaultNotRepresentable }
        guard case let .huggingFace(repo, prefix) = m.source else { throw .notImportable("local") }
        var body = Body(.mlx)
        try body.repo(repo)
        if let prefix { body.push(Key.prefix, prefix.value) }
        body.rev(repo)
        return body.finish()
    case let .onnxGenai(m):
        if case .vaultArchive = m.source { throw .vaultNotRepresentable }
        guard case let .huggingFace(repo, prefix) = m.source else { throw .notImportable("local") }
        var body = Body(.onnxGenai)
        try body.repo(repo)
        if let prefix { body.push(Key.prefix, prefix.value) }
        body.rev(repo)
        return body.finish()
    case .appleFoundationText:
        throw .notImportable("apple-foundation")
    }
}

/// A JSON `Model` when the trimmed argument starts with `{`, else the compact URI.
nonisolated func parseModelArg(_ arg: String) throws -> Model {
    let trimmed = arg.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.hasPrefix("{") {
        return try JSONDecoder().decode(Model.self, from: Data(trimmed.utf8))
    }
    return try parseModelUri(trimmed)
}
