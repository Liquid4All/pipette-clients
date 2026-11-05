import Foundation

// iOS's counterpart of `crates/pipette-cli/src/hf_auth.rs`. The crate has one tier
// (`PIPETTE_HF_TOKEN`); a device has three, because a claim can carry a repo's token and an
// interrupted transfer outlives the claim that brought it.

/// A `claimToken` wins over the Keychain: a claim that carries a credential names the one
/// that can read that repo, and falling back to whatever the device has stored answers 401
/// for a gated repo the plan could have opened. The stored tier covers a transfer whose
/// in-memory claim is gone (a relaunch). The user's own token is written only by
/// `auth import`.
///
/// Both transports resolve through here, so a repo that opens for one format can't 401 for
/// the other. Why iOS stores a token at all is in `docs/pipette-ios/execution-alignment.md`.
func resolveHfToken(claimToken: AuthToken?, model: Model?) -> HfCredential? {
    if let claimToken { return HfCredential(token: claimToken, source: .claim) }
    if let stored = model.flatMap({ KeychainHelper.loadHfToken(forModel: $0) }) {
        return HfCredential(token: stored, source: .stored)
    }
    return KeychainHelper.loadHfToken().map { HfCredential(token: $0, source: .user) }
}

/// The tier is carried so the log line can name it: `source=claim` shows a plan's token
/// reached the fetch rather than the device quietly falling back to something else.
struct HfCredential {
    let token: AuthToken
    let source: HfTokenSource
}

/// Most specific first. `rawValue` is the log spelling.
enum HfTokenSource: String {
    case claim
    /// Stored for this model when its claim arrived, for a transfer that outlived its claim.
    case stored
    /// The device's own token, iOS's analogue of `PIPETTE_HF_TOKEN`.
    case user
}

/// Only huggingface.co hosts get the token, so it isn't leaked to third parties.
///
/// Logs which it did: an anonymous fetch of a public repo looks like an authenticated one
/// until HF starts answering 429. Only the tier's name is logged, never the token.
func attachHfAuth(_ request: inout URLRequest, claimToken: AuthToken? = nil, model: Model? = nil) {
    guard let host = request.url?.host, host.hasSuffix("huggingface.co") else { return }
    let path = request.url?.path ?? "?"
    guard let credential = resolveHfToken(claimToken: claimToken, model: model) else {
        AppLog.storage.info("hf auth: no token available — requesting \(host)\(path) anonymously")
        return
    }
    request.setValue("Bearer \(credential.token.value)", forHTTPHeaderField: "Authorization")
    AppLog.storage.info(
        "hf auth: attached bearer token source=\(credential.source.rawValue) to \(host)\(path)")
}
