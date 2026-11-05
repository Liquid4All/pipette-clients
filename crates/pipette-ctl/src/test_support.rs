//! Helpers the unit tests of more than one module share.

use httpmock::HttpMockRequest;

/// Signed RFC 9421 by `key_id`, which alone names the account, with none of
/// the `v2` headers.
pub(crate) fn signed_as(request: &HttpMockRequest, key_id: &str) -> bool {
    let headers = request.headers();
    let keyid = format!(";keyid=\"{key_id}\";");
    headers
        .get("Signature-Input")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|input| input.contains(&keyid))
        && ["X-Client-Id", "X-User-Key-Id", "X-Signature"]
            .iter()
            .all(|name| !headers.contains_key(*name))
}
