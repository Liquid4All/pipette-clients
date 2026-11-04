//! RFC 9421 HTTP Message Signatures in mgmt's one profile (`pipette-mgmt`'s
//! `docs/authentication.md` §2.4), which a **user** signs every request with:
//! Ed25519 over the method, authority, path, query and, when there is a body,
//! its RFC 9530 `Content-Digest`, with `created`, `nonce`, `keyid`,
//! `alg="ed25519"` and `tag="pipette"`.
//!
//! The server reads this profile and nothing else, so the shape is fixed here
//! rather than assembled from parts. mgmt's `crates/pipette-httpsig` and its
//! app's `ui/src/lib/httpsig.ts` pin one shared vector, and so does this module.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use sha2::{Digest, Sha256};

use pipette_plan_types::UserKeyId;

use crate::error::Result;

/// The components every signature covers, in the order it lists them.
const COMPONENTS: [&str; 4] = ["@method", "@authority", "@path", "@query"];

/// The component a request with a body adds.
const CONTENT_DIGEST: &str = "content-digest";

/// The label the two signature fields share. The server takes any label.
const LABEL: &str = "sig";

/// The request a user is about to send, as the server will see it.
pub(crate) struct Outgoing<'a> {
    pub method: &'a str,
    /// The assembled URL: its authority, path and query are what the server
    /// rebuilds from the request line and `Host`.
    pub url: &'a url::Url,
    pub body_sha256: Option<[u8; 32]>,
}

/// The fields a signature adds to a request.
pub(crate) struct Signed {
    pub signature_input: String,
    pub signature: String,
    pub content_digest: Option<String>,
}

/// Sign `request` as `key_id` at `created` (Unix seconds). `sign` holds the
/// key and signs the signature base as given.
pub(crate) fn sign(
    key_id: &UserKeyId,
    nonce: &str,
    created: i64,
    request: &Outgoing<'_>,
    sign: impl FnOnce(&[u8]) -> Result<[u8; 64]>,
) -> Result<Signed> {
    let content_digest = request.body_sha256.map(|digest| content_digest(&digest));
    let mut lines = vec![
        (COMPONENTS[0], request.method.to_string()),
        (COMPONENTS[1], authority(request.url)),
        (COMPONENTS[2], path(request.url)),
        (
            COMPONENTS[3],
            format!("?{}", request.url.query().unwrap_or("")),
        ),
    ];
    if let Some(field) = &content_digest {
        lines.push((CONTENT_DIGEST, field.clone()));
    }
    let names: Vec<String> = lines.iter().map(|(name, _)| quoted(name)).collect();
    let params = format!(
        "({});created={created};nonce={};keyid={};alg=\"ed25519\";tag=\"pipette\"",
        names.join(" "),
        quoted(nonce),
        quoted(key_id.as_ref()),
    );
    let signature = sign(signature_base(&lines, &params).as_bytes())?;
    Ok(Signed {
        signature_input: format!("{LABEL}={params}"),
        signature: format!("{LABEL}=:{}:", BASE64.encode(signature)),
        content_digest,
    })
}

pub(crate) fn content_digest(digest: &[u8; 32]) -> String {
    format!("sha-256=:{}:", BASE64.encode(digest))
}

pub(crate) fn sha256(body: &[u8]) -> [u8; 32] {
    Sha256::digest(body).into()
}

/// The `Host` reqwest sends: the host, and the port when the URL names one
/// other than its scheme's default. Lowercased, as RFC 9421 §2.2.3 derives it.
fn authority(url: &url::Url) -> String {
    let host = url.host_str().unwrap_or_default();
    match url.port() {
        Some(port) => format!("{host}:{port}"),
        None => host.to_string(),
    }
    .to_ascii_lowercase()
}

/// `/` for an empty path, as RFC 9421 §2.2.6 derives it.
fn path(url: &url::Url) -> String {
    match url.path() {
        "" => "/".to_string(),
        path => path.to_string(),
    }
}

/// An RFC 9651 string: double quotes, with `\` and `"` escaped.
fn quoted(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The RFC 9421 §2.5 signature base: one `"name": value` line per covered
/// component, then `@signature-params`.
fn signature_base(lines: &[(&str, String)], params: &str) -> String {
    lines
        .iter()
        .map(|(name, value)| format!("\"{name}\": {value}\n"))
        .chain(std::iter::once(format!("\"@signature-params\": {params}")))
        .collect()
}

#[cfg(test)]
pub(crate) mod test_support {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use ed25519_dalek::{Signature, VerifyingKey};

    use crate::error::{Error, Result};
    use crate::types::PublicKey;

    fn refuse(message: impl ToString) -> Error {
        Error::BuildClient(message.to_string())
    }

    /// Check a request's RFC 9421 fields as the server does: rebuild the base
    /// from what it carries and verify it against `public_key`.
    pub(crate) fn verify_request(
        public_key: &PublicKey,
        method: &str,
        url: &str,
        header: impl Fn(&str) -> Option<String>,
    ) -> Result<()> {
        let input = header("Signature-Input").ok_or_else(|| refuse("no Signature-Input"))?;
        let signature = header("Signature").ok_or_else(|| refuse("no Signature"))?;
        let params = input
            .strip_prefix("sig=")
            .ok_or_else(|| refuse("not labeled sig"))?;
        let signature = signature
            .strip_prefix("sig=:")
            .and_then(|rest| rest.strip_suffix(':'))
            .ok_or_else(|| refuse("not a sig byte sequence"))?;
        let url = url::Url::parse(url).map_err(refuse)?;
        let covered = params
            .split_once(')')
            .map(|(list, _)| list.trim_start_matches('('))
            .ok_or_else(|| refuse("no component list"))?;
        let lines = covered
            .split(' ')
            .map(|name| {
                let name = name.trim_matches('"');
                let value = match name {
                    "@method" => method.to_string(),
                    "@authority" => super::authority(&url),
                    "@path" => super::path(&url),
                    "@query" => format!("?{}", url.query().unwrap_or("")),
                    "content-digest" => {
                        header("Content-Digest").ok_or_else(|| refuse("no Content-Digest"))?
                    }
                    other => return Err(refuse(format!("unexpected component {other}"))),
                };
                Ok((name, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let base = super::signature_base(&lines, params);

        let public: [u8; 32] = hex::decode(public_key.as_ref())
            .map_err(refuse)?
            .try_into()
            .map_err(|_| refuse("public key must be 32 bytes"))?;
        let signature: [u8; 64] = BASE64
            .decode(signature)
            .map_err(refuse)?
            .try_into()
            .map_err(|_| refuse("signature must be 64 bytes"))?;
        VerifyingKey::from_bytes(&public)
            .map_err(refuse)?
            .verify_strict(base.as_bytes(), &Signature::from_bytes(&signature))
            .map_err(refuse)
    }

    /// An `httpmock` matcher for a request signed by `public_key` as the
    /// server checks one: the signature over what arrived, a body held to its
    /// digest, and none of the `v2` headers beside it. With `key_id`, the
    /// signature must name that key. A rejection says why on stderr, which
    /// httpmock's "no matching mock" does not.
    pub(crate) fn signed_by(
        public_key: PublicKey,
        key_id: Option<String>,
    ) -> impl Fn(&httpmock::HttpMockRequest) -> bool + Send + Sync + 'static {
        move |request| {
            let headers = request.headers();
            let header = |name: &str| {
                headers
                    .get(name)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string)
            };
            let uri = request.uri();
            let url = format!(
                "http://{}{}",
                header("host").unwrap_or_default(),
                uri.path_and_query().map_or("/", |target| target.as_str())
            );
            let body = request.body_ref();
            let digest_holds = match header("Content-Digest") {
                Some(field) => field == super::content_digest(&super::sha256(body)),
                None => body.is_empty(),
            };
            let no_v2 = ["X-Client-Id", "X-User-Key-Id", "X-Signature"]
                .iter()
                .all(|name| header(name).is_none());
            let names_the_key = key_id.as_ref().is_none_or(|key_id| {
                header("Signature-Input")
                    .is_some_and(|input| input.contains(&format!(";keyid=\"{key_id}\";")))
            });
            let verified = verify_request(&public_key, request.method_str(), &url, header)
                .inspect_err(|error| eprintln!("signed_by: {error}"))
                .is_ok();
            if !(digest_holds && no_v2 && names_the_key) {
                eprintln!(
                    "signed_by: digest holds {digest_holds}, no v2 header {no_v2}, names the key {names_the_key}"
                );
            }
            digest_holds && no_v2 && names_the_key && verified
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ed25519_dalek::{Signer, SigningKey};

    use super::*;
    use crate::error::Error;

    /// RFC 9421 Appendix B.1.4's `test-key-ed25519`, by its JWK `d`.
    fn rfc_key() -> anyhow::Result<SigningKey> {
        let d: [u8; 32] = URL_SAFE_NO_PAD
            .decode("n4Ni-HpISpVObnQMW0wOhCKROaIKqKtW_2ZYb2p9KcU")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("32 bytes"))?;
        Ok(SigningKey::from_bytes(&d))
    }

    fn signed(url: &str, body: Option<&[u8]>, key: &SigningKey) -> anyhow::Result<Signed> {
        let url = url::Url::parse(url)?;
        Ok(sign(
            &UserKeyId::try_new(
                "userkey_b16c2d1bead1262639764fdb0ee4d3774599336bd493404cda4b1136c59f2062",
            )?,
            "4f1c0de5a9b7",
            1_760_000_000,
            &Outgoing {
                method: "POST",
                url: &url,
                body_sha256: body.map(sha256),
            },
            |base| Ok(key.sign(base).to_bytes()),
        )?)
    }

    /// The vector mgmt's `crates/pipette-httpsig/src/lib.rs` and
    /// `ui/test/httpsig.test.ts` pin, byte for byte, so the three signers
    /// cannot drift.
    #[test]
    fn the_shared_vector_reproduces() -> anyhow::Result<()> {
        let signed = signed(
            "https://mgmt.example/api/v1/orgs?dry_run=true",
            Some(b"{\"slug\":\"acme\"}"),
            &rfc_key()?,
        )?;
        assert_eq!(
            signed.signature_input,
            r#"sig=("@method" "@authority" "@path" "@query" "content-digest");created=1760000000;nonce="4f1c0de5a9b7";keyid="userkey_b16c2d1bead1262639764fdb0ee4d3774599336bd493404cda4b1136c59f2062";alg="ed25519";tag="pipette""#
        );
        assert_eq!(
            signed.signature,
            "sig=:xErs0vVcglQO4jeofP53oNks9pyROP4E0OOttcbMlHt5CAanP85tQV21Sy4jOZEzJg3Bm9gRrGXvqDssOFy+Bg==:"
        );
        assert_eq!(
            signed.content_digest.as_deref(),
            Some("sha-256=:Dmcct3wNRuKmSGlN9J7lDIw9DvE0Xp6DUXXIZHWJ65k=:")
        );
        Ok(())
    }

    /// RFC 9530's example: `{"hello": "world"}` and its sha-256.
    #[test]
    fn the_rfc_content_digest_example_reproduces() {
        assert_eq!(
            content_digest(&sha256(b"{\"hello\": \"world\"}")),
            "sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"
        );
    }

    /// RFC 9421 Appendix B.2.6 byte for byte: the base this module builds, and
    /// the RFC's own signature verifying over it.
    #[test]
    fn the_rfc_ed25519_example_reproduces() -> anyhow::Result<()> {
        let lines = [
            ("date", "Tue, 20 Apr 2021 02:07:55 GMT".to_string()),
            ("@method", "POST".to_string()),
            ("@path", "/foo".to_string()),
            ("@authority", "example.com".to_string()),
            ("content-type", "application/json".to_string()),
            ("content-length", "18".to_string()),
        ];
        let base = signature_base(
            &lines,
            r#"("date" "@method" "@path" "@authority" "content-type" "content-length");created=1618884473;keyid="test-key-ed25519""#,
        );
        let signature: [u8; 64] = BASE64
            .decode("wqcAqbmYJ2ji2glfAMaRy4gruYYnx2nEFN2HN6jrnDnQCK1u02Gb04v9EDgwUPiu4A0w6vuQv5lIp5WPpBKRCw==")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("64 bytes"))?;
        rfc_key()?.verifying_key().verify_strict(
            base.as_bytes(),
            &ed25519_dalek::Signature::from_bytes(&signature),
        )?;
        Ok(())
    }

    /// The authority is what reqwest sends as `Host`: a non-default port kept,
    /// a default one dropped, lowercased. The path and query are as sent, `?`
    /// alone for none.
    #[rstest::rstest]
    #[case(
        "https://Mgmt.Example/api/v1/plans",
        "mgmt.example",
        "/api/v1/plans",
        "?"
    )]
    #[case("https://mgmt.example:443/x", "mgmt.example", "/x", "?")]
    #[case(
        "http://127.0.0.1:8080/x?a=1&b=c%20d",
        "127.0.0.1:8080",
        "/x",
        "?a=1&b=c%20d"
    )]
    #[case("http://[::1]:9000/x?", "[::1]:9000", "/x", "?")]
    fn components_are_what_the_server_rebuilds(
        #[case] url: &str,
        #[case] authority_value: &str,
        #[case] path_value: &str,
        #[case] query: &str,
    ) -> anyhow::Result<()> {
        let parsed = url::Url::parse(url)?;
        assert_eq!(authority(&parsed), authority_value);
        assert_eq!(path(&parsed), path_value);
        assert_eq!(format!("?{}", parsed.query().unwrap_or("")), query);
        Ok(())
    }

    /// A bodiless request covers no digest and sends none.
    #[test]
    fn no_body_covers_no_digest() -> anyhow::Result<()> {
        let signed = signed("https://mgmt.example/api/v1/users/me", None, &rfc_key()?)?;
        assert!(signed.content_digest.is_none());
        assert!(
            signed
                .signature_input
                .starts_with(r#"sig=("@method" "@authority" "@path" "@query");"#),
            "{}",
            signed.signature_input
        );
        Ok(())
    }

    /// The holder's refusal comes back as it was, nothing signed.
    #[test]
    fn a_refusing_holder_stops_the_signature() -> anyhow::Result<()> {
        let url = url::Url::parse("https://mgmt.example/x")?;
        let result = sign(
            &UserKeyId::try_new(
                "userkey_b16c2d1bead1262639764fdb0ee4d3774599336bd493404cda4b1136c59f2062",
            )?,
            "n",
            0,
            &Outgoing {
                method: "GET",
                url: &url,
                body_sha256: None,
            },
            |_| Err(Error::BuildClient("locked".to_string())),
        );
        assert!(matches!(result, Err(Error::BuildClient(reason)) if reason == "locked"));
        Ok(())
    }
}
