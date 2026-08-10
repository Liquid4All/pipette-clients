use ed25519_dalek::{Signature, Signer, SigningKey};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct AuthIdentity {
    pub client_id: String,
    pub private_key_hex: String,
}

/// Build the signed management-auth headers (`X-Client-Id`,
/// `X-Timestamp`, `X-Nonce`, `X-Signature`) as plain name/value pairs. The
/// values are an Ed25519 client id, an RFC3339 timestamp, a per-request nonce,
/// and a hex signature over the [`signed_payload`] below — all ASCII, so the
/// transport can set them verbatim.
///
/// `path_and_query` must be the request target the server receives —
/// including any path prefix carried by the base URL, and the query string
/// when there is one. The server verifies against its own
/// `uri.path_and_query()`, so anything else fails to verify.
pub fn signed_headers(
    identity: &AuthIdentity,
    method: &str,
    path_and_query: &str,
) -> Result<Vec<(String, String)>> {
    let timestamp = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(Error::TimestampFormat)?;
    let nonce = generate_nonce()?;
    let payload = signed_payload(
        method,
        path_and_query,
        &timestamp,
        &identity.client_id,
        &nonce,
    );
    let signature = sign(identity.private_key_hex.trim(), &payload)?;

    Ok(vec![
        ("X-Client-Id".to_string(), identity.client_id.clone()),
        ("X-Timestamp".to_string(), timestamp),
        ("X-Nonce".to_string(), nonce),
        ("X-Signature".to_string(), signature),
    ])
}

/// A fresh per-request nonce: 16 CSPRNG bytes, lowercase hex.
///
/// Hex rather than an arbitrary byte string on purpose. The nonce is a field in
/// a newline-delimited payload, so a value carrying a newline could forge a
/// field boundary and make two different requests hash to one payload; hex
/// cannot. It also satisfies the server's non-empty and valid-UTF-8 rules by
/// construction. 128 bits makes a collision across the fleet negligible, so the
/// server's replay cache can reject a repeat without the client coordinating.
fn generate_nonce() -> Result<String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(Error::OsEntropy)?;
    Ok(hex::encode(bytes))
}

/// The `v1` signed payload: six newline-separated fields — scheme tag,
/// method, request target, timestamp, client id, nonce (mgmt
/// `authentication.md` §2.1). Binding the method and target scopes a signature
/// to that method and target; the nonce makes it single-use, so a captured
/// signature cannot be replayed inside the freshness window. The request body
/// is still not covered.
fn signed_payload(
    method: &str,
    path_and_query: &str,
    timestamp: &str,
    client_id: &str,
    nonce: &str,
) -> String {
    format!("v1\n{method}\n{path_and_query}\n{timestamp}\n{client_id}\n{nonce}")
}

/// Generate a new Ed25519 keypair and return (private_key_hex, public_key_hex).
/// An Ed25519 signing key is its 32-byte seed, so we read it straight from the
/// OS CSPRNG — no `rand` adapter needed.
pub fn generate_keypair_hex() -> Result<(String, String)> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).map_err(Error::OsEntropy)?;
    let signing = SigningKey::from_bytes(&seed);
    let verifying = signing.verifying_key();
    Ok((
        hex::encode(signing.to_bytes()),
        hex::encode(verifying.to_bytes()),
    ))
}

fn sign(private_key_hex: &str, payload: &str) -> Result<String> {
    let private_key = hex::decode(private_key_hex).map_err(Error::DecodePrivateKeyHex)?;
    let signing = SigningKey::from_bytes(
        &private_key
            .try_into()
            .map_err(|_| Error::InvalidPrivateKeyLength)?,
    );
    let signature: Signature = signing.sign(payload.as_bytes());
    Ok(hex::encode(signature.to_bytes()))
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::{Verifier, VerifyingKey};

    use super::*;

    /// A generated keypair signs and verifies end-to-end — the private key signs
    /// a message and the paired public key verifies it, the wire contract the
    /// mgmt server enforces. Also guards the ed25519-dalek 2 → 3 bump.
    #[test]
    fn generated_keypair_signs_and_verifies() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let (private_hex, public_hex) = generate_keypair_hex()?;
        let message = signed_payload(
            "GET",
            "/clients/me",
            "2026-01-01T00:00:00Z",
            "ev1_a3f8",
            "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
        );
        let signature_hex = sign(&private_hex, &message)?;

        let public: [u8; 32] = hex::decode(&public_hex)?
            .try_into()
            .map_err(|_| "public key must be 32 bytes")?;
        let signature: [u8; 64] = hex::decode(&signature_hex)?
            .try_into()
            .map_err(|_| "signature must be 64 bytes")?;
        VerifyingKey::from_bytes(&public)?
            .verify(message.as_bytes(), &Signature::from_bytes(&signature))?;
        Ok(())
    }

    /// The payload is a byte-for-byte wire contract with the server, which
    /// rebuilds the same string from the request it received and verifies
    /// against it. Field order, the `v1` tag, and the newline delimiters are all
    /// load-bearing: get any of them wrong and every authenticated request 401s.
    #[test]
    fn signed_payload_is_six_newline_separated_fields() {
        assert_eq!(
            signed_payload(
                "GET",
                "/clients/me?page=2",
                "2026-03-10T12:00:00Z",
                "ev1_a3f8",
                "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
            ),
            "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0"
        );
    }

    /// The server rejects an empty or repeated nonce, and reads the payload as
    /// newline-delimited fields. Hex satisfies all three: never empty, never
    /// carrying a newline that could forge a field boundary, and fresh per call.
    #[test]
    fn generate_nonce_is_fresh_hex_of_32_chars() -> Result<()> {
        let first = generate_nonce()?;
        let second = generate_nonce()?;

        assert_eq!(first.len(), 32);
        assert!(first
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(first, second);
        Ok(())
    }
}
