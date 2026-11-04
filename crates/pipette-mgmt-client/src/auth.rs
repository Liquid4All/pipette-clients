use std::fmt;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use ed25519_dalek::{Signature, Signer, SigningKey};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use pipette_plan_types::{ClientId, UserId, UserKeyId};

use crate::error::{Error, Result};
use crate::httpsig::{self, Outgoing};
use crate::types::{
    Nonce, PrivateKey, PrivateKeyError, PublicKey, PublicKeyError, RequestSignature, Timestamp,
};

/// The identity a **device** authenticates as: the client the server knows, and
/// the key it signs with. Signs the `v1` payload ([`client_payload`]).
#[derive(Debug, Clone)]
pub struct ClientIdentity {
    /// What `X-Client-Id` carries and the signature covers, typed so a padded
    /// or empty one cannot sign as a client the server cannot resolve.
    pub client_id: ClientId,
    pub private_key_hex: PrivateKey,
}

/// The identity an **operator or a service job** authenticates as: the account,
/// the registered key this machine holds, and its signer. Signs RFC 9421
/// ([`crate::httpsig`]), naming the key in `keyid`.
///
/// An account holds a key per machine (`pipette-mgmt`'s
/// `docs/authentication.md` §5.11), so the key id is a field: it names which
/// registration signed, letting the server find it in one read. `user_id`
/// never reaches the wire; it is kept for attribution and the routes that take
/// an account id.
#[derive(Debug, Clone)]
pub struct UserIdentity {
    pub user_id: UserId,
    /// The signature's `keyid`, which the server finds the account by.
    pub user_key_id: UserKeyId,
    /// What produces the signature: the key itself, or a holder such as an
    /// ssh-agent. The server cannot tell which.
    pub signer: UserSigner,
}

impl UserIdentity {
    /// The identity of a machine that holds the key itself.
    pub fn with_private_key(
        user_id: UserId,
        user_key_id: UserKeyId,
        private_key_hex: PrivateKey,
    ) -> Self {
        Self {
            user_id,
            user_key_id,
            signer: UserSigner::PrivateKey(private_key_hex),
        }
    }

    /// The identity of a machine whose key is held by something else that will
    /// sign on its behalf.
    pub fn with_signer(
        user_id: UserId,
        user_key_id: UserKeyId,
        signer: Arc<dyn SignsPayload>,
    ) -> Self {
        Self {
            user_id,
            user_key_id,
            signer: UserSigner::External(signer),
        }
    }
}

/// A key whose account is not known yet. It signs as [`UserIdentity`] does,
/// since the key alone names the account.
#[derive(Debug, Clone)]
pub struct KeyOnlyIdentity {
    /// The signature's `keyid`.
    pub user_key_id: UserKeyId,
    pub signer: UserSigner,
}

/// Where a user's signature comes from. The server cannot tell the two apart.
#[derive(Clone, Debug)]
pub enum UserSigner {
    /// The key itself, as the hex an identity file stores.
    PrivateKey(PrivateKey),
    /// A holder of the key, such as an ssh-agent through `pipette-ssh`, asked
    /// to sign each request.
    External(Arc<dyn SignsPayload>),
}

/// Something that holds a user's private key and will sign with it.
///
/// Must sign the payload bytes as given, with no hashing or framing of its own,
/// or the server rejects the result. `Debug` because an identity is printed in
/// error context: an implementor that holds a key must render without it.
pub trait SignsPayload: fmt::Debug + Send + Sync {
    fn sign_payload(
        &self,
        payload: &[u8],
    ) -> std::result::Result<[u8; SIGNATURE_LEN], Box<dyn std::error::Error + Send + Sync>>;
}

/// An Ed25519 signature, the only length either path can produce.
pub const SIGNATURE_LEN: usize = 64;

pub(crate) const NONCE_LEN: usize = 16;

/// The management-auth headers a signed request carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuthHeader {
    /// `v1` (clients only), as are the next three.
    ClientId,
    Timestamp,
    Nonce,
    Signature,
    /// RFC 9421's covered components and parameters. Users only.
    SignatureInput,
    /// RFC 9421's signature. Users only.
    MessageSignature,
    /// RFC 9530's body digest, which the signature covers. Users only.
    ContentDigest,
}

impl AuthHeader {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ClientId => "X-Client-Id",
            Self::Timestamp => "X-Timestamp",
            Self::Nonce => "X-Nonce",
            Self::Signature => "X-Signature",
            Self::SignatureInput => "Signature-Input",
            Self::MessageSignature => "Signature",
            Self::ContentDigest => "Content-Digest",
        }
    }
}

impl SignsRequest for ClientIdentity {
    /// `X-Client-Id`, `X-Timestamp`, `X-Nonce` and `X-Signature`, over the `v1`
    /// payload. The body is not covered.
    fn signed_headers(&self, request: &Outgoing<'_>) -> Result<Vec<(AuthHeader, String)>> {
        let stamp = RequestStamp::new()?;
        let payload = client_payload(
            request.method,
            &path_and_query(request.url),
            stamp.timestamp.as_ref(),
            self.client_id.as_ref(),
            &stamp.nonce.to_string(),
        );
        let signed = stamp.sign(&payload, |payload| sign(&self.private_key_hex, payload))?;
        Ok(signed.into_headers(vec![(AuthHeader::ClientId, self.client_id.to_string())]))
    }
}

impl SignsRequest for UserIdentity {
    /// The RFC 9421 fields, and `Content-Digest` with a body.
    fn signed_headers(&self, request: &Outgoing<'_>) -> Result<Vec<(AuthHeader, String)>> {
        self.signer.sign_message(&self.user_key_id, request)
    }
}

impl SignsRequest for KeyOnlyIdentity {
    /// The same fields as [`UserIdentity`]: the key names the account.
    fn signed_headers(&self, request: &Outgoing<'_>) -> Result<Vec<(AuthHeader, String)>> {
        self.signer.sign_message(&self.user_key_id, request)
    }
}

/// How the private [`crate::call`] pipeline signs, and the only thing generic
/// over the two kinds. `pub(crate)` because no endpoint takes "either
/// identity"; it exists because signing has to happen inside `call`, which
/// derives the request target from the base URL.
pub(crate) trait SignsRequest {
    /// The signed auth headers, all ASCII so the transport can set them
    /// verbatim.
    ///
    /// `request.url` must be the URL the request goes to, base-URL path prefix
    /// included: the server verifies against the target and `Host` it receives.
    fn signed_headers(&self, request: &Outgoing<'_>) -> Result<Vec<(AuthHeader, String)>>;
}

/// The request target as the server's `uri.path_and_query()` reads it.
pub(crate) fn path_and_query(url: &url::Url) -> String {
    match url.query() {
        Some(query) => format!("{}?{query}", url.path()),
        None => url.path().to_string(),
    }
}

/// The `v1` timestamp and nonce, consumed into a [`SignedStamp`] by
/// [`Self::sign`] so the values signed are the values that go on the wire.
struct RequestStamp {
    timestamp: Timestamp,
    nonce: Nonce,
}

impl RequestStamp {
    fn new() -> Result<Self> {
        let timestamp = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .map_err(Error::TimestampFormat)?;
        Ok(Self {
            timestamp: Timestamp::try_new(timestamp.clone())
                .map_err(|_| Error::UnusableTimestamp { derived: timestamp })?,
            nonce: generate_nonce()?,
        })
    }

    /// Sign `payload`, built from this stamp.
    fn sign(
        self,
        payload: &str,
        sign: impl FnOnce(&str) -> Result<RequestSignature>,
    ) -> Result<SignedStamp> {
        Ok(SignedStamp {
            signature: sign(payload)?,
            timestamp: self.timestamp,
            nonce: self.nonce,
        })
    }
}

/// A stamp and the signature over the payload it was bound into.
struct SignedStamp {
    timestamp: Timestamp,
    nonce: Nonce,
    signature: RequestSignature,
}

impl SignedStamp {
    /// Append the stamp to `headers`, the ones naming the principal.
    fn into_headers(self, mut headers: Vec<(AuthHeader, String)>) -> Vec<(AuthHeader, String)> {
        headers.push((AuthHeader::Timestamp, self.timestamp.to_string()));
        headers.push((AuthHeader::Nonce, self.nonce.to_string()));
        headers.push((AuthHeader::Signature, self.signature.to_string()));
        headers
    }
}

impl UserSigner {
    /// The RFC 9421 fields for `request`, signed as `key_id` now, under a
    /// fresh nonce.
    fn sign_message(
        &self,
        key_id: &UserKeyId,
        request: &Outgoing<'_>,
    ) -> Result<Vec<(AuthHeader, String)>> {
        let signed = httpsig::sign(
            key_id,
            &generate_nonce()?.to_string(),
            OffsetDateTime::now_utc().unix_timestamp(),
            request,
            |base| self.signature(base),
        )?;
        let mut headers = vec![
            (AuthHeader::SignatureInput, signed.signature_input),
            (AuthHeader::MessageSignature, signed.signature),
        ];
        if let Some(digest) = signed.content_digest {
            headers.push((AuthHeader::ContentDigest, digest));
        }
        Ok(headers)
    }

    /// A signature over `message`, whichever holds the key.
    fn signature(&self, message: &[u8]) -> Result<[u8; SIGNATURE_LEN]> {
        match self {
            Self::PrivateKey(private_key) => Ok(signing_key(private_key)?.sign(message).to_bytes()),
            Self::External(holder) => holder
                .sign_payload(message)
                .map_err(|source| Error::Signer { source }),
        }
    }
}

/// A fresh per-request nonce: 16 CSPRNG bytes, lowercase hex.
///
/// Hex because the payload is newline-delimited, and a newline in the nonce
/// could forge a field boundary. 128 bits makes a fleet-wide collision
/// negligible, so the server's replay cache needs no client coordination.
fn generate_nonce() -> Result<Nonce> {
    let mut bytes = [0u8; NONCE_LEN];
    getrandom::fill(&mut bytes).map_err(Error::OsEntropy)?;
    Ok(Nonce::new(bytes))
}

/// The `v1` signed payload a **client** uses (mgmt's `docs/authentication.md`
/// §2.1). The nonce makes a captured signature single-use inside the freshness
/// window. The body is not covered; a user's RFC 9421 signature covers it.
fn client_payload(
    method: &str,
    path_and_query: &str,
    timestamp: &str,
    client_id: &str,
    nonce: &str,
) -> String {
    format!("v1\n{method}\n{path_and_query}\n{timestamp}\n{client_id}\n{nonce}")
}

/// A client id and a key id are the same digest under different prefixes, so
/// the server claims that digest across both keyspaces to keep one keypair to
/// one principal (mgmt's `docs/authentication.md` §5.11).
const KEY_ID_PREFIX: &str = "userkey_";

/// The key id a public key registers under: `userkey_` + hex(SHA-256(key
/// bytes)). Derived locally so a machine can name the id before registering,
/// and compare it with the one the server files.
///
/// Takes a [`PublicKey`] so half a pasted key is refused here rather than
/// given a plausible-looking id.
pub fn key_id_for(public_key: &PublicKey) -> Result<UserKeyId> {
    // Infallible in practice, but validated rather than asserted, per the
    // workspace's panic ban.
    let bytes = hex::decode(public_key.as_ref()).map_err(PublicKeyError::NotHex)?;
    let id = format!("{KEY_ID_PREFIX}{}", hex::encode(Sha256::digest(bytes)));
    UserKeyId::try_new(id.clone()).map_err(|_| Error::UnusableKeyId { derived: id })
}

/// A new Ed25519 keypair. The signing key is its 32-byte seed, read straight
/// from the OS CSPRNG, so no `rand` adapter is needed.
pub fn generate_keypair() -> Result<(PrivateKey, PublicKey)> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed).map_err(Error::OsEntropy)?;
    let signing = SigningKey::from_bytes(&seed);
    let verifying = signing.verifying_key();
    Ok((
        PrivateKey::try_new(hex::encode(signing.to_bytes()))?,
        PublicKey::try_new(hex::encode(verifying.to_bytes()))?,
    ))
}

/// The signing key a private key names. Both failures are already refused by
/// [`PrivateKey`]'s constructor; they are mapped rather than panicked on.
fn signing_key(private_key: &PrivateKey) -> Result<SigningKey> {
    let seed: [u8; 32] = hex::decode(private_key.as_ref())
        .map_err(PrivateKeyError::NotHex)?
        .try_into()
        .map_err(|decoded: Vec<u8>| PrivateKeyError::Length {
            found: decoded.len(),
        })?;
    Ok(SigningKey::from_bytes(&seed))
}

/// The public key paired with a private key, which is derived rather than
/// stored. Hex rather than a `VerifyingKey`, to keep `ed25519_dalek` out of
/// the callers' dependencies.
pub fn public_key_hex_from_private(private_key: &PrivateKey) -> Result<PublicKey> {
    Ok(PublicKey::try_new(hex::encode(
        signing_key(private_key)?.verifying_key().to_bytes(),
    ))?)
}

fn sign(private_key: &PrivateKey, payload: &str) -> Result<RequestSignature> {
    let signature: Signature = signing_key(private_key)?.sign(payload.as_bytes());
    Ok(RequestSignature::new(signature.to_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::call::test_support::verify_signature;
    use crate::httpsig::test_support::verify_request;

    fn outgoing(url: &url::Url) -> Outgoing<'_> {
        Outgoing {
            method: "GET",
            url,
            body_sha256: None,
        }
    }

    fn value(headers: &[(AuthHeader, String)], wanted: AuthHeader) -> Option<String> {
        headers
            .iter()
            .find(|(name, _)| *name == wanted)
            .map(|(_, value)| value.clone())
    }

    #[test]
    fn generated_keypair_signs_and_verifies() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let (private_key, public_hex) = generate_keypair()?;
        let message = client_payload(
            "GET",
            "/clients/me",
            "2026-01-01T00:00:00Z",
            "ev1_a3f8",
            "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
        );
        let signature = sign(&private_key, &message)?;

        verify_signature(&public_hex, &message, &signature.to_string())?;
        Ok(())
    }

    #[test]
    fn public_key_is_derived_from_the_private_one() -> Result<()> {
        let (private_key, public_hex) = generate_keypair()?;
        assert_eq!(public_key_hex_from_private(&private_key)?, public_hex);
        Ok(())
    }

    /// The server rebuilds this string byte for byte; any difference 401s
    /// every request.
    #[test]
    fn client_payload_is_six_newline_separated_fields() {
        assert_eq!(
            client_payload(
                "GET",
                "/clients/me?page=2",
                "2026-03-10T12:00:00Z",
                "ev1_a3f8",
                "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
            ),
            "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0"
        );
    }

    /// A client names itself; a user names only its key, in `keyid`.
    #[test]
    fn a_client_signs_v1_a_user_signs_rfc_9421() -> Result<()> {
        let (private_key_hex, _) = generate_keypair()?;
        let client = ClientIdentity {
            client_id: ClientId::try_new("ev1_a3f8")
                .map_err(|e| Error::BuildClient(e.to_string()))?,
            private_key_hex: private_key_hex.clone(),
        };
        let user = UserIdentity::with_private_key(
            UserId::try_new("user_0192f0c1").map_err(|e| Error::BuildClient(e.to_string()))?,
            UserKeyId::try_new("userkey_9f2c").map_err(|e| Error::BuildClient(e.to_string()))?,
            private_key_hex,
        );
        let url = url::Url::parse("https://mgmt.example/x")
            .map_err(|e| Error::BuildClient(e.to_string()))?;

        let client_headers = client.signed_headers(&outgoing(&url))?;
        assert_eq!(
            value(&client_headers, AuthHeader::ClientId).as_deref(),
            Some("ev1_a3f8")
        );
        assert_eq!(value(&client_headers, AuthHeader::SignatureInput), None);

        // The server refuses an RFC 9421 request that also carries a header of
        // the older `v2` user scheme.
        let user_headers = user.signed_headers(&outgoing(&url))?;
        [AuthHeader::ClientId, AuthHeader::Signature]
            .into_iter()
            .for_each(|v2| assert_eq!(value(&user_headers, v2), None, "{}", v2.as_str()));
        let input = value(&user_headers, AuthHeader::SignatureInput).unwrap_or_default();
        assert!(input.contains(r#";keyid="userkey_9f2c";"#), "{input}");
        Ok(())
    }

    #[rstest::rstest]
    #[case(AuthHeader::ClientId, "X-Client-Id")]
    #[case(AuthHeader::Timestamp, "X-Timestamp")]
    #[case(AuthHeader::Nonce, "X-Nonce")]
    #[case(AuthHeader::Signature, "X-Signature")]
    #[case(AuthHeader::SignatureInput, "Signature-Input")]
    #[case(AuthHeader::MessageSignature, "Signature")]
    #[case(AuthHeader::ContentDigest, "Content-Digest")]
    fn auth_headers_are_spelled_as_the_server_reads_them(
        #[case] header: AuthHeader,
        #[case] name: &str,
    ) {
        assert_eq!(header.as_str(), name);
    }

    #[test]
    fn generate_nonce_is_fresh_hex_of_32_chars() -> Result<()> {
        let first = generate_nonce()?;
        let second = generate_nonce()?;

        let hex = first.to_string();
        assert_eq!(hex.len(), 32);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn a_key_held_elsewhere_signs_what_the_server_verifies() -> Result<()> {
        #[derive(Debug)]
        struct HeldElsewhere(SigningKey);

        impl SignsPayload for HeldElsewhere {
            fn sign_payload(
                &self,
                payload: &[u8],
            ) -> std::result::Result<[u8; SIGNATURE_LEN], Box<dyn std::error::Error + Send + Sync>>
            {
                Ok(self.0.sign(payload).to_bytes())
            }
        }

        let (private_key_hex, public_key_hex) = generate_keypair()?;
        let holder = HeldElsewhere(signing_key(&private_key_hex)?);
        let identity = UserIdentity::with_signer(
            UserId::try_new("user_0192f0c1").map_err(|e| Error::BuildClient(e.to_string()))?,
            UserKeyId::try_new("userkey_9f2c").map_err(|e| Error::BuildClient(e.to_string()))?,
            Arc::new(holder),
        );

        let url = url::Url::parse("https://mgmt.example/api/v1/users/me")
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        let headers = identity.signed_headers(&outgoing(&url))?;
        verify_request(&public_key_hex, "GET", url.as_str(), |name| {
            headers
                .iter()
                .find(|(header, _)| header.as_str() == name)
                .map(|(_, value)| value.clone())
        })
    }

    #[test]
    fn a_holder_that_refuses_carries_its_own_reason() -> Result<()> {
        #[derive(Debug)]
        struct Refuses;

        impl SignsPayload for Refuses {
            fn sign_payload(
                &self,
                _payload: &[u8],
            ) -> std::result::Result<[u8; SIGNATURE_LEN], Box<dyn std::error::Error + Send + Sync>>
            {
                Err("the vault is locked".into())
            }
        }

        let identity = UserIdentity::with_signer(
            UserId::try_new("user_0192f0c1").map_err(|e| Error::BuildClient(e.to_string()))?,
            UserKeyId::try_new("userkey_9f2c").map_err(|e| Error::BuildClient(e.to_string()))?,
            Arc::new(Refuses),
        );

        let url = url::Url::parse("https://mgmt.example/api/v1/users/me")
            .map_err(|e| Error::BuildClient(e.to_string()))?;
        let error = identity
            .signed_headers(&outgoing(&url))
            .err()
            .ok_or_else(|| Error::BuildClient("a refusal cannot produce headers".to_string()))?;

        assert!(matches!(error, Error::Signer { .. }), "got {error}");
        assert!(
            error.to_string().contains("the vault is locked"),
            "the holder's reason is what the operator acts on: {error}"
        );
        assert!(
            !error.is_transient(),
            "asking a locked vault again is not a retry"
        );
        Ok(())
    }
}
