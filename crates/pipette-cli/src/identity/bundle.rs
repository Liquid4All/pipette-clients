//! The identity token `auth export` writes and `auth import` reads.
//!
//! It carries only what an importing machine cannot rebuild: the `client_id`
//! (every signed request must already carry it, so `auth me` cannot recover
//! it, and losing it orphans every result under it), the server URL and the
//! key.
//! Device labels and the storage quota are left out because they describe the
//! exporting machine, and carrying them would mislabel the importer's results
//! and its `PATCH /clients/me`. Hugging Face tokens are left out because the
//! client never writes them to disk.
//!
//! Wire format: the JSON of the fields, base64url without padding, so the
//! token survives a shell, a URL and a copy-paste as one word.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};

use pipette_mgmt_client::{PrivateKey, PublicKey, ServerUrl, public_key_hex_from_private};
use pipette_plan_types::ClientId;

use super::IdentityRegistration;
use crate::error::{Error, Result};

/// A restorable client identity.
///
/// `Serialize` only: [`BundleWire`] is what a token deserializes into, so no
/// token becomes one of these without passing [`Self::decode`]'s checks.
#[derive(Debug, Clone, Serialize)]
pub struct IdentityBundle {
    pub version: u32,
    pub client_id: ClientId,
    pub server_url: ServerUrl,
    /// [`PrivateKey`] serializes bare (a redacted value would export a token
    /// nothing could restore) and redacts only `Debug`/`Display`.
    pub private_key_hex: PrivateKey,
}

impl IdentityBundle {
    /// The only format this build writes or reads.
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(registration: &IdentityRegistration, private_key: &PrivateKey) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            client_id: registration.client_id.clone(),
            server_url: registration.server_url.clone(),
            private_key_hex: private_key.clone(),
        }
    }

    pub fn registration(&self) -> IdentityRegistration {
        IdentityRegistration {
            client_id: self.client_id.clone(),
            server_url: self.server_url.clone(),
        }
    }

    /// Derived, never stored: the one value an operator can check against the
    /// server's record without disclosing anything.
    pub fn public_key_hex(&self) -> Result<PublicKey> {
        Ok(public_key_hex_from_private(&self.private_key_hex)?)
    }

    pub fn encode(&self) -> Result<String> {
        let json = serde_json::to_vec(self)
            .map_err(|source| Error::Other(anyhow::Error::new(source).context("encode bundle")))?;
        Ok(URL_SAFE_NO_PAD.encode(json))
    }

    /// Read a token back, validating every field. The version is read first,
    /// so a later format reports its version rather than a missing field.
    pub fn decode(token: &str) -> Result<Self> {
        let json = URL_SAFE_NO_PAD
            .decode(token.trim())
            .map_err(|_| Error::BundleNotBase64url)?;
        let probe: VersionProbe = serde_json::from_slice(&json)
            .map_err(|source| Error::BundleNotAnIdentity(source.to_string()))?;
        if probe.version != Self::CURRENT_VERSION {
            return Err(Error::BundleVersion {
                found: probe.version,
                expected: Self::CURRENT_VERSION,
            });
        }
        let wire: BundleWire = serde_json::from_slice(&json)
            .map_err(|source| Error::BundleNotAnIdentity(source.to_string()))?;
        // Each newtype trims as it validates, which keeps a padded `client_id`
        // out of the restored client's `X-Client-Id` header.
        Ok(Self {
            version: wire.version,
            client_id: ClientId::try_new(wire.client_id).map_err(Error::BundleClientId)?,
            server_url: ServerUrl::try_new(wire.server_url).map_err(Error::BundleServerUrl)?,
            private_key_hex: PrivateKey::try_new(wire.private_key_hex)
                .map_err(Error::BundlePrivateKey)?,
        })
    }
}

#[derive(Deserialize)]
struct VersionProbe {
    version: u32,
}

/// The record with every field still a plain string. Deserializing the typed
/// struct directly would report a bad field as a JSON line and column the
/// operator never saw; this lets [`IdentityBundle::decode`] name the field.
#[derive(Deserialize)]
struct BundleWire {
    version: u32,
    client_id: String,
    server_url: String,
    private_key_hex: String,
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::PrincipalIdError;

    use super::*;

    /// Any 32 bytes are a valid Ed25519 seed, so hex of the right length is a
    /// usable key without generating one.
    const KEY: &str = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";

    fn registration() -> anyhow::Result<IdentityRegistration> {
        Ok(IdentityRegistration {
            client_id: ClientId::try_new("ev1_abc")?,
            server_url: ServerUrl::try_new("https://collector.example.test")?,
        })
    }

    fn key() -> anyhow::Result<PrivateKey> {
        Ok(PrivateKey::try_new(KEY)?)
    }

    fn bundle() -> anyhow::Result<IdentityBundle> {
        Ok(IdentityBundle::new(&registration()?, &key()?))
    }

    /// For values the typed struct cannot hold.
    fn token_of(
        client_id: &str,
        server_url: &str,
        private_key_hex: &str,
    ) -> anyhow::Result<String> {
        Ok(
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
                "version": 1,
                "client_id": client_id,
                "server_url": server_url,
                "private_key_hex": private_key_hex,
            }))?),
        )
    }

    #[test]
    fn round_trips_through_a_token() -> anyhow::Result<()> {
        let decoded = IdentityBundle::decode(&bundle()?.encode()?)?;
        assert_eq!(decoded.version, IdentityBundle::CURRENT_VERSION);
        assert_eq!(decoded.client_id.as_ref(), "ev1_abc");
        assert_eq!(
            decoded.server_url.to_string(),
            "https://collector.example.test"
        );
        assert_eq!(decoded.private_key_hex.as_ref(), KEY);
        Ok(())
    }

    #[test]
    fn token_is_one_word_of_base64url() -> anyhow::Result<()> {
        let token = bundle()?.encode()?;
        assert!(
            !token.contains(['+', '/', '=']),
            "not base64url-safe: {token}"
        );
        assert!(
            token.chars().all(|c| !c.is_whitespace()),
            "token must be one word: {token}"
        );
        Ok(())
    }

    /// Surrounding whitespace is what a copy-paste or a `--from` file carries.
    #[test]
    fn decode_tolerates_surrounding_whitespace() -> anyhow::Result<()> {
        let token = format!("  {}\n", bundle()?.encode()?);
        assert_eq!(
            IdentityBundle::decode(&token)?.client_id.as_ref(),
            "ev1_abc"
        );
        Ok(())
    }

    #[test]
    fn rejects_a_token_that_is_not_base64url() {
        assert!(matches!(
            IdentityBundle::decode("not a token"),
            Err(Error::BundleNotBase64url)
        ));
    }

    #[test]
    fn rejects_base64url_that_is_not_a_bundle() {
        let token = URL_SAFE_NO_PAD.encode(b"{\"unrelated\": true}");
        assert!(matches!(
            IdentityBundle::decode(&token),
            Err(Error::BundleNotAnIdentity(_))
        ));
    }

    #[test]
    fn rejects_a_version_this_build_cannot_read() -> anyhow::Result<()> {
        let token = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
            "version": 2,
            "client_id": "ev1_abc",
            "server_url": "https://collector.example.test",
            "private_key_hex": KEY,
            "something_new": 7,
        }))?);
        match IdentityBundle::decode(&token) {
            Err(Error::BundleVersion {
                found: 2,
                expected: 1,
            }) => Ok(()),
            other => anyhow::bail!("expected BundleVersion, got {other:?}"),
        }
    }

    #[rstest::rstest]
    #[case::not_hex("zz11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899x")]
    #[case::wrong_length("aabb")]
    fn rejects_a_key_that_cannot_sign(#[case] key: &str) -> anyhow::Result<()> {
        let token = token_of("ev1_abc", "https://collector.example.test", key)?;
        match IdentityBundle::decode(&token) {
            Err(Error::BundlePrivateKey(_)) => Ok(()),
            other => anyhow::bail!("expected BundlePrivateKey, got {other:?}"),
        }
    }

    #[rstest::rstest]
    #[case::empty("", PrincipalIdError::Empty { kind: "client id" })]
    #[case::blank("   ", PrincipalIdError::Empty { kind: "client id" })]
    #[case::outside_the_charset("ev1.a", PrincipalIdError::Charset { kind: "client id", ch: '.' })]
    fn rejects_an_unusable_client_id(
        #[case] id: &str,
        #[case] expected: PrincipalIdError,
    ) -> anyhow::Result<()> {
        let token = token_of(id, "https://collector.example.test", KEY)?;
        match IdentityBundle::decode(&token) {
            Err(Error::BundleClientId(refusal)) if refusal == expected => Ok(()),
            other => anyhow::bail!("expected BundleClientId({expected:?}), got {other:?}"),
        }
    }

    #[rstest::rstest]
    #[case::empty("")]
    #[case::blank("   ")]
    #[case::bare_host("collector.example.test")]
    #[case::wrong_scheme("ftp://collector.example.test")]
    #[case::carries_a_query("https://collector.example.test?tenant=a")]
    fn rejects_a_server_url_the_client_could_not_reach(#[case] url: &str) -> anyhow::Result<()> {
        let token = token_of("ev1_abc", url, KEY)?;
        match IdentityBundle::decode(&token) {
            Err(Error::BundleServerUrl(_)) => Ok(()),
            other => anyhow::bail!("expected BundleServerUrl, got {other:?}"),
        }
    }

    #[test]
    fn decode_trims_the_fields_it_restores() -> anyhow::Result<()> {
        let token = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "client_id": "  ev1_abc  ",
            "server_url": " https://collector.example.test/\n",
            "private_key_hex": format!("  {KEY}\n"),
        }))?);
        let bundle = IdentityBundle::decode(&token)?;
        assert_eq!(bundle.client_id.as_ref(), "ev1_abc");
        assert_eq!(
            bundle.server_url.to_string(),
            "https://collector.example.test"
        );
        assert_eq!(bundle.private_key_hex.as_ref(), KEY);
        Ok(())
    }

    #[test]
    fn public_key_is_derived_from_the_private_one() -> anyhow::Result<()> {
        let public = bundle()?.public_key_hex()?;
        // The newtype guarantees the shape; this checks it is not the key echoed back.
        assert_ne!(public.as_ref() as &str, KEY);
        Ok(())
    }
}
