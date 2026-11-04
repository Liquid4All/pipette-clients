//! The things that must never reach a log line or the state directory: each
//! device's private key (typed as `pipette_mgmt_client::PrivateKey`, which
//! redacts itself), the releases bucket's secret key and the signing keychain's
//! password (each a [`Secret`]), and any `auth_token` inside a claim spec (a
//! `serde_json::Value`, so [`redacted_spec`] rewrites the tree before anything
//! prints it, and [`scrub`] removes it from each console line the app prints
//! back).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use pipette_mgmt_client::PrivateKey;
use pipette_plan_types::ClientId;

pub(crate) const REDACTED: &str = "<redacted>";

/// Spec keys whose values are replaced wherever they appear in a claim spec.
const REDACTED_KEYS: &[&str] = &["auth_token"];

/// Where a device's credentials came from, named as the operator wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CredentialSource {
    PrivateKey,
    PrivateKeyFile,
    IdentityToken,
    IdentityTokenFile,
}

impl CredentialSource {
    fn key(self) -> &'static str {
        match self {
            Self::PrivateKey => "private_key",
            Self::PrivateKeyFile => "private_key_file",
            Self::IdentityToken => "identity_token",
            Self::IdentityTokenFile => "identity_token_file",
        }
    }
}

impl std::fmt::Display for CredentialSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[[devices]] {}", self.key())
    }
}

/// Why a device's credentials could not be taken.
#[derive(Debug, thiserror::Error)]
pub(crate) enum KeyError {
    #[error(
        "more than one credential source: {}. One of them is stale, and which one won would not \
         show up anywhere, so give exactly one",
        .0.iter().map(ToString::to_string).collect::<Vec<_>>().join(" and ")
    )]
    SeveralSources(Vec<CredentialSource>),
    #[error(
        "no credentials: give the phone's `identity_token` (what `headlessrun auth export` \
         prints, inline or via identity_token_file), or its client_id with private_key / \
         private_key_file. This worker signs as the phone's own registered client and never \
         mints one"
    )]
    NoSource,
    #[error("client_id is required beside {0}; only an identity token carries its own")]
    ClientIdRequired(CredentialSource),
    #[error(
        "client_id `{configured}` does not match the identity token's `{in_token}`: the token \
         was exported from another phone, or the id is stale"
    )]
    ClientIdMismatch {
        configured: String,
        in_token: String,
    },
    #[error("the identity token is not what `headlessrun auth export` prints: {0}")]
    MalformedToken(String),
    #[error(
        "{path} is mode {mode:04o}, readable beyond its owner: a signing key at that mode has \
         already been exposed, so rotate the key ({}) rather than tightening the mode. \
         Expected 0600",
        .key.rotation()
    )]
    ReadableBeyondOwner {
        path: String,
        mode: u32,
        key: KeyKind,
    },
}

/// Whose key a file holds, which decides how it is rotated.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KeyKind {
    Device,
    /// The `[client_releases]` secret access key.
    AccessKey,
    /// `[signing] keychain_password_file`.
    KeychainPassword,
    /// `[signing] api_key_file`, the App Store Connect `.p8`.
    ApiKey,
}

impl KeyKind {
    fn rotation(self) -> &'static str {
        match self {
            Self::Device => "re-export it from the phone",
            Self::AccessKey => {
                "create a new access key for the bucket's IAM user and delete this one"
            }
            Self::KeychainPassword => "change the keychain's password",
            Self::ApiKey => "revoke the key in App Store Connect and create a new one",
        }
    }
}

/// Why `[signing] keychain_password_file` cannot be fed to `security -i`.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PasswordError {
    #[error("{path} holds no password")]
    Empty { path: PathBuf },
    /// `security -i` reads one line, with the password inside double quotes.
    #[error(
        "{path} holds a double quote, a backslash or a line break beyond one trailing newline, \
         which the `security -i` line cannot carry"
    )]
    Unquotable { path: PathBuf },
}

/// A credential that debug-prints as `<redacted>` and has no `Display`, so the
/// table it is parsed into can be printed: the `[client_releases]` secret
/// access key, or the signing keychain's password, which only ever reaches
/// `security -i` on stdin.
#[derive(Clone, serde::Deserialize)]
#[serde(from = "String")]
pub(crate) struct Secret(String);

impl From<String> for Secret {
    /// Trimmed, like an inline device key. A password comes from
    /// [`read_password_file`] instead, which keeps its spaces.
    fn from(raw: String) -> Self {
        Self(raw.trim().to_owned())
    }
}

impl Secret {
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_tests(raw: &str) -> Self {
        Self(raw.to_owned())
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Strips one trailing LF or CRLF and nothing else: a password may start or
/// end with a space.
pub(crate) fn read_password_file(path: &Path) -> anyhow::Result<Secret> {
    refuse_beyond_owner(path, KeyKind::KeychainPassword)?;
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading the credential file {}", path.display()))?;
    let password = raw
        .strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(&raw);
    if password.is_empty() {
        return Err(PasswordError::Empty {
            path: path.to_path_buf(),
        }
        .into());
    }
    if password.contains(['"', '\\', '\r', '\n']) {
        return Err(PasswordError::Unquotable {
            path: path.to_path_buf(),
        }
        .into());
    }
    Ok(Secret(password.to_owned()))
}

/// The four ways a `[[devices]]` entry can carry its credentials, as written.
#[derive(Debug, Default)]
pub(crate) struct CredentialFields<'a> {
    pub(crate) client_id: Option<&'a str>,
    pub(crate) private_key: Option<&'a str>,
    pub(crate) private_key_file: Option<&'a Path>,
    pub(crate) identity_token: Option<&'a str>,
    pub(crate) identity_token_file: Option<&'a Path>,
}

impl CredentialFields<'_> {
    /// Every source the entry wrote, for the error that names them all.
    fn sources(&self) -> Vec<CredentialSource> {
        [
            (self.private_key.is_some(), CredentialSource::PrivateKey),
            (
                self.private_key_file.is_some(),
                CredentialSource::PrivateKeyFile,
            ),
            (
                self.identity_token.is_some(),
                CredentialSource::IdentityToken,
            ),
            (
                self.identity_token_file.is_some(),
                CredentialSource::IdentityTokenFile,
            ),
        ]
        .into_iter()
        .filter_map(|(given, source)| given.then_some(source))
        .collect()
    }
}

/// A phone's client id and signing key, from exactly one source. A token carries
/// both; a key needs `client_id` beside it. Two sources is an error naming both:
/// signing with a stale key answers `401`, which reads exactly like clock drift.
/// Files are expected already `~`-expanded; contents are trimmed.
pub(crate) fn resolve_credentials(
    fields: CredentialFields<'_>,
) -> anyhow::Result<(ClientId, PrivateKey)> {
    let (source, raw, file) = match (
        fields.private_key,
        fields.private_key_file,
        fields.identity_token,
        fields.identity_token_file,
    ) {
        (Some(key), None, None, None) => {
            (CredentialSource::PrivateKey, key.trim().to_owned(), None)
        }
        (None, Some(path), None, None) => (
            CredentialSource::PrivateKeyFile,
            read_secret_file(path, KeyKind::Device)?,
            Some(path),
        ),
        (None, None, Some(token), None) => (
            CredentialSource::IdentityToken,
            token.trim().to_owned(),
            None,
        ),
        (None, None, None, Some(path)) => (
            CredentialSource::IdentityTokenFile,
            read_secret_file(path, KeyKind::Device)?,
            Some(path),
        ),
        (None, None, None, None) => return Err(KeyError::NoSource.into()),
        _ => return Err(KeyError::SeveralSources(fields.sources()).into()),
    };
    let origin = file.map_or_else(
        || source.to_string(),
        |path| format!("{source} {}", path.display()),
    );
    let configured_id = fields
        .client_id
        .map(|raw| ClientId::try_new(raw.trim()).context("client_id"))
        .transpose()?;

    let (client_id, raw_key) = match source {
        CredentialSource::PrivateKey | CredentialSource::PrivateKeyFile => {
            let client_id = configured_id.ok_or(KeyError::ClientIdRequired(source))?;
            (client_id, raw)
        }
        CredentialSource::IdentityToken | CredentialSource::IdentityTokenFile => {
            let exported = decode_identity_token(&raw).with_context(|| format!("from {origin}"))?;
            let client_id = ClientId::try_new(exported.client_id.trim())
                .with_context(|| format!("client_id in {origin}"))?;
            if let Some(configured) = configured_id
                && configured != client_id
            {
                return Err(KeyError::ClientIdMismatch {
                    configured: configured.to_string(),
                    in_token: client_id.to_string(),
                }
                .into());
            }
            (client_id, exported.private_key_hex)
        }
    };

    let private_key = PrivateKey::try_new(raw_key).with_context(|| format!("from {origin}"))?;
    Ok((client_id, private_key))
}

/// The two fields of an `auth export` token this worker uses. The token also
/// carries the server URL and any Hugging Face tokens; none of it is logged.
#[derive(Debug, serde::Deserialize)]
struct ExportedIdentity {
    client_id: String,
    private_key_hex: String,
}

/// `headlessrun auth export` prints compact JSON as base64url without padding
/// (`ios/Pipette/Pipette/Identity/Bundle.swift`).
fn decode_identity_token(token: &str) -> anyhow::Result<ExportedIdentity> {
    use base64::Engine;
    let json = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token.trim())
        .map_err(|error| KeyError::MalformedToken(format!("not base64url: {error}")))?;
    serde_json::from_slice(&json).map_err(|error| {
        KeyError::MalformedToken(format!("not the exported record: {error}")).into()
    })
}

/// Group and other permission bits; a key file carrying any is refused.
const BEYOND_OWNER: u32 = 0o077;

/// Refuse a file holding a signing key if anyone but its owner can read it.
/// Applied to every key file and to the config itself when a key is inline.
/// The message says to rotate, not chmod: the exposure has already happened.
pub(crate) fn refuse_beyond_owner(path: &Path, key: KeyKind) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .permissions()
        .mode()
        & 0o777;
    if mode & BEYOND_OWNER != 0 {
        return Err(KeyError::ReadableBeyondOwner {
            path: path.display().to_string(),
            mode,
            key,
        }
        .into());
    }
    Ok(())
}

/// Read a secret out of a file: mode-checked first, whole contents, trimmed.
pub(crate) fn read_secret_file(path: &Path, key: KeyKind) -> anyhow::Result<String> {
    refuse_beyond_owner(path, key)?;
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading the credential file {}", path.display()))?;
    Ok(raw.trim().to_owned())
}

/// A copy of `spec` with every [`REDACTED_KEYS`] value at any depth replaced by
/// [`REDACTED`]. Mirrors `crates/pipette-cli/src/client/claim.rs`.
pub(crate) fn redacted_spec(spec: &Value) -> Value {
    match spec {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| {
                let value = if REDACTED_KEYS.contains(&key.as_str()) {
                    Value::String(REDACTED.to_owned())
                } else {
                    redacted_spec(value)
                };
                (key.clone(), value)
            })
            .collect::<serde_json::Map<_, _>>()
            .into(),
        Value::Array(items) => items.iter().map(redacted_spec).collect::<Vec<_>>().into(),
        leaf @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => leaf.clone(),
    }
}

/// Every [`REDACTED_KEYS`] string inside the JSON values of an app argv, which
/// `model=` carries. The app prints its argv back on a usage error.
pub(crate) fn argv_secrets(app_args: &[String]) -> Vec<String> {
    let mut secrets = Vec::new();
    app_args
        .iter()
        .filter_map(|arg| arg.split_once('='))
        .filter_map(|(_, value)| serde_json::from_str::<Value>(value).ok())
        .for_each(|value| collect_secrets(&value, &mut secrets));
    secrets
}

pub(crate) fn scrub(text: String, secrets: &[String]) -> String {
    secrets.iter().fold(text, |text, secret| {
        if text.contains(secret.as_str()) {
            text.replace(secret.as_str(), REDACTED)
        } else {
            text
        }
    })
}

fn collect_secrets(value: &Value, secrets: &mut Vec<String>) {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                if let Value::String(secret) = value
                    && REDACTED_KEYS.contains(&key.as_str())
                {
                    if !secret.is_empty() {
                        secrets.push(secret.clone());
                    }
                } else {
                    collect_secrets(value, secrets);
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| collect_secrets(item, secrets)),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use anyhow::Context;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn key_file(contents: &str) -> anyhow::Result<tempfile::NamedTempFile> {
        let mut file = tempfile::NamedTempFile::new().context("a temp key file")?;
        write!(file, "{contents}").context("writing the temp key file")?;
        Ok(file)
    }

    const VALID_KEY: &str = "1122334455667788990011223344556677889900112233445566778899001122";
    const CLIENT: &str = "ev1_9f2c1d3e4b5a6c7d";
    /// Stands in a `#[case]` for `token_for(CLIENT, VALID_KEY)`, which is not `'static`.
    const TOKEN: &str = "<token>";

    fn with_key(fields: CredentialFields<'_>) -> CredentialFields<'_> {
        CredentialFields {
            client_id: Some(CLIENT),
            ..fields
        }
    }

    /// What `headlessrun auth export` prints: base64url, no padding, over the
    /// record with a Hugging Face token beside the key.
    fn token_for(client_id: &str, key: &str) -> String {
        use base64::Engine;
        let record = json!({
            "version": 1,
            "client_id": client_id,
            "server_url": "https://collector.pipette.liquid.ai",
            "private_key_hex": key,
            "user_token": "hf_secret",
        });
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(record.to_string())
    }

    #[test]
    fn a_key_file_supplies_the_signing_key() -> anyhow::Result<()> {
        let file = key_file(VALID_KEY)?;
        let (id, key) = resolve_credentials(with_key(CredentialFields {
            private_key_file: Some(file.path()),
            ..CredentialFields::default()
        }))?;
        assert_eq!(id.as_ref(), CLIENT);
        assert_eq!(key.to_string(), REDACTED);
        Ok(())
    }

    #[test]
    fn an_inline_key_supplies_the_signing_key() -> anyhow::Result<()> {
        let (_, key) = resolve_credentials(with_key(CredentialFields {
            private_key: Some(VALID_KEY),
            ..CredentialFields::default()
        }))?;
        assert_eq!(key.to_string(), REDACTED);
        Ok(())
    }

    #[test]
    fn an_identity_token_supplies_both_halves() -> anyhow::Result<()> {
        let token = token_for(CLIENT, VALID_KEY);
        let (id, key) = resolve_credentials(CredentialFields {
            identity_token: Some(&token),
            ..CredentialFields::default()
        })?;
        assert_eq!(id.as_ref(), CLIENT);
        assert_eq!(key.to_string(), REDACTED);

        let file = key_file(&token)?;
        let (id, _) = resolve_credentials(CredentialFields {
            identity_token_file: Some(file.path()),
            ..CredentialFields::default()
        })?;
        assert_eq!(id.as_ref(), CLIENT);
        Ok(())
    }

    #[rstest]
    #[case::mismatched_id(Some("ev1_other"), None, Some(TOKEN), &["ev1_other", CLIENT], &[VALID_KEY, "hf_secret"])]
    #[case::malformed_token(None, None, Some("not-a-token!"), &["auth export"], &["not-a-token!"])]
    #[case::key_without_client_id(None, Some(VALID_KEY), None, &["client_id is required"], &[])]
    #[case::two_sources(Some(CLIENT), Some(VALID_KEY), Some(TOKEN), &["[[devices]] private_key", "[[devices]] identity_token"], &[])]
    #[case::no_credentials(None, None, None, &["auth export"], &[])]
    fn credentials_are_refused_with_a_message_that_names_the_fault(
        #[case] client_id: Option<&str>,
        #[case] private_key: Option<&str>,
        #[case] identity_token: Option<&str>,
        #[case] must_say: &[&str],
        #[case] must_not_say: &[&str],
    ) -> anyhow::Result<()> {
        let token = token_for(CLIENT, VALID_KEY);
        let identity_token = identity_token.map(|value| {
            if value == TOKEN {
                token.as_str()
            } else {
                value
            }
        });
        let error = resolve_credentials(CredentialFields {
            client_id,
            private_key,
            identity_token,
            ..CredentialFields::default()
        })
        .err()
        .context("must be refused")?;
        let message = format!("{error:#}");
        must_say
            .iter()
            .for_each(|needle| assert!(message.contains(needle), "missing {needle:?}: {message}"));
        must_not_say
            .iter()
            .for_each(|needle| assert!(!message.contains(needle), "echoes {needle:?}: {message}"));
        Ok(())
    }

    #[test]
    fn a_client_id_beside_a_token_that_agrees_is_accepted() -> anyhow::Result<()> {
        let token = token_for(CLIENT, VALID_KEY);
        let (id, _) = resolve_credentials(with_key(CredentialFields {
            identity_token: Some(&token),
            ..CredentialFields::default()
        }))?;
        assert_eq!(id.as_ref(), CLIENT);
        Ok(())
    }

    #[test]
    fn a_key_file_readable_beyond_its_owner_is_refused() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let file = key_file(VALID_KEY)?;
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(0o644))
            .context("loosening the key file")?;
        let error = resolve_credentials(with_key(CredentialFields {
            private_key_file: Some(file.path()),
            ..CredentialFields::default()
        }))
        .err()
        .context("a group-readable key must be refused")?;
        let message = format!("{error:#}");
        assert!(
            message.contains("0644") && message.contains("rotate"),
            "{message}"
        );
        assert!(
            !message.contains(VALID_KEY),
            "never the key itself: {message}"
        );
        Ok(())
    }

    #[test]
    fn a_secret_access_key_never_prints() -> anyhow::Result<()> {
        let secret = Secret::from(" wJalrXUtnFEMI/K7MDENG\n".to_owned());
        assert_eq!(secret.expose(), "wJalrXUtnFEMI/K7MDENG");
        assert_eq!(format!("{secret:?}"), "<redacted>");
        Ok(())
    }

    #[rstest]
    #[case::trailing_newline("hunter2\n", 0o600, Ok("hunter2"))]
    #[case::trailing_crlf("hunter2\r\n", 0o600, Ok("hunter2"))]
    #[case::spaces_kept(" hunter2 \n", 0o600, Ok(" hunter2 "))]
    #[case::no_newline("hunter2", 0o600, Ok("hunter2"))]
    #[case::empty("\n", 0o600, Err("holds no password"))]
    #[case::two_lines("hunter2\n\n", 0o600, Err("cannot carry"))]
    #[case::quote("hun\"ter2\n", 0o600, Err("cannot carry"))]
    #[case::backslash("hun\\ter2\n", 0o600, Err("cannot carry"))]
    #[case::group_readable("hunter2\n", 0o640, Err("change the keychain's password"))]
    fn a_password_file_keeps_everything_but_one_newline(
        #[case] contents: &str,
        #[case] mode: u32,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let file = key_file(contents)?;
        std::fs::set_permissions(file.path(), std::fs::Permissions::from_mode(mode))?;
        match (read_password_file(file.path()), expected) {
            (Ok(password), Ok(want)) => {
                assert_eq!(password.expose(), want);
                assert_eq!(format!("{password:?}"), REDACTED);
            }
            (Err(error), Err(needle)) => {
                let message = format!("{error:#}");
                assert!(message.contains(needle), "{message}");
                assert!(!message.contains("hunter2"), "{message}");
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_key_file_that_is_not_a_key_names_its_path() -> anyhow::Result<()> {
        let file = key_file("not-a-key")?;
        let error = resolve_credentials(with_key(CredentialFields {
            private_key_file: Some(file.path()),
            ..CredentialFields::default()
        }))
        .err()
        .context("a malformed key must fail")?;
        let message = format!("{error:#}");
        assert!(
            message.contains(&file.path().display().to_string()),
            "the message must name the file: {message}"
        );
        assert!(!message.contains("not-a-key"), "{message}");
        Ok(())
    }

    #[test]
    fn redaction_replaces_an_auth_token_at_any_depth() -> anyhow::Result<()> {
        let spec = json!({
            "auth_token": "secret-top",
            "benchmark": {
                "cells": [
                    {"auth_token": "secret-nested", "model": "lfm2-1.2b"},
                    {"upload": {"auth_token": "secret-deep"}},
                ],
                "note": "auth_token appears in this text",
            },
            "runtime": {"kind": "mlx"},
        });
        let redacted = redacted_spec(&spec).to_string();
        assert!(
            !redacted.contains("secret-top")
                && !redacted.contains("secret-nested")
                && !redacted.contains("secret-deep"),
            "no token may survive at any depth: {redacted}"
        );
        assert_eq!(redacted.matches(REDACTED).count(), 3);
        // A value that merely mentions the key's name is untouched.
        assert!(redacted.contains("lfm2-1.2b"));
        assert!(redacted.contains("auth_token appears in this text"));
        assert!(redacted.contains("\"kind\":\"mlx\""));
        Ok(())
    }

    #[test]
    fn redaction_leaves_a_spec_with_no_token_alone() -> anyhow::Result<()> {
        let spec = json!({"runtime": {"kind": "mlx"}, "cells": [1, 2, null, true]});
        assert_eq!(redacted_spec(&spec), spec);
        Ok(())
    }

    #[test]
    fn scrubbing_removes_every_argv_token_from_printed_text() {
        let args = [
            "headlessrun".to_string(),
            r#"model={"auth_token":"hf_top","repo_name":"r"}"#.to_string(),
            r#"runtime-flags={"cells":[{"upload":{"auth_token":"hf_deep"}}]}"#.to_string(),
            r#"runtime={"auth_token":""}"#.to_string(),
            "benchmarks=b".to_string(),
        ];
        let secrets = argv_secrets(&args);
        assert_eq!(secrets, ["hf_top", "hf_deep"]);
        let printed = "loading r with hf_top\nretry hf_deep, hf_top\n".to_string();
        assert_eq!(
            scrub(printed, &secrets),
            "loading r with <redacted>\nretry <redacted>, <redacted>\n"
        );
    }
}
