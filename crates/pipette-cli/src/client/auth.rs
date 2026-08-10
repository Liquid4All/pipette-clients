//! Client registration and profile ops against the management server.
//!
//! Local identity files (keys, `registration.json`, `device.json`) are owned
//! by [`IdentityStore`] — this module holds only the flows that talk to the
//! server or combine both.

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    generate_keypair_hex,
    types::{ClientProfile, RegisterRequest, UpdateClientRequest},
    MgmtClient, PreauthKey,
};
use pipette_plan_types::device::{DeviceFormFactor, TrimmedString};

use crate::error::{Error, Result};
use crate::identity::{DeviceLabels, IdentityRegistration, IdentityStore};

pub struct RegisterInput {
    pub server_url: String,
    pub organization: String,
    pub contact_email: String,
    pub client_details: String,
    pub device_name: Option<String>,
    pub device_form_factor: Option<DeviceFormFactor>,
    /// Optional pre-auth key sent with the registration request; transient,
    /// never persisted.
    pub preauth_key: Option<PreauthKey>,
}

/// Register a new client into `identity`.
pub fn register(
    identity: &IdentityStore,
    client: &MgmtClient,
    http: &HttpClient,
    input: RegisterInput,
) -> Result<IdentityRegistration> {
    identity.ensure_empty()?;
    log::info!("registering client against {}", input.server_url);
    // An operator can run `auth set-device` before `auth register` to pin
    // labels for local benchmarks; preserve those when the corresponding
    // `RegisterInput` field isn't supplied.
    let prior = identity.get_device_labels()?;
    let (device_name, device_form_factor) =
        merge_labels_for_register(input.device_name, input.device_form_factor, prior);
    // `device.json` is independent of the registration (`ensure_empty` ignores
    // it, `clear_registration_material` preserves it), so write it before the
    // server call: an I/O failure aborts before a client_id is committed
    // instead of discarding an explicit `--device-name` after the fact.
    identity.put_device_labels(&DeviceLabels {
        device_name,
        device_form_factor,
    })?;
    let (private_key_hex, public_key_hex) = generate_keypair_hex()?;
    // Register before persisting the keypair: a rejected key (401/403) then
    // leaves no registration material on disk, so the operator can retry
    // without `auth reset`.
    let response = client.register(
        http,
        RegisterRequest {
            public_key: public_key_hex.clone(),
            organization: input.organization,
            contact_email: input.contact_email,
            client_details: input.client_details,
            preauth_key: input.preauth_key,
            device: Default::default(),
        },
    )?;
    let registration = IdentityRegistration {
        client_id: response.client_id,
        server_url: input.server_url,
    };
    // The server has committed (client_id assigned; a preauth key, if used, is
    // now spent). The private key + registration.json are the load-bearing
    // identity; if either fails to persist we can't recover the key, so roll
    // back to unblock a clean retry and report the client_id.
    identity
        .put_private_key(&private_key_hex)
        .and_then(|()| identity.put_public_key(&public_key_hex))
        .and_then(|()| identity.put_registration(&registration))
        .map_err(|source| {
            let _ = identity.clear_registration_material();
            Error::RegistrationPersisted {
                client_id: registration.client_id.clone(),
                source: Box::new(source),
            }
        })?;
    Ok(registration)
}

/// Merge `register`'s `RegisterInput` device fields with whatever is
/// already in `device.json`. The rule is "explicit user value beats prior;
/// otherwise inherit prior": `register` is allowed to narrow but never to
/// clobber an existing label with `None`.
///
/// Empty / whitespace-only `device_name` strings count as "no value" (via
/// [`non_empty_trimmed`]), so the operator can't accidentally erase a
/// pinned name by passing `""` either.
pub(crate) fn merge_labels_for_register(
    device_name: Option<String>,
    device_form_factor: Option<DeviceFormFactor>,
    prior: DeviceLabels,
) -> (Option<TrimmedString>, Option<DeviceFormFactor>) {
    (
        device_name
            .and_then(non_empty_trimmed)
            .or(prior.device_name),
        device_form_factor.or(prior.device_form_factor),
    )
}

/// Update local device labels in `device.json` (the single label store).
///
/// Does *not* require a prior `auth register`. A fresh workspace can call
/// `auth set-device` to record labels for local-only `benchmarks run`.
pub fn set_device(
    identity: &IdentityStore,
    device_name: Option<String>,
    device_form_factor: Option<DeviceFormFactor>,
) -> Result<DeviceLabels> {
    if device_name.is_none() && device_form_factor.is_none() {
        return Err(Error::SetDeviceEmpty);
    }
    let mut labels = identity.get_device_labels()?;
    if let Some(name) = device_name {
        labels.device_name = non_empty_trimmed(name);
    }
    if let Some(form_factor) = device_form_factor {
        labels.device_form_factor = Some(form_factor);
    }
    identity.put_device_labels(&labels)?;
    Ok(labels)
}

fn non_empty_trimmed(value: String) -> Option<TrimmedString> {
    let trimmed: TrimmedString = value.into();
    if trimmed.as_ref().is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub fn fetch_profile(
    identity: &IdentityStore,
    client: &MgmtClient,
    http: &HttpClient,
) -> Result<ClientProfile> {
    let auth = identity.signing_identity()?;
    Ok(client.me(http, &auth)?)
}

/// Update the server-side `client_details` for the registered client via
/// `PATCH /clients/me`. Unlike the device labels (`set_device`, local
/// `device.json`), `client_details` lives only on the server — so this is the
/// one `auth set-device` field that reaches the network and requires a
/// registered identity. Returns the updated profile.
pub fn update_client_details(
    identity: &IdentityStore,
    client: &MgmtClient,
    http: &HttpClient,
    client_details: String,
) -> Result<ClientProfile> {
    let auth = identity.signing_identity()?;
    Ok(client.update_me(
        http,
        &auth,
        UpdateClientRequest {
            client_details: Some(client_details),
            device: Default::default(),
        },
    )?)
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    fn identity() -> anyhow::Result<(tempfile::TempDir, IdentityStore)> {
        let tmp = tempfile::tempdir()?;
        let id = IdentityStore::new(tmp.path().join("identity"));
        Ok((tmp, id))
    }

    fn prior_labels(use_prior: bool) -> DeviceLabels {
        if use_prior {
            DeviceLabels {
                device_name: Some("prior-name".to_string().into()),
                device_form_factor: Some(DeviceFormFactor::Desktop),
            }
        } else {
            DeviceLabels::default()
        }
    }

    #[test]
    fn set_device_writes_device_json_without_registration() -> anyhow::Result<()> {
        let (_tmp, id) = identity()?;
        set_device(&id, Some("dev box".into()), Some(DeviceFormFactor::Desktop))?;
        let labels = id.get_device_labels()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("dev box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Desktop));
        assert!(id.get_registration()?.is_none());
        Ok(())
    }

    #[rstest::rstest]
    #[case::both_none(None, None, false, None, None)]
    #[case::input_wins(
        Some("new"),
        Some(DeviceFormFactor::Laptop),
        true,
        Some("new"),
        Some(DeviceFormFactor::Laptop)
    )]
    #[case::inherit_prior(None, None, true, Some("prior-name"), Some(DeviceFormFactor::Desktop))]
    #[case::empty_name_inherits(
        Some("  "),
        Some(DeviceFormFactor::Laptop),
        true,
        Some("prior-name"),
        Some(DeviceFormFactor::Laptop)
    )]
    fn merge_labels_for_register_cases(
        #[case] input_name: Option<&'static str>,
        #[case] input_form_factor: Option<DeviceFormFactor>,
        #[case] use_prior: bool,
        #[case] want_name: Option<&'static str>,
        #[case] want_form_factor: Option<DeviceFormFactor>,
    ) {
        let (got_name, got_form_factor) = merge_labels_for_register(
            input_name.map(str::to_string),
            input_form_factor,
            prior_labels(use_prior),
        );
        assert_eq!(got_name.as_ref().map(AsRef::as_ref), want_name);
        assert_eq!(got_form_factor, want_form_factor);
    }

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test/0")?)
    }

    fn register_input(
        server_url: &str,
        preauth_key: Option<&str>,
    ) -> anyhow::Result<RegisterInput> {
        Ok(RegisterInput {
            server_url: server_url.into(),
            organization: "LiquidAI".into(),
            contact_email: "user@example.com".into(),
            client_details: "ci box".into(),
            device_name: None,
            device_form_factor: None,
            preauth_key: preauth_key.map(PreauthKey::try_new).transpose()?,
        })
    }

    #[test]
    fn register_rejection_leaves_no_identity_on_disk() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(401).body("preauth key expired");
        });
        let (_tmp, id) = identity()?;
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let mut input = register_input(&server.base_url(), Some("preauth_dead.key"))?;
        input.device_name = Some("lab box".into());
        let err = register(&id, &client, &http, input)
            .err()
            .context("a 401 must fail registration")?;
        assert!(
            matches!(err, Error::Mgmt(_)),
            "expected mgmt error, got {err:?}"
        );
        assert!(id.get_private_key()?.is_none());
        assert!(id.get_registration()?.is_none());
        // Labels are written ahead of the server call and survive the failure,
        // as if the operator had run `auth set-device`.
        assert_eq!(
            id.get_device_labels()?
                .device_name
                .as_ref()
                .map(AsRef::as_ref),
            Some("lab box")
        );
        Ok(())
    }

    #[test]
    fn register_threads_preauth_key_without_persisting_it() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/clients/register")
                .body_contains("preauth_live.key");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let (_tmp, id) = identity()?;
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let registration = register(
            &id,
            &client,
            &http,
            register_input(&server.base_url(), Some("preauth_live.key"))?,
        )?;
        assert_eq!(registration.client_id, "c-1");
        mock.assert();

        let reg = id.require_registration()?;
        let persisted = serde_json::to_string(&reg)?;
        assert!(
            !persisted.contains("preauth"),
            "preauth key must not be persisted, got: {persisted}"
        );
        Ok(())
    }

    #[test]
    fn register_persists_labels_only_in_device_json() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let (_tmp, id) = identity()?;
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        let mut input = register_input(&server.base_url(), None)?;
        input.device_name = Some("lab box".into());
        input.device_form_factor = Some(DeviceFormFactor::Server);
        register(&id, &client, &http, input)?;

        // Labels land in device.json…
        let labels = id.get_device_labels()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("lab box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Server));
        // …and the registration carries no device fields at all.
        let reg = id.require_registration()?;
        let persisted = serde_json::to_string(&reg)?;
        assert!(
            !persisted.contains("device_"),
            "registration must not carry labels: {persisted}"
        );
        Ok(())
    }

    #[test]
    fn update_client_details_patches_server_profile() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let patch = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH)
                .path("/clients/me")
                .body_contains("M4 Max, 36GB");
            then.status(200).body(
                r#"{"client_id":"c-1","organization":"o","client_details":"M4 Max, 36GB","contact_email":"a@b.com","status":"approved"}"#,
            );
        });
        let (_tmp, id) = identity()?;
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        register(
            &id,
            &client,
            &http,
            register_input(&server.base_url(), None)?,
        )?;

        let profile = update_client_details(&id, &client, &http, "M4 Max, 36GB".to_string())?;

        assert_eq!(profile.client_details, "M4 Max, 36GB");
        patch.assert();
        Ok(())
    }

    #[test]
    fn register_persists_a_usable_signing_identity() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let (_tmp, id) = identity()?;
        let client = MgmtClient::new(server.base_url())?;
        let http = test_http()?;
        register(
            &id,
            &client,
            &http,
            register_input(&server.base_url(), None)?,
        )?;
        let auth = id.signing_identity()?;
        assert_eq!(auth.client_id, "c-1");
        assert!(!auth.private_key_hex.is_empty());
        Ok(())
    }
}
