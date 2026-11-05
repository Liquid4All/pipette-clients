//! Client registration and profile ops against the management server. The
//! local identity files belong to [`ClientRegistrationStore`] and
//! [`ClientDeviceStore`]; this module holds only the flows that talk to the
//! server.

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    ClientPreauthKey, ContactEmail, ServerUrl, client_api, generate_keypair,
    types::{ClientProfile, DeviceProfileFields, RegisterRequest, UpdateClientRequest},
};
use pipette_plan_types::{ClientId, DeviceFormFactor, TrimmedString};

use crate::error::{Error, Result};
use crate::identity::{
    ClientDeviceStore, ClientRegistrationStore, DeviceLabels, IdentityRegistration,
};

pub struct RegisterInput {
    pub organization: String,
    pub contact_email: ContactEmail,
    pub client_details: String,
    pub device_name: Option<String>,
    pub device_form_factor: Option<DeviceFormFactor>,
    /// Sent with the registration request, never persisted.
    pub preauth_key: Option<ClientPreauthKey>,
}

/// Register a new client into `registration_store`, pinning the labels it
/// registers with into `device`. The request carries the CPU architecture and
/// the OS family, so the server derives the `arch:` and `os_family:` flags
/// before the worker first reports the full profile.
pub fn register(
    registration_store: &ClientRegistrationStore,
    device: &ClientDeviceStore,
    server: &ServerUrl,
    http: &HttpClient,
    input: RegisterInput,
) -> Result<IdentityRegistration> {
    registration_store.ensure_empty()?;
    tracing::info!(
        url = %server,
        device_cpu_arch = pipette_device::CPU_ARCH,
        device_os_family = pipette_device::OS_FAMILY,
        "registering client"
    );
    // An operator can run `auth set-device` before `auth register` to pin
    // labels for local benchmarks; preserve those when the corresponding
    // `RegisterInput` field isn't supplied.
    let prior = device.get()?;
    let (device_name, device_form_factor) =
        merge_labels_for_register(input.device_name, input.device_form_factor, prior);
    // `device.json` is its own store, untouched by `ensure_empty` and by the
    // rollback below, so write it before the server call: an I/O failure aborts
    // before a client_id is committed instead of discarding an explicit
    // `--device-name` after the fact.
    device.put(&DeviceLabels {
        device_name,
        device_form_factor,
    })?;
    let (private_key, public_key_hex) = generate_keypair()?;
    // Register before persisting the keypair: a rejected key (401/403) then
    // leaves no registration material on disk, so the operator can retry
    // without `auth reset`.
    let response = client_api::register(
        server,
        http,
        RegisterRequest {
            public_key: public_key_hex.clone(),
            organization: input.organization,
            contact_email: input.contact_email,
            client_details: input.client_details,
            client_preauth_key: input.preauth_key,
            device: DeviceProfileFields {
                device_cpu_arch: Some(pipette_device::CPU_ARCH.to_string()),
                device_os_family: Some(pipette_device::OS_FAMILY.to_string()),
                ..Default::default()
            },
        },
    )?;
    let registration = IdentityRegistration {
        // Checked rather than trusted: the id is assigned by the server, but it
        // is this client that will sign with it, and an empty one, or one
        // outside the server's charset, would sign as nobody.
        client_id: ClientId::try_new(response.client_id).map_err(Error::RegisteredId)?,
        // The normalized URL, so what lands on disk is the base every later
        // request is built and signed against.
        server_url: server.clone(),
    };
    // The server has committed (client_id assigned; a preauth key, if used, is
    // now spent). The private key + registration.json are the load-bearing
    // identity; if either fails to persist we can't recover the key, so roll
    // back to unblock a clean retry and report the client_id.
    registration_store
        .put_private_key(&private_key)
        .and_then(|()| registration_store.put_public_key(&public_key_hex))
        .and_then(|()| registration_store.put(&registration))
        .map_err(|source| {
            let _ = registration_store.clear();
            Error::RegistrationPersisted {
                client_id: registration.client_id.to_string(),
                source: Box::new(source),
            }
        })?;
    Ok(registration)
}

/// An explicit value beats the label already in `device.json`; a missing one,
/// or a blank `device_name`, inherits it, so `register` never erases a pinned
/// label.
pub(crate) fn merge_labels_for_register(
    device_name: Option<String>,
    device_form_factor: Option<DeviceFormFactor>,
    prior: DeviceLabels,
) -> (Option<TrimmedString>, Option<DeviceFormFactor>) {
    (
        device_name
            .and_then(DeviceLabels::supplied_name)
            .or(prior.device_name),
        device_form_factor.or(prior.device_form_factor),
    )
}

/// This client's server-side profile, live from `GET /clients/me`.
///
/// The destination comes out of the same store as the signing key: a profile
/// fetched from anywhere else would describe a client this key does not sign
/// for.
pub fn fetch_profile(
    registration: &ClientRegistrationStore,
    http: &HttpClient,
) -> Result<ClientProfile> {
    let auth = registration.signing_identity()?;
    Ok(client_api::me(&registration.server_url()?, http, &auth)?)
}

/// `PATCH /clients/me` with new `client_details`. Unlike the device labels in
/// `device.json`, `client_details` lives only on the server, so it is the one
/// `auth set-device` field that needs a registered identity.
pub fn update_client_details(
    registration: &ClientRegistrationStore,
    http: &HttpClient,
    client_details: String,
) -> Result<ClientProfile> {
    let auth = registration.signing_identity()?;
    Ok(client_api::update_me(
        &registration.server_url()?,
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

    use pipette_plan_types::PrincipalIdError;

    use super::*;

    /// Both stores over one `identity/`, as the workspace mints them.
    fn stores() -> anyhow::Result<(
        tempfile::TempDir,
        ClientRegistrationStore,
        ClientDeviceStore,
    )> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("identity");
        let id = ClientRegistrationStore::new(&root);
        let device = ClientDeviceStore::new(&root);
        Ok((tmp, id, device))
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

    fn server_url(server: &httpmock::MockServer) -> anyhow::Result<ServerUrl> {
        Ok(ServerUrl::try_new(server.base_url())?)
    }

    fn register_input(preauth_key: Option<&str>) -> anyhow::Result<RegisterInput> {
        Ok(RegisterInput {
            organization: "LiquidAI".into(),
            contact_email: ContactEmail::try_new("user@example.com")?,
            client_details: "ci box".into(),
            device_name: None,
            device_form_factor: None,
            preauth_key: preauth_key.map(ClientPreauthKey::try_new).transpose()?,
        })
    }

    /// A registration the server refuses, or answers with an id this client
    /// could not sign as, leaves no identity on disk, so a retry needs no
    /// `auth reset`. Labels are written ahead of the server call and survive
    /// the failure, as if the operator had run `auth set-device`.
    #[rstest::rstest]
    #[case::rejected(401, "preauth key expired", |e: &Error| matches!(e, Error::Mgmt(_)))]
    #[case::unusable_id(
        200,
        r#"{"client_id":"ev1.a","status":"approved"}"#,
        |e: &Error| matches!(e, Error::RegisteredId(PrincipalIdError::Charset { ch: '.', .. }))
    )]
    fn register_failure_leaves_no_identity_on_disk(
        #[case] status: u16,
        #[case] body: &str,
        #[case] refused: fn(&Error) -> bool,
    ) -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(status).body(body);
        });
        let (_tmp, id, device) = stores()?;
        let mut input = register_input(Some("preauth_dead.key"))?;
        input.device_name = Some("lab box".into());
        let err = register(&id, &device, &server_url(&server)?, &test_http()?, input)
            .err()
            .context("the registration must fail")?;
        assert!(refused(&err), "got {err:?}");
        assert!(id.get_private_key()?.is_none());
        assert!(id.get()?.is_none());
        assert_eq!(
            device.get()?.device_name.as_ref().map(AsRef::as_ref),
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
                // The current field name, not the `preauth_key` alias.
                .body_includes("\"client_preauth_key\"")
                .body_includes("preauth_live.key");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let (_tmp, id, device) = stores()?;
        let server_url = server_url(&server)?;
        let http = test_http()?;
        let registration = register(
            &id,
            &device,
            &server_url,
            &http,
            register_input(Some("preauth_live.key"))?,
        )?;
        assert_eq!(registration.client_id.as_ref(), "c-1");
        mock.assert();

        let reg = id.require()?;
        let persisted = serde_json::to_string(&reg)?;
        assert!(
            !persisted.contains("preauth"),
            "preauth key must not be persisted, got: {persisted}"
        );
        Ok(())
    }

    #[test]
    fn register_sends_the_arch_and_os_family() -> anyhow::Result<()> {
        use httpmock::prelude::*;
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/clients/register")
                .json_body_includes(
                    serde_json::json!({
                        "device_cpu_arch": std::env::consts::ARCH,
                        "device_os_family": std::env::consts::OS,
                    })
                    .to_string(),
                );
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let (_tmp, id, device) = stores()?;
        register(
            &id,
            &device,
            &server_url(&server)?,
            &test_http()?,
            register_input(None)?,
        )?;
        mock.assert();
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
        let (_tmp, id, device) = stores()?;
        let server_url = server_url(&server)?;
        let http = test_http()?;
        let mut input = register_input(None)?;
        input.device_name = Some("lab box".into());
        input.device_form_factor = Some(DeviceFormFactor::Server);
        register(&id, &device, &server_url, &http, input)?;

        let labels = device.get()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("lab box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Server));
        let reg = id.require()?;
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
                .body_includes("M4 Max, 36GB");
            then.status(200).body(
                r#"{"client_id":"c-1","organization":"o","client_details":"M4 Max, 36GB","contact_email":"a@b.com","status":"approved"}"#,
            );
        });
        let (_tmp, id, device) = stores()?;
        let server_url = server_url(&server)?;
        let http = test_http()?;
        register(&id, &device, &server_url, &http, register_input(None)?)?;

        let profile = update_client_details(&id, &http, "M4 Max, 36GB".to_string())?;

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
        let (_tmp, id, device) = stores()?;
        let server_url = server_url(&server)?;
        let http = test_http()?;
        register(&id, &device, &server_url, &http, register_input(None)?)?;
        let auth = id.signing_identity()?;
        assert_eq!(auth.client_id.as_ref(), "c-1");
        // The stored halves are a pair: the public key on disk is the one the
        // key it signs with derives.
        assert_eq!(
            id.get_public_key()?,
            Some(pipette_mgmt_client::public_key_hex_from_private(
                &auth.private_key_hex
            )?)
        );
        Ok(())
    }
}
