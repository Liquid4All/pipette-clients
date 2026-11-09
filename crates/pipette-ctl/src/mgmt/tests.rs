use std::{collections::HashSet, fs, io::Read, net::TcpListener, thread, time::Duration};

use anyhow::Context;
use clap::CommandFactory;
use httpmock::prelude::*;
use serde_json::json;
use strum::{IntoEnumIterator as _, VariantNames as _};

use pipette_mgmt_client::generate_keypair;
use pipette_plan_types::{
    Answered, AuditActorKind, AuditOutcome, BenchmarkType, CampaignCreated, CampaignRevised,
    ClientPreauthUsage, ClientStatus, DescriptorEntryKind, DescriptorEntryState, DescriptorKind,
    FleetOwner, ListOrder, OrgStatus, OwnerFilter, PlanDryRun, PlanFiled, PlanStatus,
    RequestedScope, RevisionDryRun, UserAccountType, UserId, UserKeyId, UserStatus, VaultModel,
    VaultRuntime, VaultUploadFile,
};

use super::{catalog::ROUTES, *};
use crate::test_support::signed_as;

fn session(server: &MockServer, signed: bool) -> anyhow::Result<Session> {
    session_in(server, signed, Path::new("."))
}

fn session_in(server: &MockServer, signed: bool, work_dir: &Path) -> anyhow::Result<Session> {
    let identity = if signed {
        Ok(user_identity()?)
    } else {
        Err("no identity here".to_string())
    };
    Ok(Session {
        server: ServerUrl::try_new(server.base_url())?,
        http: crate::http_client()?,
        bulk: crate::http_client()?,
        identity,
        work_dir: work_dir.to_path_buf(),
    })
}

fn user_identity() -> anyhow::Result<UserIdentity> {
    let (private_key_hex, _) = generate_keypair()?;
    Ok(UserIdentity::with_private_key(
        UserId::try_new("user_0192f0c1")?,
        UserKeyId::try_new("userkey_9f2c")?,
        private_key_hex,
    ))
}

fn route(tool: &str) -> anyhow::Result<&'static Route> {
    ROUTES
        .iter()
        .find(|route| route.tool() == tool)
        .with_context(|| format!("no route {tool}"))
}

#[test]
fn catalog_reads_use_signed_routes_and_local_device_registration() -> anyhow::Result<()> {
    let server = MockServer::start();
    let page = json!({"items": [], "next_cursor": null, "unreadable": 0});
    let device_page = json!({"items": [], "next_cursor": null, "unreadable": 0, "withheld": {"member_count": 0, "entry_count": 0, "members": []}});
    let list = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/descriptor-catalog")
            .query_param("kind", "stable,daily")
            .query_param("members", "true");
        then.status(200).json_body(page.clone());
    });
    let names = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/descriptor-catalog/names")
            .query_param("names", "global/x");
        then.status(200)
            .json_body(json!({"entries": [], "unknown": ["global/x"]}));
    });
    let device = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/descriptor-catalog/devices/ev1_local");
        then.status(200).json_body(device_page.clone());
    });
    let override_device = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/descriptor-catalog/devices/ev1_override");
        then.status(200).json_body(device_page.clone());
    });
    let work = tempfile::tempdir()?;
    let registration_dir = work.path().join(".pipette/identity");
    fs::create_dir_all(&registration_dir)?;
    fs::write(
        registration_dir.join("registration.json"),
        json!({"client_id": "ev1_local", "server_url": server.base_url()}).to_string(),
    )?;
    let session = session_in(&server, true, work.path())?;
    session.invoke(
        route("catalog_list")?,
        &args(json!({"kind": ["stable", "daily"], "members": true}))?,
    )?;
    session.invoke(route("catalog_show")?, &args(json!({"names": "global/x"}))?)?;
    session.invoke(route("catalog_for_device")?, &args(json!({}))?)?;
    list.assert();
    names.assert();
    device.assert();
    session.invoke(
        route("catalog_for_device")?,
        &args(json!({"client_id": "ev1_override"}))?,
    )?;
    override_device.assert();
    fs::remove_file(registration_dir.join("registration.json"))?;
    let Err(missing) = session.invoke(route("catalog_for_device")?, &args(json!({}))?) else {
        anyhow::bail!("for-device succeeded without a registration file");
    };
    assert!(missing.to_string().contains("pass the device's client id"));
    let Err(bad_fields) =
        session.invoke(route("catalog_for_device")?, &args(json!({"fields": []}))?)
    else {
        anyhow::bail!("for-device accepted an empty `fields`");
    };
    assert!(
        bad_fields.to_string().contains("`fields`"),
        "{bad_fields:#}"
    );
    Ok(())
}

const REGISTERED: &str = "https://mgmt.example";

/// A registration that cannot name the device sends the caller to an explicit
/// `client_id`.
#[rstest::rstest]
#[case::missing_file(None)]
#[case::unparsable(Some("{".to_owned()))]
#[case::no_client_id(Some(json!({"server_url": REGISTERED}).to_string()))]
#[case::empty_client_id(Some(json!({"client_id": "", "server_url": REGISTERED}).to_string()))]
#[case::no_server_url(Some(json!({"client_id": "ev1_local"}).to_string()))]
#[case::other_server(Some(
    json!({"client_id": "ev1_local", "server_url": "https://other.example"}).to_string()
))]
fn local_catalog_device_id_refuses_an_unusable_registration(
    #[case] registration: Option<String>,
) -> anyhow::Result<()> {
    let work = tempfile::tempdir()?;
    if let Some(body) = registration {
        let dir = work.path().join(".pipette/identity");
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("registration.json"), body)?;
    }
    let Err(error) = local_catalog_device_id(work.path(), &ServerUrl::try_new(REGISTERED)?) else {
        anyhow::bail!("an unusable registration named a device");
    };
    assert!(error.to_string().contains("pass the device's client id"));
    Ok(())
}

fn args(value: Value) -> anyhow::Result<Args> {
    let Value::Object(args) = value else {
        anyhow::bail!("not an object: {value}");
    };
    Ok(args)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn plan_filed() -> Value {
    json!({
        "fleet_detail": "full",
        "plan_id": "plan_1",
        "job_count": 1,
        "merged_job_count": 0,
        "job_ids": { "0": "job-1" },
        "warnings": [],
    })
}

fn plan_dry_run() -> Value {
    json!({
        "document": "benchmarks = [\"decode_throughput_512_100\"]\n",
        "bodies": [],
        "plan_warnings": [],
        "imports": [],
    })
}

fn campaign_created() -> Value {
    json!({
        "campaign_id": "campaign_1",
        "name": "nightly",
        "owner": { "type": "user", "user_id": "user_0192f0c1" },
        "current_plan_id": "plan_1",
        "created_at": "2026-01-02T03:04:05Z",
        "updated_at": "2026-01-02T03:04:05Z",
        "plan": plan_filed(),
    })
}

fn campaign_revised(previous_plan_id: &str, change: Value) -> Value {
    let mut plan = plan_filed();
    if let Value::Object(report) = &mut plan {
        report.insert("change".into(), change);
    }
    json!({
        "previous_plan_id": previous_plan_id,
        "carried_job_count": 0,
        "rerun_job_count": 0,
        "recalculated_job_count": 0,
        "dropped_job_count": 0,
        "cancel_requested": true,
        "plan": plan,
    })
}

fn revision_dry_run() -> Value {
    json!({
        "document": "benchmarks = [\"decode_throughput_512_100\"]\n",
        "bodies": [],
        "plan_warnings": [],
        "imports": [],
        "unchanged": false,
        "carried_job_count": 0,
        "new_job_count": 0,
        "requirements_rerun_job_count": 0,
        "satisfied_job_count": 0,
        "restored_job_count": 0,
        "satisfied_job_count": 0,
        "inherited_job_count": 0,
        "rerun_job_count": 0,
        "recalculated_job_count": 0,
        "dropped_job_ids": [],
        "held_jobs": [],
        "repeated_variants": [],
    })
}

fn as_mgmt<T: serde::Serialize + serde::de::DeserializeOwned>(
    answer: Value,
) -> serde_json::Result<Value> {
    serde_json::to_value(serde_json::from_value::<T>(answer)?)
}

/// The mocks stand in for mgmt, so a re-pin that renames a field of its
/// answers fails here rather than leaving them stale.
#[rstest::rstest]
#[case::plan_filed(plan_filed(), as_mgmt::<PlanFiled>)]
#[case::plan_dry_run(plan_dry_run(), as_mgmt::<PlanDryRun>)]
#[case::campaign_created(campaign_created(), as_mgmt::<CampaignCreated>)]
#[case::campaign_revised(
    campaign_revised("plan_0", json!({ "type": "document" })),
    as_mgmt::<CampaignRevised>,
)]
#[case::revision_dry_run(revision_dry_run(), as_mgmt::<RevisionDryRun>)]
fn the_mock_answers_read_back_as_mgmts(
    #[case] answer: Value,
    #[case] read_back: fn(Value) -> serde_json::Result<Value>,
) -> anyhow::Result<()> {
    assert_eq!(read_back(answer.clone())?, answer);
    Ok(())
}

/// Catalog text names a route and an argument by name, so neither may repeat.
#[test]
fn tools_and_argument_names_are_unique() -> anyhow::Result<()> {
    let mut tools = HashSet::new();
    for route in ROUTES {
        anyhow::ensure!(
            tools.insert(route.tool()),
            "duplicate tool {}",
            route.tool()
        );
        let mut names = HashSet::new();
        for name in route
            .path_params()
            .chain(route.fields().map(|field| field.name))
        {
            anyhow::ensure!(names.insert(name), "{} repeats `{name}`", route.tool());
        }
    }
    Ok(())
}

/// clap checks the generated verbs only in debug builds, at parse time; this
/// runs the same check over every verb at once.
#[test]
fn every_generated_verb_is_a_valid_clap_command() {
    crate::Cli::command().debug_assert();
}

/// The wire words of an enum without `EnumIter`, as serde writes them. The
/// exhaustive match stops a variant mgmt adds compiling until it is listed.
macro_rules! wire_words {
    ($enum:ident: $($variant:ident),+ $(; $rest:pat)?) => {{
        let _exhaustive = |value: $enum| match value {
            $($enum::$variant)|+ => (),
            $($rest => ())?
        };
        [$($enum::$variant),+]
            .iter()
            .map(|value| match serde_json::to_value(value) {
                Ok(Value::String(word)) => word,
                other => format!("{other:?}"),
            })
            .collect::<Vec<_>>()
    }};
}

/// ctl refuses a value outside these sets before sending, and the server
/// compares a filter verbatim, so a stale list either blocks a new value or
/// lists nothing.
#[rstest::rstest]
#[case::benchmark_type(
    catalog::BENCHMARK_TYPES,
    BenchmarkType::ALL.iter().map(ToString::to_string).collect(),
)]
#[case::plan_status(
    catalog::PLAN_STATUSES,
    PlanStatus::iter().map(|status| status.label().to_owned()).collect(),
)]
#[case::plan_scope(catalog::PLAN_SCOPES, wire_words!(RequestedScope: Mine, All))]
#[case::rerun_answered(catalog::RERUN_ANSWERED, wire_words!(Answered: Failed, All))]
#[case::vault_type(
    catalog::VAULT_TYPES,
    [VaultModel::<VaultUploadFile>::VARIANTS, VaultRuntime::<VaultUploadFile>::VARIANTS]
        .concat()
        .into_iter()
        .map(str::to_owned)
        .collect(),
)]
#[case::owner_filter(
    catalog::OWNER_FILTERS,
    OwnerFilter::iter().map(|owner| owner.label().to_owned()).collect(),
)]
#[case::client_status(
    catalog::CLIENT_STATUSES,
    wire_words!(ClientStatus: Pending, Approved; ClientStatus::Unknown(_)),
)]
#[case::user_status(
    catalog::USER_STATUSES,
    wire_words!(UserStatus: Active, Pending, Disabled),
)]
#[case::org_status(catalog::ORG_STATUSES, wire_words!(OrgStatus: Active, Disabled))]
#[case::account_type(
    catalog::ACCOUNT_TYPES,
    wire_words!(UserAccountType: Human, Service),
)]
#[case::fleet_owner(catalog::FLEET_OWNERS, wire_words!(FleetOwner: Mine))]
#[case::list_order(catalog::LIST_ORDERS, wire_words!(ListOrder: Oldest, Newest))]
#[case::client_preauth_usage(
    catalog::CLIENT_PREAUTH_USAGES,
    wire_words!(ClientPreauthUsage: SingleUse, MultiUse),
)]
#[case::audit_outcome(
    catalog::AUDIT_OUTCOMES,
    wire_words!(AuditOutcome: Succeeded, Refused, Failed),
)]
#[case::audit_actor(
    catalog::AUDIT_ACTORS,
    wire_words!(AuditActorKind: User, Client, Unverified, Host),
)]
#[case::catalog_kind(
    catalog::CATALOG_KINDS,
    wire_words!(DescriptorEntryKind: Stable, Daily, HandMade),
)]
#[case::catalog_state(
    catalog::CATALOG_STATES,
    wire_words!(DescriptorEntryState: Pinned, Active, Hidden; DescriptorEntryState::Tombstoned),
)]
#[case::descriptor_kind(
    catalog::DESCRIPTOR_KINDS,
    wire_words!(DescriptorKind: Runtime, Model),
)]
fn closed_sets_match_plan_types(#[case] listed: &[&str], #[case] known: Vec<String>) {
    assert_eq!(listed, known);
}

#[test]
fn a_list_filter_repeats_its_key_and_ids_stay_one_segment() -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/clients")
            .query_param("status", "approved")
            .query_param("tag", "lab")
            .query_param("tag", "rack-2")
            .query_param("limit", "5")
            .is_true(|request| signed_as(request, "userkey_9f2c"));
        then.status(200)
            .json_body(json!({ "items": [], "next_cursor": null, "unreadable": 0 }));
    });
    session(&server, true)?.invoke(
        route("clients_list")?,
        &args(json!({ "status": "approved", "tag": ["lab", "rack-2"], "limit": 5 }))?,
    )?;
    mock.assert();
    Ok(())
}

/// Models send `null` for an argument they mean to leave out, and a query
/// string has no `null` to send.
#[test]
fn a_null_argument_is_left_out() -> anyhow::Result<()> {
    let server = MockServer::start();
    let clients = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/clients")
            .query_param_missing("cursor")
            .query_param_missing("status")
            .query_param("limit", "2");
        then.status(200)
            .json_body(json!({ "items": [], "next_cursor": null, "unreadable": 0 }));
    });
    let benchmarks = server.mock(|when, then| {
        when.method(GET)
            .path("/benchmarks")
            .query_param_missing("type");
        then.status(200).json_body(json!([]));
    });
    let create = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/users")
            .json_body(json!({ "contact_email": "a@b.test" }));
        then.status(201).json_body(json!({ "user_id": "user_1" }));
    });
    let session = session(&server, true)?;
    session.invoke(
        route("clients_list")?,
        &args(json!({ "cursor": null, "status": null, "limit": 2 }))?,
    )?;
    session.invoke(route("benchmarks_list")?, &args(json!({ "type": null }))?)?;
    session.invoke(
        route("users_create")?,
        &args(json!({ "contact_email": "a@b.test", "permissions": null, "org_id": null }))?,
    )?;
    clients.assert();
    benchmarks.assert();
    create.assert();
    Ok(())
}

#[test]
fn an_owner_id_expands_to_the_object_the_server_takes() -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/campaigns/campaign_1/owner")
            .json_body(json!({ "owner": { "type": "org", "org_id": "org_acme" } }));
        then.status(200)
            .json_body(json!({ "campaign_id": "campaign_1", "changed": true }));
    });
    let answer = session(&server, true)?.invoke(
        route("campaigns_set_owner")?,
        &args(json!({ "campaign_id": "campaign_1", "owner": "org_acme" }))?,
    )?;
    assert_eq!(answer.value["changed"], true);
    mock.assert();
    Ok(())
}

#[test]
fn a_device_owner_detaches_with_none_or_null() -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/clients/ev1_a/owner")
            .json_body(json!({ "owner": null }));
        then.status(200).json_body(json!({ "changed": true }));
    });
    let session = session(&server, true)?;
    for owner in [json!("none"), Value::Null] {
        session.invoke(
            route("clients_set_owner")?,
            &args(json!({ "client_id": "ev1_a", "owner": owner }))?,
        )?;
    }
    mock.assert_calls(2);
    Ok(())
}

/// mgmt clears a membership only on an explicit `null`, which `none` spells
/// too; an organization id goes as given.
#[test]
fn a_membership_clears_with_none_or_null() -> anyhow::Result<()> {
    let cleared = json!({
        "user_id": "user_1",
        "contact_email": "a@b.test",
        "account_type": "human",
        "status": "active",
        "created_at": "2026-01-02T03:04:05Z",
        "changed": true,
    });
    let mut joined = cleared.clone();
    joined["org_id"] = json!("org_acme");
    let server = MockServer::start();
    let clear = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/users/user_1/org")
            .json_body(json!({ "org_id": null }));
        then.status(200).json_body(cleared);
    });
    let join = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/users/user_1/org")
            .json_body(json!({ "org_id": "org_acme" }));
        then.status(200).json_body(joined);
    });
    let session = session(&server, true)?;
    [json!("none"), Value::Null, json!("org_acme")]
        .into_iter()
        .try_for_each(|org_id| {
            session
                .invoke(
                    route("users_set_org")?,
                    &args(json!({ "user_id": "user_1", "org_id": org_id }))?,
                )
                .map(drop)
        })?;
    clear.assert_calls(2);
    join.assert();
    Ok(())
}

/// Each refusal happens before a request is sent, so the mock sees none.
#[rstest::rstest]
#[case::unknown_argument(
    "orgs_create",
    json!({ "slug": "acme", "displayname": "Acme" }),
    "takes no argument named displayname",
)]
#[case::missing_required("orgs_create", json!({ "display_name": "Acme" }), "`slug` is required")]
#[case::status_outside_its_set(
    "orgs_update",
    json!({ "org_id": "acme", "status": "archived" }),
    "`status` must be one of active, disabled",
)]
#[case::missing_path_param("orgs_show", json!({}), "`org_id` is required")]
#[case::unknown_permission(
    "users_set_permissions",
    json!({ "user_id": "user_1", "permissions": ["root"] }),
    "`permissions` must be a list of",
)]
// mgmt refuses a membership change that names no `org_id`.
#[case::membership_without_org(
    "users_set_org",
    json!({ "user_id": "user_1" }),
    "`org_id` is required",
)]
#[case::owner_by_handle(
    "clients_set_owner",
    json!({ "client_id": "ev1_a", "owner": "acme" }),
    "`owner` `acme` is neither",
)]
// A campaign always has an owner.
#[case::campaign_owner_none(
    "campaigns_set_owner",
    json!({ "campaign_id": "campaign_1", "owner": "none" }),
    "`owner` cannot be cleared here",
)]
#[case::campaign_owner_null(
    "campaigns_set_owner",
    json!({ "campaign_id": "campaign_1", "owner": null }),
    "`owner` cannot be cleared here",
)]
// The URL would resolve these away and reach the parent route.
#[case::client_dot_dot(
    "clients_show",
    json!({ "client_id": ".." }),
    "`..` cannot be a path parameter",
)]
#[case::org_dot("orgs_show", json!({ "org_id": "." }), "`.` cannot be a path parameter")]
#[case::benchmark_dot_dot(
    "orgs_benchmark_delete",
    json!({ "org_id": "acme", "benchmark_id": ".." }),
    "`..` cannot be a path parameter",
)]
// Words another route holds at that position.
#[case::client_capabilities(
    "clients_show",
    json!({ "client_id": "capabilities" }),
    "`client_id` cannot be `capabilities`",
)]
#[case::user_me("users_show", json!({ "user_id": "me" }), "`user_id` cannot be `me`")]
#[case::user_register(
    "users_show",
    json!({ "user_id": "register" }),
    "`user_id` cannot be `register`",
)]
#[case::plan_validate(
    "plans_status",
    json!({ "plan_id": "validate" }),
    "`plan_id` cannot be `validate`",
)]
#[case::key_owner_me(
    "users_key_revoke",
    json!({ "user_id": "me", "key_id": "userkey_1" }),
    "`user_id` cannot be `me`",
)]
#[case::model_key_dot_dot(
    "model_params_delete",
    json!({ "key": ".." }),
    "`..` cannot be a path parameter",
)]
// A pasted token's part after `.` is its secret, which the URL would carry.
#[case::user_preauth_token(
    "users_preauth_revoke",
    json!({ "user_id": "user_1", "preauth_key_id": "user_preauth_0123456789abcdef.secret" }),
    "`preauth_key_id`: user_preauth_key_id contains invalid character '.'",
)]
#[case::device_preauth_token(
    "orgs_client_preauth_revoke",
    json!({ "org_id": "acme", "preauth_key_id": "client_preauth_fedcba9876543210.secret" }),
    "`preauth_key_id`: client_preauth_key_id contains invalid character '.'",
)]
// Values the kind or the invoker refuses.
#[case::negative_limit(
    "clients_list",
    json!({ "limit": -1 }),
    "`limit` must be a non-negative integer",
)]
#[case::negative_offset(
    "plans_jobs",
    json!({ "plan_id": "plan-1", "offset": -1 }),
    "`offset` must be a non-negative integer",
)]
#[case::all_with_cursor("clients_list", json!({ "all": true, "cursor": "ev1_b" }), "drop `cursor`")]
#[case::unknown_benchmark_type(
    "benchmarks_list",
    json!({ "type": "prefill" }),
    "`type` must be one of prefill_throughput",
)]
#[case::unknown_outcome(
    "audit_list",
    json!({ "outcome": ["denied"] }),
    "`outcome` must be a list of succeeded, refused, failed",
)]
#[case::actors_as_one_string(
    "audit_list",
    json!({ "actor": "user,host" }),
    "`actor` must be a list of user, client, unverified, host",
)]
#[case::user_permission_on_a_client(
    "clients_set_permissions",
    json!({ "client_id": "ev1_a", "permissions": ["manage_clients"] }),
    "`permissions` must be a list of",
)]
#[case::unknown_usage(
    "orgs_client_preauth_create",
    json!({ "org_id": "acme", "usage": "forever" }),
    "`usage` must be one of single_use, multi_use",
)]
#[case::principal_by_handle(
    "vault_blob_exists",
    json!({ "principal_id": "acme", "sha256": "a".repeat(64) }),
    "principal id must start with `org_` or `user_`",
)]
#[case::short_digest(
    "vault_blob_exists",
    json!({ "principal_id": "user_1", "sha256": "abc" }),
    "`sha256` must be 64 hex characters",
)]
#[case::definition_not_an_object(
    "orgs_benchmark_put",
    json!({ "org_id": "acme", "benchmark_id": "b", "definition": "[1]" }),
    "`definition` must be a JSON object",
)]
// Neither JSON, nor a file, nor TOML.
#[case::definition_not_toml(
    "orgs_benchmark_put",
    json!({ "org_id": "acme", "benchmark_id": "b", "definition": "not toml" }),
    "neither JSON nor TOML",
)]
#[case::definition_broken_json(
    "orgs_benchmark_put",
    json!({ "org_id": "acme", "benchmark_id": "b", "definition": r#"{"a": 1,}"# }),
    "is not JSON: trailing comma",
)]
#[case::job_dot_dot("plans_by_job", json!({ "job_id": ".." }), "`..` cannot be a path parameter")]
#[case::overwrite_not_boolean(
    "vault_download",
    json!({ "resource_id": "vault_1", "out": "x", "overwrite": "yes" }),
    "`overwrite` must be a boolean",
)]
#[case::empty_fields(
    "orgs_list",
    json!({ "fields": [] }),
    "`fields` must be a non-empty list of keys",
)]
// The fleet listing's owner is only ever the caller.
#[case::fleet_owner_not_mine(
    "clients_list",
    json!({ "owner": "org" }),
    "`owner` must be one of mine",
)]
#[case::unknown_account_type(
    "users_create",
    json!({ "contact_email": "a@b.test", "account_type": "robot" }),
    "`account_type` must be one of human, service",
)]
#[case::set_type_without_type(
    "users_set_type",
    json!({ "user_id": "user_1" }),
    "`account_type` is required",
)]
#[case::unknown_order(
    "vault_list",
    json!({ "order": "latest" }),
    "`order` must be one of oldest, newest",
)]
#[case::label_not_a_string(
    "campaigns_rename",
    json!({ "campaign_id": "campaign_1", "labels": { "team": 7 } }),
    "`labels` must be an object of string values",
)]
#[case::labels_not_an_object(
    "campaigns_rename",
    json!({ "campaign_id": "campaign_1", "labels": "team=ml" }),
    "`labels` must be an object of string values",
)]
#[case::inherit_from_not_a_list(
    "campaigns_revise",
    json!({ "campaign_id": "campaign_1", "base_plan_id": "plan-0", "plan": "a = 1", "inherit_from": "plan-1" }),
    "`inherit_from` must be a list of strings",
)]
// Only `campaigns_variants` takes `add_from`.
#[case::add_from_on_a_revise(
    "campaigns_revise",
    json!({ "campaign_id": "campaign_1", "base_plan_id": "plan-0", "plan": "a = 1", "add_from": ["plan-1"] }),
    "takes no argument named add_from",
)]
// Only an optional document may be empty; `revise` requires `plan`.
#[case::empty_revise_plan(
    "campaigns_revise",
    json!({ "campaign_id": "campaign_1", "base_plan_id": "plan-0", "plan": "" }),
    "holds no TOML",
)]
// A write takes no `fields`, which could only fail after it ran.
#[case::fields_on_a_create(
    "orgs_create",
    json!({ "slug": "acme", "fields": ["org_id"] }),
    "`orgs_create` takes no argument named fields",
)]
#[case::fields_on_a_delete(
    "orgs_delete",
    json!({ "org_id": "acme", "fields": ["org_id"] }),
    "`orgs_delete` takes no argument named fields",
)]
fn bad_arguments_are_refused_before_any_request(
    #[case] tool: &str,
    #[case] bad: Value,
    #[case] reason: &str,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.any_request();
        then.status(200);
    });
    let error = session(&server, true)?
        .invoke(route(tool)?, &args(bad.clone())?)
        .err()
        .with_context(|| format!("{tool} accepted {bad}"))?;
    anyhow::ensure!(
        format!("{error:#}").contains(reason),
        "{tool} {bad}: {error:#}"
    );
    mock.assert_calls(0);
    Ok(())
}

#[test]
fn a_reserved_word_is_named_with_the_route_it_would_reach() -> anyhow::Result<()> {
    let server = MockServer::start();
    let refused = session(&server, true)?
        .invoke(
            route("clients_show")?,
            &args(json!({ "client_id": "capabilities" }))?,
        )
        .err()
        .context("`capabilities` is another route")?;
    let message = refused.to_string();
    assert!(
        message.contains("/api/v1/clients/capabilities"),
        "{message}"
    );
    Ok(())
}

/// `when` that also matches each of `query`'s pairs.
fn matching_query<K: AsRef<str>, V: AsRef<str>>(
    when: httpmock::When,
    query: &[(K, V)],
) -> httpmock::When {
    query.iter().fold(when, |when, (name, value)| {
        when.query_param(name.as_ref(), value.as_ref())
    })
}

/// A listing in two pages: `first` with no cursor, `second` at `c1`, each
/// asked with exactly `query` and its cursor.
fn two_pages(server: &MockServer, path: &str, query: &[(&str, &str)], first: Value, second: Value) {
    [(None, first), (Some("c1"), second)]
        .into_iter()
        .for_each(|(cursor, page)| {
            server.mock(|when, then| {
                let when = matching_query(when.method(GET).path(path), query).query_param_count(
                    ".+",
                    ".*",
                    query.len() + usize::from(cursor.is_some()),
                );
                match cursor {
                    Some(cursor) => when.query_param("cursor", cursor),
                    None => when.query_param_missing("cursor"),
                };
                then.status(200).json_body(page);
            });
        });
}

#[rstest::rstest]
#[case::rows_and_unreadable(
    "plans_list",
    "/api/v1/plans",
    json!({ "all": true }),
    &[],
    json!({ "items": [{ "plan_id": "plan-1" }], "next_cursor": "c1", "unreadable": 1 }),
    json!({ "items": [{ "plan_id": "plan-2" }], "next_cursor": null, "unreadable": 0 }),
    json!({
        "items": [{ "plan_id": "plan-1" }, { "plan_id": "plan-2" }],
        "next_cursor": null,
        "unreadable": 1,
    }),
)]
// Every page is asked for `names`, not just the first.
#[case::names(
    "campaigns_list",
    "/api/v1/campaigns",
    json!({ "all": true, "names": true }),
    &[("names", "true")],
    json!({
        "items": [{ "campaign_id": "campaign_1" }],
        "next_cursor": "c1",
        "unreadable": 0,
        "names": { "user_1": "a@b.test" },
    }),
    json!({
        "items": [{ "campaign_id": "campaign_2" }],
        "next_cursor": null,
        "unreadable": 0,
        "names": { "org_1": "acme", "user_1": "a@b.test" },
    }),
    json!({
        "items": [{ "campaign_id": "campaign_1" }, { "campaign_id": "campaign_2" }],
        "next_cursor": null,
        "unreadable": 0,
        "names": { "org_1": "acme", "user_1": "a@b.test" },
    }),
)]
// mgmt reads its whole listing for each newest-first page, so the walk goes
// oldest-first and turns the rows around.
#[case::newest_first(
    "vault_list",
    "/api/v1/vault/resources",
    json!({ "all": true, "order": "newest" }),
    &[],
    json!({ "items": [{ "resource_id": "vault_1" }], "next_cursor": "c1", "unreadable": 1 }),
    json!({ "items": [{ "resource_id": "vault_2" }], "next_cursor": null, "unreadable": 1 }),
    json!({
        "items": [{ "resource_id": "vault_2" }, { "resource_id": "vault_1" }],
        "next_cursor": null,
        "unreadable": 2,
    }),
)]
fn all_follows_the_cursor_and_merges_the_pages(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] call: Value,
    #[case] query: &[(&str, &str)],
    #[case] first: Value,
    #[case] second: Value,
    #[case] merged: Value,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    two_pages(&server, path, query, first, second);
    let answer = session(&server, true)?.invoke(route(tool)?, &args(call)?)?;
    assert_eq!(answer.value, merged);
    Ok(())
}

/// A later page that is not one is refused, not merged as an empty page.
#[rstest::rstest]
#[case::no_items(json!({ "plans": [], "next_cursor": null, "unreadable": 0 }))]
#[case::rows_not_a_list(json!({ "items": {}, "next_cursor": null, "unreadable": 0 }))]
#[case::not_an_object(json!([]))]
fn all_refuses_an_answer_that_is_not_a_page(#[case] second: Value) -> anyhow::Result<()> {
    let server = MockServer::start();
    two_pages(
        &server,
        "/api/v1/plans",
        &[],
        json!({ "items": [{ "plan_id": "plan-1" }], "next_cursor": "c1", "unreadable": 0 }),
        second,
    );
    let error = session(&server, true)?
        .invoke(route("plans_list")?, &args(json!({ "all": true }))?)
        .err()
        .context("a body that is not a page should be refused")?;
    assert!(
        error
            .to_string()
            .contains("/api/v1/plans answered a body that is not a page"),
        "{error:#}"
    );
    Ok(())
}

/// Each whole-page listing names its own rows key, so a wrong one turns the
/// page into a record that `fields` refuses.
#[rstest::rstest]
#[case::orgs_list(
    "orgs_list",
    "/api/v1/orgs",
    json!({}),
    json!({ "org_id": "org_1", "slug": "acme", "member_count": 4 }),
    "slug",
)]
#[case::users_list(
    "users_list",
    "/api/v1/users",
    json!({}),
    json!({ "user_id": "user_1", "contact_email": "a@acme.test", "status": "active" }),
    "user_id",
)]
#[case::orgs_members(
    "orgs_members",
    "/api/v1/orgs/acme/users",
    json!({ "org_id": "acme" }),
    json!({ "user_id": "user_1", "contact_email": "a@acme.test", "status": "active" }),
    "contact_email",
)]
#[case::orgs_clients(
    "orgs_clients",
    "/api/v1/orgs/acme/clients",
    json!({ "org_id": "acme" }),
    json!({ "client_id": "ev1_a", "status": "approved" }),
    "client_id",
)]
#[case::model_params_list(
    "model_params_list",
    "/api/v1/model-params",
    json!({}),
    json!({ "key": "lfm3-1.7b", "total_millions": 1700, "active_millions": 1700 }),
    "key",
)]
fn fields_keep_only_the_named_keys_of_each_row(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] path_args: Value,
    #[case] row: Value,
    #[case] kept: &str,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path(path);
        then.status(200).json_body(json!({
            "items": [row.clone()],
            "next_cursor": null,
            "unreadable": 0,
        }));
    });
    let mut call = args(path_args)?;
    call.insert("fields".to_string(), json!([kept]));
    let listing = session(&server, true)?.invoke(route(tool)?, &call)?;
    assert_eq!(
        listing.value,
        json!({ "items": [{ kept: row[kept] }], "next_cursor": null, "unreadable": 0 })
    );
    Ok(())
}

/// A key a record lacks comes back null with a note naming the keys there
/// are; a record holding none of them is refused.
#[test]
fn fields_keep_only_the_named_keys_of_a_record() -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/benchmarks/eval_x");
        then.status(200).json_body(json!({
            "benchmark_id": "eval_x", "benchmark_type": "eval", "samples": [1, 2, 3],
        }));
    });
    let session = session(&server, true)?;
    let record = session.invoke(
        route("benchmarks_show")?,
        &args(json!({ "benchmark_id": "eval_x", "fields": ["benchmark_id", "benchmark_type"] }))?,
    )?;
    assert_eq!(
        record.value,
        json!({ "benchmark_id": "eval_x", "benchmark_type": "eval" })
    );
    assert!(record.notes.is_empty(), "{:?}", record.notes);

    let partial = session.invoke(
        route("benchmarks_show")?,
        &args(json!({ "benchmark_id": "eval_x", "fields": ["benchmark_id", "nope"] }))?,
    )?;
    assert_eq!(
        partial.value,
        json!({ "benchmark_id": "eval_x", "nope": null })
    );
    let note = partial.notes.join("\n");
    assert!(
        note.contains("no record here has nope") && note.contains("samples"),
        "{note}"
    );

    let refused = session
        .invoke(
            route("benchmarks_show")?,
            &args(json!({ "benchmark_id": "eval_x", "fields": ["nope"] }))?,
        )
        .err()
        .context("no such key")?;
    assert!(refused.to_string().contains("benchmark_type"), "{refused}");
    Ok(())
}

/// The server leaves out a key with no value, so a page whose rows lack a
/// named key answers it as null, with a note, rather than refuse a key a
/// later page holds.
#[test]
fn fields_on_a_listing_answer_a_missing_key_as_null() -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/api/v1/clients");
        then.status(200).json_body(json!({
            "items": [
                { "client_id": "ev1_a", "status": "pending" },
                { "client_id": "ev1_b", "status": "approved", "owner": { "type": "org", "org_id": "org_1" } },
            ],
            "next_cursor": "c1",
            "unreadable": 0,
        }));
    });
    let session = session(&server, true)?;
    let answer = session.invoke(
        route("clients_list")?,
        &args(json!({ "fields": ["client_id", "owner"] }))?,
    )?;
    assert_eq!(
        answer.value["items"],
        json!([
            { "client_id": "ev1_a", "owner": null },
            { "client_id": "ev1_b", "owner": { "type": "org", "org_id": "org_1" } },
        ])
    );
    assert!(answer.notes.is_empty(), "{:?}", answer.notes);

    let answer = session.invoke(
        route("clients_list")?,
        &args(json!({ "fields": ["ownr"] }))?,
    )?;
    assert_eq!(
        answer.value["items"],
        json!([{ "ownr": null }, { "ownr": null }])
    );
    let note = answer.notes.join("\n");
    assert!(
        note.contains("no record here has ownr") && note.contains("status"),
        "{note}"
    );
    Ok(())
}

#[test]
fn only_the_public_catalog_answers_without_an_identity() -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET)
            .path("/benchmarks")
            .header_missing("X-Signature")
            .header_missing("Signature");
        then.status(200).json_body(json!([]));
    });
    let session = session(&server, false)?;
    session.invoke(route("benchmarks_list")?, &Map::new())?;
    mock.assert();

    let refused = session
        .invoke(route("orgs_list")?, &Map::new())
        .err()
        .context("orgs_list needs an identity")?;
    assert!(
        refused.to_string().contains("no identity here"),
        "{refused}"
    );
    Ok(())
}

/// A stored account that does not resolve keeps its server and its reason:
/// neither a fallback to production nor advice to register again.
#[test]
fn an_unusable_identity_keeps_its_server_and_its_reason() -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET).path("/benchmarks");
        then.status(200).json_body(json!([]));
    });
    let work = tempfile::tempdir()?;
    let dir = identity::identity_dir(work.path());
    fs::create_dir_all(&dir)?;
    fs::write(
        dir.join("user.json"),
        json!({
            "user_id": "user_0192f0c1",
            "user_key_id": "userkey_9f2c",
            "server_url": server.base_url(),
        })
        .to_string(),
    )?;
    let session = Session::open(work.path(), None)?;
    session.invoke(route("benchmarks_list")?, &Map::new())?;
    mock.assert();
    let refused = session
        .invoke(route("orgs_list")?, &Map::new())
        .err()
        .context("no key to sign with")?;
    let message = format!("{refused:#}");
    // Names the missing key, not a --user-id flag to pass.
    assert!(message.contains("no key to sign with"), "{message}");
    assert!(!message.contains("--user-id"), "{message}");

    let empty = tempfile::tempdir()?;
    let refused = Session::open(empty.path(), Some(ServerUrl::try_new(server.base_url())?))?
        .invoke(route("orgs_list")?, &Map::new())
        .err()
        .context("an empty working directory has no account")?;
    assert!(refused.to_string().contains("auth register"), "{refused}");
    Ok(())
}

/// The paths a session reports are read by a model that does not know this
/// process's working directory.
#[test]
fn a_relative_work_dir_is_named_absolutely() -> anyhow::Result<()> {
    let session = Session::open(
        Path::new("no-such-work-dir"),
        Some(ServerUrl::try_new("http://127.0.0.1:9")?),
    )?;
    assert!(session.work_dir().is_absolute(), "{:?}", session.work_dir());
    Ok(())
}

/// A plan document goes out byte for byte as TOML, its options in the query
/// string the signature covers.
#[test]
fn a_plan_document_is_sent_as_toml_with_its_options_in_the_query() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let plan = dir.path().join("plan.toml");
    let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
    fs::write(&plan, document)?;
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/campaigns")
            .query_param("name", "nightly")
            .query_param("owner", "org_acme")
            .query_param("dry_run", "true")
            .header("content-type", "application/toml")
            .body(document);
        then.status(200).json_body(plan_dry_run());
    });
    let session = session(&server, true)?;
    session.invoke(
        route("campaigns_create")?,
        &args(json!({ "name": "nightly", "owner": "org_acme", "dry_run": true, "plan": plan }))?,
    )?;
    // The same document given inline, for a client that cannot write a file.
    session.invoke(
        route("campaigns_create")?,
        &args(
            json!({ "name": "nightly", "owner": "org_acme", "dry_run": true, "plan": document }),
        )?,
    )?;
    mock.assert_calls(2);

    let refused = session
        .invoke(
            route("campaigns_create")?,
            &args(json!({ "name": "nightly", "plan": "missing.toml" }))?,
        )
        .err()
        .context("neither a file nor TOML")?;
    let message = refused.to_string();
    assert!(
        message.contains("no readable file") && message.contains("not TOML"),
        "{message}"
    );
    Ok(())
}

/// A re-run's or a recalculation's base plan and `dry_run` ride the query;
/// the jobs, or the whole-plan selector, stay in the body.
#[rstest::rstest]
#[case::rerun("rerun", "answered", json!("failed"))]
#[case::recalculate("recalculate", "all_evals", json!(true))]
fn a_selection_sends_its_jobs_in_the_body(
    #[case] verb: &str,
    #[case] selector: &str,
    #[case] whole: Value,
) -> anyhow::Result<()> {
    let path = format!("/api/v1/campaigns/campaign_1/{verb}");
    let server = MockServer::start();
    let some = server.mock(|when, then| {
        when.method(POST)
            .path(&path)
            .query_param("dry_run", "true")
            .query_param_missing("base_plan_id")
            .json_body(json!({ "job_ids": ["job-1", "job-2"] }));
        then.status(200).json_body(revision_dry_run());
    });
    let all = server.mock(|when, then| {
        when.method(POST)
            .path(&path)
            .query_param("base_plan_id", "plan-1")
            .query_param_missing("dry_run")
            .json_body(json!({ selector: whole, "note": "again" }));
        then.status(201).json_body(campaign_revised(
            "plan-1",
            json!({ "type": verb, selector: whole, "job_count": 2 }),
        ));
    });
    let session = session(&server, true)?;
    let route = route(&format!("campaigns_{verb}"))?;
    session.invoke(
        route,
        &args(
            json!({ "campaign_id": "campaign_1", "job_ids": ["job-1", "job-2"], "dry_run": true }),
        )?,
    )?;
    session.invoke(
        route,
        &args(json!({
            "campaign_id": "campaign_1",
            "base_plan_id": "plan-1",
            selector: whole,
            "note": "again",
        }))?,
    )?;
    some.assert();
    all.assert();
    Ok(())
}

/// A revision replaces the document with a `PUT` of the TOML itself; the plan
/// it was read off and its options ride the signed query string.
#[test]
fn a_revision_puts_the_document_with_its_base_in_the_query() -> anyhow::Result<()> {
    let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
    let server = MockServer::start();
    let read = server.mock(|when, then| {
        when.method(GET)
            .path("/api/v1/campaigns/campaign_1/document");
        then.status(200).json_body(json!({
            "base": { "plan_id": "plan-1", "note": "nightly" },
            "document": document,
        }));
    });
    let revise = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/campaigns/campaign_1/document")
            .query_param("base_plan_id", "plan-1")
            .query_param("note", "nightly v2")
            .query_param_missing("dry_run")
            .header("content-type", "application/toml")
            .body(document);
        then.status(200)
            .json_body(json!({ "campaign_id": "campaign_1", "changed": false }));
    });
    let session = session(&server, true)?;
    let base = session.invoke(
        route("campaigns_document")?,
        &args(json!({ "campaign_id": "campaign_1" }))?,
    )?;
    read.assert();
    let answer = session.invoke(
        route("campaigns_revise")?,
        &args(json!({
            "campaign_id": "campaign_1",
            "base_plan_id": base.value["base"]["plan_id"],
            "note": "nightly v2",
            "plan": base.value["document"],
        }))?,
    )?;
    revise.assert();
    assert_eq!(answer.value["changed"], false);

    // Without the plan it was read off, the server could not refuse a stale edit.
    let refused = session
        .invoke(
            route("campaigns_revise")?,
            &args(json!({ "campaign_id": "campaign_1", "plan": document }))?,
        )
        .err()
        .context("revised with no base_plan_id")?;
    assert!(
        refused.to_string().contains("`base_plan_id` is required"),
        "{refused}"
    );
    Ok(())
}

/// What a filing verb answers when it files, and on a dry run.
struct Answers {
    filed: Value,
    report: Value,
}

/// How a progress stream ends, and what the invoker makes of it.
#[derive(Debug, Clone, Copy)]
enum Ends {
    Answer,
    DryRun,
    Refused,
    Broken,
}

/// A stream with no last line tells the user to read the campaign before
/// filing the plan again.
#[rstest::rstest]
#[case::create(
    "campaigns_create",
    POST,
    "/api/v1/campaigns",
    "plan",
    json!({ "name": "nightly" }),
    Answers { filed: campaign_created(), report: plan_dry_run() }
)]
#[case::file(
    "campaigns_file",
    POST,
    "/api/v1/campaigns/campaign_1/plans",
    "plan",
    json!({ "campaign_id": "campaign_1" }),
    Answers { filed: plan_filed(), report: plan_dry_run() }
)]
#[case::revise(
    "campaigns_revise",
    PUT,
    "/api/v1/campaigns/campaign_1/document",
    "plan",
    json!({ "campaign_id": "campaign_1", "base_plan_id": "plan_0" }),
    Answers {
        filed: campaign_revised("plan_0", json!({ "type": "document" })),
        report: revision_dry_run(),
    }
)]
#[case::variants(
    "campaigns_variants",
    POST,
    "/api/v1/campaigns/campaign_1/variants",
    "edit",
    json!({ "campaign_id": "campaign_1", "base_plan_id": "plan_0" }),
    Answers {
        filed: campaign_revised(
            "plan_0",
            json!({ "type": "variants", "removed_variant_count": 0, "added_variant_count": 1 }),
        ),
        report: revision_dry_run(),
    }
)]
fn the_filing_verbs_read_the_progress_stream(
    #[case] tool: &str,
    #[case] method: httpmock::Method,
    #[case] path: &str,
    #[case] field: &str,
    #[case] mut call: Value,
    #[case] Answers { filed, report }: Answers,
    #[values(Ends::Answer, Ends::DryRun, Ends::Refused, Ends::Broken)] ends: Ends,
) -> anyhow::Result<()> {
    let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
    call[field] = json!(document);
    call["note"] = json!("why");
    let progress = r#"{"event":"progress","hf_ref":"org/m@abc","file":"m.gguf","bytes_done":5,"bytes_total":10}"#;
    let result = json!({ "event": "result", "status": 201, "body": filed });
    let (dry_run, body) = match ends {
        Ends::Answer => (
            false,
            format!("{progress}\n{{\"event\":\"keepalive\"}}\n{result}\n"),
        ),
        Ends::DryRun => (
            true,
            format!(
                "{}\n",
                json!({ "event": "result", "status": 200, "body": report })
            ),
        ),
        Ends::Refused => (
            false,
            format!(
                "{progress}\n{{\"event\":\"error\",\"status\":400,\"error\":\"gated model refused\"}}\n"
            ),
        ),
        Ends::Broken => (false, format!("{progress}\n")),
    };
    if dry_run {
        call["dry_run"] = json!(true);
    }
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        let when = when
            .method(method)
            .path(path)
            .header("accept", "application/x-ndjson")
            .header("content-type", "application/toml")
            .query_param("note", "why")
            .body(document);
        if dry_run {
            when.query_param("dry_run", "true");
        } else {
            when.query_param_missing("dry_run");
        }
        then.status(200)
            .header("content-type", "application/x-ndjson")
            .body(body);
    });
    let mut seen = Vec::new();
    let answer =
        session(&server, true)?.invoke_reporting(route(tool)?, &args(call)?, &mut |event| {
            seen.push(event.clone())
        });
    mock.assert();
    match (ends, answer) {
        (Ends::Answer, Ok(answer)) => {
            assert_eq!(answer.value, filed);
            assert_eq!(seen.len(), 1, "{seen:?}");
        }
        (Ends::DryRun, Ok(answer)) => {
            assert_eq!(answer.value, report);
            assert!(seen.is_empty(), "{seen:?}");
        }
        (Ends::Refused, Err(error)) => {
            let message = format!("{error:#}");
            assert!(
                message.contains("400") && message.contains("gated model refused"),
                "{message}"
            );
        }
        (Ends::Broken, Err(error)) => {
            let message = format!("{error:#}");
            assert!(message.contains("Read the campaign"), "{message}");
        }
        (ends, answer) => anyhow::bail!("expected {ends:?}, got {answer:?}"),
    }
    Ok(())
}

/// A definition read back carries the ids the path holds; the server refuses
/// them in the body, so they are dropped. A JSON string, or a file holding
/// JSON, is read as JSON.
#[test]
fn a_benchmark_definition_round_trips() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("b1.json");
    let server = MockServer::start();
    let mock =
        server.mock(|when, then| {
            when.method(PUT).path("/api/v1/orgs/acme/benchmarks/b1").json_body(
            json!({ "benchmark_type": "prefill_throughput", "parameter_prefill_tokens": 256 }),
        );
            then.status(200).json_body(json!({ "benchmark_id": "b1" }));
        });
    let session = session(&server, true)?;
    let read_back = json!({
        "benchmark_id": "b1",
        "org_id": "org_1",
        "benchmark_type": "prefill_throughput",
        "parameter_prefill_tokens": 256,
    });
    let put = |definition: Value| {
        session.invoke(
            route("orgs_benchmark_put")?,
            &args(json!({ "org_id": "acme", "benchmark_id": "b1", "definition": definition }))?,
        )
    };
    fs::write(&file, read_back.to_string())?;
    [
        read_back.clone(),
        Value::String(read_back.to_string()),
        json!(file),
    ]
    .into_iter()
    .try_for_each(|definition| put(definition).map(drop))?;
    mock.assert_calls(3);

    // A file holding JSON that is not an object, or broken JSON, is refused
    // unsent, with the JSON error rather than a TOML one.
    [
        ("[1]", "must be a JSON object"),
        (r#"{"a": 1,}"#, "is not JSON"),
    ]
    .into_iter()
    .try_for_each(|(contents, reason)| {
        fs::write(&file, contents)?;
        let refused = put(json!(file)).err().context("sent a bad file")?;
        anyhow::ensure!(refused.to_string().contains(reason), "{refused}");
        Ok(())
    })?;
    mock.assert_calls(3);
    Ok(())
}

/// A catalog file's TOML, inline or by path, goes out as its own bytes.
#[test]
fn a_toml_benchmark_definition_is_sent_as_toml() -> anyhow::Result<()> {
    let document = "benchmark_type = \"prefill_throughput\"\nparameter_prefill_tokens = 256\n";
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("b1.toml");
    fs::write(&file, document)?;
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(PUT)
            .path("/api/v1/orgs/acme/benchmarks/b1")
            .header("content-type", "application/toml")
            .body(document);
        then.status(200).json_body(json!({ "benchmark_id": "b1" }));
    });
    let session = session(&server, true)?;
    let put = |definition: Value| {
        session.invoke(
            route("orgs_benchmark_put")?,
            &args(json!({ "org_id": "acme", "benchmark_id": "b1", "definition": definition }))?,
        )
    };
    [json!(document), json!(file)]
        .into_iter()
        .try_for_each(|definition| put(definition).map(drop))?;
    mock.assert_calls(2);
    Ok(())
}

/// `fields` narrows the plans that asked for the job, not the job's body.
#[test]
fn plans_by_job_lists_the_plans_and_keeps_the_body() -> anyhow::Result<()> {
    let body = json!({ "job_id": "job-1", "spec": { "benchmark": "decode_throughput_512_100" } });
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET).path("/api/v1/jobs/job-1/plans");
        then.status(200).json_body(json!({
            "plans": [{ "plan_id": "plan-1", "note": "nightly", "created_at": "2026-09-28T09:00:00Z" }],
            "body": body,
        }));
    });
    let answer = session(&server, true)?.invoke(
        route("plans_by_job")?,
        &args(json!({ "job_id": "job-1", "fields": ["plan_id"] }))?,
    )?;
    mock.assert();
    assert_eq!(
        answer.value,
        json!({ "plans": [{ "plan_id": "plan-1" }], "body": body })
    );
    Ok(())
}

/// The call less its path parameters: what a write sends as its body, and a
/// read as its query.
fn call_less_path(route: &Route, call: &Value) -> anyhow::Result<Map<String, Value>> {
    Ok(call
        .as_object()
        .context("the call is an object")?
        .iter()
        .filter(|(name, _)| !route.path_params().any(|param| param == name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect())
}

#[rstest::rstest]
#[case::user_typed_at_creation(
    "users_create",
    POST,
    "/api/v1/users",
    json!({ "contact_email": "job@b.test", "org_id": "org_acme", "account_type": "service" }),
    201,
    json!({ "user_id": "user_1" }),
)]
#[case::user_retyped(
    "users_set_type",
    PATCH,
    "/api/v1/users/user_1",
    json!({ "user_id": "user_1", "account_type": "human" }),
    200,
    json!({ "user_id": "user_1", "changed": true }),
)]
#[case::labels_alone(
    "campaigns_rename",
    PATCH,
    "/api/v1/campaigns/campaign_1",
    json!({ "campaign_id": "campaign_1", "labels": { "study": "lfm3", "team": "Edge ML" } }),
    200,
    json!({ "campaign_id": "campaign_1", "changed": true }),
)]
// `{}` clears the labels.
#[case::labels_cleared_with_a_name(
    "campaigns_rename",
    PATCH,
    "/api/v1/campaigns/campaign_1",
    json!({ "campaign_id": "campaign_1", "name": "nightly", "labels": {} }),
    200,
    json!({ "campaign_id": "campaign_1", "changed": true }),
)]
// A blank note is sent as it is: the server reads it as clearing the note.
#[case::blank_note(
    "plans_update",
    PATCH,
    "/api/v1/plans/plan-1",
    json!({ "plan_id": "plan-1", "note": "" }),
    200,
    json!({ "plan_id": "plan-1", "changed": true }),
)]
#[case::user_preauth_minted(
    "users_preauth_create",
    POST,
    "/api/v1/users/user_1/preauth-keys",
    json!({ "user_id": "user_1", "expires_in_days": 7, "note": "for Ana" }),
    201,
    json!({
        "key_id": "0123456789abcdef",
        "user_id": "user_1",
        "token": "user_preauth_0123456789abcdef.secret",
    }),
)]
#[case::device_preauth_minted(
    "orgs_client_preauth_create",
    POST,
    "/api/v1/orgs/acme/client-preauth-keys",
    json!({ "org_id": "acme", "expires_in_days": 30, "usage": "multi_use", "note": "rack 2" }),
    201,
    json!({ "key_id": "fedcba9876543210", "token": "client_preauth_fedcba9876543210.y" }),
)]
// A mint sends only what was given.
#[case::device_preauth_bare(
    "orgs_client_preauth_create",
    POST,
    "/api/v1/orgs/acme/client-preauth-keys",
    json!({ "org_id": "acme" }),
    201,
    json!({ "key_id": "0123456789abcdef", "token": "client_preauth_0123456789abcdef.z" }),
)]
#[case::permissions_replaced(
    "clients_set_permissions",
    PUT,
    "/api/v1/clients/ev1_a/permissions",
    json!({ "client_id": "ev1_a", "permissions": ["submit_results", "perform_work"] }),
    200,
    json!({
        "client_id": "ev1_a",
        "permissions": ["perform_work", "submit_results"],
        "changed": true,
    }),
)]
// `[]` clears the set, so it is sent.
#[case::permissions_cleared(
    "clients_set_permissions",
    PUT,
    "/api/v1/clients/ev1_a/permissions",
    json!({ "client_id": "ev1_a", "permissions": [] }),
    200,
    json!({ "client_id": "ev1_a", "permissions": [], "changed": true }),
)]
#[case::permissions_amended(
    "clients_amend_permissions",
    PATCH,
    "/api/v1/clients/ev1_a/permissions",
    json!({ "client_id": "ev1_a", "add": ["perform_work"], "remove": ["submit_results"] }),
    200,
    json!({ "client_id": "ev1_a", "permissions": ["perform_work"], "changed": true }),
)]
#[case::model_params_created(
    "model_params_create",
    POST,
    "/api/v1/model-params",
    json!({ "model": "LiquidAI/LFM3-1.7B-GGUF", "total_millions": 1700 }),
    201,
    json!({
        "key": "lfm3-1.7b",
        "total_millions": 1700,
        "active_millions": 1700,
        "note": null,
        "updated_at": "2026-01-02T03:04:05Z",
        "updated_by": "user_0192f0c1",
    }),
)]
#[case::model_params_replaced(
    "model_params_update",
    PUT,
    "/api/v1/model-params/lfm3-1.7b",
    json!({
        "key": "lfm3-1.7b",
        "total_millions": 8300,
        "active_millions": 1500,
        "note": "model card",
    }),
    200,
    json!({ "key": "lfm3-1.7b", "changed": true }),
)]
fn a_write_sends_its_arguments_as_the_body(
    #[case] tool: &str,
    #[case] method: Method,
    #[case] path: &str,
    #[case] call: Value,
    #[case] status: u16,
    #[case] answer: Value,
) -> anyhow::Result<()> {
    let route = route(tool)?;
    let body = call_less_path(route, &call)?;
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(method)
            .path(path)
            .json_body(Value::Object(body));
        then.status(status).json_body(answer);
    });
    session(&server, true)?.invoke(route, &args(call)?)?;
    mock.assert();
    Ok(())
}

#[rstest::rstest]
#[case::clients_unowned("clients_list", "/api/v1/clients", json!({ "owned": false }))]
#[case::clients_mine("clients_list", "/api/v1/clients", json!({ "owner": "mine" }))]
#[case::clients_of_an_org("clients_list", "/api/v1/clients", json!({ "org_id": "org_acme" }))]
#[case::orgs_without_counts("orgs_list", "/api/v1/orgs", json!({ "counts": false }))]
#[case::campaigns_list(
    "campaigns_list",
    "/api/v1/campaigns",
    json!({ "plan_id": "plan-1", "order": "newest", "names": true, "plan": true }),
)]
#[case::campaigns_show(
    "campaigns_show",
    "/api/v1/campaigns/campaign_1",
    json!({ "campaign_id": "campaign_1", "names": true, "plan": true }),
)]
#[case::campaigns_plan(
    "campaigns_plan",
    "/api/v1/campaigns/campaign_1/plan",
    json!({ "campaign_id": "campaign_1", "names": true }),
)]
#[case::vault_list(
    "vault_list",
    "/api/v1/vault/resources",
    json!({ "order": "newest", "names": true, "object": true }),
)]
#[case::vault_show(
    "vault_show",
    "/api/v1/vault/resources/vault_1",
    json!({ "resource_id": "vault_1", "names": true, "object": true }),
)]
#[case::plans_list(
    "plans_list",
    "/api/v1/plans",
    json!({ "order": "newest", "names": true }),
)]
#[case::plans_status(
    "plans_status",
    "/api/v1/plans/plan-1",
    json!({ "plan_id": "plan-1", "names": true }),
)]
#[case::plans_jobs(
    "plans_jobs",
    "/api/v1/plans/plan-1/jobs",
    json!({ "plan_id": "plan-1", "results": true, "all": true }),
)]
#[case::users_keys("users_keys", "/api/v1/users/user_1/keys", json!({ "user_id": "user_1" }))]
#[case::users_preauth_keys(
    "users_preauth_keys",
    "/api/v1/users/user_1/preauth-keys",
    json!({ "user_id": "user_1" }),
)]
#[case::orgs_client_preauth_keys(
    "orgs_client_preauth_keys",
    "/api/v1/orgs/acme/client-preauth-keys",
    json!({ "org_id": "acme" }),
)]
#[case::campaigns_list_variants(
    "campaigns_list_variants",
    "/api/v1/campaigns/campaign_1/variants",
    json!({ "campaign_id": "campaign_1" }),
)]
fn a_read_sends_its_filters_and_additions_as_the_query(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] call: Value,
) -> anyhow::Result<()> {
    let route = route(tool)?;
    let query: Vec<(String, String)> = call_less_path(route, &call)?
        .iter()
        .map(|(name, value)| (name.clone(), scalar_text(value)))
        .collect();
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        // Exactly these pairs, so a filter the call leaves out is never sent.
        matching_query(when.method(GET).path(path), &query).query_param_count(
            ".+",
            ".*",
            query.len(),
        );
        then.status(200)
            .json_body(json!({ "items": [], "next_cursor": null, "unreadable": 0 }));
    });
    session(&server, true)?.invoke(route, &args(call)?)?;
    mock.assert();
    Ok(())
}

const DOCUMENT: &str = "benchmarks = [\"decode_throughput_512_100\"]\n";

/// mgmt refuses `inherit_from` or `add_from` repeated.
#[rstest::rstest]
#[case::revise(
    "campaigns_revise",
    PUT,
    "/api/v1/campaigns/campaign_1/document",
    json!({
        "campaign_id": "campaign_1",
        "base_plan_id": "plan-0",
        "plan": DOCUMENT,
        "dry_run": true,
        "carry_earlier": true,
        "inherit_from": ["plan-1", "plan-2"],
    }),
    &[("carry_earlier", "true"), ("inherit_from", "plan-1,plan-2")],
    "add_from",
    DOCUMENT,
)]
// `add_from` alone fills the edit, so none is sent.
#[case::variants(
    "campaigns_variants",
    POST,
    "/api/v1/campaigns/campaign_1/variants",
    json!({
        "campaign_id": "campaign_1",
        "base_plan_id": "plan-0",
        "dry_run": true,
        "add_from": ["plan-3", "plan-4"],
    }),
    &[("add_from", "plan-3,plan-4")],
    "inherit_from",
    "",
)]
fn a_revision_sends_the_plans_it_takes_runs_from_as_one_value(
    #[case] tool: &str,
    #[case] method: Method,
    #[case] path: &str,
    #[case] call: Value,
    #[case] query: &[(&str, &str)],
    #[case] absent: &str,
    #[case] body: &str,
) -> anyhow::Result<()> {
    let stream = format!(
        "{}\n",
        json!({ "event": "result", "status": 200, "body": revision_dry_run() })
    );
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        query
            .iter()
            .fold(when.method(method).path(path), |when, (name, value)| {
                when.query_param(*name, *value)
                    .query_param_count(format!("^{name}$"), ".*", 1)
            })
            .query_param_missing(absent)
            .header("content-type", "application/toml")
            .body(body);
        then.status(200)
            .header("content-type", "application/x-ndjson")
            .body(&stream);
    });
    session(&server, true)?.invoke(route(tool)?, &args(call)?)?;
    mock.assert();
    Ok(())
}

#[rstest::rstest]
#[case::plan_document(
    "plans_document",
    "/api/v1/plans/plan-1/document",
    "application/toml; charset=utf-8",
    "benchmarks = []\n"
)]
#[case::plan_variants(
    "plans_variants",
    "/api/v1/plans/plan-1/variants",
    "application/toml; charset=utf-8",
    "[[variants]]\nbenchmarks = [\"decode_throughput_512_100\"]\n"
)]
#[case::plan_results(
    "plans_results",
    "/api/v1/plans/plan-1/results.csv",
    "text/csv; charset=utf-8",
    "job_id,benchmark_id,metric,value\njob-1,decode_throughput_512_100,tps,41.5\n"
)]
fn a_text_answer_comes_back_as_its_text(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] content_type: &str,
    #[case] text: &str,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path(path);
        then.status(200)
            .header("content-type", content_type)
            .body(text);
    });
    let answer =
        session(&server, true)?.invoke(route(tool)?, &args(json!({ "plan_id": "plan-1" }))?)?;
    assert_eq!(answer.value, Value::String(text.into()));
    Ok(())
}

#[rstest::rstest]
#[case::user_key(
    "users_key_revoke",
    "/api/v1/users/user_1/keys/userkey_2",
    json!({ "user_id": "user_1", "key_id": "userkey_2" }),
)]
// A pre-auth key goes by its own id, not a key registration's.
#[case::user_preauth_key(
    "users_preauth_revoke",
    "/api/v1/users/user_1/preauth-keys/0123456789abcdef",
    json!({ "user_id": "user_1", "preauth_key_id": "0123456789abcdef" }),
)]
#[case::device_preauth_key(
    "orgs_client_preauth_revoke",
    "/api/v1/orgs/acme/client-preauth-keys/fedcba9876543210",
    json!({ "org_id": "acme", "preauth_key_id": "fedcba9876543210" }),
)]
#[case::model_params(
    "model_params_delete",
    "/api/v1/model-params/lfm3-1.7b",
    json!({ "key": "lfm3-1.7b" }),
)]
fn a_delete_sends_nothing_and_answers_null(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] call: Value,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(DELETE).path(path).body("");
        then.status(204);
    });
    let answer = session(&server, true)?.invoke(route(tool)?, &args(call)?)?;
    mock.assert();
    assert_eq!(answer.value, Value::Null);
    Ok(())
}

#[rstest::rstest]
#[case::own_key("userkey_9f2c", 1)]
// mgmt trims the id, so this still revokes the session's key.
#[case::own_key_padded(" userkey_9f2c ", 1)]
#[case::another_key("userkey_7a1b", 0)]
fn revoking_a_key_by_its_account_notes_only_the_session_s(
    #[case] key_id: &str,
    #[case] lockouts: usize,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    let revoke = server.mock(|when, then| {
        when.method(DELETE)
            .path_prefix("/api/v1/users/user_0192f0c1/keys/");
        then.status(204);
    });
    let notes = session(&server, true)?
        .invoke(
            route("users_key_revoke")?,
            &args(json!({ "user_id": "user_0192f0c1", "key_id": key_id }))?,
        )?
        .notes;
    revoke.assert();
    assert_eq!(notes.len(), lockouts, "{notes:?}");
    assert!(
        notes.iter().all(|note| {
            note.contains("userkey_9f2c was this session's signing key") && note.contains("401")
        }),
        "{notes:?}"
    );
    Ok(())
}

/// mgmt refuses `outcome` or `actor` repeated, so each is one comma-separated value.
#[rstest::rstest]
#[case::every_event("audit_list", "/api/v1/audit", json!({}))]
#[case::an_organization_s("orgs_audit", "/api/v1/orgs/acme/audit", json!({ "org_id": "acme" }))]
fn an_audit_log_sends_each_filter_as_one_value(
    #[case] tool: &str,
    #[case] path: &str,
    #[case] mut call: Value,
) -> anyhow::Result<()> {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(GET)
            .path(path)
            .query_param("outcome", "refused,failed")
            .query_param_count("^outcome$", ".*", 1)
            .query_param("actor", "user")
            .query_param("target", "campaign_1")
            .query_param("names", "true")
            .query_param("limit", "20");
        then.status(200)
            .json_body(json!({ "items": [], "next_cursor": null, "unreadable": 0 }));
    });
    [
        ("outcome", json!(["refused", "failed"])),
        ("actor", json!(["user"])),
        ("target", json!("campaign_1")),
        ("names", json!(true)),
        ("limit", json!(20)),
    ]
    .into_iter()
    .for_each(|(name, value)| call[name] = value);
    session(&server, true)?.invoke(route(tool)?, &args(call)?)?;
    mock.assert();
    Ok(())
}

/// A check sends the document as filing does; a verdict with errors is still an answer.
#[test]
fn a_plan_document_is_checked_as_toml() -> anyhow::Result<()> {
    let document = "benchmarks = [\"decode_throughput_512_100\"]\n";
    let verdict = json!({
        "job_count": null,
        "gated_model_count": null,
        "issues": [{ "severity": "error", "message": "no variants", "line": null, "column": null }],
    });
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/plans/validate")
            .query_param_missing("dry_run")
            .header("content-type", "application/toml")
            .body(document);
        then.status(200).json_body(verdict.clone());
    });
    let answer = session(&server, true)?.invoke(
        route("plans_validate")?,
        &args(json!({ "plan": document }))?,
    )?;
    mock.assert();
    assert_eq!(answer.value, verdict);
    Ok(())
}

/// A vault file given as a path is uploaded for the owner only when the
/// server does not hold it yet, then named by its digest in the upload.
#[test]
fn a_vault_upload_pushes_missing_files_then_names_them() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let model = dir.path().join("m.gguf");
    let mmproj = dir.path().join("p.gguf");
    fs::write(&model, "GGUFmodel")?;
    fs::write(&mmproj, "GGUFproj")?;
    let model_sha = sha256_file(&model)?.hex();
    let mmproj_sha = sha256_file(&mmproj)?.hex();
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::HEAD)
            .path(format!("/api/v1/vault/blobs/org_acme/{model_sha}"));
        then.status(404);
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::HEAD)
            .path(format!("/api/v1/vault/blobs/org_acme/{mmproj_sha}"));
        then.status(200);
    });
    let put = server.mock(|when, then| {
        when.method(PUT)
            .path(format!("/api/v1/vault/blobs/org_acme/{model_sha}"))
            .body("GGUFmodel");
        then.status(201)
            .json_body(json!({ "sha256": model_sha, "bytes": 9 }));
    });
    let upload = server.mock(|when, then| {
        when.method(POST)
            .path("/api/v1/vault/resources")
            .json_body(json!({
                "name": "acme vl",
                "type": "gguf_vision",
                "owner": "org_acme",
                "model": { "filename": "m.gguf", "sha256": model_sha },
                "mmproj": { "filename": "p.gguf", "sha256": mmproj_sha },
            }));
        then.status(201).json_body(json!({ "toml": "{}" }));
    });
    let answer = session(&server, true)?.invoke(
        route("vault_upload")?,
        &args(json!({
            "name": "acme vl",
            "type": "gguf_vision",
            "owner": "org_acme",
            "model": model,
            "mmproj": mmproj,
        }))?,
    )?;
    put.assert_calls(1);
    upload.assert();
    let notes = answer.notes.join("\n");
    assert!(
        notes.contains("m.gguf uploaded") && notes.contains("p.gguf was already uploaded"),
        "{notes}"
    );
    Ok(())
}

/// A missing `name`, a bad `type` or a file name the server refuses costs
/// no upload.
#[test]
fn a_vault_upload_checks_its_arguments_before_sending_a_byte() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let good = dir.path().join("m.gguf");
    let spaced = dir.path().join("my model (1).gguf");
    fs::write(&good, "GGUFmodel")?;
    fs::write(&spaced, "GGUFmodel")?;
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.any_request();
        then.status(200);
    });
    let session = session(&server, true)?;
    for bad in [
        json!({ "type": "gguf_text", "model": good }),
        json!({ "name": "x", "type": "not_a_type", "model": good }),
        json!({ "name": "x", "type": "gguf_text", "model": spaced }),
        json!({ "name": "x", "type": "gguf_text", "owner": "acme", "model": good }),
        json!({ "name": "x", "type": "gguf_text", "model": { "filename": "m.gguf" } }),
    ] {
        let refused = session.invoke(route("vault_upload")?, &args(bad.clone())?);
        anyhow::ensure!(refused.is_err(), "accepted {bad}");
    }
    mock.assert_calls(0);
    Ok(())
}

#[test]
fn a_blob_put_derives_its_digest_from_the_file() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("w.bin");
    fs::write(&file, "weights")?;
    let digest = sha256_file(&file)?.hex();
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::HEAD)
            .path(format!("/api/v1/vault/blobs/user_1/{digest}"));
        then.status(404);
    });
    let mock = server.mock(|when, then| {
        when.method(PUT)
            .path(format!("/api/v1/vault/blobs/user_1/{digest}"))
            .body("weights");
        then.status(201)
            .json_body(json!({ "sha256": digest, "bytes": 7 }));
    });
    session(&server, true)?.invoke(
        route("vault_blob_put")?,
        &args(json!({ "principal_id": "user_1", "file": file }))?,
    )?;
    mock.assert();
    Ok(())
}

/// The server answers a blob it holds before reading the body, which a
/// streamed upload cannot hear; the `HEAD` finds out first.
#[test]
fn a_blob_already_there_is_not_sent_again() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("w.bin");
    fs::write(&file, "weights")?;
    let digest = sha256_file(&file)?.hex();
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(httpmock::Method::HEAD)
            .path(format!("/api/v1/vault/blobs/user_1/{digest}"));
        then.status(200);
    });
    let put = server.mock(|when, then| {
        when.method(PUT);
        then.status(200);
    });
    let answer = session(&server, true)?.invoke(
        route("vault_blob_put")?,
        &args(json!({ "principal_id": "user_1", "file": file, "sha256": digest.to_uppercase() }))?,
    )?;
    put.assert_calls(0);
    assert_eq!(answer.value, json!({ "sha256": digest, "bytes": 7 }));
    assert!(answer.notes.join("").contains("already uploaded"));
    Ok(())
}

/// The server keeps a blob already there under the digest in the path, so a
/// digest the file does not hash to would report another file as stored.
#[test]
fn a_blob_put_refuses_a_digest_the_file_does_not_hash_to() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("w.bin");
    fs::write(&file, "weights")?;
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.any_request();
        then.status(200);
    });
    let refused = session(&server, true)?
        .invoke(
            route("vault_blob_put")?,
            &args(json!({ "principal_id": "user_1", "file": file, "sha256": "b".repeat(64) }))?,
        )
        .err()
        .context("the digest is not the file's")?;
    assert!(refused.to_string().contains("hashes to"), "{refused}");
    mock.assert_calls(0);
    Ok(())
}

/// A `HEAD` refusal has no body, so the status is explained here.
#[test]
fn a_refused_head_says_what_the_status_means() -> anyhow::Result<()> {
    let server = MockServer::start();
    for (principal, status) in [("user_2", 403), ("user_3", 401)] {
        server.mock(|when, then| {
            when.method(httpmock::Method::HEAD)
                .path_includes(format!("/{principal}/"));
            then.status(status);
        });
    }
    let session = session(&server, true)?;
    for (principal, meaning) in [
        ("user_2", "outside this account's reach"),
        ("user_3", "refused this session's signature"),
    ] {
        let refused = session
            .invoke(
                route("vault_blob_exists")?,
                &args(json!({ "principal_id": principal, "sha256": "a".repeat(64) }))?,
            )
            .err()
            .with_context(|| format!("{principal} is refused"))?;
        let message = format!("{refused:#}");
        assert!(message.contains(meaning), "{message}");
    }
    Ok(())
}

#[test]
fn a_download_lands_whole_under_the_requested_name() -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/vault/resources/vault_1/content");
        then.status(200)
            .header("etag", format!("\"{}\"", sha256_hex(b"weights")))
            .body("weights");
    });
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("model.bin");
    // A `.part` beside `out` that is not its staging file is left alone.
    let sibling = dir.path().join("model.part");
    fs::write(&sibling, "mine")?;
    let answer = session(&server, true)?.invoke(
        route("vault_download")?,
        &args(json!({ "resource_id": "vault_1", "out": out }))?,
    )?;
    assert_eq!(answer.value.get("path"), Some(&json!(out)));
    assert_eq!(answer.value.get("bytes"), Some(&json!(7)));
    assert_eq!(
        answer.value.get("sha256"),
        Some(&json!(sha256_hex(b"weights")))
    );
    assert_eq!(fs::read_to_string(&out)?, "weights");
    assert_eq!(fs::read_to_string(&sibling)?, "mine");
    let mut names = fs::read_dir(dir.path())?
        .map(|entry| Ok(entry?.file_name()))
        .collect::<io::Result<Vec<_>>>()?;
    names.sort();
    assert_eq!(names, ["model.bin", "model.part"]);
    Ok(())
}

/// An existing file is replaced only on request, and a directory never.
#[test]
fn a_download_does_not_replace_a_file_unasked() -> anyhow::Result<()> {
    let server = MockServer::start();
    let content = server.mock(|when, then| {
        when.method(GET).path("/vault/resources/vault_1/content");
        then.status(200)
            .header("etag", format!("\"{}\"", sha256_hex(b"weights")))
            .body("weights");
    });
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("keep.txt");
    fs::write(&out, "precious")?;
    let session = session(&server, true)?;
    let download = |extra: Value| -> anyhow::Result<Answer> {
        let mut call = args(json!({ "resource_id": "vault_1" }))?;
        call.extend(args(extra)?);
        session.invoke(route("vault_download")?, &call)
    };
    anyhow::ensure!(
        download(json!({ "out": out })).is_err(),
        "replaced a file unasked"
    );
    anyhow::ensure!(
        download(json!({ "out": dir.path(), "overwrite": true })).is_err(),
        "wrote over a directory"
    );
    content.assert_calls(0);
    assert_eq!(fs::read_to_string(&out)?, "precious");
    download(json!({ "out": out, "overwrite": true }))?;
    assert_eq!(fs::read_to_string(&out)?, "weights");
    Ok(())
}

/// A download whose server stops sending fails once one read waits out the
/// timeout, rather than hanging, and leaves nothing behind. Without the
/// timeout it would finish when the rest arrives.
#[test]
fn a_stalled_download_fails_and_leaves_nothing() -> anyhow::Result<()> {
    let body = b"weights, then the rest";
    let etag = sha256_hex(body);
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let server = ServerUrl::try_new(format!("http://{}", listener.local_addr()?))?;
    thread::spawn(move || {
        let Ok((mut socket, _)) = listener.accept() else {
            return;
        };
        let mut request = Vec::new();
        let mut buf = [0; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            match socket.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => request.extend_from_slice(buf.get(..n).unwrap_or_default()),
            }
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\netag: \"{etag}\"\r\n\r\n",
            body.len()
        );
        let (first, rest) = body.split_at(7);
        let _ = socket.write_all(head.as_bytes());
        let _ = socket.write_all(first);
        thread::sleep(Duration::from_secs(5));
        let _ = socket.write_all(rest);
    });
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("model.bin");
    let session = Session {
        server,
        http: HttpClient::with_request_timeout("pipette-ctl", Duration::from_secs(1))?,
        bulk: crate::http_client()?,
        identity: Ok(user_identity()?),
        work_dir: dir.path().to_path_buf(),
    };
    let stalled = session.invoke(
        route("vault_download")?,
        &args(json!({ "resource_id": "vault_1", "out": out }))?,
    );
    anyhow::ensure!(stalled.is_err(), "waited out the stall: {stalled:?}");
    assert_eq!(fs::read_dir(dir.path())?.count(), 0);
    Ok(())
}

/// Bytes that do not hash to the record's digest, which mgmt sends as the
/// ETag, never land, and neither do bytes with no digest to check; neither
/// leaves anything behind.
#[test]
fn a_download_that_does_not_match_its_record_is_discarded() -> anyhow::Result<()> {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(GET).path("/vault/resources/vault_1/content");
        then.status(200)
            .header("etag", format!("\"{}\"", sha256_hex(b"other")))
            .body("weights");
    });
    server.mock(|when, then| {
        when.method(GET).path("/vault/resources/vault_2/content");
        then.status(200).body("weights");
    });
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("model.bin");
    let session = session(&server, true)?;
    for (resource_id, why) in [
        ("vault_1", "no longer matches"),
        ("vault_2", "no sha256 ETag"),
    ] {
        let refused = session
            .invoke(
                route("vault_download")?,
                &args(json!({ "resource_id": resource_id, "out": out }))?,
            )
            .err()
            .with_context(|| format!("{resource_id} landed"))?;
        assert!(refused.to_string().contains(why), "{refused}");
        assert_eq!(fs::read_dir(dir.path())?.count(), 0);
    }
    Ok(())
}

/// Revoking the key the session signs with says every later call answers 401;
/// another key says nothing.
#[test]
fn revoking_the_session_key_says_it_locks_the_session_out() -> anyhow::Result<()> {
    let server = MockServer::start();
    ["userkey_9f2c", "userkey_other"]
        .into_iter()
        .for_each(|key_id| {
            server.mock(|when, then| {
                when.method(DELETE)
                    .path(format!("/api/v1/users/user_0192f0c1/keys/{key_id}"));
                then.status(204);
            });
        });
    let session = session(&server, true)?;
    let revoke = |key_id: &str| {
        session.invoke(
            route("users_key_revoke")?,
            &args(json!({ "user_id": "user_0192f0c1", "key_id": key_id }))?,
        )
    };
    let own = revoke("userkey_9f2c")?.notes;
    assert!(
        own.iter()
            .any(|note| note.contains("signing key") && note.contains("401")),
        "{own:?}"
    );
    assert!(revoke("userkey_other")?.notes.is_empty());
    Ok(())
}
