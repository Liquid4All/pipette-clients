use std::{
    cmp::Ordering,
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use nutype::nutype;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An ETag value is invalid when it contains bytes that are not legal in
/// an HTTP header value (control characters other than tab).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidHeaderValue;

impl std::fmt::Display for InvalidHeaderValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ETag is not a valid HTTP header value")
    }
}

impl std::error::Error for InvalidHeaderValue {}

fn is_valid_header_value(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte == b'\t' || (0x20..=0x7e).contains(&byte))
}

#[derive(Debug, Clone)]
pub struct EntityTag {
    value: String,
}

impl EntityTag {
    pub fn try_new(value: impl Into<String>) -> std::result::Result<Self, InvalidHeaderValue> {
        let value = value.into();
        if is_valid_header_value(&value) {
            Ok(Self { value })
        } else {
            Err(InvalidHeaderValue)
        }
    }

    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Parse an ETag from a response header value. Returns `None` if the
    /// header is not a legal value (it can never have been a valid ETag
    /// the server sent us).
    pub(crate) fn from_header_value(value: &str) -> Option<Self> {
        Self::try_new(value).ok()
    }

    fn header_value(&self) -> &str {
        &self.value
    }
}

impl PartialEq for EntityTag {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for EntityTag {}

impl PartialOrd for EntityTag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EntityTag {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl Hash for EntityTag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Serialize for EntityTag {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for EntityTag {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfNoneMatch(String);

impl IfNoneMatch {
    pub fn from_etag(etag: &EntityTag) -> Self {
        Self(etag.header_value().to_string())
    }

    pub(crate) fn header_value(&self) -> &str {
        &self.0
    }
}

impl From<&EntityTag> for IfNoneMatch {
    fn from(etag: &EntityTag) -> Self {
        Self::from_etag(etag)
    }
}

/// A pre-auth key (`preauth_<key_id>.<secret>`) that admits a registering
/// client already `approved`. `try_new` trims surrounding whitespace — CLI
/// flags and `PIPETTE_PREAUTH_KEY` routinely carry a trailing newline — and
/// rejects an empty value; the exact format stays the server's to validate.
/// Kept a distinct type so the secret can't be crossed with the other string
/// fields. `Serialize` emits the bare value for the wire, but `Debug` and
/// `Display` are hand-written to redact it — see `docs/architecture.md`
/// (“Secrets”) — so neither a `RegisterRequest` dump nor a `{}` of the key
/// itself can leak it.
#[nutype(
    sanitize(trim),
    validate(not_empty),
    derive(Clone, PartialEq, Eq, AsRef, Serialize)
)]
pub struct PreauthKey(String);

impl std::fmt::Debug for PreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PreauthKey(<redacted>)")
    }
}

impl std::fmt::Display for PreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A pre-auth key is empty after trimming. Surfaced by `--preauth-key` at
/// the CLI boundary via `FromStr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("pre-auth key must not be empty")]
pub struct InvalidPreauthKey;

// `FromStr` is hand-written rather than derived so clap reports this message
// instead of nutype's type-name-leaking "PreauthKey is empty". Sound only
// while `not_empty` is the sole validator — the mapped message assumes empty
// is the only way construction fails; add a validator above and this lies.
impl std::str::FromStr for PreauthKey {
    type Err = InvalidPreauthKey;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::try_new(value).map_err(|_| InvalidPreauthKey)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisterRequest {
    pub public_key: String,
    pub organization: String,
    pub contact_email: String,
    pub client_details: String,
    /// Optional pre-auth key (`preauth_…`): a valid key admits the client
    /// already `approved`. Skipped on the wire when absent; never persisted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preauth_key: Option<PreauthKey>,
    /// Optional device profile + capabilities, same fields as
    /// [`UpdateClientRequest`]. When supplied at registration the server can
    /// match the client without a follow-up PATCH.
    #[serde(flatten)]
    pub device: DeviceProfileFields,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegisterResponse {
    pub client_id: String,
    pub status: String,
}

/// Device-profile + capability fields shared by `POST /clients/register` and
/// `PATCH /clients/me`. Every field is optional; absent/`null` leaves the
/// stored value unchanged on PATCH. `capabilities` is set-granular: when
/// present it replaces the stored set wholesale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceProfileFields {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_form_factor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_os_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_os_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_chip_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_ram_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_gpu_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_gpu_vram_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_npu_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_npu_vram_bytes: Option<u64>,
    /// Full replacement capability set when `Some`. Canonical lowercase flags
    /// (e.g. `runtime:llama_cpp`). Must not use a server-owned reserved
    /// namespace (`os:`, `device:`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

/// `PATCH /clients/me` request body. All fields optional; only present values
/// update. Device fields cannot be individually cleared once set.
#[derive(Debug, Clone, Default, Serialize)]
pub struct UpdateClientRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_details: Option<String>,
    #[serde(flatten)]
    pub device: DeviceProfileFields,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClientProfile {
    pub client_id: String,
    pub organization: String,
    pub client_details: String,
    pub contact_email: String,
    pub status: String,
    /// Mgmt-assigned tags (`team/mobile/ios`, …), display-only. `default`
    /// keeps this compatible with a server that predates the tags field.
    #[serde(default)]
    pub tags: Vec<String>,
    /// `true` while the client's eligible-index re-evaluation is pending
    /// after a profile/capability change. While set, plan operations are
    /// refused. Defaults to `false` for older servers that omit the field.
    #[serde(default)]
    pub reindex_pending: bool,
    /// Capability flags the client reported directly (empty when none).
    /// Server-derived `device_*` flags are not echoed here.
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub device_name: Option<String>,
    #[serde(default)]
    pub device_form_factor: Option<String>,
    #[serde(default)]
    pub device_os_name: Option<String>,
    #[serde(default)]
    pub device_os_version: Option<String>,
    #[serde(default)]
    pub device_chip_model: Option<String>,
    #[serde(default)]
    pub device_ram_bytes: Option<u64>,
    #[serde(default)]
    pub device_gpu_model: Option<String>,
    #[serde(default)]
    pub device_gpu_vram_bytes: Option<u64>,
    #[serde(default)]
    pub device_npu_model: Option<String>,
    #[serde(default)]
    pub device_npu_vram_bytes: Option<u64>,
}

/// A job leased via `POST /plans/claim`: the server-owned envelope wrapped
/// around the work payload the server never interprets.
///
/// The envelope is the subset the management server acts on: identity, lease,
/// expiry. Everything needed to *execute* is `ClientRunSpec` (in
/// `pipette-plan-types`, which this crate does not depend on), typed and
/// validated at deserialization — flags carry their own
/// `(benchmark, runtime, model)` discriminants, so a mis-authored cell is
/// rejected on arrival rather than after the benchmark body has been fetched.
///
/// The `model_*` / `runtime_*` grouping labels a job body may still carry are
/// the server's own bookkeeping — it echoes them into synthetic failure records
/// without the client's help. They are ignored here, as is any future server
/// addition: unlisted fields are dropped at deserialization, never rejected.
#[derive(Debug, Deserialize)]
pub struct ClaimedJob {
    pub job_id: String,
    pub benchmark_id: String,
    /// Lease increment as an ISO 8601 duration (e.g. `"PT10M"`). Heartbeat at
    /// half this interval; each success extends the lease by this much.
    pub time_window: String,
    #[serde(default)]
    pub expires_at: Option<String>,
    /// The cell to run, carried opaquely — this crate models the wire and takes
    /// no view of what a cell is. The runner types it (`pipette_cli::client::claim`).
    ///
    /// Left as JSON, and defaulted to `null` when absent, so a payload this
    /// client cannot understand never fails the decode of the envelope around
    /// it.
    #[serde(default)]
    pub spec: Value,
}

/// `POST /benchmarks` failure variant body for a plan-attached job.
///
/// Identity and reason only. Every `model_*` / `runtime_*` field the endpoint
/// accepts is optional, and the server already holds the job body this `job_id`
/// names — echoing the cell back would restate what it can read, in a spelling
/// this client had to invent. It also keeps the injected HuggingFace token off
/// the failure path by construction rather than by remembering to strip it.
///
/// `client_version` is the exception that proves the rule: it is not in the job
/// body, so the server cannot recover it, and a failure is precisely when
/// "which build reported this" is the question.
#[derive(Debug, Clone, Serialize)]
pub struct FailureSubmission {
    pub message_type: &'static str,
    pub job_id: String,
    pub benchmark_id: String,
    pub failure_reason: String,
    pub retriable: bool,
    pub client_version: String,
}

impl FailureSubmission {
    /// Build a failure body for a claimed job. Reads only the envelope, so a
    /// job whose spec never parsed still reports.
    ///
    /// `client_version` is the caller's build identity — this crate models the
    /// wire and has no version of its own to report.
    pub fn from_claim(
        job: &ClaimedJob,
        failure_reason: impl Into<String>,
        retriable: bool,
        client_version: impl Into<String>,
    ) -> Self {
        Self {
            message_type: "failure",
            job_id: job.job_id.clone(),
            benchmark_id: job.benchmark_id.clone(),
            failure_reason: failure_reason.into(),
            retriable,
            client_version: client_version.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkSummary {
    pub benchmark_id: String,
    pub benchmark_type: String,
    #[serde(flatten)]
    pub parameters: BTreeMap<String, Value>,
}

/// Loose, forward-compatible view of an upstream benchmark — the shape the mgmt
/// server returns and the form sync stores raw. `benchmark_type` + any unknown
/// `parameter_*` keys flow through untyped (the `parameters` bag), so a benchmark
/// type or parameter the client doesn't know yet never breaks sync. Convert it to
/// the strict, typed `BenchmarkDefinition` (via `TryFrom`) before listing/running.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RemoteBenchmark {
    pub benchmark_id: String,
    pub benchmark_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<Value>>,
    #[serde(flatten)]
    pub parameters: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubmitResponse {
    pub job_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchSubmitItemResponse {
    pub index: usize,
    pub job_id: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BatchSubmitResponse {
    #[serde(default)]
    pub results: Vec<BatchSubmitItemResponse>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkJobMetric {
    pub metric: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobResponse {
    pub status: String,
    pub scored_at: Option<String>,
    pub metrics: Option<Vec<BenchmarkJobMetric>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_tag_serializes_as_string() -> Result<(), Box<dyn std::error::Error>> {
        let etag = EntityTag::try_new("\"abc123\"")?;
        assert_eq!(serde_json::to_string(&etag)?, "\"\\\"abc123\\\"\"");
        assert_eq!(
            serde_json::from_str::<EntityTag>("\"\\\"abc123\\\"\"")?,
            etag
        );
        Ok(())
    }

    #[test]
    fn if_none_match_wraps_entity_tag_as_header_value() -> Result<(), Box<dyn std::error::Error>> {
        let etag = EntityTag::try_new("\"abc123\"")?;
        let if_none_match = IfNoneMatch::from(&etag);
        assert_eq!(if_none_match.header_value(), "\"abc123\"");
        Ok(())
    }

    // Absent key → field omitted (not `null`), keeping keyless registration's
    // wire unchanged; present key → serialized verbatim.
    #[test]
    fn register_request_preauth_key_serialization() -> Result<(), Box<dyn std::error::Error>> {
        // (preauth_key input, expected serialized value)
        let cases = [
            (None, None),
            (Some("preauth_id.secret"), Some("preauth_id.secret")),
        ];
        cases.into_iter().try_for_each(
            |(input, expected)| -> Result<(), Box<dyn std::error::Error>> {
                let request = RegisterRequest {
                    public_key: "pk".into(),
                    organization: "LiquidAI".into(),
                    contact_email: "user@example.com".into(),
                    client_details: "ci box".into(),
                    preauth_key: input.map(PreauthKey::try_new).transpose()?,
                    device: Default::default(),
                };
                let value: serde_json::Value = serde_json::to_value(&request)?;
                assert_eq!(
                    value.get("preauth_key").and_then(serde_json::Value::as_str),
                    expected,
                    "preauth_key input {input:?} should serialize to {expected:?}"
                );
                Ok(())
            },
        )
    }

    // try_new trims input and rejects only an empty value; Debug never
    // renders the secret.
    #[test]
    fn preauth_key_trim_and_redaction() -> Result<(), Box<dyn std::error::Error>> {
        let key = PreauthKey::try_new("  preauth_id.secret\n")?;
        assert_eq!(
            key.as_ref(),
            "preauth_id.secret",
            "surrounding whitespace trimmed"
        );
        assert_eq!(
            format!("{key:?}"),
            "PreauthKey(<redacted>)",
            "Debug must not leak the key"
        );
        assert_eq!(
            format!("{key}"),
            "<redacted>",
            "Display must not leak the key either — `{{}}` has to be safe so no \
             caller reaches for as_ref() to satisfy it"
        );

        ["", "   ", "\n\t"].into_iter().for_each(|empty| {
            assert!(
                PreauthKey::try_new(empty).is_err(),
                "{empty:?} should be rejected as empty"
            );
        });
        Ok(())
    }

    // FromStr (the clap boundary) surfaces the CLI-facing message, not
    // nutype's type-name-leaking one.
    #[test]
    fn preauth_key_from_str_message() -> Result<(), Box<dyn std::error::Error>> {
        "  preauth_ok  ".parse::<PreauthKey>()?;
        assert_eq!("".parse::<PreauthKey>(), Err(InvalidPreauthKey));
        assert_eq!(
            InvalidPreauthKey.to_string(),
            "pre-auth key must not be empty"
        );
        Ok(())
    }

    // A `tags` field omitted by the server (older server, or an untagged
    // client) must default to empty so `auth me` works against a pre-tags
    // server; when present it deserializes verbatim. Same for the newer
    // reindex_pending / capabilities / device_* fields.
    #[rstest::rstest]
    #[case::absent(
        r#"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved"}"#,
        &[],
        false,
        &[]
    )]
    #[case::present(
        r#"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved","tags":["team/mobile/ios","batch/2026-q3"],"reindex_pending":true,"capabilities":["runtime:llama_cpp"]}"#,
        &["team/mobile/ios", "batch/2026-q3"],
        true,
        &["runtime:llama_cpp"]
    )]
    fn client_profile_forward_compat(
        #[case] body: &str,
        #[case] expected_tags: &[&str],
        #[case] expected_reindex: bool,
        #[case] expected_caps: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let profile: ClientProfile = serde_json::from_str(body)?;
        let expected_tags: Vec<String> = expected_tags.iter().map(|s| s.to_string()).collect();
        let expected_caps: Vec<String> = expected_caps.iter().map(|s| s.to_string()).collect();
        assert_eq!(profile.tags, expected_tags);
        assert_eq!(profile.reindex_pending, expected_reindex);
        assert_eq!(profile.capabilities, expected_caps);
        Ok(())
    }

    const CLAIM_BODY: &str = r#"{
        "job_id": "job-1",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
        "model_name": "m",
        "spec": {
            "benchmark": "prefill_throughput_256",
            "model": {
                "type": "gguf_text",
                "source": "huggingface",
                "org": "o",
                "repo_name": "r",
                "path": "m-Q4_0.gguf"
            },
            "runtime": {
                "type": "llamacpp_cli_stock_tools",
                "source": "github_release",
                "version": "b5000",
                "flavor": "macos-arm64"
            }
        },
        "future_field": 42
    }"#;

    /// The fixture carries a server label and an unknown field; neither is
    /// modeled, and a claim must still decode.
    #[test]
    fn claimed_job_tolerates_server_labels_and_unknown_fields(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_BODY)?;
        assert_eq!(job.job_id, "job-1");
        assert_eq!(job.time_window, "PT10M");
        Ok(())
    }

    #[rstest::rstest]
    #[case::unreadable(serde_json::json!({"benchmark": "prefill_throughput_256", "model": "nope"}))]
    #[case::absent(serde_json::Value::Null)]
    fn an_unreadable_spec_still_yields_a_reportable_job(
        #[case] spec: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut body: serde_json::Value = serde_json::from_str(CLAIM_BODY)?;
        let obj = body.as_object_mut().ok_or("the fixture is an object")?;
        if spec.is_null() {
            obj.remove("spec");
        } else {
            obj.insert("spec".into(), spec);
        }

        // The envelope decodes whatever the payload is: without a job_id there
        // is nothing to fail against.
        let job: ClaimedJob = serde_json::from_value(body)?;
        assert_eq!(job.job_id, "job-1");

        let failure =
            FailureSubmission::from_claim(&job, "[ts] bad spec", false, "9.9.9 (build t)");
        assert_eq!(failure.job_id, "job-1");
        assert_eq!(failure.benchmark_id, "prefill_throughput_256");
        assert!(!failure.retriable);
        assert_eq!(failure.client_version, "9.9.9 (build t)");
        Ok(())
    }

    #[test]
    fn update_client_request_omits_unset_fields() -> Result<(), Box<dyn std::error::Error>> {
        let req = UpdateClientRequest {
            client_details: Some("box".into()),
            device: DeviceProfileFields {
                device_name: Some("Mac".into()),
                capabilities: Some(vec!["runtime:llama_cpp".into()]),
                ..Default::default()
            },
        };
        let value = serde_json::to_value(&req)?;
        assert_eq!(value["client_details"], "box");
        assert_eq!(value["device_name"], "Mac");
        assert_eq!(
            value["capabilities"],
            serde_json::json!(["runtime:llama_cpp"])
        );
        assert!(value.get("device_ram_bytes").is_none());
        Ok(())
    }
}
