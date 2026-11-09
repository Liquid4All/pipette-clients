use std::{
    cmp::Ordering,
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

use nutype::nutype;
use serde::{Deserialize, Serialize, de};
use serde_json::Value;

pub use pipette_plan_types::{ClientStatus, ContactEmail, JobStatus};
use pipette_plan_types::{
    ContactEmailError, FailureCode, OrgId, OrgSlug, RuntimeOutput, Sha256, UserId, UserKeyId,
    VaultResourceId,
};

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

    /// `None` for a value no valid header could carry.
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

/// A client pre-auth key (`client_preauth_<key_id>.<secret>`, or the older
/// `preauth_` spelling the server still parses) that admits a registering
/// **client** already `approved`. `try_new` trims (a `PIPETTE_PREAUTH_KEY` often
/// ends in a newline) and rejects empty; the format is the server's to check.
///
/// A user registers against a [`UserPreauthKey`] instead: a separate record in
/// mgmt (its `docs/storage.md` §3.1.5), so a separate type here.
///
/// `Serialize` emits the bare value; `Debug` and `Display` redact it (this
/// repo's `docs/architecture.md`, "Secrets").
#[nutype(
    sanitize(trim),
    validate(not_empty),
    derive(Clone, PartialEq, Eq, AsRef, Serialize)
)]
pub struct ClientPreauthKey(String);

impl std::fmt::Debug for ClientPreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ClientPreauthKey(<redacted>)")
    }
}

impl std::fmt::Display for ClientPreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A client pre-auth key is empty after trimming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("pre-auth key must not be empty")]
pub struct EmptyClientPreauthKeyError;

// Hand-written so clap reports this message instead of nutype's
// "ClientPreauthKey is empty". Sound only while `not_empty` is the sole
// validator.
impl std::str::FromStr for ClientPreauthKey {
    type Err = EmptyClientPreauthKeyError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::try_new(value).map_err(|_| EmptyClientPreauthKeyError)
    }
}

/// A **user** pre-auth key (mgmt calls its record an *invite*): admits a key
/// registration, already approved, against the account it names.
///
/// Not the same type as [`ClientPreauthKey`]: the records differ and are
/// minted by different commands (mgmt's `docs/storage.md` §3.1.5), and
/// presenting one where the other belongs is a refusal. Trimmed, refused when
/// empty, and redacted, as [`ClientPreauthKey`] is.
#[nutype(
    sanitize(trim),
    validate(not_empty),
    derive(Clone, PartialEq, Eq, AsRef, Serialize)
)]
pub struct UserPreauthKey(String);

impl std::fmt::Debug for UserPreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UserPreauthKey(<redacted>)")
    }
}

impl std::fmt::Display for UserPreauthKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A user pre-auth key is empty after trimming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("user pre-auth key must not be empty")]
pub struct EmptyUserPreauthKeyError;

// Hand-written as `ClientPreauthKey`'s is, and sound under the same condition.
impl std::str::FromStr for UserPreauthKey {
    type Err = EmptyUserPreauthKeyError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::try_new(value).map_err(|_| EmptyUserPreauthKeyError)
    }
}

/// What a registered key is, for a human deciding whether to revoke it later
/// ("work laptop"). Refused when empty: otherwise a key shows only its id.
///
/// On requests only. [`UserKey::key_details`] stays a `String`, so a blank one
/// from the server does not stop the account being read.
#[nutype(
    sanitize(trim),
    validate(not_empty),
    derive(Clone, Debug, Display, PartialEq, Eq, AsRef, Serialize)
)]
pub struct KeyDetails(String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("key details must not be empty")]
pub struct EmptyKeyDetailsError;

// Hand-written as `ClientPreauthKey`'s is, and sound under the same condition.
impl std::str::FromStr for KeyDetails {
    type Err = EmptyKeyDetailsError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::try_new(value).map_err(|_| EmptyKeyDetailsError)
    }
}

/// An Ed25519 private key is its 32-byte seed.
const PRIVATE_KEY_LEN: usize = 32;

/// An Ed25519 private key, as hex.
///
/// `try_new` trims (a key file ends in a newline) and refuses anything that
/// cannot sign, so the failure is at the file it was read from rather than at
/// the first signed request. Lowercased, so equality means "the same key" and
/// `auth import` does not demand `--force` to replace a key with itself.
///
/// `Serialize`/`Deserialize` are the bare hex, which `auth export` needs;
/// `Debug` and `Display` redact (this repo's `docs/architecture.md`,
/// "Secrets").
#[nutype(
    sanitize(trim, lowercase),
    validate(with = check_private_key, error = PrivateKeyError),
    derive(Clone, PartialEq, Eq, AsRef, Serialize, Deserialize)
)]
pub struct PrivateKey(String);

impl std::fmt::Debug for PrivateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrivateKey(<redacted>)")
    }
}

impl std::fmt::Display for PrivateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

fn check_private_key(value: &str) -> std::result::Result<(), PrivateKeyError> {
    let decoded = hex::decode(value).map_err(PrivateKeyError::NotHex)?;
    if decoded.len() != PRIVATE_KEY_LEN {
        return Err(PrivateKeyError::Length {
            found: decoded.len(),
        });
    }
    Ok(())
}

/// Why a value cannot sign. Each variant names the value's shape, never the
/// value.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PrivateKeyError {
    /// A truncated paste, a base64 token pasted into a key file, a stray
    /// character from an editor.
    #[error("private key is not hex: {0}")]
    NotHex(#[source] hex::FromHexError),

    /// Hex, but not a seed: a truncated copy, or the 64-byte expanded keypair
    /// some tooling calls a private key.
    #[error("private key must be {PRIVATE_KEY_LEN} bytes of hex, not {found}")]
    Length { found: usize },
}

/// An Ed25519 public key is 32 bytes, the same as the seed behind it.
const PUBLIC_KEY_LEN: usize = 32;

/// The Ed25519 public key a client or a user key registers under, in hex.
///
/// Validated as [`PrivateKey`] is, so an unusable key is refused here rather
/// than by the server after the keypair is on disk. Not redacted: this is the
/// half you may publish. Lowercased, so equality means "the same key".
#[nutype(
    sanitize(trim, lowercase),
    validate(with = check_public_key, error = PublicKeyError),
    derive(Clone, Debug, Display, FromStr, PartialEq, Eq, AsRef, Serialize, Deserialize)
)]
pub struct PublicKey(String);

fn check_public_key(value: &str) -> std::result::Result<(), PublicKeyError> {
    let decoded = hex::decode(value).map_err(PublicKeyError::NotHex)?;
    if decoded.len() != PUBLIC_KEY_LEN {
        return Err(PublicKeyError::Length {
            found: decoded.len(),
        });
    }
    Ok(())
}

/// Why a value cannot be a public key.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PublicKeyError {
    #[error("public key is not hex: {0}")]
    NotHex(#[source] hex::FromHexError),

    #[error("public key must be {PUBLIC_KEY_LEN} bytes of hex, not {found}")]
    Length { found: usize },
}

/// Refuses at the flag what a registration would, before it spends a pre-auth key.
pub fn parse_contact_email(raw: &str) -> std::result::Result<ContactEmail, ContactEmailParseError> {
    ContactEmail::try_new(raw).map_err(ContactEmailParseError)
}

/// Names the rules people break, which mgmt's own error does not; no list could name them all.
#[derive(Debug, thiserror::Error)]
#[error(
    "contact email must be a bare address like `ops@example.com`: one `@`, a dotted domain, \
     no display name, quotes, spaces, `/` or `..` before the `@`, at most 254 bytes, among \
     RFC 5322's rules"
)]
pub struct ContactEmailParseError(#[source] ContactEmailError);

/// An instant the server stamped, in either ISO 8601 spelling it uses: the
/// extended form on most fields, and the basic one (`20240908T000000Z`) on a
/// claim's `expires_at` (mgmt's `docs/httpapi.md` §2.9.2).
///
/// Validated, then carried as the string it arrived as: nothing here does
/// arithmetic on it, and a `time::OffsetDateTime` would re-render it
/// differently on the way out.
#[nutype(
    sanitize(trim),
    validate(with = check_timestamp, error = TimestampError),
    derive(Clone, Debug, Display, FromStr, PartialEq, Eq, AsRef, Serialize, Deserialize)
)]
pub struct Timestamp(String);

fn check_timestamp(value: &str) -> std::result::Result<(), TimestampError> {
    // `PARSING` rather than `Rfc3339`: it reads both spellings the server uses,
    // and still refuses a date with no time or a time with no offset.
    time::OffsetDateTime::parse(
        value,
        &time::format_description::well_known::Iso8601::PARSING,
    )
    .map(|_| ())
    .map_err(|source| TimestampError { source })
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("timestamp is not a date and time with an offset: {source}")]
pub struct TimestampError {
    #[source]
    source: time::error::Parse,
}

/// An optional [`Timestamp`] that reads absent, `null` and unparseable all as
/// `None`. Only for a field nothing branches on: there, refusing the response
/// costs more than losing the value.
fn lenient_timestamp<'de, D>(deserializer: D) -> std::result::Result<Option<Timestamp>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.and_then(|raw| Timestamp::try_new(raw).ok()))
}

fn lenient_versions<'de, D>(deserializer: D) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: de::Deserializer<'de>,
{
    Ok(Option::<Value>::deserialize(deserializer)?.map(|value| {
        serde_json::from_value(value).unwrap_or_else(|error| {
            tracing::warn!(
                error = %error,
                "the claim's client_versions is not an array of strings; reading it as no build"
            );
            Vec::new()
        })
    }))
}

/// The management server's base URL.
///
/// Trimmed, parsed and stripped of a trailing `/`, so `format!("{server}{path}")`
/// joins cleanly and the signed target matches what the server receives. A
/// path prefix (`https://host/api`) is kept, since a deployment behind one
/// signs over it.
///
/// `Deserialize` validates too, but its error points into JSON by line and
/// column; where the operator never saw that JSON (an identity bundle), read a
/// `String` and call `try_new`.
#[nutype(
    sanitize(with = normalize_server_url),
    validate(with = check_server_url, error = ServerUrlError),
    derive(Clone, Debug, Display, FromStr, PartialEq, Eq, Serialize, Deserialize)
)]
pub struct ServerUrl(String);

/// A value the parser rejects is passed through for [`check_server_url`] to
/// report.
fn normalize_server_url(raw: String) -> String {
    let trimmed = raw.trim();
    match url::Url::parse(trimmed) {
        Ok(parsed) => parsed.to_string().trim_end_matches('/').to_string(),
        Err(_) => trimmed.to_string(),
    }
}

/// Parsing is not enough: the value also has to be a base a path can be
/// appended to, and a scheme the HTTP client can speak.
fn check_server_url(value: &str) -> std::result::Result<(), ServerUrlError> {
    let parsed = url::Url::parse(value).map_err(|source| ServerUrlError::NotAUrl {
        url: value.to_string(),
        source,
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ServerUrlError::Scheme {
            url: value.to_string(),
        });
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(ServerUrlError::NotABase {
            url: value.to_string(),
        });
    }
    Ok(())
}

/// Why a value can't be a management server base URL.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ServerUrlError {
    /// A bare host (`mgmt.example.com`), or a typo'd scheme.
    #[error("invalid management server URL `{url}`: {source}")]
    NotAUrl {
        url: String,
        #[source]
        source: url::ParseError,
    },

    /// `mailto:`, `file:`, `ftp:`: anything the blocking client can't send.
    #[error("management server URL `{url}` must use http or https")]
    Scheme { url: String },

    /// Neither survives `format!("{server}{path}")`: the endpoint path lands
    /// inside the query or never reaches the wire, and every request goes
    /// to `/`.
    #[error("management server URL `{url}` must not carry a query or fragment")]
    NotABase { url: String },
}

/// Whether an account may act. An invited account is `active` at once; one
/// that asked for itself is `pending` until an operator admits it.
///
/// Not mgmt's `UserStatus`, which has no [`Self::Unknown`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserStatus {
    Active,
    Pending,
    /// Offboarded. Disabled rather than deleted, so the plans the account filed
    /// keep resolving to a principal that exists.
    Disabled,
    /// A status introduced after this build, carried verbatim.
    #[serde(untagged)]
    Unknown(String),
}

impl std::fmt::Display for UserStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => f.write_str("active"),
            Self::Pending => f.write_str("pending"),
            Self::Disabled => f.write_str("disabled"),
            Self::Unknown(raw) => f.write_str(raw),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisterRequest {
    pub public_key: PublicKey,
    pub organization: String,
    pub contact_email: ContactEmail,
    pub client_details: String,
    /// A valid key admits the client already `approved`. Never persisted.
    /// The server also reads the old `preauth_key` name, and both in one body
    /// are a `400` (mgmt's `docs/authentication.md` §3.2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_preauth_key: Option<ClientPreauthKey>,
    /// Sent at registration so the server can match the client without a
    /// follow-up PATCH.
    #[serde(flatten)]
    pub device: DeviceProfileFields,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegisterResponse {
    pub client_id: String,
    pub status: ClientStatus,
}

/// `POST /api/v1/users/register`: bind a public key to an account. With an
/// invite, the account exists and the registration is approved. Without one,
/// `contact_email` asks for an account, created [`UserStatus::Pending`]. Set
/// one or the other: an address beside an invite would describe a record this
/// route may not touch.
///
/// mgmt's `RegisterUserKeyRequest` in JSON, in this crate's field types so a
/// blank invite is refused at the flag.
#[derive(Debug, Clone, Serialize)]
pub struct RegisterUserRequest {
    pub public_key: PublicKey,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_preauth_key: Option<UserPreauthKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_email: Option<ContactEmail>,
    pub key_details: KeyDetails,
}

/// The `POST /api/v1/users/register` receipt: mgmt's `UserKeyRegistered`, read
/// with this crate's [`UserStatus`].
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegisterUserResponse {
    pub user_id: UserId,
    pub status: UserStatus,
    /// Required: a holder compares it with the id it derived before saving an
    /// identity, and a missing one cannot be told from agreement.
    #[serde(rename = "key_id")]
    pub user_key_id: UserKeyId,
}

/// `POST /api/v1/users/me/keys`: mgmt's `AddUserKeyRequest` in this crate's
/// field types.
#[derive(Debug, Clone, Serialize)]
pub struct AddUserKeyRequest {
    pub public_key: PublicKey,
    pub key_details: KeyDetails,
    /// When the key stops signing; absent, it never does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<Timestamp>,
}

/// The `POST /api/v1/users/me/keys` receipt, the same body as the register
/// one. The caller already knows its account, so only the key id is read.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AddUserKeyResponse {
    #[serde(rename = "key_id")]
    pub user_key_id: UserKeyId,
}

/// A request's nonce: 16 random bytes, rendered as lowercase hex, which cannot
/// carry a newline into the newline-delimited signed payload.
#[nutype(derive(Clone, Copy, Debug, PartialEq, Eq))]
pub struct Nonce([u8; crate::auth::NONCE_LEN]);

impl std::fmt::Display for Nonce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode(self.into_inner()))
    }
}

/// A request's `X-Signature`: an Ed25519 signature, rendered as lowercase hex.
#[nutype(derive(Clone, Copy, Debug, PartialEq, Eq))]
pub struct RequestSignature([u8; crate::auth::SIGNATURE_LEN]);

impl std::fmt::Display for RequestSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex::encode(self.into_inner()))
    }
}

/// `GET /api/v1/users/me`. Not mgmt's `OwnUserView`, which refuses the whole
/// account over one value this build cannot read.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UserProfile {
    pub user_id: UserId,
    pub contact_email: ContactEmail,
    /// As sent: only shown, so a type a newer server adds still reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_type: Option<String>,
    pub status: UserStatus,
    #[serde(default, deserialize_with = "lenient_timestamp")]
    pub created_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org_id: Option<OrgId>,
    /// A member may not read it off `/api/v1/orgs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org_slug: Option<OrgSlug>,
    /// Sorted, and already the implication closure narrowed to what a user may
    /// hold.
    #[serde(default)]
    pub permissions: Vec<String>,
    #[serde(default)]
    pub keys: UserKeys,
}

/// The `keys` array of a [`UserProfile`]. An entry that fails to decode is
/// dropped and counted, so the account still reads, and a dropped entry is not
/// mistaken for a revoked key.
#[derive(Debug, Clone, Default)]
pub struct UserKeys {
    pub entries: Vec<UserKey>,
    /// How many entries the server sent that this build could not read.
    pub unreadable: usize,
}

impl<'de> Deserialize<'de> for UserKeys {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let sent = Vec::<Value>::deserialize(deserializer)?;
        let total = sent.len();
        let entries: Vec<UserKey> = sent
            .into_iter()
            .filter_map(|entry| serde_json::from_value(entry).ok())
            .collect();
        Ok(Self {
            unreadable: total - entries.len(),
            entries,
        })
    }
}

impl Serialize for UserKeys {
    /// The entries as a bare array. `unreadable` does not survive, so a caller
    /// rendering a profile prints it itself, as `auth me` does.
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.entries.serialize(serializer)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UserKey {
    #[serde(rename = "key_id")]
    pub user_key_id: UserKeyId,
    #[serde(default)]
    pub key_details: String,
    #[serde(default, deserialize_with = "lenient_timestamp")]
    pub added_at: Option<Timestamp>,
    /// Strict, unlike `added_at`: an unparseable expiry drops the entry rather
    /// than reading as never expiring.
    #[serde(default)]
    pub expires_at: Option<Timestamp>,
}

/// Device-profile and capability fields shared by `POST /clients/register` and
/// `PATCH /clients/me`. An absent field leaves the stored value unchanged on
/// PATCH.
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
    /// The CPU architecture as the platform spells it (`aarch64`,
    /// `arm64-v8a`). The server derives the `arch:` flag from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_cpu_arch: Option<String>,
    /// The OS family as the platform spells it (`linux`, `iOS`). The server
    /// derives the `os_family:` flag from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_os_family: Option<String>,
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
    /// Replaces the stored set wholesale when `Some`. Canonical lowercase flags
    /// (`runtime:llama_cpp`), outside the server-owned namespaces (`os:`,
    /// `device:`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

/// `PATCH /clients/me` request body. Device fields cannot be cleared once set.
#[derive(Debug, Clone, Default, Serialize)]
pub struct UpdateClientRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_details: Option<String>,
    #[serde(flatten)]
    pub device: DeviceProfileFields,
}

/// A vault resource record, as `GET /vault/resources/{resource_id}` answers
/// it to a device. The rest of the record (owner, object key) is left unread.
///
/// The digest is a **pin, not an observation**: mgmt never reads the object,
/// so a download that hashes to something else found the object replaced.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct VaultResource {
    pub resource_id: VaultResourceId,
    pub name: String,
    pub sha256: Sha256,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClientProfile {
    pub client_id: String,
    pub organization: String,
    pub client_details: String,
    /// Plain text, as mgmt's `ClientRecord` answers it.
    pub contact_email: String,
    pub status: ClientStatus,
    /// Mgmt-assigned tags (`team/mobile/ios`), display-only.
    #[serde(default)]
    pub tags: Vec<String>,
    /// What the client may call, operator-assigned (`perform_work`,
    /// `submit_results`). `None` on a server that predates the field, so a
    /// caller can tell "holds none" from "never said".
    #[serde(default)]
    pub permissions: Option<Vec<String>>,
    /// `true` while the client's eligible-index re-evaluation is pending after
    /// a profile or capability change. While set, plan operations are refused.
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

/// A job leased via `POST /plans/claim`: the envelope the server acts on
/// (identity, lease, expiry) around the work payload it never interprets.
/// Unlisted fields, such as the server's `model_*` / `runtime_*` labels, are
/// dropped, never rejected.
#[derive(Debug, Deserialize)]
pub struct ClaimedJob {
    pub job_id: String,
    pub benchmark_id: String,
    /// Lease increment as an ISO 8601 duration (`"PT10M"`). Heartbeat at half
    /// this interval; each success extends the lease by this much.
    pub time_window: String,
    /// Lenient: nothing on the client branches on it, so a format change on
    /// the server must not stop the fleet.
    #[serde(default, deserialize_with = "lenient_timestamp")]
    pub expires_at: Option<Timestamp>,
    /// The versions (the `client_version:` capability values) the job admits, ascending;
    /// `None` from an older server. Anything but an array of strings reads as
    /// `Some(vec![])`, so the job is reported failed rather than left to lapse.
    #[serde(default, deserialize_with = "lenient_versions")]
    pub client_versions: Option<Vec<String>>,
    /// The cell to run, left as JSON (`null` when absent) so a payload this
    /// build cannot read never fails the envelope. The runner types it
    /// (`pipette_cli::client::claim`).
    #[serde(default)]
    pub spec: Value,
}

/// `POST /benchmarks` failure variant body for a plan-attached job.
///
/// Carries no `model_*` / `runtime_*` fields: the server already holds the job
/// body, and leaving the cell out keeps the injected HuggingFace token off the
/// failure path. Beyond the job's identity and the reason, it carries only what
/// the job body lacks: `client_version`, `runtime_output` and `device_time_ms`.
#[derive(Debug, Clone, Serialize)]
pub struct FailureSubmission {
    pub message_type: &'static str,
    pub job_id: String,
    pub benchmark_id: String,
    pub failure_reason: String,
    pub retriable: bool,
    /// Left out when the client cannot tell: mgmt refuses a `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<FailureCode>,
    pub client_version: String,
    /// Milliseconds from the claim to the failure being ready to submit, on a
    /// monotonic clock, excluding the submit and its retries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_time_ms: Option<u64>,
    /// What the runtime printed before it failed.
    #[serde(flatten)]
    pub runtime_output: RuntimeOutput,
}

impl FailureSubmission {
    /// Reads only the envelope, so a job whose spec never parsed still reports.
    /// `client_version` is the caller's build: this crate has none of its own.
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
            failure_code: None,
            client_version: client_version.into(),
            device_time_ms: None,
            runtime_output: RuntimeOutput::default(),
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

/// A benchmark as mgmt returns it, untyped: every key without a field (unknown
/// `parameter_*` keys, `org_id`, a corpus `dataset`) lands in `parameters`, so a
/// type or parameter this build does not know never breaks sync, and re-serializing
/// reproduces what the server sent. Convert it to the strict `BenchmarkDefinition`
/// before listing or running.
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
pub struct BenchmarkJobMetric {
    pub metric: String,
    pub value: f64,
    pub unit: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JobResponse {
    pub status: JobStatus,
    /// Present once `status` is `processed`. Lenient: polled on every `sync`,
    /// and nothing branches on it.
    #[serde(default, deserialize_with = "lenient_timestamp")]
    pub scored_at: Option<Timestamp>,
    pub metrics: Option<Vec<BenchmarkJobMetric>>,
}

#[cfg(test)]
mod tests {
    use pipette_plan_types as plan_types;

    use super::*;

    /// An unknown status round-trips as the word the server sent.
    #[rstest::rstest]
    #[case::pending(r#""pending""#, ClientStatus::Pending)]
    #[case::approved(r#""approved""#, ClientStatus::Approved)]
    #[case::future(r#""quarantined""#, ClientStatus::Unknown("quarantined".into()))]
    fn a_client_status_decodes_or_carries_the_unknown(
        #[case] body: &str,
        #[case] expected: ClientStatus,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let status: ClientStatus = serde_json::from_str(body)?;
        assert_eq!(status, expected);
        assert_eq!(serde_json::to_string(&status)?, body);
        Ok(())
    }

    #[rstest::rstest]
    #[case::active(r#""active""#, UserStatus::Active)]
    #[case::disabled(r#""disabled""#, UserStatus::Disabled)]
    #[case::future(r#""suspended""#, UserStatus::Unknown("suspended".into()))]
    fn a_user_status_decodes_or_carries_the_unknown(
        #[case] body: &str,
        #[case] expected: UserStatus,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let status: UserStatus = serde_json::from_str(body)?;
        assert_eq!(status, expected);
        assert_eq!(serde_json::to_string(&status)?, body);
        Ok(())
    }

    #[rstest::rstest]
    #[case::incoming(r#""incoming""#, JobStatus::Incoming)]
    #[case::scoring(r#""scoring""#, JobStatus::Scoring)]
    #[case::processed(r#""processed""#, JobStatus::Processed)]
    #[case::failed(r#""failed""#, JobStatus::Failed)]
    #[case::future(r#""requeued""#, JobStatus::Unknown("requeued".into()))]
    fn a_job_status_decodes_or_carries_the_unknown(
        #[case] body: &str,
        #[case] expected: JobStatus,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let status: JobStatus = serde_json::from_str(body)?;
        assert_eq!(status, expected);
        assert_eq!(serde_json::to_string(&status)?, body);
        Ok(())
    }

    /// `worker` gates claiming on `== Approved`, and `sync` its score refresh on
    /// `== Processed`.
    #[test]
    fn an_unknown_status_never_equals_a_known_one() {
        assert_ne!(
            ClientStatus::Unknown("approved-ish".into()),
            ClientStatus::Approved
        );
        assert_ne!(
            JobStatus::Unknown("processing".into()),
            JobStatus::Processed
        );
        assert_ne!(UserStatus::Unknown("active?".into()), UserStatus::Active);
    }

    #[test]
    fn a_status_displays_as_the_wire_word() {
        assert_eq!(ClientStatus::Approved.to_string(), "approved");
        assert_eq!(UserStatus::Disabled.to_string(), "disabled");
        assert_eq!(JobStatus::Processed.to_string(), "processed");
        assert_eq!(
            JobStatus::Unknown("requeued".into()).to_string(),
            "requeued"
        );
    }

    #[rstest::rstest]
    #[case::exact(&"a3f8".repeat(16), true)]
    #[case::uppercase(&"A3F8".repeat(16), true)]
    #[case::padded(&format!("  {}\n", "a3f8".repeat(16)), true)]
    #[case::truncated(&"a3f8".repeat(8), false)]
    #[case::too_long(&"a3f8".repeat(32), false)]
    #[case::not_hex(&"zz".repeat(32), false)]
    #[case::empty("", false)]
    fn public_key_accepts_only_32_bytes_of_hex(#[case] raw: &str, #[case] accepted: bool) {
        assert_eq!(PublicKey::try_new(raw).is_ok(), accepted, "{raw:?}");
    }

    #[test]
    fn public_key_normalizes_casing_and_padding() -> Result<(), Box<dyn std::error::Error>> {
        let plain = PublicKey::try_new("a3f8".repeat(16))?;
        let noisy = PublicKey::try_new(format!("  {}\n", "A3F8".repeat(16)))?;
        assert_eq!(plain, noisy, "the same key however it was pasted");
        Ok(())
    }

    /// `_` and non-ASCII domain labels pass; quotes and `..` don't.
    #[rstest::rstest]
    #[case::plain("ops@example.com", Ok("ops@example.com"))]
    #[case::plus("ops+fleet@example.com", Ok("ops+fleet@example.com"))]
    #[case::subdomain("ops@mail.corp.example.com", Ok("ops@mail.corp.example.com"))]
    #[case::padded("  ops@example.com  ", Ok("ops@example.com"))]
    #[case::underscore_label("ops@my_host.example.com", Ok("ops@my_host.example.com"))]
    #[case::non_ascii_label("ops@münchen.de", Ok("ops@münchen.de"))]
    #[case::cyrillic("ops@пример.рф", Ok("ops@пример.рф"))]
    #[case::no_at("ops.example.com", Err("one `@`"))]
    #[case::two_ats("ops@example@com", Err("one `@`"))]
    #[case::no_local("@example.com", Err("RFC 5322's rules"))]
    #[case::no_domain("ops@", Err("a dotted domain"))]
    #[case::bare_host("ops@localhost", Err("a dotted domain"))]
    #[case::trailing_dot("ops@example.com.", Err("a dotted domain"))]
    #[case::inner_space("ops @example.com", Err("spaces"))]
    #[case::trailing_comma("ops@example.com,", Err("RFC 5322's rules"))]
    #[case::display_name("Ops <ops@example.com>", Err("no display name"))]
    #[case::quoted_local("\"ops\"@example.com", Err("quotes"))]
    #[case::double_dot_local("ops..fleet@example.com", Err("`..` before the `@`"))]
    #[case::slash_local("ops/fleet@example.com", Err("`/`"))]
    #[case::too_long(&format!("ops@{}com", "a.".repeat(130)), Err("254 bytes"))]
    #[case::empty("", Err("one `@`"))]
    fn contact_email_flag_takes_mgmts_rule(
        #[case] raw: &str,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        match (parse_contact_email(raw), expected) {
            (Ok(email), Ok(stored)) => assert_eq!(email.as_str(), stored, "{raw:?}"),
            (Err(error), Err(rule)) => {
                assert!(error.to_string().contains(rule), "{raw:?}: {error}");
            }
            (got, expected) => anyhow::bail!("{raw:?}: got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    /// `_` and non-ASCII domain labels fall outside hostname characters, yet mgmt stores them.
    #[rstest::rstest]
    #[case::underscore_label("ops@my_host.example.com")]
    #[case::non_ascii_label("ops@münchen.de")]
    fn a_contact_email_mgmt_stores_reads_and_sends_as_typed(
        #[case] address: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let user: UserProfile = serde_json::from_value(serde_json::json!({
            "user_id": "user_0192f0c1", "contact_email": address, "status": "active",
        }))?;
        assert_eq!(user.contact_email.as_str(), address);
        let client: ClientProfile = serde_json::from_value(serde_json::json!({
            "client_id": "ev1_abc", "organization": "o", "client_details": "d",
            "contact_email": address, "status": "approved",
        }))?;
        assert_eq!(client.contact_email, address);
        let public_key = PublicKey::try_new("11".repeat(32))?;
        let device = serde_json::to_value(RegisterRequest {
            public_key: public_key.clone(),
            organization: "LiquidAI".into(),
            contact_email: parse_contact_email(address)?,
            client_details: "ci box".into(),
            client_preauth_key: None,
            device: Default::default(),
        })?;
        let account = serde_json::to_value(RegisterUserRequest {
            public_key,
            user_preauth_key: None,
            contact_email: Some(parse_contact_email(address)?),
            key_details: KeyDetails::try_new("work laptop")?,
        })?;
        [device, account].iter().for_each(|body| {
            assert_eq!(
                body.get("contact_email").and_then(Value::as_str),
                Some(address),
                "{body}"
            );
        });
        Ok(())
    }

    /// mgmt's request bodies refuse an unknown key, so a field either side
    /// renames fails here.
    fn mgmt_reads_back<T: Serialize + de::DeserializeOwned>(
        sent: &impl Serialize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let sent = serde_json::to_value(sent)?;
        let read: T = serde_json::from_value(sent.clone())?;
        assert_eq!(serde_json::to_value(read)?, sent);
        Ok(())
    }

    #[test]
    fn the_key_registration_bodies_are_mgmts() -> Result<(), Box<dyn std::error::Error>> {
        let public_key = PublicKey::try_new("11".repeat(32))?;
        let key_details = KeyDetails::try_new("work laptop")?;
        let invited = RegisterUserRequest {
            public_key: public_key.clone(),
            user_preauth_key: Some(UserPreauthKey::try_new(
                "user_preauth_0123456789abcdef.secret",
            )?),
            contact_email: None,
            key_details: key_details.clone(),
        };
        let asking = RegisterUserRequest {
            user_preauth_key: None,
            contact_email: Some(ContactEmail::try_new("ops@example.com")?),
            ..invited.clone()
        };
        mgmt_reads_back::<plan_types::RegisterUserKeyRequest>(&invited)?;
        mgmt_reads_back::<plan_types::RegisterUserKeyRequest>(&asking)?;
        [None, Some(Timestamp::try_new("2026-10-11T12:00:00Z")?)]
            .into_iter()
            .try_for_each(|expires_at| {
                mgmt_reads_back::<plan_types::AddUserKeyRequest>(&AddUserKeyRequest {
                    public_key: public_key.clone(),
                    key_details: key_details.clone(),
                    expires_at,
                })
            })
    }

    /// Both key routes answer mgmt's `UserKeyRegistered`.
    #[test]
    fn a_key_receipt_reads_mgmts() -> Result<(), Box<dyn std::error::Error>> {
        let sent = serde_json::to_value(plan_types::UserKeyRegistered {
            user_id: UserId::try_new("user_0192f0c1")?,
            status: plan_types::UserStatus::Pending,
            key_id: UserKeyId::try_new("userkey_9f2c")?,
        })?;
        let registered: RegisterUserResponse = serde_json::from_value(sent.clone())?;
        assert_eq!(serde_json::to_value(&registered)?, sent);
        let added: AddUserKeyResponse = serde_json::from_value(sent)?;
        assert_eq!(added.user_key_id, registered.user_key_id);
        Ok(())
    }

    /// Each status mgmt writes reads as the same one here, never as `Unknown`.
    #[test]
    fn a_user_status_reads_each_of_mgmts() -> Result<(), Box<dyn std::error::Error>> {
        [
            plan_types::UserStatus::Active,
            plan_types::UserStatus::Pending,
            plan_types::UserStatus::Disabled,
        ]
        .into_iter()
        .try_for_each(|sent| {
            // Exhaustive, so a status mgmt adds stops the build here.
            let expected = match sent {
                plan_types::UserStatus::Active => UserStatus::Active,
                plan_types::UserStatus::Pending => UserStatus::Pending,
                plan_types::UserStatus::Disabled => UserStatus::Disabled,
            };
            let read: UserStatus = serde_json::from_value(serde_json::to_value(sent)?)?;
            assert_eq!(read, expected);
            Ok(())
        })
    }

    /// mgmt's `OwnUserView` reads whole but for what `auth me` does not show.
    #[test]
    fn a_profile_reads_mgmts_own_view() -> Result<(), Box<dyn std::error::Error>> {
        let view: plan_types::OwnUserView =
            serde_json::from_value(crate::user_api::tests::own_user_view())?;
        let sent = serde_json::to_value(&view)?;
        let profile: UserProfile = serde_json::from_value(sent.clone())?;
        let mut shown = sent;
        let may = shown.as_object_mut().and_then(|view| view.remove("may"));
        let public_key = shown
            .pointer_mut("/keys/0")
            .and_then(Value::as_object_mut)
            .and_then(|key| key.remove("public_key"));
        assert!(may.is_some() && public_key.is_some(), "{shown}");
        assert_eq!(serde_json::to_value(&profile)?, shown);
        Ok(())
    }

    #[rstest::rstest]
    #[case::utc("2026-01-02T03:04:05Z", true)]
    #[case::offset("2026-01-02T03:04:05+02:00", true)]
    #[case::fractional("2026-01-02T03:04:05.123Z", true)]
    // The claim's `expires_at`, as the server's `avail/` key spells it.
    #[case::basic("20240908T000000Z", true)]
    #[case::basic_no_zone("20240908T000000", false)]
    #[case::basic_date_only("20240908", false)]
    #[case::date_only("2026-01-02", false)]
    #[case::no_zone("2026-01-02T03:04:05", false)]
    #[case::prose("last tuesday", false)]
    #[case::empty("", false)]
    fn timestamp_accepts_both_iso8601_spellings(#[case] raw: &str, #[case] accepted: bool) {
        assert_eq!(Timestamp::try_new(raw).is_ok(), accepted, "{raw:?}");
    }

    #[test]
    fn nonce_and_signature_render_as_lowercase_hex() {
        assert_eq!(Nonce::new([0xab; 16]).to_string(), "ab".repeat(16));
        assert_eq!(
            RequestSignature::new([0xcd; 64]).to_string(),
            "cd".repeat(64)
        );
    }

    #[rstest::rstest]
    #[case::valid(r#"{"expires_at":"2026-01-02T03:04:05Z"}"#, true)]
    #[case::as_the_server_sends_it(r#"{"expires_at":"20240908T000000Z"}"#, true)]
    #[case::garbage(r#"{"expires_at":"tomorrow-ish"}"#, false)]
    #[case::null(r#"{"expires_at":null}"#, false)]
    #[case::absent("{}", false)]
    fn a_claim_survives_an_unreadable_expiry(
        #[case] body: &str,
        #[case] some: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        #[derive(Deserialize)]
        struct Probe {
            #[serde(default, deserialize_with = "lenient_timestamp")]
            expires_at: Option<Timestamp>,
        }
        let probe: Probe = serde_json::from_str(body)?;
        assert_eq!(probe.expires_at.is_some(), some, "{body}");
        Ok(())
    }

    /// On the struct rather than the helpers: what breaks in practice is an
    /// attribute dropped from a field.
    #[rstest::rstest]
    #[case::entries_this_build_cannot_read(
        r#"{"user_id":"user_0192f0c1","contact_email":"ops@example.com","status":"active",
            "keys":[{"key_details":"no id at all"},
                    {"key_id":"userkey_9f2c","key_details":"work laptop"},
                    {"key_id":"not_a_user_key_id","key_details":"bad prefix"}]}"#,
        2, vec!["userkey_9f2c"], false, false
    )]
    #[case::timestamps_that_do_not_parse(
        r#"{"user_id":"user_0192f0c1","contact_email":"ops@example.com","status":"active",
            "created_at":"the beginning of time",
            "keys":[{"key_id":"userkey_9f2c","added_at":"last tuesday"}]}"#,
        0, vec!["userkey_9f2c"], false, false
    )]
    #[case::an_expiry_that_does_not_parse(
        r#"{"user_id":"user_0192f0c1","contact_email":"ops@example.com","status":"active",
            "keys":[{"key_id":"userkey_9f2c","expires_at":"next week"},
                    {"key_id":"userkey_4d0a","expires_at":"2026-10-11T12:00:00Z"}]}"#,
        1, vec!["userkey_4d0a"], false, false
    )]
    #[case::no_permissions_and_no_keys(
        r#"{"user_id":"user_0192f0c1","contact_email":"ops@example.com","status":"disabled"}"#,
        0, vec![], false, false
    )]
    #[case::timestamps_that_parse(
        r#"{"user_id":"user_0192f0c1","contact_email":"ops@example.com","status":"active",
            "created_at":"2026-01-02T03:04:05Z",
            "keys":[{"key_id":"userkey_9f2c","added_at":"2026-01-02T03:04:05Z"}]}"#,
        0, vec!["userkey_9f2c"], true, true
    )]
    fn a_user_profile_resolves_around_what_it_cannot_read(
        #[case] body: &str,
        #[case] unreadable: usize,
        #[case] entry_ids: Vec<&str>,
        #[case] created_at: bool,
        #[case] added_at: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let profile: UserProfile = serde_json::from_str(body)?;
        assert_eq!(profile.keys.unreadable, unreadable);
        assert_eq!(
            profile
                .keys
                .entries
                .iter()
                .map(|key| key.user_key_id.to_string())
                .collect::<Vec<_>>(),
            entry_ids,
        );
        assert_eq!(profile.created_at.is_some(), created_at);
        assert_eq!(
            profile
                .keys
                .entries
                .first()
                .is_some_and(|key| key.added_at.is_some()),
            added_at,
        );
        Ok(())
    }

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

    #[test]
    fn register_request_client_preauth_key_serialization() -> Result<(), Box<dyn std::error::Error>>
    {
        let cases = [
            (None, None),
            (
                Some("client_preauth_id.secret"),
                Some("client_preauth_id.secret"),
            ),
        ];
        cases.into_iter().try_for_each(
            |(input, expected)| -> Result<(), Box<dyn std::error::Error>> {
                let request = RegisterRequest {
                    public_key: PublicKey::try_new("11".repeat(32))?,
                    organization: "LiquidAI".into(),
                    contact_email: ContactEmail::try_new("user@example.com")?,
                    client_details: "ci box".into(),
                    client_preauth_key: input.map(ClientPreauthKey::try_new).transpose()?,
                    device: Default::default(),
                };
                let value: serde_json::Value = serde_json::to_value(&request)?;
                assert_eq!(
                    value
                        .get("client_preauth_key")
                        .and_then(serde_json::Value::as_str),
                    expected,
                    "client_preauth_key input {input:?} should serialize to {expected:?}"
                );
                assert!(
                    value.get("preauth_key").is_none(),
                    "both spellings in one body draw a 400 from the server"
                );
                Ok(())
            },
        )
    }

    #[test]
    fn user_preauth_key_trim_and_redaction() -> Result<(), Box<dyn std::error::Error>> {
        let invite = UserPreauthKey::try_new("  invite_id.secret\n")?;
        assert_eq!(invite.as_ref(), "invite_id.secret");
        assert_eq!(format!("{invite:?}"), "UserPreauthKey(<redacted>)");
        assert_eq!(format!("{invite}"), "<redacted>");
        assert!(UserPreauthKey::try_new("   ").is_err());
        Ok(())
    }

    #[test]
    fn preauth_key_trim_and_redaction() -> Result<(), Box<dyn std::error::Error>> {
        let key = ClientPreauthKey::try_new("  preauth_id.secret\n")?;
        assert_eq!(
            key.as_ref(),
            "preauth_id.secret",
            "surrounding whitespace trimmed"
        );
        assert_eq!(
            format!("{key:?}"),
            "ClientPreauthKey(<redacted>)",
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
                ClientPreauthKey::try_new(empty).is_err(),
                "{empty:?} should be rejected as empty"
            );
        });
        Ok(())
    }

    #[test]
    fn preauth_key_from_str_message() -> Result<(), Box<dyn std::error::Error>> {
        "  preauth_ok  ".parse::<ClientPreauthKey>()?;
        assert_eq!(
            "".parse::<ClientPreauthKey>(),
            Err(EmptyClientPreauthKeyError)
        );
        assert_eq!(
            EmptyClientPreauthKeyError.to_string(),
            "pre-auth key must not be empty"
        );
        Ok(())
    }

    /// Any 32 bytes are a valid Ed25519 seed.
    const PRIVATE_KEY: &str = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";

    #[test]
    fn private_key_normalization_and_redaction() -> Result<(), Box<dyn std::error::Error>> {
        let key = PrivateKey::try_new(format!("  {PRIVATE_KEY}\n"))?;
        assert_eq!(key.as_ref(), PRIVATE_KEY, "surrounding whitespace trimmed");
        assert_eq!(
            PrivateKey::try_new(PRIVATE_KEY.to_uppercase())?,
            key,
            "hex is case-insensitive, so the same key compares equal however it \
             was written"
        );
        assert_eq!(
            format!("{key:?}"),
            "PrivateKey(<redacted>)",
            "Debug must not leak the key a `ClientIdentity` dump would carry"
        );
        assert_eq!(
            format!("{key}"),
            "<redacted>",
            "Display must not leak it either — `{{}}` has to be safe so no caller \
             reaches for as_ref() to satisfy it"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::not_hex("zz11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899x")]
    #[case::odd_length("aab")]
    #[case::too_short("aabb")]
    #[case::expanded_keypair(concat!(
        "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899",
        "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899"
    ))]
    #[case::empty("")]
    fn private_key_refuses_what_cannot_sign(#[case] raw: &str) {
        assert!(
            PrivateKey::try_new(raw).is_err(),
            "{raw:?} should be refused"
        );
    }

    /// The identity token `auth export` writes is JSON of the bare hex.
    #[test]
    fn private_key_serializes_as_the_bare_hex_and_validates_on_the_way_back()
    -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&PrivateKey::try_new(PRIVATE_KEY)?)?;
        assert_eq!(json, format!("\"{PRIVATE_KEY}\""));
        assert_eq!(
            serde_json::from_str::<PrivateKey>(&json)?.as_ref(),
            PRIVATE_KEY
        );
        assert!(serde_json::from_str::<PrivateKey>("\"aabb\"").is_err());
        Ok(())
    }

    #[rstest::rstest]
    #[case("https://mgmt.example.com", "https://mgmt.example.com")]
    #[case("https://mgmt.example.com/", "https://mgmt.example.com")]
    #[case("  https://mgmt.example.com/\n", "https://mgmt.example.com")]
    #[case("https://mgmt.example.com/api/", "https://mgmt.example.com/api")]
    #[case("http://127.0.0.1:8080", "http://127.0.0.1:8080")]
    fn server_url_normalizes_to_a_joinable_base(
        #[case] raw: &str,
        #[case] expected: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let server = ServerUrl::try_new(raw)?;
        assert_eq!(server.to_string(), expected);
        assert_eq!(
            format!("{server}/clients/me"),
            format!("{expected}/clients/me")
        );
        Ok(())
    }

    /// Each message names the URL, for an operator running several deployments.
    #[rstest::rstest]
    #[case::bare_host("mgmt.example.com", "`mgmt.example.com`: relative URL without a base")]
    #[case::query(
        "https://mgmt.example.com?tenant=a",
        "`https://mgmt.example.com/?tenant=a` must not carry a query or fragment"
    )]
    #[case::fragment(
        "https://mgmt.example.com#section",
        "`https://mgmt.example.com/#section` must not carry a query or fragment"
    )]
    #[case::mailto(
        "mailto:ops@example.com",
        "`mailto:ops@example.com` must use http or https"
    )]
    #[case::file("file:///tmp/mgmt", "`file:///tmp/mgmt` must use http or https")]
    fn server_url_rejects_what_is_not_a_joinable_http_base(
        #[case] raw: &str,
        #[case] expected: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let error = raw
            .parse::<ServerUrl>()
            .err()
            .ok_or("expected a rejection")?
            .to_string();
        assert!(error.contains(expected), "unexpected rejection: {error}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::absent(
        r#"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved"}"#,
        &[],
        false,
        &[],
        None
    )]
    #[case::present(
        r#"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved","tags":["team/mobile/ios","batch/2026-q3"],"reindex_pending":true,"capabilities":["runtime:llama_cpp"],"permissions":["perform_work","submit_results"]}"#,
        &["team/mobile/ios", "batch/2026-q3"],
        true,
        &["runtime:llama_cpp"],
        Some(&["perform_work", "submit_results"][..])
    )]
    // "Holds none" is an empty array, distinct from a field never sent.
    #[case::no_permissions(
        r#"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved","permissions":[]}"#,
        &[],
        false,
        &[],
        Some(&[][..])
    )]
    fn client_profile_forward_compat(
        #[case] body: &str,
        #[case] expected_tags: &[&str],
        #[case] expected_reindex: bool,
        #[case] expected_caps: &[&str],
        #[case] expected_permissions: Option<&[&str]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let profile: ClientProfile = serde_json::from_str(body)?;
        let expected_tags: Vec<String> = expected_tags.iter().map(|s| s.to_string()).collect();
        let expected_caps: Vec<String> = expected_caps.iter().map(|s| s.to_string()).collect();
        let expected_permissions: Option<Vec<String>> =
            expected_permissions.map(|p| p.iter().map(|s| s.to_string()).collect());
        assert_eq!(profile.tags, expected_tags);
        assert_eq!(profile.reindex_pending, expected_reindex);
        assert_eq!(profile.capabilities, expected_caps);
        assert_eq!(profile.permissions, expected_permissions);
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
                "repository_url": "github.com/ggml-org/llama.cpp",
                "version": "b5000",
                "flavor": "macos-arm64"
            }
        },
        "future_field": 42
    }"#;

    #[test]
    fn claimed_job_tolerates_server_labels_and_unknown_fields()
    -> Result<(), Box<dyn std::error::Error>> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_BODY)?;
        assert_eq!(job.job_id, "job-1");
        assert_eq!(job.time_window, "PT10M");
        assert_eq!(job.client_versions, None);
        Ok(())
    }

    #[rstest::rstest]
    #[case::two(Some(serde_json::json!(["2.0", "3.0"])), Some(vec!["2.0", "3.0"]))]
    #[case::none_admitted(Some(serde_json::json!([])), Some(vec![]))]
    #[case::absent(None, None)]
    #[case::null(Some(serde_json::Value::Null), None)]
    #[case::not_an_array(Some(serde_json::json!("2.0")), Some(vec![]))]
    #[case::not_all_strings(Some(serde_json::json!(["2.0", 3])), Some(vec![]))]
    fn client_versions_decode_leniently(
        #[case] value: Option<serde_json::Value>,
        #[case] expected: Option<Vec<&str>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut body: serde_json::Value = serde_json::from_str(CLAIM_BODY)?;
        let obj = body.as_object_mut().ok_or("the fixture is an object")?;
        if let Some(value) = value {
            obj.insert("client_versions".into(), value);
        }
        let job: ClaimedJob = serde_json::from_value(body)?;
        assert_eq!(
            job.client_versions,
            expected.map(|versions| versions.into_iter().map(str::to_owned).collect())
        );
        assert_eq!(job.job_id, "job-1");
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

    /// A body built without the streams carries neither key.
    #[test]
    fn a_failure_flattens_its_runtime_output() -> Result<(), Box<dyn std::error::Error>> {
        let job: ClaimedJob = serde_json::from_str(CLAIM_BODY)?;
        let bare = FailureSubmission::from_claim(&job, "[ts] oom", true, "9.9.9");
        let wire = serde_json::to_value(&bare)?;
        assert!(wire.get("runtime_stdout").is_none() && wire.get("runtime_stderr").is_none());
        assert!(wire.get("device_time_ms").is_none(), "untimed: {wire}");
        assert!(
            wire.get("failure_code").is_none(),
            "mgmt refuses a null: {wire}"
        );

        let coded = FailureSubmission {
            failure_code: Some(FailureCode::AttemptTimeout),
            ..bare.clone()
        };
        assert_eq!(
            serde_json::to_value(&coded)?["failure_code"],
            "attempt_timeout"
        );

        let with_output = FailureSubmission {
            runtime_output: RuntimeOutput {
                runtime_stdout: Some(String::new()),
                runtime_stderr: Some("abort\n".into()),
            },
            ..bare
        };
        let wire = serde_json::to_value(&with_output)?;
        assert_eq!(wire["runtime_stdout"], "");
        assert_eq!(wire["runtime_stderr"], "abort\n");
        assert!(wire.get("runtime_output").is_none(), "must flatten: {wire}");
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
        assert!(value.get("device_cpu_arch").is_none());
        assert!(value.get("device_os_family").is_none());
        Ok(())
    }

    /// The field names match `DeviceProfileFields` on the management server
    /// (mgmt). The server derives the `arch:` and `os_family:` flags from these
    /// fields.
    #[test]
    fn update_client_request_names_the_arch_and_os_family() -> Result<(), Box<dyn std::error::Error>>
    {
        let req = UpdateClientRequest {
            client_details: None,
            device: DeviceProfileFields {
                device_cpu_arch: Some("x86_64".into()),
                device_os_family: Some("linux".into()),
                ..Default::default()
            },
        };
        let value = serde_json::to_value(&req)?;
        assert_eq!(value["device_cpu_arch"], "x86_64");
        assert_eq!(value["device_os_family"], "linux");
        Ok(())
    }
}
