//! Shared leaf/value newtypes and containers.
//!
//! Every string-shaped field in this crate goes through a `nutype`
//! wrapper so invalid plans are rejected at TOML deserialization time
//! rather than caught later by an explicit `validate()` pass.
//!
//! Re-exported flat from `lib.rs`, so consumers reference these as
//! `pipette_plan_types::ResourceUrl` etc. without seeing the submodule.

use nutype::nutype;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[nutype(
    validate(regex = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct HfOrg(String);

#[nutype(
    validate(regex = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct HfRepoName(String);

/// A path relative to a repo root: `<seg>/…/<seg>`. No leading slash and no
/// `.`/`..` segments, so it can't be absolute or escape the repo — but it may
/// be nested. The segment-safety basis for [`RepoSubpath`].
fn is_relative_repo_subpath(path: &str) -> bool {
    path.split('/').all(|seg| {
        !seg.is_empty()
            && seg != "."
            && seg != ".."
            && seg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    })
}

/// A path relative to an HF repo root — a file (e.g. `Q4_K_M.gguf`) or a
/// subdirectory (e.g. `4bit`, for a repo that bundles several models). No
/// leading slash and no `.`/`..` segments, so it can't be absolute or escape
/// the repo; the artifact format (and thus any extension) is the enclosing
/// model variant's concern, not this path's.
#[nutype(
    validate(predicate = is_relative_repo_subpath),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct RepoSubpath(String);

#[nutype(
    validate(not_empty),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct ClientId(String);

#[nutype(
    validate(not_empty),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct NonEmptyString(String);

/// Version string with optional semver-style build metadata after `+`.
///
/// Accepts the catalog forms used across the workspace:
/// - bare semver-ish: `"0.10.0"`, `"0.31"`, `"b9050"`
/// - with build metadata: `"0.10.0+cu121.py3.12"`, `"0.10.0+rocm6.py3.12"`, `"0.10.0+cpu.py3.12"`
/// - fork-distinguished build metadata: `"0.10.0+cu121-myfork"`
///
/// Used for composed uv runtime identities such as
/// `<server_version>+<build>.py<python_version>`. The uv runtime spec
/// stores those component fields separately and derives this value when
/// it needs a `uv://name@version` wire form. Other runtime kinds
/// (`LlamacppCliStockTools`, `MlxMacosPipette`) keep `NonEmptyString` versions today.
///
/// Character set matches torch-oai's `UvSlug` grammar (`[A-Za-z0-9._-]`)
/// per segment, with an optional single `+` separator. Empty segments
/// on either side of `+` are rejected.
#[nutype(
    validate(regex = r"^[A-Za-z0-9._-]+(\+[A-Za-z0-9._-]+)?$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct SemverWithBuild(String);

/// Server/package version for a uv-hosted OpenAI-compatible server.
/// This is the package version only, without build metadata.
#[nutype(
    validate(regex = r"^[A-Za-z0-9._-]+$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct UvServerVersion(String);

/// Build selector for uv-hosted server wheels. This doubles as the
/// source for hardware flavor derivation in torch-oai.
#[nutype(
    validate(regex = r"^(cpu|cu[A-Za-z0-9_-]*|rocm[A-Za-z0-9_-]*)$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct UvBuild(String);

/// Python selector passed to `uv venv --python` for uv-hosted runtimes.
/// Kept path-free because it is embedded in the derived runtime version.
#[nutype(
    validate(regex = r"^[0-9][A-Za-z0-9._-]*$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct UvPythonVersion(String);

/// Crate-level error covering every fallible operation in
/// `pipette-plan-types` that we own. The `nutype`-generated per-type
/// errors (`HfOrgError`, `HfRepoNameError`, etc.) compose in via
/// `#[from]`, so a single `?` from any of them lands here.
///
/// Callers typically wrap with `anyhow::Context` for call-site
/// context; the `Display` of each variant is intentionally short
/// since the source chain carries the structural detail.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    #[error("must not be empty")]
    NonEmptyVec,
    #[error("HF repo must be in the form `org/repo_name`")]
    HfRepoMissingSeparator,
    #[error("invalid HF org: {0}")]
    HfOrg(#[from] HfOrgError),
    #[error("invalid HF repo name: {0}")]
    HfRepoName(#[from] HfRepoNameError),
}

/// `Vec<T>` guaranteed non-empty at construction, serialization, and
/// deserialization.  The only way to obtain one is `try_new` (fallible)
/// or deserialization from a non-empty sequence — `Variant { models:
/// vec![], … }` will not compile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NonEmptyVec<T>(Vec<T>);

impl<T> NonEmptyVec<T> {
    pub fn try_new(items: Vec<T>) -> anyhow::Result<Self, Error> {
        if items.is_empty() {
            Err(Error::NonEmptyVec)
        } else {
            Ok(Self(items))
        }
    }

    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T> AsRef<[T]> for NonEmptyVec<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T> std::ops::Deref for NonEmptyVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Serialize> Serialize for NonEmptyVec<T> {
    fn serialize<S>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for NonEmptyVec<T> {
    fn deserialize<D>(deserializer: D) -> anyhow::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = Vec::<T>::deserialize(deserializer)?;
        Self::try_new(v).map_err(serde::de::Error::custom)
    }
}

/// A HuggingFace repo revision: a git tag, branch, or commit SHA. Left
/// unvalidated beyond "a git-ref-shaped token" — HF resolves it, and the
/// set of valid refs changes per repo. SweepPins a repo to a reproducible
/// snapshot; the model-side analog of a runtime's `repository_version`.
#[nutype(
    validate(regex = r"^[A-Za-z0-9][A-Za-z0-9._/-]*$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct HfRevision(String);

/// A HuggingFace access token for a gated/private repo, carried in the plan
/// (not read from the driver env).
///
/// Secret, per the rules in `docs/architecture.md` (“Secrets”): both `Debug`
/// and `Display` are hand-written to render `<redacted>`, so neither `{:?}` on
/// a struct that holds one nor `{}` on the token itself can publish it.
/// [`AsRef`] is the one door to the raw value — auditing where this token can
/// escape is a search for `as_ref` call sites.
#[nutype(
    validate(not_empty),
    derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef)
)]
pub struct AuthToken(String);

impl std::fmt::Debug for AuthToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AuthToken(<redacted>)")
    }
}

impl std::fmt::Display for AuthToken {
    /// Present rather than omitted so `{token}` compiles to something safe: an
    /// author who needs `{}` is never pushed to the raw-value accessor to get it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// The one environment variable the CLI consults for an HF access token. When
/// set, it is injected into a gated, tokenless model definition at the run
/// boundary (see [`inject_hf_auth_token`](crate::inject_hf_auth_token)); the plan runner forwards it
/// under this name. Deliberately pipette-namespaced — the generic `HF_TOKEN` /
/// `HUGGING_FACE_HUB_TOKEN` are not consulted, so a token only ever enters via
/// the model spec.
pub const HF_TOKEN_ENV: &str = "PIPETTE_HF_TOKEN";

/// HuggingFace repo coordinate.  Flattened into every `Model` variant
/// so TOML stays `{ type = "...", org = "...", repo_name = "...", ... }`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct HfRepo {
    pub org: HfOrg,
    pub repo_name: HfRepoName,
    /// Optional pin to a specific repo snapshot (tag/branch/commit). `None`
    /// resolves to the repo's default branch (latest, mutable) — so a plan
    /// that wants reproducible weights should set it. Part of the repo's
    /// identity: it feeds [`HfRepo::reference`] and therefore the warehouse
    /// key, so two cells pinning different revisions don't collide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<HfRevision>,
    /// Access token for a gated/private repo, carried in the plan. `None` for
    /// public repos. When set, the runner forwards it to workers as
    /// [`HF_TOKEN_ENV`], which the CLI injects back into this field.
    /// Not part of the repo's identity (excluded from `reference`/`Display`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<AuthToken>,
}

impl HfRepo {
    /// Parse an `org/repo_name` string. Canonical parser across the
    /// workspace; callers should prefer this to writing their own
    /// split-and-validate logic. Errors land in the crate-level
    /// [`enum@Error`] type via the `HfRepoMissingSeparator`/`HfOrg`/
    /// `HfRepoName` variants.
    pub fn parse_org_repo(org_repo: &str) -> anyhow::Result<Self, Error> {
        let (org, name) = org_repo
            .split_once('/')
            .ok_or(Error::HfRepoMissingSeparator)?;
        Ok(Self {
            org: HfOrg::try_new(org)?,
            repo_name: HfRepoName::try_new(name)?,
            revision: None,
            auth_token: None,
        })
    }

    /// Identity form: `org/repo_name`, or `org/repo_name@revision` when a
    /// revision is pinned. Distinct from [`Display`](std::fmt::Display), which stays
    /// the bare `org/repo_name` the HF client wants (revision is passed to it
    /// separately, e.g. `--revision`). Use this wherever the string is an
    /// *identity* — model references, warehouse keys, logs.
    pub fn reference(&self) -> String {
        match &self.revision {
            Some(rev) => format!("{self}@{rev}"),
            None => self.to_string(),
        }
    }
}

impl std::fmt::Display for HfRepo {
    /// Canonical `org/repo_name` form. Matches HuggingFace's
    /// own URL/CLI convention and what mlx-lm / HF Hub clients accept
    /// as a single string identifier. Deliberately excludes the revision —
    /// that's an identity concern, see [`HfRepo::reference`].
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.org, self.repo_name)
    }
}

/// A URL locating a fetchable resource — a file *or* a directory:
/// `https://…`/`http://…` for a remote download, `file://…` for a local path.
/// Local is just the `file://` scheme, not a separate type.
///
/// For llama.cpp remote archives, use [`RemoteArchiveUrl`] (scheme-less).
#[nutype(
    validate(regex = r"^(https?|file)://.+"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct ResourceUrl(String);

/// Strip a leading `http://` or `https://` so remote-archive coordinates are
/// stored scheme-less (same idea as [`RepositoryUrl`]). Other schemes are left
/// intact so validation can reject them.
fn strip_http_scheme(raw: String) -> String {
    let s = raw.trim();
    ["https://", "http://"]
        .into_iter()
        .find_map(|scheme| s.strip_prefix(scheme))
        .map(str::to_owned)
        .unwrap_or_else(|| s.to_owned())
}

/// Host/path form of a remote prebuilt archive, **without** a URL scheme.
///
/// Construction accepts a bare `host/path/…` or an `http(s)://…` paste (scheme
/// is stripped). `file://` and any other remaining `scheme://` are rejected.
/// Requires at least one `/` (host + path). Fetchers download with `https://`
/// prepended.
fn is_remote_archive_url(s: &str) -> bool {
    !s.is_empty()
        && !s.contains("://")
        && !s.starts_with('/')
        && !s.starts_with('.')
        && s.contains('/')
        && !s.chars().any(char::is_whitespace)
}

/// Scheme-less remote archive coordinate: `example.com/path/a.tar.gz`.
/// See `strip_http_scheme` / `is_remote_archive_url`.
#[nutype(
    sanitize(with = strip_http_scheme),
    validate(predicate = is_remote_archive_url),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct RemoteArchiveUrl(String);

impl RemoteArchiveUrl {
    /// Absolute download URL used by fetchers (`https://` + coordinate).
    pub fn download_url(&self) -> String {
        format!("https://{}", self.as_ref())
    }
}

/// Sanitizer: drop a redundant leading `./` so the stored path is normalized.
fn strip_leading_current_dir(raw: String) -> String {
    raw.strip_prefix("./").map(str::to_owned).unwrap_or(raw)
}

/// Portable store/entry-relative path only. No absolute Unix/Windows forms,
/// no `~`, no `.`/`..` segments. Leading `./` is stripped by the sanitizer first.
fn is_relative_fs_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('~') || path.starts_with('/') {
        return false;
    }
    // Windows drive absolute: `C:\…` or `C:/…`.
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return false;
    }
    // UNC `\\server\share` or `//server/share`.
    if path.starts_with("\\\\") || path.starts_with("//") {
        return false;
    }
    path.split(['/', '\\'])
        .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
}

/// Absolute host path only (Unix `/…`, Windows `C:\…` / `C:/…`, optional UNC).
/// No `~`, no `.`/`..` segments. Leading `./` is stripped (then rejected if not abs).
fn is_absolute_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('~') {
        return false;
    }
    let bytes = path.as_bytes();
    let is_abs = path.starts_with('/')
        || path.starts_with("\\\\")
        || path.starts_with("//")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'/' || bytes[2] == b'\\'));
    if !is_abs {
        return false;
    }
    // Normalize check on body after the absolute prefix (UNC before bare `/`).
    let body = if let Some(rest) = path
        .strip_prefix("\\\\")
        .or_else(|| path.strip_prefix("//"))
    {
        if !rest.split(['/', '\\']).take(2).all(|s| !s.is_empty()) {
            return false;
        }
        rest
    } else if let Some(rest) = path.strip_prefix('/') {
        rest
    } else {
        // C:\rest or C:/rest
        &path[3..]
    };
    body.is_empty()
        || body
            .split(['/', '\\'])
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
}

/// Portable relative filesystem path (store / entry layout). Never absolute.
/// Host-absolute paths use [`AbsolutePath`].
#[nutype(
    sanitize(with = strip_leading_current_dir),
    validate(predicate = is_relative_fs_path),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct RelativePath(String);

/// Absolute host filesystem path (Unix `/…`, Windows `C:\…` / `C:/…`, UNC).
/// Runnable paths after ensure/bind — not portable store coordinates.
#[nutype(
    sanitize(with = strip_leading_current_dir),
    validate(predicate = is_absolute_path),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct AbsolutePath(String);

/// A file's SHA-256 content hash (64 lowercase hex chars). Source-agnostic
/// identity + integrity check for a gguf file — the same bytes regardless of
/// where they were fetched from.
#[nutype(
    validate(regex = r"^[0-9a-f]{64}$"),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct Sha256(String);

/// An id with no catalog side glued to the front and no whitespace. The
/// separator basis for [`BenchmarkId`].
fn is_bare_benchmark_id(id: &str) -> bool {
    !id.contains('/') && !id.chars().any(char::is_whitespace)
}

/// Opaque benchmark identifier as written in plan TOMLs.
///
/// An id, never an address: a catalog side (`local/…`, `remote/…`) is a
/// client-local way of *finding* a definition, and plans distribute ids alone —
/// so a `/` is rejected rather than swallowed into the id, which is how
/// `benchmarks = ["eval_smoke"]` used to parse. Whitespace goes with it;
/// beyond those the alphabet stays open, because the ids are the server's to
/// mint and a stricter rule here would make a future one unusable client-side.
#[nutype(
    validate(not_empty, predicate = is_bare_benchmark_id),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct BenchmarkId(String);

/// A capability flag in **canonical form**: non-empty, no whitespace, and no
/// uppercase (`to_lowercase()` is a no-op). These are the eligibility tokens a
/// scheduler-mode variant lists in `requires` — e.g. `os:ios`,
/// `os_version:26.1`, `runtime:llama_cpp:b9999`, `ram_bytes:17179869184`, or a
/// free-form `job_retry`.
///
/// The server normalizes each device attribute into this same canonical form
/// (lowercasing, stripping whitespace) before matching, so a plan author must
/// write flags the way they will be compared — enforced here at deserialization
/// rather than in a later validation pass. The character set is intentionally
/// open beyond the case/whitespace rule: reserved-namespace structure
/// (`os:`, `device:`, …) is the server's concern (checked at ingestion), and
/// free-form flags may use any non-whitespace, lowercase token.
#[nutype(
    validate(predicate = is_canonical_capability_flag),
    derive(
        Debug,
        Clone,
        PartialEq,
        Eq,
        Hash,
        Serialize,
        Deserialize,
        AsRef,
        Display
    )
)]
pub struct CapabilityFlag(String);

/// Canonical form for a [`CapabilityFlag`]: non-empty, contains no whitespace,
/// and is already lowercase (equal to its own `to_lowercase()`). Kept as a
/// predicate rather than a regex so the "already lowercase" rule covers
/// non-ASCII the same way the server's normalization does.
fn is_canonical_capability_flag(s: &str) -> bool {
    !s.is_empty() && !s.chars().any(char::is_whitespace) && s.to_lowercase() == s
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    /// Both rendering traits hide the value, so a token can only escape through
    /// `as_ref` — the property every `Debug`-dumped struct that holds one relies
    /// on (see `docs/architecture.md`, “Secrets”).
    #[test]
    fn auth_token_renders_redacted_in_both_traits() -> anyhow::Result<()> {
        const SECRET: &str = "hf_tokenthatmustnotescape";
        let token = AuthToken::try_new(SECRET.to_owned())?;
        assert_eq!(format!("{token:?}"), "AuthToken(<redacted>)");
        assert_eq!(format!("{token}"), "<redacted>");
        assert_eq!(token.as_ref(), SECRET, "as_ref stays the one way in");
        Ok(())
    }

    #[rstest::rstest]
    #[case("meta-llama", true)]
    #[case("LiquidAI", true)]
    #[case("a", true)]
    #[case("x_y.z-0", true)]
    #[case("0digits-can-start", true)]
    #[case("with..double-dot", true)]
    #[case("", false)]
    #[case("bad org!", false)]
    #[case("-leading-dash", false)]
    #[case(".leading-dot", false)]
    #[case("_leading-underscore", false)]
    #[case("trailing space ", false)]
    fn hf_org_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(HfOrg::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn hf_repo_parse_accepts_canonical() -> anyhow::Result<()> {
        let parsed = HfRepo::parse_org_repo("mlx-community/LFM2-350M-4bit")?;
        assert_eq!(parsed.org.to_string(), "mlx-community");
        assert_eq!(parsed.repo_name.to_string(), "LFM2-350M-4bit");
        // Round-trip through Display.
        assert_eq!(parsed.to_string(), "mlx-community/LFM2-350M-4bit");
        Ok(())
    }

    #[test]
    fn hf_repo_parse_rejects_missing_separator() -> anyhow::Result<()> {
        let err = HfRepo::parse_org_repo("just-a-name")
            .err()
            .context("expected parse to reject missing separator")?;
        assert!(matches!(err, Error::HfRepoMissingSeparator), "got {err:?}");
        Ok(())
    }

    #[test]
    fn hf_repo_parse_reports_invalid_org() -> anyhow::Result<()> {
        // Leading dot violates the HfOrg regex.
        let err = HfRepo::parse_org_repo(".bad/repo")
            .err()
            .context("expected parse to reject invalid org")?;
        assert!(matches!(err, Error::HfOrg(_)), "got {err:?}");
        Ok(())
    }

    #[test]
    fn hf_repo_parse_reports_invalid_repo_name() -> anyhow::Result<()> {
        // Empty repo_name segment after the `/`.
        let err = HfRepo::parse_org_repo("good/")
            .err()
            .context("expected parse to reject invalid repo name")?;
        assert!(matches!(err, Error::HfRepoName(_)), "got {err:?}");
        Ok(())
    }

    #[rstest::rstest]
    #[case("main", true)]
    #[case("v1.0", true)]
    #[case("refs/pr/3", true)]
    #[case("a1b2c3d4e5f6", true)]
    #[case("", false)]
    #[case("-leading-dash", false)]
    #[case("has space", false)]
    fn hf_revision_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(HfRevision::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn hf_repo_reference_appends_revision_but_display_stays_bare() -> anyhow::Result<()> {
        let mut repo = HfRepo::parse_org_repo("mlx-community/LFM2-350M-4bit")?;
        // No revision: reference() and Display are the bare repo id.
        assert_eq!(repo.reference(), "mlx-community/LFM2-350M-4bit");
        assert_eq!(repo.to_string(), repo.reference());

        // Pinned revision: identity carries `@rev`, Display does not (that's
        // the id the HF client consumes; the revision goes to it separately).
        repo.revision = Some(HfRevision::try_new("v2".to_owned())?);
        assert_eq!(repo.reference(), "mlx-community/LFM2-350M-4bit@v2");
        assert_eq!(repo.to_string(), "mlx-community/LFM2-350M-4bit");
        Ok(())
    }

    #[rstest::rstest]
    #[case("https://example.com/a.gguf", true)]
    #[case("http://example.com/a.gguf", true)]
    #[case("file:///models/a.gguf", true)]
    #[case("ftp://example.com/a.gguf", false)] // unsupported scheme
    #[case("example.com/a.gguf", false)] // scheme-less
    #[case("", false)]
    fn resource_url_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(ResourceUrl::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[rstest::rstest]
    #[case("example.com/a.tar.gz", true)]
    #[case("https://example.com/a.tar.gz", true)] // scheme stripped
    #[case("http://example.com/a.tar.gz", true)] // scheme stripped
    #[case("file:///tmp/a.tar.gz", false)]
    #[case("ftp://example.com/a.tar.gz", false)]
    #[case("/absolute/path.tgz", false)]
    #[case("", false)]
    #[case("has space/a.tgz", false)]
    fn remote_archive_url_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(RemoteArchiveUrl::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn remote_archive_url_strips_http_scheme_and_downloads_via_https() -> anyhow::Result<()> {
        let url = RemoteArchiveUrl::try_new("https://cdn.example.com/llama.tar.gz".to_owned())?;
        assert_eq!(url.as_ref(), "cdn.example.com/llama.tar.gz");
        assert_eq!(url.download_url(), "https://cdn.example.com/llama.tar.gz");
        Ok(())
    }

    #[rstest::rstest]
    #[case("models/m", true)] // relative
    #[case("./local/m", true)] // leading ./ stripped
    #[case("entry/blobs/Q4.gguf", true)]
    #[case("/models/m", false)] // absolute unix
    #[case("C:/models/m", false)] // absolute windows
    #[case(r"C:\models\m", false)]
    #[case(r"\\server\share\m", false)] // UNC
    #[case("~/models/m", false)]
    #[case("../sibling/m", false)]
    #[case("a/../b", false)]
    #[case("a/./b", false)]
    #[case("", false)]
    fn relative_path_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(RelativePath::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn relative_path_strips_leading_dot_slash() -> anyhow::Result<()> {
        assert_eq!(
            RelativePath::try_new("./models/m".to_owned())?.as_ref(),
            "models/m"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case("/models/m", true)]
    #[case("C:/models/m", true)]
    #[case(r"C:\models\m", true)]
    #[case(r"\\server\share\m", true)]
    #[case("//server/share/m", true)]
    #[case("models/m", false)] // relative rejected
    #[case("./models/m", false)] // becomes relative after sanitize
    #[case("~/models/m", false)]
    #[case("/models/../x", false)]
    #[case("", false)]
    fn local_path_abs_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(AbsolutePath::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn local_path_keeps_unix_absolute() -> anyhow::Result<()> {
        assert_eq!(
            AbsolutePath::try_new("/models/m".to_owned())?.as_ref(),
            "/models/m"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case("4bit", true)] // a subdirectory
    #[case("variants/mlx-4bit", true)] // nested subdirectory
    #[case("Q4_K_M.gguf", true)] // a file — extension-agnostic
    #[case("dir/weights.gguf", true)] // nested file
    #[case("weights.safetensors", true)] // any extension, not just .gguf
    #[case("a.b_c-d/e", true)]
    #[case("", false)]
    #[case(r"win\weights.gguf", false)] // backslash not a valid segment char
    #[case("/abs", false)] // absolute — must be relative to the repo
    #[case("../escape", false)] // parent traversal
    #[case("dir/../seg", false)] // traversal mid-path
    #[case("dir//seg", false)] // empty segment
    #[case("dir/./seg", false)] // dot segment
    fn repo_subpath_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(RepoSubpath::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[rstest::rstest]
    #[case(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        true
    )]
    #[case(
        "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
        false
    )] // uppercase
    #[case("abc123", false)] // too short
    #[case(
        "g123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
        false
    )] // non-hex
    fn sha256_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(Sha256::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn non_empty_newtypes_reject_blank_strings() {
        assert!(NonEmptyString::try_new(String::new()).is_err());
        assert!(ClientId::try_new(String::new()).is_err());
        assert!(BenchmarkId::try_new(String::new()).is_err());
    }

    #[rstest::rstest]
    #[case("os:ios", true)]
    #[case("os_version:26.1", true)]
    #[case("runtime:llama_cpp:b9999", true)]
    #[case("ram_bytes:17179869184", true)]
    #[case("job_retry", true)] // free-form
    #[case("device:iphone17pro", true)]
    #[case("", false)] // empty
    #[case("os:iOS", false)] // uppercase
    #[case("OS:ios", false)] // uppercase in namespace
    #[case("os: ios", false)] // internal whitespace
    #[case("os:ios ", false)] // trailing whitespace
    #[case(" os:ios", false)] // leading whitespace
    #[case("os:i\tos", false)] // tab
    fn capability_flag_canonical_form(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(CapabilityFlag::try_new(s.to_owned()).is_ok(), ok, "{s:?}");
    }

    #[test]
    fn non_empty_vec_try_new_rejects_empty() {
        let empty: Vec<u32> = vec![];
        assert!(NonEmptyVec::try_new(empty).is_err());
        assert!(NonEmptyVec::try_new(vec![1]).is_ok());
    }

    #[rstest::rstest]
    #[case("0.10.0", true)] // bare semver
    #[case("0.31", true)] // two-part (in-use today)
    #[case("b9050", true)] // build label
    #[case("0.10.0+cu121.py3.12", true)] // build metadata
    #[case("0.10.0+rocm6", true)]
    #[case("0.10.0+cpu", true)]
    #[case("0.10.0+cu121-myfork", true)] // fork-distinguished build metadata
    #[case("a", true)] // single char
    #[case("", false)] // empty
    #[case("+cu121", false)] // empty pre-`+` segment
    #[case("0.10.0+", false)] // empty post-`+` segment
    #[case("0.10.0+a+b", false)] // multiple `+` separators
    #[case("weird/thing", false)] // path separator not in charset
    #[case(" 0.10.0", false)] // leading whitespace
    fn semver_with_build_validation(#[case] s: &str, #[case] ok: bool) {
        assert_eq!(SemverWithBuild::try_new(s).is_ok(), ok, "{s:?}");
    }
}
