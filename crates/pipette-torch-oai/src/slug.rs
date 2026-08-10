//! The uv runtime slug (`<server>@<version>`) and the version composition
//! that feeds it.
//!
//! Leaf module: nothing here reaches back into the crate, so the slug
//! grammar can be reasoned about (and tested) in isolation.

use std::fmt;

use anyhow::Context;

use pipette_plan_types as plan_types;
use pipette_plan_types::{
    NonEmptyString, SemverWithBuild, UvBuild, UvPythonVersion, UvServerVersion,
};

pub const VLLM_SERVER_LABEL: &str = "vllm";
pub const SGLANG_SERVER_LABEL: &str = "sglang";

pub fn uv_runtime_version(
    server_version: &UvServerVersion,
    build: &UvBuild,
    python_version: &UvPythonVersion,
) -> anyhow::Result<SemverWithBuild> {
    SemverWithBuild::try_new(plan_types::uv_runtime_version(
        server_version,
        build,
        python_version,
    ))
    .context("validated uv runtime component fields should compose into SemverWithBuild")
}

/// A uv runtime slug.
///
/// The slug is the lookup key into the bundled catalog. The public
/// install CLI receives `(server, version)` as typed fields and constructs
/// the slug as `<server>@<version>`, so the slug's name part is limited
/// to `vllm` or `sglang`. Fork/vendor distinctions belong in the
/// [`SemverWithBuild`] value, usually as build metadata.
///
/// Filesystem constraints layered on top of the grammar: no `/`
/// (collides with the runtime-dir path separator), no NUL/control
/// chars (mangles env-marker readback in the orphan reaper), no
/// leading `.` (avoids hidden / dotted dir names). The grammar
/// already excludes `/` and control chars, so those checks are
/// redundant — kept for the precise error message they emit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UvSlug(NonEmptyString);

/// Allowed characters in a slug body. `[A-Za-z0-9._@+-]` covers the
/// server label, semver-ish versions with `+build` metadata, and the
/// `@` separator. Rejects whitespace, NUL, path separators, and shell
/// metacharacters in one regex-free check.
fn slug_body_chars_ok(body: &str) -> bool {
    !body.is_empty()
        && body.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' || c == '@' || c == '+'
        })
}

/// Validate a slug body decomposes into `<server>@<version>`.
fn validate_slug_grammar(body: &str) -> anyhow::Result<()> {
    let (name, version) = body.split_once('@').with_context(|| {
        format!(
            "uv slug '{body}' must follow grammar '<server>@<version>' \
             (missing '@'); examples: 'vllm@0.21.0+cu121.py3.12', 'sglang@0.5.12.post1+rocm6.py3.12'"
        )
    })?;
    if name != VLLM_SERVER_LABEL && name != SGLANG_SERVER_LABEL {
        anyhow::bail!("uv slug '{body}' must start with 'vllm@' or 'sglang@'");
    }
    if version.contains('@') {
        anyhow::bail!("uv slug '{body}' must contain exactly one '@'");
    }
    SemverWithBuild::try_new(version.to_string())
        .with_context(|| format!("uv slug '{body}' has invalid version '{version}'"))?;
    Ok(())
}

impl UvSlug {
    /// Validate and wrap a slug body. Enforces the `<server>@<version>`
    /// grammar plus the filesystem-safety rules; rejects everything else
    /// with a clear error.
    pub fn try_new(body: &str) -> anyhow::Result<Self> {
        if body.is_empty() {
            anyhow::bail!("uv slug must not be empty");
        }
        if body.starts_with('.') {
            anyhow::bail!(
                "uv slug '{body}' must not start with '.' (reserved for hidden / dotted dir names)"
            );
        }
        // Body-level character check first so control chars / `/` /
        // whitespace surface with a uniform message before grammar
        // decomposition.
        if !slug_body_chars_ok(body) {
            anyhow::bail!(
                "uv slug '{body}' may only contain [A-Za-z0-9._@+-] \
                 (no whitespace, slashes, or shell metacharacters)"
            );
        }
        // Verify the body decomposes into the expected three parts —
        // we don't keep the parts, just confirm the shape.
        validate_slug_grammar(body)?;
        let inner = NonEmptyString::try_new(body.to_string())?;
        Ok(Self(inner))
    }

    fn from_label(server_label: &str, version: &SemverWithBuild) -> anyhow::Result<Self> {
        let body = format!("{server_label}@{version}");
        let inner = NonEmptyString::try_new(body)?;
        Ok(Self(inner))
    }

    /// Construct a vLLM uv slug from the typed version.
    pub fn vllm(version: &SemverWithBuild) -> anyhow::Result<Self> {
        Self::from_label(VLLM_SERVER_LABEL, version)
    }

    /// Construct an SGLang uv slug from the typed version.
    pub fn sglang(version: &SemverWithBuild) -> anyhow::Result<Self> {
        Self::from_label(SGLANG_SERVER_LABEL, version)
    }

    /// The slug body as a string slice. Same bytes
    /// [`Self::try_new`] received or the typed slug constructors
    /// built.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl fmt::Display for UvSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    fn uv_server_version(version: &str) -> anyhow::Result<UvServerVersion> {
        Ok(UvServerVersion::try_new(version.to_string())?)
    }

    fn uv_build(build: &str) -> anyhow::Result<UvBuild> {
        Ok(UvBuild::try_new(build.to_string())?)
    }

    fn uv_python_version(version: &str) -> anyhow::Result<UvPythonVersion> {
        Ok(UvPythonVersion::try_new(version.to_string())?)
    }

    #[test]
    fn uv_runtime_version_composes_components() -> anyhow::Result<()> {
        let version = uv_runtime_version(
            &uv_server_version("0.21.0")?,
            &uv_build("cu121")?,
            &uv_python_version("3.11")?,
        )?;
        assert_eq!(version.as_ref(), "0.21.0+cu121.py3.11");
        Ok(())
    }

    #[test]
    fn uv_runtime_version_rejects_bad_build_component() {
        assert!(uv_build("xpu").is_err());
    }

    #[test]
    fn uv_runtime_version_rejects_build_in_server_version_component() {
        assert!(uv_server_version("0.21.0+cu121").is_err());
    }

    #[test]
    fn uv_slug_parse_accepts_typical_slug() -> anyhow::Result<()> {
        let s = UvSlug::try_new("vllm@0.21.0+cu121.py3.12")?;
        assert_eq!(s.as_str(), "vllm@0.21.0+cu121.py3.12");
        Ok(())
    }

    #[test]
    fn uv_slug_parse_rejects_grammar_violations() -> anyhow::Result<()> {
        for (body, expected_marker) in [
            ("vllm-nightly-rocm6", "missing '@'"),
            ("my-vllm@0.21.0", "vllm@"),
            ("vllm+cu121", "missing '@'"),
            ("vllm@+cu121", "invalid version"),
        ] {
            let err = UvSlug::try_new(body).err().context("expected an error")?;
            let msg = format!("{err:#}");
            assert!(
                msg.contains(expected_marker),
                "slug '{body}' should error with '{expected_marker}', got: {msg}"
            );
        }
        Ok(())
    }

    #[test]
    fn uv_slug_typed_constructors_round_trip() -> anyhow::Result<()> {
        let version = SemverWithBuild::try_new("0.21.0+cu121.py3.12".to_string())?;
        let slug = UvSlug::vllm(&version)?;
        assert_eq!(slug.as_str(), "vllm@0.21.0+cu121.py3.12");

        let version = SemverWithBuild::try_new("0.5.12.post1+rocm6.py3.12".to_string())?;
        let slug = UvSlug::sglang(&version)?;
        assert_eq!(slug.as_str(), "sglang@0.5.12.post1+rocm6.py3.12");
        Ok(())
    }

    #[test]
    fn uv_slug_parse_rejects_invalid_chars_in_parts() {
        // Whitespace and shell metacharacters are out per the
        // [A-Za-z0-9._-] grammar in each of the three parts.
        for body in [
            "vllm @0.10.0+cu121", // whitespace in name
            "vllm@0.10 0+cu121",  // whitespace in version
            "vllm@0.21.0+cu 121", // whitespace in flavor
            "vllm@0.21.0+cu;rm",  // shell metachar in flavor
        ] {
            assert!(
                UvSlug::try_new(body).is_err(),
                "'{body}' should be rejected"
            );
        }
    }

    #[test]
    fn uv_slug_parse_rejects_empty() {
        assert!(UvSlug::try_new("").is_err());
    }

    #[test]
    fn uv_slug_parse_rejects_slash() -> anyhow::Result<()> {
        // `/` is forbidden because the slug becomes a filename stem under
        // `runtimes/`; a `/` would create unintended subdirs.
        let err = UvSlug::try_new("vllm/foo")
            .err()
            .context("expected an error")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("/"), "got {msg}");
        Ok(())
    }

    #[test]
    fn uv_slug_parse_rejects_control_chars() {
        // Control chars in the slug would mangle env-marker readback in the
        // orphan reaper (`PIPETTE_RUNTIME_REF=<slug>` is parsed back via
        // `\0`-split — embedded `\0` / `\n` would corrupt the read).
        assert!(UvSlug::try_new("vllm\0foo").is_err());
        assert!(UvSlug::try_new("vllm\nfoo").is_err());
        assert!(UvSlug::try_new("vllm\tfoo").is_err());
    }

    #[test]
    fn uv_slug_parse_rejects_leading_dot() {
        // Avoid hidden / dotted filenames under `runtimes/` and the
        // pathological `.` / `..` cases that would walk above the storage
        // root.
        assert!(UvSlug::try_new(".hidden").is_err());
        assert!(UvSlug::try_new(".").is_err());
        assert!(UvSlug::try_new("..").is_err());
    }
}
