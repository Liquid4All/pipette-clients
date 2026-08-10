//! Bundled uv requirements catalog.
//!
//! Reads `bundled-catalog/catalog.toml` (compiled into the binary
//! via `include_str!`) and exposes per-slug lookup plus access to the
//! bundled requirements bytes embedded in that TOML.
//!
//! The TOML file is per-variant arrays of tables; the slug is computed
//! at load time as
//! `<server>@<server_version>+<build>.py<python_version>`:
//!
//! ```toml
//! [[uv_vllm]]
//! server_version = "0.21.0"
//! build = "cu121"
//! python_version = "3.12"
//! requirements = """
//! --extra-index-url https://download.pytorch.org/whl/cu121
//! vllm==0.21.0
//! """
//!
//! [[uv_sglang]]
//! server_version = "0.5.12.post1"
//! build = "cpu"
//! python_version = "3.12"
//! requirements = """
//! --extra-index-url https://download.pytorch.org/whl/cpu
//! sglang==0.5.12.post1
//! """
//! ```
//!
//! Server identity comes from the table name (`uv_vllm` /
//! `uv_sglang`). The `build` suffix is folded into the canonical
//! runtime version and must identify a supported wheel flavor
//! (`cu*`, `rocm*`, or `cpu`); catalog load rejects malformed entries
//! loudly. The Python version is written into the runtime manifest,
//! passed to `uv venv --python`, and also folded into the canonical
//! runtime version as `.py<python_version>`.
//!
//! Single source of truth: the catalog TOML decides what's installable
//! from the bundle and carries the requirements bytes inline. There
//! are no per-slug bundled requirements files.
//!
//! Per design, **on-disk TOML is the source of truth** — the
//! `include_str!` pulls the file straight from
//! `crates/pipette-torch-oai/bundled-catalog/catalog.toml` at
//! compile time. Adding a new entry means appending a complete
//! `[[uv_vllm]]` / `[[uv_sglang]]` block; no Rust edits required.
//! See `crates/pipette-torch-oai/bundled-catalog/README.md`.

use std::{collections::BTreeMap, sync::OnceLock};

use anyhow::Context;
use serde::Deserialize;

use pipette_plan_types::{
    flavor_from_uv_build, NonEmptyString, Runtime, SemverWithBuild, UvBuild, UvPythonVersion,
    UvRuntimeSource, UvServerVersion, UvSglang, UvVllm, VllmFlavor,
};

use crate::slug::{uv_runtime_version, UvSlug, SGLANG_SERVER_LABEL, VLLM_SERVER_LABEL};

const CATALOG_TOML: &str = include_str!("../bundled-catalog/catalog.toml");

/// One row of the bundled catalog, computed at catalog-load time from
/// the table kind plus typed version. The enum variant is the runtime
/// server identity, so install callers never recover it from the slug
/// string. Flavor is derived from `build`; Python is read from
/// `python_version` and encoded into the canonical version. Requirements
/// are embedded in the same TOML row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogEntry {
    UvVllm {
        server_version: UvServerVersion,
        build: UvBuild,
        python_version: UvPythonVersion,
        requirements: NonEmptyString,
        install_flags: Vec<String>,
    },
    UvSglang {
        server_version: UvServerVersion,
        build: UvBuild,
        python_version: UvPythonVersion,
        requirements: NonEmptyString,
        install_flags: Vec<String>,
    },
}

impl CatalogEntry {
    /// Version carried by the catalog row.
    pub fn version(&self) -> anyhow::Result<SemverWithBuild> {
        match self {
            Self::UvVllm {
                server_version,
                build,
                python_version,
                ..
            }
            | Self::UvSglang {
                server_version,
                build,
                python_version,
                ..
            } => uv_runtime_version(server_version, build, python_version),
        }
    }

    /// True if this entry is a vLLM catalog row.
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::UvVllm { .. })
    }

    /// True if this entry is an SGLang catalog row.
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::UvSglang { .. })
    }

    /// Short label (`"vllm"` / `"sglang"`). Same wire form as the
    /// slug body's `<name>` part.
    pub fn server_label(&self) -> &'static str {
        match self {
            Self::UvVllm { .. } => VLLM_SERVER_LABEL,
            Self::UvSglang { .. } => SGLANG_SERVER_LABEL,
        }
    }

    /// Per-server flavor canonicalized as [`VllmFlavor`] via
    /// [`flavor_from_uv_build`].
    pub fn flavor(&self) -> VllmFlavor {
        match self {
            Self::UvVllm { build, .. } | Self::UvSglang { build, .. } => {
                flavor_from_uv_build(build)
            }
        }
    }

    /// Python interpreter pin declared by this catalog row.
    pub fn python_version(&self) -> &UvPythonVersion {
        match self {
            Self::UvVllm { python_version, .. } | Self::UvSglang { python_version, .. } => {
                python_version
            }
        }
    }

    /// Bundled requirements text embedded in `catalog.toml`.
    pub fn requirements(&self) -> &str {
        match self {
            Self::UvVllm { requirements, .. } | Self::UvSglang { requirements, .. } => {
                requirements.as_ref()
            }
        }
    }

    pub fn install_flags(&self) -> &[String] {
        match self {
            Self::UvVllm { install_flags, .. } | Self::UvSglang { install_flags, .. } => {
                install_flags
            }
        }
    }

    /// Plan [`Runtime`] for this row, as an ordinary [`UvRuntimeSource::PipRequirementsText`].
    ///
    /// The `vllm@…` / `sglang@…` slug that keys the rows stays a *lookup* key:
    /// it selects which body to use and then plays no further part, so a
    /// resolved row is indistinguishable from a hand-authored one.
    pub fn to_runtime(&self) -> anyhow::Result<Runtime> {
        let source = UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new(self.requirements().to_owned())?,
            install_flags: optional_install_flags(self.install_flags()),
        };
        Ok(match self {
            Self::UvVllm {
                server_version,
                build,
                python_version,
                ..
            } => Runtime::UvVllm(UvVllm {
                server_version: server_version.clone(),
                build: build.clone(),
                python_version: python_version.clone(),
                source,
            }),
            Self::UvSglang {
                server_version,
                build,
                python_version,
                ..
            } => Runtime::UvSglang(UvSglang {
                server_version: server_version.clone(),
                build: build.clone(),
                python_version: python_version.clone(),
                source,
            }),
        })
    }

    /// Plan runtime for this row, checking the lookup slug matches the composed
    /// `server@version` identity.
    pub fn to_runtime_for_slug(&self, slug: &UvSlug) -> anyhow::Result<Runtime> {
        let runtime = self.to_runtime()?;
        let composed = format!("{}@{}", self.server_label(), self.version()?);
        anyhow::ensure!(
            composed == slug.as_str(),
            "catalog slug mismatch: lookup key '{}' vs composed '{composed}'",
            slug.as_str()
        );
        Ok(runtime)
    }
}

fn optional_install_flags(flags: &[String]) -> Option<Vec<String>> {
    if flags.is_empty() {
        None
    } else {
        Some(flags.to_vec())
    }
}

/// File-level shape of `bundled-catalog/catalog.toml`. Each
/// `[[uv_vllm]]` / `[[uv_sglang]]` table deserializes as one
/// [`CatalogEntryRaw`]; the slug body is computed at load time.
#[derive(Debug, Deserialize)]
struct CatalogFile {
    #[serde(default)]
    uv_vllm: Vec<CatalogEntryRaw>,
    #[serde(default)]
    uv_sglang: Vec<CatalogEntryRaw>,
}

/// One bundled-catalog row before slug computation. The canonical
/// runtime version is derived from the component fields as
/// `<server_version>+<build>.py<python_version>`.
#[derive(Debug, Deserialize)]
struct CatalogEntryRaw {
    server_version: UvServerVersion,
    build: UvBuild,
    python_version: UvPythonVersion,
    requirements: NonEmptyString,
    /// Defaults to `--index-strategy unsafe-best-match` so PyTorch's
    /// `--extra-index-url` can combine with PyPI transitive deps.
    #[serde(default = "default_uv_install_flags")]
    install_flags: Vec<String>,
}

fn default_uv_install_flags() -> Vec<String> {
    vec!["--index-strategy".into(), "unsafe-best-match".into()]
}

/// Build a [`CatalogEntry`] from the raw row after composing its
/// canonical runtime version.
/// Returns `Err` if the component fields do not compose into a valid
/// slug version.
fn build_catalog_entry_vllm(raw: &CatalogEntryRaw) -> anyhow::Result<(UvSlug, CatalogEntry)> {
    let version = catalog_runtime_version(raw)?;
    let slug = UvSlug::vllm(&version)?;
    Ok((
        slug,
        CatalogEntry::UvVllm {
            server_version: raw.server_version.clone(),
            build: raw.build.clone(),
            python_version: raw.python_version.clone(),
            requirements: raw.requirements.clone(),
            install_flags: raw.install_flags.clone(),
        },
    ))
}

fn build_catalog_entry_sglang(raw: &CatalogEntryRaw) -> anyhow::Result<(UvSlug, CatalogEntry)> {
    let version = catalog_runtime_version(raw)?;
    let slug = UvSlug::sglang(&version)?;
    Ok((
        slug,
        CatalogEntry::UvSglang {
            server_version: raw.server_version.clone(),
            build: raw.build.clone(),
            python_version: raw.python_version.clone(),
            requirements: raw.requirements.clone(),
            install_flags: raw.install_flags.clone(),
        },
    ))
}

fn catalog_runtime_version(raw: &CatalogEntryRaw) -> anyhow::Result<SemverWithBuild> {
    uv_runtime_version(&raw.server_version, &raw.build, &raw.python_version)
}

/// Parse a catalog TOML body into the slug-keyed `(server, flavor)`
/// map used by [`catalog`]. Returns `Err` on malformed TOML, invalid
/// component fields, failed runtime-version composition, or a
/// duplicate computed slug.
///
/// Factored out from [`catalog`] so tests can exercise the rejection
/// paths against synthetic TOML strings — the production loader pins
/// the file via `include_str!` and turns errors into panics.
fn parse_catalog_from_str(toml_str: &str) -> anyhow::Result<BTreeMap<String, CatalogEntry>> {
    let file: CatalogFile = toml::from_str(toml_str).context("malformed bundled-catalog.toml")?;
    file.uv_vllm
        .iter()
        .map(build_catalog_entry_vllm)
        .chain(file.uv_sglang.iter().map(build_catalog_entry_sglang))
        .try_fold(BTreeMap::new(), |mut map, entry| {
            let (slug, entry) = entry?;
            let body = slug.as_str().to_string();
            if map.insert(body.clone(), entry).is_some() {
                anyhow::bail!("bundled-catalog.toml has duplicate slug '{body}'");
            }
            Ok(map)
        })
}

/// Parsed catalog: slug body -> derived entry. Lazy + cached: the TOML
/// is stable per-binary so parsing once at first use is enough. The
/// parse can fail on a malformed bundled TOML, so the accessor is
/// fallible; success is cached and the error path leaves the cell
/// uninitialized so a retry re-parses.
fn catalog() -> anyhow::Result<&'static BTreeMap<String, CatalogEntry>> {
    static CATALOG: OnceLock<BTreeMap<String, CatalogEntry>> = OnceLock::new();
    if let Some(catalog) = CATALOG.get() {
        return Ok(catalog);
    }
    let parsed = parse_catalog_from_str(CATALOG_TOML)?;
    Ok(CATALOG.get_or_init(|| parsed))
}

/// Look up a slug in the bundled catalog. Returns `Ok(None)` for slugs
/// installed via `--requirements` (custom slugs) or typos.
pub fn lookup(slug: &UvSlug) -> anyhow::Result<Option<CatalogEntry>> {
    Ok(catalog()?.get(slug.as_str()).cloned())
}

/// All bundled slug bodies, sorted. Drives `runtimes catalog` and tests.
pub fn slugs() -> anyhow::Result<impl Iterator<Item = &'static str>> {
    Ok(catalog()?.keys().map(String::as_str))
}

/// Bundled requirements text embedded in `catalog.toml`.
pub fn bundled_requirements(slug: &UvSlug) -> anyhow::Result<Option<&'static str>> {
    Ok(catalog()?
        .get(slug.as_str())
        .map(CatalogEntry::requirements))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The build-time TOML parses, has the expected 7 entries from the
    /// design / ticket, and every entry maps to a real
    /// `(runtime variant, flavor)`.
    #[test]
    fn catalog_parses_and_has_expected_slugs() -> anyhow::Result<()> {
        let cat = catalog()?;
        // Spot-check the seven slugs the ticket calls out. Adding new
        // ones is fine and shouldn't break this test, but the initial
        // matrix is the documented contract.
        let expected: &[(&str, &str, VllmFlavor)] = &[
            ("vllm@0.21.0+cu121.py3.12", "vllm", VllmFlavor::NvidiaGpu),
            ("vllm@0.21.0+cu124.py3.12", "vllm", VllmFlavor::NvidiaGpu),
            ("vllm@0.21.0+rocm6.py3.12", "vllm", VllmFlavor::AmdGpu),
            ("vllm@0.21.0+cpu.py3.12", "vllm", VllmFlavor::Cpu),
            (
                "sglang@0.5.12.post1+cu121.py3.12",
                "sglang",
                VllmFlavor::NvidiaGpu,
            ),
            (
                "sglang@0.5.12.post1+rocm6.py3.12",
                "sglang",
                VllmFlavor::AmdGpu,
            ),
            ("sglang@0.5.12.post1+cpu.py3.12", "sglang", VllmFlavor::Cpu),
        ];
        expected
            .iter()
            .try_for_each(|(slug, server_label, hardware)| {
                let entry = cat
                    .get(*slug)
                    .with_context(|| format!("catalog missing slug {slug}"))?;
                assert_eq!(
                    entry.server_label(),
                    *server_label,
                    "wrong server for {slug}"
                );
                assert_eq!(entry.flavor(), *hardware, "wrong flavor for {slug}");
                Ok::<_, anyhow::Error>(())
            })?;
        assert!(
            cat.len() >= expected.len(),
            "catalog shrank below the initial 7 entries (got {})",
            cat.len()
        );
        Ok(())
    }

    /// Embedded requirements must be non-empty and start with either a
    /// comment or `--extra-index-url` (sanity check that we didn't
    /// commit malformed text by mistake).
    #[test]
    fn bundled_requirements_look_like_requirements_files() -> anyhow::Result<()> {
        catalog()?.iter().for_each(|(slug, entry)| {
            let req = entry.requirements();
            assert!(!req.trim().is_empty(), "{slug} requirements are empty");
            let first_real_line = req
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or_default()
                .trim_start();
            assert!(
                first_real_line.starts_with('#') || first_real_line.starts_with("--"),
                "{slug} requirements should start with `#` or `--`, got {first_real_line:?}"
            );
        });
        Ok(())
    }

    /// Bundled TOML must set the flags explicitly (serde default is only a fallback).
    #[test]
    fn bundled_entries_declare_index_strategy_install_flags() -> anyhow::Result<()> {
        let slug = UvSlug::try_new("vllm@0.21.0+cu121.py3.12")?;
        let entry = lookup(&slug)?.context("slug is bundled")?;
        assert_eq!(
            entry.install_flags(),
            ["--index-strategy", "unsafe-best-match"]
        );
        Ok(())
    }

    #[test]
    fn parse_catalog_defaults_install_flags_when_absent() -> anyhow::Result<()> {
        let toml_str = r#"
            [[uv_vllm]]
            server_version = "0.21.0"
            build = "cu121"
            python_version = "3.12"
            requirements = "vllm==0.21.0"
        "#;
        let parsed = parse_catalog_from_str(toml_str)?;
        let entry = parsed
            .get("vllm@0.21.0+cu121.py3.12")
            .context("expected parsed catalog entry")?;
        assert_eq!(
            entry.install_flags(),
            ["--index-strategy", "unsafe-best-match"]
        );
        Ok(())
    }

    #[test]
    fn lookup_returns_none_for_unbundled_slug() -> anyhow::Result<()> {
        let slug = UvSlug::try_new("vllm@99.99.99+cpu.py3.12")?;
        assert!(lookup(&slug)?.is_none());
        assert!(bundled_requirements(&slug)?.is_none());
        Ok(())
    }

    #[test]
    fn lookup_returns_some_for_bundled_slug() -> anyhow::Result<()> {
        let slug = UvSlug::try_new("vllm@0.21.0+cu121.py3.12")?;
        let entry = lookup(&slug)?.context("slug is bundled")?;
        assert!(entry.is_vllm());
        assert_eq!(entry.flavor(), VllmFlavor::NvidiaGpu);
        assert_eq!(entry.python_version().as_ref(), "3.12");
        Ok(())
    }

    /// Every bundled slug's declared flavor agrees with the flavor implied by
    /// the build component of its composed version.
    #[test]
    fn catalog_flavor_matches_slug_suffix_inference() -> anyhow::Result<()> {
        catalog()?.keys().try_for_each(|slug_body| {
            let slug = UvSlug::try_new(slug_body)?;
            let entry =
                lookup(&slug)?.with_context(|| format!("bundled slug '{slug_body}' missing"))?;
            let cataloged = entry.flavor();
            let version = entry.version()?;
            let inferred = flavor_from_version_build(&version)
                .with_context(|| format!("bundled slug '{slug_body}' has no inferrable flavor"))?;
            assert_eq!(
                cataloged, inferred,
                "catalog flavor {cataloged:?} disagrees with build metadata inference {inferred:?} for '{slug_body}'"
            );
            Ok::<_, anyhow::Error>(())
        })
    }

    /// Re-derive the flavor from a composed `<server>@<ver>+<build>.py<py>`
    /// version, independently of [`CatalogEntry::flavor`], so the cross-check
    /// above can't agree with itself.
    fn flavor_from_version_build(version: &SemverWithBuild) -> Option<VllmFlavor> {
        let body = version.to_string();
        let (_, build) = body.split_once('+')?;
        let (flavor, _) = build.rsplit_once(".py")?;
        let lower = flavor.to_ascii_lowercase();
        if lower == "cpu" {
            return Some(VllmFlavor::Cpu);
        }
        if lower.starts_with("rocm") {
            return Some(VllmFlavor::AmdGpu);
        }
        if lower.starts_with("cu") {
            return Some(VllmFlavor::NvidiaGpu);
        }
        None
    }

    #[test]
    fn build_catalog_entry_accepts_derived_metadata(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let raw = CatalogEntryRaw {
            server_version: UvServerVersion::try_new("0.21.0".to_string())?,
            build: UvBuild::try_new("cu121".to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            requirements: NonEmptyString::try_new("vllm==0.21.0".to_string())?,
            install_flags: default_uv_install_flags(),
        };
        let (slug, entry) = build_catalog_entry_vllm(&raw)?;
        assert_eq!(slug.as_str(), "vllm@0.21.0+cu121.py3.12");
        assert!(entry.is_vllm());
        assert_eq!(entry.flavor(), VllmFlavor::NvidiaGpu);
        assert_eq!(entry.python_version().as_ref(), "3.12");
        assert_eq!(entry.requirements(), "vllm==0.21.0");
        Ok(())
    }

    #[test]
    fn parse_catalog_uses_component_python_version() -> anyhow::Result<()> {
        let toml_str = r#"
            [[uv_vllm]]
            server_version = "0.21.0"
            build = "cu121"
            python_version = "3.11"
            requirements = "vllm==0.21.0"
        "#;
        let parsed = parse_catalog_from_str(toml_str)?;
        let entry = parsed
            .get("vllm@0.21.0+cu121.py3.11")
            .context("expected parsed catalog entry")?;
        assert_eq!(entry.python_version().as_ref(), "3.11");
        assert_eq!(entry.requirements(), "vllm==0.21.0");
        let slug = crate::slug::UvSlug::try_new("vllm@0.21.0+cu121.py3.11")?;
        let runtime = entry.to_runtime_for_slug(&slug)?;
        let pipette_plan_types::Runtime::UvVllm(u) = runtime else {
            anyhow::bail!("expected uv_vllm, got {runtime}");
        };
        assert_eq!(u.python_version.as_ref(), "3.11");
        Ok(())
    }

    #[test]
    fn parse_catalog_rejects_malformed_components(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cases: &[(&str, &str, &str, &str)] = &[
            // `server_version` must not carry build metadata.
            ("0.21.0+cu121", "cu121", "3.12", "server_version"),
            // `build` must not carry Python metadata.
            ("0.21.0", "cpu.py3.12", "3.12", "build"),
            // `xpu` isn't in the recognised `cu*` / `rocm*` / `cpu` set.
            ("0.21.0", "xpu", "3.12", "build"),
        ];
        cases
            .iter()
            .try_for_each(|(server_version, build, python_version, field)| {
            let toml_str = format!(
                r#"
                [[uv_vllm]]
                server_version = "{server_version}"
                build = "{build}"
                python_version = "{python_version}"
                requirements = "vllm==0.21.0"
                "#
            );
            let err = parse_catalog_from_str(&toml_str)
                .err()
                .context("expected malformed-component error")?;
            let msg = format!("{err:#}");
            assert!(
                msg.contains(field),
                "expected '{field}' in error for {server_version}+{build}.py{python_version}, got: {msg}"
            );
            Ok::<_, Box<dyn std::error::Error>>(())
        })?;
        Ok(())
    }

    #[test]
    fn parse_catalog_rejects_duplicate_slug() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        // Two `[[uv_vllm]]` entries with the same components compute
        // to the same slug body — must be rejected at load time.
        let toml_str = r#"
            [[uv_vllm]]
            server_version = "0.21.0"
            build = "cu121"
            python_version = "3.12"
            requirements = "vllm==0.21.0"

            [[uv_vllm]]
            server_version = "0.21.0"
            build = "cu121"
            python_version = "3.12"
            requirements = "vllm==0.21.0"
        "#;
        let err = parse_catalog_from_str(toml_str)
            .err()
            .context("expected duplicate-slug error")?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("duplicate slug 'vllm@0.21.0+cu121.py3.12'"),
            "expected duplicate-slug error, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn slugs_iterates_in_sorted_order() -> anyhow::Result<()> {
        let names: Vec<&str> = slugs()?.collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "BTreeMap iteration must be sorted");
        Ok(())
    }

    #[test]
    fn to_runtime_carries_the_rows_requirements() -> anyhow::Result<()> {
        let slug = UvSlug::try_new("vllm@0.21.0+cu121.py3.12")?;
        let entry = lookup(&slug)?.context("expected bundled slug")?;
        let Runtime::UvVllm(rt) = entry.to_runtime()? else {
            anyhow::bail!("expected UvVllm");
        };
        match &rt.source {
            // The slug found the row and then dropped away: what survives is the
            // body, which is what the runtime is.
            UvRuntimeSource::PipRequirementsText { contents, .. } => {
                assert_eq!(contents.as_ref(), entry.requirements());
                assert!(contents.as_ref().contains("vllm==0.21.0"));
            }
            other => anyhow::bail!("expected PipRequirementsText, got {other:?}"),
        }
        Ok(())
    }
}
