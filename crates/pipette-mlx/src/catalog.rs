//! Bundled MLX runtime catalog (`bundled-catalog/catalog.toml`).
//!
//! Installs go through `pipette_artifacts::ensure_runtime` — this module
//! only maps catalog rows to plan-types [`Runtime`] values.

use std::{collections::BTreeMap, sync::OnceLock};

use pipette_plan_types::{
    MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString, Runtime, UvRuntimeSource,
};
use pipette_venv::{parse_catalog_rows, VersionCatalog};

// ---------------------------------------------------------------------------
// Bundled locked-requirements catalog
//
// The catalog ships as `bundled-catalog/catalog.toml`, compiled into the
// binary via `include_str!` — the same shape `pipette-torch-oai` uses. Each
// `[[mlx]]` table is one bundled runtime: `version` + `flavor` +
// `requirements`. The TOML is the single source of truth; adding an entry
// means appending a `[[mlx]]` block, no Rust edits required.
// See `crates/pipette-mlx/bundled-catalog/README.md`.
// ---------------------------------------------------------------------------

const CATALOG_TOML: &str = include_str!("../bundled-catalog/catalog.toml");

/// One bundled-catalog row as written in the TOML, before the flavor string is
/// parsed into its typed enum.
#[derive(Debug, serde::Deserialize)]
struct CatalogEntryRaw {
    version: NonEmptyString,
    flavor: NonEmptyString,
    requirements: NonEmptyString,
}

/// A parsed bundled-catalog entry: the mlx-lm version, its typed hardware
/// flavor, and the locked requirements text embedded in `catalog.toml`.
#[derive(Debug, Clone)]
struct CatalogEntry {
    version: String,
    flavor: MlxMacosPipetteFlavor,
    requirements: String,
}

impl CatalogEntry {
    fn version(&self) -> &str {
        &self.version
    }

    fn requirements(&self) -> &str {
        &self.requirements
    }

    fn flavor(&self) -> MlxMacosPipetteFlavor {
        self.flavor
    }

    fn flavor_label(&self) -> &'static str {
        mlx_flavor_label(self.flavor)
    }
}

/// Wire label for an [`MlxMacosPipetteFlavor`]. Mirrors the mapping in
/// `pipette-cli`'s `runtime_uri` flavor spelling so the catalog and the URI
/// grammar agree on the spelling.
fn mlx_flavor_label(flavor: MlxMacosPipetteFlavor) -> &'static str {
    match flavor {
        MlxMacosPipetteFlavor::MacosArm64 => "macos-arm64",
    }
}

/// Parse a catalog `flavor` string into its typed enum.
fn parse_mlx_flavor(s: &str) -> anyhow::Result<MlxMacosPipetteFlavor> {
    match s {
        "macos-arm64" => Ok(MlxMacosPipetteFlavor::MacosArm64),
        other => anyhow::bail!("unknown mlx flavor '{other}'; expected 'macos-arm64'"),
    }
}

/// Rows through the shared parser, then the flavor column typed — the one part
/// that is mlx's. Factored out so tests can exercise the rejection paths
/// against synthetic TOML.
fn parse_catalog_from_str(toml_str: &str) -> anyhow::Result<BTreeMap<String, CatalogEntry>> {
    let raw: BTreeMap<String, CatalogEntryRaw> =
        parse_catalog_rows(toml_str, "mlx", |e: &CatalogEntryRaw| {
            e.version.as_ref().to_string()
        })?;
    raw.into_iter()
        .map(|(version, raw)| {
            Ok((
                version.clone(),
                CatalogEntry {
                    flavor: parse_mlx_flavor(raw.flavor.as_ref())?,
                    requirements: raw.requirements.as_ref().to_string(),
                    version,
                },
            ))
        })
        .collect()
}

/// Parsed catalog: version -> entry. Lazy + cached.
fn catalog() -> anyhow::Result<&'static VersionCatalog<CatalogEntry>> {
    static CATALOG: OnceLock<VersionCatalog<CatalogEntry>> = OnceLock::new();
    if let Some(catalog) = CATALOG.get() {
        return Ok(catalog);
    }
    let parsed = VersionCatalog::from_rows(
        parse_catalog_from_str(CATALOG_TOML)?,
        "mlx_macos_pipette",
        "mlx-macos-pipette",
    );
    Ok(CATALOG.get_or_init(|| parsed))
}

/// All bundled catalog versions, sorted.
pub fn available_catalog_entries() -> anyhow::Result<Vec<&'static str>> {
    Ok(catalog()?.versions())
}

/// All bundled catalog entries as `(version, flavor-label)` pairs, sorted by
/// version. Drives `runtimes catalog mlx_macos_pipette`.
pub fn catalog_entries() -> anyhow::Result<Vec<(&'static str, &'static str)>> {
    Ok(catalog()?
        .rows()
        .map(|entry| (entry.version(), entry.flavor_label()))
        .collect())
}

/// Build a plan-types [`Runtime`] for a bundled catalog version name.
pub fn declared_from_catalog(version: &str) -> anyhow::Result<Runtime> {
    let entry = catalog()?.get(version)?;
    Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
        version: NonEmptyString::try_new(entry.version().to_string())?,
        flavor: entry.flavor(),
        // The catalog version is the *lookup* key; it does not ride along in the
        // resolved source. What comes out is an ordinary uv-defined runtime,
        // identified by the body the row pinned.
        source: UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new(entry.requirements().to_string())?,
            install_flags: None,
        },
    }))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn is_bundled_catalog_entry_recognises_known_versions() -> anyhow::Result<()> {
        assert!(catalog()?.get("0.31.3").is_ok());
        assert!(catalog()?.get("0.30").is_err());
        assert!(catalog()?.get("99.99").is_err());
        assert!(catalog()?.get("custom-name").is_err());
        assert!(catalog()?.get("").is_err());
        Ok(())
    }

    /// The build-time TOML parses and the shipped `0.31.3` entry loads
    /// with the expected flavor and non-empty requirements.
    #[test]
    fn bundled_catalog_parses_expected_entry() -> anyhow::Result<()> {
        let entry = catalog()?.get("0.31.3")?;
        assert_eq!(entry.flavor(), MlxMacosPipetteFlavor::MacosArm64);
        assert_eq!(entry.flavor_label(), "macos-arm64");
        assert!(
            entry.requirements().contains("mlx-lm==0.31.3"),
            "requirements should pin mlx-lm"
        );
        assert_eq!(available_catalog_entries()?, vec!["0.31.3"]);
        assert_eq!(catalog_entries()?, vec![("0.31.3", "macos-arm64")]);
        Ok(())
    }

    #[test]
    fn parse_catalog_reads_version_and_flavor() -> anyhow::Result<()> {
        let toml_str = r#"
            [[mlx]]
            version = "1.2.3"
            flavor = "macos-arm64"
            requirements = "mlx-lm==1.2.3"
        "#;
        let parsed = parse_catalog_from_str(toml_str)?;
        let entry = parsed.get("1.2.3").context("expected parsed entry")?;
        assert_eq!(entry.flavor(), MlxMacosPipetteFlavor::MacosArm64);
        assert_eq!(entry.requirements(), "mlx-lm==1.2.3");
        Ok(())
    }

    #[test]
    fn parse_catalog_rejects_unknown_flavor() -> anyhow::Result<()> {
        let toml_str = r#"
            [[mlx]]
            version = "1.2.3"
            flavor = "linux-x64"
            requirements = "mlx-lm==1.2.3"
        "#;
        let err = parse_catalog_from_str(toml_str)
            .err()
            .context("expected unknown-flavor error")?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("linux-x64"),
            "error should name the flavor: {msg}"
        );
        Ok(())
    }

    #[test]
    fn parse_catalog_rejects_duplicate_version() -> anyhow::Result<()> {
        let toml_str = r#"
            [[mlx]]
            version = "1.2.3"
            flavor = "macos-arm64"
            requirements = "mlx-lm==1.2.3"

            [[mlx]]
            version = "1.2.3"
            flavor = "macos-arm64"
            requirements = "mlx-lm==1.2.3"
        "#;
        let err = parse_catalog_from_str(toml_str)
            .err()
            .context("expected duplicate-version error")?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("duplicate version '1.2.3'"),
            "expected duplicate-version error, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn declared_from_catalog_carries_the_rows_requirements() -> anyhow::Result<()> {
        let Runtime::MlxMacosPipette(rt) = declared_from_catalog("0.31.3")? else {
            anyhow::bail!("expected mlx runtime");
        };
        assert_eq!(rt.version.as_ref(), "0.31.3");
        let row = catalog()?.get("0.31.3")?;
        // The version selected the row; what the runtime carries is the row's
        // body, with no residue naming where it came from.
        assert!(matches!(
            rt.source,
            UvRuntimeSource::PipRequirementsText { ref contents, install_flags: None }
                if contents.as_ref() == row.requirements()
        ));
        Ok(())
    }
}
