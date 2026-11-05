//! Bundled MLX runtime catalog (`bundled-catalog/mlx/catalog.toml`): maps rows
//! to plan-types [`MlxMacosPipette`] values. `pipette_artifacts::ensure_runtime`
//! installs them.

use std::{collections::BTreeMap, sync::OnceLock};

use pipette_plan_types::{MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString, UvRuntimeSource};

use crate::version_catalog::{VersionCatalog, parse_catalog_rows};

const CATALOG_TOML: &str = include_str!("../../bundled-catalog/mlx/catalog.toml");

/// A row as written in the TOML, before `flavor` is typed.
#[derive(Debug, serde::Deserialize)]
struct CatalogEntryRaw {
    version: NonEmptyString,
    flavor: NonEmptyString,
    requirements: NonEmptyString,
}

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

/// Must match the flavor spelling in `pipette-cli`'s `runtime_uri`, so the
/// catalog listing and the URI grammar agree.
fn mlx_flavor_label(flavor: MlxMacosPipetteFlavor) -> &'static str {
    match flavor {
        MlxMacosPipetteFlavor::MacosArm64 => "macos-arm64",
    }
}

fn parse_mlx_flavor(s: &str) -> anyhow::Result<MlxMacosPipetteFlavor> {
    match s {
        "macos-arm64" => Ok(MlxMacosPipetteFlavor::MacosArm64),
        other => anyhow::bail!("unknown mlx flavor '{other}'; expected 'macos-arm64'"),
    }
}

/// Separate from [`catalog`] so tests can feed it synthetic TOML.
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

/// Build a plan-types [`MlxMacosPipette`] for a bundled catalog version name.
pub fn declared_from_catalog(version: &str) -> anyhow::Result<MlxMacosPipette> {
    let entry = catalog()?.get(version)?;
    Ok(MlxMacosPipette {
        version: NonEmptyString::try_new(entry.version().to_string())?,
        flavor: entry.flavor(),
        // The version only selected the row: the runtime is identified by the
        // body the row pins, like any hand-authored uv runtime.
        source: UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new(entry.requirements().to_string())?,
            install_flags: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

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

    #[test]
    fn bundled_catalog_parses_expected_entry() -> anyhow::Result<()> {
        let entry = catalog()?.get("0.31.3")?;
        assert_eq!(entry.flavor(), MlxMacosPipetteFlavor::MacosArm64);
        assert_eq!(entry.flavor_label(), "macos-arm64");
        assert!(
            entry.requirements().contains("mlx-lm==0.31.3"),
            "requirements should pin mlx-lm"
        );
        assert_eq!(
            available_catalog_entries()?,
            vec![
                "0.31.3",
                "0.32.0",
                "0.32.2",
                "0.32.4",
                "0.7.1+lfm2-vl-speculative",
                "0.7.11",
                "0.7.4"
            ]
        );
        assert_eq!(
            catalog_entries()?,
            vec![
                ("0.31.3", "macos-arm64"),
                ("0.32.0", "macos-arm64"),
                ("0.32.2", "macos-arm64"),
                ("0.32.4", "macos-arm64"),
                ("0.7.1+lfm2-vl-speculative", "macos-arm64"),
                ("0.7.11", "macos-arm64"),
                ("0.7.4", "macos-arm64")
            ]
        );
        Ok(())
    }

    /// A VL row pins `mlx-vlm` on a text row's mlx-lm and mlx core, so the
    /// text cells it also serves run that row's stack.
    #[rstest]
    #[case::commit(
        "0.7.1+lfm2-vl-speculative",
        "0.32.2",
        "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@"
    )]
    #[case::release("0.7.4", "0.32.0", "mlx-vlm==0.7.4\n")]
    #[case::current("0.7.11", "0.32.4", "mlx-vlm==0.7.11\n")]
    fn a_vl_row_pins_mlx_vlm_on_a_text_rows_stack(
        #[case] vl: &str,
        #[case] text: &str,
        #[case] mlx_vlm: &str,
    ) -> anyhow::Result<()> {
        let vl_row = catalog()?.get(vl)?.requirements();
        assert!(vl_row.contains(mlx_vlm), "{vl} should pin {mlx_vlm:?}");
        let text_row = catalog()?.get(text)?.requirements();
        ["mlx==", "mlx-lm==", "mlx-metal=="]
            .into_iter()
            .try_for_each(|package| {
                let pin = text_row
                    .lines()
                    .find(|line| line.starts_with(package))
                    .with_context(|| format!("{text} pins no {package}"))?;
                assert!(
                    vl_row.lines().any(|line| line == pin),
                    "{vl} should pin the {text} row's {pin}"
                );
                anyhow::Ok(())
            })
    }

    /// A re-locked row is a new entry, not an edit of the old one, so both stay
    /// pullable. It is keyed by the mlx core version because that is what moved.
    #[test]
    fn mlx_core_row_is_a_separate_entry() -> anyhow::Result<()> {
        let shipped = catalog()?.get("0.31.3")?;
        let relocked = catalog()?.get("0.32.2")?;
        assert_eq!(relocked.flavor(), MlxMacosPipetteFlavor::MacosArm64);
        assert!(
            relocked.requirements().contains("mlx-lm==0.31.3"),
            "the re-locked row pins the same mlx-lm"
        );
        assert!(
            relocked.requirements().contains("mlx==0.32.2"),
            "the row is keyed by the mlx core version it pins"
        );
        assert_ne!(
            shipped.requirements(),
            relocked.requirements(),
            "two rows with the same body would install one environment twice"
        );
        Ok(())
    }

    /// `0.32.0` moves mlx-lm, so it is keyed by mlx-lm's version even though
    /// it sorts before the older `0.32.2` row.
    #[test]
    fn mlx_lm_row_is_keyed_by_its_mlx_lm() -> anyhow::Result<()> {
        let row = catalog()?.get("0.32.0")?.requirements();
        assert!(row.contains("mlx-lm==0.32.0\n"), "{row}");
        assert!(row.contains("mlx==0.32.3\n"), "{row}");
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
        let rt = declared_from_catalog("0.31.3")?;
        assert_eq!(rt.version.as_ref(), "0.31.3");
        let row = catalog()?.get("0.31.3")?;
        assert!(matches!(
            rt.source,
            UvRuntimeSource::PipRequirementsText { ref contents, install_flags: None }
                if contents.as_ref() == row.requirements()
        ));
        Ok(())
    }
}
