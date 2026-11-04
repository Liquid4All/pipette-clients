//! Bundled OVMS release catalog (`bundled-catalog/ovms/catalog.toml`): the sha256
//! each upstream release asset is pinned to.
//!
//! Installs go through `pipette_artifacts::ensure_runtime`, which reads the pin
//! here; the CLI lists the rows as paste-ready refs.

pub mod layout;

use std::collections::HashSet;
use std::sync::OnceLock;

use anyhow::Context;
use serde::Deserialize;

use pipette_plan_types::{NonEmptyString, OvmsFlavor, RepositoryUrl, Sha256};

const CATALOG_TOML: &str = include_str!("../../../bundled-catalog/ovms/catalog.toml");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    ovms: Vec<CatalogRow>,
}

/// One pinned upstream release asset.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogRow {
    pub tag: NonEmptyString,
    pub flavor: OvmsFlavor,
    pub sha256: Sha256,
}

/// Parsed rows, in file order. A row must name an upstream asset, and a
/// `(tag, flavor)` pair must appear once, or the pin it carries is ambiguous.
fn parse(toml_src: &str) -> anyhow::Result<Vec<CatalogRow>> {
    let file: CatalogFile = toml::from_str(toml_src).context("bundled ovms catalog")?;
    let mut seen = HashSet::new();
    file.ovms.iter().try_for_each(|row| {
        anyhow::ensure!(
            row.flavor.release_asset_name(row.tag.as_ref()).is_some(),
            "ovms catalog row `{}` `{}` names no upstream release asset",
            row.tag,
            row.flavor
        );
        anyhow::ensure!(
            seen.insert((row.tag.as_ref(), &row.flavor)),
            "ovms catalog pins `{}` `{}` twice",
            row.tag,
            row.flavor
        );
        Ok(())
    })?;
    Ok(file.ovms)
}

/// Every bundled row, in file order. Drives `runtimes catalog ovms`.
pub fn rows() -> anyhow::Result<&'static [CatalogRow]> {
    static ROWS: OnceLock<Vec<CatalogRow>> = OnceLock::new();
    if let Some(rows) = ROWS.get() {
        return Ok(rows);
    }
    let parsed = parse(CATALOG_TOML)?;
    Ok(ROWS.get_or_init(|| parsed))
}

/// The repo whose releases the catalog pins.
pub fn upstream_repository_url() -> RepositoryUrl {
    RepositoryUrl::new("github.com/openvinotoolkit/model_server")
}

/// The digest the catalog pins for the upstream asset at `tag` in `flavor`, or
/// `None` when no row names it.
pub fn pinned_sha256(tag: &str, flavor: &OvmsFlavor) -> anyhow::Result<Option<&'static Sha256>> {
    Ok(rows()?
        .iter()
        .find(|row| row.tag.as_ref() == tag && row.flavor == *flavor)
        .map(|row| &row.sha256))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn the_bundled_catalog_parses() -> anyhow::Result<()> {
        assert!(!rows()?.is_empty());
        Ok(())
    }

    /// The hash PIP-619 verified against the published `.sha256` on 2026-09-24.
    #[test]
    fn the_windows_release_is_pinned_to_its_published_digest() -> anyhow::Result<()> {
        assert_eq!(
            pinned_sha256("v2026.4.0", &OvmsFlavor::WindowsX64)?.map(AsRef::as_ref),
            Some("46d03114c97abfe05f2c5a8fde772c655aeef541ee254c23f402f81a616474e3")
        );
        Ok(())
    }

    #[test]
    fn the_latest_windows_release_is_pinned_to_its_published_digest() -> anyhow::Result<()> {
        assert_eq!(
            pinned_sha256("v2026.4.1", &OvmsFlavor::WindowsX64)?.map(AsRef::as_ref),
            Some("1bc2372d81bb4767829a7bf513745bd481ccbfec9550e278804ebe2051e3b678")
        );
        Ok(())
    }

    /// Every listed tag pins each named flavor, so no release is offered for
    /// only some of its OSes.
    #[test]
    fn every_tag_pins_every_known_flavor() -> anyhow::Result<()> {
        rows()?.iter().try_for_each(|row| {
            OvmsFlavor::known().try_for_each(|flavor| {
                anyhow::ensure!(
                    pinned_sha256(row.tag.as_ref(), &flavor)?.is_some(),
                    "no `{}` row for `{flavor}`",
                    row.tag
                );
                Ok(())
            })
        })
    }

    /// The tag is matched verbatim: `v2026.4` and `v2026.4.0` are different
    /// upstream tags, and a flavor outside the set is never pinned.
    #[rstest]
    #[case::short_tag("v2026.4", OvmsFlavor::WindowsX64)]
    #[case::no_v("2026.4.0", OvmsFlavor::WindowsX64)]
    #[case::custom("v2026.4.0", OvmsFlavor::Custom("acme-x64".to_owned()))]
    fn an_unlisted_pair_is_unpinned(
        #[case] tag: &str,
        #[case] flavor: OvmsFlavor,
    ) -> anyhow::Result<()> {
        assert_eq!(pinned_sha256(tag, &flavor)?, None);
        Ok(())
    }

    #[rstest]
    #[case::no_asset(
        "[[ovms]]\ntag = \"v2025.4\"\nflavor = \"windows-x64\"\nsha256 = \"00000000000000000000000000000000000000000000000000000000000000aa\"",
        "names no upstream release asset"
    )]
    #[case::duplicate(
        "[[ovms]]\ntag = \"v2026.4.0\"\nflavor = \"windows-x64\"\nsha256 = \"00000000000000000000000000000000000000000000000000000000000000aa\"\n[[ovms]]\ntag = \"v2026.4.0\"\nflavor = \"windows-x64\"\nsha256 = \"00000000000000000000000000000000000000000000000000000000000000bb\"",
        "twice"
    )]
    #[case::url_column(
        "[[ovms]]\ntag = \"v2026.4.0\"\nflavor = \"windows-x64\"\nsha256 = \"00000000000000000000000000000000000000000000000000000000000000aa\"\nurl = \"https://x\"",
        "unknown field `url`"
    )]
    fn a_malformed_catalog_is_refused(
        #[case] toml_src: &str,
        #[case] reason: &str,
    ) -> anyhow::Result<()> {
        let Err(err) = parse(toml_src) else {
            anyhow::bail!("should not parse: {toml_src}");
        };
        assert!(format!("{err:#}").contains(reason), "{err:#}");
        Ok(())
    }
}
