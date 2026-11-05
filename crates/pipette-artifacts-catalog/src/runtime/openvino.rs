//! Bundled OpenVINO runtime catalog (`bundled-catalog/openvino/catalog.toml`):
//! maps rows to plan-types [`Runtime`] values. `pipette_artifacts::ensure_runtime`
//! installs them.
//!
//! There is no flavor/build column: one `openvino-genai` wheel serves CPU, GPU
//! and NPU, so the device is a per-cell flag and a lookup takes only the version.

use std::sync::OnceLock;

use pipette_plan_types::{
    NonEmptyString, Runtime, UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
};

use crate::version_catalog::VersionCatalog;

const CATALOG_TOML: &str = include_str!("../../bundled-catalog/openvino/catalog.toml");

/// Interpreter every bundled entry installs against. Not a catalog column
/// because no bundled row is sensitive to it.
const PYTHON_VERSION: &str = "3.11";

/// `NonEmptyString`, not `String`, so parsing the file rejects an empty column
/// instead of the first install of that entry.
#[derive(Debug, Clone, serde::Deserialize)]
struct CatalogEntry {
    version: NonEmptyString,
    requirements: NonEmptyString,
}

fn catalog() -> anyhow::Result<&'static VersionCatalog<CatalogEntry>> {
    static CATALOG: OnceLock<VersionCatalog<CatalogEntry>> = OnceLock::new();
    if let Some(catalog) = CATALOG.get() {
        return Ok(catalog);
    }
    let parsed = VersionCatalog::parse(
        CATALOG_TOML,
        "openvino",
        |e: &CatalogEntry| e.version.as_ref().to_string(),
        "uv_openvino",
        "uv-openvino",
    )?;
    Ok(CATALOG.get_or_init(|| parsed))
}

/// All bundled catalog versions, sorted.
pub fn available_catalog_entries() -> anyhow::Result<Vec<&'static str>> {
    Ok(catalog()?.versions())
}

/// All bundled catalog versions, one row each because one venv serves every
/// device. Drives `runtimes catalog uv_openvino`.
pub fn catalog_entries() -> anyhow::Result<Vec<&'static str>> {
    Ok(catalog()?
        .rows()
        .map(|entry| entry.version.as_ref())
        .collect())
}

/// Build a plan-types [`Runtime`] for a bundled catalog version.
pub fn declared_from_catalog(version: &str) -> anyhow::Result<Runtime> {
    let entry = catalog()?.get(version)?;
    Ok(Runtime::UvOpenvino(UvOpenvino {
        server_version: UvServerVersion::try_new(entry.version.as_ref().to_string())?,
        python_version: UvPythonVersion::try_new(PYTHON_VERSION.to_string())?,
        // The version only selected the row: the runtime is identified by the
        // body the row pins, like any hand-authored uv runtime.
        source: UvRuntimeSource::PipRequirementsText {
            contents: entry.requirements.clone(),
            install_flags: None,
        },
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The three wheels must move together: a tokenizers wheel from another
    /// release cannot load the compiled tokenizer IR.
    #[rstest]
    #[case("2026.2.1")]
    #[case("2026.4.0")]
    #[case("2026.4.1")]
    fn bundled_entry_pins_one_release(#[case] version: &str) -> anyhow::Result<()> {
        let entry = catalog()?.get(version)?;
        [
            format!("openvino=={version}\n"),
            format!("openvino-genai=={version}.0\n"),
            format!("openvino-tokenizers=={version}.0\n"),
        ]
        .iter()
        .for_each(|pin| {
            assert!(
                entry.requirements.as_ref().contains(pin.as_str()),
                "{version} should pin {pin:?}"
            )
        });
        Ok(())
    }

    // Inference needs none of these, and pulling them in would drag the export
    // stack's version constraints into a runtime that does not export.
    #[test]
    fn bundled_catalog_omits_the_export_only_stack() -> anyhow::Result<()> {
        catalog_entries()?.into_iter().try_for_each(|version| {
            let entry = catalog()?.get(version)?;
            ["transformers", "optimum", "nncf", "torch"]
                .into_iter()
                .for_each(|absent| {
                    assert!(
                        !entry.requirements.as_ref().contains(absent),
                        "{version} should not carry {absent}"
                    )
                });
            anyhow::Ok(())
        })
    }

    #[test]
    fn catalog_lists_one_row_per_version() -> anyhow::Result<()> {
        assert_eq!(catalog_entries()?, vec!["2026.2.1", "2026.4.0", "2026.4.1"]);
        assert_eq!(
            available_catalog_entries()?,
            vec!["2026.2.1", "2026.4.0", "2026.4.1"]
        );
        Ok(())
    }

    /// The device is the cell's to choose, so the row must not name one.
    #[test]
    fn declared_from_catalog_carries_the_pins_and_no_device() -> anyhow::Result<()> {
        let Runtime::UvOpenvino(rt) = declared_from_catalog("2026.4.0")? else {
            anyhow::bail!("expected a UvOpenvino runtime");
        };
        assert_eq!(rt.server_version.as_ref(), "2026.4.0");
        let UvRuntimeSource::PipRequirementsText { contents, .. } = &rt.source else {
            anyhow::bail!("expected a pip requirements source");
        };
        ["cpu", "gpu", "npu"].into_iter().for_each(|device| {
            assert!(
                !contents.as_ref().contains(device),
                "the requirements body must not name {device}"
            )
        });
        Ok(())
    }

    #[test]
    fn declared_from_catalog_rejects_an_unknown_version() -> anyhow::Result<()> {
        let Err(err) = declared_from_catalog("1999.1.1") else {
            anyhow::bail!("expected an unknown-version rejection");
        };
        assert!(err.to_string().contains("2026.4.0"), "got {err}");
        Ok(())
    }
}
