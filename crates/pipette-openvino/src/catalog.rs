//! Bundled OpenVINO runtime catalog (`bundled-catalog/catalog.toml`).
//!
//! Installs go through `pipette_artifacts::ensure_runtime` — this module only
//! maps catalog rows to plan-types [`Runtime`] values.
//!
//! Unlike the mlx and torch-oai catalogs there is no flavor/build column: one
//! `openvino-genai` wheel serves CPU, GPU and NPU, so the device is a field on
//! the runtime rather than a property of the install. A catalog lookup
//! therefore takes the device alongside the version.

use std::sync::OnceLock;

use pipette_plan_types::{
    NonEmptyString, Runtime, UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
};
use pipette_venv::VersionCatalog;

const CATALOG_TOML: &str = include_str!("../bundled-catalog/catalog.toml");

/// Interpreter every bundled entry installs against.
///
/// Not a catalog column: OpenVINO publishes wheels across a range of CPython
/// versions and none of the pinned set is version-sensitive, so a per-row knob
/// would be a choice with no consequence. It becomes one the day an entry needs
/// a different interpreter.
const PYTHON_VERSION: &str = "3.11";

/// One bundled-catalog row: the openvino-genai version and the locked
/// requirements text embedded in `catalog.toml`.
///
/// `NonEmptyString` rather than `String` so an empty column is rejected when
/// the file is parsed rather than when that entry is first installed — the
/// same point `pipette-mlx` validates at.
#[derive(Debug, Clone, serde::Deserialize)]
struct CatalogEntry {
    version: NonEmptyString,
    requirements: NonEmptyString,
}

/// Parsed catalog: version -> entry. Lazy + cached; the static is ours because
/// the file is.
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

/// All bundled catalog versions — one row each, because one venv serves every
/// device. Drives `runtimes catalog uv_openvino`.
pub fn catalog_entries() -> anyhow::Result<Vec<&'static str>> {
    Ok(catalog()?
        .rows()
        .map(|entry| entry.version.as_ref())
        .collect())
}

/// Build a plan-types [`Runtime`] for a bundled catalog version and device.
pub fn declared_from_catalog(version: &str) -> anyhow::Result<Runtime> {
    let entry = catalog()?.get(version)?;
    Ok(Runtime::UvOpenvino(UvOpenvino {
        server_version: UvServerVersion::try_new(entry.version.as_ref().to_string())?,
        python_version: UvPythonVersion::try_new(PYTHON_VERSION.to_string())?,
        // The catalog version is the *lookup* key; it does not ride along in
        // the resolved source. What comes out is an ordinary uv-defined
        // runtime, identified by the body the row pinned.
        source: UvRuntimeSource::PipRequirementsText {
            contents: entry.requirements.clone(),
            install_flags: None,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_parses_the_shipped_entry() -> anyhow::Result<()> {
        let entry = catalog()?.get("2026.2.1")?;
        // The three wheels must move together — a tokenizers wheel from another
        // release cannot load the compiled tokenizer IR.
        [
            "openvino==2026.2.1",
            "openvino-genai==2026.2.1.0",
            "openvino-tokenizers==2026.2.1.0",
        ]
        .into_iter()
        .for_each(|pin| {
            assert!(
                entry.requirements.as_ref().contains(pin),
                "requirements should pin {pin}"
            )
        });
        assert_eq!(available_catalog_entries()?, vec!["2026.2.1"]);
        Ok(())
    }

    // Inference needs neither, and pulling them in would drag the export
    // stack's version constraints into a runtime that does not export.
    #[test]
    fn bundled_catalog_omits_the_export_only_stack() -> anyhow::Result<()> {
        let entry = catalog()?.get("2026.2.1")?;
        ["transformers", "optimum", "nncf", "torch"]
            .into_iter()
            .for_each(|absent| {
                assert!(
                    !entry.requirements.as_ref().contains(absent),
                    "runtime requirements should not carry {absent}"
                )
            });
        Ok(())
    }

    #[test]
    fn catalog_lists_one_row_per_version() -> anyhow::Result<()> {
        assert_eq!(catalog_entries()?, vec!["2026.2.1"]);
        Ok(())
    }

    /// The row is the artifact: a version and the body it pins, with nothing
    /// naming a device — that is the cell's to choose.
    #[test]
    fn declared_from_catalog_carries_the_pins_and_no_device() -> anyhow::Result<()> {
        let Runtime::UvOpenvino(rt) = declared_from_catalog("2026.2.1")? else {
            anyhow::bail!("expected a UvOpenvino runtime");
        };
        assert_eq!(rt.server_version.as_ref(), "2026.2.1");
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
        assert!(err.to_string().contains("2026.2.1"), "got {err}");
        Ok(())
    }
}
