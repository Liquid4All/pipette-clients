//! Bundled onnxruntime-genai runtime catalog
//! (`bundled-catalog/onnx_genai/catalog.toml`): maps rows to plan-types
//! [`Runtime`] values. `pipette_artifacts::ensure_runtime` installs them.
//!
//! The key is the `(version, execution provider)` pair: each provider is a
//! separate distribution, so one version has one row per provider.

use std::sync::OnceLock;

use pipette_plan_types::{
    NonEmptyString, OnnxExecutionProvider, Runtime, UvOnnxGenai, UvPythonVersion, UvRuntimeSource,
    UvServerVersion,
};

use crate::version_catalog::VersionCatalog;

const CATALOG_TOML: &str = include_str!("../../bundled-catalog/onnx_genai/catalog.toml");

/// Interpreter every bundled entry installs against. Not a catalog column
/// because no bundled row is sensitive to it.
const PYTHON_VERSION: &str = "3.11";

/// `NonEmptyString`, not `String`, so parsing the file rejects an empty column
/// instead of the first install of that entry.
#[derive(Debug, Clone, serde::Deserialize)]
struct CatalogEntry {
    version: NonEmptyString,
    ep: NonEmptyString,
    requirements: NonEmptyString,
}

/// The `(version, ep)` pair as one lookup key.
///
/// `+` matches the separator `UvOnnxGenai::runtime_version` uses, so a key in a
/// not-found error looks like a runtime version. The provider goes through the
/// case-insensitive [`OnnxExecutionProvider::parse`], so every caller keys on
/// one spelling of a named provider; a `Custom` one keys as written.
fn key(version: &str, ep: &str) -> String {
    format!("{version}+{}", OnnxExecutionProvider::parse(ep).as_str())
}

fn catalog() -> anyhow::Result<&'static VersionCatalog<CatalogEntry>> {
    static CATALOG: OnceLock<VersionCatalog<CatalogEntry>> = OnceLock::new();
    if let Some(catalog) = CATALOG.get() {
        return Ok(catalog);
    }
    let parsed = VersionCatalog::parse(
        CATALOG_TOML,
        "onnx_genai",
        |e: &CatalogEntry| key(e.version.as_ref(), e.ep.as_ref()),
        "uv_onnx_genai",
        "uv-onnx-genai",
    )?;
    Ok(CATALOG.get_or_init(|| parsed))
}

/// Every bundled row as its `(version, execution provider)` pair, in key order.
/// Drives `runtimes catalog uv_onnx_genai`.
pub fn catalog_entries() -> anyhow::Result<Vec<(&'static str, &'static str)>> {
    Ok(catalog()?
        .rows()
        .map(|entry| (entry.version.as_ref(), entry.ep.as_ref()))
        .collect())
}

/// Build a plan-types [`Runtime`] for a bundled catalog version and execution
/// provider.
pub fn declared_from_catalog(version: &str, ep: &str) -> anyhow::Result<Runtime> {
    let entry = catalog()?.get(&key(version, ep))?;
    Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
        server_version: UvServerVersion::try_new(entry.version.as_ref().to_string())?,
        execution_provider: OnnxExecutionProvider::parse(entry.ep.as_ref()),
        python_version: UvPythonVersion::try_new(PYTHON_VERSION.to_string())?,
        // The key only selected the row: the runtime is identified by the body
        // the row pins, like any hand-authored uv runtime.
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

    /// Each provider is a separate distribution, so the two rows must pin two
    /// different top-level packages. A row that pinned the CPU wheel under
    /// `ep = "cuda"` would install and then run on the CPU.
    #[rstest]
    #[case("0.16.0", "cpu", "onnxruntime-genai==0.16.0")]
    #[case("0.16.0", "cuda", "onnxruntime-genai-cuda==0.16.0")]
    #[case("0.17.1", "cpu", "onnxruntime-genai==0.17.1")]
    #[case("0.17.1", "cuda", "onnxruntime-genai-cuda==0.17.1")]
    #[case("0.18.0", "cpu", "onnxruntime-genai==0.18.0")]
    #[case("0.18.0", "cuda", "onnxruntime-genai-cuda==0.18.0")]
    fn each_provider_pins_its_own_distribution(
        #[case] version: &str,
        #[case] ep: &str,
        #[case] pin: &str,
    ) -> anyhow::Result<()> {
        let entry = catalog()?.get(&key(version, ep))?;
        let body = entry.requirements.as_ref();
        assert!(
            body.lines().any(|line| line.trim() == pin),
            "the {ep} row should pin `{pin}`: {body}"
        );
        Ok(())
    }

    /// The engine is a transitive dependency of the wrapper each row is named
    /// for, and it decides the measurement, so it carries an explicit pin. A
    /// floating engine would put two different builds under one storage key.
    #[rstest]
    #[case("0.16.0", "cpu", "onnxruntime==1.30.0")]
    #[case("0.16.0", "cuda", "onnxruntime-gpu==1.30.0")]
    #[case("0.17.1", "cpu", "onnxruntime==1.30.0")]
    #[case("0.17.1", "cuda", "onnxruntime-gpu==1.30.0")]
    #[case("0.18.0", "cpu", "onnxruntime==1.31.1")]
    #[case("0.18.0", "cuda", "onnxruntime-gpu==1.31.1")]
    fn each_row_pins_its_engine(
        #[case] version: &str,
        #[case] ep: &str,
        #[case] engine: &str,
    ) -> anyhow::Result<()> {
        let entry = catalog()?.get(&key(version, ep))?;
        let body = entry.requirements.as_ref();
        assert!(
            body.lines().any(|line| line.trim() == engine),
            "the {ep} row should pin `{engine}`: {body}"
        );
        Ok(())
    }

    /// Every line pins an exact version. One unpinned line is enough to make
    /// the row's storage key name a set of bytes that varies by install date.
    #[test]
    fn every_row_is_fully_pinned() -> anyhow::Result<()> {
        catalog()?.rows().try_for_each(|entry| {
            entry
                .requirements
                .as_ref()
                .lines()
                .filter(|line| !line.trim().is_empty())
                .try_for_each(|line| {
                    anyhow::ensure!(line.contains("=="), "`{line}` does not pin a version");
                    Ok(())
                })
        })
    }

    #[test]
    fn catalog_lists_one_row_per_version_and_provider() -> anyhow::Result<()> {
        assert_eq!(
            catalog_entries()?,
            vec![
                ("0.16.0", "cpu"),
                ("0.16.0", "cuda"),
                ("0.17.1", "cpu"),
                ("0.17.1", "cuda"),
                ("0.18.0", "cpu"),
                ("0.18.0", "cuda")
            ]
        );
        Ok(())
    }

    #[rstest]
    #[case("cpu", OnnxExecutionProvider::Cpu)]
    #[case("cuda", OnnxExecutionProvider::Cuda)]
    fn declared_from_catalog_carries_the_provider_and_its_pin(
        #[case] ep: &str,
        #[case] expected: OnnxExecutionProvider,
    ) -> anyhow::Result<()> {
        let Runtime::UvOnnxGenai(rt) = declared_from_catalog("0.16.0", ep)? else {
            anyhow::bail!("expected a UvOnnxGenai runtime");
        };
        assert_eq!(rt.server_version.as_ref(), "0.16.0");
        assert_eq!(rt.execution_provider, expected);
        assert_eq!(rt.python_version.as_ref(), PYTHON_VERSION);
        let UvRuntimeSource::PipRequirementsText { contents, .. } = &rt.source else {
            anyhow::bail!("expected a pip requirements source");
        };
        assert!(
            contents.as_ref().contains("onnxruntime-genai"),
            "got {contents}"
        );
        Ok(())
    }

    /// The two providers resolve to two different requirements bodies, which is
    /// what gives them two store entries rather than one shared venv.
    #[test]
    fn two_providers_are_two_environments() -> anyhow::Result<()> {
        let (Runtime::UvOnnxGenai(cpu), Runtime::UvOnnxGenai(cuda)) = (
            declared_from_catalog("0.16.0", "cpu")?,
            declared_from_catalog("0.16.0", "cuda")?,
        ) else {
            anyhow::bail!("expected two UvOnnxGenai runtimes");
        };
        assert_ne!(cpu.source, cuda.source);
        Ok(())
    }

    /// A version that exists under another provider is still a miss: the key is
    /// the pair, so half of one is not a row.
    #[rstest]
    #[case::unknown_version("1999.1.1", "cpu")]
    #[case::unknown_provider("0.16.0", "directml")]
    fn declared_from_catalog_rejects_an_unknown_pair(
        #[case] version: &str,
        #[case] ep: &str,
    ) -> anyhow::Result<()> {
        let Err(err) = declared_from_catalog(version, ep) else {
            anyhow::bail!("expected an unknown-entry rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains("0.16.0+cpu"), "should list what exists: {msg}");
        assert!(msg.contains("runtimes catalog uv_onnx_genai"), "got {msg}");
        assert!(msg.contains("uv-onnx-genai://"), "got {msg}");
        Ok(())
    }

    /// The URI grammar hands `ep` over as the operator typed it, so the key
    /// canonicalizes rather than trusting callers to lowercase first.
    #[rstest]
    #[case::lower("cuda")]
    #[case::upper("CUDA")]
    #[case::mixed("Cuda")]
    fn a_named_provider_keys_the_same_in_any_case(#[case] ep: &str) -> anyhow::Result<()> {
        let Runtime::UvOnnxGenai(rt) = declared_from_catalog("0.16.0", ep)? else {
            anyhow::bail!("expected a UvOnnxGenai runtime");
        };
        assert_eq!(rt.execution_provider, OnnxExecutionProvider::Cuda);
        Ok(())
    }

    /// The listing an operator pastes from uses the spelling the lookup keys on.
    #[test]
    fn every_row_keys_under_its_canonical_provider() -> anyhow::Result<()> {
        catalog_entries()?
            .into_iter()
            .try_for_each(|(version, ep)| {
                let canonical = OnnxExecutionProvider::parse(ep);
                anyhow::ensure!(
                    key(version, ep) == format!("{version}+{canonical}"),
                    "row `{version}+{ep}` does not key under `{canonical}`"
                );
                Ok(())
            })
    }
}
