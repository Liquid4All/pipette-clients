//! Smoke tests for the bundled uv catalog path.
//!
//! The fast tests check catalog lookup and requirements without installing; the
//! ignored install smokes drive `pipette_artifacts::ensure_runtime` end-to-end.
//!
//! ```text
//! # catalog wiring only (CI):
//! cargo test -p pipette-artifacts --test uv_catalog_smoke -- --nocapture
//!
//! # real install (Linux + uv + network; tens of minutes):
//! cargo test -p pipette-artifacts --test uv_catalog_smoke \
//!     install_smoke::case_2_cpu -- --ignored --nocapture
//! ```

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;
use rstest::rstest;

use pipette_artifacts_catalog::runtime::torch_oai::{self as catalog, slug::UvSlug};
use pipette_plan_types::VllmFlavor;

// --- Cheap tests (run on every box) -----------------------------------

/// The minimum bundled set: dropping one of these slugs fails here.
#[test]
fn ticket_slugs_are_bundled() -> anyhow::Result<()> {
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
        .try_for_each(|(slug_body, want_server_label, want_target)| {
            let slug = UvSlug::try_new(slug_body)?;
            let entry = catalog::lookup(&slug)?
                .with_context(|| format!("ticket-required slug '{slug_body}' is missing"))?;
            assert_eq!(
                entry.server_label(),
                *want_server_label,
                "wrong server for '{slug_body}'"
            );
            assert_eq!(
                entry.flavor(),
                *want_target,
                "wrong flavor for '{slug_body}'"
            );
            assert!(
                catalog::bundled_requirements(&slug)?.is_some(),
                "missing embedded requirements for bundled slug '{slug_body}'"
            );
            Ok::<_, anyhow::Error>(())
        })
}

/// Catches drift between a slug's flavor and the wheel index its embedded
/// requirements actually use.
#[test]
fn requirements_extra_index_url_matches_flavor() -> anyhow::Result<()> {
    catalog::slugs()?.try_for_each(|slug_body| {
        let slug = UvSlug::try_new(slug_body)?;
        let entry = catalog::lookup(&slug)?
            .with_context(|| format!("bundled slug '{slug_body}' missing"))?;
        let body = catalog::bundled_requirements(&slug)?
            .with_context(|| format!("missing embedded requirements for '{slug_body}'"))?;
        let expected: String;
        let expected_substring: &str = match entry.flavor() {
            VllmFlavor::NvidiaGpu => {
                // Read `<N>` off the slug's `+cu<N>` so a new CUDA entry needs
                // no change here.
                let cu = slug_body
                    .split('+')
                    .nth(1)
                    .and_then(|s| s.strip_prefix("cu"))
                    .and_then(|s| s.split('.').next())
                    .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    .with_context(|| {
                        format!("nvidia_gpu slug '{slug_body}' has no recognised +cu suffix")
                    })?;
                expected = format!("/whl/cu{cu}");
                &expected
            }
            // PyTorch's ROCm index, or vLLM's own (`wheels.vllm.ai/rocm/`).
            VllmFlavor::AmdGpu => "/rocm",
            VllmFlavor::Cpu => "/whl/cpu",
        };
        assert!(
            body.contains(expected_substring),
            "{slug_body} requirements should reference '{expected_substring}'; got:\n{body}"
        );
        Ok::<_, anyhow::Error>(())
    })
}

// --- Heavy install smokes (ignored) -----------------------------------

struct TmpDir(PathBuf);

impl TmpDir {
    fn new(label: &str) -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "pipette-torch-oai-catalog-smoke-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4(),
        ));
        fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn uv_on_path() -> bool {
    Command::new("uv")
        .arg("--version")
        .output()
        .ok()
        .is_some_and(|o| o.status.success())
}

/// Install one bundled catalog slug through `ensure_runtime` and check the
/// store's venv python exists.
#[rstest]
#[case::cu121("vllm@0.21.0+cu121.py3.12")]
#[case::cpu("vllm@0.21.0+cpu.py3.12")]
#[case::rocm("vllm@0.21.0+rocm6.py3.12")]
#[case::rocm72("vllm@0.27.1+rocm723.py3.12")]
#[case::sglang_cu130("sglang@0.5.23+cu130.py3.12")]
#[ignore = "needs Linux + network + ~tens of minutes; run with --ignored"]
fn install_smoke(#[case] slug_body: &str) -> anyhow::Result<()> {
    use pipette_artifacts::runtime::RuntimeArtifactStore;
    use pipette_artifacts::{ArtifactsContext, ensure_runtime};
    use pipette_http::HttpClient;
    use pipette_plan_types::Runtime;
    use pipette_programs::ExternalPrograms;

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_test_writer()
        .try_init();
    if !uv_on_path() {
        eprintln!("skipping: uv not on PATH");
        return Ok(());
    }
    let work = TmpDir::new(slug_body)?;
    let data_dir = work.path();
    let slug = UvSlug::try_new(slug_body)?;
    let entry = catalog::lookup(&slug)?.with_context(|| format!("catalog miss for {slug_body}"))?;
    let declared = entry.to_runtime_for_slug(&slug)?;
    assert!(matches!(
        declared,
        Runtime::UvVllm(_) | Runtime::UvSglang(_)
    ));

    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?, ExternalPrograms::from_path());
    let (installed, _) = ensure_runtime(&ctx, &store, &declared)?;
    assert!(
        matches!(&installed, Runtime::UvVllm(_) | Runtime::UvSglang(_)),
        "installed runtime should be uv_vllm or uv_sglang; got {installed}"
    );

    let found = store
        .find(&declared)?
        .with_context(|| format!("store find missing after install of '{slug_body}'"))?;
    assert_eq!(found.declared, declared);

    let python = pipette_venv::venv_python(&store.install_dir_for(&found)?);
    assert!(
        python.exists(),
        "venv python missing at {}",
        python.display()
    );
    Ok(())
}
