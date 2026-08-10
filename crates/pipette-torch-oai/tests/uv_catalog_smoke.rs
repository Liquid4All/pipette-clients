//! Smoke tests for the bundled uv catalog path.
//!
//! Fast (not-ignored) tests exercise catalog lookup / requirements without
//! installing. The `#[ignore]` install smokes drive
//! `pipette_artifacts::ensure_runtime` end-to-end (same shared-store path as
//! `runtimes pull` / `benchmarks run`).
//!
//! ```text
//! # catalog wiring only (CI):
//! cargo test -p pipette-torch-oai --test uv_catalog_smoke -- --nocapture
//!
//! # real install (Linux + uv + network + vendor host; tens of minutes):
//! cargo test -p pipette-torch-oai --test uv_catalog_smoke \
//!     cpu_install_smoke -- --ignored --nocapture
//! ```

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::VllmFlavor;
use pipette_torch_oai::{catalog, slug::UvSlug};

// --- Cheap tests (run on every box) -----------------------------------

/// Every slug the ticket calls out is in the bundled catalog. SweepPins the
/// "minimum bundled set" so a future PR that accidentally drops one
/// fails here.
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

/// Every catalog entry's embedded requirements carry the
/// `--extra-index-url` matching its flavor. Catches drift between the
/// catalog version components and the actual wheel source.
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
                // The slug's `+cu<N>` suffix tells us which CUDA flavor.
                // Extract `<N>` instead of hardcoding the few we know
                // about — new catalog entries (cu129, cu130, …) drop in
                // without revisiting this test.
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
            VllmFlavor::AmdGpu => "/whl/rocm",
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

/// Install one bundled catalog slug via the shared-store ensure path (same
/// run path `pipette benchmarks run` / `runtimes pull` uses). Asserts the engine
/// projects from declared and the store venv python exists.
fn run_install_smoke(slug_body: &str) -> anyhow::Result<()> {
    use pipette_artifacts::runtime::RuntimeArtifactStore;
    use pipette_artifacts::{ensure_runtime, ArtifactsContext};
    use pipette_http::HttpClient;
    use pipette_plan_types::Runtime;

    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .try_init();
    if !uv_on_path() {
        eprintln!("skipping: uv not on PATH");
        return Ok(());
    }
    let work = TmpDir::new(slug_body)?;
    let data_dir = work.path();
    let slug = UvSlug::try_new(slug_body)?;
    // Expand catalog → self-contained declared Runtime (same as URI parse).
    let entry = pipette_torch_oai::catalog::lookup(&slug)?
        .with_context(|| format!("catalog miss for {slug_body}"))?;
    let declared = entry.to_runtime_for_slug(&slug)?;
    assert!(matches!(
        declared,
        Runtime::UvVllm(_) | Runtime::UvSglang(_)
    ));

    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?);
    let installed = ensure_runtime(&ctx, &store, &declared)?;
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

#[test]
#[ignore = "needs CUDA + network + ~tens of minutes; run with --ignored"]
fn cu121_install_smoke() -> anyhow::Result<()> {
    run_install_smoke("vllm@0.21.0+cu121.py3.12")
}

#[test]
#[ignore = "needs CPU host + network; run with --ignored"]
fn cpu_install_smoke() -> anyhow::Result<()> {
    run_install_smoke("vllm@0.21.0+cpu.py3.12")
}

#[test]
#[ignore = "needs ROCm + network + ~tens of minutes; run with --ignored"]
fn rocm_install_smoke() -> anyhow::Result<()> {
    run_install_smoke("vllm@0.21.0+rocm6.py3.12")
}
