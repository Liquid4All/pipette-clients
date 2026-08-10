//! llama.cpp-specific fixtures for the eval integration tests (crash-recovery,
//! connection-reset, stop_reason): the fake `llama-server` install and the
//! plan-localized runtime/model that point at it.
//!
//! Engine-agnostic setup is deliberately absent; only the fake-server wiring
//! has to live here, because `CARGO_BIN_EXE_fake_llama_server` resolves in the
//! declaring crate alone.
#![allow(dead_code)] // each test crate uses a subset of these helpers

use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_ops::readiness::RepObserver;
use pipette_plan_types::benchmark::BenchmarkDefinition;
use pipette_plan_types::run::{DeclaredBound, RunRequest};
use pipette_plan_types::{
    AbsolutePath, BenchmarkFlags, GgufText, GgufTextSource, LlamaCppFlavor, LlamacppCliStockTools,
    LlamacppCliStockToolsSource, Model, Runtime,
};

/// Path to the compiled fake llama-server test binary; asserts it exists.
pub fn fake_server_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_BIN_EXE_fake_llama_server"));
    assert!(
        p.exists(),
        "fake_llama_server binary missing at {}",
        p.display(),
    );
    p
}

/// Install root containing `llama-server` (and a dummy `llama-bench`) so
/// `run_eval` can resolve tools under a plan `AbsoluteDir`.
pub fn fake_runtime_install_dir(root: &Path) -> anyhow::Result<PathBuf> {
    let install = root.join("runtime-install");
    std::fs::create_dir_all(&install)?;
    let server_src = fake_server_path();
    let server_dst = install.join(if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    std::fs::copy(&server_src, &server_dst).with_context(|| {
        format!(
            "copy fake server {} → {}",
            server_src.display(),
            server_dst.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&server_dst)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&server_dst, perms)?;
    }
    // find_tool also requires llama-bench somewhere under the root.
    let bench_dst = install.join(if cfg!(windows) {
        "llama-bench.exe"
    } else {
        "llama-bench"
    });
    std::fs::write(&bench_dst, b"#!/bin/sh\nexit 1\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bench_dst)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bench_dst, perms)?;
    }
    Ok(install)
}

/// Plan runtime localized to `install_dir` (absolute).
pub fn fake_runtime(install_dir: &Path) -> anyhow::Result<Runtime> {
    let dir = AbsolutePath::try_new(install_dir.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source: LlamacppCliStockToolsSource::AbsoluteDir { dir },
        flavor: LlamaCppFlavor::MacosArm64,
    }))
}

/// Plan GGUF text model at an absolute local path.
pub fn fake_model(model_path: &Path) -> anyhow::Result<Model> {
    let path = AbsolutePath::try_new(model_path.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    Ok(Model::GgufText(GgufText {
        source: GgufTextSource::AbsoluteFile { path },
    }))
}

/// Build a [`RunRequest`] for eval integration tests.
pub fn fake_run_request(
    runtime: Runtime,
    model: Model,
    benchmark: BenchmarkDefinition,
    benchmark_flags: Option<BenchmarkFlags>,
) -> RunRequest {
    RunRequest {
        runtime: DeclaredBound::already_bound(runtime),
        model: DeclaredBound::already_bound(model),
        runtime_flags: None,
        model_flags: None,
        benchmark_flags,
        benchmark,
    }
}

/// Eval never gates on readiness, and a test must not block on real host
/// thermals regardless — injection is what lets us say so.
pub fn no_readiness_gate() -> anyhow::Result<()> {
    Ok(())
}

/// A [`RepObserver`] that records nothing: eval marks no repetitions, and a
/// test must not probe host sensors anyway.
pub fn ignore_reps() -> RepObserver<'static> {
    RepObserver::new(|| Ok(()), || Ok(()))
}
