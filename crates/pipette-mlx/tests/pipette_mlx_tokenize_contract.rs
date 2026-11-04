#![cfg(target_os = "macos")]

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Context;

use pipette_programs::ExternalProgram;

/// Whether the interpreter actually starts.
fn runs(python: &Path) -> bool {
    #[expect(clippy::disallowed_methods, reason = "both streams are discarded")]
    let status = Command::new(python)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(status, Ok(status) if status.success())
}

#[test]
fn python_tokenize_helper_contract_tests_pass() -> anyhow::Result<()> {
    // The same lookup the client uses, so a host carrying only `python`
    // still runs this rather than reporting itself Python-less.
    let Ok(python) = ExternalProgram::PYTHON.locate() else {
        eprintln!("skipping: no host python");
        return Ok(());
    };
    // Found is not usable: a shim that cannot start would otherwise fail this
    // test rather than skip it, which is what a host without Python gets.
    if !runs(&python) {
        eprintln!("skipping: {} does not run", python.display());
        return Ok(());
    }

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_path = crate_dir.join("tests").join("pipette_mlx_tokenize_test.py");
    #[expect(
        clippy::disallowed_methods,
        reason = "the Python suite reports on this test's own output"
    )]
    let status = Command::new(&python)
        .arg(&test_path)
        .current_dir(&crate_dir)
        .status()
        .with_context(|| format!("failed to run {}", test_path.display()))?;
    if !status.success() {
        anyhow::bail!("{} failed with {status}", test_path.display());
    }
    Ok(())
}
