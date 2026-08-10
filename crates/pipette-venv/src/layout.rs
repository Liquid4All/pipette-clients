//! Where things live inside a store-installed venv, and how the run path binds
//! one.
//!
//! The layout is created by [`crate::install`], so it is stated here once
//! and read from here by every consumer — otherwise the installer and each
//! backend re-derive `bin/python` independently and can drift apart.
//!
//! Two spellings, selected here and nowhere else: `uv` writes `bin/python` on
//! Linux and macOS but `Scripts\python.exe` on Windows. Callers join paths
//! through [`venv_bin`] / [`venv_python`] rather than a literal, so a
//! Windows-hosted runtime needs no per-call-site branch.

use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_plan_types::{Runtime, RuntimeType, UvRuntimeSource};

/// `uv`'s executable dir: `Scripts` on Windows, `bin` on Linux and macOS.
const BIN_DIRNAME: &str = if cfg!(windows) { "Scripts" } else { "bin" };

/// `uv`'s interpreter filename, which carries the extension on Windows.
const PYTHON_FILENAME: &str = if cfg!(windows) {
    "python.exe"
} else {
    "python"
};

/// Executable dir inside a venv.
pub fn venv_bin(venv: &Path) -> PathBuf {
    venv.join(BIN_DIRNAME)
}

/// In-venv interpreter.
pub fn venv_python(venv: &Path) -> PathBuf {
    venv_bin(venv).join(PYTHON_FILENAME)
}

/// Venv root for a **bound** runtime, verified to be one of `expected` and to
/// hold an interpreter.
///
/// The whole bound-venv projection: every venv-backed backend wants the same
/// three checks in the same order — the runtime is one this engine runs, it was
/// bound, and its interpreter is there — and each used to spell them itself.
/// `expected` is a list because torch-oai serves two runtime types from one
/// engine.
pub fn require_bound_venv(bound: &Runtime, expected: &[RuntimeType]) -> anyhow::Result<PathBuf> {
    let actual = RuntimeType::of(bound);
    anyhow::ensure!(
        expected.contains(&actual),
        "expected {}, got `{actual}`",
        expected
            .iter()
            .map(RuntimeType::to_string)
            .collect::<Vec<_>>()
            .join(" / ")
    );
    // Exhaustive, so a new venv-backed runtime has to answer here rather than
    // being reported as one that does not use a venv.
    let source = match bound {
        Runtime::UvVllm(rt) => &rt.source,
        Runtime::UvSglang(rt) => &rt.source,
        Runtime::MlxMacosPipette(rt) => &rt.source,
        Runtime::UvOpenvino(rt) => &rt.source,
        // Installed as an archive, an image, or not at all.
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::AppleFoundation(_) => {
            anyhow::bail!("`{actual}` is not a venv-backed runtime")
        }
    };
    require_preinstalled_venv(source).with_context(|| format!("binding the {actual} runtime venv"))
}

/// Venv root for a bound runtime source, with the interpreter verified to
/// exist.
///
/// After `ensure` + `bind_under`, a venv-backed runtime's source is always
/// [`UvRuntimeSource::AbsolutePreinstalled`]; anything else means the bind
/// never happened. The source-only half of [`require_bound_venv`], which is the
/// entry point callers outside this module use.
fn require_preinstalled_venv(source: &UvRuntimeSource) -> anyhow::Result<PathBuf> {
    let UvRuntimeSource::AbsolutePreinstalled { dir } = source else {
        anyhow::bail!("expected an AbsolutePreinstalled venv after bind, got {source:?}");
    };
    let venv = PathBuf::from(dir.as_ref());
    let python = venv_python(&venv);
    if !python.is_file() {
        anyhow::bail!(
            "venv python missing at {} (ensure/bind may be incomplete)",
            python.display()
        );
    }
    Ok(venv)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AbsolutePath, NonEmptyString, RelativePath};

    use super::*;

    // The two spellings are asserted as literals, each on the host that uses
    // it, so neither arm can drift into agreeing with the implementation by
    // construction. CI runs a Linux, a macOS and a Windows leg, so both are
    // exercised.
    #[cfg(not(windows))]
    #[test]
    fn layout_is_posix_bin() {
        let venv = Path::new("/blobs/venv");
        assert_eq!(venv_bin(venv), Path::new("/blobs/venv/bin"));
        assert_eq!(venv_python(venv), Path::new("/blobs/venv/bin/python"));
    }

    #[cfg(windows)]
    #[test]
    fn layout_is_windows_scripts() {
        let venv = Path::new(r"C:\blobs\venv");
        assert_eq!(venv_bin(venv), Path::new(r"C:\blobs\venv\Scripts"));
        assert_eq!(
            venv_python(venv),
            Path::new(r"C:\blobs\venv\Scripts\python.exe")
        );
    }

    // Host-independent: whichever spelling is in force, the interpreter lives
    // inside the executable dir, which lives directly under the venv root.
    #[test]
    fn python_sits_inside_the_executable_dir() {
        let venv = Path::new("/blobs/venv");
        let bin = venv_bin(venv);
        assert_eq!(bin.parent(), Some(venv));
        assert_eq!(venv_python(venv).parent(), Some(bin.as_path()));
    }

    #[test]
    fn require_preinstalled_venv_returns_the_root_when_python_exists() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let venv = tmp.path().join("venv");
        std::fs::create_dir_all(venv_bin(&venv))?;
        std::fs::write(venv_python(&venv), "")?;

        let source = UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(venv.to_string_lossy().into_owned())?,
        };
        assert_eq!(require_preinstalled_venv(&source)?, venv);
        Ok(())
    }

    #[test]
    fn require_preinstalled_venv_rejects_a_venv_without_an_interpreter() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let source = UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
        };
        let Err(err) = require_preinstalled_venv(&source) else {
            anyhow::bail!("expected a missing-interpreter error");
        };
        assert!(err.to_string().contains("venv python missing"), "got {err}");
        Ok(())
    }

    // An unbound source means prepare never ran; the run path must not treat it
    // as a usable venv.
    #[test]
    fn require_preinstalled_venv_rejects_unbound_sources() -> anyhow::Result<()> {
        for source in [
            UvRuntimeSource::RelativePreinstalled {
                dir: RelativePath::try_new("venv".to_string())?,
            },
            UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.28.4\n".to_string())?,
                install_flags: None,
            },
        ] {
            let Err(err) = require_preinstalled_venv(&source) else {
                anyhow::bail!("expected {source:?} to be rejected");
            };
            assert!(
                err.to_string().contains("AbsolutePreinstalled"),
                "got {err}"
            );
        }
        Ok(())
    }
}
