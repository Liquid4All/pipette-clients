//! Where a store-installed venv lives, where things live inside it, how the
//! run path binds one, and how to start its interpreter.
//!
//! `uv` writes `bin/python` on Linux and macOS but `Scripts\python.exe` on
//! Windows. Callers join paths through [`venv_bin`] / [`venv_python`] rather
//! than a literal, so no call site branches on the host.

use std::path::{Path, PathBuf};
use std::process::Command;

use pipette_plan_types::UvRuntimeSource;

/// Runtime manifests record it, so renaming it breaks installed runtimes.
const VENV_DIRNAME: &str = "venv";

/// `uv`'s executable dir: `Scripts` on Windows, `bin` on Linux and macOS.
const BIN_DIRNAME: &str = if cfg!(windows) { "Scripts" } else { "bin" };

/// `uv`'s interpreter filename, which carries the extension on Windows.
const PYTHON_FILENAME: &str = if cfg!(windows) {
    "python.exe"
} else {
    "python"
};

/// Where [`crate::install_venv`] creates the venv inside `payload`.
pub(crate) fn venv_dir(payload: &Path) -> PathBuf {
    payload.join(VENV_DIRNAME)
}

/// [`venv_dir`] as a runtime manifest records it: `/` on every host, so a
/// Windows install records the same path as any other.
pub fn stored_venv_dir(payload: &str) -> String {
    format!("{payload}/{VENV_DIRNAME}")
}

pub fn venv_bin(venv: &Path) -> PathBuf {
    venv.join(BIN_DIRNAME)
}

pub fn venv_python(venv: &Path) -> PathBuf {
    venv_bin(venv).join(PYTHON_FILENAME)
}

/// How a venv launcher tells the base interpreter which venv it runs for:
/// CPython takes it as `sys.executable` and finds `pyvenv.cfg` from there
/// (`Modules/getpath.py`).
const LAUNCHER_ENV: &str = "__PYVENV_LAUNCHER__";

/// Why a venv's base interpreter could not be found.
#[derive(Debug, thiserror::Error)]
pub enum PythonCommandError {
    #[error("{0} is not inside a venv")]
    NotInVenv(PathBuf),
    #[error("reading {path}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{0} names no `home`")]
    NoHome(PathBuf),
}

/// A command whose pid is the interpreter of `python`, a venv's
/// [`venv_python`]: the process a memory sampler must watch. On Windows that
/// path is a launcher (uv's trampoline, or CPython's `venvlauncher`) that holds
/// a few MiB and runs the base interpreter as its child, so this starts the
/// base interpreter as the launcher would. Elsewhere it is a symlink to it.
pub fn python_command(python: &Path) -> Result<Command, PythonCommandError> {
    if !cfg!(windows) {
        return Ok(Command::new(python));
    }
    let mut command = Command::new(base_python(python)?);
    command.env(LAUNCHER_ENV, python);
    Ok(command)
}

/// The interpreter `python`'s venv was made from: `home` in the venv's
/// `pyvenv.cfg`, read as CPython reads it.
fn base_python(python: &Path) -> Result<PathBuf, PythonCommandError> {
    let cfg = python
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| PythonCommandError::NotInVenv(python.to_owned()))?
        .join("pyvenv.cfg");
    let text = std::fs::read_to_string(&cfg).map_err(|source| PythonCommandError::Read {
        path: cfg.clone(),
        source,
    })?;
    let home = text
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            key.trim()
                .eq_ignore_ascii_case("home")
                .then(|| value.trim())
        })
        .ok_or_else(|| PythonCommandError::NoHome(cfg.clone()))?;
    Ok(Path::new(home).join(PYTHON_FILENAME))
}

/// Venv root for a bound runtime source, with the interpreter verified to
/// exist.
///
/// After `ensure` + `bind_under`, a venv-backed runtime's source is always
/// [`UvRuntimeSource::AbsolutePreinstalled`]; anything else means the bind
/// never happened.
pub fn require_bound_venv(source: &UvRuntimeSource) -> anyhow::Result<PathBuf> {
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

    // Literals, not `BIN_DIRNAME`, so the assertion cannot drift into agreeing
    // with the implementation by construction. CI runs all three host legs.
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

    #[test]
    fn python_sits_inside_the_executable_dir() {
        let venv = Path::new("/blobs/venv");
        let bin = venv_bin(venv);
        assert_eq!(bin.parent(), Some(venv));
        assert_eq!(venv_python(venv).parent(), Some(bin.as_path()));
    }

    /// A venv whose `pyvenv.cfg` is `cfg`, with an empty launcher; returns the
    /// launcher's path.
    fn venv_from(dir: &Path, cfg: &str) -> anyhow::Result<PathBuf> {
        let venv = dir.join("venv");
        std::fs::create_dir_all(venv_bin(&venv))?;
        std::fs::write(venv_python(&venv), "")?;
        std::fs::write(venv.join("pyvenv.cfg"), cfg)?;
        Ok(venv_python(&venv))
    }

    // getpath.py's reading: any key case, spaces around both sides.
    #[test]
    fn the_base_interpreter_is_the_venvs_home() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let base = tmp.path().join("cpython-3.11");
        let python = venv_from(
            tmp.path(),
            &format!(
                "implementation = CPython\n Home =  {}  \nuv = 0.12.15\n",
                base.display()
            ),
        )?;
        assert_eq!(base_python(&python)?, base.join(PYTHON_FILENAME));
        Ok(())
    }

    #[test]
    fn a_venv_that_names_no_home_has_no_base_interpreter() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let python = venv_from(tmp.path(), "implementation = CPython\n")?;
        let no_home = base_python(&python);
        assert!(
            matches!(no_home, Err(PythonCommandError::NoHome(_))),
            "{no_home:?}"
        );

        std::fs::remove_file(tmp.path().join("venv").join("pyvenv.cfg"))?;
        let no_cfg = base_python(&python);
        assert!(
            matches!(no_cfg, Err(PythonCommandError::Read { .. })),
            "{no_cfg:?}"
        );
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn the_venv_python_is_the_interpreter_off_windows() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let python = venv_from(tmp.path(), "home = /usr/bin\n")?;
        let command = python_command(&python)?;
        assert_eq!(command.get_program(), python.as_os_str());
        assert_eq!(command.get_envs().count(), 0);
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn the_base_interpreter_runs_for_the_venv_on_windows() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let base = tmp.path().join("cpython-3.11");
        let python = venv_from(tmp.path(), &format!("home = {}\n", base.display()))?;
        let command = python_command(&python)?;
        assert_eq!(command.get_program(), base.join("python.exe").as_os_str());
        let envs: Vec<_> = command.get_envs().collect();
        assert_eq!(
            envs,
            [(std::ffi::OsStr::new(LAUNCHER_ENV), Some(python.as_os_str()))]
        );
        Ok(())
    }

    #[test]
    fn require_bound_venv_returns_the_root_when_python_exists() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let venv = tmp.path().join("venv");
        std::fs::create_dir_all(venv_bin(&venv))?;
        std::fs::write(venv_python(&venv), "")?;

        let source = UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(venv.to_string_lossy().into_owned())?,
        };
        assert_eq!(require_bound_venv(&source)?, venv);
        Ok(())
    }

    #[test]
    fn require_bound_venv_rejects_a_venv_without_an_interpreter() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let source = UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
        };
        let Err(err) = require_bound_venv(&source) else {
            anyhow::bail!("expected a missing-interpreter error");
        };
        assert!(err.to_string().contains("venv python missing"), "got {err}");
        Ok(())
    }

    #[test]
    fn require_bound_venv_rejects_unbound_sources() -> anyhow::Result<()> {
        for source in [
            UvRuntimeSource::RelativePreinstalled {
                dir: RelativePath::try_new("venv".to_string())?,
            },
            UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.28.4\n".to_string())?,
                install_flags: None,
            },
        ] {
            let Err(err) = require_bound_venv(&source) else {
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
