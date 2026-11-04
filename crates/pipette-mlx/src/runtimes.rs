//! The venv python of a bound MLX runtime, resolved from the plan's
//! [`Runtime`].
//!
//! Install is `pipette_artifacts::ensure_runtime`.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use pipette_plan_types::{Runtime, RuntimeType};
use pipette_venv::{require_bound_venv, venv_python};

/// The venv python of a bound `mlx_macos_pipette` runtime.
#[derive(Debug)]
pub struct ResolvedMlxPython(PathBuf);

impl TryFrom<&Runtime> for ResolvedMlxPython {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        let Runtime::MlxMacosPipette(rt) = runtime else {
            anyhow::bail!(
                "expected mlx_macos_pipette, got `{}`",
                RuntimeType::of(runtime)
            );
        };
        Ok(Self(venv_python(&require_bound_venv(&rt.source)?)))
    }
}

impl Deref for ResolvedMlxPython {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

#[cfg(all(test, target_os = "macos"))]
pub(crate) mod test_support {
    use std::path::{Path, PathBuf};

    use anyhow::Context;
    use serde_json::json;

    use pipette_plan_types::{Model, Runtime};
    use pipette_venv::{venv_bin, venv_python};

    use super::ResolvedMlxPython;
    use crate::models::ResolvedMlxModel;

    /// A bound `mlx_macos_pipette` runtime whose venv holds an interpreter
    /// file, never run.
    pub(crate) fn bound_runtime(dir: &Path) -> anyhow::Result<Runtime> {
        let venv = dir.join("venv");
        std::fs::create_dir_all(venv_bin(&venv))?;
        std::fs::write(venv_python(&venv), b"")?;
        Ok(serde_json::from_value(json!({
            "type": "mlx_macos_pipette",
            "version": "0.20.0",
            "flavor": "macos-arm64",
            "source": {"type": "absolute_preinstalled", "dir": venv},
        }))?)
    }

    /// The venv and VL model the live tests name in `PIPETTE_MLX_LIVE_PYTHON`
    /// (`<venv>/bin/python`) and `PIPETTE_MLX_LIVE_TARGET`.
    pub(crate) fn live_vl() -> anyhow::Result<(ResolvedMlxPython, ResolvedMlxModel)> {
        let var = |name: &str| {
            std::env::var(name)
                .map(PathBuf::from)
                .with_context(|| format!("{name} must be set to run the live test"))
        };
        let python = var("PIPETTE_MLX_LIVE_PYTHON")?;
        let venv = python
            .parent()
            .and_then(Path::parent)
            .context("PIPETTE_MLX_LIVE_PYTHON is not <venv>/bin/python")?;
        let runtime: Runtime = serde_json::from_value(json!({
            "type": "mlx_macos_pipette",
            "version": "0.7.1+lfm2-vl-speculative",
            "flavor": "macos-arm64",
            "source": {"type": "absolute_preinstalled", "dir": venv},
        }))?;
        let model: Model = serde_json::from_value(json!({
            "type": "mlx",
            "source": "absolute_dir",
            "dir": var("PIPETTE_MLX_LIVE_TARGET")?,
        }))?;
        Ok((
            ResolvedMlxPython::try_from(&runtime)?,
            ResolvedMlxModel::try_from(&model)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_mlx_python_rejects_another_runtime() -> anyhow::Result<()> {
        let other = Runtime::AppleFoundation(Default::default());
        let Err(err) = ResolvedMlxPython::try_from(&other) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            err.to_string().contains("expected mlx_macos_pipette"),
            "got {err}"
        );
        Ok(())
    }
}
