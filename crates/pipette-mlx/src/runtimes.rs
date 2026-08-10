//! Bound MLX runtime projection for a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_runtime`. This module finds
//! the venv `python` under a bound `AbsolutePreinstalled` dir.

use std::path::PathBuf;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::RuntimeType;
use pipette_venv::{require_bound_venv, venv_python};

/// Bound MLX venv python: `MlxMacosPipette` + `AbsolutePreinstalled` after bind.
pub fn require_mlx_python(req: &RunRequest) -> anyhow::Result<PathBuf> {
    let venv = require_bound_venv(&req.runtime.bound, &[RuntimeType::MlxMacosPipette])?;
    Ok(venv_python(&venv))
}
