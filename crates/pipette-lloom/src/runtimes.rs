//! Bound runtime projection: the `lloom-serve` binary under the bound root.
//!
//! Install is `pipette_artifacts::ensure_runtime` (the archive path). This
//! module finds the binary anywhere under the bound `AbsoluteDir`, the way
//! the llama.cpp runner finds `llama-server`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{LloomServeSource, Runtime};

/// The binary the runner spawns.
pub const LLOOM_SERVE: &str = "lloom-serve";

/// The bound `lloom-serve` binary: `LloomServeMacos` + `AbsoluteDir` after bind.
pub fn require_lloom_serve(req: &RunRequest) -> anyhow::Result<PathBuf> {
    let root = bound_absolute_install_root(&req.runtime.bound)?;
    find_tool(root, LLOOM_SERVE)
}

fn bound_absolute_install_root(bound: &Runtime) -> anyhow::Result<&Path> {
    let Runtime::LloomServeMacos(inner) = bound else {
        anyhow::bail!(
            "expected lloom_serve_macos, got `{}`",
            bound.headless_token()
        );
    };
    match &inner.source {
        LloomServeSource::AbsoluteDir { dir } => Ok(Path::new(dir.as_ref())),
        other => anyhow::bail!("expected AbsoluteDir (after bind_under); got {other:?}"),
    }
}

fn find_tool(root: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|n| n.to_str()) == Some(name) {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "`{name}` not found under the bound lloom-serve root {}",
        root.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_binary_anywhere_under_the_root() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let nested = tmp.path().join("lloom-serve-macos-arm64").join("bin");
        fs::create_dir_all(&nested)?;
        fs::write(nested.join(LLOOM_SERVE), b"#!/bin/sh\n")?;
        let found = find_tool(tmp.path(), LLOOM_SERVE)?;
        assert_eq!(found, nested.join(LLOOM_SERVE));
        assert!(find_tool(tmp.path(), "llama-server").is_err());
        Ok(())
    }
}
