//! Bound-runtime tool resolution for a prepared [`RunRequest`].
//!
//! Install/ensure lives in ops (`ensure_runtime`). This module finds
//! `llama-server` / `llama-bench` under a bound `AbsoluteDir`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{LlamacppCliStockToolsSource, Runtime};

/// Resolve `llama-server` under the bound runtime install and verify it exists.
///
/// Expects `req.runtime.bound` to be `LlamacppCliStockTools` + `AbsoluteDir`
/// (after ensure/bind).
pub fn require_llama_server(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_tool(req, "llama-server")
}

/// Resolve `llama-bench` under the bound runtime install and verify it exists.
pub fn require_llama_bench(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_tool(req, "llama-bench")
}

fn require_bound_tool(req: &RunRequest, name: &str) -> anyhow::Result<PathBuf> {
    let root = bound_absolute_install_root(&req.runtime.bound)?;
    find_tool(root, name)
}

/// Install root from a bound plan [`Runtime`] (`AbsoluteDir` after bind_under).
fn bound_absolute_install_root(bound: &Runtime) -> anyhow::Result<&Path> {
    let Runtime::LlamacppCliStockTools(inner) = bound else {
        anyhow::bail!(
            "expected llamacpp_cli_stock_tools, got `{}`",
            bound.headless_token()
        );
    };
    match &inner.source {
        LlamacppCliStockToolsSource::AbsoluteDir { dir } => Ok(Path::new(dir.as_ref())),
        other => anyhow::bail!("expected AbsoluteDir (after bind_under); got {other:?}"),
    }
}

/// Walk `root` for a tool binary (`name` / `name.exe` on Windows).
fn find_tool(root: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
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
            if path.file_name().and_then(|n| n.to_str()) == Some(name)
                || path.file_name().and_then(|n| n.to_str()) == Some(exe.as_str())
            {
                return Ok(path);
            }
        }
    }
    anyhow::bail!(
        "runtime tool `{name}` not found under {} (install may be corrupt)",
        root.display()
    )
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{
        default_repository_url, AbsolutePath, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, Model, ModelSource, NonEmptyString, Runtime, SourceRepository,
    };

    use super::*;

    fn dummy_model() -> anyhow::Result<Model> {
        // Absolute placeholder only; runtimes tests never touch the model axis.
        let dir = AbsolutePath::try_new(if cfg!(windows) {
            r"C:\tmp\m".to_owned()
        } else {
            "/tmp/m".to_owned()
        })?;
        Ok(Model::Mlx(pipette_plan_types::Mlx {
            source: ModelSource::AbsoluteDir { dir },
        }))
    }

    fn req_with_bound_runtime(bound: Runtime) -> anyhow::Result<RunRequest> {
        let model = dummy_model()?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(bound),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "p".into(),
                parameter_prefill_tokens: 1,
            }),
        })
    }

    fn llama_absolute_dir(root: &Path) -> anyhow::Result<Runtime> {
        let dir = AbsolutePath::try_new(root.display().to_string())?;
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::AbsoluteDir { dir },
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }

    fn write_tool(root: &Path, relative: &str) -> anyhow::Result<PathBuf> {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, b"#!/bin/sh\nexit 0\n")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&path, perms)?;
        }
        Ok(path)
    }

    #[test]
    fn require_llama_server_finds_tool_at_install_root() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), "llama-server")?;
        let req = req_with_bound_runtime(llama_absolute_dir(tmp.path())?)?;
        assert_eq!(require_llama_server(&req)?, expected);
        Ok(())
    }

    #[test]
    fn require_llama_server_finds_nested_tool() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), "bin/release/llama-server")?;
        let req = req_with_bound_runtime(llama_absolute_dir(tmp.path())?)?;
        assert_eq!(require_llama_server(&req)?, expected);
        Ok(())
    }

    #[test]
    fn require_llama_bench_finds_tool() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), "llama-bench")?;
        let req = req_with_bound_runtime(llama_absolute_dir(tmp.path())?)?;
        assert_eq!(require_llama_bench(&req)?, expected);
        Ok(())
    }

    #[test]
    fn require_llama_server_errors_when_missing() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        // Only bench present — server must still fail.
        write_tool(tmp.path(), "llama-bench")?;
        let req = req_with_bound_runtime(llama_absolute_dir(tmp.path())?)?;
        let msg = match require_llama_server(&req) {
            Ok(p) => anyhow::bail!("expected missing server, found {}", p.display()),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("llama-server") && msg.contains("not found"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    #[test]
    fn require_tool_rejects_non_llamacpp_bound_runtime() -> anyhow::Result<()> {
        let req = req_with_bound_runtime(Runtime::AppleFoundation(Default::default()))?;
        let msg = match require_llama_server(&req) {
            Ok(p) => anyhow::bail!("expected kind error, found {}", p.display()),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("llamacpp_cli_stock_tools"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    #[test]
    fn require_tool_rejects_non_absolute_dir_source() -> anyhow::Result<()> {
        let bound = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b5000".to_owned())?,
            }),
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        });
        let req = req_with_bound_runtime(bound)?;
        let msg = match require_llama_bench(&req) {
            Ok(p) => anyhow::bail!("expected AbsoluteDir error, found {}", p.display()),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("AbsoluteDir") && msg.contains("bind_under"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn require_llama_server_accepts_exe_suffix_on_windows() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), "llama-server.exe")?;
        let req = req_with_bound_runtime(llama_absolute_dir(tmp.path())?)?;
        assert_eq!(require_llama_server(&req)?, expected);
        Ok(())
    }

    #[test]
    fn find_tool_errors_when_root_is_not_a_directory() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let file = tmp.path().join("not-a-dir");
        fs::write(&file, b"x")?;
        let msg = match find_tool(&file, "llama-server") {
            Ok(p) => anyhow::bail!("expected read error, found {}", p.display()),
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("failed to read"), "unexpected error: {msg}");
        Ok(())
    }
}
