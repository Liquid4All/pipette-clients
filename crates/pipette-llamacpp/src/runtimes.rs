//! The tools under a bound llama.cpp runtime, each resolved from the plan's
//! [`Runtime`] to its binary under the runtime's `AbsoluteDir`, and the
//! environment they launch with.
//!
//! Install is `pipette_artifacts::ensure_runtime`.

use std::{
    ffi::{OsStr, OsString},
    fs,
    ops::Deref,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::{
    EnvForward, LlamaCppFlavor, LlamacppCliStockTools,
    LlamacppCliStockToolsExtraRuntimeConfiguration, LlamacppCliStockToolsSource,
    LlamacppFlashAttention, LlamacppLoadMode, Runtime,
};
use pipette_subprocess::apply_envs;

/// `llama-server` under a bound runtime's install.
#[derive(Debug, Clone)]
pub struct ResolvedLlamaServer(PathBuf);

impl TryFrom<&Runtime> for ResolvedLlamaServer {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        find_tool(bound_absolute_install_root(runtime)?, "llama-server").map(Self)
    }
}

impl Deref for ResolvedLlamaServer {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A `llama-server` path no test runs, for tests of what a launch renders.
#[cfg(test)]
pub(crate) fn test_llama_server() -> ResolvedLlamaServer {
    ResolvedLlamaServer(PathBuf::from("llama-server"))
}

/// A `llama-bench` path no test runs, for tests of what a run renders.
#[cfg(test)]
pub(crate) fn test_llama_bench() -> ResolvedLlamaBench {
    ResolvedLlamaBench(PathBuf::from("llama-bench"))
}

/// A `llama-batched-bench` path no test runs, for tests of what a run renders.
#[cfg(test)]
pub(crate) fn test_llama_batched_bench() -> ResolvedLlamaBatchedBench {
    ResolvedLlamaBatchedBench(PathBuf::from("llama-batched-bench"))
}

/// `llama-bench` under a bound runtime's install.
#[derive(Debug, Clone)]
pub struct ResolvedLlamaBench(PathBuf);

impl TryFrom<&Runtime> for ResolvedLlamaBench {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        find_tool(bound_absolute_install_root(runtime)?, "llama-bench").map(Self)
    }
}

impl Deref for ResolvedLlamaBench {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// `llama-batched-bench` under a bound runtime's install.
#[derive(Debug, Clone)]
pub struct ResolvedLlamaBatchedBench(PathBuf);

impl TryFrom<&Runtime> for ResolvedLlamaBatchedBench {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        find_tool(bound_absolute_install_root(runtime)?, "llama-batched-bench").map(Self)
    }
}

impl Deref for ResolvedLlamaBatchedBench {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// The flavor a bound llama.cpp runtime was built for, by which the peak
/// benchmarks pick their memory sampler.
pub fn flavor(runtime: &Runtime) -> anyhow::Result<&LlamaCppFlavor> {
    let Runtime::LlamacppCliStockTools(LlamacppCliStockTools { flavor, .. }) = runtime else {
        anyhow::bail!(
            "expected llamacpp_cli_stock_tools, got `{}`",
            runtime.headless_token()
        );
    };
    Ok(flavor)
}

/// A cell that set both `mmap` and `load_mode`: two answers to one question.
#[derive(Debug, thiserror::Error)]
#[error(
    "the runtime flags set both `mmap` ({mmap}) and `load_mode` ({}); set `load_mode` alone",
    .load_mode.as_cli_word()
)]
pub struct LoadModeConflictError {
    mmap: bool,
    load_mode: LlamacppLoadMode,
}

/// Refuses a cell that sets both `mmap` and `load_mode`, and warns on `mmap`,
/// upstream's deprecated spelling. It stays accepted: on a build before
/// `b10105`, which has no `--load-mode`, `mmap = true` is the only way to run.
pub fn check_load_mode(
    mmap: Option<bool>,
    load_mode: Option<LlamacppLoadMode>,
) -> Result<(), LoadModeConflictError> {
    match (mmap, load_mode) {
        (Some(mmap), Some(load_mode)) => Err(LoadModeConflictError { mmap, load_mode }),
        (Some(mmap), None) => {
            tracing::warn!(
                mmap,
                "the `mmap` runtime flag is deprecated: set `load_mode` instead (`mmap = false` \
                 is `load_mode = \"none\"`, `mmap = true` is `load_mode = \"mmap\"`); keep \
                 `mmap` only for a llama.cpp build before b10105"
            );
            Ok(())
        }
        (None, _) => Ok(()),
    }
}

/// The load mode a cell runs with: its own, or `none` (weights pinned in RAM)
/// when it set neither `mmap` nor `load_mode`, spelled `--load-mode none` rather
/// than the deprecated `--mmap 0`. `--load-mode` needs `b10105` or later, so a
/// cell on an older build or a fork opts out with `mmap = true`, which renders
/// nothing. One that set both never gets here ([`check_load_mode`]).
pub fn get_load_mode(
    mmap: Option<bool>,
    load_mode: Option<LlamacppLoadMode>,
) -> Option<LlamacppLoadMode> {
    match (mmap, load_mode) {
        (None, None) => Some(LlamacppLoadMode::Read),
        (_, load_mode) => load_mode,
    }
}

/// The knobs every llama.cpp cell carries, and its `raw` argv: what
/// llama-server, llama-bench and llama-batched-bench all take.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommonKnobs<'a> {
    pub threads: Option<u32>,
    pub number_gpu_layers: Option<u32>,
    pub mmap: Option<bool>,
    pub load_mode: Option<LlamacppLoadMode>,
    pub flash_attention: Option<LlamacppFlashAttention>,
    pub raw: &'a [String],
}

impl CommonKnobs<'_> {
    /// `--threads`, `--n-gpu-layers` and `--flash-attn` for the knobs set: every
    /// tool spells these alike. `raw` and the load knobs are the tool's to render.
    pub fn scalar_tokens(&self) -> Vec<String> {
        [
            ("--threads", self.threads.map(|n| n.to_string())),
            (
                "--n-gpu-layers",
                self.number_gpu_layers.map(|n| n.to_string()),
            ),
            (
                "--flash-attn",
                self.flash_attention.map(|fa| fa.as_str().to_owned()),
            ),
        ]
        .into_iter()
        .filter_map(|(flag, value)| Some([flag.to_owned(), value?]))
        .flatten()
        .collect()
    }

    /// The knobs in llama.cpp's common arguments (`common/arg.cpp`), which
    /// llama-server and llama-batched-bench parse; `raw` is left to the caller.
    ///
    /// `--load-mode` wins; otherwise only a bare `--no-mmap` renders, as the
    /// tools mmap by default. Never both: the parser keeps whichever came last,
    /// and [`canonicalize_flag_order`] sorts these tokens.
    pub fn common_arg_tokens(&self) -> Vec<String> {
        let mut tokens = self.scalar_tokens();
        match (self.load_mode, self.mmap) {
            (Some(mode), _) => {
                tokens.extend(["--load-mode".to_owned(), mode.as_cli_word().to_owned()]);
            }
            (None, Some(false)) => tokens.push("--no-mmap".to_owned()),
            (None, _) => {}
        }
        tokens
    }
}

/// Install root from a bound plan [`Runtime`] (`AbsoluteDir` after bind_under).
fn bound_absolute_install_root(bound: &Runtime) -> anyhow::Result<&Path> {
    let Runtime::LlamacppCliStockTools(LlamacppCliStockTools { source, .. }) = bound else {
        anyhow::bail!(
            "expected llamacpp_cli_stock_tools, got `{}`",
            bound.headless_token()
        );
    };
    match source {
        LlamacppCliStockToolsSource::AbsoluteDir { dir } => Ok(Path::new(dir.as_ref())),
        other @ (LlamacppCliStockToolsSource::GithubRelease(_)
        | LlamacppCliStockToolsSource::RemoteArchive { .. }
        | LlamacppCliStockToolsSource::VaultRemoteArchive { .. }
        | LlamacppCliStockToolsSource::RelativeDir { .. }) => {
            anyhow::bail!("expected AbsoluteDir (after bind_under); got {other:?}")
        }
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
        "runtime tool `{name}` not found under {}: the install is incomplete, or the build \
         does not ship this tool",
        root.display()
    )
}

/// Point the dynamic linker at the libraries beside `binary_path` on Linux and
/// Android: the binary's directory, and the sibling `lib/` of a `cmake
/// --install` package (upstream's Snapdragon builds) ahead of any inherited
/// `LD_LIBRARY_PATH`. macOS needs nothing (upstream binaries carry an
/// `@loader_path` rpath), and nothing is done on Windows yet.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn apply_dylib_search_env(cmd: &mut Command, binary_path: &Path) {
    let bin_dir = binary_path.parent().unwrap_or_else(|| Path::new("."));
    let lib_dir = package_lib_dir(bin_dir);

    let mut value = OsString::from(bin_dir);
    if let Some(dir) = &lib_dir {
        value.push(":");
        value.push(dir);
    }
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH")
        && !existing.is_empty()
    {
        value.push(":");
        value.push(existing);
    }
    cmd.env("LD_LIBRARY_PATH", value);

    // The Hexagon NPU's device-side libraries are loaded by FastRPC, which
    // finds them by bare name on `ADSP_LIBRARY_PATH` alone. One directory, as
    // upstream's launchers pass it; inert on a build without the backend.
    cmd.env("ADSP_LIBRARY_PATH", lib_dir.as_deref().unwrap_or(bin_dir));
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn apply_dylib_search_env(_cmd: &mut Command, _binary_path: &Path) {}

/// The `cmake --install` package `lib/` beside `bin_dir`. Only a tool in `bin/`
/// has one: in a flat archive the parent's `lib/` is outside the runtime.
pub fn package_lib_dir(bin_dir: &Path) -> Option<PathBuf> {
    bin_dir
        .file_name()
        .filter(|name| *name == "bin")
        .and_then(|_| bin_dir.parent())
        .map(|prefix| prefix.join("lib"))
        .filter(|dir| dir.is_dir())
}

/// A llama.cpp tool's environment, in the order that matters: the client's
/// configuration, then the library search paths, then the cell's forwards,
/// which can override a search path for a layout nothing here anticipated.
pub fn apply_launch_env(
    cmd: &mut Command,
    binary_path: &Path,
    envs: &[EnvForward],
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) {
    pipette_subprocess::apply_client_config(cmd, client_config);
    apply_dylib_search_env(cmd, binary_path);
    apply_envs(cmd, envs);
}

/// `name` as a tool launched through [`apply_launch_env`] sees it: what that
/// sets, or else what the tool inherits from pipette. On Windows a name matches
/// in any case, as the tool would read it.
pub fn launch_env(
    name: &str,
    binary_path: &Path,
    envs: &[EnvForward],
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> Option<OsString> {
    let mut cmd = Command::new(binary_path);
    apply_launch_env(&mut cmd, binary_path, envs, client_config);
    let same = |key: &OsStr| {
        if cfg!(windows) {
            key.eq_ignore_ascii_case(name)
        } else {
            key == name
        }
    };
    cmd.get_envs().find(|(key, _)| same(key)).map_or_else(
        || std::env::var_os(name),
        |(_, value)| value.map(OsStr::to_owned),
    )
}

/// `name` as llama.cpp looks a flag up: a long name with `_` for `-`
/// (`common/arg.cpp`), a short one as it is.
pub fn canonical_flag_name(name: &str) -> String {
    if name.starts_with("--") {
        name.replace('_', "-")
    } else {
        name.to_owned()
    }
}

/// `flags` sorted by flag name, each with its values, so two runs of the same
/// flag set record the same argv. Stable, so a repeated flag keeps last-wins. A
/// value starting with `-` (`--keep -1`) opens its own group and is sorted away
/// from its flag.
pub fn canonicalize_flag_order(flags: &[String]) -> Vec<String> {
    // A bare value at the head, with no flag before it, opens its own group so
    // nothing is lost.
    let mut groups = flags.iter().fold(
        Vec::<(&String, Vec<&String>)>::new(),
        |mut groups, token| {
            match groups.last_mut() {
                Some((_, values)) if !token.starts_with('-') => values.push(token),
                _ => groups.push((token, Vec::new())),
            }
            groups
        },
    );
    groups.sort_by(|(a, _), (b, _)| flag_key(a).cmp(flag_key(b)));
    groups
        .into_iter()
        .flat_map(|(head, values)| std::iter::once(head).chain(values))
        .cloned()
        .collect()
}

/// Sort key for [`canonicalize_flag_order`]: the part of a flag
/// token before any `=`, so `--ctx-size` and `--ctx-size=8192` sort
/// together.
fn flag_key(token: &str) -> &str {
    token.split_once('=').map(|(name, _)| name).unwrap_or(token)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsRelease,
        NonEmptyString, RepositoryUrl,
    };

    use super::*;

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

    fn server(runtime: &Runtime) -> anyhow::Result<PathBuf> {
        Ok(ResolvedLlamaServer::try_from(runtime)?.to_path_buf())
    }

    fn bench(runtime: &Runtime) -> anyhow::Result<PathBuf> {
        Ok(ResolvedLlamaBench::try_from(runtime)?.to_path_buf())
    }

    fn batched_bench(runtime: &Runtime) -> anyhow::Result<PathBuf> {
        Ok(ResolvedLlamaBatchedBench::try_from(runtime)?.to_path_buf())
    }

    #[rstest]
    #[case::server_at_root("llama-server", server)]
    #[case::server_nested("bin/release/llama-server", server)]
    #[case::bench("llama-bench", bench)]
    #[case::batched_bench("bin/llama-batched-bench", batched_bench)]
    fn resolving_a_tool_finds_it_under_the_install_root(
        #[case] relative: &str,
        #[case] resolve: fn(&Runtime) -> anyhow::Result<PathBuf>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), relative)?;
        assert_eq!(resolve(&llama_absolute_dir(tmp.path())?)?, expected);
        Ok(())
    }

    #[test]
    fn resolved_llama_server_errors_when_missing() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        // Only llama-bench is present.
        write_tool(tmp.path(), "llama-bench")?;
        let runtime = llama_absolute_dir(tmp.path())?;
        let msg = match ResolvedLlamaServer::try_from(&runtime) {
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
    fn resolving_a_tool_rejects_non_llamacpp_bound_runtime() -> anyhow::Result<()> {
        let runtime = Runtime::AppleFoundation(Default::default());
        let msg = match ResolvedLlamaServer::try_from(&runtime) {
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
    fn resolving_a_tool_rejects_non_absolute_dir_source() -> anyhow::Result<()> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b5000".to_owned())?,
                asset: None,
            }),
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        });
        let msg = match ResolvedLlamaBench::try_from(&runtime) {
            Ok(p) => anyhow::bail!("expected AbsoluteDir error, found {}", p.display()),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("AbsoluteDir") && msg.contains("bind_under"),
            "unexpected error: {msg}"
        );
        Ok(())
    }

    /// llama.cpp maps by default, so the overlay pins the weights where the cell left the
    /// choice open, in `load_mode`, the spelling upstream kept. A cell that authored
    /// either knob keeps its answer, so the record carries one.
    #[rstest]
    #[case::nothing_set(None, None, Some(LlamacppLoadMode::Read))]
    #[case::mmap_on(Some(true), None, None)]
    #[case::mmap_off(Some(false), None, None)]
    #[case::load_mode_set(None, Some(LlamacppLoadMode::Mmap), Some(LlamacppLoadMode::Mmap))]
    fn get_load_mode_pins_the_weights_unless_the_cell_said(
        #[case] mmap: Option<bool>,
        #[case] load_mode: Option<LlamacppLoadMode>,
        #[case] expected: Option<LlamacppLoadMode>,
    ) {
        assert_eq!(get_load_mode(mmap, load_mode), expected);
    }

    #[rstest]
    #[case::neither(None, None, Ok(()))]
    #[case::mmap(Some(false), None, Ok(()))]
    #[case::load_mode(None, Some(LlamacppLoadMode::Mmap), Ok(()))]
    #[case::both(
        Some(true),
        Some(LlamacppLoadMode::Read),
        Err("both `mmap` (true) and `load_mode` (none)")
    )]
    fn one_load_mode_knob_at_most(
        #[case] mmap: Option<bool>,
        #[case] load_mode: Option<LlamacppLoadMode>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        match (check_load_mode(mmap, load_mode), expected) {
            (Ok(()), Ok(())) => {}
            (Err(err), Err(message)) => {
                assert!(err.to_string().contains(message), "{err}");
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn resolved_llama_server_accepts_exe_suffix_on_windows() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let expected = write_tool(tmp.path(), "llama-server.exe")?;
        assert_eq!(server(&llama_absolute_dir(tmp.path())?)?, expected);
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

#[cfg(test)]
mod launch_env_tests {
    use std::ffi::OsStr;

    use rstest::rstest;

    use pipette_plan_types::EnvForward;

    use super::*;

    /// The variables `cmd` would launch with, as `(name, value)` pairs. `None`
    /// for a value means the entry was *removed* from the child's environment,
    /// which `Command` models distinctly from setting it empty.
    fn env_of(cmd: &Command, name: &str) -> Option<Option<String>> {
        cmd.get_envs()
            .find(|(k, _)| *k == OsStr::new(name))
            .map(|(_, v)| -> Option<String> { v.and_then(|v| v.to_str().map(str::to_owned)) })
    }

    /// The client's envs reach the tool alongside the cell's, and an
    /// unconfigured run leaves the variable to the host, neither set empty nor
    /// removed, so an operator's export still reaches the tool.
    #[rstest]
    #[case::configured(Some("CUDA_VISIBLE_DEVICES=1"), Some(Some("1")))]
    #[case::unconfigured(None, None)]
    fn the_clients_envs_reach_the_command(
        #[case] entry: Option<&str>,
        #[case] expected: Option<Option<&str>>,
    ) -> anyhow::Result<()> {
        let client_config = entry
            .map(|e| {
                anyhow::Ok(LlamacppCliStockToolsExtraRuntimeConfiguration::from(vec![
                    EnvForward::try_new(e.to_owned())?,
                ]))
            })
            .transpose()?;
        let mut cmd = Command::new("true");
        apply_launch_env(
            &mut cmd,
            Path::new("/opt/rt/bin/llama-bench"),
            &[EnvForward::try_new("GGML_HEXAGON_OPPOLL=1".to_owned())?],
            client_config.as_ref(),
        );
        assert_eq!(
            env_of(&cmd, "CUDA_VISIBLE_DEVICES"),
            expected.map(|v| v.map(str::to_owned))
        );
        assert_eq!(
            env_of(&cmd, "GGML_HEXAGON_OPPOLL"),
            Some(Some("1".to_owned()))
        );
        Ok(())
    }

    /// The tool sees the cell's forward over the client's, and pipette's own
    /// value where neither sets one.
    #[rstest]
    #[case::the_cell_wins(
        Some("PIPETTE_PROBE_ENV=CPU"),
        Some("PIPETTE_PROBE_ENV=GPU"),
        Some("GPU")
    )]
    #[case::the_client_sets_it(Some("PIPETTE_PROBE_ENV=CPU"), None, Some("CPU"))]
    #[case::neither_sets_it(None, None, None)]
    fn a_launch_env_is_what_the_tool_sees(
        #[case] client: Option<&str>,
        #[case] cell: Option<&str>,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let forwards = |entry: Option<&str>| -> anyhow::Result<Vec<EnvForward>> {
            Ok(entry
                .map(|entry| EnvForward::try_new(entry.to_owned()))
                .transpose()?
                .into_iter()
                .collect())
        };
        let client_config = LlamacppCliStockToolsExtraRuntimeConfiguration::from(forwards(client)?);
        let program = Path::new("/opt/rt/bin/llama-bench");
        let envs = forwards(cell)?;
        assert_eq!(
            launch_env("PIPETTE_PROBE_ENV", program, &envs, Some(&client_config)),
            expected.map(OsString::from)
        );
        assert_eq!(
            launch_env("PATH", program, &envs, Some(&client_config)),
            std::env::var_os("PATH"),
            "an unset variable is inherited"
        );
        Ok(())
    }

    #[cfg(windows)]
    #[test]
    fn a_launch_env_matches_any_case_on_windows() -> anyhow::Result<()> {
        let envs = vec![EnvForward::try_new(r"appdata=C:\cell".to_owned())?];
        assert_eq!(
            launch_env("APPDATA", Path::new(r"C:\rt\llama-server.exe"), &envs, None),
            Some(OsString::from(r"C:\cell"))
        );
        Ok(())
    }

    /// A cell's forward beats the derived search paths: that is what makes `envs`
    /// an escape hatch.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn a_cell_forward_overrides_a_derived_search_path() -> anyhow::Result<()> {
        let mut cmd = Command::new("true");
        apply_launch_env(
            &mut cmd,
            Path::new("/opt/rt/bin/llama-bench"),
            &[EnvForward::try_new(
                "ADSP_LIBRARY_PATH=/somewhere/else".to_owned(),
            )?],
            None,
        );
        assert_eq!(
            env_of(&cmd, "ADSP_LIBRARY_PATH"),
            Some(Some("/somewhere/else".to_owned()))
        );
        Ok(())
    }

    /// `bin/` + `lib/` (a `cmake --install` package, which is what upstream's
    /// Snapdragon build produces): the libraries live in the sibling `lib/`, so
    /// both variables must point there. `LD_LIBRARY_PATH` keeps the tool's own
    /// directory too, since a flat archive needs it.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn a_bin_lib_package_resolves_to_the_sibling_lib_dir() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let bin = root.path().join("bin");
        let lib = root.path().join("lib");
        std::fs::create_dir_all(&bin)?;
        std::fs::create_dir_all(&lib)?;

        let mut cmd = Command::new("true");
        apply_dylib_search_env(&mut cmd, &bin.join("llama-bench"));

        assert_eq!(
            env_of(&cmd, "ADSP_LIBRARY_PATH"),
            Some(Some(lib.display().to_string()))
        );
        let search = env_of(&cmd, "LD_LIBRARY_PATH")
            .flatten()
            .unwrap_or_default();
        assert!(search.contains(&lib.display().to_string()), "{search}");
        assert!(search.contains(&bin.display().to_string()), "{search}");
        Ok(())
    }

    /// A flat archive (upstream's release-tarball layout) has no `lib/`; the
    /// `<parent>/lib` beside it is outside the runtime.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn a_flat_archive_resolves_to_the_tool_dir() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let flat = root.path().join("blobs");
        std::fs::create_dir_all(&flat)?;
        // A decoy `lib/` beside the tool's dir.
        std::fs::create_dir_all(root.path().join("lib"))?;

        let mut cmd = Command::new("true");
        apply_dylib_search_env(&mut cmd, &flat.join("llama-bench"));

        assert_eq!(
            env_of(&cmd, "ADSP_LIBRARY_PATH"),
            Some(Some(flat.display().to_string())),
            "the decoy `lib/` is not part of this runtime"
        );
        Ok(())
    }
}

#[cfg(test)]
mod flag_order_tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(&[], &[])]
    #[case::single_bare(&["--no-warmup"], &["--no-warmup"])]
    #[case::single_with_value(&["--threads", "8"], &["--threads", "8"])]
    #[case::sorts_two_pairs(
        &["--threads", "8", "--mmap", "0"],
        &["--mmap", "0", "--threads", "8"],
    )]
    #[case::mixes_bare_and_valued(
        &["--no-warmup", "--ctx-size", "8448", "--no-mmap"],
        &["--ctx-size", "8448", "--no-mmap", "--no-warmup"],
    )]
    #[case::equals_form_groups_with_long_form(
        &["--ctx-size=8192", "--mmap", "0"],
        &["--ctx-size=8192", "--mmap", "0"],
    )]
    #[case::short_form_sorts_after_long_double_dash(
        &["-c", "4096", "--no-warmup"],
        &["--no-warmup", "-c", "4096"],
    )]
    #[case::repeated_flag_keeps_insertion_order(
        &["--mmap", "0", "--threads", "4", "--mmap", "1"],
        &["--mmap", "0", "--mmap", "1", "--threads", "4"],
    )]
    #[case::gnu_double_dash_sorts_as_flag(
        // llama.cpp does not honor `--`, so it sorts as any flag-shaped token.
        &["--mmap", "0", "--", "foo"],
        &["--", "foo", "--mmap", "0"],
    )]
    fn canonicalize_flag_order_cases(#[case] input: &[&str], #[case] expected: &[&str]) {
        let input: Vec<String> = input.iter().map(|s| s.to_string()).collect();
        let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(canonicalize_flag_order(&input), expected);
    }
}

#[cfg(test)]
mod common_knobs_tests {
    use rstest::rstest;

    use super::*;

    /// Each knob on its own flag, in the common arguments' spelling. `--load-mode`
    /// wins over the deprecated boolean, which renders only as bare `--no-mmap`.
    #[rstest]
    #[case::nothing_set(CommonKnobs::default(), &[])]
    // Knobs that share a type, set apart so two cannot trade places.
    #[case::every_scalar_set(
        CommonKnobs {
            threads: Some(3),
            number_gpu_layers: Some(7),
            flash_attention: Some(LlamacppFlashAttention::On),
            ..CommonKnobs::default()
        },
        &["--threads", "3", "--n-gpu-layers", "7", "--flash-attn", "on"]
    )]
    #[case::load_mode_alone(
        CommonKnobs { load_mode: Some(LlamacppLoadMode::DirectIo), ..CommonKnobs::default() },
        &["--load-mode", "dio"]
    )]
    #[case::load_mode_beats_the_boolean(
        CommonKnobs { mmap: Some(false), load_mode: Some(LlamacppLoadMode::Mmap), ..CommonKnobs::default() },
        &["--load-mode", "mmap"]
    )]
    #[case::mmap_off(CommonKnobs { mmap: Some(false), ..CommonKnobs::default() }, &["--no-mmap"])]
    #[case::mmap_on_is_the_default(CommonKnobs { mmap: Some(true), ..CommonKnobs::default() }, &[])]
    fn the_common_arguments_render_each_knob_on_its_own_flag(
        #[case] knobs: CommonKnobs<'static>,
        #[case] expected: &[&str],
    ) {
        assert_eq!(knobs.common_arg_tokens(), expected);
    }

    /// The raw argv is the caller's to place, so neither rendering carries it.
    #[test]
    fn raw_stays_out_of_the_rendered_knobs() {
        let raw = vec!["--poll".to_owned(), "0".to_owned()];
        let knobs = CommonKnobs {
            threads: Some(3),
            raw: &raw,
            ..CommonKnobs::default()
        };
        assert_eq!(knobs.scalar_tokens(), ["--threads", "3"]);
        assert_eq!(knobs.common_arg_tokens(), ["--threads", "3"]);
    }
}
