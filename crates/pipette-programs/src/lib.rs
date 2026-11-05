//! The external programs a pipette client spawns: one list of them, one way to
//! locate one, and one place the failure is raised.
//!
//! "External" means a program a person installs (`uv`, `docker`, a GPU
//! vendor's CLI), not a platform utility (`sh`, `ps`, `powershell`). Each one
//! is a const on [`ExternalProgram`], so what pipette shells out to is one
//! screen of this file. It sits below everything that spawns, so
//! `pipette-artifacts` (pulls images) and `pipette-torch-oai` (runs
//! containers) use one `docker`.
//! `pipette-device` still runs `nvidia-smi` and `lspci` itself: its probe
//! discards every failure, so a shared resolver gives it nothing.
//!
//! Every program is located once at startup, as [`ExternalPrograms`], so a
//! typo in `--uv-path` shows before the first fetch rather than an hour into a
//! sweep. The failure waits in [`ResolvedProgram`] until someone asks for the
//! path, so a host without docker still runs every uv-backed benchmark.
//!
//! Being on `PATH` is weaker than working (a `docker` with no daemon resolves),
//! so each program also carries an argv that asks whether it works, which
//! [`ExternalProgram::check`] runs under a deadline.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use pipette_subprocess::{OutputError, echo_debug, output_within, render};

/// How long a check has to answer before it is killed and reported failed.
///
/// A healthy host answers in under a second; the budget is for the unhealthy
/// one: `docker info` blocks while the daemon starts, and `nvidia-smi` blocks
/// in the kernel when the driver is wedged. Without it `worker --strict` would
/// hang at startup instead of refusing.
const CHECK_TIMEOUT: Duration = Duration::from_secs(15);

/// An external program: what to look for, where to get it, and how to ask it
/// whether it works.
///
/// Consts rather than an enum: no caller dispatches on which program it holds,
/// and each program's facts stay in one literal instead of spread across
/// `match` arms. Private fields keep the set closed to this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalProgram {
    /// Most-preferred first.
    candidates: &'static [&'static str],
    /// Where the program comes from; the operator can already see it is absent.
    install_hint: &'static str,
    /// The argv after the binary. Read-only and cheap, so a check can run at
    /// every worker start.
    check_args: &'static [&'static str],
}

impl ExternalProgram {
    /// Provisions every venv-backed runtime: vLLM, SGLang, MLX, OpenVINO, ONNX.
    pub const UV: Self = Self {
        candidates: &["uv"],
        install_hint: "install it from https://docs.astral.sh/uv/getting-started/installation/",
        check_args: &["--version"],
    };

    /// Pulls and runs the container-backed runtimes.
    pub const DOCKER: Self = Self {
        candidates: &["docker"],
        install_hint: "install Docker Engine",
        // Not `--version`, which the client answers alone: every use of docker
        // here needs a daemon, and `info` is the cheapest call that needs one.
        check_args: &["info", "--format", "{{.ServerVersion}}"],
    };

    /// NVIDIA's CLI: driver preflight.
    pub const NVIDIA_SMI: Self = Self {
        candidates: &["nvidia-smi"],
        install_hint: "install the NVIDIA driver",
        // Goes through NVML, like the GPU preflight, so it catches a kernel
        // module out of sync with its userspace library.
        check_args: &["-L"],
    };

    /// AMD's equivalent, for the ROCm preflight.
    pub const ROCM_SMI: Self = Self {
        candidates: &["rocm-smi"],
        install_hint: "install ROCm",
        check_args: &["--version"],
    };

    /// Host Python, for a caller that needs an interpreter from outside a
    /// runtime's own venv. `python3` first: a bare `python` is still Python 2
    /// on some hosts, and is the name a virtualenv shim takes over.
    pub const PYTHON: Self = Self {
        candidates: &["python3", "python"],
        install_hint: "install Python 3",
        check_args: &["--version"],
    };

    /// Every name it answers to, for a message: `python3/python`.
    pub fn label(self) -> String {
        self.candidates.join("/")
    }

    /// Locate it on `PATH` now, for a caller with no command line to take an
    /// override from. The CLI uses [`ExternalPrograms`] instead.
    pub fn locate(self) -> anyhow::Result<PathBuf> {
        self.resolve(None).path().map(Path::to_path_buf)
    }

    /// Locate it, honouring `explicit`, keeping any failure for first use.
    ///
    /// `explicit` may be the binary or a directory holding it, so both
    /// `--uv-path ~/.local/bin/uv` and `--uv-path ~/.local/bin` work.
    ///
    /// A missing explicit path is not retried against `PATH`: running a
    /// different binary would attribute a measurement to the wrong build.
    pub fn resolve(self, explicit: Option<&Path>) -> ResolvedProgram {
        ResolvedProgram {
            resolved: match explicit {
                Some(given) => self.at_explicit_path(given),
                None => self.search_path(),
            },
        }
    }

    fn at_explicit_path(self, given: &Path) -> Result<PathBuf, String> {
        if given.is_dir() {
            return self
                .candidates
                .iter()
                .find_map(|name| executable_at(&given.join(name)))
                .ok_or_else(|| {
                    format!(
                        "{} not found in the directory {}",
                        self.label(),
                        given.display()
                    )
                });
        }
        executable_at(given)
            .ok_or_else(|| format!("{} not found at {}", self.label(), given.display()))
    }

    fn search_path(self) -> Result<PathBuf, String> {
        let path = std::env::var_os("PATH").unwrap_or_default();
        self.candidates
            .iter()
            .find_map(|name| which(&path, name))
            .ok_or_else(|| self.not_on_path_message())
    }

    fn not_on_path_message(self) -> String {
        format!("{} not found on PATH; {}", self.label(), self.install_hint)
    }

    /// Run its check argv and take the exit status as the answer.
    ///
    /// Infallible: a failure is a row printed beside the programs that passed,
    /// so one pass reports the whole host.
    pub fn check(self, resolved: &ResolvedProgram) -> ProgramCheck {
        self.check_within(resolved, CHECK_TIMEOUT)
    }

    /// [`Self::check`] with the deadline as a parameter, so a test need not
    /// wait out the real one.
    fn check_within(self, resolved: &ResolvedProgram, timeout: Duration) -> ProgramCheck {
        let path = match resolved.path() {
            Ok(path) => path.to_path_buf(),
            // The lookup's message is already the one to act on.
            Err(why) => return ProgramCheck::failed(self, None, why.to_string()),
        };
        let mut cmd = Command::new(&path);
        cmd.args(self.check_args);
        echo_debug(&cmd);
        // Quoted in the failure so the operator can paste it back.
        let rendered = render(&cmd);
        ProgramCheck {
            program: self,
            path: Some(path),
            outcome: ask(cmd, &rendered, timeout),
        }
    }
}

/// Run `cmd` under `timeout`: what it answered, or why it did not.
/// `Command::output` would block forever on a program that hangs, and docker
/// and nvidia-smi are asked precisely because the thing behind them may be
/// wedged.
fn ask(cmd: Command, rendered: &str, timeout: Duration) -> Result<String, String> {
    let output = match output_within(cmd, timeout) {
        Ok(output) => output,
        // Located and still unspawnable: a dangling symlink, or a file without
        // the execute bit.
        Err(OutputError::Spawn(err)) => return Err(format!("`{rendered}` could not run: {err}")),
        Err(err) => {
            let err = anyhow::Error::from(err);
            return Err(format!(
                "`{rendered}` did not answer within {timeout:?} ({err:#})"
            ));
        }
    };
    let (status, stdout, stderr) = (output.status, output.stdout, output.stderr);
    if status.success() {
        // Some of them answer on stderr, so an empty stdout is not silence.
        return Ok(first_line(&stdout)
            .or_else(|| first_line(&stderr))
            .unwrap_or_default());
    }
    // stderr first: where each of them explains itself.
    let detail = first_line(&stderr)
        .or_else(|| first_line(&stdout))
        .unwrap_or_else(|| "no output".to_owned());
    Err(format!("`{rendered}` failed ({status}): {detail}"))
}

/// Lossy, because a working program whose banner is not UTF-8 still passed.
fn first_line(output: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(output);
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

/// The first executable `name` in `path`, a `PATH` value.
fn which(path: &OsStr, name: &str) -> Option<PathBuf> {
    std::env::split_paths(path)
        // A relative entry would resolve against the client's working directory.
        .filter(|dir| dir.is_absolute())
        .filter_map(|dir| executable_at(&dir.join(name)))
        .find(|found| is_executable(found))
}

/// Any execute bit, as `which` demands, so a file on `PATH` that cannot run
/// does not shadow the binary behind it.
#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|meta| meta.permissions().mode() & 0o111 != 0)
}

/// Windows has no execute bit.
#[cfg(not(unix))]
fn is_executable(_: &Path) -> bool {
    true
}

/// `path` itself, or the `.exe` beside it on Windows, so a bare name on `PATH`
/// and a directory override both find `docker.exe`. Only `.exe`: every program
/// in the catalog ships as one there.
fn executable_at(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path.to_path_buf());
    }
    if !cfg!(windows) {
        return None;
    }
    // Appended, not `with_extension`: that would rewrite `python3.11` into
    // `python3.exe`.
    let mut exe = path.as_os_str().to_os_string();
    exe.push(".exe");
    let exe = PathBuf::from(exe);
    exe.is_file().then_some(exe)
}

/// One program as this process located it, or the failure to.
///
/// The failure is a rendered message rather than an `anyhow::Error`, which is
/// not `Clone`: the contexts carrying this are cloned for every artifact fetch.
#[derive(Debug, Clone)]
pub struct ResolvedProgram {
    resolved: Result<PathBuf, String>,
}

impl ResolvedProgram {
    /// A program whose path the caller already knows, skipping the lookup.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            resolved: Ok(path.into()),
        }
    }

    /// The one place a missing program becomes an error, so it lands on the
    /// operation that needed the binary, not on startup.
    pub fn path(&self) -> anyhow::Result<&Path> {
        match &self.resolved {
            Ok(path) => Ok(path.as_path()),
            Err(why) => Err(anyhow::anyhow!("{why}")),
        }
    }

    /// For a caller that has something else to do when it is missing. Anything
    /// that needs the binary should ask for the path.
    pub fn is_resolved(&self) -> bool {
        self.resolved.is_ok()
    }
}

/// What one program's check found: a row of `pipette check`, and the reason
/// `--strict` refuses. It names its program, unlike [`ResolvedProgram`],
/// because a row has to say what it is about even when it passed.
#[derive(Debug)]
pub struct ProgramCheck {
    program: ExternalProgram,
    /// `None` when the lookup never found it.
    path: Option<PathBuf>,
    /// The first line it answered, or why it failed.
    outcome: Result<String, String>,
}

impl ProgramCheck {
    fn failed(program: ExternalProgram, path: Option<PathBuf>, why: String) -> Self {
        Self {
            program,
            path,
            outcome: Err(why),
        }
    }

    pub fn label(&self) -> String {
        self.program.label()
    }

    /// The argv after the binary, for a report that says which question it
    /// asked.
    pub fn check_args(&self) -> String {
        self.program.check_args.join(" ")
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The first line it answered with, or `None` when the check failed.
    pub fn reported(&self) -> Option<&str> {
        self.outcome.as_ref().ok().map(String::as_str)
    }

    pub fn failure(&self) -> Option<&str> {
        self.outcome.as_ref().err().map(String::as_str)
    }

    pub fn passed(&self) -> bool {
        self.outcome.is_ok()
    }

    /// Where it was found and what it answered, or `None` if it failed. Paired
    /// because a pass always ran a binary, so a caller need not handle a pass
    /// without a path.
    pub fn answer(&self) -> Option<(&Path, &str)> {
        Some((self.path.as_deref()?, self.reported()?))
    }

    /// Refuse if any of `checks` failed, naming every one that did, so
    /// `pipette check` and `worker --strict` refuse in the same words.
    pub fn refuse_any_failed(checks: &[Self]) -> anyhow::Result<()> {
        let failures: Vec<String> = checks
            .iter()
            .filter_map(|check| {
                check
                    .failure()
                    .map(|why| format!("{}: {why}", check.label()))
            })
            .collect();
        if failures.is_empty() {
            return Ok(());
        }
        anyhow::bail!("{}", failures.join("\n"));
    }
}

/// Every external program, as this process located it at startup.
///
/// One value per process, threaded from the CLI into the artifact fetches, the
/// preflight, and the engines, so the `docker` that pulled an image is the one
/// that runs the container. Built field by field (see `pipette_cli::commands`)
/// because a constructor taking five same-typed paths invites transposing them.
#[derive(Debug, Clone)]
pub struct ExternalPrograms {
    pub uv: ResolvedProgram,
    pub docker: ResolvedProgram,
    pub nvidia_smi: ResolvedProgram,
    pub rocm_smi: ResolvedProgram,
    pub python: ResolvedProgram,
}

impl ExternalPrograms {
    /// Every program off `PATH`, for a caller with no command line to take
    /// overrides from.
    pub fn from_path() -> Self {
        Self {
            uv: ExternalProgram::UV.resolve(None),
            docker: ExternalProgram::DOCKER.resolve(None),
            nvidia_smi: ExternalProgram::NVIDIA_SMI.resolve(None),
            rocm_smi: ExternalProgram::ROCM_SMI.resolve(None),
            python: ExternalProgram::PYTHON.resolve(None),
        }
    }

    /// Every program pointed at a path that does not exist, so a test cannot
    /// reach the host's real `docker` and pull an image. Resolved rather than
    /// failed, so a caller fails at spawn, where such a test looks for the
    /// error.
    ///
    /// Behind `test-support` because `#[cfg(test)]` does not reach a dependent
    /// crate's tests.
    #[cfg(feature = "test-support")]
    pub fn at_missing_paths() -> Self {
        fn missing(name: &str) -> ResolvedProgram {
            ResolvedProgram::at(format!("/no-such-program/{name}"))
        }
        Self {
            uv: missing("uv"),
            docker: missing("docker"),
            nvidia_smi: missing("nvidia-smi"),
            rocm_smi: missing("rocm-smi"),
            python: missing("python3"),
        }
    }

    /// Every program paired with its const, in report order.
    ///
    /// The only place the catalog is a sequence. Destructured, so a new field
    /// on [`ExternalPrograms`] fails to compile here rather than going
    /// unchecked.
    pub fn catalog(&self) -> [(ExternalProgram, &ResolvedProgram); 5] {
        let Self {
            uv,
            docker,
            nvidia_smi,
            rocm_smi,
            python,
        } = self;
        [
            (ExternalProgram::UV, uv),
            (ExternalProgram::DOCKER, docker),
            (ExternalProgram::PYTHON, python),
            (ExternalProgram::NVIDIA_SMI, nvidia_smi),
            (ExternalProgram::ROCM_SMI, rocm_smi),
        ]
    }

    /// Every check, even after one fails: an operator setting a box up would
    /// otherwise pay a round trip per program.
    pub fn check_all(&self) -> Vec<ProgramCheck> {
        self.catalog()
            .into_iter()
            .map(|(program, resolved)| program.check(resolved))
            .collect()
    }

    /// Fail unless each of `selected` passes its check, for `worker --strict`.
    ///
    /// Takes a set because which programs a host should have is the operator's
    /// fact: a Mac has neither GPU CLI and runs its MLX jobs anyway.
    pub fn require_working(&self, selected: &[ExternalProgram]) -> anyhow::Result<()> {
        let checks: Vec<ProgramCheck> = self
            .catalog()
            .into_iter()
            .filter(|(program, _)| selected.contains(program))
            .map(|(program, resolved)| program.check(resolved))
            .collect();
        ProgramCheck::refuse_any_failed(&checks)
    }

    /// [`Self::require_working`] over the whole catalog.
    pub fn require_all_working(&self) -> anyhow::Result<()> {
        ProgramCheck::refuse_any_failed(&self.check_all())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The one binary every host has, for the cases that need a real file.
    const HOST_SHELL: &str = if cfg!(windows) { "cmd" } else { "sh" };

    fn a_real_binary() -> anyhow::Result<PathBuf> {
        the_host_shell(&[]).locate()
    }

    /// On Windows the file is `cmd.exe`, so this also covers the extension a
    /// bare name has to gain there.
    #[test]
    fn the_host_shell_is_found_on_path() -> anyhow::Result<()> {
        let found = a_real_binary()?;
        assert!(found.is_absolute(), "resolved to {found:?}");
        assert_eq!(found.file_stem(), Some(OsStr::new(HOST_SHELL)));
        Ok(())
    }

    #[test]
    fn a_binary_no_host_has_is_not_on_path() {
        let path = std::env::var_os("PATH").unwrap_or_default();
        assert_eq!(which(&path, "pipette-no-such-binary-3f9a1c"), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_file_on_path_without_an_execute_bit_is_passed_over() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        // It exists, so the missing bit is what turns it down.
        assert!(executable_at(&dir.join("Cargo.toml")).is_some());
        assert_eq!(which(dir.as_os_str(), "Cargo.toml"), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_relative_path_entry_is_passed_over() -> anyhow::Result<()> {
        let shell = a_real_binary()?;
        let dir = shell
            .parent()
            .ok_or_else(|| anyhow::anyhow!("{shell:?} has no parent"))?;
        // More `..` than any working directory is deep, since they stop at `/`:
        // a relative entry that reaches the shell's directory from anywhere.
        let relative = std::iter::repeat_n("..", 64)
            .collect::<PathBuf>()
            .join(dir.strip_prefix("/")?);
        assert!(executable_at(&relative.join(HOST_SHELL)).is_some());
        assert_eq!(which(relative.as_os_str(), HOST_SHELL), None);
        Ok(())
    }

    #[rstest]
    #[case(ExternalProgram::UV)]
    #[case(ExternalProgram::DOCKER)]
    #[case(ExternalProgram::NVIDIA_SMI)]
    #[case(ExternalProgram::ROCM_SMI)]
    #[case(ExternalProgram::PYTHON)]
    fn every_program_names_itself_and_where_to_get_it(#[case] program: ExternalProgram) {
        assert!(!program.candidates.is_empty());
        assert!(!program.install_hint.is_empty());
        assert!(!program.label().is_empty());
        // Without one, `--strict` would pass a program it never ran.
        assert!(!program.check_args.is_empty());
    }

    #[test]
    fn a_program_with_several_names_reports_them_all() {
        assert_eq!(ExternalProgram::PYTHON.label(), "python3/python");
        let msg = ExternalProgram::PYTHON.not_on_path_message();
        assert!(
            msg.contains("python3/python not found on PATH"),
            "got {msg}"
        );
    }

    #[test]
    fn an_explicit_file_is_taken_as_the_binary() -> anyhow::Result<()> {
        let sh = a_real_binary()?;
        let resolved = ExternalProgram::UV.resolve(Some(&sh));
        assert_eq!(resolved.path()?, sh.as_path());
        Ok(())
    }

    /// The host's shell running `script`: a real binary and exit status that
    /// do not depend on which catalog programs this machine has.
    fn the_host_shell(script: &'static [&'static str]) -> ExternalProgram {
        ExternalProgram {
            candidates: &[HOST_SHELL],
            install_hint: "it ships with the OS",
            check_args: script,
        }
    }

    const SHELL_C: &str = if cfg!(windows) { "/c" } else { "-c" };
    const ECHO_OK: &[&str] = &[SHELL_C, "echo ok"];
    /// The one failure shape both shells spell the same way.
    const EXIT_3: &[&str] = &[SHELL_C, "exit 3"];
    /// Unix redirection syntax, hence the cfg.
    #[cfg(unix)]
    const COMPLAIN_AND_FAIL: &[&str] = &[SHELL_C, "echo boom >&2; exit 3"];
    /// Outlives any deadline a test sets. Unix: Windows `cmd` has no `sleep`.
    #[cfg(unix)]
    const HANG: &[&str] = &[SHELL_C, "sleep 30"];
    /// Hangs after starting a `sleep` in a group of its own, as a shim's
    /// child can be, and writes its pid to a file named after the test's.
    #[cfg(unix)]
    const HANG_WITH_A_CHILD: &[&str] = &[
        SHELL_C,
        r#"perl -e 'setpgrp; exec @ARGV' sleep 30 & echo $! > "${TMPDIR:-/tmp}/pipette-programs-child-$PPID"; sleep 30"#,
    ];

    #[test]
    fn an_explicit_directory_is_taken_as_the_bin_dir() -> anyhow::Result<()> {
        let sh = a_real_binary()?;
        let bin_dir = sh
            .parent()
            .ok_or_else(|| anyhow::anyhow!("{sh:?} has no parent"))?;
        let shell = the_host_shell(&["--version"]);
        let resolved = shell.resolve(Some(bin_dir)).path()?.to_path_buf();
        // Not compared with `bin_dir.join(HOST_SHELL)`: on Windows the binary
        // in there is `cmd.exe`.
        assert_eq!(resolved.parent(), Some(bin_dir));
        assert_eq!(resolved.file_stem(), Some(OsStr::new(HOST_SHELL)));
        assert!(resolved.is_file(), "{resolved:?} should be the binary");
        Ok(())
    }

    #[test]
    fn an_explicit_path_that_is_missing_does_not_fall_back_to_path() {
        let resolved = ExternalProgram::UV.resolve(Some(Path::new("/no/such/uv")));
        assert!(!resolved.is_resolved());
        let msg = message(resolved.path().err());
        assert!(msg.contains("/no/such/uv"), "got {msg}");
    }

    /// Asserted on the message, so what the test machine has installed does
    /// not decide whether this is checked.
    #[test]
    fn a_program_that_is_not_on_path_says_where_to_get_it() {
        let msg = ExternalProgram::ROCM_SMI.not_on_path_message();
        assert!(msg.contains("rocm-smi not found on PATH"), "got {msg}");
        assert!(msg.contains("install ROCm"), "got {msg}");
    }

    #[test]
    fn a_missing_program_is_only_an_error_at_first_use() {
        let docker = ExternalProgram::DOCKER.resolve(Some(Path::new("/no/such/docker")));
        assert!(!docker.is_resolved());
        assert!(docker.path().is_err());
    }

    fn missing_everything() -> ExternalPrograms {
        ExternalPrograms {
            uv: ExternalProgram::UV.resolve(Some(Path::new("/no/such/uv"))),
            docker: ExternalProgram::DOCKER.resolve(Some(Path::new("/no/such/docker"))),
            nvidia_smi: ExternalProgram::NVIDIA_SMI.resolve(Some(Path::new("/no/such/nvidia-smi"))),
            rocm_smi: ExternalProgram::ROCM_SMI.resolve(Some(Path::new("/no/such/rocm-smi"))),
            python: ExternalProgram::PYTHON.resolve(Some(Path::new("/no/such/python"))),
        }
    }

    #[rstest]
    #[case::answers(ECHO_OK, true, &["ok"])]
    #[case::exits_non_zero(EXIT_3, false, &["exit 3", "no output"])]
    fn the_exit_status_decides_a_check(
        #[case] script: &'static [&'static str],
        #[case] passes: bool,
        #[case] said: &[&str],
    ) -> anyhow::Result<()> {
        let check = the_host_shell(script).check(&ResolvedProgram::at(a_real_binary()?));
        assert_eq!(check.passed(), passes, "{:?}", check.failure());
        assert!(check.path().is_some());
        let report = if passes {
            check.reported().unwrap_or_default()
        } else {
            check.failure().unwrap_or_default()
        };
        let absent: Vec<&str> = said
            .iter()
            .copied()
            .filter(|want| !report.contains(want))
            .collect();
        assert!(absent.is_empty(), "{absent:?} missing from {report}");
        Ok(())
    }

    /// Not a row of the table above: the script is unix-only, and a
    /// `cfg_attr`-ed `#[case]` reads worse.
    #[cfg(unix)]
    #[test]
    fn a_failing_check_quotes_what_the_program_complained() -> anyhow::Result<()> {
        let shell = the_host_shell(COMPLAIN_AND_FAIL);
        let check = shell.check(&ResolvedProgram::at(a_real_binary()?));
        let why = check.failure().unwrap_or_default();
        assert!(why.contains("boom"), "got {why}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_program_that_hangs_fails_its_check_at_the_deadline() -> anyhow::Result<()> {
        let shell = the_host_shell(HANG);
        let check = shell.check_within(
            &ResolvedProgram::at(a_real_binary()?),
            Duration::from_millis(200),
        );
        // Only the deadline path says this, so reaching it is the assertion.
        let why = check.failure().unwrap_or_default();
        assert!(why.contains("did not answer within"), "got {why}");
        Ok(())
    }

    /// Including a child that moved to a process group of its own.
    #[cfg(unix)]
    #[test]
    fn a_hung_check_takes_what_it_started_with_it() -> anyhow::Result<()> {
        use anyhow::Context;

        // Without `perl` the child never starts, and is gone anyway.
        let perl = std::process::Command::new("perl")
            .args(["-e", "1"])
            .output();
        anyhow::ensure!(
            perl.is_ok_and(|output| output.status.success()),
            "this test needs `perl` on PATH"
        );
        let pid_file =
            std::env::temp_dir().join(format!("pipette-programs-child-{}", std::process::id()));
        let shell = the_host_shell(HANG_WITH_A_CHILD);
        let check = shell.check_within(
            &ResolvedProgram::at(a_real_binary()?),
            Duration::from_millis(500),
        );
        let why = check.failure().unwrap_or_default();
        assert!(why.contains("did not answer within"), "got {why}");
        let child = std::fs::read_to_string(&pid_file)?;
        std::fs::remove_file(&pid_file)?;
        let child: libc::pid_t = child.trim().parse().context("no pid written")?;
        // SAFETY: signal 0 only asks whether the pid exists.
        let alive = || unsafe { libc::kill(child, 0) } == 0;
        let started = std::time::Instant::now();
        while alive() && started.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!alive(), "the program's child {child} outlived the check");
        Ok(())
    }

    #[test]
    fn a_program_that_was_never_located_fails_its_check_with_the_lookup_message() {
        let resolved = ExternalProgram::DOCKER.resolve(Some(Path::new("/no/such/docker")));
        let check = ExternalProgram::DOCKER.check(&resolved);
        assert!(!check.passed());
        assert_eq!(check.path(), None);
        let why = check.failure().unwrap_or_default();
        assert!(why.contains("/no/such/docker"), "got {why}");
    }

    #[test]
    fn a_located_program_that_cannot_run_fails_its_check() {
        let resolved = ResolvedProgram::at("/no-such-program/uv");
        let check = ExternalProgram::UV.check(&resolved);
        assert!(!check.passed());
        assert_eq!(check.path(), Some(Path::new("/no-such-program/uv")));
        let why = check.failure().unwrap_or_default();
        assert!(why.contains("could not run"), "got {why}");
    }

    /// A program missing from the catalog is one `--strict` never checks.
    #[test]
    fn the_catalog_holds_every_program() {
        let labels: Vec<String> = ExternalPrograms::from_path()
            .catalog()
            .into_iter()
            .map(|(program, _)| program.label())
            .collect();
        assert_eq!(
            labels,
            ["uv", "docker", "python3/python", "nvidia-smi", "rocm-smi"]
        );
    }

    #[test]
    fn require_working_checks_only_what_it_was_given() -> anyhow::Result<()> {
        let programs = missing_everything();
        programs.require_working(&[])?;
        let msg = message(programs.require_working(&[ExternalProgram::ROCM_SMI]).err());
        assert!(msg.contains("/no/such/rocm-smi"), "got {msg}");
        assert!(!msg.contains("/no/such/uv"), "got {msg}");
        Ok(())
    }

    #[rstest]
    #[case("/no/such/uv")]
    #[case("/no/such/docker")]
    #[case("/no/such/python")]
    #[case("/no/such/nvidia-smi")]
    #[case("/no/such/rocm-smi")]
    fn require_all_working_names_every_failing_program(#[case] named: &str) {
        let msg = message(missing_everything().require_all_working().err());
        assert!(msg.contains(named), "got {msg}");
    }

    /// A Mac (uv and docker, no GPU CLI) fails the whole-catalog gate, which is
    /// why nothing applies it by default.
    #[test]
    fn require_all_working_refuses_a_host_that_has_only_the_dependencies() -> anyhow::Result<()> {
        let sh = a_real_binary()?;
        let programs = ExternalPrograms {
            uv: ResolvedProgram::at(sh.clone()),
            docker: ResolvedProgram::at(sh),
            ..missing_everything()
        };
        let msg = message(programs.require_all_working().err());
        assert!(msg.contains("/no/such/nvidia-smi"), "got {msg}");
        Ok(())
    }

    #[test]
    fn a_check_reports_the_arguments_it_ran() {
        let check = ExternalProgram::DOCKER
            .check(&ExternalProgram::DOCKER.resolve(Some(Path::new("/no/such/docker"))));
        assert_eq!(check.check_args(), "info --format {{.ServerVersion}}");
        assert_eq!(check.label(), "docker");
    }

    fn message(err: Option<anyhow::Error>) -> String {
        err.map(|e| e.to_string()).unwrap_or_default()
    }
}
