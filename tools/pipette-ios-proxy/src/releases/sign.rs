//! A pulled build, signed for the fleet on this Mac, cached under
//! `signed/<version>/` and copied for one phone (DESIGN section 9).

use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::Context;
use wait_timeout::ChildExt;

use super::store::{self, PARTIAL_PREFIX};
use super::{Build, Fault, UnreadyError};
use crate::app::Signals;
use crate::pool::bundle;
use crate::secret::{self, Secret};
use crate::state::{ReleasePaths, StateLock};
use crate::tool::{self, Ran, ToolError};

const UNLOCK_TIMEOUT: Duration = Duration::from_secs(30);

/// `security`, `xcodebuild -version`, `ditto` and `cp` answer in seconds; one
/// that hangs must not hold `signed.lock` or stall a start.
const TOOL_TIMEOUT: Duration = Duration::from_secs(60);

/// Between SIGTERM and SIGKILL to the export's process group.
const EXPORT_KILL_GRACE: Duration = Duration::from_secs(5);

const EXPORT_POLL: Duration = Duration::from_millis(200);

/// What `signed.lock` is held for beyond the export itself: the unlock, the
/// kill grace, the unpack, the gates and the copy.
const SIGNED_LOCK_OVERHEAD: Duration = Duration::from_secs(3 * 60);

/// Beside a cached app: the sha256 of the archive it was exported from.
const SOURCE_DIGEST_FILE: &str = "source.sha256";

const DETAIL_LINES: usize = 3;

const DETAIL_CHARS: usize = 400;

/// What signs a build for the fleet, built by `Config::signer`.
#[derive(Debug)]
pub(crate) struct Signer {
    pub(crate) team_id: String,
    pub(crate) keychain: PathBuf,
    pub(crate) password: Secret,
    pub(crate) api_key_file: PathBuf,
    pub(crate) api_key_id: String,
    pub(crate) api_issuer_id: String,
    pub(crate) export_timeout: Duration,
    /// `/usr/bin/security` and `/usr/bin/xcodebuild`, or a test's stand-ins.
    pub(crate) security: PathBuf,
    pub(crate) xcodebuild: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SignError {
    /// `security -i` read the line and refused it: a wrong password.
    #[error(
        "unlocking {keychain} with `security -i` failed ({}): {detail}",
        describe(.status)
    )]
    Unlock {
        keychain: PathBuf,
        status: ExitStatus,
        detail: String,
    },
    #[error(
        "unlocking {keychain} with `security -i` did not finish within {}s and was killed: \
         {detail}",
        UNLOCK_TIMEOUT.as_secs()
    )]
    UnlockTimedOut { keychain: PathBuf, detail: String },
    #[error("writing the unlock line for {keychain} to `security -i` failed: {detail}")]
    UnlockWrite {
        keychain: PathBuf,
        #[source]
        source: io::Error,
        detail: String,
    },
    #[error(
        "{keychain} is not on this user's keychain search list, where `codesign` looks for the \
         identity (`security list-keychains -d user`: {listed})"
    )]
    NotSearched { keychain: PathBuf, listed: String },
    #[error(
        "{keychain} holds no valid code-signing identity (`security find-identity -v -p \
         codesigning`): {detail}. A missing Apple WWDR G3 intermediate reads the same way"
    )]
    NoIdentity { keychain: PathBuf, detail: String },
    #[error(
        "`xcodebuild -version` failed ({}): {detail}. `xcode-select -p` must name an Xcode whose \
         license is accepted",
        describe(.status)
    )]
    XcodeUnusable { status: ExitStatus, detail: String },
    #[error("`xcodebuild -exportArchive` failed ({}): {detail}", describe(.status))]
    Export { status: ExitStatus, detail: String },
    #[error("`xcodebuild -exportArchive` did not finish within {secs}s and was killed")]
    ExportTimedOut { secs: u64 },
    #[error("`xcodebuild -exportArchive` was stopped")]
    Interrupted,
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error("`ditto -x -k` could not unpack {ipa}: {detail}")]
    UnpackIpa { ipa: PathBuf, detail: String },
    #[error("copying {app} into {staged}: {detail}")]
    Stage {
        app: PathBuf,
        staged: PathBuf,
        detail: String,
    },
}

impl SignError {
    fn fault(&self) -> Fault {
        match self {
            Self::Unlock { .. }
            | Self::NotSearched { .. }
            | Self::NoIdentity { .. }
            | Self::XcodeUnusable { .. } => Fault::Operator,
            Self::UnlockTimedOut { .. }
            | Self::UnlockWrite { .. }
            | Self::Export { .. }
            | Self::ExportTimedOut { .. }
            | Self::Interrupted
            | Self::Tool(_)
            | Self::UnpackIpa { .. }
            | Self::Stage { .. } => Fault::Transient,
        }
    }
}

impl From<SignError> for UnreadyError {
    fn from(error: SignError) -> Self {
        Self::new(error.fault(), error)
    }
}

impl Signer {
    /// What `run` and `check` prove before any phone needs it, with no Apple
    /// call: the password unlocks the keychain, which is on the search list and
    /// holds an identity, and `xcodebuild` runs.
    pub(crate) fn preflight(&self) -> Result<(), SignError> {
        self.unlock()?;
        self.check_searched()?;
        let found = run_tool(
            Command::new(&self.security)
                .args(["find-identity", "-v", "-p", "codesigning"])
                .arg(&self.keychain),
        )?;
        if !found.status.success() || valid_identities(&found.stdout) == 0 {
            return Err(SignError::NoIdentity {
                keychain: self.keychain.clone(),
                detail: tail(&found.stdout, &found.stderr),
            });
        }
        let xcode = run_tool(Command::new(&self.xcodebuild).arg("-version"))?;
        if !xcode.status.success() {
            return Err(SignError::XcodeUnusable {
                status: xcode.status,
                detail: tail(&xcode.stdout, &xcode.stderr),
            });
        }
        Ok(())
    }

    /// The unlock names the keychain, but `codesign` finds the identity
    /// through the search list.
    fn check_searched(&self) -> Result<(), SignError> {
        let listed = run_tool(Command::new(&self.security).args(["list-keychains", "-d", "user"]))?;
        let canonical = |path: &Path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let wanted = canonical(&self.keychain);
        let searched = listed.status.success()
            && listed
                .stdout
                .lines()
                .map(|line| line.trim().trim_matches('"'))
                .any(|line| !line.is_empty() && canonical(Path::new(line)) == wanted);
        if searched {
            return Ok(());
        }
        Err(SignError::NotSearched {
            keychain: self.keychain.clone(),
            listed: tail(&listed.stdout, &listed.stderr),
        })
    }

    /// Unlock the keychain for this process's session: `codesign` fails with
    /// `errSecInternalComponent` on a locked one. `security -i` reads the
    /// command on stdin, so the password is never in an argv.
    pub(crate) fn unlock(&self) -> Result<(), SignError> {
        #[expect(
            clippy::disallowed_methods,
            reason = "stdout is discarded, and stderr is piped and drained below"
        )]
        let mut child = Command::new(&self.security)
            .arg("-i")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ToolError::Run {
                program: self.security.clone(),
                source,
            })?;
        let line = format!(
            "unlock-keychain -p \"{}\" \"{}\"\n",
            self.password.expose(),
            self.keychain.display()
        );
        // Dropped once written: the EOF ends `security -i`. A reader already
        // gone must fail the write, not raise the SIGPIPE `main` made fatal.
        let written = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("its stdin was not piped"))
            .and_then(|mut stdin| {
                pipette_subprocess::no_sigpipe_on(&stdin)?;
                stdin.write_all(line.as_bytes())
            });
        let stderr = child.stderr.take().map(tool::drain);
        let status = tool::wait_bounded(&mut child, UNLOCK_TIMEOUT);
        let detail = tail(
            "",
            &secret::scrub(tool::collect(stderr), &[self.password.expose().to_owned()]),
        );
        let keychain = self.keychain.clone();
        match (status, written) {
            (None, _) => Err(SignError::UnlockTimedOut { keychain, detail }),
            (Some(_), Err(source)) => Err(SignError::UnlockWrite {
                keychain,
                source,
                detail,
            }),
            (Some(status), Ok(())) if status.success() => Ok(()),
            (Some(status), Ok(())) => Err(SignError::Unlock {
                keychain,
                status,
                detail,
            }),
        }
    }

    /// How long a claim waits for `signed.lock`: another phone's whole export.
    pub(crate) fn lock_budget(&self) -> Duration {
        self.export_timeout + SIGNED_LOCK_OVERHEAD
    }

    fn export(
        &self,
        archive: &Path,
        scratch: &Path,
        signals: Signals<'_>,
    ) -> Result<PathBuf, UnreadyError> {
        let options = scratch.join("ExportOptions.plist");
        export_options(&self.team_id)
            .to_file_xml(&options)
            .with_context(|| format!("writing {}", options.display()))
            .map_err(UnreadyError::transient)?;
        self.unlock()?;
        let exported = scratch.join("export");
        let argv: Vec<OsString> = vec![
            "-exportArchive".into(),
            "-archivePath".into(),
            archive.into(),
            "-exportPath".into(),
            exported.clone().into(),
            "-exportOptionsPlist".into(),
            options.into(),
            "-allowProvisioningUpdates".into(),
            "-authenticationKeyPath".into(),
            self.api_key_file.clone().into(),
            "-authenticationKeyID".into(),
            self.api_key_id.clone().into(),
            "-authenticationKeyIssuerID".into(),
            self.api_issuer_id.clone().into(),
        ];
        self.run_export(&argv, signals)?;
        let ipa = store::only_child(&exported, "ipa", "ipa").map_err(UnreadyError::transient)?;
        let unpacked = scratch.join("ipa");
        let ran = run_tool(
            Command::new("ditto")
                .args(["-x", "-k"])
                .arg(&ipa)
                .arg(&unpacked),
        )?;
        if !ran.status.success() {
            return Err(SignError::UnpackIpa {
                ipa,
                detail: ran.stderr.trim().to_owned(),
            }
            .into());
        }
        store::only_child(&unpacked.join("Payload"), "app", "app").map_err(UnreadyError::transient)
    }

    /// `xcodebuild` in its own process group, so a stop or the deadline ends its
    /// helpers too and no orphan overlaps the next export. Its output echoes the
    /// argv, so the auth arguments are scrubbed from it.
    fn run_export(&self, argv: &[OsString], signals: Signals<'_>) -> Result<(), SignError> {
        let program = &self.xcodebuild;
        #[expect(
            clippy::disallowed_methods,
            reason = "both output streams are piped and drained below"
        )]
        let mut child = Command::new(program)
            .args(argv)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|source| ToolError::Run {
                program: program.clone(),
                source,
            })?;
        let stdout = child.stdout.take().map(tool::drain);
        let stderr = child.stderr.take().map(tool::drain);
        let started = Instant::now();
        let ended = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => {}
                Err(source) => {
                    end_group(&mut child);
                    break Err(ToolError::Run {
                        program: program.clone(),
                        source,
                    }
                    .into());
                }
            }
            if signals.raised() {
                end_group(&mut child);
                break Err(SignError::Interrupted);
            }
            if started.elapsed() >= self.export_timeout {
                end_group(&mut child);
                break Err(SignError::ExportTimedOut {
                    secs: self.export_timeout.as_secs(),
                });
            }
            std::thread::sleep(EXPORT_POLL);
        };
        let status = ended?;
        if status.success() {
            return Ok(());
        }
        let auth = [
            self.api_key_file.display().to_string(),
            self.api_key_id.clone(),
            self.api_issuer_id.clone(),
        ];
        Err(SignError::Export {
            status,
            detail: tail(
                &secret::scrub(tool::collect(stdout), &auth),
                &secret::scrub(tool::collect(stderr), &auth),
            ),
        })
    }
}

/// Automatic development signing for the team, as the farm exports by hand.
fn export_options(team_id: &str) -> plist::Value {
    [
        ("method", "debugging"),
        ("teamID", team_id),
        ("signingStyle", "automatic"),
        ("signingCertificate", "Apple Development"),
        ("thinning", "<none>"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), plist::Value::String(value.to_owned())))
    .collect::<plist::Dictionary>()
    .into()
}

/// `N valid identities found`, the line `find-identity -v` ends with.
fn valid_identities(stdout: &str) -> usize {
    stdout
        .lines()
        .find_map(|line| line.trim().strip_suffix("valid identities found"))
        .and_then(|count| count.trim().parse().ok())
        .unwrap_or(0)
}

fn end_group(child: &mut Child) {
    use rustix::process::{Pid, Signal, kill_process_group};

    let Some(group) = i32::try_from(child.id()).ok().and_then(Pid::from_raw) else {
        let _ = child.kill();
        let _ = child.wait();
        return;
    };
    let _ = kill_process_group(group, Signal::TERM);
    let _ = child.wait_timeout(EXPORT_KILL_GRACE);
    // Helpers can outlive the leader; the group id stays theirs until they go.
    let _ = kill_process_group(group, Signal::KILL);
    let _ = child.wait();
}

fn run_tool(command: &mut Command) -> Result<Ran, SignError> {
    Ok(tool::run_bounded(command, TOOL_TIMEOUT)?)
}

fn describe(status: &ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "on a signal".to_owned(), |code| format!("status {code}"))
}

fn tail(stdout: &str, stderr: &str) -> String {
    let source = if stderr.trim().is_empty() {
        stdout
    } else {
        stderr
    };
    let mut lines: Vec<&str> = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .rev()
        .take(DETAIL_LINES)
        .collect();
    lines.reverse();
    let joined = lines.join(" | ");
    let count = joined.chars().count();
    if count <= DETAIL_CHARS {
        return if joined.is_empty() {
            "no output".to_owned()
        } else {
            joined
        };
    }
    let kept: String = joined.chars().skip(count - DETAIL_CHARS).collect();
    format!("...{kept}")
}

/// The signed app of `build` for one phone, copied into `staged`: the cached
/// export while it still passes `gate`, else a new one. `signed.lock` is held
/// throughout, so two exports never overlap and a re-sign never replaces a
/// copy mid-clone; the install that follows holds nothing.
///
/// `gate` judges an app with the profile expiry margin it must clear: a cached
/// copy must outlive [`bundle::EXPIRY_MARGIN`], so it is signed again before it
/// stops launching, and a fresh one only must not have expired.
pub(crate) fn signed_build(
    signer: &Signer,
    paths: &ReleasePaths,
    build: &Build,
    archive: &Path,
    gate: &dyn Fn(&Path, Duration) -> anyhow::Result<()>,
    staged: &Path,
    signals: Signals<'_>,
) -> Result<PathBuf, UnreadyError> {
    let _lock = StateLock::wait_for(&paths.signed_lock, signer.lock_budget(), signals)
        .map_err(UnreadyError::transient)?;
    let dir = paths.signed.join(build.version.as_str());
    let app = match cached(&dir, build) {
        Some(app) => match gate(&app, bundle::EXPIRY_MARGIN) {
            Ok(()) => app,
            Err(error) => {
                tracing::info!(
                    version = %build.version,
                    reason = %format!("{error:#}"),
                    "the cached signed build no longer passes; signing it again"
                );
                export_into(signer, paths, build, archive, gate, &dir, signals)?
            }
        },
        None => export_into(signer, paths, build, archive, gate, &dir, signals)?,
    };
    stage(&app, staged).map_err(UnreadyError::transient)
}

/// `signed/<version>/<Name>.app`, when it was exported from `build`'s bytes.
fn cached(dir: &Path, build: &Build) -> Option<PathBuf> {
    let digest = fs::read_to_string(dir.join(SOURCE_DIGEST_FILE)).ok()?;
    if digest.trim() != AsRef::<str>::as_ref(&build.sha256) {
        return None;
    }
    store::only_child(dir, "app", "app").ok()
}

/// Export, gate, and publish as `dir`, whole or not at all.
fn export_into(
    signer: &Signer,
    paths: &ReleasePaths,
    build: &Build,
    archive: &Path,
    gate: &dyn Fn(&Path, Duration) -> anyhow::Result<()>,
    dir: &Path,
    signals: Signals<'_>,
) -> Result<PathBuf, UnreadyError> {
    let scratch = scratch_in(&paths.signed).map_err(UnreadyError::transient)?;
    tracing::info!(version = %build.version, team = %signer.team_id, "signing the build for the fleet");
    let started = Instant::now();
    let app = signer.export(archive, scratch.path(), signals)?;
    // A fresh refusal is no stale cache: who can clear it depends on the gate.
    gate(&app, Duration::ZERO).map_err(|error| {
        UnreadyError::new(
            bundle::fault_of(&error),
            error.context(format!(
                "the build this Mac just signed for {} was refused",
                build.version
            )),
        )
    })?;
    let published = publish(&app, build, dir, scratch.path()).map_err(UnreadyError::transient)?;
    tracing::info!(
        version = %build.version,
        elapsed_secs = %started.elapsed().as_secs(),
        "signed the build for the fleet"
    );
    Ok(published)
}

/// Called under the lock, so whatever carries the prefix belongs to no one.
fn scratch_in(signed: &Path) -> anyhow::Result<tempfile::TempDir> {
    fs::create_dir_all(signed).with_context(|| format!("creating {}", signed.display()))?;
    store::sweep_partials(signed)?;
    tempfile::Builder::new()
        .prefix(PARTIAL_PREFIX)
        .tempdir_in(signed)
        .with_context(|| format!("creating a scratch directory in {}", signed.display()))
}

/// Any old `dir` moves into `scratch`, so it goes when the caller drops the
/// scratch directory.
fn publish(app: &Path, build: &Build, dir: &Path, scratch: &Path) -> anyhow::Result<PathBuf> {
    let name = app
        .file_name()
        .context("the exported app has no file name")?;
    let staging = scratch.join("publish");
    fs::create_dir(&staging).with_context(|| format!("creating {}", staging.display()))?;
    fs::rename(app, staging.join(name))
        .with_context(|| format!("moving {} aside", app.display()))?;
    fs::write(
        staging.join(SOURCE_DIGEST_FILE),
        AsRef::<str>::as_ref(&build.sha256),
    )
    .with_context(|| format!("writing {SOURCE_DIGEST_FILE}"))?;
    match fs::rename(dir, scratch.join("replaced")) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("moving {} aside", dir.display()));
        }
    }
    fs::rename(&staging, dir).with_context(|| format!("publishing {}", dir.display()))?;
    Ok(dir.join(name))
}

/// A copy of `app` in `staged`, replacing what was there: an APFS clone
/// (`cp -c`), instant and taking no extra disk.
fn stage(app: &Path, staged: &Path) -> anyhow::Result<PathBuf> {
    match fs::remove_dir_all(staged) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("removing {}", staged.display())),
    }
    fs::create_dir_all(staged).with_context(|| format!("creating {}", staged.display()))?;
    let ran = run_tool(
        Command::new("cp")
            .args(["-c", "-R", "-p"])
            .arg(app)
            .arg(staged),
    )?;
    if !ran.status.success() {
        return Err(SignError::Stage {
            app: app.to_path_buf(),
            staged: staged.to_path_buf(),
            detail: ran.stderr.trim().to_owned(),
        }
        .into());
    }
    let name = app.file_name().context("the signed app has no file name")?;
    Ok(staged.join(name))
}

#[cfg(test)]
pub(crate) mod testing {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicBool;

    use super::*;

    pub(crate) const TEAM: &str = "353N5JSDGZ";
    pub(crate) const PASSWORD: &str = "correct horse";

    pub(crate) fn stand_in(dir: &Path, name: &str, body: &str) -> anyhow::Result<PathBuf> {
        let path = dir.join(name);
        fs::write(&path, format!("#!/usr/bin/env bash\n{body}"))?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    /// Each stand-in creates its logs empty, so a missing log is an error, not
    /// a stand-in never run.
    pub(crate) fn logged(dir: &Path, name: &str) -> anyhow::Result<String> {
        fs::read_to_string(dir.join(name)).with_context(|| format!("reading the {name} log"))
    }

    pub(crate) fn ipa(dir: &Path) -> anyhow::Result<PathBuf> {
        let stage = dir.join("ipa-stage");
        fs::create_dir_all(stage.join("Payload/Pipette.app"))?;
        fs::write(stage.join("Payload/Pipette.app/Info.plist"), "<plist/>")?;
        let ipa = dir.join("Pipette.ipa");
        let packed = Command::new("ditto")
            .args(["-c", "-k"])
            .arg(&stage)
            .arg(&ipa)
            .output()?;
        anyhow::ensure!(packed.status.success(), "ditto -c -k: {packed:?}");
        Ok(ipa)
    }

    /// A `security` that logs its argv to `security.argv` and its stdin to
    /// `security.stdin`, answers `-i` with `unlock_status`, lists
    /// `dir/login.keychain-db` as the search list, and reports `identities`
    /// valid identities.
    pub(crate) fn security(
        dir: &Path,
        unlock_status: i32,
        identities: usize,
    ) -> anyhow::Result<PathBuf> {
        let log = dir.display();
        fs::write(dir.join("security.argv"), "")?;
        fs::write(dir.join("security.stdin"), "")?;
        stand_in(
            dir,
            "security",
            &format!(
                "echo \"$@\" >> \"{log}/security.argv\"\n\
                 if [[ \"$1\" == -i ]]; then\n\
                   cat >> \"{log}/security.stdin\"\n\
                   echo \"security: SecKeychainUnlock: The user name or passphrase you entered \
                 is not correct. ({PASSWORD})\" >&2\n\
                   exit {unlock_status}\n\
                 fi\n\
                 if [[ \"$1\" == list-keychains ]]; then\n\
                   echo '    \"{log}/login.keychain-db\"'\n\
                   exit 0\n\
                 fi\n\
                 echo '  1) 0A1B \"Apple Development: Fleet (GPXZA93YWA)\"'\n\
                 echo '     {identities} valid identities found'\n"
            ),
        )
    }

    /// An `xcodebuild` that answers `-version`, and otherwise logs its argv to
    /// `xcodebuild.argv`, echoes it as the real one opens its output, runs
    /// `before`, then writes `ipa` where `-exportPath` says.
    pub(crate) fn xcodebuild(dir: &Path, ipa: &Path, before: &str) -> anyhow::Result<PathBuf> {
        fs::write(dir.join("xcodebuild.argv"), "")?;
        stand_in(
            dir,
            "xcodebuild",
            &format!(
                "if [[ \"$1\" == -version ]]; then echo 'Xcode 27.0'; echo 'Build version 27A5218g'; \
                 exit 0; fi\n\
                 echo \"$@\" >> \"{log}/xcodebuild.argv\"\n\
                 echo 'Command line invocation:'\n\
                 echo \"    $0 $*\"\n\
                 for arg in \"$@\"; do\n\
                   if [[ \"$next\" == 1 ]]; then out=\"$arg\"; next=; fi\n\
                   if [[ \"$arg\" == -exportPath ]]; then next=1; fi\n\
                 done\n\
                 {before}\n\
                 mkdir -p \"$out\" && cp \"{ipa}\" \"$out/Pipette.ipa\"\n",
                log = dir.display(),
                ipa = ipa.display(),
            ),
        )
    }

    pub(crate) fn signer(dir: &Path, security: PathBuf, xcodebuild: PathBuf) -> Signer {
        Signer {
            team_id: TEAM.to_owned(),
            keychain: dir.join("login.keychain-db"),
            password: Secret::for_tests(PASSWORD),
            api_key_file: dir.join("AuthKey_TEST.p8"),
            api_key_id: "TESTKEYID1".to_owned(),
            api_issuer_id: "00000000-0000-0000-0000-000000000000".to_owned(),
            export_timeout: Duration::from_secs(30),
            security,
            xcodebuild,
        }
    }

    /// The lease and stop flags a claim holds, both down until a test raises one.
    #[derive(Debug, Default)]
    pub(crate) struct Flags {
        pub(crate) lost: AtomicBool,
        pub(crate) stop: AtomicBool,
    }

    impl Flags {
        pub(crate) fn signals(&self) -> Signals<'_> {
            Signals {
                abort: &self.lost,
                stop: &self.stop,
            }
        }
    }

    /// Signed ad hoc, as the release job signs one.
    pub(crate) fn adhoc_app(
        app: &Path,
        bundle_id: &str,
        build_version: &str,
        memory: bool,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(app)?;
        fs::copy("/usr/bin/true", app.join("Pipette"))?;
        fs::write(
            app.join("Info.plist"),
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\"><dict>\
                 <key>CFBundleExecutable</key><string>Pipette</string>\
                 <key>CFBundleIdentifier</key><string>{bundle_id}</string>\
                 <key>CFBundlePackageType</key><string>APPL</string>\
                 <key>MinimumOSVersion</key><string>18.0</string>\
                 <key>PipetteBuildVersion</key><string>{build_version}</string>\
                 </dict></plist>"
            ),
        )?;
        let entitlements = app.with_extension("entitlements");
        fs::write(
            &entitlements,
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\"><dict>{}</dict></plist>",
                if memory {
                    "<key>com.apple.developer.kernel.increased-memory-limit</key><true/>"
                } else {
                    ""
                }
            ),
        )?;
        let signed = Command::new("codesign")
            .args(["-f", "-s", "-", "--entitlements"])
            .arg(&entitlements)
            .arg(app)
            .output()?;
        anyhow::ensure!(signed.status.success(), "codesign -s -: {signed:?}");
        fs::remove_file(entitlements)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::sync::atomic::Ordering;

    use super::testing::{
        Flags, PASSWORD, TEAM, ipa, logged, security, signer, stand_in, xcodebuild,
    };
    use super::*;
    use crate::pool::bundle::BundleError;
    use crate::releases::testing::{build, sha256_of};
    use crate::state::StatePaths;

    const V1: &str = "2026.10.02-e6a9434-internal";

    fn pass(_: &Path, _: Duration) -> anyhow::Result<()> {
        Ok(())
    }

    #[test]
    fn the_unlock_feeds_the_password_on_stdin() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let signer = signer(
            dir.path(),
            security(dir.path(), 0, 1)?,
            xcodebuild(dir.path(), Path::new("/none.ipa"), "")?,
        );

        signer.preflight()?;

        let argv = logged(dir.path(), "security.argv")?;
        assert_eq!(
            argv,
            format!(
                "-i\nlist-keychains -d user\nfind-identity -v -p codesigning {}\n",
                signer.keychain.display()
            )
        );
        assert!(!argv.contains(PASSWORD));
        assert_eq!(
            logged(dir.path(), "security.stdin")?,
            format!(
                "unlock-keychain -p \"{PASSWORD}\" \"{}\"\n",
                signer.keychain.display()
            )
        );
        assert_eq!(logged(dir.path(), "xcodebuild.argv")?, "", "no export");
        Ok(())
    }

    #[rstest::rstest]
    #[case::wrong_password(51, 1, "login.keychain-db", None, "status 51")]
    #[case::no_identity(
        0,
        0,
        "login.keychain-db",
        None,
        "holds no valid code-signing identity"
    )]
    #[case::off_the_search_list(
        0,
        1,
        "other.keychain-db",
        None,
        "is not on this user's keychain search list"
    )]
    #[case::no_xcode(
        0,
        1,
        "login.keychain-db",
        Some("echo 'xcode-select: error: tool xcodebuild requires Xcode' >&2; exit 1"),
        "`xcodebuild -version` failed (status 1): xcode-select: error"
    )]
    fn a_mac_that_cannot_sign_needs_an_operator(
        #[case] unlock_status: i32,
        #[case] identities: usize,
        #[case] keychain: &str,
        #[case] broken_xcodebuild: Option<&str>,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let xcodebuild = match broken_xcodebuild {
            Some(body) => stand_in(dir.path(), "broken-xcodebuild", body)?,
            None => xcodebuild(dir.path(), Path::new("/none.ipa"), "")?,
        };
        let mut signer = signer(
            dir.path(),
            security(dir.path(), unlock_status, identities)?,
            xcodebuild,
        );
        signer.keychain = dir.path().join(keychain);

        let error = signer
            .preflight()
            .err()
            .context("the preflight must fail")?;

        let message = error.to_string();
        assert!(message.contains(needle), "{message}");
        assert!(!message.contains(PASSWORD), "{message}");
        assert_eq!(error.fault(), Fault::Operator);
        Ok(())
    }

    #[rstest::rstest]
    #[case::wrong_password(
        SignError::Unlock {
            keychain: "k".into(),
            status: ExitStatus::from_raw(51 << 8),
            detail: "bad passphrase".into(),
        },
        Fault::Operator,
        "unlocking k with `security -i` failed (status 51): bad passphrase"
    )]
    #[case::killed(
        SignError::UnlockTimedOut { keychain: "k".into(), detail: "no output".into() },
        Fault::Transient,
        "unlocking k with `security -i` did not finish within 30s and was killed: no output"
    )]
    #[case::stdin_closed(
        SignError::UnlockWrite {
            keychain: "k".into(),
            source: io::ErrorKind::BrokenPipe.into(),
            detail: "no output".into(),
        },
        Fault::Transient,
        "writing the unlock line for k to `security -i` failed: no output: broken pipe"
    )]
    fn only_a_refused_unlock_needs_an_operator(
        #[case] error: SignError,
        #[case] fault: Fault,
        #[case] message: &str,
    ) {
        assert_eq!(error.fault(), fault);
        assert_eq!(format!("{:#}", anyhow::Error::new(error)), message);
    }

    /// A signer whose `security` unlocks and finds one identity, and whose
    /// `xcodebuild` runs `before`, then exports an `.ipa` holding `Pipette.app`.
    struct Rig {
        dir: tempfile::TempDir,
        signer: Signer,
    }

    impl Rig {
        fn new(before: &str) -> anyhow::Result<Self> {
            let dir = tempfile::tempdir()?;
            let ipa = ipa(dir.path())?;
            let signer = signer(
                dir.path(),
                security(dir.path(), 0, 1)?,
                xcodebuild(dir.path(), &ipa, before)?,
            );
            Ok(Self { dir, signer })
        }

        fn path(&self) -> &Path {
            self.dir.path()
        }

        fn paths(&self) -> anyhow::Result<ReleasePaths> {
            Ok(StatePaths::resolve(Some(&self.path().join("state")))?.releases())
        }
    }

    #[test]
    fn an_export_signs_for_the_team_with_the_api_key() -> anyhow::Result<()> {
        let rig = Rig::new("")?;
        let dir = rig.path();
        let signer = &rig.signer;
        let scratch = dir.join("scratch");
        fs::create_dir(&scratch)?;
        let flags = Flags::default();

        let app = signer.export(Path::new("/r/Pipette.xcarchive"), &scratch, flags.signals())?;

        assert_eq!(app, scratch.join("ipa/Payload/Pipette.app"));
        assert_eq!(
            logged(dir, "xcodebuild.argv")?,
            format!(
                "-exportArchive -archivePath /r/Pipette.xcarchive -exportPath {s}/export \
                 -exportOptionsPlist {s}/ExportOptions.plist -allowProvisioningUpdates \
                 -authenticationKeyPath {d}/AuthKey_TEST.p8 -authenticationKeyID TESTKEYID1 \
                 -authenticationKeyIssuerID 00000000-0000-0000-0000-000000000000\n",
                s = scratch.display(),
                d = dir.display(),
            )
        );
        let options: plist::Dictionary = plist::from_file(scratch.join("ExportOptions.plist"))?;
        let option = |key: &str| options.get(key).and_then(plist::Value::as_string);
        assert_eq!(option("method"), Some("debugging"));
        assert_eq!(option("teamID"), Some(TEAM));
        assert_eq!(option("signingStyle"), Some("automatic"));
        assert_eq!(option("signingCertificate"), Some("Apple Development"));
        assert_eq!(option("thinning"), Some("<none>"));
        assert_eq!(logged(dir, "security.argv")?, "-i\n");
        Ok(())
    }

    /// `xcodebuild`'s output opens with its argv, auth arguments included.
    #[rstest::rstest]
    #[case::fails_on_stderr(
        "echo 'error: exportArchive: No signing certificate' >&2; exit 70",
        "status 70"
    )]
    #[case::fails_on_stdout_alone("echo '** EXPORT FAILED **'; exit 65", "** EXPORT FAILED **")]
    #[case::hangs("sleep 30", "did not finish within 1s")]
    fn a_failed_export_is_transient_and_names_no_key(
        #[case] before: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let mut rig = Rig::new(before)?;
        rig.signer.export_timeout = Duration::from_secs(1);
        let flags = Flags::default();

        let unready = rig
            .signer
            .export(
                Path::new("/r/Pipette.xcarchive"),
                rig.path(),
                flags.signals(),
            )
            .err()
            .context("the export must fail")?;

        let message = format!("{:#}", unready.error);
        assert!(message.contains(needle), "{message}");
        [
            "TESTKEYID1",
            "AuthKey",
            "00000000-0000-0000-0000-000000000000",
        ]
        .iter()
        .for_each(|auth| assert!(!message.contains(auth), "{message}"));
        assert_eq!(unready.fault, Fault::Transient);
        Ok(())
    }

    #[test]
    fn a_stop_kills_the_export_and_its_helpers() -> anyhow::Result<()> {
        let rig = Rig::new("sleep 30 &\necho $! > \"$(dirname \"$0\")/helper.pid\"\nwait")?;
        let helper = rig.path().join("helper.pid");
        let flags = Flags::default();

        let started = Instant::now();
        let outcome = std::thread::scope(|scope| {
            scope.spawn(|| {
                // Once the helper is up, however loaded the machine.
                let _up = (0..250).any(|_| {
                    let up = fs::read_to_string(&helper).is_ok_and(|pid| pid.ends_with('\n'));
                    if !up {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    up
                });
                flags.stop.store(true, Ordering::SeqCst);
            });
            rig.signer.export(
                Path::new("/r/Pipette.xcarchive"),
                rig.path(),
                flags.signals(),
            )
        });

        let unready = outcome.err().context("a stopped export fails")?;
        assert!(format!("{:#}", unready.error).contains("was stopped"));
        assert!(started.elapsed() < Duration::from_secs(10));
        let pid: i32 = fs::read_to_string(&helper)?.trim().parse()?;
        let gone = (0..50).any(|_| {
            let alive = rustix::process::Pid::from_raw(pid)
                .is_some_and(|pid| rustix::process::test_kill_process(pid).is_ok());
            if alive {
                std::thread::sleep(Duration::from_millis(100));
            }
            !alive
        });
        assert!(gone, "the helper {pid} outlived the stop");
        Ok(())
    }

    #[test]
    fn a_signed_build_is_reused_until_it_fails_the_gate() -> anyhow::Result<()> {
        let rig = Rig::new("")?;
        let dir = rig.path();
        let signer = &rig.signer;
        let paths = rig.paths()?;
        let staged = dir.join("phone/staged");
        let flags = Flags::default();
        let exports =
            || -> anyhow::Result<usize> { Ok(logged(dir, "xcodebuild.argv")?.lines().count()) };
        let margins = std::cell::RefCell::new(Vec::new());
        let pass = |_: &Path, margin: Duration| -> anyhow::Result<()> {
            margins.borrow_mut().push(margin);
            Ok(())
        };
        let v1 = build(V1, V1, b"first")?;
        let archive = Path::new("/r/Pipette.xcarchive");

        let first = signed_build(
            signer,
            &paths,
            &v1,
            archive,
            &pass,
            &staged,
            flags.signals(),
        )?;
        assert_eq!(first, staged.join("Pipette.app"));
        assert!(first.join("Info.plist").is_file());
        assert_eq!(
            fs::read_to_string(paths.signed.join(v1.version.as_str()).join("source.sha256"))?,
            sha256_of(b"first")
        );
        signed_build(
            signer,
            &paths,
            &v1,
            archive,
            &pass,
            &staged,
            flags.signals(),
        )?;
        assert_eq!(exports()?, 1, "the cached copy is reused");
        assert_eq!(
            margins.take(),
            [Duration::ZERO, bundle::EXPIRY_MARGIN],
            "fresh, then cached"
        );

        let refuse_once = std::cell::Cell::new(true);
        let stale = |_: &Path, _: Duration| -> anyhow::Result<()> {
            if refuse_once.replace(false) {
                anyhow::bail!("the profile expires tomorrow");
            }
            Ok(())
        };
        signed_build(
            signer,
            &paths,
            &v1,
            archive,
            &stale,
            &staged,
            flags.signals(),
        )?;
        assert_eq!(
            exports()?,
            2,
            "a cached copy failing the gate is signed again"
        );

        let rebuilt = build(V1, V1, b"other bytes")?;
        signed_build(
            signer,
            &paths,
            &rebuilt,
            archive,
            &pass,
            &staged,
            flags.signals(),
        )?;
        assert_eq!(exports()?, 3, "other source bytes are signed again");
        assert_eq!(
            fs::read_dir(&paths.signed)?.count(),
            1,
            "no scratch directory is left"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::another_team(
        BundleError::WrongTeam { path: "a".into(), place: "profile", found: "ABCDE12345".into(), expected: TEAM.into() },
        Fault::Operator
    )]
    #[case::no_memory_limit(BundleError::NoMemoryEntitlement { path: "a".into() }, Fault::Version)]
    #[case::unsigned(
        BundleError::Unsigned { path: "a".into(), detail: "d".into() },
        Fault::Transient
    )]
    fn a_fresh_refusal_is_filed_by_its_fault(
        #[case] refusal: BundleError,
        #[case] fault: Fault,
    ) -> anyhow::Result<()> {
        let rig = Rig::new("")?;
        let paths = rig.paths()?;
        let flags = Flags::default();
        let reason = refusal.to_string();
        let refusal = std::cell::RefCell::new(Some(refusal));
        let gate = |_: &Path, _: Duration| -> anyhow::Result<()> {
            refusal
                .borrow_mut()
                .take()
                .map_or(Ok(()), |error| Err(error.into()))
        };
        let v1 = build(V1, V1, b"first")?;

        let unready = signed_build(
            &rig.signer,
            &paths,
            &v1,
            Path::new("/r/Pipette.xcarchive"),
            &gate,
            &rig.path().join("staged"),
            flags.signals(),
        )
        .err()
        .context("the fresh build must be refused")?;

        let message = format!("{:#}", unready.error);
        assert_eq!(unready.fault, fault, "{message}");
        assert!(message.contains(&reason), "{message}");
        assert!(!paths.signed.join(v1.version.as_str()).exists());
        Ok(())
    }

    /// Each thread opens its own lock fd, and the stand-in fails when it finds
    /// another export running.
    #[test]
    fn two_exports_never_overlap() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let paths = StatePaths::resolve(Some(&dir.path().join("state")))?.releases();
        let ipa = ipa(dir.path())?;
        let marker = dir.path().join("exporting");
        let xcodebuild = xcodebuild(
            dir.path(),
            &ipa,
            &format!(
                "if [[ -e \"{m}\" ]]; then echo overlapping >&2; exit 3; fi\n\
                 touch \"{m}\"; sleep 0.5; rm \"{m}\"",
                m = marker.display()
            ),
        )?;
        let security = security(dir.path(), 0, 1)?;

        let outcomes: Vec<Result<PathBuf, UnreadyError>> = std::thread::scope(|scope| {
            ["2026.10.01-0000000-internal", "2026.10.02-e6a9434-internal"]
                .iter()
                .enumerate()
                .map(|(index, version)| {
                    let signer = signer(dir.path(), security.clone(), xcodebuild.clone());
                    let paths = paths.clone();
                    let staged = dir.path().join(format!("phone-{index}/staged"));
                    scope.spawn(move || {
                        let flags = Flags::default();
                        let build = build(version, version, version.as_bytes())
                            .map_err(UnreadyError::transient)?;
                        signed_build(
                            &signer,
                            &paths,
                            &build,
                            Path::new("/r/Pipette.xcarchive"),
                            &pass,
                            &staged,
                            flags.signals(),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(UnreadyError::transient(anyhow::anyhow!("panicked")))
                    })
                })
                .collect()
        });

        outcomes
            .into_iter()
            .try_for_each(|outcome| outcome.map(drop).map_err(|unready| unready.error))?;
        assert_eq!(logged(dir.path(), "xcodebuild.argv")?.lines().count(), 2);
        Ok(())
    }

    #[test]
    fn the_lock_wait_outlasts_an_export() {
        let signer = signer(Path::new("/k"), PathBuf::new(), PathBuf::new());
        assert!(
            signer.lock_budget()
                > signer.export_timeout + UNLOCK_TIMEOUT + EXPORT_KILL_GRACE + 2 * TOOL_TIMEOUT
        );
    }

    /// `create-keychain` puts the throwaway keychain on the search list until
    /// the test deletes it.
    #[test]
    #[ignore = "creates a throwaway keychain; run it by hand on a signing Mac"]
    fn a_wrong_password_fails_the_unlock() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let keychain = dir.path().join("throwaway.keychain-db");
        let created = Command::new("/usr/bin/security")
            .args(["create-keychain", "-p", "right"])
            .arg(&keychain)
            .output()?;
        anyhow::ensure!(created.status.success(), "{created:?}");
        let mut signer = signer(
            dir.path(),
            PathBuf::from("/usr/bin/security"),
            PathBuf::new(),
        );
        signer.keychain = keychain.clone();
        signer.password = Secret::for_tests("wrong");

        let wrong = signer.unlock();
        signer.password = Secret::for_tests("right");
        let right = signer.unlock();
        let deleted = Command::new("/usr/bin/security")
            .arg("delete-keychain")
            .arg(&keychain)
            .output()?;

        let error = wrong.err().context("a wrong password must not unlock")?;
        assert!(matches!(error, SignError::Unlock { .. }), "{error}");
        assert!(!error.to_string().contains("wrong"), "{error}");
        right?;
        anyhow::ensure!(deleted.status.success(), "{deleted:?}");
        Ok(())
    }
}
