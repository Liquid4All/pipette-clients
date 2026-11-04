//! Sessions a pipette process started and left running when it died.
//!
//! A stop, a drop and ^C end what a run started, but a SIGKILL, an OOM kill or
//! an abort runs none of them. So each start notes the session it made
//! ([`Session`]), in a directory only this user can write, the note goes once
//! the child is reaped, and every start first kills the sessions whose starter
//! is gone.
//!
//! A process is known by its pid and its start time, so a pid handed out again
//! matches no note. A session whose leader is gone is known by a token its
//! processes carry in their environment ([`Token`]). Linux, Android and macOS;
//! elsewhere nothing is noted. On Windows a Job Object stops the children when
//! the process that started them exits, so no child remains to reap.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::session::{Session, started};

/// The environment variable a started child carries its [`Token`] in.
pub(crate) const TOKEN_VAR: &str = "PIPETTE_SESSION_TOKEN";

/// What marks one start's processes, which inherit it: this process, by pid
/// and start time, and a count.
pub(crate) struct Token(String);

impl Token {
    /// A new token, set on `cmd`'s environment.
    pub(crate) fn mark(cmd: &mut Command) -> Self {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let token = format!("{}-{}", owner(), COUNT.fetch_add(1, Ordering::Relaxed));
        cmd.env(TOKEN_VAR, &token);
        Self(token)
    }
}

/// This process as `<pid>-<start time>`, to mark what it starts outside a
/// session, such as a container, so a later run leaves a live owner's alone.
pub fn owner() -> String {
    owner_of(std::process::id())
}

/// Process `pid` as [`owner`] spells it.
pub fn owner_of(pid: u32) -> String {
    format!("{pid}-{}", started(pid).unwrap_or_default())
}

/// Whether the process an [`owner`] names still runs: a pid handed out again
/// started at another time.
pub fn owner_runs(owner: &str) -> bool {
    owner
        .split_once('-')
        .and_then(|(pid, start)| Some(started(pid.parse().ok()?)? == start.parse::<u64>().ok()?))
        .unwrap_or(false)
}

/// A started session's note, removed when dropped.
pub(crate) struct Record {
    path: PathBuf,
}

impl Record {
    /// Note that this process started `pid`, the leader of its own session,
    /// as `name`, marked with `token`; `None` where no note can be kept.
    pub(crate) fn write(pid: u32, name: &str, token: &Token) -> Option<Self> {
        Self::write_in(&dir()?, pid, name, token)
    }

    fn write_in(dir: &Path, pid: u32, name: &str, token: &Token) -> Option<Self> {
        let owner = std::process::id();
        let note = Note {
            owner,
            owner_start: started(owner)?,
            pid,
            start: started(pid)?,
        };
        let path = dir.join(note.file_name());
        match fs::write(&path, format!("{name}\n{}", token.0)) {
            Ok(()) => Some(Self { path }),
            Err(err) => {
                tracing::debug!(
                    program = %name,
                    pid,
                    error = %err,
                    "failed to note its session"
                );
                None
            }
        }
    }
}

impl Drop for Record {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Kill the sessions that pipette processes left running when they died, and
/// drop their notes (Linux, Android and macOS). [`crate::start_process`] does
/// this first; call it too before a wait those leftovers would hold up, such
/// as a readiness gate.
pub fn reap_orphans() {
    if let Some(dir) = dir() {
        reap_in(&dir);
    }
}

fn reap_in(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    entries
        .flatten()
        .filter_map(|entry| Some((Note::parse(entry.file_name().to_str()?)?, entry.path())))
        .filter(|(note, _)| started(note.owner) != Some(note.owner_start))
        .for_each(|(note, path)| {
            let contents = fs::read_to_string(&path).unwrap_or_default();
            let (name, token) = match contents.split_once('\n') {
                Some((name, token)) => (name, Some(token)),
                None => (contents.as_str(), None),
            };
            if Session::noted(note.pid, note.start).kill_noted(token) {
                tracing::warn!(
                    program = %name,
                    pid = note.pid,
                    owner_pid = note.owner,
                    "killed a process left running by a pipette process that exited without \
                     stopping it"
                );
            }
            let _ = fs::remove_file(&path);
        });
}

/// Who started which session, as a file name: `<owner>-<start>-<pid>-<start>`.
#[derive(Debug, PartialEq)]
struct Note {
    owner: u32,
    owner_start: u64,
    pid: u32,
    start: u64,
}

impl Note {
    fn file_name(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.owner, self.owner_start, self.pid, self.start
        )
    }

    fn parse(name: &str) -> Option<Self> {
        let mut parts = name.split('-');
        let note = Self {
            owner: parts.next()?.parse().ok()?,
            owner_start: parts.next()?.parse().ok()?,
            pid: parts.next()?.parse().ok()?,
            start: parts.next()?.parse().ok()?,
        };
        // A group of 0 is the killer's own, and -1 is every process it may
        // signal.
        let signalable = note.pid > 1 && i32::try_from(note.pid).is_ok();
        (parts.next().is_none() && signalable).then_some(note)
    }
}

fn dir() -> Option<PathBuf> {
    dir_in(&std::env::temp_dir())
}

/// The notes' directory under `base`. Only this user may write it: a note names
/// a group to kill.
#[cfg(unix)]
fn dir_in(base: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    // SAFETY: `getuid` takes nothing and cannot fail.
    let uid = unsafe { libc::getuid() };
    let dir = base.join(format!("pipette-{uid}-processes"));
    if let Err(err) = fs::DirBuilder::new().mode(0o700).create(&dir)
        && err.kind() != std::io::ErrorKind::AlreadyExists
    {
        tracing::debug!(
            path = %dir.display(),
            error = %err,
            "failed to create the process notes directory"
        );
        return None;
    }
    let meta = fs::symlink_metadata(&dir).ok()?;
    (meta.is_dir() && meta.uid() == uid && meta.mode() & 0o022 == 0).then_some(dir)
}

#[cfg(not(unix))]
fn dir_in(_base: &Path) -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::a_note("12-345-67-890", Some((12, 345, 67, 890)))]
    #[case::too_few_parts("12-345-67", None)]
    #[case::too_many_parts("12-345-67-890-1", None)]
    #[case::not_a_number("12-345-x-890", None)]
    #[case::its_own_group("12-345-0-890", None)]
    #[case::every_process("12-345-1-890", None)]
    #[case::past_a_pid("12-345-4294967295-890", None)]
    fn only_a_signalable_group_is_read_from_a_name(
        #[case] name: &str,
        #[case] parsed: Option<(u32, u64, u32, u64)>,
    ) {
        let expected = parsed.map(|(owner, owner_start, pid, start)| Note {
            owner,
            owner_start,
            pid,
            start,
        });
        assert_eq!(Note::parse(name), expected);
    }

    /// A note names a group to kill, so notes live only in a directory no one
    /// else can write.
    #[cfg(unix)]
    mod dir {
        use std::os::unix::fs::PermissionsExt;

        use anyhow::Context;
        use rstest::rstest;

        use super::super::*;

        /// Each case changes the directory a first call made.
        #[rstest]
        #[case::made_private(keep, true)]
        #[case::writable_by_others(open_to_all, false)]
        #[case::a_symlink(swap_for_a_symlink, false)]
        fn notes_live_only_where_no_one_else_can_write(
            #[case] change: fn(&Path) -> std::io::Result<()>,
            #[case] kept: bool,
        ) -> anyhow::Result<()> {
            let base = tempfile::tempdir()?;
            let dir = dir_in(base.path()).context("a fresh directory was refused")?;
            assert_eq!(fs::metadata(&dir)?.permissions().mode() & 0o777, 0o700);
            change(&dir)?;
            assert_eq!(dir_in(base.path()).is_some(), kept);
            Ok(())
        }

        fn keep(_: &Path) -> std::io::Result<()> {
            Ok(())
        }

        fn open_to_all(dir: &Path) -> std::io::Result<()> {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o777))
        }

        fn swap_for_a_symlink(dir: &Path) -> std::io::Result<()> {
            let elsewhere = dir.with_extension("elsewhere");
            fs::rename(dir, &elsewhere)?;
            std::os::unix::fs::symlink(&elsewhere, dir)
        }
    }

    /// Real groups, noted in a scratch directory.
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    mod groups {
        use std::os::unix::process::ExitStatusExt;
        use std::process::{Child, Command, ExitStatus};
        use std::time::{Duration, Instant};

        use anyhow::Context;
        use rstest::rstest;

        use super::super::*;

        /// A `sleep` leading its own session, as a start leaves it.
        fn leader() -> anyhow::Result<Child> {
            let mut command = Command::new("sleep");
            command.arg("30");
            Session::lead(&mut command);
            Ok(command.spawn()?)
        }

        /// A starter that has exited and been reaped.
        fn dead_owner() -> anyhow::Result<(u32, u64)> {
            let mut owner = leader()?;
            let start = started(owner.id()).context("no start time")?;
            owner.kill()?;
            owner.wait()?;
            Ok((owner.id(), start))
        }

        fn note(dir: &Path, owner: (u32, u64), pid: u32, start: u64) -> anyhow::Result<PathBuf> {
            noted_in(dir, owner, pid, start, Some("1-2-3"))
        }

        /// A note recording `token` as its session's, or none, as older notes
        /// did.
        fn noted_in(
            dir: &Path,
            owner: (u32, u64),
            pid: u32,
            start: u64,
            token: Option<&str>,
        ) -> anyhow::Result<PathBuf> {
            let note = Note {
                owner: owner.0,
                owner_start: owner.1,
                pid,
                start,
            };
            let path = dir.join(note.file_name());
            let contents = match token {
                Some(token) => format!("llama-server\n{token}"),
                None => "llama-server".to_owned(),
            };
            fs::write(&path, contents)?;
            Ok(path)
        }

        /// Long enough for a SIGKILL the reap sent to land.
        const KILL_LANDS: Duration = Duration::from_millis(500);

        /// How `child` exited, reaping it, if it does within `limit`.
        fn exit_within(child: &mut Child, limit: Duration) -> anyhow::Result<Option<ExitStatus>> {
            let started = Instant::now();
            while started.elapsed() < limit {
                if let Some(status) = child.try_wait()? {
                    return Ok(Some(status));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(None)
        }

        /// A starter that lives keeps its note; a pid handed out again names a
        /// process that started at another time.
        #[rstest]
        #[case::its_starter_died(false, 0, true)]
        #[case::its_starter_lives(true, 0, false)]
        #[case::its_pid_was_handed_out_again(false, 1, false)]
        fn a_noted_group_is_killed_only_if_its_starter_died(
            #[case] starter_lives: bool,
            #[case] start_offset: u64,
            #[case] killed: bool,
        ) -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let mut child = leader()?;
            let start = started(child.id()).context("no start time")?;
            let owner = if starter_lives {
                let me = std::process::id();
                (me, started(me).context("no start time")?)
            } else {
                dead_owner()?
            };
            let path = note(dir.path(), owner, child.id(), start + start_offset)?;

            reap_in(dir.path());
            let limit = if killed {
                Duration::from_secs(5)
            } else {
                KILL_LANDS
            };
            let signal = exit_within(&mut child, limit)?.and_then(|status| status.signal());
            if signal.is_none() {
                child.kill()?;
                child.wait()?;
            }
            assert_eq!(signal, killed.then_some(libc::SIGKILL));
            assert_eq!(path.exists(), starter_lives);
            Ok(())
        }

        /// A leader of a session of its own running `script`, its start
        /// time, the pid `script` printed, and the token it carries.
        fn printing_leader(script: &str) -> anyhow::Result<(Child, u64, u32, Token)> {
            use std::io::{BufRead, BufReader};

            crate::session::test_support::require_perl()?;

            let mut command = Command::new("sh");
            command
                .args(["-c", script])
                .stdout(std::process::Stdio::piped());
            Session::lead(&mut command);
            let token = Token::mark(&mut command);
            let mut leader = command.spawn()?;
            let start = started(leader.id()).context("no start time")?;
            let mut line = String::new();
            BufReader::new(leader.stdout.take().context("no stdout")?).read_line(&mut line)?;
            Ok((leader, start, line.trim().parse()?, token))
        }

        /// A member left in a group of its own, as a `ninja` build step, is
        /// killed after its leader is gone while it carries the token the note
        /// recorded: the leader's pid alone could name another session by
        /// then. Some macOS releases let no one read the environment of
        /// `sleep`, a system binary, so it stays there; `tests/interrupt.rs`
        /// kills one of ours on every release.
        #[rstest]
        #[case::its_token(Recorded::Ours, true)]
        #[case::another(Recorded::Other, false)]
        #[case::unrecorded(Recorded::Nothing, false)]
        fn a_session_whose_leader_is_gone_is_killed_by_its_token(
            #[case] recorded: Recorded,
            #[case] killed: bool,
        ) -> anyhow::Result<()> {
            use crate::session::test_support::{alone_within, ends_within, environment_shown};

            let dir = tempfile::tempdir()?;
            let (mut leader, start, left, token) =
                printing_leader("perl -e 'setpgrp; exec @ARGV' sleep 300 & echo $!")?;
            anyhow::ensure!(
                alone_within(left, Duration::from_secs(10)),
                "pid {left} never left the group"
            );
            leader.wait()?;
            assert_eq!(
                started(leader.id()),
                None,
                "the leader's pid was handed out"
            );
            let token = match recorded {
                Recorded::Ours => Some(token.0.as_str()),
                Recorded::Other => Some("1-2-3"),
                Recorded::Nothing => None,
            };
            noted_in(dir.path(), dead_owner()?, leader.id(), start, token)?;
            let shown = environment_shown(left);
            assert!(shown || cfg!(target_os = "macos"), "pid {left}");
            let killed = killed && shown;

            reap_in(dir.path());
            let limit = Duration::from_millis(if killed { 5000 } else { 500 });
            assert_eq!(ends_within(left, limit), killed, "pid {left}");
            Ok(())
        }

        #[derive(Debug, Clone, Copy)]
        enum Recorded {
            Ours,
            Other,
            Nothing,
        }

        #[test]
        fn a_note_lasts_until_it_is_dropped() -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let mut command = Command::new("sleep");
            command.arg("30");
            Session::lead(&mut command);
            let token = Token::mark(&mut command);
            let mut child = command.spawn()?;
            let record = Record::write_in(dir.path(), child.id(), "llama-bench", &token);
            child.kill()?;
            child.wait()?;
            let record = record.context("nothing noted")?;
            let note = Note::parse(
                record
                    .path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .context("no file name")?,
            )
            .context("an unreadable note")?;
            assert_eq!((note.owner, note.pid), (std::process::id(), child.id()));
            assert_eq!(
                fs::read_to_string(&record.path)?,
                format!("llama-bench\n{}", token.0)
            );

            let path = record.path.clone();
            drop(record);
            assert!(!path.exists());
            Ok(())
        }
    }
}
