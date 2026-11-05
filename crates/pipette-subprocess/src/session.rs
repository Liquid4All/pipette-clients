//! A started child leads a session of its own, which keeps what it starts in
//! any process group, such as `ninja`'s build steps. The kernel reuses no
//! session id while a member lives, so the id outlives the leader. A process
//! that starts a session of its own is found below its parent while that
//! runs; once the parent exits it is init's and lost. No engine's serving
//! path starts one.
//!
//! On Linux and macOS a kill freezes the session and what is below it, then
//! SIGKILLs each, checked against its start time, on Linux through a pidfd so
//! a reused pid is never hit, and waits for them to exit. Windows kills the
//! process tree. A Job Object also stops every child when pipette exits.

#[cfg(unix)]
use std::process::Command;
use std::process::{Child, ExitStatus};

/// A session leader, known by pid and start time so a reused pid is not.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Session {
    leader: u32,
    /// `None` on Windows; on Unix an unread start leaves the session unkillable.
    start: Option<u64>,
}

impl Session {
    /// `setsid` in the child. A failure fails the exec, as a kill would
    /// otherwise reach pipette's own session.
    #[cfg(unix)]
    pub(crate) fn lead(cmd: &mut Command) {
        use std::os::unix::process::CommandExt;

        let hook = || {
            // SAFETY: takes no arguments.
            if unsafe { libc::setsid() } == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        };
        // SAFETY: `setsid` is async-signal-safe.
        unsafe { cmd.pre_exec(hook) };
    }

    pub(crate) fn led_by(pid: u32) -> Self {
        Self {
            leader: pid,
            start: started(pid),
        }
    }

    pub(crate) fn noted(leader: u32, start: u64) -> Self {
        Self {
            leader,
            start: Some(start),
        }
    }

    /// A gone leader's pid stays reserved while the session has members. On
    /// Windows the child's open handle keeps its pid.
    fn is_ours(&self) -> bool {
        match (self.start, started(self.leader)) {
            (Some(recorded), Some(now)) => recorded == now,
            (Some(_), None) => true,
            (None, _) => cfg!(not(unix)),
        }
    }

    /// SIGTERM the leader's group; Windows has none, so it kills the tree.
    pub(crate) fn terminate(&self) {
        #[cfg(unix)]
        if self.is_ours()
            && let Ok(pgid) = libc::pid_t::try_from(self.leader)
        {
            // SAFETY: no pointers.
            unsafe { libc::kill(-pgid, libc::SIGTERM) };
        }
        #[cfg(not(unix))]
        crate::cleanup::kill_tree(self.leader);
    }

    /// [`Self::kill`] for a session a note names. Once its leader is gone the
    /// pid alone could name another of the user's sessions, so it goes only
    /// while one of its processes carries `token` in its environment: the
    /// kernel hands the id out again only once none is left.
    pub(crate) fn kill_noted(&self, token: Option<&str>) -> bool {
        if self.start.is_some() && self.start == started(self.leader) {
            return self.kill();
        }
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
        if let Some(token) = token
            && self.is_ours()
            && scan::carried(self.leader, token)
        {
            return scan::kill(self.leader, &|_| true);
        }
        let _ = token;
        false
    }

    /// Whether anything was left to kill.
    pub(crate) fn kill(&self) -> bool {
        if !self.is_ours() {
            return false;
        }
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
        {
            scan::kill(self.leader, &|_| true)
        }
        #[cfg(not(unix))]
        {
            crate::cleanup::kill_tree(self.leader);
            false
        }
    }

    /// `child`'s exit status once it has exited, reaping it; `None` while it
    /// runs. On Unix the session is killed first, while the unreaped leader
    /// still holds its pid, the session's id. On Windows an exited pid names
    /// no tree, and the handle `Child` keeps holds the pid anyway.
    pub(crate) fn reap(&self, child: &mut Child) -> std::io::Result<Option<ExitStatus>> {
        #[cfg(unix)]
        {
            if !exited_unreaped(child.id())? {
                return Ok(None);
            }
            self.kill();
        }
        child.try_wait()
    }
}

/// Whether child `pid` has exited, left unreaped: until then its pid, and a
/// session id it gave, stay its own.
#[cfg(unix)]
pub(crate) fn exited_unreaped(pid: u32) -> std::io::Result<bool> {
    // Zeroed, so a call that finds no exit leaves `si_signo` 0.
    let mut info = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
    // SAFETY: `info` is a live, writable `siginfo_t`; `WNOWAIT` leaves the
    // child waitable, so its reap still finds it.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid,
            info.as_mut_ptr(),
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: it started zeroed, a valid `siginfo_t`, and `waitid` succeeded,
    // so the kernel filled it or left it so.
    Ok(unsafe { info.assume_init() }.si_signo == libc::SIGCHLD)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
pub(crate) use platform::started;

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
mod scan {
    use std::collections::{BTreeMap, BTreeSet};
    use std::time::{Duration, Instant};

    use super::platform::{environment, processes, running, signal_if_same};

    const FREEZE_ROUNDS: usize = 8;
    const KILL_ROUNDS: usize = 4;
    const EXIT_LIMIT: Duration = Duration::from_secs(10);
    const EXIT_POLL: Duration = Duration::from_millis(20);

    #[derive(Debug, Clone, Copy)]
    pub(super) struct Stat {
        pub(super) parent: u32,
        pub(super) session: u32,
        pub(super) start: u64,
    }

    /// Frozen first, so none forks out of reach, and looked for again after
    /// the kill, as one caught mid-fork or mid-exec can be missing from a list;
    /// then waited for, so none runs once it returns. A GPU can still be
    /// handing back what they held.
    pub(super) fn kill(sid: u32, keep: &dyn Fn(u32) -> bool) -> bool {
        let mut killed = BTreeMap::new();
        for _ in 0..KILL_ROUNDS {
            let frozen = freeze(sid, keep);
            if frozen.is_empty() {
                break;
            }
            killed.extend(
                frozen
                    .into_iter()
                    .filter(|&(pid, start)| signal_if_same(pid, start, libc::SIGKILL)),
            );
        }
        if !exited_within(&killed, EXIT_LIMIT, &running) {
            tracing::debug!(sid, "killed processes still running");
        }
        !killed.is_empty()
    }

    /// Whether every process in `killed` stops running within `limit`.
    pub(super) fn exited_within(
        killed: &BTreeMap<u32, u64>,
        limit: Duration,
        running: &dyn Fn(u32, u64) -> bool,
    ) -> bool {
        let deadline = Instant::now() + limit;
        loop {
            if !killed.iter().any(|(&pid, &start)| running(pid, start)) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(EXIT_POLL);
        }
    }

    fn freeze(sid: u32, keep: &dyn Fn(u32) -> bool) -> BTreeMap<u32, u64> {
        let mut frozen = BTreeMap::new();
        let settled = (0..FREEZE_ROUNDS).any(|_| {
            let more: BTreeMap<u32, u64> = members(sid, &processes())
                .into_iter()
                .filter(|(pid, _)| !frozen.contains_key(pid) && keep(*pid))
                .collect();
            more.iter().for_each(|(&pid, &start)| {
                signal_if_same(pid, start, libc::SIGSTOP);
            });
            let settled = more.is_empty();
            frozen.extend(more);
            settled
        });
        if !settled {
            tracing::debug!(
                sid,
                "still finding processes to freeze; killing those found"
            );
        }
        frozen
    }

    /// Whether a process of session `sid` carries `token` in its environment.
    pub(super) fn carried(sid: u32, token: &str) -> bool {
        let entry = format!("{}={token}", crate::orphans::TOKEN_VAR);
        processes()
            .into_iter()
            .filter(|(_, stat)| stat.session == sid)
            .filter_map(|(pid, _)| environment(pid))
            .any(|environment| {
                environment
                    .split(|&byte| byte == 0)
                    .any(|variable| variable == entry.as_bytes())
            })
    }

    /// Session `sid` and the processes below its members, with start times.
    pub(super) fn members(sid: u32, processes: &BTreeMap<u32, Stat>) -> BTreeMap<u32, u64> {
        let children_of = |parents: &BTreeSet<u32>| -> BTreeSet<u32> {
            processes
                .iter()
                .filter(|(pid, stat)| parents.contains(&stat.parent) && !parents.contains(pid))
                .map(|(&pid, _)| pid)
                .collect()
        };
        let mut found: BTreeSet<u32> = processes
            .iter()
            .filter(|(_, stat)| stat.session == sid)
            .map(|(&pid, _)| pid)
            .collect();
        loop {
            let more: BTreeSet<u32> = children_of(&found).difference(&found).copied().collect();
            if more.is_empty() {
                break;
            }
            found.extend(more);
        }
        found
            .into_iter()
            .filter_map(|pid| Some((pid, processes.get(&pid)?.start)))
            .collect()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use std::collections::BTreeMap;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

    use super::scan::Stat;

    /// A pidfd pins the process, so a matching start time means the signal
    /// reaches it and no reuse of the pid. Without pidfds (before Linux 5.3)
    /// the check and the kill race.
    pub(super) fn signal_if_same(pid: u32, start: u64, signal: libc::c_int) -> bool {
        let Ok(raw) = libc::pid_t::try_from(pid) else {
            return false;
        };
        // SAFETY: no pointers.
        let opened = unsafe { libc::syscall(libc::SYS_pidfd_open, raw, 0) };
        let Ok(fd) = RawFd::try_from(opened) else {
            return false;
        };
        if fd < 0 {
            let gone = std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
            // SAFETY: no pointers.
            return !gone && started(pid) == Some(start) && unsafe { libc::kill(raw, signal) } == 0;
        }
        // SAFETY: a fresh descriptor nothing else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        started(pid) == Some(start)
            // SAFETY: a null `siginfo` is allowed; the descriptor is open.
            && unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    signal,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            } == 0
    }

    /// The fields of `/proc/<pid>/stat` after the command name, which can hold
    /// `)`: state, ppid, pgrp, session, ..., starttime at 19.
    pub(crate) fn fields(stat: &str) -> Option<Vec<&str>> {
        Some(stat.rsplit_once(')')?.1.split_whitespace().collect())
    }

    /// A stat line's parent, session and start; a zombie, already dead, gives
    /// none.
    pub(super) fn parse(stat: &str) -> Option<Stat> {
        let fields = fields(stat)?;
        if *fields.first()? == "Z" {
            return None;
        }
        Some(Stat {
            parent: fields.get(1)?.parse().ok()?,
            session: fields.get(3)?.parse().ok()?,
            start: fields.get(19)?.parse().ok()?,
        })
    }

    pub(super) fn processes() -> BTreeMap<u32, Stat> {
        std::fs::read_dir("/proc")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
                let stat = std::fs::read_to_string(entry.path().join("stat")).ok()?;
                Some((pid, parse(&stat)?))
            })
            .collect()
    }

    /// Running, not a zombie, and still the process that started at `start`.
    pub(super) fn running(pid: u32, start: u64) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| parse(&stat))
            .is_some_and(|stat| stat.start == start)
    }

    /// What `pid` started with in its environment, NUL-separated: only one
    /// of this user's can be read.
    pub(super) fn environment(pid: u32) -> Option<Vec<u8>> {
        std::fs::read(format!("/proc/{pid}/environ")).ok()
    }

    /// In a unit only compared with itself; `None` once gone. A zombie keeps
    /// its own.
    pub(crate) fn started(pid: u32) -> Option<u64> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        fields(&stat)?.get(19)?.parse().ok()
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::collections::BTreeMap;

    use super::scan::Stat;

    /// macOS has no pidfd, so the check and the kill race.
    pub(super) fn signal_if_same(pid: u32, start: u64, signal: libc::c_int) -> bool {
        // SAFETY: no pointers.
        started(pid) == Some(start)
            && libc::pid_t::try_from(pid).is_ok_and(|pid| unsafe { libc::kill(pid, signal) } == 0)
    }

    /// Running, not a zombie, and still the process that started at `start`.
    pub(super) fn running(pid: u32, start: u64) -> bool {
        info(pid)
            .filter(|info| info.pbi_status != libc::SZOMB)
            .and_then(|info| start_of(&info))
            == Some(start)
    }

    pub(super) fn processes() -> BTreeMap<u32, Stat> {
        pids()
            .into_iter()
            .filter_map(|pid| {
                let info = info(pid).filter(|info| info.pbi_status != libc::SZOMB)?;
                // SAFETY: no pointers.
                let session = unsafe { libc::getsid(libc::pid_t::try_from(pid).ok()?) };
                Some((
                    pid,
                    Stat {
                        parent: info.pbi_ppid,
                        session: u32::try_from(session).ok()?,
                        start: start_of(&info)?,
                    },
                ))
            })
            .collect()
    }

    /// A full buffer may cut the oldest pids, the leader's among them, so it
    /// is read again with more room, up to four times.
    fn pids() -> Vec<u32> {
        // SAFETY: a null buffer asks for the count alone.
        let count = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
        let mut room = usize::try_from(count).unwrap_or(0);
        (1..=4)
            .find_map(|attempt| {
                room = room.saturating_mul(2).max(64);
                let mut pids: Vec<libc::c_int> = vec![0; room];
                let bytes = libc::c_int::try_from(room * size_of::<libc::c_int>()).ok()?;
                // SAFETY: `pids` is `bytes` writable bytes.
                let listed = unsafe { libc::proc_listallpids(pids.as_mut_ptr().cast(), bytes) };
                let listed = usize::try_from(listed).unwrap_or(0);
                (listed < room || attempt == 4).then(|| {
                    pids.truncate(listed.min(room));
                    pids.into_iter()
                        .filter_map(|pid| u32::try_from(pid).ok())
                        .collect()
                })
            })
            .unwrap_or_default()
    }

    /// Its arguments, then what it started with in its environment, each
    /// NUL-terminated: only one of this user's can be read, and a system
    /// binary's comes without its environment.
    pub(super) fn environment(pid: u32) -> Option<Vec<u8>> {
        let mut max: libc::c_int = 0;
        let mut size = size_of::<libc::c_int>();
        let mut name = [libc::CTL_KERN, libc::KERN_ARGMAX];
        // SAFETY: `max` is `size` writable bytes; nothing is written back.
        let read = unsafe {
            libc::sysctl(
                name.as_mut_ptr(),
                2,
                (&raw mut max).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if read != 0 {
            return None;
        }
        let mut buffer = vec![0_u8; usize::try_from(max).ok()?];
        let mut size = buffer.len();
        let mut name = [
            libc::CTL_KERN,
            libc::KERN_PROCARGS2,
            libc::c_int::try_from(pid).ok()?,
        ];
        // SAFETY: `buffer` is `size` writable bytes; nothing is written back.
        let read = unsafe {
            libc::sysctl(
                name.as_mut_ptr(),
                3,
                buffer.as_mut_ptr().cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (read == 0).then(|| {
            buffer.truncate(size);
            buffer
        })
    }

    pub(crate) fn info(pid: u32) -> Option<libc::proc_bsdinfo> {
        let pid = libc::c_int::try_from(pid).ok()?;
        let size = libc::c_int::try_from(size_of::<libc::proc_bsdinfo>()).ok()?;
        // SAFETY: an all-zero `proc_bsdinfo` is valid: integers and byte arrays.
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is `size` writable bytes, all the call writes.
        let written = unsafe {
            libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size)
        };
        (written == size).then_some(info)
    }

    fn start_of(info: &libc::proc_bsdinfo) -> Option<u64> {
        info.pbi_start_tvsec
            .checked_mul(1_000_000)?
            .checked_add(info.pbi_start_tvusec)
    }

    /// In a unit only compared with itself; `None` once gone.
    pub(crate) fn started(pid: u32) -> Option<u64> {
        start_of(&info(pid)?)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
pub(crate) fn started(_pid: u32) -> Option<u64> {
    None
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_os = "macos")
))]
pub(crate) mod test_support {
    use std::time::Duration;

    const POLL: Duration = Duration::from_millis(10);

    struct Ids {
        zombie: bool,
        group: u32,
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn ids(pid: u32) -> Option<Ids> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let fields = super::linux::fields(&stat)?;
        Some(Ids {
            zombie: *fields.first()? == "Z",
            group: fields.get(2)?.parse().ok()?,
        })
    }

    #[cfg(target_os = "macos")]
    fn ids(pid: u32) -> Option<Ids> {
        let info = super::macos::info(pid)?;
        Some(Ids {
            zombie: info.pbi_status == libc::SZOMB,
            group: info.pbi_pgid,
        })
    }

    fn within(limit: Duration, holds: impl Fn() -> bool) -> bool {
        (0..=limit.as_millis() / POLL.as_millis()).any(|_| {
            let held = holds();
            if !held {
                std::thread::sleep(POLL);
            }
            held
        })
    }

    /// The tests that move a process out of its group run `perl`; without it
    /// they would pass on a process that never started.
    pub(crate) fn require_perl() -> anyhow::Result<()> {
        let ran = std::process::Command::new("perl")
            .args(["-e", "1"])
            .output()
            .is_ok_and(|output| output.status.success());
        anyhow::ensure!(ran, "this test needs `perl` on PATH");
        Ok(())
    }

    /// Whether `pid`'s environment can be read, which some macOS releases
    /// refuse for a system binary.
    pub(crate) fn environment_shown(pid: u32) -> bool {
        let name = format!("{}=", crate::orphans::TOKEN_VAR);
        super::platform::environment(pid).is_some_and(|environment| {
            environment
                .split(|&byte| byte == 0)
                .any(|variable| variable.starts_with(name.as_bytes()))
        })
    }

    /// Present and not a zombie.
    pub(crate) fn runs(pid: u32) -> bool {
        ids(pid).is_some_and(|ids| !ids.zombie)
    }

    /// One still running then is SIGKILLed, so it does not outlive the test.
    pub(crate) fn ends_within(pid: u32, limit: Duration) -> bool {
        let ended = within(limit, || !runs(pid));
        if !ended && let Ok(pid) = libc::pid_t::try_from(pid) {
            // SAFETY: no pointers.
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        ended
    }

    /// A shell prints a child's pid before the child has left its group.
    pub(crate) fn alone_within(pid: u32, limit: Duration) -> bool {
        within(limit, || ids(pid).is_some_and(|ids| ids.group == pid))
    }
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "android", target_os = "macos")
))]
mod tests {
    use std::collections::BTreeMap;

    use super::Session;
    use super::platform::{signal_if_same, started};
    use super::scan::{Stat, members};
    use super::test_support::runs;

    fn stat(parent: u32, session: u32, start: u64) -> Stat {
        Stat {
            parent,
            session,
            start,
        }
    }

    /// Members in any group, and what they started in a session of its own;
    /// not what is beside them.
    #[test]
    fn a_session_is_its_processes_and_what_they_started() {
        let processes = BTreeMap::from([
            (1, stat(0, 1, 1)),
            (10, stat(1, 10, 2)),  // pipette, leading its own session
            (20, stat(10, 20, 3)), // the server, leading session 20
            (21, stat(20, 20, 4)), // its worker
            (30, stat(1, 20, 5)),  // a build step whose parent exited
            (31, stat(30, 20, 6)), // the compiler it runs
            (40, stat(20, 40, 7)), // a worker in a session of its own
            (41, stat(40, 40, 8)), // and what that runs
            (50, stat(10, 10, 9)), // something else pipette started
            (60, stat(1, 60, 10)), // another session
        ]);
        let found: Vec<(u32, u64)> = members(20, &processes).into_iter().collect();
        assert_eq!(
            found,
            [(20, 3), (21, 4), (30, 5), (31, 6), (40, 7), (41, 8)]
        );
    }

    /// Only the process that started at the time given, never a gone one.
    #[test]
    fn a_signal_reaches_only_the_process_recorded() -> anyhow::Result<()> {
        use anyhow::Context;

        let mut child = std::process::Command::new("sleep").arg("30").spawn()?;
        let pid = child.id();
        let start = started(pid).context("no start time")?;
        assert!(!signal_if_same(pid, start.wrapping_add(1), libc::SIGKILL));
        assert!(runs(pid));
        assert!(signal_if_same(pid, start, libc::SIGKILL));
        child.wait()?;
        assert!(!signal_if_same(pid, start, libc::SIGKILL));
        Ok(())
    }

    /// An unknown start time could be a reused pid, so it is not killed.
    #[test]
    fn only_a_session_of_known_start_is_killed() -> anyhow::Result<()> {
        use anyhow::Context;

        let mut command = std::process::Command::new("sleep");
        command.arg("30");
        Session::lead(&mut command);
        let mut child = command.spawn()?;
        let pid = child.id();
        started(pid).context("no start time")?;
        let unknown = Session {
            leader: pid,
            start: None,
        };
        assert!(!unknown.kill());
        assert!(runs(pid));
        assert!(Session::led_by(pid).kill());
        // The kill returns once what it killed has exited.
        assert!(!runs(pid));
        child.wait()?;
        Ok(())
    }

    /// An exit is seen and the child left for its reap.
    #[test]
    fn an_exit_is_seen_before_the_reap() -> anyhow::Result<()> {
        use std::time::{Duration, Instant};

        use super::exited_unreaped;

        let mut child = std::process::Command::new("sleep").arg("0.1").spawn()?;
        let pid = child.id();
        assert!(!exited_unreaped(pid)?);
        let since = Instant::now();
        while !exited_unreaped(pid)? {
            anyhow::ensure!(since.elapsed() < Duration::from_secs(5), "no exit seen");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(child.try_wait()?.is_some_and(|status| status.success()));
        Ok(())
    }

    /// A kill waits until what it killed stops running, and gives up at its
    /// limit.
    #[test]
    fn a_kill_waits_for_its_processes_to_exit() {
        use std::cell::Cell;
        use std::time::Duration;

        use super::scan::exited_within;

        let killed = BTreeMap::from([(41, 7)]);
        let polls = Cell::new(0);
        let dies_on_the_third = |_, _| {
            polls.set(polls.get() + 1);
            polls.get() < 3
        };
        assert!(exited_within(
            &killed,
            Duration::from_secs(5),
            &dies_on_the_third
        ));
        assert_eq!(polls.get(), 3);
        assert!(!exited_within(
            &killed,
            Duration::from_millis(50),
            &|_, _| true
        ));
    }

    /// Past a command name holding spaces and parentheses; a cut line and a
    /// zombie give none.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn a_stat_line_gives_parent_session_and_start() {
        let line = "4242 (python (vllm) x) S 41 4240 4239 0 -1 4194560 100 0 0 0 5 1 0 0 \
                    20 0 3 0 987654 1000 200 18446744073709551615";
        let parsed = super::linux::parse(line).map(|stat| (stat.parent, stat.session, stat.start));
        assert_eq!(parsed, Some((41, 4239, 987_654)));
        assert!(super::linux::parse("4242 (sh) S 41").is_none());
        assert!(super::linux::parse(&line.replacen(") S ", ") Z ", 1)).is_none());
    }
}
