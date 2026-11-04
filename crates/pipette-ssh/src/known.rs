//! Where the agents a person actually runs put their sockets.
//!
//! `SSH_AUTH_SOCK` names one agent, but a host commonly runs several (1Password
//! and gpg-agent at once), and a person asked to pick a key should be offered
//! every one, not the one that happened to win the environment variable.
//!
//! Probing is a `stat` per path on unix, so a candidate absent on some host
//! costs nothing there.

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

use crate::agent::{AUTH_SOCK_ENV, Agent};

/// An agent found on this host, and what to call it when asking a person to
/// choose.
#[derive(Debug, Clone)]
pub struct KnownAgent {
    pub label: &'static str,
    pub agent: Agent,
}

/// Every agent socket present on this host, `SSH_AUTH_SOCK` first.
///
/// Presence means a socket is there, not that anything is listening: a stale
/// socket file outlives the agent that made it. Callers learn the difference
/// from [`Agent::ed25519_identities`], which is the only thing that can.
pub fn discover() -> Vec<KnownAgent> {
    present(candidates())
}

fn candidates() -> Vec<(&'static str, PathBuf)> {
    let mut candidates: Vec<(&'static str, PathBuf)> = Vec::new();
    if let Some(socket) = env::var_os(AUTH_SOCK_ENV).filter(|socket| !socket.is_empty()) {
        candidates.push((AUTH_SOCK_ENV, PathBuf::from(socket)));
    }
    candidates.extend(vendor_defaults());
    candidates
}

/// 1Password and gpg-agent are confirmed against a live macOS host; the rest
/// are each vendor's documented default.
///
/// KeePassXC is deliberately absent: it adds keys to the OpenSSH agent rather
/// than serving its own socket, so `SSH_AUTH_SOCK` already covers it.
#[cfg(unix)]
fn vendor_defaults() -> Vec<(&'static str, PathBuf)> {
    let mut defaults = Vec::new();
    if let Some(home) = directory("HOME") {
        defaults.extend(home_defaults(&home));
    }
    if let Some(runtime) = directory("XDG_RUNTIME_DIR") {
        defaults.extend(session_defaults(&runtime));
    }
    defaults
}

#[cfg(unix)]
fn home_defaults(home: &Path) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("1Password", home.join(".1password/agent.sock")),
        ("Bitwarden", home.join(".bitwarden-ssh-agent.sock")),
        ("gpg-agent", home.join(".gnupg/S.gpg-agent.ssh")),
        (
            "Secretive",
            home.join("Library/Containers/com.maxgoedjen.Secretive.SecretAgent/Data/socket.ssh"),
        ),
    ]
}

/// gpg-agent is here as well as in [`home_defaults`]: with `/run/user/$UID`
/// present it serves from there and leaves the home path absent or a
/// `%Assuan%` redirect file; without one, the home path is the socket.
#[cfg(unix)]
fn session_defaults(runtime: &Path) -> Vec<(&'static str, PathBuf)> {
    vec![
        ("gpg-agent", runtime.join("gnupg/S.gpg-agent.ssh")),
        ("gcr-ssh-agent", runtime.join("gcr/ssh")),
        ("gnome-keyring", runtime.join("keyring/ssh")),
        ("ssh-agent service", runtime.join("ssh-agent.socket")),
    ]
}

#[cfg(unix)]
fn directory(variable: &str) -> Option<PathBuf> {
    env::var_os(variable)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Windows agents are named pipes rather than files. OpenSSH's own service
/// publishes this one; 1Password's client points `SSH_AUTH_SOCK` at its pipe,
/// so it arrives through [`candidates`] instead.
#[cfg(windows)]
fn vendor_defaults() -> Vec<(&'static str, PathBuf)> {
    vec![(
        "OpenSSH service",
        PathBuf::from(r"\\.\pipe\openssh-ssh-agent"),
    )]
}

/// Keep the candidates this host actually has, first mention winning.
///
/// Deduplicated by resolved path because one agent is often reachable under
/// two names: on macOS `SSH_AUTH_SOCK` is commonly the symlink
/// `~/.1password/agent.sock`, which is also probed directly.
fn present(candidates: Vec<(&'static str, PathBuf)>) -> Vec<KnownAgent> {
    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter(|(_, socket)| exists(socket))
        .filter(|(_, socket)| seen.insert(socket.canonicalize().unwrap_or_else(|_| socket.clone())))
        .map(|(label, socket)| KnownAgent {
            label,
            agent: Agent::at(socket),
        })
        .collect()
}

/// A named pipe does not answer `stat` the way a file does, so on Windows the
/// probe is an open: it fails at once when no agent serves the pipe, and the
/// handle is dropped unused when one does. Without it every listing on a host
/// without the OpenSSH service would report that pipe as unreadable.
#[cfg(windows)]
fn exists(socket: &Path) -> bool {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(socket)
        .is_ok()
}

/// A socket, not merely a path: gpg-agent leaves a `%Assuan%` redirect file
/// where its socket used to be, which could only fail at connect.
#[cfg(unix)]
fn exists(socket: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;

    std::fs::metadata(socket).is_ok_and(|meta| meta.file_type().is_socket())
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::net::UnixListener;

    use anyhow::Context;

    use super::*;

    /// One agent reachable under two names, plus a path nothing put there.
    #[test]
    fn keeps_what_exists_once_each() -> anyhow::Result<()> {
        let dir = tempfile::TempDir::new()?;
        let socket = dir.path().join("agent.sock");
        let _listener = UnixListener::bind(&socket)?;
        let link = dir.path().join("linked.sock");
        std::os::unix::fs::symlink(&socket, &link)?;

        let found = present(vec![
            ("env", link),
            ("vendor", socket.clone()),
            ("absent", dir.path().join("nothing.sock")),
        ]);

        let only = found.first().context("the live socket is kept")?;
        assert_eq!(found.len(), 1, "got {found:?}");
        assert_eq!(only.label, "env", "first mention names it");
        Ok(())
    }

    #[test]
    fn offers_a_linux_session_without_the_gpg_redirect_file() -> anyhow::Result<()> {
        let dir = tempfile::TempDir::new()?;
        let home = dir.path().join("home");
        let runtime = dir.path().join("run");
        let _listeners = session_defaults(&runtime)
            .into_iter()
            .map(|(_, socket)| -> anyhow::Result<UnixListener> {
                std::fs::create_dir_all(socket.parent().context("a socket has a directory")?)?;
                Ok(UnixListener::bind(&socket)?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let redirect = home.join(".gnupg/S.gpg-agent.ssh");
        std::fs::create_dir_all(redirect.parent().context("the redirect has a directory")?)?;
        std::fs::write(
            &redirect,
            b"%Assuan%\nsocket=/run/user/1000/gnupg/S.gpg-agent.ssh\n",
        )?;

        let mut candidates = home_defaults(&home);
        candidates.extend(session_defaults(&runtime));
        let found = present(candidates);

        let labels: Vec<&str> = found.iter().map(|known| known.label).collect();
        assert_eq!(
            labels,
            vec![
                "gpg-agent",
                "gcr-ssh-agent",
                "gnome-keyring",
                "ssh-agent service"
            ],
            "got {found:?}"
        );
        let gpg = found.first().context("gpg-agent is offered")?;
        assert_eq!(
            gpg.agent.socket(),
            runtime.join("gnupg/S.gpg-agent.ssh"),
            "the runtime socket, not the redirect file"
        );
        Ok(())
    }
}
