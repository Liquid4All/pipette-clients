//! Process plumbing shared by every `pipette-*` CLI: the programs a client
//! starts, waits on and stops ([`start_process`], [`start_docker_process`]), a
//! short command under a deadline ([`output_within`]), command echoing, the
//! `SIGPIPE` disposition, descriptors a parent leaked, a server's loopback port,
//! and teardown of spawned children on ^C or SIGTERM, on a closed stderr when
//! asked ([`stop_on_closed_output`]), and at the next start or [`reap_orphans`]
//! for those a killed CLI left running. On Windows a Job Object also stops them
//! when their parent exits.
//!
//! A started program leads a session of its own, so a stop or an exit takes
//! what it started with it, in any process group.
//!
//! Each line a started program prints goes to [`log_line`] unless the handle's
//! `on_line` names another callback: a debug event, target
//! `pipette_subprocess::output`, inside a `process{program, pid}` span
//! (`container{container, pid}` for docker), so `RUST_LOG` can pick one program
//! (`[process{program=vllm}]=debug`).
//!
//! The `PATH` lookup lives in `pipette-programs`.
#![expect(
    clippy::disallowed_methods,
    reason = "the crate that starts children; each shipped start pipes or discards stdout"
)]

// A session kill scans processes the way these systems list them.
#[cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]
compile_error!("pipette-subprocess runs on Linux, Android, macOS and Windows");

mod cleanup;
mod command;
mod docker_process;
mod env;
mod error;
#[cfg(any(target_os = "linux", target_os = "android"))]
mod inherited_fds;
#[cfg(windows)]
mod job;
mod orphans;
mod output;
mod port;
mod process;
mod session;
mod sigpipe;
mod spawned;

pub use cleanup::{
    BoundedTeardownGuard, LogStderr, SignalTeardown, install_signal_handler, log_stderr, on_signal,
    register_bounded_teardown, register_stop_flag, stop_on_closed_output,
};
pub use command::{echo_debug, echo_info, render};
pub use docker_process::{DockerProcess, DockerRun, start_docker_process};
pub use env::{apply_client_config, apply_envs, forward_client_config};
pub use error::{ExitError, NotReadyError, ReadyError, StartError, StatusError, Stopped};
#[cfg(any(target_os = "linux", target_os = "android"))]
pub use inherited_fds::close_inherited_fds;
pub use orphans::{owner, owner_of, owner_runs, reap_orphans};
pub use output::{OutputError, output_within};
pub use port::{NoFreePortError, pick_server_port};
pub use process::{Process, start_process};
pub use sigpipe::{no_sigpipe_on, reset_sigpipe};
pub use spawned::{Stream, log_line};
