//! Process plumbing shared by every `pipette-*` CLI: describing external
//! commands for logs and locating them on `PATH`, the `SIGPIPE` disposition a
//! CLI's `main` has to restore before its first `println!`, and the teardown
//! registry that kills spawned children on ^C.
//!
//! [`cleanup`] stays a named module: the `cleanup::` prefix is what every
//! spawn site already reads by.

mod command;
mod sigpipe;

pub mod cleanup;

pub use command::{argv, echo_debug, echo_info, render, which};
pub use sigpipe::reset_sigpipe;
