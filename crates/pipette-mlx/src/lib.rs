//! MLX-backed pipette client: runs a prepared `RunRequest` the CLI hands it.
//!
//! macOS only: `mlx_lm` and the Metal-side measurement primitives
//! (`pipette-memprobe`'s Metal DYLD shim, `phys_footprint`) exist only on Apple
//! Silicon, so the crate is empty elsewhere.

#![cfg(target_os = "macos")]

mod chat;
mod completions;
mod execute;
mod image_tokens;
mod memory;
mod models;
mod runtimes;
mod server;
mod tokenizer;

pub use execute::run;
