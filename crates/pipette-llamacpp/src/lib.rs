//! The llama.cpp engine: a file per llama.cpp program it runs (`bench`,
//! `batched_bench`, `server`, `mtmd`), and one per benchmark under `execute`,
//! each starting its tool with `pipette_subprocess` and measuring through a
//! `pipette_ops` entry.

mod asr;
mod batched_bench;
mod bench;
mod eval_pool;
mod execute;
mod gguf_meta;
mod memory;
mod models;
mod mtmd;
mod runtimes;
mod server;
mod thinking_controls;

pub use execute::run;
