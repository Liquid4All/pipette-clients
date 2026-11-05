//! NVIDIA TensorRT Edge-LLM on a Jetson: a server archive that builds TensorRT
//! engines from a GGUF vision pair and serves them over llama-server's HTTP
//! dialect.
//!
//! The unpacked layout lives in `pipette-artifacts-catalog`. [`execute`] builds
//! the engines into the runtime's compile cache outside every measured window,
//! then measures the VL cells the way `pipette-llamacpp` does, through the
//! shared `pipette_completions::llama_server` client;
//! `docs/pipette-cli/trt-edgellm.md` covers the contract.

mod engines;
mod execute;
mod models;
mod runtimes;
mod server;

pub use execute::run;
