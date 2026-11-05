//! OpenVINO-backed pipette client: LFM2 exported to OpenVINO IR, served by
//! `openvino-genai` in a uv-managed venv, on Intel CPU / iGPU / NPU.
//!
//! Runs on Linux and Windows (where Intel NPUs live). Not gated at compile
//! time, unlike `pipette-mlx`: the client refuses any other OS at run time
//! (`pipette-cli`'s `refuse_foreign_os`). Execution runs one in-house
//! OpenAI-compatible server per cell, compiling one pipeline per process;
//! [`execute`] states why. `docs/openvino-ir.md` carries the measurements
//! behind the device limits this crate encodes.

mod completions;
mod execute;
mod flags;
mod models;
mod response;
mod runtimes;
mod server;

pub use execute::run;
