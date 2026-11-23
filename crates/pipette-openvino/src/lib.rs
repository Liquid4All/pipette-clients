//! OpenVINO-backed pipette client: models exported to OpenVINO IR, served by
//! `openvino-genai` in a uv-managed venv, on Intel CPU / iGPU / NPU. Text cells
//! run `LLMPipeline`, VL cells `VLMPipeline`.
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
mod memory;
mod models;
mod response;
mod runtimes;
mod server;

pub use execute::run;
