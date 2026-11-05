//! ONNX Runtime GenAI-backed pipette client: ONNX exports served by
//! `onnxruntime-genai` in a uv-managed venv, one in-house OpenAI-compatible
//! server per cell.
//!
//! Mirrors `pipette-openvino`, except that each execution provider ships as
//! its own wheel (`onnxruntime-genai`, `onnxruntime-genai-cuda`), so install
//! picks the provider and a host carries one venv per provider. That is why
//! `Runtime::UvOnnxGenai` carries the provider, where an OpenVINO cell picks
//! its device over a single venv.

mod completions;
mod execute;
mod models;
mod response;
mod runtimes;
mod server;

pub use execute::run;
