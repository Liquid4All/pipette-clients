//! OpenVINO-backed pipette client: LFM2 exported to OpenVINO IR, served by
//! `openvino-genai` in a uv-managed venv, on Intel CPU / iGPU / NPU.
//!
//! Not platform-gated, unlike `pipette-mlx`. OpenVINO ships x86_64 wheels for
//! both Linux and Windows, and Windows is where Intel NPU hardware actually
//! lives — so this is the first venv-backed runtime that has to work off Linux
//! and macOS. The layout difference that implies (`Scripts\python.exe` rather
//! than `bin/python`) is handled once in `pipette-venv`, not here.
//!
//! Execution is a one-shot Python driver per rep rather than a long-lived
//! server; [`execute`] states why. `docs/openvino-ir.md` carries the
//! measurements behind that and the other device limits this crate encodes.

pub mod catalog;
pub mod execute;
pub mod flags;
pub mod models;
pub mod runtimes;

pub use execute::run;
