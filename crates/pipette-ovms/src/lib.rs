//! OpenVINO Model Server (OVMS): OpenVINO IR served by an upstream release
//! archive and driven over HTTP.
//!
//! The release pins and the unpacked layout live in `pipette-artifacts-catalog`.
//! [`execute`] measures a cell against a server launched on the
//! `Model::Openvino` directory, one request or one batch in flight;
//! `docs/ovms.md` covers the API contract and how its numbers relate to
//! `uv_openvino`'s.

mod completions;
mod execute;
mod flags;
mod models;
mod runtimes;
mod server;
#[cfg(test)]
mod test_support;

pub use execute::run;
