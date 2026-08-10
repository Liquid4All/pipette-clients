//! uv-provisioned Python venvs for pipette runtimes.
//!
//! Everything a venv-backed runtime — vLLM, SGLang, desktop MLX — needs from a
//! venv, in one place:
//!
//! - [`VersionCatalog`] (and the [`parse_catalog_rows`] /
//!   [`unknown_catalog_entry`] pieces under it): the bundled-catalog machinery
//!   each backend's `catalog.toml` is read through
//! - [`install_venv`]: create the venv and populate it from a declared
//!   [`UvRuntimeSource`](pipette_plan_types::UvRuntimeSource)
//! - [`venv_bin`] / [`venv_python`] / [`require_bound_venv`]: where things live
//!   inside one, and the bound-path projection the run path uses
//! - [`resolve_uv`] / [`verify_uv`]: locating and probing the `uv` binary
//! - [`assert_torch_available`]: the post-install check a GPU torch venv needs
//!
//! Split out of `pipette-ops` so the backends can depend on the venv concern
//! without the whole artifact/store/results surface, and so the layout has a
//! single owner: it is created here, so it is described here rather than
//! re-derived by each reader.
//!
//! The modules are private: everything public is re-exported here, so there is
//! one path per item rather than two.

mod catalog;
mod install;
mod layout;
mod torch;
mod uv;

pub use catalog::{parse_catalog_rows, unknown_catalog_entry, VersionCatalog};
pub use install::{install_venv, VenvInstall};
pub use layout::{require_bound_venv, venv_bin, venv_python};
pub use torch::assert_torch_available;
pub use uv::{resolve_uv, verify_uv, UvError};
