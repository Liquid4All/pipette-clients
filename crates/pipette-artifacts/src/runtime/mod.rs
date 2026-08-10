//! Runtime artifact cache: store and installers (docker, uv, mlx, llamacpp).
//!
//! Find-or-fetch entry point: [`crate::ensure_runtime`].

mod key;
mod manifest;
pub(crate) mod store;
mod stored;

pub(crate) mod docker;
pub(crate) mod llamacpp;
pub(crate) mod mlx;
pub(crate) mod openvino;
pub(crate) mod uv;

pub use key::{RuntimeStorageKey, RuntimeStorageKeyError};
pub use manifest::{RuntimeManifest, RuntimeManifestError};
pub use store::{RuntimeArtifactStore, RuntimeStoreError};
pub use stored::{to_stored, RuntimeStoredError};
