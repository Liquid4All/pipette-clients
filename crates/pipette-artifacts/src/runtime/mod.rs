//! Runtime artifact cache: store and installers (docker, uv, mlx, openvino,
//! onnx-genai, llamacpp, ovms, trt_edgellm).
//!
//! Find-or-fetch entry point: [`crate::ensure_runtime`].

mod key;
mod manifest;
pub(crate) mod store;
mod stored;

pub(crate) mod coreai;
pub(crate) mod docker;
pub(crate) mod llamacpp;
pub(crate) mod mlx;
pub(crate) mod onnx_genai;
pub(crate) mod openvino;
pub(crate) mod ovms;
pub(crate) mod trt_edgellm;
pub(crate) mod uv;

pub use key::{RuntimeStorageKey, RuntimeStorageKeyError};
pub use manifest::{RuntimeManifest, RuntimeManifestError};
pub use store::{RuntimeArtifactStore, RuntimeStoreError};
pub use stored::{RuntimeStoredError, to_stored};
