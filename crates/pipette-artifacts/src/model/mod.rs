//! Model artifact cache: store and fetch.
//!
//! Find-or-fetch entry point: [`crate::ensure_model`].

mod key;
mod manifest;
pub(crate) mod store;
mod stored;

pub(crate) mod fetch;

pub(crate) use key::{bound_to, slug_from};
pub use key::{ModelStorageKey, ModelStorageKeyError};
pub use manifest::{ModelManifest, BLOBS_DIR_NAME, MANIFEST_VERSION};
pub use store::{ModelArtifactStore, ModelStoreError};
pub use stored::{to_stored, under_root, ModelStoredError};
