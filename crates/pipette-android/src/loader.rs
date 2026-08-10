//! Backend loader registry for the Android kernel.
//!
//! Copied from the liquid_executorch `ie-jni` loading approach
//! (`ie_core::loader::engine_from_options`): a list of feature-gated
//! backend loaders is tried in order, and the first that can load the
//! model wins (`Unsupported` falls through to the next). Unlike the iOS
//! build — which statically links a single kernel because iOS can't load
//! kernels dynamically — Android selects its backend this way.
//!
//! The `Engine` trait abstraction from ie-core is intentionally *not*
//! copied; loaders return the concrete [`ModelHandle`].

use crate::{llama, ModelHandle, PipetteError};

/// Options for loading a model — a trimmed analog of ie-core
/// `EngineOptions`, limited to what the backends actually need to load.
pub struct LoadOptions {
    pub model_path: String,
    pub n_gpu_layers: u32,
    pub context_size: u32,
    /// Prefill micro-batch (llama.cpp `n_ubatch`); 0 → shim default (512).
    pub n_ubatch: u32,
}

/// Outcome of a single backend loader. `Unsupported` means "this backend
/// can't handle these options" — the registry then tries the next loader.
pub enum LoaderResult {
    Success(ModelHandle),
    Error(PipetteError),
    /// This backend can't handle the options; the registry tries the next.
    /// Unused while llama.cpp is the only backend, but part of the loader
    /// contract for additional backends.
    #[allow(dead_code)]
    Unsupported,
}

pub type LoaderFn = fn(&LoadOptions) -> LoaderResult;

/// Try each backend loader in priority order; return the first that loads
/// (or the first hard error). Mirrors `engine_from_options`.
pub fn load_model(
    loaders: &[LoaderFn],
    options: &LoadOptions,
) -> Result<ModelHandle, PipetteError> {
    loaders
        .iter()
        .find_map(|loader| match loader(options) {
            LoaderResult::Success(handle) => Some(Ok(handle)),
            LoaderResult::Error(error) => Some(Err(error)),
            LoaderResult::Unsupported => None,
        })
        .unwrap_or_else(|| {
            Err(PipetteError::ModelLoad {
                msg: format!("no backend loader available for {}", options.model_path),
            })
        })
}

/// The registered backends, in priority order. Feature-gated so a build
/// only carries the kernels it ships; today llama.cpp is the only one.
pub fn loaders() -> Vec<LoaderFn> {
    vec![
        #[cfg(feature = "llamacpp-backend")]
        llamacpp_loader,
    ]
}

#[cfg(feature = "llamacpp-backend")]
fn llamacpp_loader(options: &LoadOptions) -> LoaderResult {
    match llama::load_model(
        &options.model_path,
        options.n_gpu_layers,
        options.context_size,
        options.n_ubatch,
    ) {
        Ok(handle) => LoaderResult::Success(handle),
        Err(error) => LoaderResult::Error(error),
    }
}
