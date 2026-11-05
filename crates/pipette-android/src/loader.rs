//! Backend loader registry: feature-gated loaders are tried in order and the
//! first that does not return `Unsupported` wins.

use crate::{ModelHandle, PipetteError, llama};

pub struct LoadOptions {
    pub model_path: String,
    pub n_gpu_layers: u32,
    pub context_size: u32,
    /// Prefill micro-batch (llama.cpp `n_ubatch`); 0 means the shim default (512).
    pub n_ubatch: u32,
    /// llama's `llama_load_mode` discriminant, from the cell's `load_mode`
    /// runtime flag; `0` (`none`) is the default.
    pub load_mode: i32,
}

pub enum LoaderResult {
    Success(ModelHandle),
    Error(PipetteError),
    /// This backend can't handle the options; the registry tries the next.
    #[expect(
        dead_code,
        reason = "no second backend yet; part of the loader contract"
    )]
    Unsupported,
}

pub type LoaderFn = fn(&LoadOptions) -> LoaderResult;

/// The first loader that loads, or the first hard error.
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

/// The registered backends, in priority order.
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
        options.load_mode,
    ) {
        Ok(handle) => LoaderResult::Success(handle),
        Err(error) => LoaderResult::Error(error),
    }
}
