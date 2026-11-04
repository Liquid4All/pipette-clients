//! [`LlamaEngine`]: one loaded model, the Android-facing handle in front of the
//! measurement kernel in `android/native/benchmarks.rs`. The Kotlin
//! `EngineActor` owns its lifecycle on a single worker thread.

use std::sync::Arc;

use crate::{
    ModelHandle, PipetteError, ProgressCallback, ReadinessCallback, ThermalSampler, benchmarks,
    llama, loader,
};

/// The loaded model is freed on [`Drop`].
pub struct LlamaEngine {
    handle: ModelHandle,
    /// The context size the model was loaded with: the real KV-cache capacity.
    context_size: u32,
}

impl LlamaEngine {
    /// Load `options.model_path` through the backend loader registry and keep
    /// the model resident.
    pub fn create(options: &loader::LoadOptions) -> Result<Self, PipetteError> {
        let handle = loader::load_model(&loader::loaders(), options)?;
        Ok(Self {
            handle,
            context_size: options.context_size,
        })
    }

    pub fn context_size(&self) -> u32 {
        self.context_size
    }

    /// Run a benchmark against the loaded model. The kernel resets the KV cache
    /// and sampler first, so state cannot leak between cells. `max_memory_usage`
    /// is rejected: it must observe the load itself, so callers use [`run_fresh`].
    pub fn run_benchmark(
        &self,
        benchmark_json: &str,
        n_gpu_layers: u32,
        mmproj_path: Option<&str>,
        progress: Option<Arc<dyn ProgressCallback>>,
        readiness: Option<Arc<dyn ReadinessCallback>>,
        thermal: Option<Arc<dyn ThermalSampler>>,
    ) -> Result<String, PipetteError> {
        benchmarks::run_benchmark_on_model(
            benchmark_json,
            &self.handle,
            n_gpu_layers,
            self.context_size,
            mmproj_path,
            progress,
            readiness,
            thermal,
        )
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        let _ = llama::unload_model(&self.handle);
    }
}

/// Run a benchmark that loads its own model and unloads it: the
/// `max_memory_usage` case, where the load is part of the measurement.
pub fn run_fresh(
    benchmark_json: &str,
    options: &loader::LoadOptions,
    mmproj_path: Option<&str>,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<String, PipetteError> {
    benchmarks::run_benchmark(
        benchmark_json,
        &options.model_path,
        options.n_gpu_layers,
        options.context_size,
        options.n_ubatch,
        options.load_mode,
        mmproj_path,
        progress,
        readiness,
        thermal,
    )
}

/// The llama.cpp commit the native library was built from.
pub fn llama_cpp_commit() -> String {
    option_env!("LLAMA_CPP_COMMIT")
        .unwrap_or("unknown")
        .to_string()
}

/// The upstream `bNNNN` tag on that commit, or empty when the pin carries none
/// (not `"unknown"`, which would read like a build id and get advertised).
pub fn llama_cpp_tag() -> String {
    option_env!("LLAMA_CPP_TAG").unwrap_or_default().to_string()
}
