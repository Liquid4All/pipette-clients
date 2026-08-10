//! The concrete Android inference engine.
//!
//! `pipette-android` exposes a single engine object — [`LlamaEngine`] — rather
//! than a grab-bag of stateless functions over a raw model pointer. An engine
//! owns one loaded model and runs benchmarks against it; the Kotlin
//! `EngineActor` owns an engine's lifecycle (create / reuse / destroy) on a
//! single worker thread.
//!
//! The benchmark *measurement* kernel lives in the shared `native/benchmarks.rs`
//! (included via `#[path]` in `lib.rs`, shared verbatim with iOS) — this type is
//! just the Android-facing handle in front of it.

use std::sync::Arc;

use crate::{
    benchmarks, llama, loader, ModelHandle, PipetteError, ProgressCallback, ReadinessCallback,
    ThermalSampler,
};

/// An engine with one model loaded into memory. Created via
/// [`LlamaEngine::create`]; the loaded model is freed on [`Drop`].
pub struct LlamaEngine {
    handle: ModelHandle,
    /// The context size the model was loaded with — i.e. the real KV-cache
    /// capacity available to [`run_benchmark`](LlamaEngine::run_benchmark).
    context_size: u32,
}

impl LlamaEngine {
    /// Load `options.model_path` through the backend loader registry and keep
    /// the model resident. Mirrors the sample app's `Engine.createFromOptions`.
    pub fn create(options: &loader::LoadOptions) -> Result<Self, PipetteError> {
        let handle = loader::load_model(&loader::loaders(), options)?;
        Ok(Self {
            handle,
            context_size: options.context_size,
        })
    }

    /// The context size this engine's model was loaded with.
    pub fn context_size(&self) -> u32 {
        self.context_size
    }

    /// Run a benchmark against the already-loaded model. Delegates to the
    /// shared kernel's `run_benchmark_on_model`, which resets the KV cache and
    /// sampler first so state cannot leak between cells.
    ///
    /// `max_memory_usage` is intentionally rejected here — it must observe the
    /// model load itself, so callers route it through [`run_fresh`].
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

/// Run a benchmark that loads its own model, measures, and unloads — the
/// `max_memory_usage` case, where the load must be part of the measurement.
/// No engine is retained.
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
