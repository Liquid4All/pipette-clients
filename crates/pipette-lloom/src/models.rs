//! Bound model projection: the HF safetensors snapshot `lloom-serve` loads.

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::ModelType;

/// What `lloom-serve --model-dir` reads: the checkpoint's config, its
/// tokenizer, and the safetensors shards the config's index names.
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json"];

/// The bound local directory of a `Torch` (HF snapshot) model.
pub fn require_lloom_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_model_dir(&req.model.bound, ModelType::Torch, REQUIRED_FILES)
}
