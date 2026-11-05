//! Bound Apple Core AI model-path projection from a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_model` (an `.aimodel` bundle is a
//! directory, so it materializes like any directory-shaped model).

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::ModelType;
use pipette_plan_types::RunRequest;

/// Files the Core AI bundle needs to load.
///
/// The bundle needs `metadata.json` only. The `.aimodel` weights and the
/// tokenizer are in the same directory, and the Swift sidecar
/// (`LanguageBundle(at:)`) reads their names from the metadata. The
/// `metadata.json` check detects a download that stopped early.
const REQUIRED_FILES: &[&str] = &["metadata.json"];

/// Bound Core AI bundle directory: `Model::AppleCoreAi` + `AbsoluteDir` after
/// ensure/bind.
pub fn require_coreai_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_model_dir(&req.model.bound, ModelType::AppleCoreAi, REQUIRED_FILES)
}
