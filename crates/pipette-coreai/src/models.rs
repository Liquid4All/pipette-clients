//! Bound Apple Core AI model-path projection from a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_model` (an `.aimodel` bundle is a
//! directory, so it materializes like any directory-shaped model).

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::ModelType;

/// Files the Core AI bundle needs to load.
///
/// `metadata.json` only: the `.aimodel` weights and tokenizer live under the
/// same bundle directory, but the Swift sidecar (`LanguageBundle(at:)`)
/// resolves their exact names from metadata — so requiring `metadata.json`
/// catches a download that stopped before anything landed and leaves the rest
/// to Core AI.
const REQUIRED_FILES: &[&str] = &["metadata.json"];

/// Bound Core AI bundle directory: `Model::CoreAi` + `AbsoluteDir` after
/// ensure/bind.
pub fn require_coreai_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_model_dir(&req.model.bound, ModelType::CoreAi, REQUIRED_FILES)
}
