//! `models` — local model-store management for the unified client.
//!
//! Identity/parsing is the shared `pipette_plan_types::Model` (URI or JSON).
//! Fetch goes through the shared
//! [`ModelArtifactStore`](pipette_artifacts::model::ModelArtifactStore); GGUF, MLX, and torch
//! snapshots all land under `.pipette/models/`.

use clap::{Args, Subcommand, ValueEnum};
use tabled::Tabled;

use pipette_artifacts::model::ModelManifest;
use pipette_artifacts::{ensure_model, model_download_size};
use pipette_http::HttpClient;
use pipette_plan_types::{descriptor, ModelType};

use crate::artifact_ref::resolve_model_arg;
use crate::commands::print_table_or;
use crate::hf_auth::inject_env_hf_token;
use crate::model_uri::model_to_uri;
use crate::progress::CellProgress;
use crate::workspace::PipetteWorkspace;

/// Manage locally cached models
#[derive(Args, Debug)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List locally cached models across all runtimes
    List(ListArgs),
    /// Fetch a model into the local store from its declared source
    #[command(after_help = PULL_AFTER_HELP)]
    Pull(PullArgs),
    /// Delete a locally cached model
    Delete(DeleteArgs),
}

pub const PULL_AFTER_HELP: &str = "\
Examples:
  # a GGUF text model from Hugging Face
  pipette models pull --model 'gguf-text://repo=ggml-org/gemma-3-4b-it-GGUF&path=gemma-3-4b-it-Q4_K_M.gguf'

  # a GGUF vision model (model + mmproj, optionally pinned with rev/sha256)
  pipette models pull --model 'gguf-vision://repo=ggml-org/gemma-3-4b-it-GGUF&model=gemma-3-4b-it-Q4_K_M.gguf&mmproj=mmproj-model-f16.gguf'

  # an MLX or torch model (whole repo)
  pipette models pull --model 'mlx://repo=mlx-community/gemma-3-4b-it-4bit'
  pipette models pull --model 'torch://repo=google/gemma-3-4b-it'

  # an OpenVINO IR bundle; `prefix` picks one precision out of a multi-variant repo
  pipette models pull --model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw'

  # a GGUF from a direct URL
  pipette models pull --model 'gguf-text://url=https://example.com/model-Q4_K_M.gguf'

A URL value may not contain a query string, and no value may contain `&`. For
those, pass a JSON Model object instead.";

/// `models list` input.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// How to render the model column
    #[arg(long, value_enum, default_value_t)]
    format: ListFormat,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum ListFormat {
    /// Human identity, `org/repo:path` (default).
    #[default]
    Name,
    /// The importable model URI that round-trips through `models pull`.
    Uri,
}

/// `models pull` input — a self-contained model reference.
#[derive(Args, Debug)]
pub struct PullArgs {
    /// Model URI (`gguf-text://repo=org/r&path=Q4_K_M.gguf`, `mlx://repo=org/r`,
    /// …), a JSON `Model` object, or a `model://sha256=<prefix>` digest from
    /// `models list`. See the examples below
    #[arg(long)]
    model: String,
}

/// `models delete` input — the same self-contained reference `pull` accepts.
#[derive(Args, Debug)]
pub struct DeleteArgs {
    /// Model URI or JSON `Model` object — the same grammar as `models pull`,
    /// whose `--help` carries the notation and examples. A
    /// `model://sha256=<prefix>` digest from `models list` also works
    #[arg(long)]
    model: String,
}

impl ModelsArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        self.command.execute(ws, http)
    }
}

impl ModelsCommand {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        match self {
            ModelsCommand::List(args) => args.execute(ws),
            ModelsCommand::Pull(args) => args.execute(ws, http),
            ModelsCommand::Delete(args) => args.execute(ws),
        }
    }
}

/// One row of `models list`.
#[derive(Tabled)]
struct ModelRow {
    #[tabled(rename = "MODEL")]
    model: String,
    #[tabled(rename = "TYPE")]
    model_type: String,
    #[tabled(rename = "DIGEST")]
    digest: String,
    #[tabled(rename = "FETCHED")]
    fetched: String,
}

impl ModelRow {
    fn new(manifest: &ModelManifest, format: ListFormat) -> anyhow::Result<Self> {
        let model = match format {
            ListFormat::Name => manifest.declared.to_string(),
            // A stored model's declared source is auth-stripped, so a URI renders
            // for almost every model. The exception is a URL carrying a query
            // string: it stores fine (the query is trimmed from the on-disk name)
            // but can't be represented as a URI, so it falls back to identity here.
            ListFormat::Uri => {
                model_to_uri(&manifest.declared).unwrap_or_else(|_| manifest.declared.to_string())
            }
        };
        Ok(Self {
            model,
            model_type: ModelType::of(&manifest.declared).to_string(),
            // The prefix `--model model://sha256=<hex>` takes, and the same id
            // the warehouse stores as `model_descriptor_sha256`.
            digest: descriptor::short_digest(&descriptor::digest(
                &manifest.declared.without_auth_token(),
            )?)
            .to_owned(),
            fetched: manifest.fetched_at_rfc3339()?,
        })
    }
}

impl ListArgs {
    pub fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        let store = ws.models();
        let rows: Vec<ModelRow> = store
            .list()?
            .iter()
            .map(|manifest| ModelRow::new(manifest, self.format))
            .collect::<anyhow::Result<_>>()?;
        print_table_or(&rows, "No models cached.");
        Ok(())
    }
}

impl PullArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        let mut declared = resolve_model_arg(ws, &self.model)?;
        inject_env_hf_token(&mut declared)?;
        // No pin set: `ensure_model` pins the entry it is about to publish.
        let ctx = ws.artifacts(http);
        let store = ws.models();
        // One artifact, so the cell line and the artifact line say the same thing —
        // but a directory model is many files, and the cell line is where their sum
        // shows.
        let progress =
            CellProgress::new(&[model_download_size(&ctx, &store, &declared).unwrap_or(None)]);
        let bound = ensure_model(&ctx.with_progress(progress.sink()), &store, &declared)?;
        // Erased before the summary line, which would otherwise print under a
        // finished bar.
        drop(progress);
        println!("Fetched `{declared}` → {bound}");
        Ok(())
    }
}

impl DeleteArgs {
    pub fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        let declared = resolve_model_arg(ws, &self.model)?;
        let store = ws.models();
        if !store.remove(&declared)? {
            anyhow::bail!("model `{declared}` is not in the local store");
        }
        println!("Deleted `{declared}`");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use time::format_description::well_known::Rfc3339;
    use time::OffsetDateTime;

    use pipette_artifacts::model::to_stored;
    use pipette_artifacts::model::ModelStorageKey;
    use pipette_artifacts::model::MANIFEST_VERSION;
    use pipette_plan_types::{
        GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, Model, RepoSubpath, ResourceUrl,
    };

    use super::*;
    use crate::workspace::test_support::TempWorkspace;

    /// Minimal fixture for row rendering — not a store load path.
    fn manifest_of(declared: Model) -> anyhow::Result<ModelManifest> {
        let fetched_at = OffsetDateTime::parse("2026-01-01T00:00:00Z", &Rfc3339)?;
        let declared = declared.without_auth_token();
        let key = ModelStorageKey::of(&declared)?;
        Ok(ModelManifest {
            manifest_version: MANIFEST_VERSION,
            fetched_at,
            last_used_at: fetched_at,
            stored: to_stored(&declared, &key.blobs_dir())?,
            declared,
            blobs_bytes: Some(4096),
        })
    }

    #[test]
    fn model_row_renders_name_and_the_round_tripping_uri() -> anyhow::Result<()> {
        let manifest = manifest_of(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_owned())?,
                    repo_name: HfRepoName::try_new("repo".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("Q4_K_M.gguf".to_owned())?,
                sha256: None,
            },
        }))?;

        let name = ModelRow::new(&manifest, ListFormat::Name)?;
        assert_eq!(name.model, "org/repo:Q4_K_M.gguf");
        assert_eq!(name.model_type, "gguf_text");

        let uri = ModelRow::new(&manifest, ListFormat::Uri)?;
        assert!(uri.model.starts_with("gguf-text://"), "got: {}", uri.model);
        assert_ne!(uri.model, name.model, "the URI form differs from identity");
        Ok(())
    }

    /// A query-string URL stores fine (the query is trimmed from the on-disk
    /// filename), but `model_to_uri` can't represent it, so `ListFormat::Uri`
    /// falls back to the identity string — the raw URL.
    #[test]
    fn model_row_uri_falls_back_to_identity_for_a_query_url() -> anyhow::Result<()> {
        let manifest = manifest_of(Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://example.com/m-Q4_K_M.gguf?rev=abc".to_owned())?,
                sha256: None,
            },
        }))?;

        let uri = ModelRow::new(&manifest, ListFormat::Uri)?;
        assert_eq!(
            uri.model,
            manifest.declared.to_string(),
            "a non-representable model falls back to its identity string"
        );
        assert!(uri.model.starts_with("https://"), "got: {}", uri.model);
        Ok(())
    }

    /// Deleting a model that was never pulled is a clear error, not a silent
    /// no-op — `store.remove` reports `false` and `delete` turns that into a bail.
    #[test]
    fn delete_absent_model_reports_not_found() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("models-delete")?;
        let err = DeleteArgs {
            model: "gguf-text://repo=org/repo&path=model-Q4_K_M.gguf".to_string(),
        }
        .execute(&tw.ws)
        .err()
        .context("deleting an absent model should error")?;
        assert!(format!("{err:#}").contains("not in the local store"));
        Ok(())
    }

    /// Every URI quoted in `models pull --help` must parse — the help text and
    /// the grammar can't drift.
    #[test]
    fn after_help_examples_parse() -> anyhow::Result<()> {
        let mut failures = Vec::new();
        for line in PULL_AFTER_HELP.lines() {
            let Some((_, uri)) = line.split_once("--model '") else {
                continue;
            };
            let uri = uri.trim_end_matches('\'');
            if let Err(e) = crate::model_uri::parse_model_uri(uri) {
                failures.push(format!("`{uri}`: {e}"));
            }
        }
        assert!(
            failures.is_empty(),
            "help examples must parse:\n{}",
            failures.join("\n")
        );
        Ok(())
    }
}
