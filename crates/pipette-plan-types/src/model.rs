//! The Model family: [`Model`] and its per-format variant structs
//! ([`GgufText`], [`GgufVision`], [`Mlx`], [`Torch`], [`Openvino`]), each
//! carrying a per-format source enum ([`GgufTextSource`], [`GgufVisionSource`],
//! [`ModelSource`]), plus [`ModelFlags`] and the gguf-file entries. Re-
//! exported flat from `lib.rs`, so consumers reference these as
//! `pipette_plan_types::Model` etc. without seeing the submodule.

use serde::{Deserialize, Serialize};

use crate::{
    AbsolutePath, AuthToken, BenchmarkType, HfRepo, RelativePath, RepoSubpath, ResourceUrl, Sha256,
};

/// One model deployment, tagged by artifact format.
///
/// TOML authoring shape is a tagged table with `type`:
///
/// ```toml
/// models = [
///   { type = "gguf_text",   source = "huggingface", org = "meta-llama", repo_name = "llama-3.2-1b", path = "Q4_K_M.gguf" },
///   { type = "gguf_vision", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-Vision-3B", model = "q4_K_M.gguf", mmproj = "mmproj-f16.gguf" },
///   { type = "mlx",         source = "huggingface", org = "LiquidAI",   repo_name = "LFM2.5-350M-MLX-4bit" },
///   { type = "torch",       source = "huggingface", org = "meta-llama", repo_name = "Llama-3.2-1B" },
///   { type = "openvino",    source = "huggingface", org = "LiquidAI",   repo_name = "LFM2.5-350M-ov", prefix = "int4-sym-cw" },
/// ]
/// ```
///
/// The serving backend (vLLM, sglang, plain Transformers, etc.) is
/// the `Runtime`'s job — `Torch` does not encode the consumer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Model {
    GgufText(GgufText),
    GgufVision(GgufVision),
    Mlx(Mlx),
    Torch(Torch),
    Openvino(Openvino),
    /// Apple Foundation Models, text variant — a bare marker (the model
    /// ships with the OS, so there's no repo/filename to author). The
    /// `…Text` qualifier leaves room for a future `AppleFoundationVision`.
    AppleFoundationText,
}

/// Resolving a [`ModelFlagRef`] into a typed [`ModelFlags`] failed because the
/// `(benchmark, model)` pair names no cell that carries model flags.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelFlagError {
    /// Model-generation flags apply only to eval cells, so any non-eval
    /// `benchmark` (or an otherwise unsupported pair) is rejected here.
    #[error("no model flags defined for {model:?} on {benchmark:?}")]
    NoSuchCombination {
        benchmark: BenchmarkType,
        model: ModelType,
    },
}

/// Per-cell model-generation flags — a closed enum with one variant per
/// `(benchmark, model)` cell that carries flags, named `<Benchmark><Model>`
/// (mirroring [`crate::RuntimeFlags`]'s `<Benchmark><Runtime><Model>`).
/// Generation flags only affect the chat-templated eval path, so every variant
/// is `Eval…`; a non-eval cell — or a model with no generation knobs (Apple
/// Foundation, whose weights and template ship in the OS) — has no variant and
/// is a [`ModelFlagError::NoSuchCombination`], the structural "non-evals carry
/// no model flags" guarantee. These attach to the *cell* (resolved from the
/// variant's `model_flags`), not to the [`Model`], which stays identity-only.
///
/// Authored flat via [`ModelFlagRef`] (`{ model_type, benchmark_type, …knobs }`);
/// `TryFrom` routes the pair to the one variant. Extends like `RuntimeFlags`:
/// add a knob as a field, or a new `(benchmark, model)` cell as a variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, strum::EnumCount)]
#[serde(try_from = "ModelFlagRef", into = "ModelFlagRef")]
pub enum ModelFlags {
    EvalGgufText { enable_thinking: Option<bool> },
    EvalGgufVision { enable_thinking: Option<bool> },
    EvalMlx { enable_thinking: Option<bool> },
    EvalTorch { enable_thinking: Option<bool> },
}

impl ModelFlags {
    /// The `(benchmark, model)` cell this variant encodes — its identity.
    pub fn axes(&self) -> (BenchmarkType, ModelType) {
        let model = match self {
            ModelFlags::EvalGgufText { .. } => ModelType::GgufText,
            ModelFlags::EvalGgufVision { .. } => ModelType::GgufVision,
            ModelFlags::EvalMlx { .. } => ModelType::Mlx,
            ModelFlags::EvalTorch { .. } => ModelType::Torch,
        };
        (BenchmarkType::Eval, model)
    }

    /// Whether this entry applies to a cell running `benchmark` on `model`.
    pub fn matches(&self, benchmark: BenchmarkType, model: &Model) -> bool {
        self.axes() == (benchmark, ModelType::of(model))
    }

    /// The `enable_thinking` chat-template kwarg override, if set. `Some(_)` is
    /// carried in plan-form `--model-flags` JSON (or the CLI convenience
    /// `--model-enable-thinking`).
    ///
    /// `None` is not "no thinking" — the kwarg is omitted entirely and the
    /// engine's own default applies, which today means thinking-*on* for both
    /// `llama-server` (b9119+) and `mlx_lm` 0.31.3, whose `TokenizerWrapper`
    /// injects it when `<think>` appears in the vocab.
    pub fn enable_thinking(&self) -> Option<bool> {
        match self {
            ModelFlags::EvalGgufText { enable_thinking }
            | ModelFlags::EvalGgufVision { enable_thinking }
            | ModelFlags::EvalMlx { enable_thinking }
            | ModelFlags::EvalTorch { enable_thinking } => *enable_thinking,
        }
    }

    /// Wire-form canonical string for cell identity: `enable_thinking=<bool>`
    /// when set, else `None`. Fed into `pipette-plan`'s state events and the
    /// submission payload, so its spelling is load-bearing for row hashing.
    /// The eval-resume digest serializes this enum itself, not this string,
    /// and so is unaffected by a change here.
    pub fn canonical_string(&self) -> Option<String> {
        self.enable_thinking()
            .map(|v| format!("enable_thinking={v}"))
    }

    /// Wire-form `model_flags` value for a `BenchmarkSubmissionPayload`.
    /// `enable_thinking` only affects eval scoring — throughput/latency/memory
    /// rows are insensitive to it, so carrying the flag on those submissions
    /// would split warehouse joins on a value that had no effect.
    ///
    /// Every variant today is an eval cell carrying only `enable_thinking`, so
    /// the non-eval arms have nothing to report. Adding a second flag is a
    /// compile error in [`ModelFlags::enable_thinking`] and [`ModelFlagRef`],
    /// which both destructure the variants — but *not* here, since this matches
    /// on `BenchmarkType`. Revisit these arms when fixing those: a flag that
    /// affects throughput needs reporting rather than a blanket `None`.
    pub fn submission_string(&self, benchmark_type: BenchmarkType) -> Option<String> {
        match benchmark_type {
            BenchmarkType::Eval => self.canonical_string(),
            BenchmarkType::PrefillThroughput
            | BenchmarkType::DecodeThroughput
            | BenchmarkType::EndToEndLatency
            | BenchmarkType::MaxMemoryUsage
            | BenchmarkType::VlThroughput => None,
        }
    }
}

/// Flat wire form of [`ModelFlags`]: the two axis keys plus the knobs. Exists
/// only at the serde boundary — `TryFrom` routes it to the one variant the
/// `(benchmark, model)` pair names, so a bad pair is a parse error, not a
/// silently-accepted cell.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFlagRef {
    pub model_type: ModelType,
    pub benchmark_type: BenchmarkType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

impl TryFrom<ModelFlagRef> for ModelFlags {
    type Error = ModelFlagError;

    fn try_from(r: ModelFlagRef) -> Result<Self, Self::Error> {
        let enable_thinking = r.enable_thinking;
        Ok(match (r.benchmark_type, r.model_type) {
            (BenchmarkType::Eval, ModelType::GgufText) => {
                ModelFlags::EvalGgufText { enable_thinking }
            }
            (BenchmarkType::Eval, ModelType::GgufVision) => {
                ModelFlags::EvalGgufVision { enable_thinking }
            }
            (BenchmarkType::Eval, ModelType::Mlx) => ModelFlags::EvalMlx { enable_thinking },
            (BenchmarkType::Eval, ModelType::Torch) => ModelFlags::EvalTorch { enable_thinking },
            (benchmark, model) => {
                return Err(ModelFlagError::NoSuchCombination { benchmark, model })
            }
        })
    }
}

impl From<ModelFlags> for ModelFlagRef {
    fn from(flags: ModelFlags) -> Self {
        let (benchmark_type, model_type) = flags.axes();
        ModelFlagRef {
            model_type,
            benchmark_type,
            enable_thinking: flags.enable_thinking(),
        }
    }
}

// ---------------------------------------------------------------------------
// gguf text (single file)
// ---------------------------------------------------------------------------

/// Where a single-file gguf text model's weights live. The `source` tag
/// selects: `huggingface` (`{ org, repo_name, path }`, a file in an HF repo),
/// `local` (`{ path }`, an on-disk file), or `url` (`{ url }`, a direct
/// `http(s)` download). The fetching arms (`huggingface`, `url`) carry an
/// optional `sha256` that verifies the download; a local file is identified by
/// its path (no fetch to verify).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum GgufTextSource {
    #[serde(rename = "huggingface")]
    HuggingFace {
        #[serde(flatten)]
        repo: HfRepo,
        path: RepoSubpath,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<Sha256>,
    },
    /// Portable relative path (store layout / entry-relative). Wire: `relative_file`.
    RelativeFile { path: RelativePath },
    /// Absolute host path after bind. Wire: `absolute_file`.
    AbsoluteFile { path: AbsolutePath },
    Url {
        url: ResourceUrl,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<Sha256>,
    },
}

impl GgufTextSource {
    /// Reference form: `<org>/<repo>[@<revision>]:<path>` for an HF file, else
    /// the local path or the URL. Routes through [`HfRepo::reference`] so a
    /// pinned revision is part of the identity.
    pub fn reference(&self) -> String {
        match self {
            GgufTextSource::HuggingFace { repo, path, .. } => {
                format!("{}:{path}", repo.reference())
            }
            GgufTextSource::RelativeFile { path } => path.to_string(),
            GgufTextSource::AbsoluteFile { path } => path.to_string(),
            GgufTextSource::Url { url, .. } => url.to_string(),
        }
    }

    /// The access token for this source, if any (only the HF arm can carry one).
    pub fn auth_token(&self) -> Option<&AuthToken> {
        match self {
            GgufTextSource::HuggingFace { repo, .. } => repo.auth_token.as_ref(),
            _ => None,
        }
    }
}

/// Single-file gguf model (llama.cpp).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct GgufText {
    #[serde(flatten)]
    pub source: GgufTextSource,
}

// ---------------------------------------------------------------------------
// gguf vision (weights + projector)
// ---------------------------------------------------------------------------

/// Where a VL gguf model's two files (main weights + projector) live. The
/// `source` tag selects: `huggingface` states the repo once and names both
/// files by their repo-relative path; `local` addresses each by its own path;
/// `url` gives each a direct `http(s)` download. The fetching arms
/// (`huggingface`, `url`) carry an optional per-file `_sha256`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum GgufVisionSource {
    #[serde(rename = "huggingface")]
    HuggingFace {
        #[serde(flatten)]
        repo: HfRepo,
        model: RepoSubpath,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_sha256: Option<Sha256>,
        mmproj: RepoSubpath,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mmproj_sha256: Option<Sha256>,
    },
    /// Portable relative paths. Wire: `relative_files`.
    RelativeFiles {
        model: RelativePath,
        mmproj: RelativePath,
    },
    /// Absolute host paths. Wire: `absolute_files`.
    AbsoluteFiles {
        model: AbsolutePath,
        mmproj: AbsolutePath,
    },
    Url {
        model: ResourceUrl,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model_sha256: Option<Sha256>,
        mmproj: ResourceUrl,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mmproj_sha256: Option<Sha256>,
    },
}

impl GgufVisionSource {
    /// Identity for the main weights: `<repo-ref>:<path>` for HF, else the
    /// local path or URL.
    fn model_reference(&self) -> String {
        match self {
            GgufVisionSource::HuggingFace { repo, model, .. } => {
                format!("{}:{model}", repo.reference())
            }
            GgufVisionSource::RelativeFiles { model, .. } => model.to_string(),
            GgufVisionSource::AbsoluteFiles { model, .. } => model.to_string(),
            GgufVisionSource::Url { model, .. } => model.to_string(),
        }
    }

    /// Identity for the projector: `<repo-ref>:<path>` for HF, else the local
    /// path or URL.
    fn mmproj_reference(&self) -> String {
        match self {
            GgufVisionSource::HuggingFace { repo, mmproj, .. } => {
                format!("{}:{mmproj}", repo.reference())
            }
            GgufVisionSource::RelativeFiles { mmproj, .. } => mmproj.to_string(),
            GgufVisionSource::AbsoluteFiles { mmproj, .. } => mmproj.to_string(),
            GgufVisionSource::Url { mmproj, .. } => mmproj.to_string(),
        }
    }

    /// The access token for this source, if any (only the HF arm can carry one).
    pub fn auth_token(&self) -> Option<&AuthToken> {
        match self {
            GgufVisionSource::HuggingFace { repo, .. } => repo.auth_token.as_ref(),
            _ => None,
        }
    }
}

/// VL gguf: main weights + a projector file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct GgufVision {
    #[serde(flatten)]
    pub source: GgufVisionSource,
}

// ---------------------------------------------------------------------------
// mlx / torch / openvino (directory)
// ---------------------------------------------------------------------------

/// Where a directory-style model lives. The `source` tag selects: `huggingface`
/// (`{ org, repo_name }`, optionally under a `prefix` subdirectory for a repo
/// that bundles several variants) or `local` (`{ dir }`). Carries only the
/// location; the artifact format is the enclosing [`Model`] variant's concern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ModelSource {
    #[serde(rename = "huggingface")]
    HuggingFace {
        #[serde(flatten)]
        repo: HfRepo,
        /// Subdirectory within the repo the model lives in; `None` = repo root.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefix: Option<RepoSubpath>,
    },
    /// Portable relative directory. Wire: `relative_dir`.
    RelativeDir { dir: RelativePath },
    /// Absolute host directory. Wire: `absolute_dir`.
    AbsoluteDir { dir: AbsolutePath },
}

impl ModelSource {
    /// Identity string (the warehouse key): for HF, `org/repo[@revision]` (via
    /// [`HfRepo::reference`]) with an optional `:prefix` subdirectory, mirroring
    /// gguf's `repo:path`; for a local model, the directory path.
    pub fn reference(&self) -> String {
        match self {
            ModelSource::HuggingFace { repo, prefix } => match prefix {
                Some(prefix) => format!("{}:{prefix}", repo.reference()),
                None => repo.reference(),
            },
            ModelSource::RelativeDir { dir } => dir.to_string(),
            ModelSource::AbsoluteDir { dir } => dir.to_string(),
        }
    }

    /// The access token for this source, if any (only the HF arm can carry one).
    pub fn auth_token(&self) -> Option<&AuthToken> {
        match self {
            ModelSource::HuggingFace { repo, .. } => repo.auth_token.as_ref(),
            _ => None,
        }
    }
}

/// MLX bundle (quantized safetensors + config), directory-shaped.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Mlx {
    #[serde(flatten)]
    pub source: ModelSource,
}

/// PyTorch / HF Transformers safetensors weights, directory-shaped.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Torch {
    #[serde(flatten)]
    pub source: ModelSource,
}

/// OpenVINO IR bundle, directory-shaped: `openvino_model.xml`/`.bin` plus the
/// `openvino_tokenizer` / `openvino_detokenizer` IR pairs that
/// `openvino_genai.LLMPipeline` tokenizes with — it does not use the
/// `tokenizers` library at runtime, so a directory carrying only the model IR
/// loads but cannot generate.
///
/// A different *format* from [`Torch`], not a different consumer of the same
/// bytes: these are compiled IR, so the same coordinate is not loadable by
/// vLLM/sglang. Weight precision is part of the authored coordinate (the repo
/// name, or a `prefix` subdirectory per variant) the way a gguf quant is part
/// of its filename — it is never inferred from the directory, because
/// `openvino_config.json` records `optimum_version`/`transformers_version` but
/// not the weight format, and is absent altogether from an fp16 export.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Openvino {
    #[serde(flatten)]
    pub source: ModelSource,
}

impl Model {
    /// The access token this model carries for fetching, if any. Only a gated
    /// HF source arm can carry one; local sources and AFM never do.
    pub fn auth_token(&self) -> Option<&AuthToken> {
        match self {
            Model::GgufText(m) => m.source.auth_token(),
            Model::GgufVision(m) => m.source.auth_token(),
            Model::Mlx(m) => m.source.auth_token(),
            Model::Torch(m) => m.source.auth_token(),
            Model::Openvino(m) => m.source.auth_token(),
            Model::AppleFoundationText => None,
        }
    }

    /// A copy with every source's auth token cleared — the form persisted in the
    /// model store, since a plaintext manifest must never carry a secret.
    pub fn without_auth_token(&self) -> Model {
        let mut model = self.clone();
        match &mut model {
            Model::GgufText(m) => {
                if let GgufTextSource::HuggingFace { repo, .. } = &mut m.source {
                    repo.auth_token = None;
                }
            }
            Model::GgufVision(m) => {
                if let GgufVisionSource::HuggingFace { repo, .. } = &mut m.source {
                    repo.auth_token = None;
                }
            }
            Model::Mlx(m) => {
                if let ModelSource::HuggingFace { repo, .. } = &mut m.source {
                    repo.auth_token = None;
                }
            }
            Model::Torch(m) => {
                if let ModelSource::HuggingFace { repo, .. } = &mut m.source {
                    repo.auth_token = None;
                }
            }
            Model::Openvino(m) => {
                if let ModelSource::HuggingFace { repo, .. } = &mut m.source {
                    repo.auth_token = None;
                }
            }
            Model::AppleFoundationText => {}
        }
        model
    }
}

/// Inject `token` as `model`'s HF access token, but only when the source is a
/// gated HF arm that carries no token yet. An explicit token already in the spec
/// wins; a non-HF source (local, URL, AFM) is left untouched. Returns whether the
/// token was applied. The paired [`Model::without_auth_token`] clears it again
/// before the model is persisted or reported.
pub fn inject_hf_auth_token(model: &mut Model, token: AuthToken) -> bool {
    let repo = match model {
        Model::GgufText(m) => match &mut m.source {
            GgufTextSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        },
        Model::GgufVision(m) => match &mut m.source {
            GgufVisionSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        },
        Model::Mlx(m) => match &mut m.source {
            ModelSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        },
        Model::Torch(m) => match &mut m.source {
            ModelSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        },
        Model::Openvino(m) => match &mut m.source {
            ModelSource::HuggingFace { repo, .. } => Some(repo),
            _ => None,
        },
        Model::AppleFoundationText => None,
    };
    match repo {
        Some(repo) if repo.auth_token.is_none() => {
            repo.auth_token = Some(token);
            true
        }
        _ => false,
    }
}

impl std::fmt::Display for GgufText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source.reference())
    }
}

impl std::fmt::Display for GgufVision {
    /// One identity for both files: `{weights}+{projector}`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}+{}",
            self.source.model_reference(),
            self.source.mmproj_reference()
        )
    }
}

impl std::fmt::Display for Mlx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source.reference())
    }
}

impl std::fmt::Display for Torch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source.reference())
    }
}

impl std::fmt::Display for Openvino {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source.reference())
    }
}

/// `{repo}[:{path}]` — the canonical string identifier for a
/// model: what goes on the CLI as `--model <ref>`, what shows up in
/// logs/errors, what the warehouse keys on. Distinct from this type's
/// TOML serialization (serde's `{ type = "...", ... }` tagged-table
/// plan form). Each `Model` variant struct has its own `Display`
/// impl; this enum just dispatches.
impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::GgufText(m) => m.fmt(f),
            Model::GgufVision(m) => m.fmt(f),
            Model::Mlx(m) => m.fmt(f),
            Model::Torch(m) => m.fmt(f),
            Model::Openvino(m) => m.fmt(f),
            // Matches the AFM client's submitted `model_name`
            // (`AFMRuntime.submissionModelName`), so plan refs, warehouse
            // keys, and submissions all agree.
            Model::AppleFoundationText => write!(f, "apple/foundation-text"),
        }
    }
}

/// Model-type discriminant, mirroring the [`Model`] enum's variants by name.
/// Serde and `Display` render `snake_case` (e.g. `"gguf_text"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, strum::Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ModelType {
    GgufText,
    GgufVision,
    Mlx,
    Torch,
    Openvino,
    AppleFoundationText,
}

impl ModelType {
    /// The kind of a concrete [`Model`]. Exhaustive, so adding a `Model`
    /// variant fails to compile until this is updated — no silent drift.
    pub fn of(model: &Model) -> Self {
        match model {
            Model::GgufText(_) => Self::GgufText,
            Model::GgufVision(_) => Self::GgufVision,
            Model::Mlx(_) => Self::Mlx,
            Model::Torch(_) => Self::Torch,
            Model::Openvino(_) => Self::Openvino,
            Model::AppleFoundationText => Self::AppleFoundationText,
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;
    // `plan_toml` builds a full-`Matrix` document; the tests that assert
    // multi-model variant behavior share it. Single-`Model` parse tests
    // deserialize `Model` directly instead.
    use crate::plan::plan_toml;
    use crate::*;

    // ---- newtype constructors --------------------------------------------

    fn org(s: &str) -> anyhow::Result<HfOrg> {
        Ok(HfOrg::try_new(s.to_owned())?)
    }
    fn repo_name(s: &str) -> anyhow::Result<HfRepoName> {
        Ok(HfRepoName::try_new(s.to_owned())?)
    }
    fn bench(s: &str) -> anyhow::Result<BenchmarkId> {
        Ok(BenchmarkId::try_new(s.to_owned())?)
    }

    // ---- gguf_text: both source arms -------------------------------------

    #[test]
    fn gguf_text_from_hf_parses_and_references() -> anyhow::Result<()> {
        let model: Model = toml::from_str(
            "type = \"gguf_text\"\nsource = \"huggingface\"\norg = \"org\"\nrepo_name = \"repo\"\npath = \"Q4_K_M.gguf\"",
        )
        .context("hf gguf_text should parse")?;
        let Model::GgufText(m) = &model else {
            anyhow::bail!("expected GgufText");
        };
        assert!(matches!(m.source, GgufTextSource::HuggingFace { .. }));
        assert_eq!(m.source.reference(), "org/repo:Q4_K_M.gguf");
        assert_eq!(model.to_string(), "org/repo:Q4_K_M.gguf");
        Ok(())
    }

    #[test]
    fn gguf_text_from_local_file_parses_needs_no_auth() -> anyhow::Result<()> {
        // A bare local file addressed by path — no HF coordinate, so
        // structurally no auth; identity is the path itself.
        let model: Model = toml::from_str(
            "type = \"gguf_text\"\nsource = \"absolute_file\"\npath = \"/models/m.gguf\"",
        )
        .context("abs_local gguf_text should parse")?;
        let Model::GgufText(m) = &model else {
            anyhow::bail!("expected GgufText");
        };
        assert!(matches!(m.source, GgufTextSource::AbsoluteFile { .. }));
        assert_eq!(m.source.reference(), "/models/m.gguf");
        assert!(m.source.auth_token().is_none());
        assert!(model.auth_token().is_none());
        assert_eq!(model.to_string(), "/models/m.gguf");
        Ok(())
    }

    #[test]
    fn gguf_text_from_url_parses_with_optional_sha256() -> anyhow::Result<()> {
        // A direct http(s) download; identity is the URL, and the fetch can be
        // pinned with an optional sha256. No HF coordinate ⇒ no HF auth.
        let sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let model: Model = toml::from_str(&format!(
            "type = \"gguf_text\"\nsource = \"url\"\nurl = \"https://cdn.example/m.gguf\"\nsha256 = \"{sha}\""
        ))
        .context("url gguf_text should parse")?;
        let Model::GgufText(m) = &model else {
            anyhow::bail!("expected GgufText");
        };
        assert!(matches!(m.source, GgufTextSource::Url { .. }));
        assert_eq!(model.to_string(), "https://cdn.example/m.gguf");
        assert!(model.auth_token().is_none());
        Ok(())
    }

    // ---- gguf_vision: both source arms -----------------------------------

    #[test]
    fn gguf_vision_from_hf_references_model_and_mmproj() -> anyhow::Result<()> {
        // Repo stated once; both files named by repo-relative path.
        let model: Model = toml::from_str(
            "type = \"gguf_vision\"\nsource = \"huggingface\"\norg = \"org\"\nrepo_name = \"repo\"\nmodel = \"a.gguf\"\nmmproj = \"mmproj.gguf\"",
        )
        .context("hf gguf_vision should parse")?;
        let Model::GgufVision(m) = &model else {
            anyhow::bail!("expected GgufVision");
        };
        assert!(matches!(m.source, GgufVisionSource::HuggingFace { .. }));
        assert_eq!(model.to_string(), "org/repo:a.gguf+org/repo:mmproj.gguf");
        Ok(())
    }

    #[test]
    fn gguf_vision_from_local_needs_no_auth() -> anyhow::Result<()> {
        let model: Model = toml::from_str(
            "type = \"gguf_vision\"\nsource = \"absolute_files\"\nmodel = \"/m/a.gguf\"\nmmproj = \"/m/mmproj.gguf\"",
        )
        .context("abs_local gguf_vision should parse")?;
        let Model::GgufVision(m) = &model else {
            anyhow::bail!("expected GgufVision");
        };
        assert!(matches!(m.source, GgufVisionSource::AbsoluteFiles { .. }));
        assert!(model.auth_token().is_none());
        assert_eq!(model.to_string(), "/m/a.gguf+/m/mmproj.gguf");
        Ok(())
    }

    #[test]
    fn gguf_vision_from_url_references_each_file() -> anyhow::Result<()> {
        // Each file its own http(s) download; identity is both URLs.
        let model: Model = toml::from_str(
            "type = \"gguf_vision\"\nsource = \"url\"\nmodel = \"https://cdn.example/m.gguf\"\nmmproj = \"https://cdn.example/mmproj.gguf\"",
        )
        .context("url gguf_vision should parse")?;
        let Model::GgufVision(m) = &model else {
            anyhow::bail!("expected GgufVision");
        };
        assert!(matches!(m.source, GgufVisionSource::Url { .. }));
        assert!(model.auth_token().is_none());
        assert_eq!(
            model.to_string(),
            "https://cdn.example/m.gguf+https://cdn.example/mmproj.gguf"
        );
        Ok(())
    }

    // ---- mlx / torch / openvino: both source arms -------------------------

    #[test]
    fn openvino_prefix_selects_a_precision_variant() -> anyhow::Result<()> {
        // One repo can carry the whole export matrix, so `prefix` — not the
        // model type — is what separates fp16 from int4-sym-cw. It has to fold
        // into the identity string, or the variants collide in the warehouse.
        let model: Model = toml::from_str(
            "type = \"openvino\"\nsource = \"huggingface\"\norg = \"LiquidAI\"\nrepo_name = \"LFM2.5-350M-ov\"\nprefix = \"int4-sym-cw\"",
        )
        .context("openvino with prefix should parse")?;
        assert_eq!(ModelType::of(&model), ModelType::Openvino);
        assert_eq!(model.to_string(), "LiquidAI/LFM2.5-350M-ov:int4-sym-cw");
        Ok(())
    }

    #[test]
    fn openvino_from_local_dir_parses_needs_no_auth() -> anyhow::Result<()> {
        // The export matrix lands on disk before it is ever published, so the
        // absolute-dir arm is the one an Intel-box run uses first.
        let model: Model = toml::from_str(
            "type = \"openvino\"\nsource = \"absolute_dir\"\ndir = \"/models/lfm2.5-350m-int4-sym-cw\"",
        )?;
        let Model::Openvino(m) = &model else {
            anyhow::bail!("expected Openvino, got {model:?}");
        };
        assert!(matches!(m.source, ModelSource::AbsoluteDir { .. }));
        assert!(model.auth_token().is_none());
        Ok(())
    }

    #[test]
    fn mlx_pins_revision_and_folds_it_into_identity() -> anyhow::Result<()> {
        // A pinned revision must round-trip and appear in the model's identity
        // string (the warehouse key), so two revisions don't collide.
        let model: Model = toml::from_str(
            "type = \"mlx\"\nsource = \"huggingface\"\norg = \"LiquidAI\"\nrepo_name = \"LFM2.5-350M-MLX-4bit\"\nrevision = \"v2\"",
        )
        .context("mlx with revision should parse")?;
        let Model::Mlx(m) = &model else {
            anyhow::bail!("expected Mlx");
        };
        let ModelSource::HuggingFace { repo, .. } = &m.source else {
            anyhow::bail!("expected HuggingFace source");
        };
        assert_eq!(
            repo.revision.as_ref().map(|r| r.to_string()),
            Some("v2".to_owned())
        );
        assert_eq!(model.to_string(), "LiquidAI/LFM2.5-350M-MLX-4bit@v2");
        // Absent revision stays the bare id (back-compatible identity).
        let bare: Model = toml::from_str(
            "type = \"mlx\"\nsource = \"huggingface\"\norg = \"LiquidAI\"\nrepo_name = \"LFM2.5-350M-MLX-4bit\"",
        )?;
        assert_eq!(bare.to_string(), "LiquidAI/LFM2.5-350M-MLX-4bit");
        Ok(())
    }

    #[test]
    fn mlx_from_local_dir_parses_needs_no_auth() -> anyhow::Result<()> {
        // The `Absolute` arm: host-absolute MLX bundle directory
        // path. No HF coordinate, so structurally no auth; identity is the path.
        let model: Model = toml::from_str(
            "type = \"mlx\"\nsource = \"absolute_dir\"\ndir = \"/models/LFM2.5-350M-MLX-4bit\"",
        )
        .context("abs_local mlx should parse")?;
        let Model::Mlx(m) = &model else {
            anyhow::bail!("expected Mlx");
        };
        assert!(matches!(m.source, ModelSource::AbsoluteDir { .. }));
        assert!(m.source.auth_token().is_none());
        assert_eq!(model.to_string(), "/models/LFM2.5-350M-MLX-4bit");
        Ok(())
    }

    #[test]
    fn mlx_hf_prefix_folds_into_identity() -> anyhow::Result<()> {
        // A repo bundling several variants: `prefix` selects the subdirectory,
        // and it joins the identity as `org/repo:prefix` (mirrors gguf's
        // `repo:path`), so two prefixes under one repo don't collide.
        let model: Model = toml::from_str(
            "type = \"mlx\"\nsource = \"huggingface\"\norg = \"mlx-community\"\nrepo_name = \"bundle\"\nprefix = \"4bit\"",
        )
        .context("mlx with prefix should parse")?;
        assert_eq!(model.to_string(), "mlx-community/bundle:4bit");
        Ok(())
    }

    #[test]
    fn torch_from_hf_and_local_parse() -> anyhow::Result<()> {
        let hf: Model =
            toml::from_str("type = \"torch\"\nsource = \"huggingface\"\norg = \"meta-llama\"\nrepo_name = \"Llama-3.2-1B\"")
                .context("hf torch should parse")?;
        let Model::Torch(m) = &hf else {
            anyhow::bail!("expected Torch");
        };
        assert!(matches!(m.source, ModelSource::HuggingFace { .. }));
        assert_eq!(hf.to_string(), "meta-llama/Llama-3.2-1B");

        let local: Model =
            toml::from_str("type = \"torch\"\nsource = \"absolute_dir\"\ndir = \"/models/llama\"")
                .context("abs_local torch should parse")?;
        let Model::Torch(m) = &local else {
            anyhow::bail!("expected Torch");
        };
        assert!(matches!(m.source, ModelSource::AbsoluteDir { .. }));
        assert!(local.auth_token().is_none());
        assert_eq!(local.to_string(), "/models/llama");
        Ok(())
    }

    #[test]
    fn inject_hf_auth_token_only_fills_a_gated_tokenless_source() -> anyhow::Result<()> {
        let token = || AuthToken::try_new("hf_injected".to_owned());

        // Gated HF source with no token: injected.
        let mut hf: Model = toml::from_str(
            "type = \"torch\"\nsource = \"huggingface\"\norg = \"meta-llama\"\nrepo_name = \"Llama-3.2-1B\"",
        )?;
        assert!(inject_hf_auth_token(&mut hf, token()?));
        assert_eq!(hf.auth_token().map(|t| t.as_ref()), Some("hf_injected"));

        // Explicit token already in the spec wins — not overwritten.
        let mut owned: Model = toml::from_str(
            "type = \"torch\"\nsource = \"huggingface\"\norg = \"o\"\nrepo_name = \"r\"\nauth_token = \"hf_declared\"",
        )?;
        assert!(!inject_hf_auth_token(&mut owned, token()?));
        assert_eq!(owned.auth_token().map(|t| t.as_ref()), Some("hf_declared"));

        // Non-HF sources have nowhere to put a token.
        let mut local: Model =
            toml::from_str("type = \"torch\"\nsource = \"absolute_dir\"\ndir = \"/models/llama\"")?;
        assert!(!inject_hf_auth_token(&mut local, token()?));
        assert!(local.auth_token().is_none());

        let mut afm = Model::AppleFoundationText;
        assert!(!inject_hf_auth_token(&mut afm, token()?));
        Ok(())
    }

    // ---- full Model round-trips: one HF, one local -----------------------

    #[test]
    fn hf_model_round_trips_through_toml() -> anyhow::Result<()> {
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: org("meta-llama")?,
                    repo_name: repo_name("llama-3.2-1b")?,
                    revision: None,
                    auth_token: Some(AuthToken::try_new("hf_test_xxx".to_owned())?),
                },
                path: RepoSubpath::try_new("Q4_K_M.gguf".to_owned())?,
                sha256: None,
            },
        });
        let s = toml::to_string(&model)?;
        let round: Model = toml::from_str(&s)?;
        assert_eq!(model, round);
        Ok(())
    }

    #[test]
    fn local_model_round_trips_through_toml() -> anyhow::Result<()> {
        let model = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/models/x".to_owned())?,
            },
        });
        // Absolute = host-absolute path
        let s = toml::to_string(&model)?;
        let round: Model = toml::from_str(&s)?;
        assert_eq!(model, round);
        Ok(())
    }

    #[test]
    fn plan_round_trips_through_toml() -> anyhow::Result<()> {
        // Build a Plan covering every Model variant and every Runtime kind.
        let liquid = HfRepo {
            org: org("LiquidAI")?,
            repo_name: repo_name("LFM2.5-350M-MLX-4bit")?,
            revision: None,
            auth_token: None,
        };
        let meta = HfRepo {
            org: org("meta-llama")?,
            repo_name: repo_name("llama-3.2-1b")?,
            revision: None,
            auth_token: Some(AuthToken::try_new("hf_test_xxx".to_owned())?),
        };
        let vision = HfRepo {
            org: org("LiquidAI")?,
            repo_name: repo_name("LFM2.5-Vision-3B")?,
            revision: None,
            auth_token: None,
        };

        let all_models = NonEmptyVec::try_new(vec![
            Model::GgufText(GgufText {
                source: GgufTextSource::HuggingFace {
                    repo: meta.clone(),
                    path: RepoSubpath::try_new("Q4_K_M.gguf".to_owned())?,
                    sha256: None,
                },
            }),
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::HuggingFace {
                    repo: vision,
                    model: RepoSubpath::try_new("q4_K_M.gguf".to_owned())?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new("mmproj-f16.gguf".to_owned())?,
                    mmproj_sha256: None,
                },
            }),
            Model::Mlx(Mlx {
                source: ModelSource::HuggingFace {
                    repo: liquid.clone(),
                    prefix: None,
                },
            }),
            Model::Torch(Torch {
                source: ModelSource::HuggingFace {
                    repo: meta,
                    prefix: None,
                },
            }),
        ])?;

        let runtimes = [
            Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                    repository_url: default_repository_url(),
                    repository_version: NonEmptyString::try_new("b5000".to_owned())?,
                }),
                flavor: LlamaCppFlavor::WindowsX64Vulkan,
            }),
            Runtime::MlxMacosPipette(MlxMacosPipette {
                version: NonEmptyString::try_new("0.20.0".to_owned())?,
                flavor: MlxMacosPipetteFlavor::MacosArm64,
                source: UvRuntimeSource::PipRequirementsText {
                    contents: NonEmptyString::try_new("mlx-lm==0.20.0\n".to_owned())?,
                    install_flags: None,
                },
            }),
            Runtime::DockerVllm(DockerVllm {
                image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
                image_tag: NonEmptyString::try_new("0.7.3".to_owned())?,
                flavor: VllmFlavor::NvidiaGpu,
            }),
            Runtime::DockerSglang(DockerSglang {
                image_name: NonEmptyString::try_new("lmsysorg/sglang".to_owned())?,
                image_tag: NonEmptyString::try_new("latest-cu130-runtime".to_owned())?,
                flavor: SglangFlavor::NvidiaGpu,
            }),
            Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: UvRuntimeSource::PipRequirementsText {
                    contents: NonEmptyString::try_new("vllm==0.10.0\ntorch==2.5.0\n".to_owned())?,
                    install_flags: None,
                },
            }),
            Runtime::UvSglang(UvSglang {
                server_version: UvServerVersion::try_new("0.4.0".to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: UvRuntimeSource::PipRequirementsText {
                    contents: NonEmptyString::try_new("sglang==0.4.0\n".to_owned())?,
                    install_flags: None,
                },
            }),
        ];

        let variants = NonEmptyVec::try_new(vec![Variant {
            models: all_models,
            runtimes: NonEmptyVec::try_new(runtimes.to_vec())?,
            clients: NonEmptyVec::try_new(vec![ClientId::try_new("ev1_a".to_owned())?])?,
            benchmarks: None,
            runtime_flags: vec![],
            model_flags: vec![],
            benchmark_flags: vec![],
        }])?;

        let original = Matrix {
            benchmarks: Some(NonEmptyVec::try_new(vec![bench(
                "prefill_throughput_256",
            )?])?),
            variants,
        };

        let s = toml::to_string(&original)?;
        let round: Matrix = toml::from_str(&s)?;
        assert_eq!(original, round);
        Ok(())
    }

    #[test]
    fn parses_every_model_variant() -> anyhow::Result<()> {
        let toml_str = plan_toml(
            r#"models = [
  { type = "gguf_text", source = "huggingface", org = "meta-llama", repo_name = "llama-3.2-1b", path = "Q4_K_M.gguf" },
  { type = "gguf_vision", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-Vision-3B", model = "q4_K_M.gguf", mmproj = "mmproj-f16.gguf" },
  { type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" },
  { type = "torch", source = "huggingface", org = "meta-llama", repo_name = "Llama-3.2-1B" },
]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b5000", flavor = "windows-x64-vulkan" }]
clients = ["ev1_c"]"#,
        );

        let req: Matrix = toml::from_str(&toml_str)?;
        let models = &req.variants[0].models;
        assert!(matches!(models[0], Model::GgufText(_)));
        assert!(matches!(models[1], Model::GgufVision(_)));
        assert!(matches!(models[2], Model::Mlx(_)));
        assert!(matches!(models[3], Model::Torch(_)));
        Ok(())
    }

    #[test]
    fn apple_foundation_text_variant_round_trips() -> anyhow::Result<()> {
        let toml_str = plan_toml(
            r#"models = [{ type = "apple_foundation_text" }]
runtimes = [{ type = "apple_foundation" }]
clients = ["ev1_c"]"#,
        );

        let req: Matrix = toml::from_str(&toml_str)?;
        let model = &req.variants[0].models[0];
        let runtime = &req.variants[0].runtimes[0];
        assert!(matches!(model, Model::AppleFoundationText));
        assert!(matches!(runtime, Runtime::AppleFoundation(_)));

        // Canonical strings: `--model` / `--runtime` refs and warehouse keys.
        assert_eq!(model.to_string(), "apple/foundation-text");
        assert_eq!(runtime.to_string(), "apple_foundation");

        assert!(model.auth_token().is_none());

        let emitted = toml::to_string(&req)?;
        let reparsed: Matrix = toml::from_str(&emitted)?;
        assert_eq!(req, reparsed);
        Ok(())
    }

    #[rstest::rstest]
    #[case::gguf_text(ModelType::GgufText, ModelFlags::EvalGgufText { enable_thinking: Some(true) })]
    #[case::gguf_vision(ModelType::GgufVision, ModelFlags::EvalGgufVision { enable_thinking: Some(true) })]
    #[case::mlx(ModelType::Mlx, ModelFlags::EvalMlx { enable_thinking: Some(true) })]
    #[case::torch(ModelType::Torch, ModelFlags::EvalTorch { enable_thinking: Some(true) })]
    fn model_flag_ref_routes_eval_pair_to_its_variant(
        #[case] model_type: ModelType,
        #[case] expected: ModelFlags,
    ) -> anyhow::Result<()> {
        let flags = ModelFlags::try_from(ModelFlagRef {
            model_type,
            benchmark_type: BenchmarkType::Eval,
            enable_thinking: Some(true),
        })?;
        assert_eq!(flags, expected);
        Ok(())
    }

    /// The routing cases above cover every variant. Without this a new variant
    /// would add no coverage and nothing would say so — the case list is
    /// hand-written, so only the count ties it to the enum.
    #[test]
    fn routing_cases_cover_every_variant() -> anyhow::Result<()> {
        use strum::EnumCount as _;

        let mut seen = std::collections::HashSet::new();
        for model_type in [
            ModelType::GgufText,
            ModelType::GgufVision,
            ModelType::Mlx,
            ModelType::Torch,
        ] {
            let flags = ModelFlags::try_from(ModelFlagRef {
                model_type,
                benchmark_type: BenchmarkType::Eval,
                enable_thinking: Some(true),
            })?;
            anyhow::ensure!(
                seen.insert(std::mem::discriminant(&flags)),
                "{model_type:?} duplicates an earlier case's variant"
            );
        }
        anyhow::ensure!(
            seen.len() == ModelFlags::COUNT,
            "cases cover {} variants, enum has {}",
            seen.len(),
            ModelFlags::COUNT
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::prefill(ModelType::GgufText, BenchmarkType::PrefillThroughput)]
    #[case::decode(ModelType::GgufText, BenchmarkType::DecodeThroughput)]
    #[case::e2e(ModelType::GgufText, BenchmarkType::EndToEndLatency)]
    #[case::afm_eval(ModelType::AppleFoundationText, BenchmarkType::Eval)]
    fn model_flag_ref_rejects_unsupported_pair(
        #[case] model_type: ModelType,
        #[case] benchmark: BenchmarkType,
    ) {
        // Model flags are eval-only, and Apple Foundation carries no
        // generation knobs — neither names a variant.
        let err = ModelFlags::try_from(ModelFlagRef {
            model_type,
            benchmark_type: benchmark,
            enable_thinking: Some(true),
        });
        assert!(matches!(err, Err(ModelFlagError::NoSuchCombination { .. })));
    }

    #[test]
    fn model_flags_round_trips_through_flat_wire() -> anyhow::Result<()> {
        let flags = ModelFlags::EvalTorch {
            enable_thinking: Some(false),
        };
        let wire = toml::to_string(&flags)?;
        assert!(wire.contains(r#"model_type = "torch""#), "got: {wire}");
        assert!(wire.contains(r#"benchmark_type = "eval""#), "got: {wire}");
        let round: ModelFlags = toml::from_str(&wire)?;
        assert_eq!(flags, round);
        Ok(())
    }

    #[test]
    fn model_flag_ref_rejects_unknown_field() {
        // deny_unknown_fields catches typos like `enabel_thinking`.
        let toml_str = r#"model_type = "gguf_text"
benchmark_type = "eval"
enabel_thinking = true"#;
        assert!(toml::from_str::<ModelFlags>(toml_str).is_err());
    }

    #[test]
    fn model_flags_enable_thinking_omitted_from_wire_when_unset() -> anyhow::Result<()> {
        let wire = toml::to_string(&ModelFlags::EvalMlx {
            enable_thinking: None,
        })?;
        assert!(
            !wire.contains("enable_thinking"),
            "unset knob must not serialize, got:\n{wire}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case(Some(true), Some("enable_thinking=true"))]
    #[case(Some(false), Some("enable_thinking=false"))]
    #[case(None, None)]
    fn model_flags_canonical_string_reflects_enable_thinking(
        #[case] enable_thinking: Option<bool>,
        #[case] expected: Option<&str>,
    ) {
        let flags = ModelFlags::EvalGgufText { enable_thinking };
        assert_eq!(flags.canonical_string().as_deref(), expected);
    }

    /// Evals are the only benchmark type whose scoring depends on model flags,
    /// so the canonical string passes through unchanged for them.
    #[rstest::rstest]
    #[case(Some(true), Some("enable_thinking=true"))]
    #[case(Some(false), Some("enable_thinking=false"))]
    #[case(None, None)]
    fn model_flags_submission_string_eval_keeps_canonical_form(
        #[case] enable_thinking: Option<bool>,
        #[case] expected: Option<&str>,
    ) {
        let flags = ModelFlags::EvalGgufText { enable_thinking };
        assert_eq!(
            flags.submission_string(BenchmarkType::Eval).as_deref(),
            expected
        );
    }

    /// Throughput / latency / memory rows are insensitive to `enable_thinking`;
    /// carrying it would split warehouse joins on a value that had no effect.
    #[rstest::rstest]
    fn model_flags_submission_string_non_eval_reports_nothing(
        #[values(
            BenchmarkType::PrefillThroughput,
            BenchmarkType::DecodeThroughput,
            BenchmarkType::EndToEndLatency,
            BenchmarkType::MaxMemoryUsage,
            BenchmarkType::VlThroughput
        )]
        benchmark_type: BenchmarkType,
        #[values(Some(true), Some(false), None)] enable_thinking: Option<bool>,
    ) {
        let flags = ModelFlags::EvalGgufText { enable_thinking };
        assert_eq!(flags.submission_string(benchmark_type), None);
    }

    #[rstest::rstest]
    #[case::no_token_omits_from_wire("", false)]
    #[case::token_round_trips("\nauth_token = \"hf_xxx\"", true)]
    fn auth_token_wire_form_round_trips(
        #[case] attrs: &str,
        #[case] expect: bool,
    ) -> anyhow::Result<()> {
        let toml_str = format!(
            r#"type = "gguf_text"
source = "huggingface"
org = "meta-llama"
repo_name = "llama-3.2-1b"
path = "Q4_K_M.gguf"{attrs}"#
        );
        let model: Model = toml::from_str(&toml_str)?;
        let Model::GgufText(m) = &model else {
            return Err(anyhow::anyhow!("expected GgufText, got {model:?}"));
        };
        assert_eq!(m.source.auth_token().is_some(), expect, "parse");
        let emitted = toml::to_string(&model)?;
        assert_eq!(
            emitted.contains("auth_token"),
            expect,
            "wire form ({emitted})",
        );
        // Re-parse the emitted form to guard against formatter drift —
        // the token must survive the round-trip, not just a substring
        // match.
        let reparsed: Model = toml::from_str(&emitted)?;
        let Model::GgufText(m) = &reparsed else {
            return Err(anyhow::anyhow!("reparsed wrong variant"));
        };
        assert_eq!(m.source.auth_token().is_some(), expect, "re-parse");
        Ok(())
    }

    // ---- Doc-comment TOML example stays parseable -----------------------
    //
    // The example in `Model`'s docstring (the four-line `models = [...]`
    // block) is duplicated verbatim here so a change to `tag = "type"` or
    // `rename_all = "snake_case"` breaks this test, prompting the dev to
    // update the doc-comment alongside.
    #[test]
    fn doc_comment_model_example_parses() -> anyhow::Result<()> {
        let toml_str = plan_toml(
            r#"models = [
  { type = "gguf_text",   source = "huggingface", org = "meta-llama", repo_name = "llama-3.2-1b", path = "Q4_K_M.gguf" },
  { type = "gguf_vision", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-Vision-3B", model = "q4_K_M.gguf", mmproj = "mmproj-f16.gguf" },
  { type = "mlx",         source = "huggingface", org = "LiquidAI",   repo_name = "LFM2.5-350M-MLX-4bit" },
  { type = "torch",       source = "huggingface", org = "meta-llama", repo_name = "Llama-3.2-1B" },
]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b5000", flavor = "windows-x64-vulkan" }]
clients = ["ev1_c"]"#,
        );
        // plan-types does not enforce compatibility at deserialize time;
        // semantic validation lives in `Plan::parse` via
        // `Variant::validate_compatibility` (which would reject this
        // multi-kind example). The deserialize round-trip itself is what
        // we're asserting here.
        toml::from_str::<Matrix>(&toml_str)?;
        Ok(())
    }
}
