//! Eval sample completion store: resume state keyed by [`RunRequest`].
//!
//! Layout under `root/` (= workspace `state/evals/`):
//!
//! ```text
//! <digest16>.jsonl          # header + completion lines
//! <digest16>.jsonl.stale-*  # rotated on digest mismatch / corrupt header
//! ```
//!
//! The digest covers portable fields only, never bound host paths, so a cell
//! resumes across machines. An engine appends and never deletes; the client
//! finalizes the file once the result is saved or accepted, so a result lost
//! on its way there is rebuilt from the checkpoint instead of from sample 0.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use nutype::nutype;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkEvalCompletion, BenchmarkFlagRef, BenchmarkFlags,
    ModelFlags as PlanModelFlags, PositiveSeconds, RunRequest, Runtime, RuntimeFlagRef,
    RuntimeFlags, SamplesDigest,
};

use crate::custom_label::unlabelled;
use crate::error::{Error, Result as OpsResult};

/// Hex SHA-256 of the portable run identity. File names use the first 16 chars.
#[nutype(derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, AsRef, Display))]
pub struct EvalRunDigest(String);

/// Operator-facing header fields (`head -1 <file> | jq .meta`). Not hashed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCompletionMeta {
    pub benchmark_id: String,
    /// `Display` of declared runtime.
    pub runtime: String,
    /// `Display` of declared model.
    pub model: String,
}

#[derive(Serialize, Deserialize)]
struct Header {
    digest: EvalRunDigest,
    meta: EvalCompletionMeta,
    /// Outside the digest, but a run under another timeout sends the failed
    /// rows again: the timeout may be all that failed them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    http_timeout_seconds: Option<PositiveSeconds>,
}

/// Capability handle for eval sample completions (`root/` = `state/evals/`).
///
/// Layout is private. A run gets its store from its
/// [`RunContext`](crate::run_context::RunContext).
#[derive(Debug, Clone)]
pub struct EvalCompletionsStore {
    root: PathBuf,
}

impl EvalCompletionsStore {
    /// `root` is this store's directory; it is created on first open, so any
    /// path works. The workspace picks the location (see `PipetteWorkspace`).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn open(&self, req: &RunRequest) -> anyhow::Result<EvalCompletionSession> {
        let header = Header {
            digest: run_request_digest(req)?,
            meta: EvalCompletionMeta {
                benchmark_id: req.benchmark.benchmark_id().to_string(),
                runtime: req.runtime.declared.to_string(),
                model: req.model.declared.to_string(),
            },
            http_timeout_seconds: req
                .benchmark_flags
                .as_ref()
                .and_then(BenchmarkFlags::http_timeout),
        };
        EvalCompletionSession::open_in(&self.root, &header)
    }

    /// Retire the checkpoint of a run whose result is safe: delete it, or keep
    /// only its failed rows, one line per attempt spent, so a later run under
    /// the same HTTP timeout retries each only while it has attempts left
    /// ([`MAX_SAMPLE_ATTEMPTS`]). Nothing to do when there is no checkpoint.
    ///
    /// Reads the file rather than a session, since the engine that wrote it has
    /// returned: the last row per id is the one that counts, as on resume.
    pub fn finalize(&self, digest: &EvalRunDigest) -> anyhow::Result<()> {
        let path = session_path(&self.root, digest);
        if !path.exists() {
            return Ok(());
        }
        // A digest mismatch has already rotated the file away.
        let Some((header, completions, attempts)) = load_existing(&path, digest)? else {
            return Ok(());
        };
        // Attempts are counted from failed lines, so one line per id would
        // reset every failure to its first attempt and retry a broken sample
        // forever.
        let failed: Vec<BenchmarkEvalCompletion> = completions
            .into_iter()
            .filter(|c| c.failed)
            .flat_map(|c| {
                let spent = attempts.get(&c.id).copied().unwrap_or(1).max(1);
                std::iter::repeat_n(c, spent as usize)
            })
            .collect();
        if !failed.is_empty() {
            return write_checkpoint(&path, &header, &failed);
        }
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))
    }

    pub fn clear(&self) -> OpsResult<()> {
        if !self.root.exists() {
            return Ok(());
        }
        fs::remove_dir_all(&self.root).map_err(|source| Error::Io {
            path: self.root.clone(),
            source,
        })
    }
}

/// SHA-256 hex over the request's portable projection, never bound host paths:
/// [`ResumeIdentity`] for a text eval, [`vlm_eval_digest_view`] for a
/// `vlm_eval`, whose samples embed every base64 image and would cost a
/// multi-hundred-MB transient on every `open()` if serialized whole.
pub fn run_request_digest(req: &RunRequest) -> anyhow::Result<EvalRunDigest> {
    if req.benchmark.as_vlm_eval().is_ok() {
        return digest_value(&vlm_eval_digest_view(req)?);
    }
    if req.benchmark.as_asr_eval().is_ok() {
        return digest_value(&asr_eval_identity(req)?);
    }
    let identity = ResumeIdentity {
        // A vault source's `custom_label` names no cell: a labelled claim
        // resumes the checkpoint an unlabelled run of the cell wrote.
        runtime: unlabelled(&req.runtime.declared).context("failed to serialize the runtime")?,
        model: unlabelled(&req.model.declared).context("failed to serialize the model")?,
        runtime_flags: resume_relevant_flags(&req.runtime_flags),
        model_flags: &req.model_flags,
        benchmark_flags: resume_relevant_benchmark_flags(&req.benchmark_flags),
        benchmark: &req.benchmark,
        completion_format: completion_format(&req.runtime.declared),
        client_recipe: None,
    };
    let value = serde_json::to_value(&identity).context("failed to serialize resume identity")?;
    digest_value(&value)
}

/// The version of the client-side llama.cpp recipe an `asr_eval` request
/// runs under (`pipette-llamacpp`'s `asr_eval`: each prompt profile's
/// templates, stop words, card defaults, end-of-turn override and request
/// shape), part of the asr_eval resume identity. Unlike vlm_eval's, that
/// recipe lives in the client, so a client upgrade that changes it must not
/// resume a checkpoint written under the old one and submit transcripts made
/// under two recipes as one cell. `pipette-llamacpp` pins a fingerprint of the
/// recipe per version and fails its tests until this is bumped with it.
///
/// 2: every unpinned knob takes its card's value (plan-types 852a4bf), so an
/// unpinned Gemma 4 entry samples at its card's 1.0 / 0.95 / 64.
/// 3: every request spells its sampler chain, llama.cpp's default chain on
/// every profile but Voxtral, so neither a GGUF's sampling sequence nor a
/// server's `--samplers` can drop or reorder a stage; and a MiniCPM-o beam
/// entry's unpinned penalties are its card's (plan-types 70993d0), so an
/// unpinned beam repetition penalty is 1.2, not 1.0.
pub const ASR_EVAL_RECIPE_VERSION: u32 = 3;

/// The identity an `asr_eval` request resumes under: the text path's
/// [`ResumeIdentity`], over the definition with its corpus pin reduced to the
/// corpus's digest (`corpus_digest`), as vlm_eval's: the digest names every
/// clip and prompt and their order, so another corpus is another run, while
/// the `url` the server stamps on the pin, its `bytes` and the recipe (who
/// may run the entry, not what a run produces) are not identity. The two
/// lessons vlm_eval learnt apply from the start, as no asr_eval checkpoint
/// predates them. The flags are hashed as resolved, so a changed
/// default forks the identity, and the model without its HF auth token, a fetch
/// credential; the runtime envs as a result records them
/// ([`pipette_plan_types::EnvForward::for_record`]: names, and values only in
/// the `GGML_*` namespace), so an explicit `HF_TOKEN=...` is never hashed and
/// rotating it resumes. [`ASR_EVAL_RECIPE_VERSION`] names the client's recipe.
///
/// `parallel` and `ctx_size` are cleared, as on the text path. They decide
/// only whether a sample fits, never what a fitting sample's transcript is: a
/// sample that does not fit, or whose answer the server cuts short for want of
/// context, is never checkpointed (`EvalMeasurementReading::Unfit`), so raising
/// either runs it on the next run without orphaning the rest. vLLM's
/// `max_model_len` and `prefix_caching` are cleared as on vlm_eval
/// ([`without_throughput_knobs`]): a reply its context cuts is a retried
/// failure, never an answer, and the cell refuses the prefix cache.
fn asr_eval_identity(req: &RunRequest) -> anyhow::Result<Value> {
    let asr = req.benchmark.as_asr_eval().map_err(anyhow::Error::from)?;
    let benchmark = BenchmarkDefinition::AsrEval(pipette_plan_types::AsrEvalBenchmark {
        dataset: None,
        recipe: None,
        ..asr.clone()
    });
    let mut runtime_flags = RuntimeFlagRef::from(req.resolved_runtime_flags()?);
    runtime_flags.parallel = None;
    runtime_flags.ctx_size = None;
    runtime_flags.max_model_len = None;
    runtime_flags.prefix_caching = None;
    runtime_flags.envs = runtime_flags
        .envs
        .iter()
        .map(pipette_plan_types::EnvForward::for_record)
        .collect();
    let identity = ResumeIdentity {
        runtime: unlabelled(&req.runtime.declared).context("failed to serialize the runtime")?,
        model: unlabelled(&req.model.declared.without_auth_token())
            .context("failed to serialize the model")?,
        runtime_flags: Some(runtime_flags),
        model_flags: &req.model_flags,
        benchmark_flags: resume_relevant_benchmark_flags(&req.benchmark_flags),
        benchmark: &benchmark,
        completion_format: completion_format(&req.runtime.declared),
        client_recipe: Some(ASR_EVAL_RECIPE_VERSION),
    };
    let mut value =
        serde_json::to_value(&identity).context("failed to serialize resume identity")?;
    if let Some(dataset) = &asr.dataset
        && let Some(benchmark) = value.get_mut("benchmark").and_then(Value::as_object_mut)
    {
        benchmark.insert(
            "corpus_digest".to_owned(),
            serde_json::json!(dataset.digest),
        );
    }
    Ok(value)
}

/// The portable axes cell identity is hashed over.
///
/// A struct rather than a `json!` so a field renamed on [`RunRequest`] fails
/// to build instead of silently leaving the identity. Field order is
/// irrelevant: [`write_canonical`] sorts keys.
#[derive(Serialize)]
struct ResumeIdentity<'a> {
    runtime: Value,
    model: Value,
    runtime_flags: Option<RuntimeFlagRef>,
    model_flags: &'a Option<PlanModelFlags>,
    benchmark_flags: Option<BenchmarkFlagRef>,
    benchmark: &'a BenchmarkDefinition,
    /// Skipped when `None`, so cells whose rows did not change keep their digest.
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_format: Option<&'static str>,
    /// The client-side recipe version, asr_eval's alone
    /// ([`ASR_EVAL_RECIPE_VERSION`]); skipped when `None`, so no other cell's
    /// digest moves.
    #[serde(skip_serializing_if = "Option::is_none")]
    client_recipe: Option<u32>,
}

/// vLLM rows carry the reasoning
/// ([`crate::sse::ChatReply::completion_with_reasoning`]); naming that in the
/// identity keeps a vLLM cell off an older checkpoint of content-only rows.
fn completion_format(runtime: &Runtime) -> Option<&'static str> {
    match runtime {
        Runtime::DockerVllm(_) | Runtime::UvVllm(_) => Some("reasoning_then_answer"),
        Runtime::DockerSglang(_)
        | Runtime::UvSglang(_)
        | Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleFoundation(_)
        | Runtime::AppleCoreAiMacosPipette(_) => None,
    }
}

/// `runtime_flags` with the throughput-only knobs cleared, so retuning them
/// (dropping `parallel` after an OOM, trimming `ctx_size`) does not re-run a
/// dataset that may be hours in. Neither changes what a completion is; a
/// `ctx_size` truncation shows up per sample as `stop_reason`.
///
/// Typed fields rather than JSON keys, so a rename breaks the build.
/// `RuntimeFlags` serializes through [`RuntimeFlagRef`], so the ref changes
/// nothing else about the hash.
fn resume_relevant_flags(flags: &Option<RuntimeFlags>) -> Option<RuntimeFlagRef> {
    flags.clone().map(|flags| {
        let mut flags = RuntimeFlagRef::from(flags);
        flags.parallel = None;
        flags.ctx_size = None;
        flags
    })
}

/// [`resume_relevant_flags`]'s rule for resolved flags, which the vlm_eval
/// view's `RunRequest` literal needs back as [`RuntimeFlags`], plus the vLLM
/// cell's `max_model_len` and `prefix_caching`: vLLM holds each sample whole or
/// refuses it, so its context changes no answer, and the cell runs with the
/// prefix cache off whatever it says. The round trip cannot fail for a valid
/// variant; it is propagated so a future field with a stricter rule cannot
/// pass silently.
fn without_throughput_knobs(flags: RuntimeFlags) -> anyhow::Result<RuntimeFlags> {
    let mut r = RuntimeFlagRef::from(flags);
    r.parallel = None;
    r.ctx_size = None;
    r.max_model_len = None;
    r.prefix_caching = None;
    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

/// The digest projection for a `vlm_eval` request: [`RunRequest`]'s own
/// `Serialize` with the samples replaced by their identity. A catalog entry's
/// samples are its corpus, so only the pin's digest is hashed
/// (`corpus_digest`); a local definition's are hashed one at a time
/// ([`SamplesDigest`], the evals server's canonical form) into
/// `samples_sha256`, so the images are never one buffer. A `RunRequest`
/// literal, so a field added to the request fails to build here.
///
/// The model is hashed without its HF auth token, a fetch credential: rotating
/// it must not fork the identity, and a secret stays out of the hash input.
/// The text path keeps the token, since changing its digest would orphan every
/// text checkpoint on disk.
///
/// The runtime flags are hashed as [`RunRequest::resolved_runtime_flags`], so
/// an omitted entry hashes as the cell's all-unset variant, then without the
/// throughput knobs as on the text path. Thinking mode enters through the
/// model flags.
fn vlm_eval_digest_view(req: &RunRequest) -> anyhow::Result<Value> {
    use pipette_plan_types::VlmEvalBenchmark;

    let vlm = req.benchmark.as_vlm_eval().map_err(anyhow::Error::from)?;
    let stripped = RunRequest {
        runtime: req.runtime.clone(),
        model: pipette_plan_types::DeclaredBound {
            // Only `declared` is serialized; `bound` fills the struct.
            declared: req.model.declared.without_auth_token(),
            bound: req.model.bound.clone(),
        },
        runtime_flags: Some(without_throughput_knobs(req.resolved_runtime_flags()?)?),
        model_flags: req.model_flags.clone(),
        benchmark_flags: req.benchmark_flags.clone(),
        benchmark: BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
            dataset: None,
            recipe: None,
            samples: None,
            ..vlm.clone()
        }),
    };
    let mut view = serde_json::to_value(&stripped)?;
    if let Some(benchmark) = view.get_mut("benchmark").and_then(Value::as_object_mut) {
        // An edited prompt or image under an old id is a new run, not one to
        // resume the old answers into.
        let samples = match &vlm.dataset {
            Some(dataset) => ("corpus_digest", serde_json::json!(dataset.digest)),
            None => {
                let mut digest = SamplesDigest::new();
                for sample in vlm.samples.as_deref().unwrap_or(&[]) {
                    digest.update(sample)?;
                }
                let hex = digest.finish().map_err(anyhow::Error::from)?;
                ("samples_sha256", serde_json::json!(hex.as_ref()))
            }
        };
        benchmark.insert(samples.0.to_string(), samples.1);
    }
    Ok(view)
}

/// `benchmark_flags` without the timeouts: a timeout bounds the wait for a
/// completion, not what it is, so changing one must not restart an eval.
///
/// Flags with no knob left hash as absent, so adding or dropping a timeout
/// resumes too. Every field is named so that a new knob must pick a side to build.
fn resume_relevant_benchmark_flags(flags: &Option<BenchmarkFlags>) -> Option<BenchmarkFlagRef> {
    let mut flags = BenchmarkFlagRef::from(flags.clone()?);
    flags.http_timeout_seconds = None;
    flags.bench_timeout_seconds = None;
    let BenchmarkFlagRef {
        runtime_type: _,
        model_type: _,
        benchmark_type: _,
        http_timeout_seconds: _,
        bench_timeout_seconds: _,
        _doomloop_obsolete: _,
        readiness,
    } = &flags;
    readiness.is_some().then_some(flags)
}

/// Live append session for one eval run. Dropping it keeps the file for the
/// next open of the same digest; only [`EvalCompletionsStore::finalize`]
/// retires it.
///
/// No internal synchronisation: the eval entry shares it in a mutex held only
/// around [`Self::append`] and the progress read, never across an HTTP call or
/// a server restart. That mutex is the pool's LEAF lock: nothing else may be
/// acquired while holding it.
pub struct EvalCompletionSession {
    path: PathBuf,
    file: File,
    completions: Vec<BenchmarkEvalCompletion>,
    done_ids: HashSet<String>,
    failed_count: usize,
    /// Failed rows per id in the file, its attempts spent. The file is
    /// append-only, so every attempt left a line though `completions` keeps
    /// only the last.
    failed_attempts: HashMap<String, u32>,
    /// The ids this run recorded as not fitting its context
    /// ([`Self::record_unfit`]): failed in memory, absent from the file.
    unfit: HashSet<String>,
}

impl EvalCompletionSession {
    fn open_in(root: &Path, header: &Header) -> anyhow::Result<Self> {
        fs::create_dir_all(root)
            .with_context(|| format!("failed to create eval completions dir {}", root.display()))?;
        let path = session_path(root, &header.digest);

        let loaded = if path.exists() {
            match load_existing(&path, &header.digest) {
                Ok(loaded) => loaded,
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "eval completions unreadable; rotating stale"
                    );
                    rotate_stale(&path);
                    None
                }
            }
        } else {
            None
        };
        let (completions, failed_attempts) = match loaded {
            Some((found, completions, attempts))
                if found.http_timeout_seconds == header.http_timeout_seconds =>
            {
                (completions, attempts)
            }
            // Rewritten under the new timeout without the failed rows, so each
            // starts a fresh attempt budget under it.
            Some((_, completions, _)) => {
                let (failed, completed): (Vec<_>, Vec<_>) =
                    completions.into_iter().partition(|c| c.failed);
                tracing::info!(
                    retried = failed.len(),
                    path = %path.display(),
                    "eval http timeout changed; running the failed samples again"
                );
                write_checkpoint(&path, header, &completed)?;
                (completed, HashMap::new())
            }
            None => {
                write_checkpoint(&path, header, &[])?;
                (Vec::new(), HashMap::new())
            }
        };
        let done_ids: HashSet<String> = completions.iter().map(|c| c.id.clone()).collect();

        let file = OpenOptions::new()
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open {} for append", path.display()))?;

        if !done_ids.is_empty() {
            tracing::info!(
                completed = done_ids.len(),
                path = %path.display(),
                "resuming eval completions"
            );
        }

        let failed_count = completions.iter().filter(|c| c.failed).count();
        Ok(Self {
            path,
            file,
            completions,
            done_ids,
            failed_count,
            failed_attempts,
            unfit: HashSet::new(),
        })
    }

    /// Persist one completion (append-only on disk; last-write-wins in memory).
    pub fn append(&mut self, completion: BenchmarkEvalCompletion) -> anyhow::Result<()> {
        let mut line =
            serde_json::to_string(&completion).context("failed to serialize eval completion")?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .context("failed to write eval completion line")?;
        self.file
            .flush()
            .context("failed to flush eval completion line")?;

        // The entry appends each id once, and `load_existing` merges a file's
        // repeated ids, so this only keeps the counts right for a caller that
        // appends an id twice.
        if let Some(existing_idx) = self.completions.iter().position(|c| c.id == completion.id) {
            let old = self.completions.remove(existing_idx);
            if old.failed {
                self.failed_count = self.failed_count.saturating_sub(1);
            } else {
                tracing::warn!(
                    sample_id = %completion.id,
                    "duplicate eval completion id (last write wins)"
                );
            }
        }
        self.done_ids.insert(completion.id.clone());
        if completion.failed {
            self.failed_count += 1;
            *self
                .failed_attempts
                .entry(completion.id.clone())
                .or_insert(0) += 1;
        }
        self.completions.push(completion);
        Ok(())
    }

    /// Hold a failed row for a sample that does not fit this run's context
    /// in memory only: it counts in this run's result and its failures, but
    /// is not written, and spends no attempt. Its earlier rows in the file, if
    /// any, are what the next open reads, so the next run checks the sample
    /// again whatever context it holds.
    pub fn record_unfit(&mut self, completion: BenchmarkEvalCompletion) {
        if let Some(existing_idx) = self.completions.iter().position(|c| c.id == completion.id)
            && self.completions.remove(existing_idx).failed
        {
            self.failed_count = self.failed_count.saturating_sub(1);
        }
        self.done_ids.insert(completion.id.clone());
        if completion.failed {
            self.failed_count += 1;
        }
        self.unfit.insert(completion.id.clone());
        self.completions.push(completion);
    }

    /// The ids [`Self::record_unfit`] held this run.
    pub fn unfit_ids(&self) -> impl Iterator<Item = &str> {
        self.completions
            .iter()
            .filter(|c| c.failed && self.unfit.contains(&c.id))
            .map(|c| c.id.as_str())
    }

    pub fn contains(&self, sample_id: &str) -> bool {
        self.done_ids.contains(sample_id)
    }

    /// True for an id whose latest row failed and that has spent fewer than
    /// `max_attempts`. Bounded because some failures belong to the sample (an
    /// undecodable image, a prompt that overflows the context) and would
    /// re-fail on every resume.
    pub fn should_reattempt(&self, sample_id: &str, max_attempts: u32) -> bool {
        let currently_failed = self
            .completions
            .iter()
            .any(|c| c.id == sample_id && c.failed);
        currently_failed && self.failed_attempts.get(sample_id).copied().unwrap_or(0) < max_attempts
    }

    /// Failed ids a resume would retry, and those out of attempts, so an
    /// operator can tell which rows a re-run recovers.
    pub fn failure_split(&self, max_attempts: u32) -> (Vec<&str>, Vec<&str>) {
        let mut retriable = Vec::new();
        let mut exhausted = Vec::new();
        for c in self
            .completions
            .iter()
            .filter(|c| c.failed && !self.unfit.contains(&c.id))
        {
            if self.failed_attempts.get(&c.id).copied().unwrap_or(0) < max_attempts {
                retriable.push(c.id.as_str());
            } else {
                exhausted.push(c.id.as_str());
            }
        }
        (retriable, exhausted)
    }

    pub fn done_ids(&self) -> impl Iterator<Item = &str> {
        self.done_ids.iter().map(String::as_str)
    }

    pub fn failed_ids(&self) -> impl Iterator<Item = &str> {
        self.completions
            .iter()
            .filter(|c| c.failed)
            .map(|c| c.id.as_str())
    }

    pub fn completions(&self) -> &[BenchmarkEvalCompletion] {
        &self.completions
    }

    pub fn failed_count(&self) -> usize {
        self.failed_count
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every row, completed and failed alike. The file stays until finalized.
    pub fn into_completions(self) -> Vec<BenchmarkEvalCompletion> {
        self.completions
    }
}

/// Env var that lets an incomplete cell finish past [`ensure_complete`], so its
/// rows can be inspected. It publishes nothing: the management service refuses
/// a vlm_eval result without an answer to every sample.
/// [`ensure_every_answer`] (asr_eval) never honors it.
pub const ALLOW_INCOMPLETE_ENV: &str = "PIPETTE_ALLOW_INCOMPLETE_EVAL";

/// Whether [`ALLOW_INCOMPLETE_ENV`] may waive the failed-row refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waiver {
    /// The waiver is honored: the cell finishes, for inspection only.
    Honored,
    /// The waiver is ignored and never named: the score server scores the
    /// family only with an answer to every sample, so a waived submission
    /// would be refused there after the client had finalized its checkpoint,
    /// losing the answered rows.
    Refused,
}

/// Refuse to publish a cell with fewer rows than the dataset served, or with
/// any failed row ([`crate::measurement::eval_complete`]).
pub fn ensure_complete(
    benchmark_id: &str,
    checkpoint: &EvalCompletionSession,
    served_samples: usize,
) -> anyhow::Result<()> {
    check_complete(benchmark_id, checkpoint, served_samples, Waiver::Honored)
}

/// [`ensure_complete`] for a family scored only with an answer to every
/// sample (asr_eval): [`ALLOW_INCOMPLETE_ENV`] is ignored, so a failed row
/// always refuses the cell and the checkpoint stays for the next run.
pub fn ensure_every_answer(
    benchmark_id: &str,
    checkpoint: &EvalCompletionSession,
    served_samples: usize,
) -> anyhow::Result<()> {
    check_complete(benchmark_id, checkpoint, served_samples, Waiver::Refused)
}

fn check_complete(
    benchmark_id: &str,
    checkpoint: &EvalCompletionSession,
    served_samples: usize,
    waiver: Waiver,
) -> anyhow::Result<()> {
    // No failures does not mean complete: a sample skipped before the loop
    // leaves no failed row. `served_samples` counts distinct ids; vlm_eval's
    // `validate` checks its rows against `parameter_sample_count` and refuses a
    // duplicate id, so declared == served == submitted.
    let submitted = checkpoint.completions().len();
    anyhow::ensure!(
        submitted == served_samples,
        "eval: benchmark={benchmark_id} produced {submitted} completions for {served_samples} \
         served samples. Every sample must yield exactly one row; a shortfall means rows were \
         skipped or two dataset rows collapsed onto one id (duplicate sample ids)."
    );

    let failed = checkpoint.failed_count();
    if failed == 0 {
        return Ok(());
    }
    let total = submitted;
    let (retriable, exhausted) = checkpoint.failure_split(MAX_SAMPLE_ATTEMPTS);

    // Affirmative values only, so "false", "no" or "off" waives nothing.
    let waived = waiver == Waiver::Honored
        && std::env::var(ALLOW_INCOMPLETE_ENV)
            .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"));
    if waived {
        tracing::warn!(
            benchmark_id,
            failed,
            total,
            ?retriable,
            attempts_exhausted = ?exhausted,
            "finishing an INCOMPLETE cell because {ALLOW_INCOMPLETE_ENV} is set, for inspection \
             only: the management service refuses a result without an answer to every sample"
        );
        return Ok(());
    }

    let unfit: Vec<&str> = checkpoint.unfit_ids().collect();
    let unfit_note = if unfit.is_empty() {
        String::new()
    } else {
        format!(
            "unfit={unfit:?}: these did not fit the context and cost no attempt (each \
             failed_reason names the ctx_size that holds it); raise ctx_size or lower parallel \
             and run the cell again, and the checkpoint resumes. "
        )
    };
    let recovery = if retriable.is_empty() && exhausted.is_empty() {
        String::new()
    } else if retriable.is_empty() {
        let input = match waiver {
            Waiver::Honored => "image",
            Waiver::Refused => "audio",
        };
        format!(
            "every failure has used up its {MAX_SAMPLE_ATTEMPTS} attempts, so re-running as-is \
             will not change this -- inspect these samples (decodable {input}? ctx_size large \
             enough? model healthy?). NOTE the attempt history lives in the checkpoint, and \
             the resume digest deliberately ignores throughput knobs: raising ctx_size or \
             parallel alone will NOT give these rows fresh attempts. Discard the checkpoint (or use a \
             clean --work-dir) once the underlying cause is fixed."
        )
    } else {
        format!(
            "run this cell again: the checkpoint resumes and retries just these rows \
             (up to {MAX_SAMPLE_ATTEMPTS} attempts each)"
        )
    };
    anyhow::bail!(
        "eval: benchmark={benchmark_id} is incomplete -- {failed} of {total} samples failed, \
         and a result without an answer to every sample is not one: the management service \
         refuses it. retriable={retriable:?} attempts_exhausted={exhausted:?}. \
         {unfit_note}{recovery}"
    )
}

fn session_path(root: &Path, digest: &EvalRunDigest) -> PathBuf {
    root.join(format!("{}.jsonl", digest_prefix(digest)))
}

fn digest_prefix(digest: &EvalRunDigest) -> String {
    digest.as_ref().chars().take(16).collect()
}

fn digest_value(value: &Value) -> anyhow::Result<EvalRunDigest> {
    let canonical = canonical_json(value)?;
    let hash = Sha256::digest(canonical.as_bytes());
    Ok(EvalRunDigest::new(
        hash.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    ))
}

fn canonical_json(value: &Value) -> anyhow::Result<String> {
    let mut out = String::new();
    write_canonical(value, &mut out)?;
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut String) -> anyhow::Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push_str(
                &serde_json::to_string(s).context("failed to serialize JSON string value")?,
            );
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).context("failed to serialize JSON key")?);
                out.push(':');
                write_canonical(&map[k], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Replace the file with `header` and `rows` through a sibling, so a kill
/// mid-write cannot leave half a checkpoint.
fn write_checkpoint(
    path: &Path,
    header: &Header,
    rows: &[BenchmarkEvalCompletion],
) -> anyhow::Result<()> {
    let body = std::iter::once(serde_json::to_string(header))
        .chain(rows.iter().map(serde_json::to_string))
        .map(|line| line.map(|line| line + "\n"))
        .collect::<Result<String, _>>()
        .context("failed to serialize eval checkpoint")?;
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, body).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("failed to replace {}", path.display()))
}

/// Attempts a failed sample gets before its failure is taken as its own. The
/// failures worth retrying are transient (a recycled keep-alive connection, a
/// slot that died with the server, a stall), and a third attempt costs one
/// sample while a permanent loop costs the whole cell.
pub const MAX_SAMPLE_ATTEMPTS: u32 = 3;

/// A checkpoint file's header, the last row per id, and how many failed rows
/// each id has (its attempts spent).
type Loaded = (Header, Vec<BenchmarkEvalCompletion>, HashMap<String, u32>);

/// `None` when the header names another digest, which rotates the file away.
fn load_existing(path: &Path, expected: &EvalRunDigest) -> anyhow::Result<Option<Loaded>> {
    let mut body = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    // A kill mid-append leaves a torn last line; cut it so the next append
    // starts a line of its own instead of gluing its row onto the torn one.
    let complete = body
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |at| at + 1);
    if complete < body.len() {
        body.truncate(complete);
        OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| file.set_len(body.len() as u64))
            .with_context(|| format!("failed to cut the torn tail of {}", path.display()))?;
    }
    let mut lines = body.as_slice().lines();

    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(e)) => return Err(e).context("failed to read header"),
        None => anyhow::bail!("empty completion file"),
    };
    let header: Header =
        serde_json::from_str(&header_line).context("failed to parse completion header")?;
    if &header.digest != expected {
        tracing::info!(
            path = %path.display(),
            digest = %header.digest,
            expected_digest = %expected,
            "eval completions digest mismatch; rotating"
        );
        rotate_stale(path);
        return Ok(None);
    }

    let mut failed_attempts: HashMap<String, u32> = HashMap::new();
    let mut by_id: HashMap<String, BenchmarkEvalCompletion> = HashMap::new();
    for (idx, line) in lines.enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 2))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<BenchmarkEvalCompletion>(&line) {
            Ok(c) => {
                let c = sanitize_pre_gate_success(c);
                if c.failed {
                    *failed_attempts.entry(c.id.clone()).or_insert(0) += 1;
                }
                by_id.insert(c.id.clone(), c);
            }
            Err(e) => {
                tracing::warn!(
                    line = idx + 2,
                    path = %path.display(),
                    error = %e,
                    "skipping corrupt completion line"
                );
            }
        }
    }
    Ok(Some((
        header,
        by_id.into_values().collect(),
        failed_attempts,
    )))
}

/// The stop detail vlm_eval's chat reader attached to a stream that ended
/// with no `finish_reason`, back when such a stream was recorded as a success.
const PRE_GATE_CHAT_DETAIL: &str = "stream ended without finish_reason";

/// Recover a row a pre-gate vlm_eval binary trusted as a success: it recorded
/// `failed: false`, with whatever partial text had arrived, for a stream that
/// closed without a `finish_reason`. vlm_eval's `read_chat_stream` now fails
/// such a read, but an old checkpoint's digest is unchanged, so a resume would
/// skip the row and submit the stale answer. The row becomes a failure a
/// resume retries. Old text eval clients wrote another detail, so a text
/// checkpoint loads as it always did.
fn sanitize_pre_gate_success(mut c: BenchmarkEvalCompletion) -> BenchmarkEvalCompletion {
    let is_stale_malformed_success = !c.failed
        && c.stop_detail
            .as_deref()
            .is_some_and(|d| d == PRE_GATE_CHAT_DETAIL);
    if is_stale_malformed_success {
        tracing::warn!(
            sample_id = %c.id,
            detail = ?c.stop_detail,
            "eval completion recorded as a success by a pre-fix binary never reached a terminal \
             stop event; treating it as failed so a resume retries it instead of publishing it"
        );
        c.failed = true;
        c.failed_reason = Some(format!(
            "recovered on load: pre-fix checkpoint recorded this as a success with no terminal \
             stop event ({})",
            c.stop_detail.as_deref().unwrap_or("no detail")
        ));
    }
    c
}

fn rotate_stale(path: &Path) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dest = path.with_extension(format!("jsonl.stale-{ts}"));
    if let Err(e) = fs::rename(path, &dest) {
        tracing::warn!(
            from = %path.display(),
            to = %dest.display(),
            error = %e,
            "failed to rotate stale completions"
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use pipette_plan_types::{
        AbsolutePath, BenchmarkDefinition, BenchmarkId, BenchmarkType, EvalBenchmark, EvalId,
        EvalSample, GgufText, GgufTextSource, GgufVision, GgufVisionHfFiles, GgufVisionSource,
        HfOrg, HfRepo, HfRepoName, LlamaCppFlavor, LlamacppCliStockToolsRelease,
        LlamacppCliStockToolsSource, Model, ModelType, NonEmptyString, PositiveCount,
        PositiveSeconds, ReadinessOverrides, RepoSubpath, RepositoryUrl, Runtime, RuntimeType,
        VllmDtype, VllmRepetitionDetection,
    };

    use super::*;

    /// The `TempDir` is returned because it owns the store's directory.
    fn store_and_req() -> anyhow::Result<(tempfile::TempDir, EvalCompletionsStore, RunRequest)> {
        let root = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(root.path());
        let req = req()?;
        Ok((root, store, req))
    }

    fn runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(
            pipette_plan_types::LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                    repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                    repository_version: NonEmptyString::try_new("b1234".to_owned())?,
                    asset: None,
                }),
                flavor: LlamaCppFlavor::MacosArm64,
            },
        ))
    }

    fn model() -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_owned())?,
                    repo_name: HfRepoName::try_new("model".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("model.gguf".to_owned())?,
                sha256: None,
            },
        }))
    }

    fn benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("bench-a".to_owned())?,
            parameter_eval_id: EvalId::from("math_500"),
            parameter_dataset_name: "math_500".into(),
            parameter_max_tokens: PositiveCount::try_new(16)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: Some(vec![json!({"id": "s1", "prompt": "hi"})]),
        }))
    }

    fn req() -> anyhow::Result<RunRequest> {
        let rt = runtime()?;
        let m = model()?;
        Ok(RunRequest {
            runtime: pipette_plan_types::DeclaredBound::already_bound(rt),
            model: pipette_plan_types::DeclaredBound::already_bound(m),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: benchmark()?,
        })
    }

    fn eval_flags(
        parallel: Option<i32>,
        ctx_size: Option<u32>,
        no_cache: Option<bool>,
    ) -> anyhow::Result<RuntimeFlags> {
        Ok(RuntimeFlags::EvalLlamacppCliStockToolsGgufText(
            pipette_plan_types::EvalLlamacppCliStockToolsGgufText {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size,
                no_cache,
                parallel: parallel.map(PositiveCount::try_new).transpose()?,
                envs: Vec::new(),
                raw: Vec::new(),
            },
        ))
    }

    fn eval_timeout(seconds: Option<u64>) -> anyhow::Result<BenchmarkFlags> {
        Ok(BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
            http_timeout_seconds: seconds.map(PositiveSeconds::try_new).transpose()?,
        })
    }

    fn digest_with(
        edit: impl FnOnce(&mut RunRequest) -> anyhow::Result<()>,
    ) -> anyhow::Result<String> {
        let mut r = req()?;
        edit(&mut r)?;
        Ok(run_request_digest(&r)?.into_inner())
    }

    /// [`req`] at `parallel = 64` with no benchmark flags: the digest its
    /// checkpoints on disk were written under. Update it only with `req`.
    const EVAL_CELL_DIGEST: &str =
        "0c3d5d58b63d27a322edcad53332933a5a94f535b7b49a734214d735a075fdfe";

    /// Retuning throughput must not orphan a checkpoint.
    #[rstest]
    #[case::parallel_64(Some(64), None)]
    #[case::parallel_32(Some(32), None)]
    #[case::parallel_1(Some(1), None)]
    #[case::parallel_unset(None, None)]
    #[case::ctx_size(Some(64), Some(589_824))]
    #[case::parallel_and_ctx_size(Some(32), Some(294_912))]
    fn perf_only_flags_do_not_change_the_resume_identity(
        #[case] parallel: Option<i32>,
        #[case] ctx_size: Option<u32>,
    ) -> anyhow::Result<()> {
        let digest = digest_with(|r| {
            r.runtime_flags = Some(eval_flags(parallel, ctx_size, None)?);
            Ok(())
        })?;
        anyhow::ensure!(
            digest == EVAL_CELL_DIGEST,
            "should resume the same checkpoint, got {digest}"
        );
        Ok(())
    }

    /// A vLLM eval cell's flags at `parallel`, with liquid_evals' repetition
    /// stop if `repetition_detection`.
    fn vllm_eval_flags(parallel: i32, repetition_detection: bool) -> anyhow::Result<RuntimeFlags> {
        let mut flags =
            RuntimeFlagRef::new(BenchmarkType::Eval, RuntimeType::UvVllm, ModelType::Torch);
        flags.parallel = Some(PositiveCount::try_new(parallel)?);
        flags.repetition_detection = repetition_detection
            .then(|| -> anyhow::Result<_> {
                Ok(VllmRepetitionDetection {
                    min_pattern_size: PositiveCount::try_new(2)?,
                    max_pattern_size: PositiveCount::try_new(128)?,
                    min_count: PositiveCount::try_new(10)?,
                })
            })
            .transpose()?;
        Ok(RuntimeFlags::try_from(flags)?)
    }

    /// [`req`] under [`vllm_eval_flags`] with the stop unset: the digest vLLM
    /// eval checkpoints were written under before the knob existed.
    const VLLM_EVAL_CELL_DIGEST: &str =
        "88fbc902c10c6aab62670028950f4ccd93b8ea0d66d80dc397e6be893b3f5588";

    /// `parallel` is still stripped beside the stop, and an unset stop resumes
    /// the old checkpoint.
    #[rstest]
    #[case::unset(64, false, true)]
    #[case::unset_retuned(8, false, true)]
    #[case::set(64, true, false)]
    #[case::set_retuned(8, true, false)]
    fn repetition_detection_is_part_of_the_resume_identity(
        #[case] parallel: i32,
        #[case] repetition_detection: bool,
        #[case] resumes_the_unset_checkpoint: bool,
    ) -> anyhow::Result<()> {
        let digest = digest_with(|r| {
            r.runtime_flags = Some(vllm_eval_flags(parallel, repetition_detection)?);
            Ok(())
        })?;
        let set = digest_with(|r| {
            r.runtime_flags = Some(vllm_eval_flags(64, true)?);
            Ok(())
        })?;
        assert_eq!(
            digest == VLLM_EVAL_CELL_DIGEST,
            resumes_the_unset_checkpoint,
            "{digest}"
        );
        assert_eq!(digest == set, !resumes_the_unset_checkpoint, "{digest}");
        Ok(())
    }

    #[rstest]
    #[case::unset(None)]
    #[case::seconds_600(Some(600))]
    #[case::seconds_3600(Some(3_600))]
    fn a_timeout_does_not_change_the_resume_identity(
        #[case] seconds: Option<u64>,
    ) -> anyhow::Result<()> {
        let digest = digest_with(|r| {
            r.runtime_flags = Some(eval_flags(Some(64), None, None)?);
            r.benchmark_flags = Some(eval_timeout(seconds)?);
            Ok(())
        })?;
        anyhow::ensure!(
            digest == EVAL_CELL_DIGEST,
            "should resume the same checkpoint, got {digest}"
        );
        Ok(())
    }

    fn vault_cell_digest(
        model_label: Option<&str>,
        runtime_label: Option<&str>,
    ) -> anyhow::Result<String> {
        digest_with(|r| {
            r.model.declared = serde_json::from_value(json!({
                "type": "gguf_text",
                "source": "vault_url",
                "url": "/vault/resources/vault_1/content",
                "principal_id": "org_1",
                "custom_label": model_label,
            }))?;
            r.runtime.declared = serde_json::from_value(json!({
                "type": "llamacpp_cli_stock_tools",
                "source": "vault_remote_archive",
                "url": "/vault/resources/vault_2/content",
                "principal_id": "org_1",
                "custom_label": runtime_label,
                "flavor": "macos-arm64",
            }))?;
            Ok(())
        })
    }

    /// A torch-oai eval cell of a `torch` model on `runtime`, under the vLLM
    /// eval flags at `parallel = 64` when `flags`.
    fn torch_oai_cell_digest(runtime: Value, flags: bool) -> anyhow::Result<String> {
        digest_with(|r| {
            r.runtime.declared = serde_json::from_value(runtime)?;
            r.model.declared = serde_json::from_value(json!({
                "type": "torch",
                "source": "huggingface",
                "org": "google",
                "repo_name": "gemma-4-26B-A4B-it",
            }))?;
            r.runtime_flags = flags.then(|| vllm_eval_flags(64, false)).transpose()?;
            Ok(())
        })
    }

    fn uv_runtime(kind: &str) -> Value {
        json!({
            "type": kind,
            "server_version": "0.31.0",
            "build": "rocm723",
            "python_version": "3.12",
            "source": {"type": "pip_requirements_text", "contents": "vllm==0.31.0\n"},
        })
    }

    fn docker_runtime(kind: &str, image: &str) -> Value {
        json!({"type": kind, "image_name": image, "image_tag": "v1", "flavor": "nvidia_gpu"})
    }

    /// The pinned digests are these cells' digests before vLLM rows carried
    /// the reasoning.
    #[rstest]
    #[case::uv_vllm(
        uv_runtime("uv_vllm"),
        true,
        "2d7871c40d70192fb9eae6fe048404df123bb38412dfa1ad401252907ccd4a1a",
        false
    )]
    #[case::docker_vllm(
        docker_runtime("docker_vllm", "vllm/vllm-openai"),
        false,
        "98cf44e0e8d86326d5f7329942fe68687b00e9f3f4f5d7ed29747f860b35082d",
        false
    )]
    #[case::docker_vllm_flags(
        docker_runtime("docker_vllm", "vllm/vllm-openai"),
        true,
        "620967dedb03782993136ccad9d55dae57fe612605bcec70a91871f287c68e0c",
        false
    )]
    #[case::uv_sglang(
        uv_runtime("uv_sglang"),
        false,
        "67326ac74bbb7197bb7e638b6194a5b21b9fc1fd9134d26f72e13120b65c9bc9",
        true
    )]
    #[case::docker_sglang(
        docker_runtime("docker_sglang", "lmsysorg/sglang"),
        false,
        "783b4f131480b2118819981040231ac68c2850c77380668c92d9f8826f36fa3a",
        true
    )]
    fn only_a_vllm_cell_leaves_its_old_checkpoint(
        #[case] runtime: Value,
        #[case] flags: bool,
        #[case] main_digest: &str,
        #[case] resumes: bool,
    ) -> anyhow::Result<()> {
        let digest = torch_oai_cell_digest(runtime, flags)?;
        assert_eq!(digest == main_digest, resumes, "{digest}");
        Ok(())
    }

    #[rstest]
    #[case::on_the_model(Some("acme LFM3 q4"), None)]
    #[case::on_the_runtime(None, Some("acme llama.cpp build"))]
    fn a_custom_label_does_not_change_the_resume_identity(
        #[case] model_label: Option<&str>,
        #[case] runtime_label: Option<&str>,
    ) -> anyhow::Result<()> {
        assert_eq!(
            vault_cell_digest(model_label, runtime_label)?,
            vault_cell_digest(None, None)?
        );
        Ok(())
    }

    #[test]
    fn a_semantic_change_still_changes_the_resume_identity() -> anyhow::Result<()> {
        let no_cache = digest_with(|r| {
            r.runtime_flags = Some(eval_flags(Some(64), None, Some(true))?);
            Ok(())
        })?;
        anyhow::ensure!(
            no_cache != EVAL_CELL_DIGEST,
            "a semantic knob must still change the digest"
        );
        let longer = digest_with(|r| {
            r.runtime_flags = Some(eval_flags(Some(64), None, None)?);
            if let BenchmarkDefinition::Eval(eval) = &mut r.benchmark {
                eval.parameter_max_tokens = PositiveCount::try_new(32)?;
            }
            Ok(())
        })?;
        anyhow::ensure!(
            longer != EVAL_CELL_DIGEST,
            "a changed benchmark definition must still change the digest"
        );
        Ok(())
    }

    /// Absent flags hash as `null`, flags with only perf knobs as an object.
    #[test]
    fn no_flags_at_all_is_still_its_own_identity() -> anyhow::Result<()> {
        anyhow::ensure!(
            run_request_digest(&req()?)?.as_ref() != EVAL_CELL_DIGEST,
            "flags present vs absent should not collapse",
        );
        Ok(())
    }

    /// No eval variant has a knob besides the timeouts, so a corpus cell's
    /// stands in.
    #[test]
    fn other_benchmark_flags_stay_in_the_resume_identity() -> anyhow::Result<()> {
        let gated = |timeout: u64| {
            digest_with(|r| {
                r.benchmark_flags = Some(
                    BenchmarkFlags::VlSpecDecodeLlamacppCliStockToolsGgufVision {
                        http_timeout_seconds: Some(PositiveSeconds::try_new(timeout)?),
                        bench_timeout_seconds: Some(PositiveSeconds::try_new(timeout)?),
                        readiness: Some(ReadinessOverrides {
                            max_wait_secs: Some(1_800),
                            skip_thermal: None,
                            skin_below_c: None,
                        }),
                    },
                );
                Ok(())
            })
        };
        let kept = gated(600)?;
        anyhow::ensure!(
            kept != run_request_digest(&req()?)?.into_inner(),
            "readiness must stay in the identity"
        );
        anyhow::ensure!(
            kept == gated(3_600)?,
            "the timeouts must still leave the identity"
        );
        Ok(())
    }

    fn c(id: &str) -> BenchmarkEvalCompletion {
        BenchmarkEvalCompletion {
            id: id.into(),
            completion: format!("ans-{id}"),
            ..Default::default()
        }
    }

    #[test]
    fn digest_stable_for_same_declared_request() -> anyhow::Result<()> {
        let a = run_request_digest(&req()?)?;
        let b = run_request_digest(&req()?)?;
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    fn digest_is_pinned_across_builds() -> anyhow::Result<()> {
        assert_eq!(
            run_request_digest(&req()?)?.as_ref(),
            "57a85293a329044d001992fdcff6386012df4399c5c4c7845fe74f76bb50373f",
            "if you changed `req()`, update this constant to match; if you did \
             not, resume just broke for every checkpoint already on disk"
        );
        Ok(())
    }

    #[test]
    fn a_resume_reattempts_a_failed_sample() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;

        let mut first = store.open(&req)?;
        first.append(failed("s-1", "connection closed"))?;
        first.append(succeeded("s-2", "B"))?;
        drop(first);

        let resumed = store.open(&req)?;
        assert!(
            resumed.contains("s-1"),
            "the failed row is still on disk and still counted"
        );
        assert!(
            resumed.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS),
            "a failed sample with attempts left must be re-run"
        );
        assert!(
            !resumed.should_reattempt("s-2", MAX_SAMPLE_ATTEMPTS),
            "a SUCCEEDED sample must never be re-run -- that would be wasted work \
             and, worse, would let a second answer overwrite the first"
        );
        Ok(())
    }

    #[test]
    fn reattempts_stop_at_the_limit() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;

        for attempt in 1..=MAX_SAMPLE_ATTEMPTS {
            let mut s = store.open(&req)?;
            assert!(
                s.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS) || attempt == 1,
                "attempt {attempt} should still be offered"
            );
            s.append(failed("s-1", "still broken"))?;
        }

        let exhausted = store.open(&req)?;
        assert!(
            !exhausted.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS),
            "after {MAX_SAMPLE_ATTEMPTS} failures the sample must be left alone"
        );
        let (retriable, spent) = exhausted.failure_split(MAX_SAMPLE_ATTEMPTS);
        assert!(retriable.is_empty());
        assert_eq!(spent, vec!["s-1"]);
        Ok(())
    }

    #[test]
    fn attempts_survive_across_resumes() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut a = store.open(&req)?;
        a.append(failed("s-1", "blip"))?;
        drop(a);
        let mut b = store.open(&req)?;
        b.append(failed("s-1", "blip"))?;
        drop(b);

        let c = store.open(&req)?;
        let (retriable, spent) = c.failure_split(2);
        assert!(
            retriable.is_empty() && spent == vec!["s-1"],
            "two failures written by two different sessions must count as two attempts"
        );
        Ok(())
    }

    #[test]
    fn a_successful_reattempt_clears_the_failure() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut a = store.open(&req)?;
        a.append(failed("s-1", "blip"))?;
        assert_eq!(a.failed_count(), 1);
        a.append(succeeded("s-1", "C"))?;
        assert_eq!(
            a.failed_count(),
            0,
            "the failure must not be double-counted"
        );
        drop(a);

        let resumed = store.open(&req)?;
        assert_eq!(resumed.failed_count(), 0);
        assert!(
            !resumed.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS),
            "a sample that eventually answered must not be re-run"
        );
        assert_eq!(resumed.completions().len(), 1, "one row per sample id");
        Ok(())
    }

    #[test]
    fn finalize_preserves_attempt_history() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;

        for _ in 0..2 {
            let mut s = store.open(&req)?;
            s.append(failed("s-1", "still broken"))?;
            drop(s);
            store.finalize(&run_request_digest(&req)?)?;
        }

        let resumed = store.open(&req)?;
        let (retriable, spent) = resumed.failure_split(2);
        assert!(
            retriable.is_empty() && spent == vec!["s-1"],
            "two attempts must survive the finalize rewrite; got retriable={retriable:?} \
             spent={spent:?} (a reset would report this as retriable)"
        );
        assert!(!resumed.should_reattempt("s-1", 2));
        Ok(())
    }

    #[test]
    fn a_resume_recovers_a_pre_gate_malformed_success() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        {
            let mut s = store.open(&req)?;
            let stale = BenchmarkEvalCompletion {
                id: "s-1".into(),
                completion: String::new(),
                failed: false,
                failed_reason: None,
                stop_reason: pipette_plan_types::BenchmarkEvalCompletionStopReason::Unknown,
                stop_detail: Some(PRE_GATE_CHAT_DETAIL.to_string()),
                completion_tokens: None,
            };
            s.append(stale)?;
        }

        let resumed = store.open(&req)?;
        assert_eq!(
            resumed.failed_count(),
            1,
            "the stale success must be recovered as a failure on load"
        );
        assert!(
            resumed.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS),
            "and must be given a fresh attempt budget, not treated as already exhausted"
        );
        Ok(())
    }

    #[test]
    fn a_text_evals_old_success_loads_unchanged() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut s = store.open(&req)?;
        s.append(BenchmarkEvalCompletion {
            stop_detail: Some("stream ended without a terminal stop event".to_string()),
            stop_reason: pipette_plan_types::BenchmarkEvalCompletionStopReason::Unknown,
            ..succeeded("s-1", "A")
        })?;
        drop(s);

        let resumed = store.open(&req)?;
        assert_eq!(resumed.failed_count(), 0);
        assert!(!resumed.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS));
        Ok(())
    }

    #[test]
    fn a_genuine_success_is_not_recovered_as_failed() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut s = store.open(&req)?;
        // Details that share the sanitizer's words, so the reopen checks the
        // exact match rather than a missing detail.
        s.append(BenchmarkEvalCompletion {
            stop_detail: Some("stream ended with a stop event".to_string()),
            ..succeeded("s-1", "A")
        })?;
        s.append(BenchmarkEvalCompletion {
            stop_detail: Some(format!("{PRE_GATE_CHAT_DETAIL} (and more)")),
            ..succeeded("s-2", "B")
        })?;
        drop(s);

        // The sanitizer runs on load, so only a reopened store exercises it.
        let resumed = store.open(&req)?;
        assert_eq!(resumed.failed_count(), 0);
        assert!(!resumed.should_reattempt("s-1", MAX_SAMPLE_ATTEMPTS));
        assert!(!resumed.should_reattempt("s-2", MAX_SAMPLE_ATTEMPTS));
        Ok(())
    }

    fn failed(id: &str, reason: &str) -> BenchmarkEvalCompletion {
        BenchmarkEvalCompletion {
            id: id.to_string(),
            completion: String::new(),
            failed: true,
            failed_reason: Some(reason.to_string()),
            stop_reason: pipette_plan_types::BenchmarkEvalCompletionStopReason::Failure,
            stop_detail: Some(reason.to_string()),
            completion_tokens: None,
        }
    }

    fn succeeded(id: &str, text: &str) -> BenchmarkEvalCompletion {
        BenchmarkEvalCompletion {
            id: id.to_string(),
            completion: text.to_string(),
            failed: false,
            failed_reason: None,
            stop_reason: pipette_plan_types::BenchmarkEvalCompletionStopReason::Eos,
            stop_detail: None,
            completion_tokens: Some(1),
        }
    }

    /// The vlm digest resolves the runtime flags by the (benchmark, runtime,
    /// model) triple, so a vlm_eval fixture carries the axes of a real cell.
    fn vision_model() -> anyhow::Result<Model> {
        Ok(Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                repo: HfRepo {
                    org: HfOrg::try_new("org".to_owned())?,
                    repo_name: HfRepoName::try_new("vision".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                model: RepoSubpath::try_new("model.gguf".to_owned())?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf".to_owned())?,
                mmproj_sha256: None,
            }),
        }))
    }

    fn vlm_req(benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        let mut r = req()?;
        r.model = pipette_plan_types::DeclaredBound::already_bound(vision_model()?);
        r.benchmark = benchmark;
        Ok(r)
    }

    /// A vLLM vlm_eval cell retuned after an OOM (fewer samples in flight, a
    /// smaller context) or stating the prefix cache's fixed `false` resumes
    /// its checkpoint; a knob that can change answers forks it.
    #[rstest]
    #[case::same(Some(64), None, None, None, true)]
    #[case::parallel(Some(8), None, None, None, true)]
    #[case::context(Some(64), Some(40960), None, None, true)]
    #[case::prefix_cache_stated(Some(64), None, Some(false), None, true)]
    #[case::dtype(Some(64), None, None, Some(VllmDtype::Float16), false)]
    fn a_vllm_vlm_eval_retuned_resumes_its_checkpoint(
        #[case] parallel: Option<i32>,
        #[case] max_model_len: Option<i32>,
        #[case] prefix_caching: Option<bool>,
        #[case] dtype: Option<VllmDtype>,
        #[case] same: bool,
    ) -> anyhow::Result<()> {
        let digest = |parallel: Option<i32>,
                      max_model_len: Option<i32>,
                      prefix_caching: Option<bool>,
                      dtype: Option<VllmDtype>|
         -> anyhow::Result<String> {
            let mut r = vlm_req(BenchmarkDefinition::VlmEval(vlm_benchmark()?))?;
            r.model = pipette_plan_types::DeclaredBound::already_bound(serde_json::from_value(
                serde_json::json!({"type": "torch", "source": "huggingface", "org": "o", "repo_name": "r"}),
            )?);
            r.runtime = pipette_plan_types::DeclaredBound::already_bound(serde_json::from_value(
                serde_json::json!({
                    "type": "uv_vllm", "server_version": "0.31.0", "build": "rocm723",
                    "python_version": "3.12",
                    "source": {"type": "absolute_preinstalled", "dir": "/venv"},
                }),
            )?);
            let mut flags = RuntimeFlagRef::new(
                BenchmarkType::VlmEval,
                RuntimeType::UvVllm,
                ModelType::Torch,
            );
            flags.parallel = parallel.map(PositiveCount::try_new).transpose()?;
            flags.max_model_len = max_model_len.map(PositiveCount::try_new).transpose()?;
            flags.prefix_caching = prefix_caching;
            flags.dtype = dtype;
            r.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
            Ok(run_request_digest(&r)?.into_inner())
        };
        let base = digest(Some(64), None, None, None)?;
        assert_eq!(
            digest(parallel, max_model_len, prefix_caching, dtype)? == base,
            same
        );
        Ok(())
    }

    fn no_messages(id: &str) -> EvalSample {
        EvalSample {
            id: id.to_owned(),
            messages: Vec::new(),
            chat_template_kwargs: None,
        }
    }

    fn vlm_benchmark() -> anyhow::Result<pipette_plan_types::VlmEvalBenchmark> {
        Ok(pipette_plan_types::VlmEvalBenchmark {
            benchmark_id: BenchmarkId::try_new("vlm_eval_mmstar_2026.07.1".to_owned())?,
            parameter_lmms_task: "mmstar".into(),
            parameter_dataset_version: "2026.07.1".into(),
            parameter_sample_count: PositiveCount::try_new(1)?,
            parameter_max_tokens: PositiveCount::try_new(1024)?,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_presence_penalty: None,
            parameter_repeat_penalty: None,
            parameter_seed: None,
            parameter_seed_rule: None,
            parameter_prompt_profile: None,
            parameter_image_tokens: None,
            dataset: None,
            recipe: None,
            samples: Some(vec![no_messages("s1")]),
        })
    }

    #[test]
    fn vlm_eval_digest_is_pinned_across_builds() -> anyhow::Result<()> {
        use pipette_plan_types::VlmEvalBenchmark;

        let r = vlm_req(BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
            parameter_sample_count: PositiveCount::try_new(2)?,
            parameter_mcq_choices: Some(vec!["A".into(), "B".into()]),
            samples: Some(serde_json::from_value(json!([
                {"id": "s1", "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]}]},
                {"id": "s2", "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}
                ]}]},
            ]))?),
            ..vlm_benchmark()?
        }))?;
        assert_eq!(
            run_request_digest(&r)?.as_ref(),
            "2406597133a36b251de6d02ef72da3f4f1f4ee82c0b42168e68a4e8a355b7600",
            "if you changed this request, update this constant to match; if you \
             did not, vlm_eval resume just broke for every checkpoint already \
             on disk"
        );
        Ok(())
    }

    /// An asr_eval request on the vision axes its cell runs on, with the given
    /// slot count, context, beams and HF token.
    fn asr_req(
        parallel: Option<i32>,
        ctx_size: Option<u32>,
        num_beams: Option<i32>,
        auth_token: Option<&str>,
    ) -> anyhow::Result<RunRequest> {
        asr_req_with_envs(parallel, ctx_size, num_beams, auth_token, &[])
    }

    /// [`asr_req`] with runtime `envs`.
    fn asr_req_with_envs(
        parallel: Option<i32>,
        ctx_size: Option<u32>,
        num_beams: Option<i32>,
        auth_token: Option<&str>,
        envs: &[&str],
    ) -> anyhow::Result<RunRequest> {
        let mut r = vlm_req(serde_json::from_value(serde_json::json!({
            "benchmark_id": "asr_eval_esb_librispeech_test_clean_minicpm_o",
            "benchmark_type": "asr_eval",
            "parameter_asr_task": "asr_esb_librispeech_test_clean",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 2620,
            "parameter_prompt_profile": "minicpm_o",
            "parameter_max_tokens": 128,
            "parameter_num_beams": num_beams,
        }))?)?;
        if let Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(files),
        }) = &mut r.model.declared
        {
            files.repo.auth_token = auth_token
                .map(|token| pipette_plan_types::AuthToken::try_new(token.to_owned()))
                .transpose()?;
        }
        r.runtime_flags = Some(RuntimeFlags::AsrEvalLlamacppCliStockToolsGgufVision(
            pipette_plan_types::AsrEvalLlamacppCliStockToolsGgufVision {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size,
                no_cache: None,
                parallel: parallel.map(PositiveCount::try_new).transpose()?,
                envs: envs
                    .iter()
                    .map(|env| pipette_plan_types::EnvForward::try_new((*env).to_owned()))
                    .collect::<Result<_, _>>()?,
                raw: Vec::new(),
            },
        ));
        Ok(r)
    }

    /// The asr_eval resume identity, pinned: its definition carries no
    /// samples, so it is hashed whole, flags resolved and the HF token
    /// dropped.
    #[test]
    fn asr_eval_digest_is_pinned_across_builds() -> anyhow::Result<()> {
        assert_eq!(
            run_request_digest(&asr_req(None, None, Some(3), None)?)?.as_ref(),
            "b4c49593ab82d23e8d6e39fa61dff2b5ab35b2e72326cd7b837d9656c0d2f792",
            "if you changed this request, update this constant to match; if you \
             did not, asr_eval resume just broke for every checkpoint already \
             on disk"
        );
        Ok(())
    }

    /// An entry on the per-sample seed rule is another run than the same entry
    /// on the repeat offset, so neither resumes into the other's answers; an
    /// entry that names no rule hashes as it always has (pinned above).
    #[test]
    fn a_seed_rule_is_part_of_the_resume_identity() -> anyhow::Result<()> {
        use pipette_plan_types::{SamplingSeed, SeedRule, VlmEvalBenchmark};

        let seeded = VlmEvalBenchmark {
            parameter_seed: Some(SamplingSeed::try_new(0)?),
            ..vlm_benchmark()?
        };
        let repeat = vlm_req(BenchmarkDefinition::VlmEval(seeded.clone()))?;
        let per_sample = vlm_req(BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
            parameter_seed_rule: Some(SeedRule::PerSampleSha256),
            ..seeded
        }))?;
        assert_ne!(
            run_request_digest(&repeat)?.as_ref(),
            run_request_digest(&per_sample)?.as_ref()
        );

        let asr = |rule: Option<SeedRule>| -> anyhow::Result<RunRequest> {
            let mut r = asr_req(None, None, None, None)?;
            if let BenchmarkDefinition::AsrEval(b) = &mut r.benchmark {
                b.parameter_seed = Some(SamplingSeed::try_new(0)?);
                b.parameter_seed_rule = rule;
            }
            Ok(r)
        };
        assert_ne!(
            run_request_digest(&asr(None)?)?.as_ref(),
            run_request_digest(&asr(Some(SeedRule::PerSampleSha256))?)?.as_ref()
        );
        Ok(())
    }

    /// Retuning the slots or the context, or rotating an HF token (typed, or
    /// forwarded as a runtime env), resumes the same checkpoint; the decoding,
    /// an env's name, and a `GGML_*` value (a backend knob) are the cell's
    /// identity.
    #[test]
    fn asr_eval_digest_keeps_only_what_decides_a_transcript() -> anyhow::Result<()> {
        let base = run_request_digest(&asr_req(None, None, Some(3), None)?)?;
        for (label, other) in [
            ("parallel", asr_req(Some(4), None, Some(3), None)?),
            ("ctx_size", asr_req(None, Some(65_536), Some(3), None)?),
            (
                "auth token",
                asr_req(None, None, Some(3), Some("hf_rotated"))?,
            ),
        ] {
            assert_eq!(
                run_request_digest(&other)?,
                base,
                "{label} forked the identity"
            );
        }
        assert_ne!(
            run_request_digest(&asr_req(None, None, None, None)?)?,
            base,
            "the beams decide the transcript"
        );
        let token = |value: &str| -> anyhow::Result<EvalRunDigest> {
            run_request_digest(&asr_req_with_envs(
                None,
                None,
                Some(3),
                None,
                &[&format!("HF_TOKEN={value}")],
            )?)
        };
        assert_eq!(
            token("hf_old")?,
            token("hf_rotated")?,
            "an env secret's value is not hashed"
        );
        assert_ne!(token("hf_old")?, base, "the env's name is");
        let ggml = |value: &str| -> anyhow::Result<EvalRunDigest> {
            run_request_digest(&asr_req_with_envs(
                None,
                None,
                Some(3),
                None,
                &[&format!("GGML_METAL_NDEBUG={value}")],
            )?)
        };
        assert_ne!(ggml("0")?, ggml("1")?, "a backend knob's value is hashed");
        Ok(())
    }

    /// A vLLM asr_eval cell whose context is raised after a cut reply, or that
    /// states the prefix cache's fixed `false`, resumes its checkpoint; a knob
    /// that can change answers forks it.
    #[rstest]
    #[case::same(None, None, None, true)]
    #[case::context(Some(8192), None, None, true)]
    #[case::prefix_cache_stated(None, Some(false), None, true)]
    #[case::dtype(None, None, Some(VllmDtype::Float16), false)]
    fn a_vllm_asr_eval_retuned_resumes_its_checkpoint(
        #[case] max_model_len: Option<i32>,
        #[case] prefix_caching: Option<bool>,
        #[case] dtype: Option<VllmDtype>,
        #[case] same: bool,
    ) -> anyhow::Result<()> {
        let digest = |max_model_len: Option<i32>,
                      prefix_caching: Option<bool>,
                      dtype: Option<VllmDtype>|
         -> anyhow::Result<String> {
            let mut r = asr_req(None, None, None, None)?;
            r.model = pipette_plan_types::DeclaredBound::already_bound(serde_json::from_value(
                serde_json::json!({"type": "torch", "source": "huggingface", "org": "o", "repo_name": "r"}),
            )?);
            r.runtime = pipette_plan_types::DeclaredBound::already_bound(serde_json::from_value(
                serde_json::json!({
                    "type": "uv_vllm", "server_version": "0.32.0", "build": "rocm723",
                    "python_version": "3.12",
                    "source": {"type": "absolute_preinstalled", "dir": "/venv"},
                }),
            )?);
            let mut flags = RuntimeFlagRef::new(
                BenchmarkType::AsrEval,
                RuntimeType::UvVllm,
                ModelType::Torch,
            );
            flags.max_model_len = max_model_len.map(PositiveCount::try_new).transpose()?;
            flags.prefix_caching = prefix_caching;
            flags.dtype = dtype;
            r.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
            Ok(run_request_digest(&r)?.into_inner())
        };
        let base = digest(Some(4096), None, None)?;
        assert_eq!(digest(max_model_len, prefix_caching, dtype)? == base, same);
        Ok(())
    }

    /// A catalog asr_eval's samples are its corpus: a corpus pinned at
    /// another digest is other clips, so its checkpoint must not resume; the
    /// rest of the pin (the `url` the server stamps on it, its `bytes`) and
    /// the recipe are not identity.
    #[test]
    fn asr_eval_digest_follows_the_corpus_pin() -> anyhow::Result<()> {
        let entry = |digest: &str, url: Option<&str>, bytes: u64| -> anyhow::Result<RunRequest> {
            let mut r = asr_req(None, None, Some(3), None)?;
            if let BenchmarkDefinition::AsrEval(asr) = &mut r.benchmark {
                asr.dataset = Some(serde_json::from_value(json!({
                    "format": "asr-eval-corpus/1",
                    "adapter": "evals-server@1",
                    "digest": digest,
                    "bytes": bytes,
                    "upstream": {
                        "source": "evals_server",
                        "eval_id": "asr_esb_librispeech_test_clean",
                        "dataset_version": "2026.10.1",
                        "prompt_profile": "minicpm_o",
                        "identity": "9".repeat(64),
                    },
                    "url": url,
                }))?);
            }
            Ok(r)
        };
        let pinned = run_request_digest(&entry(&"1".repeat(64), Some("/datasets/a"), 1)?)?;
        assert_eq!(
            pinned,
            run_request_digest(&entry(&"1".repeat(64), None, 2)?)?,
            "only the corpus digest is identity"
        );
        assert_ne!(
            pinned,
            run_request_digest(&entry(&"2".repeat(64), Some("/datasets/a"), 1)?)?,
            "another corpus is another run"
        );
        assert_ne!(
            pinned,
            run_request_digest(&asr_req(None, None, Some(3), None)?)?,
            "a pinned corpus is named in the identity"
        );
        let mut covered = entry(&"1".repeat(64), Some("/datasets/a"), 1)?;
        if let BenchmarkDefinition::AsrEval(asr) = &mut covered.benchmark {
            asr.recipe = Some(serde_json::from_value(json!({
                "family": "MiniCPM-o 4.5",
                "models": [{"repo": "openbmb/MiniCPM-o-4_5-gguf"}],
                "enable_thinking": false,
                "sources": ["https://a"],
            }))?);
        }
        assert_eq!(
            pinned,
            run_request_digest(&covered)?,
            "the recipe is not identity"
        );
        Ok(())
    }

    /// The rest of the pin (the `url` the server stamps on it, its `bytes`) is
    /// not identity.
    #[test]
    fn vlm_eval_digest_follows_the_corpus_pin() -> anyhow::Result<()> {
        use pipette_plan_types::VlmEvalBenchmark;

        let entry = |digest: &str, url: Option<&str>, bytes: u64| -> anyhow::Result<RunRequest> {
            vlm_req(BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
                benchmark_id: BenchmarkId::try_new("vlm_eval_mmstar_2026.10.1".to_owned())?,
                parameter_dataset_version: "2026.10.1".into(),
                parameter_sample_count: PositiveCount::try_new(1500)?,
                dataset: Some(serde_json::from_value(json!({
                    "format": "vlm-eval-corpus/1",
                    "adapter": "evals-server@1",
                    "digest": digest,
                    "bytes": bytes,
                    "upstream": {
                        "source": "evals_server",
                        "eval_id": "mmstar",
                        "dataset_version": "2026.10.1",
                        "identity": "9".repeat(64),
                    },
                    "url": url,
                }))?),
                samples: None,
                ..vlm_benchmark()?
            }))
        };
        let pinned = run_request_digest(&entry(&"1".repeat(64), Some("/datasets/a"), 1)?)?;
        assert_eq!(
            pinned,
            run_request_digest(&entry(&"1".repeat(64), None, 2)?)?,
            "only the corpus digest is identity"
        );
        assert_ne!(
            pinned,
            run_request_digest(&entry(&"2".repeat(64), Some("/datasets/a"), 1)?)?,
            "another corpus is another run"
        );
        // Whom the recipe covers decides who may run the entry, not what a
        // run produces: a model added to it must not restart a run hours in.
        let mut covered = entry(&"1".repeat(64), Some("/datasets/a"), 1)?;
        if let BenchmarkDefinition::VlmEval(vlm) = &mut covered.benchmark {
            vlm.recipe = Some(serde_json::from_value(json!({
                "family": "F", "models": [{"repo": "o/r"}], "sources": ["https://a"],
            }))?);
        }
        assert_eq!(
            pinned,
            run_request_digest(&covered)?,
            "the recipe is not identity"
        );
        Ok(())
    }

    #[test]
    fn vlm_eval_digest_ignores_the_model_auth_token() -> anyhow::Result<()> {
        use pipette_plan_types::AuthToken;

        let vlm_req = |auth_token: Option<AuthToken>| -> anyhow::Result<RunRequest> {
            let mut r = req()?;
            let model = Model::GgufVision(GgufVision {
                source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                    repo: HfRepo {
                        org: HfOrg::try_new("org".to_owned())?,
                        repo_name: HfRepoName::try_new("vision".to_owned())?,
                        revision: None,
                        auth_token,
                    },
                    model: RepoSubpath::try_new("model.gguf".to_owned())?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new("mmproj.gguf".to_owned())?,
                    mmproj_sha256: None,
                }),
            });
            r.model = pipette_plan_types::DeclaredBound::already_bound(model);
            r.benchmark = BenchmarkDefinition::VlmEval(vlm_benchmark()?);
            Ok(r)
        };

        let with_token = vlm_req(Some(AuthToken::try_new("hf_rotated_secret".to_owned())?))?;
        let without_token = vlm_req(None)?;
        assert_eq!(
            run_request_digest(&with_token)?,
            run_request_digest(&without_token)?,
            "rotating an HF token must not fork the vlm_eval resume identity"
        );
        Ok(())
    }

    /// An omitted entry hashes as the all-unset variant, never as the `null`
    /// of the unresolved projection.
    #[test]
    fn vlm_eval_digest_resolves_omitted_runtime_flags() -> anyhow::Result<()> {
        use pipette_plan_types::RuntimeFlags;

        let vlm = |flags: Option<RuntimeFlags>| -> anyhow::Result<RunRequest> {
            let mut r = vlm_req(BenchmarkDefinition::VlmEval(vlm_benchmark()?))?;
            r.runtime_flags = flags;
            Ok(r)
        };

        let omitted = vlm(None)?;
        let explicit_default = vlm(Some(RuntimeFlags::VlmEvalLlamacppCliStockToolsGgufVision(
            pipette_plan_types::VlmEvalLlamacppCliStockToolsGgufVision {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size: None,
                no_cache: None,
                parallel: None,
                envs: Vec::new(),
                raw: vec![],
            },
        )))?;
        let resolved_digest = run_request_digest(&omitted)?;
        assert_eq!(
            resolved_digest,
            run_request_digest(&explicit_default)?,
            "an omitted runtime_flags entry must digest as the resolved default \
             variant, not as null"
        );

        let mut pre_flag_era = vlm_eval_digest_view(&omitted)?;
        pre_flag_era["runtime_flags"] = Value::Null;
        assert_ne!(
            resolved_digest,
            digest_value(&pre_flag_era)?,
            "the resolved digest must not collide with the unresolved projection"
        );
        Ok(())
    }

    /// `enable_thinking` is the control: it changes what a completion is.
    #[test]
    fn vlm_eval_throughput_knobs_do_not_orphan_the_checkpoint() -> anyhow::Result<()> {
        use pipette_plan_types::{ModelFlags, RuntimeFlags};

        let vlm = |parallel: Option<i32>,
                   ctx_size: Option<u32>,
                   enable_thinking: Option<bool>|
         -> anyhow::Result<EvalRunDigest> {
            let mut r = vlm_req(BenchmarkDefinition::VlmEval(vlm_benchmark()?))?;
            r.runtime_flags = Some(RuntimeFlags::VlmEvalLlamacppCliStockToolsGgufVision(
                pipette_plan_types::VlmEvalLlamacppCliStockToolsGgufVision {
                    threads: None,
                    number_gpu_layers: None,
                    mmap: None,
                    load_mode: None,
                    flash_attention: None,
                    ctx_size,
                    no_cache: None,
                    parallel: parallel.map(PositiveCount::try_new).transpose()?,
                    envs: Vec::new(),
                    raw: vec![],
                },
            ));
            r.model_flags = Some(ModelFlags::VlmEvalGgufVision { enable_thinking });
            run_request_digest(&r)
        };

        let serial = vlm(None, None, Some(false))?;
        for parallel in [Some(1), Some(8), Some(64)] {
            assert_eq!(
                vlm(parallel, None, Some(false))?,
                serial,
                "parallel={parallel:?} must resume the same checkpoint",
            );
        }
        for ctx_size in [Some(16384), Some(65536)] {
            assert_eq!(
                vlm(Some(4), ctx_size, Some(false))?,
                serial,
                "ctx_size={ctx_size:?} must resume the same checkpoint",
            );
        }
        assert_ne!(
            vlm(None, None, Some(true))?,
            serial,
            "a knob that changes what a completion IS must still fork the identity",
        );
        Ok(())
    }

    #[test]
    fn a_local_vlm_eval_digest_follows_its_samples_content() -> anyhow::Result<()> {
        use pipette_plan_types::VlmEvalBenchmark;

        let vlm = |img: &str, id: &str| -> anyhow::Result<RunRequest> {
            vlm_req(BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
                samples: Some(vec![serde_json::from_value(json!({
                    "id": id,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url":
                            format!("data:image/png;base64,{img}")}}
                    ]}]
                }))?]),
                ..vlm_benchmark()?
            }))
        };

        assert_eq!(
            run_request_digest(&vlm("AAAA", "s1")?)?,
            run_request_digest(&vlm("AAAA", "s1")?)?,
        );
        assert_ne!(
            run_request_digest(&vlm("AAAA", "s1")?)?,
            run_request_digest(&vlm("ZZZZ", "s1")?)?,
        );
        assert_ne!(
            run_request_digest(&vlm("AAAA", "s1")?)?,
            run_request_digest(&vlm("AAAA", "s2")?)?,
        );
        Ok(())
    }

    /// The message says whether a re-run recovers it.
    #[test]
    fn ensure_complete_refuses_a_failed_row() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut s = store.open(&req)?;
        s.append(succeeded("s-1", "A"))?;
        s.append(failed("s-2", "blip"))?;
        let Err(err) = ensure_complete("bench-a", &s, 2) else {
            anyhow::bail!("a cell with a failed row must not publish");
        };
        let message = format!("{err:#}");
        assert!(message.contains("1 of 2 samples failed"), "{message}");
        assert!(message.contains("run this cell again"), "{message}");
        // An incomplete cell is never published, so no override is offered as a way to.
        assert!(message.contains("refuses it"), "{message}");
        assert!(!message.contains(ALLOW_INCOMPLETE_ENV), "{message}");
        Ok(())
    }

    /// Once a failure's attempts are spent, neither refusal offers the
    /// waiver: no family's incomplete cell is published, so the waiver is no
    /// way to publish one. Each names its own input to inspect.
    #[test]
    fn a_spent_refusal_offers_no_waiver() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut s = store.open(&req)?;
        s.append(succeeded("s-1", "A"))?;
        for _ in 0..MAX_SAMPLE_ATTEMPTS {
            s.append(failed("s-2", "blip"))?;
        }
        let Err(waivable) = ensure_complete("bench-a", &s, 2) else {
            anyhow::bail!("a cell with a failed row must not publish");
        };
        let Err(every) = ensure_every_answer("bench-a", &s, 2) else {
            anyhow::bail!("an asr_eval with a failed row must not publish");
        };
        for (message, input) in [(waivable, "image"), (every, "audio")] {
            let message = format!("{message:#}");
            assert!(message.contains("used up its 3 attempts"), "{message}");
            assert!(
                message.contains(&format!("decodable {input}?")),
                "{message}"
            );
            assert!(message.contains("refuses it"), "{message}");
            assert!(!message.contains(ALLOW_INCOMPLETE_ENV), "{message}");
            assert!(!message.contains("scored as wrong"), "{message}");
        }
        Ok(())
    }

    /// Refused even with no failed row.
    #[test]
    fn ensure_complete_refuses_a_shortfall() -> anyhow::Result<()> {
        let (_root, store, req) = store_and_req()?;
        let mut s = store.open(&req)?;
        s.append(succeeded("s-1", "A"))?;
        assert!(ensure_complete("bench-a", &s, 1).is_ok());
        let Err(err) = ensure_complete("bench-a", &s, 2) else {
            anyhow::bail!("a short cell must not publish");
        };
        assert!(
            format!("{err:#}").contains("1 completions for 2"),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn digest_ignores_bound_paths_and_source() -> anyhow::Result<()> {
        let mut a = req()?;
        let mut b = req()?;
        if let Runtime::LlamacppCliStockTools(rt) = &mut b.runtime.bound {
            rt.source = LlamacppCliStockToolsSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/other".to_owned())?,
            };
        }
        assert_eq!(run_request_digest(&a)?, run_request_digest(&b)?);

        if let Runtime::LlamacppCliStockTools(rt) = &mut a.runtime.declared {
            rt.flavor = LlamaCppFlavor::MacosX64;
        }
        assert_ne!(run_request_digest(&a)?, run_request_digest(&b)?);
        Ok(())
    }

    #[test]
    fn open_append_resume_finalize_and_clear() -> anyhow::Result<()> {
        let (_root, store, r) = store_and_req()?;

        {
            let mut s = store.open(&r)?;
            s.append(c("s1"))?;
            assert!(s.contains("s1"));
            assert!(!s.contains("s2"));
        }
        {
            let mut s = store.open(&r)?;
            assert!(s.contains("s1"));
            s.append(c("s2"))?;
            assert_eq!(s.into_completions().len(), 2);
        }
        // Finalizing a run where every sample succeeded removes the file.
        store.finalize(&run_request_digest(&r)?)?;
        assert_eq!(
            fs::read_dir(store.root())?
                .filter(|e| e.as_ref().ok().is_some_and(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x == "jsonl")
                }))
                .count(),
            0
        );

        {
            let mut s = store.open(&r)?;
            s.append(c("s1"))?;
        }
        store.clear()?;
        assert!(!store.root().exists());
        store.clear()?; // idempotent
        Ok(())
    }

    #[test]
    fn finalize_keeps_failed_only_and_reopen_skips_completed() -> anyhow::Result<()> {
        let (_root, store, r) = store_and_req()?;
        {
            let mut s = store.open(&r)?;
            s.append(c("done"))?;
            s.append(BenchmarkEvalCompletion {
                id: "bad".into(),
                completion: String::new(),
                failed: true,
                failed_reason: Some("crash".into()),
                ..Default::default()
            })?;
            assert_eq!(s.failed_ids().collect::<Vec<_>>(), ["bad"]);
            assert_eq!(s.into_completions().len(), 2);
        }
        store.finalize(&run_request_digest(&r)?)?;
        let s = store.open(&r)?;
        assert!(!s.contains("done"));
        assert!(s.contains("bad"));
        assert_eq!(s.failed_count(), 1);
        Ok(())
    }

    /// Under a new HTTP timeout the completed rows resume and a failed one runs
    /// again; under the same timeout, finalized or not, a failed row stays in
    /// the checkpoint.
    #[test]
    fn a_new_http_timeout_runs_the_failed_rows_again() -> anyhow::Result<()> {
        let (_root, store, _) = store_and_req()?;
        let under = |seconds: u64| -> anyhow::Result<RunRequest> {
            let mut r = req()?;
            r.benchmark_flags = Some(eval_timeout(Some(seconds))?);
            Ok(r)
        };
        let timed_out = BenchmarkEvalCompletion {
            id: "slow".into(),
            failed: true,
            failed_reason: Some("timed out".into()),
            ..Default::default()
        };
        {
            let mut s = store.open(&under(60)?)?;
            s.append(c("done"))?;
            s.append(timed_out.clone())?;
        }
        anyhow::ensure!(
            store.open(&under(60)?)?.contains("slow"),
            "the same timeout must skip a failed row"
        );

        let mut raised = store.open(&under(600)?)?;
        anyhow::ensure!(
            raised.contains("done") && !raised.contains("slow") && raised.failed_count() == 0,
            "a new timeout must resume the completed rows and run the failed one again"
        );
        raised.append(timed_out)?;
        drop(raised);
        store.finalize(&run_request_digest(&under(600)?)?)?;
        anyhow::ensure!(
            store.open(&under(600)?)?.contains("slow"),
            "a row that failed under the new timeout must stay skipped"
        );
        Ok(())
    }

    /// Finalizing twice, or with no run at all, does nothing.
    #[test]
    fn a_checkpoint_outlives_its_session_until_finalized() -> anyhow::Result<()> {
        let (_root, store, r) = store_and_req()?;
        let digest = run_request_digest(&r)?;
        store.finalize(&digest)?;
        {
            let mut s = store.open(&r)?;
            s.append(c("s1"))?;
            s.append(c("s2"))?;
            assert_eq!(s.into_completions().len(), 2);
        }
        let resumed = store.open(&r)?;
        assert!(resumed.contains("s1") && resumed.contains("s2"));
        drop(resumed);

        store.finalize(&digest)?;
        store.finalize(&digest)?;
        assert!(!store.open(&r)?.contains("s1"));
        Ok(())
    }

    /// Also checks that a resumed open targets exactly the failed ids.
    #[test]
    fn concurrent_appends_under_mutex_keep_counts_consistent() -> anyhow::Result<()> {
        use std::sync::{Mutex, PoisonError};

        const WORKERS: usize = 8;
        const PER_WORKER: usize = 25;

        let root = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(root.path().join("state/evals"));
        let r = req()?;
        let session = Mutex::new(store.open(&r)?);

        let results: Vec<anyhow::Result<()>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WORKERS)
                .map(|worker| {
                    let session = &session;
                    scope.spawn(move || -> anyhow::Result<()> {
                        (0..PER_WORKER).try_for_each(|i| {
                            let n = worker * PER_WORKER + i;
                            // Every 5th row fails, so `failed_count` has to
                            // stay exact across interleaved appends.
                            let failed = n.is_multiple_of(5);
                            let entry = BenchmarkEvalCompletion {
                                id: format!("s{n:03}"),
                                completion: if failed {
                                    String::new()
                                } else {
                                    format!("ans-{n}")
                                },
                                failed,
                                failed_reason: failed.then(|| "boom".to_string()),
                                ..Default::default()
                            };
                            session
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .append(entry)
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(anyhow::anyhow!("append worker panicked")))
                })
                .collect()
        });
        results.into_iter().collect::<anyhow::Result<()>>()?;

        let session = session.into_inner().unwrap_or_else(PoisonError::into_inner);
        let total = WORKERS * PER_WORKER;
        assert_eq!(session.completions().len(), total);
        assert_eq!(session.failed_count(), total / 5);
        (0..total).for_each(|n| assert!(session.contains(&format!("s{n:03}")), "missing s{n:03}"));

        drop(session.into_completions());
        store.finalize(&run_request_digest(&r)?)?;
        let resumed = store.open(&r)?;
        assert_eq!(resumed.failed_count(), total / 5);
        (0..total).for_each(|n| {
            assert_eq!(
                resumed.contains(&format!("s{n:03}")),
                n.is_multiple_of(5),
                "resume should carry only the failed ids; s{n:03} disagreed",
            );
        });
        Ok(())
    }

    /// Older clients' files can hold a failure row then a success for one id;
    /// read as a failure, the sample would be skipped on every resume.
    #[test]
    fn a_later_success_clears_an_earlier_failure_for_the_same_id() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(root.path().join("state/evals"));
        let r = req()?;
        {
            let mut s = store.open(&r)?;
            s.append(BenchmarkEvalCompletion {
                id: "collateral".into(),
                completion: String::new(),
                failed: true,
                failed_reason: Some("llama-server crashed mid-completion".into()),
                ..Default::default()
            })?;
            s.append(c("collateral"))?;
        }

        let resumed = store.open(&r)?;
        assert!(resumed.contains("collateral"));
        assert_eq!(resumed.failed_count(), 0);
        let all = resumed.into_completions();
        assert_eq!(all.len(), 1, "the row is replaced, not doubled");
        assert!(!all[0].failed);

        // The append-only tail must not resurrect the failure.
        store.finalize(&run_request_digest(&r)?)?;
        assert!(!store.open(&r)?.contains("collateral"));
        Ok(())
    }

    #[test]
    fn a_torn_last_line_does_not_swallow_the_next_append() -> anyhow::Result<()> {
        let (_root, store, r) = store_and_req()?;
        let path = {
            let mut s = store.open(&r)?;
            s.append(c("s1"))?;
            s.path().to_owned()
        };
        // A kill mid-append leaves half a row with no newline.
        OpenOptions::new()
            .append(true)
            .open(&path)?
            .write_all(br#"{"id":"s2","compl"#)?;
        {
            let mut s = store.open(&r)?;
            assert!(s.contains("s1") && !s.contains("s2"));
            s.append(c("s3"))?;
        }
        let s = store.open(&r)?;
        assert!(
            s.contains("s1") && s.contains("s3"),
            "{:?}",
            s.done_ids().collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn digest_mismatch_rotates_and_starts_fresh() -> anyhow::Result<()> {
        let (_root, store, mut r) = store_and_req()?;
        {
            let mut s = store.open(&r)?;
            s.append(c("s1"))?;
        }
        if let Runtime::LlamacppCliStockTools(rt) = &mut r.runtime.declared {
            rt.flavor = LlamaCppFlavor::LinuxX64Cpu;
        }
        let s = store.open(&r)?;
        assert!(!s.contains("s1"), "mismatched identity must not resume");
        Ok(())
    }
}
