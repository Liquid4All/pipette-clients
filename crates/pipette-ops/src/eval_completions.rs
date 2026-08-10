//! Eval sample completion store — resume state keyed by [`RunRequest`].
//!
//! Portable [`RunRequest`]-keyed resume store (replaces the old marker-based path).
//! Layout under `root/` (= workspace `state/evals/`):
//!
//! ```text
//! <digest16>.jsonl          # header + completion lines
//! <digest16>.jsonl.stale-*  # rotated on digest mismatch / corrupt header
//! ```
//!
//! Digest is SHA-256 over **portable** [`RunRequest`] fields only (declared
//! runtime/model, flags, benchmark body) — never bound host paths — so the
//! same cell resumes across machines with the same plan coordinates.

use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use nutype::nutype;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use pipette_plan_types::result::BenchmarkEvalCompletion;
use pipette_plan_types::run::RunRequest;

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
}

/// Capability handle for eval sample completions (`root/` = `state/evals/`).
///
/// Layout is private. Mint from the workspace when wired (`ws.eval_completions()` → `EvalCompletionsStore`).
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

    /// This store's directory, for a caller that inspects the checkpoint files —
    /// mirroring `ResultsStore`'s path accessors.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Open or resume completions for this [`RunRequest`]'s portable identity.
    pub fn open(&self, req: &RunRequest) -> anyhow::Result<EvalCompletionSession> {
        let digest = run_request_digest(req)?;
        let meta = EvalCompletionMeta {
            benchmark_id: req.benchmark.benchmark_id().to_string(),
            runtime: req.runtime.declared.to_string(),
            model: req.model.declared.to_string(),
        };
        EvalCompletionSession::open_in(&self.root, &digest, meta)
    }

    /// Remove every session file and stale rotate under this store.
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

/// SHA-256 hex over the request's plan-stable projection — [`RunRequest`]'s own
/// `Serialize`, which omits the bound host paths and the catalog the body came
/// from. A field added to the request therefore enters cell identity rather than
/// being silently left out of it. Legacy marker checkpoints do not resume
/// against this identity.
pub fn run_request_digest(req: &RunRequest) -> anyhow::Result<EvalRunDigest> {
    digest_value(&serde_json::to_value(req)?)
}

/// Live append session for one eval run. Drop without [`Self::finalize`] to keep
/// resume state for the next open of the same digest.
pub struct EvalCompletionSession {
    path: PathBuf,
    file: File,
    completions: Vec<BenchmarkEvalCompletion>,
    done_ids: HashSet<String>,
    failed_count: usize,
}

impl EvalCompletionSession {
    fn open_in(
        root: &Path,
        digest: &EvalRunDigest,
        meta: EvalCompletionMeta,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(root)
            .with_context(|| format!("failed to create eval completions dir {}", root.display()))?;
        let path = root.join(format!("{}.jsonl", digest_prefix(digest)));

        let (completions, done_ids, needs_fresh_header) = if path.exists() {
            match load_existing(&path, digest) {
                Ok(loaded) => loaded,
                Err(e) => {
                    log::warn!(
                        "eval completions at {} unreadable ({e}); rotating stale",
                        path.display()
                    );
                    rotate_stale(&path);
                    (Vec::new(), HashSet::new(), true)
                }
            }
        } else {
            (Vec::new(), HashSet::new(), true)
        };

        if needs_fresh_header {
            write_fresh_header(&path, digest, &meta)?;
        }

        let file = OpenOptions::new()
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open {} for append", path.display()))?;

        if !done_ids.is_empty() {
            log::info!(
                "eval completions: resuming {} from {}",
                done_ids.len(),
                path.display()
            );
        }

        let failed_count = completions.iter().filter(|c| c.failed).count();
        Ok(Self {
            path,
            file,
            completions,
            done_ids,
            failed_count,
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

        if let Some(existing_idx) = self.completions.iter().position(|c| c.id == completion.id) {
            log::warn!(
                "eval completions: duplicate id `{}` (last write wins)",
                completion.id
            );
            let old = self.completions.remove(existing_idx);
            if old.failed {
                self.failed_count = self.failed_count.saturating_sub(1);
            }
        }
        self.done_ids.insert(completion.id.clone());
        if completion.failed {
            self.failed_count += 1;
        }
        self.completions.push(completion);
        Ok(())
    }

    /// Owned-style append for `try_fold` sample loops.
    pub fn with_append(mut self, completion: BenchmarkEvalCompletion) -> anyhow::Result<Self> {
        self.append(completion)?;
        Ok(self)
    }

    pub fn contains(&self, sample_id: &str) -> bool {
        self.done_ids.contains(sample_id)
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

    /// Finish the run: delete the file when every sample succeeded; otherwise
    /// rewrite keeping only failed rows so a retry can target them.
    pub fn finalize(self) -> anyhow::Result<Vec<BenchmarkEvalCompletion>> {
        let path = self.path.clone();
        let completions = self.completions;
        drop(self.file);

        let failed: Vec<BenchmarkEvalCompletion> =
            completions.iter().filter(|c| c.failed).cloned().collect();

        if failed.is_empty() {
            if path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        } else {
            rewrite_with_only_failed(&path, &failed)?;
        }
        Ok(completions)
    }
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

fn write_fresh_header(
    path: &Path,
    digest: &EvalRunDigest,
    meta: &EvalCompletionMeta,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let header = Header {
        digest: digest.clone(),
        meta: meta.clone(),
    };
    let mut line = serde_json::to_string(&header).context("failed to serialize header")?;
    line.push('\n');
    fs::write(path, line).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn load_existing(
    path: &Path,
    expected: &EvalRunDigest,
) -> anyhow::Result<(Vec<BenchmarkEvalCompletion>, HashSet<String>, bool)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(e)) => return Err(e).context("failed to read header"),
        None => anyhow::bail!("empty completion file"),
    };
    let header: Header =
        serde_json::from_str(&header_line).context("failed to parse completion header")?;
    if &header.digest != expected {
        log::info!(
            "eval completions digest mismatch at {} (file={}, want={}); rotating",
            path.display(),
            header.digest,
            expected
        );
        rotate_stale(path);
        return Ok((Vec::new(), HashSet::new(), true));
    }

    let mut by_id: std::collections::HashMap<String, BenchmarkEvalCompletion> =
        std::collections::HashMap::new();
    for (idx, line) in lines.enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 2))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<BenchmarkEvalCompletion>(&line) {
            Ok(c) => {
                by_id.insert(c.id.clone(), c);
            }
            Err(e) => {
                log::warn!(
                    "skipping corrupt completion line {} in {}: {e}",
                    idx + 2,
                    path.display()
                );
            }
        }
    }
    let completions: Vec<_> = by_id.into_values().collect();
    let done_ids = completions.iter().map(|c| c.id.clone()).collect();
    Ok((completions, done_ids, false))
}

fn rotate_stale(path: &Path) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dest = path.with_extension(format!("jsonl.stale-{ts}"));
    if let Err(e) = fs::rename(path, &dest) {
        log::warn!(
            "failed to rotate stale completions {} → {}: {e}",
            path.display(),
            dest.display()
        );
    }
}

fn rewrite_with_only_failed(path: &Path, failed: &[BenchmarkEvalCompletion]) -> anyhow::Result<()> {
    // Keep header; rewrite body to failed-only.
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let header_line = lines
        .next()
        .transpose()
        .context("failed to read header for rewrite")?
        .context("missing header on finalize rewrite")?;

    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut out =
            File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?;
        writeln!(out, "{header_line}").context("failed to write header")?;
        for c in failed {
            let line = serde_json::to_string(c).context("serialize failed completion")?;
            writeln!(out, "{line}").context("write failed completion")?;
        }
        out.flush().context("flush rewrite")?;
    }
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "failed to replace {} with failed-only rewrite",
            path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use pipette_plan_types::benchmark::eval_id::EvalId;
    use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark};
    use pipette_plan_types::{
        AbsolutePath, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor,
        LlamacppCliStockToolsSource, Model, NonEmptyString, RepoSubpath, RepositoryUrl, Runtime,
        SourceRepository,
    };

    use super::*;

    /// A store on a fresh temp dir plus a request to key it by. The `TempDir` is
    /// returned, not dropped: it owns the directory the store points at.
    ///
    /// No `state/evals` segment — the store creates whatever root it is given, so
    /// a test reproducing the workspace's layout would be asserting nothing.
    fn store_and_req() -> anyhow::Result<(tempfile::TempDir, EvalCompletionsStore, RunRequest)> {
        let root = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(root.path());
        let req = req()?;
        Ok((root, store, req))
    }

    fn runtime() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(
            pipette_plan_types::LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                    repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                    repository_version: NonEmptyString::try_new("b1234".to_owned())?,
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

    fn benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "bench-a".into(),
            parameter_eval_id: EvalId::from("math_500"),
            parameter_dataset_name: "math_500".into(),
            parameter_max_tokens: 16,
            parameter_mcq_choices: None,
            samples: Some(vec![json!({"id": "s1", "prompt": "hi"})]),
        })
    }

    fn req() -> anyhow::Result<RunRequest> {
        let rt = runtime()?;
        let m = model()?;
        Ok(RunRequest {
            runtime: pipette_plan_types::run::DeclaredBound::already_bound(rt),
            model: pipette_plan_types::run::DeclaredBound::already_bound(m),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: benchmark(),
        })
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

    /// Resume compatibility: a checkpoint written by any earlier build has to
    /// keep resuming, so the digest of a fixed request is a wire constant, not
    /// an implementation detail of how the projection is taken.
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
    fn digest_ignores_bound_paths_and_source() -> anyhow::Result<()> {
        let mut a = req()?;
        let mut b = req()?;
        // Mutate bound only (Absolute-style path change).
        if let Runtime::LlamacppCliStockTools(rt) = &mut b.runtime.bound {
            rt.source = LlamacppCliStockToolsSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/other".to_owned())?,
            };
        }
        assert_eq!(run_request_digest(&a)?, run_request_digest(&b)?);

        // Declared change must move the digest.
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
            let all = s.finalize()?;
            assert_eq!(all.len(), 2);
        }
        // Successful finalize removes the file.
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

        // New run after clear.
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
            let all = s.finalize()?;
            assert_eq!(all.len(), 2);
        }
        // Failed-only rewrite left on disk; completed id is gone.
        let s = store.open(&r)?;
        assert!(!s.contains("done"));
        assert!(s.contains("bad"));
        assert_eq!(s.failed_count(), 1);
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
