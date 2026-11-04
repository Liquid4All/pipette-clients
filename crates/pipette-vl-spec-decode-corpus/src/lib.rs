//! Reader for `vl-spec-decode-corpus/1`, the corpus tar the management server
//! packs for a `vl_spec_decode` job: download through the [`RunContext`],
//! verify, unpack, read back, and traverse ([`turn`]).
//!
//! **Order is part of the measurement.** The two arms are comparable only
//! because they walk `samples.jsonl` in the same order, so
//! [`VlSpecDecodeCorpus::samples`] returns the file's order and nothing here
//! sorts, shuffles or partitions it.
//!
//! Its own crate rather than a module of `pipette-ops` because
//! `pipette-android` links `pipette-ops`, and the `tar` and `reqwest` trees
//! would ship in the APK for a benchmark Android cannot run (plan-types'
//! `is_compatible` never pairs a speculative model with its engine). So this
//! crate depends on `pipette-ops`, never the other way round. iOS links no Rust
//! and reads the format with a Swift port
//! (`ios/Pipette/Pipette/Benchmarks/VlSpecDecodeCorpus.swift`).
//!
//! [`vlm_eval`] reads the server's other corpus tar, a `vlm_eval`'s prompts,
//! and lives here to keep `tar` out of the APK too.
//! [`asr_eval`] reads an `asr_eval`'s, the same layout with clips for images,
//! from disk rather than memory: one is gigabytes of audio.

pub mod asr_eval;
pub mod turn;
pub mod vlm_eval;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BenchmarkId, CorpusFormat, Dataset, DatasetAdapter, DatasetUpstream, Sha256,
};

/// One part of a user turn; maps one-to-one onto a chat-completions content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VlSpecDecodeCorpusPart {
    Text {
        text: String,
    },
    Image {
        /// Tar-relative, `media/<sha256>.<ext>`.
        path: String,
        mime: String,
        bytes: u64,
    },
}

/// One `samples.jsonl` line: a whole conversation, user turns only.
///
/// Each arm generates the replies and a later turn's prompt carries the
/// earlier ones, which is why a readiness gate may run between samples but
/// never between turns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VlSpecDecodeCorpusSample {
    pub id: String,
    pub subset: String,
    pub system: String,
    pub turns: Vec<Vec<VlSpecDecodeCorpusPart>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VlSpecDecodeCorpusSubsetCounts {
    pub samples: u32,
    pub turns: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VlSpecDecodeCorpusManifest {
    pub format: CorpusFormat,
    pub adapter: DatasetAdapter,
    pub upstream: DatasetUpstream,
    pub samples: u32,
    pub turns: u32,
    #[serde(default)]
    pub subsets: BTreeMap<String, VlSpecDecodeCorpusSubsetCounts>,
}

#[derive(Debug)]
pub struct VlSpecDecodeCorpus {
    root: PathBuf,
    manifest: VlSpecDecodeCorpusManifest,
    samples: Vec<VlSpecDecodeCorpusSample>,
    digest: Sha256,
}

impl VlSpecDecodeCorpus {
    pub fn manifest(&self) -> &VlSpecDecodeCorpusManifest {
        &self.manifest
    }

    /// In file order; see the module docs for why there is no other.
    pub fn samples(&self) -> &[VlSpecDecodeCorpusSample] {
        &self.samples
    }

    /// Turns across every sample: the `vl_spec_decode_turns` a completed
    /// traversal reports.
    pub fn turns(&self) -> usize {
        self.samples.iter().map(|s| s.turns.len()).sum()
    }

    /// The sha256 this client computed over the tar it received. Submitted as
    /// `observation_dataset_digest` whether or not it matched the catalog.
    pub fn digest(&self) -> &Sha256 {
        &self.digest
    }

    /// The host path of a media part, refusing anything outside the corpus.
    ///
    /// [`unpack`] checked the archive's entries, not the paths `samples.jsonl`
    /// asks for; without this an index pointing outside the corpus would read
    /// host files into a prompt.
    pub fn media_path(&self, part_path: &str) -> anyhow::Result<PathBuf> {
        let relative = safe_relative(part_path)
            .with_context(|| format!("corpus media path {part_path:?} escapes the corpus"))?;
        let path = self.root.join(relative);
        anyhow::ensure!(
            path.is_file(),
            "corpus media file is missing: {}",
            path.display()
        );
        Ok(path)
    }
}

/// Download `dataset` through the client, verify it against its pin, and
/// unpack it under `<datasets_dir>/<benchmark_id>/`.
pub fn download(
    run_context: &dyn RunContext,
    benchmark_id: &BenchmarkId,
    dataset: &Dataset,
) -> anyhow::Result<VlSpecDecodeCorpusLease> {
    let started = Instant::now();
    let tar = run_context
        .fetch_dataset(dataset)
        .with_context(|| format!("failed to download the corpus of `{benchmark_id}`"))?;
    if tar.len() as u64 != dataset.bytes {
        // Not fatal: the digest is the real check, this is the first clue when it fails.
        tracing::warn!(
            %benchmark_id,
            bytes = tar.len(),
            expected_bytes = dataset.bytes,
            "corpus size differs from the catalog's pin"
        );
    }
    let dir = run_context.datasets_dir().join(benchmark_id.as_ref());
    // A job that died before cleanup leaves a tree here; unpacking over it
    // would let a stale file reach this traversal.
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to clear the stale corpus at {}", dir.display()))?;
    }
    let (corpus, matched) = verify_and_open(&tar, &dataset.digest, &dir)?;
    tracing::info!(
        %benchmark_id,
        elapsed = ?started.elapsed(),
        path = %dir.display(),
        samples = corpus.samples().len(),
        turns = corpus.turns(),
        digest = %corpus.digest(),
        digest_matched = matched,
        "corpus ready"
    );
    Ok(VlSpecDecodeCorpusLease::new(corpus, dir))
}

/// Verify `tar` against `expected`, unpack it under `dest`, and read it back;
/// the bool is whether the digest matched.
///
/// A mismatch is reported, never rejected: the row records which corpus was
/// measured, and refusing would turn a server-side repack into silence.
pub fn verify_and_open(
    tar: &[u8],
    expected: &Sha256,
    dest: &Path,
) -> anyhow::Result<(VlSpecDecodeCorpus, bool)> {
    let digest = Sha256::try_new(hex::encode(Sha256Hasher::digest(tar)))
        .map_err(|e| anyhow::anyhow!("hashing the corpus produced an unusable digest: {e}"))?;
    let matched = &digest == expected;
    if !matched {
        tracing::warn!(
            %digest,
            expected_digest = %expected,
            "corpus digest does not match the catalog's; recording the measured digest and \
             continuing"
        );
    }
    unpack(tar, dest)?;
    let corpus = open(dest, digest)?;
    Ok((corpus, matched))
}

/// Extract a `vl-spec-decode-corpus/1` tar into `dest`.
///
/// Not `tar::Archive::unpack`: that only keeps writes inside `dest`, but the
/// format is closed (`manifest.json`, `samples.jsonl`, `media/*`), so any other
/// entry means this is not the format the catalog pinned.
fn unpack(tar: &[u8], dest: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("failed to create the corpus directory {}", dest.display()))?;
    let mut archive = tar::Archive::new(std::io::Cursor::new(tar));
    archive
        .entries()
        .context("the corpus is not a readable tar")?
        .try_for_each(|entry| {
            let mut entry = entry.context("failed to read a corpus tar entry")?;
            let raw = entry.path().context("a corpus tar entry has no path")?;
            let name = raw
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path in the corpus tar: {raw:?}"))?
                .to_owned();
            let relative = safe_relative(&name)
                .with_context(|| format!("corpus tar entry {name:?} escapes the archive"))?;
            anyhow::ensure!(
                name == "manifest.json" || name == "samples.jsonl" || name.starts_with("media/"),
                "corpus tar holds an unexpected entry: {name:?}"
            );
            match entry.header().entry_type() {
                // The files under it are checked on their own.
                tar::EntryType::Directory => return Ok(()),
                tar::EntryType::Regular => {}
                // A link written as a file would be an empty media file, and
                // followed it could point anywhere.
                other @ (tar::EntryType::Link
                | tar::EntryType::Symlink
                | tar::EntryType::Char
                | tar::EntryType::Block
                | tar::EntryType::Fifo
                | tar::EntryType::Continuous
                | tar::EntryType::GNULongName
                | tar::EntryType::GNULongLink
                | tar::EntryType::GNUSparse
                | tar::EntryType::XGlobalHeader
                | tar::EntryType::XHeader
                | _) => anyhow::bail!("corpus tar entry {name:?} is a {other:?}, not a file"),
            }
            let out = dest.join(relative);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            let mut file = std::fs::File::create(&out)
                .with_context(|| format!("failed to create {}", out.display()))?;
            std::io::copy(&mut entry, &mut file)
                .with_context(|| format!("failed to write {}", out.display()))?;
            Ok(())
        })
}

fn open(root: &Path, digest: Sha256) -> anyhow::Result<VlSpecDecodeCorpus> {
    let manifest_path = root.join("manifest.json");
    let manifest: VlSpecDecodeCorpusManifest = serde_json::from_slice(&read_file(&manifest_path)?)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    let samples_path = root.join("samples.jsonl");
    let samples_text = String::from_utf8(read_file(&samples_path)?)
        .with_context(|| format!("{} is not UTF-8", samples_path.display()))?;
    let samples = samples_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str::<VlSpecDecodeCorpusSample>(line).with_context(|| {
                format!(
                    "failed to parse {} line {}",
                    samples_path.display(),
                    index + 1
                )
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!samples.is_empty(), "the corpus holds no samples");

    // The manifest is the server's tally of what it packed: a disagreement
    // means the two files describe different corpora.
    let turns: usize = samples.iter().map(|s| s.turns.len()).sum();
    anyhow::ensure!(
        samples.len() == manifest.samples as usize && turns == manifest.turns as usize,
        "the corpus manifest counts {} samples / {} turns, but samples.jsonl holds {} / {turns}",
        manifest.samples,
        manifest.turns,
        samples.len(),
    );
    Ok(VlSpecDecodeCorpus {
        root: root.to_path_buf(),
        manifest,
        samples,
        digest,
    })
}

fn read_file(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .read_to_end(&mut buf)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(buf)
}

/// `name` as a path of normal components only, or `None` when it could land
/// outside the corpus.
fn safe_relative(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    let ok = !name.is_empty()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
    ok.then(|| path.to_path_buf())
}

/// An unpacked corpus that deletes itself on drop.
///
/// The server serves the tar `Cache-Control: no-store`: the corpus is the
/// job's, not the host's. A guard rather than a cleanup call because a run
/// ends in many ways (a failed arm, a wedged server, a `?`).
#[derive(Debug)]
pub struct VlSpecDecodeCorpusLease {
    corpus: VlSpecDecodeCorpus,
    /// `None` once [`Self::release`] has run, so the drop does not remove twice.
    dir: Option<PathBuf>,
}

impl VlSpecDecodeCorpusLease {
    pub(crate) fn new(corpus: VlSpecDecodeCorpus, dir: PathBuf) -> Self {
        Self {
            corpus,
            dir: Some(dir),
        }
    }

    pub fn corpus(&self) -> &VlSpecDecodeCorpus {
        &self.corpus
    }

    /// Delete the corpus now and return the error, which the drop can only log.
    pub fn release(&mut self) -> anyhow::Result<()> {
        let Some(dir) = self.dir.take() else {
            return Ok(());
        };
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to delete the corpus at {}", dir.display()))
    }
}

impl Drop for VlSpecDecodeCorpusLease {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            tracing::warn!(error = %e, "could not delete the corpus");
        }
    }
}

#[cfg(test)]
mod tests {
    use pipette_ops::EvalCompletionsStore;

    use super::*;

    const UPSTREAM: &str = r#"{
        "source": "git",
        "url": "github.com/acme/corpus",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "identity": "2923603292c7dd463ad01a119f57965817778fc3",
        "digest": "2222222222222222222222222222222222222222222222222222222222222222"
    }"#;

    /// Two samples: a single image turn and a two-turn text conversation.
    fn corpus_tar() -> anyhow::Result<Vec<u8>> {
        let manifest = format!(
            r#"{{"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1","upstream":{UPSTREAM},"samples":2,"turns":3,"subsets":{{"GQA":{{"samples":2,"turns":3}}}}}}"#
        );
        let samples = concat!(
            r#"{"id":"a","subset":"GQA","system":"be brief","turns":[[{"type":"image","path":"media/aa.png","mime":"image/png","bytes":3},{"type":"text","text":"what is this?"}]]}"#,
            "\n",
            r#"{"id":"b","subset":"GQA","system":"","turns":[[{"type":"text","text":"hello"}],[{"type":"text","text":"and again"}]]}"#,
            "\n",
        );
        let mut builder = tar::Builder::new(Vec::new());
        let mut append = |path: &str, body: &[u8]| -> anyhow::Result<()> {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, path, body)?;
            Ok(())
        };
        append("manifest.json", manifest.as_bytes())?;
        append("samples.jsonl", samples.as_bytes())?;
        append("media/aa.png", b"png")?;
        Ok(builder.into_inner()?)
    }

    fn digest_of(bytes: &[u8]) -> anyhow::Result<Sha256> {
        Ok(Sha256::try_new(hex::encode(Sha256Hasher::digest(bytes)))?)
    }

    #[test]
    fn a_verified_corpus_reads_back_in_traversal_order() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let dir = tempfile::tempdir()?;
        let (corpus, matched) =
            verify_and_open(&tar, &digest_of(&tar)?, &dir.path().join("corpus"))?;
        assert!(matched);
        assert_eq!(
            corpus.samples().iter().map(|s| &s.id).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(corpus.turns(), 3);
        assert_eq!(corpus.manifest().format, CorpusFormat::VlSpecDecodeCorpusV1);
        assert!(corpus.media_path("media/aa.png")?.is_file());
        Ok(())
    }

    #[test]
    fn a_digest_mismatch_is_reported_rather_than_rejected() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let dir = tempfile::tempdir()?;
        let (corpus, matched) = verify_and_open(
            &tar,
            &Sha256::try_new("11".repeat(32))?,
            &dir.path().join("corpus"),
        )?;
        assert!(!matched);
        assert_eq!(corpus.digest(), &digest_of(&tar)?);
        Ok(())
    }

    #[test]
    fn disagreeing_counts_fail_the_read() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("corpus");
        std::fs::create_dir_all(&root)?;
        std::fs::write(
            root.join("manifest.json"),
            format!(
                r#"{{"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1","upstream":{UPSTREAM},"samples":9,"turns":9}}"#
            ),
        )?;
        std::fs::write(
            root.join("samples.jsonl"),
            "{\"id\":\"a\",\"subset\":\"GQA\",\"system\":\"\",\"turns\":[[]]}\n",
        )?;
        let error = open(&root, digest_of(b"x")?)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("manifest counts"), "{error}");
        Ok(())
    }

    #[rstest::rstest]
    #[case("../escape.json")]
    #[case("/etc/passwd")]
    #[case("notes.txt")]
    fn an_unexpected_tar_entry_is_refused(#[case] path: &str) -> anyhow::Result<()> {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(1);
        header.set_mode(0o644);
        header.set_cksum();
        // tar-rs refuses to write `..` and absolute paths, so those cases stop
        // at the builder and only `notes.txt` reaches `unpack`.
        if builder.append_data(&mut header, path, &b"x"[..]).is_err() {
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        assert!(unpack(&builder.into_inner()?, &dir.path().join("c")).is_err());
        Ok(())
    }

    /// Most packers write a directory entry before its files, so it is skipped
    /// rather than refused; a link is refused.
    #[rstest::rstest]
    #[case::directory(tar::EntryType::Directory, "media/", true)]
    #[case::symlink(tar::EntryType::Symlink, "media/link.png", false)]
    fn only_files_carry_content(
        #[case] kind: tar::EntryType,
        #[case] path: &str,
        #[case] expected_ok: bool,
    ) -> anyhow::Result<()> {
        let mut archive = tar::Archive::new(std::io::Cursor::new(corpus_tar()?));
        let mut builder = tar::Builder::new(Vec::new());
        archive
            .entries()?
            .try_for_each(|entry| -> anyhow::Result<()> {
                let mut entry = entry?;
                let mut body = Vec::new();
                entry.read_to_end(&mut body)?;
                let mut header = entry.header().clone();
                builder.append_data(&mut header, &entry.path()?, &body[..])?;
                Ok(())
            })?;
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(kind);
        header.set_size(0);
        header.set_mode(0o755);
        if kind == tar::EntryType::Symlink {
            header.set_link_name("aa.png")?;
        }
        header.set_cksum();
        builder.append_data(&mut header, path, &b""[..])?;
        let dir = tempfile::tempdir()?;
        assert_eq!(
            unpack(&builder.into_inner()?, &dir.path().join("c")).is_ok(),
            expected_ok
        );
        Ok(())
    }

    /// Refused at read time, not only at unpack time.
    #[test]
    fn an_escaping_media_path_is_refused() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let dir = tempfile::tempdir()?;
        let (corpus, _) = verify_and_open(&tar, &digest_of(&tar)?, &dir.path().join("corpus"))?;
        assert!(corpus.media_path("../../etc/passwd").is_err());
        Ok(())
    }

    #[test]
    fn a_lease_deletes_the_corpus_when_it_drops() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let dir = tempfile::tempdir()?;
        let root = dir.path().join("corpus");
        let (corpus, _) = verify_and_open(&tar, &digest_of(&tar)?, &root)?;
        {
            let lease = VlSpecDecodeCorpusLease::new(corpus, root.clone());
            assert_eq!(lease.corpus().samples().len(), 2);
        }
        assert!(!root.exists());
        Ok(())
    }

    struct Serving {
        tar: Vec<u8>,
        dir: PathBuf,
    }

    impl RunContext for Serving {
        fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
            Ok(self.tar.clone())
        }

        fn datasets_dir(&self) -> PathBuf {
            self.dir.clone()
        }

        fn eval_completions(&self) -> EvalCompletionsStore {
            EvalCompletionsStore::new(self.dir.join("evals"))
        }
    }

    fn dataset(digest: &Sha256, bytes: usize) -> anyhow::Result<Dataset> {
        Ok(serde_json::from_value(serde_json::json!({
            "format": "vl-spec-decode-corpus/1",
            "adapter": "mmspec@1",
            "digest": digest.as_ref(),
            "bytes": bytes,
            "upstream": serde_json::from_str::<serde_json::Value>(UPSTREAM)?,
        }))?)
    }

    fn benchmark_id() -> anyhow::Result<BenchmarkId> {
        Ok(BenchmarkId::try_new("vl_spec_decode_test".to_owned())?)
    }

    #[test]
    fn a_downloaded_corpus_is_unpacked_per_benchmark_and_leased() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let temp = tempfile::tempdir()?;
        let client = Serving {
            tar: tar.clone(),
            dir: temp.path().to_path_buf(),
        };
        let root = temp.path().join("vl_spec_decode_test");
        {
            let lease = download(
                &client,
                &benchmark_id()?,
                &dataset(&digest_of(&tar)?, tar.len())?,
            )?;
            assert_eq!(lease.corpus().samples().len(), 2);
            assert_eq!(lease.corpus().digest(), &digest_of(&tar)?);
            assert!(root.join("samples.jsonl").is_file());
        }
        assert!(!root.exists());
        Ok(())
    }

    #[test]
    fn a_stale_corpus_is_replaced() -> anyhow::Result<()> {
        let tar = corpus_tar()?;
        let temp = tempfile::tempdir()?;
        let stale = temp.path().join("vl_spec_decode_test/media/stale.png");
        std::fs::create_dir_all(temp.path().join("vl_spec_decode_test/media"))?;
        std::fs::write(&stale, b"old")?;
        let client = Serving {
            tar: tar.clone(),
            dir: temp.path().to_path_buf(),
        };

        let _lease = download(
            &client,
            &benchmark_id()?,
            &dataset(&digest_of(&tar)?, tar.len())?,
        )?;
        assert!(!stale.exists());
        Ok(())
    }

    #[test]
    fn a_refused_download_names_the_benchmark() -> anyhow::Result<()> {
        struct Refusing;

        impl RunContext for Refusing {
            fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
                anyhow::bail!("HTTP 503: dataset not built")
            }

            fn datasets_dir(&self) -> PathBuf {
                std::env::temp_dir()
            }

            fn eval_completions(&self) -> EvalCompletionsStore {
                EvalCompletionsStore::new(std::env::temp_dir().join("evals"))
            }
        }

        let error = download(
            &Refusing,
            &benchmark_id()?,
            &dataset(&Sha256::try_new("11".repeat(32))?, 10)?,
        )
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(error.contains("`vl_spec_decode_test`"), "{error}");
        assert!(error.contains("dataset not built"), "{error}");
        Ok(())
    }
}
