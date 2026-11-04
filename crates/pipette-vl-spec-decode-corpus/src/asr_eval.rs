//! `asr-eval-corpus/1`: an `asr_eval`'s samples as the management server packs
//! them, each clip stored once under `media/`, read back into the exact
//! samples the evals server serves and grades.
//!
//! The checks are `vlm-eval-corpus/1`'s ([`crate::vlm_eval`]), and so is the
//! rule: **a mismatch fails the run**. The tar must hash to the catalog's pin,
//! its manifest must name the pinned upstream, and the samples it rebuilds
//! must hash to the evals server's identity for the entry's task, version and
//! prompt profile, before the first prompt is sent.
//!
//! What differs is the size. One profile's corpus of a large set is about
//! 11.5 GB of audio, so nothing here holds the corpus:
//!
//! - The download streams to a file under the run's datasets directory,
//!   hashed as it is written, once the disk is known to have room for it.
//! - The tar is indexed by its headers alone: where each media file starts and
//!   how long it is. Only `manifest.json` and `samples.jsonl` (ids, prompt
//!   text and media paths, a few MB) are read whole.
//! - The identity pass rebuilds one sample at a time, each clip read by offset
//!   and checked against its name and size, and keeps none of them.
//! - The run rebuilds each sample again when it is sent
//!   ([`AsrEvalCorpus::samples`]), so memory holds the index and the samples
//!   in flight, never the corpus.
//!
//! The file belongs to the run: each run writes its own, named for it and
//! locked while the run holds it, so two runs of one entry never share one.
//! It is deleted when the corpus drops, as a `vl_spec_decode` corpus is, and
//! by the teardown ^C and SIGTERM run. What a SIGKILL leaves, unlocked once
//! its process is gone, the next run of the entry deletes
//! ([`remove_leftover`]).
//! The server serves it `Cache-Control: no-store`.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use anyhow::Context;
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AsrEvalBenchmark, BenchmarkType, CorpusFormat, Dataset, DatasetAdapter, DatasetUpstream,
    EvalSample, Model, ModelFlags, SamplesDigest, Sha256, VlmEvalCorpusManifest,
    VlmEvalCorpusSample,
};

/// `benchmark`, if a cell running `model` with `model_flags` may run it: a
/// catalog entry (one that pins a corpus) always says whose recipe it is, and
/// runs only the models it covers, with thinking off, as its recipe says; mgmt
/// refuses such a cell before queueing it, and this is the same rule for a
/// run nothing queued. Every asr_eval cell states `enable_thinking = false`
/// in its model flags, recipe or not.
pub fn on_recipe<'a>(
    benchmark: &'a AsrEvalBenchmark,
    model: &Model,
    model_flags: Option<&ModelFlags>,
) -> anyhow::Result<&'a AsrEvalBenchmark> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    let enable_thinking = model_flags.and_then(ModelFlags::enable_thinking);
    match (&benchmark.recipe, &benchmark.dataset) {
        (Some(recipe), _) => recipe
            .check_cell(BenchmarkType::AsrEval, model, enable_thinking)
            .with_context(|| format!("asr_eval `{benchmark_id}` is off its recipe"))?,
        (None, Some(_)) => anyhow::bail!(
            "asr_eval `{benchmark_id}` pins a corpus but names no recipe; a catalog entry says \
             whose settings it holds"
        ),
        (None, None) => {}
    }
    anyhow::ensure!(
        enable_thinking == Some(false),
        "asr_eval `{benchmark_id}` runs with thinking off, and its cell must say so: \
         model_flags = {{ benchmark_type = \"asr_eval\", model_type = \"{}\", \
         enable_thinking = false }}",
        pipette_plan_types::ModelType::of(model),
    );
    Ok(benchmark)
}

/// The corpus is the dataset the entry declares, its rows where their ids
/// say: the chunks are read by position and the checkpoint is keyed by id,
/// so a row out of place would be skipped or repeated.
pub fn check_positions(
    benchmark_id: &str,
    corpus: &AsrEvalCorpus,
    declared: usize,
    id_at: impl Fn(usize) -> String,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        corpus.len() == declared,
        "asr_eval `{benchmark_id}` declares parameter_sample_count={declared} but its corpus \
         holds {} samples",
        corpus.len(),
    );
    if let Some(position) = (0..declared).find(|&at| corpus.id(at) != Some(id_at(at).as_str())) {
        anyhow::bail!(
            "asr_eval `{benchmark_id}`: the corpus's sample at position {position} is id `{}`, \
             where the dataset's is `{}`",
            corpus.id(position).unwrap_or_default(),
            id_at(position),
        );
    }
    Ok(())
}

/// A downloaded corpus is written to `<TAR_PREFIX>.<run>.tar` in its
/// benchmark's directory, one file per run ([`run_tar_name`]).
const TAR_PREFIX: &str = "asr-eval-corpus";
const TAR_SUFFIX: &str = ".tar";
/// The sidecar beside a run's tar that the run holds locked ([`lock_path`]).
const LOCK_SUFFIX: &str = ".lock";

/// This run's corpus file name: the process id and a counter, so two runs of
/// one entry, in two processes or one, never write the same file.
fn run_tar_name() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{TAR_PREFIX}.{}-{}{TAR_SUFFIX}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Room left on the disk beyond the corpus itself: the checkpoint, logs, and
/// whatever else the run writes while it downloads.
const DISK_HEADROOM: u64 = 256 * 1024 * 1024;

/// The most `manifest.json` or `samples.jsonl` may hold. The largest set's
/// index is a few MB; one past this is not a corpus mgmt packed.
const MAX_INDEX_BYTES: u64 = 512 * 1024 * 1024;

/// Where a media file's bytes sit in the tar.
#[derive(Debug, Clone, Copy)]
struct MediaEntry {
    offset: u64,
    size: u64,
}

/// A verified corpus on disk: its index in memory, its clips read on demand.
#[derive(Debug)]
pub struct AsrEvalCorpus {
    tar: Mutex<File>,
    manifest: VlmEvalCorpusManifest,
    lines: Vec<VlmEvalCorpusSample>,
    media: BTreeMap<String, MediaEntry>,
    digest: Sha256,
    /// Deletes the downloaded file when the corpus drops; `None` for a file
    /// the caller owns ([`open`]).
    _lease: Option<FileLease>,
}

impl AsrEvalCorpus {
    pub fn manifest(&self) -> &VlmEvalCorpusManifest {
        &self.manifest
    }

    /// How many samples the corpus holds.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The id of the sample at `position`, in the order the evals server
    /// serves them.
    pub fn id(&self, position: usize) -> Option<&str> {
        self.lines.get(position).map(|line| line.id.as_str())
    }

    /// The sha256 of the tar, which matched the catalog's pin. Submitted as
    /// `observation_dataset_digest`.
    pub fn digest(&self) -> &Sha256 {
        &self.digest
    }

    /// The samples at `positions`, rebuilt from the tar: each clip read by
    /// offset and checked against its name and size again, so a file changed
    /// on disk after it was verified is refused rather than sent.
    pub fn samples(&self, positions: Range<usize>) -> anyhow::Result<Vec<EvalSample>> {
        let lines = self.lines.get(positions.clone()).with_context(|| {
            format!(
                "samples {}..{} asked of a corpus of {}",
                positions.start,
                positions.end,
                self.lines.len()
            )
        })?;
        lines.iter().map(|line| self.rebuild(line)).collect()
    }

    /// The samples from `from` on, as many as fit in `max_samples` and
    /// `max_media_bytes` of clips, and at least one: how a run reads the
    /// corpus, a bounded chunk at a time. Empty at the end.
    pub fn chunk(
        &self,
        from: usize,
        max_samples: usize,
        max_media_bytes: u64,
    ) -> anyhow::Result<Vec<EvalSample>> {
        let mut end = from;
        let mut bytes = 0_u64;
        while let Some(line) = self.lines.get(end) {
            if end - from >= max_samples.max(1) {
                break;
            }
            let clips = line
                .media_paths()
                .filter_map(|path| self.media.get(path))
                .fold(0_u64, |sum, entry| sum.saturating_add(entry.size));
            if end > from && bytes.saturating_add(clips) > max_media_bytes {
                break;
            }
            bytes = bytes.saturating_add(clips);
            end += 1;
        }
        self.samples(from..end)
    }

    fn rebuild(&self, line: &VlmEvalCorpusSample) -> anyhow::Result<EvalSample> {
        let mut tar = self.tar.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(
            line.to_eval_sample_in(CorpusFormat::AsrEvalCorpusV1, |path| {
                read_media(&mut tar, &self.media, path)
            })?,
        )
    }
}

/// The downloaded tar, deleted when the run is done with it, or by the
/// signal teardown when ^C or SIGTERM ends the run (that exit skips `Drop`).
/// `held` locks the tar's sidecar ([`lock_path`]) while the run holds it, so
/// another run's [`remove_leftover`] passes it by; the lock goes with the
/// process. The lock is on the sidecar, never the tar: on Windows a lock is
/// mandatory, and would refuse the run's own reads of a locked tar.
#[derive(Debug)]
struct FileLease {
    path: PathBuf,
    held: File,
    _on_signal: pipette_subprocess::SignalTeardown,
}

impl FileLease {
    /// Lock `path`'s sidecar for this run, then create `path`, which must not
    /// exist.
    fn create(path: PathBuf) -> anyhow::Result<Self> {
        let lock = lock_path(&path);
        let held = File::create_new(&lock)
            .with_context(|| format!("failed to create {}", lock.display()))?;
        let on_signal = path.clone();
        let lease = Self {
            _on_signal: pipette_subprocess::on_signal(move || remove_corpus(&on_signal)),
            held,
            path,
        };
        lease
            .held
            .try_lock()
            .with_context(|| format!("failed to lock {}", lock.display()))?;
        File::create_new(&lease.path)
            .with_context(|| format!("failed to create {}", lease.path.display()))?;
        Ok(lease)
    }

    /// A handle to write the file through.
    fn writer(&self) -> anyhow::Result<File> {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))
    }
}

impl Drop for FileLease {
    fn drop(&mut self) {
        remove_corpus(&self.path);
    }
}

/// The sidecar a run locks while it holds the corpus at `tar`.
fn lock_path(tar: &Path) -> PathBuf {
    let mut name = tar.as_os_str().to_owned();
    name.push(LOCK_SUFFIX);
    PathBuf::from(name)
}

/// Delete the corpus at `path` and its lock sidecar.
fn remove_corpus(path: &Path) {
    for file in [path.to_owned(), lock_path(path)] {
        match std::fs::remove_file(&file) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => tracing::warn!(
                path = %file.display(),
                error = %err,
                "failed to delete the asr_eval corpus"
            ),
        }
    }
}

/// Delete the corpora killed runs of `benchmark` left (SIGKILL runs no
/// teardown). A download does this itself; a run with every sample settled
/// downloads nothing, so it calls this instead. A corpus another live run
/// holds is locked, and left to it.
pub fn remove_leftover(
    run_context: &dyn RunContext,
    benchmark: &AsrEvalBenchmark,
) -> anyhow::Result<()> {
    remove_unheld(
        &run_context
            .datasets_dir()
            .join(benchmark.benchmark_id.as_str()),
    )
}

/// Delete every corpus file in `dir` that no live run holds locked, with its
/// sidecar, and every sidecar whose corpus is gone.
fn remove_unheld(dir: &Path) -> anyhow::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", dir.display())),
    };
    for entry in entries {
        let path = entry
            .with_context(|| format!("failed to read {}", dir.display()))?
            .path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with(TAR_PREFIX) {
            continue;
        }
        // Each corpus and each orphaned sidecar is judged by its sidecar.
        let tar = match name.strip_suffix(LOCK_SUFFIX) {
            Some(tar) if tar.ends_with(TAR_SUFFIX) => {
                let tar = path.with_file_name(tar);
                if tar.exists() {
                    // Judged with its corpus.
                    continue;
                }
                tar
            }
            None if name.ends_with(TAR_SUFFIX) => path.clone(),
            _ => continue,
        };
        if !held_by_a_live_run(&lock_path(&tar))? {
            for file in [tar.clone(), lock_path(&tar)] {
                match std::fs::remove_file(&file) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => {
                        return Err(err)
                            .with_context(|| format!("failed to remove {}", file.display()));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Whether a live run holds the sidecar at `lock`. A missing sidecar is held
/// by no one: a killed run's, or one from before sidecars.
fn held_by_a_live_run(lock: &Path) -> anyhow::Result<bool> {
    let file = match File::open(lock) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("failed to open {}", lock.display())),
    };
    match file.try_lock() {
        Ok(()) => Ok(false),
        Err(std::fs::TryLockError::WouldBlock) => Ok(true),
        Err(std::fs::TryLockError::Error(err)) => {
            Err(err).with_context(|| format!("failed to lock {}", lock.display()))
        }
    }
}

/// Download the corpus `benchmark` pins and verify it, handing each rebuilt
/// sample to `visit` as the identity pass reaches it (the run sizes its
/// context from them). Every check fails the run before a prompt is sent: an
/// entry that pins no corpus, a pin for other samples, too little disk, a tar
/// that is not the pinned one, and samples that are not the evals server's.
pub fn download(
    run_context: &dyn RunContext,
    benchmark: &AsrEvalBenchmark,
    visit: impl FnMut(&EvalSample) -> anyhow::Result<()>,
) -> anyhow::Result<AsrEvalCorpus> {
    let benchmark_id = &benchmark.benchmark_id;
    let dataset = benchmark.dataset.as_deref().with_context(|| {
        format!(
            "asr_eval `{benchmark_id}` pins no corpus; an asr_eval's samples reach a run only \
             as the asr-eval-corpus/1 corpus its catalog entry pins"
        )
    })?;
    let started = Instant::now();
    check_pin(benchmark, dataset)?;
    let dir = run_context.datasets_dir().join(benchmark_id.as_str());
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    // A file a killed run left is not trusted: it is deleted, and the free
    // space it held counts. Another live run's is locked, and left alone.
    remove_unheld(&dir)?;
    check_room(&dir, dataset.bytes)
        .with_context(|| format!("asr_eval `{benchmark_id}` cannot download its corpus"))?;
    let path = dir.join(run_tar_name());
    let lease = FileLease::create(path.clone())?;
    let digest = fetch_to(run_context, dataset, lease.writer()?, &path)
        .with_context(|| format!("failed to download the corpus of `{benchmark_id}`"))?;
    let corpus = read(&path, dataset, digest, Some(lease), visit)
        .with_context(|| format!("the corpus of `{benchmark_id}` is not the one it pins"))?;
    tracing::info!(
        %benchmark_id,
        elapsed = ?started.elapsed(),
        bytes = dataset.bytes,
        samples = corpus.lines.len(),
        media = corpus.manifest.media,
        digest = %corpus.digest,
        "asr_eval corpus ready"
    );
    Ok(corpus)
}

/// The pin names the samples the entry is graded on, in the one format this
/// reads. The catalog refuses an entry that pins others; checked here too,
/// before the download, for a definition that reached the client some other
/// way. An asr_eval's dataset holds no prompt, so its pin always names the
/// profile.
fn check_pin(benchmark: &AsrEvalBenchmark, dataset: &Dataset) -> anyhow::Result<()> {
    let benchmark_id = &benchmark.benchmark_id;
    anyhow::ensure!(
        dataset.format == CorpusFormat::AsrEvalCorpusV1
            && dataset.adapter == DatasetAdapter::EvalsServerV1,
        "asr_eval `{benchmark_id}` pins a {} corpus from {}; it reads asr-eval-corpus/1 from \
         evals-server@1",
        dataset.format,
        dataset.adapter
    );
    let DatasetUpstream::EvalsServer {
        eval_id,
        dataset_version,
        prompt_profile,
        ..
    } = &dataset.upstream
    else {
        anyhow::bail!(
            "asr_eval `{benchmark_id}` pins {}, not the evals server's samples",
            dataset.upstream.reference()
        );
    };
    let pinned = (
        eval_id.as_ref(),
        dataset_version.as_ref(),
        prompt_profile.as_ref().map(AsRef::as_ref),
    );
    let runs = (
        benchmark.parameter_asr_task.as_str(),
        benchmark.parameter_dataset_version.as_str(),
        Some(benchmark.parameter_prompt_profile.as_str()),
    );
    anyhow::ensure!(
        pinned == runs,
        "asr_eval `{benchmark_id}` pins the samples of {}, but runs {}/{}?prompt_profile={}: they \
         are not the samples the scorer grades it on",
        dataset.upstream.reference(),
        runs.0,
        runs.1,
        benchmark.parameter_prompt_profile.as_str(),
    );
    Ok(())
}

/// The disk under `dir` holds `bytes` more and the headroom, where it can be
/// asked; a platform that cannot say is let through, and a short disk then
/// fails the write.
fn check_room(dir: &Path, bytes: u64) -> anyhow::Result<()> {
    let Some(available) = available_bytes(dir) else {
        return Ok(());
    };
    let needed = bytes.saturating_add(DISK_HEADROOM);
    anyhow::ensure!(
        available >= needed,
        "the corpus is {bytes} bytes, and {} has {available} free; it needs {needed} with room \
         for the run beside it. Free disk space, or run the cell on a device with more",
        dir.display(),
    );
    Ok(())
}

/// The bytes free to this process on the filesystem holding `dir`.
#[cfg(unix)]
fn available_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a NUL-terminated C string that outlives the call, and
    // `stat` points at writable memory sized for one `statvfs`.
    let status = unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) };
    if status != 0 {
        return None;
    }
    // SAFETY: `statvfs` returned 0, so it filled `stat`.
    let stat = unsafe { stat.assume_init() };
    #[allow(
        clippy::useless_conversion,
        reason = "the block count and size are u32 on some targets and u64 on others"
    )]
    let (blocks, size) = (u64::from(stat.f_bavail), u64::from(stat.f_frsize));
    Some(blocks.saturating_mul(size))
}

#[cfg(not(unix))]
fn available_bytes(_dir: &Path) -> Option<u64> {
    None
}

/// Stream `dataset` into `file` (at `path`), and the sha256 of what was
/// written, once it is the pinned size.
fn fetch_to(
    run_context: &dyn RunContext,
    dataset: &Dataset,
    file: File,
    path: &Path,
) -> anyhow::Result<Sha256> {
    let mut writer = HashingWriter {
        inner: BufWriter::new(file),
        hasher: Sha256Hasher::new(),
        written: 0,
        limit: dataset.bytes,
    };
    run_context.fetch_dataset_to(dataset, &mut writer)?;
    writer
        .inner
        .flush()
        .with_context(|| format!("failed to write {}", path.display()))?;
    anyhow::ensure!(
        writer.written == dataset.bytes,
        "the server sent {} bytes, the catalog pins a corpus of {}",
        writer.written,
        dataset.bytes
    );
    sha256(writer.hasher)
}

/// A writer that hashes and counts what passes through it, and refuses a
/// byte past `limit`: the disk was checked for the pinned size, so a server
/// that sends more stops the download there rather than filling the disk.
struct HashingWriter<W> {
    inner: W,
    hasher: Sha256Hasher,
    written: u64,
    limit: u64,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let incoming = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        if self.written.saturating_add(incoming) > self.limit {
            return Err(std::io::Error::other(format!(
                "the server sent more than the {} bytes the catalog pins",
                self.limit
            )));
        }
        let n = self.inner.write(buf)?;
        let written = buf.get(..n).unwrap_or(buf);
        self.hasher.update(written);
        self.written = self
            .written
            .saturating_add(u64::try_from(n).unwrap_or(u64::MAX));
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn sha256(hasher: Sha256Hasher) -> anyhow::Result<Sha256> {
    Sha256::try_new(hex::encode(hasher.finalize()))
        .map_err(|e| anyhow::anyhow!("hashing the corpus produced an unusable digest: {e}"))
}

/// Verify the corpus tar at `path` against `dataset`, as [`download`] does
/// after it fetched one. The file stays the caller's.
pub fn open(
    path: &Path,
    dataset: &Dataset,
    visit: impl FnMut(&EvalSample) -> anyhow::Result<()>,
) -> anyhow::Result<AsrEvalCorpus> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hashing = HashingWriter {
        inner: std::io::sink(),
        hasher: Sha256Hasher::new(),
        written: 0,
        // A file already on disk: its digest decides, whatever its size.
        limit: u64::MAX,
    };
    std::io::copy(&mut BufReader::new(file), &mut hashing)
        .with_context(|| format!("failed to read {}", path.display()))?;
    read(path, dataset, sha256(hashing.hasher)?, None, visit)
}

/// The checks: the tar's digest, then its manifest and index, then every
/// sample rebuilt against the evals server's identity.
fn read(
    path: &Path,
    dataset: &Dataset,
    digest: Sha256,
    lease: Option<FileLease>,
    mut visit: impl FnMut(&EvalSample) -> anyhow::Result<()>,
) -> anyhow::Result<AsrEvalCorpus> {
    anyhow::ensure!(
        digest == dataset.digest,
        "the corpus hashes to {digest}, the catalog pins {}",
        dataset.digest
    );
    let DatasetUpstream::EvalsServer {
        identity: pinned, ..
    } = &dataset.upstream
    else {
        anyhow::bail!(
            "{} is not the evals server's samples",
            dataset.upstream.reference()
        );
    };
    let mut tar = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let index = index_entries(&mut tar)?;
    let manifest: VlmEvalCorpusManifest = serde_json::from_slice(
        index
            .manifest
            .as_deref()
            .context("the corpus has no manifest.json")?,
    )
    .context("failed to parse manifest.json")?;
    anyhow::ensure!(
        manifest.format == CorpusFormat::AsrEvalCorpusV1
            && manifest.adapter == DatasetAdapter::EvalsServerV1,
        "the corpus manifest names format {} from {}",
        manifest.format,
        manifest.adapter
    );
    anyhow::ensure!(
        manifest.upstream == dataset.upstream,
        "the corpus was packed from {}, the catalog pins {}",
        manifest.upstream.reference(),
        dataset.upstream.reference()
    );
    anyhow::ensure!(
        index.media.len() == usize::try_from(manifest.media)?,
        "the corpus manifest counts {} media files, the tar holds {}",
        manifest.media,
        index.media.len()
    );
    let lines = index
        .samples
        .as_deref()
        .context("the corpus has no samples.jsonl")?
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(n, line)| {
            serde_json::from_slice::<VlmEvalCorpusSample>(line)
                .with_context(|| format!("samples.jsonl line {}", n + 1))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        lines.len() == usize::try_from(manifest.samples)?,
        "the corpus manifest counts {} samples, samples.jsonl holds {}",
        manifest.samples,
        lines.len()
    );
    let mut ids = HashSet::with_capacity(lines.len());
    if let Some(twice) = lines.iter().find(|line| !ids.insert(line.id.as_str())) {
        anyhow::bail!(
            "the corpus holds sample `{}` twice; every sample is answered once, by its id",
            twice.id
        );
    }

    let mut rebuilt = SamplesDigest::new();
    for line in &lines {
        let sample = line.to_eval_sample_in(CorpusFormat::AsrEvalCorpusV1, |path| {
            read_media(&mut tar, &index.media, path)
        })?;
        rebuilt.update(&sample)?;
        visit(&sample)?;
    }
    let rebuilt = rebuilt.finish()?;
    anyhow::ensure!(
        &rebuilt == pinned,
        "the corpus rebuilds samples hashing to {rebuilt}, the evals server's pin is {pinned}: \
         these are not the samples the scorer grades"
    );
    Ok(AsrEvalCorpus {
        tar: Mutex::new(tar),
        manifest,
        lines,
        media: index.media,
        digest,
        _lease: lease,
    })
}

/// A media file's bytes, read by its offset in the tar: a path the index does
/// not hold is not in the corpus.
fn read_media(
    tar: &mut File,
    media: &BTreeMap<String, MediaEntry>,
    path: &str,
) -> std::io::Result<Vec<u8>> {
    let entry = media
        .get(path)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "not in the corpus"))?;
    tar.seek(SeekFrom::Start(entry.offset))?;
    let mut bytes = Vec::new();
    Read::by_ref(tar).take(entry.size).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).ok() != Some(entry.size) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "the tar ends inside the media file",
        ));
    }
    Ok(bytes)
}

/// What the tar holds, by its headers: the two index files read whole, and
/// where each media file sits.
struct Index {
    manifest: Option<Vec<u8>>,
    samples: Option<Vec<u8>>,
    media: BTreeMap<String, MediaEntry>,
}

/// Read the tar's headers. The format is closed — `manifest.json`,
/// `samples.jsonl`, `media/*` — so anything else, or anything that is not a
/// plain file, means this is not the corpus the catalog pinned. A media
/// file's data is skipped (sought past), never read here.
fn index_entries(tar: &mut File) -> anyhow::Result<Index> {
    tar.seek(SeekFrom::Start(0))?;
    let mut archive = tar::Archive::new(BufReader::new(&*tar));
    let mut index = Index {
        manifest: None,
        samples: None,
        media: BTreeMap::new(),
    };
    let mut seen = HashSet::new();
    for entry in archive
        .entries_with_seek()
        .context("the corpus is not a readable tar")?
    {
        let mut entry = entry.context("failed to read a corpus tar entry")?;
        let name = entry
            .path()
            .context("a corpus tar entry has no path")?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path in the corpus tar"))?
            .to_owned();
        match entry.header().entry_type() {
            // The one directory the format has, which a packer may list.
            tar::EntryType::Directory if name == "media" || name == "media/" => continue,
            tar::EntryType::Directory => {
                anyhow::bail!("corpus tar holds an unexpected directory: {name:?}")
            }
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
        anyhow::ensure!(seen.insert(name.clone()), "corpus tar holds {name:?} twice");
        let size = entry.size();
        let slot = match name.as_str() {
            "manifest.json" => &mut index.manifest,
            "samples.jsonl" => &mut index.samples,
            media if media.starts_with("media/") => {
                index.media.insert(
                    name.clone(),
                    MediaEntry {
                        offset: entry.raw_file_position(),
                        size,
                    },
                );
                continue;
            }
            _ => anyhow::bail!("corpus tar holds an unexpected entry: {name:?}"),
        };
        anyhow::ensure!(
            size <= MAX_INDEX_BYTES,
            "the corpus's {name} is {size} bytes, past the {MAX_INDEX_BYTES} an index may hold"
        );
        let mut body = Vec::new();
        entry
            .read_to_end(&mut body)
            .with_context(|| format!("failed to read {name}"))?;
        *slot = Some(body);
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use base64::Engine as _;

    use pipette_ops::EvalCompletionsStore;

    use super::*;

    /// A mono 16 kHz 16-bit WAV of `samples` samples, each `value`.
    fn wav(samples: u32, value: u8) -> Vec<u8> {
        let data = samples * 2;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend((36 + data).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(16_000u32.to_le_bytes());
        bytes.extend(32_000u32.to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data.to_le_bytes());
        bytes.resize(bytes.len() + usize::try_from(data).unwrap_or(0), value);
        bytes
    }

    fn audio(bytes: &[u8]) -> serde_json::Value {
        serde_json::json!({
            "type": "input_audio",
            "input_audio": {
                "data": base64::engine::general_purpose::STANDARD.encode(bytes),
                "format": "wav",
            },
        })
    }

    /// What the evals server serves for a Voxtral profile: two clips, one
    /// played twice, each with its template arguments, and a system turn.
    fn served() -> anyhow::Result<Vec<EvalSample>> {
        let (a, b) = (wav(800, 1), wav(1600, 2));
        Ok(serde_json::from_value(serde_json::json!([
            {"id": "c0", "messages": [{"role": "user", "content": [audio(&a)]}],
             "chat_template_kwargs": {"transcription": true, "language": "nl"}},
            {"id": "c1", "messages": [
                {"role": "system", "content": "Perform ASR."},
                {"role": "user", "content": [{"type": "text", "text": "Transcribe."}, audio(&b)]}]},
            {"id": "c2", "messages": [{"role": "user", "content": [audio(&a)]}]}
        ]))?)
    }

    fn identity(samples: &[EvalSample]) -> anyhow::Result<Sha256> {
        let mut digest = SamplesDigest::new();
        samples.iter().try_for_each(|s| digest.update(s))?;
        Ok(digest.finish()?)
    }

    fn upstream(identity: &Sha256) -> anyhow::Result<DatasetUpstream> {
        Ok(serde_json::from_value(serde_json::json!({
            "source": "evals_server",
            "eval_id": "asr_fleurs_nl_nl",
            "dataset_version": "2026.10.1",
            "prompt_profile": "voxtral",
            "identity": identity.as_ref(),
        }))?)
    }

    /// The tar mgmt packs from `samples`, with `manifest_upstream` recorded,
    /// `extra` entries a server would never write after it.
    fn pack(
        samples: &[EvalSample],
        manifest_upstream: &DatasetUpstream,
        extra: &[(&str, &[u8])],
    ) -> anyhow::Result<Vec<u8>> {
        let mut media = BTreeMap::<String, Vec<u8>>::new();
        let mut lines = Vec::new();
        samples.iter().try_for_each(|s| {
            let line = VlmEvalCorpusSample::from_eval_sample_in(
                CorpusFormat::AsrEvalCorpusV1,
                s,
                |path, bytes| {
                    media.insert(path.to_owned(), bytes.to_vec());
                    Ok(())
                },
            )?;
            serde_json::to_writer(&mut lines, &line)?;
            lines.push(b'\n');
            anyhow::Ok(())
        })?;
        let manifest = serde_json::to_vec(&VlmEvalCorpusManifest {
            format: CorpusFormat::AsrEvalCorpusV1,
            adapter: DatasetAdapter::EvalsServerV1,
            upstream: manifest_upstream.clone(),
            samples: u32::try_from(samples.len())?,
            media: u32::try_from(media.len())?,
        })?;
        let mut builder = tar::Builder::new(Vec::new());
        let mut append = |path: &str, body: &[u8]| -> anyhow::Result<()> {
            let mut header = tar::Header::new_ustar();
            header.set_size(u64::try_from(body.len())?);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, path, body)?;
            Ok(())
        };
        append("manifest.json", &manifest)?;
        media
            .iter()
            .try_for_each(|(path, body)| append(path, body))?;
        append("samples.jsonl", &lines)?;
        extra
            .iter()
            .try_for_each(|(path, body)| append(path, body))?;
        Ok(builder.into_inner()?)
    }

    fn dataset(tar: &[u8], upstream: DatasetUpstream) -> anyhow::Result<Dataset> {
        Ok(serde_json::from_value(serde_json::json!({
            "format": "asr-eval-corpus/1",
            "adapter": "evals-server@1",
            "digest": hex::encode(Sha256Hasher::digest(tar)),
            "bytes": tar.len(),
            "upstream": serde_json::to_value(upstream)?,
            "url": "/datasets/asr_eval_fleurs_nl_nl_voxtral",
        }))?)
    }

    /// Where `needle` first sits in `haystack`: the manifest's format, or the
    /// first clip (media files come before `samples.jsonl`, and nothing
    /// before them spells `RIFF`).
    fn find(haystack: &[u8], needle: &[u8]) -> anyhow::Result<usize> {
        haystack
            .windows(needle.len())
            .position(|w| w == needle)
            .context("the bytes are in the tar")
    }

    /// A client that serves one tar, counting the downloads, under `root`.
    struct Serving {
        tar: Vec<u8>,
        root: PathBuf,
        fetched: RefCell<usize>,
    }

    impl Serving {
        fn new(tar: Vec<u8>) -> anyhow::Result<(Self, tempfile::TempDir)> {
            let dir = tempfile::tempdir()?;
            Ok((
                Self {
                    tar,
                    root: dir.path().to_path_buf(),
                    fetched: RefCell::new(0),
                },
                dir,
            ))
        }
    }

    impl RunContext for Serving {
        fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
            *self.fetched.borrow_mut() += 1;
            Ok(self.tar.clone())
        }

        fn datasets_dir(&self) -> PathBuf {
            self.root.join("datasets")
        }

        fn eval_completions(&self) -> EvalCompletionsStore {
            EvalCompletionsStore::new(self.root.join("evals"))
        }
    }

    /// The FLEURS Dutch Voxtral entry, pinning `dataset`.
    fn benchmark(dataset: Option<Dataset>) -> anyhow::Result<AsrEvalBenchmark> {
        let mut entry: AsrEvalBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "asr_eval_asr_fleurs_nl_nl_voxtral",
            "parameter_asr_task": "asr_fleurs_nl_nl",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 3,
            "parameter_prompt_profile": "voxtral",
            "parameter_max_tokens": 128,
        }))?;
        entry.dataset = dataset.map(Box::new);
        Ok(entry)
    }

    fn served_corpus() -> anyhow::Result<(Vec<EvalSample>, Vec<u8>, Dataset)> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let tar = pack(&served, &pinned, &[])?;
        let dataset = dataset(&tar, pinned)?;
        Ok((served, tar, dataset))
    }

    /// What the client downloads is what the entry pins: the samples come
    /// back as the evals server served them, the identity pass sees each once
    /// in order, the shared clip is one file, and the tar is the run's alone,
    /// gone once the corpus drops.
    #[test]
    fn a_download_rebuilds_the_served_samples() -> anyhow::Result<()> {
        let (served, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar.clone())?;
        let mut visited = Vec::new();
        let corpus = download(&ctx, &benchmark(Some(dataset))?, |sample| {
            visited.push(sample.clone());
            Ok(())
        })?;
        assert_eq!(visited, served);
        assert_eq!(corpus.samples(0..3)?, served);
        assert_eq!(corpus.samples(1..2)?, served.get(1..2).unwrap_or_default());
        assert_eq!(corpus.len(), 3);
        assert_eq!(corpus.id(2), Some("c2"));
        assert_eq!(
            corpus.manifest().media,
            2,
            "the clip played twice is one file"
        );
        assert_eq!(
            corpus.digest().as_ref(),
            hex::encode(Sha256Hasher::digest(&tar))
        );
        assert_eq!(*ctx.fetched.borrow(), 1);
        assert_eq!(
            corpus_files(&ctx)?.len(),
            1,
            "the corpus is on disk while the run holds it"
        );
        drop(corpus);
        assert!(
            corpus_files(&ctx)?.is_empty(),
            "and deleted once it is done"
        );
        Ok(())
    }

    /// The corpus files under the test entry's directory.
    fn corpus_files(ctx: &dyn RunContext) -> anyhow::Result<Vec<PathBuf>> {
        let dir = ctx.datasets_dir().join("asr_eval_asr_fleurs_nl_nl_voxtral");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(Vec::new());
        };
        let mut files = Vec::new();
        for entry in entries {
            let path = entry?.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(TAR_PREFIX) && name.ends_with(TAR_SUFFIX))
            {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// Every file under the test entry's directory, sidecars included.
    fn entry_files(ctx: &dyn RunContext) -> anyhow::Result<usize> {
        let dir = ctx.datasets_dir().join("asr_eval_asr_fleurs_nl_nl_voxtral");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Ok(0);
        };
        Ok(entries.count())
    }

    /// The run locks its tar's sidecar, never the tar: a lock on the tar
    /// itself is mandatory on Windows and refuses the run's own reads of it.
    #[test]
    fn a_run_locks_the_sidecar_not_its_corpus() -> anyhow::Result<()> {
        let (served, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar)?;
        let entry = benchmark(Some(dataset))?;
        let corpus = download(&ctx, &entry, |_| Ok(()))?;
        let files = corpus_files(&ctx)?;
        let [path] = files.as_slice() else {
            anyhow::bail!("one corpus file for one run");
        };
        File::open(path)?
            .try_lock()
            .map_err(|err| anyhow::anyhow!("the corpus itself must not be locked: {err}"))?;
        assert!(
            File::open(lock_path(path))?.try_lock().is_err(),
            "the run holds its sidecar"
        );
        assert_eq!(corpus.samples(0..3)?, served);
        drop(corpus);
        assert_eq!(entry_files(&ctx)?, 0, "the corpus and its sidecar are gone");
        Ok(())
    }

    /// A killed run's sidecar, its corpus gone or not, is swept with it.
    #[test]
    fn a_killed_runs_sidecar_is_swept() -> anyhow::Result<()> {
        let (_, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar)?;
        let entry = benchmark(Some(dataset))?;
        let dir = ctx.datasets_dir().join("asr_eval_asr_fleurs_nl_nl_voxtral");
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("asr-eval-corpus.1-0.tar"), b"left by a killed run")?;
        std::fs::write(dir.join("asr-eval-corpus.1-0.tar.lock"), b"")?;
        std::fs::write(dir.join("asr-eval-corpus.2-0.tar.lock"), b"")?;
        remove_leftover(&ctx, &entry)?;
        assert_eq!(entry_files(&ctx)?, 0, "nothing a killed run left survives");
        Ok(())
    }

    /// Two runs of one entry at once each write and read their own corpus:
    /// neither run's download, leftover sweep or end deletes the other's,
    /// while a file no live run holds (a killed run's) is deleted.
    #[test]
    fn two_runs_of_one_entry_keep_their_own_corpus() -> anyhow::Result<()> {
        let (served, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar)?;
        let entry = benchmark(Some(dataset))?;
        let first = download(&ctx, &entry, |_| Ok(()))?;
        let leftover = ctx
            .datasets_dir()
            .join("asr_eval_asr_fleurs_nl_nl_voxtral")
            .join("asr-eval-corpus.tar");
        std::fs::write(&leftover, b"left by a killed run")?;
        let second = download(&ctx, &entry, |_| Ok(()))?;
        assert!(!leftover.exists(), "a killed run's corpus is deleted");
        assert_eq!(corpus_files(&ctx)?.len(), 2, "one file per live run");
        remove_leftover(&ctx, &entry)?;
        assert_eq!(corpus_files(&ctx)?.len(), 2, "a live run's file is kept");
        drop(first);
        assert_eq!(second.samples(0..3)?, served, "the other run reads on");
        assert_eq!(corpus_files(&ctx)?.len(), 1);
        drop(second);
        assert!(corpus_files(&ctx)?.is_empty());
        assert_eq!(entry_files(&ctx)?, 0, "no sidecar is left either");
        Ok(())
    }

    /// A server that streams past the pinned size is cut off at that size:
    /// nothing beyond it is accepted, so the disk holds at most the corpus the
    /// room check made space for, and nothing is left behind.
    #[test]
    fn a_download_past_the_pinned_size_stops_there() -> anyhow::Result<()> {
        struct Overrunning {
            serving: Serving,
            accepted: RefCell<u64>,
        }
        impl RunContext for Overrunning {
            fn fetch_dataset(&self, dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
                self.serving.fetch_dataset(dataset)
            }
            fn fetch_dataset_to(&self, _: &Dataset, dest: &mut dyn Write) -> anyhow::Result<u64> {
                // The pinned tar, then a megabyte more, 4 KiB at a time.
                let tail = vec![0_u8; 1 << 20];
                for piece in self.serving.tar.chunks(4096).chain(tail.chunks(4096)) {
                    dest.write_all(piece)?;
                    *self.accepted.borrow_mut() += u64::try_from(piece.len())?;
                }
                Ok(*self.accepted.borrow())
            }
            fn datasets_dir(&self) -> PathBuf {
                self.serving.datasets_dir()
            }
            fn eval_completions(&self) -> EvalCompletionsStore {
                self.serving.eval_completions()
            }
        }

        let (_, tar, dataset) = served_corpus()?;
        let (serving, _dir) = Serving::new(tar)?;
        let ctx = Overrunning {
            serving,
            accepted: RefCell::new(0),
        };
        let error = download(&ctx, &benchmark(Some(dataset.clone()))?, |_| Ok(()))
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("more than the"), "{error}");
        assert!(
            *ctx.accepted.borrow() <= dataset.bytes,
            "accepted {} bytes of a {}-byte pin",
            ctx.accepted.borrow(),
            dataset.bytes
        );
        assert!(
            corpus_files(&ctx)?.is_empty(),
            "the partial file is deleted"
        );
        Ok(())
    }

    /// A chunk stops at its sample count or its clip bytes, whichever comes
    /// first, and always carries one sample.
    #[test]
    fn a_chunk_is_bounded_by_samples_and_clip_bytes() -> anyhow::Result<()> {
        let (served, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar)?;
        let corpus = download(&ctx, &benchmark(Some(dataset))?, |_| Ok(()))?;
        let ids = |chunk: Vec<EvalSample>| chunk.into_iter().map(|s| s.id).collect::<Vec<_>>();
        assert_eq!(ids(corpus.chunk(0, 2, u64::MAX)?), ["c0", "c1"]);
        assert_eq!(ids(corpus.chunk(1, 8, u64::MAX)?), ["c1", "c2"]);
        // c0's clip is 1,644 bytes and c1's 3,244: a budget of 2,000 holds one.
        assert_eq!(ids(corpus.chunk(0, 8, 2_000)?), ["c0"]);
        assert_eq!(
            ids(corpus.chunk(1, 8, 1)?),
            ["c1"],
            "one past the budget, alone"
        );
        assert!(corpus.chunk(3, 8, u64::MAX)?.is_empty());
        assert_eq!(corpus.chunk(0, 8, u64::MAX)?, served);
        Ok(())
    }

    /// Asking past the end is an error, not a short read.
    #[test]
    fn samples_past_the_corpus_are_refused() -> anyhow::Result<()> {
        let (_, tar, dataset) = served_corpus()?;
        let (ctx, _dir) = Serving::new(tar)?;
        let corpus = download(&ctx, &benchmark(Some(dataset))?, |_| Ok(()))?;
        assert!(corpus.samples(2..4).is_err());
        Ok(())
    }

    /// Each way a corpus can fail to be the pinned samples is a refusal, and
    /// none hands a sample to the run: the identity is checked before the
    /// corpus is returned.
    #[rstest::rstest]
    #[case::tar_digest("tar", "the catalog pins")]
    #[case::size("size", "the catalog pins a corpus of")]
    #[case::other_samples("samples", "not the samples the scorer grades")]
    #[case::other_upstream("upstream", "was packed from")]
    #[case::stray_entry("stray", "unexpected entry")]
    #[case::vl_corpus("vl", "names format vlm-eval-corpus/1")]
    #[case::clip_changed("clip", "does not hash to its name")]
    fn a_corpus_that_is_not_the_pin_is_refused(
        #[case] fault: &str,
        #[case] names: &str,
    ) -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let (tar, dataset) = match fault {
            "tar" => {
                let tar = pack(&served, &pinned, &[])?;
                let mut dataset = dataset(&tar, pinned)?;
                dataset.digest = Sha256::try_new("1".repeat(64))?;
                (tar, dataset)
            }
            "size" => {
                let tar = pack(&served, &pinned, &[])?;
                let mut dataset = dataset(&tar, pinned)?;
                dataset.bytes += 1;
                (tar, dataset)
            }
            "samples" => {
                // Packed from the samples in another order, claiming the served.
                let mut reordered = served.clone();
                reordered.swap(0, 1);
                let tar = pack(&reordered, &pinned, &[])?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            "upstream" => {
                let other = upstream(&Sha256::try_new("2".repeat(64))?)?;
                let tar = pack(&served, &other, &[])?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            "vl" => {
                // The manifest of the other eval format, under an ASR pin.
                let mut tar = pack(&served, &pinned, &[])?;
                let at = find(&tar, b"asr-eval-corpus/1")?;
                if let Some(format) = tar.get_mut(at..at + 3) {
                    format.copy_from_slice(b"vlm");
                }
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            "clip" => {
                // One byte of a clip flipped after packing, the tar re-pinned.
                let mut tar = pack(&served, &pinned, &[])?;
                let at = find(&tar, b"RIFF")? + 50;
                if let Some(byte) = tar.get_mut(at) {
                    *byte ^= 0xff;
                }
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            _ => {
                let tar = pack(&served, &pinned, &[("notes.txt", b"x")])?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
        };
        let (ctx, _dir) = Serving::new(tar)?;
        let mut visited = 0;
        let error = download(&ctx, &benchmark(Some(dataset))?, |_| {
            visited += 1;
            Ok(())
        })
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(error.contains(names), "{error}");
        assert!(
            error.contains("`asr_eval_asr_fleurs_nl_nl_voxtral`"),
            "the refusal names the benchmark: {error}"
        );
        assert!(
            corpus_files(&ctx)?.is_empty(),
            "a refused corpus is not left on disk"
        );
        if fault != "samples" && fault != "clip" {
            assert_eq!(visited, 0, "nothing reaches the run before the pins hold");
        }
        Ok(())
    }

    /// The control: the same corpus, unchanged, is accepted.
    #[test]
    fn the_unchanged_corpus_is_accepted() -> anyhow::Result<()> {
        let (served, tar, dataset) = served_corpus()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("corpus.tar");
        std::fs::write(&path, &tar)?;
        let corpus = open(&path, &dataset, |_| Ok(()))?;
        assert_eq!(corpus.samples(0..3)?, served);
        drop(corpus);
        assert!(path.exists(), "a file the caller opened stays the caller's");
        Ok(())
    }

    /// The pin must name the samples the entry is graded on (its task, its
    /// version and its prompt profile), in the one format this reads;
    /// refused before anything is downloaded.
    #[rstest::rstest]
    #[case::other_task(
        "eval_id",
        Some("asr_fleurs_de_de"),
        "pins the samples of evals-server:asr_fleurs_de_de"
    )]
    #[case::other_version("dataset_version", Some("2026.07.1"), "/2026.07.1")]
    #[case::other_profile("prompt_profile", Some("gemma4"), "?prompt_profile=gemma4")]
    #[case::no_profile(
        "prompt_profile",
        None,
        "but runs asr_fleurs_nl_nl/2026.10.1?prompt_profile=voxtral"
    )]
    #[case::vl_format("format", Some("vlm-eval-corpus/1"), "reads asr-eval-corpus/1")]
    #[case::other_adapter("adapter", Some("mmspec@1"), "from mmspec@1")]
    fn a_pin_for_other_samples_is_refused(
        #[case] field: &str,
        #[case] value: Option<&str>,
        #[case] names: &str,
    ) -> anyhow::Result<()> {
        let (_, tar, dataset) = served_corpus()?;
        let mut pin = serde_json::to_value(dataset)?;
        let target = match field {
            "format" | "adapter" => pin.as_object_mut(),
            _ => pin.get_mut("upstream").and_then(|u| u.as_object_mut()),
        }
        .context("the pin is an object")?;
        match value {
            Some(value) => target.insert(field.to_owned(), serde_json::json!(value)),
            None => target.remove(field),
        };
        let (ctx, _dir) = Serving::new(tar)?;
        let error = download(
            &ctx,
            &benchmark(Some(serde_json::from_value(pin)?))?,
            |_| Ok(()),
        )
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(error.contains(names), "{error}");
        assert_eq!(*ctx.fetched.borrow(), 0, "nothing is downloaded");
        Ok(())
    }

    /// An entry without a pin has no samples to run.
    #[test]
    fn an_entry_without_a_corpus_is_refused() -> anyhow::Result<()> {
        let (ctx, _dir) = Serving::new(Vec::new())?;
        let error = download(&ctx, &benchmark(None)?, |_| Ok(()))
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("pins no corpus"), "{error}");
        assert_eq!(*ctx.fetched.borrow(), 0);
        Ok(())
    }

    /// A disk short of the corpus is refused before a byte is fetched; one
    /// with room is not.
    #[test]
    fn a_short_disk_is_refused_before_the_download() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        if available_bytes(dir.path()).is_none() {
            return Ok(());
        }
        let error = check_room(dir.path(), u64::MAX / 2)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("Free disk space"), "{error}");
        check_room(dir.path(), 1)?;
        Ok(())
    }

    /// A sample listed twice would collapse two answers onto one id.
    #[test]
    fn a_sample_listed_twice_is_refused() -> anyhow::Result<()> {
        let mut served = served()?;
        let first = served.first().cloned().context("a sample")?;
        served.push(first);
        let pinned = upstream(&identity(&served)?)?;
        let tar = pack(&served, &pinned, &[])?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("corpus.tar");
        std::fs::write(&path, &tar)?;
        let error = open(&path, &dataset(&tar, pinned)?, |_| Ok(()))
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("holds sample `c0` twice"), "{error}");
        Ok(())
    }

    /// A clip the identity pass read is checked again when the run reads it:
    /// a file changed on disk after the checks is refused, not sent.
    #[test]
    fn a_clip_changed_after_the_checks_is_refused() -> anyhow::Result<()> {
        let (_, tar, dataset) = served_corpus()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("corpus.tar");
        std::fs::write(&path, &tar)?;
        let corpus = open(&path, &dataset, |_| Ok(()))?;
        let mut changed = tar.clone();
        let at = find(&changed, b"RIFF")? + 50;
        if let Some(byte) = changed.get_mut(at) {
            *byte ^= 0xff;
        }
        std::fs::write(&path, &changed)?;
        let error = corpus
            .samples(0..1)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("does not hash to its name"), "{error}");
        Ok(())
    }
}
