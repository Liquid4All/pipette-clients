//! Reader for `vlm-eval-corpus/1`, a `vlm_eval`'s prompts as the management
//! server packs them (each image once under `media/`), rebuilt into the exact
//! samples the evals server serves and grades.
//!
//! **A mismatch fails the run**, unlike `vl_spec_decode`, which measures an
//! unexpected corpus anyway: answers are graded against the evals server's own
//! prompts, so different ones would score questions the model was never asked.
//! Two digests guard that: the tar's, as the catalog pins it, and the rebuilt
//! samples', as the evals server reports it for the pinned eval, dataset
//! version and prompt profile.
//!
//! Nothing is unpacked to disk: a media path in `samples.jsonl` is only a key
//! into the tar's in-memory entries (`media/<sha256>.<ext>`, checked against
//! the bytes it names).

use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

use anyhow::Context;
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    CorpusFormat, Dataset, DatasetAdapter, DatasetUpstream, EvalSample, KnownVlmPromptProfile,
    Model, ModelFlags, SamplesDigest, Sha256, VlmEvalBenchmark, VlmEvalCorpusManifest,
    VlmEvalCorpusSample, VlmPromptProfile,
};

#[derive(Debug)]
pub struct VlmEvalCorpus {
    manifest: VlmEvalCorpusManifest,
    samples: Vec<EvalSample>,
    digest: Sha256,
}

impl VlmEvalCorpus {
    pub fn manifest(&self) -> &VlmEvalCorpusManifest {
        &self.manifest
    }

    /// In the order the evals server serves them.
    pub fn samples(&self) -> &[EvalSample] {
        &self.samples
    }

    /// The tar's sha256, which matched the catalog's pin; submitted as
    /// `observation_dataset_digest`.
    pub fn digest(&self) -> &Sha256 {
        &self.digest
    }
}

/// Download and rebuild the corpus `benchmark` pins; `None` for a local
/// definition, which pins none.
pub fn download(
    run_context: &dyn RunContext,
    benchmark: &VlmEvalBenchmark,
) -> anyhow::Result<Option<VlmEvalCorpus>> {
    let Some(dataset) = &benchmark.dataset else {
        return Ok(None);
    };
    let benchmark_id = &benchmark.benchmark_id;
    let started = Instant::now();
    check_pin(benchmark, dataset)?;
    let tar = run_context
        .fetch_dataset(dataset)
        .with_context(|| format!("failed to download the corpus of `{benchmark_id}`"))?;
    let corpus = open(&tar, dataset)
        .with_context(|| format!("the corpus of `{benchmark_id}` is not the one it pins"))?;
    tracing::info!(
        %benchmark_id,
        elapsed = ?started.elapsed(),
        bytes = tar.len(),
        samples = corpus.samples.len(),
        media = corpus.manifest.media,
        digest = %corpus.digest,
        "vlm_eval corpus ready"
    );
    Ok(Some(corpus))
}

/// A corpus is graded as the entry's task, version and profile, so it must be
/// those samples. The catalog refuses other pins; checked again before the
/// download for a definition that reached the client some other way.
fn check_pin(benchmark: &VlmEvalBenchmark, dataset: &Dataset) -> anyhow::Result<()> {
    let benchmark_id = &benchmark.benchmark_id;
    anyhow::ensure!(
        dataset.format == CorpusFormat::VlmEvalCorpusV1
            && dataset.adapter == DatasetAdapter::EvalsServerV1,
        "vlm_eval `{benchmark_id}` pins a {} corpus from {}; it reads vlm-eval-corpus/1 \
         from evals-server@1",
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
            "vlm_eval `{benchmark_id}` pins {}, not the evals server's samples",
            dataset.upstream.reference()
        );
    };
    let pinned = (
        eval_id.as_ref(),
        dataset_version.as_ref(),
        prompt_profile.as_ref().map(AsRef::as_ref),
    );
    let runs = (
        benchmark.parameter_lmms_task.as_str(),
        benchmark.parameter_dataset_version.as_str(),
        benchmark
            .parameter_prompt_profile
            .as_ref()
            .map(VlmPromptProfile::as_str),
    );
    anyhow::ensure!(
        pinned == runs,
        "vlm_eval `{benchmark_id}` pins the samples of {}, but runs {}/{}{}: they are not the \
         prompts the scorer grades it on",
        dataset.upstream.reference(),
        runs.0,
        runs.1,
        runs.2
            .map_or(String::new(), |p| format!("?prompt_profile={p}"))
    );
    Ok(())
}

/// Verify `tar` against `dataset` and rebuild its samples.
pub fn open(tar: &[u8], dataset: &Dataset) -> anyhow::Result<VlmEvalCorpus> {
    let digest = Sha256::try_new(hex::encode(Sha256Hasher::digest(tar)))
        .map_err(|e| anyhow::anyhow!("hashing the corpus produced an unusable digest: {e}"))?;
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
    let entries = read_entries(tar)?;
    let manifest: VlmEvalCorpusManifest = serde_json::from_slice(
        entries
            .get("manifest.json")
            .context("the corpus has no manifest.json")?,
    )
    .context("failed to parse manifest.json")?;
    anyhow::ensure!(
        manifest.format == CorpusFormat::VlmEvalCorpusV1
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
    let index = entries
        .get("samples.jsonl")
        .context("the corpus has no samples.jsonl")?;
    let media_entries = entries
        .keys()
        .filter(|name| name.starts_with("media/"))
        .count();
    anyhow::ensure!(
        media_entries == manifest.media as usize,
        "the corpus manifest counts {} media files, the tar holds {media_entries}",
        manifest.media
    );

    let mut rebuilt = SamplesDigest::new();
    let samples = index
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(n, line)| {
            let corpus: VlmEvalCorpusSample = serde_json::from_slice(line)
                .with_context(|| format!("samples.jsonl line {}", n + 1))?;
            let sample = corpus.to_eval_sample(|path| {
                entries.get(path).cloned().ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, "not in the corpus")
                })
            })?;
            rebuilt.update(&sample)?;
            Ok(sample)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(
        samples.len() == manifest.samples as usize,
        "the corpus manifest counts {} samples, samples.jsonl holds {}",
        manifest.samples,
        samples.len()
    );
    let rebuilt = rebuilt.finish()?;
    anyhow::ensure!(
        &rebuilt == pinned,
        "the corpus rebuilds samples hashing to {rebuilt}, the evals server's pin is {pinned}: \
         these are not the prompts the scorer grades"
    );
    Ok(VlmEvalCorpus {
        manifest,
        samples,
        digest,
    })
}

/// The format is closed (`manifest.json`, `samples.jsonl`, `media/*`), so any
/// other entry, or one that is not a plain file, means this is not the corpus
/// the catalog pinned.
fn read_entries(tar: &[u8]) -> anyhow::Result<HashMap<String, Vec<u8>>> {
    let mut archive = tar::Archive::new(std::io::Cursor::new(tar));
    let mut entries = HashMap::new();
    archive
        .entries()
        .context("the corpus is not a readable tar")?
        .try_for_each(|entry| {
            let mut entry = entry.context("failed to read a corpus tar entry")?;
            let name = entry
                .path()
                .context("a corpus tar entry has no path")?
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 path in the corpus tar"))?
                .to_owned();
            match entry.header().entry_type() {
                // The one directory the format has, which a packer may list.
                tar::EntryType::Directory if name == "media" || name == "media/" => {
                    return Ok(());
                }
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
            anyhow::ensure!(
                name == "manifest.json" || name == "samples.jsonl" || name.starts_with("media/"),
                "corpus tar holds an unexpected entry: {name:?}"
            );
            let mut body = Vec::new();
            entry
                .read_to_end(&mut body)
                .with_context(|| format!("failed to read {name}"))?;
            anyhow::ensure!(
                entries.insert(name.clone(), body).is_none(),
                "corpus tar holds {name:?} twice"
            );
            Ok(())
        })?;
    Ok(entries)
}

// ── The vlm_eval cell's rules, shared by every engine ──────────────────────

/// A catalog entry (one that pins a corpus) must name its recipe and runs only
/// the models it covers, in its thinking mode. mgmt refuses such a cell before
/// queueing it; this is the same rule for a run nothing queued.
pub fn on_recipe<'a>(
    benchmark: &'a VlmEvalBenchmark,
    model: &Model,
    model_flags: Option<&ModelFlags>,
) -> anyhow::Result<&'a VlmEvalBenchmark> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    match (&benchmark.recipe, &benchmark.dataset) {
        (Some(recipe), _) => recipe
            .check(model, model_flags.and_then(ModelFlags::enable_thinking))
            .with_context(|| format!("vlm_eval `{benchmark_id}` is off its recipe"))?,
        (None, Some(_)) => anyhow::bail!(
            "vlm_eval `{benchmark_id}` pins a corpus but names no recipe; a catalog entry says \
             whose settings it holds"
        ),
        (None, None) => {}
    }
    Ok(benchmark)
}

/// A catalog entry's prompts are its corpus, a local definition's are inline.
/// Never both: it would be unclear which the scorer grades.
pub fn samples_of<'a>(
    benchmark: &'a VlmEvalBenchmark,
    corpus: Option<&'a VlmEvalCorpus>,
) -> anyhow::Result<&'a [EvalSample]> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    match (
        benchmark.dataset.is_some(),
        corpus,
        benchmark.samples.as_deref(),
    ) {
        (true, _, Some(_)) => anyhow::bail!(
            "vlm_eval `{benchmark_id}` carries both a dataset and inline samples; a catalog \
             entry's prompts are its dataset alone"
        ),
        (true, Some(corpus), None) => Ok(corpus.samples()),
        (true, None, None) => anyhow::bail!(
            "vlm_eval `{benchmark_id}` pins a dataset that was not downloaded before the run"
        ),
        (false, None, Some(samples)) => Ok(samples),
        (false, Some(_), _) => {
            anyhow::bail!("vlm_eval `{benchmark_id}` was handed a corpus it does not pin")
        }
        (false, None, None) => anyhow::bail!(
            "vlm_eval `{benchmark_id}` has neither a dataset to download nor inline samples"
        ),
    }
}

/// The served samples, if they are the dataset the definition declares: as
/// many as it says, each id once. Every engine checks this before it runs
/// anything.
pub fn check_samples(benchmark: &VlmEvalBenchmark, samples: &[EvalSample]) -> anyhow::Result<()> {
    let benchmark_id = benchmark.benchmark_id.as_str();
    // Any other length would submit an incomparable result under the shared
    // benchmark id.
    anyhow::ensure!(
        samples.len() == benchmark.parameter_sample_count.into_inner().unsigned_abs() as usize,
        "vlm_eval `{benchmark_id}` declares parameter_sample_count={} but was served {} samples",
        benchmark.parameter_sample_count,
        samples.len(),
    );
    // The resume checkpoint is keyed by id, so a duplicate would run once and
    // the run would submit fewer answers than declared.
    let mut seen = std::collections::HashSet::with_capacity(samples.len());
    if let Some(id) = samples
        .iter()
        .map(|sample| sample.id.as_str())
        .find(|id| !seen.insert(*id))
    {
        anyhow::bail!(
            "vlm_eval `{benchmark_id}` was served a duplicate sample id `{id}`; \
             the resume checkpoint would silently drop every later occurrence",
        );
    }
    Ok(())
}

/// InternVL3.5 thinks under its R1 system prompt (`internvl_r1`), not a
/// template switch, so such a cell must declare `enable_thinking = true` for
/// its rows to say it thought.
pub fn prompt_thinks(
    benchmark: &VlmEvalBenchmark,
    enable_thinking: Option<bool>,
) -> anyhow::Result<bool> {
    let thinks = benchmark.parameter_prompt_profile
        == Some(VlmPromptProfile::Known(KnownVlmPromptProfile::InternvlR1));
    anyhow::ensure!(
        !thinks || enable_thinking == Some(true),
        "vlm_eval `{}`: the internvl_r1 prompt profile is InternVL3.5's thinking mode, so the \
         cell must declare it: set its model flags' enable_thinking = true",
        benchmark.benchmark_id
    );
    Ok(thinks)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use base64::Engine as _;

    use pipette_ops::EvalCompletionsStore;

    use super::*;

    fn data_uri(bytes: &[u8]) -> String {
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// Two questions sharing a picture, and a text-only one with a system prompt.
    fn served() -> anyhow::Result<Vec<EvalSample>> {
        let picture = data_uri(&[5_u8; 700]);
        Ok(serde_json::from_value(serde_json::json!([
            {"id": "a", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": picture}},
                {"type": "text", "text": "Which mood?"}]}]},
            {"id": "b", "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": picture}},
                {"type": "text", "text": "How many?"}]}]},
            {"id": "c", "messages": [
                {"role": "system", "content": "Think first."},
                {"role": "user", "content": "No picture?"}]}
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
            "eval_id": "mmstar",
            "dataset_version": "2026.10.1",
            "identity": identity.as_ref(),
        }))?)
    }

    /// The tar the management server packs from `samples`.
    fn pack(
        samples: &[EvalSample],
        manifest_upstream: &DatasetUpstream,
    ) -> anyhow::Result<Vec<u8>> {
        pack_with(samples, manifest_upstream, &[])
    }

    fn pack_with(
        samples: &[EvalSample],
        manifest_upstream: &DatasetUpstream,
        extra: &[(&str, &[u8])],
    ) -> anyhow::Result<Vec<u8>> {
        let mut media = BTreeMap::<String, Vec<u8>>::new();
        let mut lines = Vec::new();
        samples.iter().try_for_each(|s| {
            let corpus = VlmEvalCorpusSample::from_eval_sample(s, |path, bytes| {
                media.insert(path.to_owned(), bytes.to_vec());
                Ok(())
            })?;
            serde_json::to_writer(&mut lines, &corpus)?;
            lines.push(b'\n');
            anyhow::Ok(())
        })?;
        let manifest = serde_json::to_vec(&VlmEvalCorpusManifest {
            format: CorpusFormat::VlmEvalCorpusV1,
            adapter: pipette_plan_types::DatasetAdapter::EvalsServerV1,
            upstream: manifest_upstream.clone(),
            samples: u32::try_from(samples.len())?,
            media: u32::try_from(media.len())?,
        })?;
        let mut builder = tar::Builder::new(Vec::new());
        let mut append = |path: &str, body: &[u8]| -> anyhow::Result<()> {
            let mut header = tar::Header::new_ustar();
            header.set_size(body.len() as u64);
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
            "format": "vlm-eval-corpus/1",
            "adapter": "evals-server@1",
            "digest": hex::encode(Sha256Hasher::digest(tar)),
            "bytes": tar.len(),
            "upstream": serde_json::to_value(upstream)?,
        }))?)
    }

    #[test]
    fn a_corpus_rebuilds_the_served_samples() -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let tar = pack(&served, &pinned)?;
        let corpus = open(&tar, &dataset(&tar, pinned)?)?;
        assert_eq!(corpus.samples(), served.as_slice());
        assert_eq!(corpus.manifest().media, 1, "the shared picture is one file");
        assert_eq!(
            corpus.digest().as_ref(),
            hex::encode(Sha256Hasher::digest(&tar))
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::tar_digest("tar", "the catalog pins")]
    #[case::other_samples("samples", "not the prompts the scorer grades")]
    #[case::other_upstream("upstream", "was packed from")]
    #[case::stray_entry("stray", "unexpected entry")]
    fn a_corpus_that_is_not_the_pin_is_refused(
        #[case] fault: &str,
        #[case] names: &str,
    ) -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let (tar, dataset) = match fault {
            "tar" => {
                let tar = pack(&served, &pinned)?;
                let mut dataset = dataset(&tar, pinned)?;
                dataset.digest = Sha256::try_new("1".repeat(64))?;
                (tar, dataset)
            }
            "samples" => {
                // The same prompts reordered no longer hash to the pin.
                let mut reworded = served.clone();
                reworded.swap(0, 1);
                let tar = pack(&reworded, &pinned)?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            "upstream" => {
                let other = upstream(&Sha256::try_new("2".repeat(64))?)?;
                let tar = pack(&served, &other)?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
            _ => {
                let tar = pack_with(&served, &pinned, &[("notes.txt", b"x")])?;
                let dataset = dataset(&tar, pinned)?;
                (tar, dataset)
            }
        };
        let error = open(&tar, &dataset)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains(names), "{error}");
        Ok(())
    }

    struct Serving(Vec<u8>);

    impl RunContext for Serving {
        fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
            Ok(self.0.clone())
        }

        fn datasets_dir(&self) -> PathBuf {
            std::env::temp_dir()
        }

        fn eval_completions(&self) -> EvalCompletionsStore {
            EvalCompletionsStore::new(std::env::temp_dir().join("evals"))
        }
    }

    fn benchmark(dataset: Dataset) -> anyhow::Result<VlmEvalBenchmark> {
        Ok(serde_json::from_value(serde_json::json!({
            "benchmark_id": "vlm_eval_mmstar",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 3,
            "parameter_max_tokens": 64,
            "dataset": serde_json::to_value(dataset)?,
        }))?)
    }

    #[test]
    fn a_download_rebuilds_the_entrys_corpus() -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let tar = pack(&served, &pinned)?;
        let entry = benchmark(dataset(&tar, pinned)?)?;
        let corpus = download(&Serving(tar), &entry)?
            .ok_or_else(|| anyhow::anyhow!("the entry pins a corpus"))?;
        assert_eq!(corpus.samples(), served.as_slice());
        Ok(())
    }

    #[test]
    fn a_local_definition_downloads_nothing() -> anyhow::Result<()> {
        let mut entry = benchmark(dataset(b"", upstream(&Sha256::try_new("1".repeat(64))?)?)?)?;
        entry.dataset = None;
        assert!(download(&Serving(Vec::new()), &entry)?.is_none());
        Ok(())
    }

    /// Refused before anything is downloaded, and so is a pin from another adapter.
    #[rstest::rstest]
    #[case::other_task("eval_id", "pope", "pins the samples of evals-server:pope")]
    #[case::other_version(
        "dataset_version",
        "2026.07.1",
        "pins the samples of evals-server:mmstar/2026.07.1"
    )]
    #[case::other_profile("prompt_profile", "qwen3_5", "?prompt_profile=qwen3_5")]
    #[case::other_adapter("adapter", "mmspec@1", "from mmspec@1")]
    fn a_pin_for_other_samples_is_refused(
        #[case] field: &str,
        #[case] value: &str,
        #[case] names: &str,
    ) -> anyhow::Result<()> {
        let served = served()?;
        let tar = pack(&served, &upstream(&identity(&served)?)?)?;
        let mut pin = serde_json::to_value(dataset(&tar, upstream(&identity(&served)?)?)?)?;
        if field == "adapter" {
            pin["adapter"] = serde_json::json!(value);
        } else {
            pin["upstream"][field] = serde_json::json!(value);
        }
        let entry = benchmark(serde_json::from_value(pin)?)?;
        let error = download(&Refusing, &entry)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains(names), "{error}");
        assert!(!error.contains("never fetched"), "{error}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::media("media/", true)]
    #[case::other("notes/", false)]
    fn only_the_media_directory_may_be_listed(
        #[case] directory: &str,
        #[case] ok: bool,
    ) -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_ustar();
        header.set_entry_type(tar::EntryType::Directory);
        header.set_size(0);
        header.set_mode(0o755);
        header.set_cksum();
        builder.append_data(&mut header, directory, &b""[..])?;
        let mut tar = builder.into_inner()?;
        // Drop the builder's two end-of-archive blocks so the corpus follows.
        tar.truncate(tar.len() - 1024);
        tar.extend(pack(&served, &pinned)?);
        assert_eq!(open(&tar, &dataset(&tar, pinned)?).is_ok(), ok);
        Ok(())
    }

    struct Refusing;

    impl RunContext for Refusing {
        fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("never fetched")
        }

        fn datasets_dir(&self) -> PathBuf {
            std::env::temp_dir()
        }

        fn eval_completions(&self) -> EvalCompletionsStore {
            EvalCompletionsStore::new(std::env::temp_dir().join("evals"))
        }
    }

    #[test]
    fn a_refused_corpus_names_the_benchmark() -> anyhow::Result<()> {
        let served = served()?;
        let pinned = upstream(&identity(&served)?)?;
        let tar = pack(&served, &pinned)?;
        let mut dataset = dataset(&tar, pinned)?;
        dataset.digest = Sha256::try_new("1".repeat(64))?;
        let error = download(&Serving(tar), &benchmark(dataset)?)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("`vlm_eval_mmstar`"), "{error}");
        Ok(())
    }
}
