//! A catalog vlm_eval downloads its `vlm-eval-corpus/1` tar once, rebuilds
//! each sample's `data:` images, sends exactly the evals server's prompts, and
//! reports the tar's digest. Off-recipe cells never download it.

mod common;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_readiness_gate,
};
use serde_json::json;
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_ops::EvalCompletionsStore;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkResultData, CorpusFormat, Dataset, DatasetAdapter, EnvForward,
    EvalSample, Model, ModelFlags, RuntimeFlags, SamplesDigest, VlmEvalBenchmark,
    VlmEvalCorpusManifest, VlmEvalCorpusSample,
};

/// A valid 1×1 PNG, so calibration reads a real image header.
const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn served() -> anyhow::Result<Vec<EvalSample>> {
    Ok(serde_json::from_value(json!([
        {"id": "q1", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": PNG}},
            {"type": "text", "text": "Which mood?"}]}]},
        {"id": "q2", "messages": [
            {"role": "system", "content": "Answer with a letter."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": PNG}},
                {"type": "text", "text": "How many?"}]}]},
        {"id": "q3", "messages": [{"role": "user", "content": "No picture?"}]}
    ]))?)
}

/// The pin and tar mgmt would build from `samples`.
fn corpus(samples: &[EvalSample]) -> anyhow::Result<(Dataset, Vec<u8>)> {
    let mut digest = SamplesDigest::new();
    samples.iter().try_for_each(|s| digest.update(s))?;
    let upstream: pipette_plan_types::DatasetUpstream = serde_json::from_value(json!({
        "source": "evals_server",
        "eval_id": "mmstar",
        "dataset_version": "2026.10.1",
        "identity": digest.finish()?.as_ref(),
    }))?;
    let mut media = BTreeMap::<String, Vec<u8>>::new();
    let mut lines = Vec::new();
    samples.iter().try_for_each(|s| {
        let line = VlmEvalCorpusSample::from_eval_sample(s, |path, bytes| {
            media.insert(path.to_owned(), bytes.to_vec());
            Ok(())
        })?;
        serde_json::to_writer(&mut lines, &line)?;
        lines.push(b'\n');
        anyhow::Ok(())
    })?;
    let manifest = serde_json::to_vec(&VlmEvalCorpusManifest {
        format: CorpusFormat::VlmEvalCorpusV1,
        adapter: DatasetAdapter::EvalsServerV1,
        upstream: upstream.clone(),
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
    let tar = builder.into_inner()?;
    let dataset = serde_json::from_value(json!({
        "format": "vlm-eval-corpus/1",
        "adapter": "evals-server@1",
        "digest": hex::encode(Sha256Hasher::digest(&tar)),
        "bytes": tar.len(),
        "upstream": serde_json::to_value(&upstream)?,
        "url": "/datasets/vlm_eval_mmstar",
    }))?;
    Ok((dataset, tar))
}

struct Serving {
    tar: Vec<u8>,
    root: PathBuf,
    fetched: RefCell<usize>,
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

fn entry(
    dataset: Option<Dataset>,
    samples: Option<Vec<EvalSample>>,
) -> anyhow::Result<BenchmarkDefinition> {
    let mut benchmark: VlmEvalBenchmark = serde_json::from_value(json!({
        "benchmark_id": "vlm_eval_mmstar",
        "parameter_lmms_task": "mmstar",
        "parameter_dataset_version": "2026.10.1",
        "parameter_sample_count": 3,
        "parameter_max_tokens": 16,
        "recipe": {
            "family": "Qwen3.5",
            "models": [{"repo": "unsloth/Qwen3.5-2B-GGUF"}],
            "sources": ["https://huggingface.co/Qwen/Qwen3.5-2B"],
        },
    }))?;
    benchmark.dataset = dataset.map(Box::new);
    benchmark.samples = samples;
    Ok(BenchmarkDefinition::VlmEval(benchmark))
}

/// The model the plan named, which the recipe check reads; the run loads the
/// bound files.
fn declared(repo: &str) -> anyhow::Result<Model> {
    let (org, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("org/repo"))?;
    Ok(serde_json::from_value(json!({
        "type": "gguf_vision", "source": "huggingface", "org": org, "repo_name": name,
        "model": "Qwen3.5-2B-Q4_K_M.gguf", "mmproj": "mmproj-F16.gguf",
    }))?)
}

fn model_files(root: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let model = root.join("fake-model.gguf");
    std::fs::write(&model, b"fake-model-bytes")?;
    let mmproj = root.join("fake-mmproj.gguf");
    std::fs::write(&mmproj, b"fake-mmproj-bytes")?;
    Ok((model, mmproj))
}

#[test]
fn a_catalog_vlm_eval_runs_the_samples_its_corpus_holds() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let bodies = root.path().join("chat-bodies.jsonl");

    let served = served()?;
    let (dataset, tar) = corpus(&served)?;
    let pinned_digest = dataset.digest.clone();
    let client = Serving {
        tar,
        root: root.path().to_path_buf(),
        fetched: RefCell::new(0),
    };
    let (model_path, mmproj_path) = model_files(root.path())?;
    let mut req = fake_run_request(
        fake_runtime(&fake_runtime_install_dir(root.path())?)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        entry(Some(dataset), None)?,
        None,
    );
    req.model.declared = declared("unsloth/Qwen3.5-2B-GGUF")?;
    // Through the cell's envs, so the fake logs this run's requests alone.
    let mut flags = req.runtime_flags_ref()?;
    flags.envs = vec![EnvForward::try_new(format!(
        "FAKE_LLAMA_CHAT_BODY_FILE={}",
        bodies.display()
    ))?];
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);

    let outcome = pipette_llamacpp::run(&req, &client, &no_readiness_gate, ignore_reps(), None)?;

    assert_eq!(*client.fetched.borrow(), 1, "the corpus is downloaded once");
    assert_eq!(outcome.dataset_digest.as_ref(), Some(&pinned_digest));
    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    let mut ids: Vec<&str> = completions.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["q1", "q2", "q3"]);
    assert!(completions.iter().all(|c| !c.failed), "{completions:?}");

    let mut sent: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for line in std::fs::read_to_string(&bodies)?.lines() {
        let body: serde_json::Value = serde_json::from_str(line)?;
        let messages = body["messages"].clone();
        let sample = served
            .iter()
            .find(|s| serde_json::to_value(&s.messages).ok().as_ref() == Some(&messages))
            .ok_or_else(|| {
                anyhow::anyhow!("a request sent messages no served sample has: {messages}")
            })?;
        sent.insert(sample.id.clone(), messages);
    }
    assert_eq!(sent.len(), 3, "every served sample was sent");
    Ok(())
}

/// Both a corpus and inline samples leaves it unclear which the scorer grades.
#[test]
fn a_definition_with_both_a_corpus_and_samples_is_refused() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let served = served()?;
    let (dataset, tar) = corpus(&served)?;
    let client = Serving {
        tar,
        root: root.path().to_path_buf(),
        fetched: RefCell::new(0),
    };
    let (model_path, mmproj_path) = model_files(root.path())?;
    let mut req = fake_run_request(
        fake_runtime(&fake_runtime_install_dir(root.path())?)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        entry(Some(dataset), Some(served))?,
        None,
    );
    req.model.declared = declared("unsloth/Qwen3.5-2B-GGUF")?;
    let error = pipette_llamacpp::run(&req, &client, &no_readiness_gate, ignore_reps(), None)
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
    assert!(
        error.contains("both a dataset and inline samples"),
        "{error}"
    );
    Ok(())
}

#[test]
fn a_cell_off_its_recipe_never_downloads_the_corpus() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let served = served()?;
    let (model_path, mmproj_path) = model_files(root.path())?;
    let thinking: ModelFlags = serde_json::from_value(json!({
        "benchmark_type": "vlm_eval", "model_type": "gguf_vision", "enable_thinking": true,
    }))?;
    let local = fake_vision_model(&model_path, &mmproj_path)?;
    for (model, flags, names) in [
        (
            declared("bartowski/Qwen3.5-2B-GGUF")?,
            None,
            "does not cover bartowski/Qwen3.5-2B-GGUF",
        ),
        (
            declared("unsloth/Qwen3.5-2B-GGUF")?,
            Some(thinking),
            "enable_thinking = unset",
        ),
        // Files on disk say nothing about which model they are.
        (local, None, "from Hugging Face repos"),
    ] {
        let (dataset, tar) = corpus(&served)?;
        let client = Serving {
            tar,
            root: root.path().to_path_buf(),
            fetched: RefCell::new(0),
        };
        let mut req = fake_run_request(
            fake_runtime(&fake_runtime_install_dir(root.path())?)?,
            fake_vision_model(&model_path, &mmproj_path)?,
            entry(Some(dataset), None)?,
            None,
        );
        req.model.declared = model;
        req.model_flags = flags;
        let error = pipette_llamacpp::run(&req, &client, &no_readiness_gate, ignore_reps(), None)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(
            error.contains("is off its recipe") && error.contains(names),
            "{names}: {error}"
        );
        assert_eq!(
            *client.fetched.borrow(),
            0,
            "{names}: nothing was downloaded"
        );
    }
    Ok(())
}
