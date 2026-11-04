//! asr_eval end to end against the fake llama-server: a catalog entry runs the
//! clips its `asr-eval-corpus/1` corpus holds. The run refuses a cell off the
//! entry's recipe before it downloads anything, downloads the corpus through
//! its `RunContext` and checks it against both pins, sizes the context from
//! every clip, sends each profile's server flags and request fields, reports
//! the tar's digest, and submits a result only when every clip has an answer.

mod common;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use base64::Engine;
use common::{
    fake_run_request_with_flags, fake_runtime, fake_runtime_install_dir, fake_vision_model,
    ignore_reps, labeled_temp_dir, no_readiness_gate,
};
use serde_json::{Value, json};
use sha2::Digest;

use pipette_llamacpp::run;
use pipette_ops::EvalCompletionsStore;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AsrEvalLlamacppCliStockToolsGgufVision, AudioFormat, BenchmarkDefinition, BenchmarkFlags,
    BenchmarkResultData, ChatMessage, ChatRole, ContentPart, CorpusFormat, Dataset, DatasetAdapter,
    EnvForward, EvalSample, InputAudio, MessageContent, Model, ModelFlags, PositiveCount,
    PositiveSeconds, RunRequest, RuntimeFlags, SamplesDigest, VlmEvalCorpusManifest,
    VlmEvalCorpusSample,
};

/// The task every test entry runs.
const TASK: &str = "asr_fleurs_nl_nl";

/// A client that serves one corpus tar, counting the downloads, and
/// checkpoints under `root`.
struct Serving {
    root: PathBuf,
    tar: Vec<u8>,
    fetched: RefCell<usize>,
}

impl RunContext for Serving {
    fn fetch_dataset(&self, _dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
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

/// A mono 16 kHz 16-bit WAV of `seconds`, each sample `value`.
fn wav(seconds: u32, value: u8) -> Vec<u8> {
    let data: u32 = 32_000 * seconds;
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
    bytes.resize(bytes.len() + data as usize, value);
    bytes
}

/// The fake runtime under `root`, installed by its first run and reused by the
/// next: a test that runs several cells in one root must not copy the server
/// binary over the one a previous run executed, which macOS kills (SIGKILL) at
/// its next start for a stale code signature.
fn install_once(root: &Path) -> anyhow::Result<PathBuf> {
    let install = root.join("runtime-install");
    if install
        .join(if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        })
        .is_file()
    {
        return Ok(install);
    }
    fake_runtime_install_dir(root)
}

/// The client's audio bound for a clip of `seconds`: 25 a second, or 750 per
/// started 30-second window, whichever is more.
fn audio_bound(seconds: u32) -> u32 {
    (seconds * 25).max(750 * (seconds / 30 + 1))
}

/// The chat template's margin the client adds to every sample.
const TEMPLATE_MARGIN: u32 = 256;

/// The id the evals server gives position `index` of [`TASK`]:
/// `sha256("<eval_id>|<index>")`, 12 hex chars.
fn sample_id(index: usize) -> String {
    hex::encode(sha2::Sha256::digest(format!("{TASK}|{index}").as_bytes()))
        .chars()
        .take(12)
        .collect()
}

/// The ids of positions `0..count`, sorted as [`answered`] sorts them.
fn sample_ids(count: usize) -> Vec<String> {
    let mut ids: Vec<String> = (0..count).map(sample_id).collect();
    ids.sort_unstable();
    ids
}

/// The clip at `index`, `seconds` long (its bytes differ per index, so each
/// is a file of its own), with Voxtral's template arguments when `language`.
fn clip(index: usize, seconds: u32, language: Option<&str>) -> EvalSample {
    EvalSample {
        id: sample_id(index),
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: MessageContent::Parts(vec![ContentPart::InputAudio {
                input_audio: InputAudio {
                    data: base64::engine::general_purpose::STANDARD
                        .encode(wav(seconds, u8::try_from(index % 200).unwrap_or(0))),
                    format: AudioFormat::Wav,
                },
            }]),
        }],
        chat_template_kwargs: language.and_then(|language| {
            json!({"transcription": true, "language": language})
                .as_object()
                .cloned()
        }),
    }
}

/// `count` one-second clips.
fn clips(count: usize, language: Option<&str>) -> Vec<EvalSample> {
    (0..count).map(|i| clip(i, 1, language)).collect()
}

/// The evals server's identity of `samples`.
fn identity(samples: &[EvalSample]) -> anyhow::Result<String> {
    let mut digest = SamplesDigest::new();
    samples.iter().try_for_each(|s| digest.update(s))?;
    Ok(digest.finish()?.as_ref().to_owned())
}

/// The pin and tar mgmt would build from `samples` for `profile`, claiming
/// the identity `pinned` (the samples' own, unless a test lies).
fn corpus_claiming(
    samples: &[EvalSample],
    profile: &str,
    pinned: &str,
) -> anyhow::Result<(Dataset, Vec<u8>)> {
    let upstream: pipette_plan_types::DatasetUpstream = serde_json::from_value(json!({
        "source": "evals_server",
        "eval_id": TASK,
        "dataset_version": "2026.10.1",
        "prompt_profile": profile,
        "identity": pinned,
    }))?;
    let mut media = BTreeMap::<String, Vec<u8>>::new();
    let mut lines = Vec::new();
    samples.iter().try_for_each(|s| {
        let line =
            VlmEvalCorpusSample::from_eval_sample_in(CorpusFormat::AsrEvalCorpusV1, s, |p, b| {
                media.insert(p.to_owned(), b.to_vec());
                Ok(())
            })?;
        serde_json::to_writer(&mut lines, &line)?;
        lines.push(b'\n');
        anyhow::Ok(())
    })?;
    let manifest = serde_json::to_vec(&VlmEvalCorpusManifest {
        format: CorpusFormat::AsrEvalCorpusV1,
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
        "format": "asr-eval-corpus/1",
        "adapter": "evals-server@1",
        "digest": hex::encode(sha2::Sha256::digest(&tar)),
        "bytes": tar.len(),
        "upstream": serde_json::to_value(&upstream)?,
        "url": format!("/datasets/asr_eval_{TASK}_{profile}"),
    }))?;
    Ok((dataset, tar))
}

/// The pin and tar of `samples` for `profile`.
fn corpus(samples: &[EvalSample], profile: &str) -> anyhow::Result<(Dataset, Vec<u8>)> {
    corpus_claiming(samples, profile, &identity(samples)?)
}

/// The Hub repo each profile's test recipe covers.
fn recipe_repo(profile: &str) -> &'static str {
    match profile {
        "voxtral" => "ggml-org/Voxtral-Mini-3B-2507-GGUF",
        "minicpm_o" => "openbmb/MiniCPM-o-4_5-gguf",
        "gemma4" => "ggml-org/gemma-4-E2B-it-GGUF",
        _ => "ggml-org/Qwen2.5-Omni-3B-GGUF",
    }
}

fn recipe(profile: &str) -> Value {
    json!({
        "family": profile,
        "models": [{"repo": recipe_repo(profile)}],
        "enable_thinking": false,
        "sources": ["https://huggingface.co/example/card"],
    })
}

/// What the plan named: a quant from `repo`, on the Hub. The run loads the
/// bound files; the recipe reads this.
fn declared(repo: &str) -> anyhow::Result<Model> {
    let (org, name) = repo.split_once('/').context("org/repo")?;
    Ok(serde_json::from_value(json!({
        "type": "gguf_vision", "source": "huggingface", "org": org, "repo_name": name,
        "model": "model-Q4_K_M.gguf", "mmproj": "mmproj-F16.gguf",
    }))?)
}

/// The cell's model flags: thinking off, as every asr_eval cell states it.
fn thinking(enable_thinking: Option<bool>) -> anyhow::Result<ModelFlags> {
    Ok(serde_json::from_value(json!({
        "benchmark_type": "asr_eval", "model_type": "gguf_vision",
        "enable_thinking": enable_thinking,
    }))?)
}

/// A GGUF v3 header of `(key, type, value)` entries.
fn gguf(entries: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3_u32.to_le_bytes());
    out.extend(0_u64.to_le_bytes());
    out.extend((entries.len() as u64).to_le_bytes());
    for (key, value_type, value) in entries {
        out.extend((key.len() as u64).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend(value_type.to_le_bytes());
        out.extend(value);
    }
    out
}

fn gguf_string(value: &str) -> Vec<u8> {
    let mut out = (value.len() as u64).to_le_bytes().to_vec();
    out.extend(value.as_bytes());
    out
}

const GGUF_BOOL: u32 = 7;
const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;

/// The model files a profile's recipe accepts: a projector holding its audio
/// encoder and, for MiniCPM-o, a backbone whose vocabulary names the token
/// ids the recipe hard-codes.
fn model_files(root: &Path, profile: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    let encoder = match profile {
        "gemma4" => "gemma4a",
        "qwen25_omni" => "qwen2.5oa",
        "voxtral" => "voxtral",
        "minicpm_o" => "minicpmoa",
        "lfm25_audio" => "lfm2a",
        "lfm3" => "lfm3a",
        other => anyhow::bail!("no projector for profile {other}"),
    };
    let mmproj = gguf(&[
        ("clip.has_audio_encoder", GGUF_BOOL, vec![1]),
        (
            "clip.audio.projector_type",
            GGUF_STRING,
            gguf_string(encoder),
        ),
    ]);
    let mut tokens = GGUF_STRING.to_le_bytes().to_vec();
    let len: u32 = 151_705;
    tokens.extend(u64::from(len).to_le_bytes());
    for id in 0..len {
        let piece = match id {
            151_704 => "<|tts_eos|>".to_owned(),
            151_645 => "<|im_end|>".to_owned(),
            151_643 => "<|endoftext|>".to_owned(),
            other => format!("t{other}"),
        };
        tokens.extend(gguf_string(&piece));
    }
    let backbone = gguf(&[("tokenizer.ggml.tokens", GGUF_ARRAY, tokens)]);
    let model_path = root.join("fake-model.gguf");
    std::fs::write(&model_path, backbone)?;
    let mmproj_path = root.join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, mmproj)?;
    Ok((model_path, mmproj_path))
}

/// How a test's cell is set up beyond its entry.
#[derive(Default)]
struct Cell<'a> {
    parallel: Option<i32>,
    ctx_size: Option<u32>,
    extra_envs: &'a [String],
}

/// A catalog asr_eval request over `dataset` on the fake server installed
/// under `root`, its declared model covered by the entry's recipe and its
/// thinking off, with the fake told to log its argv and chat bodies there.
fn request(
    root: &Path,
    definition: Value,
    count: usize,
    dataset: &Dataset,
    cell: &Cell<'_>,
) -> anyhow::Result<RunRequest> {
    let profile = definition
        .get("parameter_prompt_profile")
        .and_then(Value::as_str)
        .context("the test names a profile")?
        .to_owned();
    let (model_path, mmproj_path) = model_files(root, &profile)?;
    let mut wire = json!({
        "benchmark_id": format!("asr_eval_{TASK}_{profile}"),
        "benchmark_type": "asr_eval",
        "parameter_asr_task": TASK,
        "parameter_dataset_version": "2026.10.1",
        "parameter_sample_count": count,
        "parameter_max_tokens": 16,
        "dataset": serde_json::to_value(dataset)?,
        "recipe": recipe(&profile),
    });
    if let (Some(wire), Some(definition)) = (wire.as_object_mut(), definition.as_object()) {
        wire.extend(definition.clone());
    }
    let benchmark: BenchmarkDefinition = serde_json::from_value(wire)?;
    let install = install_once(root)?;
    let envs = [
        format!("FAKE_LLAMA_ARGS_FILE={}", root.join("args").display()),
        format!("FAKE_LLAMA_CALL_LOG={}", root.join("calls").display()),
    ]
    .into_iter()
    .chain(cell.extra_envs.iter().cloned())
    .map(EnvForward::try_new)
    .collect::<Result<_, _>>()?;
    let mut req = fake_run_request_with_flags(
        fake_runtime(&install)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark,
        Some(BenchmarkFlags::AsrEvalLlamacppCliStockToolsGgufVision {
            http_timeout_seconds: Some(PositiveSeconds::try_new(30)?),
            readiness: None,
        }),
        Some(RuntimeFlags::AsrEvalLlamacppCliStockToolsGgufVision(
            AsrEvalLlamacppCliStockToolsGgufVision {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size: cell.ctx_size,
                no_cache: None,
                parallel: cell.parallel.map(PositiveCount::try_new).transpose()?,
                envs,
                raw: Vec::new(),
            },
        )),
    );
    req.model.declared = declared(recipe_repo(&profile))?;
    req.model_flags = Some(thinking(Some(false))?);
    Ok(req)
}

fn serving(root: &Path, tar: Vec<u8>) -> Serving {
    Serving {
        root: root.to_path_buf(),
        tar,
        fetched: RefCell::new(0),
    }
}

/// The one process's argv the fake logged under `root`, split on whitespace:
/// an inline chat template spans lines, so a process is counted by its
/// `--model`.
fn started_argv(root: &Path) -> anyhow::Result<Vec<String>> {
    let log = std::fs::read_to_string(root.join("args")).context("no argv logged")?;
    let argv: Vec<String> = log.split_whitespace().map(str::to_owned).collect();
    let starts = argv.iter().filter(|token| *token == "--model").count();
    anyhow::ensure!(starts == 1, "one server per run: {log}");
    Ok(argv)
}

/// The ids of the rows the checkpoint under `root` holds, one per line
/// written (a failed attempt is a line of its own).
fn checkpointed(root: &Path) -> anyhow::Result<Vec<String>> {
    let mut ids = Vec::new();
    let Ok(entries) = std::fs::read_dir(root.join("evals")) else {
        return Ok(ids);
    };
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_none_or(|ext| ext != "jsonl") {
            continue;
        }
        for line in std::fs::read_to_string(&path)?.lines().skip(1) {
            let row: Value = serde_json::from_str(line)?;
            ids.push(
                row.get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            );
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

fn value_of<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
    argv.windows(2)
        .find(|pair| pair.first().is_some_and(|f| f == flag))
        .and_then(|pair| pair.get(1))
        .map(String::as_str)
}

/// The chat requests the fake logged under `root`. Read as a stream of JSON
/// values: two slots append at once, so one line can hold two bodies.
fn chat_bodies(root: &Path) -> anyhow::Result<Vec<Value>> {
    let Ok(log) = std::fs::read_to_string(root.join("calls")) else {
        return Ok(Vec::new());
    };
    serde_json::Deserializer::from_str(&log)
        .into_iter::<Value>()
        .map(|body| body.context("a logged request is not JSON"))
        .collect()
}

/// The ids of a run's completions, sorted, after checking none failed.
fn answered(data: &BenchmarkResultData) -> anyhow::Result<Vec<String>> {
    let BenchmarkResultData::Eval { completions } = data else {
        anyhow::bail!("expected eval result data, got {data:?}");
    };
    anyhow::ensure!(completions.iter().all(|c| !c.failed), "{completions:?}");
    let mut ids: Vec<String> = completions.iter().map(|c| c.id.clone()).collect();
    ids.sort_unstable();
    Ok(ids)
}

fn error_of<T>(result: anyhow::Result<T>) -> String {
    result.err().map(|e| format!("{e:#}")).unwrap_or_default()
}

/// Voxtral over five clips on two slots: the corpus downloaded once and
/// reported as the row's dataset, the transcription template and thinking
/// off on the server, a context per slot times two, and each request
/// carrying exactly the clip and template arguments the evals server served.
#[test]
fn a_voxtral_run_lands_every_clip_of_its_corpus() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-voxtral")?;
    let root = dir.path();
    let served = clips(5, Some("nl"));
    let (dataset, tar) = corpus(&served, "voxtral")?;
    let pinned = dataset.digest.clone();
    let ctx = serving(root, tar);
    let req = request(
        root,
        json!({"parameter_prompt_profile": "voxtral"}),
        5,
        &dataset,
        &Cell {
            parallel: Some(2),
            ..Cell::default()
        },
    )?;
    let response = run(&req, &ctx, &no_readiness_gate, ignore_reps(), None)?;

    assert_eq!(*ctx.fetched.borrow(), 1, "the corpus is downloaded once");
    assert_eq!(response.dataset_digest.as_ref(), Some(&pinned));
    assert_eq!(answered(&response.result_data)?, sample_ids(5));
    let argv = started_argv(root)?;
    assert_eq!(value_of(&argv, "--reasoning"), Some("off"), "{argv:?}");
    assert_eq!(value_of(&argv, "--parallel"), Some("2"), "{argv:?}");
    assert!(argv.iter().any(|t| t == "--chat-template"), "{argv:?}");
    let ctx_size: u32 = value_of(&argv, "--ctx-size")
        .context("a context is always passed")?
        .parse()?;
    // A 60-second floor (2250 audio tokens) + margin + 16 generated, per slot.
    assert_eq!(ctx_size, (2250 + 256 + 16) * 2, "{argv:?}");

    let bodies = chat_bodies(root)?;
    assert_eq!(bodies.len(), 5);
    let served_messages: Vec<Value> = served
        .iter()
        .map(|s| serde_json::to_value(&s.messages))
        .collect::<Result<_, _>>()?;
    for body in bodies {
        assert!(
            served_messages.contains(&body["messages"]),
            "a request sent messages no served sample has"
        );
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"transcription": true, "language": "nl"}),
            "{body}"
        );
        assert_eq!(body["samplers"], json!(["temperature"]), "{body}");
        assert_eq!(body["stream"], json!(true), "{body}");
    }
    let left: Vec<_> =
        std::fs::read_dir(ctx.datasets_dir().join(format!("asr_eval_{TASK}_voxtral")))?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<_, _>>()?;
    assert!(
        left.is_empty(),
        "the corpus is the run's, deleted once it is done: {left:?}"
    );
    Ok(())
}

/// MiniCPM-o with beams: one spelled-out slot running the ASR build's beam
/// search, a context sized for the corpus's longest clip and every beam,
/// one-response requests carrying the beam keys, and every clip answered.
#[test]
fn a_beam_run_sends_the_beams_on_one_slot() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-beams")?;
    let root = dir.path();
    // The longest clip, last, sizes the context past the 60-second floor.
    let long = 95;
    let served = vec![clip(0, 1, None), clip(1, 1, None), clip(2, long, None)];
    let (dataset, tar) = corpus(&served, "minicpm_o")?;
    let ctx = serving(root, tar);
    // The fake bills each request the whole prompt the client's bound allows
    // for the longest clip, and divides what is left among the beams as the
    // ASR build does: every beam's 16 tokens must still fit.
    let prompt = audio_bound(long) + TEMPLATE_MARGIN;
    let envs = [format!("FAKE_LLAMA_PROMPT_TOKENS={prompt}")];
    let req = request(
        root,
        json!({
            "parameter_prompt_profile": "minicpm_o",
            "parameter_num_beams": 3,
            "parameter_beam_repetition_penalty": 1.2,
            "parameter_beam_length_penalty": 1.0
        }),
        3,
        &dataset,
        &Cell {
            extra_envs: &envs,
            ..Cell::default()
        },
    )?;
    let response = run(&req, &ctx, &no_readiness_gate, ignore_reps(), None)?;

    assert_eq!(answered(&response.result_data)?, sample_ids(3));
    let BenchmarkResultData::Eval { completions } = &response.result_data else {
        anyhow::bail!("expected eval result data");
    };
    for completion in completions {
        assert_eq!(completion.completion_tokens, Some(16), "{completion:?}");
    }
    let argv = started_argv(root)?;
    assert_eq!(value_of(&argv, "--beam-search"), Some("3"), "{argv:?}");
    assert_eq!(value_of(&argv, "--parallel"), Some("1"), "{argv:?}");
    assert_eq!(
        value_of(&argv, "--ctx-size"),
        Some((prompt + 3 * 16).to_string().as_str()),
        "{argv:?}"
    );
    assert_eq!(
        value_of(&argv, "--override-kv"),
        Some("tokenizer.ggml.eot_token_id=int:151704"),
        "{argv:?}"
    );
    let bodies = chat_bodies(root)?;
    assert_eq!(bodies.len(), 3);
    for body in bodies {
        assert_eq!(body["n_beams"], json!(3), "{body}");
        assert_eq!(
            body["beam_eos_token_ids"],
            json!([151_645, 151_643]),
            "{body}"
        );
        assert!(body.get("stream").is_none(), "{body}");
    }
    Ok(())
}

/// A corpus that is not the pinned samples (another order under the served
/// identity), or not the size the entry declares, is refused before any
/// server starts and leaves no row.
#[test]
fn a_corpus_off_its_pins_starts_nothing() -> anyhow::Result<()> {
    let served = clips(4, None);
    let mut reordered = served.clone();
    reordered.swap(0, 1);
    for (label, (dataset, tar), declared, names) in [
        (
            "identity",
            corpus_claiming(&reordered, "qwen25_omni", &identity(&served)?)?,
            4,
            "not the samples the scorer grades",
        ),
        (
            "count",
            corpus(&served, "qwen25_omni")?,
            5,
            "its corpus holds 4 samples",
        ),
    ] {
        let dir = labeled_temp_dir("asr-mismatch")?;
        let root = dir.path();
        let ctx = serving(root, tar);
        let req = request(
            root,
            json!({"parameter_prompt_profile": "qwen25_omni"}),
            declared,
            &dataset,
            &Cell::default(),
        )?;
        let error = error_of(run(&req, &ctx, &no_readiness_gate, ignore_reps(), None));
        assert!(error.contains(names), "{label}: {error}");
        assert!(!root.join("args").exists(), "{label}: no server may start");
        assert!(checkpointed(root)?.is_empty(), "{label}: no row");
    }
    Ok(())
}

/// The corpus's longest clip, last, sizes the context from the start, so it
/// runs on the first run. A run whose result failed to land resumes a
/// settled checkpoint: it downloads nothing, starts no server, and reports
/// the same answers from the pinned corpus.
#[test]
fn the_longest_clip_sizes_the_context_and_a_settled_run_downloads_nothing() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-longest")?;
    let root = dir.path();
    let mut served = clips(4, None);
    served.push(clip(4, 100, None));
    let (dataset, tar) = corpus(&served, "qwen25_omni")?;
    let pinned = dataset.digest.clone();
    let ctx = serving(root, tar);
    let req = request(
        root,
        json!({"parameter_prompt_profile": "qwen25_omni"}),
        5,
        &dataset,
        &Cell::default(),
    )?;
    let response = run(&req, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&response.result_data)?, sample_ids(5));
    let argv = started_argv(root)?;
    assert_eq!(
        value_of(&argv, "--ctx-size"),
        Some(
            (audio_bound(100) + TEMPLATE_MARGIN + 16)
                .to_string()
                .as_str()
        ),
        "{argv:?}"
    );

    std::fs::remove_file(root.join("args"))?;
    // What a killed run left of its corpus: the settled run deletes it.
    let leftover = ctx
        .datasets_dir()
        .join(format!("asr_eval_{TASK}_qwen25_omni"))
        .join("asr-eval-corpus.tar");
    std::fs::create_dir_all(leftover.parent().context("no parent")?)?;
    std::fs::write(&leftover, b"left by a killed run")?;
    let again = run(&req, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&again.result_data)?, sample_ids(5));
    assert_eq!(again.dataset_digest.as_ref(), Some(&pinned));
    assert_eq!(
        *ctx.fetched.borrow(),
        1,
        "the settled run downloads nothing"
    );
    assert!(!root.join("args").exists(), "no server may start");
    assert!(!leftover.exists(), "a killed run's corpus is deleted");
    Ok(())
}

/// A pinned ctx_size that cannot hold the corpus's longest clip is refused
/// before any server starts, with the pin that would; that pin runs it.
#[test]
fn a_pinned_context_short_of_the_longest_clip_starts_nothing() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-short-pin")?;
    let root = dir.path();
    let mut served = clips(2, None);
    served.push(clip(2, 100, None));
    let (dataset, tar) = corpus(&served, "gemma4")?;
    let ctx = serving(root, tar);
    let needed = audio_bound(100) + TEMPLATE_MARGIN + 16;
    let short = request(
        root,
        json!({"parameter_prompt_profile": "gemma4"}),
        3,
        &dataset,
        &Cell {
            parallel: Some(2),
            ctx_size: Some(2 * needed - 1),
            ..Cell::default()
        },
    )?;
    let error = error_of(run(&short, &ctx, &no_readiness_gate, ignore_reps(), None));
    assert!(
        error.contains(&format!("at least {}", 2 * needed)),
        "{error}"
    );
    assert!(!root.join("args").exists(), "no server may start");
    assert!(checkpointed(root)?.is_empty(), "and no row is written");

    let enough = request(
        root,
        json!({"parameter_prompt_profile": "gemma4"}),
        3,
        &dataset,
        &Cell {
            parallel: Some(2),
            ctx_size: Some(2 * needed),
            ..Cell::default()
        },
    )?;
    let response = run(&enough, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&response.result_data)?, sample_ids(3));
    Ok(())
}

/// A clip the server cuts short for want of context (a beam whose room the
/// prompt took) is not an answer: the cell fails with no result and no row
/// for it, however often it is rerun as-is, and a larger ctx_size runs it.
#[test]
fn a_clip_that_does_not_fit_fails_the_cell_until_a_larger_context_runs_it() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-unfit")?;
    let root = dir.path();
    let served = clips(3, None);
    let (dataset, tar) = corpus(&served, "minicpm_o")?;
    let ctx = serving(root, tar);
    let beams = json!({
        "parameter_prompt_profile": "minicpm_o",
        "parameter_num_beams": 3,
    });
    // The fake bills each prompt more than the context the client sized (the
    // 60-second floor, as every clip is a second long) leaves three beams of
    // 16 tokens: each beam's room is cut to 12.
    let prompt = audio_bound(60) + TEMPLATE_MARGIN + 12;
    let envs = [format!("FAKE_LLAMA_PROMPT_TOKENS={prompt}")];
    let cut = request(
        root,
        beams.clone(),
        3,
        &dataset,
        &Cell {
            extra_envs: &envs,
            ..Cell::default()
        },
    )?;
    for attempt in 1..=2 {
        let error = error_of(run(&cut, &ctx, &no_readiness_gate, ignore_reps(), None));
        assert!(error.contains("unfit="), "run {attempt}: {error}");
        assert!(error.contains("cost no attempt"), "run {attempt}: {error}");
        assert!(checkpointed(root)?.is_empty(), "run {attempt}: no row");
    }
    let larger = request(
        root,
        beams,
        3,
        &dataset,
        &Cell {
            ctx_size: Some(prompt + 3 * 16),
            extra_envs: &envs,
            ..Cell::default()
        },
    )?;
    let response = run(&larger, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&response.result_data)?, sample_ids(3));
    Ok(())
}

/// A clip whose generation fails is retried on the next run while it has
/// attempts left, and a run that leaves one unanswered submits nothing: the
/// first run fails naming it, the second sends only it again and lands the
/// cell. Once its attempts are spent no rerun publishes the cell.
#[test]
fn a_failed_clip_is_retried_and_a_partial_run_never_submits() -> anyhow::Result<()> {
    let served: Vec<EvalSample> = (0..3)
        .map(|i| clip(i, 1, Some(if i == 1 { "xx" } else { "nl" })))
        .collect();
    let (dataset, tar) = corpus(&served, "voxtral")?;
    // The fake fails, mid-stream, the one request whose body names `xx`.
    let failing = ["FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS=\"language\":\"xx\"".to_owned()];
    let healed = ["FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS=matches-nothing".to_owned()];
    let entry = json!({"parameter_prompt_profile": "voxtral"});

    let dir = labeled_temp_dir("asr-retry")?;
    let root = dir.path();
    let ctx = serving(root, tar.clone());
    let failed = request(
        root,
        entry.clone(),
        3,
        &dataset,
        &Cell {
            extra_envs: &failing,
            ..Cell::default()
        },
    )?;
    let error = error_of(run(&failed, &ctx, &no_readiness_gate, ignore_reps(), None));
    assert!(error.contains(&sample_id(1)), "{error}");
    assert_eq!(chat_bodies(root)?.len(), 3);
    let healed_req = request(
        root,
        entry.clone(),
        3,
        &dataset,
        &Cell {
            extra_envs: &healed,
            ..Cell::default()
        },
    )?;
    let response = run(&healed_req, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&response.result_data)?, sample_ids(3));
    assert_eq!(
        chat_bodies(root)?.len(),
        4,
        "only the failed clip is sent again"
    );

    // Spent: three failed attempts, then nothing more is sent, and the cell
    // is never published, healed or not.
    let dir = labeled_temp_dir("asr-spent")?;
    let root = dir.path();
    let ctx = serving(root, tar);
    let failed = request(
        root,
        entry.clone(),
        3,
        &dataset,
        &Cell {
            extra_envs: &failing,
            ..Cell::default()
        },
    )?;
    for _ in 0..3 {
        assert!(run(&failed, &ctx, &no_readiness_gate, ignore_reps(), None).is_err());
    }
    let healed_req = request(
        root,
        entry,
        3,
        &dataset,
        &Cell {
            extra_envs: &healed,
            ..Cell::default()
        },
    )?;
    let error = error_of(run(
        &healed_req,
        &ctx,
        &no_readiness_gate,
        ignore_reps(),
        None,
    ));
    assert!(error.contains("used up its 3 attempts"), "{error}");
    assert_eq!(
        chat_bodies(root)?.len(),
        3 + 1 + 1,
        "the spent clip is not sent again"
    );
    Ok(())
}

/// `PIPETTE_ALLOW_INCOMPLETE_EVAL` waives the failed-row refusal for text
/// eval and vlm_eval, never for asr_eval: the score server scores an asr_eval
/// only with an answer to every sample, so a waived partial run would be
/// refused there after the client had dropped its checkpoint. Under the
/// waiver a run with a failed clip still fails and keeps its answered rows,
/// and the next run sends only the failed clip.
///
/// The waiver is process-wide, so the test runs itself again in a child
/// process that has it set and holds no other test.
#[test]
fn the_incomplete_eval_waiver_never_submits_a_partial_asr_eval() -> anyhow::Result<()> {
    const WAIVER: &str = "PIPETTE_ALLOW_INCOMPLETE_EVAL";
    const NAME: &str = "the_incomplete_eval_waiver_never_submits_a_partial_asr_eval";
    if std::env::var_os(WAIVER).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", NAME, "--nocapture", "--test-threads=1"])
            .env(WAIVER, "1")
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::ensure!(
            output.status.success() && stdout.contains("1 passed"),
            "the run under {WAIVER}=1 failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Ok(());
    }

    let served: Vec<EvalSample> = (0..3)
        .map(|i| clip(i, 1, Some(if i == 1 { "xx" } else { "nl" })))
        .collect();
    let (dataset, tar) = corpus(&served, "voxtral")?;
    let failing = ["FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS=\"language\":\"xx\"".to_owned()];
    let healed = ["FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS=matches-nothing".to_owned()];
    let entry = json!({"parameter_prompt_profile": "voxtral"});
    let dir = labeled_temp_dir("asr-waiver")?;
    let root = dir.path();
    let ctx = serving(root, tar);
    let partial = request(
        root,
        entry.clone(),
        3,
        &dataset,
        &Cell {
            extra_envs: &failing,
            ..Cell::default()
        },
    )?;
    let error = error_of(run(&partial, &ctx, &no_readiness_gate, ignore_reps(), None));
    assert!(error.contains(&sample_id(1)), "{error}");
    assert!(
        !error.contains(WAIVER),
        "the refusal never offers the waiver: {error}"
    );
    assert_eq!(
        checkpointed(root)?,
        sample_ids(3),
        "the answered clips and the failed one stay in the checkpoint"
    );
    let healed_req = request(
        root,
        entry,
        3,
        &dataset,
        &Cell {
            extra_envs: &healed,
            ..Cell::default()
        },
    )?;
    let response = run(&healed_req, &ctx, &no_readiness_gate, ignore_reps(), None)?;
    assert_eq!(answered(&response.result_data)?, sample_ids(3));
    assert_eq!(
        chat_bodies(root)?.len(),
        4,
        "only the failed clip is sent again"
    );
    Ok(())
}

/// A cell off its entry's recipe (another repo, thinking not stated off, a
/// model off the Hub), or an entry that pins a corpus but names no recipe,
/// is refused before the corpus downloads or a server starts; the covered
/// cell, its control, runs.
#[test]
fn a_cell_off_its_recipe_never_downloads_the_corpus() -> anyhow::Result<()> {
    let served = clips(2, Some("nl"));
    let (dataset, tar) = corpus(&served, "voxtral")?;
    let entry = json!({"parameter_prompt_profile": "voxtral"});
    let cases: [(&str, &str); 6] = [
        ("repo", "does not cover bartowski/Other-GGUF"),
        ("thinking on", "enable_thinking"),
        ("thinking unstated", "enable_thinking"),
        ("local model", "from Hugging Face repos"),
        ("no recipe", "names no recipe"),
        ("covered", ""),
    ];
    for (label, names) in cases {
        let dir = labeled_temp_dir("asr-recipe")?;
        let root = dir.path();
        let ctx = serving(root, tar.clone());
        let mut req = request(root, entry.clone(), 2, &dataset, &Cell::default())?;
        match label {
            "repo" => req.model.declared = declared("bartowski/Other-GGUF")?,
            "thinking on" => req.model_flags = Some(thinking(Some(true))?),
            "thinking unstated" => req.model_flags = Some(thinking(None)?),
            "local model" => req.model.declared = req.model.bound.clone(),
            "no recipe" => {
                if let BenchmarkDefinition::AsrEval(asr) = &mut req.benchmark {
                    asr.recipe = None;
                }
            }
            _ => {}
        }
        let outcome = run(&req, &ctx, &no_readiness_gate, ignore_reps(), None);
        if names.is_empty() {
            assert_eq!(answered(&outcome?.result_data)?, sample_ids(2));
            continue;
        }
        let error = error_of(outcome);
        assert!(error.contains(names), "{label}: {error}");
        assert_eq!(*ctx.fetched.borrow(), 0, "{label}: nothing was downloaded");
        assert!(!root.join("args").exists(), "{label}: no server may start");
    }
    Ok(())
}

/// A model whose projector is another family's is refused before the corpus
/// is downloaded or a server started.
#[test]
fn a_model_of_another_family_is_refused_up_front() -> anyhow::Result<()> {
    let dir = labeled_temp_dir("asr-family")?;
    let root = dir.path();
    let (dataset, tar) = corpus(&clips(2, Some("nl")), "voxtral")?;
    let ctx = serving(root, tar);
    let req = request(
        root,
        json!({"parameter_prompt_profile": "voxtral"}),
        2,
        &dataset,
        &Cell::default(),
    )?;
    // The Voxtral entry on a Qwen2.5-Omni projector.
    model_files(root, "qwen25_omni")?;
    let error = error_of(run(&req, &ctx, &no_readiness_gate, ignore_reps(), None));
    assert!(
        error.contains("holds the `qwen2.5oa` audio encoder"),
        "{error}"
    );
    assert_eq!(*ctx.fetched.borrow(), 0, "nothing may be downloaded");
    assert!(!root.join("args").exists(), "no server may start");
    Ok(())
}
