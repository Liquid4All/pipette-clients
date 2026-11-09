//! Fixtures for this crate's integration tests: the fake `llama-server`
//! install and the plan runtime/model that point at it. They live here because
//! `CARGO_BIN_EXE_fake_llama_server` resolves only in the declaring crate.
#![allow(dead_code, reason = "each test crate uses a subset of these helpers")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AbsolutePath, BenchmarkDefinition, BenchmarkEvalCompletion, BenchmarkFlags, BenchmarkId,
    ChatMessage, ChatRole, ContentPart, Dataset, DeclaredBound, EnvForward, EvalBenchmark,
    EvalSample, GgufText, GgufTextSource, GgufVision, GgufVisionAbsoluteFiles, GgufVisionSource,
    ImageUrl, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource,
    LlamacppFlashAttention, MessageContent, Model, ModelFlags, PositiveCount, PositiveSeconds,
    RunRequest, Runtime, RuntimeFlags, RuntimeOutput, VlmEvalBenchmark,
    VlmEvalLlamacppCliStockToolsGgufVision,
};

/// A temp directory named for `label`, so a leaked one names the test that
/// leaked it.
pub fn labeled_temp_dir(label: &str) -> anyhow::Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix(&format!("pipette-eval-{label}-"))
        .tempdir()
        .with_context(|| format!("create temp dir for {label}"))
}

/// A temp directory that is also the run's [`RunContext`], so the checkpoint a
/// test seeds and the one the run resumes are the same store.
pub struct TempStore {
    pub root: PathBuf,
    _dir: tempfile::TempDir,
}

impl RunContext for TempStore {
    fn fetch_dataset(&self, _dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("this test lends no corpus")
    }

    fn datasets_dir(&self) -> PathBuf {
        self.root.join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.root.join("evals"))
    }
}

pub fn temp_store(label: &str) -> anyhow::Result<TempStore> {
    let dir = labeled_temp_dir(label)?;
    Ok(TempStore {
        root: dir.path().to_path_buf(),
        _dir: dir,
    })
}

/// Path to the compiled fake llama-server test binary; asserts it exists.
pub fn fake_server_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_BIN_EXE_fake_llama_server"));
    assert!(
        p.exists(),
        "fake_llama_server binary missing at {}",
        p.display(),
    );
    p
}

/// Install root holding the fake `llama-server` and a dummy `llama-bench`.
pub fn fake_runtime_install_dir(root: &Path) -> anyhow::Result<PathBuf> {
    let install = root.join("runtime-install");
    std::fs::create_dir_all(&install)?;
    let server_src = fake_server_path();
    let server_dst = install.join(if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    });
    std::fs::copy(&server_src, &server_dst).with_context(|| {
        format!(
            "copy fake server {} → {}",
            server_src.display(),
            server_dst.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&server_dst)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&server_dst, perms)?;
    }
    // find_tool also requires llama-bench somewhere under the root.
    let bench_dst = install.join(if cfg!(windows) {
        "llama-bench.exe"
    } else {
        "llama-bench"
    });
    std::fs::write(&bench_dst, b"#!/bin/sh\nexit 1\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&bench_dst)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&bench_dst, perms)?;
    }
    Ok(install)
}

/// A fake `llama-mtmd-cli` that merges frames in pairs, as llama.cpp builds before
/// upstream #27348 do for Qwen-VL models: `tokens_per_image` for each pair of
/// `--image` it is given, rounded up.
#[cfg(unix)]
pub fn install_fake_mtmd_cli(install_dir: &Path, tokens_per_image: u32) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let helper = install_dir.join("llama-mtmd-cli");
    std::fs::write(
        &helper,
        format!(
            "#!/bin/sh\nn=0\nfor a in \"$@\"; do [ \"$a\" = --image ] && n=$((n+1)); done\n\
             echo \"decoding image batch 1/1, n_tokens_batch = $(((n+1)/2*{tokens_per_image}))\" >&2\n"
        ),
    )?;
    let mut perms = std::fs::metadata(&helper)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&helper, perms)?;
    Ok(())
}

/// A fake `llama-mtmd-cli` that counts audio: each run appends one line to
/// `log` naming the clip after `--audio`, whether it was a WAV file while the
/// helper ran, the `PIPETTE_TEST_FORWARD` it was given and its argv (one
/// tab-separated field per argument), then prints `tokens` (or
/// `FAKE_MTMD_AUDIO_TOKENS` when the environment sets it) as one audio batch
/// and exits with `status`.
#[cfg(unix)]
pub fn install_fake_mtmd_cli_audio(
    install_dir: &Path,
    log: &Path,
    tokens: u32,
    status: i32,
) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let helper = install_dir.join("llama-mtmd-cli");
    std::fs::write(
        &helper,
        format!(
            "#!/bin/sh\nclip=\nprev=\nfor a in \"$@\"; do [ \"$prev\" = --audio ] && clip=$a; prev=$a; done\n\
             wav=no\n[ -n \"$clip\" ] && [ \"$(head -c 4 \"$clip\")\" = RIFF ] && wav=yes\n\
             {{ printf 'clip=%s\\twav=%s\\tforward=%s\\targs=' \"$clip\" \"$wav\" \"${{PIPETTE_TEST_FORWARD:-}}\"; \
             printf '%s\\t' \"$@\"; printf '\\n'; }} >> '{log}'\n\
             echo \"decoding audio batch 1/1, n_tokens_batch = ${{FAKE_MTMD_AUDIO_TOKENS:-{tokens}}}\" >&2\nexit {status}\n",
            log = log.display(),
        ),
    )?;
    let mut perms = std::fs::metadata(&helper)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&helper, perms)?;
    Ok(())
}

const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;

/// A GGUF header holding what the client reads from a projector: that it has an
/// audio encoder, and the encoder's name when `kind` gives one.
pub fn audio_projector(kind: Option<&str>) -> Vec<u8> {
    projector_header(kind, false)
}

/// [`audio_projector`] for a projector that holds a vision encoder beside the
/// audio one (Qwen-Omni's).
pub fn audio_and_vision_projector(kind: &str) -> Vec<u8> {
    projector_header(Some(kind), true)
}

fn projector_header(kind: Option<&str>, with_vision: bool) -> Vec<u8> {
    let entry = |out: &mut Vec<u8>, key: &str, value_type: u32, value: &[u8]| {
        out.extend((key.len() as u64).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend(value_type.to_le_bytes());
        out.extend(value);
    };
    let mut out = b"GGUF".to_vec();
    out.extend(3_u32.to_le_bytes());
    out.extend(0_u64.to_le_bytes());
    out.extend((1 + u64::from(kind.is_some()) + u64::from(with_vision)).to_le_bytes());
    entry(&mut out, "clip.has_audio_encoder", GGUF_TYPE_BOOL, &[1]);
    if with_vision {
        entry(&mut out, "clip.has_vision_encoder", GGUF_TYPE_BOOL, &[1]);
    }
    if let Some(kind) = kind {
        let mut value = (kind.len() as u64).to_le_bytes().to_vec();
        value.extend(kind.as_bytes());
        entry(
            &mut out,
            "clip.audio.projector_type",
            GGUF_TYPE_STRING,
            &value,
        );
    }
    out
}

/// [`install_fake_mtmd_cli`] that also appends a line to `log` each time it
/// runs, so a test can tell whether a cell calibrated.
#[cfg(unix)]
pub fn install_logged_fake_mtmd_cli(install_dir: &Path, log: &Path) -> anyhow::Result<()> {
    install_fake_mtmd_cli(install_dir, 10)?;
    let helper = install_dir.join("llama-mtmd-cli");
    let script = std::fs::read_to_string(&helper)?;
    let logged = script.replacen(
        "#!/bin/sh\n",
        &format!("#!/bin/sh\necho run >> '{}'\n", log.display()),
        1,
    );
    std::fs::write(&helper, logged)?;
    Ok(())
}

/// Plan runtime at `install_dir`, as a CPU build for the host's OS: the flavor
/// picks a memory cell's probe, and every CPU flavor of an OS picks the same one.
pub fn fake_runtime(install_dir: &Path) -> anyhow::Result<Runtime> {
    let dir = AbsolutePath::try_new(install_dir.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    let flavor = match std::env::consts::OS {
        "linux" => LlamaCppFlavor::LinuxX64Cpu,
        "windows" => LlamaCppFlavor::WindowsX64Cpu,
        _ => LlamaCppFlavor::MacosArm64,
    };
    Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source: LlamacppCliStockToolsSource::AbsoluteDir { dir },
        flavor,
    }))
}

pub fn fake_model(model_path: &Path) -> anyhow::Result<Model> {
    let path = AbsolutePath::try_new(model_path.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    Ok(Model::GgufText(GgufText {
        source: GgufTextSource::AbsoluteFile { path },
    }))
}

pub fn fake_vision_model(model_path: &Path, mmproj_path: &Path) -> anyhow::Result<Model> {
    let model = AbsolutePath::try_new(model_path.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    let mmproj = AbsolutePath::try_new(mmproj_path.display().to_string())
        .map_err(|e| anyhow::anyhow!("AbsolutePath: {e}"))?;
    Ok(Model::GgufVision(GgufVision {
        source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj }),
    }))
}

pub fn fake_run_request(
    runtime: Runtime,
    model: Model,
    benchmark: BenchmarkDefinition,
    benchmark_flags: Option<BenchmarkFlags>,
) -> RunRequest {
    fake_run_request_with_flags(runtime, model, benchmark, benchmark_flags, None)
}

/// The flags are part of the resume digest, so a test that pre-seeds a
/// checkpoint must build the request the same way.
pub fn fake_run_request_with_flags(
    runtime: Runtime,
    model: Model,
    benchmark: BenchmarkDefinition,
    benchmark_flags: Option<BenchmarkFlags>,
    runtime_flags: Option<RuntimeFlags>,
) -> RunRequest {
    RunRequest {
        runtime: DeclaredBound::already_bound(runtime),
        model: DeclaredBound::already_bound(model),
        runtime_flags,
        model_flags: None,
        benchmark_flags,
        benchmark,
    }
}

/// `req` with its flags' knobs that share a type set to values no default
/// takes, so an argument passed in another's place shows in the argv
/// ([`assert_distinct_knobs`]).
pub fn with_distinct_knobs(mut req: RunRequest) -> anyhow::Result<RunRequest> {
    let mut flags = req.runtime_flags_ref()?;
    flags.threads = Some(3);
    flags.number_gpu_layers = Some(7);
    flags.flash_attention = Some(LlamacppFlashAttention::On);
    if runs_on_server(&req.benchmark) {
        flags.no_cache = Some(true);
    }
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    Ok(req)
}

/// Each knob [`with_distinct_knobs`] set, on its own flag in `argv`, and a
/// server's context beside them.
pub fn assert_distinct_knobs(
    benchmark: &BenchmarkDefinition,
    argv: &[String],
) -> anyhow::Result<()> {
    let passes = |flag: &str, value: &str| argv.windows(2).any(|pair| pair == [flag, value]);
    anyhow::ensure!(
        passes("--threads", "3") && passes("--n-gpu-layers", "7") && passes("--flash-attn", "on"),
        "a knob is missing or on another's flag: {argv:?}"
    );
    if runs_on_server(benchmark) {
        anyhow::ensure!(
            argv.iter().any(|token| token == "--no-cache-prompt")
                && argv.iter().any(|token| token == "--ctx-size"),
            "the prompt cache or context knob is missing: {argv:?}"
        );
    }
    Ok(())
}

/// Whether `benchmark` runs on `llama-server` rather than a bench tool.
fn runs_on_server(benchmark: &BenchmarkDefinition) -> bool {
    !matches!(
        benchmark,
        BenchmarkDefinition::PrefillThroughput(_)
            | BenchmarkDefinition::DecodeThroughput(_)
            | BenchmarkDefinition::MaxMemoryUsage(_)
    )
}

/// The fake's log on a response: it prints to stderr alone, and both streams
/// are kept.
pub fn assert_server_log(output: &RuntimeOutput) -> anyhow::Result<()> {
    anyhow::ensure!(
        output.runtime_stdout.as_deref() == Some(""),
        "stdout should be kept, and empty: {:?}",
        output.runtime_stdout
    );
    let stderr = output.runtime_stderr.as_deref().unwrap_or_default();
    anyhow::ensure!(stderr.contains("fake-llama-server: listening"), "{stderr}");
    Ok(())
}

/// A readiness gate that always passes: a test must not block on host thermals.
pub fn no_readiness_gate() -> anyhow::Result<()> {
    Ok(())
}

/// A [`MeasurementObserver`] that records nothing: a test must not probe host
/// sensors.
pub fn ignore_reps() -> MeasurementObserver<'static> {
    MeasurementObserver::QUIET
}

/// A [`RunContext`] that checkpoints evals under `root` and serves no corpus,
/// recording the datasets a cell asks for.
pub fn no_corpus(root: &Path) -> NoCorpus {
    NoCorpus {
        root: root.to_path_buf(),
        asked: RefCell::new(Vec::new()),
    }
}

pub struct NoCorpus {
    root: PathBuf,
    pub asked: RefCell<Vec<Dataset>>,
}

impl RunContext for NoCorpus {
    fn fetch_dataset(&self, dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
        self.asked.borrow_mut().push(dataset.clone());
        anyhow::bail!("this test lends no corpus")
    }

    fn datasets_dir(&self) -> PathBuf {
        self.root.join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.root.join("evals"))
    }
}

/// Dataset id for sample `index`, shared so tests across files match the
/// rows the loop wrote.
pub fn sample_id(index: usize) -> String {
    format!("s-{index:02}")
}

/// The request every eval-pool test drives: the fake server under `root`, one
/// eval benchmark over `samples`, and the slot knobs under test. Shared so the
/// pool tests (separate processes for their `FAKE_LLAMA_*` env vars) keep
/// exercising the same loop.
pub fn pool_request(
    root: &Path,
    benchmark_id: &str,
    samples: Vec<Value>,
    parallel: u32,
    ctx_size: Option<u32>,
) -> anyhow::Result<RunRequest> {
    let model_path = root.join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: BenchmarkId::try_new(benchmark_id.to_owned())?,
        parameter_eval_id: benchmark_id.into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: PositiveCount::try_new(16)?,
        parameter_max_context: None,
        parameter_mcq_choices: None,
        parameter_temperature: None,
        parameter_top_p: None,
        parameter_top_k: None,
        parameter_min_p: None,
        parameter_seed: None,
        samples: Some(samples),
    });
    let install = fake_runtime_install_dir(root)?;
    Ok(fake_run_request_with_flags(
        fake_runtime(&install)?,
        fake_model(&model_path)?,
        benchmark,
        Some(BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
            http_timeout_seconds: Some(PositiveSeconds::try_new(30)?),
        }),
        Some(RuntimeFlags::EvalLlamacppCliStockToolsGgufText(
            pipette_plan_types::EvalLlamacppCliStockToolsGgufText {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size,
                no_cache: None,
                parallel: Some(PositiveCount::try_from(parallel)?),
                envs: Vec::new(),
                raw: Vec::new(),
            },
        )),
    ))
}

/// The vlm_eval sibling of [`pool_request`], not a generic over both: the
/// cells carry different typed flags and benchmark bodies, the difference the
/// vlm pool tests exist to exercise. `parameter_mcq_choices` is unset
/// so the runner takes the streaming arm, the one that tees tokens to stderr
/// and that a pool has to turn off.
pub fn vlm_pool_request(
    root: &Path,
    benchmark_id: &str,
    samples: Vec<EvalSample>,
    parallel: u32,
    ctx_size: Option<u32>,
) -> anyhow::Result<RunRequest> {
    let model_path = root.join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;
    let benchmark = BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
        benchmark_id: BenchmarkId::try_new(benchmark_id.to_owned())?,
        parameter_lmms_task: benchmark_id.into(),
        parameter_dataset_version: "local".into(),
        parameter_sample_count: PositiveCount::try_new(i32::try_from(samples.len())?)?,
        parameter_max_tokens: PositiveCount::try_new(16)?,
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
        samples: Some(samples),
    });
    let install = fake_runtime_install_dir(root)?;
    Ok(fake_run_request_with_flags(
        fake_runtime(&install)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark,
        Some(BenchmarkFlags::VlmEvalLlamacppCliStockToolsGgufVision {
            http_timeout_seconds: Some(PositiveSeconds::try_new(30)?),
            readiness: None,
        }),
        Some(RuntimeFlags::VlmEvalLlamacppCliStockToolsGgufVision(
            VlmEvalLlamacppCliStockToolsGgufVision {
                threads: None,
                number_gpu_layers: None,
                mmap: None,
                load_mode: None,
                flash_attention: None,
                ctx_size,
                no_cache: None,
                parallel: Some(PositiveCount::try_from(parallel)?),
                envs: Vec::new(),
                raw: Vec::new(),
            },
        )),
    ))
}

/// What a cell forwards in its `envs` to give the fake's chat template a
/// thinking switch.
pub const THINKING_SWITCH: &str = "FAKE_LLAMA_THINKING_SWITCH=1";

/// A one-slot [`vlm_pool_request`] over a sample per text, with no model flags
/// when `enable_thinking` is `None` (a model with no reasoning mode). The
/// cell forwards `envs` and tells the fake to log each process's arguments
/// ([`started_reasoning_flags`]) and each chat request ([`chat_bodies`]).
pub fn vlm_reasoning_request(
    root: &Path,
    enable_thinking: Option<bool>,
    texts: &[&str],
    envs: &[&str],
) -> anyhow::Result<RunRequest> {
    let samples = texts
        .iter()
        .enumerate()
        .map(|(i, text)| vlm_sample(i, text))
        .collect();
    let mut req = vlm_pool_request(root, "vlm-eval-reasoning-test", samples, 1, None)?;
    req.model_flags = enable_thinking.map(|v| ModelFlags::VlmEvalGgufVision {
        enable_thinking: Some(v),
    });
    let mut flags = req.runtime_flags_ref()?;
    flags.envs = [
        format!("FAKE_LLAMA_ARGS_FILE={}", root.join("args").display()),
        format!("FAKE_LLAMA_CALL_LOG={}", root.join("calls").display()),
    ]
    .into_iter()
    .chain(envs.iter().map(|env| (*env).to_owned()))
    .map(EnvForward::try_new)
    .collect::<Result<_, _>>()?;
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    Ok(req)
}

pub fn reasoning_flags<S: AsRef<str>>(argv: &[S]) -> Vec<&str> {
    argv.windows(2)
        .filter_map(|pair| match pair {
            [flag, value] if flag.as_ref().starts_with("--reasoning") => {
                Some([flag.as_ref(), value.as_ref()])
            }
            _ => None,
        })
        .flatten()
        .collect()
}

/// The reasoning flags each process a [`vlm_reasoning_request`] started under
/// `root` was given, in start order.
pub fn started_reasoning_flags(root: &Path) -> anyhow::Result<Vec<Vec<String>>> {
    Ok(std::fs::read_to_string(root.join("args"))
        .context("the fake server logged no arguments")?
        .lines()
        .map(|line| {
            let argv: Vec<&str> = line.split_whitespace().collect();
            reasoning_flags(&argv)
                .into_iter()
                .map(str::to_owned)
                .collect()
        })
        .collect())
}

/// The chat requests a [`vlm_reasoning_request`]'s servers under `root` were
/// sent, in arrival order.
pub fn chat_bodies(root: &Path) -> anyhow::Result<Vec<Value>> {
    std::fs::read_to_string(root.join("calls"))
        .context("the fake server logged no chat request")?
        .lines()
        .map(|line| serde_json::from_str(line).context("a logged request is not JSON"))
        .collect()
}

/// One vlm_eval sample in the shape the evals server builds.
pub fn vlm_sample(index: usize, text: &str) -> EvalSample {
    EvalSample {
        id: sample_id(index),
        messages: vec![ChatMessage {
            role: ChatRole::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: text.to_owned(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: TINY_PNG_DATA_URI.to_owned(),
                    },
                },
            ]),
        }],
        chat_template_kwargs: None,
    }
}

/// A 1x1 PNG, small enough that ctx calibration stays on its cheap
/// scan-and-floor path.
pub const TINY_PNG_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// Completions keyed by dataset id, after checking every sample landed exactly
/// once. A pool writes rows out of dataset order.
pub fn completions_by_id(
    completions: &[BenchmarkEvalCompletion],
    expected: usize,
) -> anyhow::Result<HashMap<&str, &BenchmarkEvalCompletion>> {
    anyhow::ensure!(
        completions.len() == expected,
        "expected {expected} completions, got {}",
        completions.len(),
    );
    let by_id: HashMap<_, _> = completions.iter().map(|c| (c.id.as_str(), c)).collect();
    anyhow::ensure!(
        by_id.len() == expected,
        "ids must be unique; got {:?}",
        completions.iter().map(|c| &c.id).collect::<Vec<_>>(),
    );
    Ok(by_id)
}

/// How a range of samples must have come out.
pub enum Expect<'a> {
    /// Generated, carrying exactly this completion text.
    Answered(&'a str),
    /// Recorded failed, with `failed_reason` naming this substring.
    Failed(&'a str),
}

/// Assert every sample in `indices` matches `expect`.
pub fn assert_samples(
    by_id: &HashMap<&str, &BenchmarkEvalCompletion>,
    indices: impl IntoIterator<Item = usize>,
    expect: Expect<'_>,
) -> anyhow::Result<()> {
    indices.into_iter().try_for_each(|i| {
        let id = sample_id(i);
        let entry = by_id
            .get(id.as_str())
            .with_context(|| format!("missing completion for {id}"))?;
        match expect {
            Expect::Answered(text) => {
                anyhow::ensure!(
                    !entry.failed,
                    "{id} should not have failed; reason={:?}",
                    entry.failed_reason,
                );
                anyhow::ensure!(
                    entry.completion == text,
                    "{id} should read {text:?}, got {:?}",
                    entry.completion,
                );
            }
            Expect::Failed(needle) => {
                anyhow::ensure!(entry.failed, "{id} should have been recorded failed");
                let reason = entry.failed_reason.as_deref().unwrap_or("");
                anyhow::ensure!(
                    reason.contains(needle),
                    "{id} should say {needle:?}; got {reason:?}",
                );
            }
        }
        Ok(())
    })
}
