//! Live run of the corpus benchmark on SGLang: the engine's own entry point
//! over a real venv, a real target + drafter pair and a real corpus tar, with
//! nothing of the CLI around it but what the CLI would bind.
//!
//! **Linux-only, `#[ignore]`.** It needs an NVIDIA host with an SGLang build
//! that carries the DSpark path, and the store entries the CLI's `pull`
//! commands materialize. Everything comes in by environment:
//!
//! - `PIPETTE_LIVE_SGLANG_VENV`: the venv root (`…/runtimes/<key>/blobs/venv`)
//! - `PIPETTE_LIVE_TARGET_DIR`, `PIPETTE_LIVE_DRAFT_DIR`: the pair's two bound
//!   directories under a `models/` store (the label the orphan reap keys on is
//!   derived from that path, so it has to be a store path)
//! - `PIPETTE_LIVE_CORPUS_TAR`: a `vl-spec-decode-corpus/1` tar
//! - `PIPETTE_LIVE_SPECULATION`: `on` (default) or `off`
//! - `PIPETTE_LIVE_BATCH_SIZE`: samples in flight, `parameter_batch_size` (default 1)
//! - `PIPETTE_LIVE_MAX_TOKENS`: generation cap per turn (default 2048)
//! - `PIPETTE_LIVE_RAW`: extra server argv, space-separated (default
//!   `--mem-fraction-static 0.60 --speculative-draft-attention-backend flashinfer`
//!   with speculation on, `--mem-fraction-static 0.60` off)
//! - `PIPETTE_LIVE_ENVS`: env forwards, space-separated (default
//!   `SGLANG_ENABLE_JIT_DEEPGEMM=0`)
//!
//! ```text
//! cargo test -p pipette-torch-oai --test live_sglang_vl_spec_decode -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, Dataset,
    DeclaredBound, EnvForward, Model, ModelSource, PositiveCount, RunRequest, Runtime,
    RuntimeFlags, SamplingTemperature, Sha256, SpeculativePair, Torch, UvBuild, UvPythonVersion,
    UvRuntimeSource, UvServerVersion, UvSglang, VlSpecDecodeBenchmark,
};
use pipette_programs::ExternalPrograms;

/// Serves the corpus tar from disk, as the management server would.
struct LocalCorpus {
    tar: Vec<u8>,
    dir: PathBuf,
}

impl RunContext for LocalCorpus {
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

fn env(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("set {name}; see the file's doc comment"))
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn abs(path: &str) -> anyhow::Result<AbsolutePath> {
    Ok(AbsolutePath::try_new(path.to_owned())?)
}

fn dir(path: &str) -> anyhow::Result<ModelSource> {
    Ok(ModelSource::AbsoluteDir { dir: abs(path)? })
}

#[test]
#[ignore = "needs an NVIDIA host with the DSpark SGLang build and the pulled stores"]
fn a_live_corpus_traversal_on_sglang() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let venv = env("PIPETTE_LIVE_SGLANG_VENV")?;
    let target = env("PIPETTE_LIVE_TARGET_DIR")?;
    let draft = env("PIPETTE_LIVE_DRAFT_DIR")?;
    let tar = env("PIPETTE_LIVE_CORPUS_TAR")?;
    let speculation = env_or("PIPETTE_LIVE_SPECULATION", "on") == "on";
    let batch_size: u32 = env_or("PIPETTE_LIVE_BATCH_SIZE", "1").parse()?;
    let max_tokens: u32 = env_or("PIPETTE_LIVE_MAX_TOKENS", "2048").parse()?;
    let raw_default = if speculation {
        "--mem-fraction-static 0.60 --speculative-draft-attention-backend flashinfer"
    } else {
        "--mem-fraction-static 0.60"
    };
    let raw: Vec<String> = env_or("PIPETTE_LIVE_RAW", raw_default)
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let envs = env_or("PIPETTE_LIVE_ENVS", "SGLANG_ENABLE_JIT_DEEPGEMM=0")
        .split_whitespace()
        .map(EnvForward::try_new)
        .collect::<Result<Vec<_>, _>>()?;

    let runtime = Runtime::UvSglang(UvSglang {
        server_version: UvServerVersion::try_new("live".to_owned())?,
        build: UvBuild::try_new("cu130".to_owned())?,
        python_version: UvPythonVersion::try_new("3.12".to_owned())?,
        source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(&venv)? },
    });
    // Speculation off runs the pair's target as a plain torch model, which is
    // the row a plan authors for the regular model.
    let model = if speculation {
        Model::TorchSpeculative(SpeculativePair {
            target: dir(&target)?,
            draft: dir(&draft)?,
        })
    } else {
        Model::Torch(Torch {
            source: dir(&target)?,
        })
    };
    let runtime_flags = if speculation {
        RuntimeFlags::VlSpecDecodeUvSglangTorchSpeculative {
            tensor_parallel_size: None,
            prefix_caching: None,
            envs,
            raw,
            speculative_enabled: Some(true),
        }
    } else {
        RuntimeFlags::VlSpecDecodeUvSglangTorch {
            tensor_parallel_size: None,
            prefix_caching: None,
            envs,
            raw,
            speculative_enabled: Some(false),
        }
    };

    let bytes = std::fs::read(&tar).with_context(|| format!("failed to read {tar}"))?;
    let digest = Sha256::try_new(hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes)))?;
    let corpus_dir =
        std::env::temp_dir().join(format!("pipette-live-corpus-{}", std::process::id()));
    // Read once for the pin's upstream and subset; the engine downloads its own
    // copy through the context.
    let (corpus, _) =
        pipette_vl_spec_decode_corpus::verify_and_open(&bytes, &digest, &corpus_dir.join("pin"))?;
    let dataset: Dataset = serde_json::from_value(serde_json::json!({
        "format": "vl-spec-decode-corpus/1",
        "adapter": "mmspec@1",
        "subset": corpus.samples().first().map(|s| s.subset.clone()),
        "bytes": bytes.len(),
        "digest": digest.as_ref(),
        "upstream": corpus.manifest().upstream,
    }))?;

    let req = RunRequest {
        runtime: DeclaredBound::already_bound(runtime),
        model: DeclaredBound::already_bound(model),
        runtime_flags: Some(runtime_flags),
        model_flags: None,
        benchmark_flags: None,
        benchmark: BenchmarkDefinition::VlSpecDecode(VlSpecDecodeBenchmark {
            benchmark_id: BenchmarkId::try_new("live_vl_spec_decode".to_owned())?,
            parameter_draft_length: PositiveCount::try_new(9)?,
            parameter_max_tokens: PositiveCount::try_from(max_tokens)?,
            parameter_batch_size: BatchSize::try_new(batch_size)?,
            parameter_temperature: SamplingTemperature::greedy(),
            dataset,
        }),
    };
    let observer = MeasurementObserver::QUIET;
    let corpus_turns = corpus.turns();
    let context = LocalCorpus {
        tar: bytes,
        dir: corpus_dir.clone(),
    };

    let started = Instant::now();
    // The host's own programs: this is the live test, so the uv it launches
    // from is the one on this box.
    let response = pipette_torch_oai::run(
        &req,
        &context,
        &|| Ok(()),
        observer,
        None,
        &ExternalPrograms::from_path(),
    )?;
    let elapsed = started.elapsed();
    let _ = std::fs::remove_dir_all(&corpus_dir);

    eprintln!(
        "== {} ({:.0}s wall incl. server start)",
        req.benchmark.benchmark_id(),
        elapsed.as_secs_f64()
    );
    eprintln!(
        "{}",
        serde_json::Value::Object(response.client_extras.clone())
    );
    eprintln!(
        "runtime_flags: {}",
        serde_json::to_string(&response.runtime_flags)?
    );
    eprintln!(
        "output_digest: {:?}",
        response.output_digest.as_ref().map(AsRef::as_ref)
    );
    let BenchmarkResultData::VlSpecDecode {
        vl_spec_decode_turns,
        vl_spec_decode_prompt_ms,
        vl_spec_decode_predicted_ms,
        vl_spec_decode_ttft_ms,
        vl_spec_decode_decode_ms,
        vl_spec_decode_e2e_ms,
        vl_spec_decode_generated_tokens,
        vl_spec_decode_drafted_tokens,
        vl_spec_decode_accepted_tokens,
        vl_spec_decode_turn_accept_length_sum,
    } = response.result_data
    else {
        anyhow::bail!(
            "expected a vl_spec_decode row, got {:?}",
            response.result_data
        );
    };
    let turns = f64::from(vl_spec_decode_turns);
    eprintln!(
        "turns={vl_spec_decode_turns} generated={vl_spec_decode_generated_tokens} \
         tok/s={:.1} ttft_ms/turn={:.1} decode_ms/turn={:.1} e2e_ms/turn={:.1} \
         prompt_ms/turn={:.1} predicted_ms/turn={:.1}",
        vl_spec_decode_generated_tokens as f64 / vl_spec_decode_predicted_ms * 1000.0,
        vl_spec_decode_ttft_ms / turns,
        vl_spec_decode_decode_ms / turns,
        vl_spec_decode_e2e_ms / turns,
        vl_spec_decode_prompt_ms / turns,
        vl_spec_decode_predicted_ms / turns,
    );
    if let (Some(drafted), Some(accepted), Some(sum)) = (
        vl_spec_decode_drafted_tokens,
        vl_spec_decode_accepted_tokens,
        vl_spec_decode_turn_accept_length_sum,
    ) {
        eprintln!(
            "drafted={drafted} accepted={accepted} accept_length={:.3} \
             accept_length_per_request={:.3} draft_acceptance_rate={:.3}",
            vl_spec_decode_generated_tokens as f64
                / (vl_spec_decode_generated_tokens - accepted) as f64,
            sum / turns,
            accepted as f64 / drafted as f64,
        );
    }
    assert_eq!(vl_spec_decode_turns as usize, corpus_turns);
    assert_eq!(vl_spec_decode_drafted_tokens.is_some(), speculation);
    Ok(())
}
