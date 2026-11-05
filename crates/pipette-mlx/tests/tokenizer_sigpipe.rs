//! A tokenizer helper that has gone by the time a cell writes it a count, under
//! the default `SIGPIPE` disposition the CLI restores. Its own test binary
//! because that disposition is process-wide: the harness ignores `SIGPIPE`, so
//! an in-crate test would pass without the guard.
#![cfg(target_os = "macos")]

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use serde_json::json;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BatchSize, BenchmarkDefinition, BenchmarkId, Dataset, DeclaredBound, Model, NonNegativeCount,
    PrefillThroughputBenchmark, RunRequest, Runtime,
};
use pipette_venv::{venv_bin, venv_python};

struct NoCorpus(PathBuf);

impl RunContext for NoCorpus {
    fn fetch_dataset(&self, _: &Dataset) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("this test lends no corpus")
    }

    fn datasets_dir(&self) -> PathBuf {
        self.0.join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.0.join("evals"))
    }
}

#[test]
fn a_tokenizer_helper_that_has_gone_fails_the_cell() -> anyhow::Result<()> {
    pipette_subprocess::reset_sigpipe();
    let dir = tempfile::tempdir()?;
    // The venv's interpreter: closes its stdin, then greets and stays up.
    let venv = dir.path().join("venv");
    std::fs::create_dir_all(venv_bin(&venv))?;
    let python = venv_python(&venv);
    std::fs::write(
        &python,
        "#!/bin/sh\nexec 0<&-\necho '{\"eos_ids\": [2]}'\nexec sleep 30\n",
    )?;
    std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))?;
    let model_dir = dir.path().join("model");
    std::fs::create_dir_all(&model_dir)?;
    std::fs::write(model_dir.join("config.json"), "{}")?;

    let runtime: Runtime = serde_json::from_value(json!({
        "type": "mlx_macos_pipette",
        "version": "0.32.2",
        "flavor": "macos-arm64",
        "source": {"type": "absolute_preinstalled", "dir": venv},
    }))?;
    let model: Model =
        serde_json::from_value(json!({"type": "mlx", "source": "absolute_dir", "dir": model_dir}))?;
    let request = RunRequest {
        runtime: DeclaredBound::already_bound(runtime),
        model: DeclaredBound::already_bound(model),
        runtime_flags: None,
        model_flags: None,
        benchmark_flags: None,
        benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_64")?,
            parameter_prefill_tokens: NonNegativeCount::try_new(64)?,
            parameter_batch_size: BatchSize::one(),
        }),
    };
    let Err(err) = pipette_mlx::run(
        &request,
        &NoCorpus(dir.path().to_path_buf()),
        &|| Ok(()),
        MeasurementObserver {
            on_before_rep_start: &|| Ok(()),
            on_after_rep_end: &|| Ok(()),
        },
        None,
    ) else {
        anyhow::bail!("a cell whose tokenizer helper had gone was measured");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("tokenizer helper failed to write request: Broken pipe"),
        "{message}"
    );
    Ok(())
}
