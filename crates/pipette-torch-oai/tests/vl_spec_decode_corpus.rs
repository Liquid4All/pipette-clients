//! Integration test: the corpus benchmark downloads its corpus through the
//! run's context before it launches a server.

use std::cell::RefCell;
use std::path::PathBuf;

use anyhow::Context;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    AbsolutePath, BenchmarkDefinition, Dataset, DeclaredBound, Model, ModelSource, RunRequest,
    Runtime, Torch, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang,
};
use pipette_programs::ExternalPrograms;

/// Records the datasets a cell asks for, and serves none.
struct Refusing {
    asked: RefCell<Vec<Dataset>>,
    root: PathBuf,
}

impl RunContext for Refusing {
    fn fetch_dataset(&self, dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
        self.asked.borrow_mut().push(dataset.clone());
        anyhow::bail!("no corpus server in this test")
    }

    fn datasets_dir(&self) -> PathBuf {
        self.root.join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.root.join("evals"))
    }
}

/// The cell asks for the `dataset` its definition pins before it looks for
/// `uv` or a model, neither of which exists here, so the context's refusal is
/// the cell's error.
#[test]
fn a_corpus_cell_asks_the_context_for_its_dataset_first() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let benchmark: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
        "benchmark_type": "vl_spec_decode",
        "benchmark_id": "vl_spec_decode_corpus_download_test",
        "parameter_draft_length": 8,
        "parameter_max_tokens": 64,
        "dataset": {
            "format": "vl-spec-decode-corpus/1",
            "adapter": "mmspec@1",
            "digest": "11".repeat(32),
            "bytes": 10,
            "upstream": {
                "source": "git",
                "url": "github.com/acme/corpus",
                "revision": "01".repeat(20),
                "identity": "29".repeat(20),
                "digest": "22".repeat(32)
            },
            "url": "/datasets/vl_spec_decode_corpus_download_test"
        }
    }))?;
    let pinned = benchmark
        .dataset()
        .cloned()
        .context("the fixture pins a dataset")?;
    let absent = |name: &str| AbsolutePath::try_new(root.path().join(name).display().to_string());
    let req = RunRequest {
        runtime: DeclaredBound::already_bound(Runtime::UvSglang(UvSglang {
            server_version: UvServerVersion::try_new("test".to_owned())?,
            build: UvBuild::try_new("cu130".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: absent("runtimes/venv")?,
            },
        })),
        model: DeclaredBound::already_bound(Model::Torch(Torch {
            source: ModelSource::AbsoluteDir {
                dir: absent("models/target")?,
            },
        })),
        runtime_flags: None,
        model_flags: None,
        benchmark_flags: None,
        benchmark,
    };

    let context = Refusing {
        asked: RefCell::new(Vec::new()),
        root: root.path().to_path_buf(),
    };
    let Err(err) = pipette_torch_oai::run(
        &req,
        &context,
        &|| Ok(()),
        MeasurementObserver::QUIET,
        None,
        &ExternalPrograms::at_missing_paths(),
    ) else {
        anyhow::bail!("a corpus cell without its corpus must fail");
    };

    let msg = format!("{err:#}");
    assert!(msg.contains("no corpus server in this test"), "got {msg}");
    assert_eq!(*context.asked.borrow(), [pinned]);
    Ok(())
}
