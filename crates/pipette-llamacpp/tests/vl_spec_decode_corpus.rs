//! The corpus benchmark downloads its corpus through the run's context, as the
//! first thing it does.

mod common;

use anyhow::Context;
use common::{
    fake_run_request, fake_runtime, fake_vision_model, ignore_reps, no_corpus, no_readiness_gate,
};

use pipette_plan_types::BenchmarkDefinition;

/// The cell asks for the `dataset` its definition pins before it looks for a
/// server or a model, neither of which exists here, so the context's refusal
/// is the cell's error.
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
    let req = fake_run_request(
        fake_runtime(&root.path().join("no-runtime"))?,
        fake_vision_model(
            &root.path().join("no-model.gguf"),
            &root.path().join("no-mmproj.gguf"),
        )?,
        benchmark,
        None,
    );

    let context = no_corpus(root.path());
    let Err(err) = pipette_llamacpp::run(&req, &context, &no_readiness_gate, ignore_reps(), None)
    else {
        anyhow::bail!("a corpus cell without its corpus must fail");
    };

    let msg = format!("{err:#}");
    assert!(msg.contains("this test lends no corpus"), "got {msg}");
    assert_eq!(*context.asked.borrow(), [pinned]);
    Ok(())
}
