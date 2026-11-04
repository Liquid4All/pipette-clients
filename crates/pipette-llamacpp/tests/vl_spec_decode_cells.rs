//! The corpus cell with speculation off: every turn of the corpus to the row,
//! the per-turn record and summary beside it, and the server's log.

mod common;

use std::cell::Cell;
use std::path::PathBuf;

use common::{
    assert_distinct_knobs, assert_server_log, fake_run_request, fake_runtime,
    fake_runtime_install_dir, fake_vision_model, with_distinct_knobs,
};
use serde_json::json;

use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkResultData, Dataset, Sha256, VlSpecDecodeBenchmark,
};

/// A client that serves one corpus tar, unpacked into a directory of the
/// test's, where evals would checkpoint too.
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

/// An image turn under a system prompt, then a two-turn text conversation,
/// packed.
fn corpus_tar() -> anyhow::Result<Vec<u8>> {
    let manifest = json!({
        "format": "vl-spec-decode-corpus/1",
        "adapter": "mmspec@1",
        "upstream": {
            "source": "git",
            "url": "github.com/acme/corpus",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "identity": "2923603292c7dd463ad01a119f57965817778fc3",
            "digest": "2222222222222222222222222222222222222222222222222222222222222222",
        },
        "samples": 2,
        "turns": 3,
        "subsets": { "GQA": { "samples": 2, "turns": 3 } },
    });
    let samples = [
        json!({"id": "img", "subset": "GQA", "system": "be brief", "turns": [[
            {"type": "image", "path": "media/aa.png", "mime": "image/png", "bytes": 3},
            {"type": "text", "text": "what is this?"},
        ]]}),
        json!({"id": "multi", "subset": "GQA", "system": "", "turns": [
            [{"type": "text", "text": "hello"}],
            [{"type": "text", "text": "and again"}],
        ]}),
    ]
    .iter()
    .map(|sample| format!("{sample}\n"))
    .collect::<String>();
    let mut builder = tar::Builder::new(Vec::new());
    let mut append = |path: &str, body: &[u8]| -> anyhow::Result<()> {
        let mut header = tar::Header::new_gnu();
        header.set_size(u64::try_from(body.len())?);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, body)?;
        Ok(())
    };
    append("manifest.json", manifest.to_string().as_bytes())?;
    append("samples.jsonl", samples.as_bytes())?;
    append("media/aa.png", b"png")?;
    Ok(builder.into_inner()?)
}

fn benchmark(tar: &[u8]) -> anyhow::Result<BenchmarkDefinition> {
    let digest = Sha256::try_new(hex::encode(<sha2::Sha256 as sha2::Digest>::digest(tar)))?;
    Ok(BenchmarkDefinition::VlSpecDecode(serde_json::from_value::<
        VlSpecDecodeBenchmark,
    >(json!({
        "benchmark_id": "vl_spec_decode_mmspec_test_gqa_k8",
        "parameter_draft_length": 8,
        "parameter_max_tokens": 16,
        "dataset": {
            "format": "vl-spec-decode-corpus/1",
            "adapter": "mmspec@1",
            "digest": digest,
            "bytes": tar.len(),
            "upstream": {
                "source": "git",
                "url": "github.com/acme/corpus",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "identity": "2923603292c7dd463ad01a119f57965817778fc3",
                "digest": "2222222222222222222222222222222222222222222222222222222222222222",
            },
        },
    }))?))
}

#[test]
fn a_corpus_cell_traverses_through_its_entry() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let serving = Serving {
        tar: corpus_tar()?,
        dir: root.path().join("datasets"),
    };
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;
    let req = with_distinct_knobs(fake_run_request(
        fake_runtime(&fake_runtime_install_dir(root.path())?)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark(&serving.tar)?,
        None,
    ))?;

    let (gated, started, finished) = (Cell::new(0), Cell::new(0), Cell::new(0));
    let gate = || {
        gated.set(gated.get() + 1);
        Ok(())
    };
    let observer = MeasurementObserver {
        on_before_rep_start: &|| {
            started.set(started.get() + 1);
            Ok(())
        },
        on_after_rep_end: &|| {
            finished.set(finished.get() + 1);
            Ok(())
        },
    };
    let response = pipette_llamacpp::run(&req, &serving, &gate, observer, None)?;

    // A gate per sample; the traversal is the one rep.
    assert_eq!((gated.get(), started.get(), finished.get()), (2, 1, 1));
    assert_distinct_knobs(&req.benchmark, &response.command)?;
    let BenchmarkResultData::VlSpecDecode {
        vl_spec_decode_turns,
        vl_spec_decode_generated_tokens,
        vl_spec_decode_drafted_tokens,
        ..
    } = response.result_data
    else {
        anyhow::bail!("not a VL spec-decode result: {:?}", response.result_data);
    };
    // The fake answers three tokens a turn and drafts none.
    assert_eq!(
        (
            vl_spec_decode_turns,
            vl_spec_decode_generated_tokens,
            vl_spec_decode_drafted_tokens
        ),
        (3, 9, None)
    );
    assert_eq!(
        response
            .turns
            .iter()
            .map(|turn| (turn.sample_id.as_str(), turn.turn))
            .collect::<Vec<_>>(),
        [("img", 0), ("multi", 0), ("multi", 1)]
    );
    assert!(response.output_digest.is_some());
    assert_eq!(response.client_extras.get("turns"), Some(&json!(3)));
    assert_server_log(&response.runtime_output)
}
