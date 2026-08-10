//! Companion to `eval_crash_recovery.rs`: the eval cell must also
//! recover when `/completion` fails against a server that is still
//! alive (Windows WSAECONNRESET / os error 10054, broken pipe, etc).
//! In that case `poll_child_exit` returns `None`, but the sample
//! must still be recorded as `failed: true` with a useful reason and
//! the server recycled so the next sample can run.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps,
    no_readiness_gate,
};
use serde_json::json;

use pipette_doomloop::plan::DoomloopOverrides;
use pipette_llamacpp::execute::run;
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::BenchmarkFlags;

#[test]
fn eval_recovers_from_connection_drop_with_live_server() -> anyhow::Result<()> {
    // The fake server closes the TCP connection mid-`/completion` (no
    // response written) but stays running. From the eval loop's
    // perspective the request errors out, yet `poll_child_exit`
    // never observes an exit — the "no exit observed" branch must
    // still mark the sample failed, recycle the server, and continue.
    const DROP_MARKER: &str = "PLEASE-DROP-CONNECTION";

    // SAFETY: integration tests run in their own process per file; this
    // file holds a single test so no parallel test reads these vars.
    std::env::set_var("FAKE_LLAMA_DROP_PROMPT_CONTAINS", DROP_MARKER);
    let pid_log = std::env::temp_dir().join(format!(
        "pipette-eval-connreset-pids-{}-{}.txt",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log);

    let root = tempfile::tempdir()?;
    let evals = EvalCompletionsStore::new(root.path().join("evals"));

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: "connreset-recovery-test".into(),
        parameter_eval_id: "connreset-recovery-test".into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: 16,
        parameter_mcq_choices: None,
        samples: Some(vec![
            json!({
                "id": "s-first",
                "messages": [{"role": "user", "content": "first sample"}],
            }),
            json!({
                "id": "s-drop",
                "messages": [{
                    "role": "user",
                    "content": format!("middle sample {DROP_MARKER}"),
                }],
            }),
            json!({
                "id": "s-third",
                "messages": [{"role": "user", "content": "third sample"}],
            }),
        ]),
    });

    let install = fake_runtime_install_dir(root.path())?;
    let runtime = fake_runtime(&install)?;
    let model = fake_model(&model_path)?;
    let benchmark_flags = BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
        http_timeout_seconds: Some(15),
        doomloop: DoomloopOverrides::default(),
    };
    let req = fake_run_request(runtime, model, benchmark, Some(benchmark_flags));

    let outcome = run(&req, &evals, &no_readiness_gate, &ignore_reps())
        .context("run should recover from a mid-stream connection drop")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    assert_eq!(completions.len(), 3, "all three samples must be present");
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    let first = by_id["s-first"];
    assert!(!first.failed, "first sample should succeed");
    assert_eq!(first.completion, "answer");

    let middle = by_id["s-drop"];
    assert!(middle.failed, "dropped sample must be marked failed");
    assert_eq!(middle.completion, "");
    let reason = middle.failed_reason.as_deref().unwrap_or("");
    assert!(
        reason.starts_with('['),
        "failed_reason should carry a timestamped prefix; got {reason:?}",
    );
    assert!(
        reason.contains("server still alive"),
        "failed_reason should mention the server-still-alive branch; got {reason:?}",
    );

    let third = by_id["s-third"];
    assert!(
        !third.failed,
        "post-failure sample must succeed against the same server",
    );
    assert_eq!(third.completion, "answer");

    assert!(
        outcome
            .stderr
            .starts_with("FAILED: benchmark=connreset-recovery-test failed=1 of 3"),
        "stderr should carry the FAILED summary; got: {:?}",
        outcome.stderr,
    );
    assert!(
        outcome.stderr.contains("id=s-drop"),
        "FAILED block should list the offending id; got: {:?}",
        outcome.stderr,
    );

    // Stronger discriminator: the server must NOT have been restarted
    // on the live-server path. The fake binary appends one line per
    // fresh process to FAKE_LLAMA_PID_FILE; exactly one line means
    // one server.
    let pids = std::fs::read_to_string(&pid_log).context("pid log should exist")?;
    let lines: Vec<&str> = pids.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        1,
        "live-server path must not restart llama-server; pid log: {pids:?}",
    );
    let _ = std::fs::remove_file(&pid_log);
    Ok(())
}
