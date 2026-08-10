//! End-to-end integration test for issue #103: the eval cell must not
//! abort when `llama-server` dies mid-`/completion`. Drives
//! `execute::run` with a fake `llama-server`
//! (`tests/bin/fake_llama_server.rs`) that exits with status 139 on a
//! marked sample, and asserts the run still produces a result with
//! every sample accounted for — the crashing one carrying
//! `failed: true`.

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
use pipette_plan_types::result::{BenchmarkEvalCompletionStopReason, BenchmarkResultData};
use pipette_plan_types::BenchmarkFlags;

#[test]
fn eval_recovers_from_mid_run_server_crash() -> anyhow::Result<()> {
    // The crash marker appears only in the middle sample's user message.
    // The fake server inspects /completion bodies for it and calls
    // `process::exit(139)` on match — same observable shape as a real
    // mid-`/completion` crash.
    const CRASH_MARKER: &str = "PLEASE-CRASH-NOW";

    // SAFETY: integration tests run in their own process per file; this
    // file holds a single test so no parallel test reads these vars.
    std::env::set_var("FAKE_LLAMA_CRASH_PROMPT_CONTAINS", CRASH_MARKER);
    let pid_log = std::env::temp_dir().join(format!(
        "pipette-eval-crash-pids-{}-{}.txt",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log);

    let root = tempfile::tempdir()?;
    let evals = EvalCompletionsStore::new(root.path().join("evals"));

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: "crash-recovery-test".into(),
        parameter_eval_id: "crash-recovery-test".into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: 16,
        parameter_mcq_choices: None,
        samples: Some(vec![
            json!({
                "id": "s-first",
                "messages": [{"role": "user", "content": "first sample"}],
            }),
            json!({
                "id": "s-crash",
                "messages": [{
                    "role": "user",
                    "content": format!("middle sample {CRASH_MARKER}"),
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
        .context("run should recover from the mid-stream crash")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    assert_eq!(completions.len(), 3, "all three samples must be present");
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    let first = by_id["s-first"];
    assert!(!first.failed, "first sample should succeed");
    assert_eq!(first.completion, "answer");
    // A natural stop below the cap classifies as `eos`, with the server's
    // token count carried through.
    assert_eq!(first.stop_reason, BenchmarkEvalCompletionStopReason::Eos);
    assert_eq!(first.completion_tokens, Some(3));
    assert_eq!(first.stop_detail, None);

    let middle = by_id["s-crash"];
    assert!(middle.failed, "crashing sample must be marked failed");
    assert_eq!(middle.completion, "");
    // The client owns the reason: a crashed sample is `failure`, with the
    // crash detail dual-written to `stop_detail`.
    assert_eq!(
        middle.stop_reason,
        BenchmarkEvalCompletionStopReason::Failure
    );
    assert_eq!(middle.stop_detail, middle.failed_reason);
    let reason = middle.failed_reason.as_deref().unwrap_or("");
    assert!(
        reason.starts_with('['),
        "failed_reason should carry a timestamped prefix; got {reason:?}",
    );
    assert!(
        reason.contains("crashed mid-completion"),
        "failed_reason should mention the recovery branch; got {reason:?}",
    );

    let third = by_id["s-third"];
    assert!(
        !third.failed,
        "post-restart sample must succeed against the fresh server",
    );
    assert_eq!(third.completion, "answer");
    assert_eq!(third.stop_reason, BenchmarkEvalCompletionStopReason::Eos);

    // Persisted operator signal: the FAILED block goes to stderr, which
    // the caller writes to `result_extras_path` alongside the payload.
    assert!(
        outcome
            .stderr
            .starts_with("FAILED: benchmark=crash-recovery-test failed=1 of 3"),
        "stderr should carry the FAILED summary; got: {:?}",
        outcome.stderr,
    );
    assert!(
        outcome.stderr.contains("id=s-crash"),
        "FAILED block should list the offending id; got: {:?}",
        outcome.stderr,
    );

    // Finalize keeps the checkpoint file iff there were failed entries,
    // so a re-run against the same plan_digest would skip the crasher.
    let jsonls: Vec<_> = std::fs::read_dir(evals.root())?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .collect();
    assert_eq!(
        jsonls.len(),
        1,
        "exactly one checkpoint file should survive finalize",
    );

    // Crash path must restart the server. The fake binary appends one
    // line per fresh process to FAKE_LLAMA_PID_FILE; two distinct PIDs
    // mean initial-spawn + post-crash restart.
    let pids = std::fs::read_to_string(&pid_log).context("pid log should exist")?;
    let lines: Vec<&str> = pids.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        2,
        "crash path must restart llama-server exactly once; pid log: {pids:?}",
    );
    assert_ne!(
        lines[0], lines[1],
        "the restarted server must be a fresh process"
    );
    let _ = std::fs::remove_file(&pid_log);
    Ok(())
}
