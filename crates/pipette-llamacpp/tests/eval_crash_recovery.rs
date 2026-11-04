//! An eval cell must not abort when `llama-server` dies mid-`/completion`:
//! the run restarts the server and records every sample, the crashed one
//! as failed.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps, no_corpus,
    no_readiness_gate,
};
use serde_json::json;

use pipette_llamacpp::run;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkEvalCompletionStopReason, BenchmarkFlags, BenchmarkId,
    BenchmarkResultData, EvalBenchmark, PositiveCount, PositiveSeconds,
};

#[test]
fn eval_recovers_from_mid_run_server_crash() -> anyhow::Result<()> {
    const CRASH_MARKER: &str = "PLEASE-CRASH-NOW";

    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_CRASH_PROMPT_CONTAINS", CRASH_MARKER) };
    let pid_log = std::env::temp_dir().join(format!(
        "pipette-eval-crash-pids-{}-{}.txt",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log) };

    let root = tempfile::tempdir()?;
    let run_context = no_corpus(root.path());

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: BenchmarkId::try_new("crash-recovery-test".to_owned())?,
        parameter_eval_id: "crash-recovery-test".into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: PositiveCount::try_new(16)?,
        parameter_max_context: None,
        parameter_mcq_choices: None,
        parameter_temperature: None,
        parameter_top_p: None,
        parameter_top_k: None,
        parameter_min_p: None,
        parameter_seed: None,
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
        http_timeout_seconds: Some(PositiveSeconds::try_new(15)?),
    };
    let req = fake_run_request(runtime, model, benchmark, Some(benchmark_flags));

    let outcome = run(&req, &run_context, &no_readiness_gate, ignore_reps(), None)
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
    assert_eq!(first.stop_reason, BenchmarkEvalCompletionStopReason::Eos);
    assert_eq!(first.completion_tokens, Some(3));
    assert_eq!(first.stop_detail, None);

    let middle = by_id["s-crash"];
    assert!(middle.failed, "crashing sample must be marked failed");
    assert_eq!(middle.completion, "");
    // The crash detail is written to both `failed_reason` and `stop_detail`.
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

    // The run leaves its checkpoint, crashed row included, for the client to
    // finalize once the result is safe.
    let jsonls: Vec<_> = std::fs::read_dir(run_context.eval_completions().root())?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .collect();
    assert_eq!(
        jsonls.len(),
        1,
        "exactly one checkpoint file should survive the run",
    );

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

    // The response carries the replacement server's argv and log.
    let port = outcome
        .command
        .windows(2)
        .find_map(|pair| match pair {
            [flag, port] if flag == "--port" => Some(port.as_str()),
            _ => None,
        })
        .context("the response names no --port")?;
    let stderr = outcome
        .runtime_output
        .runtime_stderr
        .as_deref()
        .unwrap_or_default();
    assert!(
        stderr.contains(&format!("listening on 127.0.0.1:{port}, pid {}", lines[1])),
        "{stderr}"
    );
    Ok(())
}
