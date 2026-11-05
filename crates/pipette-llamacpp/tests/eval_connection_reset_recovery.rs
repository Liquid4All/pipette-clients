//! An eval cell whose `/completion` loses its connection to a live server
//! (a pooled connection closed under the request, Windows WSAECONNRESET)
//! resends on the same server; a sample that keeps dropping is recorded
//! failed once its resends run out, and the next sample still runs.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps, no_corpus,
    no_readiness_gate,
};
use serde_json::json;

use pipette_llamacpp::run;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, BenchmarkId, BenchmarkResultData, EvalBenchmark,
    PositiveCount, PositiveSeconds,
};

#[test]
fn eval_recovers_from_connection_drop_with_live_server() -> anyhow::Result<()> {
    const DROP_ONCE_MARKER: &str = "DROP-ONCE-PLEASE";
    const DROP_MARKER: &str = "PLEASE-DROP-CONNECTION";

    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_DROP_ONCE_PROMPT_CONTAINS", DROP_ONCE_MARKER) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_DROP_PROMPT_CONTAINS", DROP_MARKER) };
    let scratch = |label: &str| {
        std::env::temp_dir().join(format!(
            "pipette-eval-connreset-{label}-{}-{}.txt",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    };
    let pid_log = scratch("pids");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log) };
    let body_log = scratch("bodies");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_BODY_FILE", &body_log) };

    let root = tempfile::tempdir()?;

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: BenchmarkId::try_new("connreset-recovery-test".to_owned())?,
        parameter_eval_id: "connreset-recovery-test".into(),
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
                "id": "s-drop-once",
                "messages": [{
                    "role": "user",
                    "content": format!("second sample {DROP_ONCE_MARKER}"),
                }],
            }),
            json!({
                "id": "s-drop",
                "messages": [{
                    "role": "user",
                    "content": format!("third sample {DROP_MARKER}"),
                }],
            }),
            json!({
                "id": "s-last",
                "messages": [{"role": "user", "content": "last sample"}],
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

    let outcome = run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("run should recover from a mid-stream connection drop")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    assert_eq!(completions.len(), 4, "all four samples must be present");
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    let first = by_id["s-first"];
    assert!(!first.failed, "first sample should succeed");
    assert_eq!(first.completion, "answer");

    let resent = by_id["s-drop-once"];
    assert!(
        !resent.failed,
        "a sample whose connection dropped once must succeed when sent again; reason={:?}",
        resent.failed_reason,
    );
    assert_eq!(resent.completion, "answer");

    let dropped = by_id["s-drop"];
    assert!(
        dropped.failed,
        "a sample that keeps dropping must be marked failed"
    );
    assert_eq!(dropped.completion, "");
    let reason = dropped.failed_reason.as_deref().unwrap_or("");
    assert!(
        reason.starts_with('['),
        "failed_reason should carry a timestamped prefix; got {reason:?}",
    );
    assert!(
        reason.contains("server still alive"),
        "failed_reason should mention the server-still-alive branch; got {reason:?}",
    );

    let last = by_id["s-last"];
    assert!(
        !last.failed,
        "post-failure sample must succeed against the same server",
    );
    assert_eq!(last.completion, "answer");

    // One resend for the sample dropped once, two for the one dropped every time.
    let bodies = std::fs::read_to_string(&body_log).context("body log should exist")?;
    let sends = |marker: &str| bodies.lines().filter(|l| l.contains(marker)).count();
    assert_eq!(sends(DROP_ONCE_MARKER), 2, "bodies: {bodies}");
    assert_eq!(sends(DROP_MARKER), 3, "bodies: {bodies}");

    let pids = std::fs::read_to_string(&pid_log).context("pid log should exist")?;
    let lines: Vec<&str> = pids.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        1,
        "live-server path must not restart llama-server; pid log: {pids:?}",
    );
    let _ = std::fs::remove_file(&pid_log);
    let _ = std::fs::remove_file(&body_log);
    Ok(())
}
