//! The model flags an eval cell renders its prompt under, read back from the
//! `/apply-template` bodies the fake server received: both completion arms
//! must render through `apply_template_body`.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps, no_corpus,
    no_readiness_gate,
};
use serde_json::{Map, Value, json};

use pipette_llamacpp::run;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, BenchmarkId, EvalBenchmark, ModelFlags, PositiveCount,
    PositiveSeconds, ReasoningEffort,
};

/// The messages of the cell's two samples, in the order they are sent.
fn sample_messages() -> [Value; 2] {
    [
        json!([{"role": "user", "content": "first draw"}]),
        json!([{"role": "user", "content": "second draw"}]),
    ]
}

/// Run one two-sample eval cell under `model_flags` and hand back the
/// `/apply-template` bodies the server saw, in arrival order.
fn template_bodies(
    label: &str,
    mcq_choices: Option<Vec<String>>,
    model_flags: Option<ModelFlags>,
) -> anyhow::Result<Vec<Value>> {
    let root = tempfile::tempdir()?;
    let bodies = root.path().join("template-bodies.jsonl");
    // SAFETY: this file holds a single test, so no parallel test reads or
    // writes this var.
    unsafe { std::env::set_var("FAKE_LLAMA_TEMPLATE_BODY_FILE", &bodies) };

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let [first, second] = sample_messages();
    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: BenchmarkId::try_new(label.to_owned())?,
        parameter_eval_id: "math_500".into(),
        parameter_dataset_name: "2026.06.1".into(),
        parameter_max_tokens: PositiveCount::try_new(8)?,
        parameter_max_context: None,
        parameter_mcq_choices: mcq_choices,
        parameter_temperature: None,
        parameter_top_p: None,
        parameter_top_k: None,
        parameter_min_p: None,
        parameter_seed: None,
        samples: Some(vec![
            json!({"id": "q#0", "messages": first}),
            json!({"id": "q#1", "messages": second}),
        ]),
    });

    let install = fake_runtime_install_dir(root.path())?;
    let mut req = fake_run_request(
        fake_runtime(&install)?,
        fake_model(&model_path)?,
        benchmark,
        Some(BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
            http_timeout_seconds: Some(PositiveSeconds::try_new(15)?),
        }),
    );
    req.model_flags = model_flags;
    run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("the cell should complete both samples")?;

    std::fs::read_to_string(&bodies)
        .context("the fake server recorded no /apply-template body")?
        .lines()
        .map(|line| serde_json::from_str(line).context("a recorded body is not JSON"))
        .collect()
}

fn gguf_text(
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
) -> Option<ModelFlags> {
    Some(ModelFlags::EvalGgufText {
        enable_thinking,
        reasoning_effort,
    })
}

/// Every case in one test, not one `#[test]` each: they share the env var
/// naming the recording file, and `cargo test` would run separate tests in
/// parallel and pool their bodies into one file.
#[test]
fn the_template_body_carries_the_cells_model_flags() -> anyhow::Result<()> {
    let cases = [
        // No flags: the prompt of every cell that sets none is unchanged.
        ("streaming, no flags", None, None, json!({})),
        (
            "streaming, thinking and effort",
            None,
            gguf_text(Some(true), Some(ReasoningEffort::Low)),
            json!({
                "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"},
                "reasoning_effort": "low",
            }),
        ),
        // The grammar arm renders through the same template call; an effort
        // that reached only one arm would be invisible in a scored eval.
        (
            "grammar, effort alone",
            Some(vec!["A".to_string(), "B".to_string()]),
            gguf_text(None, Some(ReasoningEffort::High)),
            json!({
                "chat_template_kwargs": {"reasoning_effort": "high"},
                "reasoning_effort": "high",
            }),
        ),
    ];

    cases.into_iter().try_for_each(
        |(label, mcq_choices, model_flags, expected)| -> anyhow::Result<()> {
            let bodies = template_bodies("eval_math_500_2026.06.1", mcq_choices, model_flags)?;
            assert_eq!(bodies.len(), 2, "{label}: {bodies:?}");
            bodies
                .iter()
                .zip(sample_messages())
                .for_each(|(body, messages)| {
                    let mut flags = body.as_object().cloned().unwrap_or_else(Map::new);
                    assert_eq!(flags.remove("messages"), Some(messages), "{label}: {body}");
                    assert_eq!(Value::Object(flags), expected, "{label}: {body}");
                });
            Ok(())
        },
    )
}
