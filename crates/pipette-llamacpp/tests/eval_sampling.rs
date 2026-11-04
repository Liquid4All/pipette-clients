//! The sampling a catalog entry pins, read back from the `/completion` bodies
//! the fake server received: checks the engine calls `eval_cell::Sampling` on
//! the request that leaves the process, and sends the preserved control tokens.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps, no_corpus,
    no_readiness_gate,
};
use serde_json::{Value, json};

use pipette_llamacpp::run;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, BenchmarkId, EvalBenchmark, MinP, PositiveCount,
    PositiveSeconds, SamplingSeed, SamplingTemperature, TopK, TopP,
};

/// Run one two-sample eval cell and hand back the `/completion` bodies the
/// server saw. The samples are the `#0` and `#1` pass@k repeats of one prompt.
fn completion_bodies(
    label: &str,
    pinned: bool,
    mcq_choices: Option<Vec<String>>,
) -> anyhow::Result<Vec<Value>> {
    let root = tempfile::tempdir()?;
    let bodies = root.path().join("completion-bodies.jsonl");
    // SAFETY: this file holds a single test, so no parallel test reads or
    // writes this var.
    unsafe { std::env::set_var("FAKE_LLAMA_BODY_FILE", &bodies) };

    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let mut eval = EvalBenchmark {
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
            json!({"id": "q#0", "messages": [{"role": "user", "content": "first draw"}]}),
            json!({"id": "q#1", "messages": [{"role": "user", "content": "second draw"}]}),
        ]),
    };
    if pinned {
        pin(&mut eval)?;
    }
    let benchmark = BenchmarkDefinition::Eval(eval);

    let install = fake_runtime_install_dir(root.path())?;
    let req = fake_run_request(
        fake_runtime(&install)?,
        fake_model(&model_path)?,
        benchmark,
        Some(BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
            http_timeout_seconds: Some(PositiveSeconds::try_new(15)?),
        }),
    );
    run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("the cell should complete both samples")?;

    std::fs::read_to_string(&bodies)
        .context("the fake server recorded no /completion body")?
        .lines()
        .map(|line| serde_json::from_str(line).context("a recorded body is not JSON"))
        .collect()
}

/// Every knob an eval entry can pin, at values none of the runtime defaults
/// share.
fn pin(eval: &mut EvalBenchmark) -> anyhow::Result<()> {
    eval.parameter_temperature = Some(SamplingTemperature::try_new(1.0)?);
    eval.parameter_top_p = Some(TopP::try_new(0.95)?);
    eval.parameter_top_k = Some(TopK::try_new(64)?);
    eval.parameter_min_p = Some(MinP::try_new(0.05)?);
    eval.parameter_seed = Some(SamplingSeed::try_new(7)?);
    Ok(())
}

/// The sampling keys present in a body, so an assertion catches an extra knob
/// as well as a missing one. The seed differs per `#k`, so it is read apart.
fn sampling_of(body: &Value) -> Value {
    Value::Object(
        ["temperature", "top_p", "top_k", "min_p"]
            .into_iter()
            .filter_map(|key| body.get(key).map(|value| (key.to_owned(), value.clone())))
            .collect(),
    )
}

/// Every case in one test, not one `#[test]` each: they share the env var
/// naming the recording file, and `cargo test` would run separate tests in
/// parallel and pool their bodies into one file.
#[test]
fn the_completion_body_carries_what_the_entry_pinned() -> anyhow::Result<()> {
    let sampled = serde_json::json!({
        "temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05,
    });

    let cases = [
        // The streaming arm and the grammar arm are two request builders; a
        // pin that reached only one would be invisible in a scored eval.
        (
            "streaming, pinned",
            "eval_math_500_2026.06.1_t1",
            true,
            None,
            sampled.clone(),
            vec![Some(7), Some(8)],
        ),
        (
            "grammar, pinned",
            "eval_mcq_t1",
            true,
            Some(vec!["A".to_string(), "B".to_string()]),
            sampled,
            vec![Some(7), Some(8)],
        ),
        // Unpinned: the eval-id policy's temperature, llama-server's defaults
        // for the rest.
        (
            "streaming, unpinned",
            "eval_math_500_2026.06.1",
            false,
            None,
            serde_json::json!({"temperature": 0.6}),
            vec![None, None],
        ),
    ];

    cases.into_iter().try_for_each(
        |(label, id, pinned, mcq_choices, expected, expected_seeds)| -> anyhow::Result<()> {
            let mcq = mcq_choices.is_some();
            let bodies = completion_bodies(id, pinned, mcq_choices)?;
            assert_eq!(bodies.len(), 2, "{label}: {bodies:?}");
            bodies.iter().for_each(|body| {
                assert_eq!(sampling_of(body), expected, "{label}: {body}");
                assert_eq!(
                    body.get("grammar").is_some(),
                    mcq,
                    "{label}: the grammar arm is what mcq_choices selects: {body}"
                );
                // North's reasoning and answer markers, which the scorer
                // splits on; llama-server drops control tokens otherwise.
                assert_eq!(
                    body["preserved_tokens"],
                    json!([
                        "<|START_THINKING|>",
                        "<|END_THINKING|>",
                        "<|START_TEXT|>",
                        "<|END_TEXT|>",
                    ]),
                    "{label}: {body}"
                );
            });
            // The base seed on both repeats would return one completion twice
            // and collapse pass@1 to the single-shot number.
            let mut seeds: Vec<Option<i64>> = bodies
                .iter()
                .map(|body| body.get("seed").and_then(Value::as_i64))
                .collect();
            seeds.sort_unstable();
            assert_eq!(seeds, expected_seeds, "{label}: {bodies:?}");
            Ok(())
        },
    )
}
