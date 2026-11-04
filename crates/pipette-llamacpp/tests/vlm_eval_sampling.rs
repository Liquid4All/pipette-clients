//! Every vlm_eval request sends the sampling its entry pins and every other
//! sampler at its neutral value, so a GGUF's `general.sampling.*` defaults
//! cannot change the decode. An entry that pins nothing decodes greedily.

mod common;

use anyhow::Context;
use common::{ignore_reps, no_readiness_gate, sample_id, temp_store, vlm_pool_request, vlm_sample};
use serde_json::{Value, json};

use pipette_llamacpp::run;
use pipette_plan_types::{
    BenchmarkDefinition, PresencePenalty, RunRequest, SamplingSeed, SamplingTemperature, SeedRule,
    TopK, TopP,
};

const NEUTRAL: [(&str, f64); 8] = [
    ("min_p", 0.0),
    ("typical_p", 1.0),
    ("top_n_sigma", -1.0),
    ("repeat_penalty", 1.0),
    ("frequency_penalty", 0.0),
    ("dry_multiplier", 0.0),
    ("xtc_probability", 0.0),
    ("mirostat", 0.0),
];

/// The request bodies of one run, read from the fake server's call log.
fn run_and_log(label: &str, edit: impl FnOnce(&mut RunRequest)) -> anyhow::Result<Vec<Value>> {
    let store = temp_store(label)?;
    let call_log = store.root.join("calls.log");
    // SAFETY: this file holds one test, and each run's server reads the var
    // when it starts.
    unsafe { std::env::set_var("FAKE_LLAMA_CALL_LOG", &call_log) };
    let samples = vec![vlm_sample(0, "first"), vlm_sample(1, "second")];
    let mut req = vlm_pool_request(&store.root, label, samples, 1, None)?;
    edit(&mut req);
    run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .with_context(|| format!("the {label} run should succeed"))?;
    let log = std::fs::read_to_string(&call_log).context("the call log should exist")?;
    let bodies = log
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(bodies.len() == 2, "{log}");
    Ok(bodies)
}

fn assert_fields(body: &Value, expected: &[(&str, f64)]) {
    for (key, value) in expected {
        assert_eq!(
            body.get(*key).and_then(Value::as_f64),
            Some(*value),
            "{key} in {body}"
        );
    }
}

#[test]
fn requests_send_the_pinned_sampling_and_neutral_rest() -> anyhow::Result<()> {
    for body in run_and_log("vlm-greedy", |_| {})? {
        assert_fields(
            &body,
            &[
                ("temperature", 0.0),
                ("top_k", 0.0),
                ("top_p", 1.0),
                ("presence_penalty", 0.0),
            ],
        );
        assert_fields(&body, &NEUTRAL);
        assert_eq!(body.get("seed"), None, "{body}");
        assert_eq!(body.get("max_tokens"), Some(&json!(16)), "{body}");
    }

    let pinned = run_and_log("vlm-pinned", |req| {
        if let BenchmarkDefinition::VlmEval(b) = &mut req.benchmark {
            b.parameter_temperature = SamplingTemperature::try_new(0.7).ok();
            b.parameter_top_p = TopP::try_new(0.8).ok();
            b.parameter_top_k = TopK::try_new(20).ok();
            b.parameter_presence_penalty = PresencePenalty::try_new(1.5).ok();
            b.parameter_seed = SamplingSeed::try_new(7).ok();
        }
    })?;
    for body in pinned {
        assert_fields(
            &body,
            &[
                ("temperature", 0.7),
                ("top_k", 20.0),
                ("top_p", 0.8),
                ("presence_penalty", 1.5),
            ],
        );
        assert_fields(&body, &NEUTRAL);
        // No seed rule: every sample of the attempt draws with the base seed.
        assert_eq!(body.get("seed"), Some(&json!(7)), "{body}");
    }

    // An entry on the per-sample rule sends each sample the seed
    // `VlmEvalBenchmark::sample_seed` derives for its id; the body names no
    // id, so the two are compared as a set.
    let mut entry = None;
    let mut seeds: Vec<_> = run_and_log("vlm-seed-rule", |req| {
        if let BenchmarkDefinition::VlmEval(b) = &mut req.benchmark {
            b.parameter_seed = SamplingSeed::try_new(7).ok();
            b.parameter_seed_rule = Some(SeedRule::PerSampleSha256);
            entry = Some(b.clone());
        }
    })?
    .into_iter()
    .map(|body| body.get("seed").and_then(Value::as_i64))
    .collect();
    let entry = entry.context("a vlm_eval request")?;
    let mut expected: Vec<_> = (0..2).map(|i| entry.sample_seed(&sample_id(i))).collect();
    seeds.sort_unstable();
    expected.sort_unstable();
    assert_eq!(seeds, expected);
    assert_ne!(seeds.first(), seeds.last(), "two samples, two seeds");
    Ok(())
}
