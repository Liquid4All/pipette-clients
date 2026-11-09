//! The end-to-end cell: a warm-up and one streamed request per rep carrying
//! the batch's prompts, and the server's log on the record. One test per file,
//! so the `FAKE_LLAMA_BODY_FILE` env var can't race a sibling.

mod common;

use std::cell::Cell;

use anyhow::Context;
use common::{
    assert_distinct_knobs, assert_server_log, fake_model, fake_run_request, fake_runtime,
    fake_runtime_install_dir, no_corpus, with_distinct_knobs,
};

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::BenchmarkResultData;
use pipette_plan_types::{BatchSize, BenchmarkDefinition, EndToEndLatencyBenchmark};
use pipette_plan_types::{BenchmarkId, NonNegativeCount, PositiveCount};

#[test]
fn an_end_to_end_rep_is_one_request_of_the_batch() -> anyhow::Result<()> {
    [1, 2].into_iter().try_for_each(one_batch)
}

fn one_batch(batch_size: u32) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let bodies = root.path().join("bodies.jsonl");
    // SAFETY: one test per file, so no parallel test reads this var.
    unsafe { std::env::set_var("FAKE_LLAMA_BODY_FILE", &bodies) };
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let id = match batch_size {
        1 => "end_to_end_latency_16_4".to_owned(),
        n => format!("end_to_end_latency_16_4_bs{n}"),
    };
    let benchmark = BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
        benchmark_id: BenchmarkId::try_new(id)?,
        parameter_prefill_tokens: NonNegativeCount::try_new(16)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_batch_size: BatchSize::try_new(batch_size)?,
    });
    let req = with_distinct_knobs(fake_run_request(
        fake_runtime(&fake_runtime_install_dir(root.path())?)?,
        fake_model(&model_path)?,
        benchmark,
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
        ..MeasurementObserver::QUIET
    };
    let response = pipette_llamacpp::run(&req, &no_corpus(root.path()), &gate, observer, None)?;

    // One gate per rep, after the warm-up: a sixth request but no rep.
    assert_eq!((gated.get(), started.get(), finished.get()), (5, 5, 5));
    let sent = std::fs::read_to_string(&bodies).context("no request reached the server")?;
    let sent = sent
        .lines()
        .map(serde_json::from_str::<serde_json::Value>)
        .collect::<Result<Vec<_>, _>>()?;
    let shapes = sent
        .iter()
        .map(|body| {
            (
                body["stream"].as_bool(),
                body["prompt"].as_array().map(Vec::len),
            )
        })
        .collect::<Vec<_>>();
    let prompts = usize::try_from(batch_size)?;
    assert_eq!(shapes, vec![(Some(true), Some(prompts)); 6], "{sent:?}");
    let parallel = response.command.windows(2).find_map(|pair| match pair {
        [flag, value] if flag == "--parallel" => Some(value.clone()),
        _ => None,
    });
    // `with_distinct_knobs` sets `no_cache`, which renders `--parallel 1` at a
    // batch of one.
    assert_eq!(parallel, Some(batch_size.to_string()));
    assert_distinct_knobs(&req.benchmark, &response.command)?;
    let BenchmarkResultData::EndToEndLatency { total_time_ms, .. } = response.result_data else {
        anyhow::bail!("not an end-to-end result: {:?}", response.result_data);
    };
    assert!(total_time_ms > 0.0, "{total_time_ms}");
    assert_server_log(&response.runtime_output)
}
