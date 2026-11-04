//! `vl_throughput`'s calibration helper loads where the cell's server does, so
//! it gets the cell's GPU layers.
#![cfg(unix)]

mod common;

use std::os::unix::fs::PermissionsExt;

use anyhow::Context;
use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate, with_distinct_knobs,
};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkId, NonNegativeCount, PositiveCount, VlThroughputBenchmark,
};

/// [`with_distinct_knobs`] sets 7 GPU layers.
#[test]
fn the_calibration_helper_gets_the_cells_gpu_layers() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;
    let benchmark = BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-helper-placement-test".to_owned())?,
        parameter_image_width: PositiveCount::try_new(256)?,
        parameter_image_height: PositiveCount::try_new(256)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    });
    let install = fake_runtime_install_dir(root.path())?;
    let helper_args = root.path().join("helper-args");
    let helper = install.join("llama-mtmd-cli");
    std::fs::write(
        &helper,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n\
             echo 'decoding image batch 1/1, n_tokens_batch = 10' >&2\n",
            helper_args.display()
        ),
    )?;
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))?;
    let req = with_distinct_knobs(fake_run_request(
        fake_runtime(&install)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark,
        None,
    ))?;

    pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("vl_throughput should complete against the fake server")?;
    let args = std::fs::read_to_string(&helper_args)?;
    let args = args.lines().collect::<Vec<_>>();
    assert!(
        args.windows(2).any(|pair| pair == ["--n-gpu-layers", "7"]),
        "{args:?}"
    );
    Ok(())
}
