//! A multi-frame `vl_throughput` cell reports the helper's marker-free count, or none
//! when the runtime has no helper or the helper fails.

mod common;

use std::path::Path;

use anyhow::Context;
use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate,
};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkId, BenchmarkResultData, NonNegativeCount, PositiveCount,
    VlThroughputBenchmark,
};

/// A three-frame cell's `image_tokens`, with `install` placing the runtime's helper.
fn three_frame_image_tokens(
    install: impl FnOnce(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<Option<u32>> {
    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;
    let benchmark = BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-image-tokens-test".to_owned())?,
        parameter_image_width: PositiveCount::try_new(256)?,
        parameter_image_height: PositiveCount::try_new(256)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_num_images: NonNegativeCount::try_new(3)?,
    });
    let install_dir = fake_runtime_install_dir(root.path())?;
    install(&install_dir)?;
    let req = fake_run_request(
        fake_runtime(&install_dir)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark,
        None,
    );
    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("vl_throughput should complete against the fake server")?;
    let BenchmarkResultData::VlThroughput {
        prompt_tokens,
        image_tokens,
        ..
    } = outcome.result_data
    else {
        anyhow::bail!("expected VlThroughput result data");
    };
    // The fake's prompt_n is 5 + 10 per image.
    assert_eq!(prompt_tokens, 35);
    Ok(image_tokens)
}

/// Three frames merge to two: 14, not 3 x 7 and not the fake server's 30.
#[cfg(unix)]
#[test]
fn multi_frame_image_tokens_are_the_helpers_marker_free_total() -> anyhow::Result<()> {
    assert_eq!(
        three_frame_image_tokens(|install| common::install_fake_mtmd_cli(install, 7))?,
        Some(14)
    );
    Ok(())
}

#[test]
fn no_helper_reports_no_image_tokens() -> anyhow::Result<()> {
    assert_eq!(three_frame_image_tokens(|_| Ok(()))?, None);
    Ok(())
}

/// A helper that fails on one frame is not run again on all of them: it would
/// fail again, and a wedged one would hold the cell for a second deadline.
#[cfg(unix)]
#[test]
fn a_failed_one_frame_run_skips_the_frames_run() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let runs = tempfile::tempdir()?;
    let log = runs.path().join("runs");
    let image_tokens = three_frame_image_tokens(|install| {
        let helper = install.join("llama-mtmd-cli");
        std::fs::write(
            &helper,
            format!("#!/bin/sh\necho run >> '{}'\nexit 3\n", log.display()),
        )?;
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))?;
        Ok(())
    })?;
    assert_eq!(image_tokens, None);
    assert_eq!(std::fs::read_to_string(&log)?.lines().count(), 1);
    Ok(())
}
