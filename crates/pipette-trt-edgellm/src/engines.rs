//! The engine build: `trt-edgellm-server --build-only` into the runtime's
//! compile cache, before any measured window and outside every memory sampler,
//! under a deadline of its own. The server keys its engines by the GGUF pair,
//! precision, TensorRT, CUDA, L4T, SoC and its own code, so a warm cache makes
//! this a check of a few seconds.

use std::time::{Duration, Instant};

use anyhow::Context;

use pipette_subprocess::start_process;

use crate::server::Launch;

/// A cold build of a 2B VLA on an AGX Orin takes about 6 minutes (conversion,
/// then the vision, language and VLA engines); this leaves room for a slower
/// board or a bigger model.
const BUILD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

pub fn build(launch: &Launch) -> anyhow::Result<()> {
    let started = Instant::now();
    tracing::info!(
        timeout = ?BUILD_TIMEOUT,
        "building TensorRT engines if the cache has none (minutes on a cold cache)"
    );
    let mut builder = start_process(
        "trt-edgellm-server --build-only",
        launch.build_only(),
        Duration::ZERO,
    )?;
    builder
        .wait_success(BUILD_TIMEOUT)
        .context("building the TensorRT engines failed")?;
    tracing::info!(elapsed = ?started.elapsed(), "TensorRT engines ready");
    Ok(())
}
