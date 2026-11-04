//! Marker-free image token count for a vision prompt, from the model's own processor.

use std::path::Path;
use std::time::Duration;

use anyhow::Context;
use serde::Deserialize;

use pipette_plan_types::MlxMacosPipetteExtraRuntimeConfiguration;
use pipette_subprocess::start_process;

use crate::server;

const IMAGE_TOKENS_SCRIPT: &str = include_str!("python/pipette_mlx_image_tokens.py");
const SCRIPT_FILENAME: &str = "pipette_mlx_image_tokens.py";

const IMAGE_TOKENS_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, Deserialize)]
struct Reply {
    image_tokens: u32,
}

/// The image tokens `num_images` solid `width`×`height` frames take in one message.
pub(crate) fn count(
    venv_python: &Path,
    model_dir: &Path,
    width: u32,
    height: u32,
    num_images: u32,
    trust_remote_code: bool,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<u32> {
    // Run from a file rather than `-c`, so the logged command is one line, not
    // the whole script.
    let script_dir = tempfile::tempdir()
        .with_context(|| format!("could not create a temp dir for {SCRIPT_FILENAME}"))?;
    let script = script_dir.path().join(SCRIPT_FILENAME);
    std::fs::write(&script, IMAGE_TOKENS_SCRIPT)
        .with_context(|| format!("could not write {}", script.display()))?;
    let mut command = server::python(venv_python, client_config);
    command
        .arg(&script)
        .arg("--model-dir")
        .arg(model_dir)
        .args(["--width", &width.to_string()])
        .args(["--height", &height.to_string()])
        .args(["--num-images", &num_images.to_string()]);
    if trust_remote_code {
        command.arg("--trust-remote-code");
    }
    let mut process = start_process("pipette_mlx_image_tokens", command, Duration::ZERO)?;
    process.wait_success(IMAGE_TOKENS_TIMEOUT)?;
    parse(&process.output().runtime_stdout.unwrap_or_default())
}

/// Parses the last non-empty stdout line, since libraries may print above it.
fn parse(stdout: &str) -> anyhow::Result<u32> {
    let line = stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .context("the image token helper printed nothing")?;
    let reply: Reply = serde_json::from_str(line)
        .with_context(|| format!("failed to parse the image token helper's reply: {line}"))?;
    anyhow::ensure!(
        reply.image_tokens > 0,
        "the processor expanded the images into no placeholder tokens"
    );
    Ok(reply.image_tokens)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use rstest::rstest;

    use super::*;
    use crate::runtimes::test_support::live_vl;

    /// The fake interpreter checks both and answers 64 tokens a frame.
    #[test]
    fn the_script_runs_from_a_file_with_the_frame_count() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let python = dir.path().join("python");
        std::fs::write(
            &python,
            "#!/bin/sh\n\
             grep -q 'def placeholder_id' \"$1\" || exit 7\n\
             while [ $# -gt 0 ]; do [ \"$1\" = --num-images ] && n=$2; shift; done\n\
             echo \"{\\\"image_tokens\\\": $((n * 64))}\"\n",
        )?;
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))?;
        assert_eq!(
            count(&python, Path::new("/model"), 512, 512, 3, false, None)?,
            192
        );
        Ok(())
    }

    /// Needs Apple silicon, a venv holding `mlx-vlm` and an MLX VL model:
    ///
    /// ```text
    /// PIPETTE_MLX_LIVE_PYTHON=<venv>/bin/python PIPETTE_MLX_LIVE_TARGET=<dir> \
    ///   cargo test -p pipette-mlx --lib live_image_tokens -- --ignored --nocapture
    /// ```
    ///
    /// `mlx_vlm` does not merge frames, so three take three times one.
    #[test]
    #[ignore]
    fn live_image_tokens_scale_with_the_frames() -> anyhow::Result<()> {
        let (venv_python, model_dir) = live_vl()?;
        let frames = |num_images| {
            count(
                &venv_python,
                &model_dir,
                512,
                512,
                num_images,
                server::VISION_TRUSTS_REMOTE_CODE,
                None,
            )
        };
        let one = frames(1)?;
        let three = frames(3)?;
        eprintln!("{one} image tokens for one 512x512 frame, {three} for three");
        anyhow::ensure!(one > 1, "{one} image tokens for one frame");
        anyhow::ensure!(three == 3 * one, "{three} for three frames, {one} for one");
        Ok(())
    }

    #[rstest]
    #[case::reply("{\"image_tokens\": 256}\n", Ok(256))]
    #[case::after_library_noise("loading...\n{\"image_tokens\": 64}\n\n", Ok(64))]
    #[case::nothing("", Err("the image token helper printed nothing"))]
    #[case::zero(
        "{\"image_tokens\": 0}",
        Err("the processor expanded the images into no placeholder tokens")
    )]
    #[case::garbled("oops", Err("failed to parse the image token helper's reply: oops"))]
    fn the_reply_is_the_last_line(
        #[case] stdout: &str,
        #[case] expected: Result<u32, &str>,
    ) -> anyhow::Result<()> {
        match (parse(stdout), expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(err), Err(want)) => assert!(format!("{err:#}").starts_with(want), "{err:#}"),
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }
}
