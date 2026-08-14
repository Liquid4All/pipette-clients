use std::{io::Write, time::Duration};

use anyhow::Context;
use base64::Engine;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;

use pipette_http::HttpClient;
use pipette_ops::measurement;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::reserved_flags::llamacpp_cli_stock_tools as reserved;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;

use crate::models::require_gguf_vision;
use crate::runtime_flags::{self, MmapPolicy};
use crate::runtimes::require_llama_server;
use crate::server;

/// Default HTTP timeout when `benchmark_flags.http_timeout` is unset.
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// Marker token that llama.cpp multimodal expects in the prompt string to
/// indicate where each image should be inserted.
const MEDIA_MARKER: &str = "<__media__>";

/// VL throughput cell: bound `llama-server` + GGUF vision, typed [`VlThroughput`] body.
pub fn run(
    req: &RunRequest,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_vl_throughput()
        .map_err(anyhow::Error::from)?;
    let llama_server = require_llama_server(req)?;
    let (model_path, mmproj_path) = require_gguf_vision(req)?;
    let flags = runtime_flags::for_server(
        req,
        {
            // ~1 token per 14x14 patch + text + decode.
            let image_tokens =
                (benchmark.parameter_image_width / 14) * (benchmark.parameter_image_height / 14);
            image_tokens
                .saturating_add(benchmark.parameter_text_tokens)
                .saturating_add(benchmark.parameter_decode_tokens)
                .max(8192)
        },
        MmapPolicy::PinInRam,
    )?;
    let extra_flags = server::args_for(&flags).build(reserved::SERVER, "vl_throughput")?;

    readiness_gate()?;
    let params = VlBenchmarkParams {
        image_width: benchmark.parameter_image_width,
        image_height: benchmark.parameter_image_height,
        text_tokens: benchmark.parameter_text_tokens,
        decode_tokens: benchmark.parameter_decode_tokens,
    };
    let http_timeout = http_timeout_from_req(req);

    let mut server = server::start(&llama_server, &model_path, Some(&mmproj_path), &extra_flags)?;
    server.observe_memory();

    let result = (|| -> anyhow::Result<RunResponse> {
        if let Err(e) = server::wait_until_ready(&server.base_url, &mut server.child, http_timeout)
        {
            let stderr = server::shutdown_and_collect_stderr(&mut server);
            if stderr.is_empty() {
                anyhow::bail!("{e}");
            } else {
                anyhow::bail!("{e}\nserver stderr:\n{stderr}");
            }
        }

        let client = HttpClient::blocking_with_timeout("pipette", http_timeout)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let eog_token_ids = server::discover_eog_token_ids(&server);
        let text_prompt = build_text_prompt(&client, &server.base_url, params.text_tokens)?;
        let image_b64 = generate_image_b64(params.image_width, params.image_height)?;

        // Warm-up run — validate to catch misconfiguration early.
        log::info!("vl_throughput: warm-up run");
        let warmup_resp = send_vl_completion(
            &client,
            &server.base_url,
            &text_prompt,
            &image_b64,
            params.decode_tokens,
            &eog_token_ids,
        )?;
        validate_response(&warmup_resp, params.decode_tokens)?;

        let measured = measurement::run(
            "vl_throughput",
            readiness_gate,
            observer,
            // No untimed per-rep setup: the server holds no state a rep resets.
            |_| Ok(()),
            |_| {
                send_vl_completion(
                    &client,
                    &server.base_url,
                    &text_prompt,
                    &image_b64,
                    params.decode_tokens,
                    &eog_token_ids,
                )
            },
            |_, rep| {
                validate_response(&rep.value, params.decode_tokens)?;
                Ok(rep.value.timings.prompt_ms)
            },
        )?;

        // Every rep sends the same prompt, so the token count is read off a
        // rep rather than reduced.
        let prompt_tokens = measured
            .first()
            .map(|resp| resp.timings.prompt_n)
            .context("vl_throughput measured no repetitions")?;
        let prompt = measured.stats();
        let predicted = measured.metric("predicted_ms", |rep| rep.value.timings.predicted_ms);

        // Both metrics' samples and means are already logged by the harness;
        // the token count is the one thing it cannot know.
        log::info!("vl_throughput: prompt_tokens={prompt_tokens}");

        Ok(RunResponse {
            executable: Some(llama_server.display().to_string()),
            command: server.command_preview.clone(),
            runtime_flags: Some(flags.clone()),
            memory: server.memory_observation(),
            ..RunResponse::new(
                BenchmarkResultData::VlThroughput {
                    prompt_tokens,
                    prompt_ms: prompt.mean_ms,
                    prompt_ms_stddev: Some(prompt.stddev_ms),
                    predicted_ms: predicted.mean_ms,
                    predicted_ms_stddev: Some(predicted.stddev_ms),
                },
                String::new(),
                String::new(),
            )
        })
    })();

    let _ = server.child.kill();
    let _ = server.child.wait();
    result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct VlBenchmarkParams {
    image_width: u32,
    image_height: u32,
    text_tokens: u32,
    decode_tokens: u32,
}

fn http_timeout_from_req(req: &RunRequest) -> Duration {
    Duration::from_secs(
        req.benchmark_flags
            .as_ref()
            .and_then(|f| f.http_timeout())
            .unwrap_or(DEFAULT_HTTP_TIMEOUT_SECS),
    )
}

fn build_text_prompt(client: &Client, base_url: &str, token_count: u32) -> anyhow::Result<String> {
    if token_count == 0 {
        return Ok(String::new());
    }
    let target = token_count as usize;

    // Strategy: tokenize a seed phrase once to learn the bytes-per-token ratio,
    // then build a string of the estimated length and adjust with at most a few
    // extra tokenize calls.
    for candidate in [" the quick brown fox", " hello world", " lorem ipsum", "\n"] {
        let seed_tokens = tokenize_text(client, base_url, candidate)?;
        if seed_tokens.is_empty() {
            continue;
        }
        let bytes_per_token = candidate.len() as f64 / seed_tokens.len() as f64;

        // Estimate: build a string slightly longer than needed.
        let estimated_bytes = ((target as f64) * bytes_per_token * 1.05) as usize;
        let repeats = estimated_bytes / candidate.len() + 1;
        let long_text = candidate.repeat(repeats);

        // Verify and adjust. Trim to estimated length first, then correct.
        let mut end = estimated_bytes.min(long_text.len());
        let mut n = tokenize_text(client, base_url, &long_text[..end])?.len();

        if n == target {
            return Ok(long_text[..end].to_string());
        }

        if n < target {
            // Too short — extend one seed-phrase at a time.
            while n < target && end < long_text.len() {
                end = (end + candidate.len()).min(long_text.len());
                n = tokenize_text(client, base_url, &long_text[..end])?.len();
            }
        }

        if n > target {
            // Too long — shrink by estimated bytes-per-token steps.
            while n > target && end > 0 {
                let overshoot = n - target;
                let shrink = ((overshoot as f64) * bytes_per_token).ceil() as usize;
                end = end.saturating_sub(shrink.max(1));
                n = tokenize_text(client, base_url, &long_text[..end])?.len();
            }
        }

        // Fine-tune: walk one character at a time in the right direction.
        let max_finetune = candidate.len() * 2;
        for _ in 0..max_finetune {
            if n == target {
                break;
            }
            if n < target && end < long_text.len() {
                end += 1;
            } else if n > target && end > 0 {
                end -= 1;
            } else {
                break;
            }
            n = tokenize_text(client, base_url, &long_text[..end])?.len();
        }

        if n == target {
            return Ok(long_text[..end].to_string());
        }
    }
    anyhow::bail!(
        "failed to build text prompt of exactly {token_count} tokens for vl_throughput benchmark"
    )
}

fn tokenize_text(client: &Client, base_url: &str, content: &str) -> anyhow::Result<Vec<u32>> {
    let response: server::TokenizeResponse = client
        .post(format!("{base_url}/tokenize"))
        .json(&json!({ "content": content }))
        .send()
        .context("failed to call /tokenize")?
        .error_for_status()
        .context("/tokenize failed")?
        .json()
        .context("failed to parse /tokenize response")?;
    Ok(response.tokens)
}

fn send_vl_completion(
    client: &Client,
    base_url: &str,
    text_prompt: &str,
    image_b64: &str,
    decode_tokens: u32,
    eog_token_ids: &[u32],
) -> anyhow::Result<VlCompletionResponse> {
    let logit_bias: Vec<(u32, bool)> = eog_token_ids.iter().map(|&id| (id, false)).collect();

    // The prompt must contain <__media__> markers matching the number of
    // images in multimodal_data. We place the marker before the text.
    let prompt_with_marker = format!("{MEDIA_MARKER}{text_prompt}");

    let resp = client
        .post(format!("{base_url}/completion"))
        .json(&json!({
            "prompt": {
                "prompt_string": prompt_with_marker,
                "multimodal_data": [image_b64],
            },
            "temperature": 0.0,
            "n_predict": decode_tokens,
            "ignore_eos": true,
            "logit_bias": logit_bias,
            "cache_prompt": false,
        }))
        .send()
        .context("failed to call /completion")?
        .error_for_status()
        .context("/completion failed")?
        .json()
        .context("failed to parse /completion response")?;
    Ok(resp)
}

fn validate_response(resp: &VlCompletionResponse, expected_decode: u32) -> anyhow::Result<()> {
    if resp.timings.predicted_n != expected_decode {
        anyhow::bail!(
            "llama-server generation mismatch: expected {} tokens, got {}",
            expected_decode,
            resp.timings.predicted_n
        );
    }
    let stopped_at_limit =
        resp.stopped_limit || (resp.stop && resp.stop_type.as_deref() == Some("limit"));
    if !stopped_at_limit {
        anyhow::bail!("llama-server did not stop at the requested generation limit");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct VlCompletionResponse {
    #[serde(default)]
    stopped_limit: bool,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    stop_type: Option<String>,
    timings: VlCompletionTimings,
}

#[derive(Debug, Deserialize)]
struct VlCompletionTimings {
    prompt_n: u32,
    prompt_ms: f64,
    predicted_n: u32,
    predicted_ms: f64,
}

// ---------------------------------------------------------------------------
// Synthetic PNG generation
// ---------------------------------------------------------------------------

/// Generate a minimal valid PNG of solid gray pixels and return it as a
/// base64-encoded string suitable for the llama.cpp multimodal API.
fn generate_image_b64(width: u32, height: u32) -> anyhow::Result<String> {
    let png_bytes = generate_png(width, height)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&png_bytes))
}

/// Generate a minimal valid PNG file in memory.
///
/// Produces a solid gray (128, 128, 128) RGB image of the given dimensions.
fn generate_png(width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    let mut buf = Vec::new();

    // PNG signature
    buf.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR chunk
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(2); // color type: RGB
    ihdr_data.push(0); // compression method
    ihdr_data.push(0); // filter method
    ihdr_data.push(0); // interlace method
    write_chunk(&mut buf, b"IHDR", &ihdr_data);

    // IDAT chunk: zlib-compressed raw scanlines
    let raw_scanlines = build_raw_scanlines(width, height);
    let compressed = zlib_compress(&raw_scanlines)?;
    write_chunk(&mut buf, b"IDAT", &compressed);

    // IEND chunk
    write_chunk(&mut buf, b"IEND", &[]);

    Ok(buf)
}

fn build_raw_scanlines(width: u32, height: u32) -> Vec<u8> {
    // Each scanline: filter byte (0 = None) + 3 bytes per pixel (RGB).
    let row_len = 1 + (width as usize) * 3;
    let mut row = vec![0u8; row_len];
    // Fill pixel bytes with solid gray (128); row[0] stays 0 (filter: None).
    row[1..].fill(128);
    row.repeat(height as usize)
}

fn zlib_compress(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    use flate2::{write::ZlibEncoder, Compression};

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

fn write_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let length = data.len() as u32;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);
    let crc = png_crc32(chunk_type, data);
    buf.extend_from_slice(&crc.to_be_bytes());
}

fn png_crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in chunk_type.iter().chain(data.iter()) {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Standard CRC-32 lookup table (polynomial 0xEDB88320).
static CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_png_produces_valid_header() -> anyhow::Result<()> {
        let png = generate_png(2, 2)?;
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        Ok(())
    }

    #[test]
    fn generate_png_roundtrips_dimensions_in_ihdr() -> anyhow::Result<()> {
        let png = generate_png(320, 240)?;
        // IHDR starts at offset 8 (after signature): 4 bytes length + 4 bytes "IHDR"
        let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
        let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
        assert_eq!(w, 320);
        assert_eq!(h, 240);
        Ok(())
    }

    #[test]
    fn generate_image_b64_produces_non_empty_base64() -> anyhow::Result<()> {
        let b64 = generate_image_b64(4, 4)?;
        assert!(!b64.is_empty());
        // Should be valid base64.
        base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .context("invalid base64")?;
        Ok(())
    }
}
