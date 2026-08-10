use std::time::Duration;

use anyhow::Context;
use reqwest::Method;
use serde::{de::DeserializeOwned, Serialize};

use pipette_http::HttpClient;

/// Client-side request timeout for MLX timing/memory HTTP cells.
/// No MLX timing variant of `BenchmarkFlags` carries `http_timeout`, so
/// throughput paths keep a fixed long budget.
const HTTP_TIMEOUT: Duration = Duration::from_secs(3600);

pub(super) fn post_json<T, U>(base_url: &str, endpoint: &str, request: &T) -> anyhow::Result<U>
where
    T: Serialize + ?Sized,
    U: DeserializeOwned,
{
    let http = HttpClient::with_request_timeout("pipette", HTTP_TIMEOUT)
        .context("failed to build MLX server HTTP client")?;
    let url = format!("{base_url}{endpoint}");
    let body = serde_json::to_value(request)
        .with_context(|| format!("failed to serialize pipette_mlx_server {endpoint} request"))?;
    http.json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {endpoint} to pipette_mlx_server failed"))
}

pub(super) fn validate_tps(metric: &str, tps: f64) -> anyhow::Result<()> {
    pipette_ops::measurement::positive_finite(metric, tps).map(|_| ())
}

pub(super) fn time_ms_from_tps(tokens: u32, tps: f64) -> anyhow::Result<f64> {
    validate_tps("throughput", tps)?;
    Ok((tokens as f64 / tps) * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_tps_to_ms() -> anyhow::Result<()> {
        // 512 tokens / 1024 tok/s = 0.5 s = 500 ms.
        assert_eq!(time_ms_from_tps(512, 1024.0)?, 500.0);
        Ok(())
    }

    #[test]
    fn rejects_invalid_tps_values() {
        assert!(time_ms_from_tps(512, 0.0).is_err());
        assert!(time_ms_from_tps(512, f64::NAN).is_err());
        assert!(time_ms_from_tps(512, -1.0).is_err());
    }
}
