use std::time::Duration;

use anyhow::Context;
use reqwest::Method;
use serde::{de::DeserializeOwned, Serialize};

use pipette_http::HttpClient;

/// Client-side request timeout for Core AI timing/memory HTTP cells.
const HTTP_TIMEOUT: Duration = Duration::from_secs(3600);

pub(super) fn post_json<T, U>(base_url: &str, endpoint: &str, request: &T) -> anyhow::Result<U>
where
    T: Serialize + ?Sized,
    U: DeserializeOwned,
{
    let http = HttpClient::with_request_timeout("pipette", HTTP_TIMEOUT)
        .context("failed to build Core AI server HTTP client")?;
    let url = format!("{base_url}{endpoint}");
    let body = serde_json::to_value(request).with_context(|| {
        format!("failed to serialize pipette-coreai-sidecar {endpoint} request")
    })?;
    http.json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {endpoint} to pipette-coreai-sidecar failed"))
}
