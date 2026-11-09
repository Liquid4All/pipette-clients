use anyhow::Context;
use reqwest::Method;
use serde::{Serialize, de::DeserializeOwned};

use crate::server::ServerHandle;

pub(super) fn post_json<T, U>(
    server: &ServerHandle,
    endpoint: &str,
    request: &T,
) -> anyhow::Result<U>
where
    T: Serialize + ?Sized,
    U: DeserializeOwned,
{
    let url = format!("{}{endpoint}", server.base_url);
    let body = serde_json::to_value(request).with_context(|| {
        format!("failed to serialize pipette-coreai-sidecar {endpoint} request")
    })?;
    server
        .http
        .json_request(Method::POST, &url, None, Some(body))
        .with_context(|| format!("POST {endpoint} to pipette-coreai-sidecar failed"))
}

/// Reset the key-value (KV) cache and let the engine settle. This step is not
/// timed. Call it before each timed request, including the warm-up. The 50 ms
/// sleep of the sidecar then stays outside the end-to-end timer.
pub(super) fn prepare(server: &ServerHandle) -> anyhow::Result<()> {
    let _: serde_json::Value = post_json(server, "/prepare", &serde_json::json!({}))?;
    Ok(())
}
