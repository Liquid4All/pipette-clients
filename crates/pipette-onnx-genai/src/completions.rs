//! The calls the cells make to the ONNX GenAI server: `/tokenize` to size a
//! prompt and `/v1/completions`, both through [`pipette_completions`]. The
//! server sends one chunk per generated token, text or not, so every chunk
//! with a choice is a token.

/// The model id requests send; the server serves the one model it loaded.
pub(crate) const MODEL: &str = "default";

pub(crate) fn url(base_url: &str) -> String {
    format!("{base_url}/v1/completions")
}

#[cfg(test)]
mod tests {
    use pipette_completions::TimingRequest;

    use super::*;

    #[test]
    fn a_request_is_greedy_past_eos() -> anyhow::Result<()> {
        let body = serde_json::to_value(TimingRequest::text(Some(MODEL), "p", 3))?;
        assert_eq!(
            body,
            serde_json::json!({
                "model": "default", "prompt": "p", "max_tokens": 3,
                "temperature": 0.0, "ignore_eos": true,
            })
        );
        Ok(())
    }
}
