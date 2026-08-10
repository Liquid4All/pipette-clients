//! CLI-side HuggingFace token injection into typed [`Model`]s.

use anyhow::Context;

use pipette_plan_types::{inject_hf_auth_token, AuthToken, Model, HF_TOKEN_ENV};

/// Fold the `PIPETTE_HF_TOKEN` env var into a gated, tokenless HF model so auth
/// travels on the model spec. No-op when unset, or when the model is
/// public/local/URL/AFM or already carries an explicit token.
pub fn inject_env_hf_token(model: &mut Model) -> anyhow::Result<()> {
    let Some(raw) = std::env::var(HF_TOKEN_ENV)
        .ok()
        .filter(|token| !token.trim().is_empty())
    else {
        return Ok(());
    };
    let token = AuthToken::try_new(raw.trim().to_owned())
        .with_context(|| format!("{HF_TOKEN_ENV} is set but is not a valid access token"))?;
    if inject_hf_auth_token(model, token) {
        eprintln!("injected {HF_TOKEN_ENV} into the model definition for the gated download");
    }
    Ok(())
}
