//! Folds the HuggingFace token from the environment into a [`Model`].

use anyhow::Context;

use pipette_plan_types::{AuthToken, HF_TOKEN_ENV, Model, inject_hf_auth_token};

/// Fold `PIPETTE_HF_TOKEN` into each HuggingFace source of `model` that carries
/// no token, so auth travels on the model spec. No-op when unset or blank.
pub fn inject_env_hf_token(model: &mut Model) -> anyhow::Result<()> {
    inject_hf_token_from(model, std::env::var(HF_TOKEN_ENV).ok().as_deref())
}

/// [`inject_env_hf_token`] over an already-read value, so tests leave the
/// process environment alone.
fn inject_hf_token_from(model: &mut Model, raw: Option<&str>) -> anyhow::Result<()> {
    let Some(raw) = raw.map(str::trim).filter(|token| !token.is_empty()) else {
        return Ok(());
    };
    let token = AuthToken::try_new(raw.to_owned())
        .with_context(|| format!("{HF_TOKEN_ENV} is set but is not a valid access token"))?;
    if inject_hf_auth_token(model, token) {
        tracing::info!(
            var = %HF_TOKEN_ENV,
            "injected the token env var into the model definition for the gated download"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn gated(auth_token: Option<&str>) -> anyhow::Result<Model> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "torch",
            "source": "huggingface",
            "org": "meta-llama",
            "repo_name": "Llama-3.2-1B",
            "auth_token": auth_token,
        }))?)
    }

    #[rstest]
    #[case::unset(None)]
    #[case::blank(Some("   "))]
    fn no_token_leaves_the_model_untouched(#[case] raw: Option<&str>) -> anyhow::Result<()> {
        let mut model = gated(None)?;
        inject_hf_token_from(&mut model, raw)?;
        assert!(model.auth_token().is_none());
        Ok(())
    }

    /// The token is trimmed, and one the spec already carries wins.
    #[test]
    fn a_token_fills_only_a_tokenless_gated_model() -> anyhow::Result<()> {
        let mut model = gated(None)?;
        inject_hf_token_from(&mut model, Some("  hf_env  "))?;
        assert_eq!(model.auth_token().map(AsRef::as_ref), Some("hf_env"));

        let mut declared = gated(Some("hf_declared"))?;
        inject_hf_token_from(&mut declared, Some("hf_env"))?;
        assert_eq!(
            declared.auth_token().map(AsRef::as_ref),
            Some("hf_declared")
        );
        Ok(())
    }
}
