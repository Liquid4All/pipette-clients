//! Addressing an **installed** artifact by the digest of its descriptor:
//! `runtime://sha256=<hex>` and `model://sha256=<hex>`. A digest refers to a
//! descriptor rather than describing one, so it resolves against the local
//! stores, above the pure parsers in [`crate::runtime_uri`] and
//! [`crate::model_uri`].
//!
//! The digest is `descriptor_sha256`, the `*_descriptor_sha256` the warehouse
//! groups rows by (except a vault copy of a HuggingFace model, recorded under the
//! HuggingFace descriptor). A model is digested without its auth token, as it is
//! submitted, and either kind without a vault source's `custom_label`, so a run
//! named by digest reports no stale label from the fetch that created the entry.
//!
//! It addresses a declaration, not an installed directory: the runtime storage
//! key leaves `install_flags` out, the digest does not. Adding a field to
//! `Runtime` or `Model` moves every digest, so this is a typing shorthand, never
//! something to write into a plan or job body.

use anyhow::Context;

use pipette_ops::custom_label::{descriptor_sha256, without_custom_label};
use pipette_plan_types::{Model, Runtime};

use crate::model_uri::parse_model_arg;
use crate::runtime_uri::parse_runtime_arg;
use crate::workspace::PipetteWorkspace;

const RUNTIME_SCHEME: &str = "runtime://";
const MODEL_SCHEME: &str = "model://";
const DIGEST_KEY: &str = "sha256=";
/// 32 bits is far past collision range for a local store; shorter reads as a typo.
const DIGEST_MIN_PREFIX_LEN: usize = 8;

/// Resolve a `--runtime` argument, accepting the digest form alongside every
/// spelling [`parse_runtime_arg`] handles.
pub fn resolve_runtime_arg(ws: &PipetteWorkspace, arg: &str) -> anyhow::Result<Runtime> {
    let Some(prefix) = digest_prefix(arg, RUNTIME_SCHEME)? else {
        return parse_runtime_arg(arg.trim());
    };
    let installed = ws
        .runtimes()
        .list()
        .context("listing installed runtimes")?
        .into_iter()
        .map(|manifest| {
            let declared = without_custom_label(&manifest.declared)
                .context("unlabelling an installed runtime descriptor")?;
            let digest = descriptor_sha256(&declared)
                .context("digesting an installed runtime descriptor")?;
            anyhow::Ok((digest, declared))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    unique_match(&prefix, installed, "runtime", Runtime::cli_ref)
}

/// Resolve a `--model` argument, accepting the digest form alongside every
/// spelling [`parse_model_arg`] handles.
pub fn resolve_model_arg(ws: &PipetteWorkspace, arg: &str) -> anyhow::Result<Model> {
    let Some(prefix) = digest_prefix(arg, MODEL_SCHEME)? else {
        return parse_model_arg(arg.trim());
    };
    let installed = ws
        .models()
        .list()
        .context("listing installed models")?
        .into_iter()
        .map(|manifest| {
            let declared = without_custom_label(&manifest.declared.without_auth_token())
                .context("unlabelling an installed model descriptor")?;
            let digest =
                descriptor_sha256(&declared).context("digesting an installed model descriptor")?;
            anyhow::Ok((digest, declared))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    unique_match(&prefix, installed, "model", Model::to_string)
}

/// `None` when `arg` is an ordinary reference. The scheme without `sha256=` gets
/// its own error rather than the URI parser's "unknown scheme": it is what
/// pasting a bare digest after the scheme looks like.
fn digest_prefix(arg: &str, scheme: &str) -> anyhow::Result<Option<String>> {
    let Some(body) = arg.trim().strip_prefix(scheme) else {
        return Ok(None);
    };
    let raw = body.strip_prefix(DIGEST_KEY).with_context(|| {
        format!(
            "`{scheme}` takes a digest: `{scheme}{DIGEST_KEY}<hex>` \
             (see the DIGEST column of `pipette runtimes list` / `pipette models list`)"
        )
    })?;
    validated_prefix(raw).map(Some)
}

fn validated_prefix(raw: &str) -> anyhow::Result<String> {
    let prefix = raw.trim().to_ascii_lowercase();
    anyhow::ensure!(
        prefix.len() >= DIGEST_MIN_PREFIX_LEN,
        "digest `{prefix}` is too short; give at least {DIGEST_MIN_PREFIX_LEN} hex chars"
    );
    anyhow::ensure!(
        prefix.chars().all(|c| c.is_ascii_hexdigit()),
        "digest `{prefix}` is not hex"
    );
    Ok(prefix)
}

/// The one candidate whose digest starts with `prefix`. Ambiguity is an error,
/// not a pick: guessing would run the wrong artifact.
fn unique_match<T>(
    prefix: &str,
    candidates: Vec<(String, T)>,
    kind: &str,
    render: impl Fn(&T) -> String,
) -> anyhow::Result<T> {
    let mut matched: Vec<T> = candidates
        .into_iter()
        .filter(|(digest, _)| digest.starts_with(prefix))
        .map(|(_, value)| value)
        .collect();

    match matched.len() {
        1 => matched.pop().context("one match was just counted"),
        0 => anyhow::bail!(
            "no installed {kind} has a descriptor digest starting `{prefix}` \
             (`pipette {kind}s list` shows the installed set)"
        ),
        n => {
            let refs: Vec<String> = matched.iter().map(&render).collect();
            anyhow::bail!(
                "digest `{prefix}` is ambiguous across {n} {kind}s ({}); use more characters",
                refs.join(", ")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::descriptor_digest;

    use super::*;

    #[rstest::rstest]
    #[case::too_short("abc", "too short")]
    #[case::not_hex("zzzzzzzzzz", "not hex")]
    #[case::hex_but_short("abc123", "too short")]
    fn malformed_prefixes_are_rejected(
        #[case] prefix: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let err = validated_prefix(prefix)
            .err()
            .with_context(|| format!("{prefix:?} should be rejected"))?;
        assert!(format!("{err:#}").contains(needle), "{prefix}: {err:#}");
        Ok(())
    }

    #[test]
    fn prefixes_are_normalized_and_trimmed() -> anyhow::Result<()> {
        assert_eq!(validated_prefix("  ABCDEF01  ")?, "abcdef01");
        Ok(())
    }

    #[test]
    fn a_scheme_only_claims_its_own_digests() -> anyhow::Result<()> {
        assert_eq!(
            digest_prefix("runtime://sha256=abcdef01", RUNTIME_SCHEME)?,
            Some("abcdef01".to_owned())
        );
        assert_eq!(
            digest_prefix("model://sha256=abcdef01", RUNTIME_SCHEME)?,
            None
        );
        assert_eq!(
            digest_prefix("model://sha256=abcdef01", MODEL_SCHEME)?,
            Some("abcdef01".to_owned())
        );
        Ok(())
    }

    #[test]
    fn the_scheme_without_a_digest_key_explains_itself() -> anyhow::Result<()> {
        let err = digest_prefix("runtime://abcdef0123", RUNTIME_SCHEME)
            .err()
            .context("a bare digest after the scheme should be rejected")?;
        assert!(format!("{err:#}").contains("takes a digest"), "{err:#}");
        Ok(())
    }

    #[test]
    fn ordinary_uris_are_not_treated_as_digests() -> anyhow::Result<()> {
        assert_eq!(
            digest_prefix("llamacpp-cli-stock-tools://version=b9050", RUNTIME_SCHEME)?,
            None
        );
        assert_eq!(
            digest_prefix("gguf-text://repo=a/b&path=c.gguf", MODEL_SCHEME)?,
            None
        );
        Ok(())
    }

    /// An ambiguous prefix names the candidates so the operator can lengthen it.
    #[test]
    fn misses_and_ambiguity_are_both_errors() -> anyhow::Result<()> {
        let none: Vec<(String, String)> = vec![];
        let err = unique_match("abcdef01", none, "runtime", Clone::clone)
            .err()
            .context("an empty store should miss")?;
        assert!(
            format!("{err:#}").contains("no installed runtime"),
            "{err:#}"
        );

        let two = vec![
            ("abcdef0111".to_owned(), "first".to_owned()),
            ("abcdef0122".to_owned(), "second".to_owned()),
        ];
        let err = unique_match("abcdef01", two, "runtime", Clone::clone)
            .err()
            .context("two matches should be ambiguous")?;
        let rendered = format!("{err:#}");
        assert!(rendered.contains("ambiguous"), "{rendered}");
        assert!(
            rendered.contains("first") && rendered.contains("second"),
            "{rendered}"
        );
        Ok(())
    }

    /// Only from the local store: the same prefix on an empty store is a miss,
    /// not a fetch.
    #[test]
    fn a_listed_digest_resolves_the_installed_artifact() -> anyhow::Result<()> {
        use crate::test_support::TempWorkspace;

        let tw = TempWorkspace::new("artifact-ref")?;
        let model_uri = "gguf-text://repo=org/repo&path=m-Q4_0.gguf";
        let runtime = parse_runtime_arg("docker-vllm://image=vllm/vllm-openai&tag=v0.10.0")?;
        let model = parse_model_arg(model_uri)?;
        let runtime_digest = descriptor_digest(&runtime)?;
        let model_digest = descriptor_digest(&model)?;
        let short = |digest: &str| -> String { digest.chars().take(12).collect() };
        let runtime_ref = format!("runtime://sha256={}", short(&runtime_digest));
        let model_ref = format!("model://sha256={}", short(&model_digest));

        let missing = resolve_runtime_arg(&tw.ws, &runtime_ref)
            .err()
            .context("an empty store should miss")?;
        assert!(format!("{missing:#}").contains("no installed runtime"));
        let missing = resolve_model_arg(&tw.ws, &model_ref)
            .err()
            .context("an empty store should miss")?;
        assert!(format!("{missing:#}").contains("no installed model"));

        tw.ws.runtimes().ensure(&runtime, |_, _| Ok(()))?;
        tw.store_gguf_model(model_uri)?;

        assert_eq!(resolve_runtime_arg(&tw.ws, &runtime_ref)?, runtime);
        assert_eq!(resolve_model_arg(&tw.ws, &model_ref)?, model);
        assert_eq!(
            resolve_runtime_arg(&tw.ws, " docker-vllm://image=vllm/vllm-openai&tag=v0.10.0 ")?,
            runtime
        );
        Ok(())
    }

    /// A store entry keeps the label of the fetch that created it; a run named
    /// by digest must not report that stale label.
    #[test]
    fn a_digest_resolves_without_the_stored_label() -> anyhow::Result<()> {
        use crate::test_support::TempWorkspace;

        let tw = TempWorkspace::new("artifact-ref-label")?;
        let plain_model = serde_json::json!({
            "type": "gguf_text",
            "source": "vault_url",
            "url": "/vault/resources/vault_1/content",
            "principal_id": "org_1",
        });
        let plain_runtime = serde_json::json!({
            "type": "llamacpp_cli_stock_tools",
            "source": "vault_remote_archive",
            "url": "/vault/resources/vault_2/content",
            "principal_id": "org_1",
            "flavor": "macos-arm64",
        });
        let labelled = |plain: &serde_json::Value| {
            let mut labelled = plain.clone();
            labelled["custom_label"] = "acme LFM3 q4".into();
            labelled
        };
        let model: Model = serde_json::from_value(labelled(&plain_model))?;
        let runtime: Runtime = serde_json::from_value(labelled(&plain_runtime))?;
        tw.store_gguf_text(&model)?;
        tw.ws.runtimes().ensure(&runtime, |_, _| Ok(()))?;

        let short = |digest: String| -> String { digest.chars().take(12).collect() };
        let model_ref = format!("model://sha256={}", short(descriptor_sha256(&model)?));
        let runtime_ref = format!("runtime://sha256={}", short(descriptor_sha256(&runtime)?));
        assert_eq!(
            serde_json::to_value(resolve_model_arg(&tw.ws, &model_ref)?)?,
            plain_model
        );
        assert_eq!(
            serde_json::to_value(resolve_runtime_arg(&tw.ws, &runtime_ref)?)?,
            plain_runtime
        );
        Ok(())
    }
}
