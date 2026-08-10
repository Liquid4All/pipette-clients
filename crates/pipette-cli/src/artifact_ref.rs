//! Addressing an **installed** artifact by the digest of its descriptor:
//! `runtime://sha256=<hex>` and `model://sha256=<hex>`.
//!
//! Every other `--runtime` / `--model` spelling is self-contained — a JSON
//! object, or a URI whose keys name the artifact — so [`crate::runtime_uri`]
//! and [`crate::model_uri`] turn one into a value with no outside help. A
//! digest is the exception: it *refers to* a descriptor rather than describing
//! one, so resolving it needs somewhere to look. That is why this layer sits
//! above the pure parsers instead of inside them.
//!
//! It looks only in the local stores, which is exactly the set a digest can
//! usefully name: something already pulled on this host, whose descriptor was
//! long enough to be worth not retyping. A catalog preset that was never pulled
//! has no entry to point at, and already has a short form
//! (`mlx-macos-pipette://version=0.31.3`).
//!
//! The digest is `pipette_plan_types::descriptor::digest` — the same
//! `runtime_descriptor_sha256` / `model_descriptor_sha256` the warehouse
//! stores, so a prefix copied out of `runtimes list` / `models list` is also
//! what groups that artifact's rows there. Models are digested with their auth
//! token stripped, matching how the descriptor is submitted; a stored model is
//! auth-stripped already, so this only makes it explicit.
//!
//! A digest addresses a *declaration*, not an installed directory: the runtime
//! storage key leaves `install_flags` out of identity while the digest covers
//! the whole value, so two declarations differing only in flags share one entry
//! and still have distinct digests. And because it hashes the descriptor's
//! exact shape, adding a field to `Runtime` or `Model` moves every digest —
//! which is why this is a shorthand for typing, never something to write into a
//! plan or job body.

use anyhow::Context;

use pipette_plan_types::{
    descriptor::{self, DIGEST_MIN_PREFIX_LEN},
    Model, Runtime,
};

use crate::model_uri::parse_model_arg;
use crate::runtime_uri::parse_runtime_arg;
use crate::workspace::PipetteWorkspace;

const RUNTIME_SCHEME: &str = "runtime://";
const MODEL_SCHEME: &str = "model://";
const DIGEST_KEY: &str = "sha256=";

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
            let digest = descriptor::digest(&manifest.declared)
                .context("digesting an installed runtime descriptor")?;
            anyhow::Ok((digest, manifest.declared))
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
            let declared = manifest.declared.without_auth_token();
            let digest =
                descriptor::digest(&declared).context("digesting an installed model descriptor")?;
            anyhow::Ok((digest, declared))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    unique_match(&prefix, installed, "model", Model::to_string)
}

/// The digest prefix `arg` carries, or `None` when it is an ordinary reference.
///
/// The scheme without a `sha256=` key gets its own error rather than falling
/// through to the URI parser's "unknown scheme": it is the shape someone lands
/// on by pasting a bare digest after the scheme.
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

/// Normalize a pasted prefix and reject the shapes that can never match.
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

/// The one candidate whose digest starts with `prefix`.
///
/// Ambiguity is an error rather than a pick: two entries sharing a prefix are
/// two different artifacts, and guessing would run the wrong one.
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

    /// Digests are hex, so case is not meaningful — a pasted upper-case prefix
    /// has to match the lower-case form the digest is rendered in.
    #[test]
    fn prefixes_are_normalized_and_trimmed() -> anyhow::Result<()> {
        assert_eq!(validated_prefix("  ABCDEF01  ")?, "abcdef01");
        Ok(())
    }

    /// Each scheme claims only its own, so a model digest is not a runtime
    /// reference and vice versa.
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

    /// The bare scheme is its own actionable error rather than an unknown one.
    #[test]
    fn the_scheme_without_a_digest_key_explains_itself() -> anyhow::Result<()> {
        let err = digest_prefix("runtime://abcdef0123", RUNTIME_SCHEME)
            .err()
            .context("a bare digest after the scheme should be rejected")?;
        assert!(format!("{err:#}").contains("takes a digest"), "{err:#}");
        Ok(())
    }

    /// Ordinary references route to the ordinary parsers untouched.
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

    /// A prefix matching nothing, and one matching several, are both refusals —
    /// the second names the candidates so the operator can lengthen it.
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
}
