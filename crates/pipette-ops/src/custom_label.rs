//! A vault source's `custom_label`, left out wherever a descriptor is an
//! identity.
//!
//! The label repeats the name of the record the source's path already names.
//! pipette-mgmt takes a job's merge key and a descriptor's id without it, so a
//! client identity that kept it would split one file into two.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use pipette_plan_types::descriptor_digest;

const CUSTOM_LABEL: &str = "custom_label";

/// `descriptor` as JSON without any vault source's `custom_label`: the form an
/// identity is taken over. A reported descriptor keeps its label.
pub fn unlabelled<T: Serialize>(descriptor: &T) -> serde_json::Result<Value> {
    let mut value = serde_json::to_value(descriptor)?;
    drop_custom_labels(&mut value);
    Ok(value)
}

pub fn without_custom_label<T: Serialize + DeserializeOwned>(
    declared: &T,
) -> serde_json::Result<T> {
    serde_json::from_value(unlabelled(declared)?)
}

/// The id pipette-mgmt takes for `declared`, as `*_descriptor_sha256`.
pub fn descriptor_sha256<T: Serialize>(declared: &T) -> serde_json::Result<String> {
    descriptor_digest(&unlabelled(declared)?)
}

/// A port of pipette-mgmt's `drop_custom_labels` (pipeline crate); keep the two
/// in step.
fn drop_custom_labels(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            if is_vault_source(fields) {
                fields.remove(CUSTOM_LABEL);
            }
            fields.values_mut().for_each(drop_custom_labels);
        }
        Value::Array(items) => items.iter_mut().for_each(drop_custom_labels),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

/// A model or runtime tags its vault arm on `source`; a uv runtime's source is
/// its own table, tagged on `type`.
fn is_vault_source(fields: &serde_json::Map<String, Value>) -> bool {
    ["source", "type"].into_iter().any(|key| {
        fields
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|tag| tag.starts_with("vault_"))
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{Model, Runtime, canonical_json};

    use super::*;

    /// pipette-mgmt's vectors for a descriptor's content id, verbatim, plus a
    /// uv runtime, whose source is tagged on `type`. The right-hand text is
    /// what the server hashes.
    #[rstest]
    #[case::labelled(
        r#"{"custom_label":"LFM3 q4","principal_id":"org_1","source":"vault_url","url":"/vault/resources/vault_1/content"}"#,
        r#"{"principal_id":"org_1","source":"vault_url","url":"/vault/resources/vault_1/content"}"#
    )]
    #[case::a_labelled_pair(
        r#"{"draft":{"custom_label":"p","source":"vault_archive","url":"/vault/resources/vault_2/content"},"target":{"custom_label":"p","source":"vault_archive","url":"/vault/resources/vault_1/content"},"type":"mlx_speculative"}"#,
        r#"{"draft":{"source":"vault_archive","url":"/vault/resources/vault_2/content"},"target":{"source":"vault_archive","url":"/vault/resources/vault_1/content"},"type":"mlx_speculative"}"#
    )]
    #[case::unlabelled(
        r#"{"principal_id":"org_1","source":"vault_url","url":"/vault/resources/vault_1/content"}"#,
        r#"{"principal_id":"org_1","source":"vault_url","url":"/vault/resources/vault_1/content"}"#
    )]
    #[case::a_public_source(
        r#"{"custom_label":"x","org":"o","repo_name":"r","source":"huggingface"}"#,
        r#"{"custom_label":"x","org":"o","repo_name":"r","source":"huggingface"}"#
    )]
    #[case::a_uv_runtime(
        r#"{"python_version":"3.12","server_version":"0.10.0","source":{"contents":"vllm==0.10.0\n","custom_label":"acme wheelhouse","principal_id":"org_1","type":"vault_wheelhouse","url":"/vault/resources/vault_3/content"},"type":"uv_vllm"}"#,
        r#"{"python_version":"3.12","server_version":"0.10.0","source":{"contents":"vllm==0.10.0\n","principal_id":"org_1","type":"vault_wheelhouse","url":"/vault/resources/vault_3/content"},"type":"uv_vllm"}"#
    )]
    fn only_a_vault_sources_label_is_dropped(
        #[case] descriptor: &str,
        #[case] hashed: &str,
    ) -> anyhow::Result<()> {
        let descriptor: Value = serde_json::from_str(descriptor)?;
        assert_eq!(canonical_json(&unlabelled(&descriptor)?), hashed);
        Ok(())
    }

    /// `canonical` is the unlabelled descriptor exactly as the server hashes
    /// it; the label goes on the vault source at `source_at`.
    fn digests_without_and_with_a_label<T>(
        canonical: &str,
        source_at: &str,
    ) -> anyhow::Result<[String; 2]>
    where
        T: DeserializeOwned + Serialize,
    {
        let plain: Value = serde_json::from_str(canonical)?;
        let mut labelled = plain.clone();
        labelled
            .pointer_mut(source_at)
            .and_then(Value::as_object_mut)
            .context("the vault source is an object")?
            .insert("custom_label".to_owned(), "acme LFM3 q4".into());
        let typed = |descriptor: &Value| -> anyhow::Result<T> {
            let typed: T = serde_json::from_value(descriptor.clone())?;
            // Neither the label nor anything else is lost on the way in.
            assert_eq!(&serde_json::to_value(&typed)?, descriptor);
            Ok(typed)
        };
        let labelled = typed(&labelled)?;
        assert_eq!(
            serde_json::to_value(without_custom_label(&labelled)?)?,
            plain
        );
        Ok([
            descriptor_sha256(&typed(&plain)?)?,
            descriptor_sha256(&labelled)?,
        ])
    }

    #[test]
    fn a_custom_label_is_outside_the_digest() -> anyhow::Result<()> {
        // shasum -a 256 of each text.
        let model = "4db8864023866a76ddce3f6d32b121e043720f322fa57bf13f80589941ca1260";
        assert_eq!(
            digests_without_and_with_a_label::<Model>(
                r#"{"principal_id":"org_01936d","source":"vault_url","type":"gguf_text","url":"/vault/resources/vault_1/content"}"#,
                "",
            )?,
            [model, model]
        );
        // A uv runtime's source is its own table.
        let runtime = "5e189bd3e1aa60526cff98300ff2d3a15842b078c25f4738125363f8ffd74d7e";
        assert_eq!(
            digests_without_and_with_a_label::<Runtime>(
                r#"{"build":"cu121","python_version":"3.12","server_version":"0.10.0","source":{"contents":"vllm==0.10.0\n","principal_id":"org_01936d","type":"vault_wheelhouse","url":"/vault/resources/vault_1/content"},"type":"uv_vllm"}"#,
                "/source",
            )?,
            [runtime, runtime]
        );
        Ok(())
    }
}
