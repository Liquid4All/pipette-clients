//! Canonical descriptor form and the digest that addresses it.
//!
//! A `runtime_descriptor` / `model_descriptor` is the full typed value rendered
//! as JSON. [`digest`] hashes its **canonical** form — object keys sorted
//! recursively, no insignificant whitespace — so the id survives a client
//! formatting its payload differently, or this crate reordering a struct's
//! fields.
//!
//! The definition is shared with `pipette-mgmt`'s `canonical_json`, which
//! already stores `runtime_descriptor_sha256` on every warehouse metric row.
//! Reproducing it here means the digest an operator reads off `runtimes list`
//! is the same string the warehouse groups by, rather than a second, private
//! id. `canonicalization_matches_pipette_mgmt` pins the two together with fixed
//! vectors — the rule is small and stable, but a silent drift would split one
//! identity into two, so it is asserted rather than assumed.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Canonical, compact JSON for `value`: object keys sorted recursively, no
/// insignificant whitespace.
///
/// Array order is preserved — it is semantically significant. Numbers keep
/// `serde_json`'s rendering, so `1` and `1.0` stay distinct; descriptors are
/// almost entirely strings, so that edge does not arise in practice.
pub fn canonicalize(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, value);
    out
}

/// Hex SHA-256 over the canonical form of `value` — the id `pipette-mgmt`
/// stores as `runtime_descriptor_sha256`.
pub fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let json = serde_json::to_value(value)?;
    Ok(hex::encode(Sha256::digest(canonicalize(&json).as_bytes())))
}

/// How many leading hex chars of a [`digest`] are shown in listings. Long
/// enough to be unambiguous over a store's worth of entries, short enough to
/// sit in a table column and be retyped.
pub const DIGEST_DISPLAY_LEN: usize = 12;

/// The shortest prefix accepted when addressing a runtime by digest. 32 bits is
/// far past collision range for a local store, and shorter prefixes start to
/// read as typos rather than references.
pub const DIGEST_MIN_PREFIX_LEN: usize = 8;

/// `digest` truncated for display.
pub fn short_digest(digest: &str) -> &str {
    let end = digest.len().min(DIGEST_DISPLAY_LEN);
    &digest[..end]
}

fn write_canonical(out: &mut String, value: &Value) {
    match value {
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            keys.iter().enumerate().for_each(|(i, key)| {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                if let Some(v) = map.get(*key) {
                    write_canonical(out, v);
                }
            });
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            items.iter().enumerate().for_each(|(i, item)| {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            });
            out.push(']');
        }
        leaf => out.push_str(&leaf.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Key order and whitespace are exactly what canonicalization erases, so
    /// two spellings of one descriptor have to land on one digest.
    #[test]
    fn spelling_does_not_change_the_digest() -> anyhow::Result<()> {
        let a = json!({"type": "mlx", "version": "0.31.3", "flavor": "macos-arm64"});
        let b = json!({"flavor": "macos-arm64", "version": "0.31.3", "type": "mlx"});
        assert_eq!(canonicalize(&a), canonicalize(&b));
        assert_eq!(digest(&a)?, digest(&b)?);
        Ok(())
    }

    /// Nested objects sort at every level, and arrays keep their order because
    /// position carries meaning (install flags are argv).
    #[test]
    fn canonical_form_sorts_deeply_and_preserves_arrays() {
        let value = json!({
            "b": {"z": 1, "a": 2},
            "a": ["second", "first"],
        });
        assert_eq!(
            canonicalize(&value),
            r#"{"a":["second","first"],"b":{"a":2,"z":1}}"#
        );
    }

    /// A different descriptor is a different id — the property the whole
    /// addressing scheme rests on.
    #[test]
    fn distinct_descriptors_differ() -> anyhow::Result<()> {
        let a = json!({"type": "mlx", "version": "0.31.3"});
        let b = json!({"type": "mlx", "version": "0.31.4"});
        assert_ne!(digest(&a)?, digest(&b)?);
        Ok(())
    }

    /// Fixed vectors shared with `pipette-mgmt`'s `canonical_json`. If either
    /// side changes its rule, one of these fails instead of the two repos
    /// quietly disagreeing about what a descriptor's id is.
    #[test]
    fn canonicalization_matches_pipette_mgmt() -> anyhow::Result<()> {
        let cases = [
            (json!({}), "{}"),
            (json!({"b": 1, "a": 2}), r#"{"a":2,"b":1}"#),
            (json!({"a": [1, 2]}), r#"{"a":[1,2]}"#),
            (json!({"a": null}), r#"{"a":null}"#),
            (json!({"a": "x y"}), r#"{"a":"x y"}"#),
        ];
        cases.iter().try_for_each(|(value, expected)| {
            assert_eq!(&canonicalize(value), expected);
            anyhow::Ok(())
        })?;
        // Anchored so a change to the hashing (not just the canonical form) is
        // also caught.
        assert_eq!(
            digest(&json!({"b": 1, "a": 2}))?,
            hex::encode(Sha256::digest(r#"{"a":2,"b":1}"#.as_bytes()))
        );
        Ok(())
    }

    #[test]
    fn short_digest_truncates_without_panicking_on_short_input() {
        assert_eq!(short_digest("abcdef0123456789abcdef"), "abcdef012345");
        assert_eq!(short_digest("abc"), "abc");
    }
}
