//! Pre-flight repair of eval benchmark submission payloads.
//!
//! The management server forwards `completions` straight to
//! `pipette-scores` `/score`, which rejects requests whose completion ids
//! are not unique (the server joins completions to dataset samples by id
//! and refuses to merge silently). Sending a payload with duplicates
//! results in a 400 that names at most ten offenders, with no per-sample
//! diagnostics — by the time the failure is visible server-side the run
//! is lost.
//!
//! This module repairs the payload client-side so a misbehaving runner —
//! or local state that already contains duplicate ids from an earlier bug
//! — does not lose the run. [`dedupe_completion_ids`] drops duplicate
//! completion entries in place (last-write-wins, matching the
//! [`pipette_ops::eval_completions`] append and load semantics) and returns a
//! report that the caller logs. Apply it to the JSON form of the payload
//! at the boundary, just before `client.submit_result`.
//!
//! Per-eval policy: every supported eval (see [`pipette_plan_types::benchmark::eval_id::KnownEvalId`])
//! follows the same rule — one completion per dataset sample id, per
//! submission. The client does
//! not generate repeats; the eval runner skips ids already in the
//! checkpoint (see `eval_completions.rs`) so the resulting `completions`
//! array stays unique by construction. If a duplicate does slip through —
//! e.g. a previously written `payload.json` from a buggy build — this
//! repair step is what keeps the submission viable.

use std::collections::HashSet;

use serde_json::Value;
use thiserror::Error;

/// Errors raised by [`dedupe_completion_ids`]. Distinct from `anyhow::Error`
/// so callers (notably `pipette_cli::client::sync::submit_pending_result`)
/// can `match` on the specific failure rather than string-match on a
/// formatted message — that's the pattern issue #227 set out to remove.
#[derive(Debug, Error)]
pub enum ScoreValidationError {
    /// A `completions` array entry was missing the string `id` field
    /// required by the scores server. The id is the join key against the
    /// dataset; an unkeyed entry can't be safely deduplicated (silent
    /// removal would mask a real bug in the runner).
    #[error("benchmark {benchmark_id}: completions entry missing string `id`; refusing to dedupe /score payload")]
    CompletionMissingId { benchmark_id: String },
}

/// Maximum number of duplicate ids included verbatim in log lines.
pub const MAX_LISTED_DUPLICATES: usize = 10;

/// What [`dedupe_completion_ids`] removed from a payload. The caller
/// logs this; nothing else consumes it. Use [`DedupReport::summary`] for
/// a paste-friendly one-liner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupReport {
    pub benchmark_id: String,
    pub before: usize,
    pub after: usize,
    /// First [`MAX_LISTED_DUPLICATES`] distinct duplicate ids (sorted).
    pub listed_duplicate_ids: Vec<String>,
    /// Total distinct duplicate ids; may exceed `listed_duplicate_ids.len()`.
    pub distinct_duplicates: usize,
}

impl DedupReport {
    pub fn summary(&self) -> String {
        let removed = self.before - self.after;
        let listed = self.listed_duplicate_ids.join(", ");
        let remainder = self
            .distinct_duplicates
            .saturating_sub(self.listed_duplicate_ids.len());
        let remainder_clause = if remainder == 0 {
            String::new()
        } else {
            format!(" (and {remainder} more)")
        };
        format!(
            "benchmark {bid}: deduplicated /score payload, removed {removed} duplicate \
             completion entries across {distinct} distinct id(s) [{listed}{remainder_clause}]; \
             kept last occurrence of each id ({after} of {before} completions)",
            bid = self.benchmark_id,
            distinct = self.distinct_duplicates,
            after = self.after,
            before = self.before,
        )
    }
}

/// Repair `payload` so its eval `completions` array has unique ids.
///
/// Non-eval payloads (no `completions` array) and already-unique payloads
/// pass through with `Ok(None)`. When duplicates are removed, the kept
/// entry for each id is the last one encountered (last-write-wins,
/// matching eval completion append / collect semantics). Errors
/// only on malformed entries (missing string `id` field) — the server
/// cannot key those to a dataset sample so silent removal would mask a
/// real bug.
pub fn dedupe_completion_ids(
    payload: &mut Value,
) -> Result<Option<DedupReport>, ScoreValidationError> {
    let benchmark_id = payload
        .get("benchmark_id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string();

    let Some(completions) = payload.get_mut("completions").and_then(Value::as_array_mut) else {
        return Ok(None);
    };

    let before = completions.len();
    // Validate ids up front and snapshot them — silent removal of
    // malformed entries would mask a real bug, since the server can't
    // key those to a dataset sample.
    let ids: Vec<String> = completions
        .iter()
        .map(|c| {
            c.get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| ScoreValidationError::CompletionMissingId {
                    benchmark_id: benchmark_id.clone(),
                })
        })
        .collect::<Result<_, _>>()?;

    // Last-wins: walk reversed, keep first occurrence per id (= last in
    // original order), then flip back. Surviving entries stay in their
    // original chronological positions.
    let entries: Vec<Value> = std::mem::take(completions);
    let mut seen: HashSet<String> = HashSet::with_capacity(before);
    let mut duplicate_ids: Vec<String> = Vec::new();
    let mut kept: Vec<Value> = entries
        .into_iter()
        .zip(ids)
        .rev()
        .filter_map(|(c, id)| {
            if seen.insert(id.clone()) {
                Some(c)
            } else {
                duplicate_ids.push(id);
                None
            }
        })
        .collect();
    kept.reverse();
    *completions = kept;

    if duplicate_ids.is_empty() {
        return Ok(None);
    }

    let after = completions.len();
    duplicate_ids.sort();
    duplicate_ids.dedup();
    let distinct_duplicates = duplicate_ids.len();
    let listed = duplicate_ids
        .into_iter()
        .take(MAX_LISTED_DUPLICATES)
        .collect::<Vec<_>>();

    Ok(Some(DedupReport {
        benchmark_id,
        before,
        after,
        listed_duplicate_ids: listed,
        distinct_duplicates,
    }))
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use serde_json::json;

    use super::*;

    fn payload(benchmark_id: &str, ids: &[&str]) -> Value {
        let completions: Vec<Value> = ids
            .iter()
            .map(|id| json!({"id": *id, "completion": format!("c-{id}")}))
            .collect();
        json!({
            "benchmark_id": benchmark_id,
            "completions": completions,
        })
    }

    fn ids_in(payload: &Value) -> anyhow::Result<Vec<String>> {
        payload
            .get("completions")
            .and_then(Value::as_array)
            .context("completions array")?
            .iter()
            .map(|c| {
                Ok(c.get("id")
                    .and_then(Value::as_str)
                    .context("completion id")?
                    .to_string())
            })
            .collect()
    }

    fn first_completion_for(payload: &Value, id: &str) -> anyhow::Result<String> {
        Ok(payload
            .get("completions")
            .and_then(Value::as_array)
            .context("completions array")?
            .iter()
            .find(|c| c.get("id").and_then(Value::as_str) == Some(id))
            .and_then(|c| c.get("completion").and_then(Value::as_str))
            .context("completion text")?
            .to_string())
    }

    #[test]
    fn unique_payload_passes_through_unchanged() -> anyhow::Result<()> {
        let mut v = payload("eval_x", &["a", "b", "c"]);
        let before = v.clone();
        let report = dedupe_completion_ids(&mut v)?;
        assert!(report.is_none());
        assert_eq!(v, before);
        Ok(())
    }

    #[test]
    fn payload_without_completions_passes_through() -> anyhow::Result<()> {
        // Non-eval payload (e.g. prefill_throughput) has no `completions`.
        let mut v = json!({"benchmark_id": "prefill_throughput_512", "prefill_time_ms": 12.3});
        let before = v.clone();
        let report = dedupe_completion_ids(&mut v)?;
        assert!(report.is_none());
        assert_eq!(v, before);
        Ok(())
    }

    #[test]
    fn keeps_last_occurrence_when_deduping() -> Result<(), Box<dyn std::error::Error>> {
        // Two entries share id "a"; the second has a distinguishable
        // completion. After last-wins dedup the surviving completion
        // must be the later one — matches
        // `eval_completions::collect_completions` and the append-only
        // log convention. Surviving entries keep their original
        // chronological positions, so the order is the input order
        // with the earlier "a" removed.
        let mut v = json!({
            "benchmark_id": "eval_x",
            "completions": [
                {"id": "a", "completion": "first-a"},
                {"id": "b", "completion": "only-b"},
                {"id": "a", "completion": "second-a"},
                {"id": "c", "completion": "only-c"},
            ],
        });
        let report = dedupe_completion_ids(&mut v)?.ok_or("expected dedup to fire")?;
        assert_eq!(report.before, 4);
        assert_eq!(report.after, 3);
        assert_eq!(report.distinct_duplicates, 1);
        assert_eq!(ids_in(&v)?, vec!["b", "a", "c"]);
        assert_eq!(first_completion_for(&v, "a")?, "second-a");
        Ok(())
    }

    #[test]
    fn report_summary_names_benchmark_and_listed_ids() -> anyhow::Result<()> {
        let mut v = payload("eval_ifstruct_original", &["a", "b", "a", "c", "b"]);
        let report = dedupe_completion_ids(&mut v)?.context("expected dedup to fire")?;
        let s = report.summary();
        assert!(s.contains("eval_ifstruct_original"), "{s}");
        assert!(s.contains("a"), "{s}");
        assert!(s.contains("b"), "{s}");
        // 2 distinct duplicates ⇒ no truncation note.
        assert!(!s.contains("(and "), "{s}");
        Ok(())
    }

    #[test]
    fn report_truncates_listed_ids_above_ten() -> anyhow::Result<()> {
        // 12 distinct ids each duplicated once → 12 distinct duplicate ids.
        let mut ids: Vec<String> = (0..12).map(|i| format!("id{i:02}")).collect();
        ids.extend(ids.clone());
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut v = payload("eval_ifstruct_original", &refs);
        let report = dedupe_completion_ids(&mut v)?.context("expected dedup to fire")?;
        assert_eq!(report.distinct_duplicates, 12);
        assert_eq!(report.listed_duplicate_ids.len(), MAX_LISTED_DUPLICATES);
        assert!(report.summary().contains("(and 2 more)"));
        Ok(())
    }

    #[test]
    fn ifstruct_multi_repeat_payload_is_deduped_to_unique_ids() -> anyhow::Result<()> {
        // Synthesize the issue #95 production shape: 2000 unique ids each
        // appearing twice. After repair the payload satisfies the
        // /score uniqueness invariant.
        let ids: Vec<String> = (0..2000).map(|i| format!("sample{i:04}")).collect();
        let mut doubled: Vec<&str> = ids.iter().map(String::as_str).collect();
        doubled.extend(ids.iter().map(String::as_str));
        let mut v = payload("eval_ifstruct_original", &doubled);

        let report = dedupe_completion_ids(&mut v)?.context("expected dedup to fire")?;
        assert_eq!(report.before, 4000);
        assert_eq!(report.after, 2000);
        assert_eq!(report.distinct_duplicates, 2000);

        let final_ids = ids_in(&v)?;
        let unique: HashSet<&String> = final_ids.iter().collect();
        assert_eq!(
            final_ids.len(),
            unique.len(),
            "deduped payload must satisfy len(ids) == len(set(ids))"
        );
        assert_eq!(final_ids.len(), 2000);
        Ok(())
    }

    #[test]
    fn missing_id_field_errors_with_benchmark_context() -> anyhow::Result<()> {
        let mut v = json!({
            "benchmark_id": "eval_x",
            "completions": [{"completion": "no id here"}],
        });
        let err = dedupe_completion_ids(&mut v)
            .err()
            .context("expected missing-id error")?
            .to_string();
        assert!(err.contains("eval_x"), "{err}");
        assert!(err.contains("missing string `id`"), "{err}");
        Ok(())
    }

    #[test]
    fn unique_payload_with_two_thousand_samples_is_noop() -> anyhow::Result<()> {
        // Post-fix happy path: ifstruct/original at full size with one
        // completion per id passes through unchanged.
        let ids: Vec<String> = (0..2000).map(|i| format!("sample{i:04}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut v = payload("eval_ifstruct_original", &refs);
        let report = dedupe_completion_ids(&mut v)?;
        assert!(report.is_none());
        assert_eq!(ids_in(&v)?.len(), 2000);
        Ok(())
    }
}
