//! Pre-flight repair of eval submission payloads. The scores server rejects
//! duplicate completion ids with a 400 that names at most ten offenders, so the
//! run would be lost; the eval runner skips checkpointed ids, so this repair is
//! for what slips through anyway (a `payload.json` from an older build).

use std::collections::HashSet;

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScoreValidationError {
    /// The id is the join key against the dataset, so dropping an unkeyed entry
    /// would mask a runner bug.
    #[error(
        "benchmark {benchmark_id}: completions entry missing string `id`; refusing to dedupe /score payload"
    )]
    CompletionMissingId { benchmark_id: String },
}

/// Duplicate ids listed verbatim in the log line.
pub const MAX_LISTED_DUPLICATES: usize = 10;

/// What [`dedupe_completion_ids`] removed from a payload.
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

/// Repair `payload` so its eval `completions` array has unique ids, keeping the
/// last entry for each id (as [`pipette_ops::eval_completions`] loads them).
/// `Ok(None)` when nothing was removed.
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

    // Last-wins: walk reversed and keep the first occurrence, then flip back.
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
        let mut v = json!({"benchmark_id": "prefill_throughput_512", "prefill_time_ms": 12.3});
        let before = v.clone();
        let report = dedupe_completion_ids(&mut v)?;
        assert!(report.is_none());
        assert_eq!(v, before);
        Ok(())
    }

    #[test]
    fn keeps_last_occurrence_when_deduping() -> Result<(), Box<dyn std::error::Error>> {
        // Survivors keep their positions: the input order minus the earlier "a".
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
        assert!(!s.contains("(and "), "{s}");
        Ok(())
    }

    #[test]
    fn report_truncates_listed_ids_above_ten() -> anyhow::Result<()> {
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
        // The production shape: 2000 ids, each twice.
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
        let ids: Vec<String> = (0..2000).map(|i| format!("sample{i:04}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut v = payload("eval_ifstruct_original", &refs);
        let report = dedupe_completion_ids(&mut v)?;
        assert!(report.is_none());
        assert_eq!(ids_in(&v)?.len(), 2000);
        Ok(())
    }
}
