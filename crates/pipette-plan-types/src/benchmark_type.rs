//! `BenchmarkType` — the closed, exhaustive set of benchmark kinds the plan
//! supports.
//!
//! It's a pure plan-type. Two parses serve different callers:
//! - [`BenchmarkType::from_id`] is **total** — `None` for anything outside the
//!   set — so a client can tolerate (list, display) benchmark types it doesn't
//!   implement, e.g. a remote benchmark of a newer kind.
//! - `FromStr` is the fallible parse (errors on unknown) behind the `--type`
//!   CLI filter and upstream benchmark-type strings. It accepts both the
//!   snake_case wire spelling and the kebab-case the CLI advertises. Its error
//!   is local here; `pipette-ops` bridges it into that crate's `Error`.

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use strum::{Display, VariantArray};
use thiserror::Error;

/// A benchmark-type string (snake_case or kebab-case) named no known
/// [`BenchmarkType`] — e.g. an unrecognized `--type` filter.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unknown benchmark type '{0}'")]
pub struct UnknownBenchmarkType(pub String);

/// Benchmark type — used for filtering, serde, display, and the
/// `(BenchmarkType, Runtime, Model)` flag key.
///
/// - Serde and Display use `snake_case` (e.g. `"prefill_throughput"`).
/// - `FromStr` accepts both the `snake_case` wire spelling and the
///   `kebab-case` the `--type` CLI advertises (e.g. `"prefill-throughput"`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, VariantArray,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum BenchmarkType {
    PrefillThroughput,
    DecodeThroughput,
    EndToEndLatency,
    MaxMemoryUsage,
    Eval,
    VlThroughput,
}

impl FromStr for BenchmarkType {
    type Err = UnknownBenchmarkType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Match against each variant's canonical snake_case spelling (strum
        // `Display`), normalizing kebab-case input first so the `--type` CLI
        // form parses too. Driven off `ALL` so it can't drift as variants grow.
        let normalized = s.replace('-', "_");
        Self::ALL
            .iter()
            .copied()
            .find(|variant| variant.to_string() == normalized)
            .ok_or_else(|| UnknownBenchmarkType(s.to_string()))
    }
}

impl BenchmarkType {
    /// All variants, in declaration order (via `#[derive(VariantArray)]`),
    /// re-projected as an inherent const so callers don't import the trait.
    pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;

    /// Infer the benchmark type from a benchmark ID by prefix matching.
    /// Strips an optional `"local/"` or `"remote/"` prefix first. Returns
    /// `None` for an ID outside the known set — callers tolerate those rather
    /// than reject them.
    ///
    /// A prefix heuristic, not an authoritative type: prefer
    /// `BenchmarkDefinition::benchmark_type()` wherever a catalog is
    /// available. This survives for the two seams that have none —
    /// `pipette-plan` cell-expansion (no synced storage at that layer) and
    /// result-record readers that persist only the id.
    // TODO: remove once benchmark definitions are threaded through those seams.
    pub fn from_id(benchmark_id: &str) -> Option<Self> {
        let id = benchmark_id
            .strip_prefix("local/")
            .or_else(|| benchmark_id.strip_prefix("remote/"))
            .unwrap_or(benchmark_id);
        if id.starts_with("prefill_throughput") {
            Some(Self::PrefillThroughput)
        } else if id.starts_with("decode_throughput") {
            Some(Self::DecodeThroughput)
        } else if id.starts_with("end_to_end_latency") {
            Some(Self::EndToEndLatency)
        } else if id.starts_with("max_memory_usage") {
            Some(Self::MaxMemoryUsage)
        } else if id.starts_with("eval_") || id == "eval" {
            Some(Self::Eval)
        } else if id.starts_with("vl_throughput") {
            Some(Self::VlThroughput)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_round_trips_every_variant() {
        BenchmarkType::ALL.iter().for_each(|&variant| {
            assert_eq!(
                variant.to_string().parse::<BenchmarkType>().ok(),
                Some(variant)
            );
        });
    }

    #[test]
    fn from_str_accepts_snake_and_kebab() -> Result<(), UnknownBenchmarkType> {
        assert_eq!(
            "prefill_throughput".parse::<BenchmarkType>()?,
            BenchmarkType::PrefillThroughput
        );
        // The kebab spelling the `--type` CLI advertises parses too.
        assert_eq!(
            "prefill-throughput".parse::<BenchmarkType>()?,
            BenchmarkType::PrefillThroughput
        );
        assert!("mystery_metric".parse::<BenchmarkType>().is_err());
        Ok(())
    }

    #[test]
    fn from_id_tolerates_unknown_as_none() {
        assert_eq!(
            BenchmarkType::from_id("prefill_throughput_256"),
            Some(BenchmarkType::PrefillThroughput)
        );
        assert_eq!(
            BenchmarkType::from_id("remote/eval_mmlu"),
            Some(BenchmarkType::Eval)
        );
        // A type the client doesn't implement is tolerated, not an error.
        assert_eq!(BenchmarkType::from_id("mystery_metric_1"), None);
    }
}
