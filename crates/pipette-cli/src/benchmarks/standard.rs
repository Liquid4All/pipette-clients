//! Built-in local ladder + smoke benchmark definitions.

use serde_json::json;

use pipette_plan_types::benchmark::{
    BenchmarkDefinition, BenchmarkSource, DecodeThroughput, EndToEndLatency, EvalBenchmark,
    MaxMemoryUsage, PrefillThroughput, VlThroughput,
};
use pipette_plan_types::BenchmarkId;
use pipette_plan_types::BenchmarkType;

use super::reference::SourcedBenchmarkId;
use super::store::BenchmarkStore;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Standard local benchmark catalog
// ---------------------------------------------------------------------------
//
// The ladder + smoke set lifted from `pipette-{llamacpp,mlx,torch-oai}/src/
// benchmarks.rs`. Each engine CLI used to carry its own copy with the same
// ladder, the same `_smoke` IDs, and the same `benchmark_id` format strings —
// the only real variation is which benchmark kinds the engine's runner can
// execute. The caller declares that capability set as a slice of
// [`BenchmarkType`] — same enum the mgmt server uses, so wire-format and
// catalog-filter share one source of truth.

/// Token ladder shared across every kind that has a ladder form.
///
/// Tuning this set is a methodology question (see #120 "out of scope") —
/// the refactor keeps the existing values verbatim.
const LADDER_TOKENS: [u32; 7] = [100, 256, 512, 1024, 2048, 4096, 8192];

/// Build the standard local benchmark set — the ladder over [`LADDER_TOKENS`]
/// plus a small `_smoke` fixed set — restricted to the kinds the caller can
/// actually execute.
///
/// `kinds` is treated as a set; duplicates are tolerated. Order does not
/// affect the output (entries are always emitted in [`BenchmarkType`]
/// declaration order so the catalog is stable).
///
/// Sizes for the three engines today:
/// - torch-oai (`[EndToEndLatency, MaxMemoryUsage, Eval]`) → 17 (7×2 ladder + 3 smoke)
/// - mlx (everything except VL) → 33 (7×4 ladder + 5 smoke)
/// - llamacpp (all variants) → 34 (7×4 ladder + 6 smoke)
///
/// Kinds without a ladder form (`Eval`, `VlThroughput`) only contribute a
/// smoke entry.
fn standard_local_benchmarks(kinds: &[BenchmarkType]) -> Vec<BenchmarkDefinition> {
    let mut benchmarks = ladder_local_benchmarks(kinds);
    benchmarks.extend(smoke_local_benchmarks(kinds));
    benchmarks
}

fn ladder_local_benchmarks(kinds: &[BenchmarkType]) -> Vec<BenchmarkDefinition> {
    LADDER_TOKENS
        .into_iter()
        .flat_map(|tokens| selected_kinds(kinds).filter_map(move |k| ladder_entry(k, tokens)))
        .collect()
}

fn smoke_local_benchmarks(kinds: &[BenchmarkType]) -> Vec<BenchmarkDefinition> {
    selected_kinds(kinds).map(smoke_entry).collect()
}

/// Iterate the kinds in canonical (declaration) order, restricted to those
/// the caller selected. Driving from [`BenchmarkType::ALL`] makes the
/// output order independent of `kinds` order and tolerant of duplicates;
/// `ALL` is auto-generated via `#[derive(VariantArray)]` so a new enum
/// variant is automatically considered, then forced through the exhaustive
/// `match` in [`ladder_entry`] / [`smoke_entry`].
fn selected_kinds(kinds: &[BenchmarkType]) -> impl Iterator<Item = BenchmarkType> + '_ {
    BenchmarkType::ALL
        .iter()
        .copied()
        .filter(|k| kinds.contains(k))
}

/// Ladder entry for `kind` at `tokens`, or `None` if the kind has no
/// ladder form. Exhaustive `match` — adding a new [`BenchmarkType`]
/// variant forces a compile-time decision here.
fn ladder_entry(kind: BenchmarkType, tokens: u32) -> Option<BenchmarkDefinition> {
    match kind {
        BenchmarkType::PrefillThroughput => {
            Some(BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: format!("prefill_throughput_{tokens}"),
                parameter_prefill_tokens: tokens,
            }))
        }
        BenchmarkType::DecodeThroughput => {
            Some(BenchmarkDefinition::DecodeThroughput(DecodeThroughput {
                benchmark_id: format!("decode_throughput_{tokens}_100"),
                parameter_prefill_tokens: tokens,
                parameter_decode_tokens: 100,
            }))
        }
        BenchmarkType::EndToEndLatency => {
            Some(BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
                benchmark_id: format!("end_to_end_latency_{tokens}_256"),
                parameter_prefill_tokens: tokens,
                parameter_decode_tokens: 256,
            }))
        }
        BenchmarkType::MaxMemoryUsage => {
            Some(BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsage {
                benchmark_id: format!("max_memory_usage_{tokens}"),
                parameter_prefill_tokens: tokens,
            }))
        }
        BenchmarkType::Eval | BenchmarkType::VlThroughput => None,
    }
}

/// Smoke entry for `kind`. Every kind has one. Exhaustive `match` — adding
/// a new [`BenchmarkType`] variant forces a compile-time decision here too.
fn smoke_entry(kind: BenchmarkType) -> BenchmarkDefinition {
    match kind {
        BenchmarkType::PrefillThroughput => {
            BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "prefill_throughput_smoke".to_string(),
                parameter_prefill_tokens: 8,
            })
        }
        BenchmarkType::DecodeThroughput => {
            BenchmarkDefinition::DecodeThroughput(DecodeThroughput {
                benchmark_id: "decode_throughput_smoke".to_string(),
                parameter_prefill_tokens: 8,
                parameter_decode_tokens: 8,
            })
        }
        BenchmarkType::EndToEndLatency => BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
            benchmark_id: "end_to_end_latency_smoke".to_string(),
            parameter_prefill_tokens: 8,
            parameter_decode_tokens: 8,
        }),
        BenchmarkType::MaxMemoryUsage => BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsage {
            benchmark_id: "max_memory_usage_smoke".to_string(),
            parameter_prefill_tokens: 8,
        }),
        BenchmarkType::Eval => BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "eval_smoke".to_string(),
            parameter_eval_id: "eval_smoke".into(),
            parameter_dataset_name: "local".to_string(),
            parameter_max_tokens: 4,
            parameter_mcq_choices: None,
            samples: Some(vec![json!({
                "id": "smoke-1",
                "messages": [
                    { "role": "user", "content": "Reply with exactly OK." }
                ]
            })]),
        }),
        BenchmarkType::VlThroughput => BenchmarkDefinition::VlThroughput(VlThroughput {
            benchmark_id: "vl_throughput_smoke".to_string(),
            parameter_image_width: 224,
            parameter_image_height: 224,
            parameter_text_tokens: 8,
            parameter_decode_tokens: 8,
        }),
    }
}

/// Counts from [`seed_standard_local`].
#[derive(Debug, Clone, Copy)]
pub struct LocalBenchmarkInitSummary {
    pub created: usize,
    pub updated: usize,
}

/// Build standard ladder+smoke defs for `kinds` and [`BenchmarkStore::put`] each as local.
///
/// Catalog policy lives here; the store only receives ready [`BenchmarkDefinition`]s.
pub fn seed_standard_local(
    store: &BenchmarkStore,
    kinds: &[BenchmarkType],
) -> Result<LocalBenchmarkInitSummary> {
    let mut created = 0usize;
    let mut updated = 0usize;
    for def in standard_local_benchmarks(kinds) {
        let id = BenchmarkId::try_new(def.benchmark_id().to_string())
            .map_err(|e| crate::error::Error::Other(e.into()))?;
        let existed = store
            .get(&SourcedBenchmarkId::new(BenchmarkSource::Local, id))?
            .is_some();
        store.put(BenchmarkSource::Local, &def)?;
        if existed {
            updated += 1;
        } else {
            created += 1;
        }
    }
    Ok(LocalBenchmarkInitSummary { created, updated })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::BenchmarkType::*;

    use super::*;

    /// torch-oai capability set: 7×2 ladder (e2e + max_mem) + 3 smoke = 17.
    #[test]
    fn torch_oai_kind_set_is_seventeen() {
        assert_eq!(
            standard_local_benchmarks(&[EndToEndLatency, MaxMemoryUsage, Eval]).len(),
            17
        );
    }
}
