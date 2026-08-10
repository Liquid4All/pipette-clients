//! The remote catalog's wire shapes: pull bookkeeping and the loose → strict
//! benchmark conversion. Only the client talks to the management server, so
//! these stay out of `pipette-ops` and off the runtime crates' compile path.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use pipette_mgmt_client::types::RemoteBenchmark;
use pipette_mgmt_client::EntityTag;
use pipette_plan_types::benchmark::BenchmarkDefinition;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemoteSyncState {
    pub pulled_at: String,
    pub benchmark_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks_etag: Option<EntityTag>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub benchmark_etags: BTreeMap<String, EntityTag>,
}

/// Validate a loose upstream benchmark into a strict, typed definition.
/// Re-serializing the loose value reassembles the flat wire object (the
/// `#[serde(flatten)]` parameters spread back out), so the tagged enum's own
/// deserialization enforces the contract — a known `benchmark_type` carrying
/// all of its required, well-typed parameters, or an error.
///
/// A free function rather than `TryFrom`: both types are foreign to this crate,
/// so the orphan rule rules out the impl.
pub(crate) fn benchmark_definition_from_remote(
    loose: RemoteBenchmark,
) -> Result<BenchmarkDefinition> {
    let benchmark_id = loose.benchmark_id.clone();
    serde_json::to_value(loose)
        .and_then(serde_json::from_value::<BenchmarkDefinition>)
        .map_err(|source| Error::BenchmarkConversion {
            benchmark_id,
            source,
        })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{DecodeThroughput, EvalBenchmark};

    use super::*;

    #[test]
    fn old_sync_state_with_dropped_server_url_still_deserializes() -> anyhow::Result<()> {
        // Existing sync.json files carry `server_url` (always written empty, never
        // read); dropping the field must not break reading a pre-existing cache.
        let old_json = serde_json::json!({
            "pulled_at": "2026-01-01T00:00:00Z",
            "server_url": "https://mgmt.example.com",
            "benchmark_count": 3
        });
        let state: RemoteSyncState = serde_json::from_value(old_json)?;
        assert_eq!(state.pulled_at, "2026-01-01T00:00:00Z");
        assert_eq!(state.benchmark_count, 3);
        Ok(())
    }

    #[test]
    fn loose_remote_benchmark_converts_to_strict_definition() -> anyhow::Result<()> {
        let loose: RemoteBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "decode_throughput_512_100",
            "benchmark_type": "decode_throughput",
            "parameter_prefill_tokens": 512,
            "parameter_decode_tokens": 100,
        }))?;
        let definition = benchmark_definition_from_remote(loose)?;
        assert!(matches!(
            definition,
            BenchmarkDefinition::DecodeThroughput(DecodeThroughput {
                parameter_decode_tokens: 100,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn unknown_benchmark_type_is_rejected() -> anyhow::Result<()> {
        let loose: RemoteBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "mystery_1",
            "benchmark_type": "mystery_metric",
            "parameter_foo": 1,
        }))?;
        match benchmark_definition_from_remote(loose) {
            Err(Error::BenchmarkConversion { benchmark_id, .. }) => {
                assert_eq!(benchmark_id, "mystery_1");
            }
            Ok(def) => return Err(anyhow::anyhow!("expected rejection, got {def:?}")),
            Err(other) => {
                return Err(anyhow::anyhow!(
                    "expected BenchmarkConversion, got {other:?}"
                ))
            }
        }
        Ok(())
    }

    #[test]
    fn missing_required_parameter_is_rejected() -> anyhow::Result<()> {
        let loose: RemoteBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "decode_throughput_512_100",
            "benchmark_type": "decode_throughput",
            "parameter_prefill_tokens": 512,
        }))?;
        assert!(benchmark_definition_from_remote(loose).is_err());
        Ok(())
    }

    #[test]
    fn loose_remote_eval_keeps_samples() -> anyhow::Result<()> {
        let loose: RemoteBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_mmlu",
            "benchmark_type": "eval",
            "parameter_eval_id": "mmlu_pro",
            "parameter_dataset_name": "edge_2026.03.1",
            "parameter_max_tokens": 64,
            "samples": [{"id": "q1", "messages": []}],
        }))?;
        match benchmark_definition_from_remote(loose)? {
            BenchmarkDefinition::Eval(EvalBenchmark { samples, .. }) => {
                assert_eq!(samples.as_deref().map(<[_]>::len), Some(1));
            }
            other => return Err(anyhow::anyhow!("expected Eval, got {other:?}")),
        }
        Ok(())
    }
}
