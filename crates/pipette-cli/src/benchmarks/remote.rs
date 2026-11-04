//! The remote catalog's wire shapes: pull bookkeeping and the loose → strict
//! benchmark conversion. Only the client talks to the management server, so
//! these stay out of `pipette-ops` and off the runtime crates' compile path.

use serde::{Deserialize, Serialize};

use pipette_mgmt_client::types::RemoteBenchmark;
use pipette_mgmt_client::{EntityTag, ServerUrl};
use pipette_plan_types::BenchmarkDefinition;

use crate::error::{Error, Result};

/// Pull bookkeeping, at `remote/state/sync.json`. No older build reads that file, so its list
/// ETag validates only what this layout stores. A detail's ETag is kept beside it, since a claim
/// rewrites a detail without touching this state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemoteSyncState {
    pub pulled_at: String,
    /// The rows of `state/index.json` the pull fetches details for: a `benchmark_type` this
    /// build knows and an id it accepts. Any other row is stored, not counted.
    pub benchmark_count: usize,
    /// The server the cached catalog and its ETags came from. `sync
    /// --server-url` can point an unregistered workspace at another host,
    /// where revalidating would let a 304 keep a catalog that host never served.
    #[serde(
        default,
        deserialize_with = "recorded_server_url",
        skip_serializing_if = "Option::is_none"
    )]
    pub server_url: Option<ServerUrl>,
    /// The list's ETag, for `state/index.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks_etag: Option<EntityTag>,
}

/// A value that does not parse is an error, not a cold pull: this program
/// writes `sync.json`, so an unusable URL in it means the file is corrupt.
///
/// `""` reads as "no server recorded": older builds wrote it for unset.
fn recorded_server_url<'de, D>(deserializer: D) -> std::result::Result<Option<ServerUrl>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)?
        .filter(|raw| !raw.trim().is_empty())
        .map(|raw| ServerUrl::try_new(raw).map_err(serde::de::Error::custom))
        .transpose()
}

/// Re-serializing the loose value spreads its `#[serde(flatten)]` parameters
/// back into the flat wire object, so the tagged enum's own deserialization
/// enforces the contract.
fn benchmark_definition_from_remote(loose: &RemoteBenchmark) -> Result<BenchmarkDefinition> {
    serde_json::to_value(loose)
        .and_then(serde_json::from_value::<BenchmarkDefinition>)
        .map_err(|source| Error::BenchmarkConversion {
            benchmark_id: loose.benchmark_id.clone(),
            source,
        })
}

/// A server benchmark that converted, kept with the object the server sent.
///
/// [`super::BenchmarkStore::put_remote`] takes only this, so a body this build cannot run is
/// never cached, and the cache keeps the keys this build does not model for a build that does.
#[derive(Debug, Clone)]
pub(crate) struct ConvertedRemote {
    definition: BenchmarkDefinition,
    sent: RemoteBenchmark,
}

impl ConvertedRemote {
    pub(crate) fn definition(&self) -> &BenchmarkDefinition {
        &self.definition
    }

    pub(crate) fn into_definition(self) -> BenchmarkDefinition {
        self.definition
    }

    pub(super) fn sent(&self) -> &RemoteBenchmark {
        &self.sent
    }
}

impl TryFrom<RemoteBenchmark> for ConvertedRemote {
    type Error = Error;

    fn try_from(sent: RemoteBenchmark) -> Result<Self> {
        let definition = benchmark_definition_from_remote(&sent)?;
        Ok(Self { definition, sent })
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{DecodeThroughputBenchmark, EvalBenchmark};

    use super::*;

    #[rstest::rstest]
    #[case::a_recorded_server(Some("https://mgmt.example.com"), Some("https://mgmt.example.com"))]
    #[case::trailing_slash_is_normalized(
        Some("https://mgmt.example.com/"),
        Some("https://mgmt.example.com")
    )]
    #[case::the_legacy_empty_value(Some(""), None)]
    #[case::no_field_at_all(None, None)]
    fn a_sync_state_reads_the_server_it_recorded(
        #[case] stored: Option<&str>,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut json = serde_json::json!({
            "pulled_at": "2026-01-01T00:00:00Z",
            "benchmark_count": 3
        });
        if let Some(stored) = stored {
            json["server_url"] = serde_json::Value::String(stored.to_string());
        }
        let state: RemoteSyncState = serde_json::from_value(json)?;
        assert_eq!(state.pulled_at, "2026-01-01T00:00:00Z");
        assert_eq!(state.benchmark_count, 3);
        assert_eq!(
            state.server_url.map(|url| url.to_string()).as_deref(),
            expected
        );
        Ok(())
    }

    #[test]
    fn an_unusable_recorded_server_is_an_error() {
        let json = serde_json::json!({
            "pulled_at": "2026-01-01T00:00:00Z",
            "benchmark_count": 3,
            "server_url": "mgmt.example.com"
        });
        let error = serde_json::from_value::<RemoteSyncState>(json)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("mgmt.example.com"), "{error}");
    }

    #[test]
    fn loose_remote_benchmark_converts_to_strict_definition() -> anyhow::Result<()> {
        let loose: RemoteBenchmark = serde_json::from_value(serde_json::json!({
            "benchmark_id": "decode_throughput_512_100",
            "benchmark_type": "decode_throughput",
            "parameter_prefill_tokens": 512,
            "parameter_decode_tokens": 100,
        }))?;
        let definition = benchmark_definition_from_remote(&loose)?;
        assert!(matches!(
            definition,
            BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                parameter_decode_tokens,
                ..
            }) if parameter_decode_tokens.into_inner() == 100
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
        match benchmark_definition_from_remote(&loose) {
            Err(Error::BenchmarkConversion { benchmark_id, .. }) => {
                assert_eq!(benchmark_id, "mystery_1");
            }
            Ok(def) => return Err(anyhow::anyhow!("expected rejection, got {def:?}")),
            Err(other) => {
                return Err(anyhow::anyhow!(
                    "expected BenchmarkConversion, got {other:?}"
                ));
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
        match benchmark_definition_from_remote(&loose) {
            Err(Error::BenchmarkConversion { benchmark_id, .. }) => {
                assert_eq!(benchmark_id, "decode_throughput_512_100");
            }
            other => {
                return Err(anyhow::anyhow!(
                    "expected BenchmarkConversion, got {other:?}"
                ));
            }
        }
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
        match benchmark_definition_from_remote(&loose)? {
            BenchmarkDefinition::Eval(EvalBenchmark { samples, .. }) => {
                assert_eq!(samples.as_deref().map(<[_]>::len), Some(1));
            }
            other @ (BenchmarkDefinition::PrefillThroughput(_)
            | BenchmarkDefinition::DecodeThroughput(_)
            | BenchmarkDefinition::EndToEndLatency(_)
            | BenchmarkDefinition::MaxMemoryUsage(_)
            | BenchmarkDefinition::VlmEval(_)
            | BenchmarkDefinition::AsrEval(_)
            | BenchmarkDefinition::VlThroughput(_)
            | BenchmarkDefinition::VlMaxMemory(_)
            | BenchmarkDefinition::VlSpecDecode(_)
            | BenchmarkDefinition::AsrThroughput(_)
            | BenchmarkDefinition::AsrMaxMemory(_)) => {
                return Err(anyhow::anyhow!("expected Eval, got {other:?}"));
            }
        }
        Ok(())
    }
}
