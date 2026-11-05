//! Turning a claimed job into the cell the runner executes.
//!
//! `pipette-mgmt-client` hands the payload over as opaque JSON, so typing it is
//! this module's job. The flags carry their own `(benchmark, runtime, model)`
//! discriminants, so a mis-authored cell is refused here rather than after the
//! benchmark body has been fetched. This host's HuggingFace token is injected
//! here too: it never travels on the wire.

use serde::Deserialize;
use serde_json::Value;

use pipette_mgmt_client::types::ClaimedJob;
use pipette_plan_types::{ClientRunSpec, RunSpecError};

use crate::hf_auth::inject_env_hf_token;
use crate::run::refuse_foreign_os;

/// A claim no client can run: its payload is unreadable or self-contradictory.
///
/// Every variant is terminal, a missing `spec` included: the queue holds only
/// job bodies this build's schema describes, so an unreadable one is
/// mis-authored and fails identically on any device. If an older body revision
/// is ever reintroduced, revisit this; the mgmt repo's `docs/plan-ingestion.md`
/// ("Schema revisions ride on capability flags") is how such bodies would be
/// kept away from this build.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UnrunnableClaimError {
    #[error("unrunnable claim: carries no `spec`")]
    NoSpec,
    #[error("unrunnable claim: spec did not parse: {source}")]
    Unparseable {
        #[source]
        source: serde_json::Error,
    },
    #[error("unrunnable claim: benchmark_id `{envelope}` disagrees with spec.benchmark `{spec}`")]
    BenchmarkMismatch { envelope: String, spec: String },
    #[error("unrunnable claim")]
    Invalid {
        #[source]
        source: RunSpecError,
    },
}

/// The claim's payload with every `auth_token` removed, for logging.
///
/// A plan carries a gated repo's token inside the model spec, and
/// `serde_json::Value`'s `Display` skips the redaction the typed form has, so
/// anything that prints a raw claim payload must strip it first.
pub(crate) fn redacted_spec(spec: &Value) -> Value {
    match spec {
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| {
                let value = if key == "auth_token" {
                    Value::String("<redacted>".to_owned())
                } else {
                    redacted_spec(value)
                };
                (key.clone(), value)
            })
            .collect::<serde_json::Map<_, _>>()
            .into(),
        Value::Array(items) => items.iter().map(redacted_spec).collect::<Vec<_>>().into(),
        leaf @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => leaf.clone(),
    }
}

/// The cell this claim asks for, ready to hand to `run_cell`.
///
/// `benchmark_id` is duplicated on the wire (the server's copy resolves the
/// catalog and attributes failures, the spec's runs). If they disagree the job
/// is refused: guessing would run the wrong work or file it under the wrong id.
pub(crate) fn run_spec_from_claim(job: &ClaimedJob) -> anyhow::Result<ClientRunSpec> {
    if job.spec.is_null() {
        return Err(UnrunnableClaimError::NoSpec.into());
    }
    // The failing payload is never quoted into the error: it reaches the
    // server as `failure_reason`, and a plan may carry an access token.
    let mut spec = ClientRunSpec::deserialize(&job.spec)
        .map_err(|source| UnrunnableClaimError::Unparseable { source })?;
    let spec_benchmark: &str = spec.benchmark.as_ref();
    if spec_benchmark != job.benchmark_id {
        return Err(UnrunnableClaimError::BenchmarkMismatch {
            envelope: job.benchmark_id.clone(),
            spec: spec_benchmark.to_owned(),
        }
        .into());
    }
    inject_env_hf_token(&mut spec.model)?;
    spec.validate()
        .map_err(|source| UnrunnableClaimError::Invalid { source })?;
    refuse_foreign_os(&spec.runtime, std::env::consts::OS)?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{Model, Runtime};

    use super::*;
    use crate::test_support::expect_gate_on_this_host;

    const MODEL: &str = r#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"m-Q4_0.gguf"}"#;
    const RUNTIME: &str = r#"{"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","version":"b5000","flavor":"macos-arm64"}"#;
    const OPENVINO_MODEL: &str = r#"{"type":"openvino","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-ov"}"#;
    const OPENVINO_RUNTIME: &str = r#"{"type":"uv_openvino","server_version":"2026.4.1","python_version":"3.11","source":{"type":"pip_requirements_text","contents":"openvino-genai==2026.4.1.0\n"}}"#;

    /// A claim for `spec_benchmark`, with `extras` merged into the spec.
    fn claim_with(
        envelope_benchmark: &str,
        spec_benchmark: &str,
        extras: Value,
    ) -> anyhow::Result<ClaimedJob> {
        let mut spec = serde_json::json!({
            "benchmark": spec_benchmark,
            "model": serde_json::from_str::<Value>(MODEL)?,
            "runtime": serde_json::from_str::<Value>(RUNTIME)?,
        });
        let target = spec
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("the spec fixture is an object"))?;
        for (key, value) in extras
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("extras must be an object"))?
        {
            target.insert(key.clone(), value.clone());
        }
        Ok(serde_json::from_value(serde_json::json!({
            "job_id": "j-1",
            "benchmark_id": envelope_benchmark,
            "time_window": "PT10M",
            "spec": spec,
        }))?)
    }

    fn claim(envelope_benchmark: &str, spec_benchmark: &str) -> anyhow::Result<ClaimedJob> {
        claim_with(envelope_benchmark, spec_benchmark, serde_json::json!({}))
    }

    /// The principal and route survive, so the fetch can sign for it.
    #[test]
    fn a_claim_carrying_a_vault_model_arrives_typed() -> anyhow::Result<()> {
        let job = claim_with(
            "prefill_throughput_256",
            "prefill_throughput_256",
            serde_json::json!({
                "model": {
                    "type": "gguf_text",
                    "source": "vault_url",
                    "url": "/vault/resources/vault_0195c3e2/content",
                    "principal_id": "org_01936d",
                },
            }),
        )?;
        let spec = run_spec_from_claim(&job)?;
        let Model::GgufText(pipette_plan_types::GgufText {
            source:
                pipette_plan_types::GgufTextSource::VaultUrl {
                    url, principal_id, ..
                },
        }) = &spec.model
        else {
            anyhow::bail!("expected a vault gguf, got {}", spec.model);
        };
        assert_eq!(url.resource_id().as_ref(), "vault_0195c3e2");
        assert_eq!(principal_id.to_string(), "org_01936d");
        Ok(())
    }

    #[test]
    fn spec_arrives_typed_and_unset_flag_groups_may_be_omitted() -> anyhow::Result<()> {
        let spec =
            run_spec_from_claim(&claim("prefill_throughput_256", "prefill_throughput_256")?)?;
        assert!(matches!(spec.model, Model::GgufText(_)));
        assert!(matches!(spec.runtime, Runtime::LlamacppCliStockTools(_)));
        assert!(spec.runtime_flags.is_none());
        assert!(spec.model_flags.is_none());
        assert!(spec.benchmark_flags.is_none());
        Ok(())
    }

    /// Values must land intact, not merely parse.
    #[test]
    fn a_spec_carrying_every_flag_group_arrives_with_its_values() -> anyhow::Result<()> {
        let job = claim_with(
            "eval_ifbench",
            "eval_ifbench",
            serde_json::json!({
                "runtime_flags": {
                    "runtime_type": "llamacpp_cli_stock_tools",
                    "model_type": "gguf_text",
                    "benchmark_type": "eval",
                    "number_gpu_layers": 99,
                    "threads": 8
                },
                "model_flags": {
                    "model_type": "gguf_text",
                    "benchmark_type": "eval",
                    "enable_thinking": true
                },
                "benchmark_flags": {
                    "runtime_type": "llamacpp_cli_stock_tools",
                    "model_type": "gguf_text",
                    "benchmark_type": "eval",
                    "http_timeout_seconds": 600,
                    "doomloop": { "exact_repeat": { "window": 4096, "required": 3 } }
                }
            }),
        )?;

        let spec = run_spec_from_claim(&job)?;
        // Each group serializes through its flat `…FlagRef` form, so its values read back as
        // flat keys.
        let runtime_flags = serde_json::to_value(
            spec.runtime_flags
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("runtime flags were dropped"))?,
        )?;
        assert_eq!(runtime_flags["number_gpu_layers"], 99);
        assert_eq!(runtime_flags["threads"], 8);

        let model_flags = serde_json::to_value(
            spec.model_flags
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model flags were dropped"))?,
        )?;
        assert_eq!(model_flags["enable_thinking"], true);

        let benchmark_flags = serde_json::to_value(
            spec.benchmark_flags
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("benchmark flags were dropped"))?,
        )?;
        assert_eq!(benchmark_flags["http_timeout_seconds"], 600);
        // mgmt still sends `doomloop`; it is absorbed and never carried out.
        assert!(benchmark_flags.get("doomloop").is_none());
        Ok(())
    }

    #[test]
    fn flags_naming_another_cell_are_refused_on_arrival() -> anyhow::Result<()> {
        let err = serde_json::from_value::<ClientRunSpec>(serde_json::json!({
            "benchmark": "eval_ifbench",
            "model": serde_json::from_str::<serde_json::Value>(MODEL)?,
            "runtime": serde_json::from_str::<serde_json::Value>(RUNTIME)?,
            "runtime_flags": {
                "runtime_type": "mlx_macos_pipette",
                "model_type": "gguf_text",
                "benchmark_type": "eval",
                "number_gpu_layers": 99
            },
        }))
        .err()
        .ok_or_else(|| anyhow::anyhow!("an mlx/gguf cell has no runtime flags"))?;
        assert!(
            err.to_string().contains("no runtime flags defined"),
            "{err}"
        );
        Ok(())
    }

    /// The rejection reason is stored on the server as `failure_reason`, so
    /// quoting the payload into it would publish a gated repo's token.
    #[test]
    fn a_rejection_never_carries_a_plan_supplied_token() -> anyhow::Result<()> {
        const TOKEN: &str = "hf_tokenthatmustnotescape";
        let gated_model: Value = serde_json::from_str(MODEL)?;
        let mut gated_model = gated_model;
        gated_model
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("the model fixture is an object"))?
            .insert("auth_token".into(), Value::String(TOKEN.to_owned()));

        // A cell that cannot resolve, so the payload is what fails.
        let job = claim_with(
            "eval_ifbench",
            "eval_ifbench",
            serde_json::json!({
                "model": gated_model,
                "runtime_flags": {
                    "runtime_type": "mlx_macos_pipette",
                    "model_type": "gguf_text",
                    "benchmark_type": "eval",
                    "number_gpu_layers": 99
                },
            }),
        )?;

        let err = run_spec_from_claim(&job)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unresolvable cell cannot produce a spec"))?;
        let reported = format!("{err:#}");
        assert!(!reported.contains(TOKEN), "token leaked into: {reported}");

        // The same holds for the payload an operator sees in the logs.
        let logged = redacted_spec(&job.spec).to_string();
        assert!(!logged.contains(TOKEN), "token leaked into: {logged}");
        Ok(())
    }

    #[test]
    fn benchmark_id_disagreeing_with_the_spec_is_rejected() -> anyhow::Result<()> {
        let err = run_spec_from_claim(&claim("prefill_throughput_256", "decode_throughput_512")?)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a claim naming two benchmarks cannot run"))?;
        let msg = format!("{err:#}");
        assert!(msg.contains("prefill_throughput_256"), "{msg}");
        assert!(msg.contains("decode_throughput_512"), "{msg}");
        Ok(())
    }

    /// An unreadable payload must survive decoding (the job it names still has
    /// to be failed) and then be refused as terminal.
    #[rstest::rstest]
    #[case::absent(serde_json::json!({
        "job_id": "j-1",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
    }))]
    #[case::unreadable(serde_json::json!({
        "job_id": "j-1",
        "benchmark_id": "prefill_throughput_256",
        "time_window": "PT10M",
        "spec": { "benchmark": "prefill_throughput_256", "model": "not-a-model" },
    }))]
    fn an_unreadable_spec_is_an_unrunnable_claim(
        #[case] body: serde_json::Value,
    ) -> anyhow::Result<()> {
        let job: ClaimedJob = serde_json::from_value(body)?;
        let err = run_spec_from_claim(&job)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unreadable spec cannot produce a cell"))?;
        assert!(
            err.chain().any(|cause| cause.is::<UnrunnableClaimError>()),
            "must be terminal, not retried: {err:#}"
        );
        Ok(())
    }

    fn openvino_claim(runtime: &str) -> anyhow::Result<ClaimedJob> {
        claim_with(
            "prefill_throughput_256",
            "prefill_throughput_256",
            serde_json::json!({
                "model": serde_json::from_str::<Value>(OPENVINO_MODEL)?,
                "runtime": serde_json::from_str::<Value>(runtime)?,
            }),
        )
    }

    /// OpenVINO runs on Linux and Windows only. A host the gate refuses
    /// leaves the claim retriable: a host on another OS can take it.
    #[test]
    fn an_openvino_claim_follows_this_hosts_os() -> anyhow::Result<()> {
        let result = run_spec_from_claim(&openvino_claim(OPENVINO_RUNTIME)?);
        if let Some(err) = expect_gate_on_this_host(result)? {
            assert!(crate::client::worker::protocol::classify_run_error(&err));
        }
        Ok(())
    }

    /// A model its runtime cannot load is terminal on every host.
    #[test]
    fn an_incompatible_claim_is_terminal() -> anyhow::Result<()> {
        let err = run_spec_from_claim(&openvino_claim(RUNTIME)?)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an incompatible pair must be refused"))?;
        assert!(
            matches!(
                err.downcast_ref::<UnrunnableClaimError>(),
                Some(UnrunnableClaimError::Invalid { .. })
            ),
            "{err:#}"
        );
        assert!(!crate::client::worker::protocol::classify_run_error(&err));
        Ok(())
    }
}
