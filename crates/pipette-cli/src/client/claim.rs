//! Turning a claimed job into the cell the runner executes.
//!
//! `pipette-mgmt-client` hands the work payload over as opaque JSON — it models
//! the wire and takes no view of what a cell is — so typing it is this module's
//! job. One `from_value` replaces the per-field parsing the flat claim needed:
//! flags carry their own `(benchmark, runtime, model)` discriminants, so a
//! mis-authored cell is refused here rather than after the benchmark body has
//! been fetched.
//!
//! Beyond the parse, two things the client owns and the server cannot supply:
//! agreeing the envelope's `benchmark_id` with the spec, and injecting this
//! host's HuggingFace token, which never travels on the wire in either
//! direction.
//!
//! A pure function over a [`ClaimedJob`] — no workspace, no network — which is
//! what makes it testable apart from the claim loop.

use serde::Deserialize;
use serde_json::Value;

use pipette_mgmt_client::types::ClaimedJob;
use pipette_plan_types::ClientRunSpec;

use crate::hf_auth::inject_env_hf_token;

/// A claim no client can run: its payload is unreadable or self-contradictory.
///
/// Every variant is terminal, including a body carrying no `spec` at all. There
/// is no earlier job-body revision in circulation to mistake one for: the queue
/// holds only bodies this build's schema describes, so a payload that cannot be
/// read is mis-authored rather than addressed to the wrong client. Retrying it,
/// here or on another device, fails identically, so the worker reports it
/// terminally rather than letting the lease lapse and the job be re-served until
/// it expires.
///
/// Were an older revision ever reintroduced, this is the decision to revisit
/// first — and `job_schema:<n>` capability flags (`pipette-mgmt`
/// `docs/plan-ingestion.md` §7) are the mechanism that would keep those bodies
/// away from this build instead.
///
/// Distinct variants so a caller can match on the rejection kind instead of
/// parsing an error string, and so the serde failure keeps its own type rather
/// than being flattened into a message.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UnrunnableClaim {
    #[error("unrunnable claim: carries no `spec`")]
    NoSpec,
    #[error("unrunnable claim: spec did not parse: {source}")]
    Unparseable {
        #[source]
        source: serde_json::Error,
    },
    #[error("unrunnable claim: benchmark_id `{envelope}` disagrees with spec.benchmark `{spec}`")]
    BenchmarkMismatch { envelope: String, spec: String },
    #[error("unrunnable claim: model `{model}` is not compatible with runtime `{runtime}`")]
    Incompatible { model: String, runtime: String },
}

/// The claim's payload with every `auth_token` removed, for logging.
///
/// A plan carries the token for a gated repo inside the model spec
/// (`HfRepo::auth_token`), and `serde_json::Value`'s `Display` honours none of
/// the redaction the typed form has — not `AuthToken`'s `Debug`, not
/// `Model::without_auth_token`. Anything that prints a raw claim payload has to
/// strip it first.
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
        leaf => leaf.clone(),
    }
}

/// The cell this claim asks for, ready to hand to `run_cell`.
///
/// `benchmark_id` is duplicated on the wire — the server needs its own copy to
/// resolve the catalog and to attribute synthetic failures, and the spec needs
/// one to run. They must agree: a job whose envelope and payload name different
/// benchmarks is mis-authored, and guessing which one was meant would silently
/// run the wrong work or file the result against the wrong id.
pub(crate) fn run_spec_from_claim(job: &ClaimedJob) -> anyhow::Result<ClientRunSpec> {
    if job.spec.is_null() {
        return Err(UnrunnableClaim::NoSpec.into());
    }
    // Borrowed: the payload is parsed, not consumed, and it can be large.
    // The failing payload is never quoted into the error — it reaches the
    // server as `failure_reason`, and a plan may carry an access token.
    let mut spec = ClientRunSpec::deserialize(&job.spec)
        .map_err(|source| UnrunnableClaim::Unparseable { source })?;
    let spec_benchmark: &str = spec.benchmark.as_ref();
    if spec_benchmark != job.benchmark_id {
        return Err(UnrunnableClaim::BenchmarkMismatch {
            envelope: job.benchmark_id.clone(),
            spec: spec_benchmark.to_owned(),
        }
        .into());
    }
    inject_env_hf_token(&mut spec.model)?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{Model, Runtime};

    use super::*;

    const MODEL: &str = r#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"m-Q4_0.gguf"}"#;
    const RUNTIME: &str = r#"{"type":"llamacpp_cli_stock_tools","source":"github_release","version":"b5000","flavor":"macos-arm64"}"#;

    /// A claim for `spec_benchmark`, with `extras` merged into the spec — the
    /// flag groups, or anything else a case needs to vary.
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

    /// The flag groups are the reason the payload is typed here rather than
    /// parsed field by field further in: each carries its own
    /// `(benchmark, runtime, model)` discriminants, so the cell is validated on
    /// arrival. Values must land intact, not merely parse.
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
        // Re-emitting is how the values are read back: each group serializes
        // through its flat `…FlagRef` form.
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
        assert_eq!(benchmark_flags["doomloop"]["exact_repeat"]["window"], 4096);
        assert_eq!(benchmark_flags["doomloop"]["exact_repeat"]["required"], 3);
        Ok(())
    }

    /// A cell whose flags name a different runtime than the spec runs is
    /// mis-authored; it must die at the claim, not at launch.
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

    /// A plan carries the access token for a gated repo inside the model spec,
    /// and the rejection reason for an unreadable payload is submitted to the
    /// server as `failure_reason` and stored there. Quoting the payload into
    /// that message would publish the token, defeating `AuthToken`'s redacting
    /// `Debug` and `Model::without_auth_token`.
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

    /// An unreadable payload must survive decoding — the job it names still has
    /// to be failed — and then be refused as terminal.
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
            err.chain().any(|cause| cause.is::<UnrunnableClaim>()),
            "must be terminal, not retried: {err:#}"
        );
        Ok(())
    }
}
