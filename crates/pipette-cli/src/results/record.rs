//! The record-and-submit flow: build the payload from an
//! [`RunResponse`] + [`RunRequest`], save it, optionally submit.

use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_device::{detect_device_info, detect_power_state};
use pipette_http::HttpClient;
use pipette_mgmt_client::MgmtClient;
use pipette_plan_types::device::DeviceInfo;
use pipette_plan_types::result::BenchmarkSubmissionPayload;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;
use pipette_plan_types::thermal::{PowerState, ThermalTelemetry};
use pipette_plan_types::{ModelType, RuntimeType};

use crate::identity::IdentityStore;
use crate::results::BenchmarkResultExtras;
use crate::results::{BenchmarkResultLocation, ResultsStore};

/// Where a recorded result landed and how it's keyed.
pub struct RecordSubmitOutcome {
    pub location: BenchmarkResultLocation,
    /// The id the result is stored under — the job id after a successful sync,
    /// otherwise the locally-generated result id.
    pub result_id: String,
    /// The management-server job id, `Some` only after a successful submit.
    pub job_id: Option<String>,
}

/// A submittable result must carry `field` as non-empty JSON — guards a payload
/// built from a record that predates the descriptor format.
fn ensure_descriptor(field: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty(),
        "result is missing `{field}`; re-run the benchmark (records predating the \
         descriptor format can't be submitted)"
    );
    serde_json::from_str::<serde_json::Value>(value)
        .map_err(|e| anyhow::anyhow!("result `{field}` is not valid JSON: {e}"))?;
    Ok(())
}

/// The response's `runtime_flags` must be the request's, resolved — same cell,
/// same variant. An engine that returned another cell's flags would submit a
/// record for a run that didn't happen, so refuse rather than record it.
fn ensure_flags_round_tripped(req: &RunRequest, outcome: &RunResponse) -> anyhow::Result<()> {
    let Some(returned) = outcome.runtime_flags.as_ref() else {
        return Ok(());
    };
    let axes = (
        req.benchmark.benchmark_type(),
        RuntimeType::of(&req.runtime.declared),
        ModelType::of(&req.model.declared),
    );
    anyhow::ensure!(
        returned.axes() == axes,
        "engine returned runtime flags for {:?}, but the run was {axes:?}",
        returned.axes()
    );
    Ok(())
}

/// What a finished run submits and what it stores, with the device and power
/// state detected here.
///
/// Both finish paths go through this: the CLI records then optionally syncs, the
/// worker completes a claim then records. Describing the run in one place is
/// what keeps the two from drifting in what they report.
///
/// Runtime flags come from `outcome.runtime_flags` — the cell's plan entry as
/// the engine ran it, serialized like the descriptors.
pub fn finished_run_payload(
    identity: &IdentityStore,
    req: &RunRequest,
    outcome: &RunResponse,
) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
    let labels = identity.get_device_labels()?;
    let device = detect_device_info(
        labels.device_name.as_ref().map(AsRef::as_ref),
        labels.device_form_factor,
    )?;
    let power = detect_power_state();
    // No CPU variant: the desktop client doesn't detect one. The payload field
    // is filled by the Android client, which builds its own submission.
    build_submission_payload_from_run(outcome, req, None, device, power)
}

/// Save a finished run under `results/{local,remote}` and — when `sync` is set
/// and the result is remote-pending — submit it. A sync failure is a warning,
/// not a run failure: the on-disk artifact is durable.
///
/// `http` is the process-shared client.
pub fn record_and_maybe_submit_run(
    results: &ResultsStore,
    identity: &IdentityStore,
    payload: &BenchmarkSubmissionPayload,
    extras: &BenchmarkResultExtras,
    location: BenchmarkResultLocation,
    sync: bool,
    http: &HttpClient,
) -> anyhow::Result<RecordSubmitOutcome> {
    let result_id = uuid::Uuid::new_v4().to_string();
    let mut location = location;
    results.save_result(location, &result_id, payload, extras)?;

    let mut job_id = None;
    if sync && location == BenchmarkResultLocation::RemotePending {
        let try_sync = || -> anyhow::Result<String> {
            let registration = identity.require_registration()?;
            let auth = identity.signing_identity()?;
            let client = MgmtClient::new(registration.server_url)?;
            crate::client::sync::submit_pending_result(
                results, &client, http, &auth, &result_id, payload,
            )
            .map_err(Into::into)
        };
        match try_sync() {
            Ok(id) => {
                location = BenchmarkResultLocation::RemoteSynced;
                job_id = Some(id);
            }
            Err(e) => log::warn!("sync failed, result kept locally: {e:#}"),
        }
    }

    Ok(RecordSubmitOutcome {
        result_id: job_id.clone().unwrap_or(result_id),
        location,
        job_id,
    })
}

/// Wire payload from [`RunRequest`] + outcome.
///
/// `model_flags` come from [`RunRequest::model_flags`] (plan-declared).
pub fn build_submission_payload_from_run(
    outcome: &RunResponse,
    req: &RunRequest,
    runtime_cpu_variant: Option<String>,
    device: DeviceInfo,
    power: PowerState,
) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
    let model_descriptor = serde_json::to_string(&req.model.declared.without_auth_token())?;
    let runtime_descriptor = serde_json::to_string(&req.runtime.declared)?;
    ensure_descriptor("model_descriptor", &model_descriptor)?;
    ensure_descriptor("runtime_descriptor", &runtime_descriptor)?;
    ensure_flags_round_tripped(req, outcome)?;
    let benchmark_type = req.benchmark.benchmark_type();
    let payload = BenchmarkSubmissionPayload {
        benchmark_id: req.benchmark.benchmark_id().to_string(),
        device,
        device_battery_level: power.battery_level,
        device_power_state: power.power_state,
        device_power_save_mode: power.power_save_mode,
        thermal: ThermalTelemetry::from_series(&outcome.thermal.before, &outcome.thermal.after),
        // Passed straight through: the engine observed it, and unlike thermal
        // there is no per-rep series for the caller to fold.
        memory: outcome.memory,
        model_descriptor,
        runtime_descriptor,
        model_flags: req
            .model_flags
            .as_ref()
            .and_then(|f| f.submission_string(benchmark_type)),
        benchmark_flags: outcome
            .benchmark_flags
            .as_ref()
            .map(|f| f.submission_value().to_string()),
        runtime_flags: outcome
            .runtime_flags
            .as_ref()
            .map(|f| f.submission_value().to_string()),
        runtime_cpu_variant,
        client_version: Some(crate::CLIENT_VERSION.to_string()),
        submitted_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
        result: outcome.result_data.clone(),
    };
    let extras = BenchmarkResultExtras {
        executable: outcome.executable.clone(),
        command: outcome.command.clone(),
        stdout: outcome.stdout.clone(),
        stderr: outcome.stderr.clone(),
    };
    Ok((payload, extras))
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark, PrefillThroughput};
    use pipette_plan_types::result::BenchmarkResultData;
    use pipette_plan_types::run::{DeclaredBound, RunRequest};
    use pipette_plan_types::{
        BenchmarkFlags, Model, ModelFlags as PlanModelFlags, ReadinessOverrides, Runtime,
        RuntimeFlags,
    };

    use super::*;

    /// A prefill outcome whose readiness resolved to a gated run on the
    /// platform default — the shape every measured cell reports.
    fn outcome() -> RunResponse {
        RunResponse {
            benchmark_flags: Some(BenchmarkFlags::PrefillLlamacppCliStockToolsGgufText {
                readiness: Some(ReadinessOverrides {
                    max_wait_secs: Some(300),
                    skip_thermal: Some(false),
                }),
            }),
            ..RunResponse::new(
                BenchmarkResultData::PrefillThroughput {
                    prefill_time_ms: 12.5,
                    prefill_time_ms_stddev: None,
                },
                "out".into(),
                "err".into(),
            )
        }
    }

    fn req(
        benchmark: BenchmarkDefinition,
        model_flags: Option<PlanModelFlags>,
    ) -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_text",
            "source": "huggingface",
            "org": "o",
            "repo_name": "r",
            "path": "m-Q4_0.gguf"
        }))?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "llamacpp_cli_stock_tools",
            "source": "github_release",
            "version": "b5000",
            "flavor": "macos-arm64"
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags,
            benchmark_flags: None,
            benchmark,
        })
    }

    /// A docker-vLLM + torch eval run — the cell shape that carries `envs`.
    fn docker_vllm_eval_req() -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "torch",
            "source": "huggingface",
            "org": "o",
            "repo_name": "r"
        }))?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "docker_vllm",
            "image_name": "vllm/vllm-openai",
            "image_tag": "v0.21.0",
            "flavor": "nvidia_gpu"
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: eval_benchmark(),
        })
    }

    /// A hermetic all-defaults `DeviceInfo` (every field is `#[serde(default)]`),
    /// so the payload builder is exercised without probing the host.
    fn device() -> anyhow::Result<DeviceInfo> {
        Ok(serde_json::from_value(serde_json::json!({}))?)
    }

    fn prefill_benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_256".into(),
            parameter_prefill_tokens: 256,
        })
    }

    fn eval_benchmark() -> BenchmarkDefinition {
        BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "ifbench".into(),
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "ifbench".into(),
            parameter_max_tokens: 64,
            parameter_mcq_choices: None,
            samples: None,
        })
    }

    /// The wire `model_flags` field is built from the *declared* flags routed
    /// through `submission_string`, so both the eval pass-through and the
    /// non-eval strip have to hold end to end — a payload builder that always
    /// emitted `None` would satisfy only the second.
    #[rstest::rstest]
    #[case::eval_carries_the_flag(eval_benchmark(), Some("enable_thinking=true"))]
    #[case::non_eval_strips_the_flag(prefill_benchmark(), None)]
    fn payload_model_flags_follow_the_benchmark_type(
        #[case] benchmark: BenchmarkDefinition,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let flags = PlanModelFlags::EvalGgufText {
            enable_thinking: Some(true),
        };
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &req(benchmark, Some(flags))?,
            None,
            device()?,
            PowerState::default(),
        )?;
        assert_eq!(payload.model_flags.as_deref(), expected);
        Ok(())
    }

    /// Every measured run names the build that produced it, and names it with
    /// the same string `pipette --version` prints — the warehouse column is
    /// only useful for attributing a shift in the numbers if it matches what a
    /// bug report quotes. The field is `Option` for older on-disk payloads, so
    /// this pins that the live path never leaves it unset.
    #[test]
    fn payload_reports_this_build_as_client_version() -> anyhow::Result<()> {
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &req(prefill_benchmark(), None)?,
            None,
            device()?,
            PowerState::default(),
        )?;
        assert_eq!(
            payload.client_version.as_deref(),
            Some(crate::CLIENT_VERSION)
        );

        let wire = serde_json::to_value(&payload)?;
        assert_eq!(wire["client_version"], crate::CLIENT_VERSION);

        // Comparing against the const alone would be the code checking itself,
        // so pin the shape too: an empty value would still satisfy the equality
        // above. The value is the release's own version string (`ci/version.sh`,
        // e.g. "2026.08.1-0-g58c2adbf16") or "dev" for a local build — nothing
        // is wrapped around it, since the point is that it equals the GitHub
        // release name rather than merely containing it.
        let reported = wire["client_version"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("client_version must be a string"))?;
        assert!(!reported.is_empty());
        assert!(
            !reported.contains(char::is_whitespace),
            "client_version must be the bare release version, got {reported:?}"
        );
        Ok(())
    }

    /// The wire `runtime_flags` field is what the response carried back — the
    /// request's flags as the run resolved them — as canonical JSON, and absent
    /// when the engine reported none (a runtime that takes no flags).
    #[rstest::rstest]
    #[case::resolved_entry_is_reported(
        Some(RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
            threads: Some(8),
            number_gpu_layers: Some(99),
            mmap: Some(false),
            flash_attention: None,
            raw: vec![],
        }),
        // The cell's own settings only — the payload names the cell through
        // its descriptors and benchmark id.
        Some(serde_json::json!({
            "threads": 8,
            "number_gpu_layers": 99,
            "mmap": false,
        })),
    )]
    #[case::no_flags_is_absent(None, None)]
    fn payload_runtime_flags_carry_what_the_response_returned(
        #[case] returned: Option<RuntimeFlags>,
        #[case] expected: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        let outcome = RunResponse {
            runtime_flags: returned,
            ..outcome()
        };
        let (payload, _) = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark(), None)?,
            None,
            device()?,
            PowerState::default(),
        )?;
        let got = payload
            .runtime_flags
            .as_deref()
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()?;
        assert_eq!(got, expected);
        Ok(())
    }

    /// An env forward reaches the payload by name only — its value may be a
    /// token, and the payload leaves the device.
    /// The two shapes the server refuses or discards: it rejects a
    /// `benchmark_flags` that is not a JSON object, and NULLs a top-level empty
    /// one. Neither can be produced here — readiness always resolves to
    /// something — and this pins that, because a payload that tripped either
    /// would lose the run's gate state without failing anything locally.
    #[test]
    fn payload_benchmark_flags_are_always_a_populated_object() -> anyhow::Result<()> {
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &req(prefill_benchmark(), None)?,
            None,
            device()?,
            PowerState::default(),
        )?;

        let raw = payload
            .benchmark_flags
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("benchmark_flags must be reported"))?;
        let value: serde_json::Value = serde_json::from_str(raw)?;
        let fields = value
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("benchmark_flags must be a JSON object"))?;
        assert!(!fields.is_empty(), "an empty object is stored as null");
        assert_eq!(
            fields.get("readiness"),
            Some(&serde_json::json!({"max_wait_secs": 300, "skip_thermal": false}))
        );
        Ok(())
    }

    #[test]
    fn payload_runtime_flags_drop_env_values() -> anyhow::Result<()> {
        let outcome = RunResponse {
            runtime_flags: Some(RuntimeFlags::EvalDockerVllmTorch {
                tensor_parallel_size: None,
                dtype: None,
                max_model_len: None,
                prefix_caching: None,
                gpus: None,
                shm_size: None,
                ipc: None,
                envs: vec!["HF_TOKEN=hunter2".into(), "NCCL_DEBUG".into()],
                raw: vec![],
            }),
            ..outcome()
        };
        let (payload, _) = build_submission_payload_from_run(
            &outcome,
            &docker_vllm_eval_req()?,
            None,
            device()?,
            PowerState::default(),
        )?;
        let flags = payload.runtime_flags.unwrap_or_default();
        assert!(!flags.contains("hunter2"), "token leaked: {flags}");
        assert!(flags.contains("HF_TOKEN"), "name dropped: {flags}");
        Ok(())
    }

    /// The response's flags are the request's, resolved. Flags naming another
    /// cell mean the engine answered a question it wasn't asked, so the record
    /// is refused rather than written.
    #[test]
    fn flags_for_another_cell_are_refused() -> anyhow::Result<()> {
        let outcome = RunResponse {
            runtime_flags: Some(RuntimeFlags::EvalDockerVllmTorch {
                tensor_parallel_size: None,
                dtype: None,
                max_model_len: None,
                prefix_caching: None,
                gpus: None,
                shm_size: None,
                ipc: None,
                envs: vec![],
                raw: vec![],
            }),
            ..outcome()
        };
        let err = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark(), None)?,
            None,
            device()?,
            PowerState::default(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("mismatched flags must fail"))?;
        assert!(
            format!("{err:#}").contains("PrefillThroughput"),
            "got {err:#}"
        );
        Ok(())
    }

    #[test]
    fn empty_descriptor_is_rejected() -> anyhow::Result<()> {
        let err = ensure_descriptor("model_descriptor", "")
            .err()
            .ok_or_else(|| anyhow::anyhow!("empty descriptor must fail"))?;
        assert!(format!("{err:#}").contains("model_descriptor"));
        Ok(())
    }
}
