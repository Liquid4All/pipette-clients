//! Build a result payload from a [`RunRequest`] and its [`RunResponse`], save
//! it, and optionally submit it.

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_device::{detect_device_info, detect_power_state};
use pipette_http::HttpClient;
use pipette_plan_types::{
    BenchmarkParams, BenchmarkSubmissionPayload, DeviceInfo, ExtraRuntimeConfiguration,
    MAX_CLIENT_EXTRAS_BYTES, MAX_DEVICE_READINGS_BYTES, MAX_READINESS_GATE_WAITS_BYTES, ModelType,
    PowerReading, RunRequest, RunResponse, RuntimeType, ThermalTelemetry, keep_last_elements,
};

use crate::identity::{ClientRegistrationStore, DeviceLabels};
use crate::rep_telemetry::RepTelemetry;
use crate::results::BenchmarkResultExtras;
use crate::results::{BenchmarkResultLocation, ResultsStore};

pub struct RecordSubmitOutcome {
    pub location: BenchmarkResultLocation,
    /// The job id after a successful sync, otherwise the local result id.
    pub result_id: String,
    pub job_id: Option<String>,
}

/// Guards against a record that predates the descriptor format.
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

/// An engine that returned another cell's flags would submit a record for a
/// run that did not happen, so refuse it.
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
/// state detected here. Both the CLI and the worker finish through this so they
/// report the same thing.
///
/// `labels` are read by the caller before the run, which can outlive the
/// workspace's files. `client_configuration` is passed separately because it
/// is deliberately absent from the request.
pub fn finished_run_payload(
    labels: &DeviceLabels,
    req: &RunRequest,
    outcome: &RunResponse,
    client_configuration: Option<&ExtraRuntimeConfiguration>,
    telemetry: &RepTelemetry,
) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
    let device = detect_device_info(
        labels.device_name.as_ref().map(AsRef::as_ref),
        labels.device_form_factor,
    )?;
    let power = detect_power_state();
    // No CPU variant: only the Android client, which builds its own
    // submission, detects one.
    build_submission_payload_from_run(
        outcome,
        req,
        None,
        device,
        power,
        client_configuration,
        telemetry,
    )
}

/// Save a finished run and, when `sync` is set and the result is
/// remote-pending, submit it. A sync failure is only a warning: the saved
/// result is durable.
pub fn record_and_maybe_submit_run(
    results: &ResultsStore,
    registration: &ClientRegistrationStore,
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
        let try_sync = || -> anyhow::Result<Option<String>> {
            let server_url = registration.server_url()?;
            let auth = registration.signing_identity()?;
            crate::client::sync::submit_pending_result(
                results,
                &server_url,
                http,
                &auth,
                &result_id,
            )
            .map_err(Into::into)
        };
        match try_sync() {
            Ok(Some(id)) => {
                location = BenchmarkResultLocation::RemoteSynced;
                job_id = Some(id);
            }
            Ok(None) => tracing::info!("another process is submitting the result"),
            Err(e) => tracing::warn!(
                error = %format_args!("{e:#}"),
                "sync failed, result kept locally"
            ),
        }
    }

    Ok(RecordSubmitOutcome {
        result_id: job_id.clone().unwrap_or(result_id),
        location,
        job_id,
    })
}

/// Over the cap is refused here: mgmt would answer `400`, and the result would
/// stay pending for good.
fn client_extras(outcome: &RunResponse) -> anyhow::Result<Option<String>> {
    if outcome.client_extras.is_empty() {
        return Ok(None);
    }
    let extras = serde_json::to_string(&outcome.client_extras)?;
    anyhow::ensure!(
        extras.len() <= MAX_CLIENT_EXTRAS_BYTES,
        "client_extras is {} bytes, over the {MAX_CLIENT_EXTRAS_BYTES} the server accepts",
        extras.len()
    );
    Ok(Some(extras))
}

/// Over its cap a field loses its first elements, as the server would cut it,
/// so the measurement still goes out.
fn telemetry_field(
    name: &str,
    elements: &[impl serde::Serialize],
    max_bytes: usize,
) -> anyhow::Result<Option<String>> {
    if elements.is_empty() {
        return Ok(None);
    }
    let sent = serde_json::to_string(elements)?;
    let kept = keep_last_elements(&sent, max_bytes)?;
    if kept.len() < sent.len() {
        tracing::warn!(
            field = name,
            sent_bytes = sent.len(),
            kept_bytes = kept.len(),
            "device telemetry over its cap: kept its last elements"
        );
    }
    Ok(Some(kept))
}

/// `client_configuration` is the entry the client ran this runtime under, if
/// any.
pub fn build_submission_payload_from_run(
    outcome: &RunResponse,
    req: &RunRequest,
    runtime_cpu_variant: Option<String>,
    device: DeviceInfo,
    power: PowerReading,
    client_configuration: Option<&ExtraRuntimeConfiguration>,
    telemetry: &RepTelemetry,
) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
    let model_descriptor = serde_json::to_string(&req.model.declared.without_auth_token())?;
    let runtime_descriptor = serde_json::to_string(&req.runtime.declared)?;
    ensure_descriptor("model_descriptor", &model_descriptor)?;
    ensure_descriptor("runtime_descriptor", &runtime_descriptor)?;
    ensure_flags_round_tripped(req, outcome)?;
    let benchmark_type = req.benchmark.benchmark_type();
    let payload = BenchmarkSubmissionPayload {
        benchmark_id: req.benchmark.benchmark_id().to_string(),
        // Checked by the server against its catalog, so a run against a stale
        // or edited definition is refused rather than filed under the id.
        benchmark_params: Some(BenchmarkParams::from(&req.benchmark)),
        device,
        device_battery_level: power.battery_level,
        device_power_state: power.power_state,
        device_power_save_mode: power.power_save_mode,
        thermal: ThermalTelemetry::from_series(&outcome.thermal.before, &outcome.thermal.after),
        // Unlike thermal, there is no per-rep series to fold.
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
        client_configuration: client_configuration.map(|c| c.submission_value().to_string()),
        device_hardware_readings: telemetry_field(
            "device_hardware_readings",
            &telemetry.hardware_readings,
            MAX_DEVICE_READINGS_BYTES,
        )?,
        device_os_throttling_readings: telemetry_field(
            "device_os_throttling_readings",
            &telemetry.os_throttling_readings,
            MAX_DEVICE_READINGS_BYTES,
        )?,
        readiness_gate_waits: telemetry_field(
            "readiness_gate_waits",
            &telemetry.gate_waits,
            MAX_READINESS_GATE_WAITS_BYTES,
        )?,
        runtime_cpu_variant,
        // The corpus that was actually traversed, matched or not: the server
        // records the verdict rather than rejecting a mismatch.
        observation_dataset_digest: outcome.dataset_digest.clone(),
        observation_output_digest: outcome.output_digest.clone(),
        client_version: Some(crate::CLIENT_VERSION.to_string()),
        // Only a claimed job is timed; the worker stamps it from its claim.
        device_time_ms: None,
        client_extras: client_extras(outcome)?,
        runtime_output: outcome.runtime_output.clone().capped(),
        submitted_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
        result: outcome.result_data.clone(),
    };
    let extras = BenchmarkResultExtras {
        executable: outcome.executable.clone(),
        command: outcome.command.clone(),
        stdout: outcome
            .runtime_output
            .runtime_stdout
            .clone()
            .unwrap_or_default(),
        stderr: outcome
            .runtime_output
            .runtime_stderr
            .clone()
            .unwrap_or_default(),
        turns: outcome.turns.clone(),
    };
    Ok((payload, extras))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_mgmt_client::types::ClaimedJob;
    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkFlags, BenchmarkId, BenchmarkResultData,
        DeclaredBound, EnvForward, EvalBenchmark, ExtraRuntimeConfiguration,
        MAX_RUNTIME_OUTPUT_BYTES, MaxMemoryUsageBenchmark, Model, ModelFlags as PlanModelFlags,
        NonNegativeCount, PositiveCount, PositiveSeconds, PrefillThroughputBenchmark,
        ReadinessOverrides, ReasoningEffort, RunRequest, Runtime, RuntimeFlags, RuntimeOutput,
        VlmEvalBenchmark,
    };

    use super::*;
    use crate::client::claim::run_spec_from_claim;
    use crate::rep_telemetry::{GateWaitElement, ReadingsElement, SamplePoint};

    fn outcome() -> RunResponse {
        RunResponse {
            benchmark_flags: Some(
                BenchmarkFlags::PrefillThroughputLlamacppCliStockToolsGgufText {
                    attempt_timeout_seconds: None,
                    readiness: Some(ReadinessOverrides {
                        max_wait_secs: Some(300),
                        skip_thermal: Some(false),
                        skin_below_c: None,
                    }),
                },
            ),
            ..RunResponse::new(
                BenchmarkResultData::PrefillThroughput {
                    prefill_time_ms: 12.5,
                    prefill_time_ms_stddev: None,
                },
                RuntimeOutput {
                    runtime_stdout: Some("out".into()),
                    runtime_stderr: Some("err".into()),
                },
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
            "repository_url": "github.com/ggml-org/llama.cpp",
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
            benchmark: eval_benchmark()?,
        })
    }

    /// All defaults, so the builder runs without probing the host.
    fn device() -> anyhow::Result<DeviceInfo> {
        Ok(serde_json::from_value(serde_json::json!({}))?)
    }

    fn prefill_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn eval_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("ifbench".to_owned())?,
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "ifbench".into(),
            parameter_max_tokens: PositiveCount::try_new(64)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: None,
        }))
    }

    fn vlm_eval_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
            benchmark_id: BenchmarkId::try_new("vlm_eval_mmstar".to_owned())?,
            parameter_lmms_task: "mmstar".into(),
            parameter_dataset_version: "2026.07.1".into(),
            parameter_sample_count: PositiveCount::try_new(1)?,
            parameter_max_tokens: PositiveCount::try_new(64)?,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_presence_penalty: None,
            parameter_repeat_penalty: None,
            parameter_seed: None,
            parameter_seed_rule: None,
            parameter_prompt_profile: None,
            parameter_image_tokens: None,
            dataset: None,
            recipe: None,
            samples: None,
        }))
    }

    fn max_memory_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::MaxMemoryUsage(
            MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new("max_memory_usage".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            },
        ))
    }

    /// A variant with no readiness field at all, unlike [`outcome`].
    fn max_memory_outcome() -> anyhow::Result<RunResponse> {
        Ok(RunResponse {
            benchmark_flags: Some(
                BenchmarkFlags::MaxMemoryUsageLlamacppCliStockToolsGgufText {
                    attempt_timeout_seconds: Some(PositiveSeconds::try_new(2400)?),
                },
            ),
            ..RunResponse::new(
                BenchmarkResultData::MaxMemoryUsage {
                    max_host_bytes: 1_024,
                    max_gpu_bytes: None,
                    max_npu_bytes: None,
                },
                RuntimeOutput {
                    runtime_stdout: Some("out".into()),
                    runtime_stderr: Some("err".into()),
                },
            )
        })
    }

    /// Both cases must hold: a builder that always emitted `None` would pass
    /// only the non-eval ones.
    #[rstest::rstest]
    #[case::eval_carries_the_flag(eval_benchmark()?, text_flags(None), Some("enable_thinking=true"))]
    #[case::eval_carries_the_effort(
        eval_benchmark()?,
        text_flags(Some(ReasoningEffort::High)),
        Some("enable_thinking=true,reasoning_effort=high")
    )]
    #[case::vlm_eval_carries_the_flag(
        vlm_eval_benchmark()?,
        PlanModelFlags::VlmEvalGgufVision { enable_thinking: Some(true) },
        Some("enable_thinking=true")
    )]
    #[case::non_eval_strips_the_flag(prefill_benchmark()?, text_flags(None), None)]
    #[case::non_eval_strips_the_effort(
        prefill_benchmark()?,
        text_flags(Some(ReasoningEffort::High)),
        None
    )]
    fn payload_model_flags_follow_the_benchmark_type(
        #[case] benchmark: BenchmarkDefinition,
        #[case] flags: PlanModelFlags,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &req(benchmark, Some(flags))?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        assert_eq!(payload.model_flags.as_deref(), expected);
        Ok(())
    }

    fn text_flags(reasoning_effort: Option<ReasoningEffort>) -> PlanModelFlags {
        PlanModelFlags::EvalGgufText {
            enable_thinking: Some(true),
            reasoning_effort,
        }
    }

    /// The field is `Option` for older on-disk payloads; the live path must
    /// always set it, to the string `pipette --version` prints, so a shift in
    /// the numbers can be tied to the build a bug report quotes.
    #[test]
    fn payload_reports_this_build_as_client_version() -> anyhow::Result<()> {
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        assert_eq!(
            payload.client_version.as_deref(),
            Some(crate::CLIENT_VERSION)
        );

        let wire = serde_json::to_value(&payload)?;
        assert_eq!(wire["client_version"], crate::CLIENT_VERSION);

        // The equality above is the code checking itself, so pin the shape too.
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

    #[rstest::rstest]
    #[case::resolved_entry_is_reported(
        Some(RuntimeFlags::PrefillThroughputLlamacppCliStockToolsGgufText(pipette_plan_types::PrefillThroughputLlamacppCliStockToolsGgufText {
            threads: Some(8),
            number_gpu_layers: Some(99),
            mmap: Some(false),
            load_mode: None,
            flash_attention: None,
            envs: Vec::new(),
            raw: vec![],
        })),
        // The cell's own settings only: descriptors and the id name the cell.
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
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        let got = payload
            .runtime_flags
            .as_deref()
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()?;
        assert_eq!(got, expected);
        Ok(())
    }

    /// The server rejects a non-object `benchmark_flags` and NULLs an empty
    /// one, losing the flags without failing anything locally. A gating cell
    /// always has a readiness block; a non-gating one has flags only when the
    /// plan authored a knob, since `run.rs`'s `resolved_flags` answers `None`
    /// rather than an empty variant.
    #[rstest]
    #[case::gating_cell(outcome(), prefill_benchmark()?, "readiness")]
    #[case::non_gating_cell(max_memory_outcome()?, max_memory_benchmark()?, "attempt_timeout_seconds")]
    fn payload_benchmark_flags_are_always_a_populated_object(
        #[case] outcome: RunResponse,
        #[case] benchmark: BenchmarkDefinition,
        #[case] expected_key: &str,
    ) -> anyhow::Result<()> {
        let (payload, _) = build_submission_payload_from_run(
            &outcome,
            &req(benchmark, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
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
        assert!(
            fields.contains_key(expected_key),
            "expected `{expected_key}` in {fields:?}",
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
                parallel: None,
                repetition_detection: None,
                envs: vec![
                    EnvForward::try_new("HF_TOKEN=hunter2")?,
                    EnvForward::try_new("NCCL_DEBUG")?,
                ],
                raw: vec![],
            }),
            ..outcome()
        };
        let (payload, _) = build_submission_payload_from_run(
            &outcome,
            &docker_vllm_eval_req()?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        let flags = payload.runtime_flags.unwrap_or_default();
        assert!(!flags.contains("hunter2"), "token leaked: {flags}");
        assert!(flags.contains("HF_TOKEN"), "name dropped: {flags}");
        Ok(())
    }

    /// The same run with and without a configuration submits identical
    /// `runtime_flags`. Its own field keeps values only for `GGML_*`.
    #[test]
    fn a_configured_run_records_the_configuration_beside_the_cells_flags() -> anyhow::Result<()> {
        let outcome = RunResponse {
            runtime_flags: Some(RuntimeFlags::EvalDockerVllmTorch {
                tensor_parallel_size: None,
                dtype: None,
                max_model_len: None,
                prefix_caching: None,
                gpus: None,
                shm_size: None,
                ipc: None,
                parallel: None,
                repetition_detection: None,
                envs: vec![EnvForward::try_new("HF_TOKEN=hunter2")?],
                raw: vec![],
            }),
            ..outcome()
        };
        let config = ExtraRuntimeConfiguration::DockerVllm(
            vec![
                EnvForward::try_new("CUDA_VISIBLE_DEVICES=1")?,
                EnvForward::try_new("GGML_CUDA_ENABLE_UNIFIED_MEMORY=0")?,
            ]
            .into(),
        );
        let built = |config| -> anyhow::Result<BenchmarkSubmissionPayload> {
            Ok(build_submission_payload_from_run(
                &outcome,
                &docker_vllm_eval_req()?,
                None,
                device()?,
                PowerReading::default(),
                config,
                &RepTelemetry::default(),
            )?
            .0)
        };
        let placed = built(Some(&config))?;
        let unplaced = built(None)?;

        assert_eq!(placed.runtime_flags, unplaced.runtime_flags);
        assert_eq!(unplaced.client_configuration, None);
        assert_eq!(
            placed.client_configuration.as_deref(),
            Some(r#"{"envs":["CUDA_VISIBLE_DEVICES","GGML_CUDA_ENABLE_UNIFIED_MEMORY=0"]}"#)
        );
        Ok(())
    }

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
                parallel: None,
                repetition_detection: None,
                envs: vec![],
                raw: vec![],
            }),
            ..outcome()
        };
        let err = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("mismatched flags must fail"))?;
        assert!(
            format!("{err:#}").contains("PrefillThroughput"),
            "got {err:#}"
        );
        Ok(())
    }

    /// Verbatim, since downstream tells entries apart by that text.
    #[rstest]
    #[case::labelled(Some("acme LFM3 q4"), Some("acme llama.cpp build"))]
    #[case::unlabelled(None, None)]
    fn a_claimed_custom_label_is_reported_as_claimed(
        #[case] model_label: Option<&str>,
        #[case] runtime_label: Option<&str>,
    ) -> anyhow::Result<()> {
        let labelled = |mut source: serde_json::Value, label: Option<&str>| {
            if let Some(label) = label {
                source["custom_label"] = label.into();
            }
            source
        };
        let model = serde_json::json!({
            "type": "gguf_text",
            "source": "vault_url",
            "url": "/vault/resources/vault_1/content",
            "principal_id": "org_01936d",
        });
        let runtime = serde_json::json!({
            "type": "llamacpp_cli_stock_tools",
            "source": "vault_remote_archive",
            "url": "/vault/resources/vault_2/content",
            "principal_id": "org_01936d",
            "flavor": "macos-arm64",
        });
        let job: ClaimedJob = serde_json::from_value(serde_json::json!({
            "job_id": "j-1",
            "benchmark_id": "prefill_throughput_256",
            "time_window": "PT10M",
            "spec": {
                "benchmark": "prefill_throughput_256",
                "model": labelled(model, model_label),
                "runtime": labelled(runtime, runtime_label),
            },
        }))?;
        let spec = run_spec_from_claim(&job)?;
        let (payload, _) = build_submission_payload_from_run(
            &outcome(),
            &RunRequest {
                runtime: DeclaredBound::already_bound(spec.runtime),
                model: DeclaredBound::already_bound(spec.model),
                runtime_flags: None,
                model_flags: None,
                benchmark_flags: None,
                benchmark: prefill_benchmark()?,
            },
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;

        let reported = |descriptor: &str| -> anyhow::Result<Option<serde_json::Value>> {
            let descriptor: serde_json::Value = serde_json::from_str(descriptor)?;
            Ok(descriptor.get("custom_label").cloned())
        };
        assert_eq!(
            reported(&payload.model_descriptor)?,
            model_label.map(Into::into)
        );
        assert_eq!(
            reported(&payload.runtime_descriptor)?,
            runtime_label.map(Into::into)
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

    use crate::test_support::TempWorkspace;

    fn payload_and_extras(
        telemetry: &RepTelemetry,
    ) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
        build_submission_payload_from_run(
            &outcome(),
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            telemetry,
        )
    }

    fn http() -> anyhow::Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|e| anyhow::anyhow!("{e}"))
    }

    #[test]
    fn a_finished_run_carries_the_device_labels_it_is_handed() -> anyhow::Result<()> {
        let labels = DeviceLabels {
            device_name: Some("bench-box".into()),
            device_form_factor: None,
        };
        let (payload, extras) = finished_run_payload(
            &labels,
            &req(prefill_benchmark()?, None)?,
            &outcome(),
            None,
            &RepTelemetry::default(),
        )?;
        assert_eq!(payload.device.device_name.as_ref(), "bench-box");
        assert_eq!(payload.benchmark_id, "prefill_throughput_256");
        assert_eq!(extras.stdout, "out");
        Ok(())
    }

    /// Type and params at the top level, params untagged: the shape mgmt
    /// checks against its catalog entry.
    #[test]
    fn a_finished_run_declares_the_benchmark_it_ran() -> anyhow::Result<()> {
        let (payload, _) = finished_run_payload(
            &DeviceLabels::default(),
            &req(prefill_benchmark()?, None)?,
            &outcome(),
            None,
            &RepTelemetry::default(),
        )?;
        let wire = serde_json::to_value(&payload)?;
        assert_eq!(wire["benchmark_type"], "prefill_throughput");
        assert_eq!(
            wire["benchmark_params"],
            serde_json::json!({"prefill_tokens": 256, "batch_size": 1})
        );
        assert_eq!(wire["benchmark_id"], "prefill_throughput_256");
        Ok(())
    }

    /// A long stream goes out as its tail, an uncaptured one not at all and an
    /// empty one as `""`, while the local extras keep the whole of each.
    #[rstest]
    #[case::uncaptured(None)]
    #[case::printed_nothing(Some(String::new()))]
    fn a_finished_run_submits_its_runtime_output(
        #[case] stderr: Option<String>,
    ) -> anyhow::Result<()> {
        let outcome = RunResponse {
            runtime_output: RuntimeOutput {
                runtime_stdout: Some(format!("head{}", "x".repeat(MAX_RUNTIME_OUTPUT_BYTES))),
                runtime_stderr: stderr.clone(),
            },
            ..outcome()
        };
        let (payload, extras) = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        let wire = serde_json::to_value(&payload)?;
        let stdout = wire["runtime_stdout"].as_str().unwrap_or_default();
        assert_eq!(stdout.len(), MAX_RUNTIME_OUTPUT_BYTES);
        assert!(stdout.ends_with('x') && !stdout.contains("head"));
        assert_eq!(
            wire.get("runtime_stderr").and_then(|v| v.as_str()),
            stderr.as_deref()
        );
        assert!(wire.get("client_extras").is_none());
        // The worker stamps the time from its claim; a built payload has none.
        assert!(wire.get("device_time_ms").is_none());
        assert_eq!(Some(extras.stdout), outcome.runtime_output.runtime_stdout);
        Ok(())
    }

    #[test]
    fn a_finished_run_submits_its_client_extras() -> anyhow::Result<()> {
        let outcome = RunResponse {
            client_extras: serde_json::json!({"wall_ms": 12.5})
                .as_object()
                .cloned()
                .unwrap_or_default(),
            ..outcome()
        };
        let (payload, _) = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        )?;
        assert_eq!(
            payload.client_extras.as_deref(),
            Some(r#"{"wall_ms":12.5}"#)
        );
        Ok(())
    }

    #[test]
    fn client_extras_over_the_cap_are_refused() -> anyhow::Result<()> {
        let mut extras = serde_json::Map::new();
        extras.insert("log".into(), "x".repeat(MAX_CLIENT_EXTRAS_BYTES).into());
        let outcome = RunResponse {
            client_extras: extras,
            ..outcome()
        };
        let refused = build_submission_payload_from_run(
            &outcome,
            &req(prefill_benchmark()?, None)?,
            None,
            device()?,
            PowerReading::default(),
            None,
            &RepTelemetry::default(),
        );
        let message = refused.err().map(|e| e.to_string());
        assert!(
            message
                .as_deref()
                .is_some_and(|m| m.contains("the server accepts")),
            "{message:?}"
        );
        Ok(())
    }

    /// An empty field is absent rather than `[]`.
    #[test]
    fn device_telemetry_rides_as_json_arrays() -> anyhow::Result<()> {
        let counters = serde_json::json!([{ "InstanceName": r"\_TZ.THZ0", "Temperature": 316 }]);
        let reading = ReadingsElement {
            iteration: 0,
            sample_point: SamplePoint::BeforeRep,
            facilities: serde_json::Map::from_iter([(
                "windows_thermal_zone_information_counters".to_string(),
                counters.clone(),
            )]),
        };
        let wait = GateWaitElement {
            iteration: 0,
            poll_count: 14,
            platform: "windows",
            phase: "proceeding",
            elapsed_ms: 87_400,
            observed: "ok".to_string(),
        };
        let (payload, _) = payload_and_extras(&RepTelemetry {
            hardware_readings: vec![reading],
            gate_waits: vec![wait],
            ..RepTelemetry::default()
        })?;
        let parsed = |field: Option<String>| -> anyhow::Result<serde_json::Value> {
            let field = field.ok_or_else(|| anyhow::anyhow!("the field should be sent"))?;
            Ok(serde_json::from_str(&field)?)
        };
        assert_eq!(
            parsed(payload.device_hardware_readings)?,
            serde_json::json!([{
                "iteration": 0, "sample_point": "before_rep",
                "windows_thermal_zone_information_counters": counters,
            }])
        );
        assert_eq!(
            parsed(payload.readiness_gate_waits)?,
            serde_json::json!([{
                "iteration": 0, "poll_count": 14, "platform": "windows",
                "phase": "proceeding", "elapsed_ms": 87_400, "observed": "ok",
            }])
        );
        assert_eq!(payload.device_os_throttling_readings, None);
        Ok(())
    }

    #[test]
    fn device_telemetry_over_the_cap_keeps_its_last_elements() -> anyhow::Result<()> {
        let element = |iteration, fill: &str| ReadingsElement {
            iteration,
            sample_point: SamplePoint::BeforeRep,
            facilities: serde_json::Map::from_iter([("x".to_string(), fill.into())]),
        };
        let half = MAX_DEVICE_READINGS_BYTES / 2;
        let (payload, _) = payload_and_extras(&RepTelemetry {
            os_throttling_readings: vec![
                element(0, &"a".repeat(half)),
                element(1, &"b".repeat(half)),
                element(2, "last"),
            ],
            ..RepTelemetry::default()
        })?;
        let kept: Vec<serde_json::Value> = serde_json::from_str(
            &payload
                .device_os_throttling_readings
                .ok_or_else(|| anyhow::anyhow!("the field should be sent"))?,
        )?;
        let iterations: Vec<_> = kept.iter().map(|e| e["iteration"].as_u64()).collect();
        assert_eq!(iterations, [Some(1), Some(2)]);
        Ok(())
    }

    #[rstest]
    #[case::no_sync(false)]
    #[case::sync_asked(true)]
    fn a_local_result_is_recorded_and_never_submitted(#[case] sync: bool) -> anyhow::Result<()> {
        let tw = TempWorkspace::new("record-local")?;
        let server = httpmock::MockServer::start();
        tw.register(&server.base_url())?;
        let submit = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });
        let (payload, extras) = payload_and_extras(&RepTelemetry::default())?;

        let done = record_and_maybe_submit_run(
            &tw.ws.results(),
            &tw.ws.registration(),
            &payload,
            &extras,
            BenchmarkResultLocation::Local,
            sync,
            &http()?,
        )?;

        assert_eq!(done.location, BenchmarkResultLocation::Local);
        assert!(done.job_id.is_none());
        assert_eq!(
            tw.ws.results().list_ids(BenchmarkResultLocation::Local)?,
            vec![done.result_id.clone()]
        );
        submit.assert_calls(0);
        Ok(())
    }

    #[test]
    fn a_remote_result_is_submitted_and_rekeyed_by_its_job() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("record-synced")?;
        let server = httpmock::MockServer::start();
        tw.register(&server.base_url())?;
        let submit = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });
        let (payload, extras) = payload_and_extras(&RepTelemetry::default())?;

        let done = record_and_maybe_submit_run(
            &tw.ws.results(),
            &tw.ws.registration(),
            &payload,
            &extras,
            BenchmarkResultLocation::RemotePending,
            true,
            &http()?,
        )?;

        submit.assert();
        assert_eq!(done.location, BenchmarkResultLocation::RemoteSynced);
        assert_eq!(done.result_id, "job-1");
        assert_eq!(done.job_id.as_deref(), Some("job-1"));
        let results = tw.ws.results();
        assert!(
            results
                .list_ids(BenchmarkResultLocation::RemotePending)?
                .is_empty()
        );
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemoteSynced)?,
            vec!["job-1".to_string()]
        );
        Ok(())
    }

    #[rstest]
    #[case::server_refuses(true)]
    #[case::unregistered(false)]
    fn a_failed_submit_keeps_the_result_pending(#[case] registered: bool) -> anyhow::Result<()> {
        let tw = TempWorkspace::new("record-sync-failed")?;
        let server = httpmock::MockServer::start();
        if registered {
            tw.register(&server.base_url())?;
        }
        server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/benchmarks");
            then.status(500).body("nope");
        });
        let (payload, extras) = payload_and_extras(&RepTelemetry::default())?;

        let done = record_and_maybe_submit_run(
            &tw.ws.results(),
            &tw.ws.registration(),
            &payload,
            &extras,
            BenchmarkResultLocation::RemotePending,
            true,
            &http()?,
        )?;

        assert_eq!(done.location, BenchmarkResultLocation::RemotePending);
        assert!(done.job_id.is_none());
        assert_eq!(
            tw.ws
                .results()
                .list_ids(BenchmarkResultLocation::RemotePending)?,
            vec![done.result_id.clone()]
        );
        Ok(())
    }
}
