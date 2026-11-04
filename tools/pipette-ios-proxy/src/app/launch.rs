//! A claim turned into the app's arguments, or refused terminally.

use serde_json::{Map, Value};

use pipette_mgmt_client::types::ClaimedJob;
use pipette_plan_types::{BenchmarkType, Runtime};

use super::{CLAIM_KEY, HEADLESS_TOKEN, SUBMIT_ON};

/// The `spec` keys this proxy turns into app arguments. The set is
/// `ClientRunSpec`'s; the argv order is `ios_headless_args`', plus
/// `benchmark-flags=`, which it does not emit.
const SPEC_BENCHMARK: &str = "benchmark";

const SPEC_MODEL: &str = "model";

const SPEC_RUNTIME: &str = "runtime";

const SPEC_RUNTIME_FLAGS: &str = "runtime_flags";

const SPEC_BENCHMARK_FLAGS: &str = "benchmark_flags";

/// Composite-key fields stripped from `runtime_flags` and `benchmark_flags` before
/// the knobs go on the wire, as `RuntimeFlags::knobs_json` strips them: the app
/// resolves its cell from `runtime=`, `model=` and `benchmarks=` first.
const FLAGS_AXIS_KEYS: [&str; 3] = ["runtime_type", "model_type", "benchmark_type"];

/// The one runtime that takes no `model=`: AFM is supplied by the OS.
const RUNTIME_TYPE_APPLE_FOUNDATION: &str = "apple_foundation";

#[derive(Debug, Clone)]
pub(crate) struct LaunchPlan {
    /// Beginning with [`HEADLESS_TOKEN`] and ending with [`SUBMIT_ON`] and the
    /// claim.
    pub(crate) app_args: Vec<String>,
    pub(crate) runtime_json: String,
    /// The same descriptor, typed, for choosing which build to install. Kept
    /// BESIDE the JSON, which is forwarded byte for byte; this copy is never sent.
    pub(crate) runtime: Runtime,
}

/// A claim this proxy cannot turn into a launch. Every variant is TERMINAL:
/// a body that cannot be read is mis-authored, and a retry fails identically.
#[derive(Debug, thiserror::Error)]
pub(crate) enum UnrunnableClaimError {
    #[error("the claim carried no `spec`")]
    MissingSpec,
    #[error("the claim's `spec` is not a JSON object")]
    SpecNotAnObject,
    #[error("the claim's `spec` names no runtime this build can be asked to run")]
    MissingRuntime,
    /// Well-formed JSON that `pipette-plan-types` cannot name: no build can be
    /// matched against it, and a newer runtime needs a newer proxy.
    #[error("the claim's `spec` names a runtime this proxy cannot read ({0})")]
    UnreadableRuntime(String),
    #[error("the claim's `spec` names no benchmark")]
    MissingBenchmark,
    /// The spec's `benchmark` and the envelope's `benchmark_id` disagree. Refused
    /// rather than resolved either way, as `pipette-cli`'s `BenchmarkMismatch` is:
    /// submissions stamp the envelope's id, but the spec describes the work.
    #[error(
        "the claim's `spec` names benchmark `{spec}` but the claim envelope says `{envelope}`; \
         the two have to agree for the result to be filed against the right job"
    )]
    BenchmarkMismatch { spec: String, envelope: String },
    #[error("the claim's `spec` carries `{key}`, which this proxy cannot interpret")]
    UnsupportedKey { key: String },
    /// Refused rather than dropped, for the reason an unknown key is.
    #[error("the claim's `{group}` is not a JSON object")]
    FlagsNotAnObject { group: &'static str },
    /// Refused rather than applied to this cell, as `pipette-cli`'s `validate_spec_flags`
    /// refuses it.
    #[error("the claim's `{group}` names `{axis}` {named}, but the claim runs `{runs}`")]
    FlagsForAnotherCell {
        group: &'static str,
        axis: &'static str,
        named: String,
        runs: String,
    },
}

/// Translate a claim into app arguments, or refuse it terminally. A pure function
/// over the claim.
///
/// `runtime` is forwarded VERBATIM: the app compares it against `Runtime.thisBuild`
/// and refuses any mismatch. An unknown `key=` is a hard refusal, so nothing is
/// added speculatively. `benchmarks=` comes from `spec.benchmark` and must
/// equal the envelope's `benchmark_id`.
pub(crate) fn launch_plan_from_claim(job: &ClaimedJob) -> Result<LaunchPlan, UnrunnableClaimError> {
    if job.spec.is_null() {
        return Err(UnrunnableClaimError::MissingSpec);
    }
    let spec = job
        .spec
        .as_object()
        .ok_or(UnrunnableClaimError::SpecNotAnObject)?;

    // A spec key with no `key=` to carry it is REFUSED, not dropped as the iOS
    // transport in `pipette-plan-types` drops `model_flags`: a knob the phone
    // never saw comes back as a measurement that looks like it honoured it. A
    // `null` value asked for nothing and is ignored.
    let interpretable = |key: &str, value: &Value| {
        matches!(
            key,
            SPEC_BENCHMARK | SPEC_MODEL | SPEC_RUNTIME | SPEC_RUNTIME_FLAGS | SPEC_BENCHMARK_FLAGS
        ) || value.is_null()
    };
    if let Some((key, _)) = spec.iter().find(|(key, value)| !interpretable(key, value)) {
        return Err(UnrunnableClaimError::UnsupportedKey { key: key.clone() });
    }

    // Must be an object: the app refuses a bare type tag as "names a runtime
    // type, not a build", and a `<scheme>://` URI outright.
    let runtime = spec
        .get(SPEC_RUNTIME)
        .filter(|runtime| runtime.is_object())
        .ok_or(UnrunnableClaimError::MissingRuntime)?;
    let runtime_json = compact_json(runtime).ok_or(UnrunnableClaimError::MissingRuntime)?;

    // From the SPEC, and it must EQUAL the envelope's `benchmark_id`. Neither
    // half is optional or normalised: trimming a padded value would run a job
    // under an id its envelope does not carry.
    let benchmark = spec
        .get(SPEC_BENCHMARK)
        .and_then(Value::as_str)
        .filter(|benchmark| !benchmark.is_empty())
        .ok_or(UnrunnableClaimError::MissingBenchmark)?;
    if benchmark != job.benchmark_id {
        return Err(UnrunnableClaimError::BenchmarkMismatch {
            spec: benchmark.to_string(),
            envelope: job.benchmark_id.clone(),
        });
    }

    let apple_foundation =
        runtime.get("type").and_then(Value::as_str) == Some(RUNTIME_TYPE_APPLE_FOUNDATION);
    let model = spec.get(SPEC_MODEL).filter(|model| !model.is_null());
    // A non-AFM cell with no model parses as the app's catalog-selector form,
    // which picks its own work: a launch that measures something nobody asked for.
    let model_json = match model {
        Some(_) if apple_foundation => None,
        Some(model) => Some(compact_json(model).ok_or(UnrunnableClaimError::MissingRuntime)?),
        None if apple_foundation => None,
        None => return Err(UnrunnableClaimError::MissingRuntime),
    };
    // In `FLAGS_AXIS_KEYS` order. `None` where the claim cannot say (no model, or an id
    // no prefix classifies), which leaves that axis to the app.
    let type_tag = |value: &Value| value.get("type").and_then(Value::as_str).map(str::to_owned);
    let cell = [
        type_tag(runtime),
        model.and_then(type_tag),
        BenchmarkType::from_id(benchmark).map(|kind| kind.to_string()),
    ];

    let mut app_args = vec![
        HEADLESS_TOKEN.to_string(),
        format!("runtime={runtime_json}"),
    ];
    if let Some(model_json) = model_json {
        app_args.push(format!("model={model_json}"));
    }
    app_args.push(format!("benchmarks={benchmark}"));
    if let Some(knobs) = group_knobs(spec, SPEC_RUNTIME_FLAGS, &cell)?
        .and_then(|knobs| compact_json(&Value::Object(knobs)))
    {
        app_args.push(format!("runtime-flags={knobs}"));
    }
    // Omitted when only the axes were set, so a build without the key still runs a
    // cell that asked for nothing.
    if let Some(knobs) = group_knobs(spec, SPEC_BENCHMARK_FLAGS, &cell)?
        .filter(|knobs| !knobs.is_empty())
        .and_then(|knobs| compact_json(&Value::Object(knobs)))
    {
        app_args.push(format!("benchmark-flags={knobs}"));
    }
    // Never omitted: the app's default is not uniform (a `model=` run defaults
    // to 0). The claim files the app's upload against this job.
    app_args.push(SUBMIT_ON.to_string());
    app_args.push(format!("{CLAIM_KEY}={}", job.job_id));

    // Parsed AFTER the argv, so the app gets the server's bytes, not a round trip.
    let runtime = serde_json::from_str::<Runtime>(&runtime_json)
        .map_err(|error| UnrunnableClaimError::UnreadableRuntime(error.to_string()))?;

    Ok(LaunchPlan {
        app_args,
        runtime_json,
        runtime,
    })
}

/// A flag group's knobs, its axes checked against the claim's `cell` and then stripped.
/// `None` when the spec carries no group.
fn group_knobs(
    spec: &Map<String, Value>,
    group: &'static str,
    cell: &[Option<String>; 3],
) -> Result<Option<Map<String, Value>>, UnrunnableClaimError> {
    let Some(flags) = spec.get(group).filter(|flags| !flags.is_null()) else {
        return Ok(None);
    };
    let entry = flags
        .as_object()
        .ok_or(UnrunnableClaimError::FlagsNotAnObject { group })?;
    let mismatch = FLAGS_AXIS_KEYS.iter().zip(cell).find_map(|(axis, runs)| {
        let (named, runs) = (entry.get(*axis)?, runs.as_deref()?);
        (named.as_str() != Some(runs)).then(|| (*axis, named.to_string(), runs.to_owned()))
    });
    if let Some((axis, named, runs)) = mismatch {
        return Err(UnrunnableClaimError::FlagsForAnotherCell {
            group,
            axis,
            named,
            runs,
        });
    }
    let mut knobs = entry.clone();
    knobs.retain(|key, _| !FLAGS_AXIS_KEYS.contains(&key.as_str()));
    Ok(Some(knobs))
}

/// Compact JSON preserving the server's key order, which needs `serde_json`'s
/// `preserve_order` feature (pinned in `Cargo.toml`): without it every object
/// comes back ALPHABETISED and the forwarded document is no longer verbatim. Do
/// not drop the feature to shrink the dependency graph.
fn compact_json(value: &Value) -> Option<String> {
    serde_json::to_string(value).ok()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::app::RUNTIME_WIRE;

    const MODEL_WIRE: &str = r#"{"type":"gguf_text","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-GGUF","path":"LFM2.5-350M-Q4_0.gguf"}"#;

    fn claim(spec: Value) -> ClaimedJob {
        ClaimedJob {
            job_id: "job-1".to_string(),
            benchmark_id: "decode_throughput_256".to_string(),
            time_window: "PT10M".to_string(),
            expires_at: None,
            client_versions: None,
            spec,
        }
    }

    fn spec() -> Value {
        json!({
            "benchmark": "decode_throughput_256",
            "model": serde_json::from_str::<Value>(MODEL_WIRE).unwrap_or(Value::Null),
            "runtime": serde_json::from_str::<Value>(RUNTIME_WIRE).unwrap_or(Value::Null),
            "runtime_flags": Value::Null,
            "model_flags": Value::Null,
            "benchmark_flags": Value::Null,
        })
    }

    fn with(key: &str, value: Value) -> Value {
        let mut spec = spec();
        if let Some(fields) = spec.as_object_mut() {
            fields.insert(key.to_string(), value);
        }
        spec
    }

    /// The argv carries the server's documents byte for byte, member order
    /// included; that needs `serde_json`'s `preserve_order`.
    #[test]
    fn the_plan_runner_argument_order_is_reproduced() -> anyhow::Result<()> {
        let plan = launch_plan_from_claim(&claim(spec()))?;
        assert_eq!(
            plan.app_args,
            vec![
                HEADLESS_TOKEN.to_string(),
                format!("runtime={RUNTIME_WIRE}"),
                format!("model={MODEL_WIRE}"),
                "benchmarks=decode_throughput_256".to_string(),
                SUBMIT_ON.to_string(),
                "claim=job-1".to_string(),
            ]
        );
        assert_eq!(plan.runtime_json, RUNTIME_WIRE);
        Ok(())
    }

    #[test]
    fn apple_foundation_carries_no_model() -> anyhow::Result<()> {
        let mut spec = spec();
        if let Some(fields) = spec.as_object_mut() {
            fields.insert("runtime".to_string(), json!({"type": "apple_foundation"}));
            fields.remove("model");
        }
        let plan = launch_plan_from_claim(&claim(spec))?;
        assert_eq!(
            plan.app_args,
            vec![
                HEADLESS_TOKEN.to_string(),
                r#"runtime={"type":"apple_foundation"}"#.to_string(),
                "benchmarks=decode_throughput_256".to_string(),
                SUBMIT_ON.to_string(),
                "claim=job-1".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn runtime_flags_travel_as_knobs_without_their_axes() -> anyhow::Result<()> {
        let spec = with(
            "runtime_flags",
            json!({
                "runtime_type": "mlx_ios_pipette",
                "model_type": "gguf_text",
                "benchmark_type": "decode_throughput",
                "batch": 512,
            }),
        );
        let plan = launch_plan_from_claim(&claim(spec))?;
        assert!(
            plan.app_args
                .contains(&r#"runtime-flags={"batch":512}"#.to_string()),
            "{:?}",
            plan.app_args
        );
        Ok(())
    }

    /// `knobs` under this claim's own axes, as the server sends a group.
    fn on_this_cell(knobs: Value) -> Value {
        let mut group = json!({
            "runtime_type": "mlx_ios_pipette",
            "model_type": "gguf_text",
            "benchmark_type": "decode_throughput",
        });
        if let (Some(group), Some(knobs)) = (group.as_object_mut(), knobs.as_object()) {
            group.extend(knobs.clone());
        }
        group
    }

    /// The knobs travel without their axes, even one no iOS cell carries: the app
    /// refuses that by name. An entry with only its axes sends nothing.
    #[rstest]
    #[case::readiness(
        on_this_cell(json!({"readiness": {"max_wait_secs": 1800, "skip_thermal": true}})),
        Some(r#"benchmark-flags={"readiness":{"max_wait_secs":1800,"skip_thermal":true}}"#),
    )]
    #[case::server_knob(
        on_this_cell(json!({"http_timeout_seconds": 600})),
        Some(r#"benchmark-flags={"http_timeout_seconds":600}"#),
    )]
    #[case::only_axes(on_this_cell(json!({})), None)]
    fn benchmark_flags_travel_as_knobs_without_their_axes(
        #[case] flags: Value,
        #[case] want: Option<&str>,
    ) -> anyhow::Result<()> {
        let plan = launch_plan_from_claim(&claim(with("benchmark_flags", flags)))?;
        let mut expected = vec![
            HEADLESS_TOKEN.to_string(),
            format!("runtime={RUNTIME_WIRE}"),
            format!("model={MODEL_WIRE}"),
            "benchmarks=decode_throughput_256".to_string(),
        ];
        expected.extend(want.map(str::to_string));
        expected.extend([SUBMIT_ON.to_string(), "claim=job-1".to_string()]);
        assert_eq!(plan.app_args, expected);
        Ok(())
    }

    #[rstest]
    #[case::missing_spec(Value::Null, "no `spec`")]
    #[case::not_an_object(json!([1, 2]), "not a JSON object")]
    fn an_unreadable_spec_is_refused(#[case] spec: Value, #[case] needle: &str) {
        let error = launch_plan_from_claim(&claim(spec))
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(needle), "{error}");
    }

    /// One key set to one value, and the claim is refused naming what was wrong.
    /// `model_flags` matters most: the iOS transport drops it silently.
    #[rstest]
    #[case::knob_with_no_argument("model_flags", json!({"temperature": 0.7}), "model_flags")]
    #[case::unknown_key("cooldown", json!(30), "cooldown")]
    #[case::non_string_benchmark("benchmark", json!(7), "no benchmark")]
    #[case::bare_runtime_tag("runtime", json!("mlx_ios_pipette"), "no runtime")]
    #[case::benchmark_flags_not_an_object(
        "benchmark_flags",
        json!([{"readiness": {"max_wait_secs": 1800}}]),
        "`benchmark_flags` is not a JSON object"
    )]
    #[case::runtime_flags_not_an_object(
        "runtime_flags",
        json!("threads=4"),
        "`runtime_flags` is not a JSON object"
    )]
    #[case::flags_for_another_runtime(
        "runtime_flags",
        json!({"runtime_type": "llamacpp_ios_pipette", "threads": 4}),
        "runs `mlx_ios_pipette`"
    )]
    #[case::flags_for_another_model(
        "benchmark_flags",
        json!({"model_type": "mlx", "readiness": {"max_wait_secs": 1800}}),
        "runs `gguf_text`"
    )]
    #[case::flags_for_another_benchmark(
        "benchmark_flags",
        json!({"benchmark_type": "prefill_throughput", "readiness": {"max_wait_secs": 1800}}),
        "runs `decode_throughput`"
    )]
    fn a_spec_with_one_bad_key_is_refused_naming_it(
        #[case] key: &str,
        #[case] value: Value,
        #[case] needle: &str,
    ) {
        let error = launch_plan_from_claim(&claim(with(key, value)))
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains(needle), "`{key}`: {error}");
    }

    #[rstest]
    #[case::no_runtime("runtime")]
    #[case::no_benchmark("benchmark")]
    #[case::no_model("model")]
    fn a_spec_missing_an_axis_is_refused(#[case] removed: &str) {
        let mut spec = spec();
        if let Some(fields) = spec.as_object_mut() {
            fields.remove(removed);
        }
        assert!(launch_plan_from_claim(&claim(spec)).is_err());
    }

    /// A padded value is a disagreement: trimming it would file the run under an
    /// id the envelope does not carry.
    #[rstest]
    #[case::another_benchmark("prefill_latency_512")]
    #[case::padded(" decode_throughput_256 ")]
    #[case::wrong_case("DECODE_THROUGHPUT_256")]
    fn a_spec_that_disagrees_with_the_envelope_is_refused(#[case] named: &str) {
        assert!(
            matches!(
                launch_plan_from_claim(&claim(with("benchmark", json!(named)))),
                Err(UnrunnableClaimError::BenchmarkMismatch { ref spec, ref envelope })
                    if spec == named && envelope == "decode_throughput_256"
            ),
            "`{named}` must be refused as a mismatch"
        );
    }

    #[test]
    fn an_empty_benchmark_is_missing_rather_than_mismatched() {
        assert!(matches!(
            launch_plan_from_claim(&claim(with("benchmark", json!("")))),
            Err(UnrunnableClaimError::MissingBenchmark)
        ));
    }

    #[test]
    fn the_benchmark_argument_is_the_agreed_id() -> anyhow::Result<()> {
        let plan = launch_plan_from_claim(&claim(spec()))?;
        assert!(
            plan.app_args
                .contains(&"benchmarks=decode_throughput_256".to_string()),
            "{:?}",
            plan.app_args
        );
        Ok(())
    }

    /// The console scrubs what the argv carries, so the gated model's token
    /// must be readable back out of the argv.
    #[test]
    fn the_console_finds_the_token_the_argv_carries() -> anyhow::Result<()> {
        let mut spec = spec();
        if let Some(model) = spec.get_mut("model").and_then(Value::as_object_mut) {
            model.insert("auth_token".to_string(), json!("hf_gated"));
        }
        let plan = launch_plan_from_claim(&claim(spec))?;
        assert_eq!(crate::secret::argv_secrets(&plan.app_args), ["hf_gated"]);
        Ok(())
    }

    /// Key order survives re-emission: the server's documents lead with `type`,
    /// so alphabetising would be visible on every real job.
    #[test]
    fn re_emitted_json_keeps_the_servers_key_order() -> anyhow::Result<()> {
        let source =
            r#"{"type":"mlx_ios_pipette","version":"0.21.0","build":"abc","alpha":{"z":1,"a":2}}"#;
        let parsed: Value = serde_json::from_str(source).context("a fixed document parses")?;
        let rendered = compact_json(&parsed).context("a parsed document re-serialises")?;
        assert_eq!(rendered, source);
        Ok(())
    }
}
