//! The contract: one `fixtures/<section>/<name>.json` example per file,
//! read by this test and by each client's mirror. Each file is an rstest case,
//! checked by its directory's name function; `pipette_contract` holds the checks.
//! The rules:
//!
//! - an example re-encodes to itself;
//! - its name spells the variants it holds, joined by dots in the order its section's
//!   name function records them (`gguf_text.huggingface`), optionally followed by a
//!   `.<note>` telling apart two examples of the same variants;
//! - every string or key that, renamed to an unknown name, fails as an unknown variant
//!   is in its name, so a new enum field fails once an example spells it;
//! - every variant of every enum an example holds has an example, with variant lists
//!   taken from serde.
//!
//! `fixtures/variants.json` records the examples holding each variant, for
//! the clients. Beside it, what the crate derives from the examples, which a client
//! derives too and must match:
//!
//! - `params.json`: each `benchmark/` example's `BenchmarkParams`;
//! - `sampling.json`: what each eval example's samples draw under;
//! - `defaults.json`: every field each example may leave out, with what it reads as then.
//!   A field not listed is required;
//! - `runtime-build-flags.json`: the `runtime_build_flag` of each `runtime/` example that
//!   has one.
//!
//! `CONTRACT_BLESS=1` rewrites all five. A maintainer writes `bounds.json` by hand.
//! It lists values that the crate must accept or refuse at each field boundary.
//!
//! Reads `pipette-plan-types` through its public API only, as a git dependency on
//! pipette-mgmt, which owns the crate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use rstest::rstest;
use serde_json::{Value, json};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkEvalCompletion, BenchmarkFlagRef, BenchmarkFlags,
    BenchmarkParams, ClientRunSpec, DeviceInfo, Model, ModelFlagRef, ModelFlags, Runtime,
    RuntimeFlagRef, RuntimeFlags, runtime_build_flag,
};

use pipette_contract::{Bounds, Contract, Key, Name, ReencodeFn, ensure_current, reencode};

#[rstest]
fn example(#[files("fixtures/*/*.json")] path: PathBuf) -> anyhow::Result<()> {
    check(&mut Contract::default(), &path)
}

#[test]
fn every_variant_of_every_enum_has_an_example() -> anyhow::Result<()> {
    Contract::read(check)?.ensure_every_variant_held()
}

#[test]
fn variants_json_records_the_examples_holding_each_variant() -> anyhow::Result<()> {
    Contract::read(check)?.ensure_variants_json_current()
}

#[test]
fn params_json_records_each_benchmarks_params() -> anyhow::Result<()> {
    let mut params = BTreeMap::new();
    for (label, definition) in benchmarks()? {
        params.insert(
            label,
            serde_json::to_value(BenchmarkParams::from(&definition))?,
        );
    }
    ensure_current("params.json", &params)
}

/// Sample ids every eval example is resolved for, beside its own samples': a bare id,
/// repeats up to the last a `u32` counts, and suffixes that are no repeat count.
const SAMPLE_IDS: &[&str] = &[
    "row",
    "row#0",
    "row#1",
    "row#+2",
    "row#4294967295",
    "row#4294967296",
    "row#-5",
    "row#draft",
    "row#",
    "a#b#3",
];

/// What a sample draws under, as `pipette_ops::eval_cell::Sampling::apply` writes it into
/// a request (its tests check it against this file): the temperature always, and a cut or
/// a seed only when the entry pins one.
#[test]
fn sampling_json_records_what_each_eval_sample_draws_under() -> anyhow::Result<()> {
    let mut sampling = BTreeMap::new();
    for (label, definition) in benchmarks()? {
        let BenchmarkDefinition::Eval(eval) = definition else {
            continue;
        };
        let own = eval.samples.iter().flatten();
        let own = own.filter_map(|sample| sample.get("id")?.as_str());
        let mut samples = BTreeMap::new();
        for id in SAMPLE_IDS.iter().copied().chain(own) {
            let mut drawn = BTreeMap::new();
            drawn.insert("temperature", json!(eval.sampling_temperature()?.as_f64()));
            if let Some(top_p) = eval.parameter_top_p {
                drawn.insert("top_p", json!(top_p.as_f64()));
            }
            if let Some(top_k) = eval.parameter_top_k {
                drawn.insert("top_k", json!(top_k.into_inner()));
            }
            if let Some(min_p) = eval.parameter_min_p {
                drawn.insert("min_p", json!(min_p.as_f64()));
            }
            if let Some(seed) = eval.sample_seed(id) {
                drawn.insert("seed", json!(seed));
            }
            samples.insert(id.to_owned(), drawn);
        }
        sampling.insert(label, samples);
    }
    ensure_current("sampling.json", &sampling)
}

#[test]
fn defaults_json_records_the_fields_each_example_may_omit() -> anyhow::Result<()> {
    let mut defaults = BTreeMap::new();
    for (label, path) in pipette_contract::examples()? {
        let reencode = reencoder(pipette_contract::section(&path)?)?;
        let omittable = pipette_contract::omittable(&authored(&path)?, reencode)
            .with_context(|| label.clone())?;
        if !omittable.is_empty() {
            defaults.insert(label, omittable);
        }
    }
    ensure_current("defaults.json", &defaults)
}

/// The build flag the catalog reads for each iOS app runtime example. The iOS app's
/// `runtimeBuildFlag` must give the same flag (its `ContractTests` read this file).
#[test]
fn runtime_build_flags_json_records_each_runtimes_build_flag() -> anyhow::Result<()> {
    let mut flags = BTreeMap::new();
    for (label, path) in pipette_contract::examples()? {
        if pipette_contract::section(&path)? != "runtime" {
            continue;
        }
        let runtime: Runtime = serde_json::from_value(authored(&path)?)?;
        if let Some(flag) = runtime_build_flag(&runtime)? {
            flags.insert(label, flag);
        }
    }
    ensure_current("runtime-build-flags.json", &flags)
}

#[test]
fn bounds_json_holds_what_the_crate_accepts_and_refuses() -> anyhow::Result<()> {
    let examples = pipette_contract::examples()?;
    let bounds: BTreeMap<String, BTreeMap<String, Bounds>> =
        pipette_contract::read_fixture("bounds.json")?;
    for (label, fields) in bounds {
        let path = examples
            .get(&label)
            .with_context(|| format!("bounds.json: {label} is no example"))?;
        let authored = authored(path)?;
        let reencode = reencoder(pipette_contract::section(path)?)?;
        for (pointer, bounds) in fields {
            pipette_contract::check_bounds(&authored, &pointer, &bounds, reencode)
                .with_context(|| format!("bounds.json: {label}"))?;
        }
    }
    Ok(())
}

fn authored(path: &Path) -> anyhow::Result<Value> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).with_context(|| path.display().to_string())
}

fn benchmarks() -> anyhow::Result<BTreeMap<String, BenchmarkDefinition>> {
    let mut benchmarks = BTreeMap::new();
    for (label, path) in pipette_contract::examples()? {
        if pipette_contract::section(&path)? == "benchmark" {
            let definition = serde_json::from_value(authored(&path)?)?;
            benchmarks.insert(label, definition);
        }
    }
    Ok(benchmarks)
}

fn reencoder(section: &str) -> anyhow::Result<ReencodeFn> {
    Ok(match section {
        "model" => reencode::<Model>,
        "runtime" => reencode::<Runtime>,
        "model_flags" => reencode::<ModelFlags>,
        "runtime_flags" => reencode::<RuntimeFlags>,
        "benchmark_flags" => reencode::<BenchmarkFlags>,
        "benchmark" => reencode::<BenchmarkDefinition>,
        "benchmark_params" => reencode::<BenchmarkParams>,
        "client_run_spec" => reencode::<ClientRunSpec>,
        "device" => reencode::<DeviceInfo>,
        "eval_completion" => reencode::<BenchmarkEvalCompletion>,
        other => anyhow::bail!("{other}/ is no section"),
    })
}

/// Checks an example with the name function of the directory it is in.
fn check(contract: &mut Contract, path: &Path) -> anyhow::Result<()> {
    match pipette_contract::section(path)? {
        "model" => contract.file(path, model_name),
        "runtime" => contract.file(path, runtime_name),
        "model_flags" => contract.file(path, model_flags_name),
        "runtime_flags" => contract.file(path, runtime_flags_name),
        "benchmark_flags" => contract.file(path, benchmark_flags_name),
        "benchmark" => contract.file(path, benchmark_name),
        "benchmark_params" => contract.file(path, benchmark_params_name),
        "client_run_spec" => contract.file(path, client_run_spec_name),
        "device" => contract.file(path, device_name),
        "eval_completion" => contract.file(path, eval_completion_name),
        other => anyhow::bail!("{other}/ is no section"),
    }
}

/// A `Model` example's name. Exhaustive, so a new variant fails to compile here
/// until it says which enums it holds.
fn model_name(model: &Model, name: &mut Name) -> anyhow::Result<()> {
    name.holds(model)?;
    match model {
        Model::GgufText(m) => name.holds(&m.source),
        Model::GgufVision(m) => name.holds(&m.source),
        Model::GgufVisionSpeculative(m) => name.holds(&m.source),
        Model::Mlx(m) => name.holds(&m.source),
        Model::AppleCoreAi(m) => name.holds(&m.source),
        Model::Torch(m) => name.holds(&m.source),
        Model::Openvino(m) => name.holds(&m.source),
        Model::OnnxGenai(m) => name.holds(&m.source),
        Model::MlxSpeculative(pair) | Model::TorchSpeculative(pair) => {
            name.holds(&pair.target)?;
            name.holds(&pair.draft)
        }
        Model::AppleFoundationText => Ok(()),
    }
}

/// A `Runtime` example's name, exhaustive as `model_name` is.
fn runtime_name(runtime: &Runtime, name: &mut Name) -> anyhow::Result<()> {
    name.holds(runtime)?;
    match runtime {
        Runtime::LlamacppCliStockTools(rt) => name.holds(&rt.source),
        Runtime::LlamacppApkPipette(rt) => name.holds(&rt.flavor),
        Runtime::LlamacppIosPipette(rt) => name.holds(&rt.flavor),
        Runtime::MlxMacosPipette(rt) => {
            name.holds(&rt.flavor)?;
            name.holds(&rt.source)
        }
        Runtime::MlxIosPipette(rt) => name.holds(&rt.flavor),
        Runtime::DockerVllm(rt) => name.holds(&rt.flavor),
        Runtime::DockerSglang(rt) => name.holds(&rt.flavor),
        Runtime::UvVllm(rt) => name.holds(&rt.source),
        Runtime::UvSglang(rt) => name.holds(&rt.source),
        Runtime::UvOpenvino(rt) => name.holds(&rt.source),
        Runtime::UvOnnxGenai(rt) => name.holds(&rt.source),
        Runtime::Ovms(rt) => name.holds(&rt.source),
        Runtime::TrtEdgellm(rt) => name.holds(&rt.source),
        Runtime::OnnxGenaiIosPipette(rt) => name.holds(&rt.flavor),
        Runtime::AppleFoundation(_) | Runtime::AppleCoreAiMacosPipette(_) => Ok(()),
    }
}

/// A `ModelFlags` example's name: `benchmark_type` and `model_type` select its variant,
/// then the reasoning effort when it sets one.
fn model_flags_name(flags: &ModelFlags, name: &mut Name) -> anyhow::Result<()> {
    let flat = ModelFlagRef::from(flags.clone());
    name.keyed(
        flags,
        &[
            Key::new("benchmark_type", &flat.benchmark_type)?,
            Key::new("model_type", &flat.model_type)?,
        ],
    )?;
    if let Some(reasoning_effort) = &flat.reasoning_effort {
        name.holds(reasoning_effort)?;
    }
    Ok(())
}

/// A `RuntimeFlags` example's name: its axes in the order the variant names them, then
/// any set knob that is a closed enum.
fn runtime_flags_name(flags: &RuntimeFlags, name: &mut Name) -> anyhow::Result<()> {
    let flat = RuntimeFlagRef::from(flags.clone());
    name.keyed(
        flags,
        &[
            Key::new("benchmark_type", &flat.benchmark_type)?,
            Key::new("runtime_type", &flat.runtime_type)?,
            Key::new("model_type", &flat.model_type)?,
        ],
    )?;
    if let Some(load_mode) = &flat.load_mode {
        name.holds(load_mode)?;
    }
    if let Some(flash_attention) = &flat.flash_attention {
        name.holds(flash_attention)?;
    }
    if let Some(dtype) = &flat.dtype {
        name.holds(dtype)?;
    }
    if let Some(generate_hint) = &flat.generate_hint {
        name.holds(generate_hint)?;
    }
    if let Some(ipc) = &flat.ipc {
        name.holds(ipc)?;
    }
    if let Some(precision) = &flat.precision {
        name.holds(precision)?;
    }
    Ok(())
}

/// A `BenchmarkFlags` example's name: its three axes, as for `RuntimeFlags`.
fn benchmark_flags_name(flags: &BenchmarkFlags, name: &mut Name) -> anyhow::Result<()> {
    let flat = BenchmarkFlagRef::from(flags.clone());
    name.keyed(
        flags,
        &[
            Key::new("benchmark_type", &flat.benchmark_type)?,
            Key::new("runtime_type", &flat.runtime_type)?,
            Key::new("model_type", &flat.model_type)?,
        ],
    )
}

/// A `BenchmarkDefinition` example's name: its kind, then the corpus a
/// `vl_spec_decode` or catalog `vlm_eval` or `asr_eval` pins, or the role of each
/// message a local `vlm_eval`'s samples carry, in order. Exhaustive, as
/// `model_name` is.
fn benchmark_name(definition: &BenchmarkDefinition, name: &mut Name) -> anyhow::Result<()> {
    name.holds(definition)?;
    match definition {
        BenchmarkDefinition::VlSpecDecode(b) => {
            name.holds(&b.dataset.format)?;
            name.holds(&b.dataset.adapter)?;
            name.holds(&b.dataset.upstream)
        }
        BenchmarkDefinition::VlmEval(b) => {
            if let Some(dataset) = &b.dataset {
                name.holds(&dataset.format)?;
                name.holds(&dataset.adapter)?;
                name.holds(&dataset.upstream)?;
            }
            b.samples
                .iter()
                .flatten()
                .flat_map(|sample| &sample.messages)
                .try_for_each(|message| name.holds(&message.role))
        }
        BenchmarkDefinition::AsrEval(b) => {
            if let Some(dataset) = &b.dataset {
                name.holds(&dataset.format)?;
                name.holds(&dataset.adapter)?;
                name.holds(&dataset.upstream)?;
            }
            Ok(())
        }
        BenchmarkDefinition::PrefillThroughput(_)
        | BenchmarkDefinition::DecodeThroughput(_)
        | BenchmarkDefinition::EndToEndLatency(_)
        | BenchmarkDefinition::MaxMemoryUsage(_)
        | BenchmarkDefinition::Eval(_)
        | BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => Ok(()),
    }
}

/// A `BenchmarkParams` example's name: its kind.
fn benchmark_params_name(params: &BenchmarkParams, name: &mut Name) -> anyhow::Result<()> {
    name.holds(params)
}

/// A `ClientRunSpec` example's name: its model's, its runtime's, then each flag
/// group's it carries.
fn client_run_spec_name(spec: &ClientRunSpec, name: &mut Name) -> anyhow::Result<()> {
    model_name(&spec.model, name)?;
    runtime_name(&spec.runtime, name)?;
    if let Some(flags) = &spec.runtime_flags {
        runtime_flags_name(flags, name)?;
    }
    if let Some(flags) = &spec.model_flags {
        model_flags_name(flags, name)?;
    }
    if let Some(flags) = &spec.benchmark_flags {
        benchmark_flags_name(flags, name)?;
    }
    Ok(())
}

/// A `DeviceInfo` example's name: its form factor.
fn device_name(device: &DeviceInfo, name: &mut Name) -> anyhow::Result<()> {
    name.holds(&device.device_form_factor)
}

/// A `BenchmarkEvalCompletion` example's name: how the sample stopped.
fn eval_completion_name(
    completion: &BenchmarkEvalCompletion,
    name: &mut Name,
) -> anyhow::Result<()> {
    name.holds(&completion.stop_reason)
}
