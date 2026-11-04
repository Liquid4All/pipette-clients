//! A vlm_eval cell's thinking mode is its model flags' `enable_thinking`, as on
//! a text eval, and goes through the model's chat template: the first server
//! starts with no reasoning flag, the runner asks it whether the template has a
//! thinking switch, and only a template with one is switched, on a server
//! restarted with the mode's flags. The restart after a crash is
//! `vlm_eval_reasoning_restart.rs`'s.

mod common;

use std::path::Path;

use common::{
    THINKING_SWITCH, chat_bodies, ignore_reps, no_corpus, no_readiness_gate, reasoning_flags,
    started_reasoning_flags, vlm_reasoning_request,
};
use rstest::rstest;
use serde_json::json;

use pipette_plan_types::{KnownVlmPromptProfile, RunRequest, RunResponse, VlmPromptProfile};

/// `--reasoning off` with its zero budget, as a probed switch gets it.
const OFF: &[&str] = &["--reasoning", "off", "--reasoning-budget", "0"];

fn run(root: &Path, req: &RunRequest) -> anyhow::Result<RunResponse> {
    pipette_llamacpp::run(
        req,
        &no_corpus(root),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
}

/// The recorded command is the one the cell ran on, and no sample carries a
/// thinking kwarg of its own.
#[rstest]
#[case::no_switch_undeclared(None, &[], &[])]
#[case::switch_off(Some(false), &[THINKING_SWITCH], OFF)]
#[case::switch_on(Some(true), &[THINKING_SWITCH], &["--reasoning", "on"])]
// Found by the probe conversation that opens with a system prompt.
#[case::switch_behind_a_system_prompt(
    Some(false),
    &["FAKE_LLAMA_THINKING_SWITCH=system"],
    OFF
)]
// The date turns between the first `on` render and the `off` one: the two `on`
// renders disagree, so the template is probed again rather than taken for one
// with a switch.
#[case::a_day_turning_mid_probe(None, &["FAKE_LLAMA_TEMPLATE_DAY_TURNS_AFTER=1"], &[])]
fn the_mode_is_applied_through_the_chat_template(
    #[case] enable_thinking: Option<bool>,
    #[case] envs: &[&str],
    #[case] flags: &[&str],
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = vlm_reasoning_request(root.path(), enable_thinking, &["one", "two"], envs)?;
    let outcome = run(root.path(), &req)?;

    assert_eq!(
        reasoning_flags(&outcome.command),
        flags,
        "{:?}",
        outcome.command
    );
    let mut want: Vec<&[&str]> = vec![&[]];
    if !flags.is_empty() {
        want.push(flags);
    }
    assert_eq!(started_reasoning_flags(root.path())?, want);
    let bodies = chat_bodies(root.path())?;
    assert_eq!(bodies.len(), 2, "{bodies:?}");
    for body in &bodies {
        assert_eq!(body.get("reasoning_format"), Some(&json!("none")), "{body}");
        assert_eq!(body.get("chat_template_kwargs"), None, "{body}");
    }
    Ok(())
}

#[rstest]
#[case::on_without_a_switch(
    Some(true),
    &[],
    &["fake-model.gguf", "no thinking switch", "enable_thinking = true", "internvl_r1"]
)]
#[case::undeclared_with_a_switch(
    None,
    &[THINKING_SWITCH],
    &["fake-model.gguf", "has a thinking switch", "enable_thinking"]
)]
#[case::off_without_a_switch(
    Some(false),
    &[],
    &["fake-model.gguf", "no reasoning mode to turn off", "leave enable_thinking unset"]
)]
#[case::renders_that_never_agree(
    None,
    &["FAKE_LLAMA_TEMPLATE_COUNTER=1"],
    &["rendered the same probe differently", "cannot be told"]
)]
fn the_cell_fails_before_any_sample(
    #[case] enable_thinking: Option<bool>,
    #[case] envs: &[&str],
    #[case] needles: &[&str],
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = vlm_reasoning_request(root.path(), enable_thinking, &["one"], envs)?;
    let Err(err) = run(root.path(), &req) else {
        anyhow::bail!("the cell must fail");
    };
    let msg = format!("{err:#}");
    for needle in needles {
        assert!(msg.contains(needle), "{needle:?} missing from: {msg}");
    }
    assert_eq!(
        started_reasoning_flags(root.path())?,
        vec![Vec::<String>::new()]
    );
    assert!(!root.path().join("calls").exists(), "a sample was sent");
    Ok(())
}

/// InternVL3.5 thinks under its R1 system prompt, not a template switch: a
/// cell on the `internvl_r1` profile that declares `enable_thinking = true`
/// runs on the first server with no reasoning flag, and its rows say it thought.
#[test]
fn the_internvl_r1_profile_thinks_without_a_switch() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut req = vlm_reasoning_request(root.path(), Some(true), &["one"], &[])?;
    with_profile(&mut req, KnownVlmPromptProfile::InternvlR1)?;
    let outcome = run(root.path(), &req)?;
    assert!(
        reasoning_flags(&outcome.command).is_empty(),
        "{:?}",
        outcome.command
    );
    assert_eq!(
        started_reasoning_flags(root.path())?,
        vec![Vec::<String>::new()]
    );
    Ok(())
}

#[rstest]
#[case::undeclared(None)]
#[case::off(Some(false))]
fn the_internvl_r1_profile_must_be_declared_thinking(
    #[case] enable_thinking: Option<bool>,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut req = vlm_reasoning_request(root.path(), enable_thinking, &["one"], &[])?;
    with_profile(&mut req, KnownVlmPromptProfile::InternvlR1)?;
    let Err(err) = run(root.path(), &req) else {
        anyhow::bail!("the cell must fail");
    };
    let msg = format!("{err:#}");
    assert!(msg.contains("internvl_r1"), "{msg}");
    assert!(msg.contains("enable_thinking = true"), "{msg}");
    assert!(!root.path().join("args").exists(), "a server was started");
    Ok(())
}

fn with_profile(req: &mut RunRequest, profile: KnownVlmPromptProfile) -> anyhow::Result<()> {
    let pipette_plan_types::BenchmarkDefinition::VlmEval(entry) = &mut req.benchmark else {
        anyhow::bail!("not a vlm_eval request");
    };
    entry.parameter_prompt_profile = Some(VlmPromptProfile::Known(profile));
    Ok(())
}

/// A cell over a parseable image, with a calibration helper that logs each
/// run under `root`, the cell's `raw`, its `envs`, and a llama.cpp config file
/// holding `config` under a forwarded `XDG_CONFIG_HOME`.
#[cfg(unix)]
fn calibrating_request(
    root: &Path,
    raw: &[&str],
    envs: &[&str],
    config: &str,
) -> anyhow::Result<RunRequest> {
    let config_dir = root.join("config");
    std::fs::create_dir_all(config_dir.join("llama.cpp"))?;
    std::fs::write(config_dir.join("llama.cpp").join("config.ini"), config)?;
    let xdg = format!("XDG_CONFIG_HOME={}", config_dir.display());
    let envs: Vec<&str> = envs.iter().copied().chain([xdg.as_str()]).collect();
    let mut req = vlm_reasoning_request(root, None, &["one"], &envs)?;
    let mut flags = req.runtime_flags_ref()?;
    flags.raw = raw.iter().map(|token| (*token).to_owned()).collect();
    req.runtime_flags = Some(flags.try_into()?);
    common::install_logged_fake_mtmd_cli(&root.join("runtime-install"), &root.join("helper"))?;
    Ok(req)
}

/// Refused before the calibration, which can load the model: a key in a config
/// file llama.cpp applies to every launch. The reserved raw flags and llama.cpp
/// option envs never get this far (see the test below).
#[cfg(unix)]
#[rstest]
#[case::config_file(&[], &[], "[*]\nreasoning = on\n", "config.ini sets `reasoning`")]
fn thinking_controls_outside_the_typed_field_are_refused(
    #[case] raw: &[&str],
    #[case] envs: &[&str],
    #[case] config: &str,
    #[case] needle: &str,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = calibrating_request(root.path(), raw, envs, config)?;
    let Err(err) = run(root.path(), &req) else {
        anyhow::bail!("{raw:?} {envs:?} {config:?} must be refused");
    };
    let msg = format!("{err:#}");
    assert!(msg.contains(needle), "{needle:?} missing from: {msg}");
    assert!(!root.path().join("helper").exists(), "the cell calibrated");
    assert!(!root.path().join("args").exists(), "a server was started");
    Ok(())
}

/// A vlm_eval's thinking mode, chat template and presets are its recipe's, so
/// its flags refuse them when they are read, in any spelling llama.cpp parses,
/// and any llama.cpp option variable in its envs: the cell never reaches the
/// client's own check.
#[cfg(unix)]
#[rstest]
#[case::raw_reasoning_flag(&["-rea", "on"], &[], "is reserved by the benchmark/tool")]
#[case::raw_template_kwargs(
    &["--chat_template_kwargs", r#"{"enable_thinking":true}"#],
    &[],
    "is reserved by the benchmark/tool"
)]
#[case::no_jinja(&["--no_jinja"], &[], "is reserved by the benchmark/tool")]
#[case::builtin_preset(&["--vision-gemma-4b-default"], &[], "is reserved by the benchmark/tool")]
#[case::hf_repo(&["-hf", "org/presets"], &[], "is reserved by the benchmark/tool")]
#[case::hf_repo_underscored(&["--hf_repo", "org/presets"], &[], "is reserved by the benchmark/tool")]
#[case::projector(&["-mm", "other.gguf"], &[], "is reserved by the benchmark/tool")]
#[case::reasoning_env(&[], &["llama_arg_reasoning=on"], "sets a llama.cpp option")]
#[case::jinja_env(&[], &["LLAMA_ARG_JINJA=false"], "sets a llama.cpp option")]
fn a_vlm_evals_thinking_controls_are_refused_when_its_flags_are_read(
    #[case] raw: &[&str],
    #[case] envs: &[&str],
    #[case] needle: &str,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let Err(err) = calibrating_request(root.path(), raw, envs, "") else {
        anyhow::bail!("{raw:?} {envs:?} must be refused when the flags are read");
    };
    let msg = format!("{err:#}");
    assert!(msg.contains(needle), "{msg}");
    Ok(())
}

/// The same cell with nothing to refuse calibrates, so the test above sees a
/// refusal's doing and not a cell that never calibrates.
#[cfg(unix)]
#[test]
fn a_cell_with_nothing_to_refuse_calibrates() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = calibrating_request(root.path(), &[], &[], "threads = 4\n")?;
    run(root.path(), &req)?;
    assert!(
        root.path().join("helper").exists(),
        "the cell did not calibrate"
    );
    Ok(())
}
