//! A vlm_eval server that crashes comes back with the reasoning flags the
//! thinking-switch probe decided, not the first server's. Its own file, as one
//! test, because it sets the env var that lets a cell with a failed row
//! publish.

mod common;

use anyhow::Context;
use common::{
    THINKING_SWITCH, ignore_reps, no_corpus, no_readiness_gate, started_reasoning_flags,
    vlm_reasoning_request,
};

#[test]
fn a_restart_after_a_crash_keeps_the_decided_flags() -> anyhow::Result<()> {
    const CRASH_MARKER: &str = "PLEASE-CRASH-NOW";
    // The crashing sample stays failed, which `ensure_complete` would refuse
    // to submit; that contract has its own tests.
    // SAFETY: one test per file, so no parallel test reads this var.
    unsafe { std::env::set_var("PIPETTE_ALLOW_INCOMPLETE_EVAL", "1") };
    let root = tempfile::tempdir()?;
    let crash = format!("FAKE_LLAMA_CRASH_PROMPT_CONTAINS={CRASH_MARKER}");
    let req = vlm_reasoning_request(
        root.path(),
        Some(false),
        &[CRASH_MARKER, "two"],
        &[THINKING_SWITCH, crash.as_str()],
    )?;
    pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )?;

    // The probe's server, the one the cell decided on, and at least the
    // replacement the second sample needed after the first crashed it.
    let started = started_reasoning_flags(root.path())?;
    let (probe, decided) = started.split_first().context("no server was started")?;
    assert!(probe.is_empty(), "{started:?}");
    assert!(
        decided.len() >= 2,
        "no restart after the crash: {started:?}"
    );
    assert!(
        decided
            .iter()
            .all(|flags| flags == &["--reasoning", "off", "--reasoning-budget", "0"]),
        "{started:?}"
    );
    Ok(())
}
