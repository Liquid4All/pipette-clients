//! `PIPETTE_READINESS_SKIP_THERMAL` has to parse through the same grammar the
//! runners use, and reading it is clap's job rather than ours — so this asserts
//! the wiring end to end rather than just re-testing the parser.
//!
//! Before the value parser was attached, clap's default `bool` parser accepted
//! only `true`/`false`: `=1` — the very value `runner::run` forwards to remote
//! cells — aborted argument parsing with "invalid value '1'", and because the
//! flag is `global = true` that killed every subcommand.
//!
//! This lives in `tests/` rather than beside `cli.rs` deliberately. It mutates
//! process-wide environment state, which is only sound while nothing else in
//! the process reads the environment concurrently; an integration test gets its
//! own binary, and every case runs in one test function so there is no second
//! thread to race with. That matches where the rest of the repo puts
//! env-mutating tests.

use clap::Parser;

use pipette_plan::cli::Cli;

/// Parse a minimal command line with `PIPETTE_READINESS_SKIP_THERMAL` set to
/// `value` (or unset when `None`) and report the resolved flag.
fn skip_thermal_with_env(value: Option<&str>) -> Result<bool, clap::Error> {
    match value {
        Some(raw) => std::env::set_var(pipette_readiness::SKIP_THERMAL_ENV, raw),
        None => std::env::remove_var(pipette_readiness::SKIP_THERMAL_ENV),
    }
    Cli::try_parse_from(["pipette-plan", "status", "--plan", "plan.toml"])
        .map(|cli| cli.readiness_skip_thermal)
}

#[test]
fn env_var_resolves_through_the_readiness_grammar() -> anyhow::Result<()> {
    // Unset is the safe default: nothing waived.
    assert!(!skip_thermal_with_env(None)?, "unset must enforce");

    let cases = [
        // The spelling the plan runner itself forwards. This one regressed.
        ("1", true),
        ("true", true),
        ("yes", true),
        ("on", true),
        // Exporting empty is how a shell clears a variable, so it must enforce
        // rather than fail to parse.
        ("", false),
        ("0", false),
        ("false", false),
        // Written as "no" by an operator who means "do not waive the gate".
        ("off", false),
        ("no", false),
        // Whitespace is trimmed, which clap's own FalseyValueParser would not do.
        ("  0  ", false),
    ];

    for (raw, want_skip) in cases {
        let got = skip_thermal_with_env(Some(raw))
            .map_err(|e| anyhow::anyhow!("{raw:?} failed to parse at all: {e}"))?;
        anyhow::ensure!(
            got == want_skip,
            "{raw:?} resolved to skip={got}, expected skip={want_skip}",
        );
        // The driver's answer must match what a runner inheriting the same
        // value would independently decide, or a fleet run splits in half.
        anyhow::ensure!(
            got == pipette_readiness::skip_thermal_from_str(raw),
            "{raw:?}: driver and runner disagree",
        );
    }

    std::env::remove_var(pipette_readiness::SKIP_THERMAL_ENV);
    Ok(())
}
