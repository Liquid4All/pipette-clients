//! clap reads `PIPETTE_READINESS_SKIP_THERMAL` through the runners' own
//! grammar; this asserts that wiring end to end, not just the parser.
//!
//! In `tests/`, with every case in one test function: it mutates the process
//! environment, which is sound only while no other thread reads it.

use clap::Parser;

use pipette_ctl::cli::Cli;

/// Parse a minimal command line with `PIPETTE_READINESS_SKIP_THERMAL` set to
/// `value` (or unset when `None`) and report the resolved flag.
fn skip_thermal_with_env(value: Option<&str>) -> Result<bool, clap::Error> {
    match value {
        // SAFETY: the only test in this binary, so no other thread reads the environment.
        Some(raw) => unsafe { std::env::set_var(pipette_readiness::SKIP_THERMAL_ENV, raw) },
        // SAFETY: as above.
        None => unsafe { std::env::remove_var(pipette_readiness::SKIP_THERMAL_ENV) },
    }
    Cli::try_parse_from(["pipette-ctl", "status", "--plan", "plan.toml"])
        .map(|cli| cli.readiness.readiness_skip_thermal)
}

#[test]
fn env_var_resolves_through_the_readiness_grammar() -> anyhow::Result<()> {
    // Unset is the safe default: nothing waived.
    assert!(!skip_thermal_with_env(None)?, "unset must enforce");

    let cases = [
        // The spelling the plan runner itself forwards.
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

    // SAFETY: as above.
    unsafe { std::env::remove_var(pipette_readiness::SKIP_THERMAL_ENV) };
    Ok(())
}
