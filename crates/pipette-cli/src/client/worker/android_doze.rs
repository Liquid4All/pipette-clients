//! Android doze, turned off at worker start. `dumpsys deviceidle disable` is
//! runtime state, so a reboot turns doze back on; the mediator restarts the
//! worker after every reboot, which makes worker start the place to turn it off.
#![cfg_attr(
    not(target_os = "android"),
    expect(
        dead_code,
        reason = "host builds compile this only for the doze_flags tests"
    )
)]

use std::time::Duration;

use anyhow::Context;

/// Turns doze off and reads it back, failing if either flag is still on.
pub fn disable_doze() -> anyhow::Result<()> {
    dumpsys(&["deviceidle", "disable"])?;
    let dump = dumpsys(&["deviceidle"])?;
    let (light, deep) = doze_flags(&dump)
        .context("`dumpsys deviceidle` shows no mLightEnabled or no mDeepEnabled")?;
    tracing::info!(
        light_enabled = light,
        deep_enabled = deep,
        "doze after `dumpsys deviceidle disable`"
    );
    anyhow::ensure!(
        !light && !deep,
        "doze still on after `dumpsys deviceidle disable`: \
         mLightEnabled={light} mDeepEnabled={deep}"
    );
    Ok(())
}

fn dumpsys(args: &[&str]) -> anyhow::Result<String> {
    // The readiness gate's probe limit (pipette-readiness PROBE_TIMEOUT).
    const DUMPSYS_TIMEOUT: Duration = Duration::from_secs(30);
    let mut cmd = std::process::Command::new("/system/bin/dumpsys");
    cmd.args(args);
    let shown = args.join(" ");
    let output = pipette_subprocess::output_within(cmd, DUMPSYS_TIMEOUT)
        .with_context(|| format!("running `dumpsys {shown}`"))?;
    anyhow::ensure!(
        output.status.success(),
        "`dumpsys {shown}` exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `(mLightEnabled, mDeepEnabled)` from a `dumpsys deviceidle` dump, `None`
/// when either is missing.
fn doze_flags(dump: &str) -> Option<(bool, bool)> {
    let flag = |name: &str| {
        dump.split_whitespace()
            .find_map(|token| token.strip_prefix(name)?.strip_prefix('=')?.parse().ok())
    };
    Some((flag("mLightEnabled")?, flag("mDeepEnabled")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doze_flags_reads_the_s26_line() {
        let dump = "Settings:\n  mLightEnabled=true  mDeepEnabled=true\n  mForceIdle=false\n";
        assert_eq!(doze_flags(dump), Some((true, true)));
        let dump = "Settings:\n  mLightEnabled=false  mDeepEnabled=false\n  mForceIdle=false\n";
        assert_eq!(doze_flags(dump), Some((false, false)));
    }

    #[test]
    fn doze_flags_is_none_when_a_flag_is_missing() {
        assert_eq!(
            doze_flags("  mLightEnabled=false  mForceIdle=false\n"),
            None
        );
    }
}
