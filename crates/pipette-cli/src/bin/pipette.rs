fn main() -> anyhow::Result<()> {
    // Restore default SIGPIPE handling so piping into `head` etc. exits cleanly
    // instead of aborting on a broken pipe (e.g. `pipette … | head`).
    pipette_subprocess::reset_sigpipe();
    pipette_cli::commands::run()
}
