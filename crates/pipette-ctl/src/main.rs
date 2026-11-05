use clap::Parser;

fn main() -> anyhow::Result<()> {
    pipette_subprocess::reset_sigpipe();
    pipette_ctl::Cli::parse().execute()
}
