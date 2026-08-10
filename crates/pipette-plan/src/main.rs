use clap::Parser;

fn main() -> anyhow::Result<()> {
    pipette_subprocess::reset_sigpipe();
    pipette_plan::Cli::parse().execute()
}
