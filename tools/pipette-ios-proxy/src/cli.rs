//! The argument surface (DESIGN section 7). `run` is the default subcommand.

use std::path::{Path, PathBuf};

use tracing::level_filters::LevelFilter;

use crate::releases::BuildVersion;

/// The release version CI bakes in (`PIPETTE_IOS_PROXY_VERSION`), else the
/// crate's own for a local build.
pub(crate) const VERSION: &str = match option_env!("PIPETTE_IOS_PROXY_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

const ABOUT: &str = "Let a Mac stand in for a tethered iPhone on the pipette management server";

#[derive(Debug, clap::Parser)]
#[command(name = "pipette-ios-proxy", version = VERSION, about = ABOUT)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) global: GlobalArgs,
    /// Absent means [`Command::Run`].
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GlobalArgs {
    /// Config file path. Falls back to $PIPETTE_IOS_PROXY_CONFIG, then
    /// ./pipette-ios-proxy.toml.
    // Not clap `env`: an exported-but-empty variable would fail the `PathBuf`
    // parser at startup. `crate::config::resolve_config_path` treats empty as unset.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// State directory. Overrides the WHOLE path, not just its base.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) state_dir: Option<PathBuf>,

    /// Act for only the configured `[[devices]]` entry whose `udid` is ID.
    #[arg(long, global = true, value_name = "ID")]
    pub(crate) udid: Option<String>,

    /// Log level, overriding `[log] level`: error, warn, info, debug or trace.
    #[arg(long, global = true, value_name = "LEVEL")]
    pub(crate) log_level: Option<LevelFilter>,

    /// Shorthand for --log-level debug; the explicit flag wins.
    #[arg(short = 'v', long, global = true)]
    pub(crate) verbose: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, clap::Subcommand)]
pub(crate) enum Command {
    /// Run the claim loop for every configured phone. The default.
    #[default]
    Run,
    /// Refresh every configured phone's server-side profile and capabilities,
    /// then exit.
    Register,
    /// Print the server-side status and whether the phone is reachable.
    Status,
    /// Read every build in the pool (installing each unread one) and record which
    /// runtimes it provides. Touches no server.
    Scan,
    /// Uninstall the configured bundle id from every phone. Touches no server.
    Uninstall,
    /// Print the DeviceProfileFields the proxy would report. Touches no server.
    Probe,
    /// Validate the config, every device's key, the build pool or this Mac's
    /// signing setup, and the `[client_releases]` secret, and list the phones
    /// another process is driving. Touches no phone and no server.
    Check,
    /// List, pull or prune the app builds published to the `[client_releases]`
    /// bucket. Touches no phone and no management server.
    Builds {
        #[command(subcommand)]
        action: BuildsCommand,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, clap::Subcommand)]
pub(crate) enum BuildsCommand {
    /// Print one line per published build: version, tag, size, upload time,
    /// short sha256, and whether it is pulled.
    List,
    /// Download one build into the state directory, check it against its
    /// published sha256, and unpack it. Prints the directory.
    Pull {
        /// The build's version, as `builds list` prints it.
        version: BuildVersion,
    },
    /// Remove every pulled and signed build but the bucket's newest, the
    /// `[client_releases]` default and what a phone last installed, and any
    /// leftover scratch directory. Refuses while another process holds a phone,
    /// pull or signing lock. Prints each directory removed.
    Prune {
        /// Print what would be removed, taking no lock and removing nothing.
        #[arg(long)]
        dry_run: bool,
    },
}

impl Cli {
    pub(crate) fn command(&self) -> Command {
        self.command.clone().unwrap_or_default()
    }

    /// The level `--log-level` / `-v` ask for, or `None` to leave `[log] level` in charge.
    pub(crate) fn requested_log_level(&self) -> Option<LevelFilter> {
        match (self.global.log_level, self.global.verbose) {
            (Some(level), _) => Some(level),
            (None, true) => Some(LevelFilter::DEBUG),
            (None, false) => None,
        }
    }

    /// `--config` as typed; the environment fallback lives in
    /// `crate::config::resolve_config_path`.
    pub(crate) fn config_path(&self) -> Option<&Path> {
        self.global.config.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use clap::error::ErrorKind;
    use clap::{CommandFactory, Parser};
    use rstest::rstest;

    use super::*;
    use crate::config::CONFIG_PATH_ENV;

    #[test]
    fn the_derived_command_is_well_formed() {
        <Cli as CommandFactory>::command().debug_assert();
    }

    #[test]
    fn no_subcommand_means_run() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette-ios-proxy"])?;
        assert_eq!(cli.command(), Command::Run);
        Ok(())
    }

    #[rstest]
    #[case("run", Command::Run)]
    #[case("register", Command::Register)]
    #[case("status", Command::Status)]
    #[case("scan", Command::Scan)]
    #[case("uninstall", Command::Uninstall)]
    #[case("probe", Command::Probe)]
    #[case("check", Command::Check)]
    fn every_verb_parses(#[case] word: &str, #[case] expected: Command) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette-ios-proxy", word])?;
        assert_eq!(cli.command(), expected, "verb `{word}`");
        Ok(())
    }

    #[test]
    fn the_builds_verbs_parse() -> anyhow::Result<()> {
        let list = Cli::try_parse_from(["pipette-ios-proxy", "builds", "list"])?;
        assert_eq!(
            list.command(),
            Command::Builds {
                action: BuildsCommand::List
            }
        );

        let version = "2026.10.02-5c435ba-internal";
        let pull = Cli::try_parse_from(["pipette-ios-proxy", "builds", "pull", version])?;
        assert_eq!(
            pull.command(),
            Command::Builds {
                action: BuildsCommand::Pull {
                    version: version.parse()?
                }
            }
        );
        Ok(())
    }

    #[rstest]
    #[case::prune(&["prune"], false)]
    #[case::dry_run(&["prune", "--dry-run"], true)]
    fn builds_prune_parses(#[case] args: &[&str], #[case] dry_run: bool) -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette-ios-proxy", "builds"].iter().chain(args))?;
        assert_eq!(
            cli.command(),
            Command::Builds {
                action: BuildsCommand::Prune { dry_run }
            }
        );
        Ok(())
    }

    /// A version is a path segment in the store, so one that is not is a usage
    /// error, like a missing action or version.
    #[rstest]
    #[case::no_action(&["builds"], ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand)]
    #[case::no_version(&["builds", "pull"], ErrorKind::MissingRequiredArgument)]
    #[case::path_for_a_version(&["builds", "pull", "../escape"], ErrorKind::ValueValidation)]
    fn a_bad_builds_verb_is_a_usage_error(
        #[case] args: &[&str],
        #[case] kind: ErrorKind,
    ) -> anyhow::Result<()> {
        let error =
            Cli::try_parse_from(std::iter::once("pipette-ios-proxy").chain(args.iter().copied()))
                .err()
                .context("a usage error")?;
        assert_eq!(error.kind(), kind, "{error}");
        Ok(())
    }

    #[test]
    fn verbose_asks_for_debug_and_the_explicit_flag_wins() -> anyhow::Result<()> {
        let verbose = Cli::try_parse_from(["pipette-ios-proxy", "-v"])?;
        assert_eq!(verbose.requested_log_level(), Some(LevelFilter::DEBUG));

        let both = Cli::try_parse_from(["pipette-ios-proxy", "-v", "--log-level", "warn"])?;
        assert_eq!(both.requested_log_level(), Some(LevelFilter::WARN));

        let neither = Cli::try_parse_from(["pipette-ios-proxy"])?;
        assert_eq!(neither.requested_log_level(), None);
        Ok(())
    }

    #[test]
    fn a_bad_log_level_is_refused() {
        let parsed = Cli::try_parse_from(["pipette-ios-proxy", "--log-level", "chatty"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn the_config_flag_is_not_bound_to_the_environment_variable() -> anyhow::Result<()> {
        let command = <Cli as CommandFactory>::command();
        let config = command
            .get_arguments()
            .find(|arg| arg.get_id() == "config")
            .context("the derived command has a --config argument")?;
        assert!(
            config.get_env().is_none(),
            "clap would validate $PIPETTE_IOS_PROXY_CONFIG as a --config value, and its PathBuf \
             parser rejects an empty one: {:?}",
            config.get_env()
        );
        assert!(
            config
                .get_help()
                .map(|help| help.to_string())
                .unwrap_or_default()
                .contains(CONFIG_PATH_ENV),
            "--help must still name the variable"
        );
        Ok(())
    }

    #[test]
    fn the_config_flag_is_the_path_it_was_given() -> anyhow::Result<()> {
        let cli = Cli::try_parse_from(["pipette-ios-proxy", "--config", "/tmp/proxy.toml"])?;
        assert_eq!(cli.config_path(), Some(Path::new("/tmp/proxy.toml")));
        assert_eq!(
            Cli::try_parse_from(["pipette-ios-proxy"])?.config_path(),
            None
        );
        assert!(Cli::try_parse_from(["pipette-ios-proxy", "--config", ""]).is_err());
        Ok(())
    }
}
