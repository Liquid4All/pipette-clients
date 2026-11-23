//! The `--config` file (`docs/pipette-cli/usage.md` §8a), read once per
//! invocation. Each section's rules are its own type's serde rules
//! ([`ExtraRuntimeConfigurationRef`], [`ReadinessConfiguration`], [`Hooks`]).

use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use pipette_plan_types::{
    ExtraRuntimeConfigurationRef, ExtraRuntimeConfigurations, ExtraRuntimeConfigurationsError,
};
use pipette_readiness::{ReadinessConfiguration, ReadinessPlatform};

/// Unknown top-level keys are refused, so a typo cannot silently apply to
/// nothing.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientConfigurationRef {
    #[serde(default)]
    runtimes: Vec<ExtraRuntimeConfigurationRef>,
    #[serde(default)]
    readiness: Option<ReadinessConfiguration>,
    #[serde(default)]
    hooks: Hooks,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClientConfiguration {
    /// `[[runtimes]]`
    pub runtimes: ExtraRuntimeConfigurations,
    /// `[readiness.<gate>]`
    pub readiness: Option<ReadinessConfiguration>,
    /// `[hooks]`
    pub hooks: Hooks,
}

/// `[hooks]`: scripts the client runs at fixed points of every cell, with the
/// cell's types in `PIPETTE_HOOK_*`, so a script picks the cells it acts on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hooks {
    /// After the downloads and before the warm-up, outside the attempt timeout.
    /// Again before each rep of a llama.cpp cell whose reps each start a
    /// process, after the rep's readiness gate.
    #[serde(default)]
    pub before_cell_script: Option<String>,
}

impl fmt::Display for ClientConfiguration {
    /// For the `run` log line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set: Vec<String> = [
            (self.runtimes != ExtraRuntimeConfigurations::default())
                .then(|| self.runtimes.to_string()),
            self.readiness.as_ref().map(ToString::to_string),
            self.hooks
                .before_cell_script
                .as_ref()
                .map(|_| "hooks.before_cell_script".to_string()),
        ]
        .into_iter()
        .flatten()
        .collect();
        if set.is_empty() {
            write!(f, "{}", self.runtimes)
        } else {
            write!(f, "{}", set.join("; "))
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClientConfigurationError {
    #[error("reading --config {}", path.display())]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("in --config {}", path.display())]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("in --config {}", path.display())]
    Resolve {
        path: PathBuf,
        #[source]
        source: ExtraRuntimeConfigurationsError,
    },
    #[error(
        "in --config {}: [readiness.{configured}] tunes the {configured} readiness gate, and \
         this client runs the {running} one",
        path.display()
    )]
    ReadinessForAnotherGate {
        path: PathBuf,
        configured: ReadinessPlatform,
        running: ReadinessPlatform,
    },
}

/// The empty configuration without a file. A malformed file fails here, before
/// a workspace is opened.
pub fn load(path: Option<&Path>) -> Result<ClientConfiguration, ClientConfigurationError> {
    let Some(path) = path else {
        return Ok(ClientConfiguration::default());
    };
    let text = std::fs::read_to_string(path).map_err(|source| ClientConfigurationError::Read {
        path: path.to_owned(),
        source,
    })?;
    let config = parse(path, &text)?;
    let running = pipette_readiness::platform();
    if let Some(readiness) = &config.readiness
        && readiness.platform() != running
    {
        return Err(ClientConfigurationError::ReadinessForAnotherGate {
            path: path.to_owned(),
            configured: readiness.platform(),
            running,
        });
    }
    Ok(config)
}

/// Every check that does not depend on the host, so a file written for another
/// gate can still be checked here.
fn parse(path: &Path, text: &str) -> Result<ClientConfiguration, ClientConfigurationError> {
    let file: ClientConfigurationRef =
        toml::from_str(text).map_err(|source| ClientConfigurationError::Parse {
            path: path.to_owned(),
            source,
        })?;
    let runtimes = ExtraRuntimeConfigurations::new(file.runtimes).map_err(|source| {
        ClientConfigurationError::Resolve {
            path: path.to_owned(),
            source,
        }
    })?;
    Ok(ClientConfiguration {
        runtimes,
        readiness: file.readiness,
        hooks: file.hooks,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn load_text(text: &str) -> anyhow::Result<ClientConfiguration> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("client.toml");
        std::fs::write(&path, text)?;
        Ok(load(Some(&path))?)
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn both_sections_load() -> anyhow::Result<()> {
        let config = load_text(
            "[[runtimes]]\nruntime_type = \"docker_vllm\"\nenvs = [\"A=1\"]\n\n\
             [readiness.windows]\nmax_cpu_busy_pct = 15\nceilings = { \"acpi:THZ0\" = 43 }\n",
        )?;
        assert_eq!(
            config.to_string(),
            "docker_vllm: envs=[A]; readiness.windows: ceilings=[acpi:THZ0<=43C] \
             max_cpu_busy_pct=15"
        );
        Ok(())
    }

    #[test]
    fn the_table_for_this_hosts_gate_loads() -> anyhow::Result<()> {
        let running = pipette_readiness::platform();
        let config = load_text(&format!("[readiness.{running}]\n"))?;
        assert!(
            config
                .to_string()
                .starts_with(&format!("readiness.{running}")),
            "{config}"
        );
        assert_eq!(config.readiness.map(|r| r.platform()), Some(running));
        Ok(())
    }

    /// A type change that invalidates a reference file fails here, whichever
    /// gate it is written for.
    #[test]
    fn shipped_example_configs_parse() -> anyhow::Result<()> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/client-configs");
        let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
            .collect();
        anyhow::ensure!(!paths.is_empty(), "no example configs in {dir}");
        paths.iter().try_for_each(|path| {
            let config = parse(path, &std::fs::read_to_string(path)?)?;
            anyhow::ensure!(
                config != ClientConfiguration::default(),
                "{} sets nothing",
                path.display()
            );
            anyhow::Ok(())
        })
    }

    /// Each gate's table parses on any host.
    #[rstest]
    #[case::android(
        "[readiness.android]\nhottest_below_c = 36\ncpu_busy_below_pct = 40\n",
        ReadinessPlatform::Android
    )]
    #[case::linux(
        "[readiness.linux]\nhottest_below_c = 80\ncpu_load_below_pct = 50\n",
        ReadinessPlatform::Linux
    )]
    #[case::pi5(
        "[readiness.pi5]\nhottest_below_c = 70\n",
        ReadinessPlatform::RaspberryPi5
    )]
    #[case::orin(
        "[readiness.orin]\ngpu_load_below_pct = 10\n",
        ReadinessPlatform::JetsonOrin
    )]
    #[case::macos(
        "[readiness.macos]\ndie_below_c = 60\ncpu_busy_below_cores = 2\n",
        ReadinessPlatform::MacOs
    )]
    #[case::windows(
        "[readiness.windows]\nmax_cpu_busy_pct = 15\nceilings = { gpu = 70 }\n",
        ReadinessPlatform::Windows
    )]
    fn each_gates_table_parses(
        #[case] text: &str,
        #[case] gate: ReadinessPlatform,
    ) -> anyhow::Result<()> {
        let config = parse(Path::new("client.toml"), text)?;
        assert_eq!(config.readiness.map(|r| r.platform()), Some(gate));
        Ok(())
    }

    #[test]
    fn another_gates_table_is_refused() -> anyhow::Result<()> {
        let other = match pipette_readiness::platform() {
            ReadinessPlatform::Windows => ReadinessPlatform::Linux,
            ReadinessPlatform::Android
            | ReadinessPlatform::Linux
            | ReadinessPlatform::RaspberryPi5
            | ReadinessPlatform::JetsonOrin
            | ReadinessPlatform::MacOs
            | ReadinessPlatform::Other => ReadinessPlatform::Windows,
        };
        let err = load_text(&format!("[readiness.{other}]\n"))
            .err()
            .ok_or_else(|| anyhow::anyhow!("[readiness.{other}] should be refused here"))?;
        assert!(
            format!("{err:#}").contains(&format!("tunes the {other} readiness gate")),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn runtimes_alone_load_anywhere() -> anyhow::Result<()> {
        let config = load_text("[[runtimes]]\nruntime_type = \"docker_vllm\"\nenvs = [\"A=1\"]\n")?;
        assert_eq!(config.to_string(), "docker_vllm: envs=[A]");
        assert_eq!(load(None)?, ClientConfiguration::default());
        Ok(())
    }

    #[rstest]
    #[case::unknown_table("[placement]\ngpu = 1\n", "placement")]
    #[case::bad_ceiling(
        "[readiness.windows]\nceilings = { \"acpi:THZ0\" = 0 }\n",
        "out of range"
    )]
    #[case::bad_sensor("[readiness.windows]\nceilings = { THZ0 = 43 }\n", "not a sensor")]
    #[case::bad_bound("[readiness.macos]\ndie_below_c = 0\n", "out of range")]
    #[case::another_gates_key(
        "[readiness.linux]\ndie_below_c = 60\n",
        "unknown field `die_below_c`"
    )]
    #[case::unknown_gate("[readiness.ios]\n", "unknown variant `ios`")]
    #[case::unknown_hook("[hooks]\nafter_cell_script = \"true\"\n", "after_cell_script")]
    #[case::hook_as_a_list("[[hooks]]\nbefore_cell_script = \"true\"\n", "invalid type")]
    #[case::duplicate_runtime(
        "[[runtimes]]\nruntime_type = \"docker_vllm\"\n[[runtimes]]\nruntime_type = \"docker_vllm\"\n",
        "appears twice"
    )]
    fn a_malformed_file_is_refused_naming_it(
        #[case] text: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let err = load_text(text)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should be refused: {text}"))?;
        let msg = format!("{err:#}");
        assert!(msg.contains("in --config"), "{msg}");
        assert!(msg.contains(needle), "{msg}");
        Ok(())
    }

    #[test]
    fn the_hook_loads_and_is_named_on_the_run_line() -> anyhow::Result<()> {
        let config = load_text("[hooks]\nbefore_cell_script = '''\necho one\necho two\n'''\n")?;
        assert_eq!(
            config.hooks.before_cell_script.as_deref(),
            Some("echo one\necho two\n")
        );
        assert_eq!(config.to_string(), "hooks.before_cell_script");
        Ok(())
    }

    #[test]
    fn a_missing_file_is_a_read_error() {
        let err = load(Some(Path::new("/nonexistent/client.toml")));
        assert!(
            matches!(err, Err(ClientConfigurationError::Read { .. })),
            "{err:?}"
        );
    }
}
