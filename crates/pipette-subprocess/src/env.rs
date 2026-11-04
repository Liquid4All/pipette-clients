//! Landing a cell's env forwards, or the client's configuration, on a child.

use std::process::Command;

use pipette_plan_types::{EngineExtraRuntimeConfiguration, EnvForward};

use crate::docker_process::DockerRun;

/// Set each forward on a command that inherits its environment: a literal
/// value as given, a bare name as the launching process's own value when it
/// has one. A bare name with nothing to inherit sets nothing: an empty value is
/// still *set* to `getenv`, which several `GGML_*` knobs branch on. Later
/// entries win, as with `Command::env`.
pub fn apply_envs(cmd: &mut Command, envs: &[EnvForward]) {
    envs.iter().for_each(|entry| match entry.value() {
        Some(value) => {
            cmd.env(entry.name(), value);
        }
        None => {
            if let Some(inherited) = std::env::var_os(entry.name()) {
                cmd.env(entry.name(), inherited);
            }
        }
    });
}

/// Apply the client configuration a run was given, if it has one: everything
/// in it that is environment, landed the way [`apply_envs`] lands a cell's.
///
/// Generic over the engine's configuration type, so a spawn site cannot be
/// handed another engine's. Engines pass the configuration whole, so what it
/// holds is answered only in `pipette_plan_types`.
pub fn apply_client_config<C: EngineExtraRuntimeConfiguration>(
    cmd: &mut Command,
    config: Option<&C>,
) {
    if let Some(config) = config {
        apply_envs(cmd, config.envs());
    }
}

/// [`apply_client_config`] for a container: the configuration goes before the
/// run's forwards, whenever it is called, so theirs win.
pub fn forward_client_config<C: EngineExtraRuntimeConfiguration>(
    run: &mut DockerRun,
    config: Option<&C>,
) {
    if let Some(config) = config {
        run.envs.splice(0..0, config.envs().iter().cloned());
    }
}

#[cfg(test)]
pub(crate) fn forwards(entries: &[&str]) -> anyhow::Result<Vec<EnvForward>> {
    Ok(entries
        .iter()
        .map(|entry| EnvForward::try_new((*entry).to_owned()))
        .collect::<Result<_, _>>()?)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use rstest::rstest;

    use pipette_plan_types::TorchOaiExtraRuntimeConfiguration;

    use super::*;

    /// The variables `cmd` would launch with. `None` for a value means the
    /// entry was *removed* from the child's environment, which `Command`
    /// models distinctly from setting it empty.
    fn env_of(cmd: &Command, name: &str) -> Option<Option<String>> {
        cmd.get_envs()
            .find(|(k, _)| *k == OsStr::new(name))
            .map(|(_, v)| v.and_then(|v| v.to_str().map(str::to_owned)))
    }

    /// A name this process will not have set, for the inherit-absent case.
    const ABSENT: &str = "PIPETTE_APPLY_ENVS_TEST_ABSENT";

    #[rstest]
    #[case::sets_a_value(&["GGML_HEXAGON_OPPOLL=1"], "GGML_HEXAGON_OPPOLL", Some(Some("1")))]
    #[case::sets_an_empty_value(&["GGML_HEXAGON_NHVX="], "GGML_HEXAGON_NHVX", Some(Some("")))]
    #[case::keeps_inner_equals(&["OPTS=a=b"], "OPTS", Some(Some("a=b")))]
    #[case::inherit_absent_is_dropped(&[ABSENT], ABSENT, None)]
    #[case::last_wins(&["X=2", "X=4"], "X", Some(Some("4")))]
    fn forwards_reach_the_command(
        #[case] entries: &[&str],
        #[case] name: &str,
        #[case] expected: Option<Option<&str>>,
    ) -> anyhow::Result<()> {
        let mut cmd = Command::new("true");
        apply_envs(&mut cmd, &forwards(entries)?);
        assert_eq!(
            env_of(&cmd, name),
            expected.map(|v| v.map(str::to_owned)),
            "{entries:?}"
        );
        Ok(())
    }

    #[test]
    fn a_client_configuration_goes_before_a_containers_forwards() -> anyhow::Result<()> {
        let config =
            TorchOaiExtraRuntimeConfiguration::from(forwards(&["CUDA_VISIBLE_DEVICES=1"])?);
        let mut run = DockerRun {
            name: "pipette-test".to_owned(),
            image: "vllm/vllm-openai:v1".to_owned(),
            options: vec![],
            args: vec![],
            envs: forwards(&["HF_TOKEN"])?,
        };
        forward_client_config(&mut run, Some(&config));
        assert_eq!(
            run.envs.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            ["CUDA_VISIBLE_DEVICES=1", "HF_TOKEN"]
        );
        Ok(())
    }
}
