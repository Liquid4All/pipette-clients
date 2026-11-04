//! The interpreter and install-time provider of a bound ONNX GenAI runtime,
//! resolved from the plan's [`Runtime`].
use std::path::PathBuf;

use pipette_plan_types::{OnnxExecutionProvider, Runtime, RuntimeType};
use pipette_venv::{require_bound_venv, venv_python};

use crate::models::genai_name;

/// A bound `uv_onnx_genai` install: its venv python and the provider its
/// wheel was installed for.
#[derive(Debug)]
pub struct ResolvedOnnxGenaiInstall {
    pub python: PathBuf,
    pub execution_provider: OnnxExecutionProvider,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum UnservableProviderError {
    #[error(
        "ONNX GenAI's `append_provider` takes no `winml` name; for DirectML, set `execution_provider` to `directml`, `server_version` to `0.13.1`, and pin `onnxruntime-genai-directml==0.13.1` in `source.contents`"
    )]
    WinMl,
    /// It would reach GenAI's DirectML past the model's provider check, and
    /// its rows would carry another spelling.
    #[error("`{0}` is GenAI's name for DirectML; set `execution_provider` to `directml`")]
    CustomDml(String),
}

impl TryFrom<&Runtime> for ResolvedOnnxGenaiInstall {
    type Error = anyhow::Error;

    fn try_from(runtime: &Runtime) -> anyhow::Result<Self> {
        let Runtime::UvOnnxGenai(uv) = runtime else {
            anyhow::bail!("expected uv_onnx_genai, got `{}`", RuntimeType::of(runtime));
        };
        // At bind, so each cell refuses before its gate, watch or server.
        check_servable(&uv.execution_provider)?;
        let venv = require_bound_venv(&uv.source)?;
        Ok(Self {
            python: venv_python(&venv),
            execution_provider: uv.execution_provider.clone(),
        })
    }
}

fn check_servable(provider: &OnnxExecutionProvider) -> Result<(), UnservableProviderError> {
    match provider {
        OnnxExecutionProvider::WinMl => Err(UnservableProviderError::WinMl),
        OnnxExecutionProvider::Custom(name)
            if name.eq_ignore_ascii_case(genai_name(&OnnxExecutionProvider::DirectMl)) =>
        {
            Err(UnservableProviderError::CustomDml(name.clone()))
        }
        OnnxExecutionProvider::Cpu
        | OnnxExecutionProvider::Cuda
        | OnnxExecutionProvider::DirectMl
        | OnnxExecutionProvider::Custom(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_venv::venv_bin;

    use super::*;

    fn runtime(dir: &std::path::Path, provider: &str) -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type":"uv_onnx_genai", "server_version":"0.16.0",
            "python_version":"3.11", "execution_provider":provider,
            "source":{"type":"absolute_preinstalled", "dir":dir}
        }))?)
    }

    #[rstest]
    #[case::cuda("cuda", Ok(OnnxExecutionProvider::Cuda))]
    #[case::winml("winml", Err(UnservableProviderError::WinMl))]
    #[case::dml("dml", Err(UnservableProviderError::CustomDml("dml".to_owned())))]
    #[case::dml_in_any_case("DmL", Err(UnservableProviderError::CustomDml("DmL".to_owned())))]
    #[case::directml("directml", Ok(OnnxExecutionProvider::DirectMl))]
    #[case::custom("QNN", Ok(OnnxExecutionProvider::Custom("QNN".to_owned())))]
    fn a_runtime_provider_resolves_or_is_refused(
        #[case] provider: &str,
        #[case] expected: Result<OnnxExecutionProvider, UnservableProviderError>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir_all(venv_bin(dir.path()))?;
        std::fs::write(venv_python(dir.path()), "")?;
        let install = ResolvedOnnxGenaiInstall::try_from(&runtime(dir.path(), provider)?);
        match (install, expected) {
            (Ok(install), Ok(provider)) => {
                assert_eq!(install.python, venv_python(dir.path()));
                assert_eq!(install.execution_provider, provider);
            }
            (Err(err), Err(refused)) => assert_eq!(
                err.downcast_ref::<UnservableProviderError>(),
                Some(&refused),
                "{err:#}"
            ),
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
        Ok(())
    }

    #[test]
    fn resolved_onnx_genai_install_rejects_another_runtime() -> anyhow::Result<()> {
        let other = Runtime::AppleFoundation(Default::default());
        let Err(err) = ResolvedOnnxGenaiInstall::try_from(&other) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            err.to_string().contains("expected uv_onnx_genai"),
            "got {err}"
        );
        Ok(())
    }
}
