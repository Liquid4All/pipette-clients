//! The directory of a bound ONNX GenAI model, resolved from the plan's
//! [`Model`]: its config, graph and external weights, then the providers it
//! declares against the runtime's.
use std::collections::BTreeMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::{Model, ModelType, OnnxExecutionProvider};

const REQUIRED_FILES: &[&str] = &["genai_config.json"];
const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Deserialize)]
struct Config {
    model: ModelConfig,
}
#[derive(Deserialize)]
struct ModelConfig {
    decoder: Decoder,
}
#[derive(Deserialize)]
struct Decoder {
    filename: String,
    #[serde(default)]
    session_options: SessionOptions,
}
#[derive(Default, Deserialize)]
struct SessionOptions {
    #[serde(default)]
    provider_options: Vec<BTreeMap<String, serde_json::Value>>,
}

/// A bound `onnx_genai` model's directory, with the graph its
/// `genai_config.json` names and that graph's external data on disk.
#[derive(Debug)]
pub struct ResolvedOnnxGenaiModel {
    dir: PathBuf,
    /// Parsed by [`Self::check_provider`], which a custom provider skips.
    provider_options: Vec<BTreeMap<String, serde_json::Value>>,
}

impl TryFrom<&Model> for ResolvedOnnxGenaiModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        let dir = require_bound_model_dir(model, ModelType::OnnxGenai, REQUIRED_FILES)
            .map_err(|e| anyhow::anyhow!("{e}. This model may be a plain ONNX export. Rebuild it with `python -m onnxruntime_genai.models.builder`."))?;
        let path = dir.join("genai_config.json");
        let config: Config = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("could not read {}", path.display()))?;
        let decoder = config.model.decoder;
        let graph = dir.join(&decoder.filename);
        anyhow::ensure!(
            graph.is_file(),
            "genai_config.json model.decoder.filename names the missing ONNX graph `{}`",
            graph.display()
        );
        validate_external_data(&graph)?;
        Ok(Self {
            dir,
            provider_options: decoder.session_options.provider_options,
        })
    }
}

impl Deref for ResolvedOnnxGenaiModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.dir
    }
}

impl ResolvedOnnxGenaiModel {
    /// Refuses a model that declares no support for `provider`, the one the
    /// runtime's wheel was installed for.
    pub fn check_provider(&self, provider: &OnnxExecutionProvider) -> anyhow::Result<()> {
        if provider.is_custom() {
            return Ok(());
        }
        let mut declared = self
            .provider_options
            .iter()
            .map(parse_provider_options)
            .collect::<anyhow::Result<Vec<_>>>()?;
        // CPU builds name no provider, so an empty list declares CPU.
        if declared.is_empty() {
            declared.push(OnnxExecutionProvider::Cpu);
        }
        anyhow::ensure!(
            declared.contains(provider),
            "ONNX GenAI provider mismatch: the model declares [{}], but the runtime uses `{provider}`. Select a compatible `ep` with `--runtime`, or select a compatible model variant with `--model`.",
            declared
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }
}

fn parse_provider_options(
    options: &BTreeMap<String, serde_json::Value>,
) -> anyhow::Result<OnnxExecutionProvider> {
    anyhow::ensure!(
        options.len() == 1,
        "each genai_config.json provider_options entry must name exactly one provider"
    );
    let key = options
        .keys()
        .next()
        .context("provider_options entry is empty")?
        .to_ascii_lowercase();
    Ok(OnnxExecutionProvider::known()
        .find(|provider| genai_name(provider) == key)
        .unwrap_or_else(|| OnnxExecutionProvider::parse(&key)))
}

/// GenAI's name for `provider`, as `append_provider` takes it; GenAI refuses
/// `directml`.
pub(crate) fn genai_name(provider: &OnnxExecutionProvider) -> &str {
    match provider {
        OnnxExecutionProvider::DirectMl => "dml",
        OnnxExecutionProvider::Cpu
        | OnnxExecutionProvider::Cuda
        | OnnxExecutionProvider::WinMl
        | OnnxExecutionProvider::Custom(_) => provider.as_str(),
    }
}

/// Scan small graphs for the builder's literal external-data sibling name.
/// This is not a protobuf parser: it cannot catch truncation before the location
/// string, arbitrary external-data filenames, or declarations in graphs above
/// 64 MiB. Those failures remain the model loader's responsibility.
fn validate_external_data(graph: &Path) -> anyhow::Result<()> {
    let mut sidecar = graph.as_os_str().to_os_string();
    sidecar.push(".data");
    let sidecar = PathBuf::from(sidecar);
    if sidecar.exists() {
        anyhow::ensure!(
            sidecar.is_file() && sidecar.metadata()?.len() > 0,
            "ONNX external-data file `{}` is empty or not a file; fetch the complete model directory",
            sidecar.display()
        );
    } else if graph.metadata()?.len() <= MAX_SCAN_BYTES {
        let name = sidecar
            .file_name()
            .context("external-data path has no filename")?
            .to_string_lossy();
        let bytes = std::fs::read(graph)?;
        anyhow::ensure!(
            !bytes.windows(name.len()).any(|w| w == name.as_bytes()),
            "The graph references the missing ONNX external-data file `{}`. Fetch the complete model directory.",
            sidecar.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn fixture(providers: serde_json::Value) -> anyhow::Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("genai_config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model": {"decoder": {"filename": "foo.onnx", "session_options": {"provider_options": providers}}},
                "ignored": true
            }))?,
        )?;
        std::fs::write(dir.path().join("foo.onnx"), b"small self-contained graph")?;
        Ok(dir)
    }

    fn resolve(dir: &Path) -> anyhow::Result<ResolvedOnnxGenaiModel> {
        let model: Model = serde_json::from_value(
            serde_json::json!({"type": "onnx_genai", "source": "absolute_dir", "dir": dir}),
        )?;
        ResolvedOnnxGenaiModel::try_from(&model)
    }

    #[rstest]
    #[case::cpu("[]", "cpu", true)]
    #[case::cpu_on_cuda("[]", "cuda", false)]
    #[case::cuda_on_cpu(r#"[{"cuda":{}}]"#, "cpu", false)]
    #[case::cuda(r#"[{"CUDA":{}}]"#, "cuda", true)]
    #[case::dml(r#"[{"dml":{}}]"#, "directml", true)]
    #[case::multiple(r#"[{"cuda":{}},{"cpu":{}}]"#, "cpu", true)]
    #[case::custom_runtime("[]", "QNN-Htp", true)]
    #[case::custom_model(r#"[{"QNN-Htp":{}}]"#, "cpu", false)]
    #[case::both_custom(r#"[{"QNN-Htp":{}}]"#, "QNN-Htp", true)]
    fn providers(
        #[case] declared: &str,
        #[case] runtime: &str,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let dir = fixture(serde_json::from_str(declared)?)?;
        let result = resolve(dir.path())?.check_provider(&OnnxExecutionProvider::parse(runtime));
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        if let Err(err) = result {
            assert!(err.to_string().contains("provider mismatch"));
        }
        Ok(())
    }

    #[rstest]
    #[case::complete(true, Some(b"weights".as_slice()), true)]
    #[case::missing(true, None, false)]
    #[case::empty(true, Some(b"".as_slice()), false)]
    #[case::single_file(false, None, true)]
    fn external_data(
        #[case] reference: bool,
        #[case] data: Option<&[u8]>,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let dir = fixture(serde_json::json!([]))?;
        if reference {
            std::fs::write(dir.path().join("foo.onnx"), b"stub foo.onnx.data")?;
        }
        if let Some(data) = data {
            std::fs::write(dir.path().join("foo.onnx.data"), data)?;
        }
        let result = resolve(dir.path());
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        if let Err(err) = result {
            assert!(err.to_string().contains("foo.onnx.data"));
        }
        Ok(())
    }

    #[test]
    fn missing_graph_names_config_key() -> anyhow::Result<()> {
        let dir = fixture(serde_json::json!([]))?;
        std::fs::remove_file(dir.path().join("foo.onnx"))?;
        let Err(err) = resolve(dir.path()) else {
            anyhow::bail!("expected missing graph rejection");
        };
        assert!(err.to_string().contains("model.decoder.filename"));
        Ok(())
    }

    #[test]
    fn missing_config_names_the_file_and_builder() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let Err(err) = resolve(dir.path()) else {
            anyhow::bail!("expected missing config error");
        };
        let message = err.to_string();
        assert!(message.contains("genai_config.json"));
        assert!(message.contains("python -m onnxruntime_genai.models.builder"));
        Ok(())
    }
}
