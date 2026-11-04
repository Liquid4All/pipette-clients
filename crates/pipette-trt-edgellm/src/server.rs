//! The server's command lines: one for the engine build, one per measured
//! server. Both name the same GGUF pair, precision and cache, so the server
//! finds the engines the build left. Also the engine stack a running server
//! reports.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::Context;
use serde_json::{Map, Value};

use pipette_completions::llama_server;
use pipette_http::HttpClient;
use pipette_plan_types::{
    EnvForward, TrtEdgellmExtraRuntimeConfiguration, TrtEdgellmGgufVision, TrtEdgellmPrecision,
};
use pipette_subprocess::{apply_client_config, apply_envs};

use crate::models::ResolvedGgufVisionModel;
use crate::runtimes::ResolvedTrtEdgellmInstall;

/// The server's default, recorded on the row so it names the precision it ran.
pub const DEFAULT_PRECISION: TrtEdgellmPrecision = TrtEdgellmPrecision::Fp16;

/// How long a measured server has to answer `/health`: it loads engines the
/// build left, not a request's HTTP timeout.
pub(crate) const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// One cell's server: the install, the weights, the cell's flags with the
/// precision resolved, and the cache the engines live in.
#[derive(Debug)]
pub struct Launch {
    server: PathBuf,
    model: PathBuf,
    mmproj: PathBuf,
    cache_dir: PathBuf,
    precision: TrtEdgellmPrecision,
    envs: Vec<EnvForward>,
    raw: Vec<String>,
    client_config: Option<TrtEdgellmExtraRuntimeConfiguration>,
}

impl Launch {
    pub fn new(
        install: &ResolvedTrtEdgellmInstall,
        model: &ResolvedGgufVisionModel,
        flags: &TrtEdgellmGgufVision,
        cache_dir: &Path,
        client_config: Option<&TrtEdgellmExtraRuntimeConfiguration>,
    ) -> Self {
        // Every field named: a flag plan-types adds fails to compile here
        // until the launch takes it or sets it aside.
        let TrtEdgellmGgufVision {
            precision,
            envs,
            raw,
        } = flags.clone();
        Self {
            server: install.server(),
            model: model.backbone.clone(),
            mmproj: model.mmproj.clone(),
            cache_dir: cache_dir.to_owned(),
            precision: precision.unwrap_or(DEFAULT_PRECISION),
            envs,
            raw,
            client_config: client_config.cloned(),
        }
    }

    /// The cell's flags as run: the precision resolved.
    pub fn flags(&self) -> TrtEdgellmGgufVision {
        TrtEdgellmGgufVision {
            precision: Some(self.precision),
            envs: self.envs.clone(),
            raw: self.raw.clone(),
        }
    }

    /// Converts the GGUF pair and builds this precision's engines into the
    /// cache, or finds them there. The cell's `raw` is the measured server's.
    pub fn build_only(&self) -> Command {
        let mut command = self.base();
        command.arg("--build-only");
        command
    }

    /// The measured server on `port`. It never builds: with no engines for its
    /// key it exits before binding the port.
    pub fn serve(&self, port: u16) -> Command {
        let mut command = self.base();
        command
            .args(["--host", "127.0.0.1", "--port"])
            .arg(port.to_string())
            // Serve-only: the build never sees them, so a flag that shapes the
            // engines must be a typed plan field, as `precision` is.
            .args(&self.raw);
        command
    }

    fn base(&self) -> Command {
        let mut command = Command::new(&self.server);
        command
            .arg("--model")
            .arg(&self.model)
            .arg("--mmproj")
            .arg(&self.mmproj)
            .args(["--precision", self.precision.as_str()])
            .arg("--cache-dir")
            .arg(&self.cache_dir);
        apply_client_config(&mut command, self.client_config.as_ref());
        apply_envs(&mut command, &self.envs);
        command
    }
}

/// `/props`' key for the engine stack, and the row's `client_extras` key.
const ENGINE_STACK: &str = "trt_edgellm";

/// The engine stack the server reports in `/props` (its cache key, the
/// Edge-LLM commit, TensorRT, CUDA, L4T and the SoC), verbatim, as the row's
/// `client_extras.trt_edgellm`.
pub(crate) fn engine_stack(http: &HttpClient, url: &str) -> anyhow::Result<Map<String, Value>> {
    let stack = llama_server::props(http, url)?
        .remove(ENGINE_STACK)
        .with_context(|| {
            format!(
                "the server's /props has no `{ENGINE_STACK}`: the archive is too old to \
                 report its engine stack; use one built from a newer Edge-LLM"
            )
        })?;
    Ok(Map::from_iter([(ENGINE_STACK.to_owned(), stack)]))
}

#[cfg(test)]
mod tests {
    use httpmock::MockServer;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn props_reporting(props: &Value) -> anyhow::Result<anyhow::Result<Map<String, Value>>> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("GET").path("/props");
            then.status(200).json_body(props.clone());
        });
        let http = HttpClient::loopback("pipette-test", Duration::from_secs(5))?;
        Ok(engine_stack(&http, &server.base_url()))
    }

    #[test]
    fn the_row_carries_the_servers_engine_stack_verbatim() -> anyhow::Result<()> {
        let stack = json!({
            "engine_key": "9c1f0e7a2b3d", "edgellm_commit": "4e3f9d23f052", "dirty": false,
            "tensorrt": "10.16.0", "cuda_runtime": "13.2", "cuda_driver": "13.2",
            "l4t": "R39.0", "soc": null,
        });
        let extras = props_reporting(&json!({"n_ctx": 8192, "trt_edgellm": stack}))??;
        assert_eq!(Value::Object(extras), json!({ "trt_edgellm": stack }));
        Ok(())
    }

    #[test]
    fn a_server_without_an_engine_stack_is_too_old() -> anyhow::Result<()> {
        let Err(err) = props_reporting(&json!({"n_ctx": 8192}))? else {
            anyhow::bail!("a /props without trt_edgellm was accepted");
        };
        assert!(format!("{err:#}").contains("too old"), "{err:#}");
        Ok(())
    }

    fn launch(precision: Option<TrtEdgellmPrecision>, raw: &[&str]) -> Launch {
        Launch {
            server: "/i/bin/trt-edgellm-server".into(),
            model: "/m/model.gguf".into(),
            mmproj: "/m/mmproj.gguf".into(),
            cache_dir: "/ws/cache/trt".into(),
            precision: precision.unwrap_or(DEFAULT_PRECISION),
            envs: Vec::new(),
            raw: raw.iter().map(|&s| s.to_owned()).collect(),
            client_config: None,
        }
    }

    fn argv(command: &Command) -> String {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[rstest]
    #[case::default(None, "fp16")]
    #[case::bf16(Some(TrtEdgellmPrecision::Bf16), "bf16")]
    #[case::weight_only(Some(TrtEdgellmPrecision::W8a16), "w8a16")]
    fn the_build_names_the_pair_precision_and_cache(
        #[case] precision: Option<TrtEdgellmPrecision>,
        #[case] spelled: &str,
    ) {
        assert_eq!(
            argv(&launch(precision, &["--robo-embodiment", "0"]).build_only()),
            format!(
                "--model /m/model.gguf --mmproj /m/mmproj.gguf --precision {spelled} \
                 --cache-dir /ws/cache/trt --build-only"
            )
        );
    }

    #[test]
    fn the_server_takes_the_cells_raw_flags_last() {
        assert_eq!(
            argv(&launch(None, &["--robo-embodiment", "0"]).serve(8080)),
            "--model /m/model.gguf --mmproj /m/mmproj.gguf --precision fp16 --cache-dir \
             /ws/cache/trt --host 127.0.0.1 --port 8080 --robo-embodiment 0"
        );
    }

    #[test]
    fn the_row_names_the_precision_it_ran() {
        assert_eq!(
            launch(None, &[]).flags().precision,
            Some(TrtEdgellmPrecision::Fp16)
        );
    }
}
