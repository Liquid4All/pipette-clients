//! Where an unpacked TensorRT Edge-LLM server archive keeps its launchers and
//! plugin. The installer checks an archive against this and `pipette-trt-edgellm`
//! launches from it, so neither re-derives the other's paths.
//!
//! There is no bundled catalog: upstream publishes no server archive, so every
//! install names a vault or remote archive.

use std::path::{Path, PathBuf};

/// The archive's top-level directory, which is the whole payload.
const PAYLOAD_DIR: &str = "trt-edgellm-server";

/// A server archive unpacked under a runtime's install directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrtEdgellmInstall {
    payload: PathBuf,
}

impl TrtEdgellmInstall {
    /// The archive unpacked into `install_dir`, the directory a stored runtime's
    /// `dir` names.
    pub fn under(install_dir: &Path) -> Self {
        Self {
            payload: install_dir.join(PAYLOAD_DIR),
        }
    }

    /// The server; `--build-only` turns it into the engine builder.
    pub fn server(&self) -> PathBuf {
        self.payload.join("bin").join("trt-edgellm-server")
    }

    /// The TensorRT plugin every engine needs at load.
    pub fn plugin(&self) -> PathBuf {
        self.payload
            .join("lib")
            .join("libNvInfer_edgellm_plugin.so")
    }

    /// The files an install must have, in the order a check reports them.
    pub fn required(&self) -> [PathBuf; 2] {
        [self.server(), self.plugin()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_payload_sits_under_its_top_level_dir() {
        let install = TrtEdgellmInstall::under(Path::new("/store/blobs"));
        assert_eq!(
            install.required(),
            [
                PathBuf::from("/store/blobs/trt-edgellm-server/bin/trt-edgellm-server"),
                PathBuf::from("/store/blobs/trt-edgellm-server/lib/libNvInfer_edgellm_plugin.so"),
            ]
        );
    }
}
