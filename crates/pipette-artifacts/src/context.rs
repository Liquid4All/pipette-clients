//! Host context for filling the artifact stores.

use std::path::PathBuf;
use std::sync::OnceLock;

use pipette_http::HttpClient;

use crate::progress::ProgressSink;
use crate::quota::{StoragePolicy, SweepPins};

/// Host context for filling the artifact stores.
///
/// Public fields only. Reuse across multiple ensures so tool paths resolve once.
#[derive(Clone)]
pub struct ArtifactsContext {
    /// Shared HTTPS client for model HF/URL downloads and runtime archives.
    pub download_http_client: HttpClient,
    /// Resolved on first UV/MLX install so docker/llama pulls work without `uv`.
    pub uv_executable: OnceLock<PathBuf>,
    /// Resolved on first docker pull so non-docker installs work without Docker.
    pub docker_executable: OnceLock<PathBuf>,
    /// Host `python3` / `python` (lazy). Callers may pre-seed; see
    /// [`crate::resolve_python_executable`].
    pub python_executable: OnceLock<PathBuf>,
    /// Disk cap for the artifact stores, enforced after each publish. `None`
    /// leaves the stores uncapped — the default for library and test callers.
    pub storage: Option<StoragePolicy>,
    /// Who to tell how a fetch is going, if anyone. `None` — the default — costs
    /// a fetch nothing, and also skips the size lookup a total would need.
    pub progress: Option<ProgressSink>,
}

impl ArtifactsContext {
    /// Default CLI/worker context: shared download client, system uv/docker/python.
    pub fn new(download_http_client: HttpClient) -> Self {
        Self {
            download_http_client,
            uv_executable: OnceLock::new(),
            docker_executable: OnceLock::new(),
            python_executable: OnceLock::new(),
            storage: None,
            progress: None,
        }
    }

    /// Report byte-level fetch progress to `sink`.
    ///
    /// Asking for progress also asks for the artifact's size, which costs a HEAD
    /// or a repo listing before the first byte — so a caller that will not render
    /// it should not install one.
    pub fn with_progress(mut self, sink: ProgressSink) -> Self {
        self.progress = Some(sink);
        self
    }

    /// Cap this context's stores at `policy`.
    pub fn with_storage(mut self, policy: StoragePolicy) -> Self {
        self.storage = Some(policy);
        self
    }

    /// Clone with `pins` merged into the storage policy (a no-op when
    /// uncapped). The run layer knows the in-flight cell; the fetch layer knows
    /// only its own artifact.
    pub fn with_pins(&self, pins: SweepPins) -> Self {
        let mut next = self.clone();
        if let Some(policy) = next.storage.as_mut() {
            policy.pins.merge(pins);
        }
        next
    }
}
