//! Host context for filling the artifact stores.

use pipette_http::HttpClient;
use pipette_http::retry::RetryPolicy;
use pipette_programs::ExternalPrograms;

use crate::progress::ProgressSink;
use crate::quota::{StoragePolicy, SweepPins};
use crate::vault::VaultAccess;

/// Host context for filling the artifact stores.
///
/// Public fields only. Cheap to clone, and cloned per artifact: the external
/// programs it carries were located once, by the caller, so a fetch never pays
/// for a lookup however many times the context is copied.
#[derive(Clone)]
pub struct ArtifactsContext {
    /// Shared HTTPS client for model HF/URL downloads and runtime archives.
    pub download_http_client: HttpClient,
    /// How each download retries a transient failure. [`RetryPolicy::DOWNLOAD`]
    /// by default.
    pub download_retry: RetryPolicy,
    /// `uv` and `docker`, as the process located them at startup. Never looked
    /// up here: an install spawns the binary its caller resolved, so a host
    /// missing one is told so by the install that needed it (see
    /// [`ResolvedProgram::path`](pipette_programs::ResolvedProgram::path)),
    /// and a host missing both still pulls a llama.cpp archive.
    pub programs: ExternalPrograms,
    /// Disk cap for the artifact stores, enforced after each publish. `None`
    /// (the default) leaves the stores uncapped.
    pub storage: Option<StoragePolicy>,
    /// Who to tell how a fetch is going. `None` (the default) also skips the
    /// size lookup a total would need.
    pub progress: Option<ProgressSink>,
    /// The device identity a `vault_*` source is fetched as. `None` for a
    /// workspace with no registration, where such a fetch is refused by name.
    pub vault: Option<VaultAccess>,
}

impl ArtifactsContext {
    pub fn new(download_http_client: HttpClient, programs: ExternalPrograms) -> Self {
        Self {
            download_http_client,
            download_retry: RetryPolicy::DOWNLOAD,
            programs,
            storage: None,
            progress: None,
            vault: None,
        }
    }

    pub fn with_download_retry(mut self, policy: RetryPolicy) -> Self {
        self.download_retry = policy;
        self
    }

    pub fn with_vault(mut self, access: VaultAccess) -> Self {
        self.vault = Some(access);
        self
    }

    /// Asking for progress also asks for the artifact's size, which costs a HEAD
    /// or a repo listing before the first byte, so a caller that will not render
    /// it should not install one.
    pub fn with_progress(mut self, sink: ProgressSink) -> Self {
        self.progress = Some(sink);
        self
    }

    pub fn with_storage(mut self, policy: StoragePolicy) -> Self {
        self.storage = Some(policy);
        self
    }

    /// Clone with `pins` merged into the storage policy (a no-op when
    /// uncapped): the run layer knows the in-flight cell, the fetch layer only
    /// its own artifact.
    pub fn with_pins(&self, pins: SweepPins) -> Self {
        let mut next = self.clone();
        if let Some(policy) = next.storage.as_mut() {
            policy.pins.merge(pins);
        }
        next
    }
}
