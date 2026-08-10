//! The unified workspace: a single flat `.pipette/` store (TOML manifest) that
//! mints one concrete store per subdirectory (`MARKER_NAME = "pipette"`).
//!
//! ```text
//! .pipette/
//!   manifest.toml
//!   identity/          # shared: keys, registration, device labels, settings
//!   benchmarks/        # shared: local/ + remote/
//!   results/           # shared: local/ + remote/{pending,synced}
//!   state/evals/       # shared: eval checkpoints
//!   runtimes/          # FLAT — builds from all runtimes co-mingled
//!   models/            # FLAT — all runtimes co-mingled
//! ```

use std::path::{Path, PathBuf};

use pipette_artifacts::quota::StoragePolicy;
use pipette_artifacts::ArtifactsContext;
use pipette_http::HttpClient;
use pipette_mgmt_client::{AuthIdentity, MgmtClient};
use pipette_workspace::{InitResult, Workspace};

use crate::error::Result;
use crate::identity::IdentityStore;
use crate::storage_quota::{resolve_storage_quota, StorageQuota};

/// A registered identity and a client pointed at its server.
pub struct MgmtSession {
    pub identity: IdentityStore,
    pub client: MgmtClient,
    pub auth: AuthIdentity,
    pub server_url: String,
}

/// Workspace directory name: `work_dir/.pipette`.
const MARKER_NAME: &str = "pipette";

#[derive(Debug)]
pub struct PipetteWorkspace {
    root: PathBuf,
    storage_quota: StorageQuota,
}

impl PipetteWorkspace {
    /// File-backed benchmark catalog under `benchmarks/`.
    pub fn benchmarks(&self) -> crate::benchmarks::BenchmarkStore {
        crate::benchmarks::BenchmarkStore::new(self.root().join("benchmarks"))
    }

    /// Runtime install tree under `runtimes/`.
    pub fn runtimes(&self) -> pipette_artifacts::runtime::RuntimeArtifactStore {
        pipette_artifacts::runtime::RuntimeArtifactStore::new(self.root().join("runtimes"))
    }

    /// Model install tree under `models/`.
    pub fn models(&self) -> pipette_artifacts::model::ModelArtifactStore {
        pipette_artifacts::model::ModelArtifactStore::new(self.root().join("models"))
    }

    /// Artifacts context bound to this workspace's stores and the disk cap they
    /// share. The policy's roots are taken from the stores themselves, so the
    /// sweep can only ever look where the fetches write.
    pub fn artifacts(&self, http: &HttpClient) -> ArtifactsContext {
        ArtifactsContext::new(http.clone()).with_storage(StoragePolicy::new(
            self.storage_quota.bytes,
            self.models().models_dir().to_path_buf(),
            self.runtimes().runtimes_dir().to_path_buf(),
        ))
    }

    pub fn eval_completions(&self) -> pipette_ops::EvalCompletionsStore {
        pipette_ops::EvalCompletionsStore::new(self.root().join("state/evals"))
    }

    /// Where a runtime's engine keeps compiled artifacts it can rebuild:
    /// `cache/<key>/`, under the same [`RuntimeStorageKey`] as
    /// `runtimes/<key>/`.
    ///
    /// The same key, so removing the runtime is what makes its cache
    /// unaddressable, and `runtimes remove` reclaims it by looking in one
    /// place. No engine segment: the key already names the runtime type, and a
    /// runtime has exactly one engine.
    ///
    /// Beside the artifact stores rather than inside them, because the quota
    /// measures an entry's `blobs/` alone — a cache grown in there would be
    /// unaccounted, and eviction would trade a venv for bytes that cost one
    /// compile to rebuild.
    ///
    /// [`RuntimeStorageKey`]: pipette_artifacts::runtime::RuntimeStorageKey
    pub fn compile_cache(
        &self,
        runtime: &pipette_plan_types::Runtime,
    ) -> anyhow::Result<std::path::PathBuf> {
        let key = pipette_artifacts::runtime::RuntimeStorageKey::of(runtime)?;
        Ok(self.root().join("cache").join(key.as_str()))
    }

    pub fn identity(&self) -> crate::identity::IdentityStore {
        crate::identity::IdentityStore::new(self.root().join("identity"))
    }

    pub fn results(&self) -> crate::results::ResultsStore {
        crate::results::ResultsStore::new(self.root().join("results"))
    }

    /// The registered identity plus a client bound to its server — the opening
    /// move of every command that reaches the management server.
    pub fn mgmt_session(&self) -> Result<MgmtSession> {
        let identity = self.identity();
        let registration = identity.require_registration()?;
        let auth = identity.signing_identity()?;
        Ok(MgmtSession {
            client: MgmtClient::new(&registration.server_url)?,
            server_url: registration.server_url,
            auth,
            identity,
        })
    }

    /// Create the workspace under `work_dir/.pipette` (idempotent, repair-safe).
    ///
    /// The instance is built first so its trait path methods can enumerate the
    /// subdirs to create, then the generic [`Workspace::init`] writes the
    /// manifest and ensures the tree.
    pub fn init(work_dir: &Path) -> anyhow::Result<InitResult> {
        let root = storage_root(work_dir);
        let dirs = [
            root.join("identity"),
            root.join("benchmarks/local"),
            root.join("benchmarks/remote"),
            root.join("results/local"),
            root.join("results/remote/pending"),
            root.join("results/remote/synced"),
            root.join("state/evals"),
            root.join("runtimes"),
            root.join("models"),
        ];
        Workspace::init(work_dir, MARKER_NAME, dirs)
    }

    /// Open an existing workspace, validating the TOML manifest is present, and
    /// resolve the disk cap it enforces.
    ///
    /// The quota is settled here rather than threaded through the commands: like
    /// the root itself it is invocation config resolved against workspace state
    /// (`identity/settings.json`), so the opened workspace is the one object that
    /// knows it. `quota_override` is already parsed — see
    /// [`crate::storage_quota::resolve_storage_quota`].
    pub fn open(work_dir: &Path, quota_override: Option<u64>) -> anyhow::Result<Self> {
        let inner = Workspace::open(work_dir, MARKER_NAME)?;
        let root = inner.root().to_path_buf();
        let storage_quota =
            resolve_storage_quota(&IdentityStore::new(root.join("identity")), quota_override)?;
        Ok(Self {
            root,
            storage_quota,
        })
    }

    /// Absolute path to `.pipette/` (workspace root).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The disk cap this workspace enforces, and which rung of the precedence
    /// chain supplied it.
    pub fn storage_quota(&self) -> StorageQuota {
        self.storage_quota
    }
}

/// The storage root (`<work_dir>/.pipette`) without opening — for existence
/// checks and diagnostics.
pub fn storage_root(work_dir: &Path) -> PathBuf {
    pipette_workspace::storage_root(work_dir, MARKER_NAME)
}

/// [`pipette_workspace::require_workspace`] bound to this crate's `.pipette` marker.
pub fn require_workspace(work_dir: &Path, work_dir_arg: Option<&Path>) -> anyhow::Result<()> {
    pipette_workspace::require_workspace(work_dir, work_dir_arg, MARKER_NAME)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::PipetteWorkspace;

    /// An initialized `PipetteWorkspace` in a per-process temp dir, removed on
    /// drop so a panicking test doesn't leak it (`tempfile` isn't a dep here).
    pub(crate) struct TempWorkspace {
        pub ws: PipetteWorkspace,
        path: PathBuf,
    }

    impl TempWorkspace {
        pub(crate) fn new(label: &str) -> anyhow::Result<Self> {
            // A per-process counter keeps the path unique even if two tests pass
            // the same label — `cargo test` runs them in parallel threads that
            // share the pid, so the label alone wouldn't disambiguate.
            static SEQ: AtomicU64 = AtomicU64::new(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("pipette-cli-{label}-{}-{seq}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            PipetteWorkspace::init(&path)?;
            let ws = PipetteWorkspace::open(&path, None)?;
            Ok(Self { ws, path })
        }

        /// The same workspace under an explicit cap. The storage tests populate
        /// the store first and derive the quota from what they stored, so they
        /// cannot pass it at construction — and this still goes through the real
        /// `open` resolution rather than hand-building a `StorageQuota`.
        pub(crate) fn reopen_with_quota(
            &self,
            quota_bytes: u64,
        ) -> anyhow::Result<PipetteWorkspace> {
            PipetteWorkspace::open(&self.path, Some(quota_bytes))
        }
    }

    impl Drop for TempWorkspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pipette-cli-ws-{}-{label}", std::process::id()))
    }

    /// `init` builds the full unified tree and writes the manifest. Guards the
    /// byte-for-byte layout the module doc promises. It cannot see an accessor
    /// that mints outside that tree — that is
    /// `eval_completions_is_minted_where_init_created_it`.
    #[test]
    fn init_creates_expected_dirs() -> anyhow::Result<()> {
        let work = temp_dir("init");
        let _ = std::fs::remove_dir_all(&work);
        let root = match PipetteWorkspace::init(&work)? {
            InitResult::Created(root) => root,
            InitResult::AlreadyExists(_) => anyhow::bail!("expected a fresh Created workspace"),
        };
        let expected = [
            "identity",
            "runtimes",
            "models",
            "benchmarks/local",
            "benchmarks/remote",
            "results/local",
            "results/remote/pending",
            "results/remote/synced",
            "state/evals",
        ];
        expected.into_iter().try_for_each(|sub| {
            anyhow::ensure!(root.join(sub).is_dir(), "missing dir: {sub}");
            anyhow::Ok(())
        })?;
        anyhow::ensure!(root.join("manifest.toml").exists(), "missing manifest");
        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }

    /// A store accessor and `init` name the same subdirectory twice, 40 lines
    /// apart. Diverge them and nothing else fails: the store lazily creates its
    /// root, so the workspace grows a second directory and every existing
    /// checkpoint is orphaned. Asserts the invariant rather than the literal.
    #[test]
    fn eval_completions_is_minted_where_init_created_it() -> anyhow::Result<()> {
        let work = temp_dir("evals-mint");
        let _ = std::fs::remove_dir_all(&work);
        PipetteWorkspace::init(&work)?;
        let ws = PipetteWorkspace::open(&work, None)?;
        let minted = ws.eval_completions().root().to_path_buf();
        anyhow::ensure!(
            minted.is_dir(),
            "eval store minted outside the initialized tree: {}",
            minted.display()
        );
        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_after_init() -> anyhow::Result<()> {
        let work = temp_dir("open");
        let _ = std::fs::remove_dir_all(&work);
        PipetteWorkspace::init(&work)?;
        let ws = PipetteWorkspace::open(&work, None)?;
        anyhow::ensure!(
            ws.root().ends_with(".pipette"),
            "unexpected root: {}",
            ws.root().display()
        );
        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }
}
