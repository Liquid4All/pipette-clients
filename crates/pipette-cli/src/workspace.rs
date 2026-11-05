//! The `.pipette/` workspace, which mints one store per subdirectory:
//!
//! ```text
//! .pipette/
//!   manifest.toml
//!   identity/          # keys, registration, device labels, settings
//!   benchmarks/        # local/ + remote/
//!   results/           # local/ + remote/{pending,synced}
//!   state/evals/       # eval checkpoints
//!   state/worker/      # the worker's in-flight job marker
//!   runtimes/          # flat: every runtime type's builds together
//!   models/            # flat: every runtime type's models together
//! ```
//!
//! `cache/` and `datasets/` appear on first use.

use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_artifacts::quota::StoragePolicy;
use pipette_artifacts::{ArtifactsContext, VaultAccess};
use pipette_http::HttpClient;
use pipette_mgmt_client::client_api;
use pipette_ops::EvalCompletionsStore;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::Dataset;
use pipette_programs::ExternalPrograms;
use pipette_workspace::{InitResult, Workspace};

use crate::error::Result;
use crate::identity::{ClientDeviceStore, ClientRegistrationStore, ClientSettingsStore};
use crate::storage_quota::{StorageQuota, resolve_storage_quota};

/// Workspace directory name: `work_dir/.pipette`.
const MARKER_NAME: &str = "pipette";

#[derive(Debug)]
pub struct PipetteWorkspace {
    root: PathBuf,
    storage_quota: StorageQuota,
}

impl PipetteWorkspace {
    pub fn benchmarks(&self) -> crate::benchmarks::BenchmarkStore {
        crate::benchmarks::BenchmarkStore::new(self.root().join("benchmarks"))
    }

    pub fn runtimes(&self) -> pipette_artifacts::runtime::RuntimeArtifactStore {
        pipette_artifacts::runtime::RuntimeArtifactStore::new(self.root().join("runtimes"))
    }

    pub fn models(&self) -> pipette_artifacts::model::ModelArtifactStore {
        pipette_artifacts::model::ModelArtifactStore::new(self.root().join("models"))
    }

    /// Artifacts context bound to this workspace's stores and their shared disk
    /// cap. The policy's roots come from the stores, so the sweep only looks
    /// where the fetches write. `programs` is the process's: which `uv` came off
    /// the command line. Unregistered, a `vault_*` source is refused at fetch
    /// time, so no registration is required here.
    pub fn artifacts(&self, http: &HttpClient, programs: &ExternalPrograms) -> ArtifactsContext {
        let ctx =
            ArtifactsContext::new(http.clone(), programs.clone()).with_storage(StoragePolicy::new(
                self.storage_quota.bytes,
                self.models().models_dir().to_path_buf(),
                self.runtimes().runtimes_dir().to_path_buf(),
            ));
        match self.vault_access() {
            Some(access) => ctx.with_vault(access),
            None => ctx,
        }
    }

    /// The device identity a vault fetch signs as, if this workspace has one.
    fn vault_access(&self) -> Option<VaultAccess> {
        let registration = self.registration();
        Some(VaultAccess {
            server: registration.server_url().ok()?,
            identity: registration.signing_identity().ok()?,
        })
    }

    /// Where a runtime's engine keeps compiled artifacts it can rebuild:
    /// `cache/<key>/`, under the same [`RuntimeStorageKey`] as `runtimes/<key>/`,
    /// so `runtimes remove` reclaims it in one place.
    ///
    /// Outside the artifact stores because the quota measures an entry's
    /// `blobs/` alone: eviction would trade a venv for bytes one compile rebuilds.
    ///
    /// [`RuntimeStorageKey`]: pipette_artifacts::runtime::RuntimeStorageKey
    pub fn compile_cache(
        &self,
        runtime: &pipette_plan_types::Runtime,
    ) -> anyhow::Result<std::path::PathBuf> {
        let key = pipette_artifacts::runtime::RuntimeStorageKey::of(runtime)?;
        Ok(self.root().join("cache").join(key.as_str()))
    }

    /// `identity/registration.json` and the keypair beside it.
    pub fn registration(&self) -> ClientRegistrationStore {
        ClientRegistrationStore::new(self.identity_dir())
    }

    /// `identity/device.json`: this machine's labels, on a different lifetime
    /// from the registration.
    pub fn device(&self) -> ClientDeviceStore {
        ClientDeviceStore::new(self.identity_dir())
    }

    /// `identity/settings.json`: client-wide local policy.
    pub fn settings(&self) -> ClientSettingsStore {
        ClientSettingsStore::new(self.identity_dir())
    }

    fn identity_dir(&self) -> PathBuf {
        self.root().join("identity")
    }

    /// Every file under `identity/`, for `auth reset`. Kept beside the accessors
    /// so a fourth store is one line away from being cleared too.
    pub fn clear_identity(&self) -> Result<()> {
        self.registration().clear()?;
        self.device().clear()?;
        self.settings().clear()
    }

    pub fn results(&self) -> crate::results::ResultsStore {
        crate::results::ResultsStore::new(self.root().join("results"))
    }

    pub fn in_flight_marker(&self) -> crate::client::worker::in_flight::MarkerStore {
        crate::client::worker::in_flight::MarkerStore::new(&self.root().join("state/worker"))
    }

    /// Create the workspace under `work_dir/.pipette` (idempotent, repair-safe).
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
            root.join("state/worker"),
            root.join("runtimes"),
            root.join("models"),
        ];
        Workspace::init(work_dir, MARKER_NAME, dirs)
    }

    /// Open an existing workspace and resolve its disk cap here, since the cap
    /// reads workspace state (`identity/settings.json`).
    pub fn open(work_dir: &Path, quota_override: Option<u64>) -> anyhow::Result<Self> {
        let inner = Workspace::open(work_dir, MARKER_NAME)?;
        let root = inner.root().to_path_buf();
        let storage_quota = resolve_storage_quota(
            &ClientSettingsStore::new(root.join("identity")),
            quota_override,
        )?;
        Ok(Self {
            root,
            storage_quota,
        })
    }

    /// Absolute path to `.pipette/`.
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn storage_quota(&self) -> StorageQuota {
        self.storage_quota
    }
}

/// Where a dataset is fetched from, and as whom.
struct DatasetRoute {
    server: pipette_mgmt_client::ServerUrl,
    auth: pipette_mgmt_client::ClientIdentity,
    http: HttpClient,
    path: String,
}

impl PipetteWorkspace {
    /// The route the management server stamped onto `dataset`, signed as this
    /// workspace's device.
    fn dataset_route(&self, dataset: &Dataset, http: HttpClient) -> anyhow::Result<DatasetRoute> {
        let path = dataset.url.as_ref().context(
            "the dataset names no download URL; the management server stamps one on \
             `GET /benchmarks/{id}`, so this definition did not come from a catalog sync",
        )?;
        let registration = self.registration();
        let server = registration.server_url().context(
            "a dataset is fetched from the management server, so this workspace has to be \
             registered with one",
        )?;
        let auth = registration.signing_identity()?;
        tracing::info!(
            bytes = dataset.bytes,
            upstream = %dataset.upstream.reference(),
            url = %format_args!("{server}{path}"),
            "downloading corpus"
        );
        Ok(DatasetRoute {
            server,
            auth,
            http,
            path: path.as_ref().to_owned(),
        })
    }
}

/// How long a corpus download may wait for the response, and then for each
/// read of its body, before it fails: long enough for a server that streams
/// from object storage, short enough that a dead connection does not hold a
/// run (and the disk the corpus is reserved) for good.
const CORPUS_STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

impl PipetteWorkspace {
    /// [`RunContext::fetch_dataset_to`] with `stall` as the longest wait for
    /// the response, and then for each read of the body.
    fn stream_dataset(
        &self,
        dataset: &Dataset,
        dest: &mut dyn std::io::Write,
        stall: std::time::Duration,
    ) -> anyhow::Result<u64> {
        let http = HttpClient::with_request_timeout("pipette", stall)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let route = self.dataset_route(dataset, http)?;
        let mut body =
            client_api::get_dataset_stream(&route.server, &route.http, &route.auth, &route.path)?;
        std::io::copy(&mut body, dest).with_context(|| {
            format!(
                "the corpus download from {}{} broke off (or sent nothing for {stall:?})",
                route.server, route.path
            )
        })
    }
}

/// What a run gets from its workspace.
impl RunContext for PipetteWorkspace {
    /// Signed as this workspace's device, on the route the management server
    /// stamped onto the benchmark.
    fn fetch_dataset(&self, dataset: &Dataset) -> anyhow::Result<Vec<u8>> {
        let http = HttpClient::new("pipette").map_err(|e| anyhow::anyhow!("{e}"))?;
        let route = self.dataset_route(dataset, http)?;
        Ok(client_api::get_dataset(
            &route.server,
            &route.http,
            &route.auth,
            &route.path,
        )?)
    }

    /// Streamed into `dest` as it arrives: a corpus may be larger than the
    /// memory of the device that runs it. A server that sends nothing for
    /// [`CORPUS_STALL_TIMEOUT`] fails the download rather than holding the run
    /// on a dead socket.
    fn fetch_dataset_to(
        &self,
        dataset: &Dataset,
        dest: &mut dyn std::io::Write,
    ) -> anyhow::Result<u64> {
        self.stream_dataset(dataset, dest, CORPUS_STALL_TIMEOUT)
    }

    /// Not an artifact store and not under the quota: the server serves a
    /// corpus `Cache-Control: no-store`, so it lives for one job.
    fn datasets_dir(&self) -> PathBuf {
        self.root().join("datasets")
    }

    fn eval_completions(&self) -> EvalCompletionsStore {
        EvalCompletionsStore::new(self.root().join("state/evals"))
    }
}

/// `<work_dir>/.pipette` without opening it.
pub fn storage_root(work_dir: &Path) -> PathBuf {
    pipette_workspace::storage_root(work_dir, MARKER_NAME)
}

/// [`pipette_workspace::require_workspace`] bound to this crate's `.pipette` marker.
pub fn require_workspace(work_dir: &Path, work_dir_arg: Option<&Path>) -> anyhow::Result<()> {
    pipette_workspace::require_workspace(work_dir, work_dir_arg, MARKER_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pipette-cli-ws-{}-{label}", std::process::id()))
    }

    /// The layout the module doc promises.
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
            "state/worker",
        ];
        expected.into_iter().try_for_each(|sub| {
            anyhow::ensure!(root.join(sub).is_dir(), "missing dir: {sub}");
            anyhow::Ok(())
        })?;
        anyhow::ensure!(root.join("manifest.toml").exists(), "missing manifest");
        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }

    /// `eval_completions` and `init` name the subdirectory twice. Diverged,
    /// nothing else fails: the store creates its root lazily, orphaning every
    /// existing checkpoint.
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
    fn artifacts_carry_vault_access_only_once_registered() -> anyhow::Result<()> {
        let temp = crate::test_support::TempWorkspace::new("vault-access")?;
        let http = HttpClient::new("pipette-test")?;
        let programs = ExternalPrograms::at_missing_paths();

        assert!(temp.ws.artifacts(&http, &programs).vault.is_none());

        temp.register("https://mgmt.example")?;
        let access = temp
            .ws
            .artifacts(&http, &programs)
            .vault
            .ok_or_else(|| anyhow::anyhow!("a registered workspace should fetch as itself"))?;
        assert_eq!(access.server.to_string(), "https://mgmt.example");
        assert_eq!(access.identity.client_id.as_ref(), "ev1_test");
        Ok(())
    }

    /// A corpus lives beside the stores, outside the quota-swept trees.
    #[test]
    fn datasets_are_unpacked_beside_the_stores() -> anyhow::Result<()> {
        let temp = crate::test_support::TempWorkspace::new("datasets-dir")?;
        assert_eq!(temp.ws.datasets_dir(), temp.ws.root().join("datasets"));
        Ok(())
    }

    const DATASET_ROUTE: &str = "/datasets/vl_spec_decode_corpus_test_gqa_k8";

    fn dataset(url: Option<&str>) -> anyhow::Result<Dataset> {
        Ok(serde_json::from_value(serde_json::json!({
            "format": "vl-spec-decode-corpus/1",
            "adapter": "mmspec@1",
            "digest": "11".repeat(32),
            "bytes": 9,
            "upstream": {
                "source": "git",
                "url": "github.com/acme/corpus",
                "revision": "01".repeat(20),
                "identity": "29".repeat(20),
                "digest": "22".repeat(32)
            },
            "url": url,
        }))?)
    }

    fn fetch_error(ws: &PipetteWorkspace, dataset: &Dataset) -> String {
        ws.fetch_dataset(dataset)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
    }

    #[test]
    fn a_dataset_is_fetched_from_its_route() -> anyhow::Result<()> {
        let server = httpmock::MockServer::start();
        let route = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path(DATASET_ROUTE);
            then.status(200)
                .header("content-type", "application/x-tar")
                .body("tar bytes");
        });
        let temp = crate::test_support::TempWorkspace::new("fetch-dataset")?;
        temp.register(&server.base_url())?;

        let bytes = temp.ws.fetch_dataset(&dataset(Some(DATASET_ROUTE))?)?;
        route.assert();
        assert_eq!(bytes, b"tar bytes");
        Ok(())
    }

    /// Streamed, the same route hands the same bytes to the writer, and a
    /// refusal still carries the server's message.
    #[test]
    fn a_dataset_is_streamed_from_its_route() -> anyhow::Result<()> {
        let server = httpmock::MockServer::start();
        let route = server.mock(|when, then| {
            when.method(httpmock::Method::GET).path(DATASET_ROUTE);
            then.status(200)
                .header("content-type", "application/x-tar")
                .body("tar bytes");
        });
        let temp = crate::test_support::TempWorkspace::new("stream-dataset")?;
        temp.register(&server.base_url())?;

        let mut written = Vec::new();
        let count = temp
            .ws
            .fetch_dataset_to(&dataset(Some(DATASET_ROUTE))?, &mut written)?;
        route.assert();
        assert_eq!(written, b"tar bytes");
        assert_eq!(count, 9);

        let refusing = httpmock::MockServer::start();
        refusing.mock(|when, then| {
            when.method(httpmock::Method::GET).path(DATASET_ROUTE);
            then.status(503).body("{\"error\": \"dataset not built\"}");
        });
        let refused = crate::test_support::TempWorkspace::new("stream-refused")?;
        refused.register(&refusing.base_url())?;
        let error = refused
            .ws
            .fetch_dataset_to(&dataset(Some(DATASET_ROUTE))?, &mut Vec::new())
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        assert!(error.contains("dataset not built"), "{error}");
        Ok(())
    }

    /// A server that answers and then sends nothing more fails the streamed
    /// download once the stall bound passes, well before the server gives up.
    #[test]
    fn a_stalled_corpus_download_fails_at_the_stall_bound() -> anyhow::Result<()> {
        use std::io::{Read, Write};

        const SERVER_HOLDS: std::time::Duration = std::time::Duration::from_secs(10);
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let base = format!("http://{}", listener.local_addr()?);
        let server = std::thread::spawn(move || -> std::io::Result<()> {
            let (mut socket, _) = listener.accept()?;
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request)?;
            socket.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/x-tar\r\ncontent-length: 9\r\n\r\ntar",
            )?;
            socket.flush()?;
            std::thread::sleep(SERVER_HOLDS);
            Ok(())
        });
        let temp = crate::test_support::TempWorkspace::new("stream-stalled")?;
        temp.register(&base)?;

        let started = std::time::Instant::now();
        let mut written = Vec::new();
        let error = temp
            .ws
            .stream_dataset(
                &dataset(Some(DATASET_ROUTE))?,
                &mut written,
                std::time::Duration::from_millis(500),
            )
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default();
        let waited = started.elapsed();
        assert!(error.contains("broke off"), "{error}");
        assert!(
            waited < SERVER_HOLDS / 2,
            "the download waited {waited:?}, as long as the server held the socket"
        );
        assert_eq!(written, b"tar");
        drop(server);
        Ok(())
    }

    /// The upstream-gated refusals carry the server's message through, so an
    /// operator reading a failed job learns the corpus is why.
    #[rstest::rstest]
    #[case(503, "dataset not built")]
    #[case(503, "upstream check stale")]
    #[case(502, "upstream no longer backs the pin")]
    fn a_refusal_names_itself(#[case] status: u16, #[case] body: &str) -> anyhow::Result<()> {
        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::GET).path(DATASET_ROUTE);
            then.status(status)
                .body(format!("{{\"error\": \"{body}\"}}"));
        });
        let temp = crate::test_support::TempWorkspace::new("fetch-refused")?;
        temp.register(&server.base_url())?;

        let error = fetch_error(&temp.ws, &dataset(Some(DATASET_ROUTE))?);
        assert!(error.contains(body), "{error}");
        Ok(())
    }

    /// Refused before any request; the route is checked first.
    #[rstest::rstest]
    #[case::no_route(None, "no download URL")]
    #[case::unregistered(Some(DATASET_ROUTE), "registered")]
    fn an_unfetchable_dataset_is_refused(
        #[case] url: Option<&str>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let temp = crate::test_support::TempWorkspace::new("fetch-unfetchable")?;
        let error = fetch_error(&temp.ws, &dataset(url)?);
        assert!(error.contains(expected), "{error}");
        Ok(())
    }

    /// Two runtimes never share one.
    #[test]
    fn the_compile_cache_is_keyed_by_runtime() -> anyhow::Result<()> {
        let temp = crate::test_support::TempWorkspace::new("compile-cache")?;
        // Inline rather than a catalog ref, so a catalog bump cannot break it.
        let onnx_genai = |ep: &str| -> anyhow::Result<pipette_plan_types::Runtime> {
            Ok(serde_json::from_value(serde_json::json!({
                "type": "uv_onnx_genai", "server_version": "0.16.0",
                "execution_provider": ep, "python_version": "3.11",
                "source": {
                    "type": "pip_requirements_text",
                    "contents": "onnxruntime-genai==0.16.0\n"
                }
            }))?)
        };

        let cpu = temp.ws.compile_cache(&onnx_genai("cpu")?)?;
        let cuda = temp.ws.compile_cache(&onnx_genai("cuda")?)?;

        assert!(
            cpu.starts_with(temp.ws.root().join("cache")),
            "{}",
            cpu.display()
        );
        assert_ne!(cpu, cuda);
        assert_eq!(cpu, temp.ws.compile_cache(&onnx_genai("cpu")?)?);
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
