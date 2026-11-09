//! The doubles and fixtures the crate's unit tests share, kept here so the
//! production modules hold production code alone.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pipette_mgmt_client::{PrivateKey, ServerUrl};
use pipette_ops::EvalShard;
use pipette_plan_types::{ClientId, GgufText, GgufTextSource, Model};

use crate::console::Console;
use crate::identity::IdentityRegistration;
use crate::workspace::PipetteWorkspace;

/// The OpenVINO/OVMS gate on this host: `Ok` on Linux and Windows, else a
/// `ForeignOsError`, returned for further checks.
pub(crate) fn expect_gate_on_this_host<T>(
    result: anyhow::Result<T>,
) -> anyhow::Result<Option<anyhow::Error>> {
    match (result, matches!(std::env::consts::OS, "linux" | "windows")) {
        (Ok(_), true) => Ok(None),
        (Err(err), false) if err.downcast_ref::<crate::run::ForeignOsError>().is_some() => {
            Ok(Some(err))
        }
        (Ok(_), false) => anyhow::bail!("admitted on {}", std::env::consts::OS),
        (Err(err), _) => anyhow::bail!("unexpected refusal: {err:#}"),
    }
}

/// An initialized `PipetteWorkspace` in a temp dir, removed on drop so a
/// panicking test doesn't leak it.
pub(crate) struct TempWorkspace {
    pub ws: PipetteWorkspace,
    path: PathBuf,
}

impl TempWorkspace {
    pub(crate) fn new(label: &str) -> anyhow::Result<Self> {
        // Parallel tests share the pid and may share a label.
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("pipette-cli-{label}-{}-{seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        PipetteWorkspace::init(&path)?;
        let ws = PipetteWorkspace::open(&path, None)?;
        Ok(Self { ws, path })
    }

    /// The same workspace under an explicit cap, through the real `open`: the
    /// storage tests derive the quota from what they stored.
    pub(crate) fn reopen_with_quota(&self, quota_bytes: u64) -> anyhow::Result<PipetteWorkspace> {
        PipetteWorkspace::open(&self.path, Some(quota_bytes))
    }

    /// The same workspace under `--eval-shard <shard>`.
    pub(crate) fn as_shard(&self, shard: &str) -> anyhow::Result<PipetteWorkspace> {
        let shard = shard.parse::<EvalShard>().map_err(anyhow::Error::msg)?;
        Ok(PipetteWorkspace::open(&self.path, None)?.with_eval_shard(Some(shard)))
    }

    /// A gguf-text model published through the real store with a stub
    /// fetch, so its manifest reads back as a live entry.
    pub(crate) fn store_gguf_model(&self, uri: &str) -> anyhow::Result<Model> {
        let declared = crate::model_uri::parse_model_uri(uri)?;
        self.store_gguf_text(&declared)?;
        Ok(declared)
    }

    pub(crate) fn store_gguf_text(&self, declared: &Model) -> anyhow::Result<()> {
        self.ws.models().ensure(declared, |_declared, into| {
            let Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path },
            }) = into
            else {
                anyhow::bail!("the stub fetch only handles gguf-text");
            };
            let path = Path::new(path.as_ref());
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, vec![0u8; 4096])?;
            Ok(())
        })?;
        Ok(())
    }

    /// Any 32 bytes of hex is a usable Ed25519 seed; no test request is
    /// verified.
    pub(crate) fn register(&self, server_url: &str) -> anyhow::Result<()> {
        let registration = self.ws.registration();
        registration.put(&IdentityRegistration {
            client_id: ClientId::try_new("ev1_test")?,
            server_url: ServerUrl::try_new(server_url)?,
        })?;
        registration.put_private_key(&PrivateKey::try_new("11".repeat(32))?)?;
        Ok(())
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A [`Console`] over two buffers, read back as text.
#[derive(Default)]
pub(crate) struct CapturedConsole {
    out: Vec<u8>,
    err: Vec<u8>,
}

impl CapturedConsole {
    pub(crate) fn out_text(&self) -> String {
        String::from_utf8_lossy(&self.out).into_owned()
    }

    pub(crate) fn err_text(&self) -> String {
        String::from_utf8_lossy(&self.err).into_owned()
    }
}

impl Console for CapturedConsole {
    fn out(&mut self) -> &mut dyn Write {
        &mut self.out
    }

    fn err(&mut self) -> &mut dyn Write {
        &mut self.err
    }
}

/// A console whose `out` is a closed pipe and whose `err` still lands.
#[derive(Default)]
pub(crate) struct BrokenPipe {
    out: ClosedPipe,
    err: Vec<u8>,
}

impl Console for BrokenPipe {
    fn out(&mut self) -> &mut dyn Write {
        &mut self.out
    }

    fn err(&mut self) -> &mut dyn Write {
        &mut self.err
    }
}

#[derive(Default)]
struct ClosedPipe;

impl Write for ClosedPipe {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "the reader went away",
        ))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// What the log subscriber writes, for reading an event back.
#[derive(Clone, Default)]
pub(crate) struct Written(Arc<Mutex<Vec<u8>>>);

impl Written {
    pub(crate) fn text(&self) -> anyhow::Result<String> {
        let bytes = self.0.lock().map_err(|e| anyhow::anyhow!("{e}"))?.clone();
        Ok(String::from_utf8(bytes)?)
    }
}

impl Write for Written {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|e| std::io::Error::other(e.to_string()))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` under the text lines `pipette` logs off a terminal, and return them
/// with its result.
pub(crate) fn logged_text<R>(f: impl FnOnce() -> R) -> anyhow::Result<(R, String)> {
    let written = Written::default();
    let writer = written.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let result = tracing::subscriber::with_default(subscriber, f);
    Ok((result, written.text()?))
}

/// The wire document the server serves for a corpus benchmark, pinned to the
/// MMSpec GQA subset.
pub(crate) fn corpus_wire(benchmark_id: &str) -> serde_json::Value {
    serde_json::json!({
        "benchmark_id": benchmark_id,
        "benchmark_type": "vl_spec_decode",
        "parameter_draft_length": 8,
        "parameter_max_tokens": 2048,
        "parameter_batch_size": 8,
        "dataset": {
            "format": "vl-spec-decode-corpus/1",
            "adapter": "mmspec@1",
            "digest": "1862eedc832710f6c4776472d00d5ee4aa0a16d85d6a9c433398109f90239810",
            "bytes": 15955456,
            "subset": "GQA",
            "upstream": {
                "source": "git",
                "url": "github.com/killthefullmoon/MMSpec",
                "revision": "290486861eabd746075ca776adb66175636c1fc0",
                "subpath": "dataset/MMSpec/test",
                "identity": "2923603292c7dd463ad01a119f57965817778fc3",
                "digest": "2bd3c7b41707c10d91726e90545b9ec5d741ac947637c90e2f9d5fe6321fa8f7",
                "license": "Apache-2.0"
            },
            "url": format!("/datasets/{benchmark_id}")
        }
    })
}

/// An 80-image VL benchmark, whose image count a pre-`detail/` cache lost.
pub(crate) const IMG80: &str = "vl_throughput_512x512_text1024_decode64_img80";

pub(crate) fn img80_wire() -> serde_json::Value {
    serde_json::json!({
        "benchmark_id": IMG80,
        "benchmark_type": "vl_throughput",
        "parameter_image_width": 512,
        "parameter_image_height": 512,
        "parameter_text_tokens": 1024,
        "parameter_decode_tokens": 64,
        "parameter_num_images": 80,
    })
}

/// What public 2026.08.1 cached for [`IMG80`]: no `parameter_num_images`, so it
/// reads back as 1 image.
pub(crate) fn write_public_2026_08_1_img80_cache(
    benchmarks_root: &Path,
) -> anyhow::Result<PathBuf> {
    let dir = benchmarks_root.join("remote");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{IMG80}.json"));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&serde_json::json!({
            "benchmark_type": "vl_throughput",
            "benchmark_id": IMG80,
            "parameter_image_width": 512,
            "parameter_image_height": 512,
            "parameter_text_tokens": 1024,
            "parameter_decode_tokens": 64,
        }))?,
    )?;
    Ok(path)
}

pub(crate) fn cache_remote(
    store: &crate::benchmarks::BenchmarkStore,
    body: serde_json::Value,
) -> anyhow::Result<()> {
    cache_remote_with_etag(store, body, None)
}

/// Caches `body` as a pull or a claim would. `etag` is as sent, quotes included.
pub(crate) fn cache_remote_with_etag(
    store: &crate::benchmarks::BenchmarkStore,
    body: serde_json::Value,
    etag: Option<&str>,
) -> anyhow::Result<()> {
    let sent: pipette_mgmt_client::types::RemoteBenchmark = serde_json::from_value(body)?;
    let etag = etag
        .map(pipette_mgmt_client::EntityTag::try_new)
        .transpose()?;
    store.put_remote(
        &crate::benchmarks::ConvertedRemote::try_from(sent)?,
        etag.as_ref(),
    )?;
    Ok(())
}

pub(crate) fn remote_detail_path(benchmarks_root: &Path, benchmark_id: &str) -> PathBuf {
    benchmarks_root.join(format!("remote/detail/{benchmark_id}.json"))
}

/// `name` is `index` or `sync`.
pub(crate) fn remote_state_path(benchmarks_root: &Path, name: &str) -> PathBuf {
    benchmarks_root.join(format!("remote/state/{name}.json"))
}

/// The list and sync state an older build keeps directly under `remote/`.
pub(crate) struct OlderBuildBookkeeping {
    pub(crate) index: PathBuf,
    pub(crate) sync: PathBuf,
}

/// What an older internal build keeps after a pull from `server` that listed `ids`,
/// with an ETag for the list and for each id.
pub(crate) fn write_older_build_bookkeeping(
    benchmarks_root: &Path,
    server: &str,
    ids: &[&str],
) -> anyhow::Result<OlderBuildBookkeeping> {
    let dir = benchmarks_root.join("remote");
    std::fs::create_dir_all(&dir)?;
    let rows: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "benchmark_id": id,
                "benchmark_type": "prefill_throughput",
                "parameter_prefill_tokens": 128,
            })
        })
        .collect();
    let etags: serde_json::Map<String, serde_json::Value> = ids
        .iter()
        .map(|id| {
            (
                (*id).to_owned(),
                serde_json::json!(format!("\"older-{id}\"")),
            )
        })
        .collect();
    let bookkeeping = OlderBuildBookkeeping {
        index: dir.join("index.json"),
        sync: dir.join("sync.json"),
    };
    std::fs::write(&bookkeeping.index, serde_json::to_string(&rows)?)?;
    std::fs::write(
        &bookkeeping.sync,
        serde_json::json!({
            "pulled_at": "2026-10-01T00:00:00Z",
            "benchmark_count": ids.len(),
            "server_url": server,
            "benchmarks_etag": "\"older-list\"",
            "benchmark_etags": etags,
        })
        .to_string(),
    )?;
    Ok(bookkeeping)
}
