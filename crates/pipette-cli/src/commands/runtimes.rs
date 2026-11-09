//! `runtimes`: the local runtime store. `pull` installs every pullable kind
//! through [`pipette_artifacts::ensure_runtime`]. It places bytes on this host
//! without demanding that this host can run them, so there is no GPU preflight.

use anyhow::Context;
use clap::{Args, Subcommand, ValueEnum};
use tabled::Tabled;

use pipette_artifacts::{ensure_runtime, runtime_download_size};
use pipette_artifacts_catalog::runtime::{llamacpp, mlx, onnx_genai, openvino, ovms, torch_oai};
use pipette_http::HttpClient;
use pipette_ops::custom_label::descriptor_sha256;
use pipette_plan_types::{LlamaCppFlavor, Runtime, VllmFlavor, short_digest};
use pipette_programs::ExternalPrograms;

use crate::artifact_ref::resolve_runtime_arg;
use crate::commands::write_table_or;
use crate::console::Console;
use crate::progress::CellProgress;
use crate::runtime_uri::runtime_to_uri;
use crate::workspace::PipetteWorkspace;

/// Manage installed runtime builds
#[derive(Args, Debug)]
pub struct RuntimeArgs {
    #[command(subcommand)]
    pub command: RuntimeCommand,
}

#[derive(Subcommand, Debug)]
pub enum RuntimeCommand {
    /// List installed runtime builds across all runtimes
    List(ListArgs),
    /// List installable runtime builds for a given runtime type
    Catalog(CatalogArgs),
    /// List the llama.cpp build flavors `--flavor` accepts, with the upstream
    /// asset each resolves to
    Flavors,
    /// Install a runtime build into the local store from its declared source
    Pull(PullArgs),
    /// Remove an installed runtime's local record
    Remove(RemoveArgs),
}

#[derive(Args, Debug)]
pub struct ListArgs {
    /// How to render the runtime column
    #[arg(long, value_enum, default_value_t)]
    format: ListFormat,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum ListFormat {
    /// Human identity (default).
    #[default]
    Name,
    /// The importable runtime URI that round-trips through `runtimes pull`.
    Uri,
}

#[derive(Args, Debug)]
pub struct CatalogArgs {
    #[command(subcommand)]
    pub command: CatalogCommand,
}

#[derive(Subcommand, Debug)]
pub enum CatalogCommand {
    /// Bundled vLLM venv builds
    #[command(name = "uv_vllm")]
    UvVllm,
    /// Bundled SGLang venv builds
    #[command(name = "uv_sglang")]
    UvSglang,
    /// Bundled MLX venv builds (macOS)
    #[command(name = "mlx_macos_pipette")]
    MlxMacosPipette,
    /// Bundled OpenVINO venv builds (Intel CPU / GPU / NPU)
    #[command(name = "uv_openvino")]
    UvOpenvino,
    /// Bundled ONNX Runtime GenAI venv builds (one per execution provider)
    #[command(name = "uv_onnx_genai")]
    UvOnnxGenai,
    /// Upstream llama.cpp CLI releases (fetched from GitHub)
    #[command(name = "llamacpp_cli_stock_tools")]
    LlamacppCliStockTools(LlamacppCatalogArgs),
    /// OpenVINO Model Server releases whose archives the bundled catalog pins
    #[command(name = "ovms")]
    Ovms,
}

#[derive(Args, Debug)]
pub struct LlamacppCatalogArgs {
    /// Runtime flavor for the emitted ref + asset-availability filter (e.g.
    /// macos-arm64, linux-x64-cpu). `runtimes flavors` lists the vocabulary
    /// and the upstream asset each name resolves to.
    // Validated against the known set: `LlamaCppFlavor::parse` accepts anything
    // as a `Custom` build, which names no upstream release asset and so lists
    // nothing.
    #[arg(long, value_parser = parse_known_flavor)]
    flavor: LlamaCppFlavor,
    /// Max releases to fetch from GitHub
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(Args, Debug)]
pub struct RemoveArgs {
    /// Runtime URI or JSON `Runtime` object — the same grammar as `pull`,
    /// whose `--help` carries the scheme table and examples. A
    /// `runtime://sha256=<prefix>` digest from `runtimes list` also works
    #[arg(long)]
    runtime: String,
}

#[derive(Args, Debug)]
#[command(after_long_help = RUNTIME_URI_EXAMPLES)]
pub struct PullArgs {
    /// Runtime URI (`llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64`,
    /// `docker-vllm://image=…&tag=…`, …) or a JSON `Runtime` object.
    /// Run `pipette runtimes pull --help` for the full notation + examples.
    #[arg(long)]
    runtime: String,
}

/// `--runtime` notations, one example per scheme, shown under `--help`. A URI is
/// `<scheme>://<key>=<value>[&<key>=<value>…]`; a URL value may not contain `?`.
/// An arg starting with `{` is a JSON `Runtime` object.
const RUNTIME_URI_EXAMPLES: &str = "\
PULLABLE SCHEMES (runtimes pull) -- a scheme is the runtime type with `-` for `_`:
  llamacpp-cli-stock-tools   version(+repo, asset) xor url; flavor   (repo defaults to github.com/ggml-org/llama.cpp; asset to the flavor's file)
  docker-vllm                image, tag; [flavor] (default nvidia_gpu)   -- pulled into the docker daemon
  docker-sglang              image, tag; [flavor] (default nvidia_gpu)   -- pulled into the docker daemon
  mlx-macos-pipette          version; [flavor] (default macos-arm64)     -- catalog: mlx_macos_pipette (macOS)
  core-ai-macos-pipette      version (default: bundled coreai-models pin)  -- the OS provides the engine; pipette builds the sidecar (macOS 27+)
  uv-vllm                    server, build, python                       -- catalog: uv_vllm (Linux)
  uv-sglang                  server, build, python                       -- catalog: uv_sglang (Linux)
  uv-openvino                version                                     -- catalog: uv_openvino (Linux/Windows)
  uv-onnx-genai              version, ep                                 -- catalog: uv_onnx_genai (one venv per provider)
  ovms                       version(+repo) xor url; flavor              -- catalog: ovms (repo defaults to github.com/openvinotoolkit/model_server)
  trt-edgellm                url; flavor                                 -- no catalog: a server archive by URL (Jetson)

EXAMPLES:
  # llama.cpp from an upstream release (repo defaults to ggml-org/llama.cpp)
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

  # llama.cpp from a named release file
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b11399&asset=llama-b11399-bin-win-cuda-13.4-x64.zip&flavor=windows-x64-cuda-13'

  # llama.cpp from a fork / explicit repo
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=linux-x64-cpu'

  # llama.cpp from a prebuilt archive URL
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://url=https://example.com/llama-b1.tar.gz&flavor=macos-arm64'

  # docker (vLLM / SGLang): pulled into the docker daemon
  pipette runtimes pull --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.10.0'
  pipette runtimes pull --runtime 'docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=amd_gpu'

  # MLX (macOS) / uv (Linux): catalog installs; list with runtimes catalog <type>
  pipette runtimes pull --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'
  pipette runtimes pull --runtime 'uv-vllm://server=0.32.0&build=cu129&python=3.12'
  pipette runtimes pull --runtime 'uv-sglang://server=0.5.23&build=cu130&python=3.12'

  # Core AI (macOS 27+): the OS provides the engine. The URI names the
  # bundled coreai-models pin. Pipette builds the Swift sidecar from source.
  pipette runtimes pull --runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'

  # OpenVINO (Linux/Windows) -- one venv per version; `device` is per cell, not per install
  pipette runtimes pull --runtime 'uv-openvino://version=2026.4.1'

  # ONNX Runtime GenAI -- one venv per version AND execution provider
  pipette runtimes pull --runtime 'uv-onnx-genai://version=0.17.1&ep=cpu'

  # OpenVINO Model Server -- an upstream release archive per OS; `device` is per cell
  pipette runtimes pull --runtime 'ovms://version=v2026.4.1&flavor=windows-x64'
  pipette runtimes pull --runtime 'ovms://version=v2026.4.1&flavor=ubuntu24-x64'

  # TensorRT Edge-LLM (Jetson) -- a server archive; engines build into the compile cache per cell
  pipette runtimes pull --runtime 'trt-edgellm://url=https://example.com/trt-edgellm-server-0.11.0-jetson-orin-cuda-13-4e3f9d23f052.tar.gz&flavor=jetson-orin-cuda-13'

  # or a JSON Runtime object
  pipette runtimes pull --runtime '{\"type\":\"docker_vllm\",\"image_name\":\"vllm\",\"image_tag\":\"v0.10.0\",\"flavor\":\"nvidia_gpu\"}'
";

impl RuntimeArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        self.command.execute(ws, http, programs, console)
    }
}

impl RuntimeCommand {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        match self {
            RuntimeCommand::List(args) => list(ws, args.format, console),
            RuntimeCommand::Catalog(args) => args.execute(http, console),
            RuntimeCommand::Flavors => write_llamacpp_flavors(console),
            RuntimeCommand::Pull(args) => args.execute(ws, http, programs, console),
            RuntimeCommand::Remove(args) => args.execute(ws, console),
        }
    }
}

impl PullArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        use anyhow::Context;

        let declared = resolve_runtime_arg(ws, &self.runtime)?;
        // No pin set: `ensure_runtime` pins the entry it is about to publish.
        // A uv or docker install spawns the program this process located, so an
        // absent one is reported to the pull that needed it.
        let ctx = ws.artifacts(http, programs);
        let store = ws.runtimes();

        // A llama.cpp archive streams bytes this process counts; a uv solve and a
        // docker pull report their own, so those draw nothing here.
        let progress =
            CellProgress::new(&[runtime_download_size(&store, &declared).unwrap_or(None)]);
        let _bound = ensure_runtime(&ctx.with_progress(progress.sink()), &store, &declared)
            .with_context(|| format!("installing runtime `{declared}`"))?;
        // Erased before the summary line, which would otherwise print under a
        // finished bar.
        drop(progress);

        let entry = store
            .find(&declared)?
            .ok_or_else(|| anyhow::anyhow!("runtime missing after ensure"))?;
        writeln!(
            console.out(),
            "Installed `{declared}` → {}",
            store.entry_dir_for(&entry)?.display()
        )?;
        Ok(())
    }
}

impl RemoveArgs {
    pub fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
        let declared = resolve_runtime_arg(ws, &self.runtime)?;
        let store = ws.runtimes();
        if store.remove(&declared)? {
            writeln!(console.out(), "Removed `{declared}`")?;
            // Same key as the store entry, so removing the runtime is what
            // makes its compiled artifacts unaddressable; leaving them would
            // strand bytes nothing can reach.
            let cache = ws.compile_cache(&declared)?;
            if cache.is_dir() {
                std::fs::remove_dir_all(&cache).with_context(|| {
                    format!("removing the compile cache at {}", cache.display())
                })?;
                writeln!(console.out(), "  reclaimed the compile cache")?;
            }
            if matches!(declared, Runtime::DockerVllm(_) | Runtime::DockerSglang(_)) {
                writeln!(
                    console.out(),
                    "  note: the docker image is still in the daemon; `docker rmi` to remove it"
                )?;
            }
            return Ok(());
        }
        writeln!(console.out(), "`{declared}` is not installed")?;
        Ok(())
    }
}

#[derive(Tabled)]
struct RuntimeRow {
    #[tabled(rename = "RUNTIME")]
    runtime: String,
    #[tabled(rename = "TYPE")]
    runtime_type: String,
    #[tabled(rename = "DIGEST")]
    digest: String,
    #[tabled(rename = "PULLED")]
    pulled: String,
}

fn list(
    ws: &PipetteWorkspace,
    format: ListFormat,
    console: &mut dyn Console,
) -> anyhow::Result<()> {
    write_table_or(
        console.out(),
        &runtime_rows(ws, format)?,
        "No runtimes installed.",
    )?;
    Ok(())
}

/// The installed runtimes as `list` rows, sorted by the rendered runtime column.
fn runtime_rows(ws: &PipetteWorkspace, format: ListFormat) -> anyhow::Result<Vec<RuntimeRow>> {
    let store = ws.runtimes();
    let stored = store
        .list()?
        .into_iter()
        .map(|m| Ok((m.declared.clone(), m.fetched_at_rfc3339()?)));
    let mut rows: Vec<RuntimeRow> = stored
        .map(|entry: anyhow::Result<(Runtime, String)>| {
            let (declared, pulled) = entry?;
            Ok(RuntimeRow {
                // A runtime with no URI form (a vault archive, say) falls back
                // to its identity rather than dropping the row.
                runtime: match format {
                    ListFormat::Name => declared.cli_ref(),
                    ListFormat::Uri => {
                        runtime_to_uri(&declared).unwrap_or_else(|_| declared.cli_ref())
                    }
                },
                runtime_type: declared.headless_token().to_owned(),
                // The prefix `--runtime runtime://sha256=<hex>` takes, and the
                // same id the warehouse stores as `runtime_descriptor_sha256`.
                digest: short_digest(&descriptor_sha256(&declared)?).to_owned(),
                pulled,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    rows.sort_by(|a, b| a.runtime.cmp(&b.runtime));
    Ok(rows)
}

fn vllm_flavor_label(flavor: VllmFlavor) -> &'static str {
    match flavor {
        VllmFlavor::NvidiaGpu => "nvidia_gpu",
        VllmFlavor::AmdGpu => "amd_gpu",
        VllmFlavor::Cpu => "cpu",
    }
}

/// A catalog row whose `REF` is pasted verbatim into `--runtime`, carrying its
/// hardware `FLAVOR`. Used by every subword but llama.cpp's.
#[derive(Tabled)]
struct CatalogRefRow {
    #[tabled(rename = "REF")]
    runtime_ref: String,
    #[tabled(rename = "FLAVOR")]
    flavor: String,
}

#[derive(Tabled)]
struct LlamacppRefRow {
    #[tabled(rename = "REF")]
    runtime_ref: String,
    #[tabled(rename = "PUBLISHED")]
    published: String,
}

/// One llama.cpp flavor with its upstream asset stem: flavor names are
/// pipette's, asset names are upstream's, and they disagree often enough
/// (`linux-x64-openvino` ships as `ubuntu-openvino-…`) that seeing both is the
/// point.
#[derive(Tabled)]
struct LlamacppFlavorRow {
    #[tabled(rename = "FLAVOR")]
    flavor: String,
    #[tabled(rename = "UPSTREAM ASSET")]
    asset: String,
}

/// Bundled uv vLLM builds as paste-ready `uv-vllm://…` refs, split from the
/// printing so a test can assert every emitted ref parses.
fn uv_vllm_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    uv_bundled_rows(torch_oai::CatalogEntry::is_vllm)
}

/// Bundled uv SGLang builds as paste-ready `uv-sglang://…` refs.
fn uv_sglang_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    uv_bundled_rows(torch_oai::CatalogEntry::is_sglang)
}

/// The bundled torch-oai catalog entries `keep` selects, as paste-ready refs.
/// A malformed slug is skipped rather than aborting the listing; the catalog's
/// own tests validate the table.
fn uv_bundled_rows(
    keep: fn(&torch_oai::CatalogEntry) -> bool,
) -> anyhow::Result<Vec<CatalogRefRow>> {
    use torch_oai::CatalogEntry;

    torch_oai::slugs()?
        .filter_map(|slug_body| torch_oai::slug::UvSlug::try_new(slug_body).ok())
        .try_fold(Vec::new(), |mut rows, slug| {
            if let Some(entry) = torch_oai::lookup(&slug)?.filter(keep) {
                let runtime_ref = match &entry {
                    CatalogEntry::UvVllm {
                        server_version,
                        build,
                        python_version,
                        ..
                    } => format!(
                        "uv-vllm://server={server_version}&build={build}&python={python_version}"
                    ),
                    CatalogEntry::UvSglang {
                        server_version,
                        build,
                        python_version,
                        ..
                    } => format!(
                        "uv-sglang://server={server_version}&build={build}&python={python_version}"
                    ),
                };
                rows.push(CatalogRefRow {
                    runtime_ref,
                    flavor: vllm_flavor_label(entry.flavor()).to_string(),
                });
            }
            Ok(rows)
        })
}

/// Bundled OpenVINO builds as paste-ready `uv-openvino://…` refs, one row per
/// version: one venv serves every device, and the device is a flag on the cell.
fn openvino_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(openvino::catalog_entries()?
        .into_iter()
        .map(|version| CatalogRefRow {
            runtime_ref: format!("uv-openvino://version={version}"),
            flavor: "any device".to_owned(),
        })
        .collect())
}

/// Bundled ONNX Runtime GenAI builds as paste-ready `uv-onnx-genai://…` refs,
/// one row per version and execution provider: each provider is a separate
/// wheel, so a separate venv.
fn onnx_genai_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(onnx_genai::catalog_entries()?
        .into_iter()
        .map(|(version, ep)| CatalogRefRow {
            runtime_ref: format!("uv-onnx-genai://version={version}&ep={ep}"),
            flavor: ep.to_owned(),
        })
        .collect())
}

/// Pinned OVMS releases as paste-ready `ovms://…` refs, one row per release
/// tag and flavor.
fn ovms_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(ovms::rows()?
        .iter()
        .map(|row| CatalogRefRow {
            runtime_ref: format!("ovms://version={}&flavor={}", row.tag, row.flavor),
            flavor: row.flavor.to_string(),
        })
        .collect())
}

/// Bundled MLX builds as paste-ready `mlx-macos-pipette://…` refs. Listable
/// anywhere: only executing one needs a Mac.
fn mlx_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(mlx::catalog_entries()?
        .into_iter()
        .map(|(version, flavor)| CatalogRefRow {
            runtime_ref: format!("mlx-macos-pipette://version={version}"),
            flavor: flavor.to_string(),
        })
        .collect())
}

/// Upstream llama.cpp CLI releases carrying an asset for `flavor`, as
/// paste-ready `llamacpp-cli-stock-tools://…` refs. Hits GitHub. The caller has
/// already rejected `Custom`, so the canonical `as_str` goes in the ref.
fn llamacpp_rows(
    http: &HttpClient,
    flavor: &LlamaCppFlavor,
    limit: usize,
) -> anyhow::Result<Vec<LlamacppRefRow>> {
    let releases = llamacpp::github_releases(http, limit)?;
    let rows = releases
        .into_iter()
        .filter(|release| llamacpp::release_asset_available(release, flavor))
        .map(|release| LlamacppRefRow {
            runtime_ref: format!(
                "llamacpp-cli-stock-tools://version={}&flavor={}",
                release.tag_name,
                flavor.as_str()
            ),
            published: release.published_at.unwrap_or_default(),
        })
        .collect();
    Ok(rows)
}

/// Holds `--flavor` to the known set, so clap rejects a typo by name. Validates
/// through `parse`, so the vocabulary is defined in one place, and points at
/// `runtimes flavors`, which also shows each name's upstream asset.
fn parse_known_flavor(value: &str) -> Result<LlamaCppFlavor, String> {
    let flavor = LlamaCppFlavor::parse(value);
    if flavor.is_custom() {
        return Err(
            "not a flavor pipette tracks: a custom flavor names no upstream release \
             asset, so nothing could be listed for it. `pipette runtimes flavors` \
             lists the known set"
                .to_owned(),
        );
    }
    Ok(flavor)
}

/// `runtimes flavors`: the `--flavor` vocabulary. Needs neither workspace nor
/// network, so a fresh box can read it before it picks what to pull.
pub fn write_llamacpp_flavors(console: &mut dyn Console) -> anyhow::Result<()> {
    write_table_or(
        console.out(),
        &llamacpp_flavor_rows(),
        "No llama.cpp flavors.",
    )?;
    Ok(())
}

/// Every flavor pipette tracks, with the upstream asset stem it resolves to.
/// Offline: this is the enum, not the release feed.
fn llamacpp_flavor_rows() -> Vec<LlamacppFlavorRow> {
    LlamaCppFlavor::known()
        .map(|flavor| LlamacppFlavorRow {
            flavor: flavor.to_string(),
            // Every known flavor names an asset (pinned by
            // `listed_llamacpp_flavors_are_usable`); the placeholder stands in
            // for the version so the column shows the stable stem.
            asset: flavor
                .release_asset_name(VERSION_PLACEHOLDER)
                .unwrap_or_else(|| "(none)".to_string()),
        })
        .collect()
}

/// Stands in for the release tag in the flavor listing's asset column, which
/// describes the *shape* of the asset name rather than one release's.
const VERSION_PLACEHOLDER: &str = "<version>";

impl CatalogArgs {
    /// Lists what is installable, not what is installed (that is `list`).
    /// Offline except the llama.cpp subword, which hits GitHub.
    pub fn execute(self, http: &HttpClient, console: &mut dyn Console) -> anyhow::Result<()> {
        self.command.execute(http, console)
    }
}

impl CatalogCommand {
    pub fn execute(self, http: &HttpClient, console: &mut dyn Console) -> anyhow::Result<()> {
        match self {
            CatalogCommand::UvVllm => {
                write_table_or(
                    console.out(),
                    &uv_vllm_rows()?,
                    "No bundled uv-vLLM builds.",
                )?;
            }
            CatalogCommand::UvSglang => {
                write_table_or(
                    console.out(),
                    &uv_sglang_rows()?,
                    "No bundled uv-SGLang builds.",
                )?;
            }
            CatalogCommand::MlxMacosPipette => {
                write_table_or(console.out(), &mlx_rows()?, "No bundled MLX builds.")?;
            }
            CatalogCommand::UvOpenvino => {
                write_table_or(
                    console.out(),
                    &openvino_rows()?,
                    "No bundled OpenVINO builds.",
                )?;
            }
            CatalogCommand::UvOnnxGenai => {
                write_table_or(
                    console.out(),
                    &onnx_genai_rows()?,
                    "No bundled ONNX Runtime GenAI builds.",
                )?;
            }
            CatalogCommand::Ovms => {
                write_table_or(console.out(), &ovms_rows()?, "No pinned OVMS releases.")?;
            }
            CatalogCommand::LlamacppCliStockTools(args) => {
                write_table_or(
                    console.out(),
                    &llamacpp_rows(http, &args.flavor, args.limit)?,
                    &format!(
                        "None of the latest {} llama.cpp releases has a `{}` file. \
                         Raise --limit, or name the file with `asset`.",
                        args.limit, args.flavor
                    ),
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    /// Every ref the offline catalogs emit, from one builder so a new catalog
    /// reaches every test that walks the rows. The network-backed llamacpp
    /// subword is appended by the tests that want it.
    fn bundled_catalog_refs() -> anyhow::Result<Vec<String>> {
        let mut refs: Vec<String> = uv_vllm_rows()?
            .iter()
            .chain(uv_sglang_rows()?.iter())
            .map(|r| r.runtime_ref.clone())
            .collect();
        refs.extend(super::mlx_rows()?.iter().map(|r| r.runtime_ref.clone()));
        refs.extend(
            super::openvino_rows()?
                .iter()
                .map(|r| r.runtime_ref.clone()),
        );
        refs.extend(
            super::onnx_genai_rows()?
                .iter()
                .map(|r| r.runtime_ref.clone()),
        );
        refs.extend(super::ovms_rows()?.iter().map(|r| r.runtime_ref.clone()));
        Ok(refs)
    }

    /// A row `runtimes list --format uri` renders must be what `runtimes pull`
    /// accepts. Driven off the bundled catalogs so the cases stay valid.
    #[test]
    fn listed_uri_round_trips_through_pull() -> anyhow::Result<()> {
        let mut refs = bundled_catalog_refs()?;
        refs.push("llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64".to_string());

        for uri in refs {
            let declared = parse_runtime_arg(&uri)?;
            let rendered = crate::runtime_uri::runtime_to_uri(&declared)
                .map_err(|e| anyhow::anyhow!("{uri} has no URI form: {e}"))?;
            // Not byte-equality: the renderer spells out keys the catalog leaves
            // defaulted (`mlx-macos-pipette://…&flavor=macos-arm64`).
            assert_eq!(
                parse_runtime_arg(&rendered)?,
                declared,
                "{uri} rendered to {rendered}, which parses to something else"
            );
        }
        Ok(())
    }
    use strum::IntoEnumIterator;

    use super::{RUNTIME_URI_EXAMPLES, uv_sglang_rows, uv_vllm_rows};
    use crate::runtime_uri::{Scheme, parse_runtime_arg};

    #[test]
    fn bundled_catalog_refs_parse() -> anyhow::Result<()> {
        assert!(
            !uv_vllm_rows()?.is_empty(),
            "bundled uv-vLLM catalog must have entries"
        );
        assert!(
            !uv_sglang_rows()?.is_empty(),
            "bundled uv-SGLang catalog must have entries"
        );

        let refs = bundled_catalog_refs()?;

        for r in &refs {
            parse_runtime_arg(r)
                .map_err(|e| anyhow::anyhow!("catalog ref `{r}` should parse: {e}"))?;
        }
        Ok(())
    }

    /// A typo is a clap error rather than a run that lists nothing, and every
    /// flavor the listing offers is accepted: a validator stricter than the
    /// listing would refuse its own advice.
    #[test]
    fn flavor_validation_accepts_the_listing_and_rejects_the_rest() -> anyhow::Result<()> {
        super::llamacpp_flavor_rows().iter().try_for_each(
            |row| match super::parse_known_flavor(&row.flavor) {
                Ok(_) => Ok(()),
                Err(e) => Err(anyhow::anyhow!(
                    "listed flavor `{}` is rejected by --flavor: {e}",
                    row.flavor,
                )),
            },
        )?;

        let rejected = super::parse_known_flavor("linux-x64-cuda");
        assert!(rejected.is_err(), "an unknown flavor must not validate");
        // The message has to name where the set lives; clap prints it verbatim.
        assert!(
            rejected
                .err()
                .is_some_and(|e| e.contains("pipette runtimes flavors")),
            "the rejection must point at the listing",
        );
        Ok(())
    }

    /// Every listed flavor parses to a known variant (not `Custom`, which lists
    /// nothing), composes into a `--runtime` URI that parses, and shows an
    /// asset stem.
    #[test]
    fn listed_llamacpp_flavors_are_usable() -> anyhow::Result<()> {
        let rows = super::llamacpp_flavor_rows();
        assert!(!rows.is_empty(), "the flavor vocabulary must not be empty");

        rows.iter().try_for_each(|row| {
            let parsed = pipette_plan_types::LlamaCppFlavor::parse(&row.flavor);
            assert!(
                !parsed.is_custom(),
                "listed flavor `{}` parses as Custom, so it would list nothing",
                row.flavor,
            );
            assert!(
                row.asset.contains(super::VERSION_PLACEHOLDER),
                "flavor `{}` shows asset `{}`, which names no version slot",
                row.flavor,
                row.asset,
            );
            let uri = format!(
                "llamacpp-cli-stock-tools://version=b9305&flavor={}",
                row.flavor
            );
            parse_runtime_arg(&uri)
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("listed flavor `{}` breaks `{uri}`: {e}", row.flavor))
        })
    }

    /// Extracts the args from the help text itself, so the examples can't drift
    /// from the grammar.
    #[test]
    fn every_documented_example_parses() -> anyhow::Result<()> {
        let args = documented_examples();
        assert!(
            args.len() >= 10,
            "expected the examples block to yield the documented args, got {}",
            args.len()
        );
        args.iter().try_for_each(|arg| {
            parse_runtime_arg(arg)
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("documented example `{arg}` should parse: {e}"))
        })
    }

    /// Catches an omission rather than a typo: a scheme the grammar accepts but
    /// the help never shows is a runtime nobody can discover.
    #[test]
    fn every_scheme_is_documented_with_an_example() -> anyhow::Result<()> {
        let examples = documented_examples();
        // The scheme column of the PULLABLE SCHEMES block, which ends where the
        // examples begin.
        let listed: Vec<&str> = RUNTIME_URI_EXAMPLES
            .lines()
            .take_while(|line| !line.starts_with("EXAMPLES:"))
            .filter_map(|line| line.split_whitespace().next())
            .collect();

        Scheme::iter().try_for_each(|scheme| {
            let name = scheme.as_str();
            anyhow::ensure!(
                listed.contains(&name),
                "scheme `{name}` is missing from the PULLABLE SCHEMES list"
            );
            anyhow::ensure!(
                examples
                    .iter()
                    .any(|arg| arg.starts_with(&format!("{name}://"))),
                "scheme `{name}` has no `--runtime` example in the help"
            );
            Ok(())
        })
    }

    /// A scheme is its runtime's type word with `-` for `_`, checked against
    /// plan-types. Fails the day a shorthand is introduced.
    #[test]
    fn a_schemes_prefix_is_its_runtime_type() -> anyhow::Result<()> {
        documented_examples()
            .iter()
            .filter_map(|arg| arg.split_once("://"))
            .try_for_each(|(scheme, _)| {
                let runtime = parse_runtime_arg(&format!("{scheme}://{}", body_of(scheme)?))?;
                anyhow::ensure!(
                    runtime.headless_token().replace('_', "-") == scheme,
                    "scheme `{scheme}` names runtime type `{}`",
                    runtime.headless_token()
                );
                Ok(())
            })
    }

    /// The body of the first documented example for `scheme`.
    fn body_of(scheme: &str) -> anyhow::Result<&'static str> {
        documented_examples()
            .into_iter()
            .filter_map(|arg| arg.split_once("://"))
            .find(|(s, _)| *s == scheme)
            .map(|(_, body)| body)
            .ok_or_else(|| anyhow::anyhow!("no documented example for `{scheme}`"))
    }

    /// The `--runtime '<arg>'` values the help block shows.
    fn documented_examples() -> Vec<&'static str> {
        RUNTIME_URI_EXAMPLES
            .lines()
            .filter_map(|line| line.split_once("--runtime '"))
            .filter_map(|(_, rest)| rest.strip_suffix('\''))
            .collect()
    }

    use pipette_plan_types::{AppleCoreAiMacosPipette, Runtime};

    use super::{ListFormat, RemoveArgs, RuntimeArgs, RuntimeCommand, list, runtime_rows};
    use crate::test_support::CapturedConsole;
    use crate::test_support::TempWorkspace;
    use crate::workspace::PipetteWorkspace;

    const DOCKER_URI: &str = "docker-vllm://image=vllm/vllm-openai&tag=v0.10.0&flavor=nvidia_gpu";
    const LLAMA_URI: &str = "llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64";

    /// An installed runtime, published through the real store with a no-op
    /// install so its manifest reads back as a live entry.
    fn install(ws: &PipetteWorkspace, uri: &str) -> anyhow::Result<pipette_plan_types::Runtime> {
        let declared = parse_runtime_arg(uri)?;
        ws.runtimes().ensure(&declared, |_, _| Ok(()))?;
        Ok(declared)
    }

    fn programs() -> pipette_programs::ExternalPrograms {
        pipette_programs::ExternalPrograms::at_missing_paths()
    }

    fn http() -> anyhow::Result<pipette_http::HttpClient> {
        Ok(pipette_http::HttpClient::new("pipette-test")?)
    }

    /// Both formats sort by the rendered column, so the listing is stable across
    /// pulls.
    #[test]
    fn list_renders_each_installed_runtime_in_the_requested_format() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("runtimes-list")?;
        let docker = install(&tw.ws, DOCKER_URI)?;
        let llama = install(&tw.ws, LLAMA_URI)?;

        let by_name = runtime_rows(&tw.ws, ListFormat::Name)?;
        let mut expected = vec![
            (docker.cli_ref(), docker.headless_token()),
            (llama.cli_ref(), llama.headless_token()),
        ];
        expected.sort();
        assert_eq!(
            by_name
                .iter()
                .map(|r| (r.runtime.clone(), r.runtime_type.as_str()))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(by_name.iter().all(|row| row.pulled.ends_with('Z')));
        assert!(by_name.iter().all(|row| !row.digest.is_empty()));

        let by_uri = runtime_rows(&tw.ws, ListFormat::Uri)?;
        let mut uris = vec![DOCKER_URI.to_owned(), LLAMA_URI.to_owned()];
        uris.sort();
        assert_eq!(
            by_uri.iter().map(|r| r.runtime.clone()).collect::<Vec<_>>(),
            uris
        );
        Ok(())
    }

    #[test]
    fn an_empty_store_says_nothing_is_installed() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("runtimes-list-empty")?;
        assert!(runtime_rows(&tw.ws, ListFormat::Uri)?.is_empty());
        let mut captured = CapturedConsole::default();
        list(&tw.ws, ListFormat::Name, &mut captured)?;
        assert_eq!(captured.out_text(), "No runtimes installed.\n");
        Ok(())
    }

    /// Core AI is listed from its store manifest.
    #[test]
    fn a_stored_core_ai_sidecar_lists_the_bundled_runtime() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("runtimes-list-coreai")?;
        let bundled = Runtime::AppleCoreAiMacosPipette(AppleCoreAiMacosPipette::bundled());
        tw.ws.runtimes().ensure(&bundled, |_, _| Ok(()))?;
        let rows = runtime_rows(&tw.ws, ListFormat::Name)?;
        assert_eq!(
            rows.iter()
                .map(|r| (r.runtime.clone(), r.runtime_type.as_str()))
                .collect::<Vec<_>>(),
            vec![(bundled.cli_ref(), bundled.headless_token())]
        );
        assert!(rows.iter().all(|row| row.pulled.ends_with('Z')));
        Ok(())
    }

    /// The compile cache shares the store entry's key, so it goes with the
    /// runtime. A docker runtime also notes the image the daemon still holds.
    #[test]
    fn remove_takes_the_compile_cache_and_reports_a_missing_runtime() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("runtimes-remove")?;
        let docker = install(&tw.ws, DOCKER_URI)?;
        let cache = tw.ws.compile_cache(&docker)?;
        std::fs::create_dir_all(&cache)?;
        std::fs::write(cache.join("kernel.bin"), b"compiled")?;

        let mut removed = CapturedConsole::default();
        RuntimeArgs {
            command: RuntimeCommand::Remove(RemoveArgs {
                runtime: DOCKER_URI.to_owned(),
            }),
        }
        .execute(&tw.ws, &http()?, &programs(), &mut removed)?;

        assert!(tw.ws.runtimes().find(&docker)?.is_none());
        assert!(!cache.exists(), "the compile cache goes with the runtime");
        let out = removed.out_text();
        assert!(
            out.starts_with(&format!("Removed `{docker}`\n")),
            "got {out}"
        );
        assert!(out.contains("reclaimed the compile cache"), "got {out}");
        assert!(out.contains("docker rmi"), "got {out}");

        let mut again = CapturedConsole::default();
        RemoveArgs {
            runtime: DOCKER_URI.to_owned(),
        }
        .execute(&tw.ws, &mut again)?;
        assert_eq!(again.out_text(), format!("`{docker}` is not installed\n"));
        Ok(())
    }
}
