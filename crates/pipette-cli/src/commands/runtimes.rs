//! `runtimes` — runtime-build management. Pulled runtimes live under the
//! workspace `runtimes/` tree and `list` reads the shared artifact store
//! (llama.cpp / docker / UV / MLX), which is the only layout written or read.
//!
//! `pull` takes one self-contained `--runtime` reference (a URI or JSON
//! `Runtime`) and installs every pullable kind through
//! [`pipette_artifacts::ensure_runtime`] (shared store stage → fetch →
//! `manifest.toml` → publish). It places bytes on this host without
//! demanding that this host be able to run them, so there is no GPU
//! preflight here; a uv install still probes the torch it just built.

use anyhow::Context;
use clap::{Args, Subcommand, ValueEnum};
use tabled::Tabled;

use pipette_artifacts::{ensure_runtime, runtime_download_size};
use pipette_http::HttpClient;
use pipette_plan_types::{descriptor, LlamaCppFlavor, Runtime, VllmFlavor};

use crate::artifact_ref::resolve_runtime_arg;
use crate::commands::print_table_or;
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

/// `runtimes list` — how to render the runtime column.
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

/// `runtimes catalog` — one subword per installable runtime *type*; each row is
/// a paste-ready `--runtime` URI.
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
    /// Upstream llama.cpp CLI releases (fetched from GitHub)
    #[command(name = "llamacpp_cli_stock_tools")]
    LlamacppCliStockTools(LlamacppCatalogArgs),
}

#[derive(Args, Debug)]
pub struct LlamacppCatalogArgs {
    /// Runtime flavor for the emitted ref + asset-availability filter (e.g.
    /// macos-arm64, linux-x64-cpu). `runtimes flavors` lists the vocabulary
    /// and the upstream asset each name resolves to.
    // Validated against the known set: `LlamaCppFlavor::parse` would otherwise
    // accept anything as a `Custom` build, and a custom flavor names no
    // upstream release asset, so it could only ever list nothing.
    #[arg(long, value_parser = parse_known_flavor)]
    flavor: LlamaCppFlavor,
    /// Max releases to fetch from GitHub
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

/// `runtimes remove` input — a self-contained runtime reference.
#[derive(Args, Debug)]
pub struct RemoveArgs {
    /// Runtime URI or JSON `Runtime` object — the same grammar as `pull`,
    /// whose `--help` carries the scheme table and examples. A
    /// `runtime://sha256=<prefix>` digest from `runtimes list` also works
    #[arg(long)]
    runtime: String,
}

/// `runtimes pull` input — a self-contained runtime reference.
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
/// `<scheme>://<key>=<value>[&<key>=<value>…]`; `&` separates pairs and a URL
/// value may not contain `?`. Any runtime is also accepted as a JSON `Runtime`
/// object (an arg starting with `{`).
const RUNTIME_URI_EXAMPLES: &str = "\
PULLABLE SCHEMES (runtimes pull) -- a scheme is the runtime type with `-` for `_`:
  llamacpp-cli-stock-tools   version(+repo) xor url; flavor   (repo defaults to github.com/ggml-org/llama.cpp)
  docker-vllm                image, tag; [flavor] (default nvidia_gpu)   -- pulled into the docker daemon
  docker-sglang              image, tag; [flavor] (default nvidia_gpu)   -- pulled into the docker daemon
  mlx-macos-pipette          version; [flavor] (default macos-arm64)     -- catalog: mlx_macos_pipette (macOS)
  uv-vllm                    server, build, python                       -- catalog: uv_vllm (Linux)
  uv-sglang                  server, build, python                       -- catalog: uv_sglang (Linux)
  uv-openvino                version                                     -- catalog: uv_openvino (Linux/Windows)

EXAMPLES:
  # llama.cpp from an upstream release (repo defaults to ggml-org/llama.cpp)
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

  # llama.cpp from a fork / explicit repo
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=linux-x64-cpu'

  # llama.cpp from a prebuilt archive URL
  pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://url=https://example.com/llama-b1.tar.gz&flavor=macos-arm64'

  # docker (vLLM / SGLang): pulled into the docker daemon
  pipette runtimes pull --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.10.0'
  pipette runtimes pull --runtime 'docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=amd_gpu'

  # MLX (macOS) / uv (Linux): catalog installs; list with runtimes catalog <type>
  pipette runtimes pull --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'
  pipette runtimes pull --runtime 'uv-vllm://server=0.22.0&build=cu129&python=3.12'
  pipette runtimes pull --runtime 'uv-sglang://server=0.5.12.post1&build=cu121&python=3.12'

  # OpenVINO (Linux/Windows) -- one venv per version; `device` is per cell, not per install
  pipette runtimes pull --runtime 'uv-openvino://version=2026.2.1'

  # or a JSON Runtime object
  pipette runtimes pull --runtime '{\"type\":\"docker_vllm\",\"image_name\":\"vllm\",\"image_tag\":\"v0.10.0\",\"flavor\":\"nvidia_gpu\"}'
";

impl RuntimeArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        self.command.execute(ws, http)
    }
}

impl RuntimeCommand {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        match self {
            RuntimeCommand::List(args) => list(ws, args.format),
            RuntimeCommand::Catalog(args) => args.execute(http),
            RuntimeCommand::Flavors => print_llamacpp_flavors(),
            RuntimeCommand::Pull(args) => args.execute(ws, http),
            RuntimeCommand::Remove(args) => args.execute(ws),
        }
    }
}

impl PullArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        use anyhow::Context;

        let declared = resolve_runtime_arg(ws, &self.runtime)?;
        // No pin set: `ensure_runtime` pins the entry it is about to publish.
        let ctx = ws.artifacts(http);
        let store = ws.runtimes();

        // A llama.cpp archive streams bytes this process counts; a uv solve and a
        // docker pull report their own, so those draw nothing here.
        let progress =
            CellProgress::new(&[runtime_download_size(&store, &declared).unwrap_or(None)]);
        // One procedure for every installable kind (llama / docker / mlx / uv).
        let _bound = ensure_runtime(&ctx.with_progress(progress.sink()), &store, &declared)
            .with_context(|| format!("installing runtime `{declared}`"))?;
        // Erased before the summary line, which would otherwise print under a
        // finished bar.
        drop(progress);

        let entry = store
            .find(&declared)?
            .ok_or_else(|| anyhow::anyhow!("runtime missing after ensure"))?;
        println!(
            "Installed `{declared}` → {}",
            store.entry_dir_for(&entry)?.display()
        );
        Ok(())
    }
}

impl RemoveArgs {
    pub fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        let declared = resolve_runtime_arg(ws, &self.runtime)?;
        let store = ws.runtimes();
        if store.remove(&declared)? {
            println!("Removed `{declared}`");
            // Same key as the store entry, so removing the runtime is what
            // makes its compiled artifacts unaddressable; leaving them would
            // strand bytes nothing can reach.
            let cache = ws.compile_cache(&declared)?;
            if cache.is_dir() {
                std::fs::remove_dir_all(&cache).with_context(|| {
                    format!("removing the compile cache at {}", cache.display())
                })?;
                println!("  reclaimed the compile cache");
            }
            if matches!(declared, Runtime::DockerVllm(_) | Runtime::DockerSglang(_)) {
                println!(
                    "  note: the docker image is still in the daemon; `docker rmi` to remove it"
                );
            }
            return Ok(());
        }
        println!("`{declared}` is not installed");
        Ok(())
    }
}

/// One row of `runtimes list`.
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

fn list(ws: &PipetteWorkspace, format: ListFormat) -> anyhow::Result<()> {
    let store = ws.runtimes();
    let mut rows: Vec<RuntimeRow> = store
        .list()?
        .into_iter()
        .map(|m| {
            Ok(RuntimeRow {
                // A runtime that predates a grammar key has no URI form; fall
                // back to the identity rather than dropping the row.
                runtime: match format {
                    ListFormat::Name => m.declared.cli_ref(),
                    ListFormat::Uri => {
                        runtime_to_uri(&m.declared).unwrap_or_else(|_| m.declared.cli_ref())
                    }
                },
                runtime_type: m.declared.headless_token().to_owned(),
                // The prefix `--runtime runtime://sha256=<hex>` takes, and the
                // same id the warehouse stores as `runtime_descriptor_sha256`.
                digest: descriptor::short_digest(&descriptor::digest(&m.declared)?).to_owned(),
                pulled: m.fetched_at_rfc3339()?,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    rows.sort_by(|a, b| a.runtime.cmp(&b.runtime));
    print_table_or(&rows, "No runtimes installed.");
    Ok(())
}

fn vllm_flavor_label(flavor: VllmFlavor) -> &'static str {
    match flavor {
        VllmFlavor::NvidiaGpu => "nvidia_gpu",
        VllmFlavor::AmdGpu => "amd_gpu",
        VllmFlavor::Cpu => "cpu",
    }
}

/// A catalog row whose `REF` is pasted verbatim into `--runtime`, carrying its
/// hardware `FLAVOR`. Used by the bundled uv-* and mlx subwords.
#[derive(Tabled)]
struct CatalogRefRow {
    #[tabled(rename = "REF")]
    runtime_ref: String,
    #[tabled(rename = "FLAVOR")]
    flavor: String,
}

/// A llama.cpp release row: a paste-ready `REF` plus the release's publish date.
#[derive(Tabled)]
struct LlamacppRefRow {
    #[tabled(rename = "REF")]
    runtime_ref: String,
    #[tabled(rename = "PUBLISHED")]
    published: String,
}

/// One llama.cpp *flavor* — the `--flavor` vocabulary, listed when the subword
/// is run without one. Carries the upstream asset stem because that is what
/// makes the choice decidable: flavor names are pipette's, asset names are
/// upstream's, and they disagree often enough (`linux-x64-openvino` ships as
/// `ubuntu-openvino-…`) that seeing both is the point.
#[derive(Tabled)]
struct LlamacppFlavorRow {
    #[tabled(rename = "FLAVOR")]
    flavor: String,
    #[tabled(rename = "UPSTREAM ASSET")]
    asset: String,
}

/// Bundled uv vLLM builds as paste-ready `uv-vllm://…` refs. Split from the
/// printing so the test can assert every emitted ref parses.
fn uv_vllm_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    uv_bundled_rows(pipette_torch_oai::catalog::CatalogEntry::is_vllm)
}

/// Bundled uv SGLang builds as paste-ready `uv-sglang://…` refs.
fn uv_sglang_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    uv_bundled_rows(pipette_torch_oai::catalog::CatalogEntry::is_sglang)
}

/// Walk the bundled torch-oai catalog, keeping the entries `keep` selects, and
/// render each as its scheme's paste-ready ref. A cosmetically malformed slug
/// is unreachable in practice (catalog tests + build.rs validate the table), so
/// skip one rather than abort the whole listing.
fn uv_bundled_rows(
    keep: fn(&pipette_torch_oai::catalog::CatalogEntry) -> bool,
) -> anyhow::Result<Vec<CatalogRefRow>> {
    use pipette_torch_oai::catalog::CatalogEntry;

    let mut rows: Vec<CatalogRefRow> = Vec::new();
    for slug_body in pipette_torch_oai::catalog::slugs()? {
        let Ok(slug) = pipette_torch_oai::slug::UvSlug::try_new(slug_body) else {
            continue;
        };
        let Some(entry) = pipette_torch_oai::catalog::lookup(&slug)? else {
            continue;
        };
        if !keep(&entry) {
            continue;
        }
        let runtime_ref = match &entry {
            CatalogEntry::UvVllm {
                server_version,
                build,
                python_version,
                ..
            } => format!("uv-vllm://server={server_version}&build={build}&python={python_version}"),
            CatalogEntry::UvSglang {
                server_version,
                build,
                python_version,
                ..
            } => {
                format!("uv-sglang://server={server_version}&build={build}&python={python_version}")
            }
        };
        rows.push(CatalogRefRow {
            runtime_ref,
            flavor: vllm_flavor_label(entry.flavor()).to_string(),
        });
    }
    Ok(rows)
}

/// Bundled OpenVINO builds as paste-ready `uv-openvino://…` refs — one row per
/// version, because one venv serves every device. Which device a cell runs on
/// is a flag on the cell, not part of what gets installed.
fn openvino_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(pipette_openvino::catalog::catalog_entries()?
        .into_iter()
        .map(|version| CatalogRefRow {
            runtime_ref: format!("uv-openvino://version={version}"),
            flavor: "any device".to_owned(),
        })
        .collect())
}

/// Bundled MLX builds as paste-ready `mlx-macos-pipette://…` refs.
///
/// Listable anywhere: the ref is authored and pulled on whatever host runs the
/// CLI, and only executing it needs a Mac.
fn mlx_rows() -> anyhow::Result<Vec<CatalogRefRow>> {
    Ok(pipette_mlx::catalog::catalog_entries()?
        .into_iter()
        .map(|(version, flavor)| CatalogRefRow {
            runtime_ref: format!("mlx-macos-pipette://version={version}"),
            flavor: flavor.to_string(),
        })
        .collect())
}

/// Upstream llama.cpp CLI releases carrying an asset for `flavor`, as
/// paste-ready `llamacpp-cli-stock-tools://…` refs. Hits GitHub.
///
/// Takes the parsed flavor: the caller has already rejected `Custom`, so the
/// canonical `as_str` is what belongs in an emitted ref.
fn llamacpp_rows(
    http: &HttpClient,
    flavor: &LlamaCppFlavor,
    limit: usize,
) -> anyhow::Result<Vec<LlamacppRefRow>> {
    let releases = pipette_llamacpp::github::github_releases(http, limit)?;
    let rows = releases
        .into_iter()
        .filter(|release| pipette_llamacpp::github::release_asset_available(release, flavor))
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

/// Holds `--flavor` to the known set, so clap rejects a typo by name and exits
/// non-zero. Validates through `parse` rather than a list of literals, so the
/// vocabulary stays defined in exactly one place; the operator is pointed at
/// `runtimes flavors`, which also shows the upstream asset each name resolves
/// to — more use than the bare set clap would print.
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

/// `runtimes flavors` — the `--flavor` vocabulary. Needs neither workspace nor
/// network, so it is dispatched before either is set up: a fresh box has to be
/// able to read this before it can pick what to pull.
pub fn print_llamacpp_flavors() -> anyhow::Result<()> {
    print_table_or(&llamacpp_flavor_rows(), "No llama.cpp flavors.");
    Ok(())
}

/// The `--flavor` vocabulary: every flavor pipette tracks, with the upstream
/// asset stem it resolves to. Offline — this is the enum, not the release feed.
fn llamacpp_flavor_rows() -> Vec<LlamacppFlavorRow> {
    LlamaCppFlavor::known()
        .map(|flavor| LlamacppFlavorRow {
            flavor: flavor.to_string(),
            // Every known flavor names an asset (pinned by
            // `every_known_flavor_is_offerable`); the placeholder stands in for the
            // version so the column shows the stable stem.
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
    /// `runtimes catalog <type>` — installable builds for one runtime type, each
    /// row a paste-ready `--runtime` ref. Offline and store-independent except
    /// the llama.cpp subword (GitHub via `http`); lists what's *installable*,
    /// not what's installed (that's `list`).
    pub fn execute(self, http: &HttpClient) -> anyhow::Result<()> {
        self.command.execute(http)
    }
}

impl CatalogCommand {
    pub fn execute(self, http: &HttpClient) -> anyhow::Result<()> {
        match self {
            CatalogCommand::UvVllm => {
                print_table_or(&uv_vllm_rows()?, "No bundled uv-vLLM builds.");
            }
            CatalogCommand::UvSglang => {
                print_table_or(&uv_sglang_rows()?, "No bundled uv-SGLang builds.");
            }
            CatalogCommand::MlxMacosPipette => {
                print_table_or(&mlx_rows()?, "No bundled MLX builds.");
            }
            CatalogCommand::UvOpenvino => {
                print_table_or(&openvino_rows()?, "No bundled OpenVINO builds.");
            }
            CatalogCommand::LlamacppCliStockTools(args) => {
                print_table_or(
                    &llamacpp_rows(http, &args.flavor, args.limit)?,
                    "No matching llama.cpp releases.",
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    /// The point of `--format uri` is that a listed row is paste-ready: what
    /// `runtimes list` renders must be what `runtimes pull` accepts. Drives off
    /// the bundled catalogs so the cases stay valid as they are revised.
    #[test]
    fn listed_uri_round_trips_through_pull() -> anyhow::Result<()> {
        let mut refs: Vec<String> = uv_vllm_rows()?
            .iter()
            .chain(uv_sglang_rows()?.iter())
            .map(|r| r.runtime_ref.clone())
            .collect();
        refs.extend(super::mlx_rows()?.iter().map(|r| r.runtime_ref.clone()));
        refs.push("llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64".to_string());

        for uri in refs {
            let declared = parse_runtime_arg(&uri)?;
            let rendered = crate::runtime_uri::runtime_to_uri(&declared)
                .map_err(|e| anyhow::anyhow!("{uri} has no URI form: {e}"))?;
            // Not byte-equality: the renderer spells out keys the catalog leaves
            // defaulted (`mlx-macos-pipette://…&flavor=macos-arm64`). What has to hold is that
            // pasting a listed row back gets you the same runtime.
            assert_eq!(
                parse_runtime_arg(&rendered)?,
                declared,
                "{uri} rendered to {rendered}, which parses to something else"
            );
        }
        Ok(())
    }
    use strum::IntoEnumIterator;

    use super::{uv_sglang_rows, uv_vllm_rows, RUNTIME_URI_EXAMPLES};
    use crate::runtime_uri::{parse_runtime_arg, Scheme};

    /// Every emitted BUNDLED ref must parse back through the `--runtime`
    /// grammar — the whole point of the catalog is that its rows paste in
    /// verbatim. Covers uv-vLLM + uv-SGLang (present on every host) and, on
    /// macOS, the mlx rows. The network-backed llamacpp subword is excluded.
    #[test]
    fn bundled_catalog_refs_parse() -> anyhow::Result<()> {
        let vllm = uv_vllm_rows()?;
        let sglang = uv_sglang_rows()?;
        assert!(
            !vllm.is_empty(),
            "bundled uv-vLLM catalog must have entries"
        );
        assert!(
            !sglang.is_empty(),
            "bundled uv-SGLang catalog must have entries"
        );

        let mut refs: Vec<String> = vllm
            .iter()
            .chain(sglang.iter())
            .map(|r| r.runtime_ref.clone())
            .collect();
        refs.extend(super::mlx_rows()?.iter().map(|r| r.runtime_ref.clone()));

        for r in &refs {
            parse_runtime_arg(r)
                .map_err(|e| anyhow::anyhow!("catalog ref `{r}` should parse: {e}"))?;
        }
        Ok(())
    }

    /// `--flavor` is held to the known set, so a typo is a clap error (exit 2)
    /// rather than a successful run that lists nothing. Both directions matter:
    /// rejecting an unknown spelling, and accepting every one the listing
    /// offers — a validator stricter than the listing would refuse its own
    /// advice.
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

    /// The flavor listing exists to be pasted into `--flavor`, so every row
    /// must survive the trip back: parse to a known variant (not `Custom`,
    /// which lists nothing) and compose into a `--runtime` URI that parses.
    /// The asset column must also show a stem, since an empty one would make
    /// a listed flavor look unfetchable.
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

    /// Guard against doc rot: every `--runtime '<arg>'` in the help's examples
    /// must actually parse. Extracts the args from the help text itself, so the
    /// examples can't drift from the grammar.
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

    /// The other half of the doc-rot guard, and the one that catches an
    /// omission rather than a typo: a scheme the grammar accepts but the help
    /// never shows is a runtime nobody can discover.
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

    /// The naming rule, checked against plan-types rather than restated: a
    /// scheme is its runtime's type word with `-` for `_`. Fails the day a
    /// shorthand is introduced.
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
}
