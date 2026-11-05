//! What two variants over the same `(benchmark, model, runtime)` expand to,
//! and which of them a run actually executes.
//!
//! `allowed_clients` is on a `RunnableCell` but not in its state key, so two
//! variants differing only by `clients` are two cells with one key: recording
//! either as done takes both out of the queue. A flag entry on one keys them
//! apart, which is why the split looks like it depends on runtime flags.
//!
//! Goes through the public surface only, so it fails if any layer between
//! `Plan` and `StateIndex` changes what a cell is.

use pipette_ctl::state::{AttemptStatus, CellState, StateEvent, StateIndex, build_cell_key};
use pipette_plan_types::{Plan, RunnableCell};

const MODEL: &str = r#"{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }"#;
const RUNTIME: &str = r#"{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }"#;
const FLAGS: &str = r#"{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4 }"#;

/// Two transports so a variant can name a pool of its own, and one benchmark
/// so every variant covers exactly the same cell.
fn plan(variants: &str) -> String {
    format!(
        r#"
plan_id    = "identity"
benchmarks = ["prefill_throughput_256"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[transports]]
client_id = "t2"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"
{variants}"#
    )
}

fn variant(clients: &str, flags: Option<&str>) -> String {
    let flags = flags.map_or_else(String::new, |flags| format!("runtime_flags = [{flags}]\n"));
    format!(
        "\n[[variants]]\nclients = [\"{clients}\"]\nmodels = [{MODEL}]\nruntimes = [{RUNTIME}]\n{flags}"
    )
}

fn cells(variants: &str) -> anyhow::Result<Vec<RunnableCell>> {
    Ok(Plan::parse(&plan(variants))?
        .runnable_cells()?
        .into_iter()
        .collect())
}

fn cell_for<'a>(cells: &'a [RunnableCell], client: &str) -> anyhow::Result<&'a RunnableCell> {
    cells
        .iter()
        .find(|cell| {
            cell.allowed_clients
                .iter()
                .any(|allowed| allowed.as_ref() == client)
        })
        .ok_or_else(|| anyhow::anyhow!("no cell is allowed on {client}"))
}

/// The state file as a run that finished `cell` on `client` would leave it.
fn state_after(cell: &RunnableCell, client: &str) -> anyhow::Result<String> {
    let event = StateEvent::new("identity", cell, AttemptStatus::Success, 1, client)?;
    Ok(serde_json::to_string(&event)?)
}

/// Expansion collects into a set, so the duplicate is dropped rather than run
/// twice.
#[test]
fn identical_variants_expand_to_one_cell() -> anyhow::Result<()> {
    let cells = cells(&format!("{}{}", variant("t1", None), variant("t1", None)))?;

    assert_eq!(cells.len(), 1, "the duplicate variant adds no cell");
    Ok(())
}

#[test]
fn variants_differing_only_by_client_share_one_state_key() -> anyhow::Result<()> {
    let cells = cells(&format!("{}{}", variant("t1", None), variant("t2", None)))?;
    assert_eq!(cells.len(), 2, "each variant carries its own client pool");

    let first = cell_for(&cells, "t1")?;
    let second = cell_for(&cells, "t2")?;
    assert_eq!(
        build_cell_key(first),
        build_cell_key(second),
        "routing is not identity, so the pools do not key apart"
    );

    let index = StateIndex::load(Some(&state_after(first, "t1")?), "identity")?;
    assert_eq!(index.state_for(first), CellState::Done);
    assert_eq!(
        index.state_for(second),
        CellState::Done,
        "running the cell on one pool takes the other variant's cell out of the queue"
    );
    Ok(())
}

#[test]
fn a_runtime_flag_entry_keys_the_cells_apart() -> anyhow::Result<()> {
    let cells = cells(&format!(
        "{}{}",
        variant("t1", None),
        variant("t2", Some(FLAGS))
    ))?;
    assert_eq!(cells.len(), 2);

    let unflagged = cell_for(&cells, "t1")?;
    let flagged = cell_for(&cells, "t2")?;
    assert!(
        unflagged.runtime_flags.is_none() && flagged.runtime_flags.is_some(),
        "the fixture must put flags on exactly one side"
    );
    assert_ne!(build_cell_key(unflagged), build_cell_key(flagged));

    let index = StateIndex::load(Some(&state_after(flagged, "t2")?), "identity")?;
    assert_eq!(index.state_for(flagged), CellState::Done);
    assert_eq!(
        index.state_for(unflagged),
        CellState::Missing,
        "the unflagged cell is still queued"
    );
    Ok(())
}

#[test]
fn a_variant_without_flags_still_expands() -> anyhow::Result<()> {
    let cells = cells(&variant("t1", None))?;

    assert_eq!(cells.len(), 1, "one benchmark × one model × one runtime");
    assert!(
        cell_for(&cells, "t1")?.runtime_flags.is_none(),
        "and it carries no flags"
    );
    Ok(())
}

/// A fleet plan's shape: several benchmarks over several models on one
/// runtime. `{FLAGS}` is replaced by a flag block, or by nothing.
const FLEET: &str = r#"
plan_id = "fleet"

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[variants]]
benchmarks = ["prefill_throughput_256", "decode_throughput_256_100", "end_to_end_latency_256_256"]
clients    = ["t1"]
models = [
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "b.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "c.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "d.gguf" },
]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
{FLAGS}"#;

const FLEET_FLAGS: &str = r#"runtime_flags = [
  { benchmark_type = "prefill_throughput",  runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 8, number_gpu_layers = 99, flash_attention = "on" },
  { benchmark_type = "decode_throughput",   runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 8, number_gpu_layers = 99, flash_attention = "on" },
  { benchmark_type = "end_to_end_latency",  runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 8, number_gpu_layers = 99, flash_attention = "on" },
]
"#;

/// The cells of `FLEET`, as `(benchmark, model, runtime)` triples with whether
/// each carries runtime flags.
fn fleet_cells(flags: &str) -> anyhow::Result<Vec<(String, bool)>> {
    let mut cells: Vec<(String, bool)> = Plan::parse(&FLEET.replace("{FLAGS}", flags))?
        .runnable_cells()?
        .into_iter()
        .map(|cell| {
            (
                format!(
                    "{} {} {}",
                    cell.benchmark.as_ref(),
                    cell.model,
                    cell.runtime
                ),
                cell.runtime_flags.is_some(),
            )
        })
        .collect();
    cells.sort();
    Ok(cells)
}

/// The matrix is benchmarks × models × runtimes; flags are resolved onto cells
/// afterwards, so they never decide which cells exist. Checked against the real
/// `2026.08.02-aa/aa-ext.toml` too: 336 cells and 336 `commands` lines either
/// way. `FLEET` is its shape.
#[test]
fn stripping_runtime_flags_keeps_every_cell() -> anyhow::Result<()> {
    let flagged = fleet_cells(FLEET_FLAGS)?;
    let bare = fleet_cells("")?;

    assert_eq!(flagged.len(), 12, "3 benchmarks × 4 models × 1 runtime");
    assert_eq!(
        flagged.iter().map(|(cell, _)| cell).collect::<Vec<_>>(),
        bare.iter().map(|(cell, _)| cell).collect::<Vec<_>>(),
        "the same cells arrange with the flag block and without it"
    );
    assert!(
        flagged.iter().all(|(_, has_flags)| *has_flags),
        "every cell resolved the entry for its benchmark_type"
    );
    assert!(
        bare.iter().all(|(_, has_flags)| !*has_flags),
        "and none does once the block is gone"
    );
    Ok(())
}

#[test]
fn a_flagless_cell_still_builds_its_invocation() -> anyhow::Result<()> {
    let plan = Plan::parse(&FLEET.replace("{FLAGS}", ""))?;
    let transport = plan
        .transports
        .first()
        .ok_or_else(|| anyhow::anyhow!("the fixture must declare a transport"))?;
    let cell = plan
        .runnable_cells()?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("the plan produced no cells"))?;

    let argv = cell.build_argv(transport)?;

    assert!(
        argv.iter().any(|arg| arg == "--model") && argv.iter().any(|arg| arg == "--runtime"),
        "the cell names its work: {argv:?}"
    );
    assert!(
        !argv.iter().any(|arg| arg == "--runtime-flags"),
        "and passes no flags rather than an empty object: {argv:?}"
    );
    Ok(())
}
