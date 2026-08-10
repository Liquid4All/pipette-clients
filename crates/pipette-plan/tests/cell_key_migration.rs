//! A plan whose state predates the runtime-flags format change resumes instead of
//! re-running.
//!
//! The unit tests cover the key functions; this covers the thing an operator actually
//! meets — a state file on disk, written by an earlier build, read by this one. It goes
//! through the public surface only (`Plan` → `RunnableCell` → `StateEvent` →
//! `StateIndex`), so it fails if the migration is dropped from any layer between them.

use pipette_plan::state::{
    build_cell_key, legacy_cell_key, AttemptStatus, CellState, StateEvent, StateIndex,
};
use pipette_plan_types::Plan;

/// One flagged cell: `runtime_flags` is what keys differently across the change, so an
/// unflagged cell would pass this test without exercising anything.
const PLAN: &str = r#"
plan_id          = "migration"
benchmarks       = ["prefill_throughput_256"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4 }]
"#;

/// A state file as an earlier build wrote it: the same event, keyed the old way.
fn state_file_keyed(cell: &pipette_plan_types::RunnableCell, key: &str) -> anyhow::Result<String> {
    let event = StateEvent::new("migration", cell, AttemptStatus::Success, 1, "t1")?;
    let mut value = serde_json::to_value(event)?;
    value["cell_key"] = serde_json::Value::String(key.to_owned());
    Ok(value.to_string())
}

#[test]
fn a_plan_resumes_across_the_runtime_flags_format_change() -> anyhow::Result<()> {
    let plan = Plan::parse(PLAN)?;
    let cells = plan.runnable_cells()?;
    let cell = cells
        .iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("plan produced no cells"))?;

    let current = build_cell_key(cell);
    let legacy = legacy_cell_key(cell).ok_or_else(|| anyhow::anyhow!("expected a legacy key"))?;
    assert_ne!(
        current, legacy,
        "a flagged cell must key differently across the change, or this proves nothing"
    );

    // What this build records.
    let written = StateEvent::new("migration", cell, AttemptStatus::Success, 1, "t1")?;
    assert_eq!(
        written.cell_key, current,
        "new state must be written in the new form"
    );

    // What it must still understand.
    for (label, key) in [("current", &current), ("legacy", &legacy)] {
        let raw = state_file_keyed(cell, key)?;
        let index = StateIndex::load(Some(&raw), "migration")?;

        assert_eq!(
            index.state_for(cell),
            CellState::Done,
            "a cell recorded under the {label} key must not be re-run"
        );
        assert_eq!(index.attempts_for(cell), 1, "{label} key: attempts");
    }
    Ok(())
}
