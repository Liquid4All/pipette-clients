use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use clap::ValueEnum;
use tabled::{settings::Style, Tabled};

use pipette_plan_types::{Plan, RunnableCell};

use super::state_io::load_state_index;
use crate::state::CellState;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListState {
    Missing,
    Failed,
    Done,
    All,
}

impl ListState {
    /// Whether a cell in `state` is selected by this filter.
    pub fn matches(self, state: CellState) -> bool {
        match self {
            ListState::All => true,
            ListState::Missing => state == CellState::Missing,
            ListState::Failed => state == CellState::Failed,
            ListState::Done => state == CellState::Done,
        }
    }
}

fn display_cell_state(state: CellState) -> &'static str {
    match state {
        CellState::Done => "done",
        CellState::Failed => "failed",
        CellState::Missing => "missing",
    }
}

/// Tabled row for a single matrix cell and its current state.
#[derive(Tabled)]
struct CellRow {
    #[tabled(rename = "STATE")]
    state: String,
    #[tabled(rename = "BENCHMARK")]
    benchmark: String,
    #[tabled(rename = "MODEL")]
    model: String,
    #[tabled(rename = "RUNTIME")]
    runtime: String,
}

/// Print matrix cells filtered by state as a psql table
/// (STATE, BENCHMARK, MODEL, RUNTIME). Prints nothing when no cells match.
pub fn list_matrix(plans_dir: &Path, plan: &Plan, filter: ListState) -> anyhow::Result<()> {
    let cells: Vec<RunnableCell> = plan.runnable_cells()?.into_iter().collect();
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    let rows: Vec<CellRow> = cells
        .iter()
        .map(|cell| (cell, state.state_for(cell)))
        .filter(|(_, cell_state)| filter.matches(*cell_state))
        .map(|(cell, cell_state)| CellRow {
            state: display_cell_state(cell_state).to_string(),
            benchmark: cell.benchmark.as_ref().to_string(),
            model: cell.model.to_string(),
            runtime: cell.runtime.to_string(),
        })
        .collect();
    if !rows.is_empty() {
        println!("{}", tabled::Table::new(&rows).with(Style::psql()));
    }
    Ok(())
}

/// Return matrix cells matching the given state filter.
pub fn cells_in_state(
    plans_dir: &Path,
    plan: &Plan,
    filter: ListState,
) -> anyhow::Result<Vec<RunnableCell>> {
    let cells: Vec<RunnableCell> = plan.runnable_cells()?.into_iter().collect();
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    Ok(cells
        .into_iter()
        .filter(|cell| filter.matches(state.state_for(cell)))
        .collect())
}

/// Tabled row that groups cells by (benchmark, runtime), listing all
/// affected models in a single multi-line `MODELS` cell.
#[derive(Tabled)]
pub struct GroupedCellRow {
    #[tabled(rename = "BENCHMARK")]
    benchmark: String,
    #[tabled(rename = "MODELS")]
    models: String,
    #[tabled(rename = "RUNTIME")]
    runtime: String,
}

pub fn group_cells_by_benchmark(cells: &[RunnableCell]) -> Vec<GroupedCellRow> {
    cells
        .iter()
        .fold(
            BTreeMap::<(String, String), BTreeSet<String>>::new(),
            |mut map, cell| {
                map.entry((
                    cell.benchmark.as_ref().to_string(),
                    cell.runtime.to_string(),
                ))
                .or_default()
                .insert(cell.model.to_string());
                map
            },
        )
        .into_iter()
        .map(|((benchmark, runtime), models)| GroupedCellRow {
            benchmark,
            models: models.into_iter().collect::<Vec<_>>().join("\n"),
            runtime,
        })
        .collect()
}
