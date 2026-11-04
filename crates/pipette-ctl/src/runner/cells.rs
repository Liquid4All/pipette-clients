use std::{collections::BTreeMap, path::Path};

use clap::ValueEnum;

use pipette_plan_types::{Plan, RunnableCell};

use super::state_io::load_state_index;
use crate::state::CellState;

/// Which cells of a matrix a command works on.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellFilter {
    /// Everything still to do: failed and missing together.
    Outstanding,
    Missing,
    Failed,
    Done,
    All,
}

impl CellFilter {
    pub fn matches(self, state: CellState) -> bool {
        match self {
            CellFilter::All => true,
            CellFilter::Outstanding => {
                matches!(state, CellState::Failed | CellState::Missing)
            }
            CellFilter::Missing => state == CellState::Missing,
            CellFilter::Failed => state == CellState::Failed,
            CellFilter::Done => state == CellState::Done,
        }
    }
}

/// How a cell's state reads in a section heading.
pub fn display_cell_state(state: CellState) -> &'static str {
    match state {
        CellState::Failed => "failed",
        CellState::Missing => "missing",
        CellState::Done => "done",
    }
}

/// Worst-first: what broke, then what is left, then what is finished.
const STATE_ORDER: [CellState; 3] = [CellState::Failed, CellState::Missing, CellState::Done];

/// Group cells by state, worst first, dropping empty states. A section per
/// state rather than one ladder: inside a ladder a cell's state could only be
/// a branch under every model, splitting each list in two.
pub fn split_by_state(
    cells: Vec<(RunnableCell, CellState)>,
) -> Vec<(CellState, Vec<RunnableCell>)> {
    cells
        .into_iter()
        .fold(
            BTreeMap::<usize, (CellState, Vec<RunnableCell>)>::new(),
            |mut groups, (cell, state)| {
                groups
                    .entry(state_rank(state))
                    .or_insert_with(|| (state, Vec::new()))
                    .1
                    .push(cell);
                groups
            },
        )
        .into_values()
        .collect()
}

/// Where `state` sorts in [`STATE_ORDER`]. A state missing from that array
/// sorts last rather than losing its section.
fn state_rank(state: CellState) -> usize {
    STATE_ORDER
        .iter()
        .position(|ordered| *ordered == state)
        .unwrap_or(STATE_ORDER.len())
}

/// Return matrix cells matching `filter`, each with the state it matched
/// on: what a cell is now decides which section it lands in.
pub fn cells_in_state(
    plans_dir: &Path,
    plan: &Plan,
    filter: CellFilter,
) -> anyhow::Result<Vec<(RunnableCell, CellState)>> {
    let cells: Vec<RunnableCell> = plan.runnable_cells()?.into_iter().collect();
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    Ok(cells
        .into_iter()
        .map(|cell| {
            let cell_state = state.state_for(&cell);
            (cell, cell_state)
        })
        .filter(|(_, cell_state)| filter.matches(*cell_state))
        .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(CellFilter::Outstanding, CellState::Failed, true)]
    #[case(CellFilter::Outstanding, CellState::Missing, true)]
    #[case(CellFilter::Outstanding, CellState::Done, false)]
    #[case(CellFilter::Missing, CellState::Missing, true)]
    #[case(CellFilter::Missing, CellState::Failed, false)]
    #[case(CellFilter::Failed, CellState::Failed, true)]
    #[case(CellFilter::Done, CellState::Done, true)]
    #[case(CellFilter::Done, CellState::Missing, false)]
    #[case(CellFilter::All, CellState::Done, true)]
    #[case(CellFilter::All, CellState::Failed, true)]
    fn a_filter_selects_its_own_states(
        #[case] filter: CellFilter,
        #[case] state: CellState,
        #[case] selected: bool,
    ) {
        assert_eq!(filter.matches(state), selected);
    }

    /// One cell per state, handed over in the order a plan happens to
    /// produce them.
    fn one_of_each() -> anyhow::Result<Vec<(RunnableCell, CellState)>> {
        const PLAN: &str = r#"
plan_id    = "sections"
benchmarks = ["prefill_throughput_256", "decode_throughput_256_100", "end_to_end_latency_256_256"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
"#;
        Ok(Plan::parse(PLAN)?
            .runnable_cells()?
            .into_iter()
            .zip([CellState::Done, CellState::Missing, CellState::Failed])
            .collect())
    }

    #[test]
    fn sections_come_worst_first() -> anyhow::Result<()> {
        let sections = split_by_state(one_of_each()?);

        assert_eq!(
            sections
                .iter()
                .map(|(state, cells)| (display_cell_state(*state), cells.len()))
                .collect::<Vec<_>>(),
            vec![("failed", 1), ("missing", 1), ("done", 1)]
        );
        Ok(())
    }

    #[test]
    fn a_state_with_no_cells_has_no_section() -> anyhow::Result<()> {
        let outstanding: Vec<_> = one_of_each()?
            .into_iter()
            .filter(|(_, state)| CellFilter::Outstanding.matches(*state))
            .collect();

        let sections = split_by_state(outstanding);

        assert_eq!(
            sections
                .iter()
                .map(|(state, _)| display_cell_state(*state))
                .collect::<Vec<_>>(),
            vec!["failed", "missing"]
        );
        Ok(())
    }
}
