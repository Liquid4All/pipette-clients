use std::{
    fmt::Display,
    path::{Path, PathBuf},
};

use pipette_plan_types::Plan;

use super::state_io::{load_state_index, state_file_path};
use crate::state::StateSummary;

/// Where a run has got to. What the plan is lives in the plan file and the
/// ladder drawn under this header.
pub struct StatusInfo {
    plan_id: String,
    state_path: PathBuf,
    summary: StateSummary,
}

/// The width the keys are padded to, so the values line up in one column.
const KEY_WIDTH: usize = 8;

impl StatusInfo {
    /// One `key value` line per field, not a table: a table pads every cell
    /// to the widest one, past a thousand columns on a fleet plan.
    pub fn render(&self) -> String {
        [
            field("plan_id", &self.plan_id),
            field("state", self.state_path.display()),
            field(
                "cells",
                format!(
                    "{} total: {} done, {} failed, {} missing",
                    self.summary.total,
                    self.summary.done,
                    self.summary.failed,
                    self.summary.missing
                ),
            ),
        ]
        .join("\n")
    }
}

fn field(key: &str, value: impl Display) -> String {
    format!("{key:KEY_WIDTH$}  {value}")
}

pub fn load_status(plans_dir: &Path, plan: &Plan) -> anyhow::Result<StatusInfo> {
    let cells: Vec<_> = plan.runnable_cells()?.into_iter().collect();
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    Ok(StatusInfo {
        plan_id: plan.plan_id.clone(),
        state_path: state_file_path(plans_dir, &plan.plan_id),
        summary: state.summary_for(&cells),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_puts_every_field_on_its_own_line() {
        let info = StatusInfo {
            plan_id: "p".to_string(),
            state_path: PathBuf::from("/s"),
            summary: StateSummary {
                total: 2,
                done: 0,
                failed: 1,
                missing: 1,
            },
        };

        assert_eq!(
            info.render(),
            "plan_id   p\nstate     /s\ncells     2 total: 0 done, 1 failed, 1 missing"
        );
    }
}
