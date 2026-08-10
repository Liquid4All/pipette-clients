use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use tabled::Tabled;

use pipette_plan_types::Plan;

use super::state_io::{load_state_index, state_file_path};
use crate::{state::StateSummary, transport::Transport};

#[derive(Tabled)]
pub struct StatusRow {
    #[tabled(rename = "KEY")]
    key: String,
    #[tabled(rename = "VALUE")]
    value: String,
}

/// Status information returned by [`load_status`].
pub struct StatusInfo {
    plan_id: String,
    target_labels: Vec<String>,
    runtimes: Vec<String>,
    state_path: PathBuf,
    /// Counts of done/failed/missing cells. Public so callers can
    /// branch on them (e.g. "only print the missing table if
    /// summary.missing > 0").
    pub summary: StateSummary,
}

impl StatusInfo {
    /// Build the standard status rows for rendering as a table.
    pub fn to_rows(&self) -> Vec<StatusRow> {
        vec![
            StatusRow {
                key: "plan_id".into(),
                value: self.plan_id.clone(),
            },
            StatusRow {
                key: "targets".into(),
                value: self.target_labels.join(", "),
            },
            StatusRow {
                key: "runtimes".into(),
                value: self.runtimes.join(", "),
            },
            StatusRow {
                key: "state".into(),
                value: self.state_path.display().to_string(),
            },
            StatusRow {
                key: "total".into(),
                value: self.summary.total.to_string(),
            },
            StatusRow {
                key: "done".into(),
                value: self.summary.done.to_string(),
            },
            StatusRow {
                key: "failed".into(),
                value: self.summary.failed.to_string(),
            },
            StatusRow {
                key: "missing".into(),
                value: self.summary.missing.to_string(),
            },
        ]
    }
}

/// Load status information for a plan.
pub fn load_status(
    plans_dir: &Path,
    plan: &Plan,
    adb_port: Option<u16>,
) -> anyhow::Result<StatusInfo> {
    let cells: Vec<_> = plan.runnable_cells()?.into_iter().collect();
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    let transports: Vec<Transport> = plan
        .transports
        .iter()
        .map(|cfg| Transport::from_config_with_adb_port(cfg, adb_port))
        .collect();
    let runtimes: Vec<String> = cells
        .iter()
        .map(|c| c.runtime.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(StatusInfo {
        plan_id: plan.plan_id.clone(),
        target_labels: transports.iter().map(|t| t.target_label()).collect(),
        runtimes,
        state_path: state_file_path(plans_dir, &plan.plan_id),
        summary: state.summary_for(&cells),
    })
}
