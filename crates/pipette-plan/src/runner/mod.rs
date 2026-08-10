//! Shared runner infrastructure.
//!
//! Split by concern:
//!
//! - `state_io` — read/append the plan's JSONL state file.
//! - `probe` — device-alive heartbeat logic used by workers.
//! - `list` — enumerate matrix cells by state, table formatting.
//! - `status` — assemble the status summary shown by `pipette-plan
//!   status`.
//! - `run` — the worker loop that actually executes cells.

mod commands;
mod host_semaphore;
mod kill;
mod list;
mod probe;
mod run;
mod shard;
mod state_io;
mod status;

pub use commands::print_commands;
pub use kill::kill_transports;
pub use list::{cells_in_state, group_cells_by_benchmark, list_matrix, GroupedCellRow, ListState};
pub use run::{run_matrix, RunOptions};
pub use shard::Shard;
pub use status::{load_status, StatusInfo, StatusRow};
