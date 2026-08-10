//! Planner-client helpers, split by what they touch: [`timing`] derives waits
//! from the server's lease window, [`profile`] advertises this client's
//! capabilities, and [`protocol`] carries the claim / heartbeat / submit
//! exchange.
//!
//! The loop that drives them lives in [`crate::commands::worker`], which needs
//! the runtime seams.

pub mod profile;
pub mod protocol;
pub mod timing;

pub use profile::{installed_runtime_capabilities, refresh_profile_at_startup};
pub use protocol::{
    attach_claim_to_success_payload, classify_run_error, failure_from_claim, format_failure_reason,
    keepalive_lease, poll_claim, submit_plan_result_with_backoff, ClaimPoll, LeaseKeepalive,
    SubmitDisposition,
};
pub use timing::{
    idle_wait_with_jitter, resolve_heartbeat_interval, DEFAULT_IDLE_JITTER, DEFAULT_IDLE_WAIT,
};
