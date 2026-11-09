//! Planner-client helpers, split by what they touch: [`timing`] derives waits
//! from the server's lease window, [`profile`] advertises this client's
//! capabilities, [`protocol`] carries the claim / heartbeat / submit exchange,
//! [`in_flight`] marks the running job so a reboot during it is caught, and
//! `android_doze` turns doze off at worker start.
//!
//! The loop that drives them lives in [`crate::commands::worker`], which needs
//! the runtime seams.

#[cfg(any(target_os = "android", test))]
pub mod android_doze;
pub mod in_flight;
pub mod profile;
pub mod protocol;
pub mod timing;

pub use profile::{
    ReindexCallbacks, ReindexPhase, ReindexStatus, ReindexWait, installed_runtime_capabilities,
    refresh_profile_at_startup,
};
pub use protocol::{
    ClaimPoll, LeaseKeepalive, SubmitDisposition, SubmitRefusedError,
    attach_claim_to_success_payload, classify_run_error, device_time_ms, failure_code,
    failure_from_claim, format_failure_reason, keepalive_lease, poll_claim,
    submit_plan_result_with_backoff,
};
pub use timing::{
    DEFAULT_IDLE_JITTER, DEFAULT_IDLE_WAIT, idle_wait_with_jitter, resolve_heartbeat_interval,
};
