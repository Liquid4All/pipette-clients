use clap::{Args, Subcommand};

use pipette_http::HttpClient;
use pipette_mgmt_client::{MgmtClient, PreauthKey};
use pipette_plan_types::device::DeviceFormFactor;

use crate::client::auth::{
    fetch_profile, register, set_device, update_client_details, RegisterInput,
};
use crate::client::clear_remote_state;
use crate::error::{Error, Result};
use crate::results::BenchmarkResultLocation;
use crate::workspace::PipetteWorkspace;

/// Render an optional label for `key: value` output.
fn opt(value: &Option<impl std::fmt::Display>) -> String {
    value
        .as_ref()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "(unset)".to_string())
}

/// Render a client's tags for `auth me`: sorted and comma-joined, or `(none)`
/// when untagged.
fn format_tags(mut tags: Vec<String>) -> String {
    tags.sort();
    if tags.is_empty() {
        "(none)".to_string()
    } else {
        tags.join(", ")
    }
}

const REGISTER_LONG_ABOUT: &str = "\
Register this machine with the management server.

All values must be supplied on the command line. Register refuses to run if the
workspace already contains any identity files (private key, public key, or
registration.json). Use `auth reset` to remove them before re-registering.

To relabel device metadata without re-registering, use `auth set-device`.";

const REGISTER_AFTER_HELP: &str = "\
Examples:

  Minimal (auto-detected name and form factor):
    pipette auth register --organization LiquidAI \\
      --contact-email user@example.com --client-details \"linux ci box\"

  With explicit overrides (rarely needed, only to disambiguate SKUs
  or correct a misdetected form factor):
    pipette auth register --organization LiquidAI \\
      --contact-email user@example.com --client-details \"mac studio\" \\
      --device-name \"Mac Studio M3 Ultra 96GB\" --device-form-factor desktop

  Pre-authorized (comes up already approved, no manual approval step):
    pipette auth register --organization LiquidAI \\
      --contact-email user@example.com --client-details \"ci box\" \\
      --preauth-key preauth_...
  The key can also be passed via PIPETTE_PREAUTH_KEY to keep it out of
  shell history; it is sent with the request and never written to disk.

Default --server-url is https://collector.pipette.liquid.ai.
Run `pipette auth me` after registration to confirm the profile.";

const RESET_LONG_ABOUT: &str = "\
Delete the local identity and every cache derived from the server relationship.

This removes the private key, registration, and device labels, plus the pulled
remote benchmarks, remote results (pending and synced), and eval score-refresh
state, all of which are bound to the identity being torn down. Local benchmarks
and local results are server-independent and kept.

If any results are still awaiting submission (`sync` has not run), reset refuses
unless --force is given, so unsubmitted runs aren't silently discarded.";

/// Manage client identity and server registration
#[derive(Args, Debug)]
pub struct AuthArgs {
    #[command(subcommand)]
    pub command: AuthCommand,
}

#[derive(Subcommand, Debug)]
pub enum AuthCommand {
    /// Register a new client identity with the management server
    #[command(long_about = REGISTER_LONG_ABOUT, after_help = REGISTER_AFTER_HELP)]
    Register(RegisterArgs),
    /// Delete the local identity and every server-derived cache
    #[command(long_about = RESET_LONG_ABOUT)]
    Reset(ResetArgs),
    /// Fetch and print the current client profile from the server
    Me,
    /// Update local device labels without re-registering
    SetDevice(SetDeviceArgs),
}

impl AuthArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> Result<()> {
        self.command.execute(ws, http)
    }
}

impl AuthCommand {
    // `reset` clears server-derived caches (benchmarks/results/eval state).
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> Result<()> {
        match self {
            AuthCommand::Register(args) => args.execute(ws, http),
            AuthCommand::Reset(args) => args.execute(ws),
            AuthCommand::Me => {
                let session = ws.mgmt_session()?;
                let profile = fetch_profile(&session.identity, &session.client, http)?;
                println!("client_id:        {}", profile.client_id);
                println!("organization:     {}", profile.organization);
                println!("client_details:   {}", profile.client_details);
                println!("contact_email:    {}", profile.contact_email);
                println!("status:           {}", profile.status);
                // Tags are mgmt-assigned (`pipette-mgmt clients tag ...`), read
                // live from /clients/me; display-only here.
                println!("tags:             {}", format_tags(profile.tags));
                println!("reindex_pending:  {}", profile.reindex_pending);
                println!(
                    "capabilities:     {}",
                    if profile.capabilities.is_empty() {
                        "(none)".to_string()
                    } else {
                        profile.capabilities.join(", ")
                    }
                );
                Ok(())
            }
            AuthCommand::SetDevice(args) => args.execute(ws, http),
        }
    }
}

#[derive(Args, Debug)]
pub struct ResetArgs {
    /// Delete unsubmitted remote results too. Required only when such results
    /// exist; a workspace with nothing pending resets without it.
    #[arg(long)]
    pub force: bool,
}

impl ResetArgs {
    /// Tear down the identity and every server-derived cache. Refuses when
    /// unsubmitted results would be lost unless `--force` is set; see
    /// [`RESET_LONG_ABOUT`] for the full list of what's removed vs. kept.
    pub fn execute(self, ws: &PipetteWorkspace) -> Result<()> {
        // Pending is load-bearing — it gates --force — so a failure to enumerate it
        // is fatal. The synced/benchmark counts and server label are display-only
        // and best-effort: a corrupt or unreadable cache must not block `auth reset`,
        // whose whole job is to delete that cache.
        let results = ws.results();
        let pending = results.list_ids(BenchmarkResultLocation::RemotePending)?;
        let synced = results
            .list_ids(BenchmarkResultLocation::RemoteSynced)
            .map(|ids| ids.len())
            .unwrap_or(0);
        let bench_store = ws.benchmarks();
        let benchmarks = bench_store
            .list(pipette_plan_types::benchmark::BenchmarkSource::Remote)
            .map(|entries| entries.len())
            .unwrap_or(0);
        let identity = ws.identity();
        let server = identity
            .require_registration()
            .ok()
            .map(|reg| reg.server_url);

        if !pending.is_empty() && !self.force {
            println!("auth reset would remove:");
            print_reset_targets(server.as_deref(), benchmarks, pending.len(), synced);
            println!("Local benchmarks and results are kept.");
            return Err(Error::ResetNeedsForce {
                unsubmitted: pending.len(),
            });
        }

        identity.clear()?;
        let evals = ws.eval_completions();
        clear_remote_state(&bench_store, &results, &evals)?;

        println!("identity deleted; removed:");
        print_reset_targets(server.as_deref(), benchmarks, pending.len(), synced);
        Ok(())
    }
}

/// Render the shared "identity + N benchmarks + M results (K unsubmitted)"
/// breakdown used by both the refusal preview and the post-reset summary.
fn print_reset_targets(server: Option<&str>, benchmarks: usize, pending: usize, synced: usize) {
    match server {
        Some(url) => println!("  • client identity (registered against {url})"),
        None => println!("  • client identity"),
    }
    println!("  • {benchmarks} remote benchmark(s)");
    println!(
        "  • {} remote result(s) ({pending} unsubmitted)",
        pending + synced
    );
    println!("  • eval score-refresh state");
}

#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Management server URL
    #[arg(long, default_value = "https://collector.pipette.liquid.ai")]
    pub server_url: String,

    /// Organization operating this client, e.g. "LiquidAI"
    #[arg(long)]
    pub organization: String,

    /// Contact email for registration
    #[arg(long)]
    pub contact_email: String,

    /// Human-readable client description
    #[arg(long)]
    pub client_details: String,

    /// Pre-auth key (`preauth_…`) that admits this client already approved.
    /// Also read from PIPETTE_PREAUTH_KEY (flag wins) to keep it out of shell
    /// history. Sent with the request; never written to disk.
    #[arg(long, env = "PIPETTE_PREAUTH_KEY", hide_env_values = true)]
    pub preauth_key: Option<PreauthKey>,

    /// Override the auto-detected device name (e.g. "Mac Studio M3 Ultra
    /// 96GB"). Rarely needed — set it only to disambiguate identical SKUs
    /// or to give a unit a friendlier label in dashboards.
    #[arg(long)]
    pub device_name: Option<String>,

    /// Override the auto-detected form factor (phone, tablet, laptop,
    /// desktop, server, embedded). Rarely needed — the value is inferred
    /// from the host; set it only when the auto-detect is wrong (e.g. a
    /// workstation used as a server).
    #[arg(long)]
    pub device_form_factor: Option<DeviceFormFactor>,
}

impl RegisterArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> Result<()> {
        let identity = ws.identity();
        let client = MgmtClient::new(&self.server_url)?;
        let registration = register(
            &identity,
            &client,
            http,
            RegisterInput {
                server_url: self.server_url,
                organization: self.organization,
                contact_email: self.contact_email,
                client_details: self.client_details,
                device_name: self.device_name,
                device_form_factor: self.device_form_factor,
                preauth_key: self.preauth_key,
            },
        )?;
        println!("client_id:  {}", registration.client_id);
        println!("server_url: {}", registration.server_url);
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct SetDeviceArgs {
    /// New override for the auto-detected device name. Same conventions
    /// as `auth register --device-name`.
    #[arg(long)]
    pub device_name: Option<String>,

    /// New override for the auto-detected form factor: phone, tablet,
    /// laptop, desktop, server, embedded.
    #[arg(long)]
    pub device_form_factor: Option<DeviceFormFactor>,

    /// New human-readable client description. Unlike the device labels above
    /// (which are local), this is server-side registration metadata, so
    /// updating it reaches the management server and requires a registered
    /// client — but it no longer forces a full reset + re-register.
    #[arg(long)]
    pub client_details: Option<String>,
}

impl SetDeviceArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> Result<()> {
        let identity = ws.identity();
        if self.device_name.is_none()
            && self.device_form_factor.is_none()
            && self.client_details.is_none()
        {
            return Err(crate::error::Error::SetDeviceEmpty);
        }

        // Device labels are local (device.json) — no network, no registration.
        if self.device_name.is_some() || self.device_form_factor.is_some() {
            let labels = set_device(&identity, self.device_name, self.device_form_factor)?;
            println!("device_name:        {}", opt(&labels.device_name));
            println!("device_form_factor: {}", opt(&labels.device_form_factor));
        }

        // `client_details` is server-side — patch it on the management server.
        if let Some(details) = self.client_details {
            let session = ws.mgmt_session()?;
            let profile = update_client_details(&session.identity, &session.client, http, details)?;
            println!("client_details:     {}", profile.client_details);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::test_support::TempWorkspace;

    #[rstest::rstest]
    #[case::empty(&[], "(none)")]
    #[case::single(&["team/mobile"], "team/mobile")]
    #[case::sorted(&["team/mobile", "batch/2026-q3"], "batch/2026-q3, team/mobile")]
    fn format_tags_cases(#[case] tags: &[&str], #[case] expected: &str) {
        let tags = tags.iter().map(|s| s.to_string()).collect();
        assert_eq!(format_tags(tags), expected);
    }

    // PIP-375: empty set-device is rejected before I/O.
    #[test]
    fn set_device_requires_at_least_one_field() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("set-device-empty")?;
        let args = SetDeviceArgs {
            device_name: None,
            device_form_factor: None,
            client_details: None,
        };
        let http = HttpClient::new("pipette-test/0")?;
        match args.execute(&tmp.ws, &http) {
            Err(crate::error::Error::SetDeviceEmpty) => Ok(()),
            other => anyhow::bail!("expected SetDeviceEmpty, got {other:?}"),
        }
    }

    fn add_pending(ws: &PipetteWorkspace, id: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(
            ws.results()
                .result_dir(BenchmarkResultLocation::RemotePending, id),
        )
    }

    #[test]
    fn reset_refuses_unsubmitted_results_without_force() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("reset-refuse")?;
        add_pending(&tmp.ws, "r-1")?;

        match (ResetArgs { force: false }).execute(&tmp.ws) {
            Err(Error::ResetNeedsForce { unsubmitted: 1 }) => {}
            other => anyhow::bail!("expected ResetNeedsForce {{ unsubmitted: 1 }}, got {other:?}"),
        }
        assert!(
            tmp.ws
                .results()
                .location_dir(BenchmarkResultLocation::RemotePending)
                .exists(),
            "refused reset must not delete pending results"
        );
        Ok(())
    }

    #[test]
    fn reset_force_clears_unsubmitted_results() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("reset-force")?;
        add_pending(&tmp.ws, "r-1")?;

        (ResetArgs { force: true }).execute(&tmp.ws)?;

        assert!(!tmp
            .ws
            .results()
            .location_dir(BenchmarkResultLocation::RemotePending)
            .exists());
        Ok(())
    }

    #[test]
    fn reset_without_pending_proceeds_without_force() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("reset-empty")?;
        std::fs::create_dir_all(tmp.ws.root().join("benchmarks/remote"))?;

        (ResetArgs { force: false }).execute(&tmp.ws)?;

        assert!(!tmp.ws.root().join("benchmarks/remote").exists());
        Ok(())
    }
}
