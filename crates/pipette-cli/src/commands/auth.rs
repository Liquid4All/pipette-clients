use std::io::Read;
use std::path::PathBuf;

use clap::{Args, Subcommand};

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    ClientPreauthKey, ContactEmail, PublicKey, ServerUrl, parse_contact_email,
    public_key_hex_from_private,
};
use pipette_ops::run_context::RunContext;
use pipette_plan_types::DeviceFormFactor;

use crate::client::auth::{RegisterInput, fetch_profile, register, update_client_details};
use crate::client::clear_remote_state;
use crate::console::Console;
use crate::error::{Error, Result};
use crate::identity::{ClientRegistrationStore, IdentityBundle, StoredRegistration};
use crate::results::BenchmarkResultLocation;
use crate::secret_file;
use crate::workspace::PipetteWorkspace;

/// Render an optional label for `key: value` output.
fn opt(value: &Option<impl std::fmt::Display>) -> String {
    value
        .as_ref()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "(unset)".to_string())
}

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
      --preauth-key client_preauth_...
  The key can also be passed via PIPETTE_PREAUTH_KEY to keep it out of
  shell history; it is sent with the request and never written to disk.
  This is a device's key. An operator account registers with
  `pipette-ctl auth register --user-preauth-key`, against its own record.

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

const EXPORT_LONG_ABOUT: &str = "\
Print this client's identity as one restorable token.

The token carries the three values nothing can rebuild: the client_id the
server assigned at registration, the server it was registered against, and the
private key that signs for it. Everything else is re-detected on the machine
that imports it (device profile), local policy for that machine (storage
quota), or already server-side and readable with `auth me` (organization,
contact, client_details, status, tags).

The token holds the private key in plaintext. On stdout it lands in scrollback,
and in shell history if you pipe it; --out writes it to a file created 0600
instead. The summary goes to stderr and the token to stdout, so a redirect
captures the token alone.";

const IMPORT_LONG_ABOUT: &str = "\
Restore an identity exported with `auth export`.

Reads the token from stdin, or from --from <path>. The whole bundle is
validated before anything is written, so a token this build cannot read leaves
the workspace untouched.

Importing the identity this workspace already holds is not a conflict: a rerun
finishes an import that failed partway. Importing a different identity needs
--force, and also drops every cache derived from the outgoing one (remote
benchmarks, remote results pending and synced, eval score-refresh state).
Those are keyed to the client_id being replaced, and keeping them would submit
one client's runs under another's id on the next `sync`.

Device labels and the storage quota are deliberately not carried: they describe
the machine, not the identity. The importing machine detects its own hardware,
and the first worker startup pushes that profile to the server under the
restored client_id.";

const SECRET_LONG_ABOUT: &str = "\
Print this client's private key hex, and nothing else.

For a prompt that asks you to prove ownership of this client by pasting the
key. Be sure it is one you trust with the key itself: whoever reads this value
can sign as this client. The management server accepts a public key at
registration and nowhere else, so there is no rotation. A disclosed key can
only be retired with `auth reset` plus `auth register`, which mints a new
client_id and orphans every result submitted under the old one.

For a backup, use `auth export` instead: it carries the client_id and server
URL alongside the key, which is what a restore needs.";

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
    /// Print this client's profile, from the server or from disk
    #[command(long_about = ME_LONG_ABOUT)]
    Me(MeArgs),
    /// Update device labels or the client description without re-registering
    SetDevice(SetDeviceArgs),
    /// Print this client's identity as one restorable token
    #[command(long_about = EXPORT_LONG_ABOUT)]
    Export(ExportArgs),
    /// Restore an identity exported with `auth export`
    #[command(long_about = IMPORT_LONG_ABOUT)]
    Import(ImportArgs),
    /// Print this client's private key, and nothing else
    #[command(long_about = SECRET_LONG_ABOUT)]
    Secret,
}

impl AuthArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        self.command.execute(ws, http, console)
    }
}

impl AuthCommand {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        match self {
            AuthCommand::Register(args) => args.execute(ws, http, console),
            AuthCommand::Reset(args) => args.execute(ws, console),
            AuthCommand::Me(args) => args.execute(ws, http, console),
            AuthCommand::SetDevice(args) => args.execute(ws, http, console),
            AuthCommand::Export(args) => args.execute(ws, console),
            AuthCommand::Import(args) => args.execute(ws, console),
            AuthCommand::Secret => secret(ws, console),
        }
    }
}

const ME_LONG_ABOUT: &str = "\
Print this client's profile, live from the management server.

Most of what it shows is server-owned and has no local copy: organization,
contact, approval status, mgmt-assigned tags, the reindex gate, and the
advertised capabilities.

--local answers from the workspace instead, with no request and no approval
needed: the client_id, the server it is registered against, and the public key
derived from the signing key. That is every field a machine can answer about
itself, and it is what a provisioning step can assert on before the server has
approved the client.";

#[derive(Args, Debug)]
pub struct MeArgs {
    /// Answer from the workspace instead of the server: client_id, server_url,
    /// and public key. No network, and none of the server-owned fields.
    #[arg(long)]
    pub local: bool,
}

impl MeArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        if self.local {
            writeln!(console.out(), "{}", local_identity(ws)?.join("\n"))?;
            return Ok(());
        }
        let profile = fetch_profile(&ws.registration(), http)?;
        writeln!(console.out(), "client_id:        {}", profile.client_id)?;
        writeln!(console.out(), "organization:     {}", profile.organization)?;
        writeln!(
            console.out(),
            "client_details:   {}",
            profile.client_details
        )?;
        writeln!(console.out(), "contact_email:    {}", profile.contact_email)?;
        writeln!(console.out(), "status:           {}", profile.status)?;
        // Tags are assigned on the management server; display-only here.
        writeln!(
            console.out(),
            "tags:             {}",
            format_tags(profile.tags)
        )?;
        writeln!(
            console.out(),
            "reindex_pending:  {}",
            profile.reindex_pending
        )?;
        writeln!(
            console.out(),
            "capabilities:     {}",
            if profile.capabilities.is_empty() {
                "(none)".to_string()
            } else {
                profile.capabilities.join(", ")
            }
        )?;
        Ok(())
    }
}

/// `auth me --local`: who this workspace says it is, from disk. Lines rather
/// than output, so a test can assert what a caller keys on. Both halves are
/// required: a registration with no key names a client this machine cannot
/// sign for.
fn local_identity(ws: &PipetteWorkspace) -> Result<Vec<String>> {
    let store = ws.registration();
    // Read leniently: a field that no longer parses is the answer this command
    // reports, not a reason to refuse. See [`StoredRegistration`].
    let registration = store.require_stored()?;
    let private_key_hex = store.get_private_key()?.ok_or(Error::PrivateKeyMissing)?;
    Ok(vec![
        format!("client_id:        {}", registration.client_id),
        format!("server_url:       {}", registration.server_url),
        format!(
            "public_key:       {}",
            public_key_hex_from_private(&private_key_hex)?
        ),
    ])
}

/// `auth secret`: the private key hex alone on stdout, so it copies or pipes
/// whole. [`SECRET_LONG_ABOUT`] says why it warns: the server cannot rotate it.
fn secret(ws: &PipetteWorkspace, console: &mut dyn Console) -> Result<()> {
    // A key that cannot sign proves nothing to whoever asked for it, so an
    // unusable file is refused here rather than in a rejected paste.
    let private_key = ws
        .registration()
        .get_private_key()?
        .ok_or(Error::PrivateKeyMissing)?;
    writeln!(
        console.err(),
        "warning: this is the key that signs for this client; anyone who reads it can \
         sign as this client, and the server has no way to rotate it"
    )?;
    writeln!(console.out(), "{}", private_key.as_ref())?;
    Ok(())
}

#[derive(Args, Debug)]
pub struct ResetArgs {
    /// Delete unsubmitted remote results too. Required only when such results
    /// exist; a workspace with nothing pending resets without it.
    #[arg(long)]
    pub force: bool,
}

impl ResetArgs {
    /// Refuses when unsubmitted results would be lost unless `--force` is set;
    /// [`RESET_LONG_ABOUT`] lists what is removed and kept.
    pub fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> Result<()> {
        // Pending gates --force, so failing to enumerate it is fatal. The other
        // counts and the server label are display-only and best-effort: a
        // corrupt cache must not block `auth reset`, whose job is to delete it.
        let results = ws.results();
        let pending = results.list_ids(BenchmarkResultLocation::RemotePending)?;
        let synced = results
            .list_ids(BenchmarkResultLocation::RemoteSynced)
            .map(|ids| ids.len())
            .unwrap_or(0);
        let bench_store = ws.benchmarks();
        let benchmarks = bench_store
            .list(pipette_plan_types::BenchmarkSource::Remote)
            .map(|entries| entries.len())
            .unwrap_or(0);
        let registration = ws.registration();
        let held = registration.stored().ok().flatten();

        if !pending.is_empty() && !self.force {
            writeln!(console.out(), "auth reset would remove:")?;
            write_reset_targets(console, held.as_ref(), benchmarks, pending.len(), synced)?;
            writeln!(console.out(), "Local benchmarks and results are kept.")?;
            return Err(Error::ResetNeedsForce {
                unsubmitted: pending.len(),
            });
        }

        ws.clear_identity()?;
        let evals = ws.eval_completions();
        clear_remote_state(&bench_store, &results, &evals)?;

        writeln!(console.out(), "identity deleted; removed:")?;
        write_reset_targets(console, held.as_ref(), benchmarks, pending.len(), synced)?;
        Ok(())
    }
}

/// The "identity + N benchmarks + M results (K unsubmitted)" breakdown shared
/// by the refusal preview and the post-reset summary. Takes the lenient
/// registration so the label survives a `server_url` that no longer parses.
fn write_reset_targets(
    console: &mut dyn Console,
    held: Option<&StoredRegistration>,
    benchmarks: usize,
    pending: usize,
    synced: usize,
) -> Result<()> {
    match held {
        Some(registration) => writeln!(
            console.out(),
            "  • client identity (registered against {})",
            registration.server_url
        )?,
        None => writeln!(console.out(), "  • client identity")?,
    }
    writeln!(console.out(), "  • {benchmarks} remote benchmark(s)")?;
    writeln!(
        console.out(),
        "  • {} remote result(s) ({pending} unsubmitted)",
        pending + synced
    )?;
    writeln!(console.out(), "  • eval score-refresh state")?;
    Ok(())
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// Write the bundle to this path, created 0600, instead of stdout.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,
}

impl ExportArgs {
    pub fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> Result<()> {
        let (registration, private_key_hex) = ws.registration().require_identity()?;
        let bundle = IdentityBundle::new(&registration, &private_key_hex);
        // Derived before anything is printed: a key that cannot produce a public
        // key cannot sign either, so an export nothing could restore is refused.
        // Printed because it is the one value an operator can check against the
        // server's record without disclosing anything.
        let public_key = bundle.public_key_hex()?;
        let token = bundle.encode()?;

        // Summary on stderr, token on stdout: `auth export > bundle.txt`
        // captures the token alone, and the operator still sees the summary.
        writeln!(console.err(), "client_id:  {}", bundle.client_id)?;
        writeln!(console.err(), "server_url: {}", bundle.server_url)?;
        writeln!(console.err(), "public_key: {public_key}")?;
        writeln!(console.err(), "bytes:      {}", token.len())?;

        let Some(path) = self.out else {
            writeln!(
                console.err(),
                "warning: this token carries the client's private key in plaintext; \
                 --out <path> writes it to a 0600 file instead"
            )?;
            writeln!(console.out(), "{token}")?;
            return Ok(());
        };
        // The same creation-mode rule the private key itself is written under,
        // because this file carries that key.
        secret_file::write(&path, token.as_bytes())?;
        // Only unix gets the mode; elsewhere the writer falls back to the ACL
        // the parent directory grants (PIP-461), so claiming 0600 would be false.
        let mode = if cfg!(unix) { " (0600)" } else { "" };
        writeln!(console.err(), "bundle:     {}{mode}", path.display())?;
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ImportArgs {
    /// Read the bundle from this path; `-` (the default) reads stdin.
    ///
    /// Deliberately not a positional argument: a token on the command line is
    /// readable by anything that can list processes for as long as the command
    /// runs, and lands in shell history besides.
    #[arg(long, value_name = "PATH", default_value = "-")]
    pub from: String,

    /// Replace a different identity, dropping every cache derived from it.
    #[arg(long)]
    pub force: bool,
}

impl ImportArgs {
    pub fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> Result<()> {
        // Decoded in full before anything writes, so a token this build cannot
        // read leaves the workspace as it was.
        let bundle = IdentityBundle::decode(&read_bundle_token(&self.from)?)?;
        let public_key = bundle.public_key_hex()?;

        let store = ws.registration();
        // Read leniently, the way `auth reset` does: a registration this build
        // cannot parse is still an identity held, and `--force` is how it gets
        // replaced. Parsing here would fail on the very file the import fixes.
        let held = store.stored()?;
        let disposition = ImportDisposition::of(&store, held.as_ref(), &bundle, &public_key)?;
        let replaces = disposition == ImportDisposition::Replaces;
        let results = ws.results();
        if replaces && !self.force {
            // Priced, not just refused: the error says how many unsubmitted
            // results `--force` would discard.
            return Err(Error::ImportIdentityConflict {
                held: match &held {
                    Some(registration) => format!("the identity {}", registration.client_id),
                    None => "a signing key it holds no registration for".to_string(),
                },
                unsubmitted: results
                    .list_ids(BenchmarkResultLocation::RemotePending)
                    .map(|ids| ids.len())
                    .unwrap_or(0),
            });
        }

        if disposition == ImportDisposition::Unchanged {
            // No write at all, so a caller keying on `changed` sees no mtime
            // churn either.
            writeln!(
                console.out(),
                "{}",
                import_summary(&bundle, &public_key, false).join("\n")
            )?;
            return Ok(());
        }

        let bench_store = ws.benchmarks();
        let mut removed = None;
        if replaces {
            // Counted before the clear, and display-only as in `auth reset`: an
            // unreadable cache must not block the restore whose job is to delete it.
            removed = Some((
                bench_store
                    .list(pipette_plan_types::BenchmarkSource::Remote)
                    .map(|entries| entries.len())
                    .unwrap_or(0),
                results
                    .list_ids(BenchmarkResultLocation::RemotePending)
                    .map(|ids| ids.len())
                    .unwrap_or(0),
                results
                    .list_ids(BenchmarkResultLocation::RemoteSynced)
                    .map(|ids| ids.len())
                    .unwrap_or(0),
            ));
            // The caches go before the identity: this is the last point that
            // knows they belong to the outgoing client, and once the new
            // registration lands the next `sync` would submit them under its id.
            clear_remote_state(&bench_store, &results, &ws.eval_completions())?;
            // Then the outgoing identity files, so a failure below leaves a
            // half-restored workspace that a rerun finishes without --force.
            store.clear()?;
        }

        store.put_private_key(&bundle.private_key_hex)?;
        // Derived, because the bundle carries only the private half; `ensure_empty`
        // and `clear` both count this file as identity material.
        store.put_public_key(&public_key)?;
        store.put(&bundle.registration())?;

        writeln!(
            console.out(),
            "{}",
            import_summary(&bundle, &public_key, true).join("\n")
        )?;
        if let Some((benchmarks, pending, synced)) = removed {
            writeln!(
                console.out(),
                "replaced the identity this workspace held; removed:"
            )?;
            write_reset_targets(console, held.as_ref(), benchmarks, pending, synced)?;
        }
        Ok(())
    }
}

/// The bundle token named by `--from`: a file, or stdin for `-`.
fn read_bundle_token(from: &str) -> Result<String> {
    if from != "-" {
        let path = PathBuf::from(from);
        return std::fs::read_to_string(&path)
            .map(|raw| raw.trim().to_string())
            .map_err(|source| pipette_ops::Error::Io { path, source }.into());
    }
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .map_err(|source| pipette_ops::Error::Io {
            path: PathBuf::from("<stdin>"),
            source,
        })?;
    Ok(raw.trim().to_string())
}

/// The three fields `auth import` reports either way, plus whether it wrote:
/// `changed` is the line a provisioning step keys on.
fn import_summary(bundle: &IdentityBundle, public_key: &PublicKey, changed: bool) -> Vec<String> {
    vec![
        format!("client_id:  {}", bundle.client_id),
        format!("server_url: {}", bundle.server_url),
        format!("public_key: {public_key}"),
        // `changed`, not `status`: `auth me` already prints a `status` field,
        // and it means the server's approval state.
        format!("changed:    {}", if changed { "yes" } else { "no" }),
    ]
}

/// What an import would do to the workspace, decided from disk before it
/// writes anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImportDisposition {
    /// Every file this import would write already holds what the bundle
    /// carries.
    Unchanged,
    /// Nothing conflicting is held: a fresh workspace, one holding part of this
    /// same bundle, or one whose registration moved servers.
    Restores,
    /// A different client is held. Its caches go with it, so this needs
    /// `--force`.
    Replaces,
}

impl ImportDisposition {
    /// Both halves decide a conflict: the caches this gates are keyed to the
    /// `client_id`, and the key is what signs for it.
    /// A bundle matching what is already here is not a replacement, which lets
    /// a rerun finish a partial import with no `--force`.
    fn of(
        store: &ClientRegistrationStore,
        held: Option<&StoredRegistration>,
        bundle: &IdentityBundle,
        public_key: &PublicKey,
    ) -> Result<Self> {
        // A file that cannot sign is absent, not foreign (see
        // [`ClientRegistrationStore::get_private_key`]): it must neither demand
        // `--force` nor let the workspace read as complete below.
        let held_key = store.get_private_key()?;
        // `PrivateKey` has already trimmed and lowercased both, so whitespace or
        // casing cannot make a key read as foreign.
        let key_differs = held_key
            .as_ref()
            .is_some_and(|held| held != &bundle.private_key_hex);
        // The `client_id` alone: the same client against a different server is
        // a move, a write to make rather than an identity to replace. Compared
        // as the file holds it, so an id differing only by padding is foreign.
        let client_differs = held.is_some_and(|held| held.client_id != bundle.client_id.as_ref());
        if key_differs || client_differs {
            return Ok(Self::Replaces);
        }
        // The public key file counts too: an import interrupted before it leaves
        // it missing, and calling that unchanged would leave it missing for good.
        let complete = held_key.is_some()
            && held.is_some_and(|held| held.server_url == bundle.server_url.to_string())
            && store
                .get_public_key()?
                .is_some_and(|held| &held == public_key);
        Ok(if complete {
            Self::Unchanged
        } else {
            Self::Restores
        })
    }
}

/// The server `auth register` reaches when the operator names none: the only
/// defaulted server URL in the CLI. Afterwards the registration record is the
/// answer, so no later command reaches a host the operator never named.
const DEFAULT_SERVER_URL: &str = "https://collector.pipette.liquid.ai";

#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Management server URL. The only defaulted server URL in the CLI: once
    /// registered, the record answers for every later command.
    #[arg(long, default_value = DEFAULT_SERVER_URL)]
    pub server_url: ServerUrl,

    /// Organization operating this client, e.g. "LiquidAI"
    #[arg(long)]
    pub organization: String,

    #[arg(long, value_parser = parse_contact_email)]
    pub contact_email: ContactEmail,

    /// Human-readable client description
    #[arg(long)]
    pub client_details: String,

    /// Pre-auth key (`client_preauth_…`) that admits this client already
    /// approved. Also read from PIPETTE_PREAUTH_KEY (flag wins) to keep it out
    /// of shell history. Sent with the request; never written to disk.
    ///
    /// A device's own credential. An account's key registers a user with
    /// `pipette-ctl auth register`, which the server admits on another route
    /// against another record.
    #[arg(long, env = "PIPETTE_PREAUTH_KEY", hide_env_values = true)]
    pub preauth_key: Option<ClientPreauthKey>,

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
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        let registration = register(
            &ws.registration(),
            &ws.device(),
            &self.server_url,
            http,
            RegisterInput {
                organization: self.organization,
                contact_email: self.contact_email,
                client_details: self.client_details,
                device_name: self.device_name,
                device_form_factor: self.device_form_factor,
                preauth_key: self.preauth_key,
            },
        )?;
        writeln!(console.out(), "client_id:  {}", registration.client_id)?;
        writeln!(console.out(), "server_url: {}", registration.server_url)?;
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
    /// client.
    #[arg(long)]
    pub client_details: Option<String>,
}

impl SetDeviceArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        if self.device_name.is_none()
            && self.device_form_factor.is_none()
            && self.client_details.is_none()
        {
            return Err(crate::error::Error::SetDeviceEmpty);
        }

        // Device labels are local (device.json): no network, no registration.
        if self.device_name.is_some() || self.device_form_factor.is_some() {
            let labels = ws
                .device()
                .update(self.device_name, self.device_form_factor)?;
            writeln!(
                console.out(),
                "device_name:        {}",
                opt(&labels.device_name)
            )?;
            writeln!(
                console.out(),
                "device_form_factor: {}",
                opt(&labels.device_form_factor)
            )?;
        }

        if let Some(details) = self.client_details {
            let profile = update_client_details(&ws.registration(), http, details)?;
            writeln!(
                console.out(),
                "client_details:     {}",
                profile.client_details
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::Parser;

    use pipette_plan_types::ClientId;

    use super::*;
    use crate::test_support::TempWorkspace;
    use crate::test_support::{BrokenPipe, CapturedConsole};

    /// A value that can never address a server is refused at parse time, not
    /// after a keypair has been generated and a request attempted.
    #[rstest::rstest]
    #[case::bare_host("mgmt.example.com")]
    #[case::query("https://mgmt.example.com?tenant=a")]
    fn an_unusable_server_url_is_refused_at_parse_time(#[case] raw: &str) -> Result<()> {
        let error = crate::commands::Cli::try_parse_from([
            "pipette",
            "auth",
            "register",
            "--server-url",
            raw,
            "--organization",
            "LiquidAI",
            "--contact-email",
            "user@example.com",
            "--client-details",
            "ci box",
        ])
        .err()
        .ok_or_else(|| Error::Other(anyhow::anyhow!("expected a parse failure for {raw}")))?;
        assert!(
            error.to_string().contains("mgmt.example.com"),
            "the message names the offending value: {error}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::empty(&[], "(none)")]
    #[case::single(&["team/mobile"], "team/mobile")]
    #[case::sorted(&["team/mobile", "batch/2026-q3"], "batch/2026-q3, team/mobile")]
    fn format_tags_cases(#[case] tags: &[&str], #[case] expected: &str) {
        let tags = tags.iter().map(|s| s.to_string()).collect();
        assert_eq!(format_tags(tags), expected);
    }

    #[test]
    fn set_device_requires_at_least_one_field() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("set-device-empty")?;
        let args = SetDeviceArgs {
            device_name: None,
            device_form_factor: None,
            client_details: None,
        };
        let http = HttpClient::new("pipette-test/0")?;
        match args.execute(&tmp.ws, &http, &mut CapturedConsole::default()) {
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

        match (ResetArgs { force: false }).execute(&tmp.ws, &mut CapturedConsole::default()) {
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

        (ResetArgs { force: true }).execute(&tmp.ws, &mut CapturedConsole::default())?;

        assert!(
            !tmp.ws
                .results()
                .location_dir(BenchmarkResultLocation::RemotePending)
                .exists()
        );
        Ok(())
    }

    /// Any 32 bytes are a valid Ed25519 seed.
    const KEY: &str = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";
    const OTHER_KEY: &str = "1122334455667788990011223344556677889900112233445566778899001122";

    fn key(hex: &str) -> anyhow::Result<pipette_mgmt_client::PrivateKey> {
        Ok(pipette_mgmt_client::PrivateKey::try_new(hex)?)
    }

    fn seed_identity(ws: &PipetteWorkspace, client_id: &str, hex: &str) -> anyhow::Result<()> {
        let registration = ws.registration();
        registration.put_private_key(&key(hex)?)?;
        registration.put(&crate::identity::IdentityRegistration {
            client_id: ClientId::try_new(client_id)?,
            server_url: ServerUrl::try_new("https://collector.example.test")?,
        })?;
        Ok(())
    }

    fn export_to_file(ws: &PipetteWorkspace, label: &str) -> anyhow::Result<PathBuf> {
        let path = ws.root().join(format!("{label}.bundle"));
        (ExportArgs {
            out: Some(path.clone()),
        })
        .execute(ws, &mut CapturedConsole::default())?;
        Ok(path)
    }

    fn disposition_of(ws: &PipetteWorkspace, path: &Path) -> anyhow::Result<ImportDisposition> {
        let bundle = IdentityBundle::decode(&std::fs::read_to_string(path)?)?;
        let registration = ws.registration();
        let held = registration.stored()?;
        Ok(ImportDisposition::of(
            &registration,
            held.as_ref(),
            &bundle,
            &bundle.public_key_hex()?,
        )?)
    }

    fn import_from(ws: &PipetteWorkspace, path: &Path, force: bool) -> Result<()> {
        (ImportArgs {
            from: path.display().to_string(),
            force,
        })
        .execute(ws, &mut CapturedConsole::default())
    }

    #[test]
    fn export_then_import_moves_the_identity_to_another_workspace() -> anyhow::Result<()> {
        let source = TempWorkspace::new("export-source")?;
        let target = TempWorkspace::new("export-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;

        let bundle = export_to_file(&source.ws, "identity")?;
        import_from(&target.ws, &bundle, false)?;

        let restored = target.ws.registration();
        assert_eq!(restored.require()?.client_id.as_ref(), "ev1_abc");
        assert_eq!(restored.get_private_key()?, Some(key(KEY)?));
        // `ensure_empty` counts the derived public half, as for a registered one.
        assert!(matches!(
            restored.ensure_empty(),
            Err(Error::IdentityExists(_))
        ));
        assert_eq!(restored.signing_identity()?.client_id.as_ref(), "ev1_abc");
        Ok(())
    }

    #[test]
    fn import_leaves_the_importing_machine_labels_alone() -> anyhow::Result<()> {
        let source = TempWorkspace::new("labels-source")?;
        let target = TempWorkspace::new("labels-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        source.ws.device().put(&crate::identity::DeviceLabels {
            device_name: Some("source box".to_string().into()),
            device_form_factor: Some(DeviceFormFactor::Server),
        })?;
        target.ws.device().put(&crate::identity::DeviceLabels {
            device_name: Some("target box".to_string().into()),
            device_form_factor: Some(DeviceFormFactor::Laptop),
        })?;

        import_from(&target.ws, &export_to_file(&source.ws, "identity")?, false)?;

        let labels = target.ws.device().get()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("target box")
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Laptop));
        Ok(())
    }

    #[test]
    fn import_refuses_a_different_identity_without_force() -> anyhow::Result<()> {
        let source = TempWorkspace::new("conflict-source")?;
        let target = TempWorkspace::new("conflict-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        seed_identity(&target.ws, "ev1_held", OTHER_KEY)?;
        add_pending(&target.ws, "r-1")?;
        let bundle = export_to_file(&source.ws, "identity")?;

        match import_from(&target.ws, &bundle, false) {
            Err(Error::ImportIdentityConflict { held, unsubmitted }) => {
                assert!(held.contains("ev1_held"), "unexpected held: {held}");
                assert_eq!(unsubmitted, 1);
            }
            other => anyhow::bail!("expected ImportIdentityConflict, got {other:?}"),
        }
        let registration = target.ws.registration();
        assert_eq!(registration.require()?.client_id.as_ref(), "ev1_held");
        assert_eq!(registration.get_private_key()?, Some(key(OTHER_KEY)?));
        Ok(())
    }

    /// A rerun after an import that failed partway finds the key that same
    /// bundle wrote.
    #[test]
    fn import_of_the_held_identity_needs_no_force() -> anyhow::Result<()> {
        let source = TempWorkspace::new("rerun-source")?;
        let target = TempWorkspace::new("rerun-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        let bundle = export_to_file(&source.ws, "identity")?;

        // What a partial import leaves: the key landed, the registration did not.
        target.ws.registration().put_private_key(&key(KEY)?)?;
        import_from(&target.ws, &bundle, false)?;

        assert_eq!(
            target.ws.registration().require()?.client_id.as_ref(),
            "ev1_abc"
        );
        import_from(&target.ws, &bundle, false)?;
        Ok(())
    }

    #[test]
    fn forced_import_replaces_the_identity_and_drops_its_caches() -> anyhow::Result<()> {
        let source = TempWorkspace::new("force-source")?;
        let target = TempWorkspace::new("force-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        seed_identity(&target.ws, "ev1_held", OTHER_KEY)?;
        add_pending(&target.ws, "r-1")?;
        let bundle = export_to_file(&source.ws, "identity")?;

        import_from(&target.ws, &bundle, true)?;

        let registration = target.ws.registration();
        assert_eq!(registration.require()?.client_id.as_ref(), "ev1_abc");
        assert_eq!(registration.get_private_key()?, Some(key(KEY)?));
        assert!(
            !target
                .ws
                .results()
                .location_dir(BenchmarkResultLocation::RemotePending)
                .exists()
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::not_base64url("not a bundle")]
    #[case::not_a_record("eyJ1bnJlbGF0ZWQiOiB0cnVlfQ")]
    fn a_bundle_this_build_cannot_read_writes_nothing(#[case] token: &str) -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("bad-bundle")?;
        let path = tmp.ws.root().join("bad.bundle");
        std::fs::write(&path, token)?;

        assert!(import_from(&tmp.ws, &path, false).is_err());
        assert!(tmp.ws.registration().get_private_key()?.is_none());
        tmp.ws.registration().ensure_empty()?;
        Ok(())
    }

    /// Asserted on the disposition rather than the key file's mtime: `Unchanged`
    /// returns before any write, and mtime granularity is the filesystem's.
    #[test]
    fn a_second_import_of_the_same_bundle_changes_nothing() -> anyhow::Result<()> {
        let source = TempWorkspace::new("unchanged-source")?;
        let target = TempWorkspace::new("unchanged-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        let path = export_to_file(&source.ws, "identity")?;
        import_from(&target.ws, &path, false)?;

        assert_eq!(
            disposition_of(&target.ws, &path)?,
            ImportDisposition::Unchanged
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::wrote(true, "changed:    yes")]
    #[case::no_op(false, "changed:    no")]
    fn the_summary_reports_whether_the_import_wrote(
        #[case] changed: bool,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let bundle = IdentityBundle::new(
            &crate::identity::IdentityRegistration {
                client_id: ClientId::try_new("ev1_abc")?,
                server_url: ServerUrl::try_new("https://collector.example.test")?,
            },
            &key(KEY)?,
        );
        let summary = import_summary(&bundle, &PublicKey::try_new("ca7e".repeat(16))?, changed);
        assert_eq!(summary.last().map(String::as_str), Some(expected));
        Ok(())
    }

    /// Blank is what a crash mid-write leaves; the other case is a file edited
    /// by hand, which a text comparison would call foreign.
    #[rstest::rstest]
    #[case::blank("")]
    #[case::not_a_key("not a key\n")]
    fn an_unusable_key_file_is_repaired_without_force(#[case] stored: &str) -> anyhow::Result<()> {
        let source = TempWorkspace::new("unusable-source")?;
        let target = TempWorkspace::new("unusable-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        let path = export_to_file(&source.ws, "identity")?;
        import_from(&target.ws, &path, false)?;
        // Written as bytes, because the store's own setter takes a key.
        std::fs::write(target.ws.root().join("identity").join("id_ed25519"), stored)?;

        assert_eq!(
            disposition_of(&target.ws, &path)?,
            ImportDisposition::Restores
        );
        import_from(&target.ws, &path, false)?;

        assert_eq!(target.ws.registration().get_private_key()?, Some(key(KEY)?));
        Ok(())
    }

    #[rstest::rstest]
    #[case::no_public_key("id_ed25519.pub")]
    #[case::no_registration("registration.json")]
    fn a_rerun_finishes_a_workspace_missing_a_file(#[case] removed: &str) -> anyhow::Result<()> {
        let source = TempWorkspace::new("partial-source")?;
        let target = TempWorkspace::new("partial-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        let bundle = export_to_file(&source.ws, "identity")?;
        import_from(&target.ws, &bundle, false)?;
        std::fs::remove_file(target.ws.root().join("identity").join(removed))?;

        import_from(&target.ws, &bundle, false)?;

        assert!(target.ws.root().join("identity").join(removed).exists());
        Ok(())
    }

    #[test]
    fn the_same_client_moving_servers_needs_no_force() -> anyhow::Result<()> {
        let source = TempWorkspace::new("moved-source")?;
        let target = TempWorkspace::new("moved-target")?;
        seed_identity(&source.ws, "ev1_abc", KEY)?;
        let bundle = export_to_file(&source.ws, "identity")?;
        let registration = target.ws.registration();
        registration.put_private_key(&key(KEY)?)?;
        registration.put(&crate::identity::IdentityRegistration {
            client_id: ClientId::try_new("ev1_abc")?,
            server_url: ServerUrl::try_new("https://old-collector.example.test")?,
        })?;

        import_from(&target.ws, &bundle, false)?;

        assert_eq!(
            registration.require()?.server_url.to_string(),
            "https://collector.example.test"
        );
        Ok(())
    }

    /// No request is made. The public key is the derived one, not the signing
    /// key it is derived from.
    #[test]
    fn me_local_reports_the_identity_on_disk() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("me-local")?;
        seed_identity(&tmp.ws, "ev1_abc", KEY)?;

        let lines = local_identity(&tmp.ws)?;

        assert_eq!(
            lines,
            vec![
                "client_id:        ev1_abc".to_string(),
                "server_url:       https://collector.example.test".to_string(),
                format!(
                    "public_key:       {}",
                    pipette_mgmt_client::public_key_hex_from_private(&key(KEY)?)?
                ),
            ]
        );
        assert!(
            !lines.concat().contains(KEY),
            "the signing key must not print"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::no_identity(false)]
    #[case::registration_without_a_key(true)]
    fn me_local_refuses_half_an_identity(#[case] with_registration: bool) -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("me-local-half")?;
        if with_registration {
            tmp.ws
                .registration()
                .put(&crate::identity::IdentityRegistration {
                    client_id: ClientId::try_new("ev1_abc")?,
                    server_url: ServerUrl::try_new("https://collector.example.test")?,
                })?;
        }
        match (local_identity(&tmp.ws), with_registration) {
            (Err(Error::PrivateKeyMissing), true) => Ok(()),
            (Err(Error::RegistrationMissing(_)), false) => Ok(()),
            (other, _) => anyhow::bail!("unexpected: {other:?}"),
        }
    }

    /// The command an operator reaches for when the workspace is wrong has to
    /// report the wrong value rather than refuse over it. Asserted at the
    /// command, because the store-level test passes either way.
    #[test]
    fn me_local_reports_a_server_url_that_no_longer_parses() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("me-local-unparsable")?;
        let store = tmp.ws.registration();
        store.put_private_key(&key(KEY)?)?;
        // Written past `put`, which the typed field would not let hold this.
        std::fs::write(
            tmp.ws.root().join("identity").join("registration.json"),
            serde_json::to_vec(&serde_json::json!({
                "client_id": "ev1_abc",
                "server_url": "collector.example.test",
            }))?,
        )?;

        let lines = local_identity(&tmp.ws)?;
        assert!(
            lines
                .iter()
                .any(|l| l.contains("client_id:        ev1_abc")),
            "expected the stored client_id, got {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("server_url:       collector.example.test")),
            "expected the unparsable URL reported as held, got {lines:?}"
        );
        Ok(())
    }

    #[test]
    fn export_refuses_half_an_identity() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("export-half")?;
        // A registration with no key cannot sign.
        tmp.ws
            .registration()
            .put(&crate::identity::IdentityRegistration {
                client_id: ClientId::try_new("ev1_abc")?,
                server_url: ServerUrl::try_new("https://collector.example.test")?,
            })?;
        match (ExportArgs { out: None }).execute(&tmp.ws, &mut CapturedConsole::default()) {
            Err(Error::PrivateKeyMissing) => Ok(()),
            other => anyhow::bail!("expected PrivateKeyMissing, got {other:?}"),
        }
    }

    #[test]
    fn secret_without_a_key_says_so() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("secret-missing")?;
        match secret(&tmp.ws, &mut CapturedConsole::default()) {
            Err(Error::PrivateKeyMissing) => Ok(()),
            other => anyhow::bail!("expected PrivateKeyMissing, got {other:?}"),
        }
    }

    /// The warning rides on stderr, where it does not corrupt the value.
    #[test]
    fn secret_pipes_the_key_alone_and_warns_beside_it() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("secret-warning")?;
        seed_identity(&tmp.ws, "ev1_abc", KEY)?;
        let mut captured = CapturedConsole::default();
        secret(&tmp.ws, &mut captured)?;

        assert_eq!(captured.out_text(), format!("{KEY}\n"));
        assert!(
            captured.err_text().contains("no way to rotate it"),
            "got {:?}",
            captured.err_text()
        );
        Ok(())
    }

    /// Exiting zero on a closed pipe would read as "the key is on your
    /// clipboard" when nothing was written.
    #[test]
    fn a_closed_pipe_fails_the_command() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("secret-broken-pipe")?;
        seed_identity(&tmp.ws, "ev1_abc", KEY)?;
        let err = secret(&tmp.ws, &mut BrokenPipe::default())
            .err()
            .ok_or_else(|| anyhow::anyhow!("a closed pipe should fail the command"))?;
        assert!(matches!(err, Error::Console(_)), "got {err:?}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn exported_bundle_file_is_owner_only() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp = TempWorkspace::new("export-mode")?;
        seed_identity(&tmp.ws, "ev1_abc", KEY)?;
        let path = export_to_file(&tmp.ws, "identity")?;

        let mode = std::fs::metadata(&path)?.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {mode:o}");
        Ok(())
    }

    /// Every file under `identity/` goes, across all three stores.
    #[test]
    fn reset_without_pending_proceeds_without_force() -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("reset-empty")?;
        seed_identity(&tmp.ws, "ev1_abc", KEY)?;
        tmp.ws.device().put(&crate::identity::DeviceLabels {
            device_name: Some("lab box".to_string().into()),
            device_form_factor: None,
        })?;
        tmp.ws.settings().put(&crate::identity::ClientSettings {
            storage_quota_bytes: 4096,
        })?;
        std::fs::create_dir_all(tmp.ws.root().join("benchmarks/remote"))?;

        (ResetArgs { force: false }).execute(&tmp.ws, &mut CapturedConsole::default())?;

        assert!(tmp.ws.registration().get()?.is_none());
        assert!(tmp.ws.registration().get_private_key()?.is_none());
        assert_eq!(
            tmp.ws.device().get()?,
            crate::identity::DeviceLabels::default()
        );
        assert!(tmp.ws.settings().get()?.is_none());
        assert!(!tmp.ws.root().join("benchmarks/remote").exists());
        Ok(())
    }

    #[test]
    fn the_registered_lifecycle_dispatches_end_to_end() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let tmp = TempWorkspace::new("auth-lifecycle")?;
        let server = MockServer::start();
        let http = HttpClient::new("pipette-test/0")?;
        let run = |command: AuthCommand, captured: &mut CapturedConsole| {
            AuthArgs { command }.execute(&tmp.ws, &http, captured)
        };

        let register = server.mock(|when, then| {
            when.method(POST).path("/clients/register");
            then.status(200)
                .body(r#"{"client_id":"c-1","status":"approved"}"#);
        });
        let mut registered = CapturedConsole::default();
        run(
            AuthCommand::Register(RegisterArgs {
                server_url: ServerUrl::try_new(server.base_url())?,
                organization: "LiquidAI".to_owned(),
                contact_email: ContactEmail::try_new("user@example.com")?,
                client_details: "ci box".to_owned(),
                preauth_key: None,
                device_name: Some("lab box".to_owned()),
                device_form_factor: None,
            }),
            &mut registered,
        )?;
        register.assert();
        assert_eq!(
            registered.out_text(),
            format!("client_id:  c-1\nserver_url: {}\n", server.base_url())
        );
        let registration = tmp.ws.registration().require()?;
        assert_eq!(registration.client_id.as_ref(), "c-1");

        let local = local_identity(&tmp.ws)?;
        assert_eq!(local[0], "client_id:        c-1");
        assert_eq!(local[1], format!("server_url:       {}", server.base_url()));
        assert!(local[2].starts_with("public_key:       "));

        let me = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(
                r#"{"client_id":"c-1","organization":"LiquidAI","client_details":"ci box","contact_email":"user@example.com","status":"approved","tags":["team/ci"],"capabilities":["docker"]}"#,
            );
        });
        let mut profile = CapturedConsole::default();
        run(AuthCommand::Me(MeArgs { local: false }), &mut profile)?;
        me.assert();
        let shown = profile.out_text();
        assert!(shown.contains("client_id:        c-1"), "got {shown}");
        assert!(shown.contains("status:           approved"), "got {shown}");
        assert!(shown.contains("tags:             team/ci"), "got {shown}");
        assert!(shown.contains("reindex_pending:  false"), "got {shown}");
        assert!(shown.contains("capabilities:     docker"), "got {shown}");

        let patch = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH)
                .path("/clients/me")
                .body_includes("rack 3");
            then.status(200).body(
                r#"{"client_id":"c-1","organization":"LiquidAI","client_details":"rack 3","contact_email":"user@example.com","status":"approved"}"#,
            );
        });
        let mut relabelled = CapturedConsole::default();
        run(
            AuthCommand::SetDevice(SetDeviceArgs {
                device_name: None,
                device_form_factor: Some(DeviceFormFactor::Server),
                client_details: Some("rack 3".to_owned()),
            }),
            &mut relabelled,
        )?;
        patch.assert();
        assert_eq!(
            relabelled.out_text(),
            "device_name:        lab box\n\
             device_form_factor: server\n\
             client_details:     rack 3\n"
        );
        let labels = tmp.ws.device().get()?;
        assert_eq!(
            labels.device_name.as_ref().map(AsRef::as_ref),
            Some("lab box"),
            "an unset label keeps the value register wrote"
        );
        assert_eq!(labels.device_form_factor, Some(DeviceFormFactor::Server));

        let mut reset = CapturedConsole::default();
        run(AuthCommand::Reset(ResetArgs { force: false }), &mut reset)?;
        assert!(tmp.ws.registration().get()?.is_none());
        let torn_down = reset.out_text();
        assert!(
            torn_down.starts_with("identity deleted; removed:"),
            "got {torn_down}"
        );
        assert!(
            torn_down.contains(&format!("registered against {}", server.base_url())),
            "got {torn_down}"
        );
        Ok(())
    }

    /// Each names the missing piece rather than printing an empty profile.
    #[rstest::rstest]
    #[case::me_local(AuthCommand::Me(MeArgs { local: true }))]
    #[case::me_remote(AuthCommand::Me(MeArgs { local: false }))]
    #[case::secret(AuthCommand::Secret)]
    #[case::export(AuthCommand::Export(ExportArgs { out: None }))]
    #[case::set_details(AuthCommand::SetDevice(SetDeviceArgs {
        device_name: None,
        device_form_factor: None,
        client_details: Some("rack 3".to_owned()),
    }))]
    fn identity_commands_refuse_an_unregistered_workspace(
        #[case] command: AuthCommand,
    ) -> anyhow::Result<()> {
        let tmp = TempWorkspace::new("auth-unregistered")?;
        let http = HttpClient::new("pipette-test/0")?;
        let err = AuthArgs { command }
            .execute(&tmp.ws, &http, &mut CapturedConsole::default())
            .err()
            .ok_or_else(|| anyhow::anyhow!("should refuse an unregistered workspace"))?;
        assert!(
            matches!(
                err,
                Error::RegistrationMissing(_) | Error::PrivateKeyMissing
            ),
            "{err:?}"
        );
        Ok(())
    }
}
