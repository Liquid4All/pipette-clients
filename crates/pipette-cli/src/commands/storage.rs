//! `storage` — what the artifact stores occupy, and reclaiming it.
//!
//! Both subcommands read the store roots through [`pipette_artifacts::quota`],
//! which classifies anything without a readable manifest as garbage. That is
//! what keeps them usable on a store a version bump stranded: `models list`
//! errors on such an entry, `storage gc` deletes it.
//!
//! Sizes cross the crate boundary as raw byte counts; the formatting is here.

use clap::{Args, Subcommand};
use tabled::Tabled;
use time::format_description::well_known::Rfc3339;

use pipette_artifacts::quota::{self, EntryKind, StorageEntry, SweepPins};

use crate::commands::print_table_or;
use crate::workspace::PipetteWorkspace;

/// Inspect and reclaim local artifact disk usage
#[derive(Args, Debug)]
pub struct StorageArgs {
    #[command(subcommand)]
    pub command: StorageCommand,
}

#[derive(Subcommand, Debug)]
pub enum StorageCommand {
    /// Show artifact disk usage against the quota
    Status,
    /// Reclaim disk: garbage first, then least-recently-used artifacts
    Gc(GcArgs),
}

/// `storage gc` input.
#[derive(Args, Debug)]
pub struct GcArgs {
    /// Report what would be reclaimed without deleting anything
    #[arg(long)]
    pub dry_run: bool,
}

impl StorageArgs {
    pub fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        match self.command {
            StorageCommand::Status => status(ws),
            StorageCommand::Gc(args) => args.execute(ws),
        }
    }
}

/// One row of `storage status`, in the order `gc` would reclaim them.
#[derive(Tabled)]
struct StorageRow {
    #[tabled(rename = "KIND")]
    kind: String,
    #[tabled(rename = "ENTRY")]
    entry: String,
    #[tabled(rename = "SIZE")]
    size: String,
    #[tabled(rename = "LAST USED")]
    last_used: String,
    #[tabled(rename = "NOTE")]
    note: String,
}

impl StorageRow {
    fn new(entry: &StorageEntry) -> Self {
        Self {
            kind: kind_label(&entry.kind).to_owned(),
            entry: entry.label.clone(),
            size: human_bytes(entry.size_bytes),
            last_used: last_used_label(&entry.kind),
            note: match &entry.kind {
                EntryKind::Garbage { reason } => reason.clone(),
                // Docker is the one live entry whose row needs explaining: it
                // measures ~0 and `gc` will never pick it.
                EntryKind::Runtime {
                    evictable: false, ..
                } => "image lives in the docker daemon".to_owned(),
                _ => String::new(),
            },
        }
    }
}

fn status(ws: &PipetteWorkspace) -> anyhow::Result<()> {
    let quota = ws.storage_quota();
    let survey = survey_stores(ws);
    println!(
        "quota: {} ({})",
        human_bytes(quota.bytes),
        quota.source.label()
    );
    println!(
        "used:  {} ({}%)",
        human_bytes(survey.used_bytes),
        percent_of(survey.used_bytes, quota.bytes)
    );
    println!(
        "free:  {}",
        human_bytes(quota.bytes.saturating_sub(survey.used_bytes))
    );
    println!();

    let rows: Vec<StorageRow> = survey.entries.iter().map(StorageRow::new).collect();
    print_table_or(&rows, "Nothing stored.");
    Ok(())
}

impl GcArgs {
    fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        let quota = ws.storage_quota();
        let survey = survey_stores(ws);
        // Nothing is in flight during `gc`, so nothing is pinned.
        let plan = quota::plan(&survey, quota.bytes, &SweepPins::default());

        if plan.evictions.is_empty() && plan.still_over_by_bytes.is_none() {
            println!(
                "Already within quota ({} of {}) with nothing to reclaim.",
                human_bytes(survey.used_bytes),
                human_bytes(quota.bytes)
            );
            return Ok(());
        }

        let freed_bytes = if self.dry_run {
            plan.evictions
                .iter()
                .for_each(|entry| println!("{}", eviction_line(entry, true)));
            plan.freed_bytes
        } else {
            let report = quota::apply_sweep(&plan);
            report
                .removed
                .iter()
                .for_each(|entry| println!("{}", eviction_line(entry, false)));
            report.failed.iter().for_each(|(entry, reason)| {
                println!("could not reclaim {}: {reason}", entry.path.display());
            });
            report.freed_bytes
        };

        let (freed, remaining) = if self.dry_run {
            ("would free", "leaving")
        } else {
            ("freed", "using")
        };
        let remaining_bytes = survey.used_bytes.saturating_sub(freed_bytes);
        println!(
            "{freed} {}; {remaining} {} of {}",
            human_bytes(freed_bytes),
            human_bytes(remaining_bytes),
            human_bytes(quota.bytes)
        );
        if let Some(warning) = still_over_warning(remaining_bytes, quota.bytes) {
            println!("{warning}");
        }
        Ok(())
    }
}

fn survey_stores(ws: &PipetteWorkspace) -> quota::StorageSurvey {
    quota::survey(ws.models().models_dir(), ws.runtimes().runtimes_dir())
}

/// The warning for a store still over quota after a sweep, or `None` when it
/// fits. Over quota with no candidate left is a warning, never a failure — `gc`
/// does not fail over disk bookkeeping any more than a run does.
///
/// Takes what the sweep actually left rather than what the plan predicted:
/// `apply_sweep` tolerates a failed unlink, so a plan that expected to fit can
/// still leave the store over quota, and `still_over_by_bytes` would say nothing.
fn still_over_warning(remaining_bytes: u64, quota_bytes: u64) -> Option<String> {
    let over = remaining_bytes.saturating_sub(quota_bytes);
    (over > 0).then(|| {
        format!(
            "warning: still {} over quota; removing the remaining entries would free nothing",
            human_bytes(over)
        )
    })
}

/// One reported eviction, verb first so no delete is silent. A plan says what
/// it would do; a finished sweep says what it did.
fn eviction_line(entry: &StorageEntry, dry_run: bool) -> String {
    match &entry.kind {
        EntryKind::Garbage { reason } => format!(
            "{} garbage {} ({}, {reason})",
            if dry_run {
                "would reclaim"
            } else {
                "reclaimed"
            },
            entry.label,
            human_bytes(entry.size_bytes)
        ),
        kind => format!(
            "{} {} {} ({}, last used {})",
            if dry_run { "would evict" } else { "evicted" },
            kind_label(kind),
            entry.label,
            human_bytes(entry.size_bytes),
            last_used_label(kind)
        ),
    }
}

fn kind_label(kind: &EntryKind) -> &'static str {
    match kind {
        EntryKind::Garbage { .. } => "garbage",
        EntryKind::Model { .. } => "model",
        EntryKind::Runtime {
            evictable: true, ..
        } => "runtime",
        EntryKind::Runtime {
            evictable: false, ..
        } => "runtime (docker)",
    }
}

/// RFC 3339, or `-` for garbage. An unformattable timestamp also renders `-`:
/// a report is not worth failing over.
fn last_used_label(kind: &EntryKind) -> String {
    kind.last_used_at()
        .and_then(|at| at.format(&Rfc3339).ok())
        .unwrap_or_else(|| "-".to_owned())
}

/// Share of the quota in use. Reads above 100 when the store is over.
fn percent_of(used_bytes: u64, quota_bytes: u64) -> u64 {
    used_bytes
        .saturating_mul(100)
        .checked_div(quota_bytes)
        .unwrap_or(100)
}

/// IEC size with one decimal (`4.1 GiB`). Raw bytes below 1 KiB.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use rstest::rstest;

    use pipette_plan_types::{GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, RepoSubpath};

    use super::*;
    use crate::workspace::test_support::TempWorkspace;

    /// A stored model, published through the real store so its manifest is one
    /// this build reads back as a live entry.
    fn store_model(ws: &PipetteWorkspace, repo_name: &str) -> anyhow::Result<()> {
        let declared = pipette_plan_types::Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("meta".to_owned())?,
                    repo_name: HfRepoName::try_new(repo_name.to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        });
        ws.models().ensure(&declared, |_declared, into| {
            let pipette_plan_types::Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path },
            }) = into
            else {
                anyhow::bail!("the stub fetch only handles gguf-text");
            };
            let path = Path::new(path.as_ref());
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, vec![0u8; 4096])?;
            Ok(())
        })?;
        Ok(())
    }

    fn orphan(ws: &PipetteWorkspace) -> anyhow::Result<std::path::PathBuf> {
        let path = ws.models().models_dir().join("no-manifest");
        fs::create_dir_all(&path)?;
        fs::write(path.join("leftover.bin"), vec![0u8; 4096])?;
        Ok(path)
    }

    #[rstest]
    #[case(0, "0 B")]
    #[case(1023, "1023 B")]
    #[case(1024, "1.0 KiB")]
    #[case(1536, "1.5 KiB")]
    #[case(4 * 1024 * 1024, "4.0 MiB")]
    #[case(200 * 1024 * 1024 * 1024, "200.0 GiB")]
    #[case(3 * 1024 * 1024 * 1024 * 1024, "3.0 TiB")]
    fn human_bytes_formats_each_unit(#[case] bytes: u64, #[case] expected: &str) {
        assert_eq!(human_bytes(bytes), expected);
    }

    #[test]
    fn a_dry_run_line_reads_as_a_plan_and_a_real_one_as_a_fact() {
        let entry = StorageEntry {
            path: std::path::PathBuf::from("/models/junk"),
            label: "junk".to_owned(),
            size_bytes: 2048,
            kind: EntryKind::Garbage {
                reason: "no manifest".to_owned(),
            },
        };
        assert_eq!(
            eviction_line(&entry, true),
            "would reclaim garbage junk (2.0 KiB, no manifest)"
        );
        assert_eq!(
            eviction_line(&entry, false),
            "reclaimed garbage junk (2.0 KiB, no manifest)"
        );
    }

    #[test]
    fn status_rows_read_as_the_eviction_queue() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-status")?;
        store_model(&tw.ws, "llama")?;
        orphan(&tw.ws)?;

        let survey = survey_stores(&tw.ws);
        let rows: Vec<StorageRow> = survey.entries.iter().map(StorageRow::new).collect();

        let kinds: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(kinds, vec!["garbage", "model"]);
        assert_eq!(rows[0].note, "no manifest");
        assert_eq!(rows[0].last_used, "-");
        assert!(rows[1].last_used.ends_with('Z'), "{}", rows[1].last_used);
        Ok(())
    }

    #[test]
    fn gc_dry_run_deletes_nothing() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-gc-dry")?;
        store_model(&tw.ws, "llama")?;
        let orphan = orphan(&tw.ws)?;
        let total = survey_stores(&tw.ws).used_bytes;

        GcArgs { dry_run: true }.execute(&tw.reopen_with_quota(total - 1)?)?;

        assert!(orphan.exists(), "a dry run reclaims nothing");
        assert_eq!(survey_stores(&tw.ws).used_bytes, total);
        Ok(())
    }

    #[test]
    fn gc_reclaims_garbage_before_a_live_model() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-gc")?;
        store_model(&tw.ws, "llama")?;
        let orphan = orphan(&tw.ws)?;
        let total = survey_stores(&tw.ws).used_bytes;

        GcArgs { dry_run: false }.execute(&tw.reopen_with_quota(total - 1)?)?;

        assert!(!orphan.exists());
        assert_eq!(
            tw.ws.models().list()?.len(),
            1,
            "the live model survives while garbage covers the overage"
        );
        Ok(())
    }

    /// The recovery path for a store a manifest version bump stranded: such a
    /// store is under quota, so `gc` has to take its garbage anyway.
    #[test]
    fn gc_reclaims_garbage_even_under_quota() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-gc-under")?;
        store_model(&tw.ws, "llama")?;
        let orphan = orphan(&tw.ws)?;

        GcArgs { dry_run: false }.execute(&tw.reopen_with_quota(u64::MAX)?)?;

        assert!(!orphan.exists(), "garbage is free to drop at any usage");
        assert_eq!(
            tw.ws.models().list()?.len(),
            1,
            "a live model under quota is not touched"
        );
        Ok(())
    }

    /// Keyed on what the sweep left, not what it planned: a failed unlink leaves
    /// the store over quota even when the plan expected to fit, and that case has
    /// to warn.
    #[rstest]
    #[case::nothing_freed_still_over(12_000, 10_000, true)]
    #[case::freed_enough_to_fit(9_000, 10_000, false)]
    #[case::exactly_full(10_000, 10_000, false)]
    fn warns_from_what_the_sweep_left(
        #[case] remaining_bytes: u64,
        #[case] quota_bytes: u64,
        #[case] expect_warning: bool,
    ) {
        let warning = still_over_warning(remaining_bytes, quota_bytes);
        assert_eq!(warning.is_some(), expect_warning);
        if let Some(warning) = warning {
            assert!(warning.contains("still 2.0 KiB over quota"));
        }
    }
}
