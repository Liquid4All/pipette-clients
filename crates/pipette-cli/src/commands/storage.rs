//! `storage`: what the artifact stores occupy, and reclaiming it.
//! [`pipette_artifacts::quota`] classifies anything without a readable manifest
//! as garbage, which keeps both subcommands usable on a store a version bump
//! stranded: `models list` errors on such an entry, `storage gc` deletes it.

use clap::{Args, Subcommand};
use tabled::Tabled;
use time::format_description::well_known::Rfc3339;

use pipette_artifacts::quota::{self, EntryKind, StorageEntry, SweepPins};

use crate::commands::write_table_or;
use crate::console::Console;
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

#[derive(Args, Debug)]
pub struct GcArgs {
    /// Report what would be reclaimed without deleting anything
    #[arg(long)]
    pub dry_run: bool,
}

impl StorageArgs {
    pub fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
        match self.command {
            StorageCommand::Status => status(ws, console),
            StorageCommand::Gc(args) => args.execute(ws, console),
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
                EntryKind::Model { .. } | EntryKind::Runtime { .. } => String::new(),
            },
        }
    }
}

fn status(ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
    let quota = ws.storage_quota();
    let survey = survey_stores(ws);
    writeln!(
        console.out(),
        "quota: {} ({})",
        human_bytes(quota.bytes),
        quota.source.label()
    )?;
    writeln!(
        console.out(),
        "used:  {} ({}%)",
        human_bytes(survey.used_bytes),
        percent_of(survey.used_bytes, quota.bytes)
    )?;
    writeln!(
        console.out(),
        "free:  {}",
        human_bytes(quota.bytes.saturating_sub(survey.used_bytes))
    )?;
    writeln!(console.out())?;

    let rows: Vec<StorageRow> = survey.entries.iter().map(StorageRow::new).collect();
    write_table_or(console.out(), &rows, "Nothing stored.")?;
    Ok(())
}

impl GcArgs {
    fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
        let quota = ws.storage_quota();
        let survey = survey_stores(ws);
        // `gc` runs no cell, so nothing is pinned; the sweep skips entries another
        // process has in use.
        let plan = quota::plan(&survey, quota.bytes, &SweepPins::default());

        if plan.evictions.is_empty() && plan.still_over_by_bytes.is_none() {
            writeln!(
                console.out(),
                "Already within quota ({} of {}) with nothing to reclaim.",
                human_bytes(survey.used_bytes),
                human_bytes(quota.bytes)
            )?;
            return Ok(());
        }

        let freed_bytes = if self.dry_run {
            plan.evictions
                .iter()
                .try_for_each(|entry| writeln!(console.out(), "{}", eviction_line(entry, true)))?;
            plan.freed_bytes
        } else {
            let (_, report) = quota::sweep(
                ws.models().models_dir(),
                ws.runtimes().runtimes_dir(),
                quota.bytes,
                SweepPins::default(),
            );
            report
                .removed
                .iter()
                .try_for_each(|entry| writeln!(console.out(), "{}", eviction_line(entry, false)))?;
            report.failed.iter().try_for_each(|(entry, reason)| {
                writeln!(
                    console.out(),
                    "could not reclaim {}: {reason}",
                    entry.path.display()
                )
            })?;
            report.freed_bytes
        };

        let (freed, remaining) = if self.dry_run {
            ("would free", "leaving")
        } else {
            ("freed", "using")
        };
        let remaining_bytes = survey.used_bytes.saturating_sub(freed_bytes);
        writeln!(
            console.out(),
            "{freed} {}; {remaining} {} of {}",
            human_bytes(freed_bytes),
            human_bytes(remaining_bytes),
            human_bytes(quota.bytes)
        )?;
        if let Some(warning) = still_over_warning(remaining_bytes, quota.bytes) {
            writeln!(console.out(), "{warning}")?;
        }
        Ok(())
    }
}

fn survey_stores(ws: &PipetteWorkspace) -> quota::StorageSurvey {
    quota::survey(ws.models().models_dir(), ws.runtimes().runtimes_dir())
}

/// The warning for a store still over quota after a sweep: a warning, never a
/// failure. Takes what the sweep left, not what the plan predicted: the sweep
/// tolerates a failed unlink, so `still_over_by_bytes` can miss it.
fn still_over_warning(remaining_bytes: u64, quota_bytes: u64) -> Option<String> {
    let over = remaining_bytes.saturating_sub(quota_bytes);
    (over > 0).then(|| {
        format!(
            "warning: still {} over quota; every remaining entry is in use or frees nothing",
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
        kind @ (EntryKind::Model { .. } | EntryKind::Runtime { .. }) => format!(
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
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    // `bytes >= 1024` here, so the exponent is at least 1; it tops out at TiB.
    let exp = (bytes.ilog2() / 10).min(4);
    let unit = match exp {
        1 => "KiB",
        2 => "MiB",
        3 => "GiB",
        _ => "TiB",
    };
    format!("{:.1} {unit}", bytes as f64 / (1u64 << (10 * exp)) as f64)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rstest::rstest;

    use super::*;
    use crate::test_support::CapturedConsole;
    use crate::test_support::TempWorkspace;

    const LLAMA: &str = "gguf-text://repo=meta/llama&path=Q4.gguf";

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
    #[case(1 << 50, "1024.0 TiB")]
    fn human_bytes_formats_each_unit(#[case] bytes: u64, #[case] expected: &str) {
        assert_eq!(human_bytes(bytes), expected);
    }

    /// Over 100 reads as over quota; a zero quota, which nothing lets through,
    /// reads full instead of dividing by zero.
    #[rstest]
    #[case::empty(0, 100, 0)]
    #[case::half(50, 100, 50)]
    #[case::over(150, 100, 150)]
    #[case::zero_quota(10, 0, 100)]
    fn percent_of_reads_as_a_share_of_the_quota(
        #[case] used: u64,
        #[case] quota: u64,
        #[case] expected: u64,
    ) {
        assert_eq!(percent_of(used, quota), expected);
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
        tw.store_gguf_model(LLAMA)?;
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
        tw.store_gguf_model(LLAMA)?;
        let orphan = orphan(&tw.ws)?;
        let total = survey_stores(&tw.ws).used_bytes;

        GcArgs { dry_run: true }.execute(
            &tw.reopen_with_quota(total - 1)?,
            &mut CapturedConsole::default(),
        )?;

        assert!(orphan.exists(), "a dry run reclaims nothing");
        assert_eq!(survey_stores(&tw.ws).used_bytes, total);
        Ok(())
    }

    #[test]
    fn gc_reclaims_garbage_before_a_live_model() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-gc")?;
        tw.store_gguf_model(LLAMA)?;
        let orphan = orphan(&tw.ws)?;
        let total = survey_stores(&tw.ws).used_bytes;

        GcArgs { dry_run: false }.execute(
            &tw.reopen_with_quota(total - 1)?,
            &mut CapturedConsole::default(),
        )?;

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
        tw.store_gguf_model(LLAMA)?;
        let orphan = orphan(&tw.ws)?;

        GcArgs { dry_run: false }.execute(
            &tw.reopen_with_quota(u64::MAX)?,
            &mut CapturedConsole::default(),
        )?;

        assert!(!orphan.exists(), "garbage is free to drop at any usage");
        assert_eq!(
            tw.ws.models().list()?.len(),
            1,
            "a live model under quota is not touched"
        );
        Ok(())
    }

    #[test]
    fn gc_evicts_past_a_model_in_use() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-gc-in-use")?;
        tw.store_gguf_model(LLAMA)?;
        tw.store_gguf_model("gguf-text://repo=meta/qwen&path=Q4.gguf")?;
        let survey = survey_stores(&tw.ws);
        let [first, second] = &survey.entries[..] else {
            anyhow::bail!("expected two models, got {:?}", survey.entries);
        };
        let models_dir = tw.ws.models().models_dir().to_path_buf();
        fs::create_dir_all(models_dir.join(".locks"))?;
        let name = first.path.file_name().unwrap_or_default().to_string_lossy();
        let in_use = fs::File::create(models_dir.join(format!(".locks/{name}.use")))?;
        in_use.lock_shared()?;

        GcArgs { dry_run: false }.execute(
            &tw.reopen_with_quota(survey.used_bytes - 1)?,
            &mut CapturedConsole::default(),
        )?;

        assert!(first.path.exists(), "evicted a model in use");
        assert!(!second.path.exists(), "kept over quota");
        Ok(())
    }

    #[test]
    fn gc_through_the_dispatch_evicts_only_over_quota() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("storage-dispatch")?;
        tw.store_gguf_model(LLAMA)?;
        let gc = |ws: &PipetteWorkspace, captured: &mut CapturedConsole| {
            StorageArgs {
                command: StorageCommand::Gc(GcArgs { dry_run: false }),
            }
            .execute(ws, captured)
        };

        let mut under = CapturedConsole::default();
        gc(&tw.reopen_with_quota(u64::MAX)?, &mut under)?;
        assert!(
            under.out_text().starts_with("Already within quota"),
            "got {}",
            under.out_text()
        );
        assert_eq!(
            tw.ws.models().list()?.len(),
            1,
            "nothing to reclaim under quota"
        );

        let mut over = CapturedConsole::default();
        gc(&tw.reopen_with_quota(1)?, &mut over)?;
        assert!(
            over.out_text().contains("evicted model"),
            "got {}",
            over.out_text()
        );
        assert!(tw.ws.models().list()?.is_empty());
        Ok(())
    }

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
