use std::collections::HashMap;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use pipette_plan_types::RunnableCell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Done,
    Failed,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptStatus {
    /// Worker has claimed the cell and is about to invoke the runner.
    /// SweepPins the cell to that worker's transport for resume; does not
    /// count toward the retry attempt cap (only terminal Success /
    /// Failed do).
    #[serde(rename = "started")]
    Started,
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "failed")]
    Failed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StateEvent {
    pub plan_id: String,
    pub cell_key: String,
    pub benchmark_ref: String,
    pub model_ref: String,
    pub runtime_ref: String,
    pub runtime_flags: Vec<String>,
    /// Canonical model_flags string (see `ModelFlags::canonical_string`).
    /// Empty when no per-model overrides were set. Skipped in JSON so
    /// pre-flag state files round-trip unchanged.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model_flags: String,
    /// Transport `name`s this cell may dispatch to — copied from
    /// the cell's `allowed_clients`, in the operator's declared order.
    /// Recorded for provenance only; it is NOT part of the cell key
    /// (see `build_cell_key`), so editing the pool never resets state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_clients: Vec<String>,
    pub status: AttemptStatus,
    pub attempt: usize,
    pub at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Transport label (e.g. `adb:R5CY…`) of the worker that wrote
    /// this event. Empty in pre-affinity state files; the index then
    /// treats the cell as unpinned for that attempt.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub transport_label: String,
}

impl StateEvent {
    pub fn new(
        plan_id: &str,
        cell: &RunnableCell,
        status: AttemptStatus,
        attempt: usize,
        transport_label: &str,
    ) -> anyhow::Result<Self> {
        let now = time::OffsetDateTime::now_utc();
        let at = now
            .format(&time::format_description::well_known::Rfc3339)
            .context("formatting timestamp")?;
        Ok(Self {
            plan_id: plan_id.to_string(),
            cell_key: build_cell_key(cell),
            benchmark_ref: cell.benchmark.as_ref().to_string(),
            model_ref: cell.model.to_string(),
            runtime_ref: cell.runtime.to_string(),
            runtime_flags: cell.runtime_flags_canonical_string().into_iter().collect(),
            model_flags: cell
                .model_flags
                .as_ref()
                .and_then(|f| f.canonical_string())
                .unwrap_or_default(),
            allowed_clients: cell
                .allowed_clients
                .iter()
                .map(|c| c.as_ref().to_string())
                .collect(),
            status,
            attempt,
            at,
            exit_code: None,
            transport_label: transport_label.to_string(),
        })
    }

    pub fn to_json_line(&self) -> anyhow::Result<String> {
        let json = serde_json::to_string(self).context("serializing state event")?;
        Ok(format!("{json}\n"))
    }
}

/// Build the cell_key used to uniquely identify a matrix cell in the state.
///
/// Returns a SHA-256 hex digest of the cell's composite identity.
pub fn build_cell_key(cell: &RunnableCell) -> String {
    cell_key_with(cell, cell.runtime_flags_canonical_string())
}

/// The key a plan recorded before the runtime-flags identity string dropped its array
/// wrapper, or `None` for a cell with no flags — which is unaffected and keys the same.
///
/// Migration only, and only ever *read*: [`StateIndex`] falls back to it so finished cells
/// from an earlier build are still recognized instead of re-run. Delete with
/// `runtime_flags_legacy_string` once no such state is in play.
pub fn legacy_cell_key(cell: &RunnableCell) -> Option<String> {
    Some(cell_key_with(
        cell,
        Some(cell.runtime_flags_legacy_string()?),
    ))
}

fn cell_key_with(cell: &RunnableCell, runtime_flags: Option<String>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(cell.benchmark.as_ref().as_bytes());
    hasher.update(b"\t");
    let model_ref = cell.model.to_string();
    hasher.update(model_ref.as_bytes());
    hasher.update(b"\t");
    let runtime_ref = cell.runtime.to_string();
    hasher.update(runtime_ref.as_bytes());
    // The canonical form, not the wire: the wire dropped its axes, and identity must not
    // move because a format did.
    if let Some(flags_json) = runtime_flags {
        hasher.update(b"\x1f");
        hasher.update(flags_json.as_bytes());
    }
    // GgufVision's Display includes the projector, so distinct mmproj pairs
    // already differ in `model_ref` above — no separate projector hash.
    // Same model_ref with different `enable_thinking` produces a
    // different prompt; treat it as a distinct cell so resume/reruns
    // don't conflate them.
    if let Some(flags_str) = cell.model_flags.as_ref().and_then(|f| f.canonical_string()) {
        hasher.update(b"\t");
        hasher.update(flags_str.as_bytes());
    }
    // `allowed_clients` is deliberately NOT hashed. Which devices are
    // eligible to run a cell is routing, not identity: widening or
    // narrowing the transport pool (e.g. adding phones to a plan) must
    // not invalidate already-completed work. The device that actually
    // ran a cell is recorded on each event's `transport_label`; per-
    // device results are keyed there and in the warehouse, not here.
    format!("{:x}", hasher.finalize())
}

pub struct StateSummary {
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub missing: usize,
}

/// Per-cell summary built during indexing.
struct CellEntry {
    has_success: bool,
    has_failed: bool,
    /// Count of *terminal* attempts (Success + Failed). `Started`
    /// without a matching terminal — the in-flight / interrupted case
    /// — does not count, so resume after a kill doesn't burn a retry.
    attempts: usize,
    /// Most recent event status for this cell. Used to distinguish a
    /// stale terminal failure from a newer in-flight `Started` that
    /// should still resume as Missing on the next run.
    latest_status: Option<AttemptStatus>,
    /// Transport label of the most recently observed event for this
    /// cell. Used to pin resumed work back to the device that has the
    /// on-device sample checkpoint. `None` for events written by older
    /// builds that didn't record a label.
    pinned_transport: Option<String>,
}

/// Index built from a state JSONL file. A cell is Done if any attempt
/// succeeded; Failed if it has only terminal failures; Missing
/// otherwise (including the in-flight case where only a `Started`
/// event was written before the worker died). Events are indexed by
/// cell_key for O(1) lookups.
pub struct StateIndex {
    cells: HashMap<String, CellEntry>,
}

impl StateIndex {
    /// Parse a state file (one JSON event per line). `raw` may be `None` if the
    /// file does not yet exist.
    pub fn load(raw: Option<&str>, plan_id: &str) -> anyhow::Result<Self> {
        let mut cells: HashMap<String, CellEntry> = HashMap::new();
        if let Some(content) = raw {
            for (i, line) in content.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let event: StateEvent = serde_json::from_str(line)
                    .with_context(|| format!("parsing state line {}", i + 1))?;
                if event.plan_id != plan_id {
                    continue;
                }
                let entry = cells.entry(event.cell_key).or_insert(CellEntry {
                    has_success: false,
                    has_failed: false,
                    attempts: 0,
                    latest_status: None,
                    pinned_transport: None,
                });
                entry.latest_status = Some(event.status);
                if !event.transport_label.is_empty() {
                    entry.pinned_transport = Some(event.transport_label.clone());
                }
                match event.status {
                    AttemptStatus::Success => {
                        entry.has_success = true;
                        entry.attempts += 1;
                    }
                    AttemptStatus::Failed => {
                        entry.has_failed = true;
                        entry.attempts += 1;
                    }
                    AttemptStatus::Started => {}
                }
            }
        }
        Ok(Self { cells })
    }

    /// The recorded entry for a cell, under its current key or the one an earlier build
    /// wrote. The fallback is what stops a format change from re-running finished work;
    /// it costs a second hash only for cells that carry runtime flags.
    fn entry_for(&self, cell: &RunnableCell) -> Option<&CellEntry> {
        self.cells
            .get(&build_cell_key(cell))
            .or_else(|| legacy_cell_key(cell).and_then(|legacy| self.cells.get(&legacy)))
    }

    pub fn state_for(&self, cell: &RunnableCell) -> CellState {
        match self.entry_for(cell) {
            Some(entry) if entry.has_success => CellState::Done,
            Some(entry) if entry.latest_status == Some(AttemptStatus::Started) => {
                // A newer in-flight attempt should remain resumable,
                // even if earlier terminal failures exist.
                CellState::Missing
            }
            Some(entry) if entry.has_failed => CellState::Failed,
            // Started-only (in-flight or interrupted) and never-seen
            // both fall through as Missing — runnable on next pass.
            _ => CellState::Missing,
        }
    }

    pub fn attempts_for(&self, cell: &RunnableCell) -> usize {
        self.entry_for(cell).map_or(0, |e| e.attempts)
    }

    /// Transport label this cell was last seen executing on, if any.
    /// Returns `None` for cells that have no events yet, or whose
    /// events all came from a build that didn't record the label.
    pub fn pinned_transport_for(&self, cell: &RunnableCell) -> Option<&str> {
        self.entry_for(cell)
            .and_then(|e| e.pinned_transport.as_deref())
    }

    pub fn summary_for(&self, cells: &[RunnableCell]) -> StateSummary {
        let (done, failed, missing) = cells.iter().map(|cell| self.state_for(cell)).fold(
            (0, 0, 0),
            |(done, failed, missing), state| match state {
                CellState::Done => (done + 1, failed, missing),
                CellState::Failed => (done, failed + 1, missing),
                CellState::Missing => (done, failed, missing + 1),
            },
        );
        StateSummary {
            total: cells.len(),
            done,
            failed,
            missing,
        }
    }
}

/// What a per-client filter removed from a state file.
pub struct FilterOutcome {
    pub kept: String,
    pub dropped_events: usize,
    /// Distinct client ids that matched, so a pattern that hit nothing — or
    /// hit more than the operator meant — is visible in the output.
    pub matched_clients: Vec<String>,
}

/// Drop every event a given client recorded, returning the remaining file.
///
/// Events carry the worker's `client_id` in `transport_label` (the runner
/// writes its `route_key` there, not the display label), so `patterns` match
/// against client ids — a leading chunk of the `ev1_…` hash is enough.
///
/// Cells left with no events read as `missing` on the next run; cells another
/// client also ran keep that client's history. That is what makes this usable
/// to re-run one device's work without disturbing the rest of the matrix.
///
/// Lines are kept verbatim rather than re-serialized: a state file written by
/// a newer build may carry fields this one does not know, and a rewrite must
/// not silently drop them.
pub fn filter_out_clients(raw: &str, patterns: &[String]) -> anyhow::Result<FilterOutcome> {
    let empty = FilterOutcome {
        kept: String::with_capacity(raw.len()),
        dropped_events: 0,
        matched_clients: Vec::new(),
    };
    raw.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .try_fold(empty, |mut acc, (i, line)| {
            let event: StateEvent = serde_json::from_str(line)
                .with_context(|| format!("parsing state line {}", i + 1))?;
            let matched = !event.transport_label.is_empty()
                && patterns.iter().any(|p| event.transport_label.contains(p));
            if matched {
                acc.dropped_events += 1;
                if !acc.matched_clients.contains(&event.transport_label) {
                    acc.matched_clients.push(event.transport_label);
                }
            } else {
                acc.kept.push_str(line);
                acc.kept.push('\n');
            }
            Ok(acc)
        })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use pipette_plan_types::{
        BenchmarkId, ClientId, GgufText, GgufTextSource, GgufVision, GgufVisionSource, HfOrg,
        HfRepo, HfRepoName, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource,
        Model, ModelFlags, NonEmptyString, RepoSubpath, RepositoryUrl, Runtime, RuntimeFlags,
        SourceRepository,
    };

    use super::*;

    /// Build a `RunnableCell` from string identifiers — small helper
    /// so tests can vary one dimension at a time. The fixed
    /// `(LlamacppCliStockTools + macos-arm64, gguf_text + Q4_K_M.gguf)` shape
    /// keeps cells consistent across tests; per-test variation lives
    /// in `benchmark`, `model_name` (HfRepoName), `runtime_ver`, and
    /// any explicit override fields.
    fn cell(benchmark: &str, model_name: &str, runtime_ver: &str) -> anyhow::Result<RunnableCell> {
        cell_full(benchmark, model_name, runtime_ver, vec![], None, &[])
    }

    fn cell_with_flags(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        number_gpu_layers: u32,
    ) -> anyhow::Result<RunnableCell> {
        // Built from the wire form — `RuntimeFlags`' fields are private and it
        // is only constructible via its validating `TryFrom`.
        let flag: RuntimeFlags = toml::from_str(&format!(
            "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\n\
             benchmark_type = \"prefill_throughput\"\nnumber_gpu_layers = {number_gpu_layers}"
        ))?;
        cell_full(benchmark, model_name, runtime_ver, vec![flag], None, &[])
    }

    fn cell_with_clients(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        clients: &[&str],
    ) -> anyhow::Result<RunnableCell> {
        cell_full(benchmark, model_name, runtime_ver, vec![], None, clients)
    }

    fn cell_full(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        runtime_flags: Vec<RuntimeFlags>,
        mmproj_filename: Option<&str>,
        clients: &[&str],
    ) -> anyhow::Result<RunnableCell> {
        let repo = HfRepo {
            org: HfOrg::try_new("org".to_string()).context("org")?,
            repo_name: HfRepoName::try_new(model_name.to_string()).context("repo_name")?,
            revision: None,
            auth_token: None,
        };
        let model = if let Some(mm) = mmproj_filename {
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::HuggingFace {
                    repo,
                    model: RepoSubpath::try_new("Q4_K_M.gguf".to_string()).context("filename")?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new(mm.to_string()).context("mmproj filename")?,
                    mmproj_sha256: None,
                },
            })
        } else {
            Model::GgufText(GgufText {
                source: GgufTextSource::HuggingFace {
                    repo,
                    path: RepoSubpath::try_new("Q4_K_M.gguf".to_string()).context("filename")?,
                    sha256: None,
                },
            })
        };
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(runtime_ver.to_string())
                    .context("version")?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        Ok(RunnableCell {
            benchmark: BenchmarkId::try_new(benchmark.to_string()).context("benchmark")?,
            model,
            runtime,
            allowed_clients: clients
                .iter()
                .map(|c| ClientId::try_new(c.to_string()).context("client"))
                .collect::<anyhow::Result<_>>()?,
            // The helper takes a Vec for caller convenience; a real cell resolves
            // to at most one, so collapse to the first (fixtures pass 0 or 1).
            runtime_flags: runtime_flags.into_iter().next(),
            model_flags: None,
            benchmark_flags: None,
        })
    }

    fn make_event_line(
        plan_id: &str,
        cell: &RunnableCell,
        status: AttemptStatus,
        attempt: usize,
    ) -> anyhow::Result<String> {
        make_event_line_with_label(plan_id, cell, status, attempt, "")
    }

    fn make_event_line_with_label(
        plan_id: &str,
        cell: &RunnableCell,
        status: AttemptStatus,
        attempt: usize,
        transport_label: &str,
    ) -> anyhow::Result<String> {
        let mut event = StateEvent::new(plan_id, cell, status, attempt, transport_label)?;
        // Pin the timestamp for deterministic round-trip in tests.
        event.at = "2026-01-01T00:00:00Z".to_string();
        event.to_json_line()
    }

    /// The cell key is persisted plan state: resume and rerun match on it, so its
    /// composition is a compatibility surface, not an implementation detail. Changing what
    /// is hashed re-keys every flagged cell of an in-flight plan and the runner re-runs
    /// work it already recorded. Pinned to a literal so that change cannot be silent — it
    /// is what kept the key stable when `--runtime-flags` dropped its axes from the wire.
    ///
    /// This digest moved once, deliberately, when the identity string stopped being a
    /// one-element array. That re-keyed flagged cells: a plan in flight across that deploy
    /// re-runs them. Any *later* move is a regression.
    ///
    /// If this fails after `cell_with_flags` changed, the fixture moved and not the key:
    /// re-pin deliberately. If it fails on its own, something re-keyed every flagged cell.
    #[test]
    fn a_flagged_cells_key_is_stable() -> anyhow::Result<()> {
        let key = build_cell_key(&cell_with_flags("bench", "model", "rt", 99)?);
        assert_eq!(
            key,
            "4f9e0e0972c7f5ae51c0ef334f115b8ea59c6692d99e21bb2371f975c49cc415"
        );
        Ok(())
    }

    #[test]
    fn empty_state_returns_missing() -> anyhow::Result<()> {
        let idx = StateIndex::load(None, "plan1")?;
        let c = cell("bench", "model", "rt")?;
        assert_eq!(idx.state_for(&c), CellState::Missing);
        assert_eq!(idx.attempts_for(&c), 0);
        Ok(())
    }

    /// Writes are the new format; reads accept either.
    ///
    /// The three properties this PR has to hold at once, asserted together so none can be
    /// satisfied while another regresses: a freshly recorded event carries the *new* key
    /// and not the legacy one, and a cell resolves to `Done` whichever of the two its
    /// event was recorded under.
    #[test]
    fn state_is_written_in_the_new_form_and_read_in_either() -> anyhow::Result<()> {
        let c = cell_with_flags("bench", "model", "rt", 99)?;
        let new_key = build_cell_key(&c);
        let old_key =
            legacy_cell_key(&c).ok_or_else(|| anyhow::anyhow!("expected a legacy key"))?;
        assert_ne!(new_key, old_key, "the formats must key differently");

        // What this build writes.
        let written: serde_json::Value =
            serde_json::from_str(&make_event_line("plan1", &c, AttemptStatus::Success, 1)?)?;
        assert_eq!(
            written["cell_key"],
            serde_json::Value::String(new_key.clone())
        );

        // Read back under each key in turn.
        for (label, key) in [("new", &new_key), ("old", &old_key)] {
            let mut event = written.clone();
            event["cell_key"] = serde_json::Value::String(key.to_string());
            let idx = StateIndex::load(Some(&event.to_string()), "plan1")?;

            assert_eq!(idx.state_for(&c), CellState::Done, "{label} key");
            assert_eq!(idx.attempts_for(&c), 1, "{label} key");
        }
        Ok(())
    }

    /// State written before the runtime-flags identity string dropped its array wrapper
    /// still resolves. Without the legacy fallback the cell keys differently, reads as
    /// never-run, and a plan mid-sweep repeats every flagged cell it had already finished.
    #[test]
    fn a_cell_recorded_under_the_old_key_is_still_done() -> anyhow::Result<()> {
        let c = cell_with_flags("bench", "model", "rt", 99)?;
        let mut event: serde_json::Value =
            serde_json::from_str(&make_event_line("plan1", &c, AttemptStatus::Success, 1)?)?;
        // Rewrite the recorded key to the one an earlier build would have produced.
        let legacy = legacy_cell_key(&c).ok_or_else(|| anyhow::anyhow!("expected a legacy key"))?;
        assert_ne!(legacy, build_cell_key(&c), "the key must actually differ");
        event["cell_key"] = serde_json::Value::String(legacy);

        let idx = StateIndex::load(Some(&event.to_string()), "plan1")?;

        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 1);
        Ok(())
    }

    #[test]
    fn success_marks_done() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let line = make_event_line("plan1", &c, AttemptStatus::Success, 1)?;
        let idx = StateIndex::load(Some(&line), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 1);
        Ok(())
    }

    #[test]
    fn failure_marks_failed() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let line = make_event_line("plan1", &c, AttemptStatus::Failed, 1)?;
        let idx = StateIndex::load(Some(&line), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Failed);
        assert_eq!(idx.attempts_for(&c), 1);
        Ok(())
    }

    #[test]
    fn success_after_failure_is_done() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut lines = make_event_line("plan1", &c, AttemptStatus::Failed, 1)?;
        lines.push_str(&make_event_line("plan1", &c, AttemptStatus::Success, 2)?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 2);
        Ok(())
    }

    #[test]
    fn failure_after_success_still_done() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut lines = make_event_line("plan1", &c, AttemptStatus::Success, 1)?;
        lines.push_str(&make_event_line("plan1", &c, AttemptStatus::Failed, 2)?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 2);
        Ok(())
    }

    #[test]
    fn filters_by_plan_id() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let line = make_event_line("other-plan", &c, AttemptStatus::Success, 1)?;
        let idx = StateIndex::load(Some(&line), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Missing);
        Ok(())
    }

    #[test]
    fn dimension_distinguishes_cells() -> anyhow::Result<()> {
        // For each dimension, the first cell is the one we mark Done;
        // the remaining cells differ along that dimension only and
        // must register as Missing (distinct keys).
        struct Case {
            name: &'static str,
            done: RunnableCell,
            missing: Vec<RunnableCell>,
        }
        let cases = vec![
            Case {
                name: "runtime_flags",
                done: cell_with_flags("bench", "model", "rt", 99)?,
                missing: vec![cell_with_flags("bench", "model", "rt", 0)?],
            },
            Case {
                // Projector is part of GgufVision Display / model_ref.
                name: "vision_mmproj",
                done: cell_full("bench", "model", "rt", vec![], Some("mmproj-F16.gguf"), &[])?,
                missing: vec![
                    cell_full(
                        "bench",
                        "model",
                        "rt",
                        vec![],
                        Some("mmproj-Q8_0.gguf"),
                        &[],
                    )?,
                    cell_full("bench", "model", "rt", vec![], None, &[])?,
                ],
            },
            Case {
                name: "model_flags",
                done: cell_with_thinking("bench", "model", "rt", Some(true))?,
                missing: vec![cell_with_thinking("bench", "model", "rt", Some(false))?],
            },
        ];
        for case in cases {
            let line = make_event_line("plan1", &case.done, AttemptStatus::Success, 1)?;
            let idx = StateIndex::load(Some(&line), "plan1")?;
            assert_eq!(
                idx.state_for(&case.done),
                CellState::Done,
                "{}: done cell should be Done",
                case.name
            );
            for (i, m) in case.missing.iter().enumerate() {
                assert_eq!(
                    idx.state_for(m),
                    CellState::Missing,
                    "{}: missing[{i}] should be Missing",
                    case.name
                );
            }
        }
        Ok(())
    }

    fn cell_with_thinking(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        enable_thinking: Option<bool>,
    ) -> anyhow::Result<RunnableCell> {
        let mut c = cell(benchmark, model_name, runtime_ver)?;
        c.model_flags = Some(ModelFlags::EvalGgufText { enable_thinking });
        Ok(c)
    }

    #[test]
    fn allowed_clients_do_not_affect_cell_key() -> anyhow::Result<()> {
        // The eligible-transport pool is routing, not identity: changing
        // it — adding/removing devices, reordering, or clearing it — must
        // not change the cell key, so editing a plan's clients never
        // resets completed work.
        let base = build_cell_key(&cell("bench", "model", "rt")?);
        let one = build_cell_key(&cell_with_clients("bench", "model", "rt", &["phone-a"])?);
        let other = build_cell_key(&cell_with_clients("bench", "model", "rt", &["phone-b"])?);
        let widened = build_cell_key(&cell_with_clients(
            "bench",
            "model",
            "rt",
            &["phone-a", "phone-b"],
        )?);
        let reordered = build_cell_key(&cell_with_clients(
            "bench",
            "model",
            "rt",
            &["phone-b", "phone-a"],
        )?);
        assert_eq!(base, one);
        assert_eq!(base, other);
        assert_eq!(base, widened);
        assert_eq!(base, reordered);
        Ok(())
    }

    #[test]
    fn summary_counts() -> anyhow::Result<()> {
        let c1 = cell("bench1", "model", "rt")?;
        let c2 = cell("bench2", "model", "rt")?;
        let c3 = cell("bench3", "model", "rt")?;
        let mut lines = make_event_line("plan1", &c1, AttemptStatus::Success, 1)?;
        lines.push_str(&make_event_line("plan1", &c2, AttemptStatus::Failed, 1)?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        let summary = idx.summary_for(&[c1, c2, c3]);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.done, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.missing, 1);
        Ok(())
    }

    #[test]
    fn started_alone_is_missing_but_pinned() -> anyhow::Result<()> {
        // A worker died after writing Started but before producing a
        // terminal — index treats the cell as runnable (Missing) and
        // remembers the device.
        let c = cell("bench", "model", "rt")?;
        let line =
            make_event_line_with_label("plan1", &c, AttemptStatus::Started, 1, "adb:DEVICE_A")?;
        let idx = StateIndex::load(Some(&line), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Missing);
        assert_eq!(idx.attempts_for(&c), 0); // started alone doesn't burn a retry
        assert_eq!(idx.pinned_transport_for(&c), Some("adb:DEVICE_A"));
        Ok(())
    }

    #[test]
    fn started_then_success_done_and_pinned() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut lines =
            make_event_line_with_label("plan1", &c, AttemptStatus::Started, 1, "ssh:host_b")?;
        lines.push_str(&make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Success,
            1,
            "ssh:host_b",
        )?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 1); // only the terminal counts
        assert_eq!(idx.pinned_transport_for(&c), Some("ssh:host_b"));
        Ok(())
    }

    #[test]
    fn started_after_failure_is_still_missing() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut lines = make_event_line_with_label("plan1", &c, AttemptStatus::Failed, 1, "adb:A")?;
        lines.push_str(&make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Started,
            2,
            "adb:A",
        )?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Missing);
        assert_eq!(idx.attempts_for(&c), 1); // interrupted retry doesn't burn attempt 2
        assert_eq!(idx.pinned_transport_for(&c), Some("adb:A"));
        Ok(())
    }

    #[test]
    fn pinned_transport_follows_latest_event() -> anyhow::Result<()> {
        // First attempt on A failed, retried on B and succeeded —
        // pinned label tracks the most recent run, not the first.
        let c = cell("bench", "model", "rt")?;
        let mut lines = make_event_line_with_label("plan1", &c, AttemptStatus::Failed, 1, "adb:A")?;
        lines.push_str(&make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Success,
            2,
            "adb:B",
        )?);
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.pinned_transport_for(&c), Some("adb:B"));
        Ok(())
    }

    #[test]
    fn legacy_events_without_label_are_unpinned() -> anyhow::Result<()> {
        // State written by an older build has no transport_label;
        // the cell is still Done/Failed/Missing as before but
        // unpinned, so any worker can pick it up.
        let c = cell("bench", "model", "rt")?;
        let line = make_event_line("plan1", &c, AttemptStatus::Failed, 1)?;
        let idx = StateIndex::load(Some(&line), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Failed);
        assert_eq!(idx.pinned_transport_for(&c), None);
        Ok(())
    }

    #[test]
    fn skips_empty_lines() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut lines = String::from("\n\n");
        lines.push_str(&make_event_line("plan1", &c, AttemptStatus::Success, 1)?);
        lines.push_str("\n\n");
        let idx = StateIndex::load(Some(&lines), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        Ok(())
    }

    #[test]
    fn filter_drops_only_the_named_client() -> anyhow::Result<()> {
        let a = cell("bench_a", "model", "rt")?;
        let b = cell("bench_b", "model", "rt")?;
        let mut raw = String::new();
        raw.push_str(&make_event_line_with_label(
            "plan1",
            &a,
            AttemptStatus::Success,
            1,
            "ev1_aaa111",
        )?);
        raw.push_str(&make_event_line_with_label(
            "plan1",
            &b,
            AttemptStatus::Success,
            1,
            "ev1_bbb222",
        )?);

        let out = filter_out_clients(&raw, &["ev1_aaa111".to_string()])?;
        assert_eq!(out.dropped_events, 1);
        assert_eq!(out.matched_clients, vec!["ev1_aaa111"]);

        // The wiped client's cell is runnable again; the other client's is not.
        let idx = StateIndex::load(Some(&out.kept), "plan1")?;
        assert_eq!(idx.state_for(&a), CellState::Missing);
        assert_eq!(idx.state_for(&b), CellState::Done);
        Ok(())
    }

    #[test]
    fn filter_matches_a_client_id_prefix() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let raw = make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Success,
            1,
            "ev1_9e1c2ad6c894467ba4a42a4fc5bb97d5621036c84fddb1dadb6b157551465e6c",
        )?;
        let out = filter_out_clients(&raw, &["ev1_9e1c2ad6".to_string()])?;
        assert_eq!(out.dropped_events, 1);
        Ok(())
    }

    #[test]
    fn filter_keeps_a_cell_another_client_also_ran() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let mut raw = String::new();
        raw.push_str(&make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Failed,
            1,
            "ev1_aaa111",
        )?);
        raw.push_str(&make_event_line_with_label(
            "plan1",
            &c,
            AttemptStatus::Success,
            2,
            "ev1_bbb222",
        )?);

        let out = filter_out_clients(&raw, &["ev1_aaa111".to_string()])?;
        let idx = StateIndex::load(Some(&out.kept), "plan1")?;
        assert_eq!(idx.state_for(&c), CellState::Done);
        assert_eq!(idx.attempts_for(&c), 1);
        Ok(())
    }

    #[test]
    fn filter_reports_a_pattern_that_matched_nothing() -> anyhow::Result<()> {
        let c = cell("bench", "model", "rt")?;
        let raw = make_event_line_with_label("plan1", &c, AttemptStatus::Success, 1, "ev1_bbb222")?;
        let out = filter_out_clients(&raw, &["ev1_nope".to_string()])?;
        assert_eq!(out.dropped_events, 0);
        assert!(out.matched_clients.is_empty());
        assert_eq!(out.kept, raw);
        Ok(())
    }

    #[test]
    fn filter_never_drops_an_unlabeled_event() -> anyhow::Result<()> {
        // Pre-affinity state files carry no transport_label. An empty pattern
        // match would wipe results no client can be shown to have produced.
        let c = cell("bench", "model", "rt")?;
        let raw = make_event_line("plan1", &c, AttemptStatus::Success, 1)?;
        let out = filter_out_clients(&raw, &["ev1_aaa111".to_string(), String::new()])?;
        assert_eq!(out.dropped_events, 0);
        assert_eq!(out.kept, raw);
        Ok(())
    }
}
