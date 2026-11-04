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
    /// Worker has claimed the cell and is about to invoke the runner. Pins
    /// the cell to that worker's transport for resume; does not count toward
    /// the retry cap (only terminal Success / Failed do).
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
    /// `ModelFlags::canonical_string`, empty (and omitted from the JSON) when
    /// no per-model overrides were set.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model_flags: String,
    /// The cell's `allowed_clients`, in declared order. Provenance only: not
    /// part of the cell key (see `build_cell_key`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_clients: Vec<String>,
    pub status: AttemptStatus,
    pub attempt: usize,
    pub at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Despite the name, the `client_id` of the transport whose worker wrote
    /// this event. Empty in older state files; the index then treats the cell
    /// as unpinned for that attempt.
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

/// SHA-256 hex digest of the cell's identity, the key state events match on.
pub fn build_cell_key(cell: &RunnableCell) -> String {
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
    if let Some(flags_json) = cell.runtime_flags_canonical_string() {
        hasher.update(b"\x1f");
        hasher.update(flags_json.as_bytes());
    }
    // GgufVision's Display includes the projector, so mmproj pairs already
    // differ in `model_ref`. Model flags change the prompt, so they are identity.
    if let Some(flags_str) = cell.model_flags.as_ref().and_then(|f| f.canonical_string()) {
        hasher.update(b"\t");
        hasher.update(flags_str.as_bytes());
    }
    // `allowed_clients` is deliberately not hashed: eligibility is routing,
    // not identity, so editing the pool must not invalidate completed work.
    hex::encode(hasher.finalize())
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
    /// Count of terminal attempts (Success + Failed). A `Started` with no
    /// terminal does not count, so resume after a kill doesn't burn a retry.
    attempts: usize,
    /// Most recent event status for this cell. Used to distinguish a
    /// stale terminal failure from a newer in-flight `Started` that
    /// should still resume as Missing on the next run.
    latest_status: Option<AttemptStatus>,
    /// Client id from the latest event that recorded one, to pin resumed work
    /// back to the device holding the on-device sample checkpoint.
    pinned_transport: Option<String>,
}

/// Index built from a state JSONL file. A cell is Done if any attempt
/// succeeded; Failed if it has only terminal failures; Missing
/// otherwise (including the in-flight case where only a `Started`
/// event was written before the worker died).
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

    fn entry_for(&self, cell: &RunnableCell) -> Option<&CellEntry> {
        self.cells.get(&build_cell_key(cell))
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
            // Never seen: runnable on the next pass.
            _ => CellState::Missing,
        }
    }

    pub fn attempts_for(&self, cell: &RunnableCell) -> usize {
        self.entry_for(cell).map_or(0, |e| e.attempts)
    }

    /// Client id this cell last ran on, if any of its events recorded one.
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
    /// Distinct client ids that matched, so a pattern that hit nothing (or
    /// more than the operator meant) is visible in the output.
    pub matched_clients: Vec<String>,
}

/// Drop every event a given client recorded, returning the remaining file.
///
/// Events carry the worker's `client_id` in `transport_label` (the runner
/// writes its `route_key` there, not the display label), so `patterns` match
/// client ids by substring: a leading chunk of the `ev1_…` hash is enough.
///
/// Cells left with no events read as `missing` on the next run; cells another
/// client also ran keep that client's history.
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
        BenchmarkId, ClientId, GgufText, GgufTextSource, GgufVision, GgufVisionHfFiles,
        GgufVisionSource, HfOrg, HfRepo, HfRepoName, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsRelease, LlamacppCliStockToolsSource, Model, ModelFlags,
        NonEmptyString, ReasoningEffort, RepoSubpath, RepositoryUrl, Runtime, RuntimeFlags,
    };

    use super::*;

    /// A fixed llama.cpp + `gguf_text` cell, so tests vary one dimension at a
    /// time.
    fn cell(benchmark: &str, model_name: &str, runtime_ver: &str) -> anyhow::Result<RunnableCell> {
        cell_full(benchmark, model_name, runtime_ver, vec![], None, &[])
    }

    fn cell_with_flags(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        number_gpu_layers: u32,
    ) -> anyhow::Result<RunnableCell> {
        // Built from the wire form: `RuntimeFlags` is only constructible via its
        // validating `TryFrom`.
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
                source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                    repo,
                    model: RepoSubpath::try_new("Q4_K_M.gguf".to_string()).context("filename")?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new(mm.to_string()).context("mmproj filename")?,
                    mmproj_sha256: None,
                }),
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
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(runtime_ver.to_string())
                    .context("version")?,
                asset: None,
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

    /// The cell key is persisted plan state that resume and rerun match on, so
    /// changing what is hashed re-runs recorded work. If this fails after
    /// `cell_with_flags` changed, re-pin deliberately; otherwise something
    /// re-keyed every flagged cell.
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
        // `done` is marked Done; each `missing` differs from it in one dimension.
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
                done: cell_with_model_flags("bench", "model", "rt", Some(true), None)?,
                missing: vec![cell_with_model_flags(
                    "bench",
                    "model",
                    "rt",
                    Some(false),
                    None,
                )?],
            },
            Case {
                name: "reasoning_effort",
                done: cell_with_model_flags(
                    "bench",
                    "model",
                    "rt",
                    Some(true),
                    Some(ReasoningEffort::Low),
                )?,
                missing: vec![
                    cell_with_model_flags(
                        "bench",
                        "model",
                        "rt",
                        Some(true),
                        Some(ReasoningEffort::High),
                    )?,
                    cell_with_model_flags("bench", "model", "rt", Some(true), None)?,
                ],
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

    fn cell_with_model_flags(
        benchmark: &str,
        model_name: &str,
        runtime_ver: &str,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> anyhow::Result<RunnableCell> {
        let mut c = cell(benchmark, model_name, runtime_ver)?;
        c.model_flags = Some(ModelFlags::EvalGgufText {
            enable_thinking,
            reasoning_effort,
        });
        Ok(c)
    }

    #[test]
    fn allowed_clients_do_not_affect_cell_key() -> anyhow::Result<()> {
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
    fn a_custom_label_does_not_affect_cell_key() -> anyhow::Result<()> {
        let vault_cell = |label: Option<&str>| -> anyhow::Result<RunnableCell> {
            let mut c = cell("bench", "model", "rt")?;
            c.model = serde_json::from_value(serde_json::json!({
                "type": "gguf_text",
                "source": "vault_url",
                "url": "/vault/resources/vault_1/content",
                "principal_id": "org_1",
                "custom_label": label,
            }))?;
            c.runtime = serde_json::from_value(serde_json::json!({
                "type": "llamacpp_cli_stock_tools",
                "source": "vault_remote_archive",
                "url": "/vault/resources/vault_2/content",
                "principal_id": "org_1",
                "custom_label": label,
                "flavor": "macos-arm64",
            }))?;
            Ok(c)
        };
        let (plain, labelled) = (vault_cell(None)?, vault_cell(Some("acme LFM3 q4"))?);
        assert_ne!(plain, labelled, "the label is on the cell");
        assert_eq!(build_cell_key(&plain), build_cell_key(&labelled));
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
        // An unlabeled event names no client, so even an empty pattern keeps it.
        let c = cell("bench", "model", "rt")?;
        let raw = make_event_line("plan1", &c, AttemptStatus::Success, 1)?;
        let out = filter_out_clients(&raw, &["ev1_aaa111".to_string(), String::new()])?;
        assert_eq!(out.dropped_events, 0);
        assert_eq!(out.kept, raw);
        Ok(())
    }
}
